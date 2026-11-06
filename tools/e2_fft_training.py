"""复用旧E2损失的梯度标定、短冒烟、同预算续训；不修改历史训练入口。"""
import hashlib
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from models.e2_compat import bare_network, compute_losses, e2_objective, load_trainable_strict
from models.e2_geometry import GeometryCache
from models.e2_local_fft import LocalFFTLoss
from tools.e2_fft_experiment import SOURCE, environment, load, read, setup, sha, write


def prepare_protocol(args):
    core, out = setup(args)
    rows = read(core.ROOT / 'training_manifest.json')
    identities = {}
    for i, row in enumerate(rows):
        identities.setdefault(Path(row['cloth']).stem.split('_')[0], []).append(i)
    # 清单索引保持不变，继续使用按原索引存储的历史mask缓存。
    ordered = sorted([k for k, v in identities.items() if len(v) == 1],
                     key=lambda k: hashlib.sha256(('e2_fft_dev64/' + k).encode()).hexdigest())
    validation = sorted(identities[k][0] for k in ordered[:64])
    excluded = set(validation)
    training = [i for i in range(len(rows)) if i not in excluded]
    patterns = {'solid': lambda s: not any(x in s for x in ['stripe', 'plaid', 'check', 'polka', 'dotted', 'floral', 'print']),
                'stripes': lambda s: 'stripe' in s,
                'checks': lambda s: any(x in s for x in ['plaid', 'checkered', 'gingham']),
                'dots': lambda s: 'polka' in s or 'dotted' in s}
    chosen = []
    for pattern, predicate in patterns.items():
        candidates = [i for i in training if predicate(str(rows[i]['caption']).lower())]
        assert candidates, '训练文本没有可用的%s候选；不得伪造纹样标签' % pattern
        index = min(candidates, key=lambda i: hashlib.sha256(('e2_fft_smoke/' + rows[i]['cloth']).encode()).hexdigest())
        chosen.append(dict(index=index, pattern_candidate=pattern, **rows[index]))
    value = dict(version=1, source_manifest=str(core.ROOT / 'training_manifest.json'),
        source_manifest_sha256=sha(core.ROOT / 'training_manifest.json'), train_indices=training,
        validation_indices=validation, smoke=chosen, mask_cache=str(core.MASKS),
        validation_pretraining_exposure=True,
        validation_note='仅用于续训开发；成熟E2已见过这些训练身份，不是独立泛化测试',
        pattern_note='按文本选固定诊断候选，需要结合实际参考图核查，不作为真实纹样标注')
    path = out / 'split.json'
    if path.exists():
        assert read(path) == value, '不覆盖已固定split'
    else:
        write(path, value)
    print('SPLIT', len(training), len(validation), [(x['pattern_candidate'], x['index']) for x in chosen], flush=True)


def dataset(pipe, core):
    from train_GAM_texture_joint import JointTextureDataset
    return JointTextureDataset(str(core.ROOT / 'training_manifest.json'), pipe.tokenizer,
        str(core.PROJECT.parent / 'datasets/BF/training'), width=384, height=512,
        texture_preprocess_mode='plain_resize', mask_cache=str(core.MASKS))


def save_diagnostics(folder, batch, captured, stats, geometry):
    from PIL import Image, ImageDraw
    folder.mkdir(parents=True, exist_ok=True)
    tiles = []
    for key, value in [('sketch', batch['vae_sketch']), ('reference', batch['texture_image']),
                       ('GT', batch['vae_cloth']), ('predicted_t100', captured['decoded'])]:
        pixels = ((value[0].detach().float().cpu() + 1) * 127.5).clamp(0, 255).round().byte()
        image = Image.fromarray(pixels.permute(1, 2, 0).numpy())
        draw = ImageDraw.Draw(image)
        draw.text((5, 5), key, fill='red')
        if key == 'predicted_t100':
            for scale in stats['samples'][0]['scales']:
                for x, y in scale['predicted_coordinates']:
                    q = scale['size']; draw.rectangle((x, y, x + q - 1, y + q - 1), outline='red', width=1)
        tiles.append(image)
    panel = Image.new('RGB', (384 * len(tiles), 512), 'white')
    for i, image in enumerate(tiles):
        panel.paste(image, (i * 384, 0))
    panel.save(folder / 'inputs_prediction_patches.png')
    torch.save(dict(decoded=captured['decoded'].detach().cpu(), body=geometry.body.cpu(),
                    signed_distance=geometry.signed_distance.cpu(), patch_stats=stats), folder / 'diagnostics.pt')


def calibrate(args):
    from train_GAM_texture_joint import VGGGramStyleLoss
    core, out = setup(args)
    assert read(out / 'regression/forward_result.json')['passed']
    assert read(out / 'regression/image_result.json')['passed']
    if not (out / 'split.json').exists():
        prepare_protocol(args)
    pipe, network, checked = load(args, core)
    network.train()
    data = dataset(pipe, core)
    style = VGGGramStyleLoss().to(args.device).eval().requires_grad_(False)
    sampler = LocalFFTLoss()
    cache = GeometryCache(out / 'geometry_cache')
    generator = torch.Generator(device=args.device).manual_seed(42000)
    records, ratios = [], []
    for choice in read(out / 'split.json')['smoke']:
        item = data[choice['index']]
        batch = {k: v[None].to(args.device) for k, v in item.items()}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            total, raw, token, captured, stats = compute_losses(pipe, network, batch, generator,
                style, core, fft=sampler, fft_weight=1., audit_mode=True)
        targets = [captured['epsilon'], captured['vae_input'], network.adapter.projections[0].weight,
                   network.adapter.fusion.downconv1[0].weight,
                   next(p for p in network.unet.parameters() if p.requires_grad)]
        names = ['epsilon', 'vae_input', 'projection0', 'fusion_down1', 'GAM']
        gradients = {}
        for name, loss in dict(old=e2_objective(raw), **{k: raw[k] for k in ['mse', 'gram', 'edge', 'color', 'body', 'rest', 'fft']}).items():
            values = torch.autograd.grad(loss, targets, retain_graph=True, allow_unused=True)
            norms = {k: float(v.detach().float().norm()) if v is not None else 0.
                     for k, v in zip(names, values)}
            # decode输入为x0/scaling_factor；链式法则换算真实x0的梯度范数。
            norms['x0_equivalent'] = norms['vae_input'] / pipe.vae.config.scaling_factor
            assert all(np.isfinite(v) for v in norms.values())
            gradients[name] = norms
        if stats['valid_n']:
            assert all(gradients['fft'][k] > 0 for k in ['epsilon', 'vae_input', 'projection0', 'fusion_down1', 'GAM'])
            ratios.append(gradients['old']['epsilon'] / gradients['fft']['epsilon'])
        geometry, key = cache.get(choice['index'], batch['garment_mask'], batch['vae_sketch'],
                                  {'original_manifest': read(out / 'split.json')['source_manifest_sha256']})
        folder = out / 'calibration' / str(choice['index'])
        save_diagnostics(folder, batch, captured, stats, geometry)
        records.append(dict(index=choice['index'], pattern_candidate=choice['pattern_candidate'],
            token=token, gradients=gradients, patch_stats=stats, geometry_cache_key=key,
            losses={k: float(v.detach()) for k, v in raw.items()}))
        write(folder / 'result.json', records[-1])
        network.adapter.clear()
        print('CALIBRATION', choice['index'], gradients['fft']['epsilon'], flush=True)
        del total, raw, captured, batch
    assert len(ratios) >= 2, '没有至少两个有效纹理参考，不能标定FFT权重'
    weight = .1 * float(np.median(ratios))
    assert np.isfinite(weight) and weight > 0
    write(out / 'calibration/result.json', dict(passed=True, weight=weight,
        target_epsilon_gradient_ratio=.1, valid_cases=len(ratios), records=records,
        initialization=checked, environment=environment(), peak_gib=torch.cuda.max_memory_allocated() / 2**30))
    print('FFT_WEIGHT', weight, flush=True)


def tensor_hash(items):
    digest = hashlib.sha256()
    for name, value in items:
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def optimizer_hash(optimizer):
    state = optimizer.state_dict()
    return tensor_hash(('%s/%s' % (index, key), value)
                       for index, values in sorted(state['state'].items())
                       for key, value in sorted(values.items()) if isinstance(value, torch.Tensor))


def frozen_hash(pipe, network):
    modules = [('unet', network.unet), ('reference', pipe.reference_unet), ('vae', pipe.vae),
               ('text', pipe.text_encoder), ('image', pipe.image_encoder), ('old_texture', pipe.bf_texture_conditioner)]
    return tensor_hash((prefix + '/' + n, p) for prefix, module in modules
                       for n, p in module.named_parameters() if not p.requires_grad)


class DataStream:
    def __init__(self, data, indices, rank=0, world=1, workers=0):
        from torch.utils.data import Subset
        self.data = Subset(data, indices)
        self.rank, self.world, self.workers = rank, world, workers
        self.epoch, self.cursor = 0, 0
        self.iterator = None

    def reset(self):
        from torch.utils.data import DataLoader, DistributedSampler
        from itertools import islice
        sampler = DistributedSampler(self.data, num_replicas=self.world, rank=self.rank,
                                     seed=42, drop_last=False)
        sampler.set_epoch(self.epoch)
        loader = DataLoader(self.data, batch_size=1, sampler=sampler, num_workers=self.workers,
            generator=torch.Generator().manual_seed(424200 + self.epoch * 1000 + self.rank))
        self.iterator = iter(islice(loader, self.cursor, None))

    def next(self):
        if self.iterator is None:
            self.reset()
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.epoch += 1; self.cursor = 0
            self.reset(); batch = next(self.iterator)
        self.cursor += 1
        return batch

    def state_dict(self):
        return dict(epoch=self.epoch, cursor=self.cursor)

    def load_state_dict(self, state):
        self.epoch, self.cursor = state['epoch'], state['cursor']
        self.iterator = None


class TrainingRun:
    def __init__(self, args, core, pipe, network, data, style, config, indices, accelerator):
        from diffusers.optimization import get_scheduler
        self.acc, self.core, self.pipe, self.style = accelerator, core, pipe, style
        self.config = config
        self.params = [p for p in network.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(self.params, lr=config['learning_rate'], betas=(.9, .999), eps=1e-8, weight_decay=.01)
        self.scheduler = get_scheduler('cosine', optimizer=optimizer,
            num_warmup_steps=config['warmup'], num_training_steps=config['updates'])
        self.network, self.optimizer = accelerator.prepare(network, optimizer)
        self.network.train()
        self.stream = DataStream(data, indices, accelerator.process_index, accelerator.num_processes, config.get('workers', 0))
        self.rng = torch.Generator(device=accelerator.device).manual_seed(42000 + accelerator.process_index)
        self.fft = LocalFFTLoss(sizes=config.get('fft_sizes', (32, 64)), count=config.get('fft_count', 4),
            t_max=config.get('fft_t_max', 249), std_min=config.get('fft_std_min', .02),
            seed=110642 + accelerator.process_index) if config['fft'] else None
        self.step, self.chain = 0, '0' * 64

    def update(self):
        started = time.monotonic()
        values, tokens, counts = {}, [], dict(active_n=0, valid_n=0)
        diagnostics = []
        accumulation = 8 // self.acc.num_processes
        for _ in range(accumulation):
            batch = {k: v.to(self.acc.device) for k, v in self.stream.next().items()}
            with self.acc.accumulate(self.network):
                with self.acc.autocast():
                    loss, raw, token, captured, stats = compute_losses(self.pipe, self.network, batch,
                        self.rng, self.style, self.core, self.fft, self.config['fft_weight'])
                assert torch.isfinite(loss)
                if self.config.get('diagnostics'):
                    for key in ['epsilon', 'vae_input', 'decoded']:
                        captured[key].retain_grad()
                    for value in captured['features']:
                        value.retain_grad()
                self.acc.backward(loss)
                if self.config.get('diagnostics'):
                    watched = [(n, p) for n, p in bare_network(self.network).named_parameters()
                               if n in ['adapter.fusion.downconv1.0.weight', 'adapter.projections.0.weight']]
                    diagnostics.append(dict(
                        forward=tensor_hash((k, captured[k]) for k in ['epsilon', 'vae_input', 'decoded']),
                        features=tensor_hash((str(i), v) for i, v in enumerate(captured['features'])),
                        feature_gradients=[tensor_hash([(str(i), v.grad)])
                                           for i, v in enumerate(captured['features'])],
                        image_gradients={k: tensor_hash([(k, captured[k].grad)])
                                         for k in ['epsilon', 'vae_input', 'decoded']},
                        gradients={n: dict(sha256=tensor_hash([(n, p.grad)]), norm=float(p.grad.float().norm()),
                                           stride=list(p.grad.stride()))
                                   for n, p in watched if p.grad is not None}))
                if self.acc.sync_gradients:
                    norm = self.acc.clip_grad_norm_(self.params, 1.)
                    assert torch.isfinite(norm)
                self.optimizer.step()
                if self.acc.sync_gradients:
                    self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
            self.chain = hashlib.sha256((self.chain + str(sorted(token.items()))).encode()).hexdigest()
            tokens.append(token)
            for k, v in dict(total=loss, **raw).items():
                values[k] = values.get(k, 0.) + float(v.detach()) / accumulation
            for k in counts:
                counts[k] += stats.get(k, 0)
            bare_network(self.network).adapter.clear()
            del captured, raw, loss
        assert self.acc.sync_gradients and not self.acc.optimizer_step_was_skipped
        self.step += 1
        return dict(step=self.step, seconds=time.monotonic() - started, losses=values, patch_counts=counts,
                    diagnostics=diagnostics,
                    grad_norm=float(norm), lr=self.scheduler.get_last_lr()[0], tokens=tokens,
                    chain=self.chain, peak_gib=torch.cuda.max_memory_allocated() / 2**30)

    def local_state(self):
        return dict(rng=self.rng.get_state().cpu(), patch=self.fft.sampling_state() if self.fft else None,
            stream=self.stream.state_dict(), chain=self.chain, python=random.getstate(), numpy=np.random.get_state(),
            torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state(self.acc.device))

    def save(self, path):
        states = [None] * self.acc.num_processes
        if len(states) > 1:
            torch.distributed.all_gather_object(states, self.local_state())
        else:
            states[0] = self.local_state()
        if self.acc.is_main_process:
            state = dict(network=self.acc.unwrap_model(self.network).trainable_state(), optimizer=self.optimizer.state_dict(),
                scheduler=self.scheduler.state_dict(), states=states, config=self.config, step=self.step,
                world=self.acc.num_processes)
            temporary = Path(path).with_suffix('.tmp')
            torch.save(state, temporary); temporary.replace(path)
        self.acc.wait_for_everyone()

    def restore(self, path):
        state = torch.load(path, map_location='cpu', weights_only=False)
        assert state['world'] == self.acc.num_processes and state['config'] == self.config
        load_trainable_strict(self.acc.unwrap_model(self.network), state['network'])
        self.optimizer.load_state_dict(state['optimizer']); self.scheduler.load_state_dict(state['scheduler'])
        local = state['states'][self.acc.process_index]
        self.rng.set_state(local['rng']); self.stream.load_state_dict(local['stream'])
        if self.fft:
            self.fft.load_sampling_state(local['patch'])
        self.step, self.chain = state['step'], local['chain']
        random.setstate(local['python']); np.random.set_state(local['numpy'])
        torch.set_rng_state(local['torch_cpu']); torch.cuda.set_rng_state(local['torch_cuda'], self.acc.device)


def parameter_changes(before, network):
    changes = dict(fusion=0., projections=0., GAM=0.)
    for name, value in network.trainable_state().items():
        group = 'fusion' if name.startswith('adapter.fusion.') else 'projections' if name.startswith('adapter.projections.') else 'GAM'
        changes[group] += float((value.float() - before[name].float()).square().sum())
    return {k: v ** .5 for k, v in changes.items()}


def backward_probe(args):
    """零学习率重复同一批，区分反传不复现与checkpoint加载问题。"""
    from accelerate import Accelerator
    from train_GAM_texture_joint import VGGGramStyleLoss
    core, out = setup(args)
    acc = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=8)
    args.device = str(acc.device)
    pipe, network, _ = load(args, core)
    style = VGGGramStyleLoss().to(args.device).eval().requires_grad_(False)
    indices = [c['index'] for c in read(out / 'split.json')['smoke']] * 2
    run = TrainingRun(args, core, pipe, network, dataset(pipe, core), style,
        dict(fft=False, fft_weight=0., learning_rate=0., updates=8, warmup=0,
             workers=0, diagnostics=True), indices, acc)
    initial = run.local_state()
    weights = network.trainable_state()
    records = []
    for mutation in ['none_first', 'none_repeat', 'load_same_weights', 'load_optimizer']:
        if mutation == 'load_same_weights':
            load_trainable_strict(network, weights)
        if mutation == 'load_optimizer':
            state = torch.load(out / 'tests/backward_probe.pt', map_location='cpu', weights_only=False)
            run.optimizer.load_state_dict(state['optimizer'])
            del state
        run.rng.set_state(initial['rng']); run.stream.load_state_dict(initial['stream'])
        run.chain = initial['chain']
        random.setstate(initial['python']); np.random.set_state(initial['numpy'])
        torch.set_rng_state(initial['torch_cpu']); torch.cuda.set_rng_state(initial['torch_cuda'], acc.device)
        row = run.update()
        assert tensor_hash(network.trainable_state().items()) == tensor_hash(weights.items())
        records.append(dict(mutation=mutation, **row))
        if mutation == 'none_first':
            run.save(out / 'tests/backward_probe.pt')
        write(out / 'tests/backward_probe.json', dict(records=records, environment=environment()))
        print('BACKWARD_PROBE', mutation, row['grad_norm'], flush=True)


def smoke(args):
    from accelerate import Accelerator
    from train_GAM_texture_joint import VGGGramStyleLoss
    core, out = setup(args)
    calibration = read(out / 'calibration/result.json')
    assert calibration['passed']
    assert read(out / 'tests/ddp_result.json')['passed']
    acc = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=8)
    assert acc.num_processes == 1, '冒烟及精确恢复比较用一张GPU'
    args.device = str(acc.device)
    pipe, network, checked = load(args, core)
    before = {k: v.clone() for k, v in network.trainable_state().items()}
    frozen_before = frozen_hash(pipe, network)
    data = dataset(pipe, core)
    indices = [c['index'] for c in read(out / 'split.json')['smoke']] * 2
    style = VGGGramStyleLoss().to(args.device).eval().requires_grad_(False)
    results = {}
    for arm in ['E2', 'E2_FFT']:
        load_trainable_strict(network, before)
        torch.cuda.reset_peak_memory_stats()
        config = dict(arm=arm, fft=arm == 'E2_FFT', fft_weight=calibration['weight'] if arm == 'E2_FFT' else 0.,
                      learning_rate=5e-5, updates=4, warmup=0, workers=0, diagnostics=True)
        run = TrainingRun(args, core, pipe, network, data, style, config, indices, acc)
        folder = out / 'smoke' / arm
        folder.mkdir(parents=True, exist_ok=True)
        logs = [run.update(), run.update()]
        state2_hash = tensor_hash(network.trainable_state().items())
        optimizer2_hash = optimizer_hash(run.optimizer)
        scheduler2_state = run.scheduler.state_dict()
        run.save(folder / 'resume_test.pt')
        expected = run.update()
        expected_state = acc.unwrap_model(run.network).trainable_state()
        expected_hash = tensor_hash(expected_state.items())
        run.restore(folder / 'resume_test.pt')
        assert tensor_hash(network.trainable_state().items()) == state2_hash, 'checkpoint参数未完整恢复'
        assert optimizer_hash(run.optimizer) == optimizer2_hash, 'optimizer状态未完整恢复'
        assert scheduler2_state == run.scheduler.state_dict(), 'scheduler状态未完整恢复'
        replay = run.update()
        actual_state = acc.unwrap_model(run.network).trainable_state()
        actual_hash = tensor_hash(actual_state.items())
        difference = {name: float((value - actual_state[name]).abs().max()) for name, value in expected_state.items()}
        failed = [name for name, value in expected_state.items()
                  if not torch.allclose(value, actual_state[name], atol=1e-7, rtol=1e-5)]
        resume_result = dict(bitwise_equal=expected_hash == actual_hash,
            parameter_max_abs=max(difference.values()), per_parameter_max_abs=difference,
            outside_tolerance=failed, atol=1e-7, rtol=1e-5,
            expected=expected, replay=replay, checkpoint_parameters_exact=True,
            optimizer_exact=True, scheduler_exact=True)
        write(folder / 'resume_comparison.json', resume_result)
        assert not failed, '恢复后参数更新超出FP32容差；见resume_comparison.json'
        assert expected['tokens'] == replay['tokens'] and expected['chain'] == replay['chain']
        assert expected['losses'] == replay['losses']
        del expected_state, actual_state
        logs += [replay, run.update()]
        changes = parameter_changes(before, acc.unwrap_model(run.network))
        assert all(v > 0 and np.isfinite(v) for v in changes.values())
        assert frozen_hash(pipe, network) == frozen_before, '冻结参数被修改'
        if config['fft']:
            assert sum(row['patch_counts']['valid_n'] for row in logs) > 0, '冒烟没有激活FFT'
        run.save(folder / 'latest.pt')
        results[arm] = dict(passed=True, updates=run.step, effective_batch=8, logs=logs,
            executed_update_calls=5, resume_replay_updates=1,
            changes_l2=changes, frozen_unchanged=True, resume_exact=expected_hash == actual_hash,
            resume_within_tolerance=True, resume_parameter_max_abs=resume_result['parameter_max_abs'],
            checkpoint_sha256=sha(folder / 'latest.pt'), final_hash=tensor_hash(network.trainable_state().items()))
        write(folder / 'result.json', results[arm])
        network = acc.unwrap_model(run.network, keep_fp32_wrapper=False)
        del run
        acc.free_memory()
        print('SMOKE_PASS', arm, changes, flush=True)
    assert [r['tokens'] for r in results['E2']['logs']] == [r['tokens'] for r in results['E2_FFT']['logs']], '两臂旧训练随机流不同'
    write(out / 'smoke/result.json', dict(passed=True, same_legacy_stream=True, arms=results,
        initialization=checked, environment=environment()))


def prepare_comparison(args):
    core, out = setup(args)
    assert read(out / 'smoke/result.json')['passed']
    weight = read(out / 'calibration/result.json')['weight']
    source = read(SOURCE / 'configs/e2_fft_regression.json')
    base = dict(status='planned_not_run', protocol='matched_E2_continuation',
        initialization=source['checkpoint'], initialization_sha256=source['checkpoint_sha256'],
        split=str(out / 'split.json'), split_sha256=sha(out / 'split.json'),
        updates=2000, effective_batch=8, world_size=1, learning_rate=5e-5, warmup=100, workers=2,
        train_seed=42, sampler_seed=42, noise_seed_per_rank='42000+rank', patch_seed_per_rank='110642+rank',
        precision='bf16', prediction='epsilon', old_losses='unchanged_all_timesteps',
        deterministic_algorithms=True, cublas_workspace=':4096:8',
        sdp_backends=dict(flash=False, efficient=False, math=True, cudnn=False),
        fft_sizes=[32, 64], fft_count=4, fft_t_max=249, fft_std_min=.02,
        inference=dict(width=384, height=512, ddim_steps=50, cfg=7., sketch_scale=.6),
        validation_pretraining_exposure=True, metrics_version='legacy_v1_keep_and_recompute_both_arms',
        historical_iou_mismatch_unresolved=True,
        code_commit=environment()['commit'])
    for arm in ['E2', 'E2_FFT']:
        write(out / 'comparison' / (arm + '.json'), dict(base, arm=arm, fft=arm == 'E2_FFT',
                                                       fft_weight=weight if arm == 'E2_FFT' else 0.))
    write(out / 'comparison/status.json', dict(status='prepared_not_run',
        actual_long_training_updates=0, smoke_updates_per_arm=4, fft_weight=weight,
        identical_fields_except=['arm', 'fft', 'fft_weight'],
        resources='prepared_world1_accumulation8; change both configs together for world2/4/8'))
    print('COMPARISON_PREPARED_NOT_RUN', flush=True)


def train(args):
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs
    from train_GAM_texture_joint import VGGGramStyleLoss
    core, out = setup(args)
    assert read(out / 'smoke/result.json')['passed']
    config = read(args.config)
    assert sha(config['split']) == config['split_sha256']
    source = read(SOURCE / 'configs/e2_fft_regression.json')
    assert config['initialization'] == source['checkpoint'] and config['initialization_sha256'] == source['checkpoint_sha256']
    assert config['effective_batch'] == 8 and config['code_commit'] == environment()['commit']
    world = int(os.environ.get('WORLD_SIZE', '1'))
    assert world in [1, 2, 4, 8] and world == config['world_size'], '两臂必须按共同配置使用相同world'
    acc = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=8 // world,
                      kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    torch.cuda.set_device(acc.local_process_index)
    args.device = str(acc.device)
    pipe, network, _ = load(args, core)
    data = dataset(pipe, core)
    style = VGGGramStyleLoss().to(args.device).eval().requires_grad_(False)
    folder = out / 'training' / config['arm']
    folder.mkdir(parents=True, exist_ok=True)
    run = TrainingRun(args, core, pipe, network, data, style, config,
                      read(config['split'])['train_indices'], acc)
    if (folder / 'latest.pt').exists():
        run.restore(folder / 'latest.pt')
    while run.step < config['updates']:
        row = run.update()
        if acc.is_main_process:
            with (folder / 'training.jsonl').open('a', encoding='utf-8') as stream:
                import json
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            if run.step % 25 == 0 or run.step < 4:
                print('TRAIN', config['arm'], run.step, row['losses'], flush=True)
        if run.step % 100 == 0 or run.step == config['updates']:
            run.save(folder / 'latest.pt')
    if acc.is_main_process:
        write(folder / 'complete.json', dict(status='completed', updates=run.step, config=config,
              environment=environment(), checkpoint_sha256=sha(folder / 'latest.pt')))
