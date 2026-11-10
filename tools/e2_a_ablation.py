"""补齐A1-A4；A0/A5经核验复用；旧预算、随机流、指标不变。"""
import argparse
import gc
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np
import torch

from models.e2_compat import compute_losses, load_trainable_strict
from models.e2_structure_ablation import attach_ablation, architecture, ARMS, NEW_ARMS, VERSION
from tools.e2_fft_experiment import SOURCE, setup, load, read, write, sha, environment
from tools.e2_fft_training import TrainingRun, dataset, tensor_hash, optimizer_hash, frozen_hash
from tools.e2_a_minimal import ATrainingRun, mask_snapshot

MATCH_FIELDS = ['initialization', 'initialization_sha256', 'updates', 'effective_batch', 'world_size',
    'learning_rate', 'warmup', 'workers', 'train_seed', 'sampler_seed', 'noise_seed_per_rank', 'precision',
    'prediction', 'old_losses', 'deterministic_algorithms', 'sdp_backends', 'training_input_map_sha256', 'split_sha256']


def a_environment():
    value = environment()
    for name in ['models/e2_structure.py', 'models/e2_structure_ablation.py', 'tools/e2_a_minimal.py',
        'tools/e2_a_evaluation.py', 'tools/e2_a_ablation.py', 'tools/e2_a_ablation_evaluation.py',
        'scripts/e2_a_ablation.sbatch', 'tests/test_e2_a_ablation.py']:
        value['source'][name] = sha(SOURCE / name)
    import PIL, scipy, skimage, cv2
    value['image_libraries'] = dict(numpy=np.__version__, Pillow=PIL.__version__, scipy=scipy.__version__,
                                   skimage=skimage.__version__, opencv=cv2.__version__)
    return value


def configuration(args, arm):
    cfg = dict(read(Path(args.a_parent) / 'comparison/A5.json'))
    cfg.update(arm=arm, structure_mode=arm, code_commit=environment()['commit'],
        protocol='A_complete_ablation_v1_exact268256', ablation_version=VERSION,
        a_pre='canny40_120' if arm == 'A2' else 'original_sketch',
        structure_width=None if arm in ('A1','A2') else 32,
        single_widths=[24,48,64] if arm in ('A1','A2') else None,
        metrics_version='A_structure_v1+v3_texture',
        split=str(Path(args.out) / 'split.json'), split_sha256=sha(Path(args.out) / 'split.json'),
        architecture=architecture(), added_parameters=268256, a_support=False, B=False, C=False)
    return cfg


def prepare(args):
    from tools import e2_a_minimal as legacy
    core, out = setup(args); parent = Path(args.a_parent)
    # 复用已验证全量输入/旧mask审计及A0复制，不改变其实现。
    legacy.prepare(args)
    for name in ['training_inputs.json', 'mask_inputs.json']:
        assert sha(out / 'audit' / name) == sha(parent / 'audit' / name)
    assert read(parent / 'audit/training_inputs_final.json')['passed']
    assert read(parent / 'audit/mask_inputs_final.json')['passed']
    complete5 = read(parent / 'training/A5/complete.json')
    critical = ['models/e2_structure.py', 'models/e2_compat.py', 'models/bf_original_fusion.py',
        'models/bf_sd_skip.py', 'train_GAM_texture_joint.py', 'tools/bf_t3_frozen_core.py', 'tools/e2_fft_training.py']
    assert all(sha(SOURCE / n) == complete5['environment']['source'][n] for n in critical)
    checkpoints = dict(A0=str(Path(args.baseline) / 'training/B0/latest.pt'),
                       A5=str(parent / 'training/A5/latest.pt'))
    records = []
    for arm in ['A0', 'A5']:
        complete = read(parent / 'training' / arm / 'complete.json')
        assert complete['status'] == 'completed' and complete['updates'] == 2000
        assert sha(checkpoints[arm]) == complete['checkpoint_sha256']
        target = out / 'training' / arm; target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent / 'training' / arm / 'training.jsonl', target / 'training.jsonl')
        write(target / 'complete.json', dict(complete, reused_from=str(parent / 'training' / arm)))
        write(out / 'comparison' / (arm + '.json'), complete['config'])
        for pair in read(out / 'strict_pairs.json')['pairs']:
            for condition in ['old', 'new']:
                src = parent / 'outputs' / arm / pair['case_id'] / (condition + '.png')
                meta = read(src.with_suffix('.json'))
                assert meta['output_sha256'] == sha(src) and meta['checkpoint_sha256'] == complete['checkpoint_sha256']
                assert meta['manifest_sha256'] == sha(out / 'strict_pairs.json')
                dest = out / 'outputs' / arm / pair['case_id'] / src.name; dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
                meta.update(parent_metadata_sha256=sha(src.with_suffix('.json')), parent_output=str(src), reused_in_run='A_complete')
                write(dest.with_suffix('.json'), meta)
                records.append(dict(arm=arm, file=str(src), output_sha256=sha(src), metadata_sha256=sha(src.with_suffix('.json'))))
    write(out / 'audit/reuse_A0_A5.json', dict(prepared=True, checkpoints=checkpoints, records=records,
        critical_source_hashes={n:sha(SOURCE / n) for n in critical},
        required_gates='new4 alloff+zeroon,old4 B0+old4 A5 update replay,A5 8PNG pixel regression'))
    for arm in NEW_ARMS: write(out / 'comparison' / (arm + '.json'), configuration(args, arm))
    cfg0 = read(out / 'training/A0/complete.json')['config']
    cfg5 = read(out / 'training/A5/complete.json')['config']
    for arm in ARMS[1:]:
        cfg = read(out / 'comparison' / (arm + '.json'))
        assert all(cfg[k] == cfg0[k] == cfg5[k] for k in MATCH_FIELDS)
    write(out / 'comparison/variants.json', architecture())
    write(out / 'protocol.json', dict(version='A_complete_ablation_v1', arms=list(ARMS), new_training=list(NEW_ARMS),
        reused=['A0', 'A5'], training='common matureE2;2000updates/effectivebatch8/world1/seed42;oldlossunchanged',
        primary='old BoundaryF1 r2;5alternatives versusA0,Holm5', secondary='old EdgeF1/IoU/innerproxy,Holm15',
        texture_guard='unchangedv3;graystrict family5,other3 family15',
        mechanism_descriptive=['A2_minus_A1', 'A5_minus_A3', 'A5_minus_A4'],
        mechanism_note='paired differences and unadjustedCI only;no additional significance claims',
        architecture=architecture(), a_support=False, B=False, C=False,
        old_pairwise_statistics='prior A0/A5 report preserved;new sixarm multiplicity recalculated',
        limitations='single trainingseed;32texturefocused targets;no human semantic line labels;upstream exposure unknown;historical fullmaskSHA unavailable',
        environment=a_environment()))
    print('A_ABLATION_PREPARE_PASS', flush=True)


def load_arm(args, core):
    pipe, network, checked = load(args, core)
    attach_ablation(network, args.arm)
    return pipe, network, checked


def regression(args):
    from tools import e2_a_minimal as legacy
    from tools.e2_a_ablation_evaluation import image_once
    core, out = setup(args); combined = {}
    for arm in NEW_ARMS:
        with patch.object(legacy, 'attach_structure', lambda n, **kw: attach_ablation(n, arm, enabled=kw.get('enabled', True))), \
             patch.object(legacy, 'a_environment', a_environment):
            legacy.regression(args)
        folder = out / 'regression' / arm; folder.mkdir(parents=True, exist_ok=True)
        for name in ['forward_result.json', 'image_result.json', 'alloff', 'zeroon']:
            dest = folder / name
            # 失败重试时保留先前证据，禁止静默覆盖。
            if dest.exists(): raise RuntimeError('已有回归证据，先核验而非覆盖: ' + str(dest))
            shutil.move(str(out / 'regression' / name), str(dest))
        combined[arm] = dict(forward=read(folder / 'forward_result.json'), image=read(folder / 'image_result.json'))
        gc.collect(); torch.cuda.empty_cache()
    pipe, network, _ = load_arm(arg_arm(args, 'A5'), core)
    state = torch.load(Path(args.a_parent) / 'training/A5/latest.pt', map_location='cpu', weights_only=False)
    load_trainable_strict(network, state['network']); del state
    network.eval().requires_grad_(False); pipe.unet.half(); pipe.vae.half(); network.adapter.half()
    pipe.image_encoder.to(args.device); pipe.bf_texture_conditioner.to(args.device)
    records = []
    from PIL import Image
    with torch.inference_mode():
        for pair in read(out / 'strict_pairs.json')['pairs'][:4]:
            for condition in ['old', 'new']:
                image, latent_sha = image_once(args, core, pipe, network, pair, condition)
                old = out / 'outputs/A5' / pair['case_id'] / (condition + '.png')
                error = int(np.abs(np.asarray(image).astype(int) - np.asarray(Image.open(old).convert('RGB')).astype(int)).max())
                path = out / 'regression/A5_reuse' / pair['case_id'] / (condition + '.png')
                path.parent.mkdir(parents=True, exist_ok=True); image.save(path)
                assert error == 0 and latent_sha == read(old.with_suffix('.json'))['initial_latent_sha256']
                records.append(dict(case_id=pair['case_id'], condition=condition, max_pixel_difference=error, latent_equal=True))
    write(out / 'regression/A5_reuse.json', dict(passed=True, records=records))
    write(out / 'regression/forward_result.json', dict(passed=True, arms={a:x['forward'] for a,x in combined.items()}))
    write(out / 'regression/image_result.json', dict(passed=True, arms={a:x['image'] for a,x in combined.items()}, A5_reuse=records))
    print('A_ABLATION_REGRESSION_PASS', flush=True)


def arg_arm(args, arm):
    import copy
    value = copy.copy(args); value.arm = arm
    return value


def reset_network(network, initial, arm, enabled=True):
    adapter = attach_ablation(network, arm, enabled)
    state = network.trainable_state(); state.update(initial)
    load_trainable_strict(network, state)
    return adapter


def smoke(args):
    from accelerate import Accelerator
    from train_GAM_texture_joint import VGGGramStyleLoss
    core, out = setup(args)
    assert read(out / 'regression/forward_result.json')['passed'] and read(out / 'regression/image_result.json')['passed']
    pipe, network, _ = load(args, core); initial = network.trainable_state()
    style = VGGGramStyleLoss().to(args.device).eval().requires_grad_(False); data = dataset(pipe, core)
    acc = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=8); assert acc.num_processes == 1
    for arm in ['A0', 'A5']:
        adapter = reset_network(network, initial, 'A5', enabled=arm == 'A5')
        if arm == 'A0': adapter.structure.requires_grad_(False)
        cfg = read(out / 'training' / arm / 'complete.json')['config']
        run = ATrainingRun(args, core, pipe, network, data, style, cfg, read(out / 'split.json')['train_indices'], acc)
        old = [json.loads(s) for s in (out / 'training' / arm / 'training.jsonl').read_text().splitlines()][:4]
        records = []
        for expected in old:
            got = run.update()
            assert got['tokens'] == expected['tokens'] and got['losses'] == expected['losses'] and got['grad_norm'] == expected['grad_norm']
            records.append(dict(step=got['step'], tokens_equal=True, losses_equal=True, grad_norm_equal=True))
        write(out / 'audit' / ('baseline_replay.json' if arm == 'A0' else 'A5_replay.json'), dict(passed=True, records=records))
        network = acc.unwrap_model(run.network, keep_fp32_wrapper=False); del run; acc.free_memory()
    results = {}
    for arm in NEW_ARMS:
        adapter = reset_network(network, initial, arm)
        before, frozen = network.trainable_state(), frozen_hash(pipe, network)
        cfg = dict(read(out / 'comparison' / (arm + '.json')), updates=4, warmup=0, workers=0, diagnostics=True)
        run = ATrainingRun(args, core, pipe, network, data, style, cfg,
            [x['index'] for x in read(out / 'split.json')['smoke']], acc)
        logs = [run.update(), run.update()]; folder = out / 'smoke' / arm; folder.mkdir(parents=True, exist_ok=True)
        run.save(folder / 'resume.pt'); expected = run.update()
        expected_state = acc.unwrap_model(run.network).trainable_state(); opt_hash = optimizer_hash(run.optimizer)
        run.restore(folder / 'resume.pt'); actual = run.update()
        assert tensor_hash(expected_state.items()) == tensor_hash(acc.unwrap_model(run.network).trainable_state().items())
        assert opt_hash == optimizer_hash(run.optimizer)
        assert actual['tokens'] == expected['tokens'] and actual['losses'] == expected['losses'] and actual['structure'] == expected['structure']
        logs.extend([actual, run.update()]); network = acc.unwrap_model(run.network)
        changes = {}
        for name, value in network.trainable_state().items():
            group = name.split('.')[2] if name.startswith('adapter.structure.') else 'old_E2'
            changes[group] = changes.get(group, 0.) + float((value.float() - before[name].float()).square().sum())
        assert all(np.isfinite(v) and v > 0 for v in changes.values()) and frozen_hash(pipe, network) == frozen
        batch = {k:v[None].to(args.device) for k,v in data[read(out / 'split.json')['smoke'][0]['index']].items()}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, raw, _, captured, _ = compute_losses(pipe, network, batch,
                torch.Generator(device=args.device).manual_seed(42000), style, core, audit_mode=True)
            image_loss = raw['gram'] + .05*raw['edge'] + .1*raw['color'] + .1*raw['body'] + .03816244801608925*raw['rest']
        named = list(network.adapter.structure.named_parameters())
        gradients = torch.autograd.grad(image_loss, [captured['epsilon'], captured['vae_input']] + [p for _,p in named], allow_unused=True)
        norms = {n:float(g.float().norm()) if g is not None else 0. for n,g in zip(['epsilon','vae_input']+[n for n,_ in named], gradients)}
        assert all(np.isfinite(v) and v > 0 for v in norms.values()), norms
        if arm == 'A4':
            grad = gradients[2 + [n for n,_ in named].index('outer.0.weight')]
            assert torch.all(grad.abs().sum((0,2,3)) > 0), 'A4首层仍有无输入的占位列'
        results[arm] = dict(passed=True, updates=4, executed_update_calls=5, resume_exact=True,
            frozen_unchanged=True, added_parameters=sum(p.numel() for _,p in named),
            changes_squared_l2=changes, all_parameter_image_gradient_norms=norms, logs=logs)
        write(folder / 'result.json', results[arm])
        network = acc.unwrap_model(run.network, keep_fp32_wrapper=False)
        del run, before, expected_state, batch, captured, raw, image_loss, gradients, named; acc.free_memory()
        print('A_ABLATION_SMOKE_PASS', arm, flush=True)
    write(out / 'smoke/result.json', dict(passed=True, arms=results, environment=a_environment()))


def train(args):
    from accelerate import Accelerator
    from train_GAM_texture_joint import VGGGramStyleLoss
    from tools.e2_b_screen_training import assert_snapshot
    core, out = setup(args)
    assert args.arm in NEW_ARMS and read(out / 'smoke/result.json')['passed']
    assert read(out / 'audit/baseline_replay.json')['passed'] and read(out / 'audit/A5_replay.json')['passed']
    cfg = read(out / 'comparison' / (args.arm + '.json'))
    assert cfg['code_commit'] == environment()['commit'] and assert_snapshot(args) == cfg['training_input_map_sha256']
    assert sha(out / 'split.json') == cfg['split_sha256']
    acc = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=8); assert acc.num_processes == 1
    pipe, network, _ = load_arm(args, core); data = dataset(pipe, core)
    style = VGGGramStyleLoss().to(args.device).eval().requires_grad_(False)
    folder = out / 'training' / args.arm; folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'complete.json').exists():
        value = read(folder / 'complete.json')
        assert value['config'] == cfg and sha(folder / 'latest.pt') == value['checkpoint_sha256']
        print('A_ABLATION_ALREADY_COMPLETED', args.arm); return
    run = ATrainingRun(args, core, pipe, network, data, style, cfg, read(out / 'split.json')['train_indices'], acc)
    if (folder / 'latest.pt').exists(): run.restore(folder / 'latest.pt')
    log = folder / 'training.jsonl'
    if log.exists():
        rows = [json.loads(s) for s in log.read_text().splitlines()]
        log.write_text(''.join(json.dumps(x)+'\n' for x in rows if x['step'] <= run.step))
    baseline = [json.loads(s) for s in (out / 'training/A0/training.jsonl').read_text().splitlines()]
    while run.step < cfg['updates']:
        row = run.update(); assert len(row['structure']) == 8 and row['tokens'] == baseline[row['step']-1]['tokens']
        with log.open('a', encoding='utf8') as stream: stream.write(json.dumps(row, allow_nan=False)+'\n')
        if run.step <= 4 or run.step % 25 == 0: print('A_ABLATION_TRAIN', args.arm, run.step, row['losses'], flush=True)
        if run.step % 100 == 0: run.save(folder / 'latest.pt')
    write(folder / 'complete.json', dict(status='completed', updates=run.step, config=cfg,
        environment=a_environment(), checkpoint_sha256=sha(folder / 'latest.pt')))
    print('A_ABLATION_TRAIN_COMPLETED', args.arm, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare','regression','smoke','train','audit-final','generate','evaluate','bundle'])
    p.add_argument('--project', default='/share/home/u2515283058/Mymodel')
    p.add_argument('--baseline', default='/share/home/u2515283058/Mymodel/output_eval/e2_b_screen_20261108')
    p.add_argument('--a-parent', default='/share/home/u2515283058/Mymodel/output_eval/e2_a_minimal_20261109')
    p.add_argument('--out', default='/share/home/u2515283058/Mymodel/output_eval/e2_a_ablation_20261110')
    p.add_argument('--data', default='/share/home/u2515283058/datasets/BF')
    p.add_argument('--candidates', default=str(SOURCE / 'configs/e2_b_candidates.json'))
    p.add_argument('--device', default='cuda:0'); p.add_argument('--arm', choices=ARMS, default='A1')
    args = p.parse_args(); args.parent = args.baseline
    if args.action in ['generate','evaluate','bundle']:
        from tools import e2_a_ablation_evaluation
        getattr(e2_a_ablation_evaluation, args.action)(args)
    elif args.action == 'audit-final':
        from tools.e2_b_screen_training import audit_training
        audit_training(args, final=True); core, _ = setup(args); mask_snapshot(args, core, final=True)
    else: globals()[args.action](args)


if __name__ == '__main__': main()
