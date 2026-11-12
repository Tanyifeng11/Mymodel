"""固定逐步读取器，仅比较原损失与新增局部参考功率谱损失。"""
import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw
import torch

from models.e2_compat import compute_losses, e2_objective, load_trainable_strict
from models.e2_local_fft import LocalFFTLoss
from models.e2_texture_diagnostics import STEPS, TrajectoryTap
from models.unet_texture_reader import ATTENTION, attach_reader, encode_reference
from tools import bfm_skip_probe as probe
from tools import e2_b_screen_evaluation as legacy
from tools import unet_spatial_texture as spatial
from tools.e2_fft_experiment import setup, read, write, sha, environment
from tools.e2_fft_training import TrainingRun, dataset, tensor_hash
from tools.e2_texture_bottleneck import image, forward_at
from tools.m1_evaluation import prepare_inference
from tools.m1_residual_followup import copy_checked, comparison_row, summarize, panels, bundle
from tools.unet_texture_reader_diagnosis import attention_stats


RUN = probe.OUTPUT / 'reader_reference_loss_20261112'
ARMS = ('OriginalLoss', 'ReferenceSpectrum')
START_SHA = '5eb55141f4e5118d673958b3005bac0c5a49e7ab4b6f38b16b8ea6709dbb141f'


def prepare(args):
    core, out = setup(args)
    for name in ('strict_pairs.json', 'split.json', 'audit/training_inputs_final.json'):
        copy_checked(args.parent / name, out / name)
    source = read(args.parent / 'protocol.json')
    checkpoint = args.parent / 'training/Spatial/latest.pt'
    assert sha(checkpoint) == START_SHA
    assert sha(core.ROOT / 'training_manifest.json') == source['training_manifest_sha256']
    pairs = read(out / 'strict_pairs.json')['pairs']
    for pair in pairs:
        for key in ('sketch', 'gt', 'original_reference', 'swapped_reference'):
            assert sha(args.data / pair[key]) == pair['hashes'][key]
        if pair['case_id'] in source['controls']:
            for name, expected in source['controls'][pair['case_id']].items():
                target = probe.control_path(out, pair, name)
                copy_checked(probe.control_path(args.parent, pair, name), target)
                assert sha(target) == expected
    protocol = dict(arms=ARMS, updates=args.updates, effective_batch=8, world=2,
        learning_rate=1e-4, warmup=100, seed=42, start_checkpoint=str(checkpoint),
        start_sha256=START_SHA, base_sha256=probe.CHECKPOINT_SHA,
        training_manifest_sha256=source['training_manifest_sha256'],
        split_sha256=sha(out / 'split.json'), manifest_sha256=sha(out / 'strict_pairs.json'),
        reference_loss=dict(sizes=[64, 128], count=4, t_min=0, t_max=249, std_min=.02,
                            gradient_ratio_at_calibration=1.),
        controls=source['controls'], margins=source['margins'], sampler=source['sampler'],
        initial_state='same trained 600-step reader; fresh AdamW and common cosine schedule',
        trained_parameters='existing 209536 reader parameters only',
        loss_difference='ReferenceSpectrum adds calibrated local reference power-spectrum loss',
        environment=environment())
    if (out / 'protocol.json').exists():
        assert read(out / 'protocol.json') == json.loads(json.dumps(protocol))
    else:
        write(out / 'protocol.json', protocol)
    print('LOSS_PAIR_PREPARED', flush=True)


def load_start(args, core):
    pipe, network = spatial.load_base(args, core)
    reader = attach_reader(network, pipe.vae)
    path = Path(read(args.out / 'protocol.json')['start_checkpoint'])
    assert sha(path) == START_SHA
    state = torch.load(path, map_location='cpu', weights_only=False)
    load_trainable_strict(network, state['network'])
    del state
    network.eval()
    assert sum(p.numel() for p in network.parameters() if p.requires_grad) == 209536
    return pipe, network, reader


def preflight(args):
    core, out = setup(args)
    pipe, network, reader = load_start(args, core)
    initial = tensor_hash(network.trainable_state().items())
    prepare_inference(args, pipe, network)
    reader.half()
    pair = read(out / 'strict_pairs.json')['pairs'][0]
    with torch.inference_mode():
        reference = core.tensor(args.data / pair['original_reference']).to(args.device).half()
        sketch = core.tensor(args.data / pair['sketch']).to(reference)
        features = probe.projected(network, sketch, reference)
        with reader.inference(encode_reference(pipe.vae, reference)):
            value, latent = image(args, core, pipe, network, pair, features)
        previous = probe.paths(args.parent, 'Spatial', pair, 'old')
        assert np.array_equal(np.asarray(value), np.asarray(Image.open(previous).convert('RGB')))
        assert latent == read(previous.with_suffix('.json'))['initial_latent_sha256']
        assert reader.calls == 50
    write(out / 'audit/preflight.json', dict(passed=True, exact_start_replay=True,
          initial_sha256=initial, case_id=pair['case_id'], conditional_calls=reader.calls))
    print('LOSS_PAIR_PREFLIGHT_PASS', flush=True)


def spectral_loss(config, rank=0):
    return LocalFFTLoss(sizes=config['sizes'], count=config['count'], t_min=config['t_min'],
                        t_max=config['t_max'], std_min=config['std_min'], seed=110642 + rank)


def calibrate(args):
    from train_GAM_texture_joint import VGGGramStyleLoss

    core, out = setup(args)
    protocol = read(out / 'protocol.json')
    pipe, network, reader = load_start(args, core)
    style = VGGGramStyleLoss().to(args.device).eval().requires_grad_(False)
    data = dataset(pipe, core)
    extra = spectral_loss(protocol['reference_loss'])
    rng = torch.Generator(device=args.device).manual_seed(91112)
    records, ratios = [], []
    for choice in read(out / 'split.json')['smoke']:
        item = data[choice['index']]
        batch = {key: value[None].to(args.device) for key, value in item.items()}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, raw, token, captured, stats = compute_losses(pipe, network, batch, rng, style, core,
                                                           extra, 1., audit_mode=True)
        if stats['valid_n']:
            old = torch.autograd.grad(e2_objective(raw), reader.output.weight, retain_graph=True)[0].float()
            new = torch.autograd.grad(raw['fft'], reader.output.weight, retain_graph=True)[0].float()
            edge = torch.autograd.grad(.05 * raw['edge'], reader.output.weight)[0].float()
            old_norm, new_norm = float(old.norm()), float(new.norm())
            assert old_norm > 0 and new_norm > 0 and np.isfinite([old_norm, new_norm]).all()
            ratios.append(old_norm / new_norm)
            cosine = lambda a, b: float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-20))
            records.append(dict(index=choice['index'], pattern=choice['pattern_candidate'], token=token,
                old_gradient=old_norm, fft_gradient=new_norm, edge_gradient=float(edge.norm()),
                old_fft_cosine=cosine(old, new), edge_fft_cosine=cosine(edge, new),
                fft_loss=float(raw['fft']), stats=stats))
        network.adapter.clear()
        del raw, captured, batch
    assert ratios, '训练校准样本中缺少有效纹理梯度'
    weight = float(np.median(ratios)) * protocol['reference_loss']['gradient_ratio_at_calibration']
    write(out / 'calibration/result.json', dict(passed=True, weight=weight, records=records,
          initial_sha256=tensor_hash(network.trainable_state().items()), target='reader output projection gradient',
          initial_old_fft_norm_ratio=protocol['reference_loss']['gradient_ratio_at_calibration']))
    print('LOSS_PAIR_WEIGHT', weight, len(records), flush=True)


def train(args):
    from accelerate import Accelerator
    from train_GAM_texture_joint import VGGGramStyleLoss

    core, out = setup(args)
    protocol = read(out / 'protocol.json')
    calibration = read(out / 'calibration/result.json')
    assert calibration['passed'] and read(out / 'audit/preflight.json')['passed']
    world = int(os.environ.get('WORLD_SIZE', '1'))
    assert world == protocol['world']
    acc = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=8 // world)
    torch.cuda.set_device(acc.local_process_index)
    args.device = str(acc.device)
    pipe, network, reader = load_start(args, core)
    initial = tensor_hash(network.trainable_state().items())
    assert initial == calibration['initial_sha256'] == read(out / 'audit/preflight.json')['initial_sha256']
    frozen = tensor_hash((n, p) for n, p in network.named_parameters() if not p.requires_grad)
    cfg = protocol['reference_loss']
    config = dict(learning_rate=protocol['learning_rate'], warmup=protocol['warmup'], updates=protocol['updates'],
        workers=2, fft=args.arm == ARMS[1], fft_weight=calibration['weight'] if args.arm == ARMS[1] else 0.,
        fft_sizes=cfg['sizes'], fft_count=cfg['count'], fft_t_min=cfg['t_min'],
        fft_t_max=cfg['t_max'], fft_std_min=cfg['std_min'])
    style = VGGGramStyleLoss().to(args.device).eval().requires_grad_(False)
    run = TrainingRun(args, core, pipe, network, dataset(pipe, core), style, config,
                      read(out / 'split.json')['train_indices'], acc)
    network.eval()
    reader.train()
    folder = out / 'training' / args.arm
    folder.mkdir(parents=True, exist_ok=True)
    latest = folder / 'latest.pt'
    history = folder / 'training.jsonl'
    if latest.exists():
        run.restore(latest)
    if acc.is_main_process and history.exists():
        history.write_text(''.join(line + '\n' for line in history.read_text().splitlines()
                                   if json.loads(line)['step'] <= run.step))
    while run.step < config['updates']:
        gradients, handles = {}, []
        if run.step < 3 or (run.step + 1) % 100 == 0:
            def capture(name):
                def record(value):
                    gradients[name] = gradients.get(name, 0.) + float(value.detach().float().square().sum())
                return record
            handles = [p.register_hook(capture(name)) for name, p in reader.named_parameters()]
        row = run.update()
        for handle in handles:
            handle.remove()
        if gradients:
            row['reader_gradient_norms'] = {name: value ** .5 for name, value in gradients.items()}
        assert reader.calls == 1
        if acc.is_main_process:
            with history.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            if run.step <= 3 or run.step % 20 == 0:
                write(folder / 'progress.json', row)
                print('LOSS_PAIR_TRAIN', args.arm, run.step, row['losses'], row['seconds'], flush=True)
        if run.step % 100 == 0 or run.step == config['updates']:
            run.save(latest)
            if acc.is_main_process and run.step in (600, 1200, config['updates']):
                shutil.copy2(latest, folder / ('step%04d.pt' % run.step))
    unchanged = frozen == tensor_hash((n, p) for n, p in network.named_parameters() if not p.requires_grad)
    assert unchanged
    if acc.is_main_process:
        write(folder / 'complete.json', dict(status='completed', updates=run.step, world=world,
              checkpoint_sha256=sha(latest), initial_sha256=initial, frozen_unchanged=True, config=config,
              protocol_sha256=sha(out / 'protocol.json'), calibration_sha256=sha(out / 'calibration/result.json')))
    acc.wait_for_everyone()


def audit_training(out):
    protocol = read(out / 'protocol.json')
    logs, states, complete = [], [], []
    for arm in ARMS:
        folder = out / 'training' / arm
        record = read(folder / 'complete.json')
        rows = [json.loads(line) for line in (folder / 'training.jsonl').read_text().splitlines()]
        assert record['status'] == 'completed' and record['updates'] == protocol['updates'] and record['frozen_unchanged']
        assert sha(folder / 'latest.pt') == record['checkpoint_sha256']
        assert [r['step'] for r in rows] == list(range(1, protocol['updates'] + 1))
        assert all(np.isfinite(r['grad_norm']) and all(np.isfinite(v) for v in r['losses'].values()) for r in rows)
        logs.append([(r['tokens'], r['chain'], r['lr']) for r in rows])
        checkpoint = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=False)
        states.append([(s['chain'], s['stream'], s['rng']) for s in checkpoint['states']])
        complete.append(record)
    assert logs[0] == logs[1] and complete[0]['initial_sha256'] == complete[1]['initial_sha256']
    for a, b in zip(states[0], states[1]):
        assert a[:2] == b[:2] and torch.equal(a[2], b[2])
    ignored = {'fft', 'fft_weight'}
    assert {k: v for k, v in complete[0]['config'].items() if k not in ignored} == {
            k: v for k, v in complete[1]['config'].items() if k not in ignored}
    second = [json.loads(line) for line in (out / 'training' / ARMS[1] / 'training.jsonl').read_text().splitlines()]
    assert sum(r['patch_counts']['valid_n'] for r in second) > 0
    write(out / 'audit/training_completion.json', dict(passed=True, identical_logged_tokens_and_learning_rates=True,
          identical_all_rank_streams_and_rng=True, identical_initial_weights=True, only_loss_differs=True,
          arms=dict(zip(ARMS, complete))))


def load_final(args, core):
    pipe, network, reader = load_start(args, core)
    folder = args.out / 'training' / args.arm
    complete = read(folder / 'complete.json')
    assert complete['frozen_unchanged'] and sha(folder / 'latest.pt') == complete['checkpoint_sha256']
    state = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=False)
    load_trainable_strict(network, state['network'])
    del state
    prepare_inference(args, pipe, network)
    reader.half()
    return pipe, network, reader, complete['checkpoint_sha256']


def generate(args):
    core, out = setup(args)
    protocol = read(out / 'protocol.json')
    pipe, network, reader, checkpoint_sha = load_final(args, core)
    pairs = read(out / 'strict_pairs.json')['pairs']
    with torch.inference_mode():
        for pair in pairs:
            sketch = core.tensor(args.data / pair['sketch']).to(args.device).half()
            latent = read(probe.paths(args.parent, 'M0', pair, 'old').with_suffix('.json'))['initial_latent_sha256']
            jobs = [(c, args.data / pair[k], pair['hashes'][k], out)
                    for c, k in [('old', 'original_reference'), ('new', 'swapped_reference')]]
            if pair['case_id'] in protocol['controls']:
                jobs += [(c, probe.control_path(out, pair, c), expected, out / 'synthetic')
                         for c, expected in protocol['controls'][pair['case_id']].items()]
            for condition, path, expected, folder in jobs:
                reference = core.tensor(path).to(sketch)
                features = probe.projected(network, sketch, reference)
                spatial.render(args, core, pipe, network, reader, pair, args.arm, condition, path, expected,
                               folder, sketch, features, encode_reference(pipe.vae, reference), latent, checkpoint_sha)
            print('LOSS_PAIR_GENERATED', args.arm, pair['case_id'], flush=True)
    write(out / 'generation' / (args.arm + '.json'), dict(passed=True, checkpoint_sha256=checkpoint_sha,
          identities=[p['case_id'] for p in pairs], outputs=64 + 3 * len(protocol['controls'])))


def controlled_metrics(out, pairs):
    values, rows = {}, []
    destination = out / 'synthetic/metrics/panels'
    destination.mkdir(parents=True, exist_ok=True)
    for pair in pairs:
        canvas = Image.new('RGB', (480, 705), 'white')
        draw = ImageDraw.Draw(canvas)
        for i, condition in enumerate(spatial.CONTROLS):
            tiles = [('Reference', probe.control_path(out, pair, condition))]
            for arm in ARMS:
                path = probe.paths(out / 'synthetic', arm, pair, condition)
                meta = read(path.with_suffix('.json'))
                assert sha(path) == meta['output_sha256']
                assert meta['texture_reads'] == 50 and meta['input_reference_sha256'] == sha(tiles[0][1])
                pixels = np.asarray(Image.open(path).convert('RGB'))
                bands = [probe.patch_bands(pixels[y:y+64, x:x+64]) for x, y in pair['patch_coordinates']]
                means = {name: float(np.mean([b[name] for b in bands])) for name in spatial.CONTROLS}
                values[arm, pair['case_id'], condition] = means
                rows.append(dict(arm=arm, case=pair['case_id'], condition=condition, bands=means))
                tiles.append((arm, path))
            for j, (name, path) in enumerate(tiles):
                draw.text((j * 160 + 3, i * 235 + 3), name + ' ' + condition, fill='black')
                with Image.open(path) as source:
                    canvas.paste(source.convert('RGB').resize((160, 213)), (j * 160, i * 235 + 19))
        canvas.save(destination / (pair['case_id'] + '.jpg'), quality=92)
    comparisons = []
    for condition in ('h16', 'v32'):
        scores, switches = {}, {}
        for arm in ARMS:
            a = [(values[arm, p['case_id'], 'v16'], values[arm, p['case_id'], condition]) for p in pairs]
            scores[arm] = [(b[condition] - b['v16']) - (x[condition] - x['v16']) for x, b in a]
            switches[arm] = sum(probe.follow(x, 'v16', condition) and probe.follow(b, condition, 'v16') for x, b in a)
        row = comparison_row(ARMS[1], ARMS[0], 'controlled', condition, 'target_band_margin',
                             list(zip(scores[ARMS[1]], scores[ARMS[0]])))
        comparisons.append(dict(row, candidate_switches=switches[ARMS[1]], baseline_switches=switches[ARMS[0]]))
    write(out / 'synthetic/metrics/per_image.json', rows)
    write(out / 'synthetic/metrics/comparisons.json', comparisons)


def mechanisms(args, core, arm):
    args.arm = arm
    pipe, network, reader, checkpoint_sha = load_final(args, core)
    protocol = read(args.out / 'protocol.json')
    pairs = [p for p in read(args.out / 'strict_pairs.json')['pairs'] if p['case_id'] in list(protocol['controls'])[:2]]
    records = []
    with torch.inference_mode():
        for pair in pairs:
            reference = core.tensor(probe.control_path(args.out, pair, 'v16')).to(args.device).half()
            sketch = core.tensor(args.data / pair['sketch']).to(reference)
            features = probe.projected(network, sketch, reference)
            with TrajectoryTap(pipe, network.adapter) as trajectory:
                image(args, core, pipe, network, pair, features)
            for step in STEPS:
                captured = []
                def capture(_module, inputs, kwargs, output):
                    if network.adapter.conditional:
                        captured.append((inputs[0] if inputs else kwargs['hidden_states'], output))
                handle = network.unet.get_submodule(ATTENTION).register_forward_hook(capture, with_kwargs=True)
                forward_at(pipe, network, trajectory.states[step], features)
                handle.remove()
                hidden, original = captured[0]
                for condition in spatial.CONTROLS:
                    ref = core.tensor(probe.control_path(args.out, pair, condition)).to(reference)
                    stats = attention_stats(reader, hidden, original, encode_reference(pipe.vae, ref))
                    records.append(dict(case_id=pair['case_id'], step=step, reference=condition, **stats))
    write(args.out / 'mechanisms' / (arm + '.json'), dict(checkpoint_sha256=checkpoint_sha, records=records))
    del pipe, network, reader
    torch.cuda.empty_cache()


def evaluate(args):
    core, out = setup(args)
    audit_training(out)
    protocol = read(out / 'protocol.json')
    pairs = read(out / 'strict_pairs.json')['pairs']
    for arm in ARMS:
        record = read(out / 'generation' / (arm + '.json'))
        assert record['passed'] and len(record['identities']) == 32 and record['outputs'] == 88
    legacy.evaluate(args, arms=ARMS, audit=lambda _: None)
    summarize(args, pairs, [(ARMS[1], ARMS[0])], protocol['margins'], include_historical_failures=False)
    controls = [p for p in pairs if p['case_id'] in protocol['controls']]
    controlled_metrics(out, controls)
    panels(args, controls, ARMS)
    for arm in ARMS:
        mechanisms(args, core, arm)
    bundle(out, 'reader_reference_loss_results.zip', (
        out / 'protocol.json', out / 'strict_pairs.json', out / 'audit', out / 'calibration',
        out / 'metrics', out / 'synthetic/metrics', out / 'mechanisms', out / 'generation',
        *(out / 'training' / arm / name for arm in ARMS for name in ('complete.json', 'training.jsonl'))))
    print('LOSS_PAIR_REVIEW_COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'preflight', 'calibrate', 'train', 'generate', 'evaluate'))
    parser.add_argument('--arm', choices=ARMS, default=ARMS[0])
    parser.add_argument('--project', type=Path, default=probe.PROJECT)
    parser.add_argument('--previous', type=Path, default=probe.PREVIOUS)
    parser.add_argument('--parent', type=Path, default=spatial.RUN)
    parser.add_argument('--out', type=Path, default=RUN)
    parser.add_argument('--data', type=Path, default=probe.PROJECT.parent / 'datasets/BF')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--updates', type=int, default=2000)
    args = parser.parse_args()
    globals()[args.action](args)


if __name__ == '__main__':
    main()
