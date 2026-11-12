"""短训并评测 M0 的逐步参考纹理注意力。"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from models.e2_compat import load_trainable_strict
from models.e2_texture_diagnostics import reference_control
from models.unet_texture_reader import ATTENTION, attach_reader, encode_reference
from tools import bfm_skip_probe as probe
from tools import e2_b_screen_evaluation as legacy
from tools.e2_fft_experiment import environment, read, setup, sha, write
from tools.e2_fft_training import TrainingRun, dataset, tensor_hash
from tools.e2_texture_bottleneck import image
from tools.m1_evaluation import prepare_inference
from tools.m1_experiment import PAIRS_SHA, load_m
from tools.m1_residual_followup import bundle, comparison_row, copy_checked, panels, summarize


RUN = probe.OUTPUT / 'unet_spatial_texture_20261112'
BASELINE = 'M0'
READER = 'Spatial'
SHUFFLED = 'Shuffled'
CONTROLS = ('v16', 'h16', 'v32')


def prepare(args):
    core, out = setup(args)
    copy_checked(args.previous / 'strict_pairs.json', out / 'strict_pairs.json')
    copy_checked(args.previous / 'audit/training_inputs_final.json', out / 'audit/training_inputs_final.json')
    copy_checked(args.previous / 'split.json', out / 'split.json')
    assert read(out / 'audit/training_inputs_final.json')['passed']
    assert sha(out / 'strict_pairs.json') == PAIRS_SHA
    pairs = read(out / 'strict_pairs.json')['pairs']
    assert len(pairs) == len({p['identity_group'] for p in pairs}) == 32
    split = read(out / 'split.json')
    manifest = core.ROOT / 'training_manifest.json'
    assert sha(manifest) == split['source_manifest_sha256']
    training_ids = {Path(row['cloth']).stem.split('_')[0] for row in read(manifest)}
    excluded = set(read(probe.RUN / 'protocol.json')['diagnostic_cases'])
    controls = {}
    for pair in pairs:
        for key in ('sketch', 'gt', 'original_reference', 'swapped_reference'):
            assert sha(args.data / pair[key]) == pair['hashes'][key]
        assert Path(pair['gt']).stem.split('_')[0] not in training_ids
        for condition in ('old', 'new'):
            probe.copy_output(args.previous, BASELINE, out, BASELINE, pair, condition)
        case = pair['case_id']
        if case in excluded or len(controls) == 8:
            continue
        with Image.open(args.data / pair['original_reference']) as source:
            rgb = np.asarray(source.convert('RGB').resize((384, 512), Image.Resampling.BILINEAR))
        if (rgb @ [.2126, .7152, .0722]).std() / 255. < .02:
            continue
        values = {c: reference_control(rgb, c, pair['seed']) for c in CONTROLS}
        if not all(all(probe.follow(probe.patch_bands(pixels[:64, :64]), c, other)
                       for other in CONTROLS if other != c) for c, pixels in values.items()):
            continue
        controls[case] = {}
        for c, pixels in values.items():
            path = probe.control_path(out, pair, c)
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pixels).save(path)
            controls[case][c] = sha(path)
    assert len(controls) == 8
    protocol = dict(steps=args.updates, batch=8, learning_rate=1e-4, warmup=30,
                    checkpoint_sha256=probe.CHECKPOINT_SHA, manifest_sha256=PAIRS_SHA,
                    training_manifest_sha256=sha(manifest), controls=controls,
                    sampler=dict(width=384, height=512, steps=50, cfg=7., sketch_scale=.6),
                    attention=ATTENTION, control_selection='first eight eligible cases outside previous skip probe',
                    training='paired real E2 objective; all M0 weights frozen; new reader only',
                    margins=read(args.previous / 'protocol.json')['inherited_exploratory_noninferiority'],
                    source=environment(),
                    implementation_sha256={name: sha(Path(__file__).resolve().parents[1] / name)
                        for name in ('models/unet_texture_reader.py', 'tools/unet_spatial_texture.py',
                                     'scripts/unet_spatial_texture.sbatch')})
    write(out / 'protocol.json', protocol)
    print('SPATIAL_PREPARED', list(controls), flush=True)


def load_base(args, core):
    path = args.previous / 'training/M0/latest.pt'
    assert sha(path) == probe.CHECKPOINT_SHA
    pipe, network, _ = load_m(args, core, BASELINE)
    state = torch.load(path, map_location='cpu', weights_only=False)
    load_trainable_strict(network, state['network'])
    network.eval().requires_grad_(False)
    return pipe, network


def replay(args, core, pipe, network, pair, reader=None):
    out = args.out
    reference = core.tensor(args.data / pair['original_reference']).to(args.device).half()
    sketch = core.tensor(args.data / pair['sketch']).to(reference)
    features = probe.projected(network, sketch, reference)
    original, latent = image(args, core, pipe, network, pair, features)
    if reader is not None:
        assert reader.calls == 0
        with reader.inference(encode_reference(pipe.vae, reference)):
            enabled, enabled_latent = image(args, core, pipe, network, pair, features)
        assert reader.calls == 50 and enabled_latent == latent
        assert np.array_equal(np.asarray(original), np.asarray(enabled)), 'zero residual changed output'
    source = probe.paths(out, BASELINE, pair, 'old')
    with Image.open(source) as expected:
        assert np.array_equal(np.asarray(original), np.asarray(expected)), 'historical M0 replay differs'
    assert latent == read(source.with_suffix('.json'))['initial_latent_sha256']


def preflight(args):
    core, out = setup(args)
    pipe, network = load_base(args, core)
    prepare_inference(args, pipe, network)
    reader = attach_reader(network, pipe.vae)
    pair = read(out / 'strict_pairs.json')['pairs'][0]
    with torch.inference_mode():
        replay(args, core, pipe, network, pair, reader)
    write(out / 'audit/preflight.json', dict(passed=True, case_id=pair['case_id'],
          exact_replay=True, zero_residual_exact=True, conditional_calls=reader.calls,
          parameters=sum(p.numel() for p in reader.parameters()), environment=environment()))
    print('SPATIAL_REPLAY_PASS', flush=True)


def train(args):
    from accelerate import Accelerator
    from train_GAM_texture_joint import VGGGramStyleLoss

    core, out = setup(args)
    assert read(out / 'audit/preflight.json')['passed']
    protocol = read(out / 'protocol.json')
    world = int(__import__('os').environ.get('WORLD_SIZE', '1'))
    acc = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=protocol['batch'] // world)
    torch.cuda.set_device(acc.local_process_index)
    args.device = str(acc.device)
    pipe, network = load_base(args, core)
    frozen = tensor_hash(network.named_parameters())
    reader = attach_reader(network, pipe.vae)
    style = VGGGramStyleLoss().to(acc.device).eval().requires_grad_(False)
    config = dict(learning_rate=protocol['learning_rate'], warmup=protocol['warmup'],
                  updates=protocol['steps'], fft=False, fft_weight=0., workers=2)
    run = TrainingRun(args, core, pipe, network, dataset(pipe, core), style, config,
                      read(out / 'split.json')['train_indices'], acc)
    network.eval()
    reader.train()
    folder = out / 'training' / READER
    folder.mkdir(parents=True, exist_ok=True)
    latest = folder / 'latest.pt'
    history = folder / 'training.jsonl'
    if latest.exists():
        run.restore(latest)
    if acc.is_main_process and history.exists():
        rows = [line for line in history.read_text().splitlines() if json.loads(line)['step'] <= run.step]
        history.write_text(''.join(line + '\n' for line in rows))
    while run.step < config['updates']:
        gradients, handles = {}, []
        if run.step < 3 or (run.step + 1) % 100 == 0:
            def capture(name):
                def record(gradient):
                    gradients[name] = gradients.get(name, 0.) + float(gradient.detach().float().square().sum())
                return record
            handles = [p.register_hook(capture(name)) for name, p in reader.named_parameters()]
        row = run.update()
        for handle in handles:
            handle.remove()
        if gradients:
            row['reader_gradient_norms'] = {name: value ** .5 for name, value in gradients.items()}
        assert reader.calls == 1
        if run.step <= 3 and not torch.distributed.is_initialized():
            print('SPATIAL_INITIAL_GRAD', run.step, row['grad_norm'], flush=True)
        if acc.is_main_process:
            with history.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            if run.step <= 3 or run.step % 20 == 0:
                write(folder / 'progress.json', row)
                print('SPATIAL_TRAIN', run.step, row['losses']['total'], row['seconds'], row['peak_gib'], flush=True)
        if run.step % 25 == 0 or run.step == config['updates']:
            run.save(latest)
    unchanged = tensor_hash((n, p) for n, p in network.named_parameters() if not p.requires_grad) == frozen
    assert unchanged and reader.output.weight.detach().abs().max().item() > 0
    if acc.is_main_process:
        write(folder / 'complete.json', dict(status='completed', checkpoint_sha256=sha(latest),
              base_sha256=probe.CHECKPOINT_SHA, updates=run.step, world=world,
              frozen_unchanged=unchanged, parameters=sum(p.numel() for p in reader.parameters())))
    acc.wait_for_everyone()


def render(args, core, pipe, network, reader, pair, arm, condition, ref_path, ref_sha, out,
           sketch, features, grid, latent_expected, checkpoint_sha):
    assert sha(ref_path) == ref_sha
    path = probe.paths(out, arm, pair, condition)
    if path.exists() and path.with_suffix('.json').exists():
        record = read(path.with_suffix('.json'))
        assert (record['output_sha256'] == sha(path) and record['checkpoint_sha256'] == checkpoint_sha
                and record['input_reference_sha256'] == ref_sha and record['initial_latent_sha256'] == latent_expected)
        assert record['texture_reads'] == (0 if arm == BASELINE else 50)
        return
    reader.calls = 0
    shuffle_seed = pair['seed'] + 1112 if arm == SHUFFLED else None
    context = nullcontext() if arm == BASELINE else reader.inference(grid, shuffle_seed)
    with context:
        value, latent = image(args, core, pipe, network, pair, features)
    assert latent == latent_expected and reader.calls == (0 if arm == BASELINE else 50)
    probe.save_output(out, arm, pair, condition, value, latent, ref_sha,
                      reader_checkpoint_sha256=None if arm == BASELINE else checkpoint_sha,
                      texture_reads=reader.calls, shuffle_seed=shuffle_seed)
    if arm != BASELINE:
        record = read(path.with_suffix('.json'))
        write(path.with_suffix('.json'), dict(record, checkpoint_sha256=checkpoint_sha))


def generate(args):
    core, out = setup(args)
    protocol = read(out / 'protocol.json')
    pairs = read(out / 'strict_pairs.json')['pairs'][args.shard::args.shards]
    complete = read(out / 'training' / READER / 'complete.json')
    assert complete['frozen_unchanged']
    path = out / 'training' / READER / 'latest.pt'
    checkpoint_sha = sha(path)
    assert checkpoint_sha == complete['checkpoint_sha256']
    pipe, network = load_base(args, core)
    prepare_inference(args, pipe, network)
    reader = attach_reader(network, pipe.vae)
    state = torch.load(path, map_location='cpu', weights_only=False)
    load_trainable_strict(network, state['network'])
    del state
    reader.eval().requires_grad_(False)
    with torch.inference_mode():
        replay(args, core, pipe, network, pairs[0])
        for pair in pairs:
            sketch_path = args.data / pair['sketch']
            assert sha(sketch_path) == pair['hashes']['sketch']
            sketch = core.tensor(sketch_path).to(args.device).half()
            latent = read(probe.paths(out, BASELINE, pair, 'old').with_suffix('.json'))['initial_latent_sha256']
            jobs = [(name, args.data / pair[key], pair['hashes'][key], out, (READER,))
                    for name, key in [('old', 'original_reference'), ('new', 'swapped_reference')]]
            controls = protocol['controls'].get(pair['case_id'])
            if controls:
                jobs += [(c, probe.control_path(out, pair, c), controls[c], out / 'synthetic',
                          (BASELINE, READER, SHUFFLED)) for c in CONTROLS]
            for name, reference_path, ref_sha, folder, arms in jobs:
                reference = core.tensor(reference_path).to(sketch)
                features = probe.projected(network, sketch, reference)
                grid = encode_reference(pipe.vae, reference)
                for arm in arms:
                    render(args, core, pipe, network, reader, pair, arm, name, reference_path, ref_sha,
                           folder, sketch, features, grid, latent, probe.CHECKPOINT_SHA if arm == BASELINE else checkpoint_sha)
            print('SPATIAL_GENERATED', args.shard, pair['case_id'], bool(controls), flush=True)
    write(out / f'audit/shard_{args.shard}.json', dict(passed=True, cases=[p['case_id'] for p in pairs],
          checkpoint_sha256=checkpoint_sha, replay_case=pairs[0]['case_id'], environment=environment()))


def score_controls(args, pairs, protocol, checkpoint_sha):
    out = args.out
    arms = (BASELINE, READER, SHUFFLED)
    rows, results = [], []
    lookup = {}
    dest = out / 'synthetic/metrics/panels'
    dest.mkdir(parents=True, exist_ok=True)
    for pair in pairs:
        case = pair['case_id']
        expected_latent = read(probe.paths(out, BASELINE, pair, 'old').with_suffix('.json'))['initial_latent_sha256']
        canvas = Image.new('RGB', (4 * 160, 3 * 235), 'white')
        labels = ImageDraw.Draw(canvas)
        for row_index, condition in enumerate(CONTROLS):
            reference = probe.control_path(out, pair, condition)
            tiles = [('Reference', reference)]
            for arm in arms:
                path = probe.paths(out / 'synthetic', arm, pair, condition)
                meta = read(path.with_suffix('.json'))
                assert (meta['output_sha256'] == sha(path) and meta['initial_latent_sha256'] == expected_latent
                        and meta['texture_reads'] == (0 if arm == BASELINE else 50)
                        and meta['input_sketch_sha256'] == pair['hashes']['sketch']
                        and meta['input_reference_sha256'] == protocol['controls'][case][condition] == sha(reference)
                        and meta['checkpoint_sha256'] == (probe.CHECKPOINT_SHA if arm == BASELINE else checkpoint_sha))
                with Image.open(path) as source:
                    pixels = np.asarray(source.convert('RGB'))
                bands = [probe.patch_bands(pixels[y:y+64, x:x+64]) for x, y in pair['patch_coordinates']]
                means = {key: float(np.mean([b[key] for b in bands])) for key in CONTROLS}
                lookup[arm, case, condition] = means
                rows.append(dict(arm=arm, case=case, condition=condition, bands=means, patches=len(bands)))
                tiles.append((arm, path))
            for column, (name, path) in enumerate(tiles):
                x, y = 160 * column, 235 * row_index
                labels.text((x + 4, y + 4), f'{name} {condition}', fill='black')
                with Image.open(path) as source:
                    canvas.paste(source.convert('RGB').resize((160, 213)), (x, y + 19))
        canvas.save(dest / (case + '.jpg'), quality=92)

    def switching(arm, case, condition):
        anchor, changed = (lookup[arm, case, c] for c in ('v16', condition))
        score = (changed[condition] - changed['v16']) - (anchor[condition] - anchor['v16'])
        success = probe.follow(anchor, 'v16', condition) and probe.follow(changed, condition, 'v16')
        return score, success

    for condition in ('h16', 'v32'):
        values = {arm: [switching(arm, p['case_id'], condition) for p in pairs] for arm in arms}
        for arm, baseline in [(READER, BASELINE), (READER, SHUFFLED), (SHUFFLED, BASELINE)]:
            a, b = values[arm], values[baseline]
            result = comparison_row(arm, baseline, 'controlled', condition, 'target_band_margin',
                                    [(x[0], y[0]) for x, y in zip(a, b)])
            results.append(dict(result, candidate_switches=sum(x[1] for x in a), baseline_switches=sum(y[1] for y in b)))
    write(out / 'synthetic/metrics/per_image.json', rows)
    write(out / 'synthetic/metrics/comparisons.json', results)


def evaluate(args):
    _, out = setup(args)
    protocol = read(out / 'protocol.json')
    pairs = read(out / 'strict_pairs.json')['pairs']
    audits = [read(out / f'audit/shard_{i}.json') for i in range(args.shards)]
    complete = read(out / 'training' / READER / 'complete.json')
    assert all(a['passed'] and a['checkpoint_sha256'] == complete['checkpoint_sha256'] for a in audits)
    assert sorted(c for a in audits for c in a['cases']) == sorted(p['case_id'] for p in pairs)
    write(out / 'training/M0/complete.json', dict(checkpoint_sha256=probe.CHECKPOINT_SHA))
    controlled = [p for p in pairs if p['case_id'] in protocol['controls']]
    score_controls(args, controlled, protocol, complete['checkpoint_sha256'])
    legacy.evaluate(args, arms=(BASELINE, READER), audit=lambda _: None)
    summarize(args, pairs, [(READER, BASELINE)], protocol['margins'], include_historical_failures=False)
    panels(args, controlled, (BASELINE, READER))
    bundle(out, 'unet_spatial_texture_results.zip', (
        out / 'protocol.json', out / 'strict_pairs.json', out / 'audit', out / 'metrics', out / 'synthetic/metrics',
        out / 'training/Spatial/complete.json', out / 'training/Spatial/training.jsonl'))
    print('SPATIAL_REVIEW_COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'preflight', 'train', 'generate', 'evaluate'))
    parser.add_argument('--project', type=Path, default=probe.PROJECT)
    parser.add_argument('--previous', type=Path, default=probe.PREVIOUS)
    parser.add_argument('--data', type=Path, default=probe.PROJECT.parent / 'datasets/BF')
    parser.add_argument('--out', type=Path, default=RUN)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--updates', type=int, default=600)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=2)
    args = parser.parse_args()
    globals()[args.action](args)


if __name__ == '__main__':
    main()
