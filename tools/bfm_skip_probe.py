"""冻结 M0，比较参考插值，并隔离两个 skip 尺度与三个去噪区间。"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from models.e2_compat import load_trainable_strict
from models.e2_texture_diagnostics import reference_control
from texture_preprocess import preprocess_texture_image
from tools import e2_b_screen_evaluation as legacy
from tools.e2_fft_experiment import environment, read, setup, sha, write
from tools.e2_texture_bottleneck import image, weights_digest
from tools.m1_evaluation import prepare_inference
from tools.m1_experiment import PAIRS_SHA, load_m
from tools.m1_residual_followup import (
    EXPECTED_OLD, bundle, comparison_row, copy_checked, panels, summarize,
)

PROJECT = Path('/share/home/u2515283058/Mymodel')
OUTPUT = PROJECT / 'output_eval'
RUN = OUTPUT / 'bfm_skip_probe_20261112'
PREVIOUS = OUTPUT / 'm1_s2t_patch_20261111'
CHECKPOINT_SHA = EXPECTED_OLD['M0']
WINDOWS = ((0, 17), (17, 33), (33, 50))
CELLS = {f'S{scale}_{name}': (scale, *window)
         for scale in (0, 1) for name, window in zip(('early', 'mid', 'late'), WINDOWS)}
CONTROLS = ('v16', 'h16', 'v32', 'v16_shift')
SWAPS = ('h16', 'v32')
PATCH_TARGETS = {'v16': (0, 4), 'h16': (4, 0), 'v32': (0, 2)}
MIN_BAND_POWER = .075


def paths(out, arm, pair, condition):
    return out / 'outputs' / arm / pair['case_id'] / (condition + '.png')


def control_path(out, pair, name):
    return out / 'synthetic/inputs' / pair['case_id'] / (name + '.png')


def patch_bands(pixels):
    gray = np.asarray(pixels, dtype=np.float64) @ [.2126, .7152, .0722] / 255.
    if gray.std() < 2 / 255.:
        return dict.fromkeys(PATCH_TARGETS, 0.)
    window = np.outer(np.hanning(64), np.hanning(64))
    power = abs(np.fft.fft2((gray - gray.mean()) * window)) ** 2
    power[0, 0] = 0
    return {name: float((power[y, x] + power[-y, -x]) / max(power.sum(), 1e-12))
            for name, (y, x) in PATCH_TARGETS.items()}


def follow(scores, wanted, other):
    return scores[wanted] >= MIN_BAND_POWER and scores[wanted] > scores[other]


def prepare(args):
    _, out = setup(args)
    previous = args.previous
    copy_checked(previous / 'strict_pairs.json', out / 'strict_pairs.json')
    assert sha(out / 'strict_pairs.json') == PAIRS_SHA
    pairs = read(out / 'strict_pairs.json')['pairs']
    assert len(pairs) == len({p['identity_group'] for p in pairs}) == 32
    old_model = previous / 'training/M0'
    assert read(old_model / 'complete.json')['checkpoint_sha256'] == CHECKPOINT_SHA
    assert sha(old_model / 'latest.pt') == CHECKPOINT_SHA
    source_audit = previous / 'audit/training_inputs_final.json'
    assert read(source_audit)['passed']
    for directory in (out, out / 'real_window'):
        copy_checked(out / 'strict_pairs.json', directory / 'strict_pairs.json')
        copy_checked(source_audit, directory / 'audit/training_inputs_final.json')
    eligible = {}
    for p in pairs:
        for key in ('sketch', 'gt', 'original_reference', 'swapped_reference'):
            assert sha(args.data / p[key]) == p['hashes'][key]
        for condition in ('old', 'new'):
            src = paths(previous, 'M0', p, condition)
            dst = paths(out, 'M0', p, condition)
            assert read(src.with_suffix('.json'))['output_sha256'] == sha(src)
            copy_checked(src, dst)
            copy_checked(src.with_suffix('.json'), dst.with_suffix('.json'))
        with Image.open(args.data / p['original_reference']) as source:
            rgb = np.asarray(source.convert('RGB').resize((384, 512), Image.Resampling.BICUBIC))
        if (rgb @ [.2126, .7152, .0722]).std() / 255. >= .02:
            eligible[p['case_id']] = rgb
    subset = []
    for pattern, count in [('stripes', 4), ('checks', 2), ('dots', 2)]:
        group = [p for p in pairs if p['qualification']['old_pattern'] == pattern
                 and p['case_id'] in eligible][:count]
        assert len(group) == count
        subset.extend(group)
    controls = {}
    for p in subset:
        case = p['case_id']
        rgb = eligible[case]
        packed = np.sort(np.ascontiguousarray(rgb).view('V3'), axis=None)
        controls[case] = {}
        for name in CONTROLS:
            value = reference_control(rgb, name, seed=p['seed'])
            assert np.array_equal(packed, np.sort(np.ascontiguousarray(value).view('V3'), axis=None))
            wanted = name.replace('_shift', '')
            scores = patch_bands(value[:64, :64])
            assert all(follow(scores, wanted, other) for other in PATCH_TARGETS if other != wanted)
            path = control_path(out, p, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(value).save(path)
            controls[case][name] = sha(path)
    write(out / 'protocol.json', dict(
        source=str(previous), checkpoint_sha256=CHECKPOINT_SHA, manifest_sha256=PAIRS_SHA,
        real_pairs=32, diagnostic_cases=[p['case_id'] for p in subset],
        selection='first eligible 4 stripes / 2 checks / 2 dots in frozen manifest; raw luma std >= .02',
        controls=controls, generator_updates=0,
        reference_resize=['historical bilinear', 'training bicubic'], canvas=[384, 512],
        sampler=dict(steps=50, cfg=7., sketch_scale=.6),
        windows=CELLS, scales={'0': [64, 48], '1': [32, 24]},
        intervention='replace one projected skip from swap in selected conditional DDIM steps; all other steps and scales use anchor',
        spectral_metric='Hann exact-bin fraction at (v16,h16,v32); flat patches score zero; invalid outputs count as failures',
        minimum_band_power=MIN_BAND_POWER,
        margins=read(previous / 'protocol.json')['inherited_exploratory_noninferiority'],
        environment=environment(), source_sha256=sha(__file__)))
    print('BFM_PREPARED', list(controls), flush=True)


class WindowSwitch:
    """只在指定去噪调用替换指定尺度的投影，保持原注入钩子。"""

    def __init__(self, network, anchor, swap, cell):
        self.network = network
        self.anchor = anchor
        scale, self.start, self.end = cell
        self.mixed = list(anchor)
        self.mixed[scale] = swap[scale]
        self.steps = []
        self.active_steps = []
        self.calls = [0, 0]

    def __enter__(self):
        def choose(_, args, kwargs):
            conditional = 'sa_hidden_states' in (kwargs.get('cross_attention_kwargs') or {})
            if conditional:
                index = len(self.steps)
                self.steps.append(args[1])
                active = self.start <= index < self.end
                if active:
                    self.active_steps.append(index)
                self.network.adapter.features = self.mixed if active else self.anchor

        def verify(_, args, kwargs, result):
            conditional = 'sa_hidden_states' in (kwargs.get('cross_attention_kwargs') or {})
            expected = [3, 2, 2, 3] if conditional else [0, 0, 0, 0]
            assert self.network.adapter.injection_counts == expected
            self.calls[0 if conditional else 1] += 1

        self.before = self.network.unet.register_forward_pre_hook(choose, with_kwargs=True)
        self.after = self.network.unet.register_forward_hook(verify, with_kwargs=True)
        return self

    def __exit__(self, *_):
        self.before.remove()
        self.after.remove()

    def audit(self):
        assert self.calls == [50, 50] and len(self.steps) == 50
        assert self.active_steps == list(range(self.start, self.end))
        return dict(conditional_steps=50, unconditional_steps=50, indices=self.active_steps,
                    timesteps=torch.stack(self.steps).cpu().tolist()[self.start:self.end])


def save_output(out, arm, p, condition, value, latent, ref_sha, **details):
    path = paths(out, arm, p, condition)
    path.parent.mkdir(parents=True, exist_ok=True)
    value.save(path)
    write(path.with_suffix('.json'), dict(
        arm=arm, condition=condition, case_id=p['case_id'], index=p['index'], seed=p['seed'],
        caption=p['caption'], initial_latent_sha256=latent, output_sha256=sha(path),
        checkpoint_sha256=CHECKPOINT_SHA, input_sketch_sha256=p['hashes']['sketch'],
        input_reference_sha256=ref_sha, manifest_sha256=PAIRS_SHA,
        GT_used_in_inference=False, **details))


def copy_output(source_root, source_arm, target_root, target_arm, p, condition):
    source = paths(source_root, source_arm, p, condition)
    target = paths(target_root, target_arm, p, condition)
    copy_checked(source, target)
    meta = read(source.with_suffix('.json'))
    assert sha(source) == meta['output_sha256']
    write(target.with_suffix('.json'), dict(meta, arm=target_arm, reused_from=str(source)))


def projected(network, sketch, texture):
    network.adapter.condition(sketch, texture)
    features = network.adapter.features
    network.adapter.clear()
    return features


def render(args, core, pipe, network, p, features, swap=None, cell=None):
    if cell is None:
        value, latent = image(args, core, pipe, network, p, features)
        return value, latent, {}
    with WindowSwitch(network, features, swap, cell) as switch:
        value, latent = image(args, core, pipe, network, p, features)
    return value, latent, switch.audit()


def run_real(args, core, pipe, network, p, sketch, probe, replay):
    out = args.out
    values = {}
    for condition, key in [('old', 'original_reference'), ('new', 'swapped_reference')]:
        ref = args.data / p[key]
        assert sha(ref) == p['hashes'][key]
        with Image.open(ref) as source:
            texture = preprocess_texture_image(source.convert('RGB'), 384, 512, mode='plain_resize')
        features = projected(network, sketch, texture[None].to(sketch))
        values[condition] = features
        historical = paths(out, 'M0', p, condition)
        expected_latent = read(historical.with_suffix('.json'))['initial_latent_sha256']
        if replay and condition == 'old':
            old_features = projected(network, sketch, core.tensor(ref).to(sketch))
            old, latent, _ = render(args, core, pipe, network, p, old_features, features, (0, 50, 50))
            with Image.open(historical) as original:
                assert np.array_equal(np.asarray(old), np.asarray(original.convert('RGB')))
            assert latent == expected_latent
        value, latent, _ = render(args, core, pipe, network, p, features)
        assert latent == expected_latent
        save_output(out, 'Cubic', p, condition, value, latent, p['hashes'][key], preprocessing='training_bicubic')
    if not probe:
        return
    window_out = out / 'real_window'
    for condition in ('old', 'new'):
        copy_output(out, 'Cubic', window_out, 'Cubic', p, condition)
    for arm, cell in CELLS.items():
        copy_output(out, 'Cubic', window_out, arm, p, 'old')
        value, latent, audit = render(args, core, pipe, network, p, values['old'], values['new'], cell)
        assert latent == expected_latent
        save_output(window_out, arm, p, 'new', value, latent, p['hashes']['swapped_reference'],
                    anchor_sha256=p['hashes']['original_reference'], switch=audit, preprocessing='training_bicubic')


def run_synthetic(args, core, pipe, network, p, sketch, hashes):
    out = args.out / 'synthetic'
    features = {}
    for name in CONTROLS:
        path = control_path(args.out, p, name)
        assert sha(path) == hashes[name]
        features[name] = projected(network, sketch, core.tensor(path).to(sketch))
        value, latent, _ = render(args, core, pipe, network, p, features[name])
        save_output(out, 'Full', p, name, value, latent, hashes[name], preprocessing='model_grid_png')
    for arm, cell in CELLS.items():
        for name in SWAPS:
            value, latent, audit = render(args, core, pipe, network, p, features['v16'], features[name], cell)
            save_output(out, arm, p, name, value, latent, hashes[name], anchor_sha256=hashes['v16'],
                        switch=audit, preprocessing='model_grid_png')


def generate(args):
    core, out = setup(args)
    spec = read(out / 'protocol.json')
    pairs = read(out / 'strict_pairs.json')['pairs'][args.shard::args.shards]
    checkpoint = args.previous / 'training/M0/latest.pt'
    assert sha(checkpoint) == spec['checkpoint_sha256'] == CHECKPOINT_SHA
    pipe, network, _ = load_m(args, core, 'M0')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    load_trainable_strict(network, state['network'])
    del state
    prepare_inference(args, pipe, network)
    before = weights_digest(network)
    with torch.inference_mode():
        for index, p in enumerate(pairs):
            assert sha(args.data / p['sketch']) == p['hashes']['sketch']
            sketch = core.tensor(args.data / p['sketch']).to(args.device).half()
            controls = spec['controls'].get(p['case_id'])
            run_real(args, core, pipe, network, p, sketch, controls is not None, replay=index == 0)
            if controls:
                run_synthetic(args, core, pipe, network, p, sketch, controls)
            print('BFM_CASE', args.shard, p['case_id'], 'controlled' if controls else 'interpolation', flush=True)
    assert before == weights_digest(network)
    write(out / f'audit/shard_{args.shard}.json', dict(passed=True, cases=[p['case_id'] for p in pairs],
          historical_replay=pairs[0]['case_id'], replay_bitwise=True, weights_sha256=before, environment=environment()))


def probe_images(out, p, hashes, arms, latent):
    rows = []
    for arm in arms:
        for condition in CONTROLS if arm == 'Full' else SWAPS:
            path = paths(out / 'synthetic', arm, p, condition)
            meta = read(path.with_suffix('.json'))
            assert (meta['output_sha256'] == sha(path) and meta['checkpoint_sha256'] == CHECKPOINT_SHA
                    and meta['input_reference_sha256'] == hashes[condition] and not meta['GT_used_in_inference']
                    and meta['initial_latent_sha256'] == latent and meta['input_sketch_sha256'] == p['hashes']['sketch'])
            if arm != 'Full':
                scale, start, end = CELLS[arm]
                assert meta['anchor_sha256'] == hashes['v16']
                assert meta['switch']['indices'] == list(range(start, end))
            with Image.open(path) as file:
                pixels = np.asarray(file.convert('RGB'))
            bands = [patch_bands(pixels[y:y + 64, x:x + 64]) for x, y in p['patch_coordinates']]
            mean = {key: float(np.mean([x[key] for x in bands])) for key in PATCH_TARGETS}
            rows.append(dict(arm=arm, case_id=p['case_id'], condition=condition, bands=mean,
                             patch_count=len(bands), switch=meta.get('switch')))
    return rows


def assess_controls(out, pairs, spec):
    rows = []
    for p in pairs:
        latent = read(paths(out, 'Cubic', p, 'old').with_suffix('.json'))['initial_latent_sha256']
        rows.extend(probe_images(out, p, spec['controls'][p['case_id']], ('Full', *CELLS), latent))
    lookup = {(r['arm'], r['condition'], r['case_id']): r['bands'] for r in rows}
    summary = []
    for arm in ('Full', *CELLS):
        for condition in SWAPS:
            values = []
            for p in pairs:
                case = p['case_id']
                anchor = lookup['Full', 'v16', case]
                switched = lookup[arm, condition, case]
                values.append((switched[condition] - switched['v16'], anchor[condition] - anchor['v16']))
            row = comparison_row(arm, 'Full_v16', 'all', condition, 'target_band_margin', values)
            row['target_matches'] = sum(follow(lookup[arm, condition, p['case_id']], condition, 'v16') for p in pairs)
            row['anchor_target_matches'] = sum(follow(lookup['Full', 'v16', p['case_id']], condition, 'v16') for p in pairs)
            row['paired_switches'] = sum(follow(lookup['Full', 'v16', p['case_id']], 'v16', condition)
                                         and follow(lookup[arm, condition, p['case_id']], condition, 'v16') for p in pairs)
            summary.append(row)
    phase = [(abs(lookup['Full', 'v16_shift', p['case_id']]['v16'] - lookup['Full', 'v16', p['case_id']]['v16']), 0.)
             for p in pairs]
    summary.append(comparison_row('Full_v16_shift', 'Full_v16', 'all', 'phase', 'absolute_v16_band_change', phase))
    write(out / 'synthetic/metrics/per_image.json', rows)
    write(out / 'synthetic/metrics/comparisons.json', summary)


def synthetic_panel(args, pair):
    out = args.out
    tiles = [('sketch', args.data / pair['sketch'])]
    tiles += [(f'ref {c}', control_path(out, pair, c)) for c in CONTROLS]
    tiles += [(f'{a} {c}', paths(out / 'synthetic', a, pair, c))
              for a in ('Full', *CELLS) for c in (CONTROLS if a == 'Full' else SWAPS)]
    width, height, columns = 160, 235, 7
    canvas = Image.new('RGB', (columns * width, ((len(tiles) + columns - 1) // columns) * height), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (label, path) in enumerate(tiles):
        x, y = i % columns * width, i // columns * height
        draw.text((x + 4, y + 4), label, fill='black')
        with Image.open(path) as source:
            canvas.paste(source.convert('RGB').resize((width, 213)), (x, y + 19))
    dest = out / 'synthetic/metrics/panels'
    dest.mkdir(exist_ok=True)
    canvas.save(dest / (pair['case_id'] + '.jpg'), quality=93)


def evaluate(args):
    _, out = setup(args)
    spec = read(out / 'protocol.json')
    audits = [read(out / f'audit/shard_{i}.json') for i in range(args.shards)]
    pairs = read(out / 'strict_pairs.json')['pairs']
    assert all(a['passed'] and a['replay_bitwise'] for a in audits)
    assert len({a['weights_sha256'] for a in audits}) == 1
    assert sorted(c for a in audits for c in a['cases']) == sorted(p['case_id'] for p in pairs)
    subset = [p for p in pairs if p['case_id'] in spec['diagnostic_cases']]
    assess_controls(out, subset, spec)
    for p in subset:
        synthetic_panel(args, p)
    for folder, arms, chosen in [(out, ('M0', 'Cubic'), pairs), (out / 'real_window', ('Cubic', *CELLS), subset)]:
        scoped = argparse.Namespace(**dict(vars(args), out=folder))
        for arm in arms:
            write(folder / 'training' / arm / 'complete.json', dict(checkpoint_sha256=CHECKPOINT_SHA))
        legacy.evaluate(scoped, arms=arms, audit=lambda _: None,
                        case_ids=None if folder == out else spec['diagnostic_cases'])
        summarize(scoped, chosen, [(a, arms[0]) for a in arms[1:]], spec['margins'],
                  include_historical_failures=False)
        panels(scoped, chosen if folder != out else subset, arms)
    bundle(out, 'bfm_skip_probe_results.zip', (
        out / 'protocol.json', out / 'strict_pairs.json', out / 'audit', out / 'metrics',
        out / 'real_window/metrics', out / 'synthetic/metrics'))
    print('BFM_REVIEW_COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'generate', 'evaluate'))
    parser.add_argument('--project', type=Path, default=PROJECT)
    parser.add_argument('--previous', type=Path, default=PREVIOUS)
    parser.add_argument('--out', type=Path, default=RUN)
    parser.add_argument('--data', type=Path, default=PROJECT.parent / 'datasets/BF')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=2)
    args = parser.parse_args()
    globals()[args.action](args)


if __name__ == '__main__':
    main()
