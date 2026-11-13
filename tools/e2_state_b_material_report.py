"""同一11个身份上的颜色分布、纹样细节和残留，保留逐图对照。"""
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

from tools import e2_state_b_pilot as pilot
from tools.bf_t3_three_arm_report import gray, paired
from tools.e2_fft_experiment import read, sha, write
from tools.e2_state_b_generalization_metrics import palette, pixels

ARMS = ('m0', 'old_raw', 'old_lab', 'b0', 'b400')


def lab(image):
    return cv2.cvtColor(np.asarray(image, np.float32) / 255, cv2.COLOR_RGB2LAB)


def aligned(picture, goal, positive, anchor, mask):
    """以已知平面目标和VAE重建衡量纹样、色度和旧模板，指标不是实拍真值。"""
    inside = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5) >= 16
    current, target, vae, base = map(lab, (picture, goal, positive, anchor))
    high = lambda field: field - cv2.blur(field, (17, 17))
    a, b, p, old = [high(field)[inside] for field in (current, target, vae, base)]
    # 移除模板中与新纹样共线的分量，避免把共享条纹计成残留。
    template = old - p
    template -= p * (np.sum(template * p) / max(float(np.sum(p * p)), 1e-6))
    energy = float(np.mean(template ** 2))
    projection = float(abs(np.mean((a - p) * template)) / energy) if energy >= .25 else None
    ref_dist, ref_spread = palette(target[inside, 1:])
    distribution, spread = palette(current[inside, 1:])
    _, positive_spread = palette(vae[inside, 1:])
    rms = lambda field: float(np.sqrt(np.mean(field ** 2)))
    contrast = rms(a) / max(rms(p), 1e-6)
    correlation = float(np.sum(a * b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-6))
    low_l = cv2.GaussianBlur((current[..., 0] - target[..., 0]) * mask, (0, 0), 24)
    low_l /= cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 24).clip(.001)
    return dict(correlation=correlation, contrast_ratio=contrast, contrast_error=abs(1 - contrast),
                pattern_hit=correlation >= .5 and .75 <= contrast <= 1.25,
                palette_swd=float(np.abs(distribution - ref_dist).mean()), palette_ratio=spread / max(ref_spread, 1e-6),
                spread=spread, goal_spread=ref_spread, positive_spread=positive_spread,
                low_l=float(np.abs(low_l[inside]).mean()), old_projection=projection,
                old_energy=energy, interior=int(inside.sum()))


def evaluate(args, spec):
    from tools import t3_response_run as response
    response.ROOT = args.out / f'metrics{args.shard}'
    metric = response.load_dists('cuda' if args.device.startswith('cuda') else args.device)
    sanity = response.sanity(metric)
    refcoords = [(int(x), int(y)) for y in np.linspace(0, 448, 4).round() for x in np.linspace(0, 320, 4).round()]

    def features(image, coords, grayscale):
        parts = [image[y:y+64, x:x+64] for x, y in coords]
        if grayscale:
            parts = [gray(part) for part in parts]
        tensor = torch.from_numpy(np.stack(parts).transpose(0, 3, 1, 2).copy()).float().to(args.device) / 255
        return metric.forward_once(tensor)

    cases = []
    with torch.inference_mode():
        for p in spec['holdout'][args.shard::2]:
            folder = args.out / 'cases' / p['case_id']
            mask = np.asarray(Image.open(folder / 'mask.png')) > 0
            inside = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5) >= 16
            coords = p['patch_coordinates']
            conditions = spec['validation_conditions']
            refs = {c: pixels(folder / (c + '_ref.png')) for c in conditions}
            ref_features = {(c, g): features(refs[c], refcoords, g) for c in ('real_a', 'real_b') for g in (False, True)}
            ref_palettes = {c: palette(lab(ref)[..., 1:]) for c, ref in refs.items()}
            rows, switches = {}, {}
            generation = read(folder / 'generation.json')['rows']
            assert len(generation) == 12
            for meta in generation:
                key = meta['key']
                c, s = key.rsplit('_', 1)
                goal, positive, m0 = (pixels(folder / f'{key}_{a}.png') for a in ('goal', 'positive', 'm0'))
                anchor_c = 'real_a' if c.startswith('real') else 'v16'
                anchor = pixels(folder / f'{anchor_c}_{s}_m0.png')
                rows[key] = {}
                for arm in (*ARMS, 'positive'):
                    path = folder / f'{key}_{arm}.png'
                    if arm in ('b0', 'b400'):
                        assert sha(path) == meta['output_sha256'][arm]
                    picture = pixels(path)
                    result = aligned(picture, goal, positive, anchor, mask)
                    ref_dist, ref_spread = ref_palettes[c]
                    out_dist, _ = palette(lab(picture)[inside, 1:])
                    result['reference_swd'] = float(np.abs(out_dist - ref_dist).mean())
                    result['reference_spread'] = ref_spread
                    result['outside'] = float(np.abs(picture[~mask].astype(np.float32) - m0[~mask]).mean() / 255)
                    if c.startswith('real'):
                        foil = 'real_b' if c == 'real_a' else 'real_a'
                        for grayscale, label in ((False, 'rgb'), (True, 'gray')):
                            f = features(picture, coords, grayscale)
                            target = response.patch_distance(metric, f, ref_features[c, grayscale])
                            other = response.patch_distance(metric, f, ref_features[foil, grayscale])
                            result.update({label + '_target': target, label + '_foil': other, label + '_margin': other - target})
                    else:
                        result['texture'] = pilot.texture_metrics(picture, coords, c)
                    rows[key][arm] = result
            for seed in range(2):
                for k, a, b in [('direction_v16', 'v16', 'h16'), ('direction_v32', 'v32', 'h32'),
                                ('period_v16', 'v16', 'v32'), ('period_h16', 'h16', 'h32')]:
                    switches[f'{k}_s{seed}'] = {arm: all(rows[f'{c}_s{seed}'][arm]['texture']['hit'] for c in (a, b))
                                                 for arm in (*ARMS, 'positive')}
                for name, metric_key in [('real_response', 'gray_margin'), ('real_pattern', 'pattern_hit')]:
                    switches[f'{name}_s{seed}'] = {arm: all(rows[f'{c}_s{seed}'][arm][metric_key] > 0 for c in ('real_a', 'real_b'))
                                                    for arm in (*ARMS, 'positive')}
            write(folder / 'result.json', dict(case=p['case_id'], stratum=p['stratum'], rows=rows, switches=switches))
            print('MATERIAL_EVALUATED', p['case_id'], flush=True)
            cases.append(p['case_id'])
    write(response.ROOT / 'complete.json', dict(cases=cases, sanity=sanity, protocol_sha256=sha(args.out / 'protocol.json')))


def panel(root, p):
    d = root / 'cases' / p['case_id']
    arms = ('reference', 'm0', 'old_raw', 'old_lab', 'b0', 'b400', 'goal', 'positive')
    canvas = Image.new('RGB', (len(arms) * 154, 4 * 226), 'white')
    draw = ImageDraw.Draw(canvas)
    for row, (c, s) in enumerate((c, s) for s in range(2) for c in ('real_a', 'real_b')):
        for col, arm in enumerate(arms):
            name = c + '_ref.png' if arm == 'reference' else f'{c}_s{s}_{arm}.png'
            x, y = col * 154, row * 226
            draw.text((x + 2, y + 2), f'{c} s{s} {arm}', fill='black')
            canvas.paste(Image.open(d / name).convert('RGB').resize((150, 200)), (x, y + 22))
    canvas.save(d / 'real_panel.jpg', quality=95)


def summarize(args, spec):
    root = args.out
    train = read(root / 'training/complete.json')
    assert train['updates'] == 400 and train['frozen_unchanged']
    done = [read(root / f'metrics{s}/complete.json') for s in range(2)]
    assert all(d['sanity']['passed'] and d['protocol_sha256'] == sha(root / 'protocol.json') for d in done)
    cases = [read(root / 'cases' / p['case_id'] / 'result.json') for p in spec['holdout']]
    assert sorted(r['case'] for r in cases) == sorted(c for d in done for c in d['cases'])
    groups = {}
    for group in ('synthetic', 'all_real', 'low_real', 'multi_real'):
        real = group != 'synthetic'
        selected = [r for r in cases if group in ('synthetic', 'all_real') or group.startswith(r['stratum'])]
        rows = [{key: row for key, row in r['rows'].items() if key.startswith('real') == real} for r in selected]
        kinds = ('real_response', 'real_pattern') if real else ('direction', 'period')
        fields = ['correlation', 'contrast_error', 'old_projection', 'palette_swd', 'reference_swd',
                  'spread', 'goal_spread', 'positive_spread', 'low_l', 'outside']
        if real:
            fields += ['gray_margin', 'gray_target', 'gray_foil', 'rgb_margin']
        statistics = {}
        for arm in (*ARMS, 'positive'):
            def values(key):
                return [row[arm][key] for identity in rows for row in identity.values() if row[arm][key] is not None]
            stats = {key: float(np.mean(values(key))) if values(key) else None for key in fields}
            stats['counts'] = {k: sum(v[arm] for r in selected for key, v in r['switches'].items() if key.startswith(k)) for k in kinds}
            stats['retain_75pct_positive_spread'] = sum(row[arm]['spread'] >= .75 * row[arm]['positive_spread']
                                                        for identity in rows for row in identity.values())
            statistics[arm] = stats
        differences = {}
        for field in fields:
            per_identity = []
            for identity in rows:
                delta = [v['b400'][field] - v['b0'][field] for v in identity.values()
                         if v['b400'][field] is not None and v['b0'][field] is not None]
                if delta:
                    per_identity.append(float(np.mean(delta)))
            differences[field] = paired(per_identity)
        groups[group] = dict(n=len(selected), images_per_arm=sum(len(r) for r in rows), arms=statistics, paired=differences)
    real = groups['all_real']['arms']
    previous, current = real['b0'], real['b400']
    multicolour = groups['multi_real']['arms']
    gates = dict(response=current['counts']['real_response'] >= max(previous['counts']['real_response'], real['old_raw']['counts']['real_response'])
                          and current['gray_margin'] >= previous['gray_margin'],
                 pattern=current['counts']['real_pattern'] > previous['counts']['real_pattern']
                         and current['correlation'] > previous['correlation'] and current['contrast_error'] < previous['contrast_error'],
                 old=current['old_projection'] is not None and previous['old_projection'] is not None
                     and current['old_projection'] < previous['old_projection'],
                 palette=current['palette_swd'] < previous['palette_swd'] and current['reference_swd'] < previous['reference_swd']
                         and multicolour['b400']['retain_75pct_positive_spread'] == groups['multi_real']['images_per_arm'],
                 light=current['low_l'] < previous['low_l'], outside=current['outside'] <= previous['outside'])
    write(root / 'summary.json', dict(groups=groups, gates=gates, all_gates=all(gates.values()),
        training=train, holdout=spec['holdout'], controlled_goal='flat real crop plus bounded low-frequency shade',
        texture_proxy='registered controlled-target correlation/contrast and unaligned DISTS margin',
        residue_proxy='projection against original M0 highpass residual orthogonalized to the controlled target',
        new_generated_images=2 * 12 * len(cases)))
    for p in spec['holdout']:
        panel(root, p)
    archive = root / 'material_b_review.zip'
    files = [root / 'protocol.json', root / 'summary.json', *root.glob('training/*.json'),
             *root.glob('training/*.jsonl'), *root.glob('metrics*/**/*.json')]
    for p in spec['train']:
        files.append(root / 'cases' / p['case_id'] / 'training_targets.jpg')
    for p in spec['holdout']:
        files.extend((root / 'cases' / p['case_id']).iterdir())
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(set(files)):
            if path.is_file():
                bundle.write(path, path.relative_to(root))
    print('MATERIAL_GATES', gates, flush=True)
    print('MATERIAL_ARCHIVE', archive.stat().st_size, sha(archive), flush=True)
