"""新T3消融统一评价；原文本质量与中性文本参考响应分别统计。"""
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

from tools.bf_t3_ablation import ROOT, PARENT, SOURCE, ARMS, NEW_ARMS, COEFFICIENTS, core, read, write, sha
from tools.bf_t3_three_arm_report import gray, paired


PRIMARY = ['edge_f1', 'iou', 'edge_l1', 'leak', 'lpips', 'tcf_lab']
RESPONSE = ['dists_margin', 'gray_margin', 'gray_dists']
METRICS = PRIMARY + RESPONSE + ['color_margin']


def table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def holm(rows):
    family = sorted([r for r in rows if r['p'] is not None], key=lambda r: r['p'])
    running = 0.
    for i, row in enumerate(family):
        running = max(running, min(1., row['p'] * (len(family) - i)))
        row['p_holm'] = running


def audit_training():
    protocol = read(ROOT / 'protocol.json')
    parent = read(ROOT / 'audit/parent_training.json')
    complete = dict(parent)
    for arm in NEW_ARMS:
        value = read(ROOT / 'training' / arm / 'complete.json')
        assert value['updates'] == 28365 and value['world'] == 2 and value['effective_batch'] == 8
        assert value['initial_sha256'] == protocol['initial_sha256']
        assert value['stream_hashes'] == parent['BF-SD']['stream_hashes']
        assert value['source_sha256'] == protocol['core_source_sha256']
        assert value['coefficients'] == protocol['coefficients'][arm]
        logs = list(csv.DictReader((ROOT / 'training' / arm / 'training_log.csv').open(encoding='utf-8')))
        old = list(csv.DictReader((PARENT / 'training/BF-SD/training_log.csv').open(encoding='utf-8')))
        assert len(logs) == len(old) == 28365
        assert [int(r['step']) for r in logs] == list(range(1, 28366))
        assert [float(r['lr']) for r in logs] == [float(r['lr']) for r in old]
        assert all(np.isfinite(float(v)) for r in logs for v in r.values())
        assert all(abs(float(r['total']) - (float(r['base']) + float(r['weighted_body']) + float(r['weighted_rest']))) < 1e-5 for r in logs)
        zero_key = 'weighted_body' if arm.endswith('no-body') else 'weighted_rest'
        assert all(float(r[zero_key]) == 0 for r in logs)
        complete[arm] = value
    write(ROOT / 'audit/final_comparability.json', dict(passed=True,
        same_initialization=True, same_random_stream=True, same_lr=True,
        updates=28365, world=2, coefficients=protocol['coefficients'], training=complete))
    params = []
    for arm in ARMS:
        value = complete[arm]
        params.append(dict(arm=arm, training_reused=arm not in NEW_ARMS,
            trainable_parameters=value['trainable_parameters'], initial_sha256=value['initial_sha256'],
            checkpoint_sha256=value['checkpoint_sha256'], updates=value['updates'],
            effective_batch=8, world=2, seed=42, learning_rate=5e-5, warmup=500,
            scheduler='cosine', optimizer='AdamW betas(.9,.999) eps1e-8 wd.01',
            precision='bf16 autocast/trainablefp32', body_coefficient=COEFFICIENTS[arm][0],
            rest_coefficient=COEFFICIENTS[arm][1], width=384, height=512,
            sampler='DDIM', steps=50, cfg=7., sketch_scale=.6, texture_tokens=False))
    table(ROOT / 'metrics/model_parameters.csv', params)


def evaluate():
    import torch
    from eval.eval_utils import prepare_evaluation_masks
    from eval.metrics import compute_structure_preservation, compute_texture_color_fidelity, compute_texture_leakage
    from tools.e2a_attr_eval import frozen_lpips
    from tools import t3_response_run as response
    from tools.t3_response_common import patches
    audit_training()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    response.ROOT = ROOT
    dists = response.load_dists('cuda')
    lpips, assets = frozen_lpips()
    write(ROOT / 'audit/lpips_assets.json', assets)
    rows = read(ROOT / 'evaluation.json')
    original = {r['id']: r for r in rows if r['condition'] == 'neutral_original'}
    reference_cache = {}

    def features(parts):
        value = torch.from_numpy(np.stack(parts).transpose(0, 3, 1, 2).copy()).float().cuda() / 255
        return dists.forward_once(value)

    def ref(path, is_gray):
        key = (path, is_gray)
        if key not in reference_cache:
            image = np.asarray(Image.open(path).convert('RGB').resize((384, 512), Image.Resampling.LANCZOS))
            parts = [image[y:y + 64, x:x + 64] for y in np.linspace(0, 448, 4).round().astype(int)
                     for x in np.linspace(0, 320, 4).round().astype(int)]
            reference_cache[key] = features([gray(p) for p in parts] if is_gray else parts)
        return reference_cache[key]

    def distance(a, b):
        matrix = response.distances(dists, a, b)
        return float(.5 * (matrix.min(0).values.mean() + matrix.min(1).values.mean()))

    records, failures = [], []
    with torch.inference_mode():
        for row in rows:
            masks = prepare_evaluation_masks((384, 512), sketch_path=row['sketch'], mask_policy='sketch_only')
            coords, candidates = patches(masks['garment'])
            latents = []
            for arm in ARMS:
                path = ROOT / 'outputs' / arm / row['condition'] / (row['id'] + '.png')
                meta = read(path.with_suffix('.json'))
                assert sha(path) == meta['output_sha256'] and not meta['GT_used_in_inference']
                assert meta['seed'] == row['seed']
                assert meta['input_sketch_sha256'] == row['sketch_sha256']
                assert meta['input_reference_sha256'] == row['reference_sha256']
                if row['condition'] != 'original': assert meta['caption'] == row['caption']
                latents.append(meta['initial_latent_sha256'])
                kwargs = dict(gen_path=str(path), sketch_path=row['sketch'], target_path=row['gt'], mask_bundle=masks)
                structure = compute_structure_preservation(**kwargs)
                record = dict(arm=arm, id=row['id'], category=row['category'], split=row['split'], condition=row['condition'],
                    edge_f1=structure.get('edge_f1'), iou=structure.get('sketch_iou'), edge_l1=structure.get('edge_l1'),
                    leak=compute_texture_leakage(**kwargs).get('leak_colored_frac'), lpips=None, tcf_lab=None,
                    dists_margin=None, gray_margin=None, gray_dists=None, color_margin=None,
                    patch_n=len(coords), patch_candidates=candidates)
                if row['condition'] == 'original':
                    record['lpips'] = float(lpips(core.tensor(path).cuda(), core.tensor(row['gt']).cuda()).mean())
                    record['tcf_lab'] = compute_texture_color_fidelity(texture_path=row['reference'], **kwargs).get('tcf_lab_delta')
                if row['condition'] == 'neutral_swapped':
                    old_color = compute_texture_color_fidelity(texture_path=original[row['id']]['reference'], **kwargs).get('tcf_lab_delta')
                    new_color = compute_texture_color_fidelity(texture_path=row['reference'], **kwargs).get('tcf_lab_delta')
                    if old_color is not None and new_color is not None: record['color_margin'] = old_color - new_color
                if row['split'] == 'neutral32' and len(coords) >= 4:
                    image = np.asarray(Image.open(path).convert('RGB'))
                    parts = [image[y:y + 64, x:x + 64] for x, y in coords]
                    for is_gray in [False, True]:
                        generated = features([gray(p) for p in parts] if is_gray else parts)
                        new_distance = distance(generated, ref(row['reference'], is_gray))
                        if is_gray: record['gray_dists'] = new_distance
                        if row['condition'] == 'neutral_swapped':
                            old_distance = distance(generated, ref(original[row['id']]['reference'], is_gray))
                            record['gray_margin' if is_gray else 'dists_margin'] = old_distance - new_distance
                elif row['split'] == 'neutral32':
                    failures.append(dict(arm=arm, id=row['id'], condition=row['condition'],
                        reason='fewer_than_four_fixed64px_patches', patch_n=len(coords)))
                records.append(record)
            assert len(set(latents)) == 1
            print('EVALUATED', row['condition'], row['id'], flush=True)
    lookup = {(r['arm'], r['id'], r['condition']): r for r in records}
    for row in rows:
        if row['condition'] != 'neutral_swapped': continue
        for arm in ARMS:
            old_meta = read((ROOT / 'outputs' / arm / 'neutral_original' / (row['id'] + '.json')))
            new_meta = read((ROOT / 'outputs' / arm / 'neutral_swapped' / (row['id'] + '.json')))
            assert old_meta['initial_latent_sha256'] == new_meta['initial_latent_sha256']
    safe = [{k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.items()} for r in records]
    assert len(safe) == 768
    write(ROOT / 'metrics/per_image.json', safe); table(ROOT / 'metrics/per_image.csv', safe)
    write(ROOT / 'audit/metric_failures.json', failures)
    summary = []
    for arm in ARMS:
        for condition in ['original', 'neutral_original', 'neutral_swapped']:
            part = [r for r in safe if r['arm'] == arm and r['condition'] == condition]
            for metric in METRICS:
                values = [r[metric] for r in part if r[metric] is not None]
                summary.append(dict(arm=arm, condition=condition, metric=metric, n=len(values), planned_n=len(part),
                    mean=float(np.mean(values)) if values else None,
                    std=float(np.std(values, ddof=1)) if len(values) > 1 else None))
    write(ROOT / 'metrics/summary.json', summary); table(ROOT / 'metrics/summary.csv', summary)
    lookup = {(r['arm'], r['id'], r['condition']): r for r in safe}
    comparisons = []
    for condition, metrics, family in [('original', PRIMARY, 'primary18'),
                                        ('neutral_swapped', RESPONSE, 'response9'),
                                        ('neutral_swapped', ['edge_f1', 'iou', 'leak', 'color_margin'], 'secondary12')]:
        family_rows = []
        for other in ARMS[:-1]:
            for metric in metrics:
                values = []
                for row in rows:
                    if row['condition'] != condition: continue
                    a, b = lookup[ARMS[-1], row['id'], condition][metric], lookup[other, row['id'], condition][metric]
                    if a is not None and b is not None: values.append(a - b)
                family_rows.append(dict(comparison='BF-SD-T3 minus ' + other, condition=condition,
                    metric=metric, family=family, **paired(values)))
        holm(family_rows); comparisons.extend(family_rows)
    write(ROOT / 'metrics/paired_statistics.json', comparisons)
    table(ROOT / 'metrics/paired_statistics.csv', [dict(r, ci95=json.dumps(r['ci95'])) for r in comparisons])
    # 交互项仅作描述，不能以单seed声称跨训练重复的协同作用。
    interactions = []
    for metric in PRIMARY + RESPONSE:
        condition = 'original' if metric in PRIMARY else 'neutral_swapped'
        values = []
        for row in rows:
            if row['condition'] != condition: continue
            v = [lookup[a, row['id'], condition][metric] for a in ARMS]
            if all(x is not None for x in v): values.append(v[3] - v[1] - v[2] + v[0])
        interactions.append(dict(metric=metric, condition=condition, **paired(values)))
    holm(interactions); write(ROOT / 'metrics/factorial_interactions.json', interactions)
    chosen = sorted([r for r in rows if r['condition'] == 'original'], key=lambda r: sha(r['sketch']))[:4]
    chosen += sorted([r for r in rows if r['condition'] == 'neutral_swapped'], key=lambda r: sha(r['sketch']))[:4]
    for page, start in enumerate(range(0, len(chosen), 4), 1):
        panel = Image.new('RGB', (1260, 260 * min(4, len(chosen) - start)), 'white'); draw = ImageDraw.Draw(panel)
        for j, row in enumerate(chosen[start:start + 4]):
            paths = [row['reference'], row['sketch'], row['gt'] if row['condition'] == 'original' else None]
            paths += [ROOT / 'outputs' / arm / row['condition'] / (row['id'] + '.png') for arm in ARMS]
            for col, (label, path) in enumerate(zip(['Reference', 'Sketch', 'GT original only'] + ARMS, paths)):
                draw.text((col * 180 + 3, j * 260 + 2), label, fill='black')
                if path:
                    image = Image.open(path).convert('RGB'); image.thumbnail((174, 226), Image.Resampling.LANCZOS)
                    panel.paste(image, (col * 180, j * 260 + 22))
            draw.text((3, j * 260 + 246), row['id'] + ' ' + row['condition'], fill='black')
        (ROOT / 'figures').mkdir(exist_ok=True); panel.save(ROOT / 'figures' / ('fixed_input_hash_%02d.png' % page))
    write(ROOT / 'evaluation_complete.json', dict(complete=True, outputs=len(safe), identities=128,
        primary128=True, neutral_response32=True, new_training_arms=NEW_ARMS,
        training_updates_per_new_arm=28365, independent_confirmation=False))


if __name__ == '__main__': evaluate()
