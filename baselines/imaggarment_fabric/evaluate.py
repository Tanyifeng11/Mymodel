"""复用上一轮冻结评价函数和固定身份，新增模型仅替换生成图路径。"""
import csv
import json
import sys
import tarfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .common import CONFIG, ROOT, OLD, HERE, NAME, read, write, output, sha


def table(path, rows):
    if rows:
        with output(path).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def gray(patch):
    # 与 tools/bf_t3_three_arm_report.py 完全相同的灰度/对比度标准化。
    array = np.asarray(patch, float) / 255
    luminance = array @ np.array([.2126, .7152, .0722])
    luminance = np.clip((luminance - luminance.mean()) / max(luminance.std(), 2 / 255) * .18 + .5, 0, 1)
    return np.repeat(np.round(luminance[..., None] * 255).astype('uint8'), 3, axis=-1)


def paired(values):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return dict(n=0, mean=None, ci95=None, p=None)
    rng = np.random.default_rng(110142)
    bootstrap = x[rng.integers(len(x), size=(10000, len(x)))].mean(1)
    permutation = (x * rng.choice([-1, 1], size=(10000, len(x)))).mean(1)
    return dict(n=len(x), mean=float(x.mean()), ci95=np.quantile(bootstrap, [.025, .975]).tolist(),
                p=float((1 + (np.abs(permutation) >= abs(x.mean())).sum()) / 10001))


def evaluate():
    import torch
    frozen_source = OLD / 'source'
    assert frozen_source.is_dir()
    sys.path.insert(0, str(frozen_source))
    from eval.eval_utils import prepare_evaluation_masks
    from eval.metrics import compute_structure_preservation, compute_texture_color_fidelity, compute_texture_leakage
    from tools.e2a_attr_eval import frozen_lpips
    from tools import t3_response_run as response
    from tools.t3_response_common import patches
    response.ROOT = ROOT
    dists = response.load_dists('cuda')
    lpips, lpips_assets = frozen_lpips()
    rows = read(ROOT / 'evaluation.json')
    originals = {r['id']: r for r in rows if r['condition'] == 'original'}
    reference_cache = {}

    def tensor(path):
        value = np.asarray(Image.open(path).convert('RGB').resize((384, 512), Image.Resampling.BILINEAR), np.float32)
        return torch.from_numpy(value.transpose(2, 0, 1).copy())[None].cuda() / 127.5 - 1

    def features(parts):
        value = torch.from_numpy(np.stack(parts).transpose(0, 3, 1, 2).copy()).float().cuda() / 255
        return dists.forward_once(value)

    def reference(path, is_gray):
        key = (path, is_gray)
        if key not in reference_cache:
            image = np.asarray(Image.open(path).convert('RGB').resize((384, 512), Image.Resampling.LANCZOS))
            parts = [image[y:y + 64, x:x + 64] for y in np.linspace(0, 448, 4).round().astype(int)
                     for x in np.linspace(0, 320, 4).round().astype(int)]
            reference_cache[key] = features([gray(p) for p in parts] if is_gray else parts)
        return reference_cache[key]

    def distance(first, second):
        matrix = response.distances(dists, first, second)
        return float(.5 * (matrix.min(0).values.mean() + matrix.min(1).values.mean()))

    def measure(row, path, arm):
        masks = prepare_evaluation_masks((384, 512), sketch_path=row['sketch'], mask_policy='sketch_only')
        coords, _ = patches(masks['garment'])
        kwargs = dict(gen_path=str(path), sketch_path=row['sketch'], target_path=row['gt'], mask_bundle=masks)
        structure = compute_structure_preservation(**kwargs)
        result = dict(arm=arm, id=row['id'], category=row['category'], split=row['split'], condition=row['condition'],
            edge_f1=structure.get('edge_f1'), iou=structure.get('sketch_iou'), edge_l1=structure.get('edge_l1'),
            leak=compute_texture_leakage(**kwargs).get('leak_colored_frac'), lpips=None,
            tcf_lab=compute_texture_color_fidelity(texture_path=row['reference'], **kwargs).get('tcf_lab_delta'),
            dists_margin=None, gray_margin=None, gray_dists=None, patch_n=len(coords))
        if row['condition'] == 'original':
            result['lpips'] = float(lpips(tensor(path), tensor(row['gt'])).mean())
        elif len(coords) >= 4:
            image = np.asarray(Image.open(path).convert('RGB'))
            parts = [image[y:y + 64, x:x + 64] for x, y in coords]
            for is_gray in [False, True]:
                generated = features([gray(p) for p in parts] if is_gray else parts)
                before = distance(generated, reference(originals[row['id']]['reference'], is_gray))
                after = distance(generated, reference(row['reference'], is_gray))
                result['gray_margin' if is_gray else 'dists_margin'] = before - after
                if is_gray:
                    result['gray_dists'] = after
        return {key: (None if isinstance(value, float) and not np.isfinite(value) else value) for key, value in result.items()}

    old_records = read(OLD / 'metrics/per_image.json')
    lookup = {(r['arm'], r['id'], r['condition']): r for r in old_records}
    # 在评价新模型前，对冻结的旧T3输出做原参考/换参考回归检查。
    selected = [rows[0], next(r for r in rows if r['condition'] == 'swapped' and
                             lookup['BF-SD-T3', r['id'], 'swapped']['gray_margin'] is not None)]
    regression = []
    records = []
    with torch.inference_mode():
        for row in selected:
            path = OLD / 'outputs/BF-SD-T3' / row['condition'] / (row['id'] + '.png')
            measured = measure(row, path, 'BF-SD-T3')
            expected = lookup['BF-SD-T3', row['id'], row['condition']]
            differences = {key: abs(measured[key] - expected[key]) for key in
                           ['edge_f1', 'iou', 'edge_l1', 'leak', 'lpips', 'tcf_lab', 'dists_margin', 'gray_margin', 'gray_dists']
                           if expected[key] is not None}
            assert max(differences.values()) < 2e-6, differences
            regression.append(dict(id=row['id'], condition=row['condition'], absolute_differences=differences))
        write(ROOT / 'audit/metric_regression.json', dict(passed=True, tolerance=2e-6, cases=regression,
            source_sha256={str(p.relative_to(frozen_source)): sha(p) for p in
                [frozen_source / 'eval/metrics.py', frozen_source / 'eval/eval_utils.py',
                 frozen_source / 'tools/t3_response_common.py', frozen_source / 'tools/t3_response_run.py',
                 frozen_source / 'tools/bf_t3_three_arm_report.py']}))
        for row in rows:
            path = ROOT / 'outputs' / NAME / row['condition'] / (row['id'] + '.png')
            meta = read(path.with_suffix('.json'))
            assert sha(path) == meta['output_sha256'] and not meta['GT_used_in_inference']
            records.append(measure(row, path, NAME))
            print('EVALUATED', row['id'], row['condition'], flush=True)
        # 历史表没有新参考TCF，按同一冻结函数补算三组已有输出，不重新生成。
        conditions = {(r['id'], r['condition']): r for r in rows}
        for record in old_records:
            record = dict(record, category=conditions[record['id'], record['condition']]['category'])
            if record['condition'] == 'swapped':
                row = conditions[record['id'], record['condition']]
                masks = prepare_evaluation_masks((384, 512), sketch_path=row['sketch'], mask_policy='sketch_only')
                path = OLD / 'outputs' / record['arm'] / 'swapped' / (record['id'] + '.png')
                record['tcf_lab'] = compute_texture_color_fidelity(gen_path=str(path), sketch_path=row['sketch'],
                    target_path=row['gt'], texture_path=row['reference'], mask_bundle=masks).get('tcf_lab_delta')
            records.append(record)
    write(ROOT / 'metrics/per_image.json', records)
    table(ROOT / 'metrics/per_image.csv', records)
    metrics = ['edge_f1', 'iou', 'edge_l1', 'leak', 'lpips', 'tcf_lab', 'dists_margin', 'gray_margin', 'gray_dists']
    arms = [NAME, 'BF-SD', 'BF-SD-T3', 'BF-Fashion']
    summaries = []
    for arm in arms:
        for split, condition in [('heldout128', 'original'), ('swap32', 'original'), ('swap32', 'swapped')]:
            part = [r for r in records if r['arm'] == arm and r['split'] == split and r['condition'] == condition]
            for category in ['all'] + sorted({r['category'] for r in part}):
                chosen = part if category == 'all' else [r for r in part if r['category'] == category]
                for metric in metrics:
                    values = [r[metric] for r in chosen if r[metric] is not None]
                    summaries.append(dict(arm=arm, split=split, condition=condition, category=category, metric=metric,
                        n=len(values), planned_n=len(chosen), mean=float(np.mean(values)) if values else None))
    write(ROOT / 'metrics/summary.json', summaries)
    table(ROOT / 'metrics/summary.csv', summaries)
    lookup = {(r['arm'], r['id'], r['condition']): r for r in records}
    comparisons = []
    for other in arms[1:]:
        for metric in metrics + ['new_reference_tcf_lab']:
            condition = 'swapped' if metric in ['dists_margin', 'gray_margin', 'gray_dists', 'new_reference_tcf_lab'] else 'original'
            key = 'tcf_lab' if metric == 'new_reference_tcf_lab' else metric
            chosen = [r for r in rows if r['condition'] == condition and (condition == 'swapped' or r['split'] == 'heldout128')]
            differences = []
            for row in chosen:
                first = lookup[NAME, row['id'], condition][key]
                second = lookup[other, row['id'], condition][key]
                if first is not None and second is not None:
                    differences.append(first - second)
            comparisons.append(dict(comparison=NAME + ' minus ' + other, metric=metric, condition=condition, **paired(differences)))
        family = sorted([r for r in comparisons if r['comparison'].endswith(' ' + other) and r['p'] is not None], key=lambda r: r['p'])
        running = 0.
        for i, row in enumerate(family):
            running = max(running, min(1., row['p'] * (len(family) - i)))
            row['p_holm'] = running
    write(ROOT / 'metrics/paired_statistics.json', comparisons)
    table(ROOT / 'metrics/paired_statistics.csv', [dict(r, ci95=json.dumps(r['ci95'])) for r in comparisons])
    chosen = sorted([r for r in rows if r['split'] == 'heldout128'], key=lambda r: r['sketch_sha256'])[:4]
    chosen += sorted([r for r in rows if r['condition'] == 'swapped'], key=lambda r: r['sketch_sha256'])[:4]
    for page in range(2):
        panel = Image.new('RGB', (1260, 1040), 'white')
        draw = ImageDraw.Draw(panel)
        for j, row in enumerate(chosen[page * 4:page * 4 + 4]):
            paths = [row['sketch'], row['reference'], row['gt'] if row['condition'] == 'original' else None]
            paths += [OLD / 'outputs' / arm / row['condition'] / (row['id'] + '.png') for arm in arms[1:]]
            paths += [ROOT / 'outputs' / NAME / row['condition'] / (row['id'] + '.png')]
            for col, (label, path) in enumerate(zip(['Sketch', 'Fabric', 'GT original only'] + arms[1:] + ['Adapted IMAGGarment'], paths)):
                draw.text((col * 180 + 3, j * 260 + 2), label, fill='black')
                if path:
                    image = Image.open(path).convert('RGB')
                    image.thumbnail((174, 226), Image.Resampling.LANCZOS)
                    panel.paste(image, (col * 180, j * 260 + 22))
            draw.text((3, j * 260 + 246), row['id'] + ' ' + row['condition'], fill='black')
        panel.save(output(ROOT / 'figures' / ('fixed_inputs_%02d.png' % (page + 1))))
    from .prepare import protected_weights
    assert protected_weights() == read(ROOT / 'audit/protected_before.json'), '既有模型权重状态发生变化'
    write(ROOT / 'audit/weight_isolation_complete.json', dict(passed=True, old_weight_states_unchanged=True,
                                                           initialization_independently_copied=True))
    write(ROOT / 'evaluation_complete.json', dict(passed=True, generated_outputs=192, original_identities=128,
        swap_identities=32, lpips_assets=lpips_assets, independent_confirmation=False,
        comparable_metric_implementation=True, equal_training_budget=False,
        caveats=['official pooled CLIP appearance adaptation, not claimed exact texture transfer',
                 'GAM-only; no Logo or LEM', 'previously inspected tests; one training seed',
                 'legacy Leak uses fixed sketch exterior and old-GT background deviation proxy',
                 'Color Adapter and GAM combined exposure is twice T3 single-stage exposure']))


def package():
    assert read(ROOT / 'evaluation_complete.json')['passed']
    paths = []
    for folder in ['audit', 'datasets', 'outputs', 'generation', 'metrics', 'figures', 'training']:
        paths += [p for p in (ROOT / folder).rglob('*') if p.is_file() and p.suffix in {'.json', '.jsonl', '.csv', '.png'}]
    paths += [ROOT / 'protocol.json', ROOT / 'evaluation.json', ROOT / 'evaluation_complete.json', ROOT / 'cache/complete.json']
    write(ROOT / 'bundle_manifest.json', {str(p.relative_to(ROOT)): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in paths})
    archive = output(ROOT / 'delivery/results.tar.gz')
    with tarfile.open(archive, 'w:gz') as tar:
        for path in paths + [ROOT / 'bundle_manifest.json']:
            tar.add(path, arcname=str(path.relative_to(ROOT)))
    parts = []
    with archive.open('rb') as stream:
        index = 0
        while block := stream.read(24 * 1024 * 1024):
            part = output(ROOT / 'delivery' / ('results.tar.gz.part%02d' % index))
            part.write_bytes(block)
            parts.append(dict(file=part.name, sha256=sha(part), bytes=len(block)))
            index += 1
    write(ROOT / 'delivery/bundle_info.json', dict(file=archive.name, sha256=sha(archive), bytes=archive.stat().st_size,
                                                 parts=parts, excluded='model/optimizer weights and representation cache'))
    print('PACKAGED', archive.stat().st_size, len(parts), flush=True)


if __name__ == '__main__':
    package()
