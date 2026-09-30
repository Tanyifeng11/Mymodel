"""E30-A3：冻结输入、四组 pilot、必要消融、条件式确认及完整性报告。"""

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment

from models.apacc_affinity_adapter import PatternAffinityAdapter
from models.apacc_canonicality import select_canonical_crop
from models.apacc_hierarchical_aggregation import DEFAULT_CONFIG, hierarchical_region_aggregation, boundary_edges
from models.apacc_region_prototype import node_features
from models.apacc_global_reconciliation import reconcile_pattern_groups
from models.apacc_region_cleanup import cleanup_small_regions
from tools.e30_a2_affinity import (compute_labels, load_feature, norm_at, regions_from_labels,
                                  tensor_field, partition_stability)


ARMS = ('G0_learned_fixed', 'G1_learned_adaptive', 'G2_hierarchical', 'G3_reconciled')
METRICS = ('orientation_consistency', 'period_consistency', 'identity', 'homogeneity', 'contamination')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def ci(values, seed=42, median=False):
    values = np.asarray(values, float)
    if not len(values):
        return None
    draws = np.random.default_rng(seed).choice(values, (2000, len(values)), replace=True)
    samples = np.median(draws, axis=1) if median else draws.mean(1)
    return np.percentile(samples, [2.5, 97.5]).tolist()


def stability(a, b, occupancy):
    # 按全部前景计算；未匹配、uncertain 和丢失支持不会被排除。
    valid = occupancy >= .5
    left = [r for r in np.unique(a[valid]) if r >= 0]
    right = [r for r in np.unique(b[valid]) if r >= 0]
    n = max(len(left), len(right))
    if not n:
        return 0.
    matrix = np.zeros((n, n))
    for i, x in enumerate(left):
        for j, y in enumerate(right):
            am, bm = valid & (a == x), valid & (b == y)
            matrix[i, j] = (am & bm).sum() / max((am | bm).sum(), 1)
    i, j = linear_sum_assignment(-matrix)
    return float(matrix[i, j].sum() / n)


def summarize(rows, seed):
    summary = {}
    for split in ('dev', 'stress_dev', 'combined'):
        subset = [r for r in rows if split == 'combined' or r['split'] == split]
        counts = [r['region_count'] for r in subset]
        entry = {'count': len(subset), 'valid_crop_count': sum(r['valid_crop'] for r in subset),
            'valid_crop_coverage': float(np.mean([r['valid_crop'] for r in subset])),
            'mean_region_count': float(np.mean(counts)), 'median_region_count': float(np.median(counts)),
            'median_region_count_ci95': ci(counts, seed, median=True),
            'over_10_fraction': float(np.mean(np.array(counts) > 10)),
            'collapse_violations': sum(r['collapse_violation'] for r in subset),
            'one_region_count': sum(r['region_count'] == 1 for r in subset)}
        for key in ('region_count', 'pattern_group_count', 'foreground_coverage',
                    'eligible_foreground_coverage', 'uncertain_region_count',
                    'augmentation_stability', 'legacy_stability_iou'):
            values = [r[key] for r in subset]
            entry[key], entry[key + '_ci95'] = float(np.mean(values)), ci(values, seed)
        for key in METRICS:
            values = [r['crop']['metrics'][key] for r in subset if r['valid_crop']]
            entry[key] = float(np.mean(values)) if values else None
            entry[key + '_ci95'] = ci(values, seed)
            entry[key + '_reference_count'] = len(values)
        summary[split] = entry
    return summary


def gate(summary, paired_delta=None, canonical=False):
    checks = {}
    for split in ('dev', 'stress_dev'):
        m = summary[split]
        checks[split + '/valid_crop'] = m['valid_crop_count'] >= 13
        for key, threshold, lower in (('orientation_consistency', .8775, True),
                ('period_consistency', .8817, True), ('identity', .9584, True),
                ('homogeneity', .9095, True), ('contamination', .0094, False)):
            checks[split + '/' + key] = m[key] is not None and (m[key] >= threshold if lower else m[key] <= threshold)
        if not canonical:
            checks[split + '/median_regions'] = m['median_region_count'] <= 6
            checks[split + '/over_10_fraction'] = m['over_10_fraction'] <= .25
            checks[split + '/stability'] = m['augmentation_stability'] >= .80
            checks[split + '/anti_collapse'] = m['collapse_violations'] == 0
    if paired_delta is not None and not canonical:
        for split in ('dev', 'stress_dev'):
            checks[split + '/lower_than_G1'] = paired_delta[split]['region_count_delta_ci95'][1] < 0
    return {'checks': checks, 'pass': all(checks.values())}


def prepared(args):
    src, out = args.source, args.out
    manifest = read(src / 'split_manifest.json')
    assert len(manifest['dev']) == len(manifest['stress_dev']) == 16
    assert not set(manifest['dev']) & set(manifest['stress_dev'])
    audit = read(src / 'A21_embedding/summary.json')
    assert all(audit['seeds'][str(seed)]['gate_pass'] for seed in (42, 43, 44))
    protocol = read(src / 'protocol.json')
    assert sha(Path('models/local_pattern_field.py')) == protocol['e26_sha256']
    out.mkdir(parents=True, exist_ok=True)
    write(out / 'split_manifest.json', manifest)
    inputs = {'source': str(src), 'source_protocol': protocol, 'training_steps': 0,
        'manual_annotations_used': False, 'E29_causal_test_used': False,
        'checkpoint_sha256': {str(s): sha(src / 'A21_embedding' / ('seed%d' % s) / 'adapter.pt') for s in (42, 43, 44)},
        'normalization_sha256': sha(src / 'train_normalization.npz'),
        'feature_cache_sha256': {str(i): sha(src / 'feature_cache' / ('%03d.npz' % i)) for i in range(64, 96)},
        'frozen_source_sha256': {str(p): sha(p) for p in [Path('models/apacc_affinity_adapter.py'),
            Path('models/apacc_features.py'), Path('models/local_pattern_field.py'),
            Path('models/apacc_canonicality.py'), Path('inference_IMAGGarment-1.py')]},
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'notes': ['复用 A2 固定特征与真实颜色增强 DINO，几何/颜色沿用原图，与 A2 口径一致。',
                  'G0/G1/G2/G3 的 crop 指标沿用 A2 handcrafted selector；后续 canonicality 复核不构成独立增益证据。',
                  'stress-dev 未按纹样类别分层；方向/频率和污染均为无标注代理。',
                  'Gate 同时检查 dev 与 stress-dev；稳定性包含未匹配区域，另报告 A2 legacy IoU。',
                  '逻辑 reconciliation 不降低 spatial region count；所有 uncertain region 仍计入区域数。']}
    write(out / 'input_manifest.json', inputs)
    write(out / 'aggregation_config.json', DEFAULT_CONFIG)
    return inputs


def embedded_inputs(args, seed):
    checkpoint = args.source / 'A21_embedding' / ('seed%d' % seed) / 'adapter.pt'
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = PatternAffinityAdapter(ckpt['input_dim'] - 384, not ckpt['no_geometry'])
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval().requires_grad_(False)
    norm = norm_at(args.source)
    names = read(args.source / 'split_manifest.json')
    names = names['dev'] + names['stress_dev']
    for index in range(64, 96):
        f = load_feature(args.source, index)
        aug = dict(f, appearance=f['appearance_aug'])
        with torch.inference_mode():
            z, az = [model(torch.from_numpy(tensor_field(q, norm).reshape(-1, ckpt['input_dim']))).reshape(64, 64, 64).numpy() for q in (f, aug)]
        image = Image.open(args.dataset / 'training' / names[index - 64]).convert('RGB').resize((256, 256))
        yield index, names[index - 64], f, aug, z, az, image


def partition(features, z, arm, config):
    if arm in ARMS[:2]:
        labels = compute_labels(features, z, 'fixed' if arm == ARMS[0] else 'adaptive')
        # 基线也审计同一 strong discontinuity。
        edges = boundary_edges(node_features(features, z), config)
        strong = [(u, v) for u, v, *rest in edges if rest[-1]]
        detail = {'strong_edge_count': len(strong),
            'strong_discontinuity': len(strong) >= max(8, int((features['occupancy'] >= .5).sum() * .015)),
            'preserved_strong_edge_count': int(sum(labels.ravel()[u] != labels.ravel()[v] for u, v in strong))}
        detail['collapse_violation'] = bool(detail['strong_discontinuity'] and labels.max() < 1)
        return labels, labels.copy(), np.zeros_like(labels, bool), detail
    state = hierarchical_region_aggregation(features, z, config)
    global_logs, group_lineage = [], []
    if arm != 'G2_hierarchical':
        if config['cleanup']:
            cleanup_small_regions(state)
        if arm == 'no_global_reconciliation':
            groups = None
        else:
            groups, global_logs, group_lineage = reconcile_pattern_groups(state)
    else:
        groups = None
    labels, groups_map, uncertain, detail = state.export(groups)
    detail['global_merge_logs'] = global_logs
    detail['pattern_group_lineage'] = group_lineage
    return labels, groups_map, uncertain, detail


def palette_map(labels):
    rng = np.random.default_rng(100)
    colors = rng.integers(45, 240, (max(int(labels.max()) + 1, 1), 3), dtype=np.uint8)
    rgb = colors[np.maximum(labels, 0)]
    rgb[labels < 0] = 245
    result = Image.fromarray(rgb).resize((256, 256), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(result)
    for region in np.unique(labels):
        if region < 0:
            continue
        yy, xx = np.where(labels == region)
        draw.text((int(xx.mean() * 4), int(yy.mean() * 4)), str(region), fill=(0, 0, 0), stroke_width=1, stroke_fill=(255, 255, 255))
    return result


def pca_map(z, occupancy):
    x = z.reshape(-1, z.shape[-1])
    valid = occupancy.ravel() >= .5
    center = x[valid].mean(0)
    covariance = (x[valid] - center).T @ (x[valid] - center)
    _, vectors = np.linalg.eigh(covariance)
    v = ((x - center) @ vectors[:, -3:]).reshape(64, 64, 3)
    lo, hi = np.percentile(v[occupancy >= .5], [2, 98], axis=0)
    rgb = np.uint8(np.clip((v - lo) / np.maximum(hi - lo, 1e-9), 0, 1) * 255)
    rgb[occupancy < .5] = 245
    return Image.fromarray(rgb).resize((256, 256))


def visualization(folder, index, image, f, z, results):
    images = [('Reference', image), ('Learned PCA', pca_map(z, f['occupancy']))]
    geo = f['geometry']
    angle = np.arctan2(geo[..., 1], geo[..., 0]) / (2 * np.pi) + .5
    for title, field in [('Orientation', angle), ('Log frequency', np.clip((geo[..., 2] + 8) / 7, 0, 1))]:
        color = cv2.applyColorMap(np.uint8(field * 255), cv2.COLORMAP_TURBO)[..., ::-1].copy()
        color[f['occupancy'] < .5] = 245
        images.append((title, Image.fromarray(color).resize((256, 256))))
    for arm in ARMS:
        images.append((arm, palette_map(results[arm][0])))
    images.append(('G3 pattern groups', palette_map(results[ARMS[3]][1])))
    row = results[ARMS[3]][4]
    crop = image.crop(row['crop']['box']).resize((256, 256)) if row['valid_crop'] else Image.new('RGB', (256, 256), 'white')
    images.append(('Candidate crop' if row['valid_crop'] else 'No valid crop', crop))
    sheet = Image.new('RGB', (1280, 568), 'white')
    draw = ImageDraw.Draw(sheet)
    for n, (title, pic) in enumerate(images):
        x, y = (n % 5) * 256, (n // 5) * 284
        sheet.paste(pic, (x, y + 28))
        draw.text((x + 4, y + 6), title, fill='black')
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / ('%03d_audit.png' % index)
    sheet.save(path)
    return str(path)


def paired(before, after, seed):
    lookup = {r['index']: r for r in before}
    report = {}
    for split in ('dev', 'stress_dev', 'combined'):
        rows = [r for r in after if split == 'combined' or r['split'] == split]
        delta = [r['region_count'] - lookup[r['index']]['region_count'] for r in rows]
        report[split] = {'region_count_mean_delta': float(np.mean(delta)),
                         'region_count_delta_ci95': ci(delta, seed), 'reference_count': len(delta)}
    return report


def evaluate(args, seed, folder, configs, previews=False):
    rows = {arm: [] for arm in configs}
    for index, name, f, aug, z, az, image in embedded_inputs(args, seed):
        results = {}
        for arm, config in configs.items():
            output = folder / arm
            output.mkdir(parents=True, exist_ok=True)
            labels, groups, uncertain, detail = partition(f, z, arm, config)
            alabels, agroups, auncertain, _ = partition(aug, az, arm, config)
            eligible = labels.copy()
            eligible[uncertain] = -1
            regions = regions_from_labels(eligible, f['foreground'].astype(bool))
            selected = select_canonical_crop(image, regions, f['foreground'].astype(bool)) if regions else None
            valid = f['occupancy'] >= .5
            row = {'index': index, 'image': name, 'seed': seed,
                'split': 'dev' if index < 80 else 'stress_dev',
                'region_count': len(np.unique(labels[labels >= 0])),
                'pattern_group_count': len(np.unique(groups[groups >= 0])),
                'uncertain_region_count': len(np.unique(labels[uncertain])),
                'foreground_coverage': float(((labels >= 0) & valid).sum() / max(valid.sum(), 1)),
                'eligible_foreground_coverage': float(((eligible >= 0) & valid).sum() / max(valid.sum(), 1)),
                'valid_crop': selected is not None,
                'augmentation_stability': stability(labels, alabels, f['occupancy']),
                'pattern_group_stability': stability(groups, agroups, f['occupancy']),
                'legacy_stability_iou': partition_stability(labels, alabels, f['occupancy'])['iou'],
                'strong_discontinuity': detail['strong_discontinuity'],
                'strong_edge_count': detail['strong_edge_count'],
                'preserved_strong_edge_count': detail['preserved_strong_edge_count'],
                'collapse_violation': detail['collapse_violation'],
                'crop': None if selected is None else {'box': selected['box'], 'metrics': selected['metrics'], 'score': selected['score']}}
            rows[arm].append(row)
            np.savez_compressed(output / ('%03d_regions.npz' % index), labels=labels, pattern_groups=groups,
                uncertain=uncertain, augmented_labels=alabels, augmented_pattern_groups=agroups)
            if arm not in ARMS[:2]:
                write(output / ('%03d_aggregation.json' % index), detail)
            results[arm] = labels, groups, uncertain, detail, row
        if previews:
            visualization(folder / 'visualizations', index, image, f, z, results)
        print('[A3]', seed, index - 63, '/32', {arm: rows[arm][-1]['region_count'] for arm in rows}, flush=True)
    report = {'seed': seed, 'reference_unit': 'reference image', 'bootstrap_resamples': 2000,
              'arms': {arm: summarize(value, seed) for arm, value in rows.items()}}
    for arm, value in rows.items():
        write(folder / arm / 'rows.json', value)
    if ARMS[3] in rows:
        baseline = rows.get(ARMS[1]) or read(args.out / 'pilot' / ARMS[1] / 'rows.json')
        report['G1_to_G3'] = paired(baseline, rows[ARMS[3]], seed)
        report['gate'] = gate(report['arms'][ARMS[3]], report['G1_to_G3'])
    write(folder / 'summary.json', report)
    return report


def run_pilot(args):
    prepared(args)
    config = read(args.out / 'aggregation_config.json')
    report = evaluate(args, 42, args.out / 'pilot', {arm: config for arm in ARMS}, previews=True)
    baseline = read(args.source / 'A22_segmentation/seed42/summary.json')
    reproduction = {}
    for arm, old in zip(ARMS[:2], ('learned_fixed', 'learned_adaptive')):
        reproduction[arm] = {split: {'A2': baseline['arms'][old][split],
            'A3': report['arms'][arm][split]} for split in ('dev', 'stress_dev')}
    write(args.out / 'pilot/baseline_reproduction.json', reproduction)
    before = read(args.out / 'pilot' / ARMS[1] / 'rows.json')
    after = read(args.out / 'pilot' / ARMS[3] / 'rows.json')
    audit = {'G1_top_fragmented': [r['index'] for r in sorted(before, key=lambda r: -r['region_count'])[:5]],
        'G3_most_merged': [r['index'] for r in sorted(after, key=lambda r:
            next(b['region_count'] for b in before if b['index'] == r['index']) - r['region_count'], reverse=True)[:5]],
        'G3_over_10': [r['index'] for r in after if r['region_count'] > 10],
        'G3_one_region': [r['index'] for r in after if r['region_count'] == 1],
        'G3_no_valid_crop': [r['index'] for r in after if not r['valid_crop']],
        'G3_collapse_violations': [r['index'] for r in after if r['collapse_violation']],
        'stress_visualizations': [str(Path('pilot/visualizations') / ('%03d_audit.png' % i)) for i in range(80, 96)]}
    audit['required_visual_indices'] = sorted(set(sum([v for k, v in audit.items() if k != 'stress_visualizations'], [])))
    write(args.out / 'visual_audit/selection.json', audit)
    print('[A3 pilot gate]', report['gate'], flush=True)


def run_ablations(args):
    config = read(args.out / 'aggregation_config.json')
    variants = {'no_global_reconciliation': config,
                'no_coherence': dict(config, coherence=False),
                'no_geometry': dict(config, geometry=False),
                'no_boundary': dict(config, boundary=False),
                'no_cleanup': dict(config, cleanup=False)}
    report = evaluate(args, 42, args.out / 'ablations', variants)
    g2 = read(args.out / 'pilot' / ARMS[2] / 'rows.json')
    g3 = read(args.out / 'pilot' / ARMS[3] / 'rows.json')
    report['G2_without_global_or_cleanup'] = summarize(g2, 42)
    report['G2_to_G3'] = paired(g2, g3, 42)
    report['note'] = 'A 同时报告文档 G2 vs G3 和仅关闭 global 的独立消融；E 单独关闭 cleanup。'
    write(args.out / 'ablations/summary.json', report)


def run_confirm(args):
    pilot = read(args.out / 'pilot/summary.json')
    if not pilot['gate']['pass']:
        write(args.out / 'multiseed/status.json', {'status': 'not_run_pilot_gate_failed'})
        return
    config = read(args.out / 'aggregation_config.json')
    for seed in (42, 43, 44):
        if seed == 42:
            folder = args.out / 'multiseed/seed42'
            folder.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.out / 'pilot/summary.json', folder / 'summary.json')
            shutil.copytree(args.out / 'pilot' / ARMS[3], folder / ARMS[3], dirs_exist_ok=True)
        else:
            evaluate(args, seed, args.out / 'multiseed' / ('seed%d' % seed), {ARMS[1]: config, ARMS[3]: config})
    seeds = {str(s): read(args.out / 'multiseed' / ('seed%d' % s) / 'summary.json') for s in (42, 43, 44)}
    write(args.out / 'multiseed/status.json', {'status': 'completed', 'pass': sum(r['gate']['pass'] for r in seeds.values()) >= 2,
        'seed_gate_pass': {s: r['gate']['pass'] for s, r in seeds.items()},
        'seed_summary': {s: r['arms'][ARMS[3]] for s, r in seeds.items()}})


def run_canonical(args):
    multiseed = read(args.out / 'multiseed/status.json')
    if not multiseed.get('pass', False):
        write(args.out / 'canonicality_check/status.json', {'status': 'not_run_segmentation_gate_failed'})
        return
    seeds = {}
    for seed in (42, 43, 44):
        folder = args.out / 'multiseed' / ('seed%d' % seed)
        rows = read(folder / ARMS[3] / 'rows.json')
        # 对已冻结的空间 region 重新调用现有 handcrafted selector，不再改变分区。
        for row in rows:
            image = Image.open(args.dataset / 'training' / row['image']).convert('RGB').resize((256, 256))
            f = load_feature(args.source, row['index'])
            with np.load(folder / ARMS[3] / ('%03d_regions.npz' % row['index'])) as data:
                labels = data['labels'].copy()
                labels[data['uncertain']] = -1
            regions = regions_from_labels(labels, f['foreground'].astype(bool))
            selected = select_canonical_crop(image, regions, f['foreground'].astype(bool)) if regions else None
            row['valid_crop'] = selected is not None
            row['crop'] = None if selected is None else {'box': selected['box'], 'metrics': selected['metrics']}
        summary = summarize(rows, seed)
        seeds[str(seed)] = {'summary': summary, 'gate': gate(summary, canonical=True)}
        write(args.out / 'canonicality_check' / ('seed%d_rows.json' % seed), rows)
    write(args.out / 'canonicality_check/status.json', {'status': 'completed',
        'pass': sum(s['gate']['pass'] for s in seeds.values()) >= 2, 'seeds': seeds,
        'note': '冻结 region 上重复现有 selector 的兼容性检查，非独立的评分器改进实验。'})


def report(args):
    if args.visual_review_json:
        write(args.out / 'visual_audit/manual_review.json', json.loads(args.visual_review_json))
    pilot = read(args.out / 'pilot/summary.json')
    multi = read(args.out / 'multiseed/status.json')
    canon = read(args.out / 'canonicality_check/status.json')
    m = pilot['arms'][ARMS[3]]['combined']
    geometry_failed = any(not value for key, value in pilot['gate']['checks'].items()
                          if key.endswith(('orientation_consistency', 'period_consistency')))
    route = 'E30_B_self_supervised_canonicality' if multi.get('pass') and canon.get('pass') else (
        'canonicality_geometry_revision' if multi.get('pass') else 'stop_apacc_region_discovery')
    decision = {'pilot_region_gate_pass': pilot['gate']['pass'],
        'multiseed_region_gate_pass': multi.get('pass'), 'canonicality_compatibility_pass': canon.get('pass'),
        'median_region_count': m['median_region_count'],
        'valid_crop_coverage_dev': pilot['arms'][ARMS[3]]['dev']['valid_crop_coverage'],
        'valid_crop_coverage_stress': pilot['arms'][ARMS[3]]['stress_dev']['valid_crop_coverage'],
        **{k: m[k] for k in METRICS}, 'augmentation_stability': m['augmentation_stability'],
        'next_route': route, 'seed_summary': multi.get('seed_summary'),
        'geometry_revision_permitted': bool(geometry_failed and not pilot['gate']['pass']),
        'geometry_revision_used': read(args.out / 'aggregation_config.json')['revision'],
        'training_steps': 0, 'diffusion_generation_run': False}
    write(args.out / 'decision_summary.json', decision)
    inputs = read(args.out / 'input_manifest.json')
    frozen = {'checkpoint_unchanged': all(sha(args.source / 'A21_embedding' / ('seed%s' % s) / 'adapter.pt') == digest
              for s, digest in inputs['checkpoint_sha256'].items()),
        'frozen_source_unchanged': all(sha(Path(p)) == digest for p, digest in inputs['frozen_source_sha256'].items()),
        'normalization_unchanged': sha(args.source / 'train_normalization.npz') == inputs['normalization_sha256'],
        'feature_cache_unchanged': all(sha(args.source / 'feature_cache' / ('%03d.npz' % int(i))) == digest
                                      for i, digest in inputs['feature_cache_sha256'].items())}
    write(args.out / 'frozen_check.json', frozen)
    required = ['split_manifest.json', 'input_manifest.json', 'aggregation_config.json',
        'pilot/summary.json', 'ablations/summary.json', 'multiseed/status.json',
        'canonicality_check/status.json', 'visual_audit/selection.json', 'decision_summary.json', 'frozen_check.json']
    for arm in ARMS:
        required.append('pilot/' + arm + '/rows.json')
        required += ['pilot/' + arm + '/%03d_regions.npz' % i for i in range(64, 96)]
    required += ['pilot/visualizations/%03d_audit.png' % i for i in range(64, 96)]
    for arm in ARMS[2:]:
        required += ['pilot/' + arm + '/%03d_aggregation.json' % i for i in range(64, 96)]
    missing = [p for p in required if not (args.out / p).exists()]
    audit_path = args.out / 'visual_audit/manual_review.json'
    completion = {'required_missing': missing, 'numeric_artifacts_complete': not missing,
        'frozen_inputs_pass': all(frozen.values()), 'pilot_pass': pilot['gate']['pass'],
        'conditional_multiseed_status': multi['status'], 'conditional_canonicality_status': canon['status'],
        'visual_review_completed': audit_path.exists(),
        'experiment_complete': not missing and all(frozen.values()) and audit_path.exists()}
    write(args.out / 'completion_check.json', completion)
    artifact_paths = [str(p.relative_to(args.out)) for p in args.out.rglob('*') if p.is_file() and p.suffix not in ('.log', '.err', '.gz')]
    write(args.out / 'artifact_manifest.json', {'files': artifact_paths,
        'visualization_manifest': [p for p in artifact_paths if p.endswith('_audit.png')],
        'reference_unit': 'reference image', 'bootstrap_resamples': 2000})
    with tarfile.open(args.out / 'local_review_bundle.tar.gz', 'w:gz') as archive:
        for p in args.out.rglob('*'):
            if p.is_file() and p.suffix in ('.json', '.png') and ('aggregation' not in p.name or p.name == 'aggregation_config.json'):
                archive.add(p, arcname=str(p.relative_to(args.out)))
    print(json.dumps({'decision': decision, 'completion': completion}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('pilot', 'ablation', 'confirm', 'canonical', 'report', 'all'))
    parser.add_argument('--source', type=Path, default=Path('output_eval/e30_a2_affinity_20260930'))
    parser.add_argument('--out', type=Path, default=Path('output_eval/e30_a3_region_aggregation_20260930'))
    parser.add_argument('--dataset', type=Path, default=Path('/share/home/u2515283058/datasets/BF'))
    parser.add_argument('--visual-review-json', default=None)
    args = parser.parse_args()
    torch.set_num_threads(4)
    actions = {'pilot': run_pilot, 'ablation': run_ablations, 'confirm': run_confirm,
               'canonical': run_canonical, 'report': report}
    for action in actions if args.action == 'all' else (args.action,):
        actions[action](args)
