"""E30-A：训练 split 上的无标注特征、区域和裁剪可行性检查。"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from models.apacc_canonicality import crop_metrics, select_canonical_crop
from models.apacc_features import estimated_foreground, extract_dense_features, load_dino
from models.apacc_region_graph import discover_pattern_regions
from models.e29_crop_selector import select_crop_e29


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def summarize(rows):
    keys = ('contamination', 'identity', 'homogeneity', 'period_consistency', 'orientation_consistency',
            'foreground_purity', 'region_purity', 'tile_seam')
    result = {key: float(np.mean([r[key] for r in rows])) for key in keys}
    rotations = [r['rot90_success'] for r in rows if r['rot90_success'] is not None]
    result['rot90_equivariance'] = float(np.mean(rotations)) if rotations else None
    result['readable_rot90_count'] = len(rotations)
    result['count'] = len(rows)
    return result


def run(args):
    root, out = args.root.resolve(), args.out.resolve()
    records = json.loads((root / 'data/train_bf_texture.json').read_text(encoding='utf-8'))
    # 只取现有 training split；固定顺序和数量，绝不读取 E27/E29 人工标注。
    records = sorted(records, key=lambda row: hashlib.sha256(row['cloth'].encode()).hexdigest())[:args.count]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, model_sha = load_dino(device, out / 'dinov2_vits14_pretrain.pth')
    arms = ('A0_e29_rule', 'A1_dino', 'A2_geometry', 'A3_combined', 'A4_canonicality')
    rows = {arm: [] for arm in arms}
    regions_folder = out / 'A_feature_feasibility/regions'
    crops_folder = out / 'A_feature_feasibility/crops'
    regions_folder.mkdir(parents=True, exist_ok=True)
    crops_folder.mkdir(parents=True, exist_ok=True)
    failures = []
    for n, record in enumerate(records):
        path = args.dataset / 'training' / record['cloth']
        try:
            image = Image.open(path).convert('RGB').resize((256, 256))
            mask = estimated_foreground(image)
            features = extract_dense_features(image, mask, model, device)
            proposals = {}
            baseline_box, _ = select_crop_e29(image, mask)
            baseline_metrics = crop_metrics(image, mask, mask, baseline_box)
            rows[arms[0]].append(dict(baseline_metrics, image=record['cloth'], box=baseline_box))
            for arm, mode in zip(arms[1:], ('appearance', 'geometry', 'combined', 'combined')):
                if mode not in proposals:
                    proposals[mode] = discover_pattern_regions(features, mask, mode)
                regions, labels = proposals[mode]
                if arm == 'A4_canonicality':
                    selected = select_canonical_crop(image, regions, mask)
                elif regions:
                    region_index = int(np.argmax([r['area_ratio'] for r in regions]))
                    box, _ = select_crop_e29(image, regions[region_index]['mask'])
                    selected = {'region': region_index, 'box': box,
                                'metrics': crop_metrics(image, mask, regions[region_index]['mask'], box)}
                else:
                    selected = None
                if selected is None:
                    raise ValueError('没有满足 95% 前景和 90% 区域纯度的候选裁剪')
                metrics = selected['metrics']
                rows[arm].append(dict(metrics, image=record['cloth'], box=selected['box'],
                                      region_count=len(regions),
                                      coverage=float((labels >= 0).sum() / max((features['occupancy'] >= .5).sum(), 1)),
                                      region_homogeneity=float(np.mean([r['homogeneity'] for r in regions])),
                                      orientation_consistency=float(np.mean([r['orientation_consistency'] for r in regions])),
                                      region_period_consistency=float(np.mean([r['period_consistency'] for r in regions]))))
                if n < args.previews:
                    crop = image.crop(selected['box'])
                    crop.save(crops_folder / f'{n:03d}_{arm}.png')
                    palette = np.array([[225, 225, 225], [230, 75, 70], [80, 140, 220],
                                        [55, 180, 120], [210, 145, 50], [130, 90, 195]], np.uint8)
                    colored = palette[np.maximum(labels, 0) % len(palette)]
                    colored[labels < 0] = 255
                    Image.fromarray(cv2.resize(colored, image.size, interpolation=cv2.INTER_NEAREST)).save(
                        regions_folder / f'{n:03d}_{arm}.png')
            print('[E30 A]', n + 1, '/', len(records), record['cloth'], flush=True)
        except Exception as exc:
            failures.append({'image': record['cloth'], 'error': str(exc)})
            print('[E30 A failed]', record['cloth'], str(exc), flush=True)
    summary = {arm: summarize(value) for arm, value in rows.items() if value}
    base, chosen = summary.get(arms[0]), summary.get(arms[-1])
    checks = {}
    if base and chosen and len(rows[arms[-1]]) >= max(8, len(records) * .8):
        checks = {'contamination_20pct_lower': chosen['contamination'] <= .8 * base['contamination'],
                  'period_not_lower': chosen['period_consistency'] >= base['period_consistency'],
                  'identity_drop_at_most_0_02': chosen['identity'] >= base['identity'] - .02,
                  'homogeneity_drop_at_most_0_02': chosen['homogeneity'] >= base['homogeneity'] - .02,
                  'rot90_at_least_80pct': chosen['rot90_equivariance'] is not None and
                                          chosen['rot90_equivariance'] >= .80,
                  'orientation_not_lower': chosen['orientation_consistency'] >= base['orientation_consistency']}
    passed = bool(checks) and all(checks.values())
    report = {'stage': 'A', 'split': 'BF/training', 'requested': len(records),
              'completed': len(rows[arms[-1]]), 'failures': failures,
              'dino_model': 'dinov2_vits14', 'dino_sha256': model_sha,
              'frozen': True, 'uses_manual_labels': False, 'summary': summary,
              'gate_checks': checks, 'gate_pass': passed,
              'metric_note': 'contamination=estimated foreground 3px erosion; identity=shifted descriptor cosine; orientation/period=E26 quadrant geometry consistency'}
    write(out / 'A_feature_feasibility/report.json', report)
    write(out / 'A_feature_feasibility/rows.json', rows)
    write(out / 'decision_summary.json', {'feature_feasibility_pass': passed,
          'canonicality_pass': None, 'region_causal_pass': None, 'crop_causal_pass': None,
          'full_apacc_pass': None, 'confirmation_pass': None,
          'next_route': 'self_supervised_canonicality' if passed else 'revise_frozen_features_or_affinity'})
    write(out / 'artifact_manifest.json', {'stage_a_report': 'A_feature_feasibility/report.json',
          'stage_a_rows': 'A_feature_feasibility/rows.json',
          'dino_sha256': model_sha})
    print('[E30 A gate]', passed, checks, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--dataset', type=Path, default=Path('/share/home/u2515283058/datasets/BF'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--count', type=int, default=64)
    parser.add_argument('--previews', type=int, default=8)
    run(parser.parse_args())
