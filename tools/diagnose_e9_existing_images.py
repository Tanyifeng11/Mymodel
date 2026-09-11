"""现有图像离线审计：背景变化、分割敏感性和外轮廓证据；不生成图像。

border-only 是去掉原分割器亮度 OR 条件的敏感性对照，不是真值分割。
轮廓距离使用草图 mask 外边界与生成图梯度边缘，也不能替代人工判断。
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import distance_transform_cdt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from garment_mask_utils import build_sketch_garment_mask, estimate_cloth_foreground_mask, mask_backend_info


def morph(mask, size, grow=True):
    filt = ImageFilter.MaxFilter(size) if grow else ImageFilter.MinFilter(size)
    return np.asarray(Image.fromarray(mask.astype(np.uint8)*255).filter(filt)) > 127


def mean(values, mask):
    return float(values[mask].mean()) if mask.any() else None


def iou(a, b):
    return float((a & b).sum()/max(1, (a | b).sum()))


def edge_map(image):
    gray = np.asarray(image.convert('L'), dtype=float)/255
    dy, dx = np.gradient(gray)
    return np.hypot(dx, dy) > 0.06


def contour_distance(boundary, edges, radius=12):
    # 未在 radius 像素内找到边缘者记 radius+1；不把缺失边缘丢弃。
    if not edges.any():
        return float(radius+1) if boundary.any() else None
    distances = np.minimum(distance_transform_cdt(~edges, metric='chessboard'), radius+1)
    return mean(distances, boundary)


def measure(image, baseline, garment):
    rgb = np.asarray(image, dtype=float)
    base = np.asarray(baseline, dtype=float)
    outside = ~morph(garment, 19)
    boundary = garment & ~morph(garment, 3, False)
    # 去掉画布边缘的截断轮廓，避免把裁切边界当成服装边界。
    boundary[:2] = boundary[-2:] = False
    boundary[:, :2] = boundary[:, -2:] = False
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    bg = np.median(border, axis=0)
    distance = np.linalg.norm(rgb-bg, axis=2)
    dark = rgb.mean(axis=2) < 235
    border_only = distance > 18
    old_raw = border_only | dark
    old_image, _ = estimate_cloth_foreground_mask(image, *image.size)
    old_mask = np.asarray(old_image) > 127
    edges = edge_map(image)
    values = {
        'background_mae_vs_off': mean(np.abs(rgb-base).mean(axis=2)/255, outside),
        'background_white_distance': mean(np.abs(rgb-255).mean(axis=2)/255, outside),
        'background_dark_fraction': mean(dark.astype(float), outside),
        'background_saturation': mean((rgb.max(2)-rgb.min(2))/np.maximum(rgb.max(2), 1), outside),
        'background_old_mask_fraction': mean(old_mask.astype(float), outside),
        'background_brightness_only_fraction': mean((dark & ~border_only).astype(float), outside),
        'old_processed_iou': iou(old_mask, garment),
        'old_raw_iou': iou(old_raw, garment),
        'border_only_raw_iou': iou(border_only, garment),
        'outline_edge_recall_3px': mean(morph(edges, 7).astype(float), boundary),
        'outline_edge_distance_capped_px': contour_distance(boundary, edges),
        'outside_area': float(outside.mean()),
    }
    return values, old_mask, border_only, boundary


def panel(image, mask, boundary):
    rgb = np.asarray(image).copy()
    rgb[mask] = (rgb[mask]*0.5 + np.array([255, 0, 255])*0.5).astype(np.uint8)
    rgb[boundary] = [0, 255, 0]
    return Image.fromarray(rgb)


def save_review(path, images, results):
    width, height = next(iter(images.values())).size
    canvas = Image.new('RGB', (width*3, (height+30)*len(images)), 'white')
    draw = ImageDraw.Draw(canvas)
    for row, (name, image) in enumerate(images.items()):
        _, old, alt, boundary = results[name]
        for col, (label, tile) in enumerate([
            ('image', panel(image, np.zeros_like(old), boundary)),
            ('original mask', panel(image, old, boundary)),
            ('border-only raw (NOT ground truth)', panel(image, alt, boundary)),
        ]):
            y = row*(height+30)
            canvas.paste(tile, (col*width, y+30))
            draw.text((col*width+4, y+6), name+' | '+label, fill='black')
    canvas.save(path)


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-root', required=True, help='包含三组条件的已完成作业目录')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--expected-count', type=int, default=100)
    parser.add_argument('--variants', default='alpha_000,alpha_100,window_early')
    args = parser.parse_args()
    root, out = Path(args.eval_root), Path(args.output_dir)
    names = args.variants.split(',')
    if names[0] != 'alpha_000':
        raise ValueError('第一个条件必须为 alpha_000 基线')
    tables, paths = {}, {}
    for name in names:
        folder = root/name/'e9_b_diagnosis'
        rows = json.loads((folder/'metrics_per_sample.json').read_text(encoding='utf-8'))
        tables[name] = {str(r['sample_id']): r for r in rows}
        assert len(rows) == len(tables[name]) == args.expected_count, name+' 样本数不完整'
        paths[name] = {}
        for sample_id in tables[name]:
            found = list((folder/'token').glob('*/generated_'+sample_id+'.png'))
            if len(found) != 1:
                raise ValueError(f'{name}/{sample_id} 需要唯一生成图，找到 {len(found)} 个')
            paths[name][sample_id] = found[0]
    ids = sorted(tables[names[0]])
    for name in names[1:]:
        assert sorted(tables[name]) == ids, '样本集合不同'
        for sid in ids:
            for field in ['dataset_index', 'generation_seed', 'prompt']:
                assert tables[name][sid][field] == tables[names[0]][sid][field], (name, sid, field)
    out.mkdir(parents=True, exist_ok=True)
    (out/'review').mkdir(exist_ok=True)
    (out/'audit_config.json').write_text(json.dumps({
        'eval_root': str(root.resolve()), 'expected_count': args.expected_count,
        'mask_backend': mask_backend_info(), 'background_exclusion_radius_px': 9,
        'outline_tolerance_px': 3, 'edge_gradient_threshold': 0.06,
        'bootstrap_samples': 10000, 'bootstrap_seed': 42,
        'interpretation': '替代分割和边缘距离都是诊断代理，不是真值；核对 previous_iou 与 old_processed_iou 的后端差异。',
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    all_rows = []
    for sid in ids:
        images = {n: Image.open(paths[n][sid]).convert('RGB') for n in names}
        size = images[names[0]].size
        assert all(im.size == size for im in images.values()), '图像尺寸不同'
        # 比较图第一栏就是当次推理使用的草图，不依赖服务器绝对数据路径。
        sketches = []
        for n in names:
            comparison = Image.open(paths[n][sid].parent/('comparison_'+sid+'.png')).convert('RGB')
            assert comparison.size == (size[0]*3, size[1]), '比较图应为草图/纹理/生成三栏'
            sketches.append(comparison.crop((0, 0, *size)))
        assert all(np.array_equal(np.asarray(sketches[0]), np.asarray(s)) for s in sketches[1:]), '草图不同'
        mask, info = build_sketch_garment_mask(sketches[0], *size)
        garment = np.asarray(mask) > 127
        results = {}
        for n in names:
            results[n] = measure(images[n], images['alpha_000'], garment)
            all_rows.append({'sample_id': sid, 'variant': n,
                             'sketch_mask_confidence': info.get('mask_confidence'),
                             'sketch_mask_fallback': info.get('mask_fallback'),
                             'previous_iou': tables[n][sid]['struct_iou'],
                             **results[n][0]})
        save_review(out/'review'/(sid+'.png'), images, results)
        print(f'已检查 {sid}', flush=True)
    write_csv(out/'per_sample.csv', all_rows)
    metrics = list(results[names[0]][0])
    summary = []
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(ids), size=(10000, len(ids)))
    for n in names[1:]:
        for metric in metrics:
            off = np.array([r[metric] if r[metric] is not None else np.nan for r in all_rows if r['variant']=='alpha_000'])
            current = np.array([r[metric] if r[metric] is not None else np.nan for r in all_rows if r['variant']==n])
            delta = current-off
            ci = np.nanquantile(np.nanmean(delta[indices], axis=1), [.025, .975])
            summary.append({'variant': n, 'metric': metric, 'valid_pairs': int(np.isfinite(delta).sum()),
                            'off_mean': float(np.nanmean(off)), 'variant_mean': float(np.nanmean(current)),
                            'paired_delta': float(np.nanmean(delta)), 'ci_low': float(ci[0]), 'ci_high': float(ci[1])})
    write_csv(out/'paired_summary.csv', summary)
    # 人工审阅队列按背景变化排序；保留全部样本，不自动输出“真实形变”标签。
    queue = sorted([r for r in all_rows if r['variant']!='alpha_000'],
                   key=lambda r: r['background_mae_vs_off'] or 0, reverse=True)
    write_csv(out/'review_queue.csv', [{**r, 'review_image': 'review/'+r['sample_id']+'.png',
                                      'human_background_change': '', 'human_outline_change': '',
                                      'human_segmentation_error': ''} for r in queue])
    print(f'完成：{out}；绿色为草图边界，紫色为分割前景。替代分割不是轮廓真值。')


if __name__ == '__main__':
    main()
