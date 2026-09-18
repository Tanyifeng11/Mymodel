"""已有图像的轮廓代理评估：固定草图 mask、白底分割、双向距离及边界带错误。"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

MODES = ('baseline', 'boundary', 'weaken_texture_matched', 'strengthen_sketch_matched')


def contour(mask):
    return mask & ~ndi.binary_erosion(mask, border_value=0)


def silhouette(rgb, threshold):
    # 针对白底深色服装；保留最大连通块并填内部孔洞，避免把纹理当轮廓。
    labels, count = ndi.label(np.min(rgb, axis=2) < threshold)
    if count == 0:
        raise ValueError('未提取到服装前景')
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return ndi.binary_fill_holes(labels == sizes.argmax())


def measure(reference, predicted, margin=16):
    if reference.shape != predicted.shape:
        raise ValueError('原尺寸不一致，禁止隐式缩放')
    valid = np.ones(reference.shape, bool)
    if margin:
        valid[:margin] = valid[-margin:] = False
        valid[:, :margin] = valid[:, -margin:] = False
    ref_edge, pred_edge = contour(reference), contour(predicted)
    # 距离变换前去掉画面截断边，避免把图框误当服装边界。
    ref_edge &= valid
    pred_edge &= valid
    if not ref_edge.any() or not pred_edge.any():
        raise ValueError('有效轮廓为空')
    to_ref = ndi.distance_transform_edt(~ref_edge)
    to_pred = ndi.distance_transform_edt(~pred_edge)
    recall_dist, precision_dist = to_pred[ref_edge], to_ref[pred_edge]
    stats = dict(ref_to_gen_mean_px=float(recall_dist.mean()), gen_to_ref_mean_px=float(precision_dist.mean()),
                 symmetric_distance_px=float((recall_dist.mean()+precision_dist.mean())/2),
                 ref_to_gen_p95_px=float(np.percentile(recall_dist, 95)),
                 gen_to_ref_p95_px=float(np.percentile(precision_dist, 95)))
    for tolerance in (1, 2, 4):
        precision = float(np.mean(precision_dist <= tolerance))
        recall = float(np.mean(recall_dist <= tolerance))
        stats[f'contour_f1_{tolerance}px'] = 2*precision*recall/(precision+recall) if precision+recall else 0.
    for radius in (4, 8, 16):
        band = (to_ref <= radius) & valid
        stats[f'band_{radius}px_error'] = float(np.mean((reference ^ predicted)[band]))
        stats[f'band_{radius}px_missing'] = float(np.mean((reference & ~predicted)[band]))
        stats[f'band_{radius}px_excess'] = float(np.mean((~reference & predicted)[band]))
    return stats, (to_ref <= 8) & valid, ref_edge, pred_edge


def evaluate(root):
    root = Path(root)
    out = root/'report/local_contours'
    out.mkdir(parents=True, exist_ok=True)
    report = dict(reference='saved sketch mask; automatic proxy, not manual ground truth',
                  foreground='min RGB < threshold; largest connected component; fill holes',
                  thresholds=[235, 245, 250], frame_margins=[8, 16],
                  distance_unit='original image pixel', samples={})
    for sid in (5, 18, 23):
        name = f'sample_{sid:06d}'
        folders = {m: next((root/name/m/'e5/token').glob(f'{sid:06d}_*')) for m in MODES}
        reference = np.asarray(Image.open(next(folders['baseline'].glob('*_mask.png'))).convert('L')) > 127
        comp = Image.open(folders['baseline']/f'comparison_{sid:06d}.png').convert('RGB')
        height, width = reference.shape
        if comp.size != (width*3, height):
            raise ValueError('草图拼图尺寸与 mask 不一致')
        sketch = np.asarray(comp.crop((0, 0, width, height)))
        sample = {}
        canvas = Image.new('RGB', (width*5, height+55), 'white')
        draw = ImageDraw.Draw(canvas)
        ref_view = sketch.copy()
        ref_view[contour(reference)] = [0, 180, 255]
        canvas.paste(Image.fromarray(ref_view), (0, 55))
        draw.text((5, 5), 'Sketch + reference contour', fill='black')
        for col, mode in enumerate(MODES, 1):
            other_mask = np.asarray(Image.open(next(folders[mode].glob('*_mask.png'))).convert('L')) > 127
            if not np.array_equal(reference, other_mask):
                raise ValueError('组间草图 mask 不一致')
            rgb = np.asarray(Image.open(folders[mode]/f'generated_{sid:06d}.png').convert('RGB'))
            sample[mode] = {}
            for threshold in report['thresholds']:
                predicted = silhouette(rgb, threshold)
                for margin in report['frame_margins']:
                    stats, band, ref_edge, pred_edge = measure(reference, predicted, margin)
                    key = f't{threshold}_margin{margin}'
                    sample[mode][key] = dict(metrics=stats)
                    if mode != 'baseline':
                        sample[mode][key]['delta_vs_baseline'] = {
                            k: v-sample['baseline'][key]['metrics'][k] for k, v in stats.items()}
                    if threshold == 245 and margin == 16:
                        overlay = rgb.copy()
                        overlay[band & reference & ~predicted] = [255, 50, 50]
                        overlay[band & ~reference & predicted] = [255, 160, 0]
                        overlay[ref_edge] = [0, 180, 255]
                        overlay[pred_edge] = [180, 0, 255]
                        canvas.paste(Image.fromarray(overlay), (col*width, 55))
                        draw.text((col*width+5, 5), mode, fill='black')
                        draw.text((col*width+5, 22), f"distance={stats['symmetric_distance_px']:.3f}px F1@2={stats['contour_f1_2px']:.4f}", fill='black')
            sample[mode]['robustness'] = {
                k: sum(sample[mode][config]['delta_vs_baseline'][k] < 0 for config in sample[mode] if config.startswith('t'))
                for k in ('symmetric_distance_px', 'band_8px_error')} if mode != 'baseline' else {}
        draw.text((5, 38), 'Cyan: reference; purple: generated; red: missing; orange: excess (8px band). Frame excluded: 16px.', fill='black')
        canvas.save(out/f'{name}.png')
        report['samples'][name] = sample
    (out/'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(out)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    evaluate(parser.parse_args().run_dir)
