"""汇总实际 mask 和同 latent 传播探针，输出人工核对图，不自动推断 mask 正确性。"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)


def overlay(image, mask, color):
    mask = np.asarray(Image.fromarray(mask.astype(np.uint8)*255).resize(image.size, Image.Resampling.NEAREST)) > 127
    array = np.asarray(image).copy()
    array[mask] = (array[mask]*0.5+np.asarray(color)*0.5).astype(np.uint8)
    return Image.fromarray(array)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--expected-count', type=int, default=100)
    args = p.parse_args()
    root, out = Path(args.run_dir), Path(args.output_dir)
    files = sorted(root.glob('token/*/propagation_probe/probe.json'))
    if len(files) != args.expected_count:
        raise ValueError(f'探针不完整：{len(files)}/{args.expected_count}')
    out.mkdir(parents=True, exist_ok=True)
    (out/'review').mkdir(exist_ok=True)
    all_rows, reviews = [], []
    for file in files:
        sample = file.parent.parent
        sid = sample.name.split('_')[0]
        rows = json.loads(file.read_text())
        if [r['step_index'] for r in rows] != [0, 5, 15, 25, 49]:
            raise ValueError(f'{sid} 探针步数不完整')
        all_rows.extend([{'sample_id': sid, **r} for r in rows])
        with Image.open(sample/f'comparison_{sid}.png') as im:
            w, h = im.width//3, im.height
            sketch = im.convert('RGB').crop((0, 0, w, h))
            generated = im.convert('RGB').crop((2*w, 0, 3*w, h))
        canvas = Image.new('RGB', (4*w, len(rows)*(h+25)), 'white')
        draw = ImageDraw.Draw(canvas)
        for index, r in enumerate(rows):
            with np.load(file.parent/f"step_{r['step_index']:02d}.npz") as data:
                mask, garment = data['injection_mask'], data['garment_mask']
                delta = data['noise_delta_rms']
                maximum = float(delta.max())
                heat = np.zeros((*delta.shape, 3), dtype=np.uint8)
                heat[..., 0] = np.clip(delta/max(maximum, 1e-12)*255, 0, 255).astype(np.uint8)
                tiles = [overlay(sketch, garment, [0, 255, 0]),
                         overlay(sketch, mask, [255, 0, 255]),
                         overlay(generated, mask, [255, 0, 255]),
                         Image.fromarray(heat).resize((w, h), Image.Resampling.NEAREST)]
                labels = ['sketch / garment mask', 'sketch / actual injection', 'result / actual injection', f'noise delta max={maximum:.3g}']
                y = index*(h+25)
                for col, (tile, label) in enumerate(zip(tiles, labels)):
                    canvas.paste(tile, (col*w, y+25))
                    draw.text((col*w+3, y+5), f"step {r['step_index']} | {label}", fill='black')
        canvas.save(out/'review'/f'{sid}.png')
        reviews.append({'sample_id': sid, 'review_image': f'review/{sid}.png',
                        'mask_covers_true_background': '', 'notes': ''})
    write_csv(out/'per_step.csv', all_rows)
    write_csv(out/'mask_review.csv', reviews)
    summary = []
    for step in [0, 5, 15, 25, 49]:
        group = [r for r in all_rows if r['step_index']==step]
        row = {'step_index': step, 'samples': len(group)}
        for key in group[0]:
            if key in ('sample_id', 'step_index'):
                continue
            values = [r[key] for r in group if r[key] is not None]
            row[key+'_mean'] = float(np.mean(values)) if values else None
        summary.append(row)
    write_csv(out/'summary.csv', summary)
    print(f'已汇总 {len(files)} 个样本：{out}。mask 真值需人工判断；noise delta 为 CFG 前条件预测差。')


if __name__ == '__main__':
    main()
