"""生成 E14 受控几何参考图；这是诊断集，不是自然服装测试集。"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tools.e14_pattern_probe import FIELDS, pixel_hash, image_baselines, write_json


def pattern_mask(kind, size, cycles, angle, phase, fraction=.25):
    y, x = np.mgrid[:size, :size].astype(np.float64)
    radians = np.deg2rad(angle)
    u = (x * np.cos(radians) + y * np.sin(radians)) * cycles / size + phase
    v = (-x * np.sin(radians) + y * np.cos(radians)) * cycles / size + phase
    du, dv = np.abs((u + .5) % 1 - .5), np.abs((v + .5) % 1 - .5)
    # 距离场对应条带、交叉格线和圆点；固定前景像素总数，保证原图直方图严格一致。
    score = {'stripe': du, 'plaid': np.minimum(du, dv), 'dots': du ** 2 + dv ** 2}[kind]
    n = int(round(size * size * fraction))
    order = np.argsort(score.ravel(), kind='stable')
    mask = np.zeros(size * size, dtype=bool)
    mask[order[:n]] = True
    return mask.reshape(size, size)


def generate(output):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'images').mkdir()
    rows, histograms = [], []
    preview = Image.new('RGB', (8 * 128, 6 * 148), 'white')
    draw = ImageDraw.Draw(preview)
    for kind in ('stripe', 'plaid', 'dots'):
        for cycles in (3, 5, 8, 12):
            for angle in (0, 30):
                for phase in (0., .27):
                    mask = pattern_mask(kind, 256, cycles, angle, phase)
                    rgb = np.where(mask[..., None], np.array([32, 32, 32]), np.array([224, 224, 224])).astype('uint8')
                    image = Image.fromarray(rgb)
                    name = f'{kind}_f{cycles}_a{angle}_p{int(phase*100):02d}'
                    texture = f'images/{name}.png'
                    image.save(out / texture)
                    row = dict.fromkeys(FIELDS, '')
                    row.update(sample_id=name, texture=texture, caption='a garment', pattern=kind,
                               color_group='gray32_gray224_fraction025', source_group=f'{kind}_f{cycles}',
                               confirmed='1', pattern_visible='1', pixel_sha256=pixel_hash(image),
                               notes='程序生成标签；同类别同频率的角度/相位变体同组；非自然来源独立样本')
                    rows.append(row)
                    histograms.append(image_baselines(image)['image__color_hist'])
                    j = len(rows) - 1
                    preview.paste(image.resize((128, 128)), ((j % 8)*128, (j // 8)*148))
                    draw.text(((j % 8)*128+2, (j // 8)*148+128), name, fill='black')
    assert all(np.array_equal(histograms[0], h) for h in histograms)
    with (out / 'labels.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    preview.save(out / 'preview.png')
    write_json(out / 'generation_audit.json', dict(samples=len(rows), template_groups=12,
        size=[256, 256], foreground_rgb=[32]*3, background_rgb=[224]*3, foreground_pixels=16384,
        foreground_fraction=.25, raw_histograms_exact_equal=True,
        neutral_caption='a garment',
        limits=['严格匹配仅保证原图直方图；CLIP裁剪、插值与CNN缩放可重新引入直方图差异。',
                '标签由生成规则定义，confirmed不代表人工标注。',
                '每类4个频率模板组；相位和旋转不是独立来源。',
                '原caption与中性文本相同；本实验不能评估文本差异。',
                '探针成功只验证受控粗图案可读性，不能代替真实布料验证。']))
    print(f'生成 {len(rows)} 张参考图：{out}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    generate(parser.parse_args().output)
