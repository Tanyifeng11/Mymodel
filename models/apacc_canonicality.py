"""E30 候选裁剪与不训练的 canonicality 评分。"""

import cv2
import numpy as np
from PIL import Image

from models.local_pattern_field import patch_geometry
from tools.e27_correspondence import descriptor


def _cos(a, b):
    return float(np.clip(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9), 0, 1))


def crop_metrics(image, foreground, region, box):
    x0, y0, x1, y1 = box
    patch = image.crop(box)
    rgb = np.asarray(patch.convert('RGB'))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    fg = float(foreground[y0:y1, x0:x1].mean())
    purity = float(region[y0:y1, x0:x1].mean())
    vectors = [descriptor(patch.crop((x * patch.width // 2, y * patch.height // 2,
                                      (x + 1) * patch.width // 2,
                                      (y + 1) * patch.height // 2)))[0]
               for y in range(2) for x in range(2)]
    homogeneity = float(np.mean([_cos(vectors[i], vectors[j])
                                 for i in range(4) for j in range(i + 1, 4)]))
    shifts = []
    for dx, dy in ((8, 0), (-8, 0), (0, 8), (0, -8)):
        moved = (x0 + dx, y0 + dy, x1 + dx, y1 + dy)
        if moved[0] >= 0 and moved[1] >= 0 and moved[2] <= image.width and moved[3] <= image.height:
            shifts.append(_cos(descriptor(patch)[0], descriptor(image.crop(moved))[0]))
    shift = float(np.mean(shifts)) if shifts else 0
    geometry = patch_geometry(patch)
    rotated = patch_geometry(patch.transpose(Image.Transpose.ROTATE_90))
    rotation_error = abs((rotated['orientation'] - geometry['orientation'] + 0) % 180 - 90)
    rot_success = bool(rotation_error <= 15) if geometry['valid'] else None
    seam = (np.abs(rgb[:, 0].astype(float) - rgb[:, -1]).mean() +
            np.abs(rgb[0].astype(float) - rgb[-1].astype(float)).mean()) / 510
    # 长直边缘是衣领、下摆和门襟污染的代理。
    edges = cv2.Canny(gray, 50, 130)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 20,
                             minLineLength=max(16, min(patch.size) // 2), maxLineGap=3)
    structural = min(1., len(lines) / 8) if lines is not None else 0.
    interior = cv2.erode(foreground.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    contamination = 1 - float(interior[y0:y1, x0:x1].mean())
    return {'foreground_purity': fg, 'region_purity': purity,
            'homogeneity': homogeneity, 'identity': shift,
            'shift_consistency': shift, 'rot90_success': rot_success,
            'rotation_error': float(rotation_error),
            'period_consistency': float(geometry['confidence']),
            'tile_seam': float(1 - seam), 'structural_edge': float(structural),
            'contamination': contamination}


def score_canonical_candidate(metrics, contamination_penalty=True):
    value = (.17 * metrics['foreground_purity'] + .15 * metrics['region_purity'] +
             .17 * metrics['homogeneity'] + .17 * metrics['shift_consistency'] +
             .12 * metrics['period_consistency'] + .10 * metrics['tile_seam'] +
             .12 * (1 - metrics['structural_edge']))
    if contamination_penalty:
        value -= .25 * metrics['contamination']
    return float(value)


def generate_canonical_candidates(image, region_mask, foreground):
    h, w = region_mask.shape
    candidates = []
    for cw, ch in ((48, 48), (64, 64), (80, 80), (96, 96), (64, 96), (96, 64)):
        if cw > w or ch > h:
            continue
        for y in range(0, h - ch + 1, 16):
            for x in range(0, w - cw + 1, 16):
                fg = foreground[y:y + ch, x:x + cw].mean()
                occupancy = region_mask[y:y + ch, x:x + cw].mean()
                if fg >= .95 and occupancy >= .90:
                    candidates.append(((x, y, x + cw, y + ch), float(occupancy)))
    return candidates


def select_canonical_crop(image, regions, foreground, contamination_penalty=True):
    candidates = []
    for index, region in enumerate(regions):
        for box, occupancy in generate_canonical_candidates(image, region['mask'], foreground):
            candidates.append((index, box, occupancy))
    # 先用前景纯度与内部距离缩小候选集，再计算较贵的纹样描述符。
    distance = cv2.distanceTransform(foreground.astype(np.uint8), cv2.DIST_L2, 5)
    candidates.sort(key=lambda item: (item[2] + .01 *
                    np.mean(distance[item[1][1]:item[1][3], item[1][0]:item[1][2]])), reverse=True)
    best = None
    for index, box, _ in candidates[:64]:
        metrics = crop_metrics(image, foreground, regions[index]['mask'], box)
        score = score_canonical_candidate(metrics, contamination_penalty)
        if best is None or score > best['score']:
            best = {'region': index, 'box': box, 'metrics': metrics, 'score': score}
    return best
