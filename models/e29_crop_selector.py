"""E29 固定规则 crop 选择器；只读取图像和给定 panel mask。"""

import cv2
import numpy as np
from PIL import Image

from models.confidence_local_correspondence import select_crop
from models.local_pattern_field import patch_geometry
from tools.e27_correspondence import descriptor


def _homogeneity(patch):
    vectors = [descriptor(patch.crop((x * patch.width // 2, y * patch.height // 2,
                                      (x + 1) * patch.width // 2,
                                      (y + 1) * patch.height // 2)))[0]
               for y in range(2) for x in range(2)]
    scores = [float(np.dot(vectors[i], vectors[j]) /
                    max(np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]), 1e-9))
              for i in range(4) for j in range(i + 1, 4)]
    return float(np.clip(np.mean(scores), 0, 1))


def select_crop_e29(image, panel_mask):
    """48/64/80/96px、stride16；分数在观察测试结果前固定。"""
    rgb = np.asarray(image.convert('RGB'))
    mask = np.asarray(panel_mask, dtype=bool)
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError('empty panel mask')
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    panel_color = np.median(rgb[mask], axis=0)
    best = None
    for size in (48, 64, 80, 96):
        if size > min(image.size):
            continue
        y_stop = min(image.height - size, int(ys.max()))
        x_stop = min(image.width - size, int(xs.max()))
        for y in range(max(0, int(ys.min())), y_stop + 1, 16):
            for x in range(max(0, int(xs.min())), x_stop + 1, 16):
                area = mask[y:y + size, x:x + size]
                occupancy = float(area.mean())
                if occupancy < .95:
                    continue
                box = (x, y, x + size, y + size)
                patch = image.crop(box)
                homogeneity = _homogeneity(patch)
                geometry = patch_geometry(patch)
                margin = float(np.clip(np.percentile(distance[y:y + size, x:x + size], 20) /
                                       (size / 4), 0, 1))
                color = float(np.clip(1 - np.abs(rgb[y:y + size, x:x + size].mean((0, 1)) -
                                               panel_color).mean() / 96, 0, 1))
                score = (.25 * homogeneity + .20 * geometry['confidence'] + .20 * margin +
                         .20 * color + .10 * size / 96 + .05 * occupancy)
                candidate = (score, box, {'score': float(score), 'homogeneity': homogeneity,
                                          'geometry_confidence': geometry['confidence'],
                                          'boundary_margin': margin,
                                          'panel_color_similarity': color,
                                          'occupancy': occupancy, 'fallback': False})
                if best is None or candidate[0] > best[0]:
                    best = candidate
    if best is None:
        box, geometry, occupancy = select_crop(image, mask)
        return tuple(box), {'score': 0., 'homogeneity': _homogeneity(image.crop(tuple(box))),
                            'geometry_confidence': geometry['confidence'],
                            'boundary_margin': 0., 'panel_color_similarity': 0.,
                            'occupancy': occupancy, 'fallback': True}
    return best[1], best[2]
