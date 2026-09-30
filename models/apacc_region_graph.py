"""E30 无标注局部纹样区域：邻接亲和力与连通域。"""

import cv2
import numpy as np


WEIGHTS = {'appearance': .40, 'orientation': .20, 'frequency': .15,
           'color': .10, 'self_similarity': .15}


def _affinity(a, b, mode):
    app = float(np.dot(a[0], b[0]))
    confidence = float(min(a[1][3], b[1][3]))
    orientation = (1 + float(np.dot(a[1][:2], b[1][:2]))) / 2
    frequency = np.exp(-abs(float(a[1][2] - b[1][2])))
    color = np.exp(-4 * float(np.linalg.norm(a[2] - b[2])))
    selfsim = np.exp(-12 * abs(float(a[3] - b[3])))
    if mode == 'appearance':
        return app
    if mode == 'geometry':
        return ((.50 * orientation + .30 * frequency) * confidence + .20 * selfsim) / (.80 * confidence + .20)
    return (.40 * app + (.20 * orientation + .15 * frequency) * confidence +
            .10 * color + .15 * selfsim) / (.65 + .35 * confidence)


def discover_pattern_regions(features, mask, mode='combined'):
    """固定阈值建图后取连通域；小于前景 2% 的区域标记不确定。"""
    assert mode in ('appearance', 'geometry', 'combined')
    app, geo = features['appearance'], features['geometry']
    color, selfsim = features['color'], features['self_similarity']
    valid = features['occupancy'] >= .5
    size = valid.shape[0]
    threshold = {'appearance': .89, 'geometry': .66, 'combined': .77}[mode]
    parent = np.arange(size * size)

    def root(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    for y in range(size):
        for x in range(size):
            if not valid[y, x]:
                continue
            a = app[y, x], geo[y, x], color[y, x], selfsim[y, x]
            for ny, nx in ((y + 1, x), (y, x + 1)):
                if ny >= size or nx >= size or not valid[ny, nx]:
                    continue
                b = app[ny, nx], geo[ny, nx], color[ny, nx], selfsim[ny, nx]
                if _affinity(a, b, mode) >= threshold:
                    parent[root(y * size + x)] = root(ny * size + nx)
    labels = np.full((size, size), -1, np.int32)
    groups = {}
    for y, x in zip(*np.where(valid)):
        groups.setdefault(root(y * size + x), []).append((y, x))
    regions = []
    minimum = max(1, int(valid.sum() * .02))
    for cells in sorted(groups.values(), key=len, reverse=True):
        if len(cells) < minimum:
            continue
        index = len(regions)
        for y, x in cells:
            labels[y, x] = index
        area = cv2.resize((labels == index).astype(np.uint8),
                          (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool) & mask
        if not area.any():
            continue
        ys, xs = np.where(area)
        g = np.array([geo[y, x] for y, x in cells])
        v = np.array([app[y, x] for y, x in cells])
        regions.append({'mask': area, 'bbox': (int(xs.min()), int(ys.min()),
                                              int(xs.max() + 1), int(ys.max() + 1)),
                        'area_ratio': float(len(cells) / max(valid.sum(), 1)),
                        'homogeneity': float(np.linalg.norm(v.mean(0))),
                        'orientation_consistency': float(np.linalg.norm(g[:, :2].mean(0))),
                        'period_consistency': float(np.exp(-g[:, 2].std()))})
    return regions, labels
