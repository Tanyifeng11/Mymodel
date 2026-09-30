"""Image-adaptive supercell graph partition with prototype coherence check."""

import cv2
import numpy as np


def segment(features, embedding=None):
    # 4x4 cells of the 64x64 field give 16x16 local supercells.
    def small(a):
        return cv2.resize(a.astype(np.float32), (16, 16), interpolation=cv2.INTER_AREA)

    app = small(features['appearance'] if embedding is None else embedding)
    app /= np.maximum(np.linalg.norm(app, axis=-1, keepdims=True), 1e-8)
    geo, color = small(features['geometry']), small(features['color'])
    fft, occ = small(features['fft']), small(features['occupancy'])
    valid = occ >= .5
    n = 256
    parent, area = np.arange(n), np.ones(n, np.float32)
    vectors = [a.reshape(n, -1).copy() for a in (app, geo, color, fft)]
    sums = [a.copy() for a in vectors]

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def distance(i, j, source):
        a, g, c, f = (v[i] / area[i] for v in source)
        b, h, d, k = (v[j] / area[j] for v in source)
        sim = np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-8)
        conf = min(g[3], h[3])
        return (0.50 * (1 - sim) + 0.20 * conf * (1 - np.dot(g[:2], h[:2])) / 2 +
                0.10 * conf * min(abs(g[2] - h[2]), 2) +
                0.10 * min(np.linalg.norm(c - d), 1) +
                0.10 * min(np.linalg.norm(f - k), 1))

    edges = []
    for y in range(16):
        for x in range(16):
            if not valid[y, x]:
                continue
            i = y * 16 + x
            for yy, xx in ((y + 1, x), (y, x + 1)):
                if yy < 16 and xx < 16 and valid[yy, xx]:
                    j = yy * 16 + xx
                    edges.append((distance(i, j, vectors), i, j))
    if not edges:
        return np.full((64, 64), -1, np.int32), {'threshold': None, 'regions': 0}
    values = np.array([e[0] for e in edges])
    med = float(np.median(values))
    mad = float(np.median(abs(values - med)))
    # Threshold follows each image's local edge distribution; prototype check
    # limits chaining across a large pattern discontinuity.
    threshold = med + .60 * max(mad, .01)
    for edge_cost, i, j in sorted(edges):
        a, b = root(i), root(j)
        if a == b or edge_cost > threshold:
            continue
        if distance(a, b, sums) > threshold * 1.15:
            continue
        if area[a] < area[b]:
            a, b = b, a
        parent[b] = a
        for total in sums:
            total[a] += total[b]
        area[a] += area[b]
    roots = np.full((16, 16), -1, np.int32)
    for y, x in zip(*np.where(valid)):
        roots[y, x] = root(y * 16 + x)
    labels = np.full((16, 16), -1, np.int32)
    unique = sorted(set(roots[valid]), key=lambda r: -area[r])
    minimum = max(1, int(valid.sum() * .015))
    for r in unique:
        if area[r] >= minimum:
            labels[roots == r] = int((labels.max() + 1))
    result = cv2.resize(labels, (64, 64), interpolation=cv2.INTER_NEAREST)
    return result, {'threshold': threshold, 'median_edge': med, 'mad_edge': mad,
                    'regions': int(labels.max() + 1), 'foreground_coverage':
                    float((labels >= 0).sum() / max(valid.sum(), 1))}
