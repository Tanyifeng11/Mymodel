"""冻结节点特征的区域充分统计量；合并时不重新扫描整张网格。"""

import numpy as np


def node_features(features, embedding):
    geo = features['geometry'].astype(np.float64)
    direction = geo[..., :2]
    direction /= np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-9)
    z = embedding.astype(np.float64)
    z /= np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-9)
    yy, xx = np.indices(z.shape[:2])
    return {'z': z, 'direction': direction, 'freq': geo[..., 2],
            'conf': np.clip(geo[..., 3], 0, 1), 'color': features['color'][..., :3],
            'color_std': features['color'][..., 3:], 'selfsim': features['self_similarity'],
            'occupancy': features['occupancy'], 'boundary_dist': features['boundary_distance'],
            'xy': np.stack([xx, yy], -1).astype(float)}


def build_region_prototype(nodes, region_mask):
    ids = np.flatnonzero(region_mask)
    values = {k: v.reshape((-1,) + v.shape[2:])[ids] for k, v in nodes.items()}
    c = values['conf']
    return {'ids': ids, 'area': len(ids), 'z_sum': values['z'].sum(0),
            'z_sq': float((values['z'] ** 2).sum()),
            'dir_sum': (values['direction'] * c[:, None]).sum(0), 'conf_sum': float(c.sum()),
            'freq_sum': float((values['freq'] * c).sum()),
            'freq_sq': float((values['freq'] ** 2 * c).sum()),
            'color_sum': values['color'].sum(0), 'color_sq': float((values['color'] ** 2).sum()),
            'color_std_sum': values['color_std'].sum(0),
            'selfsim_sum': float(values['selfsim'].sum()), 'xy_sum': values['xy'].sum(0),
            'occupancy_sum': float(values['occupancy'].sum()),
            'boundary_dist_sum': float(values['boundary_dist'].sum())}


def merge_prototypes(a, b):
    return {k: np.concatenate([a[k], b[k]]) if k == 'ids' else a[k] + b[k] for k in a}


def prototype(p):
    n, weight = max(p['area'], 1), max(p['conf_sum'], 1e-9)
    z, color = p['z_sum'] / n, p['color_sum'] / n
    freq = p['freq_sum'] / weight
    direction = p['dir_sum'] / weight
    return {'mean_z': z, 'var_z': max(0., p['z_sq'] / n - float(z @ z)),
            'orientation_mean': float(np.arctan2(direction[1], direction[0]) / 2),
            'direction': direction / max(np.linalg.norm(direction), 1e-9),
            'orientation_var': float(1 - min(np.linalg.norm(direction), 1.)),
            'frequency_mean': freq,
            'frequency_std': float(np.sqrt(max(0., p['freq_sq'] / weight - freq * freq))),
            'color_mean': color, 'color_std': p['color_std_sum'] / n,
            'color_var': max(0., p['color_sq'] / n - float(color @ color)),
            'self_similarity_mean': p['selfsim_sum'] / n,
            'geometry_confidence_mean': p['conf_sum'] / n,
            'foreground_coverage': p['occupancy_sum'] / n,
            'boundary_distance_mean': p['boundary_dist_sum'] / n,
            'area': n, 'centroid': p['xy_sum'] / n}


def coherence(p, config):
    q = prototype(p)
    conf = q['geometry_confidence_mean'] if config['geometry'] else 0.
    return (.55 * q['var_z'] + conf * (.25 * q['orientation_var'] +
            .15 * q['frequency_std']) + .05 * q['color_var'])


def compute_merge_coherence_delta(a, b, merged, config):
    return coherence(merged, config) - (a['area'] * coherence(a, config) +
            b['area'] * coherence(b, config)) / merged['area']


def similarities(a, b):
    a, b = prototype(a), prototype(b)
    z = float(a['mean_z'] @ b['mean_z'] / max(np.linalg.norm(a['mean_z']) *
              np.linalg.norm(b['mean_z']), 1e-9))
    return {'z': z, 'theta': float((1 + a['direction'] @ b['direction']) / 2),
            'freq': float(np.exp(-abs(a['frequency_mean'] - b['frequency_mean']))),
            'color': float(np.exp(-np.linalg.norm(a['color_mean'] - b['color_mean']) / .25)),
            'selfsim': float(np.exp(-12 * abs(a['self_similarity_mean'] - b['self_similarity_mean']))),
            'conf': min(a['geometry_confidence_mean'], b['geometry_confidence_mean'])}
