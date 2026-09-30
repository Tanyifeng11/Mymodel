"""空间连通区域与逻辑纹样组分离；完整组对复核防止相似度链式误归并。"""

import heapq

import numpy as np

from models.apacc_region_prototype import (prototype, similarities, merge_prototypes,
                                          compute_merge_coherence_delta)


def _path_barrier(state, start, end):
    # minimax 路径：任意前景路径上最弱的强边界链。
    heap, best = [(0., start)], {start: 0.}
    while heap:
        value, region = heapq.heappop(heap)
        if region == end:
            return value
        if value > best[region]:
            continue
        for other, stat in state.neighbors[region].items():
            cost = max(value, stat[0] / stat[2])
            if cost < best.get(other, float('inf')):
                best[other] = cost
                heapq.heappush(heap, (cost, other))
    # 只容许 1–2 个低置信前景节点的 gap；背景不是桥。
    labels, conf, valid = state.labels, state.nodes['conf'], state.nodes['occupancy'] >= .5
    frontier = set(state.regions[start]['ids'].tolist())
    targets = set(state.regions[end]['ids'].tolist())
    h, w = labels.shape
    for _ in range(3):
        nxt = set()
        for node in frontier:
            y, x = divmod(node, w)
            for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if not (0 <= yy < h and 0 <= xx < w):
                    continue
                index = yy * w + xx
                if index in targets:
                    return .25
                if valid[yy, xx] and labels[yy, xx] < 0 and conf[yy, xx] < .35:
                    nxt.add(index)
        frontier = nxt
    return 1.


def reconcile_pattern_groups(state):
    config = state.config
    regions = list(state.regions)
    groups = {r: r for r in regions}
    candidates, logs, lineage = [], [], []
    foreground_xy = np.argwhere(state.nodes['occupancy'] >= .5)
    diagonal = max(float(np.linalg.norm(np.ptp(foreground_xy, axis=0))), 1.) if len(foreground_xy) else 1.
    for i, a in enumerate(regions):
        for b in regions[i + 1:]:
            s = similarities(state.regions[a], state.regions[b])
            distance = float(np.linalg.norm(prototype(state.regions[a])['centroid'] -
                                           prototype(state.regions[b])['centroid']) / diagonal)
            if distance >= config['global_max_distance'] and s['z'] < config['tau_global_candidate']:
                continue
            conf = s['conf'] if config['geometry'] else 0.
            score = (.65 * s['z'] + conf * (.20 * s['theta'] + .10 * s['freq']) + .05 * s['selfsim']) / (.70 + .30 * conf)
            barrier = _path_barrier(state, a, b)
            spatial = float(np.exp(-2 * distance) * (1 - barrier))
            accepted = score >= config['tau_pattern'] and spatial >= config['tau_spatial'] and barrier < config['strong_boundary']
            merged = merge_prototypes(state.regions[a], state.regions[b])
            delta = compute_merge_coherence_delta(state.regions[a], state.regions[b], merged, config)
            logs.append({'stage': 'global', 'region_ids': [a, b], 'merge_score': score,
                'delta_coherence': float(delta), 'boundary_evidence': barrier, 'spatial_score': spatial,
                'normalized_centroid_distance': distance, 'accepted': False,
                'pair_eligible': bool(accepted), 'reason': 'candidate' if accepted else 'pattern_or_spatial_or_boundary'})
            if accepted:
                candidates.append((-score, a, b, len(logs) - 1))
    eligible = {tuple(sorted((a, b))) for _, a, b, _ in candidates}
    for _, a, b, log_index in sorted(candidates):
        ga, gb = groups[a], groups[b]
        if ga == gb:
            logs[log_index]['reason'] = 'already_grouped'
            continue
        left, right = [r for r in regions if groups[r] == ga], [r for r in regions if groups[r] == gb]
        if not all(tuple(sorted((x, y))) in eligible for x in left for y in right):
            logs[log_index]['reason'] = 'complete_group_recheck'
            continue
        for r in right:
            groups[r] = ga
        logs[log_index].update(accepted=True, reason='logical_merge')
        lineage.append({'group': ga, 'parents': [ga, gb], 'members': left + right})
    return groups, logs, lineage
