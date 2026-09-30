"""2×2 super-region + 动态原型队列；参数在读取 pilot 结果前固定。"""

import heapq

import numpy as np

from models.apacc_region_prototype import (build_region_prototype, node_features, prototype,
    merge_prototypes, similarities, coherence, compute_merge_coherence_delta)


DEFAULT_CONFIG = {'supercell': 2, 'tau_merge': .78, 'boundary_alpha': .08,
    'pattern_beta': .06, 'tau_coherence': .055, 'max_energy': .20,
    'geometry': True, 'boundary': True, 'coherence': True, 'cleanup': True,
    'small_area': .015, 'tau_global_candidate': .90, 'tau_pattern': .94,
    'tau_spatial': .40, 'strong_boundary': .55, 'global_max_distance': .50,
    'bootstrap_resamples': 2000, 'revision': 0}


def boundary_edges(nodes, config):
    h, w = nodes['occupancy'].shape
    valid = nodes['occupancy'] >= .5
    z, g, f, c = nodes['z'], nodes['direction'], nodes['freq'], nodes['conf']
    color = nodes['color']
    edges = []
    for y, x in zip(*np.where(valid)):
        for yy, xx in ((y + 1, x), (y, x + 1)):
            if yy >= h or xx >= w or not valid[yy, xx]:
                continue
            affinity = float(z[y, x] @ z[yy, xx])
            conf = min(c[y, x], c[yy, xx])
            orientation = float((1 - g[y, x] @ g[yy, xx]) / 2)
            frequency = float(1 - np.exp(-abs(f[y, x] - f[yy, xx])))
            lab = min(float(np.linalg.norm(color[y, x] - color[yy, xx]) / .25), 1.)
            # Lab 局部梯度是 RGB gradient 的代理，绝不当作独立 GT。
            effective_conf = conf if config['geometry'] else 0.
            evidence = .55 * (1 - affinity) + effective_conf * (.25 * orientation + .15 * frequency) + .05 * lab
            strong = affinity < .65 or (conf >= .5 and (orientation > .65 or frequency > .5))
            edges.append((y * w + x, yy * w + xx, evidence, affinity, orientation,
                          frequency, lab, bool(strong)))
    return edges


def compute_region_merge_score(a, b, boundary, config):
    s = similarities(a, b)
    confidence = s['conf'] if config['geometry'] else 0.
    boundary = boundary if config['boundary'] else 0.
    score = (.45 * s['z'] + confidence * (.20 * s['theta'] + .15 * s['freq']) +
             .05 * s['color']) / (.50 + .35 * confidence) - .15 * boundary
    pattern_conf = max(0., s['z']) * min(np.linalg.norm(prototype(a)['mean_z']),
                                        np.linalg.norm(prototype(b)['mean_z']))
    threshold = config['tau_merge'] + config['boundary_alpha'] * boundary - config['pattern_beta'] * pattern_conf
    merged = merge_prototypes(a, b)
    delta = compute_merge_coherence_delta(a, b, merged, config)
    energy = coherence(merged, config)
    return float(score), float(threshold), float(delta), float(energy), merged


class RegionState:
    def __init__(self, features, embedding, config):
        self.config = config
        self.nodes = node_features(features, embedding)
        self.shape = features['occupancy'].shape
        self.edges = boundary_edges(self.nodes, config)
        self.labels = np.full(self.shape, -1, np.int32)
        self.regions, self.neighbors, self.logs, self.lineage = {}, {}, [], []
        self.uncertain = set()
        valid = features['occupancy'] >= .5
        size = config['supercell']
        for y in range(0, self.shape[0], size):
            for x in range(0, self.shape[1], size):
                block = features['occupancy'][y:y + size, x:x + size]
                if block.mean() < .5:
                    continue
                mask = np.zeros(self.shape, bool)
                mask[y:y + size, x:x + size] = valid[y:y + size, x:x + size]
                if not mask.any():
                    continue
                index = len(self.regions)
                self.labels[mask] = index
                self.regions[index] = build_region_prototype(self.nodes, mask)
                self.neighbors[index] = {}
                self.lineage.append({'id': index, 'parents': [], 'stage': 'supercell',
                                     'node_ids': self.regions[index]['ids'].tolist()})
        self.next_id = len(self.regions)
        flat = self.labels.ravel()
        for u, v, evidence, affinity, *_ in self.edges:
            a, b = int(flat[u]), int(flat[v])
            if min(a, b) < 0 or a == b:
                continue
            old = self.neighbors[a].get(b, (0., 0., 0))
            value = old[0] + evidence, old[1] + affinity, old[2] + 1
            self.neighbors[a][b] = self.neighbors[b][a] = value

    def merge(self, a, b, merged, stage):
        new = self.next_id
        self.next_id += 1
        neighbors = {}
        for source in (a, b):
            for other, stat in self.neighbors[source].items():
                if other in (a, b):
                    continue
                old = neighbors.get(other, (0., 0., 0))
                neighbors[other] = tuple(x + y for x, y in zip(old, stat))
                self.neighbors[other].pop(source, None)
        for other, stat in neighbors.items():
            self.neighbors[other][new] = stat
        self.neighbors[new] = neighbors
        for old in (a, b):
            self.regions.pop(old)
            self.neighbors.pop(old)
            self.uncertain.discard(old)
        self.regions[new] = merged
        self.labels.ravel()[merged['ids']] = new
        self.lineage.append({'id': new, 'parents': [a, b], 'stage': stage, 'area': merged['area']})
        return new

    def assess(self, a, b, stage, log=True):
        stat = self.neighbors[a][b]
        boundary = stat[0] / stat[2]
        score, threshold, delta, energy, merged = compute_region_merge_score(
            self.regions[a], self.regions[b], boundary, self.config)
        accepted = score >= threshold and (not self.config['coherence'] or
            (delta <= self.config['tau_coherence'] and energy <= self.config['max_energy']))
        if log:
            self.logs.append({'stage': stage, 'region_ids': [a, b], 'merge_score': score,
                'threshold': threshold, 'delta_coherence': delta, 'energy': energy,
                'boundary_evidence': boundary, 'boundary_affinity_mean': stat[1] / stat[2],
                'boundary_length': stat[2], 'accepted': bool(accepted),
                'reason': 'accepted' if accepted else 'score' if score < threshold else 'coherence'})
        return accepted, merged, score - threshold

    def export(self, groups=None):
        labels = np.full(self.shape, -1, np.int32)
        group_map = labels.copy()
        uncertain_map = np.zeros(self.shape, bool)
        details = []
        groups = groups or {r: r for r in self.regions}
        group_ids = {g: i for i, g in enumerate(sorted(set(groups.values())))}
        for index, region in enumerate(sorted(self.regions, key=lambda r: -self.regions[r]['area'])):
            p = self.regions[region]
            labels.ravel()[p['ids']] = index
            group = group_ids[groups[region]]
            group_map.ravel()[p['ids']] = group
            uncertain_map.ravel()[p['ids']] = region in self.uncertain
            q = prototype(p)
            ys, xs = np.unravel_index(p['ids'], self.shape)
            stat = list(self.neighbors[region].values())
            q.update(id=index, lineage_id=region, pattern_group_id=group,
                bbox=[int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)],
                uncertain=region in self.uncertain,
                boundary_length=sum(s[2] for s in stat),
                boundary_affinity_mean=sum(s[1] for s in stat) / max(sum(s[2] for s in stat), 1),
                confidence=float(np.linalg.norm(q['mean_z']) * q['geometry_confidence_mean'] *
                    np.exp(-coherence(p, self.config)) * q['foreground_coverage']))
            details.append({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in q.items()})
        strong = [(u, v) for u, v, *rest in self.edges if rest[-1]]
        support = max(8, int((self.nodes['occupancy'] >= .5).sum() * .015))
        detected = len(strong) >= support
        preserved = int(sum(labels.ravel()[u] != labels.ravel()[v] or
                        group_map.ravel()[u] != group_map.ravel()[v] for u, v in strong))
        return labels, group_map, uncertain_map, {
            'regions': details, 'merge_logs': self.logs, 'region_lineage': self.lineage,
            'strong_discontinuity': detected, 'strong_edge_count': len(strong),
            'preserved_strong_edge_count': preserved,
            'collapse_violation': bool(detected and len(details) < 2 and len(group_ids) < 2),
            'initial_region_count': sum(not entry['parents'] for entry in self.lineage)}


def hierarchical_region_aggregation(features, embedding, config):
    state = RegionState(features, embedding, config)
    heap = []
    def push(a, b):
        accepted, _, priority = state.assess(a, b, 'queue', log=False)
        # 拒绝对仍入队；每个实际决策写日志，新原型会重新评估邻居。
        heapq.heappush(heap, (-priority, min(a, b), max(a, b)))
    for a in state.regions:
        for b in state.neighbors[a]:
            if a < b:
                push(a, b)
    seen = set()
    while heap:
        _, a, b = heapq.heappop(heap)
        if (a, b) in seen or a not in state.regions or b not in state.regions:
            continue
        seen.add((a, b))
        accepted, merged, _ = state.assess(a, b, 'hierarchical')
        if accepted:
            new = state.merge(a, b, merged, 'hierarchical')
            for other in state.neighbors[new]:
                push(new, other)
    return state
