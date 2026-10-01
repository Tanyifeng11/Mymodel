"""E31 的可变 K 贪心选择；去重并以实际 coverage 增益停止。"""

import numpy as np

from models.e31_soft_support import anchor_support, probabilities, coverage


def compatible(a, b):
    x, y = a['metrics'], b['metrics']
    if x['geometry_readable'] and y['geometry_readable']:
        angle = abs((x['orientation_mean']-y['orientation_mean']+90)%180-90)
        period = abs(np.log(max(x['frequency_mean'], 1e-4)/max(y['frequency_mean'], 1e-4)))
        return angle <= 25 and period <= np.log(1.25)
    return True


def select_anchors(candidates, features, embedding, mask, mode='pattern', top1=False,
                   diversity=True, no_richness=False, geometry=True, spatial=True,
                   dino=False, temperature=.10, unknown=True):
    key = 'mean_dino' if dino else 'mean_embedding'
    quality = 'canonicality_no_richness' if no_richness else 'canonicality'
    pool = list(candidates)
    chosen, scores, events = [], [], []
    old_coverage = 0.
    while pool and len(chosen) < (1 if top1 or mode == 'low_frequency_fallback' else 4):
        def ranking(c):
            q = c['metrics'][quality]
            if not chosen or not diversity:
                return q
            similarity = max(np.dot(c[key], a[key]) for a in chosen)
            distance = min(np.linalg.norm(np.asarray(c['center_uv'])-a['center_uv'])/np.sqrt(2) for a in chosen)
            return q+.30*(1-similarity)+.10*distance
        c = max(pool, key=ranking)
        pool.remove(c)
        if any(np.dot(c[key], a[key]) >= .95 and compatible(c, a) for a in chosen):
            events.append(dict(bbox=c['bbox'], reason='redundant'))
            continue
        s, _ = anchor_support(features, embedding, c, geometry, spatial, dino)
        now = coverage(probabilities(scores+[s], mask, temperature, unknown), mask)
        gain = now-old_coverage
        if chosen and gain < .05:
            events.append(dict(bbox=c['bbox'], reason='coverage_gain_below_5pp', gain=gain))
            break
        chosen.append(c)
        scores.append(s)
        events.append(dict(bbox=c['bbox'], reason='accepted', coverage=now, gain=gain))
        old_coverage = now
    return chosen, events
