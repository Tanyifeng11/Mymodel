"""固定颜色 hard negative；只读取 reference histogram，不看 target RGB。"""

import hashlib

import numpy as np


def wrong_references(records, features, hashes):
    hist = np.stack([features[r['id']]['histogram'] for r in records])
    output = {}
    for i, record in enumerate(records):
        valid = [j for j, r in enumerate(records) if r['id'] != record['id']
                 and hashes[r['id']]['reference'] != hashes[record['id']]['reference']
                 and hashes[r['id']]['target'] != hashes[record['id']]['target']]
        assert len(valid) >= 2
        near = min(valid, key=lambda j:(float(np.linalg.norm(hist[i]-hist[j])), records[j]['id']))
        random_candidates = [j for j in valid if j != near]
        seed = int(hashlib.sha256(('E32 wrong/'+record['id']).encode()).hexdigest()[:8], 16)
        random = np.random.default_rng(seed).choice(random_candidates)
        output[record['id']] = dict(color_near=records[near]['id'], random=records[random]['id'],
                                   color_distance=float(np.linalg.norm(hist[i]-hist[near])))
    return output
