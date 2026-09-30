"""小区域只按已有 coherence 合并；拒绝后保留空间 ID 并标为 uncertain。"""

from models.apacc_region_prototype import similarities, compute_merge_coherence_delta, coherence, merge_prototypes


def cleanup_small_regions(state):
    threshold = state.config['small_area'] * (state.nodes['occupancy'] >= .5).sum()
    for region in sorted(list(state.regions), key=lambda r: state.regions[r]['area']):
        if region not in state.regions or state.regions[region]['area'] >= threshold:
            continue
        neighbors = sorted(state.neighbors[region],
            key=lambda r: similarities(state.regions[region], state.regions[r])['z'], reverse=True)
        for other in neighbors:
            a, b = state.regions[region], state.regions[other]
            merged = merge_prototypes(a, b)
            delta = compute_merge_coherence_delta(a, b, merged, state.config)
            energy = coherence(merged, state.config)
            stat = state.neighbors[region][other]
            boundary = stat[0] / stat[2]
            score = similarities(a, b)['z']
            # 小区域允许较低 prototype 分数，但绝不绕过 coherence 或强边界。
            accepted = score >= .70 and (not state.config['boundary'] or boundary < state.config['strong_boundary']) and (
                not state.config['coherence'] or (delta <= state.config['tau_coherence'] and energy <= state.config['max_energy']))
            state.logs.append({'stage': 'cleanup', 'region_ids': [region, other],
                'merge_score': score, 'delta_coherence': delta, 'energy': energy,
                'boundary_evidence': boundary, 'accepted': bool(accepted),
                'reason': 'accepted' if accepted else 'coherence_or_boundary_or_affinity'})
            if accepted:
                state.merge(region, other, merged, 'cleanup')
                break
        else:
            state.uncertain.add(region)
    # 合并形成的新区域也可能仍然过小。
    state.uncertain.update(r for r, p in state.regions.items() if p['area'] < threshold)
