"""训练侧固定回放锚点及最终像素门槛，不读取验证身份。"""
import numpy as np

GROUPS = ('neutral', 'pastel', 'multi', 'synthetic')
FIELDS = ('old_projection', 'palette_swd', 'low_l', 'l_amplitude', 'l_correlation')


def anchors(spec):
    families = ('stripes', 'checks', 'dots')
    rows = []
    for j, pair in enumerate(spec['train']):
        for n, group in enumerate(GROUPS[:3]):
            condition = next(c for c, v in spec['descriptions'].items()
                             if v['variant'] == 'native' and v['group'] == group
                             and v['family'] == families[(j + n) % 3])
            rows.append(dict(case=pair['case_id'], condition=condition, group=group))
        rows.append(dict(case=pair['case_id'], condition=('v16', 'h16', 'v32', 'h32')[j], group='synthetic'))
    assert len(rows) == 16 and len({(r['case'], r['condition']) for r in rows}) == 16
    return rows


def group_metrics(bank, values):
    groups = {}
    for group in GROUPS:
        current = [v['metrics'] for row, v in zip(bank, values) if row['group'] == group]
        fields = {key: float(np.mean([v[key] for v in current])) for key in FIELDS}
        fields['amplitude_error'] = float(np.mean([abs(v['l_amplitude'] - 1) for v in current]))
        groups[group] = fields
    return groups


def joint_gate(bank, before, after, initial):
    """逐图起点上限防累计漂移；分组把关避免以颜色换残留或亮度细节。"""
    reasons = []
    for row, current, baseline in zip(bank, after, initial):
        cap = baseline['metrics']['old_projection']
        if current['metrics']['old_projection'] > cap + 1e-5 + 1e-4 * abs(cap):
            reasons.append(row['case'] + ':' + row['condition'] + ':initial_residue_cap')
    previous, current = group_metrics(bank, before), group_metrics(bank, after)
    for group in GROUPS:
        a, b = previous[group], current[group]
        for key in ('old_projection', 'palette_swd', 'low_l', 'amplitude_error'):
            # 残留分组不接受向上漂移；其余像素指标保留极小数值容差。
            tolerance = 1e-8 if key == 'old_projection' else 1e-5
            if b[key] > a[key] + tolerance:
                reasons.append(group + ':' + key)
        if b['l_correlation'] < a['l_correlation'] - 1e-5:
            reasons.append(group + ':l_correlation')
    residue_change = np.mean([current[g]['old_projection'] - previous[g]['old_projection'] for g in GROUPS])
    if residue_change >= -1e-8:
        reasons.append('no_measurable_joint_residue_descent')
    return dict(passed=not reasons, reasons=reasons, before=previous, after=current)
