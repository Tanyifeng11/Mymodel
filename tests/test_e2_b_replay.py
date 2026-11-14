"""检查联合保护和固定起点上限，防止把平均下降当成所有参考安全。"""
import unittest

from tools.e2_b_replay import anchors, joint_gate, GROUPS


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.bank = [dict(case=str(i), condition=g, group=g) for i in range(4) for g in GROUPS]
        self.values = [dict(metrics=dict(old_projection=.1, palette_swd=2., low_l=1.,
                                       l_amplitude=.8, l_correlation=.7)) for _ in self.bank]

    def changed(self, residue=.09):
        return [dict(metrics=dict(v['metrics'], old_projection=residue, palette_swd=1.9,
                                 l_amplitude=.81, l_correlation=.71)) for v in self.values]

    def test_shared_descent_passes_but_one_group_regression_fails(self):
        updated = self.changed()
        self.assertTrue(joint_gate(self.bank, self.values, updated, self.values)['passed'])
        updated[0]['metrics']['old_projection'] = .15
        self.assertFalse(joint_gate(self.bank, self.values, updated, self.values)['passed'])

    def test_initial_cap_prevents_repeated_tolerance_drift(self):
        prior = self.changed(.10004)
        updated = self.changed(.100035)
        result = joint_gate(self.bank, prior, updated, self.values)
        self.assertTrue(any('initial_residue_cap' in reason for reason in result['reasons']))

    def test_joint_improvement_cannot_hide_fading_or_color_regression(self):
        for key, value in (('l_amplitude', .5), ('palette_swd', 3.), ('l_correlation', .5)):
            updated = self.changed()
            updated[0]['metrics'][key] = value
            self.assertFalse(joint_gate(self.bank, self.values, updated, self.values)['passed'])

    def test_bank_covers_all_identities_groups_and_real_sources(self):
        descriptions = {g + f: dict(group=g, family=f, variant='native')
                        for g in GROUPS[:3] for f in ('stripes', 'checks', 'dots')}
        rows = anchors(dict(train=[dict(case_id=str(i)) for i in range(4)], descriptions=descriptions))
        self.assertEqual(len(rows), 16)
        self.assertTrue(all(sum(r['case'] == str(i) for r in rows) == 4 for i in range(4)))
        self.assertTrue(all(sum(r['group'] == g for r in rows) == 4 for g in GROUPS))
        self.assertEqual(len({r['condition'] for r in rows if r['group'] != 'synthetic'}), 9)


if __name__ == '__main__':
    unittest.main()
