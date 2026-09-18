"""验证修正方向、软边界投影及固定时间对照的隔离性。"""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from models.condition_intervention import ConditionIntervention, fixed_trigger_steps, projected_correction, match_correction


class InterventionTests(unittest.TestCase):
    def make(self, folder, mode):
        sketch = SimpleNamespace(scale=0.6)
        texture = SimpleNamespace(scale=1.0, last_gate=7, balanced_gate_trace_enabled=True)
        metadata = dict(sketch_path='s', texture_path='t', prompt='p', seed=60, gam_ckpt='g',
                        guidance_scale=7., texture_scale=1., num_inference_steps=50, mask_info={})
        records = [dict(step_index=i, timestep=999-i, sketch_scales=[0.6], texture_scales=[1.0],
                        regions={'boundary': {'responses': {h: dict(above_repeat_floor=True,
                        cosine=-0.2 if i == 8 else 0.2) for h in ('0.1', '0.2')}}}) for i in range(50)]
        source = dict(steps=list(range(50)), fractions=[0.1, 0.2], records=records,
                      metadata=metadata, prediction_space='epsilon_before_cfg', region_kernel_input_pixels=3)
        path = Path(folder) / 'source.json'
        path.write_text(json.dumps(source), encoding='utf-8')
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, 2:6, 2:6] = 1
        budget_path = None
        if mode.endswith('_matched'):
            budget_path = Path(folder) / 'budget.json'
            budget = dict(mode='boundary', metadata=metadata, trigger_steps=[8], records=[
                dict(step_index=i, timestep=999-i, applied=i == 8, correction_rms=.02 if i == 8 else 0.)
                for i in range(50)])
            budget_path.write_text(json.dumps(budget), encoding='utf-8')
        intervention = ConditionIntervention([sketch], [texture], mask, mode, path, Path(folder)/mode, metadata, budget_path)
        def forward():
            torch.rand(1)
            texture.last_gate = 99
            texture.last_new = 'extra'
            return torch.ones(1, 4, 4, 4) * (sketch.scale - texture.scale)
        return intervention, sketch, texture, forward

    def test_soft_projection_removes_negative_dot_and_stays_in_support(self):
        a = torch.tensor([1., 2., 3.]).reshape(1, 1, 1, 3)
        b = torch.tensor([-2., -1., 4.]).reshape_as(a)
        w = torch.tensor([0.25, 0.75, 0.]).reshape_as(a)
        correction, coefficient = projected_correction(a, b, w)
        self.assertLess(coefficient, 0)
        self.assertAlmostEqual(float((w*a*(b+correction)).sum()), 0., places=6)
        self.assertEqual(float(correction[..., 2]), 0.)
        self.assertTrue(torch.equal(projected_correction(a, a, w)[0], torch.zeros_like(a)))

    def test_modes_and_state_restoration(self):
        for mode, expected in [('baseline', -0.4), ('weaken_texture', -0.2),
                               ('strengthen_sketch', -0.28), ('global', -0.2)]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                obj, sketch, texture, forward = self.make(folder, mode)
                pred = torch.full((1, 4, 4, 4), -0.4)
                state = torch.get_rng_state().clone()
                result = obj.apply(pred, forward, 8, 991)
                self.assertTrue(torch.allclose(result, torch.full_like(result, expected)))
                self.assertTrue(torch.equal(state, torch.get_rng_state()))
                self.assertEqual((sketch.scale, texture.scale, texture.last_gate), (0.6, 1., 7))
                self.assertFalse(hasattr(texture, 'last_new'))
                self.assertTrue(texture.balanced_gate_trace_enabled)
                self.assertTrue(torch.equal(pred, torch.full_like(pred, -0.4)))

    def test_no_trigger_no_forward_and_exact_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            obj, _, _, _ = self.make(folder, 'boundary')
            pred = torch.randn(1, 4, 4, 4)
            self.assertIs(obj.apply(pred, lambda: self.fail('不能额外前向'), 9, 990), pred)

    def test_exception_restores_processors_and_rng(self):
        with tempfile.TemporaryDirectory() as folder:
            obj, sketch, texture, forward = self.make(folder, 'strengthen_sketch')
            state = torch.get_rng_state().clone()
            def fail():
                forward()
                raise RuntimeError('模拟失败')
            with self.assertRaises(RuntimeError):
                obj.apply(torch.zeros(1, 4, 4, 4), fail, 8, 991)
            self.assertEqual((sketch.scale, texture.scale, texture.last_gate), (0.6, 1., 7))
            self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_timestep_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            obj, _, _, forward = self.make(folder, 'baseline')
            with self.assertRaises(ValueError):
                obj.apply(torch.zeros(1, 4, 4, 4), forward, 8, 990)

    def test_fixed_window_and_two_fraction_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            self.make(folder, 'baseline')
            source = json.loads((Path(folder)/'source.json').read_text())
            for r in source['records']:
                for stats in r['regions']['boundary']['responses'].values():
                    stats['cosine'] = -.2
            source['records'][12]['regions']['boundary']['responses']['0.1']['cosine'] = .1
            self.assertEqual(fixed_trigger_steps(source), [i for i in range(8, 36) if i != 12])

    def test_matched_modes_use_fixed_budget_and_restore_state(self):
        for mode in ('weaken_texture_matched', 'strengthen_sketch_matched'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                obj, sketch, texture, forward = self.make(folder, mode)
                pred = torch.full((1, 4, 4, 4), -.4)
                state = torch.get_rng_state().clone()
                result = obj.apply(pred, forward, 8, 991)
                self.assertTrue(torch.allclose(result, torch.full_like(result, -.38)))
                row = obj.report['records'][0]
                self.assertLess(row['match_relative_error'], 1e-5)
                self.assertAlmostEqual(row['correction_rms'], .02, places=6)
                self.assertEqual((sketch.scale, texture.scale, texture.last_gate), (.6, 1., 7))
                self.assertTrue(torch.equal(state, torch.get_rng_state()))
                obj.budget[8]['correction_rms'] = 0.
                self.assertIs(obj.apply(pred, lambda: self.fail('零预算不应前向'), 8, 991), pred)

    def test_matching_preserves_direction_and_reports_rounding(self):
        pred = torch.zeros(1, 1, 1, 2)
        candidate = torch.tensor([[[[3., -4.]]]])
        result, stats = match_correction(pred, candidate, .01)
        self.assertAlmostEqual(float(result.square().mean().sqrt()), .01, places=7)
        self.assertTrue(torch.allclose(result, candidate * stats['match_scale']))
        self.assertIs(match_correction(pred, candidate, 0.)[0], pred)
        with self.assertRaises(ValueError):
            match_correction(pred, pred, .01)
        low_precision = torch.ones(1, 1, 1, 2, dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, '量化后无法匹配'):
            match_correction(low_precision, low_precision * 2, 1e-6)

    def test_fp16_calibration_matches_actual_not_analytic_rms(self):
        generator = torch.Generator().manual_seed(42)
        pred = torch.randn(1, 4, 64, 48, generator=generator).half()
        candidate = (pred.float() + .01 * torch.randn(pred.shape, generator=generator)).half()
        target = 1e-4
        result, stats = match_correction(pred, candidate, target)
        self.assertEqual(result.dtype, pred.dtype)
        self.assertGreater(stats['match_initial_relative_error'], .05)
        self.assertLessEqual(stats['match_relative_error'], .01)
        self.assertAlmostEqual(float((result.float()-pred.float()).square().mean().sqrt()), target, delta=target*.01)

    def test_budget_from_different_sample_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            obj, sketch, texture, _ = self.make(folder, 'weaken_texture_matched')
            path = Path(folder)/'budget.json'
            budget = json.loads(path.read_text())
            budget['metadata']['seed'] = 999
            path.write_text(json.dumps(budget))
            with self.assertRaisesRegex(ValueError, '预算与当前'):
                ConditionIntervention([sketch], [texture], obj.mask, obj.mode,
                    Path(folder)/'source.json', Path(folder)/'bad', obj.report['metadata'], path)


if __name__ == '__main__':
    unittest.main()
