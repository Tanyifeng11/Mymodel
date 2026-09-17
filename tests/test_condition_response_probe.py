"""CPU 验证响应几何、区域划分及额外前向的无副作用性。"""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from models.condition_response_probe import ConditionResponseProbe, pair_stats, region_weights


class ResponseProbeTests(unittest.TestCase):
    def setup_probe(self, folder):
        sketch = SimpleNamespace(scale=0.6)
        texture = SimpleNamespace(scale=1.0, last_gate=7,
                                  balanced_gate_trace_enabled=True, balanced_gate_trace=[])
        mask = torch.zeros(1, 1, 16, 16)
        mask[:, :, 4:12, 4:12] = 1
        probe = ConditionResponseProbe([sketch], [texture], mask, folder, steps=[0], region_kernel=3)
        def forward():
            torch.rand(3)  # 验证探针不推进 CPU 随机状态
            texture.last_gate = 9
            texture.last_new = 10
            if texture.balanced_gate_trace_enabled:
                texture.balanced_gate_trace.append(1)
            return torch.ones(1, 4, 4, 4) * (sketch.scale - texture.scale)
        return probe, sketch, texture, forward

    def test_geometry(self):
        a = torch.tensor([1., 0.]).reshape(1, 2, 1, 1)
        w = torch.ones(1, 1, 1, 1)
        self.assertEqual(pair_stats(a, -a, w)['removed_norm_fraction'], 1)
        self.assertEqual(pair_stats(a, a.flip(1), w)['cosine'], 0)
        self.assertEqual(pair_stats(a, a, w)['removed_norm_fraction'], 0)
        self.assertIsNone(pair_stats(a, a * 0, w)['cosine'])
        self.assertIsNone(pair_stats(a, a, w * 0)['a_rms'])

    def test_region_partition_and_thin_boundary(self):
        mask = torch.zeros(1, 1, 32, 32)
        mask[:, :, 8:24, 8:24] = 1
        regions = region_weights(mask, (4, 4), 3)
        self.assertTrue(torch.allclose(sum(regions[k] for k in ('inner', 'boundary', 'background')),
                                       torch.ones(1, 1, 4, 4)))
        self.assertGreater(float(regions['boundary'].sum()), 0)

    def test_observe_preserves_prediction_state_rng_and_trace(self):
        with tempfile.TemporaryDirectory() as folder:
            probe, sketch, texture, forward = self.setup_probe(folder)
            pred = forward()
            original = pred.clone()
            texture.last_gate = 7
            del texture.last_new
            state = torch.get_rng_state().clone()
            probe.observe(pred, forward, 0, 999)
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            self.assertTrue(torch.equal(pred, original))
            self.assertEqual((sketch.scale, texture.scale, texture.last_gate), (0.6, 1., 7))
            self.assertFalse(hasattr(texture, 'last_new'))
            self.assertEqual(texture.balanced_gate_trace, [1])
            data = json.loads((Path(folder) / 'probe.json').read_text(encoding='utf-8'))
            row = data['records'][0]
            self.assertEqual(row['repeat_max_abs'], 0)
            stats = row['regions']['global']
            self.assertAlmostEqual(stats['responses']['0.1']['cosine'], -1)
            self.assertTrue(stats['responses']['0.1']['above_repeat_floor'])
            self.assertAlmostEqual(stats['stability']['0.1_vs_0.2']['sketch']['cosine'], 1)
            self.assertTrue((Path(folder) / 'step_00.npz').exists())

    def test_exception_restores_state(self):
        with tempfile.TemporaryDirectory() as folder:
            probe, sketch, texture, forward = self.setup_probe(folder)
            state = torch.get_rng_state().clone()
            def failing():
                forward()
                raise RuntimeError('模拟前向失败')
            with self.assertRaises(RuntimeError):
                probe.observe(torch.zeros(1, 4, 4, 4), failing, 0, 999)
            self.assertEqual((sketch.scale, texture.scale, texture.last_gate), (0.6, 1., 7))
            self.assertTrue(texture.balanced_gate_trace_enabled)
            self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_unselected_step_does_not_forward_or_write(self):
        with tempfile.TemporaryDirectory() as folder:
            probe, _, _, _ = self.setup_probe(folder)
            probe.observe(torch.zeros(1, 4, 4, 4), lambda: self.fail('不应调用'), 1, 800)
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_sampling_trajectory_is_identical(self):
        with tempfile.TemporaryDirectory() as folder:
            probe, sketch, texture, _ = self.setup_probe(folder)
            def sample(observe):
                latent = torch.ones(1, 4, 4, 4)
                for step in range(3):
                    def forward():
                        return latent.sin() * sketch.scale - texture.scale * latent.cos()
                    pred = forward()
                    if observe:
                        probe.observe(pred, forward, step, 999 - step * 100)
                    # 用原预测推进轨迹，确保探针前向不能改变下一步。
                    latent = latent - pred * 0.1
                return latent
            self.assertTrue(torch.equal(sample(False), sample(True)))


if __name__ == '__main__':
    unittest.main()
