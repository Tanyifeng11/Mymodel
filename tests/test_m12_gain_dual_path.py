"""验证实际 M10_GAIN 端点及半精度下的基底/偏置边界。"""
import unittest
import torch
from models.m54_m10_gain import M10Gain
from models.m12_gain_dual_path import GainDualPathProcessor
from test_m12_fixed_dual_path import FixedFusion, base_processor


class GainFusion(FixedFusion):
    def processor(self, alpha_b, alpha_m):
        gain = M10Gain(32)
        gain.a.data.copy_(torch.linspace(-.08, .08, 32))
        gain.b.data.copy_(torch.linspace(.05, -.05, 32))
        processor = GainDualPathProcessor(self.source, self.attention, alpha_b, alpha_m, gain)
        processor.features = self.source.features
        processor.enabled = True
        processor.geometry = torch.ones(1, 3, 6, 4)
        processor.timestep = torch.tensor([501])
        return processor

    def test_three_native_endpoints_are_bit_exact(self):
        for alpha_b, alpha_m, mode in [(0, 0, None), (1, 0, 'B1'), (0, 1, 'M10_GAIN')]:
            processor = self.processor(alpha_b, alpha_m)
            self.source.mode = mode
            self.source.timestep = processor.timestep
            self.source.m10_gain = processor.native_m10.m10_gain
            expected = base_processor(self.attention, self.target) if mode is None else self.source(self.attention, self.target, **self.options)
            self.assertTrue(torch.equal(expected, processor(self.attention, self.target, **self.options)), mode)

    def test_final_increment_is_scaled_without_readding_projection_bias(self):
        processor = self.processor(0, .5)
        base = base_processor(self.attention, self.target)
        # 验证取出的未加回增量，避免从两个 half 完整输出相减制造舍入误差。
        captured = []
        observe = processor.observe
        processor.observe = lambda b, m, combined, active: (captured.append((b, m, combined)), observe(b, m, combined, active))[-1]
        actual = processor(self.attention, self.target, **self.options)
        self.assertTrue(torch.equal(actual, base + .5 * captured[-1][1]))
        processor.alpha_b = .25
        actual = processor(self.attention, self.target, **self.options)
        b, m, combined = captured[-1]
        self.assertTrue(torch.equal(combined, .25 * b + .5 * m))
        self.assertTrue(torch.equal(actual, base + combined))

    def test_half_precision_gain_endpoint_matches_native(self):
        self.attention.half()
        self.target = self.target.half()
        processor = self.processor(0, 1)
        self.source.mode = 'M10_GAIN'
        self.source.timestep = processor.timestep
        self.source.m10_gain = processor.native_m10.m10_gain
        expected = self.source(self.attention, self.target, **self.options)
        actual = processor(self.attention, self.target, **self.options)
        self.assertTrue(torch.equal(expected, actual))


if __name__ == '__main__':
    unittest.main()
