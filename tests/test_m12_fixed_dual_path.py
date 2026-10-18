"""验证原生端点、并行计算与独立冻结投影。"""
import unittest
import torch
from torch import nn
from models.reference_attention_variants import ReferenceAttentionProcessor
from models.m54_candidates import M54Processor
from models.m12_fixed_dual_path import FixedDualPathProcessor


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 4
        self.spatial_norm = self.group_norm = None
        self.rescale_output_factor = 1.
        self.to_q = nn.Linear(32, 32, bias=False)
        self.to_k = nn.Linear(32, 32, bias=False)
        self.to_v = nn.Linear(32, 32, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(32, 32), nn.Dropout(0)])


def base_processor(attention, hidden_states, **kwargs):
    return hidden_states * .7 + .1


class FixedFusion(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.attention = Attention().eval().requires_grad_(False)
        self.source = ReferenceAttentionProcessor(base_processor, 'test', 32, 32, 'B1')
        self.source.__class__ = M54Processor
        self.source.mode = 'B1'
        self.source.gate.data.fill_(.7)
        self.source.enabled = True
        self.source.features = torch.randn(1, 16, 32)
        self.source.eval().requires_grad_(False)
        self.target = torch.randn(1, 24, 32)
        self.options = dict(sa_hidden_states=[torch.ones(1)])

    def processor(self, alpha_b, alpha_m):
        processor = FixedDualPathProcessor(self.source, self.attention, alpha_b, alpha_m)
        processor.features = self.source.features
        processor.enabled = True
        processor.geometry = torch.ones(1, 3, 6, 4)
        return processor

    def test_three_native_endpoints_are_bit_exact(self):
        for alpha_b, alpha_m, mode in [(0, 0, None), (1, 0, 'B1'), (0, 1, 'M10')]:
            self.source.mode = mode
            expected = base_processor(self.attention, self.target) if mode is None else self.source(self.attention, self.target, **self.options)
            actual = self.processor(alpha_b, alpha_m)(self.attention, self.target, **self.options)
            self.assertTrue(torch.equal(expected, actual), mode)

    def test_parallel_sum_uses_same_target_and_preserves_reference(self):
        hidden = self.target.clone()
        reference = self.source.features.clone()
        delta_b = self.processor(1, 0)(self.attention, hidden, **self.options) - base_processor(self.attention, hidden)
        delta_m = self.processor(0, 1)(self.attention, hidden, **self.options) - base_processor(self.attention, hidden)
        processor = self.processor(.5, .5)
        actual = processor(self.attention, hidden, **self.options)
        torch.testing.assert_close(actual, base_processor(self.attention, hidden) + .5 * delta_b + .5 * delta_m, atol=2e-7, rtol=2e-6)
        self.assertTrue(torch.equal(hidden, self.target))
        self.assertTrue(torch.equal(reference, self.source.features))
        self.assertEqual(processor.features.data_ptr(), self.source.features.data_ptr())
        self.assertNotEqual(processor.b1.projection.weight.data_ptr(), processor.m10.projection.weight.data_ptr())
        self.assertTrue(all(not parameter.requires_grad for parameter in processor.parameters()))

    def test_cfg_and_invalid_gate_return_exact_base(self):
        processor = self.processor(1, 1)
        expected = base_processor(self.attention, self.target)
        self.assertTrue(torch.equal(processor(self.attention, self.target), expected))
        processor.enabled = False
        self.assertTrue(torch.equal(processor(self.attention, self.target, **self.options), expected))
        self.assertTrue(all(record['fusion_energy'] == 0 for record in processor.records))

    def test_zero_coefficient_isolates_other_projection_and_cache(self):
        for alpha_b, alpha_m, inactive in [(1, 0, 'm10'), (0, 1, 'b1')]:
            processor = self.processor(alpha_b, alpha_m)
            expected = processor(self.attention, self.target, **self.options)
            reference = processor.features.clone()
            getattr(processor, inactive).projection.weight.data.zero_()
            actual = processor(self.attention, self.target, **self.options)
            self.assertTrue(torch.equal(expected, actual))
            self.assertTrue(torch.equal(reference, processor.features))


if __name__ == '__main__':
    unittest.main()
