"""核验 E9 局部旁路模块：零门起点、区域外严格为零、形状与批次约束。"""

import unittest

import torch
import torch.nn as nn

from models.local_detail_adapter import (
    DEFAULT_LOCAL_DETAIL_LAYER, LocalDetailAdapter, attach_local_detail_adapter,
)


class FakeProcessor(nn.Module):
    def __init__(self, hidden_size=16, cross_attention_dim=768):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.to_k_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.local_detail_adapter = None


class FakeUNet:
    def __init__(self, layers=None):
        self.attn_processors = {name: FakeProcessor() for name in (layers or [DEFAULT_LOCAL_DETAIL_LAYER])}


def adapter(hidden_dim=16, context_dim=768, inner_dim=8, num_heads=2):
    torch.manual_seed(0)
    return LocalDetailAdapter(hidden_dim, context_dim, inner_dim, num_heads)


def inputs(batch=2, height=4, width=3, hidden_dim=16, context_dim=768, tokens=5):
    torch.manual_seed(1)
    hidden = torch.randn(batch, height * width, hidden_dim)
    context = torch.randn(batch, tokens, context_dim)
    mask = torch.zeros(batch, 1, height * 2, width * 2)
    mask[:, :, : height, : width] = 1.0
    return hidden, context, mask, (height, width)


class LocalDetailAdapterTests(unittest.TestCase):
    def test_alpha_starts_at_zero_so_first_step_matches_e5(self):
        module = adapter()
        self.assertEqual(float(module.alpha), 0.0)
        hidden, context, mask, shape = inputs()
        residual = module(hidden, context, mask, shape)
        self.assertEqual(residual.shape, hidden.shape)
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))

    def test_alpha_receives_gradient_from_first_step(self):
        module = adapter()
        hidden, context, mask, shape = inputs()
        # 用一阶目标模拟去噪损失对残差的首步梯度；平方范数在零门处梯度恒为零。
        module(hidden, context, mask, shape).mean().backward()
        # 门为零但投影随机初始化，alpha 的一阶梯度非零，第一步就能学。
        self.assertIsNotNone(module.alpha.grad)
        self.assertNotEqual(float(module.alpha.grad), 0.0)

    def test_residual_is_zero_outside_garment_region(self):
        module = adapter()
        with torch.no_grad():
            module.alpha.fill_(0.5)
        hidden, context, mask, shape = inputs()
        residual = module(hidden, context, mask, shape)
        flat_mask = nn.functional.interpolate(mask, size=shape, mode="nearest")
        flat_mask = flat_mask.flatten(2).transpose(1, 2)
        self.assertTrue(torch.equal(residual[flat_mask.expand_as(residual) == 0],
                                    torch.zeros_like(residual[flat_mask.expand_as(residual) == 0])))
        self.assertGreater(float(residual.abs().max()), 0.0)

    def test_zero_mask_disables_branch_for_image_dropout(self):
        module = adapter()
        with torch.no_grad():
            module.alpha.fill_(0.5)
        hidden, context, mask, shape = inputs()
        residual = module(hidden, context, torch.zeros_like(mask), shape)
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))

    def test_stats_report_alpha_and_relative_scale(self):
        module = adapter()
        with torch.no_grad():
            module.alpha.fill_(0.25)
        hidden, context, mask, shape = inputs()
        module(hidden, context, mask, shape)
        self.assertEqual(set(module.last_stats), {"alpha", "residual_relative_rms"})
        self.assertAlmostEqual(float(module.last_stats["alpha"]), 0.25, places=6)
        self.assertGreater(float(module.last_stats["residual_relative_rms"]), 0.0)

    def test_highpass_constraint_changes_only_the_residual_output(self):
        plain = adapter()
        highpass = LocalDetailAdapter(16, 768, 8, 2, output_constraint="highpass")
        highpass.load_state_dict(plain.state_dict())
        with torch.no_grad():
            plain.alpha.fill_(0.5)
            highpass.alpha.fill_(0.5)
        hidden, context, mask, shape = inputs()
        plain_residual = plain(hidden, context, mask, shape)
        highpass_residual = highpass(hidden, context, mask, shape)
        self.assertEqual(highpass_residual.shape, plain_residual.shape)
        self.assertFalse(torch.equal(highpass_residual, plain_residual))
        self.assertGreater(float(highpass_residual.abs().max()), 0.0)

    def test_bad_output_constraint_is_rejected(self):
        with self.assertRaises(ValueError):
            LocalDetailAdapter(16, 768, 8, 2, output_constraint="unknown")

    def test_four_dim_hidden_states_round_trip(self):
        module = adapter()
        with torch.no_grad():
            module.alpha.fill_(0.5)
        _, context, mask, _ = inputs()
        spatial = torch.randn(2, 16, 4, 3)
        residual = module(spatial, context, mask, None)
        self.assertEqual(residual.shape, spatial.shape)

    def test_invalid_shapes_and_batches_are_rejected(self):
        module = adapter()
        hidden, context, mask, shape = inputs()
        with self.assertRaises(ValueError):
            module(hidden, context, mask, (5, 5))          # 序列长度与 spatial_shape 不符
        with self.assertRaises(ValueError):
            module(hidden, context, None, shape)           # 必须提供服装区域 mask
        with self.assertRaises(ValueError):
            module(hidden, context, mask[:1], shape)       # mask 未对齐 CFG batch
        with self.assertRaises(ValueError):
            module(hidden, context[:1], mask, shape)       # token batch 与主分支不一致
        with self.assertRaises(ValueError):
            module(hidden, context, mask, None)            # 三维输入缺少 spatial_shape

    def test_inner_dim_must_divide_heads(self):
        for inner_dim, num_heads in ((9, 2), (0, 2), (8, 0)):
            with self.subTest(inner_dim=inner_dim, num_heads=num_heads):
                with self.assertRaises(ValueError):
                    LocalDetailAdapter(16, 768, inner_dim, num_heads)


class AttachTests(unittest.TestCase):
    def test_attach_uses_processor_dims_and_registers_module(self):
        unet = FakeUNet()
        module = attach_local_detail_adapter(unet, inner_dim=8, num_heads=2)
        processor = unet.attn_processors[DEFAULT_LOCAL_DETAIL_LAYER]
        self.assertIs(processor.local_detail_adapter, module)
        self.assertEqual(module.hidden_dim, processor.hidden_size)
        self.assertEqual(module.context_dim, processor.cross_attention_dim)

    def test_context_dim_falls_back_to_hidden_size(self):
        unet = FakeUNet()
        unet.attn_processors[DEFAULT_LOCAL_DETAIL_LAYER].cross_attention_dim = None
        module = attach_local_detail_adapter(unet, inner_dim=8, num_heads=2)
        self.assertEqual(module.context_dim, unet.attn_processors[DEFAULT_LOCAL_DETAIL_LAYER].hidden_size)

    def test_unknown_layer_and_double_attach_are_rejected(self):
        unet = FakeUNet()
        with self.assertRaises(ValueError):
            attach_local_detail_adapter(unet, layer="missing.layer")
        attach_local_detail_adapter(unet, inner_dim=8, num_heads=2)
        with self.assertRaises(ValueError):
            attach_local_detail_adapter(unet, inner_dim=8, num_heads=2)

    def test_layer_without_ip_attention_is_rejected(self):
        unet = FakeUNet()
        del unet.attn_processors[DEFAULT_LOCAL_DETAIL_LAYER].to_k_ip
        with self.assertRaises(ValueError):
            attach_local_detail_adapter(unet)


if __name__ == "__main__":
    unittest.main()
