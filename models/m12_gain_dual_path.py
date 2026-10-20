"""M12 的 B1 原路保留；统计增量调用现有 M10_GAIN 完整 forward。"""
import copy
import torch
from torch import nn
from models.m12_fixed_dual_path import FixedDualPathProcessor


class ZeroBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = None

    def forward(self, *args, **kwargs):
        # 原生 forward 仍计算统计、gamma、gain、无 bias 投影和 rescale。
        # 只把其加回的基底设为零，避免相减提取增量的半精度舍入损失。
        return torch.zeros_like(self.base)


class GainDualPathProcessor(FixedDualPathProcessor):
    def __init__(self, source, attention, alpha_b, alpha_m, learned_gain):
        super().__init__(source, attention, alpha_b, alpha_m)
        self.native_m10 = copy.deepcopy(source)
        self.native_m10.original = ZeroBase()
        self.native_m10.mode = 'M10_GAIN'
        self.native_m10.m10_gain = learned_gain
        self.native_m10.effect_scale = 1.
        self.eval().requires_grad_(False)

    def statistics_delta(self, attn, target, query, base, shape, native_arguments):
        hidden_states, kwargs = native_arguments
        path = self.native_m10
        path.enabled, path.features = self.enabled, self.features
        path.mask, path.geometry = self.mask, self.geometry
        path.timestep, path.branch = self.timestep, self.branch
        path.original.base = base
        try:
            delta = path(attn, hidden_states, **kwargs)
        finally:
            path.original.base = None
        assert torch.isfinite(delta).all()
        return delta
