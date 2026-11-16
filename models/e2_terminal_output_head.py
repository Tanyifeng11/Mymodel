"""原f68全部冻结；仅第25次active调用(t=1/index49)增加输出卷积权重增量。"""
import torch
from torch import nn
from torch.nn import functional as F


class TerminalOutputHead(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base.eval().requires_grad_(False)
        self.delta_weight = nn.Parameter(torch.zeros_like(base.output.weight))
        self.delta_bias = nn.Parameter(torch.zeros_like(base.output.bias))
        self.reset_capture()

    def reset_capture(self):
        self.active_calls = 0
        self.last_record = None

    def increment_parameters(self):
        return [self.delta_weight, self.delta_bias]

    def from_features(self, feature, mask):
        """完整采样与缓存复用唯一表达式；合并权重后卷积，mask只乘一次。"""
        layer = self.base.output
        return F.conv2d(feature, layer.weight+self.delta_weight, layer.bias+self.delta_bias,
            layer.stride, layer.padding, layer.dilation, layer.groups)*mask

    def forward(self, zt, x0, reference, mask, structure, timestep, color=None):
        local_index = self.active_calls
        self.active_calls += 1
        values = dict(reference=reference, mask=mask, structure=structure, timestep=timestep, color=color)
        if local_index != 24 or int(timestep) != 1:
            return self.base(zt, x0, **values)
        captured = []
        def capture(module, inputs):
            captured.append(inputs[0].detach())
        with self.base.output.register_forward_pre_hook(capture):
            base_delta = self.base(zt, x0, **values)
        assert len(captured) == 1
        feature = captured[0]
        result = self.from_features(feature, mask)
        self.last_record = dict(feature=feature.detach().clone(), mask=mask.detach().clone(),
            base_delta=base_delta.detach().clone(), full_delta=result.detach().clone(),
            index=49, timestep=1)
        return result
