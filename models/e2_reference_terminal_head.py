"""冻结f68，仅末次B增加两组参考统计调制的输出卷积增量。"""
import torch
from torch import nn
from torch.nn import functional as F


LOG_RMS_MEAN = -2.922891616821289
LOG_RMS_STD = 1.1377462148666382
PARAMETERS = 4616


def reference_statistics(reference, mean=LOG_RMS_MEAN, std=LOG_RMS_STD):
    """只读真实reference前64通道；按样本统计，固定校准值不随设备重估。"""
    assert reference.dtype == torch.float32 and reference.ndim == 4 and reference.shape[1] >= 64
    with torch.no_grad():
        y = (F.pixel_shuffle(reference[:, :64], 8)+1)/2
        high = y-F.avg_pool2d(y, 17, 1, 8, count_include_pad=False)
        rms = high.square().mean((1, 2, 3), keepdim=True).sqrt()
        log_rms = (rms+1e-6).log()
        mean, std = [torch.as_tensor(v, dtype=torch.float32, device=reference.device) for v in (mean, std)]
        return dict(rms=rms, log_rms=log_rms, phi=((log_rms-mean)/std.clamp_min(1e-6)).tanh())


class ReferenceTerminalHead(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base.eval().requires_grad_(False)
        self.delta_weight0 = nn.Parameter(torch.zeros_like(base.output.weight))
        self.delta_bias0 = nn.Parameter(torch.zeros_like(base.output.bias))
        self.delta_weight1 = nn.Parameter(torch.zeros_like(base.output.weight))
        self.delta_bias1 = nn.Parameter(torch.zeros_like(base.output.bias))
        self.register_buffer('log_rms_mean', torch.tensor(LOG_RMS_MEAN, dtype=torch.float32))
        self.register_buffer('log_rms_std', torch.tensor(LOG_RMS_STD, dtype=torch.float32))
        self.reset_capture()

    def reset_capture(self):
        self.active_calls, self.last_record = 0, None

    def increment_parameters(self):
        return [self.delta_weight0, self.delta_bias0, self.delta_weight1, self.delta_bias1]

    def statistics(self, reference):
        return reference_statistics(reference, self.log_rms_mean, self.log_rms_std)

    def from_features(self, feature, mask, phi):
        """完整/缓存共用融合卷积；当前原S0每次一个样本，mask只乘一次。"""
        assert feature.shape[0] == 1 and phi.numel() == 1 and phi.dtype == torch.float32
        layer, scalar = self.base.output, phi.detach().reshape(())
        weight = layer.weight+(self.delta_weight0+scalar*self.delta_weight1)
        bias = layer.bias+(self.delta_bias0+scalar*self.delta_bias1)
        return F.conv2d(feature, weight, bias, layer.stride, layer.padding, layer.dilation, layer.groups)*mask

    def forward(self, zt, x0, reference, mask, structure, timestep, color=None):
        index = self.active_calls
        self.active_calls += 1
        values = dict(reference=reference, mask=mask, structure=structure, timestep=timestep, color=color)
        if index != 24 or int(timestep) != 1:
            return self.base(zt, x0, **values)
        captured = []
        def capture(module, inputs):
            captured.append(inputs[0].detach())
        with self.base.output.register_forward_pre_hook(capture):
            base_delta = self.base(zt, x0, **values)
        assert len(captured) == 1
        feature, statistics = captured[0], self.statistics(reference)
        result = self.from_features(feature, mask, statistics['phi'])
        self.last_record = dict(feature=feature.detach().clone(), mask=mask.detach().clone(),
            reference=reference.detach().clone(), statistics=statistics,
            base_delta=base_delta.detach().clone(), full_delta=result.detach().clone(), index=49, timestep=1)
        return result
