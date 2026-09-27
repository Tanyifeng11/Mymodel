"""E22: frozen, explicit reference geometry and a zero-initialized spatial residual."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.harmonic_period import harmonic_period
from models.target_pattern_score import gray


@torch.no_grad()
def geometry_map(images):
    """Six channels: cos2θ, sin2θ, fx, fy, orientation coherence, period confidence."""
    images = images.float()
    mono = gray(images)[:, None]
    sx = mono.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])[None, None] / 8
    gx = F.conv2d(mono, sx, padding=1)
    gy = F.conv2d(mono, sx.transpose(-1, -2), padding=1)
    xx = F.avg_pool2d(gx.square(), 17, stride=1, padding=8)
    yy = F.avg_pool2d(gy.square(), 17, stride=1, padding=8)
    xy = F.avg_pool2d(gx * gy, 17, stride=1, padding=8)
    magnitude = ((xx - yy).square() + 4 * xy.square() + 1e-12).sqrt()
    coherence = (magnitude / (xx + yy + 1e-6)).clamp(0, 1)
    cos2 = (xx - yy) / magnitude
    sin2 = 2 * xy / magnitude
    weights = (xx + yy).flatten(2)
    pooled_cos = (cos2.flatten(2) * weights).sum(-1) / weights.sum(-1).clamp_min(1e-6)
    pooled_sin = (sin2.flatten(2) * weights).sum(-1) / weights.sum(-1).clamp_min(1e-6)
    # Harmonic-period voting is the E19.2-A3-3 frozen positive control.
    period = harmonic_period(images)
    frequency = period["scalar"].float()[:, None, None, None]
    fx = frequency * ((1 + pooled_cos.clamp(-1, 1)) / 2).sqrt()[:, :, None, None]
    fy = frequency * ((1 - pooled_cos.clamp(-1, 1)) / 2).sqrt()[:, :, None, None]
    confidence = period["confidence"].clamp(0, 1).mean(-1)[:, None, None, None]
    size = images.shape[-2:]
    expand = lambda x: x.expand(-1, -1, *size)
    return torch.cat([cos2, sin2, expand((fx / 16).clamp(0, 1)),
                      expand((fy / 16).clamp(0, 1)), coherence, expand(confidence)], 1)


def spatial_gate(garment_mask, size):
    """Core=1; inside boundary decays smoothly; outside the garment is exactly 0."""
    mask = garment_mask.float().clamp(0, 1)
    soft = mask * F.avg_pool2d(mask, 33, stride=1, padding=16)
    soft = F.interpolate(soft, size=size, mode="area")
    support = F.interpolate(mask, size=size, mode="area")
    return soft * (support > 0).to(soft.dtype)


class SpatialAdapter(nn.Module):
    def __init__(self, channels, input_channels=3, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(input_channels, hidden, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(hidden, channels, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.alpha = nn.Parameter(torch.ones(()))

    def forward(self, geometry, gate, size):
        x = F.interpolate(geometry, size=size, mode="bilinear", align_corners=False)
        g = F.interpolate(gate, size=size, mode="area")
        return self.alpha * self.net((x * g).float()) * g.float()


class SpatialInjection:
    """Use one mid-high-resolution up-block ResNet output; the base U-Net stays frozen."""

    def __init__(self, unet, adapter, block=2, resnet=0):
        self.adapter = adapter
        self.geometry = None
        self.gate = None
        self.module = unet.up_blocks[block].resnets[resnet]
        assert self.module.out_channels == adapter.net[-1].out_channels
        self.handle = self.module.register_forward_hook(self._inject)

    def _inject(self, module, inputs, output):
        if self.geometry is None:
            return output
        residual = self.adapter(self.geometry, self.gate, output.shape[-2:])
        return output + residual.to(output.dtype)

    def set(self, geometry, gate):
        self.geometry, self.gate = geometry, gate

    def close(self):
        self.handle.remove()
