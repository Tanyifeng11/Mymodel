"""E22.1: compare masking placement with identical frozen adapter weights."""

import torch
import torch.nn.functional as F

from models.spatial_geometry import SpatialAdapter, spatial_gate


def feature_mask(mask, size, policy):
    # A cell is safe only if its full source footprint is inside the garment.
    coverage = F.interpolate(mask.float(), size=size, mode="area")
    safe = (coverage >= 1 - 1e-6).float()

    def erode(x):
        return 1 - F.max_pool2d(F.pad(1-x, (1, 1, 1, 1), value=1), 3, stride=1)

    if policy == "eroded":
        return erode(safe)
    if policy == "feather":
        # Rebuild a three-cell inward ramp at the actual feature resolution.
        core = erode(safe)
        return (core + erode(core) + erode(erode(core))) / 3
    return safe


class LocalizedAdapter(SpatialAdapter):
    def __init__(self, policy="post", **kwargs):
        super().__init__(**kwargs)
        self.policy = policy

    def forward(self, geometry, mask, size):
        x = F.interpolate(geometry.float(), size=size, mode="bilinear", align_corners=False)
        if self.policy == "original":
            # Reproduce E22's input+output soft gate exactly, including 32x32 cache.
            gate = F.interpolate(spatial_gate(mask, (32, 32)), size=size, mode="area")
            return self.alpha * self.net(x * gate) * gate
        gate = feature_mask(mask, size, self.policy)
        if self.policy == "input":
            return self.alpha * self.net(x * gate)
        # Apply the mask after all adapter convolutions and immediately before addition.
        return self.alpha * self.net(x) * gate


class LocalizedInjection:
    def __init__(self, unet, adapter, site="up_blocks.2.resnets.0", control=None):
        self.adapter, self.control = adapter, control
        self.geometry = self.mask = None
        self.module = unet.get_submodule(site)
        self.handle = self.module.register_forward_hook(self._inject)
        self.site = site

    def set(self, geometry, mask):
        self.geometry, self.mask = geometry, mask

    def _inject(self, module, inputs, output):
        if self.geometry is None:
            return output
        geometry = self.geometry
        if self.control == "zero":
            geometry = torch.zeros_like(geometry)
        elif self.control == "constant":
            geometry = torch.ones_like(geometry)
        residual = self.adapter(geometry, self.mask, output.shape[-2:])
        assert residual.shape == output.shape
        return output + residual.to(output.dtype)

    def close(self):
        self.handle.remove()
