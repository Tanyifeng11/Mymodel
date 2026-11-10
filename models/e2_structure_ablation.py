"""A1/A2严格参数匹配单分支；A3/A4/A5原实现保持不变。"""
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

from models.bf_sd_skip import BFSkipFusion
from models.e2_structure import structure_maps, attach_structure, StructureResidual

ARMS = ('A0', 'A1', 'A2', 'A3', 'A4', 'A5')
NEW_ARMS = ('A1', 'A2', 'A3', 'A4')
VERSION = 'A_ablation_v1_single5conv_exact268256_canny40_120'


def condition_maps(sketch, mode):
    maps = structure_maps(sketch)
    if mode == 'A2':
        import cv2
        # 从原草图的uint8灰度取边缘，不从bf16量化后的L反推灰度。
        values = ((sketch.detach().float().cpu() + 1) * 127.5).clamp(0, 255).round().byte()
        edges = []
        for item in values:
            gray = np.asarray(Image.fromarray(item.permute(1, 2, 0).numpy()).convert('L'))
            edges.append(cv2.Canny(gray, 40, 120, apertureSize=3, L2gradient=False).astype(np.float32) / 255.)
        maps[:, 0] = torch.from_numpy(np.stack(edges)).to(device=maps.device, dtype=maps.dtype)
    return maps


class SingleStructureResidual(nn.Module):
    def __init__(self, channels=(320, 640, 1280, 1280), seed=110909):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 24, 3, stride=2, padding=1), nn.GroupNorm(8, 24), nn.SiLU(),
                # 216+600=816个活跃参数；保持单条串行特征流，不注册占位参数。
                nn.Conv2d(24, 24, 3, padding=1, groups=24, bias=False), nn.SiLU(),
                nn.Conv2d(24, 24, 1, bias=True), nn.SiLU(),
                nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.GroupNorm(8, 48), nn.SiLU(),
                nn.Conv2d(48, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.SiLU())
            self.heads = nn.ModuleList([nn.Conv2d(64, c, 1) for c in channels])
            for head in self.heads:
                nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)

    def forward(self, maps, sizes):
        x = self.encoder(maps[:, :1])
        valid = maps[:, 5:6, :1, :1]
        return [head(F.interpolate(x, size=size, mode='bilinear', align_corners=False)) * valid
                for head, size in zip(self.heads, sizes)]


class SingleStructureAdapter(BFSkipFusion):
    def __init__(self, original, mode, enabled=True):
        nn.Module.__init__(self)
        self.fusion, self.projections, self.channels = original.fusion, original.projections, original.channels
        self.features, self.conditional = None, False
        self.handles, self.injection_counts = [], [0] * 4
        self.structure = SingleStructureResidual(self.channels).to(next(self.projections.parameters()).device)
        self.structure_enabled, self.mode = enabled, mode
        self.structure_statistics, self.structure_records = None, []

    def condition(self, sketch, reference):
        super().condition(sketch, reference)
        if not self.structure_enabled:
            return
        maps = condition_maps(sketch, self.mode)
        deltas = self.structure(maps, [f.shape[-2:] for f in self.features])
        self.features = [f + d.to(f.dtype) for f, d in zip(self.features, deltas)]
        self.structure_statistics = dict(valid=int(maps[:, 5, 0, 0].sum()),
            outer_pixels=float(maps[:, 1].sum()), inner_pixels=float(maps[:, 4].sum()),
            residual_rms=[float(d.detach().float().square().mean().sqrt()) for d in deltas])
        self.structure_records.append(self.structure_statistics)


class UnlayeredStructureResidual(StructureResidual):
    """A4无外/内语义分层；第四通道使用原灰度，原配置的零通道不计作活跃参数。"""
    def __init__(self, channels=(320, 640, 1280, 1280)):
        super().__init__(channels=channels, mode='A4')

    def forward(self, maps, sizes):
        features = []
        for scale in (1., .5, .25):
            x = maps if scale == 1 else F.interpolate(maps, scale_factor=scale, mode='bilinear', align_corners=False)
            if scale != 1:
                x = x.clone(); x[:, 2:3] = F.interpolate(maps[:, 2:3], scale_factor=scale, mode='nearest')
            a = torch.cat([x[:, [0, 2, 3]], 1 - x[:, :1]], 1)
            features.append(torch.cat([self.outer(a), self.inner(x[:, [0, 2, 3]])], 1))
        return [head(sum(F.interpolate(f, size=size, mode='bilinear', align_corners=False)
                         for f in features) / 3) * maps[:, 5:6, :1, :1]
                for head, size in zip(self.heads, sizes)]


def attach_ablation(network, mode, enabled=True):
    if mode in ('A3', 'A4', 'A5'):
        adapter = attach_structure(network, mode=mode, width=32, enabled=enabled)
        if mode == 'A4':
            adapter.structure = UnlayeredStructureResidual(adapter.channels).to(next(adapter.projections.parameters()).device)
        return adapter
    assert mode in ('A1', 'A2')
    original = network.adapter; original.unbind()
    network.adapter = SingleStructureAdapter(original, mode, enabled)
    network.adapter.bind(network.unet)
    return network.adapter


def architecture():
    single = sum(p.numel() for p in SingleStructureResidual().parameters())
    counts = {a: single if a in ('A1', 'A2') else sum(p.numel() for p in
              (UnlayeredStructureResidual() if a == 'A4' else StructureResidual(mode=a)).parameters()) for a in ARMS[1:]}
    assert set(counts.values()) == {268256}
    return dict(added_parameters=counts, single=dict(widths=[24, 48, 64], encoder_conv_layers=5,
        heads=4, output_stride=8, convolution_geometric_receptive_field=19,
        extra_block=dict(depthwise_parameters=216, pointwise_parameters=600, parameters=816)),
        dual=dict(width=32, encoder_conv_layers_per_branch=3, encoder_conv_layers_total=6,
            heads=4, output_stride=8, per_input_scale_convolution_geometric_receptive_field=15),
        normalization_note='GN spatial statistics add global dependence;19/15 describe convolution geometry,not full effectiveRF',
        A1_A2='same single-scale single-branch encoder/init;only rawL versus fixedCanny40/120 changes',
        A3='original layered single input scale;four output projections retained',
        A4='unlayered two-lane multi-input-scale control;oldzero input channel replaced by originalgray=1-L;noEo/Es;rawgray redundant with L,not new semantic information',
        A5='original layered multi-input-scale implementation and checkpoint;not retrained',
        parameter_note='total trainable counts equal;architecture/compute/RF are not claimed identical')
