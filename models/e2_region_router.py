"""C-R：目标草图特征残差与旧融合条件路由；不把旧融合输出称为纯纹理流。"""
import torch
from torch import nn
import torch.nn.functional as F

from models.bf_sd_skip import BFSkipFusion
from models.e2_structure import structure_maps

VERSION = 'C_R_v1_sketch_encoder_reuse_edt32_square9_eta100'
LEARNED = ('C3', 'C4', 'C5', 'C6')


def sketch_features(fusion, gray):
    """共享旧草图卷积权重、单独前向；不经过参考FiLM，不增添A编码器。"""
    x = fusion.downconv_channel_unchanged(fusion.resblocks(fusion.downconv1(gray)))
    outputs = []
    for level in range(2, 6):
        x = getattr(fusion, 'downconv%d' % level)(x)
        # 下一层旧LeakyReLU是inplace；保存独立张量，不能让后续层改写本尺度。
        outputs.append(x.clone())
    return outputs


def geometry_level(maps, size, w=4., tau=1.):
    hard = F.interpolate(maps[:, 2:3].float(), size=size, mode='nearest')
    d = maps[:, 3:4].float() * 32.
    fg = F.adaptive_avg_pool2d(torch.sigmoid(d / tau), size)
    inside = F.adaptive_avg_pool2d(torch.sigmoid((d - w) / tau), size)
    edges = F.adaptive_max_pool2d(maps[:, [1, 4]].float(), size)
    distance = F.interpolate(maps[:, 3:4].float(), size=size, mode='bilinear', align_corners=False)
    return hard, fg, inside, edges[:, :1], edges[:, 1:], distance


def region_weights(maps, size):
    m = maps[:, 2:3].float()
    body = -F.max_pool2d(-m, 9, 1, 4)
    return {k: F.adaptive_avg_pool2d(v, size) for k, v in
            [('body', body), ('inner_band', m - body), ('background', 1 - m)]}


def gate_statistics(gates, gain, maps):
    g = gates.detach().float()
    gain = gain.detach().float()
    result = dict(mean=g.mean((0, 2, 3)).cpu().tolist(),
                  variance=g.var((0, 2, 3), unbiased=False).cpu().tolist(),
                  saturation=((g < .05) | (g > .95)).float().mean((0, 2, 3)).cpu().tolist(),
                  app_gain_range=[float(gain.min()), float(gain.max())], regions={})
    for name, weight in region_weights(maps, g.shape[-2:]).items():
        mass = float(weight.sum())
        result['regions'][name] = dict(weight=mass,
            mean=((g * weight).sum((0, 2, 3)) / weight.sum()).cpu().tolist() if mass else None)
    return result


class RegionRouter(nn.Module):
    def __init__(self, incoming=(288, 576, 1152, 1152), channels=(320, 640, 1280, 1280),
                 width=16, seed=110919):
        super().__init__()
        self.save_maps, self.gate_maps = False, []
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.structure = nn.ModuleList([nn.Conv2d(a, b, 1) for a, b in zip(incoming, channels)])
            self.reduce_s = nn.ModuleList([nn.Conv2d(a, width, 1) for a in incoming])
            self.reduce_u = nn.ModuleList([nn.Conv2d(a, width, 1) for a in channels])
            self.gates = nn.ModuleList([nn.Sequential(nn.Conv2d(width * 2 + 4, width, 3, padding=1),
                nn.SiLU(), nn.Conv2d(width, 2, 1)) for _ in channels])
            for layer in list(self.structure) + [g[-1] for g in self.gates]:
                nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)

    def forward(self, features, sketch, maps, mode, eta, record=True):
        if mode not in LEARNED:
            raise ValueError(mode)
        if eta == 0:
            return features, []
        self.gate_maps = []
        routed, statistics = [], []
        for i, (u, s) in enumerate(zip(features, sketch)):
            hard, fg, inside, outer, inner, distance = geometry_level(maps, u.shape[-2:])
            if mode in ('C3', 'C4'):
                # C3/C4连外部距离系数也置中性，不能暗含C5/C6的几何先验。
                fg = torch.ones_like(fg)
                geometry = torch.zeros_like(torch.cat([outer, inner, distance, hard], 1))
                prior = torch.ones_like(torch.cat([fg, fg], 1))
            else:
                if mode == 'C5':
                    inner = torch.zeros_like(inner)
                geometry = torch.cat([outer, inner, distance, hard], 1)
                p_app = fg * (.15 + .85 * inside) * (1 - .5 * inner)
                p_str = fg * (.2 + .8 * torch.maximum(outer, inner))
                prior = torch.cat([p_str, p_app], 1)
            z = torch.cat([self.reduce_s[i](s), self.reduce_u[i](u), geometry.to(u.dtype)], 1)
            logits = self.gates[i](z).float()
            if mode == 'C3':
                # 卷积参数全部参与计算，再池化logits；空间各位置使用同一门控。
                logits = logits.mean((2, 3), keepdim=True).expand(-1, -1, *u.shape[-2:])
            gates = (logits + (prior + 1e-6).log()).softmax(1)
            g_str, g_app = gates[:, :1], gates[:, 1:]
            gain = 1 - eta + eta * (.5 + .5 * fg * g_app)
            v = self.structure[i](s)
            value = (1 - eta) * u + eta * ((.5 + .5 * fg * g_app).to(u.dtype) * u
                    + (fg * g_str).to(u.dtype) * v)
            valid = maps[:, 5:6, :1, :1] > .5
            routed.append(torch.where(valid, value, u))
            if record:
                statistics.append(dict(level=i, **gate_statistics(gates, gain, maps)))
                if self.save_maps:
                    self.gate_maps.append(gates.detach().float().cpu())
        return routed, statistics


class CRegionAdapter(BFSkipFusion):
    def __init__(self, original, mode='C6', enabled=True, width=16):
        nn.Module.__init__(self)
        self.fusion, self.projections, self.channels = original.fusion, original.projections, original.channels
        self.features, self.conditional = None, False
        self.handles, self.injection_counts = [], [0] * 4
        self.mode, self.enabled, self.eta, self.records = mode, enabled, 1., []
        self.record_enabled = True
        self.router = RegionRouter(channels=self.channels, width=width).to(next(self.projections.parameters()).device) if mode in LEARNED else None
        self.last_gates = None

    def condition(self, sketch, reference):
        super().condition(sketch, reference)
        if not self.enabled or self.mode == 'C0' or self.eta == 0:
            if self.record_enabled:
                self.records.append(dict(eta=self.eta, active=False, valid=None, scales=[]))
            return
        maps = structure_maps(sketch)
        if self.mode in ('C1', 'C2'):
            result, statistics = [], []
            self.fixed_maps = []
            for i, f in enumerate(self.features):
                hard, soft, *_ = geometry_level(maps, f.shape[-2:])
                weight = hard if self.mode == 'C1' else soft
                result.append(torch.where(maps[:, 5:6, :1, :1] > .5, f * weight.to(f.dtype), f))
                self.fixed_maps.append(weight.detach().float().cpu())
                statistics.append(dict(level=i, kind='fixed_projection_multiplier', mean=float(weight.mean()),
                    variance=float(weight.var(unbiased=False)), regions={name:
                        float((weight * region).sum() / region.sum()) if region.sum() > 0 else None
                        for name, region in region_weights(maps, f.shape[-2:]).items()}))
            self.features = result
            if self.record_enabled:
                self.records.append(dict(eta=1., active=True, valid=int(maps[:, 5, 0, 0].sum()), scales=statistics))
            return
        gray = (sketch * sketch.new_tensor([.2989, .5870, .1140])[None, :, None, None]).sum(1, keepdim=True)
        structural = sketch_features(self.fusion, gray)
        self.features, stats = self.router(self.features, structural, maps, self.mode, self.eta, self.record_enabled)
        if self.record_enabled:
            self.records.append(dict(eta=self.eta, valid=int(maps[:, 5, 0, 0].sum()), scales=stats))


def attach_router(network, mode='C6', enabled=True, width=16):
    original = network.adapter
    original.unbind()
    adapter = CRegionAdapter(original, mode, enabled, width)
    network.adapter = adapter
    adapter.bind(network.unet)
    return adapter
