"""基于 BF 编码器的区域非对称融合；原始 BF 实现保留为对照。"""
import torch
from torch import nn
import torch.nn.functional as F

from models.bf_original_fusion import bFMFusion
from models.bf_sd_skip import BFSkipFusion, BFSDNetwork


def spatial_normalize(value):
    # 小批量、半精度下按样本/通道统计，保留二维排列。
    x = value.float()
    mean = x.mean((-2, -1), keepdim=True)
    variance = x.var((-2, -1), keepdim=True, unbiased=False)
    return ((x - mean) * torch.rsqrt(variance + 1e-5)).to(value.dtype)


class DualScaleEncoder(bFMFusion):
    def encode(self, sketch, texture):
        s = self.downconv_channel_unchanged(self.resblocks(self.downconv1(sketch)))
        t = self.G_downconv_channel_unchanged(self.G_resblocks(self.G_downconv1(texture)))
        sources = []
        for level in range(2, 6):
            s = getattr(self, 'downconv%d' % level)(s)
            t = getattr(self, 'G_downconv%d' % level)(t)
            sources.append((s, t))
        # 编码与交互分离：推理只编码一次，每个时间步重新融合。
        return sources


class RegionAsymmetricFusion(DualScaleEncoder):
    def __init__(self, ngf=144, bottleneck_depth=100, boundary_weight=.25,
                 structure_limit=.25, region_kernel=9):
        super().__init__(1, 3, ngf=ngf, bottleneck_depth=bottleneck_depth, num_resblocks=0)
        self.boundary_weight = boundary_weight
        self.structure_limit = structure_limit
        self.region_kernel = region_kernel
        channels = (ngf * 2, ngf * 4, ngf * 8, ngf * 8)
        self.region_gates = nn.ModuleList(nn.Conv2d(c * 2, 2, 1) for c in channels)
        self.time_gates = nn.ModuleList(
            nn.Sequential(nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, 2)) for _ in channels)
        for gate, time_gate in zip(self.region_gates, self.time_gates):
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)
            nn.init.zeros_(time_gate[-1].weight)
            nn.init.zeros_(time_gate[-1].bias)

    def regions(self, mask):
        mask = mask.float().clamp(0, 1)
        k = self.region_kernel
        body = -F.max_pool2d(-mask, k, 1, k // 2)
        dilated = F.max_pool2d(mask, k, 1, k // 2)
        boundary = (dilated - body).clamp(0, 1)
        return body + self.boundary_weight * boundary

    def fuse(self, sources, region, timestep, reference_keep):
        sample = sources[0][0]
        time = torch.as_tensor(timestep, device=sample.device, dtype=torch.float32).reshape(-1, 1)
        time = time.expand(len(sample), 1) / 999.
        time = torch.cat([time, time.square()], dim=1).to(sample.dtype)
        keep = reference_keep.to(sample.dtype).reshape(-1, 1, 1, 1)
        outputs = []
        for index, (s, t) in enumerate(sources):
            s_norm, t_norm = spatial_normalize(s), spatial_normalize(t)
            logits = self.region_gates[index](torch.cat([s_norm, t_norm], dim=1))
            logits = logits + self.time_gates[index](time)[:, :, None, None]
            s_to_t, t_to_s = logits.sigmoid().chunk(2, dim=1)
            t_to_s = self.structure_limit * t_to_s
            area = F.interpolate(region, s.shape[-2:], mode='area').to(s.dtype) * keep
            alpha_s, beta_s = self.get_FiLM_param_(s, str(index + 2))
            alpha_t, beta_t = self.get_FiLM_param_(t, str(index + 2), texture=True)
            # 两路均保留恒等路径；纹理改写草图的调制另有限幅。
            t_updated = t + area * s_to_t * (alpha_s.tanh() * t_norm + beta_s.tanh())
            s_updated = s + area * t_to_s * (alpha_t.tanh() * s_norm + beta_t.tanh())
            outputs.append(s_updated + area * s_to_t * spatial_normalize(t_updated))
        return outputs

    def forward(self, sketch, texture, mask, timestep, reference_keep=None):
        if reference_keep is None:
            reference_keep = sketch.new_ones(len(sketch))
        return self.fuse(self.encode(sketch, texture), self.regions(mask), timestep, reference_keep)


class ConcatFusion(DualScaleEncoder):
    """共用双编码器的简单拼接对照，不把参数量差异隐藏为容量匹配。"""
    def __init__(self, ngf=144, bottleneck_depth=100):
        super().__init__(1, 3, ngf=ngf, bottleneck_depth=bottleneck_depth, num_resblocks=0)
        for name in list(self._modules):
            if 'bottleneck' in name:
                delattr(self, name)
        self.mix = nn.ModuleList(nn.Conv2d(c * 2, c, 1)
                                 for c in (ngf * 2, ngf * 4, ngf * 8, ngf * 8))

    def fuse(self, sources, region, timestep, reference_keep):
        keep = reference_keep.reshape(-1, 1, 1, 1)
        return [layer(torch.cat([s, t * keep], dim=1))
                for layer, (s, t) in zip(self.mix, sources)]


class RegionFusionAdapter(BFSkipFusion):
    def __init__(self, variant='region', ngf=144, bottleneck_depth=100,
                 channels=(320, 640, 1280, 1280), boundary_weight=.25,
                 structure_limit=.25, region_kernel=9):
        nn.Module.__init__(self)
        if variant not in ('bf', 'concat', 'region'):
            raise ValueError('融合类型必须为 bf、concat 或 region')
        if not 0 <= boundary_weight <= 1 or not 0 <= structure_limit <= 1:
            raise ValueError('边界权重与结构调制上限必须在 [0,1] 内')
        if region_kernel < 1 or region_kernel % 2 == 0:
            raise ValueError('区域核大小必须为正奇数')
        self.config = dict(variant=variant, ngf=ngf, bottleneck_depth=bottleneck_depth,
                           channels=list(channels), boundary_weight=boundary_weight,
                           structure_limit=structure_limit, region_kernel=region_kernel)
        if variant == 'bf':
            self.fusion = bFMFusion(1, 3, ngf=ngf, bottleneck_depth=bottleneck_depth, num_resblocks=0)
        elif variant == 'concat':
            self.fusion = ConcatFusion(ngf, bottleneck_depth)
        else:
            self.fusion = RegionAsymmetricFusion(ngf, bottleneck_depth, boundary_weight,
                                                 structure_limit, region_kernel)
        self.channels = tuple(channels)
        self.projections = nn.ModuleList(nn.Conv2d(a, b, 1) for a, b in zip(
            (ngf * 2, ngf * 4, ngf * 8, ngf * 8), channels))
        for layer in self.projections:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.features = self.context = self.region = self.reference_keep = None
        self.enabled = True
        self.conditional = False
        self.handles = []
        self.injection_counts = [0] * 4

    def condition(self, sketch, reference, mask=None, reference_keep=None):
        self.clear()
        gray = (sketch * sketch.new_tensor([.2989, .5870, .1140])[None, :, None, None]).sum(1, keepdim=True)
        if self.config['variant'] == 'bf':
            self.features = [p(f) for p, f in zip(self.projections, self.fusion(gray, reference))]
            return
        if self.config['variant'] == 'region' and mask is None:
            raise ValueError('区域融合需要由草图生成的 mask')
        self.context = self.fusion.encode(gray, reference)
        self.region = self.fusion.regions(mask) if self.config['variant'] == 'region' else None
        self.reference_keep = (sketch.new_ones(len(sketch)) if reference_keep is None else reference_keep)

    def bind(self, unet):
        # 复用原 BF 四尺度 skip 注入及通道不匹配跳过规则。
        super().bind(unet)
        def update(module, args, kwargs):
            if not self.enabled:
                self.conditional = False
            if not self.conditional or self.context is None:
                return
            timestep = kwargs.get('timestep', args[1] if len(args) > 1 else None)
            fused = self.fusion.fuse(self.context, self.region, timestep, self.reference_keep)
            self.features = [p(f) for p, f in zip(self.projections, fused)]
        self.handles.append(unet.register_forward_pre_hook(update, with_kwargs=True))

    def clear(self):
        self.features = self.context = self.region = self.reference_keep = None

    def initialize_from_bf(self, state):
        # 复用编码器及兼容的调制头；新门控或拼接头使用自己的初始化。
        own = self.fusion.state_dict()
        shared = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        result = self.fusion.load_state_dict(shared, strict=False)
        return dict(reused=list(shared), new=result.missing_keys)


class RegionFusionNetwork(BFSDNetwork):
    def forward(self, noisy, timestep, text, sketch, reference, sketch_states,
                garment_mask=None, reference_keep=None):
        self.adapter.condition(sketch, reference, garment_mask, reference_keep)
        return self.unet(noisy, timestep, encoder_hidden_states=text,
                         cross_attention_kwargs={'sa_hidden_states': sketch_states}).sample

    def checkpoint(self):
        return dict(format_version=1, fusion_config=self.adapter.config,
                    network=self.trainable_state())

    def load_checkpoint(self, state):
        if state['fusion_config'] != self.adapter.config:
            raise ValueError('checkpoint 融合配置与当前网络不一致')
        # 加载实验权重时禁止静默遗漏新门控或投影；冻结主干不在此包内。
        expected = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        if expected != set(state['network']):
            raise ValueError('checkpoint 的可训练参数不完整或来自不同模型')
        self.load_trainable(state['network'])
