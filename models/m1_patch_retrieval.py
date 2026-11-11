"""M1：结构查询参考纹理，保持旧 E2 的双 CNN、反向 FiLM 和四尺度接口。"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from models.bf_original_fusion import bFMFusion, affine_transformation


class StructureToTexturePatchAttention(nn.Module):
    def __init__(self, channels, attn_dim=128, num_heads=4, patch_stride=2):
        super().__init__()
        if attn_dim % num_heads:
            raise ValueError('注意力维度必须能被头数整除')
        self.attn_dim, self.num_heads = attn_dim, num_heads
        self.q_proj = nn.Conv2d(channels, attn_dim, 1, bias=False)
        self.patch_embed = nn.Conv2d(channels, channels, 3, stride=patch_stride,
                                     padding=1, groups=channels, bias=False)
        self.kv_proj = nn.Conv2d(channels, 2 * attn_dim, 1)
        self.q_norm, self.k_norm = nn.LayerNorm(attn_dim), nn.LayerNorm(attn_dim)
        self.out_norm = nn.LayerNorm(attn_dim)
        self.out_proj = nn.Conv2d(attn_dim, channels, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.capture = False
        self.diagnostic = None

    def heads(self, x):
        b, n, _ = x.shape
        return x.reshape(b, n, self.num_heads, self.attn_dim // self.num_heads).transpose(1, 2).contiguous()

    def forward(self, structure_feat, texture_feat):
        # [B,C,H,W] -> Q:[B,heads,HW,d], K/V:[B,heads,HtWt,d] -> [B,C,H,W]
        b, c, h, w = structure_feat.shape
        if texture_feat.ndim != 4 or texture_feat.shape[:2] != (b, c):
            raise ValueError('结构与纹理必须为 batch、通道一致的 BCHW 张量')
        q = self.heads(self.q_norm(self.q_proj(structure_feat).flatten(2).transpose(1, 2)))
        patches = self.patch_embed(texture_feat)
        k, v = self.kv_proj(patches).chunk(2, 1)
        k = self.heads(self.k_norm(k.flatten(2).transpose(1, 2)))
        v = self.heads(v.flatten(2).transpose(1, 2))
        context = F.scaled_dot_product_attention(q, k, v, dropout_p=0., is_causal=False)
        context = context.transpose(1, 2).reshape(b, h*w, self.attn_dim)
        context = self.out_norm(context).transpose(1, 2).reshape(b, self.attn_dim, h, w)
        delta = self.out_proj(context)
        if self.capture:
            # 固定网格包含左右袖与衣身；区域标签只用于离线分析，不参与条件输入。
            with torch.no_grad():
                xy = [(min(w-1, int(x*w)), min(h-1, int(y*h)))
                      for y in (.25, .5, .75) for x in (.2, .5, .8)]
                ids = torch.tensor([y*w+x for x, y in xy], device=q.device)
                a = (q.float()[:, :, ids] @ k.float().transpose(-1, -2) / math.sqrt(q.shape[-1])).softmax(-1)
                top, indices = a.topk(min(5, a.shape[-1]), -1)
                self.diagnostic = dict(query_count=h*w, key_count=k.shape[-2],
                    query_hw=[h,w], key_hw=list(patches.shape[-2:]), query_xy=xy,
                    normalized_entropy=(-a*a.clamp_min(1e-20).log()).sum(-1).div(math.log(k.shape[-2])).cpu().tolist(),
                    top1=top[...,0].cpu().tolist(), top5_sum=top.sum(-1).cpu().tolist(),
                    top5_indices=indices.cpu().tolist(), attention=a.cpu().numpy(),
                    residual_rms=float(delta.float().square().mean().sqrt()),
                    structure_rms=float(structure_feat.float().square().mean().sqrt()))
        return structure_feat + delta


class M1Fusion(bFMFusion):
    def __init__(self, sketch_nc=1, texture_nc=3, ngf=144, bottleneck_depth=100, num_resblocks=0):
        super().__init__(sketch_nc, texture_nc, ngf=ngf, bottleneck_depth=bottleneck_depth,
                         num_resblocks=num_resblocks)
        for level, channels, stride in zip(range(2,6), [ngf*2,ngf*4,ngf*8,ngf*8], [2,2,1,1]):
            setattr(self, 's2t_patch_attn%d'%level, StructureToTexturePatchAttention(channels, patch_stride=stride))

    def forward(self, sketch, texture):
        s = self.downconv_channel_unchanged(self.resblocks(self.downconv1(sketch)))
        t = self.G_downconv_channel_unchanged(self.G_resblocks(self.G_downconv1(texture)))
        result = []
        for level in range(2,6):
            s = getattr(self, 'downconv%d'%level)(s)
            t = getattr(self, 'G_downconv%d'%level)(t)
            # 必须先读取融合前的草图，保持原反向 FiLM 的依赖关系。
            alpha, beta = self.get_FiLM_param_(s, str(level))
            s = getattr(self, 's2t_patch_attn%d'%level)(s, t)
            t = affine_transformation(t, alpha, beta)
            result.append(s)
        return result


def attach_m1(network, enabled=True):
    old = network.adapter.fusion
    if enabled:
        # 新模块初始化不能扰动旧训练 RNG；共享权重必须逐键完整加载。
        with torch.random.fork_rng(devices=[]):
            # torch.manual_seed 还会重置 CUDA；这里只初始化 CPU 上的新权重。
            torch.random.default_generator.manual_seed(111142)
            new = M1Fusion().to(device=next(old.parameters()).device, dtype=next(old.parameters()).dtype)
        state = old.state_dict()
        incompatible = new.load_state_dict(state, strict=False)
        expected = {n for n in new.state_dict() if n.startswith('s2t_patch_attn')}
        assert set(incompatible.missing_keys) == expected and not incompatible.unexpected_keys
        assert all(torch.equal(new.state_dict()[n], value) for n, value in state.items())
        network.adapter.fusion = new
        for name, parameter in new.named_parameters():
            if name.startswith('G_bottleneck_'):
                parameter.requires_grad_(False)
    # 原反向第五尺度的输出没有消费者，两臂都冻结这组原无效参数。
    for name in ['bottleneck_alpha_5', 'bottleneck_beta_5']:
        getattr(network.adapter.fusion, name).requires_grad_(False)
    counts = dict(total=sum(p.numel() for p in network.parameters()),
        trainable=sum(p.numel() for p in network.parameters() if p.requires_grad),
        new=sum(p.numel() for n,p in network.named_parameters() if '.s2t_patch_attn' in n),
        frozen=sum(p.numel() for p in network.parameters() if not p.requires_grad))
    return counts
