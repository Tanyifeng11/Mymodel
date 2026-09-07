"""E9：将独立的参考局部信息以零初始化残差注入一个细节层。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_LOCAL_DETAIL_LAYER = "up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor"


class LocalDetailAdapter(nn.Module):
    def __init__(self, hidden_dim, context_dim=768, inner_dim=128, num_heads=4):
        super().__init__()
        if inner_dim <= 0 or num_heads <= 0 or inner_dim % num_heads:
            raise ValueError("local detail 的 inner_dim 必须是 num_heads 的正整数倍")
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(context_dim)
        self.to_q = nn.Linear(hidden_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, hidden_dim, bias=False)
        # 只将门置零；投影保留随机初始化，第一步即可学习 alpha。
        self.alpha = nn.Parameter(torch.zeros(()))
        self.last_stats = {}

    def forward(self, hidden_states, local_tokens, garment_mask, spatial_shape):
        input_shape = hidden_states.shape
        if hidden_states.ndim == 4:
            spatial_shape = hidden_states.shape[-2:]
            hidden_states = hidden_states.flatten(2).transpose(1, 2)
        if hidden_states.ndim != 3 or spatial_shape is None:
            raise ValueError("local detail 需要三维序列以及对应的二维 spatial_shape")
        height, width = map(int, spatial_shape)
        batch, sequence, _ = hidden_states.shape
        if sequence != height * width:
            raise ValueError("local detail 目标层的序列长度必须等于 spatial_shape 的高乘宽")
        if garment_mask is None:
            raise ValueError("local detail 必须提供服装区域 mask")
        mask = garment_mask.to(device=hidden_states.device, dtype=hidden_states.dtype)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.ndim != 4 or mask.shape[:2] != (batch, 1):
            raise ValueError("local detail mask 必须为 [batch, 1, height, width]，并已对齐 CFG batch")
        if local_tokens.ndim != 3 or local_tokens.shape[0] != batch:
            raise ValueError("local detail tokens 必须与 hidden_states 的 batch 一致")
        mask = F.interpolate(mask, size=(height, width), mode="nearest")
        mask = mask.flatten(2).transpose(1, 2).clamp(0.0, 1.0)

        hidden = self.query_norm(hidden_states.to(dtype=self.to_q.weight.dtype))
        context = self.context_norm(local_tokens.to(device=hidden.device, dtype=self.to_k.weight.dtype))
        head_dim = self.inner_dim // self.num_heads
        query = self.to_q(hidden).view(batch, sequence, self.num_heads, head_dim).transpose(1, 2)
        key = self.to_k(context).view(batch, -1, self.num_heads, head_dim).transpose(1, 2)
        value = self.to_v(context).view(batch, -1, self.num_heads, head_dim).transpose(1, 2)
        detail = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
        detail = detail.transpose(1, 2).reshape(batch, sequence, self.inner_dim)
        detail = self.to_out(detail)
        # 门乘法用 FP32，最终恢复主分支 dtype；区域外残差严格为零。
        residual = (self.alpha.float() * detail.float() * mask.float()).to(hidden_states.dtype)
        with torch.no_grad():
            base_rms = hidden_states.detach().float().square().mean().sqrt().clamp_min(1e-8)
            residual_rms = residual.detach().float().square().mean().sqrt()
            self.last_stats = {
                "alpha": self.alpha.detach().float(),
                "residual_relative_rms": residual_rms / base_rms,
            }
        if len(input_shape) == 4:
            residual = residual.transpose(1, 2).reshape(input_shape)
        return residual


def attach_local_detail_adapter(unet, layer=DEFAULT_LOCAL_DETAIL_LAYER, inner_dim=128, num_heads=4):
    """在原处理器下注册新模块，原 E5 的参数名称保持不变。"""
    if layer not in unet.attn_processors:
        raise ValueError(f"未找到 local detail 目标层：{layer}")
    processor = unet.attn_processors[layer]
    if not hasattr(processor, "to_k_ip"):
        raise ValueError("local detail 目标层必须使用 IP 注意力处理器")
    if getattr(processor, "local_detail_adapter", None) is not None:
        raise ValueError(f"目标层已挂载 local detail：{layer}")
    adapter = LocalDetailAdapter(
        hidden_dim=processor.hidden_size,
        context_dim=processor.cross_attention_dim or processor.hidden_size,
        inner_dim=inner_dim,
        num_heads=num_heads,
    ).to(device=processor.to_k_ip.weight.device, dtype=processor.to_k_ip.weight.dtype)
    processor.local_detail_adapter = adapter
    return adapter
