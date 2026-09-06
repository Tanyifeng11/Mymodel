"""完整文本调节纹理查询；视觉重采样的 key/value 仍来自参考图。"""

import torch
from torch import nn


def text_content_mask(input_ids, eos_token_id):
    """仅选择 BOS 与首个 EOS 之间的内容，空文本得到全 False。"""
    positions = torch.arange(input_ids.shape[-1], device=input_ids.device)[None, :]
    is_eos = input_ids.eq(eos_token_id)
    end = torch.where(
        is_eos.any(dim=-1), is_eos.int().argmax(dim=-1), input_ids.shape[-1]
    )
    return (positions > 0) & (positions < end[:, None])


def guidance_config_from_checkpoint(state_dict, metadata):
    key = "text_guidance.to_q.weight"
    dim = int(metadata.get("text_guidance_dim", 0))
    if key not in state_dict:
        if dim:
            raise ValueError("checkpoint 声明了文本查询模块，但没有对应权重")
        return {"text_guidance_dim": 0}
    if not dim:
        raise ValueError("文本查询 checkpoint 缺少架构元数据")
    if state_dict[key].shape[0] != dim:
        raise ValueError("文本查询权重与元数据维度不一致")
    return {
        "text_guidance_dim": dim,
        "text_guidance_heads": int(metadata["text_guidance_heads"]),
        "text_guidance_max_ratio": float(metadata["text_guidance_max_ratio"]),
    }


class TextGuidedQueries(nn.Module):
    def __init__(self, hidden_dim=768, inner_dim=256, num_heads=4, max_ratio=0.3):
        super().__init__()
        if inner_dim <= 0 or num_heads <= 0 or inner_dim % num_heads:
            raise ValueError("文本注意力维度必须能被头数整除")
        if max_ratio <= 0:
            raise ValueError("查询扰动上限必须大于零")
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.max_ratio = max_ratio
        self.norm_query = nn.LayerNorm(hidden_dim)
        self.norm_text = nn.LayerNorm(hidden_dim)
        self.to_q = nn.Linear(hidden_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(hidden_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(hidden_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, hidden_dim, bias=False)
        # 只将门控置零，避免与输出投影同时置零造成梯度死锁。
        nn.init.normal_(self.to_out.weight, std=0.02)
        self.gate = nn.Parameter(torch.zeros(()))
        self.last_stats = {}

    def forward(self, queries, text_embeds, text_mask):
        if text_embeds is None or text_mask is None:
            raise ValueError("文本引导重采样需要对应的文本特征及内容 mask")
        if text_embeds.shape[:2] != text_mask.shape or queries.shape[0] != text_embeds.shape[0]:
            raise ValueError("文本特征、mask 和纹理查询的 batch/长度不一致")
        b, n, _ = queries.shape
        h, d = self.num_heads, self.inner_dim // self.num_heads
        text = self.norm_text(text_embeds)
        q = self.to_q(self.norm_query(queries)).view(b, n, h, d).transpose(1, 2)
        k = self.to_k(text).view(b, -1, h, d).transpose(1, 2)
        v = self.to_v(text).view(b, -1, h, d).transpose(1, 2)
        logits = (q.float() @ k.float().transpose(-1, -2)) * (d ** -0.5)
        mask = text_mask.to(device=queries.device, dtype=torch.bool)
        # 全空行先保持有限值，最后将其残差显式清零；所有参数仍连接计算图。
        logits = logits.masked_fill(~mask[:, None, None, :], torch.finfo(torch.float32).min)
        weights = logits.softmax(dim=-1).to(v.dtype)
        delta = (weights @ v).transpose(1, 2).reshape(b, n, self.inner_dim)
        delta = self.to_out(delta).float()
        query_rms = queries.detach().float().square().mean(dim=-1, keepdim=True).sqrt()
        delta_rms = delta.square().mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
        gate = self.max_ratio * self.gate.float().tanh()
        residual = gate * delta / delta_rms * query_rms
        residual = residual * mask.any(dim=-1)[:, None, None]
        with torch.no_grad():
            ratio = residual.square().mean(dim=-1).sqrt() / query_rms.squeeze(-1).clamp_min(1e-6)
            self.last_stats = {
                "gate": gate.item(),
                "query_relative_rms_mean": ratio.mean().item(),
                "query_relative_rms_max": ratio.max().item(),
                "text_present_frac": mask.any(dim=-1).float().mean().item(),
            }
        return queries + residual.to(queries.dtype)
