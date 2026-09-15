"""E12：二维纹理作为 Q，CLIP token 序列作为 K/V。"""

import random

import torch
from torch import nn


class NexusTextureAdapter(nn.Module):
    def __init__(self, text_dim=768, channels=128, inner_dim=256, num_heads=4):
        super().__init__()
        if min(text_dim, channels, inner_dim) <= 0 or num_heads != 4 or inner_dim % num_heads:
            raise ValueError("E12 要求正维度、4 heads，attention 维度可被 heads 整除")
        self.inner_dim, self.num_heads = inner_dim, num_heads
        self.text_dim, self.channels = text_dim, channels
        self.local_conv = nn.Sequential(
            nn.Conv2d(channels, inner_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(inner_dim, inner_dim, 1),
            nn.GroupNorm(4, inner_dim),
        )
        self.attention = nn.MultiheadAttention(
            inner_dim, num_heads, kdim=text_dim, vdim=text_dim,
            dropout=0.0, batch_first=True,
        )
        self.output_proj = nn.Conv2d(inner_dim, channels, 1)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        self.last_stats = {}

    def _delta(self, features, text_embeds):
        if text_embeds is None or text_embeds.ndim != 3:
            raise ValueError("E12 需要完整 CLIP token-level hidden states")
        if text_embeds.shape[0] != features.shape[0] or text_embeds.shape[-1] != self.text_dim:
            raise ValueError("E12 文本 batch 或维度与纹理不匹配")
        dtype = self.local_conv[0].weight.dtype
        refined = self.local_conv(features.to(dtype))
        query = refined.flatten(2).transpose(1, 2)
        # 沿用 CLIP 全序列，包括空提示词编码，不套用 E11 的内容池化或 FiLM。
        text = text_embeds.to(device=features.device, dtype=dtype)
        attended, _ = self.attention(query, text, text, need_weights=False)
        attended = attended.transpose(1, 2).reshape_as(refined)
        return self.output_proj(attended)

    def forward(self, features, text_embeds):
        # 原生 stage3 网格：512×384 输入对应 64×48，直接处理全部 3072 个位置。
        # 不进行池化或插值；原 stage4 和 token 压缩继续使用相同尺寸的输出。
        delta = self._delta(features, text_embeds).to(features.dtype)
        output = features + delta
        with torch.no_grad():
            feature_rms = features.detach().float().square().mean().sqrt()
            self.last_stats = {
                "feature_height": features.shape[-2],
                "feature_width": features.shape[-1],
                "visual_tokens": features.shape[-2] * features.shape[-1],
                "feature_relative_rms": (delta.detach().float().square().mean().sqrt()
                                         / feature_rms.clamp_min(1e-6)).item(),
            }
        return output


def nexus_config_from_checkpoint(state_dict, metadata):
    config = {
        "nexus_dim": int(metadata.get("nexus_dim", 0)),
        "nexus_heads": int(metadata.get("nexus_heads", 4)),
    }
    keys = {key for key in state_dict if key.startswith("nexus.")}
    if bool(keys) != bool(config["nexus_dim"]):
        raise ValueError("E12 架构元数据与 Adapter 权重不一致")
    if config["nexus_dim"] < 0 or config["nexus_heads"] != 4:
        raise ValueError("E12 要求非负 attention 维度和 4 heads")
    if config["nexus_dim"]:
        if config["nexus_dim"] % config["nexus_heads"]:
            raise ValueError("E12 attention 维度必须能被 heads 整除")
        # 用 meta 张量检查完整结构，不消耗随机数或分配训练权重。
        with torch.device("meta"):
            expected = NexusTextureAdapter(
                state_dict["resampler_queries"].shape[-1],
                state_dict["stage3.1.weight"].numel(),
                config["nexus_dim"], config["nexus_heads"],
            ).state_dict()
        if keys != {"nexus." + key for key in expected} or any(
            state_dict["nexus." + key].shape != value.shape for key, value in expected.items()
        ):
            raise ValueError("E12 checkpoint 权重缺失或维度不一致")
    return config


def shuffled_adapter_prompts(samples, seed=42):
    """固定清单上的无自身配对乱序；不改变全局 RNG 或原 prompt。"""
    if len(samples) < 2:
        raise ValueError("E12 shuffled 对照至少需要两个样本")
    order = list(range(len(samples)))
    random.Random(seed).shuffle(order)
    return {
        samples[index]["sample_id"]: samples[order[(offset + 1) % len(order)]]["prompt"]
        for offset, index in enumerate(order)
    }
