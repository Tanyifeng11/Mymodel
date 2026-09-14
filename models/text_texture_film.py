"""文本在非线性激活前调制二维纹理特征；零初始化保持原模型行为。"""

import torch
from torch import nn


def film_config_from_checkpoint(state_dict, metadata):
    """旧 checkpoint 默认关闭 FiLM；新 checkpoint 必须同时保存完整权重和维度。"""
    dim = int(metadata.get("film_hidden_dim", 0))
    film_keys = {key for key in state_dict if key.startswith("film.")}
    if dim < 0:
        raise ValueError("FiLM 隐藏维度不能为负数")
    if not film_keys:
        if dim:
            raise ValueError("checkpoint 声明了 FiLM 模块，但没有对应权重")
        return {"film_hidden_dim": 0}
    if not dim:
        raise ValueError("FiLM checkpoint 缺少架构元数据")

    required = {
        "film.mlp.0.weight", "film.mlp.0.bias", "film.mlp.1.weight",
        "film.mlp.1.bias", "film.mlp.3.weight", "film.mlp.3.bias",
    }
    if film_keys != required:
        raise ValueError("FiLM checkpoint 权重不完整或包含未知参数")
    text_dim = state_dict["film.mlp.0.weight"].numel()
    output_dim = state_dict["film.mlp.3.bias"].numel()
    shapes = {
        "film.mlp.0.weight": (text_dim,), "film.mlp.0.bias": (text_dim,),
        "film.mlp.1.weight": (dim, text_dim), "film.mlp.1.bias": (dim,),
        "film.mlp.3.weight": (output_dim, dim), "film.mlp.3.bias": (output_dim,),
    }
    if output_dim % 2 or any(tuple(state_dict[key].shape) != shape for key, shape in shapes.items()):
        raise ValueError("FiLM 权重与元数据维度不一致")
    if "stage3.1.weight" in state_dict and output_dim != 2 * state_dict["stage3.1.weight"].numel():
        raise ValueError("FiLM 输出维度与 stage3 通道数不一致")
    if "resampler_queries" in state_dict and text_dim != state_dict["resampler_queries"].shape[-1]:
        raise ValueError("FiLM 文本维度与 BF 条件维度不一致")
    return {"film_hidden_dim": dim}


class TextTextureFiLM(nn.Module):
    def __init__(self, text_dim=768, channels=128, hidden_dim=128):
        super().__init__()
        if min(text_dim, channels, hidden_dim) <= 0:
            raise ValueError("FiLM 文本、特征通道与隐藏维度必须大于零")
        self.text_dim = text_dim
        self.channels = channels
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, channels * 2),
        )
        # 仅将输出层置零；首步输出层可学习，后续梯度进入前面的文本投影。
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.last_stats = {}

    def forward(self, features, text_embeds, text_mask):
        if text_embeds is None or text_mask is None:
            raise ValueError("FiLM 需要对应的文本特征及内容 mask")
        if (text_embeds.ndim != 3 or text_embeds.shape[:2] != text_mask.shape
                or text_embeds.shape[0] != features.shape[0]
                or text_embeds.shape[-1] != self.text_dim):
            raise ValueError("FiLM 文本特征、mask 和纹理特征的 batch/长度/维度不一致")
        if features.ndim != 4 or features.shape[1] != self.channels:
            raise ValueError("FiLM 需要匹配通道数的二维纹理特征")
        mask = text_mask.to(device=text_embeds.device, dtype=torch.bool)
        # 仅池化真实内容；BOS/EOS/padding 由 text_content_mask 排除。
        pooled = text_embeds.float().masked_fill(~mask[..., None], 0).sum(dim=1)
        pooled = pooled / mask.sum(dim=1, keepdim=True).clamp_min(1)
        modulation = self.mlp(pooled.to(self.mlp[0].weight.dtype))
        # 空文本严格使用恒等变换，同时保留参数到损失的计算图。
        modulation = modulation.masked_fill(~mask.any(dim=1, keepdim=True), 0)
        delta_gamma, beta = modulation.to(features.dtype).chunk(2, dim=-1)
        delta_gamma, beta = delta_gamma[..., None, None], beta[..., None, None]
        output = (1 + delta_gamma) * features + beta
        with torch.no_grad():
            feature_rms = features.detach().float().square().mean(dim=(1, 2, 3)).sqrt()
            residual_rms = (output.detach().float() - features.detach().float()).square().mean(dim=(1, 2, 3)).sqrt()
            self.last_stats = {
                "delta_gamma_abs_mean": delta_gamma.detach().float().abs().mean().item(),
                "beta_abs_mean": beta.detach().float().abs().mean().item(),
                "feature_relative_rms_mean": (residual_rms / feature_rms.clamp_min(1e-6)).mean().item(),
                "text_present_frac": mask.any(dim=1).float().mean().item(),
            }
        return output
