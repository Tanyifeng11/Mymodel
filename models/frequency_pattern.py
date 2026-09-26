"""E19.1：保留频率坐标的混合分支；显式 FFT 能力须与训练收益分开报告。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.explicit_pattern import pattern_features
from models.learned_pattern import LearnedPatternEncoder


class FrequencyBiasedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = LearnedPatternEncoder()
        # 横/纵轴共用一个局部频率滤波器，不为每个频率 bin 学单独的类别权重。
        self.spectral_filter = nn.Conv1d(1, 1, 3, padding=1)
        nn.init.zeros_(self.spectral_filter.weight)
        nn.init.zeros_(self.spectral_filter.bias)

    def forward(self, images):
        learned = self.cnn(images)
        spectrum = F.normalize(pattern_features(images)[..., 4:], dim=-1)
        residual = self.spectral_filter(spectrum.reshape(-1, 1, 16)).reshape_as(spectrum)
        spectrum = F.normalize((spectrum + .1 * torch.tanh(residual)).clamp_min(0), dim=-1)
        # 保留 CNN 学习的方向和方向/频率能量比例；频率位置由显式响应保留。
        amplitude = learned[..., 4:].norm(dim=-1, keepdim=True).clamp_min(.1)
        return F.normalize(torch.cat([learned[..., :4], spectrum * amplitude], -1), dim=-1)


def frequency_loss(pred, teacher):
    """完整 MSE + 独立频谱 cosine + 主峰 CE；只用训练样本 teacher。"""
    mse = F.mse_loss(pred, teacher)
    spectral = 1 - F.cosine_similarity(pred[..., 4:], teacher[..., 4:], dim=-1).mean()
    # 选择 teacher 能量较大的轴；横竖条纹的主峰 CE 均在 16 个频率 bin 上计算。
    horizontal = teacher[..., 1] > teacher[..., 0]
    target = torch.where(horizontal[..., None], teacher[..., 20:], teacher[..., 4:20])
    output = torch.where(horizontal[..., None], pred[..., 20:], pred[..., 4:20])
    logits = F.normalize(output, dim=-1) / .1
    peak = F.cross_entropy(logits.reshape(-1, 16), target.argmax(-1).reshape(-1))
    return mse + spectral + .1 * peak, {"mse": mse, "spectrum": spectral, "peak": peak}
