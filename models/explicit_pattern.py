"""E19-A：无标签、无可训练 encoder 的梯度/FFT 正向对照。"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def pattern_features(images):
    """RGB [0,1] -> 四个局部窗口的 36 维特征；不输入颜色均值或标签。

    前四维是梯度结构张量，后 32 维是 x/y 频率边缘分布。
    使用正方形参考图，避免服装画幅改变横竖纹的周期。
    """
    images = F.interpolate(images.float(), (128, 128), mode="bilinear", align_corners=False)
    gray = (images * images.new_tensor([.299, .587, .114])[None, :, None, None]).sum(1)
    windows = torch.stack([gray[:, y:y + 64, x:x + 64]
                           for y in (0, 64) for x in (0, 64)], 1)
    windows = windows - windows.mean((-2, -1), keepdim=True)
    windows = windows / windows.square().mean((-2, -1), keepdim=True).sqrt().clamp_min(1e-6)
    dx = windows[..., :-1, 1:] - windows[..., :-1, :-1]
    dy = windows[..., 1:, :-1] - windows[..., :-1, :-1]
    xx, yy, xy = dx.square().mean((-2, -1)), dy.square().mean((-2, -1)), (dx * dy).mean((-2, -1))
    total = (xx + yy).clamp_min(1e-6)
    gradient = torch.stack([xx / total, yy / total, (xx - yy) / total, 2 * xy / total], -1)
    # DC 已去掉；不使用方向标签，也不把原图/rot90 标志输入特征。
    power = torch.fft.fft2(windows).abs().square()
    fx = torch.log1p(power.sum(-2)[..., 1:17])
    fy = torch.log1p(power.sum(-1)[..., 1:17])
    spectrum = F.normalize(torch.cat([fx, fy], -1), dim=-1)
    return F.normalize(torch.cat([gradient, spectrum], -1), dim=-1)


class ExplicitPatternTokens(nn.Module):
    """等距线性映射保留手工特征几何；幅度只由训练集 BF RMS 标定。"""

    def __init__(self, dim=768, seed=42, rms=1.0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        matrix = torch.randn(dim, 36, generator=generator)
        basis = torch.linalg.qr(matrix, mode="reduced").Q.T
        self.register_buffer("basis", basis)
        self.register_buffer("rms", torch.tensor(float(rms)))

    def forward(self, features):
        return (features.float() @ self.basis.float()) * (math.sqrt(self.basis.shape[1]) * self.rms)
