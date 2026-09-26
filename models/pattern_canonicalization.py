"""A3-1 固定 FFT 角度读出；不更新原 geometry，不读取类别或真值角度。"""

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def fft_orientation(images):
    """二维功率谱四阶角矩；格纹/点阵轴只在模 90 度下可辨识。"""
    gray = (images.float() * images.new_tensor([.299, .587, .114])[None, :, None, None]).sum(1)
    gray = gray - gray.mean((-2, -1), keepdim=True)
    size = gray.shape[-1]
    window = torch.hann_window(size, periodic=False, device=images.device)
    power = torch.fft.fftshift(torch.fft.fft2(gray * window[:, None] * window[None, :]), dim=(-2, -1)).abs().square()
    freq = torch.fft.fftshift(torch.fft.fftfreq(size, device=images.device)) * size
    fy, fx = torch.meshgrid(freq, freq, indexing="ij")
    radius = (fx.square() + fy.square()).sqrt()
    angle = torch.atan2(fy, fx)
    # 排除 DC/窗泄漏和高频像素边缘；不随样本预测周期调整尺度。
    weights = power * ((radius >= 2) & (radius <= size / 4))
    real = (weights * torch.cos(4 * angle)).sum((-2, -1))
    imag = (weights * torch.sin(4 * angle)).sum((-2, -1))
    theta = torch.atan2(imag, real) / 4
    confidence = (real.square() + imag.square()).sqrt() / weights.sum((-2, -1)).clamp_min(1e-8)
    return theta, confidence


def rotation_only(images, angles):
    """保持像素尺度，绕中心旋转；取固定 90/128 内接方形避免填充角。

    所有训练臂都取同一中心裁剪。identity encoder 的固定 resize 也保持一致。
    affine_grid 是输出到输入的采样变换，所以使用 +theta 消除输入 +theta。
    """
    c, s = angles.cos(), angles.sin()
    affine = images.new_zeros((len(images), 2, 3))
    affine[:, 0, 0], affine[:, 0, 1] = c, -s
    affine[:, 1, 0], affine[:, 1, 1] = s, c
    grid = F.affine_grid(affine, images.shape, align_corners=False)
    rotated = F.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    size = images.shape[-1]
    crop = int(size / math.sqrt(2))
    start = (size - crop) // 2
    return rotated[..., start:start + crop, start:start + crop]
