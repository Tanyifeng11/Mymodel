"""A3-3：用多窗口最短稳定重复位移选基频，参数固定，无类别/频率标签输入。"""

import torch

from models.pattern_canonicalization import fft_orientation, rotation_only


def _axis_period(gray, axis):
    # 沿测量轴去均值，移除格纹另一方向的常量分量。
    signal = gray - gray.mean(axis, keepdim=True)
    if axis == -2:
        signal = signal.transpose(-2, -1)
    n = signal.shape[-1]
    energy = signal.square().mean((-2, -1))
    lags = list(range(2, n // 2 + 1))
    scores = []
    for lag in lags:
        left, right = signal[..., :-lag], signal[..., lag:]
        numerator = (left * right).sum((-2, -1))
        denominator = (left.square().sum((-2, -1)) * right.square().sum((-2, -1))).sqrt()
        scores.append(numerator / denominator.clamp_min(1e-8))
    corr = torch.stack(scores, -1)
    # 包括搜索末端的峰，允许最低训练频率对应的长周期。
    # 搜索起点不能当峰：零位移附近的相关性本来就高，需先下降再上升。
    previous = torch.cat([corr.new_full((len(gray), 1), 1), corr[:, :-1]], -1)
    following = torch.cat([corr[:, 1:], corr.new_full((len(gray), 1), -1)], -1)
    peak = (corr >= previous) & (corr >= following)
    stable = peak & (corr >= .60) & (corr >= .80 * corr.max(-1, keepdim=True).values)
    index = stable.float().argmax(-1)
    valid = stable.any(-1)
    index = torch.where(valid, index, corr.argmax(-1))
    selected = corr.gather(1, index[:, None]).squeeze(1)
    # 三点抛物线细化，减轻整数像素滞后的频率量化误差。
    center = corr.gather(1, index[:, None]).squeeze(1)
    left = corr.gather(1, (index - 1).clamp_min(0)[:, None]).squeeze(1)
    right = corr.gather(1, (index + 1).clamp_max(corr.shape[1] - 1)[:, None]).squeeze(1)
    denominator = left - 2 * center + right
    offset = torch.where(denominator.abs() > 1e-6, .5 * (left - right) / denominator, torch.zeros_like(center)).clamp(-.5, .5)
    offset = torch.where((index > 0) & (index < corr.shape[1] - 1), offset, torch.zeros_like(offset))
    lag = index.to(gray.dtype) + 2 + offset
    return lag, selected, energy, valid


@torch.no_grad()
def harmonic_period(images, angles=None):
    """返回规范坐标 x/y 频率；不存在纹理变化的轴标记 inactive，不能硬报一个周期。"""
    if angles is None:
        angles, _ = fft_orientation(images)
    canonical = rotation_only(images, angles)
    gray = (canonical * images.new_tensor([.299, .587, .114])[None, :, None, None]).sum(1)
    n = gray.shape[-1]
    frequencies, confidences, energies, votes = [], [], [], []
    for axis in (-1, -2):
        # 保持沿测量轴的完整长度；沿垂直轴分成三个局部条带。
        local = [gray] + list(gray.chunk(3, dim=-2 if axis == -1 else -1))
        results = [_axis_period(part, axis) for part in local]
        fs = torch.stack([images.shape[-1] / item[0] for item in results], -1)
        scores = torch.stack([item[1] for item in results], -1)
        frequency = fs.median(-1).values
        agreement = ((fs.round() - frequency.round()[:, None]).abs() <= 0).float().mean(-1)
        frequencies.append(frequency)
        confidences.append(scores.median(-1).values)
        energies.append(results[0][2])
        votes.append(agreement)
    frequency = torch.stack(frequencies, -1)
    confidence, energy, agreement = torch.stack(confidences, -1), torch.stack(energies, -1), torch.stack(votes, -1)
    active = (energy > .02 * energy.max(-1, keepdim=True).values.clamp_min(1e-8)) & (confidence >= .60)
    # 对等轴格纹/点阵/印花采用活跃轴平均；条纹只用有效轴。
    scalar = (frequency * active).sum(-1) / active.sum(-1).clamp_min(1)
    return {"xy": frequency.round(), "continuous_xy": frequency, "active": active,
            "confidence": confidence, "window_agreement": agreement,
            "scalar": scalar.round(), "canonical_size": n}
