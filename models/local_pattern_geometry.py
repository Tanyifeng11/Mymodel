"""E21.2：仅在已腐蚀衣身内部 ROI 上测量局部方向与周期。"""

import torch
import torch.nn.functional as F

from models.target_pattern_score import gray


def _tiles(images):
    g = gray(images)
    return torch.stack([g[:, y:y+64, x:x+64] for y in (0, 64) for x in (0, 64)], 1)


def orientation_hist(images):
    tiles = _tiles(images).flatten(0, 1)[:, None]
    sx = tiles.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])[None, None] / 8
    sy = sx.transpose(-1, -2)
    dx = F.conv2d(tiles, sx, padding=1)[:, 0, 2:-2, 2:-2]
    dy = F.conv2d(tiles, sy, padding=1)[:, 0, 2:-2, 2:-2]
    strength = (dx.square() + dy.square() + 1e-8).sqrt()
    theta = torch.atan2(dy, dx)
    bins = torch.arange(8, device=theta.device) * torch.pi / 8
    affinity = torch.exp(4 * (torch.cos(2 * (theta[:, None]-bins[None, :, None, None])) - 1))
    hist = (affinity * strength[:, None]).sum((-2, -1))
    return F.normalize(hist.reshape(len(images), 4, 8), p=1, dim=-1)


def local_spectra(images):
    tiles = _tiles(images)
    tiles = tiles - tiles.mean((-2, -1), keepdim=True)
    window = torch.hann_window(64, device=tiles.device)
    power = torch.fft.fft2(tiles * window[:, None] * window[None, :]).abs().square()
    axes = torch.stack([power.sum(-2)[..., 1:17], power.sum(-1)[..., 1:17]], -2)
    f = torch.fft.fftfreq(64, device=tiles.device) * 64
    radial_idx = (f[:, None].square() + f[None, :].square()).sqrt().round().long().clamp_max(16)
    radial = power.new_zeros((*power.shape[:2], 17))
    radial.scatter_add_(-1, radial_idx.flatten()[None, None].expand(len(images), 4, -1), power.flatten(-2))
    # 三路：径向、横向、纵向；每个局部 tile 单独归一化。
    spectrum = torch.cat([radial[..., 1:17].unsqueeze(-2), axes], -2)
    return F.normalize(torch.log1p(spectrum), dim=-1)


def losses(predicted, target, frequency):
    po, to = orientation_hist(predicted), orientation_hist(target)
    ps, ts = local_spectra(predicted), local_spectra(target)
    orientation = (po-to).abs().mean()
    frequency_cosine = 1 - (ps*ts).sum(-1).mean()
    # 已知受控目标的基频，64 像素局部 tile 对应全图周期数的一半。
    bins = torch.arange(1, 17, device=ps.device).float()
    fundamental = torch.as_tensor(float(frequency)/2, device=ps.device)
    target_peak = torch.exp(-.5*((bins-fundamental)/.75).square())
    target_peak = target_peak / target_peak.sum()
    # 径向谱的峰位监督与 cosine 独立，避免只匹配频谱总形状。
    predicted_peak = F.log_softmax(ps[:, :, 0] * 8, dim=-1)
    peak = -(target_peak * predicted_peak).sum(-1).mean()
    return {"orientation": orientation, "frequency": frequency_cosine, "peak": peak}
