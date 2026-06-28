import torch
import torch.nn.functional as F
from torch import nn


def _rgb01(images):
    return ((images.float() + 1.0) * 0.5).clamp(0.0, 1.0)


def _rgb_to_lab_normalized(rgb):
    mask = rgb > 0.04045
    linear = torch.where(mask, ((rgb + 0.055) / 1.055).pow(2.4), rgb / 12.92)
    r, g, b = linear[:, 0:1], linear[:, 1:2], linear[:, 2:3]
    x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883

    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0

    def lab_f(t):
        return torch.where(t > eps, t.clamp_min(1e-8).pow(1.0 / 3.0), (kappa * t + 16.0) / 116.0)

    fx, fy, fz = lab_f(x), lab_f(y), lab_f(z)
    l = (116.0 * fy - 16.0) / 100.0
    a = 500.0 * (fx - fy) / 128.0
    b = 200.0 * (fy - fz) / 128.0
    return torch.cat([l, a, b], dim=1).contiguous()


def _rgb_to_hsv(rgb):
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    maxc = torch.max(rgb, dim=1, keepdim=True).values
    minc = torch.min(rgb, dim=1, keepdim=True).values
    delta = maxc - minc

    safe_delta = delta.clamp_min(1e-6)
    hue_r = ((g - b) / safe_delta) % 6.0
    hue_g = ((b - r) / safe_delta) + 2.0
    hue_b = ((r - g) / safe_delta) + 4.0
    hue = torch.where(maxc == r, hue_r, torch.where(maxc == g, hue_g, hue_b)) / 6.0
    hue = torch.where(delta > 1e-6, hue, torch.zeros_like(hue))
    sat = torch.where(maxc > 1e-6, delta / maxc.clamp_min(1e-6), torch.zeros_like(maxc))
    val = maxc
    return torch.cat([hue, sat, val], dim=1).contiguous()


def palette_stats_from_texture(texture_images):
    rgb = _rgb01(texture_images)
    lab = _rgb_to_lab_normalized(rgb)
    hsv = _rgb_to_hsv(rgb)
    features = []
    for space in (rgb, lab, hsv):
        features.append(space.mean(dim=(2, 3)))
        features.append(space.std(dim=(2, 3), unbiased=False))
    return torch.cat(features, dim=1).contiguous()


class PaletteTokenMLP(nn.Module):
    stats_dim = 18

    def __init__(self, cross_attention_dim, num_palette_tokens=4, hidden_dim=256):
        super().__init__()
        self.cross_attention_dim = int(cross_attention_dim)
        self.num_palette_tokens = int(num_palette_tokens)
        self.net = nn.Sequential(
            nn.LayerNorm(self.stats_dim),
            nn.Linear(self.stats_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.num_palette_tokens * self.cross_attention_dim),
        )
        self.norm = nn.LayerNorm(self.cross_attention_dim)

    def forward(self, texture_images):
        stats = palette_stats_from_texture(texture_images)
        param = next(self.net.parameters())
        stats = stats.to(device=param.device, dtype=param.dtype)
        tokens = self.net(stats)
        tokens = tokens.view(-1, self.num_palette_tokens, self.cross_attention_dim)
        return self.norm(tokens)
