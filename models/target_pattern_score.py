"""E21：固定目标侧测量；不改变三路条件表示。"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.pattern_canonicalization import fft_orientation, rotation_only


def interior_rectangle(mask):
    """找完全位于 eroded interior 的最大 3:4 矩形；不根据模型输出选区域。"""
    a = (mask.squeeze().cpu().numpy() > .99).astype(np.int32)
    sat = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    for w in range(min(a.shape[1] // 3, a.shape[0] // 4) * 3, 47, -3):
        h = w * 4 // 3
        sums = sat[h:, w:] - sat[:-h, w:] - sat[h:, :-w] + sat[:-h, :-w]
        ys, xs = np.where(sums == h * w)
        if len(xs):
            j = np.argmin((xs + w / 2 - a.shape[1] / 2) ** 2 + (ys + h / 2 - a.shape[0] / 2) ** 2)
            return [int(xs[j]), int(ys[j]), w, h]
    return None


def patch(images, roi):
    x, y, w, h = roi
    return F.interpolate(images[..., y:y+h, x:x+w], (128, 128), mode="bilinear", align_corners=False)


def gray(images):
    return (images.float() * images.new_tensor([.299, .587, .114])[None, :, None, None]).sum(1)


def direction(images):
    g = gray(images)
    dx, dy = g[:, 1:, 1:] - g[:, 1:, :-1], g[:, 1:, 1:] - g[:, :-1, 1:]
    xx, yy, xy = dx.square().mean((-2, -1)), dy.square().mean((-2, -1)), (dx * dy).mean((-2, -1))
    return torch.stack([xx - yy, 2 * xy], -1) / (xx + yy).clamp_min(1e-6)[:, None]


def spectrum(images):
    """径向功率谱单独监督周期，不用完整 36 维 feature MSE。"""
    g = gray(images)
    g = g - g.mean((-2, -1), keepdim=True)
    n = g.shape[-1]
    window = torch.hann_window(n, device=g.device)
    power = torch.fft.fft2(g * window[:, None] * window[None, :]).abs().square()
    f = torch.fft.fftfreq(n, device=g.device) * n
    bins = (f[:, None].square() + f[None, :].square()).sqrt().round().long()
    radial = power.new_zeros((len(g), n))
    radial.scatter_add_(1, bins.flatten()[None].expand(len(g), -1), power.flatten(1))
    return F.normalize(radial[:, 1:41], dim=-1)


class TargetScorer(nn.Module):
    def __init__(self, identity):
        super().__init__()
        self.identity = identity.eval().requires_grad_(False)
        self.register_buffer("center", torch.zeros(72))
        self.register_buffer("scale", torch.ones(72))
        self.head = nn.Linear(72, 4).requires_grad_(False)

    def features(self, images):
        tokens = self.identity(rotation_only(images, fft_orientation(images)[0]))
        return F.normalize(torch.cat([tokens.mean(1), tokens.std(1)], -1), dim=-1)

    def forward(self, images):
        return self.head((self.features(images) - self.center) / self.scale)

    def fit(self, features, labels):
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        scaler = StandardScaler().fit(features)
        clf = LogisticRegression(C=1., max_iter=3000).fit(scaler.transform(features), labels)
        with torch.no_grad():
            for tensor, value in ((self.center, scaler.mean_), (self.scale, scaler.scale_),
                                  (self.head.weight, clf.coef_), (self.head.bias, clf.intercept_)):
                tensor.copy_(torch.as_tensor(value, device=tensor.device, dtype=tensor.dtype))

    def losses(self, images, target, label):
        return {"identity": F.cross_entropy(self(images), torch.tensor([label], device=images.device)),
                "orientation": (direction(images) - direction(target)).square().mean(),
                "period": 1 - (spectrum(images) * spectrum(target)).sum(-1).mean()}
