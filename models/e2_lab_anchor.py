"""同色条纹任务的解码后Lab约束；只读取M0锚点和衣身mask。"""
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def rgb_to_lab(rgb):
    linear = torch.where(rgb > .04045, ((rgb + .055) / 1.055).pow(2.4), rgb / 12.92)
    matrix = rgb.new_tensor([[.4124564, .3575761, .1804375],
                             [.2126729, .7151522, .0721750],
                             [.0193339, .1191920, .9503041]])
    xyz = torch.einsum('ij,bjhw->bihw', matrix, linear)
    xyz = xyz / rgb.new_tensor([.95047, 1., 1.08883])[None, :, None, None]
    f = torch.where(xyz > .008856, xyz.clamp_min(.008856).pow(1 / 3), 7.787037 * xyz + 16 / 116)
    return torch.cat((116 * f[:, 1:2] - 16, 500 * (f[:, :1] - f[:, 1:2]),
                      200 * (f[:, 1:2] - f[:, 2:])), 1)


def lab_to_rgb(lab):
    fy = (lab[:, :1] + 16) / 116
    f = torch.cat((fy + lab[:, 1:2] / 500, fy, fy - lab[:, 2:] / 200), 1)
    xyz = torch.where(f > 6 / 29, f.pow(3), (f - 16 / 116) / 7.787037)
    xyz = xyz * lab.new_tensor([.95047, 1., 1.08883])[None, :, None, None]
    matrix = lab.new_tensor([[.4124564, .3575761, .1804375],
                             [.2126729, .7151522, .0721750],
                             [.0193339, .1191920, .9503041]])
    linear = torch.einsum('ij,bjhw->bihw', torch.linalg.inv(matrix), xyz)
    return torch.where(linear > .0031308,
                       1.055 * linear.clamp_min(1e-8).pow(1 / 2.4) - .055, 12.92 * linear)


class LabAnchor(nn.Module):
    def __init__(self, anchor_rgb, mask, sigma=24.):
        super().__init__()
        mask = np.asarray(mask, dtype=np.float32)
        weight = np.clip((cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5) - 8) / 8, 0, 1)
        lab = cv2.cvtColor(np.asarray(anchor_rgb, np.float32), cv2.COLOR_RGB2LAB)
        ab = (lab[..., 1:] * weight[..., None]).sum((0, 1)) / weight.sum()
        mean_l = float((lab[..., 0] * weight).sum() / weight.sum())
        scan = torch.zeros((1, 3, 1001, 1), dtype=torch.float64)
        scan[0, 0, :, 0] = torch.linspace(0, 100, 1001, dtype=torch.float64)
        scan[0, 1:, :, 0] = torch.tensor(ab.copy(), dtype=torch.float64)[:, None]
        gamut = lab_to_rgb(scan)[0, :, :, 0].T.numpy()
        valid = ((gamut >= .001) & (gamut <= .999)).all(1)
        levels = np.linspace(0, 100, 1001)
        self.lo, self.hi = float(levels[valid].min() + .2), float(levels[valid].max() - .2)
        den = cv2.GaussianBlur(mask, (0, 0), sigma).clip(.001)
        base = cv2.GaussianBlur(lab[..., 0] * mask, (0, 0), sigma) / den
        room = max(.1, min(15., (self.hi - self.lo) / 4,
                           .8 * (mean_l - self.lo), .8 * (self.hi - mean_l)))
        lower, upper = -100., 100.
        for _ in range(24):
            offset = (lower + upper) / 2
            current = np.clip(base + offset, self.lo + room, self.hi - room)
            if (current * weight).sum() / weight.sum() < mean_l:
                lower = offset
            else:
                upper = offset
        base = np.clip(base + (lower + upper) / 2, self.lo + room, self.hi - room)
        self.sigma, self.mean_l = sigma, mean_l
        self.radius = int(round(4 * sigma))
        kernel = cv2.getGaussianKernel(2 * self.radius + 1, sigma, cv2.CV_32F).reshape(-1)
        tensor = lambda v: torch.from_numpy(np.asarray(v, np.float32).copy())[None, None]
        self.register_buffer('mask', tensor(mask))
        self.register_buffer('weight', tensor(weight))
        self.register_buffer('base_l', tensor(base))
        self.register_buffer('denominator', tensor(den))
        self.register_buffer('ab', torch.tensor(ab.copy()).view(1, 2, 1, 1))
        self.register_buffer('kernel', torch.from_numpy(kernel.copy()))
        self.last_audit = {}

    def smooth(self, value):
        radius = self.radius
        value = value * self.mask
        value = F.conv2d(F.pad(value, (radius, radius, 0, 0), mode='reflect'),
                         self.kernel.view(1, 1, 1, -1))
        value = F.conv2d(F.pad(value, (0, 0, radius, radius), mode='reflect'),
                         self.kernel.view(1, 1, -1, 1))
        return value / self.denominator

    def forward(self, rgb, mode='ab_l'):
        raw = rgb.float()
        lab = rgb_to_lab(raw)
        light = lab[:, :1]
        if mode == 'ab_l':
            light = light + self.smooth(self.base_l - light)
        else:
            assert mode == 'ab'
        safe_light = light.clamp(self.lo, self.hi)
        corrected = lab_to_rgb(torch.cat((safe_light, self.ab.expand(raw.shape[0], -1, *raw.shape[-2:])), 1))
        # 校正在可用L*色域内完成；RGB羽化保持边界颜色有效，不裁剪RGB。
        out = raw * (1 - self.weight) + corrected * self.weight
        out = torch.where(self.weight > 0, out, raw)
        active = self.weight > 0
        self.last_audit = dict(rgb_min=float(out.min().detach()), rgb_max=float(out.max().detach()),
            constrained_l_fraction=float((((light != safe_light) * active).sum() / active.sum()).detach()),
            outside_max_error=float((out - raw).abs().masked_select(~active.expand_as(raw)).max().detach()))
        assert torch.isfinite(out).all() and out.min() >= 0 and out.max() <= 1
        return out

    def describe(self):
        return dict(sigma=self.sigma, ab=self.ab.flatten().tolist(), mean_l=self.mean_l,
                    gamut_l=[self.lo, self.hi], parameters=sum(p.numel() for p in self.parameters()),
                    source='same-seed fixed M0 anchor and mask only; no evaluation target')
