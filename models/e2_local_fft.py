"""无配准参考与成衣内部的局部功率谱均值监督；无相位/RGB逐点匹配。"""
import math

import torch
from torch import nn

from models.e2_geometry import InteriorPatchSampler, build_geometry, extract_patches


def gray(patches):
    return (patches.float() * patches.new_tensor([.299, .587, .114], dtype=torch.float32)
            [None, :, None, None]).sum(1)


class PowerSpectrumDescriptor(nn.Module):
    def __init__(self, radial_bins=12, angle_bins=8, std_min=.02):
        super().__init__()
        self.radial_bins, self.angle_bins, self.std_min = radial_bins, angle_bins, std_min
        self.grids = {}

    def grid(self, size, device):
        key = (size, str(device))
        if key not in self.grids:
            frequency = torch.fft.fftfreq(size, device=device, dtype=torch.float32)
            fy, fx = torch.meshgrid(frequency, frequency, indexing='ij')
            radius = (fx.square() + fy.square()).sqrt()
            # 未fftshift网格；仅保留圆形[1/q, .5]cycles/pixel，排除DC。
            band = (radius >= 1 / size) & (radius <= .5)
            radial = ((radius - 1 / size) / (.5 - 1 / size) * self.radial_bins).long()
            radial = radial.clamp(0, self.radial_bins - 1)
            angle = (torch.remainder(torch.atan2(fy, fx), math.pi) / math.pi * self.angle_bins).long()
            index = (radial * self.angle_bins + angle.clamp(0, self.angle_bins - 1))[band]
            present = torch.bincount(index, minlength=self.radial_bins * self.angle_bins) > 0
            window = torch.hann_window(size, periodic=False, device=device, dtype=torch.float32)
            self.grids[key] = band, index, present, window[:, None] * window[None, :]
        return self.grids[key]

    def forward(self, patches):
        with torch.autocast(device_type=patches.device.type, enabled=False):
            g = gray(patches)
            mean = g.mean((-2, -1), keepdim=True)
            std = g.std((-2, -1), keepdim=True, unbiased=False).clamp_min(self.std_min)
            band, indices, present, window = self.grid(g.shape[-1], g.device)
            z = (g - mean) / std * window
            power = torch.fft.fft2(z, norm='ortho').abs().square()
            energy = power[:, band]
            histogram = energy.new_zeros(len(g), self.radial_bins * self.angle_bins)
            histogram = histogram.scatter_add(1, indices[None].expand(len(g), -1), energy)
            histogram = histogram / energy.sum(1, keepdim=True).clamp_min(1e-8)
            # 每尺度固定剔除空bin；不因预测图的频谱内容选择频带。
            return histogram[:, present]


class LocalFFTLoss(nn.Module):
    def __init__(self, sizes=(32, 64), count=4, t_max=249, std_min=.02, seed=110642):
        super().__init__()
        self.sizes, self.count, self.t_max = tuple(sizes), int(count), int(t_max)
        self.std_min = float(std_min)
        self.sampler = InteriorPatchSampler(seed)
        self.descriptor = PowerSpectrumDescriptor(std_min=std_min)

    def sampling_state(self):
        return self.sampler.state_dict()

    def load_sampling_state(self, value):
        self.sampler.load_state_dict(value)

    def forward(self, predicted, reference, mask, timesteps, sample_ids, reference_keep=True):
        geometry = build_geometry(mask, details=False)
        # 只做线性范围变换，不新增硬clamp；越界预测保留梯度。
        y = (predicted.float() + 1) * .5
        r = (reference.detach().float() + 1) * .5
        terms, records = [], []
        active = (timesteps >= 0) & (timesteps <= self.t_max) & geometry.valid
        keep = torch.as_tensor(reference_keep, device=active.device, dtype=torch.bool).expand_as(active)
        active = active & keep
        for i in range(len(y)):
            row = dict(sample_id=int(sample_ids[i]), t=int(timesteps[i]), active=bool(active[i]), scales=[])
            if not bool(active[i]):
                row['reason'] = ('reference_dropout' if not bool(keep[i]) else
                                 'timestep' if int(timesteps[i]) > self.t_max else 'invalid_mask')
                terms.append(y[i].sum() * 0)
                records.append(row)
                continue
            scale_terms = []
            for size in self.sizes:
                yc, candidates = self.sampler.coordinates(geometry.body[i], size, self.count)
                rc, _ = self.sampler.coordinates(r.new_ones(r.shape[-2:]), size, self.count)
                item = dict(size=size, candidates=candidates, predicted_coordinates=yc, reference_coordinates=rc,
                            predicted_n=len(yc), reference_n=len(rc), reference_valid_n=0)
                if yc and rc:
                    yp = extract_patches(y[i], yc, size)
                    rp = extract_patches(r[i], rc, size)
                    # 仅按参考原始灰度std过滤；预测低方差不允许成为跳过条件。
                    with torch.no_grad():
                        reference_valid = gray(rp).std((-2, -1), unbiased=False) >= self.std_min
                        item['reference_valid_n'] = int(reference_valid.sum())
                        target = self.descriptor(rp[reference_valid]).mean(0) if reference_valid.any() else None
                    if target is not None:
                        term = (self.descriptor(yp).mean(0) - target).abs().mean()
                        scale_terms.append(term)
                        item['loss'] = float(term.detach())
                    else:
                        item['reason'] = 'low_texture_reference'
                else:
                    item['reason'] = 'no_interior_patch' if not yc else 'reference_too_small'
                row['scales'].append(item)
            # 各尺度等权；最后按整个batch归一化，无效/高t样本仍计入分母。
            terms.append(torch.stack(scale_terms).mean() if scale_terms else y[i].sum() * 0)
            row['valid_scales'] = len(scale_terms)
            records.append(row)
        loss = torch.stack(terms).mean()
        return loss, dict(samples=records, batch=len(y), active_n=int(active.sum()),
                          valid_n=sum(bool(r.get('valid_scales', 0)) for r in records))
