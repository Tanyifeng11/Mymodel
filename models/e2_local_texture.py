"""B1/B4/B5局部统计对照；复用原FFT采样/门控，不修改旧模型或损失。"""
import math

import torch
from torch import nn
import torch.nn.functional as F

from models.e2_local_fft import LocalFFTLoss, gray


class GaborDescriptor(nn.Module):
    """固定复Gabor；theta是载波变化方向，条纹延伸方向为theta+90度。"""
    def __init__(self, orientations=8, std_min=.02):
        super().__init__()
        self.orientations, self.std_min = orientations, std_min
        self.kernels = {}

    def kernel(self, wavelength, device):
        key = (wavelength, str(device))
        if key not in self.kernels:
            sigma = wavelength * .5
            radius = math.ceil(3 * sigma)
            axis = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
            y, x = torch.meshgrid(axis, axis, indexing='ij')
            envelope = torch.exp(-(x.square() + y.square()) / (2 * sigma ** 2))
            parts = []
            for i in range(self.orientations):
                theta = math.pi * i / self.orientations
                phase = 2 * math.pi / wavelength * (x * math.cos(theta) + y * math.sin(theta))
                real, imag = envelope * phase.cos(), envelope * phase.sin()
                real, imag = real - real.mean(), imag - imag.mean()
                energy = (real.square().sum() + imag.square().sum()).sqrt()
                parts += [real / energy, imag / energy]
            self.kernels[key] = torch.stack(parts)[:, None]
        return self.kernels[key]

    def forward(self, patches):
        with torch.autocast(device_type=patches.device.type, enabled=False):
            g = gray(patches)
            z = (g - g.mean((-2, -1), keepdim=True)) / g.std(
                (-2, -1), keepdim=True, unbiased=False).clamp_min(self.std_min)
            features = []
            for wavelength in ([4, 8] if g.shape[-1] == 32 else [4, 8, 16]):
                # 只统计valid卷积，不引入padding边界；q32/64的最大核分别25/49。
                response = F.conv2d(z[:, None], self.kernel(wavelength, g.device))
                real, imag = response[:, 0::2], response[:, 1::2]
                amplitude = (real.square() + imag.square() + 1e-8).sqrt()
                features += [amplitude.mean((-2, -1)), amplitude.std((-2, -1), unbiased=False)]
            return torch.cat(features, 1)


class ReferenceGramDescriptor(nn.Module):
    """冻结既有VGG19的relu3_4/relu4_4，原生32/64 patch，无resize/位置对齐。"""
    def __init__(self, style):
        super().__init__()
        self.l3, self.l4 = style.l3, style.l4
        assert not any(p.requires_grad for p in self.parameters())

    def forward(self, patches):
        with torch.autocast(device_type=patches.device.type, enabled=False):
            # 仅新参考Gram使用ImageNet通道规范化；旧GT Gram的预处理保持原样。
            value = patches.float()
            mean = value.new_tensor([.485, .456, .406])[None, :, None, None]
            std = value.new_tensor([.229, .224, .225])[None, :, None, None]
            value = (value - mean) / std
            parts = []
            for layer in [self.l3, self.l4]:
                feature = layer(value).flatten(2)
                channels, area = feature.shape[1:]
                gram = feature @ feature.transpose(1, 2) / (channels * area + 1e-6)
                # 两层各自平均L1，避免通道更多的一层因特征数更多而自动占优。
                parts.append(gram.flatten(1))
            dimensions = sum(part.shape[1] for part in parts)
            return torch.cat([part * (dimensions / (2 * part.shape[1])) for part in parts], 1)


class DescriptorLocalLoss(LocalFFTLoss):
    def __init__(self, descriptor, **kwargs):
        super().__init__(**kwargs)
        self.descriptor = descriptor


class LocalTextureObjective(nn.Module):
    """同预算各臂共享patch流；联合项的每个描述子复用完全相同的坐标。"""
    def __init__(self, weights, style=None, **kwargs):
        super().__init__()
        self.weights = {k: float(v) for k, v in weights.items() if v != 0}
        assert self.weights and set(self.weights) <= {'fft', 'gabor', 'gram'}
        assert all(v > 0 and math.isfinite(v) for v in self.weights.values())
        self.terms = nn.ModuleDict()
        for name in sorted(self.weights):
            if name == 'fft':
                term = LocalFFTLoss(**kwargs)
            else:
                descriptor = GaborDescriptor(std_min=kwargs.get('std_min', .02)) if name == 'gabor' else ReferenceGramDescriptor(style)
                term = DescriptorLocalLoss(descriptor, **kwargs)
            self.terms[name] = term

    def sampling_state(self):
        return {name: term.sampling_state() for name, term in self.terms.items()}

    def load_sampling_state(self, state):
        assert set(state) == set(self.terms)
        for name, term in self.terms.items():
            term.load_sampling_state(state[name])

    def forward(self, *args, **kwargs):
        losses, stats = {}, None
        for name, term in self.terms.items():
            value, current = term(*args, **kwargs)
            losses[name] = value
            if stats is not None:
                for a, b in zip(stats['samples'], current['samples']):
                    assert a['active'] == b['active']
                    for x, y in zip(a['scales'], b['scales']):
                        assert x['predicted_coordinates'] == y['predicted_coordinates']
                        assert x['reference_coordinates'] == y['reference_coordinates']
                assert stats['valid_n'] == current['valid_n']
            else:
                stats = current
        # 临时张量随当前调用返回，由兼容层取出；不在模块上保留上一批计算图。
        stats['_component_tensors'] = {'aux_' + name: loss for name, loss in losses.items()}
        stats['raw_components'] = {k: float(v.detach()) for k, v in losses.items()}
        stats['weights'] = self.weights
        return sum(self.weights[k] * v for k, v in losses.items()), stats
