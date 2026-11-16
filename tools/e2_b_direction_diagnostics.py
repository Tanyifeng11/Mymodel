"""固定权重方向的数值诊断；不更新参数，不提供采用阈值。"""
import torch

from models.e2_lab_anchor import rgb_to_lab
from tools.e2_b_material_losses import highpass, weighted


def interpolate(start, target, scale):
    """先用 FP64 计算真实端点差，再舍入到参数dtype；端点逐位保留。"""
    assert start.shape == target.shape and start.dtype == target.dtype
    if scale == 0:
        return start.detach().clone()
    if scale == 1:
        return target.detach().to(start.device).clone()
    origin = start.detach().double()
    delta = target.detach().to(device=start.device, dtype=torch.float64) - origin
    return (origin + float(scale) * delta).to(start.dtype)


class FrozenPattern:
    """冻结 f68 的频谱focus与无梯度base，保留原pattern表达式用于FD。"""
    @torch.no_grad()
    def __init__(self, rgb0, positive, base_rgb0, weight, coords):
        self.positive = positive.detach().clone()
        self.target = rgb_to_lab(self.positive) / 100
        self.base = base_rgb0.detach().clone()
        self.weight = weight.detach().clone()
        self.interior = (self.weight >= .999).to(self.weight.dtype)
        self.coords = tuple(tuple(xy) for xy in coords)
        current = rgb_to_lab(rgb0.detach()) / 100
        self.focus = []
        self._frequency(current, capture=True)

    def _frequency(self, current, capture=False):
        crops = lambda field: torch.stack([field[0, :, y:y+64, x:x+64] for x, y in self.coords])
        a, b = crops(current), crops(self.target)
        losses = []
        for index, size in enumerate((64, 32)):
            def tiles(field):
                if size == 64:
                    return field
                return field.reshape(-1, 3, 2, 32, 2, 32).permute(0, 2, 4, 1, 3, 5).reshape(-1, 3, 32, 32)
            p, q = tiles(a), tiles(b)
            p, q = p - p.mean((-2, -1), keepdim=True), q - q.mean((-2, -1), keepdim=True)
            scale = q.square().mean((-2, -1), keepdim=True).sqrt().clamp_min(.02).detach()
            difference = torch.fft.rfft2(p / scale, norm='ortho') - torch.fft.rfft2(q / scale, norm='ortho')
            energy = difference.abs().square()
            if capture:
                focus = difference.detach().abs()
                focus = focus / focus.amax((-2, -1), keepdim=True).clamp_min(1e-8)
                self.focus.append(focus.clone())
            losses.append((self.focus[index] * energy).mean())
        return sum(losses) / len(losses)

    def pattern(self, rgb, latent_term):
        current = rgb_to_lab(rgb) / 100
        high = weighted(highpass(current - self.target).abs(), self.interior)
        rgb_term = weighted((rgb - self.positive).abs(), self.weight)
        frequency = self._frequency(current)
        protection = weighted((rgb - self.base).abs(), 1 - self.weight)
        return latent_term + 2 * rgb_term + frequency + high + 2 * protection


def finite_difference(values_plus, values_minus, h, gradient, theta_plus, theta_minus):
    """FD与g·实际舍入参数弦比较；两者均除以同一个2h。"""
    assert h > 0
    g = gradient.detach().double().reshape(-1)
    chord = (theta_plus.detach().to(device=g.device, dtype=torch.float64)
             - theta_minus.detach().to(device=g.device, dtype=torch.float64)).reshape(-1)
    central = (float(values_plus) - float(values_minus)) / (2 * h)
    predicted = float(torch.dot(g, chord)) / (2 * h)
    return dict(central_fd=central, rounded_chord_prediction=predicted,
                discrepancy=central - predicted, chord_norm=float(chord.norm()), h=float(h))
