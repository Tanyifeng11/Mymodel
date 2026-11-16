"""连续目标零点、RGB方向导数及独立直接高通投影。"""
import numpy as np
import torch
from torch.nn import functional as F

from models.e2_lab_anchor import rgb_to_lab
from tools.e2_b_consistent_projection import ConsistentGarmentProjection


def inputs(dtype):
    y, x = np.indices((80, 72))
    new = .035 * np.sin(x * 2 * np.pi / 13)
    rgb = np.stack((.43 + new, .49 + .7 * new, .54 + .4 * new), axis=-1)
    positive = torch.from_numpy(rgb.transpose(2, 0, 1).copy())[None].to(dtype)
    old = .09 * (y % 16 < 8)
    anchor = np.rint((rgb + old[..., None]) * 255).astype(np.uint8)
    # 衣内区域接触图像上/左边界，直接参考同时检查反射边界伴随。
    mask = np.ones((80, 72), dtype=bool)
    mask[-8:] = False
    mask[:, -8:] = False
    current = positive + torch.from_numpy((.4 * old)[None, None].copy()).to(dtype)
    return positive, anchor, mask, current


def direct_projection(rgb, positive, anchor, inside):
    """独立前向参考：实际执行反射高通，不读取helper的模板或系数。"""
    hp = lambda value: value - F.avg_pool2d(F.pad(value, (8, 8, 8, 8), mode='reflect'), 17, 1)
    anchor_rgb = torch.from_numpy(anchor.copy()).permute(2, 0, 1)[None].to(positive) / 255
    select = lambda value: value[0, :, torch.from_numpy(inside).to(value.device)].T
    target, old, current = [select(hp(rgb_to_lab(value))) for value in (positive, anchor_rgb, rgb)]
    template = old - target
    template = template - target * ((template * target).sum() / target.square().sum().clamp_min(1e-6))
    energy = template.square().mean()
    return ((current - target) * template).mean().abs() / energy


def test_continuous_float32_positive_is_exact_zero_and_detached():
    positive, anchor, mask, _ = inputs(torch.float32)
    positive.requires_grad_()
    metric = ConsistentGarmentProjection(positive, anchor, mask)
    assert metric.energy >= .25
    assert not torch.equal(positive.detach() * 255, (positive.detach() * 255).round())
    value = metric.surrogate(positive)
    assert value.item() == 0.
    assert not metric.positive_lab.requires_grad and not metric.coefficients.requires_grad
    assert torch.count_nonzero(torch.autograd.grad(value, positive)[0]).item() == 0


def test_adjoint_matches_direct_reflected_highpass_value_and_gradient():
    positive, anchor, mask, current = inputs(torch.float64)
    metric = ConsistentGarmentProjection(positive, anchor, mask)
    current.requires_grad_()
    actual = metric.surrogate(current)
    expected = direct_projection(current, positive, anchor, metric.inside)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    actual_gradient = torch.autograd.grad(actual, current, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, current)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-12, rtol=1e-10)
    anchor_rgb = torch.from_numpy(anchor.copy()).permute(2, 0, 1)[None].to(positive) / 255
    torch.testing.assert_close(metric.surrogate(anchor_rgb), positive.new_tensor(1.), atol=1e-12, rtol=0)


def test_rgb_double_finite_difference_matches_autograd():
    positive, anchor, mask, current = inputs(torch.float64)
    metric = ConsistentGarmentProjection(positive, anchor, mask)
    current.requires_grad_()
    generator = torch.Generator().manual_seed(1729)
    direction = torch.randn(current.shape, generator=generator, dtype=torch.float64) * .03
    gradient = torch.autograd.grad(metric.surrogate(current), current)[0]
    predicted = (gradient * direction).sum()
    h = 1e-5
    with torch.no_grad():
        fd = (metric.surrogate(current + h * direction) - metric.surrogate(current - h * direction)) / (2 * h)
    torch.testing.assert_close(fd, predicted, atol=1e-9, rtol=1e-6)


def test_insufficient_fixed_template_energy_returns_connected_zero():
    positive = torch.full((1, 3, 64, 64), 128 / 255, dtype=torch.float32)
    anchor = np.full((64, 64, 3), 128, dtype=np.uint8)
    metric = ConsistentGarmentProjection(positive, anchor, np.ones((64, 64), dtype=bool))
    current = torch.full_like(positive, .6, requires_grad=True)
    value = metric.surrogate(current)
    assert metric.energy < .25 and value.item() == 0.
    assert torch.count_nonzero(torch.autograd.grad(value, current)[0]).item() == 0
