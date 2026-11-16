"""核验真实舍入弦，以及原停止梯度量在FD中的冻结语义。"""
import torch

from models.e2_lab_anchor import rgb_to_lab
from tools import e2_b_balanced_losses as original
from tools.e2_b_joint_losses import spectrum
from tools.e2_b_material_losses import highpass, weighted
from tools.e2_b_direction_diagnostics import FrozenPattern, finite_difference, interpolate


def inputs(dtype):
    generator = torch.Generator().manual_seed(1729)
    positive = .35 + .2 * torch.rand(1, 3, 64, 64, generator=generator, dtype=dtype)
    rgb = positive + .035 * torch.randn(1, 3, 64, 64, generator=generator, dtype=dtype)
    base = rgb - .07
    weight = torch.full((1, 1, 64, 64), .3, dtype=dtype)
    weight[..., 8:-8, 8:-8] = 1
    return rgb, positive, base, weight, [(0, 0)]


def original_pattern(rgb, positive, base, weight, coords, latent):
    # 直接调用原频谱函数；不计算pattern未使用的palette(float32限定)等项。
    a, b = [rgb_to_lab(value) / 100 for value in (rgb, positive)]
    return (latent + 2 * weighted((rgb - positive).abs(), weight)
            + spectrum(a, b, coords)
            + weighted(highpass(a - b).abs(), (weight >= .999).to(weight.dtype))
            + 2 * weighted((rgb - base.detach()).abs(), 1 - weight))


def test_interpolation_preserves_exact_endpoints_and_direction_sign():
    start = torch.tensor([-0., 1., -2.], dtype=torch.float32)
    target = torch.tensor([0., 1.5, -1.], dtype=torch.float32)
    for scale, expected in ((0, start), (1, target)):
        actual = interpolate(start, target, scale)
        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
        assert actual.data_ptr() != expected.data_ptr()
    assert torch.equal(interpolate(start, target, -1), torch.tensor([-0., .5, -3.]))


def test_interpolation_forms_endpoint_difference_before_fp32_rounding():
    start = torch.tensor([-1e8], dtype=torch.float32)
    target = torch.tensor([1e8 + 8], dtype=torch.float32)
    assert float(interpolate(start, target, .5)) == 4.
    assert float(start + .5 * (target - start)) == 0.
    assert interpolate(start, target, .5).dtype == start.dtype


def test_fd_prediction_uses_real_rounded_chord_instead_of_ideal_direction():
    start = torch.tensor([1e8, 0.], dtype=torch.float32)
    target = torch.tensor([1e8 + 8, 1.], dtype=torch.float32)
    gradient = torch.tensor([2., -3.])
    h = .1
    plus, minus = interpolate(start, target, h), interpolate(start, target, -h)
    loss = lambda theta: (gradient.double() * (theta.double() - start.double())).sum()
    report = finite_difference(loss(plus), loss(minus), h, gradient, plus, minus)
    assert abs(report['central_fd'] - report['rounded_chord_prediction']) < 1e-12
    assert abs(report['central_fd'] + 3) < 1e-6
    assert float(gradient.double() @ (target.double() - start.double())) == 13.


def test_frozen_pattern_matches_original_value_and_rgb_gradient_at_capture():
    torch.set_num_threads(2)
    rgb0, positive, base, weight, coords = inputs(torch.float32)
    frozen = FrozenPattern(rgb0, positive, base, weight, coords)
    rgb = rgb0.clone().requires_grad_()
    latent = (rgb - .5).square().mean()
    terms = original.terms(rgb, positive, positive, base.detach(), weight, None, coords)
    terms['latent'] = latent
    actual_original = original.groups(terms)['pattern']
    actual_frozen = frozen.pattern(rgb, latent)
    torch.testing.assert_close(actual_frozen, actual_original, rtol=0, atol=0)
    grad_original = torch.autograd.grad(actual_original, rgb, retain_graph=True)[0]
    grad_frozen = torch.autograd.grad(actual_frozen, rgb)[0]
    torch.testing.assert_close(grad_frozen, grad_original, rtol=0, atol=0)
    assert len(frozen.focus) == 2 and all(not value.requires_grad for value in frozen.focus)
    assert not frozen.base.requires_grad


def test_frozen_semantics_fd_matches_ad_while_recomputed_stops_change_fd():
    rgb0, positive, base, weight, coords = inputs(torch.float64)
    generator = torch.Generator().manual_seed(217)
    direction = .05 * torch.randn(rgb0.shape, generator=generator, dtype=torch.float64)
    frozen = FrozenPattern(rgb0, positive, base, weight, coords)
    latent = lambda rgb: (rgb - .5).square().mean()
    rgb = rgb0.clone().requires_grad_()
    frozen_value = frozen.pattern(rgb, latent(rgb))
    original_value = original_pattern(rgb, positive, base, weight, coords, latent(rgb))
    torch.testing.assert_close(frozen_value, original_value, rtol=0, atol=0)
    gradient = torch.autograd.grad(frozen_value, rgb)[0]
    expected = float((gradient * direction).sum())
    h = 1e-7
    plus, minus = rgb0 + h * direction, rgb0 - h * direction
    frozen_fd = float((frozen.pattern(plus, latent(plus)) - frozen.pattern(minus, latent(minus))) / (2 * h))
    # 普通重算不仅更新focus，还重算原本对梯度停止的base；FD因此是另一函数。
    dynamic = lambda value: original_pattern(value, positive, base + .3 * (value - rgb0),
                                              weight, coords, latent(value))
    dynamic_fd = float((dynamic(plus) - dynamic(minus)) / (2 * h))
    assert abs(frozen_fd - expected) < 1e-7
    assert abs(dynamic_fd - expected) > 1e-4
