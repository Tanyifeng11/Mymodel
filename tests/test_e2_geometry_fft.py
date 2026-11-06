import random

import numpy as np
import pytest
import torch

from garment_mask_utils import build_region_masks
from models.e2_geometry import GeometryCache, InteriorPatchSampler, build_geometry, extract_patches
from models.e2_local_fft import LocalFFTLoss, PowerSpectrumDescriptor


@pytest.fixture(autouse=True)
def small_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def rectangle(side=128):
    mask = torch.zeros(1, 1, side, side)
    mask[:, :, 8:-8, 8:-8] = 1
    return mask


def stripe(side=128, period=8, horizontal=False, phase=0):
    coord = torch.arange(side).float()
    line = .7 * torch.sin(coord * (2 * torch.pi / period) + phase)
    value = line[:, None].expand(side, side) if horizontal else line[None].expand(side, side)
    return value[None, None].expand(1, 3, side, side).clone()


@pytest.mark.parametrize('kind', ['rectangle', 'ring', 'thin', 'multiple', 'empty', 'full'])
def test_geometry_keeps_legacy_masks_and_distance_sign(kind):
    mask = rectangle(96)
    if kind == 'ring':
        mask[:, :, 32:64, 32:64] = 0
    elif kind == 'thin':
        mask.zero_(); mask[:, :, 10:86, 45:48] = 1
    elif kind == 'multiple':
        mask[:, :, :, 40:56] = 0
    elif kind == 'empty':
        mask.zero_()
    elif kind == 'full':
        mask.fill_(1)
    old_body = build_region_masks(mask, 9)[0]
    geometry = build_geometry(mask)
    assert torch.equal(geometry.body, old_body)
    assert torch.equal(geometry.body + geometry.rest, torch.ones_like(mask))
    assert (geometry.body <= geometry.foreground).all()
    assert torch.equal(geometry.body + geometry.inner_band, mask)
    assert (geometry.signed_distance[mask.bool()] > 0).all()
    assert (geometry.signed_distance[~mask.bool()] < 0).all()
    assert torch.isfinite(geometry.signed_distance).all()
    if kind in ['empty', 'full']:
        assert not geometry.valid.any()
    if kind == 'thin':
        assert geometry.body.sum() == 0


def test_inner_lines_are_candidates_and_cache_is_versioned(tmp_path):
    mask = rectangle(96)
    sketch = torch.ones(1, 3, 96, 96)
    sketch[:, :, 47:49, 30:66] = -1
    cache = GeometryCache(tmp_path)
    before = torch.get_rng_state().clone()
    geometry, key = cache.get('cloth-1', mask, sketch, {'sketch': 'abc'})
    restored, same = cache.get('cloth-1', mask, sketch, {'sketch': 'abc'})
    assert key == same
    assert torch.equal(geometry.body, restored.body)
    assert torch.equal(geometry.inner_edge, restored.inner_edge)
    assert geometry.inner_edge.sum() > 0
    assert not (geometry.inner_edge.bool() & geometry.outer_edge.bool()).any()
    assert cache.key('cloth-1', mask, sketch, {'sketch': 'changed'}) != key
    assert torch.equal(torch.get_rng_state(), before)
    no_line = build_geometry(mask, torch.ones_like(sketch))
    assert no_line.metadata['has_inner_line'] == [False]


def test_patch_windows_never_cross_body_or_fill_background_and_restore_rng():
    mask = rectangle(96)
    body = build_geometry(mask, details=False).body[0]
    image = torch.arange(3 * 96 * 96).reshape(3, 96, 96).float().requires_grad_()
    sampler = InteriorPatchSampler(123)
    global_state = torch.get_rng_state().clone()
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    state = sampler.state_dict()
    points, number = sampler.coordinates(body, 32)
    assert number > 4 and len(points) == len(set(points)) == 4
    for x, y in points:
        assert body[:, y:y + 32, x:x + 32].sum() == 32 ** 2
    patches = extract_patches(image, points, 32)
    patches.square().mean().backward()
    assert image.grad.abs().sum() > 0
    sampler.load_state_dict(state)
    assert sampler.coordinates(body, 32)[0] == points
    assert torch.equal(global_state, torch.get_rng_state())
    assert python_state == random.getstate()
    assert np.array_equal(numpy_state[1], np.random.get_state()[1])
    assert sampler.coordinates(body, 128) == ([], 0)
    exact = torch.ones(1, 32, 32)
    assert sampler.coordinates(exact, 32) == ([(0, 0)], 1)


def test_fft_identical_period_and_direction_have_expected_response():
    descriptor = PowerSpectrumDescriptor()
    for side in [32, 64]:
        x = (stripe(side, period=8) + 1) * .5
        same = descriptor(x)
        assert torch.equal(same, descriptor(x))
        period = descriptor((stripe(side, period=4) + 1) * .5)
        rotated = descriptor((stripe(side, period=8, horizontal=True) + 1) * .5)
        shifted = descriptor((stripe(side, period=8, phase=.2) + 1) * .5)
        assert (same - period).abs().mean() > .005
        assert (same - rotated).abs().mean() > .005
        assert (same - shifted).abs().mean() < (same - period).abs().mean()
        present = descriptor.grid(side, x.device)[2].nonzero().flatten()
        assert int(present[same.argmax()] % 8) == 0
        assert int(present[rotated.argmax()] % 8) == 4
        assert torch.allclose(same.sum(1), torch.ones(1), atol=1e-6)


@pytest.mark.parametrize('level', [0., .5, 1.])
def test_constant_descriptor_is_finite(level):
    value = torch.full((2, 3, 32, 32), level, requires_grad=True)
    output = PowerSpectrumDescriptor()(value)
    output.square().sum().backward()
    assert torch.isfinite(output).all() and torch.isfinite(value.grad).all()


def test_fft_backward_agrees_with_finite_difference():
    rng = torch.Generator().manual_seed(118)
    value = torch.rand(1, 3, 32, 32, generator=rng).requires_grad_()
    target = PowerSpectrumDescriptor()((stripe(32) + 1) * .5).detach()
    descriptor = PowerSpectrumDescriptor()
    direction = torch.randn(value.shape, generator=rng)
    direction /= direction.norm()

    def objective(x):
        return (descriptor(x) - target).square().sum()

    gradient = torch.autograd.grad(objective(value), value)[0]
    analytic = (gradient * direction).sum()
    epsilon = 2e-3
    numeric = (objective(value.detach() + epsilon * direction)
               - objective(value.detach() - epsilon * direction)) / (2 * epsilon)
    assert gradient.norm() > 0 and torch.isfinite(gradient).all()
    torch.testing.assert_close(analytic, numeric, atol=2e-5, rtol=.03)


def test_texture_reference_is_detached_and_flat_prediction_is_not_skipped():
    reference = stripe().requires_grad_()
    prediction = torch.zeros_like(reference, requires_grad=True)
    objective = LocalFFTLoss()
    loss, stats = objective(prediction, reference, rectangle(), torch.tensor([100]), torch.tensor([7]))
    assert stats['valid_n'] == 1 and loss > 0
    loss.backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
    assert reference.grad is None


def test_invalid_t_pure_reference_empty_body_and_small_reference():
    predicted = stripe().requires_grad_()
    mask = rectangle()
    for reference, geometry, timestep in [
            (torch.zeros_like(predicted), mask, 100),
            (stripe(), mask, 250),
            (stripe(), torch.zeros_like(mask), 100),
            (stripe(16), mask, 100)]:
        objective = LocalFFTLoss()
        loss, stats = objective(predicted, reference, geometry, torch.tensor([timestep]), torch.tensor([0]))
        assert loss == 0 and stats['valid_n'] == 0 and torch.isfinite(loss)
        assert torch.isfinite(torch.autograd.grad(loss, predicted, retain_graph=True)[0]).all()


def test_batch_reduction_counts_inactive_sample_in_denominator():
    prediction = stripe(period=4).requires_grad_()
    reference = stripe(period=8)
    mask = rectangle()
    one = LocalFFTLoss()
    two = LocalFFTLoss()
    loss, _ = one(prediction, reference, mask, torch.tensor([100]), torch.tensor([0]))
    doubled, stats = two(prediction.expand(2, -1, -1, -1), reference.expand(2, -1, -1, -1),
                         mask.expand(2, -1, -1, -1), torch.tensor([100, 999]), torch.tensor([0, 1]))
    torch.testing.assert_close(doubled, loss / 2, atol=0, rtol=0)
    assert stats['active_n'] == stats['valid_n'] == 1


def test_reference_dropout_and_high_t_leave_sampling_rng_unchanged():
    objective = LocalFFTLoss()
    state = objective.sampling_state()['rng'].clone()
    for keep, timestep in [(False, 100), (True, 999)]:
        loss, stats = objective(stripe(), stripe(), rectangle(), torch.tensor([timestep]),
                                torch.tensor([0]), reference_keep=keep)
        assert loss == 0 and stats['active_n'] == 0
    assert torch.equal(state, objective.sampling_state()['rng'])
