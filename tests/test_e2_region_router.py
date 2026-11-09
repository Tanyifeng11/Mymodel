import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from models.e2_region_router import RegionRouter, geometry_level, sketch_features, LEARNED


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def inputs():
    maps = torch.zeros(1, 6, 32, 24)
    maps[:, 2, 4:28, 4:20] = 1
    maps[:, 1, 4, 4:20] = 1
    maps[:, 4, 10:22, 12] = 1
    maps[:, 3] = -.5
    maps[:, 3, 4:28, 4:20] = .5
    maps[:, 5] = 1
    sizes = [(8, 6), (4, 3), (2, 2), (1, 1)]
    u = [torch.randn(1, c, *s) for c, s in zip([4, 8, 8, 8], sizes)]
    sk = [torch.randn(1, c, *s) for c, s in zip([3, 6, 6, 6], sizes)]
    return u, sk, maps


def module():
    return RegionRouter((3, 6, 6, 6), (4, 8, 8, 8), width=4)


def test_init_rng_and_parameter_matching():
    state = torch.get_rng_state().clone(); counts = []
    for mode in LEARNED:
        m = module(); counts.append(sum(p.numel() for p in m.parameters()))
        assert torch.equal(state, torch.get_rng_state())
        assert all(torch.count_nonzero(h.weight) == 0 for h in m.structure)
    assert len(set(counts)) == 1


def test_eta_zero_is_exact_legacy_identity_and_needs_no_geometry():
    m = module(); u, _, _ = inputs()
    result, stats = m(u, None, None, 'C6', 0)
    assert result is u and stats == []


@pytest.mark.parametrize('mode', LEARNED)
def test_finite_gains_softmax_and_update_gradients(mode):
    m = module(); u, s, maps = inputs(); m.save_maps = True
    opt = torch.optim.SGD(m.parameters(), lr=.1)
    for step in range(3):
        output, stats = m(u, s, maps, mode, 1.)
        assert all(torch.isfinite(x).all() for x in output)
        assert all(.5 <= st['app_gain_range'][0] <= st['app_gain_range'][1] <= 1. for st in stats)
        assert all(torch.allclose(g.sum(1), torch.ones_like(g[:, 0])) for g in m.gate_maps)
        loss = sum((x - 1).square().mean() for x in output); loss.backward()
        assert m.structure[0].weight.grad.norm() > 0
        assert m.gates[0][-1].weight.grad.norm() > 0
        if step:
            assert m.reduce_s[0].weight.grad.norm() > 0 and m.reduce_u[0].weight.grad.norm() > 0
        opt.step(); opt.zero_grad()
    if mode == 'C3':
        assert all(max(st['variance']) < 1e-12 for st in stats)
        assert all(torch.equal(g, g[:, :, :1, :1].expand_as(g)) for g in m.gate_maps)


def test_invalid_geometry_falls_back_and_restore_exact():
    m = module(); u, sk, maps = inputs(); maps[:, 5] = 0
    for h in m.structure: nn.init.ones_(h.bias)
    actual, _ = m(u, sk, maps, 'C6', 1.)
    assert all(torch.equal(a, b) for a, b in zip(actual, u))
    restored = module(); restored.load_state_dict(copy.deepcopy(m.state_dict()), strict=True)
    maps[:, 5] = 1
    a, _ = m(u, sk, maps, 'C6', .7); b, _ = restored(u, sk, maps, 'C6', .7)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_c4_ignores_all_geometric_priors_c5_ignores_inner_only():
    m = module(); u, sk, maps = inputs()
    a, _ = m(u, sk, maps, 'C4', 1.)
    other = maps.clone(); other[:, :5] = torch.rand_like(other[:, :5])
    b, _ = m(u, sk, other, 'C4', 1.)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    a, _ = m(u, sk, maps, 'C5', 1.)
    other = maps.clone(); other[:, 4] = 1 - other[:, 4]
    b, _ = m(u, sk, other, 'C5', 1.)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_fixed_distance_soft_is_not_hard_mask():
    _, _, maps = inputs(); maps[:, 3] = .01
    hard, soft, *_ = geometry_level(maps, maps.shape[-2:])
    assert torch.any(hard == 0) and torch.all((soft > 0) & (soft < 1))


def test_shared_sketch_path_has_no_inplace_autograd_breakage():
    fusion = SimpleNamespace(downconv1=nn.Conv2d(1, 2, 3, padding=1), resblocks=nn.Identity(),
        downconv_channel_unchanged=nn.Conv2d(2, 2, 3, padding=1))
    for i in range(2, 6):
        setattr(fusion, 'downconv%d' % i, nn.Sequential(nn.LeakyReLU(.2, inplace=True), nn.Conv2d(2, 2, 3, padding=1)))
    x = torch.randn(1, 1, 16, 16, requires_grad=True)
    features = sketch_features(fusion, x)
    sum(f.square().mean() for f in features).backward()
    assert torch.isfinite(x.grad).all() and x.grad.norm() > 0
