import torch
import pytest

from models.e2_structure import StructureResidual, structure_maps


def fixture_maps():
    x = torch.zeros(1, 6, 64, 48)
    x[:, 2, 8:56, 8:40] = 1
    x[:, 1, 8, 8:40] = 1
    x[:, 4, 16:48, 24] = 1
    x[:, 0] = x[:, 1] + x[:, 4]
    x[:, 5] = 1
    return x


@pytest.mark.parametrize('mode', ['U1', 'A3', 'A4', 'A5'])
def test_zero_heads_and_shapes(mode):
    state = torch.get_rng_state().clone()
    module = StructureResidual((4, 8, 8, 8), mode=mode)
    assert torch.equal(state, torch.get_rng_state())
    sizes = [(8, 6), (4, 3), (2, 2), (1, 1)]
    result = module(fixture_maps(), sizes)
    assert [x.shape for x in result] == [(1, c, *s) for c, s in zip([4, 8, 8, 8], sizes)]
    assert all(torch.count_nonzero(x) == 0 for x in result)


def test_heads_then_backbone_gradients_and_restore():
    module = StructureResidual((4, 8, 8, 8))
    opt = torch.optim.SGD(module.parameters(), lr=.1)
    sizes = [(8, 6), (4, 3), (2, 2), (1, 1)]
    for step in range(2):
        loss = sum((x - 1).square().mean() for x in module(fixture_maps(), sizes))
        loss.backward()
        assert module.heads[0].weight.grad.norm() > 0
        if step:
            assert module.outer[0].weight.grad.norm() > 0
            assert module.inner[0].weight.grad.norm() > 0
        opt.step(); opt.zero_grad()
    state = {k: v.clone() for k, v in module.state_dict().items()}
    restored = StructureResidual((4, 8, 8, 8)); restored.load_state_dict(state, strict=True)
    for a, b in zip(module(fixture_maps(), sizes), restored(fixture_maps(), sizes)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_invalid_geometry_no_residual():
    module = StructureResidual((4, 8, 8, 8))
    for h in module.heads:
        torch.nn.init.ones_(h.bias)
    maps = fixture_maps(); maps[:, 5] = 0
    assert all(torch.count_nonzero(x) == 0 for x in module(maps, [(8, 6)] * 4))


def test_preprocessing_uses_sketch_and_is_deterministic():
    x = torch.ones(1, 3, 64, 48)
    x[:, :, 8:56, 8] = -1; x[:, :, 8:56, 39] = -1
    x[:, :, 8, 8:40] = -1; x[:, :, 55, 8:40] = -1
    x[:, :, 16:48, 24] = -1
    a = structure_maps(x); b = structure_maps(x)
    assert a.shape == (1, 6, 64, 48)
    assert torch.equal(a, b) and torch.isfinite(a).all()
    assert a[:, 1].sum() > 0 and a[:, 4].sum() > 0
    assert a[:, 5, 0, 0] == 1


def test_matching_parameter_counts():
    counts = [sum(p.numel() for p in StructureResidual(mode=m).parameters()) for m in ['U1', 'A3', 'A4', 'A5']]
    assert len(set(counts)) == 1
