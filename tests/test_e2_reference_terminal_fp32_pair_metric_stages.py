import torch

from tools.e2_b_pair_finetune_losses import projection
from tools.e2_reference_terminal_fp32_pair_metric_stages import projection_typed


def test_typed_float32_matches_original_projection_values_and_gradient():
    torch.manual_seed(21)
    rgb = (.2+.6*torch.rand(1, 3, 32, 32)).requires_grad_()
    positive = .2+.6*torch.rand_like(rgb)
    base = .2+.6*torch.rand_like(rgb)
    weight = torch.ones(1, 1, 32, 32)
    original = projection(rgb, positive, base, weight)
    current = projection_typed(rgb, positive, base, weight, torch.float32)
    for name in original:
        assert torch.equal(original[name], current[name])
        if not original[name].requires_grad:
            assert not current[name].requires_grad
            continue
        a = torch.autograd.grad(original[name], rgb, retain_graph=True)[0]
        b = torch.autograd.grad(current[name], rgb, retain_graph=True)[0]
        assert torch.equal(a, b)


def test_float64_target_identity_has_unit_gain_and_zero_orthogonal_error():
    torch.manual_seed(22)
    positive = .2+.6*torch.rand(1, 3, 32, 32)
    weight = torch.ones(1, 1, 32, 32)
    value = projection_typed(positive, positive, positive, weight, torch.float64)
    assert value['g'].dtype == torch.float64
    assert value['x'] == value['y'] == value['g'] == 1
    assert value['orth'] == 0
