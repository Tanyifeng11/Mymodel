import torch

from tools.e2_reference_terminal_fp32_direction import continuous_latent
from tools.e2_reference_terminal_fp32_delta_probe import anchored_rgb


def test_zero_f68_anchor_and_small_increment_survive_half_grid():
    base = torch.tensor([.5], dtype=torch.float16)
    b = torch.tensor([.2]); parameter = torch.tensor([1e-5], requires_grad=True)
    z = continuous_latent(base, torch.tensor(.3), b+parameter, b)
    assert z.dtype == torch.float32 and z > base.float()
    assert torch.equal(z.half(), base)
    zero = continuous_latent(base, torch.tensor(.3), b, b)
    assert torch.equal(zero, base.float())


def test_nonlinear_anchored_branch_derivative_agrees_with_central_probe():
    base = torch.tensor([.5]); b = torch.tensor([.2]); theta = torch.zeros(1, requires_grad=True)
    def response(value):
        z = continuous_latent(base, torch.tensor(.3), b+value, b)
        return anchored_rgb(base, base.sigmoid(), z.sigmoid()).square().sum()
    gradient = torch.autograd.grad(response(theta), theta)[0]
    h = 1e-3
    central = (response(torch.tensor([h]))-response(torch.tensor([-h])))/(2*h)
    assert torch.allclose(gradient, central, rtol=.002, atol=1e-5)
