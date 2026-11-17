import torch

from tools.e2_reference_terminal_post_latent_probe import coefficient, post_latent


def test_coefficient_matches_unclipped_eta0_ddim_B_change():
    a, ap = torch.tensor(.9983, dtype=torch.float64), torch.tensor(.99915, dtype=torch.float64)
    z = torch.linspace(-1, 1, 16, dtype=torch.float64)
    raw = z*.4+.2
    delta = z*.03
    def ddim(value):
        eps = raw-(a/(1-a)).sqrt()*value
        x0 = (z-(1-a).sqrt()*eps)/a.sqrt()
        return ap.sqrt()*x0+(1-ap).sqrt()*eps
    observed = ddim(delta)-ddim(torch.zeros_like(delta))
    assert torch.allclose(observed, coefficient(a, ap)*delta, atol=3e-16, rtol=1e-12)


def test_zero_saved_B_change_preserves_half_latent_and_one_mask():
    z = torch.linspace(-1, 1, 48).reshape(1, 4, 3, 4).half()
    base_delta = z.float()*.1
    a, ap = torch.tensor(.9983), torch.tensor(.99915)
    zero = post_latent(z, base_delta, base_delta, a, ap)
    assert torch.equal(z.view(torch.int16), zero.view(torch.int16))
    masked_change = torch.zeros_like(base_delta); masked_change[..., :2] = .01
    out = post_latent(z, base_delta, base_delta+masked_change, a, ap)
    assert torch.equal(out[..., 2:], z[..., 2:])
    assert torch.count_nonzero(out[..., :2]-z[..., :2]) > 0
