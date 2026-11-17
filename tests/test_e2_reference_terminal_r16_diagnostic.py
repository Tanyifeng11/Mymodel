from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from tools.e2_reference_terminal_r16_diagnostic import native_rgb_half, quantize, repeat_report, scalar_stations
from tools.e2_state_b_scaled_retry import scaled_scalar


class ToyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('gain', torch.tensor(.7, dtype=torch.float16))
        self.config = SimpleNamespace(scaling_factor=.18215)
        self.calls = []
    @property
    def dtype(self):
        return self.gain.dtype
    def decode(self, value, return_dict=False):
        self.calls.append((value.detach().clone(), self.dtype))
        return (value*self.gain+.03,)


def test_native_rgb_preserves_half_division_normalization_clamp_and_png():
    vae = ToyVAE()
    latent = torch.linspace(-.4, .4, 30).reshape(1, 3, 2, 5)
    output = native_rgb_half(vae, latent)
    expected_input = latent.half()/.18215
    decoded = expected_input*vae.gain+.03
    expected = (decoded/2+.5).clamp(0, 1)
    assert output.dtype == torch.float16 and torch.equal(output, expected)
    assert torch.equal(vae.calls[0][0], expected_input)
    early_float = (decoded.float()/2+.5).clamp(0, 1)
    assert not torch.equal(output.float(), early_float)
    png = np.rint(expected[0].permute(1, 2, 0).float().numpy()*255).astype(np.uint8)
    assert np.array_equal(quantize(output), png)


def test_checkpoint_and_scaled_backward_keep_vae_half_until_graph_finishes():
    vae = ToyVAE(); value = nn.Parameter(torch.tensor([[[[.01, .03]]]], dtype=torch.float32))
    before = vae.gain.clone()
    rgb = checkpoint(lambda x: native_rgb_half(vae, x), value, use_reentrant=False)
    gradient = scaled_scalar(rgb.float().square().mean(), [value])
    assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient) == 2
    assert len(vae.calls) == 2 and all(dtype == torch.float16 for _, dtype in vae.calls)
    assert value.grad is None and not any(p.requires_grad for p in vae.parameters())
    del rgb
    vae.float()
    assert torch.equal(vae.gain, before.float())
    with pytest.raises(AssertionError):
        native_rgb_half(vae, value)


def test_repeat_gate_rejects_large_difference_and_nonfinite():
    reference = torch.tensor([1., 2., 3.])
    assert repeat_report(reference.clone(), reference)['passed']
    assert not repeat_report(reference*1.01, reference)['passed']
    assert not repeat_report(torch.tensor([float('inf'), 2., 3.]), reference)['passed']


def test_stations_keep_saved_r32_metadata_and_isolate_uint8_rounding():
    from tools.e2_reference_terminal_global_amplitude import global_loss
    native = torch.linspace(.2, .7, 3*20*20).reshape(1, 3, 20, 20).half()
    positive = native.float()*.8+.1; weight = torch.ones(1, 1, 20, 20)
    metric = SimpleNamespace(surrogate=lambda rgb: (rgb.square().mean()-.1).abs())
    image = quantize(native); fp32 = dict(global_amplitude=123., residue=456.)
    official = dict(l_amplitude=.5, l_correlation=.6, l_target_gain=.3, old_projection=.2)
    values = scalar_stations(native, image, positive, weight, metric, fp32, official)
    quantized = torch.from_numpy(image).permute(2, 0, 1)[None].float()/255
    assert values['R32_metadata'] == fp32 and values['official_PNG'] == official
    assert values['R16']['global_amplitude'] == float(global_loss(native.float(), positive, weight))
    assert values['R16_uint8_analytic']['residue'] == float(metric.surrogate(quantized))
    assert values['R16']['residue'] != values['R16_uint8_analytic']['residue']
