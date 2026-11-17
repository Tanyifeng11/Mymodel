import numpy as np
import torch

from tools.e2_reference_terminal_fp32_delta_probe import anchored_rgb, quantize


def test_zero_float_difference_preserves_native_midgray_png_ties():
    native = torch.tensor([0., .5, 1.], dtype=torch.float16).reshape(1, 3, 1, 1)
    reference = torch.tensor([-.2, .7, 1.3]).reshape_as(native)
    output = anchored_rgb(native, reference, reference.clone())
    assert torch.equal(output, native.float())
    assert np.array_equal(quantize(output), np.array([[[0, 128, 255]]], dtype=np.uint8))


def test_difference_branch_keeps_fp32_increment_and_gradient():
    native = torch.full((1, 3, 2, 2), .5, dtype=torch.float16)
    reference = torch.full_like(native.float(), .7)
    current = (reference+1e-5).requires_grad_()
    output = anchored_rgb(native, reference, current)
    assert output.dtype == torch.float32
    assert (output > native.float()).all() and torch.equal(output.half(), native)
    assert torch.equal(torch.autograd.grad(output.sum(), current)[0], torch.ones_like(current))


def test_png_anchor_exposes_rounding_phase_without_extra_decode():
    native = torch.full((1, 3, 1, 1), .5, dtype=torch.float16)
    reference = torch.full_like(native.float(), .7)
    png = torch.from_numpy(quantize(native).copy()).permute(2, 0, 1)[None].float()/255
    assert np.array_equal(quantize(native), quantize(png))
    shifted_native = anchored_rgb(native, reference, reference-1e-5)
    shifted_png = anchored_rgb(png, reference, reference-1e-5)
    assert (quantize(shifted_native) == 127).all()
    assert (quantize(shifted_png) == 128).all()
