from types import SimpleNamespace

import numpy as np
import torch

from tools.e2_state_b_scaled_direction_bridge import (
    HEAD_TASKS, TASKS, head_derivatives, half_rgb, residue_bridge, three_level_fd)
from tools.e2_b_garment_projection import GarmentProjection


def test_three_layers_use_actual_stored_chords_and_sign():
    # 首个参数的名义1e-8变化写回FP32后消失，不能被计入预测。
    theta0 = torch.tensor([1., 0.], dtype=torch.float32)
    theta_plus = (theta0.double()+torch.tensor([1e-8, .1], dtype=torch.float64)).float()
    theta_minus = (theta0.double()-torch.tensor([1e-8, .1], dtype=torch.float64)).float()
    zplus, zminus = theta_plus[1:]*2, theta_minus[1:]*2
    row = three_level_fd(float(zplus*3), float(zminus*3), .1,
        torch.tensor([3.]), torch.tensor([1e9, 6.]), zplus, zminus, torch.tensor([0.]), theta_plus, theta_minus)
    assert row['parameter_chord_nonzero'] == 1 and row['latent_midpoint_offset_norm_fp64'] == 0
    assert row['B_head_q_latent_chord'] == row['C_cached_G_parameter_chord']
    assert row['total_difference']['same_strict_sign'] and row['A_head_value_fd'] > 0


def test_only_six_head_gradients_are_returned_unscaled():
    class Objective:
        def metric(self, entry):
            return SimpleNamespace(surrogate=lambda value: value.sum())
        def score_latents(self, entry, leaf, base, residue):
            return {task: leaf.square().sum()*(index+1) for index, task in enumerate(TASKS)}, {}
    z = torch.tensor([[[[.25, -.5]]]])
    q, values = head_derivatives(Objective(), {}, z, z)
    assert q.shape == (6,)+z.shape and len(values) == 7
    for index, task in enumerate(HEAD_TASKS):
        assert torch.equal(q[index], 2*z*(TASKS.index(task)+1))
    assert not z.requires_grad and z.grad is None


def test_half_normalization_precedes_float_cast_and_dtype_restores():
    class Vae:
        dtype = torch.float32
        config = SimpleNamespace(scaling_factor=.5)
        def half(self):
            self.dtype = torch.float16
        def to(self, dtype):
            self.dtype = dtype
        def decode(self, latent, return_dict):
            assert latent.dtype == torch.float16 and self.dtype == torch.float16
            return (torch.tensor([.1234, -.7777], dtype=torch.float16),)
    vae = Vae()
    result = half_rgb(vae, torch.ones(1))
    decoded = torch.tensor([.1234, -.7777], dtype=torch.float16)
    assert torch.equal(result, (decoded/2+.5).clamp(0, 1).float())
    assert not torch.equal(result, (decoded.float()/2+.5).clamp(0, 1))
    assert vae.dtype == torch.float32


def test_signed_bridge_preserves_original_abs_and_exact_png():
    random = np.random.default_rng(17)
    positive = random.integers(40, 210, size=(48, 48, 3), dtype=np.uint8)
    anchor = random.integers(20, 230, size=(48, 48, 3), dtype=np.uint8)
    mask = np.zeros((48, 48), dtype=bool)
    mask[2:-2, 2:-2] = True
    metric = GarmentProjection(positive, anchor, mask, 'cpu')
    rgb32 = torch.from_numpy(anchor).permute(2, 0, 1)[None].float()/255
    rgb16 = rgb32.half().float()
    png = np.rint(rgb16[0].permute(1, 2, 0).numpy()*255).astype(np.uint8)
    result = residue_bridge(metric, rgb32, rgb16, png)
    assert result['png_equal']
    for station in result['stations'].values():
        assert abs(station['signed']) == station['absolute']
    assert result['stations']['uint8_opencv_exact']['absolute'] == metric.exact(png)
