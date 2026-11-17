from types import SimpleNamespace

import pytest
import torch

from tools.e2_b_endpoint_losses import luminance
from tools.e2_reference_terminal_global_amplitude import global_loss, replace_direction, source_context


def test_global_amplitude_preserves_original_formula_and_gradient():
    torch.manual_seed(51)
    positive = torch.rand(1, 3, 80, 80)*.4+.3
    rgb = (positive*.7+.12).requires_grad_()
    weight = torch.ones(1, 1, 80, 80); weight[..., :4, :] = .5
    actual = global_loss(rgb, positive, weight)
    expected = luminance(rgb, positive, weight, [])[0]['amplitude']
    assert torch.equal(actual, expected)
    ga = torch.autograd.grad(actual, rgb, retain_graph=True)[0]
    gb = torch.autograd.grad(expected, rgb)[0]
    assert torch.equal(ga, gb) and torch.count_nonzero(ga) > 0


def test_direction_only_replaces_mean_amplitude_without_auxiliary_renormalization():
    source_g = torch.zeros(48, 7, 4616)
    source_g[:, 1, :2] = torch.tensor([2., 4.])
    global_g = source_g[:, 1].clone(); global_g[:, :2] += torch.tensor([1., -2.])
    saved = torch.arange(4616, dtype=torch.float32)/32
    original = saved.clone(); weights = [1., .25, 3., 1., 2., 1., 1.]
    changed, replacement = replace_direction(saved, source_g, global_g, weights)
    expected = torch.zeros_like(saved); expected[:2] = torch.tensor([.25, -.5])
    assert torch.equal(replacement, expected) and torch.equal(changed, original+expected)
    assert torch.equal(saved, original) and torch.equal(source_g[:, 1, :2], torch.tensor([2., 4.]).expand(48, 2))
    unchanged, zero = replace_direction(saved, source_g, source_g[:, 1], weights)
    assert torch.equal(unchanged, saved) and torch.count_nonzero(zero) == 0


def test_source_qualification_failure_blocks_new_experiment(tmp_path, monkeypatch):
    from tools import e2_reference_terminal_global_amplitude as module
    qualification = {'passed': False}
    monkeypatch.setattr(module.source, 'verify', lambda args: {})
    monkeypatch.setattr(module.source, 'gradient_inputs', lambda args, overlay: ([], {}, {}, qualification))
    monkeypatch.setattr(module, 'read', lambda path: qualification)
    args = SimpleNamespace(source_run=tmp_path/'source', out=tmp_path/'new')
    with pytest.raises(AssertionError):
        source_context(args)
    assert not args.out.exists()
