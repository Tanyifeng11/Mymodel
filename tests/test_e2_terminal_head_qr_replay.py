import pytest
import torch

from tools.e2_terminal_head_projection_retry import original_step
from tools.e2_terminal_head_qr_replay import qualified_step


def config():
    return dict(scale=.5, displacement_cap=.007854645)


def test_saved_projection_is_applied_with_original_step_and_no_optimizer(monkeypatch):
    monkeypatch.setattr(torch.optim, 'AdamW', lambda *a, **k: (_ for _ in ()).throw(AssertionError('optimizer called')))
    raw = torch.tensor([-.001, -.002], dtype=torch.float32)
    delta = torch.tensor([.002, .003], dtype=torch.float32)
    gradients = torch.eye(2)
    rows, margins = gradients.double(), torch.tensor([.01, .18]).double()
    expected, old_report = original_step(raw, delta, gradients, config())
    actual, report = qualified_step(raw, delta, rows, margins, float(raw.norm()), gradients, config())
    assert torch.equal(actual, expected) and all(report[k] == v for k, v in old_report.items())
    assert report['projection']['feasible'] and report['parameter_updates'] == 1
    assert report['actual_norm'] <= float(raw.norm())*.500001


def test_fp32_projection_gate_is_checked_before_original_shrink():
    raw = torch.tensor([.001, 0.])
    delta = torch.tensor([.00001, 0.])
    gradients = torch.tensor([[1., 0.]])
    rows, margins = gradients.double(), torch.tensor([.01]).double()
    vector, report = qualified_step(raw, delta, rows, margins, float(raw.norm()), gradients, config())
    assert report['projection']['feasible']
    assert (rows@(-vector).double()/float(raw.norm())-margins).min() < -1e-6


def test_infeasible_saved_projection_cannot_create_candidate():
    raw = torch.tensor([.001, 0.])
    gradients = torch.tensor([[1., 0.]])
    with pytest.raises(AssertionError, match='FP32'):
        qualified_step(raw, torch.zeros_like(raw), gradients.double(), torch.tensor([.01]).double(),
            float(raw.norm()), gradients, config())
