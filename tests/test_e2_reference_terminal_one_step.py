from types import SimpleNamespace

import pytest
import torch
from torch import nn

from models.e2_reference_terminal_head import ReferenceTerminalHead
from tools.e2_reference_terminal_one_step import raw_once, effective_difference, qualified_preflight
from tools.e2_b_endpoint_losses import flatten


class ToyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.output = nn.Conv2d(64, 4, 3, padding=1)


def test_raw_once_matches_four_shape_adamw_and_only_one_independent_step():
    torch.manual_seed(11)
    shapes = [(4, 3, 2, 2), (4,), (4, 3, 2, 2), (4,)]
    actual = [nn.Parameter(torch.zeros(shape)) for shape in shapes]
    expected = [nn.Parameter(torch.zeros(shape)) for shape in shapes]
    direction = torch.randn(sum(p.numel() for p in actual))
    cfg = dict(lr=3e-5, weight_decay=.01, gradient_clip=1.)
    raw, report = raw_once(actual, direction, cfg)
    offset = 0
    for parameter in expected:
        parameter.grad = direction[offset:offset+parameter.numel()].reshape_as(parameter).clone()
        offset += parameter.numel()
    optimizer = torch.optim.AdamW(expected, lr=3e-5, weight_decay=.01)
    clipped = torch.nn.utils.clip_grad_norm_(expected, 1.)
    optimizer.step()
    assert torch.equal(raw, torch.zeros_like(raw)-flatten(expected))
    assert report['optimizer_steps'] == 1 and report['gradient_clip_return'] == float(clipped)
    assert report['raw_norm'] == float(raw.norm())


def test_effective_difference_uses_actual_fp32_merge_before_double_subtraction():
    head = ReferenceTerminalHead(ToyBase())
    with torch.no_grad():
        head.base.output.weight.fill_(1.)
        head.base.output.bias.fill_(1.)
        head.delta_weight0.fill_(2**-25)
        head.delta_bias0.fill_(2**-25)
        head.delta_weight1.fill_(2**-24)
        head.delta_bias1.fill_(2**-24)
    effective, ideal = effective_difference(head, torch.tensor(.25))
    assert torch.count_nonzero(ideal) == 2308
    assert torch.count_nonzero(effective) == 0  # 小增量确实被FP32权重合并吞掉。
    assert effective.dtype == torch.float64 and ideal.dtype == torch.float32
    other, _ = effective_difference(head, torch.tensor(2.))
    assert torch.count_nonzero(other) == 2308
    assert all(not p.requires_grad for p in head.base.parameters())


def test_preflight_failure_is_a_hard_gate_before_new_run(tmp_path, monkeypatch):
    from tools import e2_reference_terminal_one_step as module
    basis = tmp_path/'reference_terminal_preflight_v2'
    basis.mkdir()
    protocol = {'parameter_count': 4616}
    monkeypatch.setattr(module.preflight, 'verify', lambda args: protocol)
    monkeypatch.setattr(module, 'read', lambda path: {'protocol': protocol, 'passed': False})
    args = SimpleNamespace(basis=basis, out=tmp_path/'new_run')
    with pytest.raises(AssertionError, match='资格未通过'):
        qualified_preflight(args)
    assert not args.out.exists()
