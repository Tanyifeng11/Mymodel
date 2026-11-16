"""固定缩放只改变反传精度路径，资格阈值严格使用预设值。"""
import pytest
import torch

from tools.e2_b_endpoint_losses import TASKS
from tools.e2_state_b_scaled_retry import (
    SCALE, SENTINELS, NumericalFailure, finite, replica_report, scaled_scalar, scaled_tasks)


def test_full_task_scaling_preserves_values_and_matches_joint_gradient():
    parameter = torch.nn.Parameter(torch.tensor([.25, -.5]))
    tasks = {key: (i + 1) * parameter.square().sum() for i, key in enumerate(TASKS)}
    before = {key: value.item() for key, value in tasks.items()}
    gradients = scaled_tasks(tasks, [parameter])
    expected = torch.stack([(i + 1) * 2 * parameter.detach() for i in range(7)])
    assert torch.equal(gradients, expected)
    assert {key: value.item() for key, value in tasks.items()} == before
    assert parameter.grad is None


def test_late_y_and_absolute_both_use_scaling_without_changing_value():
    parameter = torch.nn.Parameter(torch.tensor([.25, -.5]))
    y = parameter.square().sum()
    g = parameter.sum()
    gy = scaled_scalar(y, [parameter], retain_graph=True)
    ga = scaled_scalar((g - 1).square(), [parameter])
    assert torch.equal(gy, 2 * parameter.detach())
    assert torch.equal(ga, torch.full_like(parameter, 2 * (float(parameter.detach().sum()) - 1)))
    assert float(y.detach()) == .3125 and float(g.detach()) == -.25


def test_scale_restores_half_boundary_gradient_and_nonfinite_is_failure():
    parameter = torch.nn.Parameter(torch.tensor([.5]))
    tiny = scaled_scalar(parameter.half().float().sum() * 1e-9, [parameter])
    assert tiny.item() > 0 and SCALE == 1024
    bad = scaled_scalar(parameter.half().float().sum() * 100, [parameter])
    with pytest.raises(NumericalFailure) as failure:
        finite(bad, 'late_y')
    assert failure.value.details['summary']['nonfinite'] == 1


def test_replica_gate_rejects_relative_error_cosine_and_nonfinite():
    baseline = torch.tensor([[1., 0.]]).repeat(7, 1)
    assert replica_report(baseline, baseline)['passed']
    assert replica_report(baseline * 1.004, baseline)['passed']
    assert not replica_report(baseline * 1.006, baseline)['passed']
    rotated = baseline.clone()
    rotated[3, 1] = .0049
    assert not replica_report(rotated, baseline)['passed']
    bad = baseline.clone()
    bad[0, 0] = float('nan')
    assert not replica_report(bad, baseline)['passed']


def test_eight_sentinels_have_two_fixed_conditions_per_case():
    assert len(SENTINELS) == len(set(SENTINELS)) == 8
    for case in {row[0] for row in SENTINELS}:
        assert sum(row[0] == case for row in SENTINELS) == 2


def test_aggregate_writes_rejection_before_nonzero_exit(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tools import e2_state_b_scaled_retry as module
    monkeypatch.setattr(module, 'verify', lambda args: {})
    monkeypatch.setattr(module, 'sha', lambda path: 'checked')
    (tmp_path / 'gradients').mkdir()
    cases = sorted({row[0] for row in SENTINELS})
    for index, case in enumerate(cases):
        replicas = [dict(case=c, condition=condition, arm=arm,
            max_relative_l2=.006, minimum_cosine=1.)
            for c, condition, arm in SENTINELS if c == case]
        module.write(tmp_path / 'gradients' / ('shard%d.json' % index), dict(
            tensor_sha256='checked', overlay_sha256='checked', replicas=replicas,
            all_finite=True, numerical_qualification=False))
    with pytest.raises(SystemExit) as failure:
        module.aggregate(SimpleNamespace(out=tmp_path))
    assert failure.value.code == 2
    assert not module.read(tmp_path / 'numerical_qualification.json')['passed']
