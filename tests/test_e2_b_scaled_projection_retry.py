"""固定预算顺序、双臂共用预算及局部包装恢复；不读取真实大梯度。"""
from types import SimpleNamespace

import pytest
import torch

from tools.e2_b_scaled_projection_retry import (
    ARMS, budget_diagnostic, override_budget, same_numbers, tensor_sha, reproduction_comparison)


def inputs():
    raw = torch.tensor([1., -.1])
    raws = {arm: dict(raw=raw, start=torch.tensor([.2, .3]), raw_norm=float(raw.norm()),
        gradient_norm=3., raw_sha256=tensor_sha(raw), direction_sha256='direction') for arm in ARMS}
    cfg = dict(scale=.5, displacement_cap=.1)
    return raws, torch.eye(2), torch.tensor([.01, .18]), cfg


def solver_and_baseline(pass_at):
    calls = []
    def solver(raw, constraints, margins, max_sweeps, tolerance):
        arm = ARMS[len(calls) % 2]
        calls.append((max_sweeps, arm))
        feasible = max_sweeps >= pass_at[arm]
        report = dict(feasible=feasible, maximum_iterations=max_sweeps,
            actual_slack_min=0. if feasible else -1e-4)
        return (raw.clone() if feasible else torch.zeros_like(raw)), report
    raws, _, _, _ = inputs()
    baseline = dict(arms={arm: dict(raw_norm=raws[arm]['raw_norm'], gradient_norm=3.,
        actual_norm=0., parameter_updates=0, projection=dict(feasible=False,
            maximum_iterations=4096, actual_slack_min=-1e-4)) for arm in ARMS})
    return solver, baseline, calls


def test_common_first_feasible_budget_stops_at8192():
    solver, baseline, calls = solver_and_baseline(dict(C0=8192, C1=8192))
    result = budget_diagnostic(*inputs(), baseline, solver)
    assert result['selected_budget'] == 8192 and result['all_feasible']
    assert calls == [(4096, 'C0'), (4096, 'C1'), (8192, 'C0'), (8192, 'C1')]


def test_one_arm_failure_requires_both_arms_at16384():
    solver, baseline, calls = solver_and_baseline(dict(C0=16384, C1=8192))
    result = budget_diagnostic(*inputs(), baseline, solver)
    assert result['selected_budget'] == 16384
    assert calls[-2:] == [(16384, 'C0'), (16384, 'C1')]
    assert result['records'][1]['arms']['C1']['projection']['feasible']


def test_no_expansion_after16384_failure_and_zero_saved_candidates():
    solver, baseline, calls = solver_and_baseline(dict(C0=32768, C1=8192))
    result = budget_diagnostic(*inputs(), baseline, solver)
    assert result['selected_budget'] is None and not result['all_feasible']
    assert len(calls) == 6 and max(b for b, _ in calls) == 16384
    assert result['records'][0]['arms']['C0']['parameter_updates'] == 0


def test_original4096_mismatch_stops_before_budget_retry():
    solver, baseline, calls = solver_and_baseline(dict(C0=8192, C1=8192))
    baseline['arms']['C0']['raw_norm'] += .1
    with pytest.raises(AssertionError):
        budget_diagnostic(*inputs(), baseline, solver)
    assert calls == [(4096, 'C0')]


def test_local_override_restores_function_even_on_failure():
    def original():
        return 4096
    module = SimpleNamespace(project_displacement=original)
    with pytest.raises(RuntimeError):
        with override_budget(module, 8192, lambda fn, budget: lambda: budget):
            assert module.project_displacement() == 8192
            raise RuntimeError('stop')
    assert module.project_displacement is original


def test_comparison_preserves_booleans_and_projection_tolerance():
    assert same_numbers(dict(feasible=False, tolerance=1e-6), dict(feasible=False, tolerance=1e-6))
    assert not same_numbers(dict(feasible=True, tolerance=1e-6), dict(feasible=False, tolerance=1e-6))
    assert not same_numbers(dict(tolerance=1e-5), dict(tolerance=1e-6))


def test_reproduction_roundoff_is_reported_without_relaxing_classes():
    result = reproduction_comparison(dict(raw_norm=.01 + 1e-10, feasible=False), dict(raw_norm=.01, feasible=False))
    assert result['passed'] and not result['exact']
    assert result['floats']['raw_norm']['absolute_difference'] > 0
    assert not reproduction_comparison(dict(feasible=True), dict(feasible=False))['passed']
