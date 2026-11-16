"""有限步符号、实际FP32端点和相对差分母的小向量核验。"""
import torch

from tools.e2_b_finite_step_audit import actual_direction, compare_step, summarize


def test_direction_uses_stored_fp32_endpoints_not_nominal_step():
    start = torch.tensor([1., 1e-6], dtype=torch.float32)
    target = (start.double() + torch.tensor([1e-8, .1], dtype=torch.float64)).float()
    actual = actual_direction(start, target)
    assert actual.dtype == torch.float64 and actual[0] == 0
    assert torch.equal(actual, target.double() - start.double())
    assert actual[1] != (target - start).double()[1]


def test_prediction_sign_and_original_tolerance_match_observation():
    row = compare_step(torch.tensor([1., -2.]), torch.tensor([-.1, .2]), baseline=2., observed=1.6)
    assert row['predicted_delta'] < 0 and row['observed_delta'] < 0 and row['sign_agrees']
    assert not row['observed_task_failure'] and row['tolerance'] == 1e-5 + 1e-4 * 2.
    opposite = compare_step(torch.tensor([1.]), torch.tensor([-1.]), baseline=1., observed=1.1)
    assert opposite['predicted_sign'] == 'decrease' and opposite['observed_task_failure']
    assert not opposite['sign_agrees']


def test_zero_floor_and_relative_mismatch_summary_are_explicit():
    zero = compare_step(torch.tensor([0.]), torch.tensor([0.]), baseline=1., observed=1.)
    assert zero['relative_floor_used'] and zero['relative_mismatch'] == 0
    assert zero['relative_denominator'] == 1e-12
    opposite = compare_step(torch.tensor([1.]), torch.tensor([-1.]), baseline=1., observed=2.)
    assert opposite['relative_mismatch'] == 2
    result = summarize([zero, opposite])
    assert result['relative_floor_count'] == 1 and result['observed_task_failures'] == 1
    assert result['statistics']['relative_mismatch']['mean'] == 1
