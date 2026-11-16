"""用已知凸包几何验证独立于SLSQP状态的上下界。"""
import numpy as np
import pytest

from tools.e2_b_gradient_geometry import geometry


@pytest.mark.parametrize('angle', [30., 90., 150., 179.])
def test_two_unit_rows_known_half_angle(angle):
    radians = np.deg2rad(angle)
    rows = np.array([[1., 0.], [np.cos(radians), np.sin(radians)]])
    result = geometry(rows)
    expected = np.cos(radians/2)
    assert result['lower_bound'] == pytest.approx(expected, abs=1e-10)
    assert result['upper_bound'] == pytest.approx(expected, abs=1e-10)
    assert result['interval_resolved'] and result['positive_common_direction_found']
    assert result['kkt']['frank_wolfe_gap'] < 1e-12
    assert result['reconstruction']['gram_dot_max_abs_error'] < 1e-12


def test_opposing_rows_have_zero_upper_bound_without_dividing_by_zero():
    result = geometry(np.array([[1., 0.], [-1., 0.]]))
    assert result['upper_bound'] == 0
    assert result['lower_bound'] is None
    assert result['near_zero_mix'] and not result['positive_common_direction_found']
    assert result['mixed_direction_cosines'] is None


def test_duplicate_rows_nonunique_weights_still_give_valid_geometry():
    result = geometry(np.array([[1., 0.], [1., 0.], [0., 1.]]))
    assert result['upper_bound'] == pytest.approx(2**-.5, abs=1e-8)
    assert result['lower_bound'] == pytest.approx(2**-.5, abs=1e-8)
    assert result['simplex']['sum_error'] < 1e-12


def test_iteration_limit_keeps_honest_not_yet_tight_bounds():
    rows = np.array([[1., 0.], [1., 0.], [0., 1.]])
    result = geometry(rows, maxiter=1)
    expected = 2**-.5
    assert result['lower_bound'] <= expected <= result['upper_bound']
    assert not result['solver']['success']
    assert not result['interval_resolved']
