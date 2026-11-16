import numpy as np
import pytest

from tools.e2_b_weighted_geometry import candidate_margin, weighted_geometry


def test_unequal_margins_have_known_two_dimensional_optimum():
    result = weighted_geometry(np.eye(2), np.array([1., 2.]))
    expected = 1 / np.sqrt(5)
    assert result['lower_bound'] == pytest.approx(expected, abs=1e-9)
    assert result['upper_bound'] == pytest.approx(expected, abs=1e-9)
    assert result['interval_resolved'] and result['positive_common_direction_found']
    assert result['weights'] == pytest.approx([.2, .8], abs=1e-9)
    assert result['kkt']['frank_wolfe_gap'] < 1e-10
    assert result['reconstruction']['gram_dot_max_abs_error'] < 1e-12


def test_candidate_gamma_uses_each_margin_and_actual_fp32_endpoints():
    start = np.array([1., 0.], dtype=np.float32)
    target = np.array([1.-1e-8, -.1], dtype=np.float32)
    result = candidate_margin(np.eye(2), np.array([.01, .18]), start, target)
    assert result['gamma'] == 0  # 首个名义位移写回FP32后消失。
    result = candidate_margin(np.eye(2), np.array([.01, .18]), np.zeros(2), -np.ones(2))
    assert result['gamma'] == pytest.approx(1/np.sqrt(2)/.18)
    assert result['ordinary_minimum_cosine'] == pytest.approx(1/np.sqrt(2))
    assert result['minimum_weighted_row'] == 1
