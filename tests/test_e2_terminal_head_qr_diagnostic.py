import numpy as np
import torch

from tools.e2_terminal_head_qr_diagnostic import normalization, factorize, solve_arm


def test_normalization_preserves_original_fp32_order_and_active_rule():
    g = torch.tensor([[.1234567, .7654321, .3333333], [0., 0., 0.]], dtype=torch.float32)
    norms, active, rows = normalization(g)
    assert torch.equal(norms, g.norm(dim=1)) and active.tolist() == [True, False]
    assert rows.dtype == torch.float32 and torch.equal(rows, (g/g.norm(dim=1)[:, None])[:1])
    assert not torch.equal(rows.double(), (g.double()/g.double().norm(dim=1)[:, None])[:1])


def test_qr_slsqp_matches_analytic_projection_and_original_space_kkt():
    rows = np.array([[1., 0., 0.]])
    q, r, qr_report = factorize(rows)
    raw = torch.tensor([-1., .25, 0.], dtype=torch.float32)
    margin = np.array([float(torch.tensor(.1))])
    report, tensors = solve_arm(rows, q, raw, 1., margin, qr_report)
    expected = np.array([margin[0], .25, 0.])
    assert report['numerical_qualification'] and qr_report['passed']
    assert np.allclose(tensors['delta64'].numpy(), expected, atol=1e-12)
    assert abs(report['kkt']['primal']-.5*(1+margin[0])**2) < 1e-12
    assert abs(report['kkt']['gap']) < 1e-12 and report['kkt']['dual_feasible']
    assert report['kkt']['stationarity_relative'] < 1e-12
    assert report['kkt']['objective_equivalence_abs'] < 1e-12


def test_rank_deficient_rows_are_kept_and_feasible_raw_has_zero_correction():
    rows = np.array([[1., 0., 0.], [1., 0., 0.]])
    q, r, qr_report = factorize(rows)
    assert q.shape == (3, 2) and r.shape == (2, 2)
    report, tensors = solve_arm(rows, q, torch.tensor([1., .5, 0.]), 1., np.array([.1, .2]), qr_report)
    assert report['numerical_qualification'] and report['kkt']['active_count'] == 0
    assert torch.equal(tensors['v'], torch.zeros(2, dtype=torch.float64))
    assert torch.equal(tensors['lambda_'], torch.zeros(2, dtype=torch.float64))


def test_slsqp_failure_is_not_overridden_by_feasible_point(monkeypatch):
    import tools.e2_terminal_head_qr_diagnostic as module
    from types import SimpleNamespace
    monkeypatch.setattr(module, 'minimize', lambda *a, **kw: SimpleNamespace(x=np.array([0.]), fun=0.,
        success=False, status=9, message='fixed failure', nit=500, nfev=501))
    rows = np.array([[1., 0.]])
    q, _, qr_report = factorize(rows)
    report, _ = solve_arm(rows, q, torch.tensor([1., 0.]), 1., np.array([.1]), qr_report)
    assert report['checks']['original_fp32_feasible']
    assert not report['numerical_qualification'] and 'slsqp_success' in report['failed_checks']
