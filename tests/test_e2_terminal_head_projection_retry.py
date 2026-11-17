import torch

from tools import e2_terminal_head_one_step as old
from tools.e2_state_b_backward_repro import tensor_sha
from tools.e2_terminal_head_projection_retry import reconstruct_raw, original_step, budget_diagnostic


def cfg():
    return dict(lr=3e-5, weight_decay=.01, gradient_clip=1., scale=.5,
        displacement_cap=.007854645, projection_sweeps=4096)


def test_raw_reconstruction_matches_original_two_parameter_adamw_bitwise(monkeypatch):
    fresh = lambda: [torch.nn.Parameter(torch.zeros(4, 64, 3, 3)), torch.nn.Parameter(torch.zeros(4))]
    direction = torch.linspace(.01, 4., 2308)
    captured = []
    def solver(raw, constraints, margins, max_sweeps, tolerance):
        captured.append(raw.clone())
        return raw.clone(), dict(feasible=True)
    monkeypatch.setattr(old, 'project_displacement', solver)
    reference = old.one_update(fresh(), direction, torch.ones(1, 2308), torch.tensor([.01]), cfg())
    reconstructed = reconstruct_raw(fresh(), direction, cfg())
    assert torch.equal(reconstructed['raw'], captured[0])
    assert reconstructed['raw_sha256'] == reference['raw_sha256']
    assert reconstructed['gradient_clip_return'] == reference['gradient_clip_return']


def fixtures(pass_first=False):
    raws = {}
    for arm, value in (('C0', 1.), ('C1', 2.)):
        vector = torch.tensor([value, 0.])*.001
        raws[arm] = dict(raw=vector, raw_sha256=tensor_sha(vector), raw_norm=float(vector.norm()),
            gradient_norm=1., gradient_clip_return=1.)
    calls = []
    def solver(raw, constraints, margins, max_sweeps, tolerance):
        arm = 'C0' if raw[0] < .0015 else 'C1'
        calls.append((max_sweeps, arm))
        passed = max_sweeps >= (16384 if pass_first or arm == 'C0' else 32768)
        report = dict(feasible=passed, actual_slack_min=0. if passed else -.0001,
            maximum_iterations=max_sweeps, iterations=max_sweeps, tolerance=tolerance)
        return raw.clone() if passed else torch.zeros_like(raw), report
    history = {}
    for arm, raw in raws.items():
        projected, projection = solver(raw['raw'], None, None, 8192, 1e-6)
        _, step = original_step(raw['raw'], projected, torch.tensor([[1., 0.]]), cfg())
        history[arm] = dict(projection=projection, **step, raw_norm=raw['raw_norm'],
            gradient_norm=1., gradient_clip_return=1., raw_sha256=raw['raw_sha256'])
    calls.clear()
    return raws, history, solver, calls


def test_budgets_restart_from_same_raw_and_require_shared_feasibility():
    raws, history, solver, calls = fixtures()
    result, vectors = budget_diagnostic(raws, torch.tensor([[1., 0.]]), torch.tensor([.01]), cfg(), history, solver)
    assert result['selected_budget'] == 32768 and result['all_feasible']
    assert calls == [(budget, arm) for budget in (8192, 16384, 32768) for arm in ('C0', 'C1')]
    assert not result['records'][1]['all_feasible']
    assert all(torch.equal(vectors[arm], -raws[arm]['raw']*.5) for arm in raws)


def test_first_shared_feasible_budget_stops_and_bad_historical_class_stops():
    raws, history, solver, calls = fixtures(pass_first=True)
    result, _ = budget_diagnostic(raws, torch.tensor([[1., 0.]]), torch.tensor([.01]), cfg(), history, solver)
    assert result['selected_budget'] == 16384 and len(calls) == 4
    calls.clear()
    history['C0']['gradient_norm'] = 2.
    failed, vectors = budget_diagnostic(raws, torch.tensor([[1., 0.]]), torch.tensor([.01]), cfg(), history, solver)
    assert failed['status'] == 'historical_reproduction_mismatch' and not failed['all_feasible']
    assert failed['selected_budget'] is None and vectors is None and len(calls) == 2


def test_original_step_preserves_zero_descent_sha_and_shrinks_nonzero():
    raw = torch.tensor([.001, 0.])
    zero, report = original_step(raw, torch.zeros_like(raw), torch.tensor([[1., 0.]]), cfg())
    assert report['parameter_updates'] == 0
    assert report['actual_descent_sha256'] == tensor_sha(torch.zeros_like(zero)-zero)
    assert report['actual_descent_sha256'] != tensor_sha(-zero)
    nonzero, report = original_step(raw, raw, torch.tensor([[1., 0.]]), cfg())
    assert report['actual_norm'] == float(raw.norm())*.5 and report['cap_scale'] == 1.
    assert torch.equal(nonzero, torch.zeros_like(raw)-raw*.5)
