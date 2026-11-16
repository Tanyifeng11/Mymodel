"""复用合格scaled梯度，只增加原Hildreth预算；CPU诊断与候选生成分开。"""
import argparse
from contextlib import contextmanager
import math
import subprocess
import time
from pathlib import Path

import torch

from tools.e2_state_b_amplitude_response import CHECKPOINT_SHA, read, write, sha
from tools.e2_state_b_pair_retry import PROTOCOL_SHA, copy_verified
from tools.e2_state_b_backward_repro import tensor_sha
from tools.e2_b_pair_finetune_losses import project_displacement


SOURCE = Path(__file__).resolve().parents[1]
BUDGETS = (4096, 8192, 16384)
ARMS = ('C0', 'C1')
TOLERANCE = 1e-6
REPRO_ATOL, REPRO_RTOL = 1e-10, 1e-6
EXTRA_SOURCES = ('tools/e2_b_scaled_projection_retry.py',
                 'scripts/e2_b_scaled_projection_retry.sbatch', 'tools/e2_b_scaled_replay_review.py',
                 'scripts/e2_b_projection_replay.sbatch')


class ReproductionMismatch(AssertionError):
    def __init__(self, arm, record):
        self.details = dict(arm=arm, record=record)
        super().__init__('4096结果超出预设CPU数值对照容差：' + arm)


def same_numbers(actual, expected):
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(same_numbers(actual[k], v) for k, v in expected.items())
    if isinstance(expected, float):
        return math.isclose(actual, expected, rel_tol=0., abs_tol=1e-12)
    return actual == expected


def reproduction_comparison(actual, expected):
    """只允许CPU重建的浮点末位差；布尔、整数和投影验收门槛不放宽。"""
    differences = {}
    def visit(left, right, prefix):
        if isinstance(right, dict):
            return left.keys() == right.keys() and all(visit(left[k], v, prefix + k + '.') for k, v in right.items())
        if isinstance(right, float):
            difference = abs(left - right)
            limit = REPRO_ATOL + REPRO_RTOL * abs(right)
            differences[prefix[:-1]] = dict(actual=left, reference=right, absolute_difference=difference,
                                          comparison_limit=limit, equal=left == right)
            return difference <= limit
        return left == right
    passed = visit(actual, expected, '')
    return dict(passed=passed, exact=actual == expected, absolute_tolerance=REPRO_ATOL,
                relative_tolerance=REPRO_RTOL, floats=differences)


def prepare(args):
    assert not args.out.exists(), '禁止覆盖已有重试目录'
    prior = read(args.source_run / 'scaled_overlay.json')
    qualification = read(args.source_run / 'numerical_qualification.json')
    spec = read(args.source_run / 'protocol.json')
    failed = read(args.source_run / 'training/proposals.json')
    assert qualification['passed'] and qualification['all_finite']
    assert qualification['maximum_relative_l2'] <= .005 and qualification['minimum_cosine'] >= .99999
    assert len(qualification['replicas']) == 8 and all(r['passed'] for r in qualification['replicas'])
    assert sha(args.source_run / 'protocol.json') == PROTOCOL_SHA
    assert sha(args.checkpoint) == spec['checkpoint_sha256'] == CHECKPOINT_SHA
    assert spec['training']['projection_sweeps'] == 4096
    assert spec['training']['default_margin'] == .01 and spec['training']['protected_amplitude_margin'] == .18
    # Linux冻结source须与新G生成时一致；不修改本地dirty或换行形式来绕过校验。
    assert all(sha(SOURCE / name) == value for name, value in prior['source_sha256'].items())
    selected = dict(prior['copied_sha256'])
    selected.update({name: sha(args.source_run / name)
                     for name in ('scaled_overlay.json', 'numerical_qualification.json')})
    inputs = dict(prior['input_sha256'])
    inputs[str(args.checkpoint)] = CHECKPOINT_SHA
    for index in range(4):
        pt, js = 'gradients/shard%d.pt' % index, 'gradients/shard%d.json' % index
        metadata = read(args.source_run / js)
        selected[pt] = qualification['new_gradient_sha256']['shard%d.pt' % index]
        assert selected[pt] == metadata['tensor_sha256']
        assert metadata['all_finite'] and metadata['numerical_qualification']
        assert metadata['overlay_sha256'] == selected['scaled_overlay.json']
        assert metadata['source_sha256'] == prior['source_sha256']
        selected[js] = sha(args.source_run / js)
    for arm in ARMS:
        row = failed['arms'][arm]
        assert row['parameter_updates'] == 0 and row['actual_norm'] == 0 and not row['adopted']
        assert row['initial_b_hash'] == row['final_b_hash']
        assert not row['projection']['feasible'] and row['projection']['maximum_iterations'] == 4096
        inputs[str(args.source_run / 'training' / (arm + '.pt'))] = row['checkpoint_sha256']
    inputs[str(args.source_run / 'training/proposals.json')] = sha(args.source_run / 'training/proposals.json')
    assert all(sha(Path(name)) == value for name, value in inputs.items())
    for name, expected in sorted(selected.items()):
        copy_verified(args.source_run / name, args.out / name, expected)
        inputs[str(args.source_run / name)] = expected
    copy_verified(args.source_run / 'training/proposals.json', args.out / 'source_failure/proposals.json',
                  inputs[str(args.source_run / 'training/proposals.json')])
    selected['source_failure/proposals.json'] = inputs[str(args.source_run / 'training/proposals.json')]
    overlay = dict(source_run=str(args.source_run), original_protocol_sha256=PROTOCOL_SHA,
        original_protocol_unchanged=True, source_gradient_job='125421', gradient_cache_reused=True,
        new_reverse_passes=0, source_gradient_seconds=failed['gradient_seconds'],
        source_gradient_sha256=qualification['new_gradient_sha256'], input_sha256=inputs,
        copied_sha256=selected, source_sha256=dict(prior['source_sha256'], **{
            name: sha(SOURCE / name) for name in EXTRA_SOURCES}),
        execution_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip(),
        budgets=list(BUDGETS), tolerance=TOLERANCE, internal_tolerance=TOLERANCE / 4,
        reproduction_comparison=dict(absolute_tolerance=REPRO_ATOL, relative_tolerance=REPRO_RTOL,
            scope='CPU historical numeric comparison only; feasibility and zero-update classes remain exact'),
        selection='4096 must reproduce both historical failures; then8192, only if either arm fails then16384; first shared feasible budget',
        algorithm='unchanged project_displacement Hildreth iteration from zero dual for each budget',
        only_change='maximum solver sweeps; no loss/gradient/margin/tolerance/optimizer/step/scale/cap/S0 change',
        new_candidates=0, adopted=False, holdout_used_for_selection=False)
    write(args.out / 'solver_overlay.json', overlay)
    verify_inputs(args, include_holdout=True)
    print('SOLVER_RETRY_PREPARED', flush=True)


def verify_inputs(args, include_holdout=False):
    overlay = read(args.out / 'solver_overlay.json')
    assert overlay['budgets'] == list(BUDGETS) and overlay['tolerance'] == TOLERANCE
    assert sha(args.out / 'protocol.json') == PROTOCOL_SHA
    assert sha(args.checkpoint) == CHECKPOINT_SHA
    for name, expected in overlay['input_sha256'].items():
        if include_holdout or 'holdout' not in Path(name).parts:
            assert sha(Path(name)) == expected, name
    for name, expected in overlay['copied_sha256'].items():
        if include_holdout or 'holdout' not in Path(name).parts:
            assert sha(args.out / name) == expected, name
    assert all(sha(SOURCE / name) == expected for name, expected in overlay['source_sha256'].items())
    qualification = read(args.out / 'numerical_qualification.json')
    assert qualification['passed'] and qualification['all_finite'] and qualification['loss_scale'] == 1024
    assert qualification['maximum_relative_l2'] <= .005 and qualification['minimum_cosine'] >= .99999
    assert qualification['new_gradient_sha256'] == overlay['source_gradient_sha256']
    return overlay


def raw_inputs(args):
    """逐式复用旧propose的方向/首步AdamW；临时参数不保存为候选。"""
    from tools import e2_state_b_pair_finetune as old
    shards = [torch.load(args.out / 'gradients' / ('shard%d.pt' % i), map_location='cpu', weights_only=False)
              for i in range(4)]
    assert all(s['protocol_sha256'] == PROTOCOL_SHA and s['all_finite'] and s['numerical_qualification'] for s in shards)
    assert len({s['b_hash'] for s in shards}) == len({s['net_hash'] for s in shards}) == 1
    cfg = read(args.out / 'protocol.json')['training']
    gradients = torch.cat([s['gradients'] for s in shards])
    assert gradients.shape[:2] == (48, 7) and torch.isfinite(gradients).all()
    rows = [row for s in shards for row in s['items']]
    weights = torch.tensor(read(args.out / 'protocol.json')['replay_weights'])
    merged = (gradients.mean(0) * weights[:, None]).sum(0)
    old_abs = torch.stack([s['absolute_gradient'] for s in shards if s['case'] in old.LEGACY_FOCUS]).mean(0)
    new_abs = torch.stack([s['absolute_gradient'] for s in shards if s['case'] not in old.LEGACY_FOCUS]).mean(0)
    pair = torch.stack([s['pair_gradient'] for s in shards]).mean(0)
    normalized = lambda x: x * (merged.norm() / x.norm().clamp_min(1e-12))
    directions = dict(C0=merged + normalized(old_abs),
        C1=merged + normalized(old_abs) + cfg['pair_gradient_norm_fraction'] * normalized(pair)
            + cfg['extra_absolute_gradient_norm_fraction'] * normalized(new_abs))
    constraints = gradients.flatten(0, 1)
    margins = torch.full((len(constraints),), cfg['default_margin'])
    for i, row in enumerate(rows):
        if [row['case'], row['condition']] == cfg['protected_condition']:
            margins[i * 7 + old.legacy_loss.TASKS.index('amplitude')] = cfg['protected_amplitude_margin']
    result = {}
    for arm, direction in directions.items():
        b = old.StateTextureB(color_channels=2, reference_channels=192)
        b.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=False)['b'])
        assert old.weights_digest(b) == shards[0]['b_hash']
        parameters = list(b.parameters())
        start = old.legacy_loss.flatten(parameters)
        optimizer = old.endpoint.optimizer(b)
        old.balanced_loss.assign(parameters, direction)
        torch.nn.utils.clip_grad_norm_(parameters, 1.)
        optimizer.step()
        raw = start - old.legacy_loss.flatten(parameters)
        result[arm] = dict(raw=raw, start=start, raw_norm=float(raw.norm()),
            gradient_norm=float(direction.norm()), raw_sha256=tensor_sha(raw), direction_sha256=tensor_sha(direction))
    return result, constraints, margins, cfg


def apply_original_step(start, raw, delta, cfg):
    norm_scale = min(1., float(raw.norm() / delta.norm().clamp_min(1e-18)))
    delta = delta * norm_scale
    delta = delta * cfg['scale']
    cap_scale = min(1., cfg['displacement_cap'] / float(delta.norm().clamp_min(1e-18)))
    delta = delta * cap_scale
    actual = start - (start - delta)
    return dict(norm_scale=norm_scale, fixed_scale=cfg['scale'], cap_scale=cap_scale,
        actual_norm=float(actual.norm()), parameter_updates=int(bool(actual.abs().max() > 0)),
        actual_sha256=tensor_sha(actual))


def budget_diagnostic(raws, constraints, margins, cfg, baseline, solver=project_displacement):
    records, selected = [], None
    for budget in BUDGETS:
        arms = {}
        for arm in ARMS:
            raw = raws[arm]
            started = time.monotonic()
            delta, projection = solver(raw['raw'], constraints, margins,
                max_sweeps=budget, tolerance=TOLERANCE)
            assert torch.isfinite(delta).all()
            stepped = apply_original_step(raw['start'], raw['raw'], delta, cfg)
            row = dict(projection=projection, **stepped, raw_norm=raw['raw_norm'],
                gradient_norm=raw['gradient_norm'], raw_sha256=raw['raw_sha256'],
                direction_sha256=raw['direction_sha256'], returned_delta_sha256=tensor_sha(delta),
                seconds=time.monotonic()-started)
            if budget == 4096:
                expected = baseline['arms'][arm]
                fields = ('raw_norm', 'gradient_norm', 'projection', 'actual_norm', 'parameter_updates')
                comparison = reproduction_comparison({k: row[k] for k in fields}, {k: expected[k] for k in fields})
                row['historical_comparison'] = comparison
                if not comparison['passed']:
                    raise ReproductionMismatch(arm, row)
                assert row['actual_norm'] == expected['actual_norm'] == 0
                assert row['parameter_updates'] == expected['parameter_updates'] == 0
                assert not projection['feasible']
            arms[arm] = row
        records.append(dict(budget=budget, arms=arms, all_feasible=all(r['projection']['feasible'] for r in arms.values())))
        print('SOLVER_BUDGET', budget, {k: v['projection']['actual_slack_min'] for k, v in arms.items()}, flush=True)
        if budget != 4096 and records[-1]['all_feasible']:
            selected = budget
            break
    return dict(records=records, selected_budget=selected, all_feasible=selected is not None)


def diagnose(args):
    started = time.monotonic()
    verify_inputs(args)
    path = args.out / 'solver_diagnostic.json'
    assert not path.exists(), '禁止覆盖诊断'
    try:
        raws, constraints, margins, cfg = raw_inputs(args)
        result = budget_diagnostic(raws, constraints, margins, cfg, read(args.out / 'source_failure/proposals.json'))
        result.update(overlay_sha256=sha(args.out / 'solver_overlay.json'),
            constraints_sha256=tensor_sha(constraints), margins_sha256=tensor_sha(margins),
            raw_sha256={k: v['raw_sha256'] for k, v in raws.items()},
            gradient_cache_reused=True, source_gradient_job='125421', new_reverse_passes=0,
            virtual_optimizer_proposals=2, saved_candidate_parameter_updates=0, new_candidates=0,
            holdout_used_for_selection=False, seconds=time.monotonic()-started,
            interpretation='feasibility at unchanged tolerance, not an optimality certificate; no new model weights saved')
        verify_inputs(args)
        write(path, result)
        if not result['all_feasible']:
            raise SystemExit(2)
    except Exception as error:
        write(args.out / 'solver_diagnostic.failed.json', dict(error_type=type(error).__name__, error=str(error),
            details=getattr(error, 'details', None), new_candidates=0, adopted=False,
            new_reverse_passes=0, seconds=time.monotonic()-started))
        raise


@contextmanager
def override_budget(module, budget, wrapper):
    """只替换调用方导入别名；退出时恢复，不改任何磁盘源文件。"""
    original = module.project_displacement
    assert budget in BUDGETS[1:]
    module.project_displacement = wrapper(original, budget)
    try:
        yield
    finally:
        module.project_displacement = original


def propose(args):
    from tools import e2_state_b_pair_finetune as old
    started = time.monotonic()
    overlay = verify_inputs(args)
    diagnostic = read(args.out / 'solver_diagnostic.json')
    budget = diagnostic['selected_budget']
    assert diagnostic['all_feasible'] and budget in BUDGETS[1:]
    assert diagnostic['overlay_sha256'] == sha(args.out / 'solver_overlay.json')
    assert not (args.out / 'training').exists(), '禁止覆盖候选'
    expected = next(row['arms'] for row in diagnostic['records'] if row['budget'] == budget)
    calls = []
    def factory(original, selected):
        def wrapped(raw, gradients, margins, max_sweeps=4096, tolerance=1e-6):
            arm = ARMS[len(calls)]
            assert max_sweeps == 4096 and tolerance == TOLERANCE
            assert tensor_sha(raw) == diagnostic['raw_sha256'][arm]
            assert tensor_sha(gradients) == diagnostic['constraints_sha256']
            assert tensor_sha(margins) == diagnostic['margins_sha256']
            delta, report = original(raw, gradients, margins, max_sweeps=selected, tolerance=tolerance)
            assert report['feasible'] and same_numbers(report, expected[arm]['projection'])
            assert tensor_sha(delta) == expected[arm]['returned_delta_sha256']
            report.update(configured_maximum_iterations=max_sweeps, effective_maximum_iterations=selected,
                solver_overlay_sha256=sha(args.out / 'solver_overlay.json'),
                diagnostic_sha256=sha(args.out / 'solver_diagnostic.json'), raw_sha256=tensor_sha(raw))
            calls.append(arm)
            return delta, report
        return wrapped
    try:
        with override_budget(old, budget, factory):
            old.propose(args)
    except Exception as error:
        write(args.out / 'solver_proposal.failed.json', dict(error_type=type(error).__name__,
            error=str(error), completed_solver_calls=calls, adopted=False, new_reverse_passes=0))
        raise
    assert calls == list(ARMS)
    proposals = read(args.out / 'training/proposals.json')
    proposals.update(effective_projection_sweeps=budget, gradient_cache_reused=True,
        source_gradient_job='125421', new_reverse_passes=0,
        source_gradient_seconds=proposals['gradient_seconds'], gradient_seconds=0.,
        historical_reverse_counts_are_source_work=True,
        new_cpu_proposal_seconds=time.monotonic()-started,
        solver_overlay_sha256=sha(args.out / 'solver_overlay.json'),
        solver_diagnostic_sha256=sha(args.out / 'solver_diagnostic.json'))
    for arm in ARMS:
        report = proposals['arms'][arm]
        assert report['parameter_updates'] == 1 and report['projection']['feasible'] and not report['adopted']
        assert report['actual_norm'] == expected[arm]['actual_norm']
    write(args.out / 'training/proposals.json', proposals)
    write(args.out / 'solver_proposal_receipt.json', dict(
        overlay_sha256=sha(args.out / 'solver_overlay.json'), diagnostic_sha256=sha(args.out / 'solver_diagnostic.json'),
        proposals_sha256=sha(args.out / 'training/proposals.json'), selected_budget=budget,
        candidate_sha256={arm: proposals['arms'][arm]['checkpoint_sha256'] for arm in ARMS},
        input_gradient_sha256=overlay['source_gradient_sha256'], gradient_cache_reused=True,
        source_gradient_job='125421', new_reverse_passes=0, new_gradient_seconds=0.,
        saved_candidate_parameter_updates=2, adopted=False, seconds=time.monotonic()-started))
    verify(args)
    print('SOLVER_CANDIDATES_COMPLETE', budget, flush=True)


def verify(args):
    """只读硬门：供独立replay脚本在评测前后调用；不读取holdout指标。"""
    from tools.e2_oracle_texture import weights_digest
    from tools.e2_b_endpoint_losses import flatten
    from models.e2_state_texture_b import StateTextureB
    overlay = verify_inputs(args)
    diagnostic = read(args.out / 'solver_diagnostic.json')
    receipt = read(args.out / 'solver_proposal_receipt.json')
    proposals = read(args.out / 'training/proposals.json')
    assert receipt['overlay_sha256'] == diagnostic['overlay_sha256'] == sha(args.out / 'solver_overlay.json')
    assert receipt['diagnostic_sha256'] == sha(args.out / 'solver_diagnostic.json')
    assert receipt['proposals_sha256'] == sha(args.out / 'training/proposals.json')
    selected = diagnostic['selected_budget']
    assert diagnostic['all_feasible'] and selected == receipt['selected_budget'] == proposals['effective_projection_sweeps']
    assert selected in BUDGETS[1:]
    assert [r['budget'] for r in diagnostic['records']] == list(BUDGETS[:BUDGETS.index(selected)+1])
    assert all(not r['all_feasible'] for r in diagnostic['records'][:-1]) and diagnostic['records'][-1]['all_feasible']
    assert receipt['input_gradient_sha256'] == overlay['source_gradient_sha256']
    b = StateTextureB(color_channels=2, reference_channels=192)
    b.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=False)['b'])
    start, initial = flatten(list(b.parameters())), weights_digest(b)
    for arm in ARMS:
        report = proposals['arms'][arm]
        assert report['parameter_updates'] == 1 and report['actual_norm'] > 0 and not report['adopted']
        assert report['checkpoint_start'] == CHECKPOINT_SHA and report['protocol_sha256'] == PROTOCOL_SHA
        assert report['projection']['feasible'] and report['projection']['maximum_iterations'] == selected
        assert report['projection']['tolerance'] == TOLERANCE
        assert report['projection']['actual_slack_min'] >= -TOLERANCE and report['projection']['actual_violations'] == 0
        assert report['minimum_cosine'] is not None and report['minimum_cosine'] >= -1e-5
        assert report['initial_b_hash'] == initial
        path = args.out / 'training' / (arm + '.pt')
        assert sha(path) == report['checkpoint_sha256'] == receipt['candidate_sha256'][arm]
        payload = torch.load(path, map_location='cpu', weights_only=False)
        assert payload['report'] == {k: v for k, v in report.items() if k != 'checkpoint_sha256'}
        b.load_state_dict(payload['b'])
        assert weights_digest(b) == report['final_b_hash'] != report['initial_b_hash']
        actual = start - flatten(list(b.parameters()))
        assert float(actual.norm()) == report['actual_norm']
        assert tensor_sha(actual) == diagnostic['records'][-1]['arms'][arm]['actual_sha256']
        evaluation = args.out / 'evaluation' / arm / 'replay/result.json'
        if evaluation.exists():
            result = read(evaluation)
            assert result['checkpoint_sha256'] == report['checkpoint_sha256'] and result['split'] == 'replay'
            assert result['protocol_sha256'] == PROTOCOL_SHA and not result['heldout_used_for_evaluation']
            assert result['source_sha256'] == sha(SOURCE / 'tools/e2_state_b_pair_evaluation.py')
    decision = args.out / 'replay_decision.json'
    if decision.exists():
        report = read(decision)
        assert report['holdout_results_read'] == 0 and report['adopted_checkpoints'] == []
        assert all('holdout' not in Path(name).parts and sha(Path(name)) == value
                   for name, value in report['input_sha256'].items())
        assert report['proposals'] == proposals
    print('SOLVER_VERIFY_PASSED', selected, 'new_reverse_passes=0 adopted=False', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'diagnose', 'propose', 'verify'))
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--project', type=Path, default=Path('/share/home/u2515283058/Mymodel'))
    args = parser.parse_args()
    args.prior = args.prior.resolve()
    args.source_run = args.prior / 'autonomous_repair_20261116/scaled_retry'
    args.out = (args.out or args.prior / 'autonomous_repair_20261116/scaled_projection_retry').resolve()
    args.checkpoint = args.project / 'output_eval/e2_state_b_pastel_followup_20261115/trust_retry/training/step1.pt'
    torch.set_num_threads(2)
    {'prepare': prepare, 'diagnose': diagnose, 'propose': propose, 'verify': verify}[args.action](args)


if __name__ == '__main__':
    main()
