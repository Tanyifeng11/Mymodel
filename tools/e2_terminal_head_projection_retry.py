"""复用2308维合格梯度，只延长原投影预算；不重新反传或改变原S0。"""
import argparse
import copy
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import time

import cv2
import torch

from tools import e2_terminal_head_one_step as old
from tools.e2_b_balanced_losses import assign, alignment
from tools.e2_b_endpoint_losses import flatten, set_parameters
from tools.e2_b_pair_finetune_losses import project_displacement
from tools.e2_b_scaled_projection_retry import reproduction_comparison
from tools.e2_state_b_backward_repro import tensor_sha, vector_summary
from tools.e2_state_b_amplitude_response import read, write, sha, CHECKPOINT_SHA
from tools.e2_state_b_pair_retry import PROTOCOL_SHA
from tools.e2_oracle_texture import weights_digest


SOURCE = Path(__file__).resolve().parents[1]
ARMS, BUDGETS = ('C0', 'C1'), (8192, 16384, 32768)
PARAMETERS, TOLERANCE = 2308, 1e-6
EXTRA = ('tools/e2_terminal_head_projection_retry.py', 'tools/e2_terminal_head_projection_summary.py',
         'scripts/e2_terminal_head_projection_retry.sbatch', 'scripts/e2_terminal_head_projection_replay.sbatch')


def make_args(out, prior, project=None):
    prior = Path(prior).resolve()
    project = Path(project or '/share/home/u2515283058/Mymodel').resolve()
    return argparse.Namespace(out=Path(out).resolve(), prior=prior, project=project,
        source_run=prior/'autonomous_repair_20261116/terminal_head_one_step',
        preflight=prior/'autonomous_repair_20261116/terminal_head_preflight',
        fixed=prior/'autonomous_repair_20261116/scaled_projection_retry',
        checkpoint=project/'output_eval/e2_state_b_pastel_followup_20261115/trust_retry/training/step1.pt',
        data=project.parent/'datasets/BF', device='cuda:0', arm='C0')


def source_args(args):
    result = copy.copy(args)
    result.out = args.source_run
    return result


def prepare(args):
    assert not args.out.exists(), '不覆盖已有重试目录'
    source = source_args(args)
    original = old.verify(source)
    _, _, _, qualification = old.gradient_inputs(source, original)
    assert qualification == read(args.source_run/'gradient_qualification.json') and qualification['passed']
    failed = read(args.source_run/'training/proposals.json')
    assert failed['stop_before_replay'] and failed['new_reverse_passes'] == 352
    for arm in ARMS:
        report = failed['arms'][arm]
        assert report['parameter_updates'] == 0 and report['actual_norm'] == 0
        assert report['effective_projection_sweeps'] == 8192 and not report['projection']['feasible']
        assert report['projection']['maximum_iterations'] == 8192
        head = old.load_candidate(source, arm, 'cpu')
        assert not torch.count_nonzero(flatten(head.increment_parameters()))
        assert not torch.count_nonzero(old.effective_difference(head))
    inputs, copied = {}, {}
    names = dict(original['copied_sha256'])
    for name in ('one_step_overlay.json', 'gradient_qualification.json'):
        names[name] = sha(args.source_run/name)
    for index in range(4):
        names['gradients/shard%d.pt' % index] = qualification['new_gradient_sha256']['shard%d.pt' % index]
        names['gradients/shard%d.json' % index] = qualification['gradient_json_sha256']['shard%d.json' % index]
    def duplicate(source_path, name, digest):
        assert sha(source_path) == digest
        target = args.out/name
        target.parent.mkdir(parents=True, exist_ok=True)
        assert not target.exists()
        shutil.copyfile(source_path, target)
        assert sha(target) == digest
        inputs[Path(source_path).resolve().as_posix()] = copied[name] = digest
    for name, digest in names.items():
        duplicate(args.source_run/name, name, digest)
    duplicate(args.source_run/'training/proposals.json', 'source_failure/proposals.json', sha(args.source_run/'training/proposals.json'))
    for arm in ARMS:
        duplicate(args.source_run/'training'/(arm+'.pt'), 'source_failure/'+arm+'.pt', failed['arms'][arm]['checkpoint_sha256'])
    assert sha(args.checkpoint) == CHECKPOINT_SHA
    inputs[args.checkpoint.resolve().as_posix()] = CHECKPOINT_SHA
    sources = dict(original['source_sha256'])
    sources.update({name: sha(SOURCE/name) for name in EXTRA})
    overlay = dict(source_run=str(args.source_run), source_gradient_job='125463', source_reverse_passes=352,
        source_gradient_seconds=failed['gradient_seconds'], new_reverse_passes=0, new_gradient_seconds=0.,
        gradient_cache_reused=True, parameter_count=PARAMETERS, original_protocol_sha256=PROTOCOL_SHA,
        original_configured_budget=4096, source_failed_budget=8192, budgets=list(BUDGETS), tolerance=TOLERANCE,
        selection='reproduce8192; then16384; only if either fails32768; first shared feasible budget; no further retry',
        only_change='maximum Hildreth sweeps; original losses/G/margins/tolerance/AdamW/clip/scale/cap/S0 unchanged',
        raw_policy='same two increment tensors; each independent AdamW reconstructed once; exact source raw SHA required',
        source_gradient_sha256=qualification['new_gradient_sha256'], source_qualification_sha256=sha(args.source_run/'gradient_qualification.json'),
        source_overlay_sha256=sha(args.source_run/'one_step_overlay.json'), source_failure_sha256=sha(args.source_run/'training/proposals.json'),
        input_sha256=inputs, copied_sha256=copied, source_sha256=sources,
        execution_commit=subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip(),
        adopted_checkpoints=[], holdout_results_read=0)
    write(args.out/'solver_overlay.json', overlay)
    verify_inputs(args)


def verify_inputs(args):
    overlay = read(args.out/'solver_overlay.json')
    assert overlay['parameter_count'] == PARAMETERS and overlay['budgets'] == list(BUDGETS)
    assert overlay['new_reverse_passes'] == 0 and overlay['source_reverse_passes'] == 352
    assert overlay['tolerance'] == TOLERANCE and overlay['source_failed_budget'] == 8192
    assert sha(args.out/'protocol.json') == PROTOCOL_SHA and sha(args.checkpoint) == CHECKPOINT_SHA
    for path, digest in overlay['input_sha256'].items():
        assert sha(Path(path)) == digest, path
    for path, digest in overlay['copied_sha256'].items():
        assert sha(args.out/path) == digest, path
    for path, digest in overlay['source_sha256'].items():
        assert sha(SOURCE/path) == digest, path
    original = old.verify(args)
    assert sha(args.out/'one_step_overlay.json') == overlay['source_overlay_sha256']
    assert original['effective_projection_sweeps'] == 8192  # 此文件是原梯度来源，不能改写为本轮预算。
    assert sha(args.out/'gradient_qualification.json') == overlay['source_qualification_sha256']
    return overlay


def load_state(args):
    overlay = verify_inputs(args)
    shards, _, _, qualification = old.gradient_inputs(args, read(args.out/'one_step_overlay.json'), tensors=True)
    assert qualification == read(args.out/'gradient_qualification.json') and qualification['passed']
    spec = read(args.out/'protocol.json')
    directions, constraints, margins, norms = old.combine_gradients(shards, spec)
    assert constraints.shape == (336, PARAMETERS)
    source_proposals = read(args.out/'source_failure/proposals.json')
    return dict(overlay=overlay, spec=spec, qualification=qualification, source_proposals=source_proposals,
        shards=shards, directions=directions, constraints=constraints, margins=margins, component_norms=norms)


def reconstruct_raw(parameters, direction, cfg):
    """与原one_update的两个形状参数和归约顺序相同；不可替换成一个flat参数。"""
    start = flatten(parameters)
    assert not torch.count_nonzero(start)
    optimizer = torch.optim.AdamW(parameters, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    assign(parameters, direction)
    clipped = torch.nn.utils.clip_grad_norm_(parameters, cfg['gradient_clip'])
    assert torch.isfinite(clipped)
    optimizer.step()
    raw = start-flatten(parameters)
    assert torch.isfinite(raw).all()
    return dict(raw=raw, raw_sha256=tensor_sha(raw), raw_norm=float(raw.norm()),
        gradient_norm=float(direction.norm()), gradient_clip_return=float(clipped), direction_sha256=tensor_sha(direction))


def original_step(raw, projected, constraints, cfg):
    norm_scale = min(1., float(raw.norm()/projected.norm().clamp_min(1e-18)))
    delta = projected*norm_scale*cfg['scale']
    cap_scale = min(1., cfg['displacement_cap']/float(delta.norm().clamp_min(1e-18)))
    delta = delta*cap_scale
    start = torch.zeros_like(raw)
    candidate = start-delta
    actual = start-candidate  # 不用一元负号替代，保留零元素的signed-zero字节。
    changed = bool(torch.count_nonzero(actual))
    report = dict(norm_scale=norm_scale, fixed_scale=cfg['scale'], cap_scale=cap_scale,
        actual_norm=float(actual.norm()), parameter_updates=int(changed), head_sha256=tensor_sha(candidate),
        actual_descent_sha256=tensor_sha(actual), minimum_cosine=min(alignment(constraints, actual)) if changed else None)
    return candidate, report


def budget_diagnostic(raws, constraints, margins, cfg, historical, solver=project_displacement):
    records, selected, vectors = [], None, None
    for budget in BUDGETS:
        arms, candidates = {}, {}
        for arm in ARMS:
            raw = raws[arm]
            started = time.monotonic()
            projected, report = solver(raw['raw'], constraints, margins, max_sweeps=budget, tolerance=TOLERANCE)
            assert torch.isfinite(projected).all()
            candidate, step = original_step(raw['raw'], projected, constraints, cfg)
            row = dict(projection=report, **step, raw_norm=raw['raw_norm'], raw_sha256=raw['raw_sha256'],
                gradient_norm=raw['gradient_norm'], gradient_clip_return=raw['gradient_clip_return'],
                projected_sha256=tensor_sha(projected), seconds=time.monotonic()-started)
            if budget == 8192:
                fields = ('projection', 'raw_norm', 'gradient_norm', 'gradient_clip_return', 'actual_norm', 'parameter_updates')
                check = reproduction_comparison({k: row[k] for k in fields}, {k: historical[arm][k] for k in fields})
                row['historical_comparison'] = check
                row['historical_class_equal'] = (not report['feasible'] and step['parameter_updates'] == 0 and step['actual_norm'] == 0)
                assert raw['raw_sha256'] == historical[arm]['raw_sha256']
            arms[arm], candidates[arm] = row, candidate
        common = all(row['projection']['feasible'] and row['parameter_updates'] == 1 for row in arms.values())
        records.append(dict(budget=budget, arms=arms, all_feasible=common))
        print('TERMINAL_SOLVER_BUDGET', budget, {arm: row['projection']['actual_slack_min'] for arm, row in arms.items()}, flush=True)
        if budget == 8192 and not all(row['historical_comparison']['passed'] and row['historical_class_equal'] for row in arms.values()):
            return dict(status='historical_reproduction_mismatch', records=records, selected_budget=None, all_feasible=False), None
        if budget != 8192 and common:
            selected, vectors = budget, candidates
            break
    return dict(status='shared_feasible' if selected else 'no_shared_feasible_budget', records=records,
        selected_budget=selected, all_feasible=selected is not None), vectors


def diagnose(args):
    assert not (args.out/'solver_diagnostic.json').exists() and not (args.out/'raw.pt').exists()
    started = time.monotonic()
    state = load_state(args)
    raws = {}
    for arm in ARMS:
        head = old.load_head(args, 'cpu')
        base_hash = weights_digest(head.base)
        raws[arm] = reconstruct_raw(head.increment_parameters(), state['directions'][arm], state['spec']['training'])
        assert weights_digest(head.base) == base_hash
    raw_path = args.out/'raw.pt'
    torch.save(dict(raws=raws, directions=state['directions'], constraints=state['constraints'], margins=state['margins'],
        parameter_count=PARAMETERS, source_gradient_sha256=state['overlay']['source_gradient_sha256']), raw_path)
    exact = {arm: raws[arm]['raw_sha256'] == state['source_proposals']['arms'][arm]['raw_sha256'] for arm in ARMS}
    vectors = None
    if all(exact.values()):
        result, vectors = budget_diagnostic(raws, state['constraints'], state['margins'], state['spec']['training'], state['source_proposals']['arms'])
    else:
        result = dict(status='raw_sha_mismatch', records=[], selected_budget=None, all_feasible=False)
    result.update(raw_sha_match=exact, raw_sha256={arm: raws[arm]['raw_sha256'] for arm in ARMS},
        source_raw_sha256={arm: state['source_proposals']['arms'][arm]['raw_sha256'] for arm in ARMS},
        raw_pt_sha256=sha(raw_path), constraints_sha256=tensor_sha(state['constraints']), margins_sha256=tensor_sha(state['margins']),
        solver_overlay_sha256=sha(args.out/'solver_overlay.json'), new_reverse_passes=0, source_reverse_passes=352,
        raw_reconstruction_optimizer_steps=2, candidate_saved_parameter_updates=0, gradient_cache_reused=True,
        source_gradient_job='125463', adopted_checkpoints=[], holdout_results_read=0, seconds=time.monotonic()-started,
        environment=dict(hostname=socket.gethostname(), node=os.environ.get('SLURM_JOB_NODELIST'), job=os.environ.get('SLURM_JOB_ID'),
            torch=str(torch.__version__), python=platform.python_version(), threads=torch.get_num_threads()),
        interpretation='fixed-budget feasibility check, not an optimality certificate or evidence of pixel improvement')
    if vectors is not None:
        torch.save(vectors, args.out/'solver_selected.pt')
        result['selected_vectors_sha256'] = sha(args.out/'solver_selected.pt')
    verify_inputs(args)
    write(args.out/'solver_diagnostic.json', result)
    if not result['all_feasible']:
        raise SystemExit(2)


def propose(args):
    started = time.monotonic()
    state = load_state(args)
    diagnostic = read(args.out/'solver_diagnostic.json')
    assert diagnostic['all_feasible'] and all(diagnostic['raw_sha_match'].values())
    budget = diagnostic['selected_budget']
    assert budget in BUDGETS[1:] and diagnostic['solver_overlay_sha256'] == sha(args.out/'solver_overlay.json')
    assert sha(args.out/'raw.pt') == diagnostic['raw_pt_sha256']
    assert sha(args.out/'solver_selected.pt') == diagnostic['selected_vectors_sha256']
    selected = torch.load(args.out/'solver_selected.pt', map_location='cpu', weights_only=False)
    destination = args.out/'training'
    assert not destination.exists()
    destination.mkdir()
    arms = {}
    for arm in ARMS:
        head = old.load_head(args, 'cpu')
        base_hash = weights_digest(head.base)
        initial_hash = tensor_sha(flatten(head.increment_parameters()))
        row = diagnostic['records'][-1]['arms'][arm]
        assert row['projection']['feasible'] and row['parameter_updates'] == 1
        set_parameters(head.increment_parameters(), selected[arm])
        actual = flatten(head.increment_parameters())
        effective = old.effective_difference(head)
        assert tensor_sha(actual) == row['head_sha256'] and weights_digest(head.base) == base_hash
        report = {k: v for k, v in row.items() if k not in ('seconds', 'projected_sha256')}
        report.update(arm=arm, optimizer_proposals=1, optimizer_steps_executed_in_propose=0,
            configured_projection_sweeps=4096, source_failed_projection_sweeps=8192, effective_projection_sweeps=budget,
            trainable_parameters=PARAMETERS, base_checkpoint_sha256=CHECKPOINT_SHA, base_b_hash=base_hash,
            initial_head_sha256=initial_hash, effective_weight_difference_sha256=tensor_sha(effective),
            effective_weight_difference=vector_summary(effective), frozen_unchanged=True, heldout_used_for_training=False,
            adopted=False, acceptance='unassessed candidate; requires original complete S0 replay',
            protocol_sha256=PROTOCOL_SHA, overlay_sha256=sha(args.out/'one_step_overlay.json'),
            gradient_qualification_sha256=sha(args.out/'gradient_qualification.json'),
            solver_overlay_sha256=sha(args.out/'solver_overlay.json'), solver_diagnostic_sha256=sha(args.out/'solver_diagnostic.json'))
        path = destination/(arm+'.pt')
        torch.save(dict(delta_weight=head.delta_weight.detach(), delta_bias=head.delta_bias.detach(), report=report,
            base_checkpoint_sha256=CHECKPOINT_SHA), path)
        arms[arm] = dict(report, checkpoint_sha256=sha(path))
        old.load_candidate(args, arm, 'cpu', arms[arm])
    proposals = dict(arms=arms, conditions=48, parameter_count=PARAMETERS, new_reverse_passes=0, source_reverse_passes=352,
        source_full_task_reverse_passes=336, source_late_y_reverse_passes=12, source_late_absolute_reverse_passes=4,
        gradient_cache_reused=True, source_gradient_job='125463', gradient_seconds=0.,
        source_gradient_seconds=state['overlay']['source_gradient_seconds'], raw_reconstruction_optimizer_steps=2,
        optimizer_steps_executed_in_propose=0, independent_adamw_proposals=2, sequential_training_updates=0,
        new_gradient_sha256=state['qualification']['new_gradient_sha256'], gradient_json_sha256=state['qualification']['gradient_json_sha256'],
        component_norms=state['component_norms'], constraint_sha256=tensor_sha(state['constraints']), margins=state['margins'].tolist(),
        effective_projection_sweeps=budget, stop_before_replay=False, frozen_unchanged=True,
        heldout_used_for_training=False, adopted_checkpoints=[], overlay_sha256=sha(args.out/'one_step_overlay.json'),
        solver_overlay_sha256=sha(args.out/'solver_overlay.json'), solver_diagnostic_sha256=sha(args.out/'solver_diagnostic.json'),
        proposal_seconds=time.monotonic()-started)
    write(destination/'proposals.json', proposals)
    receipt = dict(solver_overlay_sha256=sha(args.out/'solver_overlay.json'), diagnostic_sha256=sha(args.out/'solver_diagnostic.json'),
        raw_sha256=sha(args.out/'raw.pt'), selected_vectors_sha256=sha(args.out/'solver_selected.pt'),
        proposals_sha256=sha(destination/'proposals.json'), candidate_sha256={arm: arms[arm]['checkpoint_sha256'] for arm in ARMS},
        selected_budget=budget, source_gradient_sha256=state['overlay']['source_gradient_sha256'],
        new_reverse_passes=0, source_reverse_passes=352, raw_reconstruction_optimizer_steps=2,
        saved_candidate_parameter_updates=2, adopted_checkpoints=[])
    write(args.out/'solver_receipt.json', receipt)
    verify(args)


def verify(args):
    overlay = verify_inputs(args)
    diagnostic = read(args.out/'solver_diagnostic.json')
    receipt, proposals = read(args.out/'solver_receipt.json'), read(args.out/'training/proposals.json')
    budget = diagnostic['selected_budget']
    assert diagnostic['all_feasible'] and diagnostic['status'] == 'shared_feasible' and all(diagnostic['raw_sha_match'].values())
    assert budget in BUDGETS[1:] and budget == receipt['selected_budget'] == proposals['effective_projection_sweeps']
    assert [r['budget'] for r in diagnostic['records']] == list(BUDGETS[:BUDGETS.index(budget)+1])
    assert all(not r['all_feasible'] for r in diagnostic['records'][:-1]) and diagnostic['records'][-1]['all_feasible']
    assert all(r['historical_comparison']['passed'] and r['historical_class_equal'] for r in diagnostic['records'][0]['arms'].values())
    assert diagnostic['solver_overlay_sha256'] == receipt['solver_overlay_sha256'] == sha(args.out/'solver_overlay.json')
    assert receipt['diagnostic_sha256'] == sha(args.out/'solver_diagnostic.json')
    assert receipt['raw_sha256'] == diagnostic['raw_pt_sha256'] == sha(args.out/'raw.pt')
    assert receipt['selected_vectors_sha256'] == diagnostic['selected_vectors_sha256'] == sha(args.out/'solver_selected.pt')
    assert receipt['proposals_sha256'] == sha(args.out/'training/proposals.json')
    assert receipt['source_gradient_sha256'] == overlay['source_gradient_sha256']
    assert proposals['new_reverse_passes'] == receipt['new_reverse_passes'] == 0 and proposals['source_reverse_passes'] == 352
    for arm in ARMS:
        report = proposals['arms'][arm]
        expected = diagnostic['records'][-1]['arms'][arm]
        assert report['projection'] == expected['projection'] and report['projection']['maximum_iterations'] == budget
        assert report['projection']['feasible'] and report['projection']['actual_slack_min'] >= -TOLERANCE
        assert report['projection']['tolerance'] == TOLERANCE and report['projection']['actual_violations'] == 0
        assert report['parameter_updates'] == 1 and report['actual_norm'] > 0 and report['minimum_cosine'] >= -1e-5
        assert report['head_sha256'] == expected['head_sha256'] and report['raw_sha256'] == diagnostic['raw_sha256'][arm]
        assert report['checkpoint_sha256'] == receipt['candidate_sha256'][arm]
        assert report['solver_overlay_sha256'] == sha(args.out/'solver_overlay.json')
        assert report['solver_diagnostic_sha256'] == sha(args.out/'solver_diagnostic.json')
        head = old.load_candidate(args, arm, 'cpu', report)
        vector = flatten(head.increment_parameters())
        assert tensor_sha(torch.zeros_like(vector)-vector) == report['actual_descent_sha256']
    return receipt


def evaluate(args):
    verify(args)
    old.evaluate(args)
    verify(args)


def review(args):
    from tools.e2_terminal_head_projection_summary import review as summarize
    return summarize(args.out, args.prior)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'diagnose', 'propose', 'verify', 'evaluate', 'review'))
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--project', type=Path, default=Path('/share/home/u2515283058/Mymodel'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--arm', choices=ARMS, default='C0')
    options = parser.parse_args()
    out = options.out or options.prior/'autonomous_repair_20261116/terminal_head_projection_retry'
    args = make_args(out, options.prior, options.project)
    args.action, args.arm, args.device = options.action, options.arm, options.device
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    try:
        dict(prepare=prepare, diagnose=diagnose, propose=propose, verify=verify, evaluate=evaluate, review=review)[args.action](args)
    except Exception as error:
        if args.out.exists():
            path = args.out/('%s_%s.failure.json' % (args.action, args.arm))
            if not path.exists():
                write(path, dict(error_type=type(error).__name__, message=str(error), complete=False,
                    new_reverse_passes=0, adopted_checkpoints=[]))
        raise


if __name__ == '__main__':
    main()
