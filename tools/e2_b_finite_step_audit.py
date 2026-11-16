"""只读比较新scaled G的有限步一阶预测与完整replay；不生成或更新模型。"""
import argparse
from collections import Counter
import math
from pathlib import Path
import statistics
import time

import torch

from tools.e2_b_endpoint_losses import TASKS
from tools.e2_state_b_backward_repro import tensor_sha
from tools.e2_state_b_amplitude_response import CHECKPOINT_SHA, write, sha


SOURCE = Path(__file__).resolve().parents[1]
RELATIVE_FLOOR = 1e-12
SOURCES = ('tools/e2_b_finite_step_audit.py', 'scripts/e2_b_finite_step_audit.sbatch',
    'tools/e2_b_scaled_projection_retry.py', 'tools/e2_b_scaled_replay_review.py',
    'tools/e2_b_endpoint_losses.py', 'tools/e2_state_b_pair_evaluation.py',
    'tools/e2_state_b_pair_summary.py', 'tools/e2_state_b_backward_repro.py',
    'tools/e2_state_b_amplitude_response.py', 'tools/e2_oracle_texture.py', 'models/e2_state_texture_b.py')


def actual_direction(start, target):
    """先提升端点再相减；保留已写回FP32端点的真实差，不使用名义提案。"""
    assert start.shape == target.shape and start.dtype == target.dtype
    return target.detach().double() - start.detach().double()


def sign(value):
    return 'increase' if value > 0 else 'decrease' if value < 0 else 'zero'


def within_tolerance(value, tolerance):
    return 'increase' if value > tolerance else 'decrease' if value < -tolerance else 'within_tolerance'


def compare_step(gradient, direction, baseline, observed):
    predicted = float(gradient.detach().double().reshape(-1) @ direction.detach().double().reshape(-1))
    delta = float(observed) - float(baseline)
    assert all(math.isfinite(value) for value in (predicted, delta, baseline, observed))
    tolerance = 1e-5 + 1e-4 * abs(baseline)
    mismatch = delta - predicted
    denominator = max(abs(delta), abs(predicted), RELATIVE_FLOOR)
    return dict(baseline_task=float(baseline), observed_task=float(observed), predicted_delta=predicted,
        observed_delta=delta, tolerance=tolerance, signed_mismatch=mismatch,
        absolute_mismatch=abs(mismatch), relative_mismatch=abs(mismatch) / denominator,
        relative_denominator=denominator, relative_floor_used=max(abs(delta), abs(predicted)) < RELATIVE_FLOOR,
        absolute_mismatch_over_tolerance=abs(mismatch) / tolerance,
        predicted_sign=sign(predicted), observed_sign=sign(delta), sign_agrees=sign(predicted) == sign(delta),
        predicted_tolerance_class=within_tolerance(predicted, tolerance),
        observed_tolerance_class=within_tolerance(delta, tolerance),
        predicted_task_failure=predicted > tolerance, observed_task_failure=delta > tolerance)


def summarize(rows):
    assert rows
    stats = {}
    for key in ('predicted_delta', 'observed_delta', 'signed_mismatch', 'absolute_mismatch',
                'relative_mismatch', 'absolute_mismatch_over_tolerance'):
        values = [row[key] for row in rows]
        stats[key] = dict(mean=statistics.mean(values), median=statistics.median(values),
                          minimum=min(values), maximum=max(values))
    counts = lambda key: dict(Counter(row[key] for row in rows))
    return dict(count=len(rows), predicted_sign_counts=counts('predicted_sign'),
        observed_sign_counts=counts('observed_sign'), sign_agreements=sum(r['sign_agrees'] for r in rows),
        predicted_tolerance_counts=counts('predicted_tolerance_class'),
        observed_tolerance_counts=counts('observed_tolerance_class'),
        predicted_task_failures=sum(r['predicted_task_failure'] for r in rows),
        observed_task_failures=sum(r['observed_task_failure'] for r in rows),
        predicted_decrease_observed_increase=sum(r['predicted_delta'] < 0 < r['observed_delta'] for r in rows),
        relative_floor_count=sum(r['relative_floor_used'] for r in rows), statistics=stats)


def grouped(rows):
    return dict(all=summarize(rows), without_stop_gradient_pattern=summarize([r for r in rows if r['task'] != 'pattern']),
        by_task={task: summarize([r for r in rows if r['task'] == task]) for task in TASKS},
        by_group={group: dict(all=summarize([r for r in rows if r['group'] == group]),
            by_task={task: summarize([r for r in rows if r['group'] == group and r['task'] == task]) for task in TASKS})
            for group in sorted({r['group'] for r in rows})})


def run(args):
    from tools import e2_b_scaled_projection_retry as retry
    from tools.e2_b_scaled_replay_review import Inputs, load_inputs
    from tools.e2_b_endpoint_losses import flatten
    from tools.e2_oracle_texture import weights_digest
    from models.e2_state_texture_b import StateTextureB

    started = time.monotonic()
    destination = args.out / 'finite_step_audit.json'
    assert not destination.exists(), '不覆盖已有有限步审计'
    retry.verify(args)
    source_hashes = {name: sha(SOURCE / name) for name in SOURCES}
    audit = Inputs()
    spec, qualification, proposals, reports = load_inputs(args.out, args.prior, audit)
    for name in ('solver_overlay.json', 'solver_diagnostic.json', 'solver_proposal_receipt.json', 'scaled_overlay.json'):
        audit.sha(args.out / name)
    assert audit.sha(args.checkpoint) == CHECKPOINT_SHA
    model = StateTextureB(color_channels=2, reference_channels=192)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=False)['b'])
    start, initial = flatten(list(model.parameters())), weights_digest(model)
    directions, displacement = {}, {}
    for arm in ('C0', 'C1'):
        report = proposals['arms'][arm]
        path = args.out / 'training' / (arm + '.pt')
        assert audit.sha(path) == report['checkpoint_sha256']
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['b'])
        target = flatten(list(model.parameters()))
        assert report['initial_b_hash'] == initial and weights_digest(model) == report['final_b_hash']
        fp32_norm = float((start - target).norm())
        assert fp32_norm == report['actual_norm']
        direction = actual_direction(start, target)
        assert torch.isfinite(direction).all() and direction.norm() > 0
        directions[arm] = direction
        displacement[arm] = dict(definition='theta_candidate.double()-theta_f68.double()',
            vector_sha256=tensor_sha(direction), norm_fp64=float(direction.norm()),
            legacy_fp32_subtraction_norm=fp32_norm, changed_parameters=int((target != start).sum()),
            checkpoint_sha256=report['checkpoint_sha256'], candidate_parameter_updates=report['parameter_updates'])
    gradients, baselines, items = {}, {}, {}
    for index in range(4):
        path = args.out / 'gradients' / ('shard%d.pt' % index)
        assert audit.sha(path) == qualification['new_gradient_sha256'][path.name]
        shard = torch.load(path, map_location='cpu', weights_only=False)
        assert shard['loss_scale'] == 1024 and shard['all_finite'] and shard['numerical_qualification']
        assert shard['b_hash'] == initial and shard['protocol_sha256'] == audit.sha(args.out / 'protocol.json')
        assert shard['overlay_sha256'] == audit.sha(args.out / 'scaled_overlay.json')
        values = {(r['case'], r['sample_condition']): r for r in shard['values']}
        for i, item in enumerate(shard['items']):
            key = item['case'], item['sample_condition']
            assert key not in gradients
            gradients[key], baselines[key], items[key] = shard['gradients'][i], values[key], item
    assert len(gradients) == 48
    rows_by_arm = {}
    for arm in ('C0', 'C1'):
        replay = reports[arm]['replay']
        rows = []
        for observed in replay['rows']:
            key = observed['case'], observed['sample_condition']
            baseline, item = baselines[key], items[key]
            assert observed['noise_digest'] == baseline['noise_digest']
            for index, task in enumerate(TASKS):
                comparison = compare_step(gradients[key][index], directions[arm],
                    baseline['tasks'][task], observed['training_tasks'][task])
                assert comparison['observed_delta'] == observed['training_task_comparison']['delta'][task]
                assert comparison['tolerance'] == observed['training_task_comparison']['tolerance'][task]
                assert comparison['observed_task_failure'] == (task in observed['training_task_comparison']['failures'])
                rows.append(dict(candidate=arm, case=item['case'], condition=item['condition'],
                    reference_arm=item['arm'], sample_condition=item['sample_condition'], group=item['group'],
                    task=task, stop_gradient_semantics=task == 'pattern', **comparison))
        assert len(rows) == 336
        rows_by_arm[arm] = rows
    retry.verify(args)
    assert all(sha(Path(name)) == expected for name, expected in audit.hashes.items())
    assert all(sha(SOURCE / name) == expected for name, expected in source_hashes.items())
    result = dict(analysis='first-order finite-step endpoint prediction; not central finite difference',
        protocol_sha256=sha(args.out / 'protocol.json'), source_gradient_job='125421',
        gradient_cache_reused=True, source_gradient_scale=1024, input_gradients_already_unscaled=True,
        displacement=displacement, counts=dict(candidates=2, conditions_per_candidate=48,
            tasks_per_candidate=336, total_comparisons=672),
        summaries={arm: grouped(rows) for arm, rows in rows_by_arm.items()}, rows=rows_by_arm,
        task_failures={arm: [r for r in rows if r['observed_task_failure']] for arm, rows in rows_by_arm.items()},
        definitions=dict(prediction='g_f68^T(theta_candidate.double()-theta_f68.double())',
            observation='replay_task_candidate - new_scaled_shard_f68_task',
            original_task_tolerance='1e-5 + 1e-4*abs(new_scaled_shard_f68_task)',
            signed_mismatch='observed_delta - predicted_delta',
            relative_mismatch='abs(observed_delta-predicted_delta)/max(abs(observed_delta),abs(predicted_delta),1e-12)',
            relative_summary='mean/median of per-row symmetric ratios; floor usage counted; not ratio of aggregated sums',
            pooled_summary='descriptive unweighted pooling of different task units; per-task statistics are primary',
            sign_counts='strict numeric sign; tolerance classes reported separately'),
        pattern_semantics='pattern contains detached spectrum focus and no-grad base; raw endpoint change is not the frozen-pattern derivative; no semantic correction performed',
        interpretation='agreement does not certify a true gradient; mismatch may include finite-step curvature, mixed-precision execution/rounding, stop-gradient semantics and gradient execution error; this audit does not separate them',
        frozen_pattern_recomputed=False, scaled_gradients_are_ground_truth=False,
        new_forward_passes=0, new_backward_passes=0, optimizer_updates=0, projection_calls=0,
        new_candidates=0, adopted_checkpoints=[], holdout_results_read=0,
        source_sha256=source_hashes, input_sha256=audit.hashes, seconds=time.monotonic()-started)
    write(destination, result)
    print('FINITE_STEP_AUDIT_COMPLETE', {arm: len(rows) for arm, rows in result['task_failures'].items()},
          sha(destination), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--project', type=Path, default=Path('/share/home/u2515283058/Mymodel'))
    args = parser.parse_args()
    args.out, args.prior = args.out.resolve(), args.prior.resolve()
    args.checkpoint = args.project / 'output_eval/e2_state_b_pastel_followup_20261115/trust_retry/training/step1.pt'
    torch.set_num_threads(2)
    run(args)


if __name__ == '__main__':
    main()
