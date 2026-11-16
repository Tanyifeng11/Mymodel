"""核验replay资格边界、完整性和不读取holdout；不运行模型或训练。"""
import copy
from pathlib import Path

import pytest

from tools import e2_b_scaled_replay_review as review
from tools import e2_state_b_pair_summary as summary
from tools.e2_b_endpoint_losses import TASKS
from tools.e2_state_b_pair_evaluation import aggregate, strict_replay_protection, training_comparison


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.write(path, value)


def update(path, change):
    value = summary.read(path)
    change(value)
    write(path, value)


@pytest.fixture
def completed_replay(tmp_path, monkeypatch):
    """构造48项完整记录，PT/PNG仅作为SHA字节输入，不假装真实GPU结果。"""
    out, prior = tmp_path / 'scaled', tmp_path / 'prior'
    fixed = prior / 'projection_fp64_retry'
    groups = ('pastel', 'multi', 'neutral', 'synthetic')
    inputs = [dict(case='case%d' % case, condition=group, group=group, arm=arm,
        sample_condition=group + '_' + arm) for case in range(4) for group in groups for arm in summary.ARMS]
    spec = dict(checkpoint_sha256='f68-checkpoint', inputs=inputs)
    write(out / 'protocol.json', spec)
    write(fixed / 'protocol.json', spec)
    protocol_sha = summary.sha(out / 'protocol.json')
    new_hashes, old_hashes, pt_hashes = {}, {}, {}
    for shard in range(4):
        pt, js = 'shard%d.pt' % shard, 'shard%d.json' % shard
        tensor = out / 'gradients' / pt
        tensor.parent.mkdir(parents=True, exist_ok=True)
        tensor.write_bytes(('new-gradient-%d' % shard).encode())
        pt_hashes[pt] = summary.sha(tensor)
        values = [dict(item, tasks={task: 1. for task in TASKS}, noise_digest='noise-' + item['case'])
                  for item in inputs if item['case'] == 'case%d' % shard]
        write(out / 'gradients' / js, dict(tensor_sha256=pt_hashes[pt], numerical_qualification=True,
            all_finite=True, protocol_sha256=protocol_sha, checkpoint_sha256='f68-checkpoint',
            frozen_unchanged=True, training_updates=0, new_gradient_generation=True,
            cached_gradients_reused=False, values=values, b_hash='b-f68', net_hash='frozen-net'))
        write(fixed / 'gradients' / js, dict(historical=True, shard=shard))
        new_hashes[js] = summary.sha(out / 'gradients' / js)
        old_hashes[js] = summary.sha(fixed / 'gradients' / js)
    write(out / 'numerical_qualification.json', dict(passed=True, all_finite=True, loss_scale=1024,
        optimizer_updates=0, replicas=[dict(passed=True) for _ in range(8)],
        maximum_relative_l2=.002, minimum_cosine=.999999, new_gradient_sha256=pt_hashes))
    proposals = dict(arms={}, conditions=48, full_task_reverse_passes=336, frozen_unchanged=True,
                     heldout_used_for_training=False)
    for model in ('C0', 'C1'):
        checkpoint = out / 'training' / (model + '.pt')
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(model.encode())
        proposals['arms'][model] = dict(arm=model, protocol_sha256=protocol_sha, adopted=False,
            optimizer_proposals=1, parameter_updates=1, checkpoint_start='f68-checkpoint',
            initial_b_hash='b-f68', final_b_hash='b-' + model,
            checkpoint_sha256=summary.sha(checkpoint), projection=dict(feasible=True), minimum_cosine=.01)
    write(out / 'training/proposals.json', proposals)
    base_rows = None
    for index, model in enumerate(summary.MODELS):
        rows = []
        for item in inputs:
            amplitude, correlation, gain, slope = .6 + .2 * index, .5, .3 + .1 * index, .5 + .1 * index
            scores = {metric: .1 - .001 * index for metric in summary.METRICS}
            scores.update(l_amplitude=amplitude, l_amplitude_error=abs(amplitude - 1),
                l_correlation=correlation, l_target_gain=gain, l_target_gain_error=abs(gain - 1),
                amplitude_pass=.75 <= amplitude <= 1.25, l_quality_pass=.75 <= amplitude <= 1.25 and correlation >= .5)
            path = out / 'evaluation' / model / 'replay/images' / ('_'.join((item['case'], item['sample_condition'])) + '.png')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((model + item['case'] + item['sample_condition']).encode())
            x = dict(a075=.75, a100=1., a125=1.25)[item['arm']]
            tasks = {task: 1. - index * .001 for task in TASKS}
            rows.append(dict(item, model_arm=model, seed=123, noise_digest='noise-' + item['case'],
                b_calls=25, scores=scores, training_tasks=tasks,
                training_task_comparison=training_comparison(tasks, {task: 1. for task in TASKS}),
                image=path.relative_to(out).as_posix(), png_sha256=summary.sha(path),
                projection=dict(fixed_target_projection=x, fixed_output_projection=slope * (x - 1) + gain)))
        if model == 'f68':
            base_rows = {summary.row_key(row, 'replay'): row for row in rows}
        report = dict(arm=model, split='replay', rows=rows, **aggregate(rows, 'replay'),
            protocol_sha256=protocol_sha, frozen_unchanged=True, evaluation_optimizer_updates=0,
            heldout_used_for_training=False, heldout_used_for_evaluation=False,
            checkpoint_sha256='f68-checkpoint' if model == 'f68' else proposals['arms'][model]['checkpoint_sha256'],
            b_hash='b-' + model, net_hash='frozen-net', checkpoint_flags={},
            training_task_protection=dict(passed=True, failures=[], comparisons=336,
                baseline_gradient_json_sha256=old_hashes if model == 'f68' else new_hashes),
            strict_replay_protection=strict_replay_protection(rows, base_rows))
        write(out / 'evaluation' / model / 'replay/result.json', report)
        if model == 'f68':
            write(fixed / 'evaluation/f68/replay/result.json', report)
    # 任何误读均会JSON解析失败；正常汇总不访问这些文件。
    for root in (out, fixed):
        for model in summary.MODELS:
            poison = root / 'evaluation' / model / 'holdout/result.json'
            poison.parent.mkdir(parents=True, exist_ok=True)
            poison.write_text('HOLDOUT_MUST_NOT_BE_READ', encoding='utf-8')
    monkeypatch.setattr(review.subprocess, 'check_output', lambda *args, **kwargs: 'test-commit\n')
    return out, prior


def test_complete_replay_is_only_holdout_candidate(completed_replay):
    out, prior = completed_replay
    result = review.review(out, prior)
    assert all(value['holdout_candidate'] and not value['adopted'] for value in result['replay_decisions'].values())
    assert result['adopted_checkpoints'] == [] and result['old_holdout_not_read']
    assert result['holdout_results_read'] == 0 and not result['holdout_evaluated']
    assert all('holdout' not in Path(path).parts for path in result['input_sha256'])
    assert len(result['pastel_conditions']) == 4
    assert result['groups']['f68']['pastel']['a100']['amplitude_pass'] == 0
    assert result['groups']['C1']['pastel']['a100']['l_quality_pass'] == 4
    assert result['innovation']['paired_and_absolute_improvement']
    assert set(result['innovation']['versus']) == {'f68', 'C0'}
    assert len(result['focus_metrics']) == 9
    assert summary.read(out / 'replay_decision.json')['tool_git_commit'] == 'test-commit'


def test_original_residue_and_protection_boundaries(completed_replay):
    out, _ = completed_replay
    proposal = summary.read(out / 'training/proposals.json')['arms']['C0']
    base = summary.read(out / 'evaluation/f68/replay/result.json')
    candidate = copy.deepcopy(base)
    for row in candidate['rows']:
        row['scores']['old_projection'] -= 5e-9
    assert not review.replay_eligibility(proposal, candidate, base)['holdout_candidate']
    for row in candidate['rows']:
        row['scores']['old_projection'] -= 2e-8
    proposal['minimum_cosine'] = -1e-5
    assert review.replay_eligibility(proposal, candidate, base)['holdout_candidate']
    proposal['minimum_cosine'] = -1.0001e-5
    assert review.replay_eligibility(proposal, candidate, base)['failed_checks'] == ['rounded_displacement_protected']
    proposal['minimum_cosine'] = 0.
    candidate['training_task_protection']['passed'] = False
    assert review.replay_eligibility(proposal, candidate, base)['failed_checks'] == ['training_tasks_protected']


@pytest.mark.parametrize('damage', ('qualification', 'missing_row', 'duplicate_row', 'comparisons', 'pt_sha', 'png_sha'))
def test_rejects_ineligible_or_incomplete_evidence(completed_replay, damage):
    out, prior = completed_replay
    report_path = out / 'evaluation/C1/replay/result.json'
    if damage == 'qualification':
        update(out / 'numerical_qualification.json', lambda value: value.update(passed=False))
    elif damage == 'missing_row':
        update(report_path, lambda value: value['rows'].pop())
    elif damage == 'duplicate_row':
        update(report_path, lambda value: value['rows'].__setitem__(-1, value['rows'][0]))
    elif damage == 'comparisons':
        update(report_path, lambda value: value['training_task_protection'].update(comparisons=335))
    elif damage == 'pt_sha':
        (out / 'gradients/shard2.pt').write_bytes(b'changed')
    else:
        row = summary.read(report_path)['rows'][0]
        (out / row['image']).write_bytes(b'changed')
    with pytest.raises(AssertionError):
        review.review(out, prior)
    assert not (out / 'replay_decision.json').exists()


def test_innovation_requires_both_references(completed_replay):
    out, _ = completed_replay
    reports = {model: {'replay': summary.read(out / 'evaluation' / model / 'replay/result.json')}
               for model in summary.MODELS}
    # C1仍优于f68，但C0更接近理想斜率，不能称两参照创新成立。
    for row in reports['C0']['replay']['responses']:
        row['response']['center_slope'] = .95
    result = summary.innovation(reports, {'C1': {'eligible': True}})
    assert result['versus']['f68']['passed'] and not result['versus']['C0']['passed']
    assert not result['paired_and_absolute_improvement']


def test_numeric_tail_tolerance_cannot_hide_gate_change():
    assert review.same_numbers({'delta': 1. + 2e-15, 'passed': True}, {'delta': 1., 'passed': True})
    assert not review.same_numbers({'delta': 1., 'passed': False}, {'delta': 1., 'passed': True})


def test_reader_rejects_holdout_path(tmp_path):
    with pytest.raises(AssertionError, match='holdout'):
        review.Inputs().read(tmp_path / 'evaluation/C0/holdout/result.json')
