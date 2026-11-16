"""只读汇总固定缩放候选的完整 replay；通过仅允许进入 holdout，不采用权重。"""
import argparse
import math
import subprocess
from pathlib import Path

from tools import e2_state_b_pair_summary as summary
from tools.e2_b_endpoint_losses import TASKS
from tools.e2_state_b_pair_evaluation import aggregate, strict_replay_protection, training_comparison


SOURCE = Path(__file__).resolve().parents[1]


def same_numbers(actual, expected):
    """历史CPU重算可有约1e-15尾差；判定布尔、失败项和原门槛必须完全一致。"""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(
            same_numbers(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            same_numbers(a, b) for a, b in zip(actual, expected))
    if isinstance(expected, float):
        return isinstance(actual, (int, float)) and math.isclose(actual, expected, rel_tol=0., abs_tol=1e-12)
    return actual == expected


class Inputs:
    """显式记录实际读取的文件；禁止从任何 holdout 路径取结果或图片。"""
    def __init__(self):
        self.hashes = {}

    def sha(self, path):
        path = Path(path).resolve()
        assert 'holdout' not in path.parts, 'replay review 禁止读取 holdout'
        value = summary.sha(path)
        self.hashes[path.as_posix()] = value
        return value

    def read(self, path):
        self.sha(path)
        return summary.read(Path(path))


def replay_eligibility(proposal, replay, baseline):
    """与原 eligibility 的 replay 部分相同，不伪造 holdout 保护通过。"""
    original = {summary.row_key(row, 'replay'): row for row in baseline['rows']}
    delta = summary.mean([row['scores']['old_projection'] -
        original[summary.row_key(row, 'replay')]['scores']['old_projection'] for row in replay['rows']])
    minimum = proposal['minimum_cosine']
    checks = dict(projection_feasible=bool(proposal['projection']['feasible']),
        nonzero_parameter_update=proposal['parameter_updates'] > 0,
        rounded_displacement_protected=minimum is not None and minimum >= -1e-5,
        training_tasks_protected=bool(replay['training_task_protection']['passed']),
        strict_replay_pixels_protected=bool(replay['strict_replay_protection']['passed']),
        mean_residue_strictly_decreased=delta < -1e-8)
    return dict(holdout_candidate=all(checks.values()), adopted=False, checks=checks,
        failed_checks=[key for key, passed in checks.items() if not passed],
        mean_replay_old_projection_delta=delta,
        training_task_protection=replay['training_task_protection'],
        strict_replay_protection=replay['strict_replay_protection'])


def validate_report(report, model, spec, protocol_sha, task_baselines, baseline_hashes, audit, out):
    assert report['arm'] == model and report['split'] == 'replay'
    assert report['protocol_sha256'] == protocol_sha and report['frozen_unchanged']
    assert report['evaluation_optimizer_updates'] == 0
    assert not report['heldout_used_for_training'] and not report['heldout_used_for_evaluation']
    expected = {summary.row_key(row, 'replay'): row for row in spec['inputs']}
    rows = report['rows']
    actual = {summary.row_key(row, 'replay'): row for row in rows}
    assert len(expected) == len(rows) == len(actual) == 48 and set(actual) == set(expected), '需要完整且唯一的48项replay'
    assert len({key[:2] for key in actual}) == 16
    failures = []
    for key, row in actual.items():
        item = expected[key]
        assert row['group'] == item['group'] and row['sample_condition'] == item['sample_condition']
        assert row['model_arm'] == model and row['b_calls'] == 25
        relative = Path(row['image'])
        assert relative.parts[:3] == ('evaluation', model, 'replay')
        image = (out / relative).resolve()
        image.relative_to(out.resolve())
        assert audit.sha(image) == row['png_sha256'], '输出PNG SHA不符'
        assert set(row['training_tasks']) == set(TASKS)
        assert all(math.isfinite(value) for value in row['training_tasks'].values())
        assert all(row['scores'][name] is not None and math.isfinite(row['scores'][name])
                   for name in summary.METRICS), '完整replay缺少有限指标'
        amplitude, correlation = row['scores']['l_amplitude'], row['scores']['l_correlation']
        assert row['scores']['amplitude_pass'] == (.75 <= amplitude <= 1.25)
        assert row['scores']['l_quality_pass'] == (.75 <= amplitude <= 1.25 and correlation >= .5)
        original = task_baselines[row['case'], row['sample_condition']]
        assert row['noise_digest'] == original['noise_digest']
        comparison = training_comparison(row['training_tasks'], original['tasks'])
        assert row['training_task_comparison'] == comparison, '逐任务比较与对应梯度基线不符'
        if comparison['failures']:
            failures.append(dict(case=row['case'], condition=row['condition'], arm=row['arm'],
                                 failures=comparison['failures']))
    protection = report['training_task_protection']
    assert protection['comparisons'] == 336, '需要336项任务比较'
    assert protection['baseline_gradient_json_sha256'] == baseline_hashes
    assert protection['failures'] == failures and protection['passed'] == (not failures)
    recomputed = aggregate(rows, 'replay')
    assert same_numbers(report['groups'], recomputed['groups']) and same_numbers(report['responses'], recomputed['responses'])
    pastel = [row for row in rows if row['group'] == 'pastel']
    assert len(pastel) == 12 and len({(row['case'], row['condition']) for row in pastel}) == 4


def load_inputs(out, prior, audit):
    fixed = prior / 'projection_fp64_retry'
    protocol_sha = audit.sha(out / 'protocol.json')
    assert protocol_sha == audit.sha(fixed / 'protocol.json'), '原协议发生改变'
    spec = audit.read(out / 'protocol.json')
    qualification = audit.read(out / 'numerical_qualification.json')
    assert qualification['passed'] and qualification['all_finite'], '新梯度未通过数值资格'
    assert qualification['loss_scale'] == 1024 and qualification['optimizer_updates'] == 0
    assert len(qualification['replicas']) == 8 and all(row['passed'] for row in qualification['replicas'])
    assert qualification['maximum_relative_l2'] <= .005 and qualification['minimum_cosine'] >= .99999
    expected_names = {'shard%d.pt' % index for index in range(4)}
    assert set(qualification['new_gradient_sha256']) == expected_names
    new_baselines, new_hashes, old_hashes, shards = {}, {}, {}, []
    for index in range(4):
        pt, js = 'shard%d.pt' % index, 'shard%d.json' % index
        digest = audit.sha(out / 'gradients' / pt)
        assert digest == qualification['new_gradient_sha256'][pt], '资格对应PT SHA不符'
        shard = audit.read(out / 'gradients' / js)
        assert shard['tensor_sha256'] == digest and shard['numerical_qualification'] and shard['all_finite']
        assert shard['protocol_sha256'] == protocol_sha and shard['checkpoint_sha256'] == spec['checkpoint_sha256']
        assert shard['frozen_unchanged'] and shard['training_updates'] == 0
        assert shard['new_gradient_generation'] and not shard['cached_gradients_reused']
        assert len(shard['values']) == 12
        for row in shard['values']:
            key = row['case'], row['sample_condition']
            assert key not in new_baselines and set(row['tasks']) == set(TASKS)
            new_baselines[key] = row
        new_hashes[js] = audit.sha(out / 'gradients' / js)
        old_hashes[js] = audit.sha(fixed / 'gradients' / js)
        shards.append(shard)
    assert len(new_baselines) == 48
    proposals = audit.read(out / 'training/proposals.json')
    assert set(proposals['arms']) == {'C0', 'C1'} and proposals['conditions'] == 48
    assert proposals['full_task_reverse_passes'] == 336 and proposals['frozen_unchanged']
    assert not proposals['heldout_used_for_training']
    baseline_path = out / 'evaluation/f68/replay/result.json'
    assert audit.sha(baseline_path) == audit.sha(fixed / 'evaluation/f68/replay/result.json'), '复用f68记录改变'
    reports = {model: {'replay': audit.read(out / 'evaluation' / model / 'replay/result.json')}
               for model in summary.MODELS}
    baseline = reports['f68']['replay']
    old_baselines = {(row['case'], row['sample_condition']): dict(tasks=row['training_tasks'],
        noise_digest=row['noise_digest']) for row in baseline['rows']}
    for key, row in new_baselines.items():
        assert row['noise_digest'] == old_baselines[key]['noise_digest']
        assert max(abs(row['tasks'][task] - old_baselines[key]['tasks'][task]) for task in TASKS) <= 1e-6
    base_rows = {summary.row_key(row, 'replay'): row for row in baseline['rows']}
    for model in summary.MODELS:
        report = reports[model]['replay']
        validate_report(report, model, spec, protocol_sha,
            old_baselines if model == 'f68' else new_baselines,
            old_hashes if model == 'f68' else new_hashes, audit, out)
        assert same_numbers(report['strict_replay_protection'], strict_replay_protection(report['rows'], base_rows))
        if model == 'f68':
            assert report['checkpoint_sha256'] == spec['checkpoint_sha256']
        else:
            proposal = proposals['arms'][model]
            assert proposal['arm'] == model and proposal['protocol_sha256'] == protocol_sha
            assert not proposal['adopted'] and proposal['optimizer_proposals'] == 1
            assert proposal['checkpoint_start'] == spec['checkpoint_sha256']
            assert proposal['initial_b_hash'] == baseline['b_hash'] and proposal['final_b_hash'] == report['b_hash']
            assert audit.sha(out / 'training' / (model + '.pt')) == proposal['checkpoint_sha256'] == report['checkpoint_sha256']
            for key, value in report['checkpoint_flags'].items():
                assert proposal[key] == value
    assert len({report['replay']['net_hash'] for report in reports.values()} |
               {shard['net_hash'] for shard in shards}) == 1
    assert all(shard['b_hash'] == baseline['b_hash'] for shard in shards)
    return spec, qualification, proposals, reports


def review(out, prior):
    out, prior = Path(out).resolve(), Path(prior).resolve()
    audit = Inputs()
    spec, qualification, proposals, reports = load_inputs(out, prior, audit)
    decisions = {model: replay_eligibility(proposals['arms'][model], reports[model]['replay'],
                                         reports['f68']['replay']) for model in ('C0', 'C1')}
    changes = {}
    for candidate, baseline in (('C0', 'f68'), ('C1', 'f68'), ('C1', 'C0')):
        changes[candidate + '_minus_' + baseline] = dict(
            replay=summary.paired_changes(reports[baseline]['replay'], reports[candidate]['replay'], 'replay'),
            response=summary.response_changes(reports[baseline]['replay'], reports[candidate]['replay']))
    pastel = summary.pastel_rows(reports)
    for condition in pastel:
        for model, values in condition['models'].items():
            for arm, metrics in values['arms'].items():
                row = next(row for row in reports[model]['replay']['rows'] if
                    (row['case'], row['condition'], row['arm']) == (condition['case'], condition['condition'], arm))
                metrics.update(amplitude_pass=row['scores']['amplitude_pass'], l_quality_pass=row['scores']['l_quality_pass'])
    # 复用旧创新判据，但明确去掉“完整资格/有效改进”的标签。
    original = summary.innovation(reports, {'C1': {'eligible': decisions['C1']['holdout_candidate']}})
    innovation = dict(versus=original['versus'], paired_and_absolute_improvement=original['paired_and_absolute_improvement'],
        C1_holdout_candidate=decisions['C1']['holdout_candidate'],
        replay_improvement_with_protection=original['effective'],
        improvement_tolerance=original['improvement_tolerance'], correlation_tolerance=original['correlation_tolerance'],
        scope='仅本次固定候选replay两参照对照；不代表采用、独立泛化或holdout通过')
    result = dict(protocol_sha256=audit.sha(out / 'protocol.json'), numerical_qualification=qualification,
        proposals=proposals, replay_decisions=decisions, innovation=innovation,
        focus_metrics=list(summary.PROTECTION), comparisons=changes, pastel_conditions=pastel,
        groups={model: report['replay']['groups'] for model, report in reports.items()},
        response_groups={model: summary.response_groups(report['replay']) for model, report in reports.items()},
        original_amplitude_quality=dict(amplitude_interval_inclusive=[.75, 1.25], quality_minimum_correlation=.5,
            count_scope='浅彩4条件×3档逐项见pastel_conditions；每档计数见groups[model][pastel]'),
        old_holdout_not_read=True, holdout_results_read=0, holdout_evaluated=False,
        adopted_checkpoints=[], frozen_unchanged=True, evaluation_optimizer_updates=0,
        interpretation='holdout_candidate只表示数值资格与原replay保护通过；任何候选尚未采用，不生成或读取holdout',
        historical_task_baseline='复用f68的零差比较绑定旧shard JSON；C0/C1的336项比较绑定本轮新shard JSON，标量与历史差不超过1e-6',
        tool_git_commit=subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip(),
        tool_sha256=audit.sha(Path(__file__)), summary_source_sha256=audit.sha(Path(summary.__file__)),
        evaluation_source_sha256=audit.sha(SOURCE / 'tools/e2_state_b_pair_evaluation.py'))
    result['input_sha256'] = audit.hashes
    summary.write(out / 'replay_decision.json', result)
    print('SCALED_REPLAY_REVIEW', {arm: value['holdout_candidate'] for arm, value in decisions.items()}, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prior', type=Path, required=True)
    args = parser.parse_args()
    review(args.out, args.prior)


if __name__ == '__main__':
    main()
