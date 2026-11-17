"""只读汇总末步头的投影预算重试；复用原梯度，不新增反传或采用候选。"""
import argparse
import subprocess
import zipfile
from pathlib import Path

import torch

from models.e2_state_texture_b import StateTextureB
from tools import e2_state_b_pair_summary as summary
from tools.e2_b_scaled_replay_review import Inputs, replay_eligibility, same_numbers, validate_report
from tools.e2_state_b_pair_evaluation import strict_replay_protection
from tools.e2_terminal_head_step_summary import validate_candidate


SOURCE = Path(__file__).resolve().parents[1]
ARMS = ('C0', 'C1')
FAILURES = ('raw_sha_mismatch', 'historical_reproduction_mismatch', 'no_shared_feasible_budget')


def load_inputs(out, prior, audit):
    # 延迟导入，允许主入口薄调本模块而不产生顶层循环。
    from tools import e2_terminal_head_projection_retry as retry
    args = retry.make_args(out, prior)
    overlay = retry.verify_inputs(args)
    state = retry.load_state(args)
    diagnostic = audit.read(out / 'solver_diagnostic.json')
    assert audit.read(out / 'solver_overlay.json') == overlay == state['overlay']
    assert audit.read(out / 'protocol.json') == state['spec']
    assert audit.read(out / 'gradient_qualification.json') == state['qualification']
    assert audit.read(out / 'source_failure/proposals.json') == state['source_proposals']
    assert state['qualification']['passed'] and state['source_proposals']['new_reverse_passes'] == 352
    assert state['source_proposals']['stop_before_replay']
    for index in range(4):
        meta = audit.read(out / 'gradients' / ('shard%d.json' % index))
        assert audit.sha(out / 'gradients' / ('shard%d.pt' % index)) == meta['tensor_sha256']
    for arm in ARMS:
        assert audit.sha(out / 'source_failure' / (arm + '.pt')) == state['source_proposals']['arms'][arm]['checkpoint_sha256']
    if 'raw_pt_sha256' in diagnostic:
        assert audit.sha(out / 'raw.pt') == diagnostic['raw_pt_sha256']
    if diagnostic['all_feasible']:
        assert diagnostic['selected_budget'] is not None
        retry.verify(args)
    else:
        assert diagnostic['status'] in FAILURES and diagnostic['selected_budget'] is None
        assert not (out / 'training').exists() and not (out / 'solver_receipt.json').exists()
    return args, state, diagnostic


def package(out, complete, reports):
    names = ['protocol.json', 'one_step_overlay.json', 'gradient_qualification.json',
             'solver_overlay.json', 'solver_diagnostic.json', 'source_failure/proposals.json', 'result.json']
    names += ['gradients/shard%d.%s' % (index, ext) for index in range(4) for ext in ('json', 'pt')]
    names += ['source_failure/%s.pt' % arm for arm in ARMS]
    names += [name for name in ('raw.pt', 'solver_selected.pt') if (out / name).exists()]
    if complete:
        names += ['solver_receipt.json', 'training/proposals.json']
        names += ['training/%s.pt' % arm for arm in ARMS]
        for model, report in reports.items():
            names.append('evaluation/%s/replay/result.json' % model)
            names.extend(row['image'] for row in report['replay']['rows'])
    # 显式白名单：旧cache与holdout不读取、不归档。
    names = sorted(set(names))
    for name in names:
        assert 'holdout' not in Path(name).parts and not name.startswith('cache/')
        (out / name).resolve().relative_to(out.resolve())
    manifest = {name: summary.sha(out / name) for name in names}
    summary.write(out / 'review_manifest.json', manifest)
    with zipfile.ZipFile(out / 'review.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(out / name, name)
        archive.write(out / 'review_manifest.json', 'review_manifest.json')
    return manifest


def replay_results(out, args, state, diagnostic, audit):
    one_overlay = audit.read(out / 'one_step_overlay.json')
    spec = state['spec']
    protocol_sha = audit.sha(out / 'protocol.json')
    proposals = audit.read(out / 'training/proposals.json')
    receipt = audit.read(out / 'solver_receipt.json')
    assert proposals['new_reverse_passes'] == 0 and proposals['source_reverse_passes'] == 352
    assert not proposals['stop_before_replay'] and not proposals['adopted_checkpoints']
    assert all(proposals['arms'][arm]['parameter_updates'] == 1 and
               proposals['arms'][arm]['projection']['feasible'] for arm in ARMS)
    baseline = audit.read(out / 'evaluation/f68/replay/result.json')
    assert baseline['checkpoint_sha256'] == one_overlay['base_checkpoint_sha256'] == spec['checkpoint_sha256']
    assert audit.sha(out / 'evaluation/f68/replay/result.json') == audit.sha(args.source_run / 'evaluation/f68/replay/result.json')
    values, hashes = {}, {}
    for index, shard in enumerate(state['shards']):
        hashes['shard%d.json' % index] = audit.sha(out / 'gradients' / ('shard%d.json' % index))
        for row in shard['values']:
            key = row['case'], row['sample_condition']
            assert key not in values
            values[key] = row
    old_values = {(row['case'], row['sample_condition']): dict(tasks=row['training_tasks'], noise_digest=row['noise_digest'])
                  for row in baseline['rows']}
    assert len(values) == 48 and set(values) == set(old_values)
    assert all(value['tasks'] == old_values[key]['tasks'] and value['noise_digest'] == old_values[key]['noise_digest']
               for key, value in values.items())
    validate_report(baseline, 'f68', spec, protocol_sha, old_values,
                    baseline['training_task_protection']['baseline_gradient_json_sha256'], audit, out)
    base_rows = {summary.row_key(row, 'replay'): row for row in baseline['rows']}
    assert same_numbers(baseline['strict_replay_protection'], strict_replay_protection(baseline['rows'], base_rows))
    base = StateTextureB(color_channels=2, reference_channels=192)
    assert audit.sha(args.checkpoint) == one_overlay['base_checkpoint_sha256']
    base.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=False)['b'])
    reports, decisions, tensor_checks = {'f68': {'replay': baseline}}, {}, {}
    for arm in ARMS:
        proposal = proposals['arms'][arm]
        path = out / 'training' / (arm + '.pt')
        assert audit.sha(path) == proposal['checkpoint_sha256']
        tensor_checks[arm] = validate_candidate(torch.load(path, map_location='cpu', weights_only=False), proposal, base)
        replay = audit.read(out / 'evaluation' / arm / 'replay/result.json')
        assert replay['overlay_sha256'] == audit.sha(out / 'one_step_overlay.json')
        assert replay['source_sha256'] == one_overlay['source_sha256']
        assert replay['base_b_hash'] == proposal['base_b_hash'] == baseline['b_hash']
        assert replay['head_hash'] == proposal['head_sha256'] and replay['net_hash'] == baseline['net_hash']
        assert replay['base_checkpoint_sha256'] == one_overlay['base_checkpoint_sha256']
        assert replay['checkpoint_sha256'] == proposal['checkpoint_sha256']
        assert replay['complete_s0_images'] == 48 and replay['new_reverse_passes'] == 0 and not replay['adopted_checkpoints']
        validate_report(replay, arm, spec, protocol_sha, values, hashes, audit, out)
        assert same_numbers(replay['strict_replay_protection'], strict_replay_protection(replay['rows'], base_rows))
        assert all(all(row[key] for key in ('full_cache_latent_equal', 'full_cache_tasks_equal',
                   'full_cache_png_equal', 'prefix_unchanged')) for row in replay['rows'])
        reports[arm] = {'replay': replay}
        decisions[arm] = replay_eligibility(proposal, replay, baseline)
    comparisons = {}
    for current, reference in (('C0', 'f68'), ('C1', 'f68'), ('C1', 'C0')):
        comparisons[current + '_minus_' + reference] = dict(
            replay=summary.paired_changes(reports[reference]['replay'], reports[current]['replay'], 'replay'),
            response=summary.response_changes(reports[reference]['replay'], reports[current]['replay']))
    original = summary.innovation(reports, {'C1': {'eligible': decisions['C1']['holdout_candidate']}})
    innovation = dict(versus=original['versus'], paired_and_absolute_improvement=original['paired_and_absolute_improvement'],
        C1_holdout_candidate=decisions['C1']['holdout_candidate'], replay_improvement_with_protection=original['effective'],
        improvement_tolerance=original['improvement_tolerance'], correlation_tolerance=original['correlation_tolerance'],
        scope='仅原replay两参照判据；不代表holdout通过、模型采用或独立泛化')
    pastel = summary.pastel_rows(reports)
    for condition in pastel:
        for model, item in condition['models'].items():
            for arm, metrics in item['arms'].items():
                row = next(row for row in reports[model]['replay']['rows'] if
                           (row['case'], row['condition'], row['arm']) == (condition['case'], condition['condition'], arm))
                metrics.update(amplitude_pass=row['scores']['amplitude_pass'], l_quality_pass=row['scores']['l_quality_pass'])
    result = dict(proposals=proposals, solver_receipt=receipt, decisions=decisions, comparisons=comparisons,
        candidate_tensor_checks=tensor_checks, pastel=pastel, innovation=innovation,
        groups={model: report['replay']['groups'] for model, report in reports.items()},
        response_groups={model: summary.response_groups(report['replay']) for model, report in reports.items()},
        focus_metrics=list(summary.PROTECTION), evaluated_candidate_arms=list(ARMS),
        amplitude_quality=dict(amplitude_interval_inclusive=[.75, 1.25], quality_minimum_correlation=.5),
        capture_evidence=dict(candidate_rows_with_25_B_calls=96, full_cache_and_prefix_checks=96),
        interpretation='只改变求解预算；原replay资格仅允许后续考虑holdout，任何候选尚未采用')
    return result, reports


def review(out, prior):
    out, prior = Path(out).resolve(), Path(prior).resolve()
    assert not (out / 'result.json').exists(), '不覆盖已有汇总'
    audit = Inputs()
    args, state, diagnostic = load_inputs(out, prior, audit)
    complete = bool(diagnostic['all_feasible'])
    result = dict(phase='terminal head fixed-gradient projection budget retry', complete=complete,
        status='complete_replay' if complete else 'stopped_before_replay',
        parameter_count=2308, solver_overlay=state['overlay'], solver_diagnostic=diagnostic,
        qualification=state['qualification'], source_proposals=state['source_proposals'], source_job='125463',
        new_reverse_passes=0, source_reverse_passes=352, new_full_B_or_UNet_reverse_passes=0,
        complete_s0_images=96 if complete else 0, reused_f68_images=48 if complete else 0,
        source_optimizer_proposals=2, raw_reconstruction_optimizer_steps=diagnostic['raw_reconstruction_optimizer_steps'],
        optimizer_steps_executed_in_propose=0, saved_candidate_parameter_updates=2 if complete else 0,
        adopted_checkpoints=[], holdout_results_read=0, old_holdout_not_read=True, holdout_evaluated=False,
        f68_unchanged=True, old_full_B_gradient_directions_used=False,
        tool_git_commit=subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip(),
        tool_sha256=audit.sha(Path(__file__)))
    if complete:
        update, reports = replay_results(out, args, state, diagnostic, audit)
        result.update(update)
    else:
        reports = {}
        result.update(evaluated_candidate_arms=[], decisions={}, stop_reason=diagnostic['status'],
                      interpretation='未得到两臂共同可行预算，未生成新候选或评测；保留诊断证据，不填造保护、改善或采用结论')
    for path, digest in list(audit.hashes.items()):
        assert audit.sha(path) == digest
    result['review_input_sha256'] = dict(audit.hashes)
    summary.write(out / 'result.json', result)
    package(out, complete, reports)
    print('TERMINAL_PROJECTION_REVIEW', result['status'], summary.sha(out / 'review.zip'), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prior', type=Path, required=True)
    args = parser.parse_args()
    review(args.out, args.prior)


if __name__ == '__main__':
    main()
