"""只读汇总末步头的一次C0/C1提案；原replay门槛不变，不启动holdout。"""
import argparse
import hashlib
import math
import subprocess
import zipfile
from pathlib import Path

import torch

from models.e2_state_texture_b import StateTextureB
from tools import e2_state_b_pair_summary as summary
from tools.e2_b_endpoint_losses import TASKS
from tools.e2_b_scaled_replay_review import Inputs, replay_eligibility, same_numbers, validate_report
from tools.e2_oracle_texture import weights_digest
from tools.e2_state_b_pair_evaluation import strict_replay_protection


SOURCE = Path(__file__).resolve().parents[1]
ARMS = ('C0', 'C1')
PARAMETERS = 2308


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def validate_candidate(payload, proposal, base):
    """增量坐标与合并至FP32底座后的真实权重差分别核验，不能混作同一SHA。"""
    assert set(payload) == {'delta_weight', 'delta_bias', 'report', 'base_checkpoint_sha256'}
    assert payload['base_checkpoint_sha256'] == proposal['base_checkpoint_sha256']
    assert payload['report'] == {k: v for k, v in proposal.items() if k != 'checkpoint_sha256'}
    weight, bias = payload['delta_weight'], payload['delta_bias']
    assert weight.shape == base.output.weight.shape and bias.shape == base.output.bias.shape
    assert weight.dtype == bias.dtype == torch.float32
    assert torch.isfinite(weight).all() and torch.isfinite(bias).all()
    vector = torch.cat((weight.flatten(), bias.flatten()))
    assert vector.numel() == PARAMETERS and tensor_sha(vector) == proposal['head_sha256']
    assert tensor_sha(torch.zeros_like(vector)) == proposal['initial_head_sha256']
    descent = torch.zeros_like(vector) - vector
    assert tensor_sha(descent) == proposal['actual_descent_sha256']
    assert math.isclose(float(vector.norm()), proposal['actual_norm'], rel_tol=1e-6, abs_tol=1e-12)
    assert proposal['parameter_updates'] == int(bool(torch.count_nonzero(vector)))
    original = torch.cat((base.output.weight.detach().flatten(), base.output.bias.detach().flatten())).double()
    merged = torch.cat(((base.output.weight.detach() + weight).flatten(),
                        (base.output.bias.detach() + bias).flatten())).double()
    effective = merged - original
    assert tensor_sha(effective) == proposal['effective_weight_difference_sha256']
    assert weights_digest(base) == proposal['base_b_hash']
    return dict(head_sha256=tensor_sha(vector), base_b_hash=weights_digest(base),
                actual_descent_sha256=tensor_sha(descent), actual_head_norm=float(vector.norm()),
                effective_weight_difference_sha256=tensor_sha(effective),
                effective_weight_difference_norm=float(effective.norm()))


def load_inputs(out, prior, audit):
    preflight = prior / 'autonomous_repair_20261116/terminal_head_preflight'
    overlay = audit.read(out / 'one_step_overlay.json')
    overlay_sha = audit.sha(out / 'one_step_overlay.json')
    spec = audit.read(out / 'protocol.json')
    protocol_sha = audit.sha(out / 'protocol.json')
    assert protocol_sha == overlay['original_protocol_sha256'] == audit.sha(preflight / 'protocol.json')
    assert overlay['trainable_parameters'] == overlay['input_gradient_space'] == PARAMETERS
    assert overlay['head_start'] == 'zero' and overlay['gradient_loss_scale'] == 1024.
    assert overlay['original_losses_and_gates_unchanged'] and not overlay['old_full_B_gradient_directions_used']
    for path, digest in overlay['input_sha256'].items():
        assert audit.sha(path) == digest
    for name, digest in overlay['copied_sha256'].items():
        assert audit.sha(out / name) == digest
    for name, digest in overlay['source_sha256'].items():
        assert audit.sha(SOURCE / name) == digest
    assert 'tools/e2_terminal_head_step_summary.py' in overlay['source_sha256']
    preflight_qualification = audit.read(preflight / 'qualification.json')
    assert audit.sha(preflight / 'qualification.json') == overlay['source_qualification_sha256']
    assert preflight_qualification['passed'] and not preflight_qualification['training_started']
    assert preflight_qualification['maximum_relative_l2'] <= .005
    assert preflight_qualification['minimum_cosine'] >= .99999
    assert preflight_qualification['cache_sha256'] == overlay['preflight_cache_sha256']
    assert preflight_qualification['comparisons']['zero_conditions'] == 48
    assert preflight_qualification['comparisons']['repeated_head_task_gradients'] == 56
    baseline_path = out / 'evaluation/f68/replay/result.json'
    assert audit.sha(baseline_path) == audit.sha(preflight / 'historical_f68.json')
    baseline = audit.read(baseline_path)
    assert baseline['checkpoint_sha256'] == overlay['base_checkpoint_sha256'] == spec['checkpoint_sha256']
    zero_hash = tensor_sha(torch.zeros(PARAMETERS, dtype=torch.float32))
    values, json_hashes, pt_hashes, replicas = {}, {}, {}, []
    for index, pair in enumerate(spec['train']):
        js = out / 'gradients' / ('shard%d.json' % index)
        pt = js.with_suffix('.pt')
        meta = audit.read(js)
        assert audit.sha(pt) == meta['tensor_sha256']
        payload = torch.load(pt, map_location='cpu', weights_only=False)
        assert {k: v for k, v in meta.items() if k not in ('tensor_file', 'tensor_sha256')} == {
            k: v for k, v in payload.items() if k not in ('gradients', 'pair_gradient', 'absolute_gradient')}
        assert meta['shard'] == index and meta['case'] == pair['case_id']
        assert meta['protocol_sha256'] == protocol_sha and meta['overlay_sha256'] == overlay_sha
        assert meta['source_sha256'] == overlay['source_sha256'] and meta['loss_scale'] == 1024.
        assert meta['parameter_count'] == PARAMETERS and meta['head_hash'] == zero_hash
        assert meta['fresh_gradients'] and meta['gradients_unscaled'] and meta['frozen_unchanged']
        assert meta['all_finite'] and meta['numerical_qualification'] and meta['optimizer_updates'] == 0
        assert meta['base_b_hash'] == baseline['b_hash'] and meta['net_hash'] == baseline['net_hash']
        assert meta['preflight_cache_sha256'] == overlay['preflight_cache_sha256']['cache/shard%d.pt' % index]
        assert meta['reverse_passes'] == dict(tasks=84, y=3, absolute=1)
        assert payload['gradients'].shape == (12, 7, PARAMETERS)
        assert payload['pair_gradient'].shape == payload['absolute_gradient'].shape == (PARAMETERS,)
        assert all(torch.isfinite(payload[k]).all() for k in ('gradients', 'pair_gradient', 'absolute_gradient'))
        assert len(meta['items']) == len(meta['values']) == 12
        for row in meta['values']:
            key = row['case'], row['sample_condition']
            assert key not in values and set(row['tasks']) == set(TASKS) and row['zero_latent_equal']
            assert all(math.isfinite(v) for v in row['tasks'].values())
            values[key] = row
        json_hashes[js.name], pt_hashes[pt.name] = audit.sha(js), audit.sha(pt)
        replicas.extend(meta['replicas'])
    assert len(values) == 48 and len(replicas) == 8
    assert {(r['case'], r['condition'], r['arm']) for r in replicas} == set(
        map(tuple, preflight_qualification['protocol']['sentinels']))
    qualification = audit.read(out / 'gradient_qualification.json')
    assert qualification['replicas'] == replicas and qualification['passed'] and qualification['all_finite']
    assert all(r['passed'] and r['max_relative_l2'] <= .005 and r['minimum_cosine'] >= .99999 for r in replicas)
    assert all(r['additional_reference_graph1']['passed'] and
               r['additional_reference_graph1']['max_relative_l2'] <= .005 and
               r['additional_reference_graph1']['minimum_cosine'] >= .99999 for r in replicas)
    assert qualification['new_gradient_sha256'] == pt_hashes and qualification['gradient_json_sha256'] == json_hashes
    assert qualification['parameter_count'] == PARAMETERS and qualification['loss_scale'] == 1024.
    assert qualification['optimizer_updates'] == 0 and not qualification['old_full_B_gradient_directions_used']
    assert qualification['overlay_sha256'] == overlay_sha
    assert (qualification['new_task_reverse_passes'], qualification['new_y_reverse_passes'],
            qualification['new_absolute_reverse_passes']) == (336, 12, 4)
    old_values = {(r['case'], r['sample_condition']): dict(tasks=r['training_tasks'],
                  noise_digest=r['noise_digest']) for r in baseline['rows']}
    assert set(values) == set(old_values)
    for key, value in values.items():
        assert value['tasks'] == old_values[key]['tasks'] and value['noise_digest'] == old_values[key]['noise_digest']
    validate_report(baseline, 'f68', spec, protocol_sha, old_values,
                    baseline['training_task_protection']['baseline_gradient_json_sha256'], audit, out)
    base_rows = {summary.row_key(r, 'replay'): r for r in baseline['rows']}
    assert same_numbers(baseline['strict_replay_protection'], strict_replay_protection(baseline['rows'], base_rows))
    proposals = audit.read(out / 'training/proposals.json')
    assert set(proposals['arms']) == set(ARMS) and proposals['conditions'] == 48
    assert proposals['parameter_count'] == PARAMETERS and proposals['new_reverse_passes'] == 352
    assert proposals['independent_adamw_proposals'] == 2 and proposals['sequential_training_updates'] == 0
    assert proposals['frozen_unchanged'] and not proposals['heldout_used_for_training'] and not proposals['adopted_checkpoints']
    assert proposals['new_gradient_sha256'] == pt_hashes and proposals['gradient_json_sha256'] == json_hashes
    assert proposals['overlay_sha256'] == overlay_sha
    base_paths = [Path(path) for path, digest in overlay['input_sha256'].items()
                  if digest == overlay['base_checkpoint_sha256']]
    assert len(base_paths) == 1
    base = StateTextureB(color_channels=2, reference_channels=192)
    base.load_state_dict(torch.load(base_paths[0], map_location='cpu', weights_only=False)['b'])
    assert weights_digest(base) == baseline['b_hash']
    candidate_checks = {}
    for arm, proposal in proposals['arms'].items():
        assert proposal['arm'] == arm and proposal['protocol_sha256'] == protocol_sha
        assert proposal['overlay_sha256'] == overlay_sha and proposal['trainable_parameters'] == PARAMETERS
        assert proposal['gradient_qualification_sha256'] == audit.sha(out / 'gradient_qualification.json')
        assert proposal['base_checkpoint_sha256'] == overlay['base_checkpoint_sha256']
        assert proposal['optimizer_proposals'] == 1 and proposal['frozen_unchanged']
        assert not proposal['heldout_used_for_training'] and not proposal['adopted']
        path = out / 'training' / (arm + '.pt')
        assert audit.sha(path) == proposal['checkpoint_sha256']
        candidate_checks[arm] = validate_candidate(torch.load(path, map_location='cpu', weights_only=False), proposal, base)
    assert proposals['stop_before_replay'] == all(p['parameter_updates'] == 0 for p in proposals['arms'].values())
    return overlay, spec, preflight_qualification, qualification, proposals, baseline, values, json_hashes, candidate_checks


def package(out, preflight, complete, reports):
    names = ['one_step_overlay.json', 'protocol.json', 'gradient_qualification.json', 'training/proposals.json', 'result.json']
    names += ['gradients/shard%d.%s' % (i, ext) for i in range(4) for ext in ('json', 'pt')]
    names += ['training/%s.pt' % arm for arm in ARMS]
    files = {name: out / name for name in names}
    files['provenance/preflight_qualification.json'] = preflight / 'qualification.json'
    files['provenance/preflight_protocol.json'] = preflight / 'preflight_protocol.json'
    if complete:
        for model, report in reports.items():
            files['evaluation/%s/replay/result.json' % model] = out / 'evaluation' / model / 'replay/result.json'
            files.update({row['image']: out / row['image'] for row in report['replay']['rows']})
    # 不递归打包；旧cache大PT与任何holdout均不进入归档。
    assert all('holdout' not in Path(name).parts and not name.startswith('cache/') for name in files)
    manifest = {name: summary.sha(path) for name, path in sorted(files.items())}
    summary.write(out / 'review_manifest.json', manifest)
    with zipfile.ZipFile(out / 'review.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
        archive.write(out / 'review_manifest.json', 'review_manifest.json')
    return manifest


def review(out, prior):
    out, prior = Path(out).resolve(), Path(prior).resolve()
    assert not (out / 'result.json').exists(), '不覆盖已有汇总'
    audit = Inputs()
    overlay, spec, preflight_q, qualification, proposals, baseline, values, hashes, candidate_checks = load_inputs(out, prior, audit)
    complete = not proposals['stop_before_replay']
    result = dict(phase='terminal head independent one-step C0/C1', complete=complete,
        status='complete_replay' if complete else 'stopped_both_zero', parameter_count=PARAMETERS,
        overlay=overlay, preflight_qualification=preflight_q, qualification=qualification, proposals=proposals,
        candidate_tensor_checks=candidate_checks, complete_s0_images=96 if complete else 0,
        reused_f68_images=48, new_reverse_passes=352, new_full_B_or_UNet_reverse_passes=0,
        independent_optimizer_proposals=2, adopted_checkpoints=[], holdout_results_read=0,
        old_holdout_not_read=True, holdout_evaluated=False, f68_unchanged=True,
        old_full_B_gradient_directions_used=False, tool_git_commit=subprocess.check_output(
            ['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip(),
        tool_sha256=audit.sha(Path(__file__)))
    reports = {'f68': {'replay': baseline}}
    if not complete:
        result.update(evaluated_candidate_arms=[], decisions={}, stop_reason='both proposals have zero parameter updates; replay not executed',
                      interpretation='保留失败提案证据；未完成候选评测，不填造保护、改善或采用结论')
    else:
        base_rows = {summary.row_key(r, 'replay'): r for r in baseline['rows']}
        decisions = {}
        for arm in ARMS:
            replay = audit.read(out / 'evaluation' / arm / 'replay/result.json')
            proposal = proposals['arms'][arm]
            assert replay['overlay_sha256'] == audit.sha(out / 'one_step_overlay.json')
            assert replay['source_sha256'] == overlay['source_sha256']
            assert replay['base_b_hash'] == proposal['base_b_hash'] == baseline['b_hash']
            assert replay['head_hash'] == proposal['head_sha256'] and replay['net_hash'] == baseline['net_hash']
            assert replay['base_checkpoint_sha256'] == overlay['base_checkpoint_sha256']
            assert replay['checkpoint_sha256'] == proposal['checkpoint_sha256']
            assert replay['complete_s0_images'] == 48 and replay['new_reverse_passes'] == 0 and not replay['adopted_checkpoints']
            validate_report(replay, arm, spec, audit.sha(out / 'protocol.json'), values, hashes, audit, out)
            assert same_numbers(replay['strict_replay_protection'], strict_replay_protection(replay['rows'], base_rows))
            assert all(all(row[k] for k in ('full_cache_latent_equal', 'full_cache_tasks_equal', 'full_cache_png_equal', 'prefix_unchanged'))
                       for row in replay['rows'])
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
                    row = next(r for r in reports[model]['replay']['rows'] if
                        (r['case'], r['condition'], r['arm']) == (condition['case'], condition['condition'], arm))
                    metrics.update(amplitude_pass=row['scores']['amplitude_pass'], l_quality_pass=row['scores']['l_quality_pass'])
        result.update(decisions=decisions, comparisons=comparisons, pastel=pastel, innovation=innovation,
            groups={model: report['replay']['groups'] for model, report in reports.items()},
            response_groups={model: summary.response_groups(report['replay']) for model, report in reports.items()},
            focus_metrics=list(summary.PROTECTION), evaluated_candidate_arms=list(ARMS),
            amplitude_quality=dict(amplitude_interval_inclusive=[.75, 1.25], quality_minimum_correlation=.5),
            capture_evidence=dict(candidate_rows_with_25_B_calls=96, full_cache_and_prefix_checks=96),
            interpretation='原replay资格只允许后续考虑holdout；任何候选尚未采用，数值资格不等于质量收益')
    # 读取结束再次核SHA，避免打包期间输入状态改变而来源记录过时。
    for path, digest in list(audit.hashes.items()):
        assert audit.sha(path) == digest
    result['review_input_sha256'] = dict(audit.hashes)
    summary.write(out / 'result.json', result)
    package(out, prior / 'autonomous_repair_20261116/terminal_head_preflight', complete, reports)
    print('TERMINAL_ONE_STEP_REVIEW', result['status'], summary.sha(out / 'review.zip'), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prior', type=Path, required=True)
    args = parser.parse_args()
    review(args.out, args.prior)


if __name__ == '__main__':
    main()
