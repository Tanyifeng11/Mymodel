"""末步头证据汇总边界；不运行采样、反传或训练。"""
import copy
import zipfile

import pytest
import torch

from models.e2_state_texture_b import StateTextureB
from tools import e2_terminal_head_step_summary as review
from tools import e2_state_b_pair_summary as summary
from tools.e2_oracle_texture import weights_digest
from test_e2_b_scaled_replay_review import completed_replay, update, write


@pytest.mark.parametrize('nonzero', (False, True))
def test_actual_delta_signed_zero_and_fp32_merged_weights(nonzero):
    base = StateTextureB(color_channels=2, reference_channels=192)
    with torch.no_grad():
        base.output.weight.fill_(.125)
        base.output.bias.fill_(.25)
    weight, bias = torch.zeros_like(base.output.weight), torch.zeros_like(base.output.bias)
    if nonzero:
        weight.flatten()[0] = .001
        weight.flatten()[1] = 1e-10  # 合并至FP32后消失，不能把增量SHA当真实弦SHA。
    vector = torch.cat((weight.flatten(), bias.flatten()))
    original = torch.cat((base.output.weight.detach().flatten(), base.output.bias.detach().flatten())).double()
    merged = torch.cat(((base.output.weight.detach() + weight).flatten(),
                        (base.output.bias.detach() + bias).flatten())).double()
    proposal = dict(base_checkpoint_sha256='base-file', head_sha256=review.tensor_sha(vector),
        initial_head_sha256=review.tensor_sha(torch.zeros_like(vector)),
        actual_descent_sha256=review.tensor_sha(torch.zeros_like(vector) - vector),
        actual_norm=float(vector.norm()), parameter_updates=int(nonzero),
        effective_weight_difference_sha256=review.tensor_sha(merged - original), base_b_hash=weights_digest(base),
        checkpoint_sha256='candidate-file')
    payload = dict(delta_weight=weight, delta_bias=bias, base_checkpoint_sha256='base-file',
                   report={k: v for k, v in proposal.items() if k != 'checkpoint_sha256'})
    checked = review.validate_candidate(payload, proposal, base)
    assert checked['actual_head_norm'] == float(vector.norm())
    assert checked['actual_descent_sha256'] != review.tensor_sha(-vector)
    if nonzero:
        assert (merged - original)[1] == 0 and vector[1] != 0
    for field in ('head_sha256', 'effective_weight_difference_sha256', 'base_b_hash'):
        bad = dict(proposal, **{field: 'corrupted'})
        bad_payload = dict(payload, report={k: v for k, v in bad.items() if k != 'checkpoint_sha256'})
        with pytest.raises(AssertionError):
            review.validate_candidate(bad_payload, bad, base)


@pytest.fixture
def terminal_replay(completed_replay, monkeypatch):
    """沿用完整48项评分夹具；只隔离来源加载，候选张量由上面的真实CPU测试覆盖。"""
    out, prior = completed_replay
    overlay = dict(source_sha256={}, base_checkpoint_sha256='f68-checkpoint')
    write(out / 'one_step_overlay.json', overlay)
    write(out / 'gradient_qualification.json', dict(passed=True))
    preflight = prior / 'autonomous_repair_20261116/terminal_head_preflight'
    write(preflight / 'qualification.json', dict(passed=True))
    write(preflight / 'preflight_protocol.json', dict(sentinels=[]))
    spec = summary.read(out / 'protocol.json')
    baseline = summary.read(out / 'evaluation/f68/replay/result.json')
    proposals = summary.read(out / 'training/proposals.json')
    proposals['stop_before_replay'] = False
    values, hashes = {}, {}
    for shard in range(4):
        path = out / 'gradients' / ('shard%d.json' % shard)
        hashes[path.name] = summary.sha(path)
        values.update({(row['case'], row['sample_condition']): row for row in summary.read(path)['values']})
    for arm in review.ARMS:
        proposal = proposals['arms'][arm]
        proposal.update(base_b_hash=baseline['b_hash'], head_sha256='head-' + arm)
        def adapt(report):
            report.update(overlay_sha256=summary.sha(out / 'one_step_overlay.json'), source_sha256={},
                base_b_hash=baseline['b_hash'], head_hash=proposal['head_sha256'],
                base_checkpoint_sha256='f68-checkpoint', complete_s0_images=48, new_reverse_passes=0,
                adopted_checkpoints=[])
            for row in report['rows']:
                row.update(full_cache_latent_equal=True, full_cache_tasks_equal=True,
                           full_cache_png_equal=True, prefix_unchanged=True)
        update(out / 'evaluation' / arm / 'replay/result.json', adapt)
    write(out / 'training/proposals.json', proposals)
    monkeypatch.setattr(review, 'load_inputs', lambda *_: (
        overlay, spec, {'passed': True}, {'passed': True}, proposals, baseline, values, hashes, {}))
    return out, prior, proposals


def test_complete_review_preserves_original_gates_and_never_adopts(terminal_replay):
    out, prior, proposals = terminal_replay
    result = review.review(out, prior)
    assert result['complete'] and result['complete_s0_images'] == 96
    assert result['new_reverse_passes'] == 352 and result['adopted_checkpoints'] == []
    assert result['old_holdout_not_read'] and result['holdout_results_read'] == 0
    assert all(v['holdout_candidate'] and not v['adopted'] for v in result['decisions'].values())
    assert len(result['pastel']) == 4 and len(result['focus_metrics']) == 9
    assert result['innovation']['paired_and_absolute_improvement']
    with zipfile.ZipFile(out / 'review.zip') as archive:
        names = archive.namelist()
    assert len([name for name in names if name.endswith('.png')]) == 144
    assert len([name for name in names if name.startswith('gradients/') and name.endswith('.pt')]) == 4
    assert not any('/holdout/' in name or name.startswith('cache/') for name in names)
    baseline = summary.read(out / 'evaluation/f68/replay/result.json')
    candidate = copy.deepcopy(baseline)
    proposal = proposals['arms']['C0']
    for row in candidate['rows']:
        row['scores']['old_projection'] -= 5e-9
    assert not review.replay_eligibility(proposal, candidate, baseline)['holdout_candidate']
    for row in candidate['rows']:
        row['scores']['old_projection'] -= 2e-8
    proposal['minimum_cosine'] = -1e-5
    assert review.replay_eligibility(proposal, candidate, baseline)['holdout_candidate']
    proposal['minimum_cosine'] = -1.0001e-5
    assert review.replay_eligibility(proposal, candidate, baseline)['failed_checks'] == ['rounded_displacement_protected']


def test_both_zero_is_incomplete_and_never_reads_candidate_replays(terminal_replay):
    out, prior, proposals = terminal_replay
    proposals['stop_before_replay'] = True
    for arm in review.ARMS:
        proposals['arms'][arm]['parameter_updates'] = 0
        (out / 'evaluation' / arm / 'replay/result.json').write_text('MUST_NOT_READ', encoding='utf-8')
    write(out / 'training/proposals.json', proposals)
    result = review.review(out, prior)
    assert not result['complete'] and result['status'] == 'stopped_both_zero'
    assert result['complete_s0_images'] == 0 and result['decisions'] == {}
    assert result['evaluated_candidate_arms'] == [] and result['adopted_checkpoints'] == []
    assert 'innovation' not in result and 'groups' not in result
    with zipfile.ZipFile(out / 'review.zip') as archive:
        assert not any(name.startswith('evaluation/') for name in archive.namelist())


@pytest.mark.parametrize('damage', ('missing_row', 'comparisons', 'png_sha', 'prefix'))
def test_rejects_partial_or_inconsistent_candidate(terminal_replay, damage):
    out, prior, _ = terminal_replay
    path = out / 'evaluation/C1/replay/result.json'
    if damage == 'missing_row':
        update(path, lambda report: report['rows'].pop())
    elif damage == 'comparisons':
        update(path, lambda report: report['training_task_protection'].update(comparisons=335))
    elif damage == 'prefix':
        update(path, lambda report: report['rows'][0].update(prefix_unchanged=False))
    else:
        row = summary.read(path)['rows'][0]
        (out / row['image']).write_bytes(b'corrupt')
    with pytest.raises(AssertionError):
        review.review(out, prior)
    assert not (out / 'result.json').exists()
