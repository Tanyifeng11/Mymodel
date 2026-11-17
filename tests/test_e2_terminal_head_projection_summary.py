"""预算失败不能冒充完整评测；仅检查新汇总边界，不运行求解、采样或反传。"""
from types import SimpleNamespace
import zipfile

import pytest

from tools import e2_state_b_pair_summary as summary
from tools import e2_terminal_head_projection_retry as retry
from tools import e2_terminal_head_projection_summary as review


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.write(path, value)


@pytest.fixture
def failure_evidence(tmp_path, monkeypatch):
    """来源加载由stub隔离；PT只作为归档字节，不冒充真实张量资格。"""
    out, prior = tmp_path / 'retry', tmp_path / 'prior'
    overlay, spec, qualification = {'new_reverse_passes': 0}, {}, {'passed': True}
    failed = dict(new_reverse_passes=352, stop_before_replay=True, arms={})
    for arm in ('C0', 'C1'):
        path = out / 'source_failure' / (arm + '.pt')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(('source-failed-' + arm).encode())
        failed['arms'][arm] = {'checkpoint_sha256': summary.sha(path)}
    for index in range(4):
        path = out / 'gradients' / ('shard%d.pt' % index)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(('reused-gradient-%d' % index).encode())
        write(path.with_suffix('.json'), {'tensor_sha256': summary.sha(path)})
    for name, value in [('solver_overlay.json', overlay), ('protocol.json', spec),
                        ('gradient_qualification.json', qualification), ('source_failure/proposals.json', failed),
                        ('one_step_overlay.json', {'unchanged_original': True})]:
        write(out / name, value)
    (out / 'raw.pt').write_bytes(b'exact-or-rejected-reconstruction-evidence')
    diagnostic = dict(status='raw_sha_mismatch', all_feasible=False, selected_budget=None,
        raw_pt_sha256=summary.sha(out / 'raw.pt'), raw_reconstruction_optimizer_steps=2)
    write(out / 'solver_diagnostic.json', diagnostic)
    # 若误读旧holdout或候选replay，将触发JSON解析失败。
    for relative in ('evaluation/C0/replay/result.json', 'evaluation/C1/holdout/result.json'):
        path = out / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('MUST_NOT_READ', encoding='utf-8')
    state = dict(overlay=overlay, spec=spec, qualification=qualification, source_proposals=failed)
    args = SimpleNamespace(out=out, prior=prior)
    monkeypatch.setattr(retry, 'make_args', lambda *_: args)
    monkeypatch.setattr(retry, 'verify_inputs', lambda *_: overlay)
    monkeypatch.setattr(retry, 'load_state', lambda *_: state)
    def unexpected(*_):
        raise AssertionError('失败分支不得读取成功候选或回放')
    monkeypatch.setattr(retry, 'verify', unexpected)
    monkeypatch.setattr(review, 'replay_results', unexpected)
    monkeypatch.setattr(review.subprocess, 'check_output', lambda *a, **k: 'unit-test\n')
    return out, prior, diagnostic


@pytest.mark.parametrize('status', review.FAILURES)
def test_controlled_failure_is_incomplete_and_archives_only_existing_evidence(failure_evidence, status):
    out, prior, diagnostic = failure_evidence
    diagnostic['status'] = status
    write(out / 'solver_diagnostic.json', diagnostic)
    result = review.review(out, prior)
    assert not result['complete'] and result['complete_s0_images'] == 0
    assert result['new_reverse_passes'] == 0 and result['source_reverse_passes'] == 352
    assert result['raw_reconstruction_optimizer_steps'] == 2 and result['saved_candidate_parameter_updates'] == 0
    assert result['decisions'] == {} and result['evaluated_candidate_arms'] == []
    assert result['adopted_checkpoints'] == [] and result['old_holdout_not_read']
    assert 'innovation' not in result and 'groups' not in result
    with zipfile.ZipFile(out / 'review.zip') as archive:
        names = archive.namelist()
    assert 'raw.pt' in names and all('source_failure/%s.pt' % arm in names for arm in ('C0', 'C1'))
    assert len([name for name in names if name.startswith('gradients/') and name.endswith('.pt')]) == 4
    assert not any(name.startswith(('training/', 'cache/', 'evaluation/')) for name in names)


def test_corrupted_raw_file_is_rejected(failure_evidence):
    out, prior, _ = failure_evidence
    (out / 'raw.pt').write_bytes(b'changed')
    with pytest.raises(AssertionError):
        review.review(out, prior)
    assert not (out / 'result.json').exists()


def test_no_shared_budget_cannot_have_new_training(failure_evidence):
    out, prior, _ = failure_evidence
    (out / 'training').mkdir()
    with pytest.raises(AssertionError):
        review.review(out, prior)
    assert not (out / 'result.json').exists()


def test_success_requires_a_selected_common_budget(failure_evidence):
    out, prior, diagnostic = failure_evidence
    diagnostic['all_feasible'] = True
    write(out / 'solver_diagnostic.json', diagnostic)
    with pytest.raises(AssertionError):
        review.review(out, prior)
    assert not (out / 'result.json').exists()
