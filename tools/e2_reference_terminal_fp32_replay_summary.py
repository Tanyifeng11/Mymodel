"""只修复125693已存96观测的seed/digest汇总元数据；不重跑模型或补造冻结后检。"""
import argparse
import copy
import hashlib
from pathlib import Path
import subprocess
import zipfile

from tools import e2_reference_terminal_fp32_propose as proposal
from tools import e2_state_b_pair_summary as summary
from tools import e2_state_b_pair_evaluation as evaluation
from tools.e2_b_scaled_replay_review import replay_eligibility
from tools.e2_state_b_amplitude_response import read, write, sha


SOURCE = Path(__file__).resolve().parents[1]
EXTRA = ('tools/e2_reference_terminal_fp32_replay_summary.py',
         'scripts/e2_reference_terminal_fp32_replay_summary.sbatch')


def repaired_models(observed, baseline, proposals):
    """只补继承前缀元数据；所有分数、图像、任务值和原分类必须保持。"""
    originals = {(r['case'], r['condition'], r['arm']): r for r in baseline['rows']}
    models = copy.deepcopy(observed['models'])
    for arm, report in models.items():
        assert len(report['rows']) == 48
        for row in report['rows']:
            old = originals[row['case'], row['condition'], row['arm']]
            assert all(row[n] == old[n] for n in ('sample_condition', 'group', 'factor', 'assets'))
            row.update(seed=old['seed'], noise_digest=old['noise_digest'], prefix_metadata_inherited=True)
        aggregate = evaluation.aggregate(report['rows'], 'replay')
        assert all(aggregate[n] == report[n] for n in ('groups', 'responses'))
        protected = evaluation.strict_replay_protection(report['rows'], originals)
        assert protected == report['strict_replay_protection']
        assert replay_eligibility(proposals['arms'][arm], report, baseline) == report['original_replay_checks']
    reports = dict(f68=dict(replay=baseline), **{a: dict(replay=r) for a, r in models.items()})
    innovation = summary.innovation(reports, dict(C1=dict(eligible=False)))
    return models, innovation


def run(args):
    parent_args = argparse.Namespace(**vars(args)); parent_args.out = args.proposals
    proposal.inputs(parent_args); proposal.verify(parent_args)
    old_protocol = read(args.replay/'protocol.json')
    original_commit = old_protocol['execution_commit']
    assert original_commit == 'edb4126d378cd254cda973a3953fa6603e24bd96'
    assert all(sha(args.replay/n) == h for n, h in read(args.replay/'manifest.json').items())
    assert all(sha(Path(n)) == h for n, h in old_protocol['input_sha256'].items())
    for name, digest in old_protocol['source_sha256'].items():
        original = subprocess.check_output(['git', '-C', str(SOURCE), 'show', original_commit+':'+name])
        assert hashlib.sha256(original).hexdigest() == digest
        if name != 'tools/e2_reference_terminal_fp32_replay.py':
            assert sha(SOURCE/name) == digest
    observed = read(args.replay/'result.json')
    assert observed['status'] == 'failed' and observed['error_type'] == 'KeyError' and observed['message'] == "'seed'"
    assert observed['source_final_verification']
    assert (observed['completed_cache_observations'], observed['zero_PNG_and_seven_task_controls']) == (96, 48)
    assert 'frozen_unchanged' not in observed, '不能把缺失的历史冻结后检补成已执行'
    baseline_file = args.source_run/'evaluation/f68/replay/result.json'
    baseline, proposals = read(baseline_file), read(args.proposals/'proposals.json')
    models, innovation = repaired_models(observed, baseline, proposals)
    assert not args.out.exists(), '保留原失败结果，不覆盖已有修复汇总'
    args.out.mkdir()
    files = [args.replay/n for n in ('protocol.json', 'result.json', 'manifest.json')]
    files += [baseline_file, args.proposals/'proposals.json']
    sources = set(old_protocol['source_sha256']) | set(EXTRA)
    protocol = dict(kind='metadata-only derived summary of saved96 observations',
        original_runtime_commit=original_commit, original_runtime_exit='FAILED 1:0',
        original_archive_sha256=sha(args.replay/'review.zip'),
        input_sha256={str(p.resolve()): sha(p) for p in files},
        original_source_sha256=old_protocol['source_sha256'],
        source_sha256={n: sha(SOURCE/n) for n in sorted(sources)},
        execution_commit=subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip(),
        model_or_metric_values_changed=False, VAE_decodes=0, reverse_passes=0, optimizer_steps=0, new_S0=0,
        holdout_read=0, frozen_runtime_postcheck_recovered=False)
    write(args.out/'protocol.json', protocol)
    result = dict(status='summary_complete_original_runtime_final_hash_checks_not_reached', summary_completed=True,
        models=models, innovation=innovation, source_integrity_verified=True,
        original_runtime=dict(status=observed['status'], error_type=observed['error_type'], message=observed['message'],
            generated_cache_observations=96, zero_PNG_and_seven_task_controls=48,
            initial_frozen_hashes=observed['frozen_hashes'], final_frozen_hash_checks_executed=False),
        images_and_tensors_relative_to=str(args.replay), tensor_file_sha256=observed['tensor_file_sha256'],
        model_or_metric_values_changed=False, new_S0=0, VAE_decodes=0, reverse_passes=0, optimizer_steps=0,
        holdout_read=0, adopted_checkpoints=[], formal_adoption_eligibility=False,
        interpretation='original images/scores/constraints preserved;missing historical frozen hash checks are not fabricated')
    assert all(sha(Path(n)) == h for n, h in protocol['input_sha256'].items())
    assert all(sha(SOURCE/n) == h for n, h in protocol['source_sha256'].items())
    write(args.out/'result.json', result)
    names = ['protocol.json', 'result.json']
    write(args.out/'manifest.json', {n: sha(args.out/n) for n in names})
    with zipfile.ZipFile(args.out/'review.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for n in names+['manifest.json']:
            archive.write(args.out/n, n)
    print('FP32_REPLAY_SUMMARY', sha(args.out/'review.zip'), flush=True)


if __name__ == '__main__':
    args = proposal.arguments(); args.proposals = args.out
    args.replay = args.prior/'autonomous_repair_20261116/reference_terminal_fp32_replay'
    args.out = args.prior/'autonomous_repair_20261116/reference_terminal_fp32_replay_summary'
    run(args)
