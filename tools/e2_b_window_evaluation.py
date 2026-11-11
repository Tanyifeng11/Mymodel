"""沿用原v3逐图指标，单独检验FFT时间窗与原B2的配对差。"""
from pathlib import Path

import numpy as np

from tools import e2_b_screen_evaluation as legacy
from tools.e2_b_screen import bundle as legacy_bundle
from tools.e2_b_window import ARM, ARMS, CHECKPOINTS, MIN_T, log_rows
from tools.e2_fft_experiment import read, sha, write


def training_audit(out):
    records = {}
    legacy_rows = log_rows(out / 'training/B2/training.jsonl')
    for arm in ARMS:
        folder = out / 'training' / arm
        complete = read(folder / 'complete.json')
        assert complete['status'] == 'completed' and complete['updates'] == 2000
        if arm in CHECKPOINTS:
            assert complete['checkpoint_sha256'] == CHECKPOINTS[arm]
        else:
            assert complete['checkpoint_sha256'] == sha(folder / 'latest.pt')
            config = read(out / 'comparison' / (ARM + '.json'))
            assert complete['config'] == config and config['fft_t_min'] == MIN_T
            assert config['calibration_sha256'] == sha(out / 'calibration/result.json')
        rows = log_rows(folder / 'training.jsonl')
        assert [r['step'] for r in rows] == list(range(1, 2001))
        assert all(r['tokens'] == ref['tokens'] and r['chain'] == ref['chain'] for r, ref in zip(rows, legacy_rows))
        assert all(np.isfinite(r['grad_norm']) and all(np.isfinite(v) for v in r['losses'].values()) for r in rows)
        if arm == ARM:
            for r in rows:
                eligible = sum(MIN_T <= t['t'] <= 249 and not .2 <= t['drop'] < .3 for t in r['tokens'])
                assert 0 <= r['patch_counts']['valid_n'] <= r['patch_counts']['active_n'] <= eligible
            assert sum(r['patch_counts']['valid_n'] for r in rows) > 0
        records[arm] = dict(checkpoint_sha256=complete['checkpoint_sha256'], updates=len(rows),
            input_map_sha256=complete['config']['training_input_map_sha256'],
            active_samples=sum(r['patch_counts']['active_n'] for r in rows),
            valid_samples=sum(r['patch_counts']['valid_n'] for r in rows),
            training_seconds=sum(r['seconds'] for r in rows), reused=arm in CHECKPOINTS)
    assert len({r['input_map_sha256'] for r in records.values()}) == 1
    write(out / 'audit/training_completion.json', dict(passed=True, same_legacy_stream=True, arms=records))


def compare_responses(responses, pairs, arms):
    assert tuple(arms) == ARMS
    rows = legacy.response_comparisons(responses, pairs, ['B2', ARM])
    from tools.bf_t3_three_arm_report import paired

    values = {(r['arm'], r['index']): r for r in responses}
    metric = 'gray_strict_response_change'
    delta = [values[ARM, p['index']][metric] - values['B0', p['index']][metric] for p in pairs]
    rows.append(dict(arm=ARM, comparison=ARM+'_minus_B0', metric=metric,
                     family='descriptive', **paired(delta)))
    assert all(row['n'] == len(pairs) for row in rows)
    return rows


def protection_statistics(records, pairs):
    from tools.bf_t3_ablation_report import holm
    from tools.bf_t3_three_arm_report import paired

    lookup = {(r['arm'], r['case_id'], r['condition']): r for r in records}
    directions = {'edge_f1': 1, 'iou': 1, 'edge_l1': -1, 'leak': -1, 'tcf_lab': -1}
    rows = []
    for condition in ['old', 'new']:
        metrics = dict(directions, lpips=-1) if condition == 'old' else directions
        for metric, direction in metrics.items():
            deltas = []
            for p in pairs:
                values = [lookup[arm, p['case_id'], condition][metric] for arm in ['B2', ARM]]
                if all(v is not None and np.isfinite(v) for v in values):
                    deltas.append(values[1] - values[0])
            row = dict(condition=condition, metric=metric, comparison=ARM+'_minus_B2',
                       better_sign=direction, missing=len(pairs)-len(deltas), **paired(deltas))
            rows.append(row)
    holm(rows)
    return rows


def generate(args):
    out = Path(args.out)
    assert all(read(out / 'audit' / name)['passed'] for name in ['training_inputs_final.json', 'mask_inputs_final.json'])
    training_audit(out)
    legacy.generate(args)


def evaluate(args):
    out = Path(args.out)
    baseline = Path(args.baseline)
    assert read(out / 'audit/mask_inputs_final.json')['passed']
    legacy.evaluate(args, arms=ARMS, audit=training_audit, compare=compare_responses)
    pairs = read(out / 'strict_pairs.json')['pairs']
    rows = read(out / 'metrics/per_image.json')
    prior = {(r['arm'], r['case_id'], r['condition']): r for r in read(baseline / 'metrics/per_image.json')}
    reused = [r for r in rows if r['arm'] in CHECKPOINTS]
    assert len(rows) == 192 and len(reused) == 128
    differences = [dict(arm=r['arm'], case_id=r['case_id'], condition=r['condition'],
                        keys=sorted(k for k, v in r.items() if prior[r['arm'], r['case_id'], r['condition']][k] != v))
                   for r in reused if r != prior[r['arm'], r['case_id'], r['condition']]]
    write(out / 'audit/reused_metric_comparison.json', dict(passed=not differences,
        expected=len(reused), differences=differences, parent_metrics_sha256=sha(baseline / 'metrics/per_image.json')))
    assert not differences, '复用输出逐图指标未重现，请检查运行环境与冻结资产'
    version = read(out / 'metrics/version.json')
    assert version['lpips_assets'] == read(baseline / 'metrics/version.json')['lpips_assets']
    previous_assets = read(baseline / 'audit/dists_assets.json')
    current_assets = read(out / 'audit/dists_assets.json')
    assert {k: v for k, v in previous_assets.items() if k.endswith('_sha256')} == {
        k: v for k, v in current_assets.items() if k.endswith('_sha256')}
    protections = protection_statistics(rows, pairs)
    write(out / 'metrics/protection_statistics.json', protections)
    version.update(protocol=read(out / 'audit/protocol.json'), inherited_v3_metric_definition=True,
                   protection_comparisons=len(protections), reused_metric_records=len(reused))
    write(out / 'metrics/version.json', version)
    print('B_WINDOW_EVALUATION_PASS', len(rows), flush=True)


def bundle(args):
    root = Path(args.out)
    assert read(root / 'audit/reused_metric_comparison.json')['passed']
    assert len(read(root / 'metrics/per_image.json')) == 192
    legacy_bundle(args, extra_suffixes=('.jpg', '.jpeg'))
