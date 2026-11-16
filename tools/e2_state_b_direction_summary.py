"""固定方向的只读导数/缩步诊断汇总；子集回放不代替完整采用门槛。"""
import argparse
import json
import statistics
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from tools.e2_b_endpoint_losses import TASKS
from tools.e2_state_b_pair_evaluation import aggregate, metric_means, strict_replay_protection, training_comparison
from tools.e2_state_b_pair_summary import metric_change, response_groups, METRICS
from tools.e2_fft_experiment import read, write, sha


DIRECTIONS = ('C0', 'C1')
SCALES = (0., .125, .25, 1.)


def key(row):
    return row['case'], row['condition'], row['arm']


def scale_key(scale):
    return format(scale, 'g')


def means(values):
    return statistics.mean(values) if values else None


def median(values):
    return statistics.median(values) if values else None


def subset_summary(rows, baselines):
    comparisons, residue = [], []
    for row in rows:
        previous = baselines[key(row)]
        tasks = training_comparison(row['training_tasks'], previous['training_tasks'])
        comparisons.append(dict(case=row['case'], condition=row['condition'], arm=row['arm'], **tasks))
        old, current = previous['scores']['old_projection'], row['scores']['old_projection']
        tolerance = 1e-5 + 1e-4 * abs(old)
        residue.append(dict(case=row['case'], condition=row['condition'], arm=row['arm'],
                            delta=current - old, tolerance=tolerance, failed=current - old > tolerance))
    task_failures = Counter(task for row in comparisons for task in row['failures'])
    changes = {metric: metric_change([baselines[key(row)]['scores'][metric] for row in rows],
                                     [row['scores'][metric] for row in rows], metric) for metric in METRICS}
    return dict(count=len(rows), task_comparisons=len(rows) * len(TASKS),
        failed_task_comparisons=sum(task_failures.values()), failed_conditions=sum(bool(r['failures']) for r in comparisons),
        failures_by_task={task: task_failures[task] for task in TASKS}, training_task_comparisons=comparisons,
        residue_cap_failures=sum(r['failed'] for r in residue), residue_caps=residue,
        metrics=metric_means(rows), changes=changes,
        scope='固定16个训练侧条件，其中参考幅度档不均衡；112项task仅是该子集，不是全量336项门槛')


def pastel_summary(rows, baselines):
    selected = [row for row in rows if row['group'] == 'pastel']
    assert len(selected) == 12
    previous = {k: v for k, v in baselines.items() if v['group'] == 'pastel'}
    assert len(previous) == 12
    report = aggregate(selected, 'replay')
    return dict(**report, response=response_groups(report)['pastel'],
        protection=strict_replay_protection(selected, previous),
        scope='只有4浅彩×3参考档完整覆盖；保留原9项分组和逐图残留规则，不代表其他组或holdout通过')


def fd_statistics(rows, frozen=False):
    """仅描述给定FD与预测差异，不据此增设采用阈值。"""
    if frozen:
        valid = [r for r in rows if r['frozen_pattern_fd'] is not None]
        errors = [r['frozen_pattern_error'] for r in valid]
        values = [r['frozen_pattern_fd'] for r in valid]
        predictions = [r['prediction_cached'] for r in valid]
        relative = [abs(e) / max(abs(v), abs(p), 1e-12) for e, v, p in zip(errors, values, predictions)]
        agrees = [v * p > 0 for v, p in zip(values, predictions)]
    else:
        valid = rows
        errors = [r['error'] for r in valid]
        relative = [r['symmetric_relative_error'] for r in valid]
        agrees = [r['sign_agrees'] for r in valid]
    # 冻结pattern没有单独的重复噪声测量，不能冒用真实pattern的noise_abs。
    measured_noise = [] if frozen else [r for r in valid if r['noise_abs'] is not None]
    signals = ([abs(r['frozen_pattern_fd']) * 2 * r['h'] for r in valid]
               if frozen else [r['signal_abs'] for r in valid])
    return dict(count=len(valid), sign_agrees=sum(v is True for v in agrees),
        sign_disagrees=sum(v is False for v in agrees),
        median_symmetric_relative_error=median(relative), median_absolute_error=median([abs(e) for e in errors]),
        maximum_absolute_error=max([abs(e) for e in errors], default=None),
        median_signal_abs=median(signals),
        repeat_noise_measured_count=len(measured_noise),
        median_repeat_noise_abs=median([r['noise_abs'] for r in measured_noise]),
        signal_not_above_repeat_noise=sum(r['signal_abs'] <= r['noise_abs'] for r in measured_noise),
        relative_error_definition='abs(fd-prediction_cached)/max(abs(fd),abs(prediction_cached),1e-12)',
        repeat_noise_scope='not_measured_for_frozen_pattern' if frozen else 'same_training_task_repeated_f68',
        scope='重复噪声为0不排除确定性FP16/PNG量化台阶；pattern冻结语义只供导数诊断')


def finite_difference_summary(records, steps):
    result = {}
    for direction in DIRECTIONS:
        result[direction] = {}
        for h in steps:
            current = [r for r in records if r['direction'] == direction and r['h'] == h]
            core = [r for r in current if r['task'] != 'pattern']
            pattern = [r for r in current if r['task'] == 'pattern']
            assert len(core) == 48 and len(pattern) == 8
            result[direction][scale_key(h)] = dict(core_six_tasks=fd_statistics(core),
                by_task={task: fd_statistics([r for r in core if r['task'] == task]) for task in TASKS if task != 'pattern'},
                pattern_original=fd_statistics(pattern), pattern_frozen_semantics=fd_statistics(pattern, frozen=True))
    return result


def panel(out, history, shrink):
    pastel = [r for r in history['f68']['rows'] if r['group'] == 'pastel' and r['arm'] == 'a100']
    assert len(pastel) == 4
    board = Image.new('RGB', (6 * 160, 4 * 252), 'white')
    draw = ImageDraw.Draw(board)
    for j, row in enumerate(pastel):
        files = [out / 'cases' / row['case'] / (row['condition'] + '_a100_positive.png'), out / row['image']]
        files += [out / shrink[direction, scale][key(row)]['image']
                  for direction, scale in (('C0', .25), ('C0', .125), ('C1', .25), ('C1', .125))]
        labels = ('target', 'f68', 'C0 lambda=.25', 'C0 lambda=.125', 'C1 lambda=.25', 'C1 lambda=.125')
        for i, (label, path) in enumerate(zip(labels, files)):
            x, y = i * 160, j * 252
            with Image.open(path) as source:
                board.paste(source.convert('RGB').resize((160, 213)), (x, y + 36))
            draw.text((x + 2, y + 2), row['case'].replace('validation_', '') + '/' + row['condition'], fill='black')
            draw.text((x + 2, y + 18), label, fill='black')
    path = out / 'pastel_a100_comparison.png'
    board.save(path)
    return path.name


def package(out, diagnostic, history, cases, comparison):
    files = {out / 'protocol.json', out / 'diagnostic_protocol.json', out / 'result.json', out / comparison}
    files.update(out / ('case_%d.json' % i) for i in range(4))
    files.update(out / 'history' / (model + '.json') for model in ('f68',) + DIRECTIONS)
    files.add(out / 'fixed_proposals.json')
    image_rows = [row for model in history.values() for row in model['rows']]
    image_rows += [row for case in cases for row in case['rows']]
    for row in image_rows:
        path = out / row['image']
        path.resolve().relative_to(out.resolve())
        assert sha(path) == row['png_sha256']
        files.add(path)
    for item in diagnostic['selected']:
        folder = out / 'cases' / item['case']
        names = ['mask.png', 'train_anchor.png', item['condition'] + '_positive.png',
                 item['sample_condition'] + '_goal.png', item['sample_condition'] + '_positive.png']
        files.update(folder / name for name in names)
    ordered = sorted(files, key=lambda path: path.relative_to(out).as_posix())
    manifest = {path.relative_to(out).as_posix(): sha(path) for path in ordered}
    write(out / 'manifest.json', manifest)
    archive = out / 'review.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in ordered + [out / 'manifest.json']:
            bundle.write(path, path.relative_to(out).as_posix())
    return dict(files=len(manifest), bytes=archive.stat().st_size, sha256=sha(archive))


def summarize(out):
    spec, diagnostic = read(out / 'protocol.json'), read(out / 'diagnostic_protocol.json')
    protocol_sha, diagnostic_sha = sha(out / 'protocol.json'), sha(out / 'diagnostic_protocol.json')
    assert diagnostic['shrink_scales'] == [.25, .125] and diagnostic['difference_steps'] == [.125, .03125]
    selected = {key(row) for row in diagnostic['selected']}
    assert len(selected) == len(diagnostic['selected']) == 16 and len(diagnostic['sentinels']) == 8
    history = {model: read(out / 'history' / (model + '.json')) for model in ('f68',) + DIRECTIONS}
    for report in history.values():
        assert len(report['rows']) == 16 and {key(row) for row in report['rows']} == selected
    baselines = {key(row): row for row in history['f68']['rows']}
    cases = [read(out / ('case_%d.json' % i)) for i in range(4)]
    assert {r['case'] for r in cases} == {p['case_id'] for p in spec['train']}
    assert len({r['net_hash'] for r in cases}) == 1
    assert all(r['b_hash'] == diagnostic['initial_b_hash'] for r in cases)
    for case in cases:
        assert case['protocol_sha256'] == protocol_sha and case['diagnostic_protocol_sha256'] == diagnostic_sha
        assert case['frozen_unchanged'] and case['b_restored'] and case['optimizer_updates'] == 0
        assert case['new_reverse_passes'] == 14
        assert len(case['controls']) == 7 and len(case['fresh_gradient_checks']) == 2
        assert Counter(r['kind'] for r in case['controls']) == dict(fresh_f68=4, repeat_f68=1, saved_endpoint=2)
        for control in case['controls']:
            assert control['png_equal']
            if control['kind'] == 'repeat_f68':
                assert max(control['task_noise'].values()) == 0
            else:
                assert control['task_max_abs_difference'] <= 1e-6
        for check in case['fresh_gradient_checks']:
            assert check['png_equal'] and check['latent_max_abs_difference'] <= 2e-4
            assert check['task_max_abs_difference'] <= 1e-6 and set(check['tasks']) == set(TASKS)
    rows = [row for case in cases for row in case['rows']]
    assert len(rows) == len({row['image'] for row in rows}) == 140
    roles = dict(Counter(row['role'] for row in rows))
    assert roles == dict(baseline=16, repeat=4, endpoint_control=8, shrink=64, difference=48), roles
    assert {key(row) for row in rows if row['role'] == 'baseline'} == selected
    shrink = {}
    for direction in DIRECTIONS:
        for scale in diagnostic['shrink_scales']:
            current = [row for row in rows if row['direction'] == direction and row['scale'] == scale and row['role'] == 'shrink']
            assert len(current) == 16 and {key(row) for row in current} == selected
            shrink[direction, scale] = {key(row): row for row in current}
    finite = [row for case in cases for row in case['finite_differences']]
    fd_keys = {(key(r), r['direction'], r['h'], r['task']) for r in finite}
    expected = {(key(row), direction, h, task) for row in diagnostic['sentinels']
                for direction in DIRECTIONS for h in diagnostic['difference_steps'] for task in TASKS}
    assert len(finite) == len(fd_keys) == 224 and fd_keys == expected
    summaries = {}
    for direction in DIRECTIONS:
        summaries[direction] = {}
        for scale in SCALES:
            current = (history['f68']['rows'] if scale == 0 else history[direction]['rows'] if scale == 1
                       else list(shrink[direction, scale].values()))
            summaries[direction][scale_key(scale)] = dict(subset=subset_summary(current, baselines),
                                                         pastel=pastel_summary(current, baselines))
    comparison = panel(out, history, shrink)
    result = dict(protocol_sha256=protocol_sha, diagnostic_protocol_sha256=diagnostic_sha,
        execution_commit=diagnostic['execution_commit'], summary_source_sha256=sha(Path(__file__)),
        diagnostic_protocol=diagnostic, rows=rows, summaries=summaries,
        finite_difference_summary=finite_difference_summary(finite, diagnostic['difference_steps']),
        finite_differences=finite, fresh_gradient_checks=[r for case in cases for r in case['fresh_gradient_checks']],
        controls=[r for case in cases for r in case['controls']],
        counts=dict(selected_conditions=16, pastel_conditions=4, reference_arms_per_pastel=3, sentinels=8,
                    new_images=140, historical_images=48, roles=roles, finite_difference_records=224,
                    new_reverse_passes=sum(case['new_reverse_passes'] for case in cases), optimizer_updates=0),
        case_seconds={case['case']: case['seconds'] for case in cases}, frozen_unchanged=True, b_restored=True,
        adopted_checkpoints=[], holdout_evaluated=False, full_eligibility_evaluated=False,
        comparison_images=[comparison],
        interpretation='固定方向的诊断，不进行连续微调或候选采用；16项子集不能证明48项回放或holdout门槛通过',
        archive='review.zip', manifest='manifest.json')
    write(out / 'result.json', result)
    archive = package(out, diagnostic, history, cases, comparison)
    print('DIRECTION_SUMMARY_COMPLETE', json.dumps(dict(counts=result['counts'], archive=archive)), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    summarize(args.out.resolve())


if __name__ == '__main__':
    main()
