"""CPU 汇总一次 f68/C0/C1 对照；只判断候选资格，不改权重或指针。"""
import argparse
import hashlib
import json
import statistics
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw


MODELS = ('f68', 'C0', 'C1')
ARMS = ('a075', 'a100', 'a125')
TOLERANCE = 1e-6
PROTECTION = ('palette_swd', 'spatial_ab_mae_goal', 'low_l', 'old_projection',
              'old_l_projection', 'edge', 'l_edge', 'l_amplitude_error', 'l_correlation')
METRICS = ('l_amplitude', 'l_amplitude_error', 'l_correlation', 'l_target_gain',
           'l_target_gain_error', 'l_hp_mae', 'palette_swd', 'spatial_ab_mae_goal',
           'spatial_ab_mae_positive', 'low_l', 'low_l_positive', 'old_projection',
           'old_projection_channel_abs_sum', 'old_l_projection', 'edge', 'l_edge')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def mean(values):
    return statistics.mean(values) if values else None


def row_key(row, split):
    return row['case'], row['condition'], row['arm'] if split == 'replay' else row['seed_index']


def metric_change(before, after, key):
    """原值差与好坏分开；幅度/增益按离1的距离，不能简单认为越大越好。"""
    assert len(before) == len(after)
    valid = [(a, b) for a, b in zip(before, after) if a is not None and b is not None]
    delta = [b - a for a, b in valid]
    if key in ('l_amplitude', 'l_target_gain'):
        error = [abs(b - 1) - abs(a - 1) for a, b in valid]
        criterion = 'absolute_distance_to_1'
    elif key == 'l_correlation':
        error = [-d for d in delta]
        criterion = 'higher_is_better'
    else:
        error, criterion = delta, 'lower_is_better'
    return dict(count=len(valid), before_mean=mean([a for a, _ in valid]),
                after_mean=mean([b for _, b in valid]), mean_delta=mean(delta),
                improvement_criterion=criterion, mean_error_delta=mean(error),
                improved=sum(e < -TOLERANCE for e in error),
                worsened=sum(e > TOLERANCE for e in error),
                unchanged=sum(abs(e) <= TOLERANCE for e in error), tolerance=TOLERANCE)


def paired_changes(baseline, candidate, split):
    by_key = {row_key(row, split): row for row in baseline['rows']}
    assert len(by_key) == len(baseline['rows']) == len(candidate['rows'])
    assert set(by_key) == {row_key(row, split) for row in candidate['rows']}
    for row in candidate['rows']:
        old = by_key[row_key(row, split)]
        assert (row['seed'], row['noise_digest'], row['group']) == (old['seed'], old['noise_digest'], old['group'])
    result = {}
    for group in ['all'] + sorted({row['group'] for row in candidate['rows']}):
        selected = [r for r in candidate['rows'] if group == 'all' or r['group'] == group]
        arms = ARMS if split == 'replay' else ('all',)
        groups = {}
        for arm in arms:
            current = [r for r in selected if split != 'replay' or r['arm'] == arm]
            prior = [by_key[row_key(row, split)] for row in current]
            metrics = {key: metric_change([r['scores'][key] for r in prior],
                                           [r['scores'][key] for r in current], key)
                       for key in METRICS}
            groups[arm] = dict(count=len(current), metrics=metrics,
                amplitude_pass_before=sum(r['scores']['amplitude_pass'] for r in prior),
                amplitude_pass_after=sum(r['scores']['amplitude_pass'] for r in current),
                quality_pass_before=sum(r['scores']['l_quality_pass'] for r in prior),
                quality_pass_after=sum(r['scores']['l_quality_pass'] for r in current))
        result[group] = groups if split == 'replay' else groups['all']
    return result


def response_groups(report):
    result = {}
    for group in ['all'] + sorted({r['group'] for r in report['responses']}):
        rows = [r for r in report['responses'] if group == 'all' or r['group'] == group]
        metrics = {}
        for name in ('low_slope', 'high_slope', 'center_slope'):
            slopes = [r['response'][name] for r in rows if r['response'][name] is not None]
            metrics[name] = dict(count=len(slopes), mean=mean(slopes),
                mean_absolute_error=mean([abs(s - 1) for s in slopes]),
                median=statistics.median(slopes) if slopes else None,
                minimum=min(slopes) if slopes else None, maximum=max(slopes) if slopes else None)
        result[group] = dict(count=len(rows), slopes=metrics,
            target_monotonic=sum(r['response']['target_monotonic'] is True for r in rows),
            output_monotonic=sum(r['response']['output_monotonic'] is True for r in rows))
    return result


def response_changes(before, after):
    original = {(r['case'], r['condition']): r for r in before['responses']}
    result = {}
    for group in ['all'] + sorted({r['group'] for r in after['responses']}):
        rows = [r for r in after['responses'] if group == 'all' or r['group'] == group]
        result[group] = {}
        for name in ('low_slope', 'high_slope', 'center_slope'):
            # 斜率与gain一样，以距离1评价。
            value = metric_change([original[r['case'], r['condition']]['response'][name] for r in rows],
                                  [r['response'][name] for r in rows], 'l_target_gain')
            result[group][name] = value
    return result


def holdout_gate(before, after):
    checks = []
    assert set(before['groups']) == set(after['groups'])
    for group in sorted(k for k in after['groups'] if k != 'all'):
        a, b = before['groups'][group], after['groups'][group]
        assert a['count'] == b['count'] > 0
        for key in PROTECTION:
            previous, current = a['metrics'][key], b['metrics'][key]
            delta = None if previous is None or current is None else current - previous
            passed = delta is not None and (-delta if key == 'l_correlation' else delta) <= 1e-5
            checks.append(dict(group=group, metric=key, delta=delta, tolerance=1e-5, passed=passed))
    return dict(passed=all(row['passed'] for row in checks) and bool(checks), checks=checks,
                scope='11个已使用过的训练隔离身份；按分层保护静态alpha1，不检查响应斜率泛化')


def eligibility(proposal, replay, holdout, base_replay, base_holdout):
    source = {row_key(r, 'replay'): r for r in base_replay['rows']}
    residue_delta = mean([r['scores']['old_projection'] - source[row_key(r, 'replay')]['scores']['old_projection']
                          for r in replay['rows']])
    regression = holdout_gate(base_holdout, holdout)
    minimum = proposal['minimum_cosine']
    checks = dict(projection_feasible=bool(proposal['projection']['feasible']),
        nonzero_parameter_update=proposal['parameter_updates'] > 0,
        rounded_displacement_protected=minimum is not None and minimum >= -1e-5,
        training_tasks_protected=bool(replay['training_task_protection']['passed']),
        strict_replay_pixels_protected=bool(replay['strict_replay_protection']['passed']),
        mean_residue_strictly_decreased=residue_delta < -1e-8,
        holdout_strata_protected=regression['passed'])
    return dict(eligible=all(checks.values()), adopted=False, checks=checks,
                failed_checks=[key for key, passed in checks.items() if not passed],
                mean_replay_old_projection_delta=residue_delta, holdout_protection=regression,
                purpose='仅报告固定候选资格，不改变模型指针或启动后续训练')


def innovation(reports, eligibility_report):
    candidate = reports['C1']['replay']
    checks = {}
    for reference in ('f68', 'C0'):
        baseline = reports[reference]['replay']
        paired = response_changes(baseline, candidate)['pastel']['center_slope']
        gains = paired_changes(baseline, candidate, 'replay')['pastel']['a100']['metrics']
        slope_delta = paired['mean_error_delta']
        gain_delta = gains['l_target_gain']['mean_error_delta']
        correlation_delta = gains['l_correlation']['mean_delta']
        valid = all(value is not None for value in (slope_delta, gain_delta, correlation_delta))
        passed = valid and slope_delta < -TOLERANCE and gain_delta < -TOLERANCE and correlation_delta >= -1e-5
        checks[reference] = dict(passed=passed, slope_mean_absolute_error_delta=slope_delta,
            base_gain_mean_absolute_error_delta=gain_delta, base_correlation_delta=correlation_delta)
    improved = all(value['passed'] for value in checks.values())
    return dict(versus=checks, paired_and_absolute_improvement=improved,
                C1_eligible=eligibility_report['C1']['eligible'],
                effective=improved and eligibility_report['C1']['eligible'],
                improvement_tolerance=TOLERANCE, correlation_tolerance=1e-5,
                scope='组合策略的一次候选对照；不能拆分归因，也不等于保留集响应泛化')


def pastel_rows(reports):
    result = []
    for key in [(r['case'], r['condition']) for r in reports['f68']['replay']['responses'] if r['group'] == 'pastel']:
        models = {}
        for model in MODELS:
            report = reports[model]['replay']
            response = next(r['response'] for r in report['responses'] if (r['case'], r['condition']) == key)
            rows = [r for r in report['rows'] if (r['case'], r['condition']) == key]
            models[model] = dict(response=response, arms={r['arm']: dict(
                gain=r['scores']['l_target_gain'], amplitude=r['scores']['l_amplitude'],
                correlation=r['scores']['l_correlation'], projection=r['projection']) for r in rows})
        result.append(dict(case=key[0], condition=key[1], models=models))
    assert len(result) == 4
    return result


def comparisons(out, spec, reports):
    paths = []
    by_model = {model: {row_key(r, 'replay'): r for r in reports[model]['replay']['rows']} for model in MODELS}
    for pair in spec['train']:
        case = pair['case_id']
        conditions = [r['condition'] for r in spec['replay_bank'] if r['case'] == case]
        assert len(conditions) == 4
        board = Image.new('RGB', (4 * 160, 4 * 252), 'white')
        draw = ImageDraw.Draw(board)
        for j, condition in enumerate(conditions):
            files = [out / 'cases' / case / (condition + '_a100_positive.png')]
            files += [out / by_model[model][case, condition, 'a100']['image'] for model in MODELS]
            for i, (label, path) in enumerate(zip(('target',) + MODELS, files)):
                x, y = i * 160, j * 252
                with Image.open(path) as source:
                    board.paste(source.convert('RGB').resize((160, 213)), (x, y + 36))
                draw.text((x + 2, y + 2), condition, fill='black')
                draw.text((x + 2, y + 18), 'a100 ' + label, fill='black')
        path = out / (case + '_comparison.png')
        board.save(path)
        paths.append(path.relative_to(out).as_posix())
    return paths


def package(out, reports, panels):
    # 显式白名单，避免打包权重、梯度tensor、源码或重复评测素材。
    files = {out / 'protocol.json', out / 'result.json', out / 'training/proposals.json'}
    files.update(out / name for name in ('run_provenance.json', 'rounded_projection.json')
                 if (out / name).exists())
    files.update((out / 'gradients').glob('*.json'))
    files.update(out / name for name in panels)
    files.update(p for p in (out / 'cases').rglob('*') if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg'))
    for model in MODELS:
        for split in ('replay', 'holdout'):
            files.add(out / 'evaluation' / model / split / 'result.json')
            for row in reports[model][split]['rows']:
                path = out / row['image']
                path.resolve().relative_to(out.resolve())
                assert sha(path) == row['png_sha256']
                files.add(path)
    ordered = sorted(files, key=lambda p: p.relative_to(out).as_posix())
    manifest = {p.relative_to(out).as_posix(): sha(p) for p in ordered}
    write(out / 'manifest.json', manifest)
    archive = out / 'review.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in ordered + [out / 'manifest.json']:
            bundle.write(path, path.relative_to(out).as_posix())
    return dict(files=len(manifest), archive=archive.name, bytes=archive.stat().st_size, sha256=sha(archive))


def summarize(out):
    spec, proposals = read(out / 'protocol.json'), read(out / 'training/proposals.json')
    protocol_sha = sha(out / 'protocol.json')
    reports, source_hashes = {}, {}
    for model in MODELS:
        reports[model] = {}
        for split in ('replay', 'holdout'):
            path = out / 'evaluation' / model / split / 'result.json'
            report = read(path)
            assert report['arm'] == model and report['split'] == split
            assert report['protocol_sha256'] == protocol_sha and report['frozen_unchanged']
            assert report['evaluation_optimizer_updates'] == 0 and not report['heldout_used_for_training']
            assert len(report['rows']) == (48 if split == 'replay' else 44)
            reports[model][split] = report
            source_hashes[path.relative_to(out).as_posix()] = sha(path)
    assert len({r['net_hash'] for model in reports.values() for r in model.values()}) == 1
    for model in MODELS:
        assert reports[model]['replay']['checkpoint_sha256'] == reports[model]['holdout']['checkpoint_sha256']
        expected = spec['checkpoint_sha256'] if model == 'f68' else proposals['arms'][model]['checkpoint_sha256']
        assert reports[model]['replay']['checkpoint_sha256'] == expected
    changes = {}
    for candidate, baseline in (('C0', 'f68'), ('C1', 'f68'), ('C1', 'C0')):
        changes[candidate + '_minus_' + baseline] = dict(
            replay=paired_changes(reports[baseline]['replay'], reports[candidate]['replay'], 'replay'),
            holdout=paired_changes(reports[baseline]['holdout'], reports[candidate]['holdout'], 'holdout'),
            response=response_changes(reports[baseline]['replay'], reports[candidate]['replay']))
    gates = {model: eligibility(proposals['arms'][model], reports[model]['replay'], reports[model]['holdout'],
                               reports['f68']['replay'], reports['f68']['holdout']) for model in ('C0', 'C1')}
    panels = comparisons(out, spec, reports)
    result = dict(protocol_sha256=protocol_sha, source_commit=spec['source_commit'],
        evaluation_source_sha256=source_hashes, summary_source_sha256=sha(Path(__file__)),
        proposals=proposals, groups={model: {split: reports[model][split]['groups'] for split in ('replay', 'holdout')}
                                    for model in MODELS},
        response_groups={model: response_groups(reports[model]['replay']) for model in MODELS},
        comparisons=changes, pastel_conditions=pastel_rows(reports), eligibility=gates,
        innovation=innovation(reports, gates), comparison_images=panels,
        frozen_unchanged=True, adopted_checkpoints=[], heldout_used_for_training=False,
        interpretation='固定一次候选更新，不搜索或续训；完整报告失败项；eligible仅为资格，未改模型指针',
        archive='review.zip', manifest='manifest.json')
    if (out / 'run_provenance.json').exists():
        result['execution_provenance'] = read(out / 'run_provenance.json')
        result['rounded_projection'] = read(out / 'rounded_projection.json')
    write(out / 'result.json', result)
    archive = package(out, reports, panels)
    print('PAIR_SUMMARY', json.dumps(dict(eligibility={m: g['eligible'] for m, g in gates.items()},
          innovation=result['innovation'], archive=archive), ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.out.resolve())


if __name__ == '__main__':
    main()
