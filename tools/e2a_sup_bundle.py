"""打包新实验的可下载证据；统计产物齐全不代表人工审查或协议已完成。"""
import csv
import math
import tarfile
from pathlib import Path
from tools.e2a_sup_common import *

ALL_ARMS = ['B', 'C'] + ARMS
TEXT = {'.json', '.csv', '.log', '.txt'}


def csv_rows(path):
    with Path(path).open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def identities(rows, count):
    values = [row['id'] for row in rows]
    assert len(values) == len(set(values)) == count
    return set(values)


def image_metadata(directory, row, reference='reference'):
    path = directory / 'images' / (row['id'] + '.png')
    metadata = read(path.with_suffix('.json'))
    assert metadata['id'] == row['id']
    assert metadata['explicit_latent_verified'] and metadata['gt_used_in_inference'] is False
    assert sha(path) == metadata['output_sha256']
    assert metadata['inputs'] == {key: sha(row[reference if key == 'reference' else key])
                                  for key in ['gt', 'sketch', 'reference']}
    return metadata


def validate():
    from tools.e2a_sup_reference import approved_reference_manifest
    reference_manifest, swaps = approved_reference_manifest()
    rows = read(ROOT / 'splits/P2b.json')
    ids = identities(rows, 96)
    assert identities(swaps, 32) <= ids
    assert read(ROOT / 'audit/preparation_complete.json')['passed']
    assert read(ROOT / 'calibration_choice.json')['gradient_target'] == .10
    for name, digest in read(ROOT / 'audit/historical_source_sha256.json').items():
        assert sha(SOURCE / name) == digest
    for name, digest in read(ROOT / 'evaluation/metric_protocol.json')['evaluation_code'].items():
        assert sha(SOURCE / name) == digest
    decision = read(ROOT / 'statistics/P2b/decision.json')
    assert set(decision['comparisons']) == set(ARMS) and not decision['LOCK_opened']
    state = read(ROOT / 'status.json')
    assert state['stage'] == 'STATISTICS_COMPLETE_QUALITATIVE_REVIEW_PENDING'
    assert not state['complete'] and not state.get('engineering_failure')
    baseline = csv_rows(OLD / 'training/seed42/F000/training_log.csv')
    assert len(baseline) == CONFIG['updates']
    streams = read(OLD / 'training/seed42/F000/seed_and_rng_manifest.json')
    models = {}
    for arm in ALL_ARMS:
        run = model_path(arm).parent.parent
        count = read(run / 'optimizer_update_count.json')
        check = read(run / 'checkpoint_verification.json')
        assert check['passed'] and read(run / 'training_complete.json')['complete']
        assert count['actual_successful_updates'] == CONFIG['updates'] and count['skipped_updates'] == 0
        trace = csv_rows(run / 'training_log.csv')
        assert [int(row['update']) for row in trace] == list(range(1, CONFIG['updates'] + 1))
        assert all(a[key] == b[key] for a, b in zip(trace, baseline)
                   for key in ['update', 'lr', 'microbatches_per_rank', 'rng_rank0'])
        assert read(run / 'seed_and_rng_manifest.json') == streams
        assert len(streams['streams']) == 2
        digest = sha(model_path(arm))
        assert digest == check['sha256']
        if arm in EXPECTED:
            assert digest == EXPECTED[arm]
        else:
            config = read(run / 'config.json')
            assert config['world_size'] == 2 and config['updates'] == CONFIG['updates']
            assert all(sha(SOURCE / name) == value for name, value in config['additions'].items())
            cal = read(folder(arm, 'calibrate') / 'calibration_complete.json')
            assert cal['passed'] and cal['samples'] == 512 and cal['effective_batches'] == 64
            assert cal['optimizer_updates'] == 0
            assert read(ROOT / 'audit' / ('pilot_release_' + arm + '.json'))['passed']
        models[arm] = dict(path=str(model_path(arm)), sha256=digest, actual_updates=CONFIG['updates'],
                           skipped_updates=0, full_trace_equal_B=True, both_rank_final_streams_equal_B=True)
    assert sha(model_path('E1')) == EXPECTED['E1']
    paired = {}
    for arm in ALL_ARMS:
        base = ROOT / 'evaluation/P2b' / arm
        swapped = ROOT / 'evaluation/reference_swap' / arm
        for directory, n in [(base, 96), (swapped, 32)]:
            for name in ['generation_complete.json', 'evaluation_complete.json']:
                complete = read(directory / name)
                assert complete['complete'] and complete['n'] == n and complete['arm'] == arm
        assert identities(read(base / 'per_identity_metrics.json'), 96) == ids
        assert identities(read(swapped / 'per_identity_swap_metrics.json'), 32) == identities(swaps, 32)
        evaluation = read(swapped / 'evaluation_complete.json')
        assert evaluation['reference_manifest_sha256'] == sha(reference_manifest)
        assert evaluation['old_GT_LPIPS_not_computed'] and evaluation['matched_reference_condition_count'] == 2
        for row in rows:
            metadata = image_metadata(base, row)
            assert Path(metadata['checkpoint']) == model_path(arm)
            values = {key: metadata[key] for key in ['seed', 'initial_latent_sha256', 'inputs']}
            if row['id'] in paired:
                assert values == paired[row['id']]
            paired[row['id']] = values
        for row in swaps:
            original = read(base / 'images' / (row['id'] + '.json'))
            alternative = image_metadata(swapped, row, 'reference_alternative')
            assert Path(alternative['checkpoint']) == model_path(arm)
            assert all(original[key] == alternative[key] for key in ['seed', 'initial_latent_sha256'])
            assert original['inputs']['reference'] != alternative['inputs']['reference']
    return models, rows, swaps


def build():
    models, rows, swaps = validate()
    # 只收录明确目录；不打开新 LOCK 清单，不携带训练状态/巨大权重或缓存。
    members = {}
    def add(path, name):
        path = Path(path)
        assert path.is_file() and not path.is_symlink(), str(path)
        name = Path(name).as_posix()
        assert name not in members or members[name] == path
        members[name] = path
    for name in ['config.json', 'checkpoint_registry.json', 'checkpoint_manifest.json', 'calibration_choice.json',
                 'loss_registry.json', 'jobs.json', 'status.json', 'resource_ledger.csv']:
        add(ROOT / name, name)
    for directory in ['audit', 'statistics', 'training', 'superseded', 'logs', 'evaluation', 'masks/evaluation']:
        for path in sorted((ROOT / directory).rglob('*')):
            if path.is_file() and not path.is_symlink() and path.suffix.lower() in TEXT | {'.png'}:
                add(path, path.relative_to(ROOT))
    for name in ['P2a.json', 'P2b.json', 'reference_swap.json', 'reference_swap_v2.json']:
        add(ROOT / 'splits' / name, 'splits/' + name)
    # 旧 B/C 只携带其完成证据，旧实验其它臂和旧指标不混入新结果。
    for arm in ['B', 'C']:
        run = model_path(arm).parent.parent
        for name in ['config.json', 'optimizer_update_count.json', 'seed_and_rng_manifest.json',
                     'checkpoint_verification.json', 'training_complete.json', 'training_log.csv']:
            add(run / name, 'reused_training/' + arm + '/' + name)
    add(OLD / 'mask_audit.csv', 'reused_training/mask_audit.csv')
    inputs = {}
    for row in read(ROOT / 'splits/P2a.json') + rows + swaps:
        for key in ['gt', 'sketch', 'reference', 'reference_alternative']:
            if key not in row:
                continue
            path = Path(row[key]); digest = sha(path)
            name = 'inputs/' + digest + path.suffix.lower()
            if name not in members:
                add(path, name)
            inputs[str(path)] = dict(member=name, sha256=digest)
    sources = set(read(ROOT / 'audit/historical_source_sha256.json'))
    sources.update(path.relative_to(SOURCE).as_posix() for path in (SOURCE / 'tools').glob('e2a_*.py'))
    sources.update(read(ROOT / 'evaluation/metric_protocol.json')['evaluation_code'])
    sources.add('scripts/e2a_sup.sbatch')
    for name in sorted(sources):
        add(SOURCE / name, 'source/' + name)
    assets = read(ROOT / 'assets/registry.json')
    add(ROOT / 'assets/registry.json', 'assets/registry.json')
    for arm in ['T1', 'T2']:
        for path, digest in assets[arm]['files'].items():
            path = Path(path)
            assert sha(path) == digest
            if path.suffix == '.py' or 'license' in path.name.lower():
                add(path, path.relative_to(ROOT))
    ledger = csv_rows(ROOT / 'resource_ledger.csv')
    jobs = read(ROOT / 'jobs.json')
    assert {str(job['id']) for job in jobs} <= {row['JobIDRaw'] for row in ledger}
    gpu_ids = {str(job['id']) for job in jobs if job['gpus']}
    gpu_rows = [row for row in ledger if row['JobIDRaw'] in gpu_ids]
    assert all(row['State'].split()[0] in ['COMPLETED', 'FAILED', 'OUT_OF_MEMORY', 'TIMEOUT', 'CANCELLED', 'NODE_FAIL']
               for row in gpu_rows)
    assert all(math.isfinite(float(row['gpu_hours'])) and float(row['gpu_hours']) >= 0 for row in ledger)
    manifest = dict(protocol=CONFIG['protocol'], plan_sha256=PLAN_SHA, source_commit=commit(),
        protocol_complete=False, stage='STATISTICS_COMPLETE_QUALITATIVE_REVIEW_PENDING',
        pending=['human blind scores', 'swap orientation/scale review', 'tail review', 'local report'],
        human_blind_scores=None, LOCK_opened=False, model_weights_included=False,
        optimizer_and_RNG_binary_states_included=False, explicit_latent_binary_included=False,
        latent_reproducibility='per-image seed and initial_latent_sha256; no saved latent tensor',
        checkpoints=models, E1_sha256=EXPECTED['E1'], portable_inputs=inputs,
        resource_gpu_hours=sum(float(row['gpu_hours']) for row in ledger),
        resource_scope='actual Slurm allocation time including failures, calibration, pilot and pauses; current CPU advance may be running',
        files={name: dict(sha256=sha(path), bytes=path.stat().st_size) for name, path in sorted(members.items())})
    write(ROOT / 'review_bundle_manifest.json', manifest)
    add(ROOT / 'review_bundle_manifest.json', 'review_bundle_manifest.json')
    expected = {name: record['sha256'] for name, record in manifest['files'].items()}
    expected['review_bundle_manifest.json'] = sha(ROOT / 'review_bundle_manifest.json')
    target = ROOT / 'review_results.tar.gz'; temporary = target.with_suffix('.gz.tmp')
    with tarfile.open(temporary, 'w:gz') as archive:
        for name, path in sorted(members.items()):
            archive.add(path, arcname=name, recursive=False)
    # 发布前重新流式核每个成员，文件缺失/损坏时不替换已发布包。
    with tarfile.open(temporary, 'r:gz') as archive:
        assert set(archive.getnames()) == set(members)
        for member in archive:
            digest = hashlib.sha256()
            with archive.extractfile(member) as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                    digest.update(block)
            assert digest.hexdigest() == expected[member.name], member.name
    temporary.replace(target)
    digest = sha(target)
    target.with_name(target.name + '.sha256').write_text(digest + '  ' + target.name + '\n', encoding='ascii')
    print('REVIEW_BUNDLE_READY', digest, 'QUALITATIVE_REVIEW_PENDING', flush=True)
    return dict(path=str(target), sha256=digest, protocol_complete=False)


if __name__ == '__main__':
    build()
