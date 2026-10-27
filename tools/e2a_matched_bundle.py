"""第一阶段可下载证据包；只读训练和评价，保留开发审查及人工评分缺项。"""
import csv
import hashlib
import math
import tarfile
from pathlib import Path
from tools.e2a_matched_common import *
from tools.e2a_sup_bundle import csv_rows, identities, image_metadata

ARMS = ['B', 'C', 'T3', 'T3M']
KEYS = ['edge_f1', 'leak', 'iou', 'edge_l1', 'lpips', 'tcf_lab']
TEXT = {'.json', '.csv', '.log', '.txt', '.png'}
TERMINAL = {'COMPLETED', 'FAILED', 'OUT_OF_MEMORY', 'TIMEOUT', 'CANCELLED', 'NODE_FAIL'}


def validate():
    from tools.e2a_sup_reference import approved_reference_manifest
    manifest, swaps = approved_reference_manifest()
    rows = read(ROOT/'splits/P2b.json')
    ids = identities(rows, 96)
    assert identities(swaps, 32) <= ids
    assert read(ROOT/'splits/reference_swap.json') == swaps
    state = read(ROOT/'status.json')
    assert state['stage'] == 'P2B_DEVELOPMENT_REVIEW_REQUIRED'
    assert not state['complete'] and not state.get('engineering_failure')
    decision = read(ROOT/'statistics/P2b/decision.json')
    assert decision['primary_family'] == CONFIG['primary_family']
    assert not decision['LOCK_opened'] and not decision['seed43_started']
    assert read(ROOT/'audit/pilot_release_T3M.json')['passed']
    prep = read(ROOT/'audit/preparation_complete.json')
    assert prep['passed'] and all(sha(SOURCE/p) == h for p, h in prep['source_sha256'].items())
    for p, h in read(ROOT/'audit/historical_source_sha256.json').items():
        assert sha(SOURCE/p) == h
    for p, h in read(ROOT/'audit/reused_metric_protocol.json')['evaluation_code'].items():
        assert sha(SOURCE/p) == h
    cal = read(folder('T3M', 'calibrate')/'calibration_complete.json')
    assert cal['passed'] and cal['samples'] == 512 and cal['effective_batches'] == 64
    assert cal['optimizer_updates'] == 0
    gamma = read(ROOT/'gamma_calibration.json')
    assert not gamma['used_P2_or_LOCK']
    assert read(ROOT/'loss_registry.json')['T3M']['coefficients'] == {'body_rgb': .1, 'rest': gamma['gamma']}
    baseline = csv_rows(OLD/'training/seed42/F000/training_log.csv')
    streams = read(OLD/'training/seed42/F000/seed_and_rng_manifest.json')
    models, paired = {}, {}
    for arm in ARMS:
        run = model_path(arm).parent.parent
        count = read(run/'optimizer_update_count.json')
        check = read(run/'checkpoint_verification.json')
        assert check['passed'] and read(run/'training_complete.json')['complete']
        assert count['actual_successful_updates'] == 28365 and count['skipped_updates'] == 0
        trace = csv_rows(run/'training_log.csv')
        assert len(trace) == len(baseline) == 28365
        assert [int(r['update']) for r in trace] == list(range(1, 28366))
        assert all(a[k] == b[k] for a, b in zip(trace, baseline)
                   for k in ['update', 'lr', 'microbatches_per_rank', 'rng_rank0'])
        assert read(run/'seed_and_rng_manifest.json') == streams and len(streams['streams']) == 2
        digest = sha(model_path(arm))
        assert digest == check['sha256']
        if arm != 'T3M':
            assert digest == (T3_SHA if arm == 'T3' else INITIAL_SHA[arm])
        else:
            aux = csv_rows(run/'auxiliary_log.csv')
            assert len(aux) == 28365 and not (run/'overflow_failure.json').exists()
            assert all(math.isfinite(float(v)) for r in aux for k, v in r.items() if k != 'job')
        models[arm] = dict(path=str(model_path(arm)), sha256=digest, successful_updates=28365,
                           skipped_updates=0, four_columns_equal_B=True, final_double_rank_manifest_equal_B=True)
        base, swapped = [ROOT/'evaluation'/stage/arm for stage in ['P2b', 'reference_swap']]
        complete = read(base/'evaluation_complete.json')
        assert complete['complete'] and complete['n'] == 96
        assert all(complete['valid_counts'][k] == 96 for k in KEYS)
        metrics = read(base/'per_identity_metrics.json')
        assert identities(metrics, 96) == ids
        assert all(math.isfinite(float(r[k])) for r in metrics for k in KEYS)
        for row in rows:
            metadata = image_metadata(base, row)
            assert Path(metadata['checkpoint']) == model_path(arm)
            assert metadata['seed'] == 42 + int(hashlib.sha256(row['id'].encode()).hexdigest()[:7], 16)
            values = {k: metadata[k] for k in ['seed', 'initial_latent_sha256', 'inputs']}
            assert paired.get(row['id'], values) == values
            paired[row['id']] = values
        complete = read(swapped/'evaluation_complete.json')
        assert complete['complete'] and complete['n'] == 32
        assert complete['reference_manifest_sha256'] == sha(manifest)
        assert complete['matched_reference_condition_count'] == 2 and complete['old_GT_LPIPS_not_computed']
        assert identities(read(swapped/'per_identity_swap_metrics.json'), 32) == identities(swaps, 32)
        for row in swaps:
            first = read(base/'images'/(row['id']+'.json'))
            second = image_metadata(swapped, row, 'reference_alternative')
            assert Path(second['checkpoint']) == model_path(arm)
            assert all(first[k] == second[k] for k in ['seed', 'initial_latent_sha256'])
            assert first['inputs']['reference'] != second['inputs']['reference']
    assert sha(model_path('E1')) == INITIAL_SHA['E1']
    return models, rows, swaps


def build():
    from tools.e2a_matched_dispatch import accounting
    accounting()
    models, rows, swaps = validate()
    ledger = csv_rows(ROOT/'resource_ledger.csv')
    jobs = read(ROOT/'jobs.json')
    assert {str(j['id']) for j in jobs} <= {r['JobIDRaw'] for r in ledger}
    gpu_ids = {str(j['id']) for j in jobs if j['gpus']}
    assert all(r['State'].split()[0] in TERMINAL for r in ledger if r['JobIDRaw'] in gpu_ids)
    assert all(math.isfinite(float(r['gpu_hours'])) and float(r['gpu_hours']) >= 0 for r in ledger)
    target = ROOT/'development_review.tar.gz'
    assert not target.exists(), 'Preserve an existing published bundle; do not overwrite'
    members = {}

    def add(path, name):
        path, name = Path(path), Path(name).as_posix()
        assert path.is_file() and not path.is_symlink()
        assert 'LOCK' not in name and not Path(name).is_absolute() and '..' not in Path(name).parts
        assert name not in members or members[name] == path
        members[name] = path

    def tree(path, prefix):
        for p in sorted(path.rglob('*')):
            if p.is_file() and not p.is_symlink() and p.suffix.lower() in TEXT and 'LOCK' not in p.name:
                add(p, str(Path(prefix)/p.relative_to(path)))

    for name in ['config.json', 'loss_definition.json', 'checkpoint_registry.json', 'checkpoint_manifest.json',
                 'gamma_calibration.json', 'loss_registry.json', 'jobs.json', 'status.json', 'resource_ledger.csv']:
        add(ROOT/name, name)
    for directory in ['audit', 'statistics', 'training', 'engineering', 'logs', 'masks/evaluation']:
        tree(ROOT/directory, directory)
    for stage in ['P2b', 'reference_swap']:
        for arm in ARMS:
            tree(ROOT/'evaluation'/stage/arm, 'evaluation/'+stage+'/'+arm)
    for stage in ['P2a', 'pilot_P2a']:
        tree(ROOT/'evaluation'/stage/'T3M', 'evaluation/'+stage+'/T3M')
    for arm in ['B', 'C', 'T3']:
        run = model_path(arm).parent.parent
        for name in ['config.json', 'optimizer_update_count.json', 'seed_and_rng_manifest.json',
                     'checkpoint_verification.json', 'training_complete.json', 'training_log.csv']:
            add(run/name, 'reused_training/'+arm+'/'+name)
    add(OLD/'mask_audit.csv', 'reused_training/mask_audit.csv')
    for name in ['P2a.json', 'P2b.json', 'reference_swap.json']:
        add(ROOT/'splits'/name, 'splits/'+name)
    for name in ['splits/reference_swap.json', 'splits/reference_swap_v2.json',
                 'audit/reference_swap_amendment_v2.json', 'audit/reference_swap_release.json']:
        add(SUP/name, 'reused_reference/'+name)
    inputs = {}
    for row in read(ROOT/'splits/P2a.json') + rows + swaps:
        for key in ['gt', 'sketch', 'reference', 'reference_alternative']:
            if key not in row:
                continue
            path = Path(row[key]); digest = sha(path)
            name = 'inputs/'+digest+path.suffix.lower()
            if name not in members:
                add(path, name)
            inputs[str(path)] = dict(member=name, sha256=digest)
    sources = set(read(ROOT/'audit/historical_source_sha256.json'))
    sources.update(p.relative_to(SOURCE).as_posix() for p in (SOURCE/'tools').glob('e2a_*.py'))
    sources.update(read(ROOT/'audit/reused_metric_protocol.json')['evaluation_code'])
    sources.update(['scripts/e2a_matched.sbatch', 'scripts/e2a_sup.sbatch'])
    for name in sorted(sources):
        add(SOURCE/name, 'source/'+name)
    manifest = dict(protocol=CONFIG['protocol'], plan_sha256=PLAN_SHA, source_commit=commit(),
        training_source_commit=read(ROOT/'audit/formal_start_120479.json')['source_commit'],
        protocol_complete=False, stage='P2B_DEVELOPMENT_REVIEW_REQUIRED', LOCK_opened=False,
        human_blind_scores=None, seed43_started=False, gamma=read(ROOT/'gamma_calibration.json')['gamma'],
        pending=['independent local statistics', 'AI texture and tail review', 'development decision', 'local report'],
        checkpoints=models, portable_inputs=inputs, reused_arms=['B', 'C', 'T3'], new_arm='T3M',
        model_weights_included=False, optimizer_and_RNG_binary_states_included=False,
        explicit_latent_binary_included=False, latent_reproducibility='seed and initial latent SHA only; no saved tensor',
        resource_gpu_hours=sum(float(r['gpu_hours']) for r in ledger),
        resource_scope='new protocol allocations including failures; old reused protocol costs reported separately',
        files={name: dict(sha256=sha(p), bytes=p.stat().st_size) for name, p in sorted(members.items())})
    manifest_path = ROOT/'development_review_manifest.json'
    write(manifest_path, manifest)
    add(manifest_path, manifest_path.name)
    expected = {name: r['sha256'] for name, r in manifest['files'].items()}
    expected[manifest_path.name] = sha(manifest_path)
    temporary = target.with_suffix('.gz.tmp')
    with tarfile.open(temporary, 'w:gz') as archive:
        for name, path in sorted(members.items()):
            archive.add(path, arcname=name, recursive=False)
    with tarfile.open(temporary, 'r:gz') as archive:
        assert len(archive.getnames()) == len(members) and set(archive.getnames()) == set(members)
        for member in archive:
            assert member.isfile()
            h = hashlib.sha256()
            with archive.extractfile(member) as handle:
                for block in iter(lambda: handle.read(8*1024*1024), b''):
                    h.update(block)
            assert h.hexdigest() == expected[member.name], member.name
    temporary.replace(target)
    digest = sha(target)
    target.with_name(target.name+'.sha256').write_text(digest+'  '+target.name+'\n', encoding='ascii')
    print('DEVELOPMENT_REVIEW_READY', digest, target.stat().st_size, len(members), flush=True)


if __name__ == '__main__':
    build()
