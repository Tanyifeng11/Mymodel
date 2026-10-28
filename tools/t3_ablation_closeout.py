"""按用户最新范围只收尾两项消融；不改变已冻结训练或原确认结论。"""
import csv
import datetime
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tools import t3_response_extension as ext
from tools.t3_response_common import CONFIG, SUP, SOURCE, read, write, freeze, sha, commit, csv_write
from tools.t3_response_run import statistics, holm
from tools.t3_response_report import place

ROOT = ext.ROOT
OUT = ROOT / 'ablation_closeout'
ARMS = ['T3_no_body_s42', 'T3_no_rest_s42']
KEYS = {'C_s42': ('C', ext.PARENT), 'T3_s42': ('T3', ext.PARENT),
        **{k: (k, ROOT) for k in ARMS}}


def analyse():
    ext.verify_parent()
    freeze(OUT / 'scope.json', dict(
        user_instruction='做完消融实验总结整理完就结束', retained_arms=ARMS,
        cancelled_new_seeds=[43, 44], scope_changed_before_ablation_statistics_inspection=True,
        original_exploratory_plan_preserved=True, no_new_training=True,
        evidence_status='post-confirmation exploratory ablations; one training seed',
        human_review='deferred; no ratings supplied'))
    for name, expected in read(ROOT / 'protocol/training_source_sha256.json').items():
        assert sha(SOURCE / name) == expected, name
    rows = read(ROOT / 'splits/confirmation.json')
    ids = [r['id'] for r in rows]
    lookup = {}
    summaries = []
    failures = []
    for key, (arm, root) in KEYS.items():
        assert read(root / f'metrics/{arm}/complete.json')['passed']
        records = read(root / f'metrics/{arm}/per_identity.json')
        assert len(records) == 256
        for rec in records:
            lookup[key, rec['stage'], rec['identity']] = rec
            if rec['failure']:
                failures.append(dict(rec, key=key))
        if root == ext.PARENT:
            freeze(OUT / f'baseline_metrics/{key}.json', records)
        for stage in ['original', 'swapped']:
            for metric in ['margin', 'dists_old', 'dists_new', 'color_margin', 'iou', 'edge_f1', 'leak']:
                values = [r[metric] for r in records if r['stage'] == stage and r.get(metric) is not None]
                summaries.append(dict(key=key, stage=stage, metric=metric, **statistics(values)))
    pairing = []
    for row in rows:
        identity = row['id']
        bindings = []
        for key, (arm, root) in KEYS.items():
            for stage in ['original', 'swapped']:
                image = root / f'evaluation/{stage}/{arm}/images/{identity}.png'
                meta = read(image.with_suffix('.json'))
                rec = lookup[key, stage, identity]
                assert sha(image) == meta['output_sha256'] == rec['output_sha256']
                assert not meta['gt_used_in_inference'] and meta['explicit_latent_verified']
                assert meta['inputs']['reference'] == sha(row['reference_alternative' if stage == 'swapped' else 'reference'])
                bindings.append((meta['seed'], meta['initial_latent_sha256'], meta['inputs']['sketch'], meta['inputs']['gt']))
        assert len(set(bindings)) == 1
        pairing.append(dict(identity=identity, eight_outputs_paired=True, latent_sha256=bindings[0][1]))
    csv_write(OUT / 'audit/eight_output_pairing.csv', pairing)
    contrasts = []
    per_identity = []
    pairs = [('T3_s42', 'C_s42')] + [('T3_s42', k) for k in ARMS]
    for a, b in pairs:
        differences = []
        for identity in ids:
            x = lookup[a, 'swapped', identity]['margin']
            y = lookup[b, 'swapped', identity]['margin']
            ok = x is not None and y is not None
            per_identity.append(dict(a=a, b=b, identity=identity, valid=ok, difference=x-y if ok else None))
            if ok:
                differences.append(x-y)
        contrasts.append(dict(a=a, b=b, planned_n=128, **statistics(differences)))
    # 原计划五项族保持不缩小。未实施的两个 seed 对比仅以1作保守校正占位，绝不当作实测 p。
    correction = [dict(r) for r in contrasts] + [dict(p=1.0, not_measured=True) for _ in range(2)]
    holm(correction)
    for result, corrected in zip(contrasts, correction[:3]):
        result['p_holm_planned5'] = corrected['p_holm']
    structures = []
    for a, b in pairs[1:]:
        for stage in ['original', 'swapped']:
            for metric, margin in CONFIG['structure_margins'].items():
                sign = -1 if metric == 'leak' else 1
                values = [sign*(lookup[a, stage, i][metric]-lookup[b, stage, i][metric]) for i in ids
                          if lookup[a, stage, i].get(metric) is not None and lookup[b, stage, i].get(metric) is not None]
                structures.append(dict(a=a, b=b, stage=stage, metric=metric,
                    positive_favors_full_T3=True, original_margin_for_context=margin,
                    descriptive_exploratory_only=True, **statistics(values)))
    streams = []
    historical = SUP / 'training/formal/T3'
    with (historical / 'training_log.csv').open(encoding='utf-8-sig') as f:
        old_trace = list(csv.DictReader(f))
    assert len(old_trace) == 28365
    for key in ARMS:
        run = ext.folder(key)
        count = read(run / 'optimizer_update_count.json')
        assert count['actual_successful_updates'] == 28365 and count['skipped_updates'] == 0 and count['complete']
        assert read(run / 'checkpoint_verification.json')['passed']
        assert read(run / 'seed_and_rng_manifest.json') == read(historical / 'seed_and_rng_manifest.json')
        with (run / 'training_log.csv').open(encoding='utf-8-sig') as f:
            trace = list(csv.DictReader(f))
        assert len(trace) == 28365
        for current, previous in zip(trace, old_trace):
            assert all(current[k] == previous[k] for k in ['update', 'lr', 'microbatches_per_rank', 'rng_rank0']), (key, current['update'])
        streams.append(dict(key=key, updates=28365, skipped=0, all_rank_final_rng_equal_historical_T3=True,
            all_update_lr_and_rng_trace_equal_historical_T3=True))
    write(OUT / 'audit/paired_training_streams.json', streams)
    write(OUT / 'statistics/results.json', dict(exploratory=True, training_seeds=[42],
        contrasts=contrasts, original_planned_family_size=5,
        unimplemented_contrasts=['T3-C seed43', 'T3-C seed44'],
        multiplicity='Holm original five-item family; missing tests padded by 1 solely for conservative correction',
        new_training_runs=2, new_generated_images=512, human_reviewers=0, human_blind_scores=None,
        original_confirmatory_decision=read(ext.PARENT / 'statistics/decision.json'), commit=commit()))
    write(OUT / 'statistics/model_summary.json', summaries)
    write(OUT / 'statistics/failures.json', failures)
    csv_write(OUT / 'statistics/contrasts.csv', contrasts)
    csv_write(OUT / 'statistics/per_identity.csv', per_identity)
    csv_write(OUT / 'statistics/model_summary.csv', summaries)
    csv_write(OUT / 'statistics/structure_descriptive.csv', structures)
    fixed = read(ROOT / 'splits/qualitative_fixed24.json')
    assert len(fixed) == 24
    write(OUT / 'figures/selection.json', dict(fixed24=fixed, selected_before_generation=True))
    labels = {'C_s42': 'C', 'T3_s42': 'T3-full', 'T3_no_body_s42': 'no-body', 'T3_no_rest_s42': 'no-rest'}
    for identity in fixed:
        row = next(r for r in rows if r['id'] == identity)
        canvas = Image.new('RGB', (1280, 520), 'white')
        draw = ImageDraw.Draw(canvas)
        for j, field in enumerate(['sketch', 'reference', 'reference_alternative']):
            draw.text((j*400+5, 5), identity+' '+field, fill='black')
            place(canvas, row[field], j*400+80, 25)
        for j, (key, stage) in enumerate((a, s) for a in KEYS for s in ['original', 'swapped']):
            arm, root = KEYS[key]
            draw.text((j*160+3, 252), labels[key]+' '+stage, fill='black')
            place(canvas, root / f'evaluation/{stage}/{arm}/images/{identity}.png', j*160, 284)
        path = OUT / f'figures/fixed24/{identity}.jpg'
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path, quality=94)
    ext.verify_parent()
    write(OUT / 'statistics/analysis_complete.json', dict(passed=True, panels=24, new_images=512,
        original_confirmation_unchanged=True, rng_audit_passed=True, commit=commit()))


def package():
    assert read(OUT / 'statistics/analysis_complete.json')['passed']
    ext.verify_parent()
    jobs = read(ROOT / 'jobs.json')
    for job in jobs:
        job['state'] = ext.job_state(job['id'])
    assert all(j['state'] == 'COMPLETED' for j in jobs if j['key'] in ARMS)
    assert all(j['state'].startswith('CANCELLED') for j in jobs if j['key'] not in ARMS)
    write(ROOT / 'jobs.json', jobs)
    identifiers = [str(j['id']) for j in jobs] + ['120764']
    accounting = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', ','.join(identifiers),
        '--format=JobID,JobName,State,ExitCode,ElapsedRaw,AllocTRES,NodeList'], text=True)
    (OUT / 'audit/sacct.txt').write_text(accounting, encoding='utf-8')
    for key in list(dict.fromkeys(ARMS+[j['key'] for j in jobs])):
        run = ext.folder(key)
        if not run.exists():
            continue
        dst = OUT / 'training_records' / key
        dst.mkdir(parents=True, exist_ok=True)
        for path in run.iterdir():
            if path.is_file() and path.suffix in ['.json', '.csv', '.txt']:
                shutil.copy2(path, dst / path.name)
    for name in ['tools/t3_response_extension.py', 'scripts/t3_response_extension.sbatch',
                 'tools/t3_ablation_closeout.py', 'scripts/t3_ablation_closeout.sbatch']:
        dst = OUT / 'code' / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE / name, dst)
    files = [p for folder in ['audit', 'protocol', 'splits', 'ablation_closeout']
             for p in (ROOT / folder).rglob('*') if p.is_file()]
    files += [p for key in ARMS for folder in [ROOT / f'metrics/{key}', ROOT / f'evaluation/original/{key}',
               ROOT / f'evaluation/swapped/{key}'] for p in folder.rglob('*') if p.is_file()]
    files += [p for p in (ROOT / 'logs').iterdir() if p.is_file() and not p.name.startswith('ablation_closeout_')]
    files += [ROOT / 'jobs.json']
    files = sorted(set(files))
    assert sum(p.suffix == '.png' for p in files) == 512
    manifest = [dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=sha(p)) for p in files]
    write(ROOT / 'ablation_artifact_manifest.json', manifest)
    bundle = ROOT / 'ablation_results.tar.gz'
    with tarfile.open(bundle, 'w:gz', compresslevel=3) as tar:
        for path in files + [ROOT / 'ablation_artifact_manifest.json']:
            tar.add(path, arcname=path.relative_to(ROOT), recursive=False)
    write(ROOT / 'ablation_bundle_info.json', dict(path=str(bundle), sha256=sha(bundle), bytes=bundle.stat().st_size,
        manifest_count=len(manifest), generated_images=512, completed_training_runs=2,
        cancelled_seed_extensions=True, human_reviewers=0, human_blind_scores=None,
        local_report_complete=False, original_confirmation_unchanged=True))


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    analyse()
    package()
    ext.status(stage='ABLATION_SERVER_COMPLETE_LOCAL_REVIEW_PENDING', server_execution_complete=True,
               retained_arms=ARMS, cancelled_seed_extensions=True, complete=False)
    print('ABLATION_CLOSEOUT_SERVER_COMPLETE', flush=True)
