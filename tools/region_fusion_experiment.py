"""三组融合实验的准备、真实预检、统一评价和交付打包。"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import numpy as np
from PIL import Image, ImageDraw
from tools import bf_t3_frozen_core as core

PROJECT = Path('/share/home/u2515283058/Mymodel')
ROOT = PROJECT / 'output_eval/region_asymmetric_fusion_20261104'
HISTORY = PROJECT / 'output_eval/bf_fusion_t3_three_arm_20261101'
PARENT = PROJECT / 'output_eval/bf_t3_ablation_20261102'
ARMS = ['bf', 'concat', 'region']
PRIMARY = ['edge_f1', 'iou', 'edge_l1', 'leak', 'lpips', 'tcf_lab']
RESPONSE = ['dists_margin', 'gray_margin', 'gray_dists']
read, write, sha = core.read, core.write, core.sha


def prepare(gpus):
    if gpus not in (1, 2, 4, 8):
        raise ValueError('每臂 GPU 数必须整除有效 batch8')
    if (ROOT / 'protocol.json').exists():
        assert read(ROOT / 'protocol.json')['gpus_per_arm'] == gpus
        return
    original = read(PARENT / 'protocol.json')
    history = read(HISTORY / 'protocol.json')
    assert sha(core.E1) == history['e1_sha256']
    fusion = HISTORY / 'initialization/bf_fusion.pt'
    assert sha(fusion) == history['fusion_sha256']
    (ROOT / 'initialization').mkdir(parents=True, exist_ok=True)
    shutil.copy2(fusion, ROOT / 'initialization/bf_fusion.pt')
    train = read(PARENT / 'training_manifest.json')
    assert len(train) == 45384
    write(ROOT / 'training_manifest.json', train)
    train_ids = {Path(row['cloth']).stem.split('_')[0] for row in train}
    rows = read(PARENT / 'evaluation.json')
    assert len(rows) == 192 and len({(row['id'], row['condition']) for row in rows}) == 192
    for row in rows:
        assert row['id'].split('_')[0] not in train_ids
        for key in ('sketch', 'reference', 'gt'):
            source = Path(row[key])
            assert sha(source) == row[key + '_sha256']
            target = ROOT / 'inputs' / row[key + '_sha256']
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(source, target)
            row[key] = str(target)
    write(ROOT / 'evaluation.json', rows)
    write(ROOT / 'audit/parent_protocol.json', original)
    write(ROOT / 'protocol.json', dict(name=ROOT.name, arms=ARMS, gpus_per_arm=gpus,
        updates=28365, samples=len(train), effective_batch=8, training_seed=42,
        learning_rate=5e-5, warmup=500, precision='bf16 autocast; trainable fp32',
        loss='epsilon MSE + GT masked Gram + .05 sketch edge + .1 reference garment RGB + .1 body RGB + .03816244801608925 rest GT L1',
        initialization='same E1, same original BF encoder, zero SD channel projections; compatible original FiLM heads reused',
        e1_sha256=sha(core.E1), bf_fusion_sha256=sha(fusion),
        training_manifest_sha256=sha(ROOT / 'training_manifest.json'),
        evaluation_sha256=sha(ROOT / 'evaluation.json'),
        preflight='each arm: 3 real T3 updates with finite gradients and actual four-scale hooks, then 2 full DDIM50 outputs',
        inference=dict(width=384, height=512, steps=50, cfg=7., sketch_scale=.6, texture_tokens=False),
        primary=dict(identities=128, condition='original', metrics=PRIMARY, correction='Holm12: region minus both baselines'),
        response=dict(planned_identities=32, neutral_captions=True, metrics=RESPONSE,
                      correction='Holm6: region minus both baselines'),
        source_commit=(ROOT / 'source_commit.txt').read_text().strip(),
        limitations=['exploratory one training seed; previously inspected test identities',
            'shared initialization applies to common encoder, projections and GAM parameters; architectures differ',
            'concat is a simple mechanism baseline, not parameter matched',
            'gray patch DISTS is not complete motif identity or direction accuracy',
            'region mask limits feature interaction, not guaranteed final background invariance']))
    print('PREPARED', ROOT, gpus, flush=True)


def arguments(variant, output=None):
    return ['--variant', variant, '--output-dir', str(output or ROOT / 'training'),
            '--project-root', str(PROJECT), '--initialization-root', str(ROOT)]


def preflight():
    if (ROOT / 'audit/preflight_complete.json').exists():
        assert read(ROOT / 'audit/preflight_complete.json')['passed']
        return
    import torch
    assert torch.cuda.is_available()
    results = {}
    for variant in ARMS:
        folder = ROOT / 'preflight'
        subprocess.run([sys.executable, '-m', 'tools.train_region_fusion', 'train'] + arguments(variant, folder)
            + ['--max-steps', '3', '--warmup', '0', '--accumulation', '8', '--workers', '0',
               '--gradient-audit', '--checkpoint-every', '3'], check=True)
        result = read(folder / variant / 'complete.json')
        gradients = read(folder / variant / 'gradients.json')
        assert result['updates'] == 3 and all(value > 0 and np.isfinite(value) for value in gradients.values())
        subprocess.run([sys.executable, '-m', 'tools.train_region_fusion', 'generate'] + arguments(variant, folder)
            + ['--checkpoint', str(folder / variant / 'step00003.pt'), '--eval-manifest', str(ROOT / 'evaluation.json'),
               '--limit', '2'], check=True)
        results[variant] = dict(training=result, gradients=gradients,
            generated=len(list((folder / variant / 'outputs').rglob('*.png'))))
        assert results[variant]['generated'] == 2
    write(ROOT / 'audit/preflight_complete.json', dict(passed=True, arms=results))


def audit_training():
    protocol = read(ROOT / 'protocol.json')
    complete = {arm: read(ROOT / 'training' / arm / 'complete.json') for arm in ARMS}
    runs = [value['run'] for value in complete.values()]
    assert all(run == runs[0] for run in runs)
    assert runs[0]['world'] == protocol['gpus_per_arm']
    assert runs[0]['world'] * runs[0]['accumulation'] == 8
    assert all(value['updates'] == 28365 for value in complete.values())
    assert all(value['stream_hashes'] == complete['bf']['stream_hashes'] for value in complete.values())
    lr = []
    for arm in ARMS:
        checkpoint = ROOT / 'training' / arm / 'step28365.pt'
        assert sha(checkpoint) == complete[arm]['checkpoint_sha256']
        rows = list(csv.DictReader((checkpoint.parent / 'training_log.csv').open()))
        assert [int(row['step']) for row in rows] == list(range(1, 28366))
        assert all(np.isfinite(float(value)) for row in rows for value in row.values())
        lr.append([float(row['lr']) for row in rows])
    assert lr[0] == lr[1] == lr[2]
    write(ROOT / 'audit/training_comparability.json', dict(passed=True,
        same_common_initialization=True, same_training_stream=True, same_lr=True, arms=complete))


def evaluate():
    import torch
    from eval.eval_utils import prepare_evaluation_masks
    from eval.metrics import compute_structure_preservation, compute_texture_color_fidelity, compute_texture_leakage
    from tools.e2a_attr_eval import frozen_lpips
    from tools import t3_response_run as response
    from tools.t3_response_common import patches
    from tools.bf_t3_three_arm_report import paired, gray, table
    from tools.bf_t3_ablation_report import holm
    audit_training()
    response.ROOT = ROOT
    dists = response.load_dists('cuda')
    lpips, assets = frozen_lpips()
    write(ROOT / 'audit/lpips_assets.json', assets)
    rows = read(ROOT / 'evaluation.json')
    originals = {row['id']: row for row in rows if row['condition'] == 'neutral_original'}
    cache = {}
    def features(parts):
        value = torch.from_numpy(np.stack(parts).transpose(0, 3, 1, 2).copy()).float().cuda() / 255
        return dists.forward_once(value)
    def reference(path, grayscale):
        key = (path, grayscale)
        if key not in cache:
            pixels = np.asarray(Image.open(path).convert('RGB').resize((384, 512), Image.Resampling.LANCZOS))
            parts = [pixels[y:y+64, x:x+64] for y in np.linspace(0, 448, 4).round().astype(int)
                     for x in np.linspace(0, 320, 4).round().astype(int)]
            cache[key] = features([gray(part) for part in parts] if grayscale else parts)
        return cache[key]
    def distance(a, b):
        matrix = response.distances(dists, a, b)
        return float(.5 * (matrix.min(0).values.mean() + matrix.min(1).values.mean()))
    records = []
    with torch.inference_mode():
        for row in rows:
            masks = prepare_evaluation_masks((384, 512), sketch_path=row['sketch'], mask_policy='sketch_only')
            coords, _ = patches(masks['garment'])
            latents = []
            for arm in ARMS:
                path = ROOT / 'training' / arm / 'outputs' / row['condition'] / (row['id'] + '.png')
                meta = read(path.with_suffix('.json'))
                assert sha(path) == meta['output_sha256'] and not meta['GT_used_in_inference']
                assert meta['checkpoint_sha256'] == read(ROOT / 'training' / arm / 'complete.json')['checkpoint_sha256']
                assert meta['input_sketch_sha256'] == row['sketch_sha256']
                assert meta['input_reference_sha256'] == row['reference_sha256']
                assert meta['caption'] == row['caption'] and meta['seed'] == row['seed']
                latents.append(meta['initial_latent_sha256'])
                kwargs = dict(gen_path=str(path), sketch_path=row['sketch'], target_path=row['gt'], mask_bundle=masks)
                structure = compute_structure_preservation(**kwargs)
                record = dict(arm=arm, id=row['id'], condition=row['condition'], category=row['category'],
                    edge_f1=structure.get('edge_f1'), iou=structure.get('sketch_iou'), edge_l1=structure.get('edge_l1'),
                    leak=compute_texture_leakage(**kwargs).get('leak_colored_frac'), lpips=None, tcf_lab=None,
                    dists_margin=None, gray_margin=None, gray_dists=None, patch_n=len(coords))
                if row['condition'] == 'original':
                    record['lpips'] = float(lpips(core.tensor(path).cuda(), core.tensor(row['gt']).cuda()).mean())
                    record['tcf_lab'] = compute_texture_color_fidelity(texture_path=row['reference'], **kwargs).get('tcf_lab_delta')
                if row['condition'].startswith('neutral_') and len(coords) >= 4:
                    pixels = np.asarray(Image.open(path).convert('RGB'))
                    parts = [pixels[y:y+64, x:x+64] for x, y in coords]
                    for grayscale in (False, True):
                        generated = features([gray(part) for part in parts] if grayscale else parts)
                        new = distance(generated, reference(row['reference'], grayscale))
                        if grayscale:
                            record['gray_dists'] = new
                        if row['condition'] == 'neutral_swapped':
                            old = distance(generated, reference(originals[row['id']]['reference'], grayscale))
                            record['gray_margin' if grayscale else 'dists_margin'] = old - new
                records.append({key: None if isinstance(value, float) and not np.isfinite(value) else value
                                for key, value in record.items()})
            assert len(set(latents)) == 1
            print('EVALUATED', row['condition'], row['id'], flush=True)
    assert len(records) == 576
    write(ROOT / 'metrics/per_image.json', records)
    table(ROOT / 'metrics/per_image.csv', records)
    summary = []
    for arm in ARMS:
        for condition in ('original', 'neutral_original', 'neutral_swapped'):
            part = [row for row in records if row['arm'] == arm and row['condition'] == condition]
            for metric in PRIMARY + RESPONSE:
                values = [row[metric] for row in part if row[metric] is not None]
                summary.append(dict(arm=arm, condition=condition, metric=metric, n=len(values), planned_n=len(part),
                                    mean=float(np.mean(values)) if values else None))
    write(ROOT / 'metrics/summary.json', summary)
    table(ROOT / 'metrics/summary.csv', summary)
    lookup = {(row['arm'], row['id'], row['condition']): row for row in records}
    statistics = []
    for condition, metrics in [('original', PRIMARY), ('neutral_swapped', RESPONSE)]:
        family = []
        for baseline in ARMS[:2]:
            for metric in metrics:
                values = []
                for row in rows:
                    if row['condition'] != condition:
                        continue
                    a = lookup['region', row['id'], condition][metric]
                    b = lookup[baseline, row['id'], condition][metric]
                    if a is not None and b is not None:
                        values.append(a - b)
                family.append(dict(comparison='region minus ' + baseline, condition=condition, metric=metric, **paired(values)))
        holm(family)
        statistics.extend(family)
    write(ROOT / 'metrics/paired_statistics.json', statistics)
    table(ROOT / 'metrics/paired_statistics.csv', [dict(row, ci95=json.dumps(row['ci95'])) for row in statistics])
    chosen = sum([sorted([row for row in rows if row['condition'] == condition],
                        key=lambda row: sha(row['sketch']))[:4] for condition in ('original', 'neutral_swapped')], [])
    for index in range(2):
        canvas = Image.new('RGB', (1080, 1040), 'white')
        draw = ImageDraw.Draw(canvas)
        for line, row in enumerate(chosen[index*4:index*4+4]):
            paths = [row['reference'], row['sketch'], row['gt'] if index == 0 else None]
            paths += [ROOT / 'training' / arm / 'outputs' / row['condition'] / (row['id'] + '.png') for arm in ARMS]
            for column, (label, path) in enumerate(zip(['Reference', 'Sketch', 'GT original only'] + ARMS, paths)):
                draw.text((column*180+3, line*260+3), label, fill='black')
                if path:
                    image = Image.open(path).convert('RGB')
                    image.thumbnail((174, 226), Image.Resampling.LANCZOS)
                    canvas.paste(image, (column*180, line*260+23))
            draw.text((3, line*260+246), row['id'] + ' ' + row['condition'], fill='black')
        (ROOT / 'figures').mkdir(exist_ok=True)
        canvas.save(ROOT / 'figures' / ('fixed_%02d.png' % (index + 1)))
    write(ROOT / 'evaluation_complete.json', dict(complete=True, outputs=576, arms=ARMS,
          independent_confirmation=False, primary128=True, neutral_response32=True))


def package():
    assert read(ROOT / 'evaluation_complete.json')['complete']
    jobs = read(ROOT / 'submitted_jobs.json')
    identifiers = [job for history in jobs.values() for job in history]
    accounting = subprocess.check_output(['sacct', '-X', '-j', ','.join(identifiers), '-P',
        '--format=JobID,JobName,State,ExitCode,ElapsedRaw,AllocTRES'], text=True)
    (ROOT / 'audit/final_sacct.psv').write_text(accounting)
    files = [path for folder in ('audit', 'preflight', 'training', 'metrics', 'figures', 'inputs', 'logs')
             for path in (ROOT / folder).rglob('*') if path.is_file() and path.suffix not in ('.pt', '.tmp')]
    files += [ROOT / name for name in ('protocol.json', 'evaluation.json', 'training_manifest.json',
              'source_commit.txt', 'submitted_jobs.json', 'job_resources.json', 'evaluation_complete.json')]
    files += [ROOT / 'source' / name for name in (
        'models/region_aware_fusion.py', 'models/bf_original_fusion.py', 'models/bf_sd_skip.py',
        'tools/train_region_fusion.py', 'tools/region_fusion_experiment.py',
        'scripts/region_fusion.sbatch', 'scripts/submit_region_fusion.py')]
    write(ROOT / 'bundle_manifest.json', {str(path.relative_to(ROOT)): dict(bytes=path.stat().st_size,
          sha256=sha(path)) for path in files})
    with tarfile.open(ROOT / 'review_results.tar.gz', 'w:gz') as archive:
        for path in files + [ROOT / 'bundle_manifest.json']:
            archive.add(path, arcname=str(path.relative_to(ROOT)))
    write(ROOT / 'bundle_info.json', dict(bytes=(ROOT / 'review_results.tar.gz').stat().st_size,
          sha256=sha(ROOT / 'review_results.tar.gz'), members=len(files) + 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'preflight', 'evaluate', 'package'])
    parser.add_argument('--gpus-per-arm', type=int, default=2)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.gpus_per_arm)
    else:
        globals()[args.action]()
