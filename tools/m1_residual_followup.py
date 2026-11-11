"""用固定八对诊断 M1，再以同预算筛选保留正向 FiLM 的残差版。"""
import argparse
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import torch

from models.e2_compat import load_trainable_strict
from models.m1_patch_retrieval import attach_m1
from tools import e2_b_screen_evaluation as legacy
from tools import m1_evaluation, m1_experiment
from tools.bf_t3_three_arm_report import paired
from tools.e2_fft_experiment import SOURCE, read, setup, sha, write
from tools.e2_fft_training import tensor_hash

PROJECT = Path('/share/home/u2515283058/Mymodel')
OUTPUT = PROJECT / 'output_eval'
FAILURES = ('validation_1174', 'validation_6372', 'validation_870', 'validation_621')
DIAGNOSTIC = ('E2', 'M1Init', 'M1Full', 'M1FG')
FORMAL = ('M0', 'M1', 'M1R')
PRIMARY = 'gray_strict_response_change'
EXPECTED_OLD = {
    'M0': '0505969102c98d243ca6deae8439c0280d54c7ff451bb3362d677cec00373400',
    'M1': '8054027054ef75f89ce39e3b74cd28d9abec884be8a2b76a4cc3f1a22ff1d839',
}


def diagnostic_pairs(pairs):
    chosen = list(FAILURES)
    for pattern in ('stripes', 'checks', 'dots', 'stripes'):
        choice = next(p for p in pairs if p['case_id'] not in chosen
                      and p['qualification']['old_pattern'] == pattern)
        chosen.append(choice['case_id'])
    return [p for p in pairs if p['case_id'] in chosen]


def copy_checked(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(source, target)
    assert sha(source) == sha(target), str(target)


def prepare(args):
    from tools.e2_a_minimal import mask_snapshot
    from tools.e2_b_screen_training import assert_snapshot

    core, out = setup(args)
    old = Path(args.previous)
    assert old.resolve() != out.resolve()
    for name in ('strict_pairs.json', 'split.json', 'audit/training_inputs.json',
                 'audit/mask_inputs.json'):
        copy_checked(old / name, out / name)
    assert sha(out / 'strict_pairs.json') == m1_experiment.PAIRS_SHA
    assert_snapshot(args)
    mask_snapshot(args, core, final=True)
    pairs = read(out / 'strict_pairs.json')['pairs']
    case_ids = [p['case_id'] for p in diagnostic_pairs(pairs)]
    assert len(case_ids) == 8
    for p in pairs:
        for key in ('sketch', 'gt', 'original_reference', 'swapped_reference'):
            assert sha(Path(args.data) / p[key]) == p['hashes'][key]
    old_audit = read(old / 'audit/training_completion.json')
    assert old_audit['passed'] and old_audit['same_legacy_stream']
    assert read(old / 'audit/training_inputs_final.json')['passed']
    for arm in FORMAL[:2]:
        complete = read(old / 'training' / arm / 'complete.json')
        assert (complete['checkpoint_sha256'] == EXPECTED_OLD[arm]
                == old_audit['arms'][arm]['checkpoint_sha256']
                == sha(old / 'training' / arm / 'latest.pt'))
        for name in ('complete.json', 'training.jsonl'):
            copy_checked(old / 'training' / arm / name, out / 'training' / arm / name)
        for p in pairs:
            for name in ('old.png', 'old.json', 'new.png', 'new.json'):
                source = old / 'outputs' / arm / p['case_id'] / name
                copy_checked(source, out / 'outputs' / arm / p['case_id'] / name)
    config = m1_experiment.configuration(args, 'M1R')
    baseline = read(old / 'comparison/M0.json')
    allowed = {'arm', 'code_commit', 'protocol', 'split', 'm1',
               'preserve_forward_film', 'residual_mask'}
    assert {k: v for k, v in config.items() if k not in allowed} == {
        k: v for k, v in baseline.items() if k not in allowed}
    write(out / 'comparison/M1R.json', config)
    write(out / 'protocol.json', dict(
        experiment='M1_residual_followup_v1', seed=42, updates=2000, effective_batch=8,
        diagnostic_case_ids=case_ids, historical_failures=list(FAILURES),
        control_selection='first unused manifest case by stripes, checks, dots, stripes',
        diagnostic_arms=DIAGNOSTIC, formal_arms=FORMAL,
        foreground='sketch-only garment area pooled to each scale; gates M1 delta only',
        residual='pre-fusion structure queries texture; delta added after original forward FiLM',
        reverse_film='reads pre-fusion structure', formal_mask=False,
        sampling=dict(steps=50, cfg=7, sketch_scale=.6, size=[384, 512]),
        historical_comparators=dict(source=str(old), checkpoint_sha256=EXPECTED_OLD),
        primary=PRIMARY, primary_rule='M1R - M0 paired 95% lower > 0 and raw signflip p < .05',
        protection_margins=read(old / 'protocol.json')['inherited_exploratory_noninferiority'],
        source=m1_experiment.m_environment(), followup_source_sha256=sha(__file__)))
    print('M1R_PREPARED', case_ids, flush=True)


def checkpoint(out, arm, checkpoint_sha, **extra):
    write(out / 'training' / arm / 'complete.json',
          dict(checkpoint_sha256=checkpoint_sha, **extra))


def features(adapter, core, args, pair):
    sketch = core.tensor(Path(args.data) / pair['sketch']).to(args.device).half()
    reference = core.tensor(Path(args.data) / pair['original_reference']).to(args.device).half()
    gray = (sketch * sketch.new_tensor([.2989, .5870, .1140])[None, :, None, None]).sum(1, keepdim=True)
    with torch.inference_mode():
        fusion = adapter.fusion(gray, reference)
        adapter.condition(sketch, reference)
        result = [tensor_hash([('feature', value)]) for value in [*fusion, *adapter.features]]
    adapter.clear()
    return result


def generation_sha(out, arm, pair):
    return [read(out / 'outputs' / arm / pair['case_id'] / (condition + '.json'))['output_sha256']
            for condition in ('old', 'new')]


def pilot(args):
    core, out = setup(args)
    proof = argparse.Namespace(**vars(args))
    proof.out = out / 'regression'
    copy_checked(out / 'strict_pairs.json', proof.out / 'strict_pairs.json')
    pair = diagnostic_pairs(read(out / 'strict_pairs.json')['pairs'])[0]
    pipe, network, _ = m1_experiment.load_m(args, core, 'M0')
    parent_sha = read(SOURCE / 'configs/e2_fft_regression.json')['checkpoint_sha256']
    m1_evaluation.prepare_inference(args, pipe, network)
    original = features(network.adapter, core, args, pair)
    m1_evaluation.generate_images(proof, core, proof.out, 'E2', pipe, network, parent_sha, [pair])
    attach_m1(network, preserve_film=True)
    restored = features(network.adapter, core, args, pair)
    assert restored == original, 'M1R 初始融合或 skip 张量与父模型不同'
    m1_evaluation.generate_images(proof, core, proof.out, 'M1RInit', pipe, network, parent_sha, [pair])
    png_sha = generation_sha(proof.out, 'E2', pair)
    assert png_sha == generation_sha(proof.out, 'M1RInit', pair), 'M1R 初始成图与父模型不同'
    write(proof.out / 'image_result.json', dict(
        passed=True, case_id=pair['case_id'], parent_sha256=parent_sha,
        bitwise_fusion_and_skip=True, tensor_sha256=original, png_sha256=png_sha,
        environment=m1_experiment.m_environment()))
    print('M1R_PARENT_EQUIVALENT', pair['case_id'], png_sha, flush=True)


def diagnose(args):
    import gc
    from eval.eval_utils import prepare_evaluation_masks

    core, root = setup(args)
    out = root / 'diagnostic'
    args = argparse.Namespace(**vars(args))
    args.out = out
    copy_checked(root / 'strict_pairs.json', out / 'strict_pairs.json')
    pairs = diagnostic_pairs(read(out / 'strict_pairs.json')['pairs'])
    case_ids = read(root / 'protocol.json')['diagnostic_case_ids']
    assert case_ids == [p['case_id'] for p in pairs]
    parent_sha = read(SOURCE / 'configs/e2_fft_regression.json')['checkpoint_sha256']
    pipe, network, _ = m1_experiment.load_m(args, core, 'M0')
    m1_evaluation.generate_images(args, core, out, 'E2', pipe, network, parent_sha, pairs)
    attach_m1(network)
    m1_evaluation.generate_images(args, core, out, 'M1Init', pipe, network, parent_sha, pairs)
    network.adapter.unbind()
    del pipe, network
    gc.collect()
    torch.cuda.empty_cache()
    pipe, network, _ = m1_experiment.load_m(args, core, 'M1')
    ckpt = Path(args.previous) / 'training/M1/latest.pt'
    ckpt_sha = sha(ckpt)
    assert ckpt_sha == EXPECTED_OLD['M1']
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    load_trainable_strict(network, state['network'])
    del state
    m1_evaluation.generate_images(args, core, out, 'M1Full', pipe, network, ckpt_sha, pairs)
    plain = features(network.adapter, core, args, pairs[0])
    fusion = network.adapter.fusion
    fusion.residual_mask = torch.ones((1, 1, 512, 384), device=args.device)
    assert features(network.adapter, core, args, pairs[0]) == plain
    fusion.residual_mask = None

    def masked_condition(sketch, sketch_tensor, reference_tensor):
        masks = prepare_evaluation_masks((384, 512), sketch_path=str(sketch), mask_policy='sketch_only')
        mask = masks['garment']
        assert mask is not None and 0 < mask.mean() < 1
        fusion.residual_mask = torch.from_numpy(mask.astype(np.float32))[None, None].to(args.device)
        try:
            network.adapter.condition(sketch_tensor, reference_tensor)
        finally:
            fusion.residual_mask = None

    m1_evaluation.generate_images(args, core, out, 'M1FG', pipe, network, ckpt_sha, pairs, masked_condition)
    for arm in DIAGNOSTIC:
        checkpoint(out, arm, parent_sha if arm in DIAGNOSTIC[:2] else ckpt_sha)
    checkpoint_match = sum(a == b for p in pairs for a, b in zip(
        generation_sha(out, 'M1Full', p), generation_sha(root, 'M1', p)))
    write(out / 'generation/audit.json', dict(
        all_one_mask_feature_equivalence=True, historical_M1_png_match=checkpoint_match,
        png_comparisons=2 * len(pairs), sketch_only_mask=True))
    network.adapter.unbind()
    del pipe, network, fusion
    gc.collect()
    torch.cuda.empty_cache()
    copy_checked(Path(args.previous) / 'audit/training_inputs_final.json',
                 out / 'audit/training_inputs_final.json')
    legacy.evaluate(args, arms=DIAGNOSTIC, case_ids=case_ids,
                    audit=lambda _: None, compare=lambda *_: [])
    summarize(args, pairs, (('M1Init', 'E2'), ('M1FG', 'M1Full')),
              read(root / 'protocol.json')['protection_margins'])
    panels(args, pairs, DIAGNOSTIC)
    bundle(out, 'diagnostic_results.zip', (out / 'metrics', out / 'generation'))
    print('M1R_DIAGNOSED', checkpoint_match, flush=True)


def audit_training(args):
    from tools.e2_a_minimal import mask_snapshot
    from tools.e2_b_screen_training import audit_training as audit_inputs

    core, out = setup(args)
    audit_inputs(args, final=True)
    mask_snapshot(args, core, final=True)
    records = {}
    for arm in FORMAL:
        folder = out / 'training' / arm
        complete = read(folder / 'complete.json')
        assert complete['status'] == 'completed' and complete['updates'] == 2000
        assert complete['frozen_unchanged']
        assert complete['config']['initialization_sha256'] == read(out / 'comparison/M1R.json')['initialization_sha256']
        assert complete['config']['training_input_map_sha256'] == sha(out / 'audit/training_inputs.json')
        assert complete['config']['split_sha256'] == sha(out / 'split.json')
        ckpt = folder / 'latest.pt' if arm == 'M1R' else Path(args.previous) / 'training' / arm / 'latest.pt'
        assert sha(ckpt) == complete['checkpoint_sha256']
        rows = [json.loads(line) for line in (folder / 'training.jsonl').read_text(encoding='utf-8').splitlines()]
        assert [r['step'] for r in rows] == list(range(1, 2001))
        assert all(np.isfinite([r['grad_norm'], *r['losses'].values()]).all() for r in rows)
        stream = [(r['tokens'], r['chain']) for r in rows]
        if arm == 'M0':
            baseline_stream = stream
        assert stream == baseline_stream
        records[arm] = dict(
            checkpoint_sha256=complete['checkpoint_sha256'], updates=len(rows),
            reused=arm != 'M1R', total_seconds=sum(r['seconds'] for r in rows),
            peak_gib=max(r['peak_gib'] for r in rows),
            first100={k: float(np.mean([r['losses'][k] for r in rows[:100]])) for k in rows[0]['losses']},
            last100={k: float(np.mean([r['losses'][k] for r in rows[-100:]])) for k in rows[0]['losses']})
    write(out / 'audit/training_completion.json', dict(passed=True, identical_training_stream=True, arms=records))
    print('M1R_TRAINING_AUDITED', flush=True)


def generate(args):
    core, out = setup(args)
    audit = read(out / 'audit/training_completion.json')
    assert audit['passed'] and read(out / 'audit/mask_inputs_final.json')['passed']
    pipe, network, _ = m1_experiment.load_m(args, core, 'M1R')
    ckpt = out / 'training/M1R/latest.pt'
    ckpt_sha = sha(ckpt)
    assert ckpt_sha == audit['arms']['M1R']['checkpoint_sha256']
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    load_trainable_strict(network, state['network'])
    del state
    m1_evaluation.generate_images(args, core, out, 'M1R', pipe, network, ckpt_sha,
                                  read(out / 'strict_pairs.json')['pairs'])


def summarize(args, pairs, comparisons, margins):
    from PIL import Image
    from eval.eval_utils import prepare_evaluation_masks, estimate_foreground_mask
    from tools.e2_a_evaluation import boundary_f1

    out = Path(args.out)
    rows = read(out / 'metrics/per_image.json')
    responses = read(out / 'metrics/responses.json')
    by_image = {(r['arm'], r['case_id'], r['condition']): r for r in rows}
    by_response = {(r['arm'], r['case_id']): r for r in responses}
    arms = dict.fromkeys(arm for comparison in comparisons for arm in comparison)
    for p in pairs:
        case_id = p['case_id']
        masks = prepare_evaluation_masks((384, 512), sketch_path=str(Path(args.data) / p['sketch']),
                                         mask_policy='sketch_only')
        inside, outside = masks['garment'], masks['outside']
        for arm in arms:
            images = []
            for condition in ('old', 'new'):
                row = by_image[arm, case_id, condition]
                with Image.open(out / 'outputs' / arm / case_id / (condition + '.png')) as image:
                    image = image.convert('RGB')
                    pixels = np.asarray(image)
                    images.append(pixels)
                    background = pixels[outside].astype(np.float32) / 255
                    row['outside_luma'] = float((background @ np.array([.2126, .7152, .0722])).mean())
                    row['outside_sat'] = float(np.asarray(image.convert('HSV'))[..., 1][outside].mean() / 255)
                    row['outside_pixels'] = int(outside.sum())
                    row['boundary_f1_r2'] = boundary_f1(estimate_foreground_mask(image, (384, 512)), inside, 2)
            difference = np.abs(images[1].astype(np.float32) - images[0]) / 255
            by_response[arm, case_id]['inside_reference_rgb_mae'] = float(difference[inside].mean())

    stats = []
    for candidate, baseline in comparisons:
        for subset, selected in [('all', pairs), ('historical_failures', [p for p in pairs if p['case_id'] in FAILURES])]:
            for metric in (PRIMARY, 'rgb_strict_response_change', 'gray_tolerant_response_change',
                           'rgb_tolerant_response_change', 'inside_reference_rgb_mae'):
                values = [(by_response[candidate, p['case_id']][metric], by_response[baseline, p['case_id']][metric])
                          for p in selected]
                stats.append(comparison_row(candidate, baseline, subset, 'paired', metric, values))
            for condition in ('old', 'new'):
                for metric in ('edge_f1', 'iou', 'edge_l1', 'leak', 'tcf_lab', 'lpips',
                               'boundary_f1_r2', 'outside_luma', 'outside_sat'):
                    if metric == 'lpips' and condition == 'new':
                        continue
                    values = [(by_image[candidate, p['case_id'], condition][metric],
                               by_image[baseline, p['case_id'], condition][metric]) for p in selected]
                    row = comparison_row(candidate, baseline, subset, condition, metric, values)
                    if metric in margins or metric == 'boundary_f1_r2':
                        direction = 1 if metric in ('edge_f1', 'iou', 'boundary_f1_r2') else -1
                        margin = margins.get(metric, 0.)
                        row['exploratory_margin'] = margin
                        row['protection_pass'] = (row['n'] == row['planned_n'] and row['ci95'] is not None
                                                  and min(direction * v for v in row['ci95']) >= -margin)
                    stats.append(row)
    write(out / 'metrics/per_image.json', rows)
    write(out / 'metrics/responses.json', responses)
    write(out / 'metrics/followup_comparisons.json', stats)
    return stats


def comparison_row(candidate, baseline, subset, condition, metric, values):
    complete = [(a, b) for a, b in values if a is not None and b is not None and np.isfinite([a, b]).all()]
    return dict(candidate=candidate, baseline=baseline, subset=subset, condition=condition, metric=metric,
                planned_n=len(values), candidate_mean=float(np.mean([a for a, _ in complete])) if complete else None,
                baseline_mean=float(np.mean([b for _, b in complete])) if complete else None,
                **paired([a - b for a, b in complete]))


def panels(args, pairs, arms):
    from PIL import Image, ImageDraw

    out = Path(args.out)
    folder = out / 'metrics/panels'
    folder.mkdir(parents=True, exist_ok=True)
    for p in pairs:
        canvas = Image.new('RGB', ((3 + 2 * len(arms)) * 192, 284), 'white')
        draw = ImageDraw.Draw(canvas)
        files = [(k, Path(args.data) / p[k]) for k in ('sketch', 'original_reference', 'swapped_reference')]
        files += [(f'{arm} {condition}', out / 'outputs' / arm / p['case_id'] / (condition + '.png'))
                  for arm in arms for condition in ('old', 'new')]
        for i, (label, path) in enumerate(files):
            draw.text((i * 192 + 4, 4), label, fill='black')
            with Image.open(path) as image:
                canvas.paste(image.convert('RGB').resize((192, 256)), (i * 192, 24))
        canvas.save(folder / (p['case_id'] + '.jpg'), quality=92)


def bundle(out, name, paths):
    with zipfile.ZipFile(out / name, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            for file in sorted(path.rglob('*')) if path.is_dir() else [path]:
                if file.is_file() and not file.name.endswith('_progress.json'):
                    archive.write(file, file.relative_to(out))


def evaluate(args):
    out = Path(args.out)

    def check_audit(root):
        assert read(root / 'audit/training_completion.json')['passed']

    legacy.evaluate(args, arms=FORMAL, audit=check_audit)
    pairs = read(out / 'strict_pairs.json')['pairs']
    protocol = read(out / 'protocol.json')
    stats = summarize(args, pairs, (('M1R', 'M0'), ('M1R', 'M1'), ('M1', 'M0')),
                      protocol['protection_margins'])
    main = [r for r in stats if (r['candidate'], r['baseline'], r['subset']) == ('M1R', 'M0', 'all')]
    primary = next(r for r in main if r['metric'] == PRIMARY)
    passed = primary['n'] == 32 and primary['ci95'][0] > 0 and primary['p'] < .05
    protected = all(r['protection_pass'] for r in main if 'protection_pass' in r)
    write(out / 'metrics/decision.json', dict(primary=primary, texture_evidence=passed,
          protection_pass=protected, effective_screening=passed and protected, exploratory_single_seed=True))
    response = {(r['arm'], r['case_id']): r[PRIMARY] for r in read(out / 'metrics/responses.json')}
    worst = sorted(pairs, key=lambda p: response['M1R', p['case_id']] - response['M0', p['case_id']])[:4]
    write(out / 'metrics/worst_cases.json', [p['case_id'] for p in worst])
    panel_ids = set(FAILURES) | {p['case_id'] for p in worst}
    panels(args, [p for p in pairs if p['case_id'] in panel_ids], FORMAL)
    bundle(out, 'followup_results.zip', (
        out / 'protocol.json', out / 'strict_pairs.json', out / 'audit/training_completion.json',
        out / 'audit/training_inputs_final.json', out / 'audit/mask_inputs_final.json',
        out / 'regression/image_result.json', out / 'smoke/result.json', out / 'metrics',
        out / 'comparison/M1R.json', out / 'training/M1R/complete.json',
        out / 'training/M1R/training.jsonl', out / 'generation/M1R.json',
        out / 'diagnostic/diagnostic_results.zip'))
    print('M1R_REVIEWED', passed, protected, primary, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'pilot', 'smoke', 'diagnose',
                                           'train', 'audit-training', 'generate', 'evaluate'))
    parser.add_argument('--project', type=Path, default=PROJECT)
    parser.add_argument('--out', type=Path, default=OUTPUT / 'm1_residual_followup_20261112')
    parser.add_argument('--previous', type=Path, default=OUTPUT / 'm1_s2t_patch_20261111')
    parser.add_argument('--parent', type=Path, default=OUTPUT / 'e2_fft_20261106')
    parser.add_argument('--baseline', type=Path, default=OUTPUT / 'e2_b_screen_20261108')
    parser.add_argument('--data', type=Path, default=PROJECT.parent / 'datasets/BF')
    parser.add_argument('--candidates', type=Path, default=SOURCE / 'configs/e2_b_candidates.json')
    parser.add_argument('--device', default='cuda:0')
    options = parser.parse_args()
    action = options.action
    args = argparse.Namespace(**vars(options), arm='M1R')
    if action == 'smoke':
        m1_experiment.smoke(args, arms=('M1R',))
    elif action == 'train':
        m1_experiment.train(args)
    else:
        globals()[action.replace('-', '_')](args)


if __name__ == '__main__':
    main()
