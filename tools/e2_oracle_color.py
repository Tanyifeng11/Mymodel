"""Colour-calibrated oracle stripes, with raw versus high-pass latent residuals."""
import argparse
import collections
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

from eval.eval_utils import prepare_evaluation_masks
from models.e2_compat import load_trainable_strict
from tools.bfm_skip_probe import CHECKPOINT_SHA, PAIRS_SHA, PREVIOUS, PROJECT, RUN, control_path
from tools.e2_fft_experiment import environment, read, setup, sha, write
from tools.e2_oracle_texture import (
    ACTIVE, ARMS, SIZE, VARIANTS, decode, encode, follow, image, inject,
    oracle_image, projected, protects, rgb_image, stripes, target_hit, weights_digest,
)
from tools.m1_evaluation import prepare_inference
from tools.m1_experiment import load_m
from tools.m1_residual_followup import comparison_row

PRIOR = PROJECT / 'output_eval/e2_oracle_texture_20261112'
METHODS = ('raw', 'highpass')
SIGMA = 2.0


def lab_image(picture):
    rgb = np.asarray(picture, dtype=np.float32) / 255
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)


def color_metrics(picture, baseline, garment):
    current, anchor = lab_image(picture), lab_image(baseline)
    difference = current[garment].mean(0) - anchor[garment].mean(0)
    return dict(lab_shift=float(np.linalg.norm(difference)), lab_delta=difference.tolist(),
                chroma_shift=float(np.linalg.norm(difference[1:])))


def calibrate_color(picture, baseline, garment, weight):
    """Match garment mean Lab using only the oracle's interior soft mask."""
    target = lab_image(baseline)[garment].mean(0)
    current = picture.copy()
    mean_weight = weight[garment].mean()
    for _ in range(20):
        lab = lab_image(current)
        offset = (target - lab[garment].mean(0)) / mean_weight
        lab += weight[..., None] * offset
        converted = rgb_image(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
        values = np.asarray(current).copy()
        values[weight > 0] = np.asarray(converted)[weight > 0]
        current = Image.fromarray(values)
    return current


def latent_residual(difference, mask, method):
    if method == 'highpass':
        axis = torch.arange(-6, 7, device=difference.device, dtype=difference.dtype)
        kernel = torch.exp(-axis.square() / (2 * SIGMA ** 2))
        kernel /= kernel.sum()
        weights = (kernel[:, None] * kernel[None, :])[None, None]
        weights = weights.expand(difference.shape[1], 1, 13, 13).contiguous()
        blurred = F.conv2d(F.pad(difference, (6, 6, 6, 6), mode='reflect'),
                           weights, groups=difference.shape[1])
        difference = difference - blurred
    return difference * mask


def record(picture, path, baseline, pair, masks, args):
    picture.save(path)
    protection = protects(picture, path, baseline, args.data / pair['sketch'],
                          args.data / pair['gt'], masks)
    return dict(output_sha256=sha(path), stripes=stripes(picture, pair['patch_coordinates']),
                protection=protection,
                color=color_metrics(picture, baseline, masks['garment']))


def select_pairs():
    assert sha(RUN / 'strict_pairs.json') == PAIRS_SHA
    spec = read(RUN / 'protocol.json')
    pairs = [p for p in read(RUN / 'strict_pairs.json')['pairs']
             if p['case_id'] in spec['diagnostic_cases']]
    assert len(pairs) == 8
    return pairs, spec


def prepare(args):
    pairs, source = select_pairs()
    audit = []
    for pair in pairs:
        case = pair['case_id']
        directory = args.out / 'cases' / case
        directory.mkdir(parents=True, exist_ok=True)
        for key in ('sketch', 'gt'):
            assert sha(args.data / pair[key]) == pair['hashes'][key]
        masks = prepare_evaluation_masks(SIZE, sketch_path=str(args.data / pair['sketch']),
                                         mask_policy='sketch_only')
        baseline = Image.open(PRIOR / 'cases' / case / 'baseline.png').convert('RGB')
        source_image = RUN / 'synthetic/outputs/Full' / case / 'v16.png'
        assert sha(source_image) == read(source_image.with_suffix('.json'))['output_sha256']
        assert np.array_equal(np.asarray(baseline), np.asarray(Image.open(source_image)))
        baseline.save(directory / 'baseline.png')
        Image.fromarray(masks['garment'].astype(np.uint8) * 255).save(directory / 'mask.png')
        row = dict(case_id=case, pair=pair, baseline_sha256=sha(directory / 'baseline.png'), inputs={})
        for variant in VARIANTS:
            path = control_path(RUN, pair, variant)
            assert sha(path) == source['controls'][case][variant]
            uncalibrated, weight = oracle_image(baseline, Image.open(path).convert('RGB'), masks['garment'])
            previous_target = PRIOR / 'cases' / case / (variant + '_oracle.png')
            assert np.array_equal(np.asarray(uncalibrated), np.asarray(Image.open(previous_target)))
            calibrated = calibrate_color(uncalibrated, baseline, masks['garment'], weight)
            assert np.array_equal(np.asarray(calibrated)[weight == 0], np.asarray(baseline)[weight == 0])
            target = directory / (variant + '_calibrated.png')
            result = record(calibrated, target, baseline, pair, masks, args)
            assert result['color']['lab_shift'] <= .25, (case, variant, result['color'])
            assert target_hit(result['stripes'], variant), (case, variant, result['stripes'])
            assert all(follow(result['stripes']['bands'], variant, other)
                       for other in VARIANTS if other != variant)
            row['inputs'][variant] = dict(result, uncalibrated_color=color_metrics(
                uncalibrated, baseline, masks['garment']), reference_sha256=sha(path))
        audit.append(row)
    write(args.out / 'protocol.json', dict(
        source_commit=environment()['commit'], pairs_sha256=PAIRS_SHA, checkpoint_sha256=CHECKPOINT_SHA,
        cases=audit, sampler=dict(steps=50, cfg=7., sketch_scale=.6),
        methods=list(METHODS), gains=ARMS, active_steps=list(ACTIVE), sigma_latent_pixels=SIGMA,
        calibration='20 fixed mean-Lab corrections within the same interior soft mask',
        residual='raw=(E(calibrated)-E(M0))*mask; highpass=(D-G_sigma(D))*mask',
        normalization='no amplitude normalization after high-pass filtering',
        color_endpoint='garment mean-Lab displacement relative to paired M0; not reference colour error',
        analysis='paired comparisons on the same eight previously inspected cases; exploratory',
        primary_comparisons='calibrated_raw minus prior; highpass minus calibrated_raw at each fixed gain',
        training_updates=0, GT_used_in_inference=False))
    print('COLOR_PREPARED', len(audit), max(i['color']['lab_shift'] for r in audit for i in r['inputs'].values()), flush=True)


def make_panel(directory):
    pictures = [('M0', directory / 'baseline.png')]
    for variant in VARIANTS:
        pictures.append((variant + ' calibrated', directory / (variant + '_calibrated.png')))
        pictures.append((variant + ' VAE', directory / (variant + '_vae.png')))
        for gain in ARMS:
            for method in METHODS:
                label = variant + ' ' + method + ' ' + gain
                pictures.append((label, directory / (variant + '_' + method + '_' + gain + '.png')))
    canvas = Image.new('RGB', (6 * 150, 4 * 222), 'white')
    draw = ImageDraw.Draw(canvas)
    for index, (label, path) in enumerate(pictures):
        x, y = index % 6 * 150, index // 6 * 222
        draw.text((x + 3, y + 3), label, fill='black')
        with Image.open(path) as picture:
            canvas.paste(picture.convert('RGB').resize((148, 197)), (x, y + 21))
    canvas.save(directory / 'panel.jpg', quality=94)


def generate(args):
    protocol = read(args.out / 'protocol.json')
    assert protocol['source_commit'] == environment()['commit']
    checkpoint = PREVIOUS / 'training/M0/latest.pt'
    assert sha(checkpoint) == protocol['checkpoint_sha256'] == CHECKPOINT_SHA
    core, _ = setup(args)
    pipe, network, _ = load_m(args, core, 'M0')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    load_trainable_strict(network, payload['network'])
    del payload
    prepare_inference(args, pipe, network)
    before = weights_digest(network)
    with torch.inference_mode():
        for item in protocol['cases'][args.shard::2]:
            pair, case = item['pair'], item['case_id']
            directory = args.out / 'cases' / case
            baseline = Image.open(directory / 'baseline.png').convert('RGB')
            assert sha(directory / 'baseline.png') == item['baseline_sha256']
            masks = prepare_evaluation_masks(SIZE, sketch_path=str(args.data / pair['sketch']),
                                             mask_policy='sketch_only')
            features = projected(network, core.tensor(args.data / pair['sketch']).to(args.device).half(),
                                 core.tensor(control_path(RUN, pair, 'v16')).to(args.device).half())
            replay, latent = image(args, core, pipe, network, pair, features)
            old = read(PRIOR / 'cases' / case / 'result.json')
            assert old['seed'] == pair['seed'] and latent == old['latent']
            assert np.array_equal(np.asarray(replay), np.asarray(baseline))
            base_z = encode(pipe.vae, baseline, args.device)
            _, weight = oracle_image(baseline, Image.open(control_path(RUN, pair, 'v16')).convert('RGB'), masks['garment'])
            mask = F.avg_pool2d(torch.from_numpy(weight)[None, None].to(base_z), 8)
            rows = {}
            for variant in VARIANTS:
                target = directory / (variant + '_calibrated.png')
                assert sha(target) == item['inputs'][variant]['output_sha256']
                z = encode(pipe.vae, Image.open(target).convert('RGB'), args.device)
                vae_picture = decode(pipe.vae, z)
                rows[variant] = dict(vae=record(vae_picture, directory / (variant + '_vae.png'),
                                               baseline, pair, masks, args))
                for method in METHODS:
                    delta = latent_residual(z - base_z, mask, method)
                    assert torch.isfinite(delta).all() and delta.square().mean() > 0
                    energy = float(delta.square().mean().sqrt().item())
                    for gain, value in ARMS.items():
                        handle, steps, rms = inject(network.unet, pipe.scheduler, delta, value)
                        try:
                            picture, paired_latent = image(args, core, pipe, network, pair, features)
                        finally:
                            handle.remove()
                        assert paired_latent == latent and len(steps) == 50 and len(rms) == len(ACTIVE)
                        label = method + '_' + gain
                        result = record(picture, directory / (variant + '_' + label + '.png'),
                                        baseline, pair, masks, args)
                        result.update(latent_delta_rms=energy,
                                      guided_delta_rms_max=float(torch.stack(rms).max().item()))
                        rows[variant][label] = result
            make_panel(directory)
            write(directory / 'result.json', dict(case_id=case, seed=pair['seed'],
                  initial_latent_sha256=latent, rows=rows, replay_bitwise=True))
            print('COLOR_CASE', case, flush=True)
    assert before == weights_digest(network)
    write(args.out / ('shard_' + str(args.shard) + '.json'),
          dict(weights_sha256=before, updates=0, source_commit=protocol['source_commit'],
               environment=environment()))


def paired_switch(rows, label, variant):
    count = collections.Counter(spectrum=0, direction_period=0)
    for row in rows:
        anchor, swap = row['v16'][label]['stripes'], row[variant][label]['stripes']
        count['spectrum'] += follow(anchor['bands'], 'v16', variant) and follow(swap['bands'], variant, 'v16')
        if variant == 'h16':
            count['direction_period'] += anchor['vertical'] and swap['horizontal']
        else:
            count['direction_period'] += target_hit(anchor, 'v16') and target_hit(swap, variant)
    return dict(count)


def summarize(args):
    protocol = read(args.out / 'protocol.json')
    audits = [read(args.out / ('shard_' + str(i) + '.json')) for i in range(2)]
    assert audits[0]['weights_sha256'] == audits[1]['weights_sha256']
    cases = [item['case_id'] for item in protocol['cases']]
    current = [read(args.out / 'cases' / case / 'result.json') for case in cases]
    previous = [read(PRIOR / 'cases' / case / 'result.json') for case in cases]
    metrics = ('lab_shift', 'outside_mae', 'iou', 'boundary', 'leak')
    rows = [row['rows'] for row in current]
    results = {}
    for gain in ARMS:
        for method in METHODS:
            label = method + '_' + gain
            score = {variant: paired_switch(rows, label, variant) for variant in ('h16', 'v32')}
            score['protection'] = {}
            for variant in VARIANTS:
                score['protection'][variant] = {}
                for metric in metrics:
                    values = [row[variant][label]['protection'][metric] for row in rows]
                    score['protection'][variant][metric] = dict(mean=float(np.mean(values)),
                         minimum=float(min(values)), maximum=float(max(values)))
            results[label] = score
    comparisons = []
    for gain in ARMS:
        for variant in VARIANTS:
            for metric in metrics:
                raw = [row[variant]['raw_' + gain]['protection'][metric] for row in rows]
                high = [row[variant]['highpass_' + gain]['protection'][metric] for row in rows]
                old = [row['rows'][variant][gain]['protection'][metric] for row in previous]
                comparisons.append(comparison_row('raw_' + gain, 'prior_' + gain, 'all', variant,
                                                  metric, list(zip(raw, old))))
                comparisons.append(comparison_row('highpass_' + gain, 'raw_' + gain, 'all', variant,
                                                  metric, list(zip(high, raw))))
    write(args.out / 'summary.json', dict(cases=cases, results=results, comparisons=comparisons,
                                         updates=0, exploratory=True))
    for item in current:
        assert item['replay_bitwise']
    archive = args.out / 'e2_oracle_color_review.zip'
    files = [args.out / 'protocol.json', args.out / 'summary.json', *args.out.glob('shard_*.json')]
    files += list((args.out / 'cases').glob('*/*'))
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(files):
            if path.is_file():
                bundle.write(path, path.relative_to(args.out))
    print('COLOR_SUMMARY', results, flush=True)
    print('COLOR_ARCHIVE', archive.stat().st_size, sha(archive), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'generate', 'summarize'))
    parser.add_argument('--project', type=Path, default=PROJECT)
    parser.add_argument('--data', type=Path, default=PROJECT.parent / 'datasets/BF')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--shard', type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    globals()[args.action](args)


if __name__ == '__main__':
    main()
