"""Frozen M0 DDIM probe with an explicit garment-aligned texture residual."""
import argparse
import collections
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from scipy.signal import find_peaks

from eval.eval_utils import estimate_foreground_mask, prepare_evaluation_masks
from eval.metrics import compute_texture_leakage
from models.e2_compat import load_trainable_strict
from tools.bfm_skip_probe import (
    CHECKPOINT_SHA, PAIRS_SHA, PREVIOUS, PROJECT, RUN, control_path, follow,
    image, patch_bands, paths, projected,
)
from tools.e2_a_evaluation import boundary_f1
from tools.e2_fft_experiment import read, setup, sha, write
from tools.e2_texture_bottleneck import weights_digest
from tools.m1_evaluation import prepare_inference
from tools.m1_experiment import load_m

VARIANTS = ('v16', 'h16', 'v32')
ARMS = {'gain1': 1.0, 'gain2': 2.0}
ACTIVE = range(25, 45)
SIZE = (384, 512)
RGB_LUMA = np.array([.2126, .7152, .0722], dtype=np.float32)


def rgb_image(array):
    return Image.fromarray(np.rint(255 * np.clip(array, 0, 1)).astype(np.uint8))


def oracle_image(base, reference, garment):
    distance = cv2.distanceTransform(garment.astype(np.uint8), cv2.DIST_L2, 5)
    weight = np.clip((distance - 8) / 8, 0, 1)
    ref = np.asarray(reference, dtype=np.float32) @ RGB_LUMA / 255
    texture = ref - cv2.GaussianBlur(ref, (0, 0), 8)
    texture = (texture - np.sum(texture * weight) / weight.sum()) * (.13 / texture.std())
    src = np.asarray(base, dtype=np.float32) / 255
    soft = weight[..., None]
    inside = garment.astype(np.float32)
    blur = cv2.GaussianBlur(src * inside[..., None], (0, 0), 12)
    smooth = blur / cv2.GaussianBlur(inside, (0, 0), 12).clip(.01)[..., None]
    return rgb_image(src * (1 - soft) + (smooth + texture[..., None]) * soft), weight


def encode(vae, picture, device):
    values = np.asarray(picture, dtype=np.float32).transpose(2, 0, 1) / 127.5 - 1
    tensor = torch.from_numpy(values.copy())[None].to(device=device, dtype=vae.dtype)
    return vae.encode(tensor).latent_dist.mean.float() * vae.config.scaling_factor


def decode(vae, latent):
    pred = vae.decode((latent / vae.config.scaling_factor).to(dtype=vae.dtype)).sample
    return rgb_image(pred[0].permute(1, 2, 0).float().cpu().numpy() / 2 + .5)


def stripes(picture, coords):
    rgb = np.asarray(picture)
    patches = [rgb[y:y+64, x:x+64] for x, y in coords]
    bands = {k: float(np.mean([s[k] for s in map(patch_bands, patches)])) for k in VARIANTS}
    vertical = horizontal = 0
    periods = []
    for patch in patches:
        gray = patch.astype(np.float32) @ RGB_LUMA / 255
        dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)[1:-1, 1:-1]
        dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)[1:-1, 1:-1]
        xx, yy, xy = (dx * dx).mean(), (dy * dy).mean(), (dx * dy).mean()
        strength = np.hypot(xx - yy, 2 * xy) / max(xx + yy, 1e-12)
        angle = abs(np.degrees(.5 * np.arctan2(2 * xy, xx - yy)))
        if strength < .35 or gray.std() < 2 / 255:
            continue
        horizontal += int(angle >= 70)
        if angle > 20:
            continue
        vertical += 1
        profile = gray.mean(axis=0)
        axis = np.arange(64)
        profile = profile - np.polyval(np.polyfit(axis, profile, 1), axis)
        corr = [1.]
        for lag in range(1, 42):
            a, b = profile[:-lag], profile[lag:]
            corr.append(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
        peaks, info = find_peaks(corr, height=.25, prominence=.20)
        picks = [(int(i), float(h)) for i, h in zip(peaks, info['peak_heights']) if 8 <= i <= 40]
        if picks:
            periods.append(next(i for i, h in picks if h >= .8 * max(v for _, v in picks)))
    minimum = (len(patches) + 1) // 2
    return dict(bands=bands, vertical=vertical >= minimum, horizontal=horizontal >= minimum,
                period=float(np.median(periods)) if len(periods) >= minimum else None)


def protects(picture, file, baseline, sketch, gt, masks):
    garment, outside = masks['garment'], masks['outside']
    pred = estimate_foreground_mask(picture, SIZE)
    iou = (pred & garment).sum() / (pred | garment).sum()
    rgb = np.asarray(picture, dtype=np.float32) / 255
    other = np.asarray(baseline, dtype=np.float32) / 255
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[garment].mean(axis=0)
    base_lab = cv2.cvtColor(other, cv2.COLOR_RGB2LAB)[garment].mean(axis=0)
    leak = compute_texture_leakage(gen_path=str(file), sketch_path=str(sketch),
                                  target_path=str(gt), mask_bundle=masks)['leak_colored_frac']
    return dict(iou=float(iou), boundary=boundary_f1(pred, garment, 2), leak=float(leak),
                outside_mae=float(np.abs(rgb[outside] - other[outside]).mean()),
                lab_shift=float(np.linalg.norm(lab - base_lab)))


def inject(unet, scheduler, delta, gain):
    steps, rms = [], []
    alpha = scheduler.alphas_cumprod.to(device=delta.device, dtype=torch.float32)

    def hook(_, args, kwargs, result):
        if 'sa_hidden_states' not in (kwargs.get('cross_attention_kwargs') or {}):
            return None
        index = len(steps)
        steps.append(args[1])
        if index not in ACTIVE:
            return None
        a = alpha[args[1]]
        # Match a clean-latent offset in the guided epsilon, compensating CFG=7.
        guided = -gain * (a / (1 - a)).sqrt() * delta
        rms.append(guided.square().mean().sqrt())
        return ((result[0].float() + guided / 7).to(result[0].dtype), *result[1:])

    return unet.register_forward_hook(hook, with_kwargs=True), steps, rms


def probe_case(args, core, pipe, network, pair, hashes):
    case = pair['case_id']
    directory = args.out / 'cases' / case
    directory.mkdir(parents=True, exist_ok=True)
    sketch = args.data / pair['sketch']
    assert sha(sketch) == pair['hashes']['sketch']
    masks = prepare_evaluation_masks(SIZE, sketch_path=str(sketch), mask_policy='sketch_only')
    coords = pair['patch_coordinates']
    assert len(coords) >= 4 and masks['garment'].any()
    assert sha(control_path(RUN, pair, 'v16')) == hashes['v16']
    features = projected(network, core.tensor(sketch).to(args.device).half(),
                         core.tensor(control_path(RUN, pair, 'v16')).to(args.device).half())
    baseline, latent = image(args, core, pipe, network, pair, features)
    baseline_file = directory / 'baseline.png'
    baseline.save(baseline_file)
    previous = paths(RUN / 'synthetic', 'Full', pair, 'v16')
    previous_meta = read(previous.with_suffix('.json'))
    assert sha(previous) == previous_meta['output_sha256'] and latent == previous_meta['initial_latent_sha256']
    assert np.array_equal(np.asarray(baseline), np.asarray(Image.open(previous).convert('RGB')))
    base_z = encode(pipe.vae, baseline, args.device)
    gt = args.data / pair['gt']
    assert sha(gt) == pair['hashes']['gt']
    base_metrics = protects(baseline, baseline_file, baseline, sketch, gt, masks)
    rows = {'baseline': dict(stripes=stripes(baseline, coords), protection=base_metrics)}
    tiles = [('M0 replay', baseline)]
    for variant in VARIANTS:
        ref = control_path(RUN, pair, variant)
        assert sha(ref) == hashes[variant]
        target, mask = oracle_image(baseline, Image.open(ref).convert('RGB'), masks['garment'])
        target_z = encode(pipe.vae, target, args.device)
        vae_target = decode(pipe.vae, target_z)
        target.save(directory / f'{variant}_oracle.png')
        vae_target.save(directory / f'{variant}_vae.png')
        rows[variant] = dict(oracle=stripes(target, coords), vae=stripes(vae_target, coords))
        tiles += [(variant + ' oracle', target), (variant + ' VAE', vae_target)]
        latent_mask = torch.nn.functional.avg_pool2d(torch.from_numpy(mask)[None, None].to(target_z), 8)
        delta = (target_z - base_z) * latent_mask
        for arm, gain in ARMS.items():
            handle, steps, rms = inject(network.unet, pipe.scheduler, delta, gain)
            try:
                picture, paired_latent = image(args, core, pipe, network, pair, features)
            finally:
                handle.remove()
            assert latent == paired_latent and len(steps) == 50 and len(rms) == len(ACTIVE)
            maximum = float(torch.stack(rms).max().item())
            assert np.isfinite(maximum) and maximum > 0
            file = directory / f'{variant}_{arm}.png'
            picture.save(file)
            rows[variant][arm] = dict(stripes=stripes(picture, coords), output_sha256=sha(file),
                                      protection=protects(picture, file, baseline, sketch, gt, masks),
                                      guided_delta_rms_max=maximum)
            tiles.append((variant + ' ' + arm, picture))
    panel = Image.new('RGB', (5 * 116, 3 * 174), 'white')
    draw = ImageDraw.Draw(panel)
    for index, (label, picture) in enumerate(tiles):
        x, y = index % 5 * 116, index // 5 * 174
        draw.text((x + 3, y + 3), label, fill='black')
        panel.paste(picture.resize((114, 152)), (x, y + 18))
    panel.save(directory / 'panel.jpg', quality=93)
    write(directory / 'result.json', dict(case_id=case, seed=pair['seed'], latent=latent, rows=rows))


def generate(args, pairs, spec):
    checkpoint = PREVIOUS / 'training/M0/latest.pt'
    assert sha(checkpoint) == spec['checkpoint_sha256'] == CHECKPOINT_SHA
    core, _ = setup(args)
    pipe, network, _ = load_m(args, core, 'M0')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    load_trainable_strict(network, payload['network'])
    del payload
    prepare_inference(args, pipe, network)
    assert pipe.scheduler.config.prediction_type == 'epsilon'
    weights = weights_digest(network)
    with torch.inference_mode():
        for pair in pairs[args.shard::2]:
            probe_case(args, core, pipe, network, pair, spec['controls'][pair['case_id']])
            print('ORACLE_CASE', pair['case_id'], flush=True)
    assert weights == weights_digest(network)
    write(args.out / f'shard_{args.shard}.json', dict(weights=weights, updates=0, steps=list(ACTIVE)))


def target_hit(metrics, variant):
    if variant == 'h16':
        return metrics['horizontal']
    desired = int(variant[1:])
    return metrics['vertical'] and metrics['period'] is not None and abs(metrics['period'] / desired - 1) <= .25


def summarize(args, pairs):
    audits = [read(args.out / f'shard_{i}.json') for i in range(2)]
    assert audits[0] == audits[1]
    rows = [read(args.out / 'cases' / p['case_id'] / 'result.json')['rows'] for p in pairs]
    result = {}
    for label in ('oracle', 'vae', *ARMS):
        def metrics(row, variant):
            record = row[variant][label]
            return record['stripes'] if label in ARMS else record
        score = {}
        for variant in ('h16', 'v32'):
            count = collections.Counter()
            for row in rows:
                anchor, swap = metrics(row, 'v16'), metrics(row, variant)
                count['spectrum'] += follow(anchor['bands'], 'v16', variant) and follow(swap['bands'], variant, 'v16')
                count['direction_period'] += target_hit(anchor, 'v16') and target_hit(swap, variant) if variant == 'v32' else anchor['vertical'] and swap['horizontal']
            score[variant] = dict(count)
        result[label] = score
        if label in ARMS:
            limits = dict(iou=-.01, boundary=-.02, leak=.005, outside_mae=.01, lab_shift=1.)
            protections = {}
            for variant in VARIANTS:
                measures = {}
                for metric, limit in limits.items():
                    deltas = [row[variant][label]['protection'][metric] - row['baseline']['protection'][metric] for row in rows]
                    mean = float(np.mean(deltas))
                    passed = mean >= limit if limit < 0 else mean <= limit
                    measures[metric] = dict(mean=mean, min=min(deltas), max=max(deltas), passed=bool(passed))
                protections[variant] = measures
            score['protection'] = protections
            score['passed'] = bool(all(n >= 6 for v in ('h16', 'v32') for n in score[v].values())
                                   and all(m['passed'] for v in protections.values() for m in v.values()))
    write(args.out / 'summary.json', dict(cases=len(rows), results=result, audit=audits[0]))
    print('ORACLE_SUMMARY', result, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('generate', 'summarize'))
    parser.add_argument('--project', type=Path, default=PROJECT)
    parser.add_argument('--data', type=Path, default=PROJECT.parent / 'datasets/BF')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--shard', type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    spec = read(RUN / 'protocol.json')
    manifest = RUN / 'strict_pairs.json'
    assert sha(manifest) == PAIRS_SHA
    pairs = [p for p in read(manifest)['pairs'] if p['case_id'] in spec['diagnostic_cases']]
    assert len(pairs) == 8
    if args.action == 'generate':
        generate(args, pairs, spec)
    else:
        summarize(args, pairs)


if __name__ == '__main__':
    main()
