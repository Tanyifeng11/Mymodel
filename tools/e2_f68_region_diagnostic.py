"""只读拆解 f68 真实原配对的 LPIPS、SSIM 和外观差异。"""
import argparse
import os
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity
import torch
from torch.nn import functional as F

from eval.eval_utils import estimate_foreground_mask
from tools.e2_a_evaluation import boundary_f1
from tools.e2_f68_real_paired import ARMS, SIZE, metric_assets, metric_models, paired_interval, rgb, verify, write
from tools.e2_fft_experiment import environment, read, sha


RADII = (4, 8, 16)
REGIONS = ('inside', 'edge', 'background')
SWAPS = ('inside_only', 'edge_only', 'background_only')
COAT_EDITS = ('restore_inside', 'anchor_l', 'anchor_ab', 'anchor_l_ab', 'anchor_amp')
ARCHIVE_SHA = '524cc6bf74e48329af2044a6bf016d16473ad896e6a5cf969183bd62199c8927'


def kernel(radius):
    diameter = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))


def regions(mask, radius):
    u8 = mask.astype('uint8')
    inside = cv2.erode(u8, kernel(radius)).astype(bool)
    exterior = ~cv2.dilate(u8, kernel(radius)).astype(bool)
    return np.stack([inside, ~(inside | exterior), exterior])


def gaussian(value, sigma):
    return cv2.GaussianBlur(value, (0, 0), sigma)


def lab(image):
    return cv2.cvtColor(image.astype('float32') / 255, cv2.COLOR_RGB2LAB)


def as_rgb(value):
    value = value.copy()
    value[..., 0] = np.clip(value[..., 0], 0, 100)
    return np.rint(np.clip(cv2.cvtColor(value, cv2.COLOR_LAB2RGB), 0, 1) * 255).astype('uint8')


def apply_lab_edit(original, roundtrip, corrected):
    delta = as_rgb(corrected).astype('int16') - roundtrip
    return np.clip(original.astype('int16') + delta, 0, 255).astype('uint8')


def band(value):
    return gaussian(value, 1) - gaussian(value, 4)


def rms(value):
    return float(np.sqrt(np.square(value, dtype='float64').mean()))


def appearance(x, target, region):
    if not region.any():
        return dict.fromkeys(('l_bias', 'chroma_bias', 'low_l_error', 'low_ab_error',
                              'l_rms', 'gt_l_rms', 'l_log_ratio', 'l_abs_log_ratio', 'ab_rms'))
    current, low, texture = x
    gt, gt_low, gt_texture = target
    current, gt = current[region], gt[region]
    low_diff = low[region] - gt_low[region]
    s = rms(texture[region, 0])
    t = rms(gt_texture[region, 0])
    ratio = float(np.log((s + .25) / (t + .25)))
    return dict(l_bias=float((current[:, 0] - gt[:, 0]).mean()),
                chroma_bias=float((np.linalg.norm(current[:, 1:], axis=1)
                                   - np.linalg.norm(gt[:, 1:], axis=1)).mean()),
                low_l_error=float(np.abs(low_diff[:, 0]).mean()),
                low_ab_error=float(np.linalg.norm(low_diff[:, 1:], axis=1).mean()),
                l_rms=s, gt_l_rms=t, l_log_ratio=ratio, l_abs_log_ratio=abs(ratio),
                ab_rms=rms(np.linalg.norm(texture[region, 1:], axis=1)))


def prepare_pixels(images, mask, outwear):
    roi = regions(mask, 8)
    weights = np.stack([gaussian(which.astype('float32'), 2) for which in roi])
    assert np.allclose(weights.sum(axis=0), 1, atol=1e-6)
    off, on = (images[a].astype('float32') for a in ARMS)
    d = on - off
    edited = {name: np.rint(off + fade[..., None] * d).astype('uint8')
              for name, fade in zip(SWAPS, weights)}
    appearance_inputs = {}
    for name, picture in images.items():
        value = lab(picture)
        appearance_inputs[name] = (value, gaussian(value, 8), band(value))
    core = cv2.erode(mask.astype('uint8'), kernel(20)).astype(bool)
    if outwear:
        on_lab, on_low, on_band = appearance_inputs['on']
        _, off_low, off_band = appearance_inputs['off']
        roundtrip = as_rgb(on_lab).astype('int16')
        core_weight = gaussian(core.astype('float32'), 2)
        interior = weights[0, ..., None]
        low_diff = interior * (on_low - off_low)
        edited['restore_inside'] = np.rint(on - interior * d).astype('uint8')
        for name, channels in (('anchor_l', slice(0, 1)), ('anchor_ab', slice(1, 3)), ('anchor_l_ab', slice(0, 3))):
            corrected = on_lab.copy()
            corrected[..., channels] -= low_diff[..., channels]
            edited[name] = apply_lab_edit(images['on'], roundtrip, corrected)
        corrected = on_lab.copy()
        if core.any():
            gain = np.clip((rms(off_band[core, 0]) + .25) / (rms(on_band[core, 0]) + .25), .5, 2.)
            corrected[..., 0] += (gain - 1) * core_weight * on_band[..., 0]
        edited['anchor_amp'] = apply_lab_edit(images['on'], roundtrip, corrected)
    return dict(**images, **edited), appearance_inputs, core


def tensors(images):
    return torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous().float() / 255


def lpips_features(metric, batch):
    from lpips import normalize_tensor
    return [normalize_tensor(layer) for layer in metric.net(metric.scaling_layer(batch * 2 - 1))]


def maps(metric, reference, batch):
    return [head((value - target).square()).squeeze(1)
            for head, value, target in zip(metric.lins, lpips_features(metric, batch), reference)]


def apportion(layer_maps, roi):
    pieces = 0
    for heat in layer_maps:
        weight = F.interpolate(roi, size=heat.shape[-2:], mode='area')
        pieces = pieces + (weight * heat[:, None]).mean(dim=(-2, -1))
    return pieces.numpy()


def ssim_regions(image, gt, roi):
    score, heat = structural_similarity(image, gt, channel_axis=2, data_range=255, full=True)
    heat = heat[3:-3, 3:-3].mean(axis=2)
    trimmed = roi[:, 3:-3, 3:-3]
    contribution = (trimmed * heat).mean(axis=(1, 2))
    assert abs(float(contribution.sum()) - score) < 1e-10
    return float(score), contribution.tolist(), trimmed.mean(axis=(1, 2)).tolist()


def shape_score(image, mask):
    predicted = estimate_foreground_mask(Image.fromarray(image), SIZE)
    if predicted is None:
        predicted = np.zeros_like(mask)
    return dict(contour_f1=boundary_f1(predicted, mask, 3),
                iou=float((predicted & mask).sum() / (predicted | mask).sum()))


def checked_case(baseline, data, pair, recorded):
    case = pair['case_id']
    folder = baseline / 'cases' / case
    gt = data / pair['gt']
    mask = folder / 'mask.png'
    assert sha(gt) == pair['input_sha256']['gt'] and sha(mask) == pair['mask_sha256'], case
    images = dict(gt=np.asarray(rgb(gt).resize(SIZE, Image.Resampling.BILINEAR)))
    for arm in ARMS:
        path = baseline / 'images' / arm / (case + '.png')
        assert sha(path) == recorded['arms'][arm]['png_sha256'], path
        images[arm] = np.asarray(rgb(path))
    assert all(image.shape == (SIZE[1], SIZE[0], 3) for image in images.values())
    return images, np.asarray(rgb(mask))[..., 0] > 0


def paired(before, after, categories):
    before, after = np.array(before, dtype='float64'), np.array(after, dtype='float64')
    finite = np.isfinite(before) & np.isfinite(after)
    before, after, categories = before[finite], after[finite], categories[finite]
    assert len(before)
    diff = after - before
    return dict(n=len(diff), off=float(before.mean()), on=float(after.mean()), delta=float(diff.mean()),
                ci95=paired_interval(diff, categories))


def fid_terms(x, gt):
    x, gt = x.astype('float64'), gt.astype('float64')
    center, target = x.mean(0), gt.mean(0)
    x, gt = ((x - center) / np.sqrt(len(x) - 1), (gt - target) / np.sqrt(len(gt) - 1))
    mean = float(np.square(center - target).sum())
    cov = float(np.square(x).sum() + np.square(gt).sum() - 2 * np.linalg.svd(x @ gt.T, compute_uv=False).sum())
    return dict(fid=mean + cov, mean_term=mean, covariance_term=cov, n=len(x))


def summarize(rows, features):
    result = {}
    labels = np.array([r['category'] for r in rows])
    for group, select in (('all', np.ones(len(rows), bool)), ('outwear', labels == 'outwear')):
        subset = [r for r, keep in zip(rows, select) if keep]
        categories = labels[select]
        variants = (ARMS + SWAPS + COAT_EDITS) if group == 'outwear' else (ARMS + SWAPS)
        stats = dict(n=len(subset), variants={}, fid={}, region_metrics={}, radius_lpips={}, amp_core=None)
        for name in variants:
            common = tuple(subset[0]['scores'][name])
            stats['variants'][name] = {k: paired([r['scores']['off'][k] for r in subset],
                [r['scores'][name][k] for r in subset], categories) for k in common}
            value = features[name][select] if name in ARMS + SWAPS else features[name]
            stats['fid'][name] = fid_terms(value, features['gt'][select])
        for region_id, name in enumerate(REGIONS):
            def scores(arm, key):
                return [r['regional'][arm][name][key] for r in subset]
            stats['region_metrics'][name] = {key: paired(scores('off', key), scores('on', key), categories)
                for key in subset[0]['regional']['off'][name]}
            for radius in RADII:
                scale = stats['radius_lpips'].setdefault(str(radius), {})
                scale[name] = paired(*([r['radius_lpips'][arm][str(radius)][region_id] for r in subset]
                    for arm in ARMS), categories)
        valid_amp = [r for r in subset if r['amp_core']]
        if valid_amp:
            stats['amp_core'] = {key: paired(*([r['amp_core'][arm][key] for r in valid_amp] for arm in ARMS),
                np.array([r['category'] for r in valid_amp])) for key in valid_amp[0]['amp_core']['off']}
        result[group] = stats
    return result


def examples(out, baseline, data, pairs, previous, rows):
    coats = sorted((r for r in rows if r['category'] == 'outwear'),
                   key=lambda r: r['scores']['on']['lpips'] - r['scores']['off']['lpips'])
    labels = ('GT', 'B off', 'B on', 'Edge only', 'L + ab anchor')
    sheet = Image.new('RGB', (5 * 168, 3 * 246), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate((coats[0], coats[len(coats)//2], coats[-1])):
        j = row['index']
        originals, mask = checked_case(baseline, data, pairs[j], previous[j])
        combinations, _, _ = prepare_pixels(originals, mask, True)
        draw.text((3, i * 246 + 3), row['case_id'], fill='black')
        for col, (key, label) in enumerate(zip(('gt', 'off', 'on', 'edge_only', 'anchor_l_ab'), labels)):
            value = Image.fromarray(combinations[key]).resize((156, 208), Image.Resampling.BILINEAR)
            sheet.paste(value, (col * 168, i * 246 + 20))
            draw.text((col * 168 + 3, i * 246 + 229), label, fill='black')
    sheet.save(out / 'outwear_examples.png')


@torch.no_grad()
def run(prior):
    baseline = prior / 'real_paired_B_on_off_20261118'
    out = baseline / 'region_diagnostic_20261118'
    out.mkdir(exist_ok=True)
    assert not (out / 'review.zip').exists(), '诊断已归档'
    assert sha(baseline / 'review.zip') == ARCHIVE_SHA
    original_manifest = read(baseline / 'review_manifest.json')
    for name in ('protocol.json', 'per_image.json', 'fid_pool3.npz', 'result.json'):
        assert sha(baseline / name) == original_manifest[name]
    frozen = verify(argparse.Namespace(out=baseline))
    assert metric_assets() == frozen['metric_assets']
    data = prior.parents[2] / 'datasets/BF'
    pairs = frozen['pairs']
    previous = read(baseline / 'per_image.json')
    assert len(previous) == len(pairs) == 500
    metric, inception = metric_models('cpu', frozen['metric_assets'])
    with np.load(baseline / 'fid_pool3.npz', allow_pickle=False) as archive:
        features = dict(archive)
    new_features = {name: [] for name in SWAPS + COAT_EDITS}
    rows = []
    ranges = [(start, min(start+4, first+125)) for first in range(0, len(pairs), 125)
              for start in range(first, first+125, 4)]
    for start, end in ranges:
        selection = pairs[start:end]
        outwear = selection[0]['category'] == 'outwear'
        assert all((p['category'] == 'outwear') == outwear for p in selection)
        batch, masks, colors, cores = [], [], [], []
        for i, pair in enumerate(selection, start):
            recorded = previous[i]
            assert recorded['index'] == i and recorded['case_id'] == pair['case_id']
            raw, mask = checked_case(baseline, data, pair, recorded)
            images, appearance_inputs, core = prepare_pixels(raw, mask, outwear)
            batch.append(images)
            masks.append(mask)
            colors.append(appearance_inputs)
            cores.append(core)
        roi = {str(r): np.stack([regions(mask, r) for mask in masks]) for r in RADII}
        roi_tensors = {radius: torch.from_numpy(value.astype('float32')) for radius, value in roi.items()}
        target = tensors([r['gt'] for r in batch])
        ref_features = lpips_features(metric, target)
        if start == 0:
            equal_backend = inception(target)[0].flatten(1).numpy()
            np.testing.assert_allclose(equal_backend, features['gt'][:len(selection)], rtol=1e-4, atol=7e-5)
        scores = [dict() for _ in batch]
        partition = [dict() for _ in batch]
        variants = ARMS + SWAPS + (COAT_EDITS if outwear else ())
        for name in variants:
            tensor = tensors([r[name] for r in batch])
            layer_maps = maps(metric, ref_features, tensor)
            values = sum(layer.mean(dim=(-2, -1)) for layer in layer_maps).numpy()
            if name not in ARMS:
                new_features[name].append(inception(tensor)[0].flatten(1).numpy())
            pieces = {radius: apportion(layer_maps, mask) for radius, mask in roi_tensors.items()} if name in ARMS else {}
            for j, value in enumerate(values):
                image, gt = batch[j][name], batch[j]['gt']
                result = dict(lpips=float(value))
                if name in ARMS:
                    baseline_score = previous[start+j]['scores'][name]
                    assert abs(value - baseline_score['lpips']) < 5e-6
                    for v in pieces.values():
                        assert abs(float(v[j].sum()) - value) < 1e-6
                    partition[j][name] = {radius: v[j].tolist() for radius, v in pieces.items()}
                    result.update({k: baseline_score[k] for k in ('ssim', 'contour_f1', 'iou')})
                elif outwear:
                    result['ssim'] = float(structural_similarity(image, gt, channel_axis=2, data_range=255))
                    result.update(shape_score(image, masks[j]))
                scores[j][name] = result
        for j, pair in enumerate(selection):
            regional, amp = {}, {}
            for arm in ARMS:
                score, pieces, fraction = ssim_regions(batch[j][arm], batch[j]['gt'], roi['8'][j])
                assert abs(score - scores[j][arm]['ssim']) < 1e-8
                regional[arm] = {}
                for r, region in enumerate(REGIONS):
                    stats = appearance(colors[j][arm], colors[j]['gt'], roi['8'][j][r])
                    stats.update(ssim_share=pieces[r], lpips_share=partition[j][arm]['8'][r], ssim_area=fraction[r])
                    regional[arm][region] = stats
                if cores[j].sum() >= 256:
                    amp[arm] = appearance(colors[j][arm], colors[j]['gt'], cores[j])
            rows.append(dict(index=start+j, case_id=pair['case_id'], category=pair['category'],
                             scores=scores[j], radius_lpips=partition[j], regional=regional, amp_core=amp))
        if len(rows) % 20 < 4:
            print('REGION_PAIRS', len(rows), flush=True)
    assert len(rows) == 500
    new_features = {name: np.concatenate(v) for name, v in new_features.items()}
    assert all(x.shape == (500 if name in SWAPS else 125, 2048) for name,x in new_features.items())
    result = summarize(rows, {**features, **new_features})
    old = read(baseline / 'result.json')
    for arm in ARMS:
        assert abs(result['all']['fid'][arm]['fid'] - old['fid'][arm]) < 1e-8
    metadata = dict(parent_archive_sha256=ARCHIVE_SHA, frozen_protocol_sha256=sha(baseline/'protocol.json'),
                    source_sha256=sha(Path(__file__)), environment=environment(),
                    images_read=1000, model_generations=0, optimizer_updates=0,
                    roi_radii_px=RADII, feather_sigma=2, lowpass_sigma=8, bandpass_sigmas=[1,4],
                    amp_core_erosion_radius=20, outwear_only_edits=COAT_EDITS)
    write(out / 'meta.json', metadata)
    write(out / 'per_pair.json', rows)
    write(out / 'result.json', result)
    np.savez_compressed(out / 'edit_pool3.npz', **new_features)
    examples(out, baseline, data, pairs, previous, rows)
    names = ('meta.json', 'per_pair.json', 'result.json', 'edit_pool3.npz', 'outwear_examples.png')
    write(out / 'review_manifest.json', {name: sha(out/name) for name in names})
    temp = out/'review.zip.tmp'
    with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in (*names, 'review_manifest.json'):
            archive.write(out/name, name)
    temp.replace(out/'review.zip')
    print('REGION_COMPLETE', len(rows), sha(out/'review.zip'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prior', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))
    cv2.setNumThreads(1)
    run(args.prior)
