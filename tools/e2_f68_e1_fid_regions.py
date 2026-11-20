"""只读复用原 64 对图像，测量区域替换对 Inception 分布的影响。"""
import argparse
import os
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from PIL import Image

from tools import e2_f68_e1_a_fused25_continue as fused
from tools import e2_f68_e1_a_fused25_lstar_ab as light
from tools import e2_f68_region_diagnostic as roi
from tools import e2_f68_tail6_real_ab as tail


SOURCE_SHA = 'f6d00513f127932070d9fca943258078eb28a3247b8d3451ce9289b608cdbe3f'
RUN = 'fid_regions_64_20261121'
COHORT = 'tail6_real_ab_e1_epoch10_20261119'
BASE_ARMS = ('offline25', 'continued_A', 'B')
REGION_B_SHA = '40cbbe5df7cad9234ab64763acf4bc2020da951a943633dcbf7fea5d8361cac7'
RETENTION_RUN = 'retention_125_250'
RETENTION = {'retain_125': .125, 'retain_250': .25}


def checked_png(path, digest):
    assert tail.sha(path) == digest, path
    return np.asarray(tail.rgb(path))


def regions(mask):
    return tuple(roi.gaussian(a.astype('float32'), 2) for a in roi.regions(mask, 8))


def interior_probes(image, offline, mask, interior):
    """Keep the original RGB where the intervention has zero support."""
    lab = roi.lab(image)
    src = roi.lab(offline)
    core = roi.regions(mask, 8)[0].astype('float32')
    low = roi.gaussian((src[..., 0] - lab[..., 0]) * core, 8)
    low /= np.maximum(roi.gaussian(core, 8), 1e-6)
    shifted = lab.copy()
    shifted[..., 0] = np.clip(lab[..., 0] + low, 0, 100)
    before = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    after = cv2.cvtColor(shifted, cv2.COLOR_LAB2RGB)
    delta_l = (after - before) * 255
    delta_other = offline.astype('float32') - image - delta_l
    deep = np.minimum(interior, roi.gaussian(roi.regions(mask, 20)[0].astype('float32'), 2))
    return dict(
        low_L_from_offline=fused.pixels_mixed(image, image + delta_l, interior),
        other_from_offline=fused.pixels_mixed(image, image + delta_other, interior),
        other_half=fused.pixels_mixed(image, image + delta_other, .5 * interior),
        interior_half=fused.pixels_mixed(image, offline, .5 * interior),
        deep_from_offline=fused.pixels_mixed(image, offline, deep),
        inner_rim_from_offline=fused.pixels_mixed(image, offline, interior - deep),
    )


def cases(args, matched, rows):
    for row in rows:
        idx, case = row['index'], row['case_id']
        pair = matched['pairs'][idx]
        assert pair['case_id'] == case and pair['category'] == row['category']
        mask = tail.case_mask(args, pair)
        target_path = args.data / pair['gt']
        assert tail.sha(target_path) == pair['input_sha256']['gt']
        target = tail.rgb(target_path).resize(tail.SIZE, Image.Resampling.BILINEAR)
        train_base = checked_png(args.parent / 'test/A' / (case + '.png'), row['pngs']['A'])
        donor = checked_png(args.parent / 'test/blend' / (case + '.png'), row['pngs']['blend'])
        images = {
            'offline25': fused.pixels_mixed(train_base, donor, fused.alpha(mask)),
            'continued_A': checked_png(args.parent / fused.RUN / 'images/eval64' / (case + '.png'),
                                       row['pngs']['continued_A']),
            'B': checked_png(args.parent / light.RUN / 'images/eval64' / (case + '.png'),
                             row['pngs']['B']),
        }
        yield row, mask, target, target_path, images


def edits(images, mask, carrier):
    x, y = images[carrier], images['offline25']
    ws = regions(mask)
    np.testing.assert_allclose(sum(ws), 1, atol=1e-6, rtol=0)
    images = dict(images)
    for bits in range(1, 7):
        weight = sum(ws[j] for j in range(3) if bits & (1 << j))
        images[f'swap_{bits}'] = fused.pixels_mixed(x, y, weight)
    if carrier == 'B':
        images.update(interior_probes(x, y, mask, ws[0]))
    return images


def retention_edits(images, mask, _carrier):
    before, offline = images['B'], images['offline25']
    inside = regions(mask)[0]
    result = {k: fused.pixels_mixed(before, offline, (1-value)*inside)
              for k, value in RETENTION.items()}
    for image in result.values():
        assert np.array_equal(image[inside == 0], before[inside == 0])
    return result


def summarize_retention(source, previous, measured, arrays):
    controls = {'offline25': 'offline25', 'retain_100': 'B'}
    regional = {'retain_0': 'swap_1', 'retain_50': 'interior_half'}
    history = tail.read(source / 'per_pair.json')
    old_region = tail.read(previous / 'per_pair.json')
    for name, path, mapping in (('source', source, controls), ('regions', previous, regional)):
        with np.load(path / 'pool3.npz', allow_pickle=False) as saved:
            arrays.update({arm: saved[key] for arm, key in mapping.items()})
            if name == 'source':
                arrays['gt'] = saved['gt']
    assert len(measured) == len(history) == len(old_region) == 64
    for new, a, b in zip(measured, history, old_region):
        for key in ('index', 'case_id', 'category', 'interior_px'):
            assert new[key] == a[key] == b[key]
        for name in ('scores', 'texture'):
            new[name].update({arm: a[name][key] for arm, key in controls.items()})
            new[name].update({arm: b[name][key] for arm, key in regional.items()})
    arms = ('offline25', 'retain_0', *RETENTION, 'retain_50', 'retain_100')
    assert all(x.shape == (64, 2048) and np.isfinite(x).all() for x in arrays.values())
    result = tail.summarize(measured, arrays, arms, ('offline25',),
                            arms[1:], fid_reference='offline25')
    original = tail.read(source / 'result.json')
    for label, group in result['groups'].items():
        valid = [row for row in measured if row['interior_px'] >= 256
                 and (label == 'all' or row['category'] == label)]
        group['texture_count'] = len(valid)
        group['texture'] = {arm: {key: float(np.mean([r['texture'][arm][key] for r in valid]))
                                  for key in valid[0]['texture'][arm]} for arm in arms}
    group = result['groups']['all']
    truth = original['groups']['all']
    limits = {key: v['threshold'] for key, v in original['acceptance']['B']['seven']['checks'].items()}
    band_limit = min(truth['texture'][key]['band_mae'] for key in ('off', 'on'))
    acceptance = {}
    for arm in arms[1:]:
        scores = group['arms'][arm]
        checks = {}
        for key, limit in limits.items():
            value = scores[key]['fid' if key == 'fid' else 'mean']
            checks[key] = bool(value < limit if key == 'fid' else
                               value <= limit if key in ('lpips', 'leak') else value >= limit)
        band = group['texture'][arm]['band_mae'] < band_limit
        delta = {key: scores[key]['vs_offline25'] if key == 'fid' else
                 scores[key]['mean']-group['arms']['offline25'][key]['mean']
                 for key in limits}
        target = delta['lpips'] < 0 and delta['fid'] < 0 and delta['ssim'] >= 0 and delta['contour_f1'] >= 0
        acceptance[arm] = dict(seven=checks, band=band, delta_vs_offline25=delta,
                               passed=all(checks.values()) and band and target)
    result.update(acceptance=acceptance, thresholds=limits, band_threshold=band_limit)
    return result


def pack(out, names):
    manifest = {name: tail.sha(out / name) for name in names}
    tail.write(out / 'review_manifest.json', manifest)
    tmp = out / 'review.tmp'
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in (*names, 'review_manifest.json'):
            archive.write(out / name, name)
    tmp.replace(out / 'review.zip')


@torch.inference_mode()
def run(prior, carrier, retention=False):
    base = prior / tail.distill.BASE
    parent = base / COHORT
    source = parent / light.RUN / 'eval64'
    out = parent / light.RUN / RUN / carrier
    previous = out
    if retention:
        out = out.parent / RETENTION_RUN
        tail.verified_files(previous, REGION_B_SHA, ('per_pair.json', 'pool3.npz', 'audit.json'))
    out.mkdir(exist_ok=True, parents=True)
    args = SimpleNamespace(out=base, data=prior.parents[2] / 'datasets/BF', parent=parent)
    tail.verified_files(source, SOURCE_SHA, ('per_pair.json', 'pool3.npz', 'result.json'))
    protocol = tail.verify(args)
    rows = tail.read(source / 'per_pair.json')
    assert [r['index'] for r in rows] == tail.subset(protocol)
    assert tail.metric_assets() == protocol['metric_assets']
    lpips, inception = tail.metric_models('cpu', protocol['metric_assets'])
    feature_chunks, measured = {}, []
    originals = np.load(source / 'pool3.npz', allow_pickle=False)
    records = cases(args, protocol, rows)
    make_edits = retention_edits if retention else edits
    for start in range(0, len(rows), 4):
        group = [next(records) for _ in range(min(4, len(rows) - start))]
        batch = {}
        with tempfile.TemporaryDirectory(prefix='fid_regions_', dir=os.environ.get('SLURM_TMPDIR')) as temp:
            for row, mask, gt, target_path, images in group:
                deep = roi.regions(mask, 20)[0]
                truth_lab = roi.lab(np.asarray(gt))
                item = {k: row[k] for k in ('index', 'case_id', 'category', 'interior_px')}
                item.update(scores={}, texture={}, deltas={})
                for region_name, region in zip(('inside', 'edge', 'background'), roi.regions(mask, 8)):
                    diff = (images[carrier].astype('float32') - images['offline25'])[region]
                    item['deltas'][region_name] = dict(
                        pixels=len(diff), mae=float(np.abs(diff).mean()) if len(diff) else None,
                        changed=float(np.any(diff != 0, axis=1).mean()) if len(diff) else None)
                for name, img in make_edits(images, mask, carrier).items():
                    batch.setdefault(name, []).append(img)
                    if name in BASE_ARMS:
                        continue
                    path = Path(temp) / 'current.png'
                    picture = Image.fromarray(img)
                    if retention:
                        path = out / 'images' / name / (row['case_id'] + '.png')
                        fused.save_png(path, img)
                        item.setdefault('pngs', {})[name] = tail.sha(path)
                    else:
                        picture.save(path)
                    item['scores'][name] = tail.image_scores(picture, gt, path, target_path, mask)
                    item['texture'][name] = tail.appearance(img, truth_lab, deep)
                measured.append(item)
        gt = roi.tensors([np.asarray(pair[2]) for pair in group])
        for name, images in batch.items():
            tensor = roi.tensors(images)
            features = inception(tensor)[0].flatten(1).numpy()
            if name in BASE_ARMS:
                np.testing.assert_allclose(features, originals[name][start:start+len(group)],
                                           rtol=1e-5, atol=2e-5, err_msg=name)
                continue
            feature_chunks.setdefault(name, []).append(features)
            value = lpips(tensor*2-1, gt*2-1).flatten().tolist()
            for item, score in zip(measured[start:], value):
                item['scores'][name]['lpips'] = score
        print('FID_REGION_BATCH', carrier, start+len(group), len(rows), flush=True)
    originals.close()
    arrays = {k: np.concatenate(v) for k, v in feature_chunks.items()}
    names = ['pool3.npz', 'per_pair.json', 'audit.json']
    metadata = {}
    if retention:
        result = summarize_retention(source, previous, measured, arrays)
        tail.write(out / 'result.json', result)
        names.append('result.json')
        metadata = dict(new_retention=RETENTION, reused_region_sha256=REGION_B_SHA,
                        reused_arms=['offline25', 'retain_0', 'retain_50', 'retain_100'],
                        pixels_outside_feathered_mask_unchanged=True)
    np.savez_compressed(out / 'pool3.npz', **arrays)
    tail.write(out / 'per_pair.json', measured)
    tail.write(out / 'audit.json', dict(samples=len(rows), carrier=carrier, source_sha256=SOURCE_SHA,
        code_sha256=tail.sha(Path(__file__)), sigma=2, radius=8, verified_original_features=not retention,
        mode='RGB composites and frozen metric forward only', diffusion_rollouts=0, optimizer_updates=0,
        **metadata))
    pack(out, names)
    print('FID_REGION_DONE', carrier, (out / 'review.zip').stat().st_size, tail.sha(out / 'review.zip'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prior', type=Path)
    parser.add_argument('carrier', choices=('B', 'continued_A'))
    parser.add_argument('--retention', action='store_true')
    args = parser.parse_args()
    if args.retention and args.carrier != 'B':
        parser.error('--retention requires carrier B')
    cv2.setNumThreads(1)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '8')))
    run(args.prior, args.carrier, args.retention)


if __name__ == '__main__':
    main()
