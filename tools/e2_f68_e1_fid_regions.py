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


def pack(out, names):
    manifest = {name: tail.sha(out / name) for name in names}
    tail.write(out / 'review_manifest.json', manifest)
    tmp = out / 'review.tmp'
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in (*names, 'review_manifest.json'):
            archive.write(out / name, name)
    tmp.replace(out / 'review.zip')


@torch.inference_mode()
def run(prior, carrier):
    base = prior / tail.distill.BASE
    parent = base / COHORT
    source = parent / light.RUN / 'eval64'
    out = parent / light.RUN / RUN / carrier
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
                for name, img in edits(images, mask, carrier).items():
                    batch.setdefault(name, []).append(img)
                    if name in BASE_ARMS:
                        continue
                    path = Path(temp) / 'current.png'
                    picture = Image.fromarray(img)
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
    np.savez_compressed(out / 'pool3.npz', **{k:np.concatenate(v) for k,v in feature_chunks.items()})
    tail.write(out / 'per_pair.json', measured)
    tail.write(out / 'audit.json', dict(samples=len(rows), carrier=carrier, source_sha256=SOURCE_SHA,
        code_sha256=tail.sha(Path(__file__)), sigma=2, radius=8, verified_original_features=True,
        mode='RGB composites and frozen metric forward only', diffusion_rollouts=0, optimizer_updates=0))
    pack(out, ('pool3.npz', 'per_pair.json', 'audit.json'))
    print('FID_REGION_DONE', carrier, (out / 'review.zip').stat().st_size, tail.sha(out / 'review.zip'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prior', type=Path)
    parser.add_argument('carrier', choices=('B', 'continued_A'))
    args = parser.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '8')))
    run(args.prior, args.carrier)


if __name__ == '__main__':
    main()
