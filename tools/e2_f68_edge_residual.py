"""用已保存的 f68 开关 PNG 验收边缘与衣内残差。"""
import argparse
import os
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from tools.e2_f68_real_paired import ARMS, CATEGORIES, METRICS, image_scores, metric_assets, metric_models, verify, write
from tools.e2_f68_region_diagnostic import (ARCHIVE_SHA, checked_case, fid_terms, gaussian, lpips_features,
                                            maps, paired, regions, tensors)
from tools.e2_fft_experiment import environment, read, sha


MIXES = {'edge_only': 0., 'edge_a0125': .125, 'edge_a025': .25}
VARIANTS = ARMS + tuple(MIXES)
SWEEP = {'edge_a025': .25, 'edge_a03125': .3125, 'edge_a0375': .375, 'edge_a050': .5}
ROI_ARCHIVE_SHA = 'c1afedc22835b1761ac0ec99f0f3aef35ce8dca4fe018d0ee4ec55431e80cd38'
ACCEPTANCE_SHA = '2a39cc9e7194f7a9ad7731e93b8d455f155e2b29b99283d30e2411ecdefc00ca'
LOWER_BETTER = {'lpips', 'leak'}


def mix_pixels(off, on, mask, mixes=MIXES):
    inside, edge, background = regions(mask, 8)
    wi, we, wb = (gaussian(r.astype('float32'), 2) for r in (inside, edge, background))
    assert np.allclose(wi + we + wb, 1, atol=1e-6)
    off = off.astype('float32')
    delta = on.astype('float32') - off
    return {name: np.rint(np.clip(off + (we + alpha * wi)[..., None] * delta, 0, 255)).astype('uint8')
            for name, alpha in mixes.items()}


def verified_files(directory, archive_hash, names):
    assert sha(directory / 'review.zip') == archive_hash, directory
    manifest = read(directory / 'review_manifest.json')
    for name in names:
        assert sha(directory / name) == manifest[name], name


def summarize(rows, features, variants=VARIANTS, controls=ARMS, mixes=MIXES, fid_reference='off'):
    classes = np.array([r['category'] for r in rows])
    groups = {}
    for group, select in (('all', np.ones(len(rows), bool)), ('outwear', classes == 'outwear')):
        selected = [r for r, keep in zip(rows, select) if keep]
        labels = classes[select]
        arms = {}
        for name in variants:
            arm = dict(fid=fid_terms(features[name][select], features['gt'][select]))
            for metric in METRICS:
                values = np.array([r['scores'][name][metric] for r in selected], dtype='float64')
                arm[metric] = dict(n=int(np.isfinite(values).sum()), mean=float(np.nanmean(values)))
                for control in controls:
                    ref = np.array([r['scores'][control][metric] for r in selected], dtype='float64')
                    comparison = paired(ref, values, labels)
                    delta = values - ref
                    oriented = -delta if metric in LOWER_BETTER else delta
                    comparison.update(improved=int((oriented > 0).sum()), worse=int((oriented < 0).sum()),
                                      equal=int((oriented == 0).sum()))
                    arm[metric]['vs_' + control] = comparison
            arms[name] = arm
        for name in variants:
            value = arms[name]['fid']
            for control in controls:
                value['vs_' + control] = value['fid'] - arms[control]['fid']['fid']
        groups[group] = dict(n=len(selected), arms=arms)
    sensitivity = {}
    for category in CATEGORIES:
        keep = classes != category
        gt = features['gt'][keep]
        fids = {arm: fid_terms(features[arm][keep], gt)['fid'] for arm in dict.fromkeys((fid_reference, *mixes))}
        sensitivity[category] = {arm: fids[arm] - fids[fid_reference] for arm in mixes}
    return {'groups':groups, 'fid_leave_one_category_out_delta_vs_' + fid_reference:sensitivity}


@torch.no_grad()
def score(baseline, data, frozen, previous, prior_scores, features, scratch, mixes=MIXES, reference='edge_only'):
    model, inception = metric_models('cpu', frozen['metric_assets'])
    scores, new_features = [], {name: [] for name in mixes if name != reference}
    png = scratch / 'scored.png'
    pairs = frozen['pairs']
    for start in range(0, len(pairs), 4):
        batch, target, masks, paths = [], [], [], []
        for i, pair in enumerate(pairs[start:start+4], start):
            raw, mask = checked_case(baseline, data, pair, previous[i])
            assert previous[i]['index'] == prior_scores[i]['index'] == i
            assert pair['case_id'] == previous[i]['case_id'] == prior_scores[i]['case_id']
            batch.append(mix_pixels(raw['off'], raw['on'], mask, mixes))
            target.append(raw['gt'])
            masks.append(mask)
            paths.append(data / pair['gt'])
        ref = lpips_features(model, tensors(target))
        records = [dict(off=r['scores']['off'], on=r['scores']['on']) for r in previous[start:start+len(batch)]]
        for name in mixes:
            images = [p[name] for p in batch]
            batch_tensor = tensors(images)
            lpips = sum(layer.mean(dim=(-1, -2)) for layer in maps(model, ref, batch_tensor)).numpy()
            if name != reference:
                new_features[name].append(inception(batch_tensor)[0].flatten(1).numpy())
            elif start == 0:
                exact = inception(batch_tensor)[0].flatten(1).numpy()
                np.testing.assert_allclose(exact, features[name][:len(batch)], atol=7e-5, rtol=1e-4)
            for j, image in enumerate(images):
                actual = Image.fromarray(image)
                actual.save(png, compress_level=1)
                value = image_scores(actual, Image.fromarray(target[j]), png, paths[j], masks[j])
                value['lpips'] = float(lpips[j])
                if name == reference:
                    prior = prior_scores[start+j]['scores'][name]
                    for metric, expected in prior.items():
                        if expected is None:
                            assert value[metric] is None
                        else:
                            assert abs(value[metric] - expected) < (5e-6 if metric == 'lpips' else 1e-9)
                    value.update(prior)
                records[j][name] = value
        for j, row in enumerate(records):
            item = pairs[start+j]
            scores.append(dict(index=start+j, case_id=item['case_id'], category=item['category'], scores=row))
        if len(scores) % 20 == 0:
            print('EDGE_PAIRS_SCORED', len(scores), flush=True)
    return scores, {name: np.concatenate(chunks) for name, chunks in new_features.items()}


def archive_result(out, rows, features, result, meta):
    write(out/'meta.json', meta)
    write(out/'per_pair.json', rows)
    write(out/'result.json', result)
    np.savez_compressed(out/'inner_mix_pool3.npz', **features)
    names = ('meta.json', 'per_pair.json', 'result.json', 'inner_mix_pool3.npz')
    write(out/'review_manifest.json', {name: sha(out/name) for name in names})
    temporary = out/'review.zip.tmp'
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in (*names, 'review_manifest.json'):
            archive.write(out/name, name)
    temporary.replace(out/'review.zip')


def run(prior):
    baseline = prior / 'real_paired_B_on_off_20261118'
    roi = baseline / 'region_diagnostic_20261118'
    out = baseline / 'edge_residual_acceptance_20261118'
    out.mkdir(exist_ok=True)
    assert not (out / 'review.zip').exists()
    verified_files(baseline, ARCHIVE_SHA, ('protocol.json', 'per_image.json', 'fid_pool3.npz', 'result.json'))
    verified_files(roi, ROI_ARCHIVE_SHA, ('per_pair.json', 'edit_pool3.npz', 'result.json'))
    frozen = verify(argparse.Namespace(out=baseline))
    assert metric_assets() == frozen['metric_assets']
    previous, old_roi = read(baseline/'per_image.json'), read(roi/'per_pair.json')
    assert len(previous) == len(old_roi) == len(frozen['pairs']) == 500
    with np.load(baseline/'fid_pool3.npz', allow_pickle=False) as original:
        features = dict(original)
    with np.load(roi/'edit_pool3.npz', allow_pickle=False) as counterfactual:
        features['edge_only'] = counterfactual['edge_only']
    with tempfile.TemporaryDirectory(prefix='f68_edge_', dir=os.environ.get('SLURM_TMPDIR')) as scratch:
        rows, new_features = score(baseline, prior.parents[2]/'datasets/BF', frozen, previous, old_roi,
                                   features, Path(scratch))
    features.update(new_features)
    assert all(v.shape == (500, 2048) and np.isfinite(v).all() for v in features.values())
    result = summarize(rows, features)
    previous_result = read(baseline/'result.json')
    for arm in ARMS:
        stats = result['groups']['all']['arms'][arm]
        assert abs(stats['fid']['fid'] - previous_result['fid'][arm]) < 1e-8
        for metric in METRICS:
            assert abs(stats[metric]['mean'] - previous_result['summary'][metric][arm]) < 1e-10
    assert abs(result['groups']['all']['arms']['edge_only']['fid']['fid']
               - read(roi/'result.json')['all']['fid']['edge_only']['fid']) < 1e-8
    meta = dict(baseline_archive_sha256=ARCHIVE_SHA, region_archive_sha256=ROI_ARCHIVE_SHA,
                frozen_protocol_sha256=sha(baseline/'protocol.json'), source_sha256=sha(Path(__file__)),
                environment=environment(), backgrounds='B off outside feather support', radius_px=8,
                feather_sigma_px=2, inner_alphas=MIXES, original_images_read=1000, compared_pairs=len(rows),
                model_generations=0, optimizer_updates=0, edge_only_inception_features_reused=True)
    archive_result(out, rows, new_features, result, meta)
    print('EDGE_ACCEPTANCE_COMPLETE', len(rows), sha(out/'review.zip'), flush=True)


def run_sweep(prior):
    baseline = prior/'real_paired_B_on_off_20261118'
    previous = baseline/'edge_residual_acceptance_20261118'
    out = baseline/'edge_inner_sweep_20261119'
    out.mkdir(exist_ok=True)
    assert not (out/'review.zip').exists()
    reference = 'edge_a025'
    verified_files(baseline, ARCHIVE_SHA, ('protocol.json', 'per_image.json', 'fid_pool3.npz'))
    verified_files(previous, ACCEPTANCE_SHA, ('per_pair.json', 'inner_mix_pool3.npz', 'result.json'))
    frozen = verify(argparse.Namespace(out=baseline))
    assert metric_assets() == frozen['metric_assets']
    original, old = read(baseline/'per_image.json'), read(previous/'per_pair.json')
    assert len(original) == len(old) == len(frozen['pairs']) == 500
    with np.load(baseline/'fid_pool3.npz', allow_pickle=False) as data:
        features = dict(data)
    with np.load(previous/'inner_mix_pool3.npz', allow_pickle=False) as data:
        features[reference] = data[reference]
    with tempfile.TemporaryDirectory(prefix='f68_inner_', dir=os.environ.get('SLURM_TMPDIR')) as scratch:
        rows, newest = score(baseline, prior.parents[2]/'datasets/BF', frozen, original, old,
                             features, Path(scratch), SWEEP, reference)
    features.update(newest)
    assert all(f.shape == (500, 2048) and np.isfinite(f).all() for f in features.values())
    result = summarize(rows, features, ARMS + tuple(SWEEP), (reference, *ARMS), SWEEP, reference)
    previous_result = read(previous/'result.json')
    for group, s in result['groups'].items():
        expected = previous_result['groups'][group]['arms'][reference]
        current = s['arms'][reference]
        assert abs(current['fid']['fid'] - expected['fid']['fid']) < 1e-8
        assert all(abs(current[k]['mean'] - expected[k]['mean']) < 1e-12 for k in METRICS)
    meta = dict(original_archive_sha256=ARCHIVE_SHA, reference_archive_sha256=ACCEPTANCE_SHA,
                frozen_protocol_sha256=sha(baseline/'protocol.json'), source_sha256=sha(Path(__file__)),
                environment=environment(), reference=reference, radius_px=8, feather_sigma_px=2,
                inner_alphas=SWEEP, backgrounds='B off outside feather support',
                images_read=1000, compared_pairs=len(rows), model_generations=0, optimizer_updates=0)
    archive_result(out, rows, newest, result, meta)
    print('EDGE_SWEEP_COMPLETE',len(rows),sha(out/'review.zip'),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prior', type=Path)
    parser.add_argument('--sweep', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))
    cv2.setNumThreads(1)
    (run_sweep if args.sweep else run)(args.prior)
