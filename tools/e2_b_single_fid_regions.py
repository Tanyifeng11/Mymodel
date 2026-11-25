"""复用已有生成图，拆解类别和区域对整图 FID 的影响。"""
import argparse
import math
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from tools import e2_b_single_pants_ab as experiment
from tools.e2_f68_e1_retain125_diagnostic import appearance, shapley

fusion, init, tail, roi = experiment.fusion, experiment.init, experiment.tail, experiment.roi
CATEGORIES = tail.distill.CATEGORIES
REGIONS = ('inside', 'edge', 'background')
ARMS = ('step0', 'h2_768', 'a64', 'b64', 'b128')
FID_FIELDS = ('fid', 'mean_term', 'covariance_term')
RUN = 'fid_category_regions_127459'


def load(prior):
    args = fusion.paths(prior, 'cpu')
    args.out = args.run
    cohort_path = args.out/'protocol.json'
    cohort = tail.read(cohort_path)
    pairs = [p for p in cohort['pairs'] if p['split'] == 'val']
    assert Counter(p['category'] for p in pairs) == dict.fromkeys(CATEGORIES, 64)
    parent = args.baseline/experiment.single.RUN
    trained = parent/experiment.RUN
    assert tail.sha(trained/'review.zip') == 'a908a1a339d27db7323c8f349b41d982f6c324d515075b70c087a1f53df64ce2'
    sources = [(parent, 0, 'H2'), (parent, 768, 'H2'), (trained, 64, 'A'),
               (trained, 64, 'B'), (trained, 128, 'B')]
    albums, features, expected = {}, {}, {}
    for name, (run, step, key) in zip(ARMS, sources):
        manifest = tail.read(run/'review_manifest.json')
        protocol = fusion.read_checked(run/'protocol.json', manifest['protocol.json'])
        assert protocol['cohort_sha256'] == tail.sha(cohort_path)
        rel = f'validation/step{step}'
        result = fusion.read_checked(run/rel/'result.json', manifest[rel+'/result.json'])
        assert result['pairs'] == len(pairs) and result['protocol_sha256'] == manifest['protocol.json']
        assert tail.sha(run/rel/'pool3.npz') == result['pool3_sha256']
        rows = tail.read(run/rel/'per_pair.json')
        assert [(p['case_id'],p['category']) for p in pairs] == [(r['case_id'],r['category']) for r in rows]
        with np.load(run/rel/'pool3.npz', allow_pickle=False) as pool:
            if name == 'step0':
                features['gt'] = pool['gt']
            np.testing.assert_array_equal(features['gt'], pool['gt'])
            features[name] = pool[key]
        albums[name] = (run/'renders'/result['head_sha256'][key], key, rows)
        expected[name] = {g: v['arms'][key]['fid'] for g,v in result['groups'].items()}
    assert all(v.shape == (len(pairs),2048) and np.isfinite(v).all() for v in features.values())
    source_assets = tail.read(parent/'protocol.json')['metrics']
    assert tail.metric_assets() == source_assets == cohort['metric_assets']
    return args, trained/RUN, pairs, cohort['gt_sha256'], albums, features, expected, source_assets


def inputs(args, pair, hashes, albums, index):
    case = pair['case_id']
    target = args.data/pair['gt']
    assert tail.sha(target) == hashes[case]
    gt = np.asarray(tail.rgb(target).resize(tail.SIZE, Image.Resampling.BILINEAR))
    mask = tail.case_mask(args, pair)
    masks = roi.regions(mask, 8)
    weights = np.stack([roi.gaussian(m.astype('float32'), 2) for m in masks])
    np.testing.assert_allclose(weights.sum(0), 1, atol=1e-6, rtol=0)
    core = int(roi.regions(mask,20)[0].sum())
    images, scores, texture = {}, {}, {}
    for name, (folder, key, rows) in albums.items():
        row = rows[index]
        path = folder/(case+'.png')
        assert tail.sha(path) == row['pngs'][key] and core == row['interior_px'], path
        images[name] = np.asarray(tail.rgb(path))
        scores[name], texture[name] = row['scores'][key], row['texture'][key]
    return images, gt, masks, weights, dict(case_id=case,category=pair['category'],
                interior_px=core,scores=scores,texture=texture,regional={})


def stats(group):
    arrays = {}
    for arm in ARMS:
        regional = [r['regional'][arm] for r in group]
        deep = [r['texture'][arm] for r in group if r['interior_px'] >= 256]
        arrays[arm] = dict(
            metrics={k:float(np.nanmean(np.array([r['scores'][arm][k] for r in group], dtype='float64'))) for k in tail.METRICS},
            deep_texture={k:float(np.mean([r[k] for r in deep])) for k in deep[0]},
            regional={name:dict(
                lpips_contribution=float(np.mean([r[name]['lpips'] for r in regional])),
                ssim_contribution=float(np.mean([r[name]['ssim'] for r in regional])),
                pixels=float(np.mean([r[name]['area'] for r in regional])),
                **{k:float(np.mean([r[name]['appearance'][k] for r in regional
                                   if r[name]['appearance'][k] is not None]))
                   for k in regional[0][name]['appearance']})
                for name in REGIONS})
    return dict(n=len(group), valid_deep=sum(r['interior_px']>=256 for r in group), arms=arrays)


def vector(x, gt):
    value = roi.fid_terms(x, gt)
    return np.array([value[k] for k in FID_FIELDS])


def contribution(features, labels, gt):
    """先对类别顺序平均，再对该类区域的替换顺序平均。"""
    masks = [labels==c for c in CATEGORIES]
    regions = {}
    for i, category in enumerate(CATEGORIES):
        effect = np.zeros((len(REGIONS),len(FID_FIELDS)))
        for subset in range(1 << len(masks)):
            if subset & (1 << i):
                continue
            size = bin(subset).count('1')
            weight = math.factorial(size)*math.factorial(len(masks)-1-size)/math.factorial(len(masks))
            mixed = features[0].copy()
            for j, keep in enumerate(masks):
                if subset & (1 << j):
                    mixed[keep] = features[7][keep]
            values = []
            for bits in range(8):
                mixed[masks[i]] = features[bits][masks[i]]
                values.append(vector(mixed,gt))
            effect += weight*np.array(list(shapley(values,REGIONS).values()))
        regions[category] = effect
    total = sum(regions.values())
    gap = vector(features[7],gt)-vector(features[0],gt)
    np.testing.assert_allclose(total.sum(0),gap,atol=1e-8,rtol=1e-8)
    describe = lambda v: dict(zip(FID_FIELDS,v.tolist()))
    return dict(gap=describe(gap),
                by_category={c:describe(v.sum(0)) for c,v in regions.items()},
                by_region={r:describe(v) for r,v in zip(REGIONS,total)},
                category_region={c:{r:describe(v) for r,v in zip(REGIONS,values)}
                                 for c,values in regions.items()},
                whole_hybrids={str(b):describe(vector(features[b],gt)) for b in range(8)})


@torch.inference_mode()
def run(prior, device):
    args, dest, pairs, hashes, albums, features, expected, assets = load(prior)
    if (dest/'review.zip').is_file():
        assert tail.sha(dest/'result.json') == tail.read(dest/'review_manifest.json')['result.json']
        return
    dest.mkdir(exist_ok=True, parents=True)
    lpips, inception = tail.metric_models(device,assets)
    combined = {n:{0:features['step0'],7:features[n],**{b:np.empty_like(features[n])
                for b in range(1,7)}} for n in ARMS[1:]}
    all_rows, probe = [], {}
    for offset in range(0,len(pairs),4):
        batch = [inputs(args,p,hashes,albums,i) for i,p in enumerate(pairs[offset:offset+4],offset)]
        images, targets, masks, weights, rows = zip(*batch)
        target = roi.tensors(targets).to(device)
        target_maps = roi.lpips_features(lpips,target)
        mask_tensor = torch.from_numpy(np.stack(masks)).float()
        for arm in ARMS:
            original = roi.tensors([x[arm] for x in images]).to(device)
            lp = roi.apportion([x.cpu() for x in roi.maps(lpips,target_maps,original)],mask_tensor)
            for i,(x,gt,m,row) in enumerate(zip(images,targets,masks,rows)):
                _,ssim,area = roi.ssim_regions(x[arm],gt,m)
                np.testing.assert_allclose([lp[i].sum(),sum(ssim)],
                    [row['scores'][arm]['lpips'],row['scores'][arm]['ssim']],atol=2e-5,rtol=2e-4)
                visual = appearance(x[arm],gt,m)
                row['regional'][arm] = {k:dict(lpips=float(lp[i,j]),ssim=ssim[j],
                    area=area[j],appearance=visual[k]) for j,k in enumerate(REGIONS)}
            if offset == 0:
                observed = inception(original)[0].flatten(1).cpu().numpy()
                np.testing.assert_allclose(observed, features[arm][:4],atol=2e-4,rtol=2e-4)
                probe[arm] = float(np.max(np.abs(observed-features[arm][:4])))
            if arm == 'step0':
                continue
            hybrid = []
            for bits in range(1,7):
                for picture, fade in zip(images,weights):
                    base, now = picture['step0'],picture[arm]
                    alpha = sum(fade[j] for j in range(3) if bits & (1 << j))
                    pixels = np.rint(base + alpha[...,None]*(now.astype('float32')-base))
                    hybrid.append(pixels.astype('uint8'))
            tensor = roi.tensors(hybrid).to(device)
            pooled = torch.cat([inception(chunk)[0].flatten(1) for chunk in tensor.split(8)]).cpu().numpy()
            for b, split in enumerate(np.split(pooled,6),1):
                combined[arm][b][offset:offset+4] = split
        all_rows.extend(rows)
        if len(all_rows)%32 == 0:
            print('PANTS_FID_REGIONS',len(all_rows),len(pairs),flush=True)

    labels = np.array([r['category'] for r in all_rows])
    groups = {}
    for group in ('all',*CATEGORIES):
        keep = np.ones(len(labels),bool) if group=='all' else labels==group
        s = stats([r for r,k in zip(all_rows,keep) if k])
        for n in ARMS:
            fid = roi.fid_terms(features[n][keep],features['gt'][keep])
            np.testing.assert_allclose(list(fid.values()),[expected[n][group][k] for k in fid],atol=1e-9,rtol=1e-9)
            s['arms'][n]['fid'] = fid
        groups[group] = s
    attribution = {}
    for n, f in combined.items():
        attribution[n] = contribution(f,labels,features['gt'])
        print('PANTS_FID_ATTRIBUTION',n,attribution[n]['gap'],flush=True)
    result = dict(groups=groups,attribution=attribution,metric_device=device,inception_probe_max_abs=probe,
                  pairs=len(pairs),new_diffusion_samples=0,optimizer_steps=0,
                  region_definition='8 px erode/dilate; pixel swaps use sigma 2 partition of unity',
                  baseline='step0')
    pool_path = dest/'region_pool3.npz'
    np.savez_compressed(pool_path, **{
        f'{n}_{b}':f[b] for n,f in combined.items() for b in range(1,7)})
    result['region_pool3_sha256'] = tail.sha(pool_path)
    tail.write(dest/'result.json',result)
    tail.write(dest/'per_pair.json',all_rows)
    init.fid.pack(dest,['result.json','per_pair.json'])
    print('PANTS_FID_REGIONS_DONE',tail.sha(dest/'review.zip'),flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('prior',type=Path)
    cli.add_argument('--device',default='cuda:0')
    cli.add_argument('--check',action='store_true')
    opts = cli.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if opts.check:
        a, _, p, h, albums, _, _, _ = load(opts.prior)
        inputs(a,p[0],h,albums,0)
        print('PANTS_FID_INPUTS_OK',len(p),flush=True)
    else:
        run(opts.prior,opts.device)
