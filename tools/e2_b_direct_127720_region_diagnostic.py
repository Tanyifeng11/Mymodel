"""复用 127720 已有图和特征，分解区域 FID 与位置对应的带通误差。"""
import argparse
from collections import Counter
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from tools import e2_b_single_fid_deep_rim as regions
from tools import e2_b_single_pants_lstar_equal50 as lstar
from tools.e2_f68_e1_independent_band import details

previous, fusion, init, tail, roi = regions.previous, regions.fusion, regions.init, regions.tail, regions.roi
CATEGORIES = regions.CATEGORIES
ARMS = ('off', 'a64', 'step768', 'step960')
RUN = 'fid_texture_regions_127720'
BASELINE_SHA = '4de6fe244e5c5c319a8074a177d5f8946c24dcdfe4b61165dfab01af83c3ec0a'
A64_SHA = '897246cc75c29e11c124d952b3e4cbe49c8aeec90866e385c086809a99d0b0f7'
DIRECT_SHA = 'c1725c2ba8ceda4097b8cbbb9638a4b085083a3449a802d0bc6060e0a7032c82'
PROBES = ('pants_band50', 'pants_low50', 'all_band50', 'pants_deep_off', 'pants_rim_off')
FIELDS = ('mae', 'amplitude_l1', 'position_l1', 'rms_bias', 'abs_rms_error', 'cosine',
          'amp_mse', 'position_mse', 'l_bias', 'low_l_error', 'low_ab_error', 'lpips', 'ssim')


def load(prior):
    args = fusion.paths(prior, 'cpu')
    args.out = args.run
    path = args.out / 'protocol.json'
    cohort = tail.read(path)
    pairs = [p for p in cohort['pairs'] if p['split'] == 'val']
    assert Counter(p['category'] for p in pairs) == dict.fromkeys(CATEGORIES, 64)
    root = args.baseline / 'b_single_s0_e2_epoch20_real_ab'
    a64 = root / 'h2_step768_pants_band_ab/a64_pants_feature_cov_ab/b64_pants_inception_band50_ab'
    trained = root / 'direct_h2_alexnet_960/scaled_qualification_retry'
    configs = {run:tail.read(run/'review_manifest.json') for run in (root, a64, trained)}
    for run, digest in ((root, BASELINE_SHA), (a64, A64_SHA), (trained, DIRECT_SHA)):
        assert tail.sha(run/'review.zip') == digest
        cfg = fusion.read_checked(run/'protocol.json', configs[run]['protocol.json'])
        assert cfg['cohort_sha256'] == tail.sha(path)
    assets = tail.read(root/'protocol.json')['metrics']
    assert tail.metric_assets() == assets == cohort['metric_assets']
    levels = [('off',root,0,'H2'), ('a64',a64,64,'A'), ('step768',trained,768,'direct'), ('step960',trained,960,'direct')]
    albums, features, expected = {}, {}, {}
    for name, run, step, arm in levels:
        rel = f'validation/step{step}'
        source = run/rel
        manifest = configs[run]
        value = fusion.read_checked(source/'result.json', manifest[rel+'/result.json'])
        assert value['pairs'] == len(pairs) == 256 and value['protocol_sha256'] == manifest['protocol.json']
        rows = tail.read(source/'per_pair.json')
        assert [(r['case_id'],r['category']) for r in rows] == [(p['case_id'],p['category']) for p in pairs]
        assert tail.sha(source/'pool3.npz') == value['pool3_sha256']
        with np.load(source/'pool3.npz',allow_pickle=False) as z:
            if name == 'off':
                features['gt'] = z['gt']
            np.testing.assert_array_equal(z['gt'],features['gt'])
            features[name] = z[arm]
        albums[name] = (run/'renders'/value['head_sha256'][arm], arm, rows)
        expected[name] = value['groups']
    limits = tail.read(trained/'protocol.json')['thresholds']
    return args, trained/RUN, pairs, cohort['gt_sha256'], albums, features, expected, assets, limits


def region_measure(image, truth, hard, masks, core, lpips, ssim):
    lab = roi.lab(image)
    value = roi.band(lab[...,0])
    target, target_band = truth
    low = roi.gaussian(lab-target,8)
    registers = {k:m for k,m in zip(hard,masks)}
    registers['inside'] = masks[0] | masks[1] if 'inner_rim' in hard else masks[0]
    registers['deep'] = core
    out = {}
    for name, select in registers.items():
        if not select.any() or name=='deep' and int(select.sum())<256:
            continue
        f = details(value[select],target_band[select])
        f.update(l_bias=float((lab[...,0]-target[...,0])[select].mean()),
                 low_l_error=float(np.abs(low[select,0]).mean()),
                 low_ab_error=float(np.linalg.norm(low[select,1:],axis=1).mean()))
        if name in hard:
            k = hard.index(name)
            f.update(lpips=float(lpips[k]),ssim=ssim[k])
        elif name == 'inside':
            f.update(lpips=float(lpips[:2].sum()),ssim=sum(ssim[:2]))
        out[name] = f
    return out


def summarize_regions(rows):
    result = {}
    for category in CATEGORIES:
        subset = [r for r in rows if r['category']==category]
        region_names = tuple(dict.fromkeys((*regions.region_names(category),'inside','deep')))
        arms = {}
        for arm in ARMS:
            stats = {}
            for name in region_names:
                values = [r['regional'][arm][name] for r in subset if name in r['regional'][arm]]
                stats[name] = dict(n=len(values), **{
                    k:float(np.mean(take)) for k in FIELDS
                    if (take:=[v[k] for v in values if v.get(k) is not None])
                })
            arms[arm] = stats
        result[category] = arms
    return result


def paired_texture(rows):
    rng = np.random.default_rng(127720)
    result = {}
    for category in CATEGORIES:
        subset = [r for r in rows if r['category']==category and r['interior_px']>=256]
        record = {}
        for k in ('mae','amplitude_l1','position_l1','rms_bias','abs_rms_error','cosine'):
            raw = [(r['regional']['a64']['deep'][k],r['regional']['step960']['deep'][k]) for r in subset]
            delta = np.array([b-a for a,b in raw if a is not None and b is not None])
            boot = rng.integers(0,len(delta),size=(10000,len(delta)))
            record[k] = dict(n=len(delta),mean=float(delta.mean()),
                             ci95=np.quantile(delta[boot].mean(axis=1),[.025,.975]).tolist())
        result[category] = record
    return result


def probe_metrics(rows, features, labels, limits):
    result = {}
    for name in PROBES:
        measured = [r['probes'].get(name,r['scores']['step960']) for r in rows]
        overall = roi.fid_terms(features[name],features['gt'])
        metrics = {k:float(np.mean([r[k] for r in measured])) for k in ('lpips','ssim','contour_f1')}
        bands = {c:float(np.mean([r['probes'].get(name,r['texture']['step960'])['band_mae']
                 for r in rows if r['category']==c and r['interior_px']>=256])) for c in CATEGORIES}
        keep = labels=='pants'
        pants = roi.fid_terms(features[name][keep],features['gt'][keep])
        checks = dict(fid=overall['fid']<limits['fid'],lpips=metrics['lpips']<=limits['lpips'],
                      ssim=metrics['ssim']>=limits['ssim'],contour_f1=metrics['contour_f1']>=limits['contour_f1'])
        checks.update({c:bands[c]<=limits['band_mae'][c] for c in CATEGORIES})
        result[name] = dict(fid=overall['fid'],pants_fid=pants['fid'],metrics=metrics,
                            band_mae=bands,checks=checks,edited=sum(name in r['probes'] for r in rows))
    return result


@torch.inference_mode()
def run(data):
    args,dest,pairs,hashes,albums,features,expected,assets,limits = data
    if (dest/'review.zip').is_file():
        assert tail.sha(dest/'result.json') == tail.read(dest/'review_manifest.json')['result.json']
        return
    dest.mkdir(parents=True,exist_ok=True)
    lpips,inception = tail.metric_models('cpu',assets)
    labels = np.array([p['category'] for p in pairs])
    rows = [None]*len(pairs)
    pools = {arm:{} for arm in ARMS[1:]}
    features.update({name:features['step960'].copy() for name in PROBES})
    counts = Counter()

    for category in CATEGORIES:
        indexes = np.flatnonzero(labels==category)
        names = regions.region_names(category)
        end = (1<<len(names))-1
        for arm in ARMS[1:]:
            pools[arm][category] = {0:features['off'][indexes],end:features[arm][indexes]}
            pools[arm][category].update({b:np.empty_like(features[arm][indexes]) for b in range(1,end)})
        for offset in range(0,len(indexes),4):
            ids = indexes[offset:offset+4]
            items = []
            for i in ids:
                pair = pairs[i]
                x,y,m,w,row = previous.inputs(args,pair,hashes,albums,i)
                mask = tail.case_mask(args,pair)
                hard,soft = regions.split_regions(mask,m,w,category)
                core = roi.regions(mask,20)[0]
                row['probes'] = {}
                rows[i] = row
                items.append((x,y,w,hard,soft,mask,core,row))
            images,targets,coarse,hard,soft,masks,cores,batch = zip(*items)
            target_lab = [roi.lab(y) for y in targets]
            truth = [(y,roi.band(y[...,0])) for y in target_lab]
            target = roi.tensors(targets)
            target_maps = roi.lpips_features(lpips,target)
            mask_tensor = torch.from_numpy(np.stack(hard)).float()
            for arm in ARMS:
                input_tensor = roi.tensors([x[arm] for x in images])
                apportioned = roi.apportion(roi.maps(lpips,target_maps,input_tensor),mask_tensor)
                for j,(x,y,row) in enumerate(zip(images,targets,batch)):
                    score,parts,_ = roi.ssim_regions(x[arm],y,hard[j])
                    np.testing.assert_allclose([apportioned[j].sum(),score],
                        [row['scores'][arm]['lpips'],row['scores'][arm]['ssim']],atol=2e-5,rtol=2e-4)
                    out = region_measure(x[arm],truth[j],names,hard[j],cores[j],apportioned[j],parts)
                    if row['interior_px']>=256:
                        np.testing.assert_allclose(out['deep']['mae'],row['texture'][arm]['band_mae'],atol=1e-9,rtol=0)
                    row['regional'][arm] = out
                if arm=='off':
                    continue
                for bits in range(1,end):
                    mixed = [regions.mix((x['off'],x[arm]),w,c,bits,category)
                             for x,w,c in zip(images,soft,coarse)]
                    vector = inception(roi.tensors(mixed))[0].flatten(1).numpy()
                    pools[arm][category][bits][offset:offset+len(ids)] = vector
                    counts['new_swap_features'] += len(ids)
                    if arm=='step960' and category=='pants' and bits in (13,14):
                        key = 'pants_deep_off' if bits==14 else 'pants_rim_off'
                        features[key][ids] = vector
                        distances = lpips(roi.tensors(mixed)*2-1,target*2-1).flatten().tolist()
                        for x,y,m,core,row,dist in zip(mixed,targets,masks,cores,batch,distances):
                            row['probes'][key] = dict(lpips=dist,ssim=roi.ssim_regions(x,y,roi.regions(m,8))[0],
                                contour_f1=roi.shape_score(x,m)['contour_f1'],
                                band_mae=tail.appearance(x,roi.lab(y),core)['band_mae'])
            # 使用配对真图作因果诊断，所有变化严格限制在 r20 深衣内。
            arms = ('band50','low_l50') if category=='pants' else ('band50',)
            edits = [lstar.edit(x['step960'],y,m)[0] if row['interior_px']>=256 else {}
                     for x,y,m,row in zip(images,targets,masks,batch)]
            for name in arms:
                changed = [j for j,e in enumerate(edits) if name in e and e[name][1]['changed']]
                if not changed:
                    continue
                samples = [edits[j][name][0] for j in changed]
                xx = roi.tensors(samples)
                vec = inception(xx)[0].flatten(1).numpy()
                distances = lpips(xx*2-1,target[changed]*2-1).flatten().tolist()
                keys = ('all_band50','pants_band50') if name=='band50' and category=='pants' else (
                    ('all_band50',) if name=='band50' else ('pants_low50',))
                for k in keys:
                    features[k][ids[changed]] = vec
                for x,j,dist in zip(samples,changed,distances):
                    m,core,row = masks[j],cores[j],batch[j]
                    score = dict(lpips=dist,ssim=roi.ssim_regions(x,targets[j],roi.regions(m,8))[0],
                                 contour_f1=roi.shape_score(x,m)['contour_f1'],
                                 band_mae=tail.appearance(x,target_lab[j],core)['band_mae'])
                    for k in keys:
                        row['probes'][k] = score
                counts['new_oracle_features'] += len(changed)
        print('REGION_127720_CATEGORY',category,int(len(indexes)),dict(counts),flush=True)

    assert all(r is not None for r in rows)
    stats = summarize_regions(rows)
    groups = {}
    for name in ARMS:
        arm = {}
        for category in ('all',*CATEGORIES):
            keep = np.ones(len(labels),bool) if category=='all' else labels==category
            fid = roi.fid_terms(features[name][keep],features['gt'][keep])
            old = expected[name][category]['arms'][albums[name][1]]['fid']
            np.testing.assert_allclose(list(fid.values()),[old[k] for k in fid],atol=1e-9,rtol=1e-9)
            arm[category] = fid
        groups[name] = arm
    attribution = {name:regions.contribution(pools[name],dict(step0=features['off'],candidate=features[name]),
                                             labels,features['gt']) for name in ARMS[1:]}
    probes = probe_metrics(rows,features,labels,limits)
    result = dict(pairs=len(pairs),groups=groups,regions=stats,attribution=attribution,
                  texture_delta_960_vs_a64=paired_texture(rows),probes=probes,thresholds=limits,
                  counts=dict(counts),new_diffusion_samples=0,optimizer_steps=0)
    cache = dest/'region_pool3.npz'
    np.savez_compressed(cache,**{f'{arm}_{c}_{b}':v for arm,cs in pools.items()
                                for c,parts in cs.items() for b,v in parts.items() if b not in (0,len(parts)-1)})
    result['region_pool3_sha256'] = tail.sha(cache)
    tail.write(dest/'result.json',result)
    tail.write(dest/'per_pair.json',rows)
    init.fid.pack(dest,['result.json','per_pair.json'])
    print('REGION_127720_DONE',tail.sha(dest/'review.zip'),flush=True)


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('prior',type=Path)
    cli.add_argument('--check',action='store_true')
    opt = cli.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
    data = load(opt.prior)
    if opt.check:
        args,_,pairs,hashes,albums,*_ = data
        for i in (0,next(i for i,p in enumerate(pairs) if p['category']=='pants')):
            previous.inputs(args,pairs[i],hashes,albums,i)
        print('REGION_127720_PRECHECK',len(pairs),flush=True)
    else:
        run(data)
