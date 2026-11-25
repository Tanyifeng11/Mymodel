"""诊断 127556 的剩余 FID，将裤装衣内分为深衣内与靠边内沿。"""
import argparse
import math
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

from tools import e2_b_single_fid_regions as previous
from tools import e2_b_single_pants_cov_ab as short

fusion, init, tail, roi = previous.fusion, previous.init, previous.tail, previous.roi
CATEGORIES, REGIONS, FID_FIELDS = previous.CATEGORIES, previous.REGIONS, previous.FID_FIELDS
PANTS = ('deep', 'inner_rim', 'edge', 'background')
ARMS = ('step0', 'start', 'a64', 'b64')
RUN = 'fid_deep_rim_127556'
PROBE_FIELDS = ('lpips', 'ssim', 'contour_f1')
PROBES = dict(pants={'deep':14,'inner_rim':13,'inside':12},dress={'inside':6})


def region_names(category):
    return PANTS if category=='pants' else REGIONS


def load(prior):
    args = fusion.paths(prior,'cpu')
    args.out = args.run
    cohort_sha = tail.sha(args.out/'protocol.json')
    cohort = tail.read(args.out/'protocol.json')
    pairs = [p for p in cohort['pairs'] if p['split']=='val']
    assert Counter(p['category'] for p in pairs)==dict.fromkeys(CATEGORIES,64)
    parent = args.baseline/short.single.RUN
    first = parent/short.previous.RUN
    trained = first/short.RUN
    for run,digest in (
        (first,'a908a1a339d27db7323c8f349b41d982f6c324d515075b70c087a1f53df64ce2'),
        (trained,'2b0fddcac11cf66d9cd436b21850ebf04b9e9ac72513a40b0017c89763867f05'),
    ):
        assert tail.sha(run/'review.zip')==digest
    albums, features, expected = {}, {}, {}
    for n,(run,s,key) in zip(ARMS,((parent,0,'H2'),(first,64,'A'),(trained,64,'A'),(trained,64,'B'))):
        manifest = tail.read(run/'review_manifest.json')
        protocol = fusion.read_checked(run/'protocol.json',manifest['protocol.json'])
        assert protocol['cohort_sha256']==cohort_sha
        rel = f'validation/step{s}'
        score = fusion.read_checked(run/rel/'result.json',manifest[rel+'/result.json'])
        assert score['pairs']==256 and score['protocol_sha256']==manifest['protocol.json']
        assert tail.sha(run/rel/'pool3.npz')==score['pool3_sha256']
        rows = tail.read(run/rel/'per_pair.json')
        assert [(p['case_id'],p['category']) for p in pairs]==[(r['case_id'],r['category']) for r in rows]
        with np.load(run/rel/'pool3.npz',allow_pickle=False) as pool:
            if n=='step0':
                features['gt']=pool['gt']
            np.testing.assert_array_equal(pool['gt'],features['gt'])
            features[n]=pool[key]
        albums[n] = (run/'renders'/score['head_sha256'][key],key,rows)
        expected[n] = {g:v['arms'][key]['fid'] for g,v in score['groups'].items()}
    assert all(v.shape==(256,2048) and np.isfinite(v).all() for v in features.values())
    assets = tail.read(parent/'protocol.json')['metrics']
    assert tail.metric_assets()==assets==cohort['metric_assets']
    cache = first/previous.RUN
    manifest = tail.read(cache/'review_manifest.json')
    old = fusion.read_checked(cache/'result.json',manifest['result.json'])
    old_rows = fusion.read_checked(cache/'per_pair.json',manifest['per_pair.json'])
    assert [r['case_id'] for r in old_rows]==[p['case_id'] for p in pairs]
    prior_fid = old['groups']['all']['arms']['a64']['fid']
    np.testing.assert_allclose([prior_fid[k] for k in FID_FIELDS],
        [expected['start']['all'][k] for k in FID_FIELDS],atol=1e-9,rtol=1e-9)
    assert tail.sha(cache/'region_pool3.npz')==old['region_pool3_sha256']
    with np.load(cache/'region_pool3.npz',allow_pickle=False) as pool:
        cached = {b:pool[f'a64_{b}'] for b in range(1,7)}
    return args,trained/RUN,pairs,cohort['gt_sha256'],albums,features,expected,assets,cached


def coarse_bits(bits,category):
    if category!='pants':
        return bits
    if (bits & 3) in (0,3):
        return ((bits >> 1) & 6) | int(bool(bits & 3))
    return None


def split_regions(mask,masks,weights,category):
    if category!='pants':
        return masks,weights
    deep = roi.regions(mask,20)[0]
    assert not (deep & ~masks[0]).any()
    hard = np.stack([deep,masks[0] & ~deep,*masks[1:]])
    soft_deep = np.minimum(roi.gaussian(deep.astype('float32'),2),weights[0])
    soft = np.stack([soft_deep,weights[0]-soft_deep,*weights[1:]])
    np.testing.assert_allclose(soft.sum(0),1,atol=1e-6,rtol=0)
    return hard,soft


def mix(images,weights,coarse,bits,category):
    parent_bits = coarse_bits(bits,category)
    source,selected = (coarse,parent_bits) if parent_bits is not None else (weights,bits)
    alpha = sum(w for j,w in enumerate(source) if selected & (1 << j))[...,None]
    x,y = images
    return np.rint(x + alpha*(y.astype('float32')-x)).astype('uint8')


def visuals(x,y,masks):
    if len(masks)==3:
        return list(previous.appearance(x,y,masks).values())
    return list(previous.appearance(x,y,masks[:2]).values())+list(
        previous.appearance(x,y,masks[2:]).values())


def contribution(pools,features,labels,gt):
    masks = {n:labels==n for n in CATEGORIES}
    describe = lambda x:dict(zip(FID_FIELDS,x.tolist()))
    effects = {}
    for i,category in enumerate(CATEGORIES):
        names = region_names(category)
        result = np.zeros((len(names),len(FID_FIELDS)))
        for subset in range(1<<len(CATEGORIES)):
            if subset & (1<<i):
                continue
            size = bin(subset).count('1')
            weight = math.factorial(size)*math.factorial(len(CATEGORIES)-1-size)/math.factorial(len(CATEGORIES))
            x = features['step0'].copy()
            for j,other in enumerate(CATEGORIES):
                if subset & (1<<j):
                    x[masks[other]] = features['candidate'][masks[other]]
            values = []
            for b in range(1<<len(names)):
                x[masks[category]] = pools[category][b]
                values.append(previous.vector(x,gt))
            result += weight*np.array(list(previous.shapley(values,names).values()))
        effects[category] = {n:describe(v) for n,v in zip(names,result)}
    gap = previous.vector(features['candidate'],gt)-previous.vector(features['step0'],gt)
    total = np.sum([[v[k] for k in FID_FIELDS] for c in effects.values() for v in c.values()],axis=0)
    np.testing.assert_allclose(total,gap,atol=1e-8,rtol=1e-8)
    return dict(gap=describe(gap),category_region=effects)


def group_stats(rows,features,expected,labels):
    groups = {}
    for category in ('all',*CATEGORIES):
        keep = np.ones(len(labels),bool) if category=='all' else labels==category
        subset = [r for r,m in zip(rows,keep) if m]
        arms = {}
        for n in ARMS:
            arms[n] = dict(metrics={k: float(np.nanmean(np.array(
                [r['scores'][n][k] for r in subset],dtype='float64'))) for k in tail.METRICS})
            fid = roi.fid_terms(features[n][keep],features['gt'][keep])
            np.testing.assert_allclose(list(fid.values()),[expected[n][category][k] for k in fid],atol=1e-9,rtol=1e-9)
            sx,sy = (np.std(features[a][keep].astype('float64'),axis=0,ddof=1)
                     for a in (n,'gt'))
            scale = float((np.linalg.norm(sx)-np.linalg.norm(sy))**2)
            deep = [r['texture'][n] for r in subset if r['interior_px']>=256]
            arms[n].update(fid=fid,covariance_scale=scale,covariance_shape=fid['covariance_term']-scale,
                           deep_texture={k:float(np.mean([v[k] for v in deep])) for k in deep[0]})
            stats = {}
            names = REGIONS if category!='pants' else ('inside',*PANTS)
            for reg in names:
                values = [r['regional'][n][reg] for r in subset]
                stats[reg] = {k:float(np.mean([v[k] for v in values if v[k] is not None]))
                              for k in values[0] if any(v[k] is not None for v in values)}
            arms[n]['regions'] = stats
        groups[category] = dict(n=len(subset),valid_deep=sum(r['interior_px']>=256 for r in subset),arms=arms)
    return groups


def restore_probes(rows,pools,features,labels,thresholds):
    current = features['b64']
    reference = {(c,k): b for c,opts in PROBES.items() for k,b in opts.items()}
    edits = [{key:b} for key,b in reference.items()]
    edits += [{('pants',r):b,('dress','inside'):6} for r,b in PROBES['pants'].items()]
    result = {}
    for edit in edits:
        by_category = {c:(r,b) for (c,r),b in edit.items()}
        f = current.copy()
        for c,(_,b) in by_category.items():
            f[labels==c] = pools['b64'][c][b]
        scores, pants_band = [], []
        for r in rows:
            category = r['category']
            key = by_category.get(category)
            alt = r['probes'][key[0]] if key else None
            scores.append(alt if alt is not None else r['scores']['b64'])
            if category=='pants' and r['interior_px']>=256:
                pants_band.append(alt['band_mae'] if alt else r['texture']['b64']['band_mae'])
        name = '+'.join(f'{c}_{r}' for c,(r,_) in by_category.items())
        fid = roi.fid_terms(f,features['gt'])
        metrics = {k:float(np.mean([r[k] for r in scores])) for k in PROBE_FIELDS}
        metrics['pants_band_mae'] = float(np.mean(pants_band))
        checks = dict(fid=fid['fid']<thresholds['fid'],lpips=metrics['lpips']<=thresholds['lpips'],
                      ssim=metrics['ssim']>=thresholds['ssim'],contour_f1=metrics['contour_f1']>=thresholds['contour_f1'],
                      pants_band_mae=metrics['pants_band_mae']<=thresholds['pants_band_mae'])
        result[name] = dict(fid=fid,metrics=metrics,checks=checks,passed=all(checks.values()))
    return result


@torch.inference_mode()
def run(data):
    args,dest,pairs,hashes,albums,features,expected,assets,cached = data
    if (dest/'review.zip').exists():
        assert tail.sha(dest/'result.json')==tail.read(dest/'review_manifest.json')['result.json']
        return
    dest.mkdir(exist_ok=True,parents=True)
    lpips,inception = tail.metric_models('cpu',assets)
    labels = np.array([p['category'] for p in pairs])
    pools = {n:{} for n in ARMS[1:]}
    rows = [None]*256
    counts = Counter()
    for category in CATEGORIES:
        indexes = np.flatnonzero(labels==category)
        names = region_names(category)
        end = (1<<len(names))-1
        for n in ARMS[1:]:
            pools[n][category] = {0:features['step0'][indexes],end:features[n][indexes]}
            for bits in range(1,end):
                old = coarse_bits(bits,category)
                pools[n][category][bits] = (cached[old][indexes] if n=='start' and old is not None
                                            else np.empty_like(features[n][indexes]))
        for offset in range(0,len(indexes),4):
            ids = indexes[offset:offset+4]
            samples = []
            for i in ids:
                x,gt,m,w,row = previous.inputs(args,pairs[i],hashes,albums,i)
                mask = tail.case_mask(args,pairs[i])
                hard,soft = split_regions(mask,m,w,category)
                row['probes'] = {}
                samples.append((x,gt,hard,w,soft,mask,row))
                rows[i] = row
            images,targets,masks,coarse,weights,sketch,batch = zip(*samples)
            target = roi.tensors(targets)
            target_maps = roi.lpips_features(lpips,target)
            mask_tensor = torch.from_numpy(np.stack(masks)).float()
            for n in ARMS:
                pictures = roi.tensors([x[n] for x in images])
                lp = roi.apportion(roi.maps(lpips,target_maps,pictures),mask_tensor)
                for j,(x,y,m,row) in enumerate(zip(images,targets,masks,batch)):
                    _,ssim,area = roi.ssim_regions(x[n],y,m)
                    np.testing.assert_allclose([lp[j].sum(),sum(ssim)],
                        [row['scores'][n]['lpips'],row['scores'][n]['ssim']],atol=2e-5,rtol=2e-4)
                    region = {r:dict(lpips=float(lp[j,k]),ssim=ssim[k],area=area[k],**a)
                        for k,(r,a) in enumerate(zip(names,visuals(x[n],y,m)))}
                    if category=='pants':
                        union = m[0]|m[1]
                        region['inside'] = dict(lpips=float(lp[j,:2].sum()),ssim=sum(ssim[:2]),area=sum(area[:2]),
                                                **previous.appearance(x[n],y,(union,))['inside'])
                    row['regional'][n] = region
                if n=='step0':
                    continue
                for bits in range(1,end):
                    old = coarse_bits(bits,category)
                    if n=='start' and old is not None:
                        counts['reused_region_features'] += len(ids)
                        continue
                    synthetic = [mix((x['step0'],x[n]),w,c,bits,category) for x,w,c in zip(images,weights,coarse)]
                    t = roi.tensors(synthetic)
                    pools[n][category][bits][offset:offset+4] = inception(t)[0].flatten(1).numpy()
                    counts['computed_region_features'] += len(ids)
                    if n=='b64':
                        probe_name = next((k for k,b in PROBES.get(category,{}).items() if b==bits),None)
                        if probe_name:
                            distances = lpips(t*2-1,target*2-1).flatten().tolist()
                            for x,y,m,mask,row,distance in zip(synthetic,targets,masks,sketch,batch,distances):
                                score,_,_ = roi.ssim_regions(x,y,m)
                                record = dict(lpips=distance,ssim=score,contour_f1=roi.shape_score(x,mask)['contour_f1'])
                                if category=='pants':
                                    record['band_mae'] = tail.appearance(x,roi.lab(y),m[0])['band_mae']
                                row['probes'][probe_name] = record
        print('DEEP_RIM_CATEGORY',category,len(indexes),dict(counts),flush=True)
    groups = group_stats(rows,features,expected,labels)
    attributions = {}
    for n in ARMS[1:]:
        attributions[n] = contribution(pools[n],dict(step0=features['step0'],candidate=features[n]),labels,features['gt'])
        print('DEEP_RIM_FID',n,attributions[n]['gap'],flush=True)
    v = groups['all']['arms']['start']['metrics']
    threshold = dict(fid=expected['step0']['all']['fid'],lpips=v['lpips'],ssim=v['ssim']*.995,
                     contour_f1=v['contour_f1']*.995,pants_band_mae=groups['pants']['arms']['start']['deep_texture']['band_mae'])
    probes = restore_probes(rows,pools,features,labels,threshold)
    dest_cache = dest/'region_pool3.npz'
    np.savez_compressed(dest_cache,**{f'{n}_{c}_{b}':z for n,cs in pools.items()
        for c,bs in cs.items() for b,z in bs.items() if b not in (0,len(bs)-1)})
    result = dict(groups=groups,attribution=attributions,restore_probes=probes,thresholds=threshold,
        counts=dict(counts),pairs=256,new_diffusion_samples=0,optimizer_steps=0,
        pants_region='deep=erode20, inner_rim=erode8 minus erode20, edge and background as previous',
        region_pool3_sha256=tail.sha(dest_cache))
    tail.write(dest/'result.json',result)
    tail.write(dest/'per_pair.json',rows)
    init.fid.pack(dest,['result.json','per_pair.json'])
    print('DEEP_RIM_DONE',tail.sha(dest/'review.zip'),flush=True)


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('prior',type=Path)
    cli.add_argument('--check',action='store_true')
    options = cli.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
    data = load(options.prior)
    if options.check:
        args,_,pairs,hashes,albums,*_ = data
        for p in (pairs[0],next(p for p in pairs if p['category']=='pants')):
            previous.inputs(args,p,hashes,albums,pairs.index(p))
        print('DEEP_RIM_INPUTS_OK',len(pairs),flush=True)
    else:
        run(data)
