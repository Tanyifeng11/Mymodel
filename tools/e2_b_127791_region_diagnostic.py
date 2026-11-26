"""用已归档的 127791 图和特征诊断 FID，仅读取扩散端点。"""
import argparse
from collections import Counter
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from tools import e2_b_direct_127720_region_diagnostic as prior
from tools.e2_b_single_pants_inception_ab import target_band50

regions, roi, tail, init = prior.regions, prior.roi, prior.tail, prior.init
CATEGORIES, FIELDS = prior.CATEGORIES, regions.FID_FIELDS
ARMS = ('off', 'step960', 'A', 'B')
COMPARISONS = {'off_A': ('off', 'A'), 'off_B': ('off', 'B'), 'A_B': ('A', 'B')}
PROBES = ('pants_deep_off', 'pants_rim_off', 'pants_band50', 'pants_low50', 'all_band50')
METRICS = ('lpips', 'ssim', 'contour_f1')
RUN = 'fid_texture_regions_127791'
RUN_SHA = '61c6b393e0c9e2377bc6445d881fa84fca0165f97e0b369084ebd5ff5d086999'
PRIOR_SHA = 'fd1ee858982694c4d1a43a4413eabd90c1f1dd068ffddc72aab098ab68d73a90'


def load(root):
    args, cache, pairs, hashes, old_albums, old_features, old_groups, assets, limits = prior.load(root)
    run = cache.parent/'fixed_band50_ab_64/gradient_qualification_retry'
    for path, digest in ((run, RUN_SHA), (cache, PRIOR_SHA)):
        assert tail.sha(path/'review.zip') == digest
    manifest = tail.read(run/'review_manifest.json')
    checked = prior.fusion.read_checked
    protocol = checked(run/'protocol.json', manifest['protocol.json'])
    train = checked(run/'training.json', manifest['training.json'])
    assert protocol['thresholds'] == limits and protocol['cohort_sha256'] == tail.sha(args.out/'protocol.json')
    assert train['updates_per_arm'] == len(train['trace']) == 64 and train['frozen_unchanged']
    rel = 'validation/step64'
    result = checked(run/rel/'result.json', manifest[rel+'/result.json'])
    rows = tail.read(run/rel/'per_pair.json')
    ids = [(p['case_id'], p['category']) for p in pairs]
    assert [(p['case_id'], p['category']) for p in rows] == ids and result['pairs'] == len(pairs)
    assert result['protocol_sha256'] == manifest['protocol.json']
    assert tail.sha(run/rel/'pool3.npz') == result['pool3_sha256']
    albums = {k: old_albums[k] for k in ARMS[:2]}
    features = {k: old_features[k] for k in ('gt', *ARMS[:2])}
    expected = {k: old_groups[k] for k in ARMS[:2]}
    with np.load(run/rel/'pool3.npz', allow_pickle=False) as pool:
        np.testing.assert_array_equal(pool['gt'], features['gt'])
        for k in ARMS[2:]:
            albums[k] = (run/'renders'/result['head_sha256'][k], k, rows)
            features[k] = pool[k]
            expected[k] = result['groups']
    assert all(x.shape == (256, 2048) and np.isfinite(x).all() for x in features.values())

    archive = tail.read(cache/'review_manifest.json')
    old = checked(cache/'result.json', archive['result.json'])
    prior_rows = checked(cache/'per_pair.json', archive['per_pair.json'])
    assert [(p['case_id'], p['category']) for p in prior_rows] == ids
    assert tail.sha(cache/'region_pool3.npz') == old['region_pool3_sha256']
    return args, run/RUN, pairs, hashes, albums, features, expected, assets, limits, cache, old, prior_rows


def response(images, target, core):
    """Measure actual endpoint changes along the fixed 960-step target direction."""
    if core.sum() < 256:
        return {}
    band = lambda image: roi.band(roi.lab(image)[..., 0])[core].astype('float64')
    anchor = band(images['step960'])
    goal = band(target)-anchor
    goal_sq = float(goal @ goal)
    if goal_sq <= 1e-10:
        return {}
    measured = {k: band(images[k])-anchor for k in ARMS[2:]}
    measured['B_A'] = measured['B']-measured['A']
    output = {}
    for k, value in measured.items():
        dot = float(value @ goal)
        norm_sq = float(value @ value)
        output[k] = dict(dot=dot, norm_sq=norm_sq, goal_sq=goal_sq, projection=dot/goal_sq,
            energy_ratio=float(np.sqrt(norm_sq/goal_sq)), mae_to_goal=float(np.abs(value-goal).mean()))
    return output


def probe_stats(rows, features, labels, limits):
    out = {}
    for name in ('B', *PROBES):
        scores = [r['edits'].get(name, r['scores']['B']) for r in rows]
        metrics = {k: float(np.mean([x[k] for x in scores])) for k in METRICS}
        bands = {c: float(np.mean([r['edits'].get(name, r['texture']['B'])['band_mae']
            for r in rows if r['category']==c and r['interior_px']>=256])) for c in CATEGORIES}
        fid = roi.fid_terms(features[name], features['gt'])
        keep = labels=='pants'
        pants = roi.fid_terms(features[name][keep], features['gt'][keep])
        checks = dict(fid=fid['fid']<limits['fid'], lpips=metrics['lpips']<=limits['lpips'],
            ssim=metrics['ssim']>=limits['ssim'], contour_f1=metrics['contour_f1']>=limits['contour_f1'])
        checks.update({k: bands[k]<=limits['band_mae'][k] for k in CATEGORIES})
        out[name] = dict(fid=fid, pants_fid=pants, metrics=metrics, band_mae=bands,
                         checks=checks, edited=sum(name in r['edits'] for r in rows))
    return out


def summaries(rows):
    output = {}
    for category in CATEGORIES:
        subset = [r for r in rows if r['category']==category]
        regional = {}
        for arm in ARMS:
            regional[arm] = {}
            for name in dict.fromkeys((*regions.region_names(category), 'deep')):
                values = [r['regional'][arm][name] for r in subset if name in r['regional'][arm]]
                regional[arm][name] = dict(n=len(values), **{k: float(np.mean(take)) for k in prior.FIELDS
                    if (take := [v[k] for v in values if v.get(k) is not None])})
        directions = {}
        for arm in ('A', 'B', 'B_A'):
            items = [r['response'][arm] for r in subset if arm in r['response']]
            directions[arm] = dict(n=len(items),
                pooled_projection=sum(v['dot'] for v in items)/sum(v['goal_sq'] for v in items),
                median_energy_ratio=float(np.median([v['energy_ratio'] for v in items])),
                median_projection=float(np.median([v['projection'] for v in items])),
                mae_to_goal=float(np.mean([v['mae_to_goal'] for v in items])))
        output[category] = dict(regional=regional, directions=directions)
    return output


@torch.inference_mode()
def run(data):
    args, dest, pairs, hashes, albums, features, expected, assets, limits, cache, old, prior_rows = data
    if (dest/'review.zip').exists():
        assert tail.sha(dest/'result.json') == tail.read(dest/'review_manifest.json')['result.json']
        return
    dest.mkdir(exist_ok=True)
    lpips, inception = tail.metric_models('cpu', assets)
    labels = np.array([p['category'] for p in pairs])
    rows = [None]*len(pairs)
    pools = {key: {} for key in COMPARISONS}
    features.update({k: features['B'].copy() for k in PROBES})
    counts = Counter()

    for category in CATEGORIES:
        indexes = np.flatnonzero(labels==category)
        names = regions.region_names(category)
        end = (1<<len(names))-1
        for key, (left, right) in COMPARISONS.items():
            pools[key][category] = {0: features[left][indexes], end: features[right][indexes]}
            for b in range(1, end):
                pools[key][category][b] = np.empty_like(features[right][indexes])
        for offset in range(0, len(indexes), 4):
            ids = indexes[offset:offset+4]
            batch = []
            for i in ids:
                pair = pairs[i]
                images, gt, coarse_mask, coarse, row = prior.previous.inputs(args,pair,hashes,albums,i)
                mask = tail.case_mask(args,pair)
                hard, soft = regions.split_regions(mask,coarse_mask,coarse,category)
                core = roi.regions(mask,20)[0]
                before = prior_rows[i]
                assert before['interior_px'] == row['interior_px']
                assert all(before['scores'][k]==row['scores'][k] and before['texture'][k]==row['texture'][k]
                           for k in ARMS[:2])
                row['regional'].update({k: before['regional'][k] for k in ARMS[:2]})
                row['edits'] = {}
                goal = target_band50(images['step960'],gt,core)[0] if core.sum()>=256 else None
                row['response'] = {} if goal is None else response(images,goal,core)
                rows[i] = row
                batch.append((images,gt,mask,core,hard,soft,coarse,row))
            pictures, targets, masks, cores, hard, soft, coarse, out = zip(*batch)
            target_tensor = roi.tensors(targets)
            target_maps = roi.lpips_features(lpips,target_tensor)
            target_lab = [roi.lab(y) for y in targets]
            truth = [(y, roi.band(y[...,0])) for y in target_lab]
            hard_tensor = torch.from_numpy(np.stack(hard)).float()
            for arm in ARMS[2:]:
                tensor = roi.tensors([x[arm] for x in pictures])
                local = roi.apportion(roi.maps(lpips,target_maps,tensor),hard_tensor)
                for j, (x,y,row) in enumerate(zip(pictures,targets,out)):
                    ssim,part,_ = roi.ssim_regions(x[arm],y,hard[j])
                    np.testing.assert_allclose([local[j].sum(),ssim],
                        [row['scores'][arm]['lpips'],row['scores'][arm]['ssim']],atol=2e-5,rtol=2e-4)
                    stats = prior.region_measure(x[arm],truth[j],names,hard[j],cores[j],local[j],part)
                    if row['interior_px']>=256:
                        np.testing.assert_allclose(stats['deep']['mae'],row['texture'][arm]['band_mae'],atol=1e-9,rtol=0)
                    row['regional'][arm] = stats

            def probe(names_for_edit, samples, selected, tensor, vector):
                distances = lpips(tensor*2-1,target_tensor[selected]*2-1).flatten().tolist()
                for name in names_for_edit:
                    features[name][ids[selected]] = vector
                for sample,j,dist in zip(samples,selected,distances):
                    score = dict(lpips=dist, ssim=roi.ssim_regions(sample,targets[j],roi.regions(masks[j],8))[0],
                        contour_f1=roi.shape_score(sample,masks[j])['contour_f1'],
                        band_mae=tail.appearance(sample,target_lab[j],cores[j])['band_mae'])
                    out[j]['edits'].update(dict.fromkeys(names_for_edit,score))

            for key,(left,right) in COMPARISONS.items():
                for bits in range(1,end):
                    mixed = [regions.mix((x[left],x[right]),w,c,bits,category)
                             for x,w,c in zip(pictures,soft,coarse)]
                    tensor = roi.tensors(mixed)
                    vector = inception(tensor)[0].flatten(1).numpy()
                    pools[key][category][bits][offset:offset+len(ids)] = vector
                    counts['new_swap_features'] += len(ids)
                    if key=='off_B' and category=='pants' and bits in (14,13):
                        probe((('pants_deep_off' if bits==14 else 'pants_rim_off'),),
                              mixed,list(range(len(ids))),tensor,vector)

            edits = [prior.lstar.edit(x['B'],y,m)[0] if r['interior_px']>=256 else {}
                     for x,y,m,r in zip(pictures,targets,masks,out)]
            for key in (('band50','low_l50') if category=='pants' else ('band50',)):
                selected = [j for j,e in enumerate(edits) if key in e and e[key][1]['changed']]
                if not selected:
                    continue
                samples = [edits[j][key][0] for j in selected]
                tensor = roi.tensors(samples)
                vector = inception(tensor)[0].flatten(1).numpy()
                variants = (('all_band50','pants_band50') if category=='pants' else ('all_band50',)) if key=='band50' else ('pants_low50',)
                probe(variants,samples,selected,tensor,vector)
                counts['new_paired_probe_features'] += len(selected)
        print('REGION_127791_CATEGORY', category, dict(counts), flush=True)

    with np.load(cache/'region_pool3.npz', allow_pickle=False) as saved:
        reused = sum(len(saved[k]) for k in saved.files if k.startswith('step960_'))
    assert reused == 2048 and counts['new_swap_features'] == 6144
    groups = {}
    for name in ARMS:
        groups[name] = {}
        for category in ('all', *CATEGORIES):
            keep = np.ones(len(labels),bool) if category=='all' else labels==category
            value = roi.fid_terms(features[name][keep],features['gt'][keep])
            previous_value = expected[name][category]['arms'][albums[name][1]]['fid']
            np.testing.assert_allclose([value[k] for k in FIELDS], [previous_value[k] for k in FIELDS],atol=1e-9,rtol=1e-9)
            groups[name][category] = value
    attribution = {'off_step960': old['attribution']['step960']}
    for key, (left, right) in COMPARISONS.items():
        attribution[key] = regions.contribution(pools[key],dict(step0=features[left],candidate=features[right]),labels,features['gt'])
    feature_path = dest/'region_pool3.npz'
    np.savez_compressed(feature_path, **{f'{name}_{c}_{b}': v for name, cats in pools.items()
        for c, vals in cats.items() for b, v in vals.items() if b not in (0,len(vals)-1)},
        **{k: features[k] for k in PROBES})
    result = dict(pairs=len(pairs),groups=groups,attribution=attribution,texture=summaries(rows),
        probes=probe_stats(rows,features,labels,limits),thresholds=limits,counts=dict(counts),
        reused_960_region_features=reused, source_review_sha256=RUN_SHA, old_region_review_sha256=PRIOR_SHA,
        region_pool3_sha256=tail.sha(feature_path),new_diffusion_samples=0,optimizer_steps=0)
    tail.write(dest/'result.json',result)
    tail.write(dest/'per_pair.json',rows)
    init.fid.pack(dest,['result.json','per_pair.json'])
    print('REGION_127791_DONE',tail.sha(dest/'review.zip'),flush=True)


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('prior',type=Path)
    cli.add_argument('--check',action='store_true')
    opt = cli.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
    data = load(opt.prior)
    if opt.check:
        args, _, pairs, hashes, albums, *_ = data
        for category in CATEGORIES:
            i = next(i for i,p in enumerate(pairs) if p['category']==category)
            prior.previous.inputs(args,pairs[i],hashes,albums,i)
        print('REGION_127791_PRECHECK',len(pairs),flush=True)
    else:
        run(data)
