"""对 127556 的 B 第64步做等幅裤装深衣内 L* 离线干预。"""
import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from models.e2_lab_anchor import lab_to_rgb, rgb_to_lab
from tools import e2_b_single_fid_deep_rim as prior_regions
from tools.e2_b_single_pants_roi import CORE_RADIUS, FEATHER_WIDTH, LOW_SIGMA

fusion, init, tail, roi = prior_regions.fusion, prior_regions.init, prior_regions.tail, prior_regions.roi
RUN = 'pants_deep_lstar_equal50_127556'
ARMS = ('step0', 'start', 'b64', 'band50', 'low_l50')
EDITS = ARMS[-2:]
FRACTION = .5


def lab_pixels(image):
    rgb = torch.from_numpy(np.asarray(image).copy()).to(torch.float64)
    return rgb_to_lab(rgb.permute(2,0,1)[None]/255)[0].permute(1,2,0).numpy()


def rgb_pixels(lab):
    x = torch.from_numpy(np.ascontiguousarray(lab.T))[None,:,:,None]
    return lab_to_rgb(x)[0,:,:,0].T.numpy()


def apply_lstar(image, lab, delta, core):
    out = image.copy()
    if not core.any():
        return out,dict(pixels=0,gamut_limited=0,changed=0,requested_abs_l=0.,applied_abs_l=0.,
                        chroma_sum=0.,chroma_max=0.)
    values = lab[core].copy()
    light = values[:,0].copy()
    request = delta[core].astype('float64')
    values[:,0] = light+request
    rgb = rgb_pixels(values)
    limit = ((rgb < -1e-6) | (rgb > 1+1e-6)).any(axis=1)
    if limit.any():
        candidate = values[limit].copy()
        base, change = light[limit],request[limit]
        lo,hi = np.zeros_like(base),np.ones_like(base)
        for _ in range(15):
            mid = (lo+hi)*.5
            candidate[:,0] = base+mid*change
            trial = rgb_pixels(candidate)
            legal = ((trial >= -1e-6) & (trial <= 1+1e-6)).all(axis=1)
            lo = np.where(legal,mid,lo)
            hi = np.where(legal,hi,mid)
        values[limit,0] = base+lo*change
        rgb[limit] = rgb_pixels(values[limit])
    np.testing.assert_array_equal(values[:,1:],lab[core,1:])
    assert ((rgb >= -1e-6) & (rgb <= 1+1e-6)).all()
    out[core] = np.rint(rgb.clip(0,1)*255).astype('uint8')
    np.testing.assert_array_equal(out[~core],image[~core])
    delta_ab = np.linalg.norm(lab_pixels(out)[core,1:]-lab[core,1:],axis=1)
    return out,dict(pixels=int(core.sum()),gamut_limited=int(limit.sum()),
        changed=int(np.any(out[core]!=image[core],axis=1).sum()),requested_abs_l=float(np.abs(request).sum()),
        applied_abs_l=float(np.abs(values[:,0]-light).sum()),chroma_sum=float(delta_ab.sum()),
        chroma_max=float(delta_ab.max()))


def edit(image,target,mask):
    core = roi.regions(mask,CORE_RADIUS)[0]
    weight = np.minimum(cv2.distanceTransform(core.astype('uint8'),cv2.DIST_L2,5)/FEATHER_WIDTH,1)
    source,truth = lab_pixels(image),lab_pixels(target)
    difference = truth[...,0]-source[...,0]
    support = roi.regions(mask,8)[0].astype('float64')
    low = roi.gaussian(difference*support,LOW_SIGMA)/np.maximum(roi.gaussian(support,LOW_SIGMA),1e-6)
    values = (roi.band(difference),low)
    return {n:apply_lstar(image,source,FRACTION*weight*x,core) for n,x in zip(EDITS,values)},core


def load(prior):
    args,regional,pairs,hashes,albums,features,expected,assets,_ = prior_regions.load(prior)
    assert tail.sha(regional/'review.zip')=='cf8e70092513f68e61df6bd47818fee46285fdcc02f3bb15a14d7cfe765f81db'
    manifest = tail.read(regional/'review_manifest.json')
    d = fusion.read_checked(regional/'result.json',manifest['result.json'])
    assert d['pairs']==256
    thresholds = d['thresholds']
    assert thresholds['fid']==expected['step0']['all']['fid']
    return args,regional.parent/RUN,pairs,hashes,albums,features,expected,assets,thresholds


def gates(metrics,pants,thresholds):
    checks = dict(fid=metrics['fid']<thresholds['fid'],lpips=metrics['lpips']<=thresholds['lpips'],
        ssim=metrics['ssim']>=thresholds['ssim'],contour_f1=metrics['contour_f1']>=thresholds['contour_f1'],
        pants_band_mae=pants['band_mae']<=thresholds['pants_band_mae'])
    return dict(checks=checks,passed=all(checks.values()))


@torch.inference_mode()
def run(data):
    args,dest,pairs,hashes,albums,raw,expected,assets,thresholds = data
    if (dest/'review.zip').exists():
        assert tail.sha(dest/'result.json')==tail.read(dest/'review_manifest.json')['result.json']
        return
    dest.mkdir(exist_ok=True,parents=True)
    features = {k:raw[k] for k in (*ARMS[:3],'gt')}
    features.update({k:raw['b64'].copy() for k in EDITS})
    lpips,inception = tail.metric_models('cpu',assets)
    rows, counts = [], Counter()
    for offset in range(0,len(pairs),4):
        pending = []
        for i,p in enumerate(pairs[offset:offset+4],offset):
            case = p['case_id']
            row = dict(case_id=case,category=p['category'],pngs={},scores={},texture={},audit={})
            for n in ARMS[:3]:
                folder,key,source = albums[n]
                old = source[i]
                row['pngs'][n],row['scores'][n],row['texture'][n] = (
                    old['pngs'][key],old['scores'][key],old['texture'][key])
                row['interior_px'] = old['interior_px']
                row['texture']['gt'] = old['texture']['gt']
            for n in EDITS:
                for field in ('pngs','scores','texture'):
                    row[field][n] = row[field]['b64']
            path = albums['b64'][0]/(case+'.png')
            assert tail.sha(path)==row['pngs']['b64']
            rows.append(row)
            if p['category']!='pants':
                continue
            target_path = args.data/p['gt']
            assert tail.sha(target_path)==hashes[case]
            gt = tail.rgb(target_path).resize(tail.SIZE,Image.Resampling.BILINEAR)
            mask = tail.case_mask(args,p)
            edited,core = edit(np.asarray(tail.rgb(path)),np.asarray(gt),mask)
            assert int(core.sum())==row['interior_px']
            row['audit'] = {n:v[1] for n,v in edited.items()}
            pending.append((i,row,gt,target_path,mask,core,edited))
        if not pending:
            continue
        assert len(pending)==4
        target = roi.tensors([np.asarray(v[2]) for v in pending])
        for n in EDITS:
            if not any(v[-1][n][1]['changed'] for v in pending):
                continue
            tensors = roi.tensors([v[-1][n][0] for v in pending])
            value = inception(tensors)[0].flatten(1).numpy()
            distances = lpips(tensors*2-1,target*2-1).flatten().tolist()
            for j,(i,row,gt,path,mask,core,edited) in enumerate(pending):
                image,audit = edited[n]
                if not audit['changed']:
                    continue
                png = dest/'images'/n/(row['case_id']+'.png')
                prior_regions.short.single.ft.fused.save_png(png,image)
                row['pngs'][n] = tail.sha(png)
                row['scores'][n] = dict(tail.image_scores(Image.fromarray(image),gt,png,path,mask),lpips=distances[j])
                row['texture'][n] = tail.appearance(image,roi.lab(np.asarray(gt)),core)
                features[n][i] = value[j]
                counts[n]+=1
        print('PANTS_EQUAL50_SCORED',offset+4,dict(counts),flush=True)
    assert len(rows)==256
    result = fusion.summarize(rows,features,ARMS)
    for c,g in result['groups'].items():
        for n in ARMS[:3]:
            observed = g['arms'][n]['fid']
            np.testing.assert_allclose([observed[k] for k in prior_regions.FID_FIELDS],
                [expected[n][c][k] for k in prior_regions.FID_FIELDS],atol=1e-9,rtol=1e-9)
    result['acceptance'] = {
        n:gates(init.metrics_from_group(result['groups']['all'],n),result['groups']['pants']['texture'][n],thresholds)
        for n in ARMS[2:]}
    color = {}
    for n in EDITS:
        audits = [r['audit'][n] for r in rows if r['category']=='pants']
        total = {k:sum(a[k] for a in audits) for k in audits[0] if k!='chroma_max'}
        color[n] = dict(**total,mean_png_chroma_drift=total['chroma_sum']/total['pixels'],
                        max_png_chroma_drift=max(a['chroma_max'] for a in audits))
    pool = dest/'pool3.npz'
    np.savez_compressed(pool,**features)
    result.update(pairs=256,pants_pairs=64,changed_cases=dict(counts),thresholds=thresholds,
        pool3_sha256=tail.sha(pool),chroma_audit=color,outside_core_max_rgb_abs=0,
        optimizer_steps=0,new_diffusion_samples=0)
    tail.write(dest/'protocol.json',dict(source_job=127556,candidate='B step64',source_diagnostic=127590,
        fraction=FRACTION,core_radius=CORE_RADIUS,feather_px=FEATHER_WIDTH,band_sigma=[1,4],low_sigma=LOW_SIGMA,
        low_support_radius=8,edit='hold Lab a*,b*; cap L* along same chroma when outside sRGB gamut',
        gt_use='paired validation diagnostic only',thresholds=thresholds))
    tail.write(dest/'result.json',result)
    tail.write(dest/'per_pair.json',rows)
    init.fid.pack(dest,['protocol.json','result.json','per_pair.json'])
    print('PANTS_EQUAL50_DONE',result['acceptance'],tail.sha(dest/'review.zip'),flush=True)


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('prior',type=Path)
    cli.add_argument('--check',action='store_true')
    opts=cli.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    data=load(opts.prior)
    if opts.check:
        _,_,pairs,_,albums,*_ = data
        i,p=next((i,p) for i,p in enumerate(pairs) if p['category']=='pants')
        folder,key,rows=albums['b64']
        assert tail.sha(folder/(p['case_id']+'.png'))==rows[i]['pngs'][key]
        print('PANTS_EQUAL50_PRECHECK_OK',len(pairs),flush=True)
    else:
        run(data)
