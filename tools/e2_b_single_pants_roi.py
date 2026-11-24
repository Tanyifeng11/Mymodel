"""复用 H2 第768步的验证图，分开干预裤装内部的低频亮度与带通纹理。"""
import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from tools import e2_b_epoch20_single as single

fusion, init, tail, roi = single.fusion, single.init, single.tail, single.roi
RUN = 'pants_lstar_band_step768'
EDITS = ('low_lstar', 'band_texture')
ARMS = ('step0', 'h2_768', *EDITS)
CORE_RADIUS, FEATHER_WIDTH, LOW_SIGMA = 20, 4, 8


def edit(image, target, mask):
    core = roi.regions(mask,CORE_RADIUS)[0]
    w = np.minimum(cv2.distanceTransform(core.astype('uint8'),cv2.DIST_L2,5)/FEATHER_WIDTH,1)
    lab = roi.lab(image)
    diff = target[...,0]-lab[...,0]
    if not core.any():
        return {k:image.copy() for k in EDITS},core
    support = roi.regions(mask,8)[0].astype('float32')
    low = roi.gaussian(diff*support,LOW_SIGMA)/np.maximum(roi.gaussian(support,LOW_SIGMA),1e-6)
    band = roi.band(diff)
    roundtrip = roi.as_rgb(lab)
    images = {}
    for name,delta in zip(EDITS,(low,band)):
        corrected = lab.copy()
        corrected[...,0] += w*delta
        images[name] = roi.apply_lab_edit(image,roundtrip,corrected)
        np.testing.assert_array_equal(images[name][~core],image[~core])
    return images,core


def load_validation(run,manifest,step):
    rel = f'validation/step{step}'
    results = fusion.read_checked(run/rel/'result.json',manifest[rel+'/result.json'])
    assert tail.sha(run/rel/'pool3.npz') == results['pool3_sha256']
    rows = tail.read(run/rel/'per_pair.json')
    with np.load(run/rel/'pool3.npz',allow_pickle=False) as pool:
        return results,rows,pool['H2'],pool['gt']


@torch.no_grad()
def run(prior):
    args = fusion.paths(prior,'cpu')
    cohort_path = args.run/'protocol.json'
    cohort = tail.read(cohort_path)
    args.out = args.run
    source = args.baseline/single.RUN
    dest = source/RUN
    manifest = tail.read(source/'review_manifest.json')
    protocol = fusion.read_checked(source/'protocol.json',manifest['protocol.json'])
    assert protocol['cohort_sha256'] == tail.sha(cohort_path)
    assert protocol['base_sha256'] == init.BASE_SHA
    assert tail.metric_assets() == protocol['metrics']
    zero,zrows,zfeatures,gt = load_validation(source,manifest,0)
    h2,rows,hfeatures,hgt = load_validation(source,manifest,768)
    assert zero['protocol_sha256'] == h2['protocol_sha256'] == manifest['protocol.json']
    np.testing.assert_array_equal(gt,hgt)
    pairs = [p for p in cohort['pairs'] if p['split']=='val']
    assert len(pairs) == len(rows) == len(zrows) == 256
    assert Counter(p['category'] for p in pairs) == dict.fromkeys(tail.distill.CATEGORIES,64)
    assert [p['case_id'] for p in pairs] == [r['case_id'] for r in rows] == [r['case_id'] for r in zrows]
    assert hfeatures.shape == zfeatures.shape == gt.shape == (256,2048)
    summary_rows = []
    features = dict(step0=zfeatures,h2_768=hfeatures,gt=gt)
    features.update({n:hfeatures.copy() for n in EDITS})
    counts = Counter()
    models = tail.metric_models('cpu',protocol['metrics'])
    # 原评分固定 batch=4，保留相同顺序和批量。三类别的图分数和特征直接复用。
    for start in range(0,len(rows),4):
        pending = []
        for i in range(start,start+4):
            p,old,h = pairs[i],zrows[i],rows[i]
            assert p['category'] == h['category'] == old['category']
            case = p['case_id']
            picture = source/'renders'/h2['head_sha256']['H2']/(case+'.png')
            assert tail.sha(picture) == h['pngs']['H2']
            row = dict(case_id=case,category=p['category'],interior_px=h['interior_px'],
                pngs=dict(step0=old['pngs']['H2'],h2_768=h['pngs']['H2']),
                scores=dict(step0=old['scores']['H2'],h2_768=h['scores']['H2']),
                texture=dict(step0=old['texture']['H2'],h2_768=h['texture']['H2'],gt=h['texture']['gt']),
                changed_pixels=dict.fromkeys(EDITS,0))
            for n in EDITS:
                for field in ('pngs','scores','texture'):
                    row[field][n] = row[field]['h2_768']
            summary_rows.append(row)
            if p['category'] != 'pants':
                continue
            path = args.data/p['gt']
            assert tail.sha(path) == cohort['gt_sha256'][case]
            target = tail.rgb(path).resize(tail.SIZE,Image.Resampling.BILINEAR)
            truth = roi.lab(np.asarray(target))
            image = np.asarray(tail.rgb(picture))
            mask = tail.case_mask(args,p)
            edited,core = edit(image,truth,mask)
            assert int(core.sum()) == h['interior_px'] == old['interior_px']
            pending.append((i,row,mask,path,target,truth,image,core,edited))
        if not pending:
            continue
        assert len(pending) == 4  # 原归档每类连续64图。
        lpips,inception = models
        original = roi.tensors([item[6] for item in pending])
        targets = roi.tensors([np.asarray(item[4]) for item in pending])
        indices = [item[0] for item in pending]
        np.testing.assert_allclose(inception(original)[0].flatten(1).numpy(),hfeatures[indices],atol=7e-5,rtol=1e-4)
        np.testing.assert_allclose(inception(targets)[0].flatten(1).numpy(),gt[indices],atol=7e-5,rtol=1e-4)
        for name in EDITS:
            tensors = roi.tensors([item[-1][name] for item in pending])
            distances = lpips(tensors*2-1,targets*2-1).flatten().tolist()
            features[name][indices] = inception(tensors)[0].flatten(1).numpy()
            for item,dist in zip(pending,distances):
                i,row,mask,path,target,truth,image,core,edited = item
                output = edited[name]
                count = int(np.any(output!=image,axis=-1).sum())
                row['changed_pixels'][name] = count
                counts[name] += int(count>0)
                if not count:
                    features[name][i] = hfeatures[i]
                    continue
                png = dest/'images'/name/(row['case_id']+'.png')
                single.ft.fused.save_png(png,output)
                row['pngs'][name] = tail.sha(png)
                measurements = tail.image_scores(Image.fromarray(output),target,png,path,mask)
                row['scores'][name] = dict(measurements,lpips=dist)
                row['texture'][name] = tail.appearance(output,truth,core)
        print('PANTS_ROI_SCORED',start+4,dict(counts),flush=True)
    result = fusion.summarize(summary_rows,features,ARMS)
    for group,summary in result['groups'].items():
        for name,stored in (('step0',zero),('h2_768',h2)):
            observed = init.metrics_from_group(summary,name)
            expected = init.metrics_from_group(stored['groups'][group],'H2')
            np.testing.assert_allclose(list(observed.values()),list(expected.values()),atol=1e-9,rtol=1e-9)
    overall = result['groups']['all']
    base = init.metrics_from_group(overall,'h2_768')
    threshold = init.metrics_from_group(overall,'step0')['fid']
    tolerance = single.rules.TOLERANCE
    acceptance = {}
    for name in EDITS:
        now = init.metrics_from_group(overall,name)
        checks = dict(fid_below_step0=now['fid']<threshold,lpips_kept=now['lpips']<=base['lpips'],
                      ssim_kept=now['ssim']>=base['ssim']*(1-tolerance),
                      f1_kept=now['contour_f1']>=base['contour_f1']*(1-tolerance))
        acceptance[name] = dict(checks,passed=all(checks.values()),band_mae_delta=now['band_mae']-base['band_mae'])
    result.update(acceptance=acceptance,pairs=256,pants_pairs=64,changed_cases=dict(counts),
                  edge_background_max_rgb_difference=0,new_diffusion_samples=0,optimizer_steps=0)
    tail.write(dest/'protocol.json',dict(source=single.RUN,source_step=768,arms=list(ARMS),source_hashes={
        'cohort':tail.sha(cohort_path),'original_result':manifest['validation/step768/result.json'],
        'step0_result':manifest['validation/step0/result.json'],'source_protocol':manifest['protocol.json']},
        target='paired GT, diagnostic only',core_radius=CORE_RADIUS,feather_width=FEATHER_WIDTH,
        low_sigma=LOW_SIGMA,band_sigma=[1,4],lab_change='L only before RGB roundtrip',
        reference='FID strictly below step0; LPIPS no worse than H2; SSIM and F1 within 0.5% of H2'))
    tail.write(dest/'result.json',result)
    tail.write(dest/'per_pair.json',summary_rows)
    np.savez_compressed(dest/'pool3.npz',**features)
    init.fid.pack(dest,['protocol.json','result.json','per_pair.json','pool3.npz'])
    print('PANTS_ROI_DONE',acceptance,'review_sha256',tail.sha(dest/'review.zip'),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prior',type=Path)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    run(parser.parse_args().prior)
