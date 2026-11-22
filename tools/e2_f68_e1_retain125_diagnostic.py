"""复用 126686 的终图和记录，拆解损失、感知、FID 与前景边界变化。"""
import argparse
import math
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from tools import e2_f68_e1_fid_regions as fid
from tools import e2_f68_e1_retain125_finetune as ft
from tools import e2_f68_e1_retention500 as expand
from tools import e2_f68_region_diagnostic as roi
from tools import e2_f68_tail6_real_ab as tail


RUN = 'loss_region_diagnostic_20261122'
SOURCE_SHA = 'ed5b0d1cff1b8cde198732058402db4db0e2f899dd45cffe1f00c4797102b311'
BASE, NEW = ft.BASE, ft.ARM
REGIONS = ('inside','edge','background')
CATEGORIES = ('dress','outwear','pants','top')
WEIGHTS = dict(rgb_l1=1.,lpips=.05,inside_lpips=.05,low_lstar=.10)
METRICS = ('lpips','ssim','psnr','contour_f1','iou','leak')
FIELDS = ('fid','fid_mean','fid_covariance',*METRICS,'band_mae')


def shapley(values, names):
    n = len(names)
    result = {}
    for i,name in enumerate(names):
        contribution = np.zeros_like(values[0],dtype='float64')
        for subset in range(1<<n):
            if subset & (1<<i):
                continue
            size = bin(subset).count('1')
            weight = math.factorial(size)*math.factorial(n-size-1)/math.factorial(n)
            contribution += weight*(values[subset|(1<<i)]-values[subset])
        result[name] = contribution
    np.testing.assert_allclose(sum(result.values()),values[-1]-values[0],atol=1e-9,rtol=1e-8)
    return result


def loss_report(log):
    report = {}
    for category in ('all',*CATEGORIES):
        groups = {step:[r for r in doc['rows'] if category=='all' or r['case'].startswith(category+'_')]
                  for step,doc in log['validation'].items()}
        initial = {r['case']:r for r in groups['0']}
        start = {k:float(np.mean([r[k] for r in initial.values() if r[k] is not None]))
                 for k in (*WEIGHTS,'loss','band_mae')}
        steps = {}
        for step,rows in groups.items():
            means = {k:float(np.mean([r[k] for r in rows if r[k] is not None]))
                     for k in (*WEIGHTS,'loss','band_mae')}
            contributions = {k:WEIGHTS[k]*(means[k]-start[k]) for k in WEIGHTS}
            np.testing.assert_allclose(sum(contributions.values()),means['loss']-start['loss'],atol=2e-8,rtol=0)
            guard = {k:float(np.mean([max(0.,r[k]-initial[r['case']][k]) for r in rows]))
                     for k in ('lpips','inside_lpips')}
            steps[step] = dict(n=len(rows),means=means,weighted_delta=contributions,
                perceptual_guard=guard,guard_weight_one_delta=means['loss']-start['loss']+sum(guard.values()),
                perception_eligible=means['lpips']<start['lpips'] and means['inside_lpips']<=start['inside_lpips'])
        report[category] = steps
    trace = log['trace']
    for row in trace:
        np.testing.assert_allclose(sum(WEIGHTS[k]*row[k] for k in WEIGHTS),row['loss'],atol=1e-8,rtol=5e-7)
    return dict(validation=report,trace_samples=len(trace),
                trace_means={k:float(np.mean([r[k] for r in trace])) for k in (*WEIGHTS,'loss')},
                fallback_step=0,eligible_logged_steps=[int(k) for k,v in report['all'].items()
                                                      if k!='0' and v['perception_eligible']])


def appearance(image, truth, regions):
    color,gt = roi.lab(image),roi.lab(truth)
    diff = color-gt
    low = roi.gaussian(diff,8)
    band,gt_band = roi.band(color[...,0]),roi.band(gt[...,0])
    result = {}
    for name,zone in zip(REGIONS,regions):
        rms = roi.rms(band[zone])
        target = roi.rms(gt_band[zone])
        result[name] = dict(rgb_mae=float(np.abs(image.astype('float32')-truth)[zone].mean()),
            l_bias=float(diff[zone,0].mean()),low_l_mae=float(np.abs(low[zone,0]).mean()),
            low_ab_error=float(np.linalg.norm(low[zone,1:],axis=1).mean()),
            band_mae=float(np.abs(band-gt_band)[zone].mean()),band_rms=rms,
            gt_band_rms=target,band_rms_error=abs(rms-target))
    return result


def foreground(image):
    mask = roi.estimate_foreground_mask(Image.fromarray(image),tail.SIZE)
    if mask is None:
        mask = np.zeros(image.shape[:2],dtype=bool)
    pixels = image.astype('float32')
    border = np.concatenate([pixels[0],pixels[-1],pixels[:,0],pixels[:,-1]],axis=0)
    background = np.median(border,axis=0)
    raw = (np.linalg.norm(pixels-background,axis=2)>18.) | (pixels.mean(axis=2)<235.)
    return mask,raw,background


def vector(rows, features, gt, subset):
    terms = roi.fid_terms(features[subset],gt[subset])
    good = [r for r in rows if r['interior_px']>=256]
    scores = [terms['fid'],terms['mean_term'],terms['covariance_term']]
    for metric in METRICS:
        scores.append(float(np.mean([r['scores'][metric] for r in rows])))
    scores.append(float(np.mean([r['texture']['band_mae'] for r in good])))
    return np.array(scores)


def summarize(rows, features, gt):
    labels = np.array([r['category'] for r in rows])
    groups = {}
    for category in ('all',*CATEGORIES):
        keep = np.ones(len(rows),bool) if category=='all' else labels==category
        picked = [r for r,k in zip(rows,keep) if k]
        coalitions = []
        for bits in range(8):
            converted = [dict(interior_px=r['interior_px'],scores=r['scores'][str(bits)],
                              texture=r['texture'][str(bits)]) for r in picked]
            coalitions.append(vector(converted,features[bits],gt,keep))
        effects = shapley(coalitions,REGIONS)
        info = dict(n=len(picked),arms={str(b):dict(zip(FIELDS,v.tolist())) for b,v in enumerate(coalitions)},
            region_shapley={name:dict(zip(FIELDS,v.tolist())) for name,v in effects.items()})
        info['pixel_difference'] = {reg:{k:float(np.mean([r['difference'][reg][k] for r in picked]))
                                        for k in ('rgb_mae','changed_fraction','changed_pixels','pixels')}
                                    for reg in REGIONS}
        info['native_lpips_delta'] = dict(zip(REGIONS,np.mean([
            np.array(r['lpips_region'][NEW])-r['lpips_region'][BASE] for r in picked],axis=0).tolist()))
        info['native_ssim_delta'] = dict(zip(REGIONS,np.mean([
            np.array(r['ssim_region'][NEW])-r['ssim_region'][BASE] for r in picked],axis=0).tolist()))
        info['appearance'] = {arm:{reg:{k:float(np.mean([r['appearance'][arm][reg][k] for r in picked]))
                                       for k in picked[0]['appearance'][arm][reg]} for reg in REGIONS}
                              for arm in (BASE,NEW)}
        info['foreground_changed_pixels'] = {reg:{key:float(np.mean([r['foreground'][reg][key] for r in picked]))
                                                   for key in ('raw','final')} for reg in REGIONS}
        groups[category] = info

    # 类别按全部替换顺序平均，类内三个区域再按全部顺序平均。
    # 所有类别区域贡献加总等于同批整组 FID 差值。
    contribution = {}
    for i,category in enumerate(CATEGORIES):
        keep = labels==category
        region_effect = {reg:np.zeros(3) for reg in REGIONS}
        for subset in range(16):
            if subset & (1<<i):
                continue
            size = bin(subset).count('1')
            weight = math.factorial(size)*math.factorial(3-size)/math.factorial(4)
            mixed = features[0].copy()
            for j,other in enumerate(CATEGORIES):
                if subset & (1<<j):
                    mixed[labels==other] = features[7][labels==other]
            values = []
            for bits in range(8):
                mixed[keep] = features[bits][keep]
                f = roi.fid_terms(mixed,gt)
                values.append(np.array([f['fid'],f['mean_term'],f['covariance_term']]))
            for reg,v in shapley(values,REGIONS).items():
                region_effect[reg] += weight*v
        contribution[category] = {reg:dict(zip(FIELDS[:3],v.tolist())) for reg,v in region_effect.items()}
    total = np.array([sum(v[reg][key] for v in contribution.values() for reg in REGIONS) for key in FIELDS[:3]])
    expected = np.array([groups['all']['arms']['7'][k]-groups['all']['arms']['0'][k] for k in FIELDS[:3]])
    np.testing.assert_allclose(total,expected,atol=2e-8,rtol=1e-8)
    return dict(groups=groups,category_region_fid=contribution)


@torch.inference_mode()
def run(prior):
    args = expand.setup(prior)
    root = args.light_run/ft.RUN
    tail.verified_files(root,SOURCE_SHA,('training.json','protocol.json','eval64/review.zip','completion.json'))
    source = root/'eval64'
    tail.verified_files(source,tail.sha(source/'review.zip'),('per_pair.json','candidate_pool3.npz','result.json'))
    tail.verified_files(args.archived,expand.CONTROL_SHA,('per_pair.json','pool3.npz'))
    args.out = args.baseline
    old = tail.read(source/'per_pair.json')
    assert len(old)==64 and [r['index'] for r in old]==tail.subset(args.protocol)
    out = root/RUN
    out.mkdir(exist_ok=True)
    assert tail.metric_assets()==args.protocol['metric_assets']
    metric,inception = tail.metric_models('cpu',args.protocol['metric_assets'])
    with np.load(args.archived/'pool3.npz',allow_pickle=False) as f:
        initial = f[BASE]
    with np.load(source/'candidate_pool3.npz',allow_pickle=False) as f:
        current = f[NEW]
        np.testing.assert_array_equal(f['indices'],[r['index'] for r in old])
    with np.load(args.light_run/'eval64/pool3.npz',allow_pickle=False) as f:
        gt_features = f['gt']
    features = {0:initial,7:current,**{i:np.zeros_like(initial) for i in range(1,7)}}
    rows = []
    with tempfile.TemporaryDirectory(prefix='retain125_regions_',dir=os.environ.get('SLURM_TMPDIR')) as temp:
        scratch = Path(temp)/'score.png'
        for start in range(0,len(old),4):
            previous = old[start:start+4]
            variants = {i:[] for i in range(8)}
            truth,parts = [],[]
            for record in previous:
                pair = args.protocol['pairs'][record['index']]
                case = record['case_id']
                assert (pair['case_id'],pair['category'])==(case,record['category'])
                mask = tail.case_mask(args,pair)
                core = roi.regions(mask,20)[0]
                assert int(core.sum())==record['interior_px']
                path = args.data/pair['gt']
                assert tail.sha(path)==pair['input_sha256']['gt']
                gt = tail.rgb(path).resize(tail.SIZE,Image.Resampling.BILINEAR)
                target = np.asarray(gt)
                images = {BASE:fid.checked_png(args.archived/'images'/BASE/(case+'.png'),record['pngs'][BASE]),
                          NEW:fid.checked_png(source/'images'/(case+'.png'),record['pngs'][NEW])}
                hard = roi.regions(mask,8)
                weights = fid.regions(mask)
                np.testing.assert_allclose(sum(weights),1,atol=1e-6,rtol=0)
                item = {k:record[k] for k in ('index','case_id','category','interior_px')}
                item.update(scores={'0':record['scores'][BASE],'7':record['scores'][NEW]},
                            texture={'0':record['texture'][BASE],'7':record['texture'][NEW]},
                            difference={},appearance={},foreground={},lpips_region={},ssim_region={})
                difference = images[NEW].astype('float32')-images[BASE]
                for name,zone in zip(REGIONS,hard):
                    change = np.any(difference[zone]!=0,axis=1)
                    item['difference'][name] = dict(rgb_mae=float(np.abs(difference[zone]).mean()),
                        changed_fraction=float(change.mean()),changed_pixels=int(change.sum()),pixels=int(zone.sum()))
                masks = {a:foreground(img) for a,img in images.items()}
                for name,zone in zip(REGIONS,hard):
                    item['foreground'][name] = dict(
                        raw=int(np.count_nonzero((masks[BASE][1]^masks[NEW][1]) & zone)),
                        final=int(np.count_nonzero((masks[BASE][0]^masks[NEW][0]) & zone)))
                for name,img in images.items():
                    item['appearance'][name] = appearance(img,target,hard)
                    ssim,contributions,_ = roi.ssim_regions(img,target,hard)
                    np.testing.assert_allclose(ssim,record['scores'][name]['ssim'],atol=1e-10,rtol=0)
                    item['ssim_region'][name] = contributions
                for bits in range(8):
                    if bits in (0,7):
                        img = images[BASE if bits==0 else NEW]
                    else:
                        img = ft.fused.pixels_mixed(images[BASE],images[NEW],
                            sum(weights[i] for i in range(3) if bits & (1<<i)))
                        Image.fromarray(img).save(scratch)
                        item['scores'][str(bits)] = tail.image_scores(Image.fromarray(img),gt,scratch,path,mask)
                        item['texture'][str(bits)] = tail.appearance(img,roi.lab(target),core)
                    variants[bits].append(img)
                truth.append(target)
                parts.append(hard)
                rows.append(item)
            tensors = roi.tensors(truth)
            reference = roi.lpips_features(metric,tensors)
            partition = torch.from_numpy(np.stack(parts).astype('float32'))
            for bits in range(8):
                prediction = roi.tensors(variants[bits])
                layers = roi.maps(metric,reference,prediction)
                values = sum(x.mean(dim=(-2,-1)) for x in layers).numpy()
                if bits in (0,7):
                    name = BASE if bits==0 else NEW
                    split = roi.apportion(layers,partition)
                    for i,item in enumerate(rows[start:start+len(previous)]):
                        np.testing.assert_allclose(values[i],item['scores'][str(bits)]['lpips'],atol=2e-7,rtol=1e-6)
                        np.testing.assert_allclose(split[i].sum(),values[i],atol=2e-7,rtol=1e-6)
                        item['lpips_region'][name] = split[i].tolist()
                else:
                    features[bits][start:start+len(previous)] = inception(prediction)[0].flatten(1).numpy()
                    for i,item in enumerate(rows[start:start+len(previous)]):
                        item['scores'][str(bits)]['lpips'] = float(values[i])
            print('RETAIN125_REGION',len(rows),64,flush=True)

    result = summarize(rows,features,gt_features)
    endpoints = tail.read(source/'result.json')['groups']['all']['arms']
    for bits,name in ((0,BASE),(7,NEW)):
        for key in ('lpips','ssim','fid','contour_f1'):
            expected = endpoints[name][key]['fid' if key=='fid' else 'mean']
            np.testing.assert_allclose(result['groups']['all']['arms'][str(bits)][key],expected,atol=1e-9,rtol=1e-10)
    result['loss'] = loss_report(tail.read(root/'training.json'))
    tail.write(out/'result.json',result)
    tail.write(out/'per_pair.json',rows)
    np.savez_compressed(out/'pool3.npz',**{str(i):v for i,v in features.items()},gt=gt_features)
    tail.write(out/'audit.json',dict(source_sha256=SOURCE_SHA,pairs=64,variants=8,metric_forwards_only=True,
        composite_pairs=384,diffusion_rollouts=0,optimizer_updates=0,mask_radius=8,feather_sigma=2,
        lpips_native_regions=REGIONS,fid_attribution='exact category then region Shapley',
        code_sha256=tail.sha(Path(__file__))))
    fid.pack(out,('result.json','per_pair.json','pool3.npz','audit.json'))
    print('RETAIN125_DIAGNOSTIC_DONE',tail.sha(out/'review.zip'),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prior',type=Path)
    cv2.setNumThreads(1)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
    run(parser.parse_args().prior)
