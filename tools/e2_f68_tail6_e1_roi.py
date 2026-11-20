"""复用 E1 第十轮原 64 对 PNG，对 LPIPS 和轮廓 F1 作区域诊断。"""
import argparse
import os
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt
import torch

from tools import e2_f68_region_diagnostic as roi
from tools import e2_f68_tail6_e1_epoch10 as epoch
from tools import e2_f68_tail6_real_ab as tail
from tools.e2_f68_rgb_student_roi import appearance_inputs, band_alignment, pixels_mixed


SOURCE_SHA = '3472521b98bfba47ccb06fe000512bdb7017176a0d42fd7dc1981aa877155373'
RUN = 'roi_64_20261120'
ORIGINAL = ('off', 'A', 'B', 'gate', 'blend', 'on')
SIDES = ('inside', 'edge', 'background')
EDITS = ('A_inside_blend', 'A_edge_blend', 'A_background_blend', 'A_inside_edge_blend',
         'A_edge_gate', 'A_edge_blend_r4', 'A_edge_blend_r16', 'B_edge_blend')


def edge_of(mask):
    return mask & ~binary_erosion(mask, structure=np.ones((3,3)), border_value=0)


def contour(picture, mask, partition, target_edge, distance_to_target):
    from eval.eval_utils import estimate_foreground_mask

    predicted = estimate_foreground_mask(Image.fromarray(picture), tail.SIZE)
    if predicted is None:
        predicted = np.zeros_like(mask)
    p_edge = edge_of(predicted)
    dist = distance_transform_edt(~p_edge)[target_edge] if p_edge.any() else np.full(target_edge.sum(), np.inf)
    near_p = distance_to_target[p_edge]
    scores = {}
    for radius in (1, 3, 5, 8):
        precision = float(np.mean(near_p<=radius)) if near_p.size else 0.
        recall = float(np.mean(dist<=radius)) if dist.size else 0.
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.
        scores.update({f'f1_r{radius}':f1})
        if radius == 3:
            scores.update(precision=precision, recall=recall)
    unmatched = p_edge & (distance_to_target>3)
    pred_pixels = max(int(p_edge.sum()),1)
    scores.update(predicted_boundary_px=int(p_edge.sum()), target_boundary_px=int(target_edge.sum()),
                  foreground_bias_pp=float(100*(predicted.mean()-mask.mean())))
    for name, part in zip(SIDES,partition):
        n=max(int(part.sum()),1)
        scores[f'unmatched_pred_{name}']=float((unmatched & part).sum()/pred_pixels)
        scores[f'foreground_fp_{name}']=float((predicted & ~mask & part).sum()/n)
        scores[f'foreground_fn_{name}']=float((~predicted & mask & part).sum()/n)
    return scores


def swaps(images, mask, partition):
    inside, edge, background = (roi.gaussian(area.astype('float32'),2) for area in partition)
    assert np.allclose(inside+edge+background,1,atol=1e-6)
    a, b, blend = (images[name] for name in ('A','B','blend'))
    return dict(A_inside_blend=pixels_mixed(a,blend,inside),
                A_edge_blend=pixels_mixed(a,blend,edge),
                A_background_blend=pixels_mixed(a,blend,background),
                A_inside_edge_blend=pixels_mixed(a,blend,inside+edge),
                A_edge_gate=pixels_mixed(a,images['gate'],edge),
                A_edge_blend_r4=pixels_mixed(a,blend,roi.gaussian(roi.regions(mask,4)[1].astype('float32'),2)),
                A_edge_blend_r16=pixels_mixed(a,blend,roi.gaussian(roi.regions(mask,16)[1].astype('float32'),2)),
                B_edge_blend=pixels_mixed(b,blend,edge))


def source_cases(base, run, data, protocol, saved):
    for row in saved:
        i, case = row['index'], row['case_id']
        p = protocol['pairs'][i]
        assert p['case_id']==case
        mask_path = base/'cases'/case/'mask.png'
        target_path = data/p['gt']
        assert tail.sha(mask_path)==p['mask_sha256'] and tail.sha(target_path)==p['input_sha256']['gt']
        mask = np.asarray(tail.rgb(mask_path))[...,0]>0
        gt = np.asarray(tail.rgb(target_path).resize(tail.SIZE,Image.Resampling.BILINEAR))
        images = {}
        for name in ORIGINAL:
            path = run/'test'/name/(case+'.png')
            assert tail.sha(path)==row['pngs'][name], (case,name)
            images[name] = np.asarray(tail.rgb(path))
        yield row, images, gt, mask, target_path


def mean_keys(rows, key, section=None):
    result = {}
    for name in rows[0][key]:
        group = [row[key][name] if section is None else row[key][name].get(section) for row in rows]
        group = [item for item in group if item is not None]
        if not group:
            continue
        result[name] = {}
        for field in group[0]:
            values = [item[field] for item in group if item[field] is not None]
            result[name][field] = float(np.mean(values)) if values else None
    return result


def local_stats(rows):
    labels=np.array([r['category'] for r in rows])
    regional = {k:mean_keys(rows,'regional',k) for k in (*SIDES,'deep_inside')}
    contours=mean_keys(rows,'contour')
    texture=mean_keys(rows,'texture')
    gaps = {}
    for arm in ('A','B','gate'):
        gaps[arm] = {}
        for section in SIDES:
            rows_a = [r['regional'][arm][section]['lpips_share'] for r in rows]
            rows_b = [r['regional']['blend'][section]['lpips_share'] for r in rows]
            gaps[arm][section]=roi.paired(rows_b,rows_a,labels)
        arm_gt_gap = [r['scores'][arm]['lpips']-r['scores']['blend']['lpips'] for r in rows]
        assert abs(sum(p['delta'] for p in gaps[arm].values())-float(np.mean(arm_gt_gap))) < 2e-6
    return dict(n=len(rows), deep_inside_count=sum(r['core_valid'] for r in rows),regional=regional,
                contour=contours,texture=texture,lpips_vs_blend=gaps)


@torch.no_grad()
def run(args):
    project=args.prior.parents[1]
    data=project.parent/'datasets/BF'
    base=args.prior/tail.distill.BASE
    run=base/epoch.RUN
    old=run/'eval64'
    out=run/RUN
    out.mkdir(exist_ok=True)
    if (out/'review.zip').is_file():
        print('E1_ROI_ARCHIVE',tail.sha(out/'review.zip'),flush=True)
        return
    tail.verified_files(old,SOURCE_SHA,('result.json','per_pair.json','pool3.npz'))
    protocol=tail.verify(argparse.Namespace(out=base))
    assert tail.metric_assets()==protocol['metric_assets']
    saved=tail.read(old/'per_pair.json')
    assert [r['index'] for r in saved]==tail.subset(protocol)
    control_result=tail.read(old/'result.json')
    with np.load(old/'pool3.npz',allow_pickle=False) as f:
        features={k:f[k] for k in (*ORIGINAL,'gt')}
    metric,inception=tail.metric_models('cpu',protocol['metric_assets'])
    edit_features={key:[] for key in EDITS}
    variants=(*ORIGINAL,*EDITS)
    rows=[]
    with tempfile.TemporaryDirectory(prefix='e1_roi64_',dir=os.environ.get('SLURM_TMPDIR')) as temp:
        scratch=Path(temp)/'edit.png'
        for row,images,gt,mask,target_path in source_cases(base,run,data,protocol,saved):
            case=row['case_id']
            partition=roi.regions(mask,8)
            core=roi.regions(mask,20)[0]
            valid=bool(core.sum()>=256)
            weights=torch.from_numpy(partition[None].astype('float32'))
            target_edge=edge_of(mask)
            dist=distance_transform_edt(~target_edge)
            all_images={**images,**swaps(images,mask,partition)}
            scores,region_scores,shape,texture={},{},{},{}
            slices=roi.lpips_features(metric,roi.tensors([gt]))
            gt_color=appearance_inputs(gt)
            for start in range(0,len(variants),4):
                names=variants[start:start+4]
                batch=roi.tensors([all_images[n] for n in names])
                layers=roi.maps(metric,slices,batch)
                lpips=sum(layer.mean(dim=(-2,-1)) for layer in layers).numpy()
                pieces=roi.apportion(layers,weights)
                assert np.allclose(pieces.sum(axis=1),lpips,atol=1e-7,rtol=0)
                changed=[j for j,n in enumerate(names) if n in EDITS]
                if changed:
                    for j,vec in zip(changed,inception(batch[changed])[0].flatten(1).numpy()):
                        edit_features[names[j]].append(vec)
                for j,name in enumerate(names):
                    array=all_images[name]
                    picture=Image.fromarray(array)
                    shape[name]=contour(array,mask,partition,target_edge,dist)
                    texture[name]=tail.appearance(array,gt_color[0],core)
                    if name in ORIGINAL:
                        old_scores=row['scores'][name]
                        score,share,area=roi.ssim_regions(array,gt,partition)
                        assert abs(old_scores['lpips']-float(lpips[j]))<5e-6,(case,name)
                        assert abs(old_scores['ssim']-score)<1e-9,(case,name)
                        regional={}
                        lab=appearance_inputs(array)
                        for k,part in enumerate(SIDES):
                            regional[part]=dict(roi.appearance(lab,gt_color,partition[k]),
                                                lpips_share=float(pieces[j,k]),ssim_share=share[k],area=area[k])
                        if valid:
                            regional['deep_inside']=roi.appearance(lab,gt_color,core)
                            regional['deep_inside']['band_alignment_gt']=band_alignment(lab,gt_color,core)
                        region_scores[name]=regional
                        scores[name]=old_scores
                    else:
                        picture.save(scratch)
                        edited=tail.image_scores(picture,Image.fromarray(gt),scratch,target_path,mask)
                        scores[name]=dict(edited,lpips=float(lpips[j]))
                    assert abs(shape[name]['f1_r3']-scores[name]['contour_f1'])<1e-12,(case,name)
            npixels=int(partition[2].sum())
            background={n:dict(changed_pixels=int(np.any(images[n]!=images['off'],axis=2)[partition[2]].sum()),
                               background_pixels=npixels) for n in ORIGINAL}
            rows.append(dict(index=row['index'],case_id=case,category=row['category'],scores=scores,
                             regional=region_scores,contour=shape,texture=texture,background=background,core_valid=valid))
            if len(rows)%8==0:
                print('E1_ROI_CASES',len(rows),flush=True)
    features.update({name:np.stack(vec) for name,vec in edit_features.items()})
    assert len(rows)==64 and all(f.shape==(64,2048) and np.isfinite(f).all() for f in features.values())
    result=tail.summarize(rows,features,variants,('off','A','blend'),('A','B',*EDITS))
    for name,group in result['groups'].items():
        selected=[r for r in rows if name=='all' or r['category']==name]
        group.update(local_stats(selected))
        for arm in ORIGINAL:
            assert abs(group['arms'][arm]['fid']['fid']-control_result['groups'][name]['arms'][arm]['fid']['fid']) < 1e-8
    result['acceptance']={arm:epoch.gate_check(result,arm) for arm in variants}
    result['meta']=dict(source_sha256=SOURCE_SHA,no_s0_trajectories=True,optimizer_updates=0,
                        edge_radius_px=8,feather_sigma_px=2,deep_inside_radius_px=20,
                        contour_radii_px=[1,3,5,8])
    tail.write(out/'result.json',result)
    tail.write(out/'per_pair.json',rows)
    np.savez_compressed(out/'edit_pool3.npz',**{k:features[k] for k in EDITS})
    files=('result.json','per_pair.json','edit_pool3.npz')
    tail.write(out/'review_manifest.json',{n:tail.sha(out/n) for n in files})
    temporary=out/'review.zip.tmp'
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in (*files,'review_manifest.json'):
            archive.write(out/name,name)
    temporary.replace(out/'review.zip')
    print('E1_ROI_ARCHIVE',tail.sha(out/'review.zip'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prior',type=Path)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','8')))
    cv2.setNumThreads(1)
    run(parser.parse_args())
