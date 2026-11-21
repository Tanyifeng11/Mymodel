"""从已归档的独立 500 对分离带通幅度和位置误差，并评分局部衰减探针。"""
import argparse
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from tools import e2_f68_e1_outwear_independent as independent
from tools import e2_f68_e1_fid_regions as fid_regions
from tools import e2_f68_region_diagnostic as roi
from tools import e2_f68_tail6_e1_epoch10 as epoch
from tools import e2_f68_tail6_real_ab as tail


RUN = 'band_attribution_20261121'
SOURCE_SHA = '8f83d9e9fa15205cd9a1e06398ddcab752c96633001e2c78e8e4c736c89953eb'
ORIGINAL = ('offline25', 'retain_125', 'on')
CONTROLS = ('off', 'on', 'blend', 'offline25', 'retain_125')
LEVELS = (.05, .10, .15, .20, .25, .30)
GROUPS = {
    'dress': ('dress',),
    'dress_outwear': ('dress','outwear'),
    'dress_outwear_top': ('dress','outwear','top'),
}


def details(pred, target):
    x, y = pred.astype('float64'), target.astype('float64')
    a, b = roi.rms(x), roi.rms(y)
    # 齐对原验收的 float32 绝对误差和均值，再用 float64 分解平方误差。
    mae = float(np.abs(pred-target).mean())
    cos = float(np.mean(x*y)/(a*b)) if a*b>1e-12 else None
    amplitude = (a-b)**2
    position = 2*a*b*(1-cos) if cos is not None else 0.
    squared = float(np.square(x-y).mean())
    np.testing.assert_allclose(amplitude+position,squared,atol=1e-8,rtol=1e-8)
    same_amp_error = float(np.abs(x*(b/a)-y).mean()) if a>1e-9 else None
    target_mae = float(np.abs(y).mean())
    if same_amp_error is not None and b>1e-9:
        same_position_error = abs(a/b-1)*target_mae
        amp_l1 = (same_position_error+mae-same_amp_error)/2
        pos_l1 = mae-amp_l1
    else:
        same_position_error,amp_l1,pos_l1 = None,None,None
    return dict(mae=mae,pred_rms=a,target_rms=b,rms_bias=a-b,abs_rms_error=abs(a-b),cosine=cos,
                zero_band_mae=target_mae,amp_mse=amplitude,position_mse=position,mse=squared,
                matched_amplitude_mae=same_amp_error,matched_location_mae=same_position_error,
                amplitude_l1=amp_l1,position_l1=pos_l1)


def attenuate(orig, lab, field, roundtrip, weight, coefficient):
    adjusted = lab.copy()
    adjusted[...,0] -= coefficient*weight*field
    result = roi.apply_lab_edit(orig,roundtrip,adjusted)
    assert np.array_equal(result[weight==0],orig[weight==0])
    return result


def score_edits(args, rows, choices, prior_features):
    lpips,inception = tail.metric_models('cpu',args.protocol['metric_assets'])
    features = {name:prior_features['retain_125'].copy() for name in choices}
    with tempfile.TemporaryDirectory(prefix='band_probe_',dir=os.environ.get('SLURM_TMPDIR')) as scratch:
        png = Path(scratch)/'probe.png'
        for start in range(0,len(rows),4):
            pict = {name:[] for name in choices}
            batch = rows[start:start+4]
            for row in batch:
                i,case = row['index'],row['case_id']
                pair = args.protocol['pairs'][i]
                active = [name for name,p in choices.items()
                          if row['category'] in GROUPS[name] and row['interior_px']>=256]
                row['edited_pixels'] = dict.fromkeys(choices,0)
                for k in ('scores','texture'):
                    for name in choices:
                        row[k][name] = dict(row[k]['retain_125'])
                if not active:
                    continue
                orig = fid_regions.checked_png(args.run/'images/retain_125'/(case+'.png'),row['pngs']['retain_125'])
                mask = tail.case_mask(args,pair)
                core = roi.regions(mask,20)[0]
                weight = roi.gaussian(core.astype('float32'),2)
                path = args.data/pair['gt']
                assert tail.sha(path)==pair['input_sha256']['gt']
                gt = tail.rgb(path).resize(tail.SIZE,Image.Resampling.BILINEAR)
                gt_rgb = np.asarray(gt)
                gt_lab = roi.lab(gt_rgb)
                lab = roi.lab(orig)
                field = roi.band(lab[...,0])
                roundtrip = roi.as_rgb(lab).astype('int16')
                for name in active:
                    coefficient = choices[name]['gain']
                    image = attenuate(orig,lab,field,roundtrip,weight,coefficient)
                    row['edited_pixels'][name] = int(np.count_nonzero(np.any(image!=orig,axis=2)))
                    if not row['edited_pixels'][name]:
                        continue
                    picture = Image.fromarray(image)
                    picture.save(png)
                    row['scores'][name] = tail.image_scores(picture,gt,png,path,mask)
                    row['texture'][name] = tail.appearance(image,gt_lab,core)
                    pict[name].append((i,image,gt_rgb))
            for name, values in pict.items():
                if not values:
                    continue
                ix, x, y = zip(*values)
                image, truth = roi.tensors(x),roi.tensors(y)
                features[name][list(ix)] = inception(image)[0].flatten(1).numpy()
                metrics = lpips(image*2-1,truth*2-1).flatten().tolist()
                for i,v in zip(ix,metrics):
                    rows[i]['scores'][name]['lpips'] = v
            if start%100==0 or start+4>=len(rows):
                print('BAND_PROBE_SCORED',start+len(batch),len(rows),flush=True)
    return features


@torch.inference_mode()
def run(prior):
    args = independent.paths(prior)
    source = args.run/'eval500'
    out = args.run/RUN
    out.mkdir(exist_ok=True)
    if (out/'review.zip').is_file():
        return
    tail.verified_files(source,SOURCE_SHA,('protocol.json','per_pair.json','pool3.npz','result.json'))
    args.protocol = tail.read(source/'protocol.json')
    rows = tail.read(source/'per_pair.json')
    assert len(rows)==500 and [r['index'] for r in rows]==list(range(500))
    audited = tail.read(source/'result.json')
    threshold = min(audited['groups']['all']['texture'][name]['band_mae'] for name in ('off','on'))
    valid = [r for r in rows if r['interior_px']>=256]
    decomposed = []
    curves = {}
    for row in rows:
        if row['interior_px']<256:
            continue
        i,case = row['index'],row['case_id']
        pair = args.protocol['pairs'][i]
        assert (case,row['category'])==(pair['case_id'],pair['category'])
        mask = tail.case_mask(args,pair)
        inner = roi.regions(mask,20)[0]
        assert int(inner.sum())==row['interior_px']
        path = args.data/pair['gt']
        assert tail.sha(path)==pair['input_sha256']['gt']
        gt = roi.lab(np.asarray(tail.rgb(path).resize(tail.SIZE,Image.Resampling.BILINEAR)))
        target = roi.band(gt[...,0])[inner]
        record = dict(index=i,case_id=case,category=row['category'],interior_px=row['interior_px'],arms={})
        for name in ORIGINAL:
            directory = args.run/'test/on' if name=='on' else args.run/'images'/name
            image = fid_regions.checked_png(directory/(case+'.png'),row['pngs'][name])
            lab = roi.lab(image)
            band = roi.band(lab[...,0])
            stats = details(band[inner],target)
            for key,metric in (('mae','band_mae'),('pred_rms','band_rms'),('abs_rms_error','band_rms_error')):
                np.testing.assert_allclose(stats[key],row['texture'][name][metric],atol=2e-7,rtol=1e-7,
                                           err_msg=f'{case} {name} {metric}')
            record['arms'][name] = stats
            if name=='retain_125' and row['category']!='pants':
                weight = roi.gaussian(inner.astype('float32'),2)
                roundtrip = roi.as_rgb(lab).astype('int16')
                curves[i] = {c:tail.appearance(
                    attenuate(image,lab,band,roundtrip,weight,c),gt,inner)['band_mae'] for c in LEVELS}
        decomposed.append(record)
        if len(decomposed)%80==0:
            print('BAND_DECOMPOSED',len(decomposed),len(valid),flush=True)

    scans = {name:[dict(gain=c,band_mae=float(np.mean([
                curves[r['index']][c] if r['category'] in cats else r['texture']['retain_125']['band_mae']
                for r in valid]))) for c in LEVELS] for name,cats in GROUPS.items()}
    selected = {name:next((x for x in scan if x['band_mae']<threshold),
                          min(scan,key=lambda x:x['band_mae'])) for name,scan in scans.items()}
    print('BAND_PROBE_SELECTED',selected,flush=True)
    assert tail.metric_assets()==args.protocol['metric_assets']
    with np.load(source/'pool3.npz',allow_pickle=False) as old:
        history = {name:old[name] for name in (*CONTROLS,'gt')}
    new = score_edits(args,rows,selected,history)
    result = tail.summarize(rows,{**history,**new},(*CONTROLS,*selected),
        ('offline25','retain_125'),tuple(selected),fid_reference='offline25')
    for name,p in selected.items():
        gate = epoch.gate_check(dict(groups=dict(all=dict(arms=result['groups']['all']['arms'],
            texture={a:{'band_mae':float(np.mean([r['texture'][a]['band_mae'] for r in valid]))}
                     for a in ('off','on',name)}))),name)
        true_mae = float(np.mean([r['texture'][name]['band_mae'] for r in valid]))
        assert abs(true_mae-p['band_mae'])<1e-7
        gate['band_mae'] = true_mae
        gate['edited_pairs'] = sum(r['edited_pixels'][name]>0 for r in rows)
        arms = result['groups']['all']['arms']
        fields = dict(lpips='mean',fid='fid',ssim='mean',contour_f1='mean')
        for ref in ('offline25','retain_125'):
            gate['delta_'+ref] = {k:arms[name][k][f]-arms[ref][k][f] for k,f in fields.items()}
        p['acceptance'] = gate
    result.update(scans=scans,selected=selected,threshold=threshold)
    tail.write(out/'result.json',result)
    tail.write(out/'decomposition.json',decomposed)
    tail.write(out/'per_pair.json',rows)
    np.savez_compressed(out/'probe_pool3.npz',**new)
    tail.write(out/'audit.json',dict(source_sha256=SOURCE_SHA,samples=500,valid_core=len(valid),
        code_sha256=tail.sha(Path(__file__)),levels=LEVELS,groups=GROUPS,
        radius=20,feather_sigma=2,diffusion_rollouts=0,optimizer_updates=0))
    fid_regions.pack(out,('result.json','decomposition.json','per_pair.json','probe_pool3.npz','audit.json'))
    file = out/'review.zip'
    print('BAND_DIAGNOSED',file.stat().st_size,tail.sha(file),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('prior',type=Path)
    a = p.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','8')))
    run(a.prior)


if __name__=='__main__':
    main()
