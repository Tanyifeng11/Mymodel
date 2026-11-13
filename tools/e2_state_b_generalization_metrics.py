"""新身份的固定评价；真实换布只用参考响应代理，不把旧GT当反事实真值。"""
import cv2
import numpy as np
from PIL import Image
import torch

from tools import e2_state_b_pilot as pilot
from tools.e2_fft_experiment import read, write, sha
from tools.bf_t3_three_arm_report import gray, paired

ARMS=('m0','raw','ab_l')
CONDITIONS=('v16','h16','v32','h32','real_a','real_b')
ANGLES=np.arange(32)*np.pi/32
DIRECTIONS=np.stack((np.cos(ANGLES),np.sin(ANGLES)))
QUANTILES=(np.arange(1024)+.5)/1024


def mean(values):
    valid=[v for v in values if v is not None]
    return float(np.mean(valid)) if valid else None


def pixels(path):
    return np.asarray(Image.open(path).convert('RGB'))


def palette(ab):
    ab=np.asarray(ab).reshape(-1,2)
    # 均匀固定抽样控制计算量；三个臂与参考使用同一规则。
    if len(ab)>4096:ab=ab[np.linspace(0,len(ab)-1,4096).round().astype(int)]
    distribution=np.quantile(ab@DIRECTIONS,QUANTILES,axis=0)
    spread=float(np.linalg.norm(np.quantile(ab,.9,axis=0)-np.quantile(ab,.1,axis=0)))
    return distribution,spread


def color(picture,reference,anchor,raw,mask):
    rgb=np.asarray(picture,np.float32)/255
    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)
    ref=cv2.cvtColor(np.asarray(reference,np.float32)/255,cv2.COLOR_RGB2LAB)
    base=cv2.cvtColor(np.asarray(anchor,np.float32)/255,cv2.COLOR_RGB2LAB)
    interior=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,5)>=16
    assert interior.sum()>0
    mean=lab[interior].mean(0);goal=base[interior].mean(0)
    dist,spread=palette(lab[interior,1:]);refdist,refspread=palette(ref[...,1:])
    den=cv2.GaussianBlur(mask.astype(np.float32),(0,0),24).clip(.001)
    smooth=lambda a:cv2.GaussianBlur(a*mask,(0,0),24)/den
    ldiff=smooth(lab[...,0])-smooth(base[...,0])
    abdiff=lab[...,1:]-base[...,1:]
    local=np.stack([smooth(abdiff[...,i]) for i in range(2)],-1)
    return dict(anchor_chroma=float(np.linalg.norm(mean[1:]-goal[1:])),
        anchor_local_chroma=float(np.linalg.norm(local[interior],axis=1).mean()),
        anchor_lightness=float(abs(mean[0]-goal[0])),anchor_low_l=float(np.abs(ldiff[interior]).mean()),
        reference_chroma=float(np.linalg.norm(mean[1:]-ref.reshape(-1,3).mean(0)[1:])),
        palette_swd=float(np.abs(dist-refdist).mean()),ab_spread=spread,reference_ab_spread=refspread,
        ab_spread_ratio=spread/max(refspread,1e-6),ab_spread_error=abs(spread-refspread),
        multicolour_collapse=bool(refspread>=12 and spread<.25*refspread),
        outside_from_raw=float(np.abs(rgb[~mask]-np.asarray(raw,np.float32)[~mask]/255).mean()))


def evaluate(args):
    from tools import t3_response_run as response
    from tools.e2_state_b_generalization import source_hashes
    from eval.eval_utils import estimate_foreground_mask
    from tools.e2_a_evaluation import boundary_f1
    spec=read(args.out/'protocol.json');assert spec['code']==source_hashes()
    response.ROOT=args.out/f'metrics{args.shard}'
    model=response.load_dists(args.device)
    sanity=response.sanity(model)
    refcoords=[(int(x),int(y)) for y in np.linspace(0,448,4).round() for x in np.linspace(0,320,4).round()]
    def features(image,coords,is_gray):
        parts=[image[y:y+64,x:x+64] for x,y in coords]
        if is_gray:parts=[gray(part) for part in parts]
        value=torch.from_numpy(np.stack(parts).transpose(0,3,1,2).copy()).float().to(args.device)/255
        return model.forward_once(value)
    records=[]
    with torch.inference_mode():
        for p in spec['pairs'][args.shard::2]:
            folder=args.out/'cases'/p['case_id'];generation=read(folder/'generation.json')
            assert generation['paired_noise'] and len(generation['rows'])==12
            assert sha(folder/'mask.png')==p['mask_sha256']
            mask=np.asarray(Image.open(folder/'mask.png'))>0
            coords=p['patch_coordinates'];assert len(coords)>=4
            refs={c:pixels(folder/(c+'_ref.png')) for c in CONDITIONS}
            rf={(c,g):features(refs[c],refcoords,g) for c in ('real_a','real_b') for g in (False,True)}
            # 输入参考间距离用于检查任务可辨识度，只报告，不看输出后筛样本。
            reference_separation={('gray' if g else 'rgb'):response.patch_distance(model,rf['real_a',g],rf['real_b',g])
                                  for g in (False,True)}
            rows={};switches={}
            for seed in range(2):
                anchor_synthetic=pixels(folder/f'v16_s{seed}_m0.png')
                anchor_real=pixels(folder/f'real_a_s{seed}_m0.png')
                for c in CONDITIONS:
                    key=f'{c}_s{seed}'
                    meta=next(r for r in generation['rows'] if r['condition']==c and r['seed']==p['seed']+100000*seed)
                    assert len({r['digest'] for r in generation['rows'] if r['seed']==meta['seed']})==1
                    raw=pixels(folder/(key+'_raw.png'));rows[key]={}
                    real=c.startswith('real');anchor=anchor_real if real else anchor_synthetic
                    for arm in ARMS:
                        path=folder/(key+'_'+arm+'.png');assert sha(path)==meta['output_sha256'][arm]
                        picture=pixels(path)
                        values=color(picture,refs[c],anchor,raw,mask)
                        foreground=estimate_foreground_mask(Image.fromarray(picture),(384,512))
                        values.update(iou=float((foreground&mask).sum()/(foreground|mask).sum()),
                            boundary=float(boundary_f1(foreground,mask,2)))
                        if real:
                            foil='real_b' if c=='real_a' else 'real_a'
                            for g in (False,True):
                                prefix='gray' if g else 'rgb'
                                f=features(picture,coords,g)
                                target=response.patch_distance(model,f,rf[c,g])
                                old=response.patch_distance(model,f,rf[foil,g])
                                values.update({prefix+'_target':target,prefix+'_foil':old,prefix+'_margin':old-target})
                        else:
                            goal=Image.open(folder/(key+'_target.png')).convert('RGB')
                            positive=Image.open(folder/(key+'_vae.png')).convert('RGB')
                            pic=Image.fromarray(picture)
                            values.update(texture=pilot.texture_metrics(pic,coords,c),
                                **pilot.old_pattern(pic,Image.fromarray(anchor),positive,coords,c))
                            goal_lab=cv2.cvtColor(np.asarray(goal,np.float32)/255,cv2.COLOR_RGB2LAB)
                            picture_lab=cv2.cvtColor(picture.astype(np.float32)/255,cv2.COLOR_RGB2LAB)
                            delta=picture_lab[mask].mean(0)-goal_lab[mask].mean(0)
                            values.update(goal_chroma=float(np.linalg.norm(delta[1:])),goal_lightness=float(abs(delta[0])))
                        rows[key][arm]=values
                for kind,a,b in [('direction','v16','h16'),('direction','v32','h32'),
                                 ('period','v16','v32'),('period','h16','h32')]:
                    switches[f'{kind}_{a}_s{seed}']={arm:all(rows[f'{c}_s{seed}'][arm]['texture']['hit'] for c in (a,b)) for arm in ARMS}
                switches[f'real_s{seed}']={arm:all(rows[f'{c}_s{seed}'][arm]['gray_margin']>0 for c in ('real_a','real_b')) for arm in ARMS}
            aggregate={}
            for group in ('synthetic','real'):
                selected=[v for k,v in rows.items() if k.startswith('real')==(group=='real')]
                aggregate[group]={}
                for arm in ARMS:
                    keys=[k for k,v in selected[0][arm].items() if v is None or isinstance(v,(int,float,bool))]
                    values={k:mean([r[arm][k] for r in selected]) for k in keys}
                    if group=='synthetic':
                        for k in ('direction','period'):
                            values[k+'_strict']=float(np.mean([v[arm] for key,v in switches.items() if key.startswith(k)]))
                    else:values['real_strict']=float(np.mean([v[arm] for key,v in switches.items() if key.startswith('real')]))
                    aggregate[group][arm]=values
            result=dict(case_id=p['case_id'],stratum=p['stratum'],rows=rows,switches=switches,
                aggregate=aggregate,reference_separation=reference_separation)
            write(folder/'result.json',result);records.append(p['case_id'])
            print('NEW_REAL_EVALUATED',p['case_id'],flush=True)
    write(args.out/f'metrics{args.shard}/complete.json',dict(passed=True,cases=records,sanity=sanity,
        protocol_sha256=sha(args.out/'protocol.json')))


def aggregate(args,spec):
    complete=[read(args.out/f'metrics{s}/complete.json') for s in range(2)]
    assert all(r['passed'] and r['protocol_sha256']==sha(args.out/'protocol.json') for r in complete)
    identities=[read(args.out/'cases'/p['case_id']/'result.json') for p in spec['pairs']]
    assert sorted(r['case_id'] for r in identities)==sorted(x for r in complete for x in r['cases'])
    groups={}
    for stratum in ('all','low','multi'):
        cases=[p for p in identities if stratum=='all' or p['stratum']==stratum]
        for group in ('synthetic','real'):
            fields=cases[0]['aggregate'][group]['raw']
            arms={a:{k:mean([p['aggregate'][group][a][k] for p in cases]) for k in fields} for a in ARMS}
            valid_counts={a:{k:sum(p['aggregate'][group][a][k] is not None for p in cases) for k in fields} for a in ARMS}
            differences={ref:{k:paired([p['aggregate'][group]['ab_l'][k]-p['aggregate'][group][ref][k] for p in cases
                if p['aggregate'][group]['ab_l'][k] is not None and p['aggregate'][group][ref][k] is not None])
                for k in fields} for ref in ('m0','raw')}
            counts={kind:{a:sum(v[a] for p in cases for k,v in p['switches'].items() if k.startswith(kind)) for a in ARMS}
                    for kind in (('direction','period') if group=='synthetic' else ('real',))}
            a,r=arms['ab_l'],arms['raw']
            gates=dict(improve_anchor_colour=all(a[k]<r[k] for k in ('anchor_chroma','anchor_local_chroma')),
                       improve_brightness=all(a[k]<r[k] for k in ('anchor_lightness','anchor_low_l')),
                       outside_unchanged=a['outside_from_raw']==0)
            if group=='synthetic':gates.update(retain_switches=all(a[k]>=r[k] for k in ('direction_strict','period_strict')),
                retain_old_removal=a['old_projection'] is not None and r['old_projection'] is not None and a['old_projection']<=r['old_projection'])
            else:gates.update(retain_gray_response=a['real_strict']>=r['real_strict'] and a['gray_margin']>=r['gray_margin'],
                improve_reference_palette=a['palette_swd']<r['palette_swd'] and a['ab_spread_error']<=r['ab_spread_error'],
                no_multicolour_collapse=a['multicolour_collapse']==0)
            groups[stratum+'_'+group]=dict(n=len(cases),arms=arms,valid_counts=valid_counts,paired_difference=differences,
                strict_counts=counts,strict_denominator=len(cases)*(4 if group=='synthetic' else 2),
                gates=gates,all_gates=all(gates.values()))
    return dict(groups=groups)
