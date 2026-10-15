"""GPU 旧指标回放；合成纹样指标另列，不能用 LPIPS 替代。"""
import csv
import cv2
import numpy as np
import torch
from PIL import Image
from tools.e52_protocol import *
from tools.e44_metrics import contour_f1,external_contour

def replay_metrics(rows):
    from tools.e33tmoc_appearance_eval import frozen_lpips
    from eval.eval_utils import prepare_evaluation_masks,estimate_foreground_mask
    from tools.e44_runtime import sketch_masks
    metric,backend=frozen_lpips();metric=metric.cuda()
    tensor=lambda im:torch.from_numpy(np.asarray(im,np.float32).transpose(2,0,1)/127.5-1)[None].cuda()
    old={(x['id'],x['variant']):x for x in read(SOURCE/'dev32/metrics.json')}
    records=[];errors=[]
    with torch.inference_mode():
        for row in rows:
            gt=Image.open(row['gt']).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR)
            garment=prepare_evaluation_masks(SIZE,sketch_path=row['sketch'],mask_policy='sketch_only')['garment']
            valid=sketch_masks(Image.open(row['sketch']).convert('RGB').resize(SIZE),'cpu')[2]['valid']
            for arm,historical in [('E5','B'),('B1','D')]:
                image=Image.open(image_path('p0',arm,row['id'])).convert('RGB')
                rec=dict(id=row['id'],arm=arm,lpips_gt=float(metric(tensor(image),tensor(gt)).mean()),contour_f1=None,iou=None)
                if valid:
                    fg=estimate_foreground_mask(image,SIZE)
                    if fg is None:fg=np.zeros((SIZE[1],SIZE[0]),bool)
                    rec.update(contour_f1=contour_f1(fg,garment),iou=float((fg&garment).sum()/max((fg|garment).sum(),1)))
                for key in ['lpips_gt','contour_f1','iou']:
                    expected=old[(row['id'],historical)][key]
                    assert (expected is None)==(rec[key] is None)
                    if expected is not None:
                        error=abs(rec[key]-expected);errors.append(error)
                        assert error<=2e-6,('GPU_METRIC_REPLAY_FAIL',row['id'],arm,key,error)
                records.append(rec)
    write(OUT/'audit/replay_metrics.json',records)
    return dict(backend=backend,n_images=len(records),max_abs_error=max(errors),tolerance=2e-6)

def paired(values):
    values=np.asarray([v for v in values if v is not None],float)
    if not len(values):return dict(n=0,mean=None,ci95=None,p_two_sided=None)
    rng=np.random.default_rng(32042)
    draws=np.concatenate([rng.choice(values,(500,len(values)),replace=True).mean(1) for _ in range(20)])
    p=min(1.,2*min((np.count_nonzero(draws<=0)+1)/10001,(np.count_nonzero(draws>=0)+1)/10001))
    return dict(n=len(values),mean=float(values.mean()),ci95=np.quantile(draws,[.025,.975]).tolist(),
        p_two_sided=float(p),method='identity paired percentile bootstrap; bootstrap sign-tail p')

def get_delta(index,a,b,key):
    if index.get(a,{}).get(key) is None or index.get(b,{}).get(key) is None:return None
    return index[a][key]-index[b][key]

def summarize(records,stage,rows):
    arms=list(dict.fromkeys(r['arm'] for r in records));keys=['f1_3px','iou','leak','lpips','texture_retrieval_correct','texture_margin','orientation_error']
    byid={r['id']:{x['arm']:x for x in records if x['id']==r['id']} for r in rows}
    means={a:{k:float(np.mean([r[k] for r in records if r['arm']==a and r.get(k) is not None]))
        if any(r['arm']==a and r.get(k) is not None for r in records) else None for k in keys} for a in arms}
    counts={a:{k:sum(r['arm']==a and r.get(k) is not None for r in records) for k in keys} for a in arms}
    summary=dict(stage=stage,means=means,valid_counts=counts,n_identities=len(rows),primary={},contrasts={},direction=None)
    for arm in arms:
        if arm=='E5':continue
        summary['contrasts'][arm]={k:paired([get_delta(index,arm,'E5',k) for index in byid.values()]) for k in keys}
    if stage in ['pilot','lock']:
        contrasts=[];directions=[]
        for row in rows:
            index=byid[row['id']]
            geom=[get_delta(index,'G1T%d'%t,'G0T%d'%t,'f1_3px') for t in [0,1]]
            tex=[get_delta(index,'G%dT1'%g,'G%dT0'%g,'f1_3px') for g in [0,1]]
            gammaG=float(np.mean(geom)) if all(v is not None for v in geom) else None
            gammaT=float(np.mean(tex)) if all(v is not None for v in tex) else None
            trace={a:read(image_path(stage,a,row['id']).with_name(row['id']+'_trace.json')) for a in ['G0T0','G1T0']}
            epsilon={}
            for a in trace:
                epsilon[a]={s['index']:s['teacher_state_effect']['boundary']['rms'] for s in trace[a]['steps']
                    if s.get('teacher_state_effect') is not None}
            early=[i for i in [0,4,9,14] if all(i in epsilon[a] for a in epsilon)]
            late=[i for i in [39,49] if all(i in epsilon[a] for a in epsilon)]
            gammaD=float(np.mean([epsilon['G0T0'][i]-epsilon['G1T0'][i] for i in early])
                -np.mean([epsilon['G0T0'][i]-epsilon['G1T0'][i] for i in late])) if early and late else None
            gammaC=get_delta(index,'ORACLE','G0T0','f1_3px') if 'ORACLE' in arms else None
            item=dict(id=row['id'],gamma_G=gammaG,gamma_T=gammaT,gamma_D=gammaD,gamma_C=gammaC,
                interaction_G_T=(geom[1]-geom[0]) if gammaG is not None else None,
                sham_delta=get_delta(index,'SHAM','G0T0','f1_3px'),kind=row['kind'])
            contrasts.append(item)
            from tools.e52_data import signed_distance
            geomdata=np.load(row['geometry']);expected=geomdata['expected_shift']
            mask=geomdata['target_labels']>0;contour=external_contour(mask)
            relevant=(cv2.dilate(contour.astype('uint8'),np.ones((9,9),np.uint8))>0)&(np.abs(expected)>=1)
            shifts=[]
            for t in [0,1]:
                foreground=[]
                for g in [0,1]:
                    p=OUT/'metrics'/stage/'foreground'/('G%dT%d'%(g,t))/(row['id']+'.png')
                    foreground.append(np.asarray(Image.open(p))>127)
                shifts.append(signed_distance(foreground[1])-signed_distance(foreground[0]))
            generated=np.mean(shifts,axis=0)
            denominator=float(np.linalg.norm(expected[relevant])*np.linalg.norm(generated[relevant]))
            cosine=float(np.dot(expected[relevant],generated[relevant])/denominator) if denominator>1e-8 else None
            directions.append(dict(id=row['id'],kind=row['kind'],sign=row['sign'],cosine=cosine,
                agreement=float((expected[relevant]*generated[relevant]>0).mean()) if relevant.any() else None,
                generated_signed_amplitude=float(np.mean(generated[relevant]*np.sign(expected[relevant]))) if relevant.any() else None))
        summary['primary']={k:paired([r['gamma_'+k] for r in contrasts]) for k in ['G','C','D']}
        summary['gamma_T']=paired([r['gamma_T'] for r in contrasts]);summary['G_T']=paired([r['interaction_G_T'] for r in contrasts])
        summary['sham']=paired([r['sham_delta'] for r in contrasts]);summary['direction']=directions
        summary['geometry_minus_sham']=paired([r['gamma_G']-r['sham_delta'] for r in contrasts
            if r['gamma_G'] is not None and r['sham_delta'] is not None])
        summary['direction_amplitude']=paired([r['generated_signed_amplitude'] for r in directions])
        # 各部位类别内打乱参考变化正负标签，保留几何类别分布。
        amplitudes=np.array([r['generated_signed_amplitude'] or 0. for r in directions]);signs=np.array([r['sign'] for r in directions])
        rng=np.random.default_rng(32042);null=[]
        for _ in range(10000):
            shuffled=signs.copy()
            for kind in set(r['kind'] for r in directions):
                positions=[i for i,r in enumerate(directions) if r['kind']==kind];shuffled[positions]=rng.permutation(signs[positions])
            null.append(float(np.mean(amplitudes*shuffled*signs)))
        observed=float(amplitudes.mean());summary['direction_random_label']=dict(observed=observed,
            null_ci95=np.quantile(null,[.025,.975]).tolist(),p_upper=(sum(v>=observed for v in null)+1)/10001)
        rawp={k:v['p_two_sided'] for k,v in summary['primary'].items() if v['p_two_sided'] is not None}
        adjusted={};previous=0.
        for i,(key,value) in enumerate(sorted(rawp.items(),key=lambda x:x[1])):
            previous=max(previous,min(1.,(3-i)*value));adjusted[key]=previous
        summary['holm_p']=adjusted
        summary['pilot_MDE_at_lock64']={k:float((1.96+.84)*np.std([r['gamma_'+k] for r in contrasts if r['gamma_'+k] is not None],ddof=1)/8)
            if sum(r['gamma_'+k] is not None for r in contrasts)>1 else None for k in ['G','C','D']}
        write(OUT/'metrics'/stage/'identity_primary_contrasts.json',contrasts)
    else:
        gates={};gain=means['E5']['lpips']-means['B1']['lpips'] if 'B1' in means else None
        for arm in arms:
            if arm=='E5':continue
            d=summary['contrasts'][arm]
            tails=[r['id'] for r in records if r['arm']==arm and r['tail_event']]
            retention=(means['E5']['lpips']-means[arm]['lpips'])/gain if gain and abs(gain)>1e-8 else None
            checks=dict(f1=d['f1_3px']['ci95'][0]>=-.02,iou=d['iou']['ci95'][0]>=-.01,
                leak=d['leak']['ci95'][1]<=.005,tail=not tails,lpips=d['lpips']['ci95'][1]<0,
                retention=retention is not None and retention>=.50 and gain>0)
            gates[arm]=dict(checks=checks,passed=all(checks.values()),tails=tails,retention=retention)
        summary['model_gates']=gates
    write(OUT/'metrics'/stage/'summary.json',summary)
    return summary

@torch.inference_mode()
def evaluate(stage):
    from tools.e33tmoc_appearance_eval import frozen_lpips
    from eval.eval_utils import prepare_evaluation_masks,estimate_foreground_mask,json_safe
    from eval.metrics import compute_texture_leakage
    from tools.e44_runtime import sketch_masks
    from tools.e44_b1_mask import overlay
    from tools.e52_texture import score
    from tools.e45_eval import foreground_at
    rows=read(OUT/'splits'/(stage+'.json'))
    synthetic=stage in ['pilot','lock'];provenance=read(OUT/'audit/token_provenance.json')
    arms=A_ARMS+C_ARMS+(B_ARMS if provenance['oracle_valid'] else []) if synthetic else CONFIG['natural_arms'] if stage=='natural' else ['E5','B1']
    metric=backend=None
    if not synthetic:metric,backend=frozen_lpips();metric=metric.cuda()
    tensor=lambda im:torch.from_numpy(np.asarray(im,np.float32).transpose(2,0,1)/127.5-1)[None].cuda()
    records=[];sensitivity=[];time_rows=[]
    for row in rows:
        sketch=Image.open(row['sketch']).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR)
        _,_,info=sketch_masks(sketch,'cpu')
        bundle=prepare_evaluation_masks(SIZE,sketch_path=row['sketch'],mask_policy='sketch_only');garment=bundle['garment']
        gt=Image.open(row['gt']).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR) if not synthetic else None
        initial=set();baseF=None
        for arm in arms:
            path=image_path(stage,arm,row['id']);meta=read(path.with_suffix('.json'));initial.add(meta['initial_sha256'])
            image=Image.open(path).convert('RGB');fg=estimate_foreground_mask(image,SIZE)
            if fg is None:fg=np.zeros((SIZE[1],SIZE[0]),bool)
            folder=OUT/'metrics'/stage/'foreground'/arm;folder.mkdir(parents=True,exist_ok=True)
            Image.fromarray(fg.astype('uint8')*255).save(folder/(row['id']+'.png'))
            if garment is not None:overlay(image,fg,garment).save(folder/(row['id']+'_overlay.png'))
            rec=dict(stage=stage,id=row['id'],arm=arm,seed=42,valid_sketch_mask=info['valid'],f1_3px=None,iou=None,leak=None,
                lpips=float(metric(tensor(image),tensor(gt)).mean()) if metric else None,
                texture_retrieval_correct=None,texture_margin=None,orientation_error=None,period_error=None,phase_error=None,
                base_reference_sha256=meta['base_reference_sha256'],writer_reference_sha256=meta['writer_reference_sha256'],
                b1_sha256=B1_SHA,generated_png_sha256=sha(path),gt_used_in_inference=False,metric_na_reason=None)
            if info['valid']:
                rec.update(f1_3px=contour_f1(fg,garment),iou=float((fg&garment).sum()/max((fg|garment).sum(),1)),
                    leak=json_safe(compute_texture_leakage(str(path),target_path=row['gt'],mask_bundle=bundle)).get('leak_colored_frac'))
                for distance,brightness in [(12,230),(18,235),(24,240)]:
                    alternative=foreground_at(image,distance,brightness)
                    sensitivity.append(dict(id=row['id'],arm=arm,distance=distance,brightness=brightness,
                        **{'f1_%dpx'%tol:contour_f1(alternative,garment,tol) for tol in [3,5,9]}))
            else:rec['metric_na_reason']='invalid_sketch_mask'
            if synthetic:
                correct=1 if arm in ['G0T1','G1T1'] else 0
                rec.update(score(image,row,correct));rec.update(G=1 if arm in ['G1T0','G1T1','G1_EARLY','SHRINK_EARLY'] else 0,T=correct)
            if arm=='E5':baseF=rec['f1_3px']
            rec['delta_f1_vs_e5']=rec['f1_3px']-baseF if rec['f1_3px'] is not None and baseF is not None else None
            rec['tail_event']=rec['delta_f1_vs_e5']<-.15 if rec['delta_f1_vs_e5'] is not None else None
            trace=read(path.with_name(row['id']+'_trace.json'))
            for step in trace['steps']:
                time_rows.append(dict(identity_id=row['id'],group_id=arm,seed=42,reference_id=meta['writer_reference_sha256'],
                    checkpoint_sha256=B1_SHA,**step))
            records.append(rec)
        assert len(initial)==1 and len({r['base_reference_sha256'] for r in records if r['id']==row['id']})==1
        write(OUT/'metrics'/stage/'per_image.json',records)
        print('E52 METRICS',stage,row['id'],len(records),flush=True)
    write(OUT/'metrics'/stage/'threshold_sensitivity.json',sensitivity)
    write(OUT/'metrics'/stage/'time_steps.json',time_rows)
    with (OUT/'metrics'/stage/'per_image.csv').open('w',newline='',encoding='utf-8') as stream:
        columns=list(dict.fromkeys(k for r in records for k in r));writer=csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader();writer.writerows(records)
    summary=summarize(records,stage,rows)
    write(OUT/'metrics'/stage/'backend.json',dict(lpips=backend,synthetic_lpips='NA; no natural GT' if synthetic else None,
        foreground='historical estimate_foreground_mask distance18/brightness235',outline_tolerance=3,
        texture='synthetic calibrated retrieval; geometry sensitive; natural identity human evaluation not_run'))
    decision(**{dict(pilot='P2',lock='P4',natural='P5').get(stage,'P0'):'complete_'+stage})
    return summary
