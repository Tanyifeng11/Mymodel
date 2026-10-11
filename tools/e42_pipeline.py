"""E42一次性流程。锁定控制校准与R/GT资格后才解码已有E5图像。"""
import argparse
import csv
import itertools
import os
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from tools.e42_repeat import (OUT, CONFIG, read, write, sha, digest, sketch_geometry,
    windows, highpass, measure_window, summarize, decision, gt_stable,
    compatibility, compatible_rois)

HISTORY=Path('eval_outputs/phase1_full_500/e5_tcpm_lite')
E5=Path('output/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt')
EXPECTED_E5='aea56b2d7b02492f0e58fc2abc45b6a5a996b9fd962f31e30aa26ff5b0ed28f0'
MODES=['joint','autocorr_only','self_similarity_only']


def now():return datetime.now(timezone.utc).isoformat()


def csv_write(path,rows,fields=None):
    rows=list(rows);fields=fields or list(rows[0])
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)


def event(stage,**values):
    with (OUT/'events.jsonl').open('a',encoding='utf-8') as f:
        f.write(__import__('json').dumps(dict(time=now(),stage=stage,**values),ensure_ascii=False)+'\n')
    print(stage,values,flush=True)


def build_manifest():
    if (OUT/'audit.json').exists():raise RuntimeError('本轮目录已存在；禁止覆盖一次性实验')
    OUT.mkdir(parents=True,exist_ok=True)
    records=read(HISTORY/'benchmark_samples.json');manifest=read(HISTORY/'experiment_manifest.json')
    dataset=read('data/train_bf_texture.json');split=read('eval/benchmarks/phase1_bf_val_split.json')
    hashes={};rows=[];geometries={}
    for i,r in enumerate(records):
        row=dict(sample_id=Path(r['target_path']).stem,benchmark_id=r['sample_id'],
            dataset_index=r['dataset_index'],uid=r['uid'],reference_path=r['texture_path'],
            sketch_path=r['sketch_path'],caption_ref='data/train_bf_texture.json#'+str(r['dataset_index'])+'/caption',
            caption=r['prompt'],gt_path=r['target_path'],e5_rgb_path=r['gen_path'],split='historical_phase1_bf_val_500',seed=manifest['seed'])
        original=dataset[r['dataset_index']];root=Path(manifest['data_root'])
        matched=(original['caption']==r['prompt'] and split[i]['idx']==r['dataset_index'] and
            all((root/original[k]).resolve()==Path(row[v]).resolve() for k,v in
                [('cloth','gt_path'),('texture','reference_path'),('sketch','sketch_path')]))
        missing=[k for k in ['reference_path','sketch_path','gt_path','e5_rgb_path'] if not Path(row[k]).is_file()]
        row['asset_status']='excluded_pair_mismatch' if not matched else ('excluded_missing_asset' if missing else 'ok')
        row['missing_assets']=';'.join(missing)
        for k in ['reference_path','sketch_path','gt_path','e5_rgb_path']:
            p=row[k]
            if Path(p).is_file():hashes.setdefault(p,sha(p))
            row[k.replace('_path','_sha256')]=hashes.get(p,'')
        row['group_id']=row['reference_sha256']
        if not missing:
            with Image.open(row['e5_rgb_path']) as im:size=im.size
            row['image_size']=str(size[0])+'x'+str(size[1])
            mask,rois,g=sketch_geometry(row['sketch_path'],size)
            g.update(rois=rois,size=list(size));geometries[row['sample_id']]=g
            row['roi_status']=g['reason'];row['window_hash']=g['window_hash']
        else:row.update(image_size='',roi_status='excluded_missing_asset',window_hash='')
        rows.append(row)
    assert len({r['sample_id'] for r in rows})==len(rows),'原BF ID重复，不能以序号覆盖'
    weight_hash=sha(E5)
    # 历史manifest记模型和代码版本；采样参数由对应历史代码路径查证，保留证据强度。
    commit=manifest['git_commit']
    source=subprocess.check_output(['git','show',commit+':inference_IMAGGarment-1.py'],text=True)
    benchmark_source=subprocess.check_output(['git','show',commit+':tools/run_fixed_benchmark.py'],text=True)
    protocol_ok=(weight_hash==EXPECTED_E5 and Path(manifest['gam_ckpt']).resolve()==E5.resolve() and
        'default=7.0' in source and 'default=50' in source and 'DDIMScheduler' in source and
        "'--guidance_scale'" not in benchmark_source and "'--num_inference_steps'" not in benchmark_source and
        manifest['generated_images']==len(records))
    audit=dict(started=now(),n_total=len(rows),asset_counts=dict(Counter(r['asset_status'] for r in rows)),
        roi_counts=dict(Counter(r['roi_status'] for r in rows)),historical_manifest=manifest,
        historical_code_commit=commit,protocol_verified=protocol_ok,E5_sha256=weight_hash,
        historical_checkpoint_digest_limit='No digest was saved with the July PNGs; current named E5 digest matches E39/E41. Historical output-to-weight provenance uses the original completed manifest.',
        sampling_evidence='Historical inference code defaults CFG7/DDIM50, benchmark has no override; manifest seed42 and all image headers checked. No per-image runtime sampling log is asserted.',
        inference_source_sha256=__import__('hashlib').sha256(source.encode()).hexdigest(),
        source_hashes={str(p):sha(p) for p in [HISTORY/'benchmark_samples.json',HISTORY/'experiment_manifest.json',Path('data/train_bf_texture.json'),Path('eval/benchmarks/phase1_bf_val_split.json')]},
        config=CONFIG,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        slurm={k:v for k,v in os.environ.items() if k.startswith('SLURM_')},
        training_steps=0,new_model_generations=0,gpu_count=0,
        truth_limit='Procedural controls only; no independently labelled real RGB repeat or motif identity truth.')
    csv_write(OUT/'manifest.csv',rows);write(OUT/'manifest_records.json',rows)
    write(OUT/'input_hashes.json',hashes);write(OUT/'geometry.json',geometries);write(OUT/'audit.json',audit)
    event('manifest_locked',manifest_sha256=sha(OUT/'manifest.csv'),n=len(rows),protocol_verified=protocol_ok)
    return rows,geometries,audit


def procedural(kind,period,angle,phase,contrast,color,size):
    # 直接复用E40条纹/格纹和E41椭圆renderer，非生成模型图像。
    if kind=='ellipse':
        from tools.e41_attributes import render
        image=render(dict(period=period,orientation=angle,scale=period*.45,density=1.),phase=phase,color=0,size=size)
    else:
        from tools.e40_protocol import controlled
        image=controlled(kind,size/period,angle,phase).resize((size,size),Image.Resampling.BILINEAR)
    a=np.asarray(image,float);t=(a[:,:,0]-48)/160
    if color:
        lab=np.stack([np.full_like(t,60),contrast*.22*(t-.5),-contrast*.18*(t-.5)],axis=-1).astype('float32')
        return np.rint(cv2.cvtColor(lab,cv2.COLOR_LAB2RGB)*255).clip(0,255).astype('uint8')
    return np.repeat(np.rint(128+contrast*(t-.5)).clip(0,255).astype('uint8')[:,:,None],3,axis=2)


def controls(rows,geometry,split_name):
    candidates=[r for r in rows if r['asset_status']=='ok' and r['roi_status']=='ok'][:8]
    if not candidates:raise RuntimeError('无草图支持程序控制；停止测量')
    periods=[12,20,28] if split_name=='development' else [14,18,24]
    angles=[0,30] if split_name=='development' else [15,45]
    output=[];gallery=[];seed=420 if split_name=='development' else 421
    for i,(kind,p,angle,contrast) in enumerate(itertools.product(['stripe','plaid','ellipse'],periods,angles,[16,64,160])):
        row=candidates[i%len(candidates)];g=geometry[row['sample_id']];w,h=g['size']
        mask,rois,_=sketch_geometry(row['sketch_path'],(w,h))
        rgb=procedural(kind,p,angle,.13 if split_name=='development' else .37,contrast,i%2,max(w,h))[:h,:w]
        positive=np.where(mask[:,:,None],rgb,255).astype('uint8')
        # 每个阳性对应一个阴性，三个阴性类别等量、与外形/颜色边缘相匹配。
        negative=np.full_like(positive,160);negative[~mask]=255
        neg_kind=['solid','single_logo','shuffled'][i%3]
        if neg_kind=='single_logo':
            points=np.argwhere(mask);cy,cx=points[len(points)//2]
            cv2.putText(negative,'X',(max(0,int(cx)-12),int(cy)),cv2.FONT_HERSHEY_SIMPLEX,1.,(50,90,120),3)
        if neg_kind=='shuffled':
            rng=np.random.default_rng(seed+i);block=8
            tiled=rgb[:h//block*block,:w//block*block].reshape(h//block,block,w//block,block,3).transpose(0,2,1,3,4)
            shuffled=tiled.reshape(-1,block,block,3)[rng.permutation((h//block)*(w//block))]
            shuffled=shuffled.reshape(h//block,w//block,block,block,3).transpose(0,2,1,3,4).reshape(h//block*block,w//block*block,3)
            negative[:shuffled.shape[0],:shuffled.shape[1]]=shuffled;negative[~mask]=255
        for label,image,family in [(1,positive,kind),(0,negative,neg_kind)]:
            ws=[measure_window(image,q) for q in rois]
            record=dict(id=split_name+'_'+str(i)+'_'+str(label),positive=bool(label),family=family,
                period=p,angle=angle,contrast=contrast,chroma_only=bool(i%2),carrier=row['sample_id'],windows=ws)
            output.append(record)
            if i<12:
                thumb=Image.fromarray(image);thumb.thumbnail((126,168))
                gallery.append((record['id']+' '+family,thumb))
    sheet=Image.new('RGB',(8*144,((len(gallery)+7)//8)*200),'white');draw=ImageDraw.Draw(sheet)
    for i,(label,thumb) in enumerate(gallery):
        x=i%8*144;y=i//8*200;sheet.paste(thumb,(x,y));draw.text((x,y+172),label,fill='black')
    dest=OUT/'review_contact_sheets';dest.mkdir(exist_ok=True);sheet.save(dest/(split_name+'_controls.jpg'))
    return output


def rates(records,t,mode):
    pos=[r for r in records if r['positive']];neg=[r for r in records if not r['positive']]
    score=lambda r:summarize(r['windows'],t,mode)['status']=='repeat'
    tp=sum(score(r) for r in pos);fp=sum(score(r) for r in neg)
    return dict(positive_n=len(pos),negative_n=len(neg),true_positive=tp,false_positive=fp,
        tpr=tp/len(pos),fpr=fp/len(neg),
        positive_NA=sum(summarize(r['windows'],t,mode)['status']=='NA' for r in pos),
        negative_NA=sum(summarize(r['windows'],t,mode)['status']=='NA' for r in neg),
        negative_by_type={k:dict(n=sum(r['family']==k for r in neg),false_positive=sum(score(r) for r in neg if r['family']==k)) for k in sorted({r['family'] for r in neg})})


def matcher_pairs(records):
    positive=[r for r in records if r['positive']];pairs=[]
    for r in positive:
        ws=[w for w in r['windows'] if w['valid']]
        transformed=next(q for q in positive if q['family']==r['family'] and q['period']!=r['period'] and q['angle']!=r['angle'] and q['contrast']!=r['contrast'])
        tws=[w for w in transformed['windows'] if w['valid']]
        if ws and tws:pairs.append(dict(positive=True,score=compatibility(ws[0]['signature'],tws[0]['signature']),left=r['id'],right=transformed['id']))
        other=next(q for q in positive if q['family']!=r['family'] and q['period']==r['period'] and q['angle']==r['angle'] and q['contrast']==r['contrast'])
        vs=[w for w in other['windows'] if w['valid']]
        if ws and vs:pairs.append(dict(positive=False,score=compatibility(ws[0]['signature'],vs[0]['signature']),left=r['id'],right=other['id']))
    return pairs


def matcher_rates(pairs,t):
    p=[r for r in pairs if r['positive']];n=[r for r in pairs if not r['positive']]
    return dict(positive_n=len(p),negative_n=len(n),tpr=sum(r['score']>=t for r in p)/max(len(p),1),fpr=sum(r['score']>=t for r in n)/max(len(n),1))


def calibrate(rows,geometry):
    assert not (OUT/'calibration.json').exists(),'禁止重复调阈值'
    event('procedural_development_begin',real_RGB_decoded=False)
    train=controls(rows,geometry,'development');write(OUT/'controls_development.json',train)
    settings=[]
    for z,corr,ncc in itertools.product([3.,5.,7.],[.15,.25,.35],[.45,.60,.75]):
        t=dict(z=z,corr=corr,ncc=ncc);r=rates(train,t,'joint')
        settings.append(dict(threshold=t,rates=r))
    best=max(settings,key=lambda r:(r['rates']['fpr']<=.05,r['rates']['tpr']-r['rates']['fpr'],r['threshold']['ncc'],r['threshold']['z']))
    pairs=matcher_pairs(train)
    mt=max(np.linspace(.35,.99,65),key=lambda t:(matcher_rates(pairs,t)['fpr']<=.05,matcher_rates(pairs,t)['tpr']-matcher_rates(pairs,t)['fpr'],t))
    locked=dict(config=CONFIG,threshold=best['threshold'],compatibility_threshold=float(mt),locked_at=now(),
        development_control_sha256=sha(OUT/'controls_development.json'),selection='development only; heldout evaluated once; real RGB values unopened')
    write(OUT/'detector_lock.json',locked);event('detector_locked',sha256=sha(OUT/'detector_lock.json'))
    held=controls(rows,geometry,'heldout');write(OUT/'controls_heldout.json',held)
    results={m:rates(held,locked['threshold'],m) for m in MODES}
    held_pairs=matcher_pairs(held);mr=matcher_rates(held_pairs,float(mt))
    detector_pass=(results['joint']['tpr']>=.90 and results['joint']['fpr']<=.05 and
        all(r['tpr']>r['fpr'] for r in results.values()))
    matcher_pass=mr['tpr']>=.90 and mr['fpr']<=.05
    cal=dict(**locked,lock_sha256=sha(OUT/'detector_lock.json'),development_grid=settings,
        development={m:rates(train,locked['threshold'],m) for m in MODES},heldout=results,
        compatibility_development=matcher_rates(pairs,float(mt)),compatibility_heldout=mr,
        compatibility_pairs_development=pairs,compatibility_pairs_heldout=held_pairs,
        detector_pass=detector_pass,compatibility_pass=matcher_pass,pass_calibration=detector_pass and matcher_pass,
        heldout_split='periods14/18/24 and angles15/45; development12/20/28 and0/30; phases held out; all colors/carriers within split',
        limitation='Program controls establish engineering calibration only; source sharing across carriers and image-domain limitations remain. B has independent unwindowed proposals, shares Lab preprocessing with A.')
    write(OUT/'calibration.json',cal);event('calibration_complete',passed=cal['pass_calibration'],heldout=results,matcher=mr)
    return cal


def reference_windows(path):
    with Image.open(path) as im:rgb=np.asarray(im.convert('RGB'))
    h,w=rgb.shape[:2];dx=round(w*.05);dy=round(h*.05)
    rgb=rgb[dy:h-dy,dx:w-dx];mask=np.ones(rgb.shape[:2],bool)
    rois,_=windows(mask)
    return [measure_window(rgb,q) for q in rois]


def observability(rows,geometry,cal):
    assert cal['pass_calibration']
    t=cal['threshold'];mt=cal['compatibility_threshold'];records=[];details={}
    event('reference_GT_begin',calibration_sha256=sha(OUT/'calibration.json'))
    for i,row in enumerate(rows):
        record=dict(sample_id=row['sample_id'],group_id=row['group_id'],asset_status=row['asset_status'],roi_status=row['roi_status'],modes={})
        if row['asset_status']!='ok' or row['roi_status']!='ok':
            records.append(record);continue
        g=geometry[row['sample_id']];size=tuple(g['size'])
        rw=reference_windows(row['reference_path'])
        with Image.open(row['gt_path']) as im:rgb=np.asarray(im.convert('RGB').resize(size,Image.Resampling.BILINEAR))
        yw=[measure_window(rgb,q) for q in g['rois']]
        for mode in MODES:
            r=summarize(rw,t,mode);y=summarize(yw,t,mode);stable=gt_stable(yw,t,mode)
            a=r['status']=='repeat' and y['status']=='repeat' and stable
            comp=compatible_rois(rw,yw,t,mt,mode) if a else dict(compatible=False,scores=[])
            layer='A+' if a and comp['compatible'] else ('A' if a else ('B' if r['status']=='repeat' else 'C'))
            record['modes'][mode]=dict(reference=r,gt=y,gt_stable=stable,layer=layer,compatibility=comp,
                reason='GT_repeat_unstable' if y['status']=='repeat' and not stable else 'ok')
        details[row['sample_id']]=dict(reference=rw,gt=yw,window_hash=g['window_hash'])
        records.append(record)
        if i%25==0:print('R/GT',i,len(rows),flush=True)
    write(OUT/'eligibility_records.json',records);write(OUT/'reference_gt_windows.json',details)
    csv_write(OUT/'eligibility.csv',[dict(sample_id=r['sample_id'],group_id=r['group_id'],asset_status=r['asset_status'],roi_status=r['roi_status'],
        **{m: r['modes'].get(m,{}).get('layer','NA') for m in MODES}) for r in records])
    lock=dict(time=now(),calibration_sha256=sha(OUT/'calibration.json'),eligibility_sha256=sha(OUT/'eligibility_records.json'),
        csv_sha256=sha(OUT/'eligibility.csv'),reference_gt_windows_sha256=sha(OUT/'reference_gt_windows.json'))
    write(OUT/'eligibility_lock.json',lock);event('eligibility_locked',**lock)
    return records,details


def evaluate_e5(rows,geometry,cal,eligibility,details):
    t=cal['threshold'];mt=cal['compatibility_threshold'];output=[];gwindows={};by_id={r['sample_id']:r for r in rows}
    lock=read(OUT/'eligibility_lock.json')
    assert sha(OUT/'eligibility_records.json')==lock['eligibility_sha256'] and sha(OUT/'calibration.json')==lock['calibration_sha256']
    event('E5_RGB_begin',eligibility_lock_sha256=sha(OUT/'eligibility_lock.json'))
    for e in eligibility:
        sid=e['sample_id'];r=by_id[sid]
        # A/B代理各有独立资格层；处理其A/A+并集，避免只报联合规则筛过的样本。
        if not any(v['layer'] in ['A','A+'] for v in e['modes'].values()):continue
        g=geometry[sid]
        with Image.open(r['e5_rgb_path']) as im:rgb=np.asarray(im.convert('RGB'))
        assert tuple(g['size'])==(rgb.shape[1],rgb.shape[0])
        ws=[measure_window(rgb,q) for q in g['rois']];gwindows[sid]=ws
        assert digest([w['roi'] for w in ws])==details[sid]['window_hash']
        from eval.eval_utils import prepare_evaluation_masks,json_safe
        from eval.metrics import compute_structure_preservation
        bundle=prepare_evaluation_masks(tuple(g['size']),sketch_path=r['sketch_path'],mask_policy='sketch_only')
        quality=json_safe(compute_structure_preservation(r['e5_rgb_path'],r['sketch_path'],mask_bundle=bundle))
        result=dict(sample_id=sid,group_id=r['group_id'],window_hash=g['window_hash'],modes={},
            quality_diagnostic=quality,quality_mask_policy='sketch_only; independent of repeat ROI selection')
        for mode in MODES:
            source=e['modes'].get(mode,{})
            if source.get('layer') not in ['A','A+']:continue
            yws=details[sid]['gt'];common=[i for i,(a,b) in enumerate(zip(yws,ws)) if a['valid'] and b['valid']]
            y=summarize([yws[i] for i in common],t,mode);pred=summarize([ws[i] for i in common],t,mode)
            measurable=y['coverage'] is not None and pred['coverage'] is not None
            comp=compatible_rois(details[sid]['reference'],ws,t,mt,mode) if pred['status']=='repeat' else dict(compatible=False,scores=[])
            result['modes'][mode]=dict(layer=source['layer'],gt=y,e5=pred,paired_window_count=len(common),
                C_GT=y['coverage'],C_E5=pred['coverage'],
                delta=y['coverage']-pred['coverage'] if measurable else None,
                e5_no_repeat=pred['coverage']==0 if measurable else None,
                ref_compatible_E5=comp,
                has_repeat_but_mismatch=pred['status']=='repeat' and not comp['compatible'] if measurable else None)
        output.append(result)
    write(OUT/'e5_records.json',output);write(OUT/'e5_windows.json',gwindows)
    csv_write(OUT/'e5_results.csv',[dict(sample_id=r['sample_id'],group_id=r['group_id'],window_hash=r['window_hash'],
        **{m+'_'+k:v.get(k) for m,v in r['modes'].items() for k in ['layer','C_GT','C_E5','delta','e5_no_repeat','paired_window_count','has_repeat_but_mismatch']}) for r in output],
        ['sample_id','group_id','window_hash']+[m+'_'+k for m in MODES for k in ['layer','C_GT','C_E5','delta','e5_no_repeat','paired_window_count','has_repeat_but_mismatch']])
    event('E5_RGB_complete',n=len(output));return output


def bootstrap(records):
    if not records:return dict(n=0,groups=0,delta_mean=None,delta_ci95=None,no_repeat=None,no_repeat_ci95=None)
    groups={}
    for r in records:groups.setdefault(r['group_id'],[]).append(r)
    # 同来源多行先组内均值，再按来源等权；抽样单位绝不是窗口。
    values=np.asarray([[np.mean([r[k] for r in v]) for k in ['delta','e5_no_repeat']] for v in groups.values()],float)
    rng=np.random.default_rng(CONFIG['seed']);sample=values[rng.integers(len(values),size=(2000,len(values)))].mean(axis=1)
    ci=np.quantile(sample,[.025,.975],axis=0)
    return dict(n=len(records),groups=len(values),delta_mean=float(values[:,0].mean()),delta_ci95=ci[:,0].tolist(),
        no_repeat=float(values[:,1].mean()),no_repeat_ci95=ci[:,1].tolist())


def gate(rows,cal,eligibility=None,results=None):
    audit=read(OUT/'audit.json');funnel=dict(N_total=len(rows),N_assets=sum(r['asset_status']=='ok' for r in rows),
        N_valid_roi=sum(r['asset_status']=='ok' and r['roi_status']=='ok' for r in rows),N_R_repeat=None,N_R_GT_repeat=None,N_Aplus=None)
    stats=dict(funnel=funnel,calibration_pass=cal['pass_calibration'],status='INCONCLUSIVE_MEASUREMENT',
        stop_reason='measurement_invalid' if not cal['pass_calibration'] else '',
        protocol_verified=audit['protocol_verified'],G0=False,G1=None,G2=None,modes={},
        stages=dict(manifest='complete',calibration='complete',reference_gt='not_run',e5='not_run',statistics='not_applicable'),
        na_reasons=dict(Counter(r['roi_status'] for r in rows if r['roi_status']!='ok')))
    if not audit['protocol_verified']:
        stats.update(stop_reason='protocol_unverified');return stats
    if not cal['pass_calibration']:return stats
    for mode in MODES:
        available=[r['modes'][mode] for r in eligibility if mode in r['modes']]
        a=[r for r in eligibility if r['modes'].get(mode,{}).get('layer') in ['A','A+']]
        plus=[r for r in a if r['modes'][mode]['layer']=='A+']
        paired={layer:[] for layer in ['A','A+']}
        for r in results or []:
            v=r['modes'].get(mode)
            if v and v['delta'] is not None:
                paired['A'].append(dict(group_id=r['group_id'],**v))
                if v['layer']=='A+':paired['A+'].append(dict(group_id=r['group_id'],**v))
        m=dict(layers=dict(Counter(v['layer'] for v in available)),A_n=len(a),Aplus_n=len(plus),
            Aplus_groups=len({r['group_id'] for r in plus}),Aplus_fraction=len(plus)/len(rows),
            A=bootstrap(paired['A']),Aplus=bootstrap(paired['A+']),
            GT_repeat_unstable=sum(v['reason']=='GT_repeat_unstable' for v in available),
            na_reference=sum(v['reference']['status']=='NA' for v in available),
            na_GT=sum(v['gt']['status']=='NA' for v in available),
            e5_unmeasurable=sum(mode in r['modes'] and r['modes'][mode]['delta'] is None for r in results or []),
            has_repeat_but_mismatch=sum(v['has_repeat_but_mismatch'] for v in paired['A+']))
        stats['modes'][mode]=m
    main=stats['modes']['joint'];funnel.update(N_R_repeat=sum(r['modes'].get('joint',{}).get('reference',{}).get('status')=='repeat' for r in eligibility),
        N_R_GT_repeat=main['A_n'],N_Aplus=main['Aplus_n'])
    stats['G0']=True # 不稳定GT以明确原因排除，主集合只含通过独立ROI复核者。
    stats['G1']=main['Aplus_groups']>=30
    if not stats['G1']:
        stats.update(status='INCONCLUSIVE_MEASUREMENT',stop_reason='insufficient_support')
        stats['stages'].update(reference_gt='complete',e5='not_run',statistics='support_gate_complete')
        return stats
    directions=[stats['modes'][m]['Aplus']['delta_mean'] for m in MODES]
    s=main['Aplus'];direction_ok=all(v is not None and v>=0 for v in directions)
    stats['G2']=s['groups']>=30 and s['delta_ci95'][0]>.15 and s['no_repeat']>=.40 and direction_ok
    stats.update(status='GO_REPEAT_FORMATION_DEFICIT' if stats['G2'] else 'NO_STRONG_REPEAT_FORMATION_DEFICIT',stop_reason='')
    if not direction_ok:stats.update(status='INCONCLUSIVE_MEASUREMENT',stop_reason='proxy_disagreement_or_insufficient_measurement')
    stats['stages'].update(reference_gt='complete',e5='complete',statistics='complete')
    return stats


def contact_sheets(rows,eligibility,geometry):
    dest=OUT/'review_contact_sheets';dest.mkdir(exist_ok=True)
    # 固定按原ID抽24条，选择规则与纹样/生成结果无关；图板不是新标签。
    chosen=sorted([r for r in rows if r['asset_status']=='ok'],key=lambda r:r['sample_id'])[:24]
    for page in range((len(chosen)+5)//6):
        canvas=Image.new('RGB',(4*180,6*218),'white');draw=ImageDraw.Draw(canvas)
        for j,r in enumerate(chosen[page*6:page*6+6]):
            for col,key in enumerate(['reference_path','sketch_path','gt_path','e5_rgb_path']):
                with Image.open(r[key]) as im:thumb=im.convert('RGB');thumb.thumbnail((176,190))
                x=col*180;y=j*218;canvas.paste(thumb,(x,y));draw.text((x,y+193),r['sample_id']+' '+key,fill='black')
        canvas.save(dest/('real_fixed_%02d.jpg'%page))


def run():
    cv2.setNumThreads(1)
    rows,geometry,audit=build_manifest()
    if not audit['protocol_verified']:
        cal=dict(pass_calibration=False,detector_pass=False,compatibility_pass=False,stop_reason='protocol_unverified',heldout={})
        write(OUT/'calibration.json',cal)
    else:cal=calibrate(rows,geometry)
    eligibility=None;results=None
    if cal['pass_calibration']:
        eligibility,details=observability(rows,geometry,cal)
        support=gate(rows,cal,eligibility)
        if support['G1']:
            results=evaluate_e5(rows,geometry,cal,eligibility,details)
            contact_sheets(rows,eligibility,geometry)
    stats=gate(rows,cal,eligibility,results);write(OUT/'paired_statistics.json',stats)
    if eligibility is None:csv_write(OUT/'eligibility.csv',[],['sample_id','status','reason'])
    if results is None:csv_write(OUT/'e5_results.csv',[],['sample_id','status','reason'])
    audit['input_hashes_unchanged']=all(Path(p).is_file() and sha(p)==h for p,h in read(OUT/'input_hashes.json').items())
    audit['finished']=now();audit['status']=stats['status'];audit['stage_status']=stats['stages'];write(OUT/'audit.json',audit)
    event('E42_complete',status=stats['status'],stop_reason=stats['stop_reason'],unchanged=audit['input_hashes_unchanged'])


if __name__=='__main__':run()
