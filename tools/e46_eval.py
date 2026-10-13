"""e46 终图评价与预注册效应量判定；不在开发结果上选门控配方。"""
import csv
import cv2
import numpy as np
import torch
from PIL import Image,ImageDraw
from tools.e46_protocol import (OUT,OLD,E45,ARMS,HISTORY,BAD,KEYS,SIZE,CONFIG,
    read,write,sha,protocol,image_path)
from tools.e44_metrics import contour_f1,highpass_error,paired
from tools.e45_eval import foreground_at,F_normalize


def assess(records,arms):
    means={a:{k:float(np.mean([r[k] for r in records if r['variant']==a and r.get(k) is not None]))
        for k in KEYS} for a in arms}
    counts={a:{k:sum(r['variant']==a and r.get(k) is not None for r in records) for k in KEYS} for a in arms}
    assert all(counts[a]['lpips_gt']==counts[a]['clip_text']==32 for a in arms)
    assert all(counts[a]['iou']==counts[a]['contour_f1']==counts[a]['leak']==28 for a in arms)
    comparisons={a:{b:{k:paired(records,a,b,k) for k in KEYS} for b in ['G0','G1','G2','G3','H45_C3','H45_C4','H45_C5'] if a!=b}
        for a in ARMS}
    baseline={r['id']:r for r in records if r['variant']=='G0'}
    gates={}
    for arm in arms:
        if arm=='G0':continue
        d={k:paired(records,arm,'G0',k) for k in KEYS}
        tail=[dict(id=r['id'],f1_delta=r['contour_f1']-baseline[r['id']]['contour_f1']) for r in records
            if r['variant']==arm and r['contour_f1'] is not None
            and r['contour_f1']-baseline[r['id']]['contour_f1']<-.15]
        retention=(means['G0']['lpips_gt']-means[arm]['lpips_gt'])/(means['G0']['lpips_gt']-means['G1']['lpips_gt'])
        checks=dict(iou=d['iou']['ci95'][0]>=-.01,contour=d['contour_f1']['ci95'][0]>=-.02,
            leak=d['leak']['ci95'][1]<=.005,tail=not tail,
            lpips_mean=d['lpips_gt']['mean']<0,appearance_retention=retention>=.5,
            text=d['clip_text']['mean']>=-.005)
        gates[arm]=dict(checks=checks,lpips_retention=retention,catastrophic=tail,
            new_bad_ids=[v['id'] for v in tail if v['id'] not in BAD],
            lpips_ci_upper_negative=d['lpips_gt']['ci95'][1]<0,comparisons_to_e5=d)
    difference=comparisons['G4']['G3']
    structure=difference['contour_f1']['mean']>=.005 and difference['lpips_gt']['mean']<=.002
    appearance=difference['lpips_gt']['mean']<=-.002 and difference['contour_f1']['mean']>=-.003
    independent=structure or appearance
    checks=gates['G4']['checks']
    safe=all(checks[k] for k in ['iou','contour','leak','tail','text'])
    looks=checks['lpips_mean'] and checks['appearance_retention']
    reasons=[]
    if not checks['tail']:reasons.append('存在相对 E5 的 F1<-0.15 尾部事件')
    if not looks:reasons.append('LPIPS 均值收益或原 B1 收益保留50%不通过')
    if not checks['text']:reasons.append('CLIP-T 均值安全门槛不通过')
    if not independent:reasons.append('相对固定能量缩放的预注册开发效应量未通过')
    for key in ['iou','contour','leak']:
        if not checks[key]:reasons.append(key+' 配对95%CI安全门槛不通过')
    if not checks['tail'] or not looks or not checks['text'] or not independent:decision='NO_GO'
    elif not safe:decision='INCONCLUSIVE'
    else:decision='CONDITIONAL_GO'
    groups={}
    for name,select in [('old_bad4',lambda r:r['id'] in BAD),('other24',lambda r:r['mask_valid'] and r['id'] not in BAD)]:
        group=[r for r in records if select(r)]
        groups[name]={a:{b:{k:paired(group,a,b,k) for k in KEYS} for b in ['G0','G1']} for a in ARMS}
    for name,data in [('means',means),('valid_counts',counts),('paired_deltas',comparisons),('subgroups',groups),
        ('gates',gates),('independent_advantage',dict(comparisons=difference,structure_pareto=structure,
            appearance_pareto=appearance,pass_effect_size=independent,confirmatory_significance=False))]:
        write(OUT/'p1'/(name+'.json'),data)
    write(OUT/'decision.json',dict(engineering_pass=True,safety_pass=safe,appearance_pass=looks,
        vs_const_energy_pass=independent,decision=decision,reasons=reasons,primary='G4_SARG',
        visual_review='pending',human_blind_ratings='not_run',not_run_stages=['P3_learnable_gate','confirm96','validation500'],
        next_stage='stop_no_tuning' if decision=='NO_GO' else 'review_and_freeze_independent_protocol_before_any_confirmation',
        strict_unseen_generalization=False,paper_claim_supported=False))


def panels(rows,arms,folder,cell=128,overlays=False):
    folder.mkdir(parents=True,exist_ok=True);per_page=4 if cell==128 else 1
    height=round(cell*4/3)
    for start in range(0,len(rows),per_page):
        page=rows[start:start+per_page]
        canvas=Image.new('RGB',((3+len(arms))*cell,len(page)*(height+25)),'white');draw=ImageDraw.Draw(canvas)
        for y,row in enumerate(page):
            entries=[('Sketch',row['sketch']),('Reference',row['reference']),('GT',row['gt'])]
            entries += [(a,OUT/'p1/masks'/row['id']/(a+'_overlay.png') if overlays else image_path(a,row['id'])) for a in arms]
            for x,(label,path) in enumerate(entries):
                draw.text((x*cell+2,y*(height+25)+2),row['id']+' '+label,fill='black')
                canvas.paste(Image.open(path).convert('RGB').resize((cell,height)),(x*cell,y*(height+25)+24))
        canvas.save(folder/('page%02d.png'%start))


def residual_summary(rows):
    values=[]
    for arm in ARMS:
        for row in rows:
            v=read(image_path(arm,row['id']).with_name(row['id']+'_gate.json'))
            valid=[s for s in v['steps'] if s['enabled']]
            result=dict(id=row['id'],variant=arm,mask_valid=bool(valid),steps=len(valid))
            for region in ['interior','boundary','background']:
                result[region]={kind:float(np.mean([s[kind][region]['rms'] for s in valid
                    if s[kind][region]['rms'] is not None])) if valid else None for kind in ['raw','applied']}
            result['raw_energy']=sum(s['raw_energy'] for s in valid)
            result['applied_energy']=sum(s['applied_energy'] for s in valid)
            values.append(result)
    write(OUT/'p1/residual_by_identity.json',values)
    aggregate={a:{region:{k:float(np.mean([r[region][k] for r in values if r['variant']==a and r['mask_valid']]))
        for k in ['raw','applied']} for region in ['interior','boundary','background']} for a in ARMS}
    write(OUT/'p1/residual_summary.json',dict(means=aggregate,
        interpretation='same fixed B1 weights; different free trajectories; RMS is diagnostic, not final structure proof',
        energy_scope='G3 matches training-calibration B1 trajectory, not necessarily dev free-trajectory energy'))


@torch.inference_mode()
def evaluate():
    from tools.e46_run import import_history
    from tools.e44_runtime import sketch_masks
    from tools.e44_b1_mask import overlay
    from tools.e33tmoc_appearance_eval import frozen_lpips
    from eval.eval_utils import prepare_evaluation_masks,estimate_foreground_mask,json_safe
    from eval.metrics import compute_texture_leakage,compute_texture_color_fidelity
    from transformers import CLIPConfig,CLIPModel,CLIPProcessor
    rows,_=protocol();arms=HISTORY[:2]+ARMS+HISTORY[2:]
    assert read(OUT/'p0/audit.json')['engineering_pass']
    for arm in ARMS:assert read(OUT/'p1'/(arm+'_complete.json'))['n']==32
    import_history(rows)
    metric,info=frozen_lpips();metric=metric.cuda()
    config=CLIPConfig.from_pretrained('models/clip/models/image_encoder',local_files_only=True)
    model=CLIPModel.from_pretrained('models/clip',config=config,local_files_only=True).cuda().eval().requires_grad_(False)
    processor=CLIPProcessor.from_pretrained('models/clip/models/image_encoder',local_files_only=True)
    old_p1={(r['variant'],r['id']):r for r in read(OLD/'p1/metrics.json')}
    old45={(r['variant'],r['id']):r for r in read(E45/'s2/metrics.json')}
    records=[];sensitivity=[];errors=[]
    for row in rows:
        gt=Image.open(row['gt']).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR)
        sketch=Image.open(row['sketch']).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR)
        _,_,mask=sketch_masks(sketch,'cpu')
        bundle=prepare_evaluation_masks(SIZE,sketch_path=row['sketch'],mask_policy='sketch_only')
        garment=bundle['garment']
        if garment is None:garment=np.zeros((SIZE[1],SIZE[0]),bool)
        folder=OUT/'p1/masks'/row['id'];folder.mkdir(parents=True,exist_ok=True)
        Image.fromarray(garment.astype('uint8')*255).save(folder/'sketch_foreground.png')
        ref=Image.open(row['reference']).convert('RGB')
        for arm in arms:
            path=image_path(arm,row['id']);a=Image.open(path).convert('RGB')
            tensor=lambda im:torch.from_numpy(np.asarray(im,np.float32).transpose(2,0,1)/127.5-1)[None].cuda()
            rec=dict(id=row['id'],variant=arm,mask_valid=mask['valid'],**{k:None for k in KEYS})
            rec['lpips_gt']=float(metric(tensor(a),tensor(gt)).mean())
            fg=estimate_foreground_mask(a,SIZE)
            if fg is None:fg=np.zeros((SIZE[1],SIZE[0]),bool)
            assert np.array_equal(fg,foreground_at(a,18,235)),'原阈值扫描没有复现原分割器'
            Image.fromarray(fg.astype('uint8')*255).save(folder/(arm+'_foreground.png'))
            overlay(a,fg,garment).save(folder/(arm+'_overlay.png'))
            if mask['valid']:
                rec['iou']=float((fg&garment).sum()/max((fg|garment).sum(),1))
                rec['contour_f1']=contour_f1(fg,garment)
                rec['leak']=json_safe(compute_texture_leakage(str(path),target_path=row['gt'],mask_bundle=bundle)).get('leak_colored_frac')
                rec['lab']=json_safe(compute_texture_color_fidelity(str(path),row['reference'],mask_bundle=bundle)).get('tcf_lab_delta')
                inner=cv2.erode(garment.astype('uint8'),np.ones((19,19),np.uint8))>0
                rec['hp_gray']=highpass_error(a,gt,inner) if inner.any() else None
            inputs=processor(text=[row['caption']],images=[a,ref,gt],return_tensors='pt',padding=True,truncation=True)
            inputs={k:v.cuda() for k,v in inputs.items()}
            tf=F_normalize(model.get_text_features(input_ids=inputs['input_ids'],attention_mask=inputs.get('attention_mask')))
            imf=F_normalize(model.get_image_features(pixel_values=inputs['pixel_values']))
            rec['clip_text']=float((tf[0]*imf[0]).sum());rec['clip_reference']=float((imf[0]*imf[1]).sum())
            rec['clip_real']=float((imf[0]*imf[2]).sum());records.append(rec)
            for distance in CONFIG['threshold_scan']['distance']:
                for brightness in CONFIG['threshold_scan']['brightness']:
                    f=foreground_at(a,distance,brightness)
                    sensitivity.append(dict(id=row['id'],variant=arm,distance=distance,brightness=brightness,
                        iou=float((f&garment).sum()/max((f|garment).sum(),1)) if mask['valid'] else None,
                        contour_f1=contour_f1(f,garment) if mask['valid'] else None))
            if arm in HISTORY:
                old=old_p1[('B0' if arm=='G0' else 'B1',row['id'])] if arm in ['G0','G1'] else old45[(arm.replace('H45_',''),row['id'])]
                for k in KEYS:
                    if k not in old:continue
                    assert (rec[k] is None)==(old[k] is None)
                    if rec[k] is not None:
                        error=abs(rec[k]-old[k]);errors.append(error)
                        assert error<=2e-6,(arm,row['id'],k,'历史指标不一致',error)
        write(OUT/'p1/metrics.json',records)
        print('E46 METRICS',row['id'],len(records),flush=True)
    write(OUT/'p1/metric_protocol.json',dict(lpips=info,original_E44_definition=True,
        baseline_metric_max_error=max(errors),baseline_tolerance=2e-6,gt_not_used_in_generation=True,
        synthetic_proxy_limitation=True,counts='global32/regional28; per metric valid N reported'))
    write(OUT/'p1/threshold_sensitivity.json',sensitivity)
    summaries={}
    for distance in CONFIG['threshold_scan']['distance']:
        for brightness in CONFIG['threshold_scan']['brightness']:
            group=[r for r in sensitivity if r['distance']==distance and r['brightness']==brightness]
            summaries['%d_%d'%(distance,brightness)]={a:{b:{k:paired(group,a,b,k) for k in ['iou','contour_f1']}
                for b in ['G0','G3'] if a!=b} for a in arms if a!='G0'}
    write(OUT/'p1/threshold_summary.json',summaries)
    with (OUT/'p1/metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    assess(records,arms);residual_summary(rows)
    panels(rows,arms,OUT/'p1/panels')
    critical=[r for r in rows if r['id'] in BAD+['174397224','140540361']]
    panels(critical,arms,OUT/'p1/critical',cell=384)
    panels(critical,['G0','G1','G3','G4'],OUT/'p1/critical_overlays',cell=384,overlays=True)
    base={r['id']:r for r in records if r['variant']=='G0'};lookup={r['id']:r for r in rows}
    for arm in ARMS:
        worst=sorted([r for r in records if r['variant']==arm and r['contour_f1'] is not None],
            key=lambda r:r['contour_f1']-base[r['id']]['contour_f1'])[:10]
        write(OUT/'p1/worst_cases'/(arm+'.json'),worst)
        panels([lookup[r['id']] for r in worst],['G0','G1',arm],OUT/'p1/worst_cases'/arm)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    means=read(OUT/'p1/means.json');fig,ax=plt.subplots(figsize=(7,5))
    for arm in arms:
        x,y=means[arm]['lpips_gt'],means[arm]['contour_f1'];ax.scatter(x,y);ax.annotate(arm,(x,y),xytext=(4,4),textcoords='offset points')
    ax.set(xlabel='LPIPS-GT (lower is better)',ylabel='Contour F1 (higher is better)',title='e46 / reused dev32 / seed42')
    ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'p1/pareto.png',dpi=180);plt.close(fig)
    write(OUT/'p1/evaluation_complete.json',dict(complete=True,identities=32,arms=arms,
        newly_generated_images=96,historical_images=160,protocol_sha256=sha(OUT/'protocol.json'),
        calibrated_protocol_sha256=sha(OUT/'protocol_calibrated.json')))
