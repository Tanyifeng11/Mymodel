"""完整终图评估、预锁定阈值扫描与匿名审核材料；不生成真人评分。"""
import csv
import random
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from tools.e45_protocol import (OUT, OLD, ARMS, BAD, KEYS, SIZE, CONFIG, read, write,
    sha, protocol, image_path, historical)
from tools.e44_metrics import contour_f1, highpass_error, paired


def foreground_at(image, distance, brightness):
    from garment_mask_utils import _binary_close, _filter_components
    array=np.asarray(image.convert('RGB').resize(SIZE,Image.BICUBIC),np.float32)
    border=np.concatenate([array[0],array[-1],array[:,0],array[:,-1]],axis=0)
    background=np.median(border,axis=0)
    foreground=(np.linalg.norm(array-background[None,None,:],axis=2)>distance)|(array.mean(2)<brightness)
    foreground=_binary_close(foreground,5)
    if foreground.any():
        contours,_=cv2.findContours(foreground.astype('uint8')*255,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            minimum=max(16.,max(cv2.contourArea(c) for c in contours)*.02)
            filled=np.zeros(foreground.shape,np.uint8)
            for contour in contours:
                if cv2.contourArea(contour)>=minimum:cv2.drawContours(filled,[contour],-1,255,-1)
            foreground=filled>127
    return _filter_components(foreground)


def adjusted_pair(records,a,b,key):
    aa={r['id']:r[key] for r in records if r['variant']==a and r.get(key) is not None}
    bb={r['id']:r[key] for r in records if r['variant']==b and r.get(key) is not None}
    ids=sorted(set(aa)&set(bb))
    delta=np.array([aa[i]-bb[i] for i in ids])
    rng=np.random.default_rng(32042)
    draws=np.concatenate([rng.choice(delta,(500,len(delta)),replace=True).mean(1) for _ in range(20)])
    # 两个对照 × 三个机制指标，六项同时区间；保守 Bonferroni。
    ci=np.quantile(draws,[.05/12,1-.05/12]).tolist()
    return dict(mean=float(delta.mean()),n=len(ids),ci_familywise=ci,method='Bonferroni6-bootstrap')


def assess(records, arms):
    def mean(arm,key):
        return float(np.mean([r[key] for r in records if r['variant']==arm and r.get(key) is not None]))
    means={a:{k:mean(a,k) for k in KEYS} for a in arms}
    deltas={a:{b:{k:paired(records,a,b,k) for k in KEYS} for b in ['C0','C1']}
        for a in arms if a!='C0'}
    counts={a:{k:sum(r['variant']==a and r.get(k) is not None for r in records) for k in KEYS} for a in arms}
    base={r['id']:r for r in records if r['variant']=='C0'}
    gates={}
    for arm in arms[1:]:
        d=deltas[arm]['C0']
        retention=(means['C0']['lpips_gt']-means[arm]['lpips_gt'])/(means['C0']['lpips_gt']-means['C1']['lpips_gt'])
        bad=[dict(id=r['id'],f1_delta=r['contour_f1']-base[r['id']]['contour_f1']) for r in records
            if r['variant']==arm and r.get('contour_f1') is not None
            and r['contour_f1']-base[r['id']]['contour_f1']<-.15]
        checks=dict(iou=d['iou']['ci95'][0]>=-.01,contour=d['contour_f1']['ci95'][0]>=-.02,
            tail=len(bad)==0,appearance_retention=retention>=.5,
            appearance_ci=d['lpips_gt']['ci95'][1]<0,leak=d['leak']['ci95'][1]<=.005,
            text=d['clip_text']['mean']>=-.005)
        gates[arm]=dict(checks=checks,pass_base=all(checks.values()),lpips_retention=retention,
            catastrophic=bad,new_bad_ids=[r['id'] for r in bad if r['id'] not in BAD])
    mechanism={}
    for control in ['C3','C4']:
        d={k:adjusted_pair(records,'C5',control,k) for k in ['lpips_gt','iou','contour_f1']}
        appearance=(d['lpips_gt']['ci_familywise'][1]<0 and d['iou']['ci_familywise'][0]>=-.005
            and d['contour_f1']['ci_familywise'][0]>=-.01)
        structure=(d['contour_f1']['ci_familywise'][0]>0 and d['lpips_gt']['ci_familywise'][1]<=.002
            and d['iou']['ci_familywise'][0]>=-.005)
        mechanism[control]=dict(comparisons=d,appearance_pareto=appearance,structure_pareto=structure,
            pass_increment=appearance or structure)
    increment=all(v['pass_increment'] for v in mechanism.values())
    base_pass=gates['C5']['pass_base']
    status='DEV_GO_PENDING_VISUAL_REVIEW' if base_pass and increment else ('INCONCLUSIVE' if base_pass else 'NO_GO')
    comparisons={a:{b:{k:paired(records,a,b,k) for k in KEYS} for b in ['C3','C4']}
        for a in ['C5']+(['C7'] if 'C7' in arms else [])}
    if 'C7' in arms:
        comparisons['C7'].update({b:{k:paired(records,'C7',b,k) for k in KEYS} for b in ['C5','C6']})
    subgroups={}
    for name,select in [('old_bad4',lambda r:r['id'] in BAD),('other24',lambda r:r['id'] not in BAD and r['mask_valid'])]:
        group=[r for r in records if select(r)]
        subgroups[name]={a:{k:paired(group,a,'C1',k) for k in KEYS} for a in arms if a!='C1'}
    write(OUT/'s2/means.json',means)
    write(OUT/'s2/paired_deltas.json',deltas)
    write(OUT/'s2/valid_counts.json',counts)
    write(OUT/'s2/subgroups.json',subgroups)
    write(OUT/'s2/comparisons.json',comparisons)
    write(OUT/'s2/gates.json',dict(arms=gates,mechanism=mechanism))
    write(OUT/'s2/decision.json',dict(result=status,primary='C5',base_pass=base_pass,
        mechanism_pass=increment,numerical_pass=base_pass and increment,
        S3='not_run',visual_review='pending',human_blind_ratings='not_run',
        paper_claim_supported=False,historical_e44='unchanged NO_GO'))


def panels(rows,arms,folder,cell=128):
    folder.mkdir(parents=True,exist_ok=True)
    for start in range(0,len(rows),4 if cell==128 else 1):
        page=rows[start:start+(4 if cell==128 else 1)]
        height=round(cell*4/3)
        canvas=Image.new('RGB',((3+len(arms))*cell,len(page)*(height+25)),'white')
        draw=ImageDraw.Draw(canvas)
        for y,row in enumerate(page):
            entries=[('Sketch',row['sketch']),('Reference',row['reference']),('GT',row['gt'])]
            entries += [(a,image_path(a,row['id'])) for a in arms]
            for x,(label,path) in enumerate(entries):
                draw.text((x*cell+2,y*(height+25)+2),row['id']+' '+label,fill='black')
                canvas.paste(Image.open(path).convert('RGB').resize((cell,height)),(x*cell,y*(height+25)+24))
        canvas.save(folder/('page%02d.png'%start))


def blind_pack(rows,arms):
    rng=random.Random(45043)
    key=[]
    for row in rows:
        shuffled=list(arms);rng.shuffle(shuffled)
        cell=192;height=256
        image=Image.new('RGB',((3+len(arms))*cell,height+30),'white');draw=ImageDraw.Draw(image)
        entries=[('Sketch',row['sketch']),('Reference',row['reference']),('GT',row['gt'])]
        for i,arm in enumerate(shuffled):
            label=chr(65+i);entries.append((label,image_path(arm,row['id'])))
            key.append(dict(id=row['id'],label=label,arm=arm))
        for i,(label,path) in enumerate(entries):
            draw.text((i*cell+2,2),label,fill='black')
            image.paste(Image.open(path).convert('RGB').resize((cell,height)),(i*cell,25))
        folder=OUT/'s2/blind_pack';folder.mkdir(parents=True,exist_ok=True)
        image.save(folder/(row['id']+'.png'))
    write(OUT/'s2/blind_key.json',key)
    with (OUT/'s2/blind_pack/ratings_blank.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['reviewer','id','label','contour','halo','connections','pattern','comment'])
        writer.writeheader()
        for r in key:writer.writerow({k:v for k,v in r.items() if k!='arm'})


@torch.inference_mode()
def evaluate():
    from tools.e44_runtime import sketch_masks
    from tools.e44_b1_mask import overlay
    from tools.e33tmoc_appearance_eval import frozen_lpips
    from eval.eval_utils import prepare_evaluation_masks,estimate_foreground_mask,json_safe
    from eval.metrics import compute_texture_leakage,compute_texture_color_fidelity
    from transformers import CLIPConfig,CLIPModel,CLIPProcessor
    rows,frozen=protocol()
    arms=read(OUT/'s1/smoke_complete.json')['arms']
    for arm in arms:assert read(OUT/'s2'/(arm+'_complete.json'))['n']==32
    metric,info=frozen_lpips();metric=metric.cuda()
    clip_config=CLIPConfig.from_pretrained('models/clip/models/image_encoder',local_files_only=True)
    model=CLIPModel.from_pretrained('models/clip',config=clip_config,local_files_only=True).cuda().eval().requires_grad_(False)
    processor=CLIPProcessor.from_pretrained('models/clip/models/image_encoder',local_files_only=True)
    records=[];sensitivity=[];source_checks=[]
    old_p1={(r['variant'],r['id']):r for r in read(OLD/'p1/metrics.json')}
    old_mask={r['id']:r for r in read(OLD/'b1_mask_intervention_20261012/metrics.json')
        if r['variant']=='interior'}
    baseline_metric_errors=[]
    for row in rows:
        gt=Image.open(row['gt']).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR)
        sketch=Image.open(row['sketch']).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR)
        _,_,mask=sketch_masks(sketch,'cpu')
        bundle=prepare_evaluation_masks(SIZE,sketch_path=row['sketch'],mask_policy='sketch_only')
        garment=bundle['garment']
        if garment is None:garment=np.zeros((SIZE[1],SIZE[0]),bool)
        mask_folder=OUT/'s2/masks'/row['id'];mask_folder.mkdir(parents=True,exist_ok=True)
        Image.fromarray(garment.astype('uint8')*255).save(mask_folder/'sketch_foreground.png')
        ref=Image.open(row['reference']).convert('RGB')
        for arm in arms:
            path=image_path(arm,row['id']);a=Image.open(path).convert('RGB')
            tensor=lambda im:torch.from_numpy(np.asarray(im,np.float32).transpose(2,0,1)/127.5-1)[None].cuda()
            rec=dict(id=row['id'],variant=arm,mask_valid=mask['valid'],
                **{k:None for k in KEYS})
            rec['lpips_gt']=float(metric(tensor(a),tensor(gt)).mean())
            fg=estimate_foreground_mask(a,SIZE)
            if fg is None:fg=np.zeros((SIZE[1],SIZE[0]),bool)
            assert np.array_equal(fg,foreground_at(a,18,235)), '敏感性扫描的原阈值未精确复现原分割器'
            Image.fromarray(fg.astype('uint8')*255).save(mask_folder/(arm+'_foreground.png'))
            overlay(a,fg,garment).save(mask_folder/(arm+'_overlay.png'))
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
            rec['clip_text']=float((tf[0]*imf[0]).sum())
            rec['clip_reference']=float((imf[0]*imf[1]).sum())
            rec['clip_real']=float((imf[0]*imf[2]).sum())
            records.append(rec)
            for distance in CONFIG['threshold_scan']['color_distance']:
                for brightness in CONFIG['threshold_scan']['brightness']:
                    f=foreground_at(a,distance,brightness)
                    sensitivity.append(dict(id=row['id'],variant=arm,distance=distance,brightness=brightness,
                        iou=float((f&garment).sum()/max((f|garment).sum(),1)) if mask['valid'] else None,
                        contour_f1=contour_f1(f,garment) if mask['valid'] else None))
            if arm in ['C0','C1','C2']:
                source_checks.append(dict(id=row['id'],arm=arm,source=str(historical(arm,row['id'])),
                    source_sha256=sha(historical(arm,row['id'])),replay_sha256=sha(path)))
                assert source_checks[-1]['source_sha256']==source_checks[-1]['replay_sha256']
                baseline=old_mask[row['id']] if arm=='C2' else old_p1[('B0' if arm=='C0' else 'B1',row['id'])]
                for key in KEYS:
                    if key not in baseline:continue
                    assert (rec[key] is None)==(baseline[key] is None), (arm,row['id'],key,'有效分母变化')
                    if rec[key] is not None:
                        error=abs(rec[key]-baseline[key])
                        baseline_metric_errors.append(error)
                        assert error<=2e-6, (arm,row['id'],key,'历史指标回放不一致',error)
        write(OUT/'s2/metrics.json',records)
        print('E45 METRICS',row['id'],len(records),flush=True)
    write(OUT/'s2/metric_protocol.json',dict(lpips=info,original_E44_definition=True,
        synthetic_proxy_limitation=True,gt_not_used_in_generation=True,counts='global32/regional28',
        baseline_metric_max_error=max(baseline_metric_errors),baseline_metric_tolerance=2e-6))
    write(OUT/'s2/baseline_source_audit.json',source_checks)
    write(OUT/'s2/threshold_sensitivity.json',sensitivity)
    threshold_summary={}
    for distance in CONFIG['threshold_scan']['color_distance']:
        for brightness in CONFIG['threshold_scan']['brightness']:
            group=[r for r in sensitivity if r['distance']==distance and r['brightness']==brightness]
            threshold_summary['%d_%d'%(distance,brightness)]={a:{k:paired(group,a,'C0',k) for k in ['iou','contour_f1']}
                for a in arms if a!='C0'}
    write(OUT/'s2/threshold_summary.json',threshold_summary)
    with (OUT/'s2/metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    assess(records,arms)
    panels(rows,arms,OUT/'s2/panels')
    critical=set(BAD+['140540361','174397224'])
    panels([r for r in rows if r['id'] in critical],arms,OUT/'s2/critical',cell=384)
    base={r['id']:r for r in records if r['variant']=='C0'}
    for arm in arms[1:]:
        worst=sorted([r for r in records if r['variant']==arm and r['contour_f1'] is not None],
            key=lambda r:r['contour_f1']-base[r['id']]['contour_f1'])[:10]
        write(OUT/'s2/worst_cases'/(arm+'.json'),worst)
        lookup={r['id']:r for r in rows}
        panels([lookup[r['id']] for r in worst],['C0','C1',arm],OUT/'s2/worst_cases'/arm)
    blind_pack([r for r in rows if r['id'] in frozen['smoke_ids']],arms)
    write(OUT/'s2/evaluation_complete.json',dict(complete=True,identities=32,arms=arms,
        images=32*len(arms),protocol_sha256=sha(OUT/'protocol.json')))


def F_normalize(x):
    return torch.nn.functional.normalize(x,dim=-1)
