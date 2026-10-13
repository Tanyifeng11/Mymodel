"""e47 固定 dev32：统一完整图像、身份配对交互及结构安全门槛。"""
import csv
import gc
import hashlib
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image,ImageDraw
from tools.e44_runtime import sketch_masks
from tools.e44_metrics import contour_f1,highpass_error,paired
from tools.e35_freeu_protocol import bootstrap
from tools.e47_protocol import *
from tools.e47_runtime import runtime,close

ARMS=['A','B','C','D','D_safe']
KEYS=['lpips_gt','contour_f1','iou','leak','lab','hp_gray','clip_text','clip_reference','clip_real']
BAD=['172600892','180633937','193488637','204446309']


def image_path(arm,identity):return OUT/'p2/images'/arm/(identity+'.png')


def generate(arm):
    assert read(OUT/'decision.json')['P0']=='pass'
    if arm=='C':assert read(OUT/'p1/C/complete.json')['updates']==800
    rows=read(OUT/'splits/dev32.json')
    if arm in ['B','D','D_safe']:
        import_history(rows,[arm]);return
    r=runtime('A',arm=='C','p2/'+arm+'_runtime')
    if arm=='C':r.load(OUT/'p1/C/checkpoints/step0800.pt')
    try:
        for i,row in enumerate(rows):
            path=image_path(arm,row['id'])
            if path.exists():assert read(path.with_suffix('.json'))['png_sha256']==sha(path)
            else:r.generate(row,path)
            old=read(OLD/'p1/images/B0'/(row['id']+'.json'))
            new=read(path.with_suffix('.json'));assert old['initial_sha256']==new['initial_sha256']
            print('e47 RGB',arm,i+1,32,flush=True)
    finally:close(r)
    write(OUT/'p2'/(arm+'_complete.json'),dict(identities=32,seed=42,commit=commit()))


def import_history(rows,arms=('B','D','D_safe')):
    frozen=read(OLD/'manifests/input_hashes.json')
    assert sha(E5)==E5_SHA and sha(B1)==B1_SHA
    assert read(OUT/'splits/dev32.json')==read(OLD/'splits/dev32.json')
    assert sha(OLD/'splits/dev32.json')==frozen[str(OLD/'splits/dev32.json')]
    provenance=[]
    for arm in arms:
        for row in rows:
            source=(E46/'p1/images/G3' if arm=='D_safe' else OLD/'p1/images'/('B0' if arm=='B' else 'B1'))/(row['id']+'.png')
            meta=read(source.with_suffix('.json'));assert sha(source)==meta['png_sha256']
            assert sha(source.with_suffix('.pt'))==meta['diagnostics_sha256']
            assert meta['seed']==42 and len(meta['steps'])==50 and not meta['gt_used_in_inference']
            expected=hashlib.sha256(Image.open(row['reference']).convert('RGB').tobytes()).hexdigest()
            assert meta['reference_sha256']==expected
            assert all(sha(row[k])==frozen[row[k]] for k in ['gt','sketch','reference'])
            if arm=='D_safe':
                calibrated=read(E46/'protocol_calibrated.json')
                assert calibrated['lambda_E']==SAFE_SCALE and calibrated['frozen_before_dev']
            target=image_path(arm,row['id']);target.parent.mkdir(parents=True,exist_ok=True)
            for ext in ['.png','.json','.pt']:
                src=source.with_suffix(ext);dst=target.with_suffix(ext)
                if dst.exists():assert sha(dst)==sha(src)
                else:shutil.copy2(src,dst)
            provenance.append(dict(arm=arm,id=row['id'],source=str(source),png_sha256=sha(source),
                trace_sha256=sha(source.with_suffix('.pt')),source_metadata_sha256=sha(source.with_suffix('.json'))))
        write(OUT/'p2'/(arm+'_complete.json'),dict(identities=32,reused=True,source='E46 G3' if arm=='D_safe' else 'E44'))
    write(OUT/'p2'/'history_provenance.json',provenance)


@torch.inference_mode()
def evaluate():
    from eval.eval_utils import prepare_evaluation_masks,estimate_foreground_mask,json_safe
    from eval.metrics import compute_texture_leakage,compute_texture_color_fidelity
    from transformers import CLIPConfig,CLIPModel,CLIPProcessor
    from tools.e33tmoc_appearance_eval import frozen_lpips
    from tools.e44_b1_mask import overlay
    assert read(OUT/'decision.json')['P0']=='pass'
    assert read(OUT/'p1/C/complete.json')['updates']==800
    rows=read(OUT/'splits/dev32.json');import_history(rows)
    metric,lpips_info=frozen_lpips();metric=metric.cuda()
    config=CLIPConfig.from_pretrained('models/clip/models/image_encoder',local_files_only=True)
    model=CLIPModel.from_pretrained('models/clip',config=config,local_files_only=True).cuda().eval().requires_grad_(False)
    processor=CLIPProcessor.from_pretrained('models/clip/models/image_encoder',local_files_only=True)
    old44={(r['variant'],r['id']):r for r in read(OLD/'p1/metrics.json')}
    old46={(r['variant'],r['id']):r for r in read(E46/'p1/metrics.json')}
    records=[];errors=[]
    tensor=lambda im:torch.from_numpy(np.asarray(im,np.float32).transpose(2,0,1)/127.5-1)[None].cuda()
    for row in rows:
        gt=Image.open(row['gt']).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR)
        ref=Image.open(row['reference']).convert('RGB')
        sketch=Image.open(row['sketch']).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR)
        _,_,mask=sketch_masks(sketch,'cpu')
        bundle=prepare_evaluation_masks(SIZE,sketch_path=row['sketch'],mask_policy='sketch_only')
        garment=bundle['garment'];folder=OUT/'p2/masks'/row['id'];folder.mkdir(parents=True,exist_ok=True)
        for arm in ARMS:
            path=image_path(arm,row['id']);a=Image.open(path).convert('RGB')
            rec=dict(id=row['id'],variant=arm,mask_valid=bool(mask['valid']),**{k:None for k in KEYS})
            rec['lpips_gt']=float(metric(tensor(a),tensor(gt)).mean())
            fg=estimate_foreground_mask(a,SIZE)
            if fg is None:fg=np.zeros((SIZE[1],SIZE[0]),bool)
            if garment is not None:overlay(a,fg,garment).save(folder/(arm+'_overlay.png'))
            if mask['valid']:
                rec['iou']=float((fg&garment).sum()/max((fg|garment).sum(),1))
                rec['contour_f1']=contour_f1(fg,garment)
                rec['leak']=json_safe(compute_texture_leakage(str(path),target_path=row['gt'],mask_bundle=bundle)).get('leak_colored_frac')
                rec['lab']=json_safe(compute_texture_color_fidelity(str(path),row['reference'],mask_bundle=bundle)).get('tcf_lab_delta')
                inner=cv2.erode(garment.astype('uint8'),np.ones((19,19),np.uint8))>0
                rec['hp_gray']=highpass_error(a,gt,inner) if inner.any() else None
            inputs=processor(text=[row['caption']],images=[a,ref,gt],return_tensors='pt',padding=True,truncation=True)
            inputs={k:v.cuda() for k,v in inputs.items()}
            text=torch.nn.functional.normalize(model.get_text_features(input_ids=inputs['input_ids'],attention_mask=inputs.get('attention_mask')),dim=-1)
            images=torch.nn.functional.normalize(model.get_image_features(pixel_values=inputs['pixel_values']),dim=-1)
            rec['clip_text']=float((text[0]*images[0]).sum());rec['clip_reference']=float((images[0]*images[1]).sum())
            rec['clip_real']=float((images[0]*images[2]).sum());records.append(rec)
            historical=old44[('B0' if arm=='B' else 'B1',row['id'])] if arm in ['B','D'] else old46[('G3',row['id'])] if arm=='D_safe' else None
            if historical:
                for k in KEYS:
                    assert (rec[k] is None)==(historical[k] is None)
                    if rec[k] is not None:
                        error=abs(rec[k]-historical[k]);assert error<=2e-6,(arm,row['id'],k,error)
                        errors.append(error)
        write(OUT/'p2/metrics.json',records);print('e47 METRICS',row['id'],len(records),flush=True)
    with (OUT/'p2/metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    write(OUT/'p2/metric_protocol.json',dict(lpips=lpips_info,history_max_error=max(errors),
        mask_backend='opencv',tolerance=2e-6,contour_tolerance_px=3,seed_unit='identity',
        fid_kid='not_run at dev32',motif_identity_validated=False))
    assess(records);panels(rows,ARMS,OUT/'p2/panels')
    panels([r for r in rows if r['id'] in BAD],ARMS,OUT/'p2/critical',cell=256)
    for arm,base in [('B','A'),('C','A'),('D','B'),('D_safe','B')]:
        index={r['id']:r for r in records if r['variant']==base};lookup={r['id']:r for r in rows}
        worst=sorted([r for r in records if r['variant']==arm and r['contour_f1'] is not None],
            key=lambda r:r['contour_f1']-index[r['id']]['contour_f1'])[:10]
        write(OUT/'p2/worst_cases'/(arm+'.json'),worst)
        panels([lookup[r['id']] for r in worst],[base,arm],OUT/'p2/worst_cases'/arm)
    write(OUT/'p2/evaluation_complete.json',dict(identities=32,arms=ARMS,seed=42,commit=commit()))


def assess(records):
    means={arm:{k:float(np.mean([r[k] for r in records if r['variant']==arm and r[k] is not None])) for k in KEYS} for arm in ARMS}
    counts={arm:{k:sum(r['variant']==arm and r[k] is not None for r in records) for k in KEYS} for arm in ARMS}
    assert all(counts[a]['lpips_gt']==32 and counts[a]['contour_f1']==28 for a in ARMS)
    index={(r['variant'],r['id']):r for r in records};ids=[r['id'] for r in records if r['variant']=='A']
    contributions=[]
    for identity in ids:
        a,b,c,d=[index[(arm,identity)] for arm in ['A','B','C','D']]
        contributions.append(dict(id=identity,TCPM_LPIPS=a['lpips_gt']-b['lpips_gt'],
            Gain_noT=a['lpips_gt']-c['lpips_gt'],Gain_T=b['lpips_gt']-d['lpips_gt'],
            Interaction_LPIPS=(b['lpips_gt']-d['lpips_gt'])-(a['lpips_gt']-c['lpips_gt']),
            LossF1_noT=a['contour_f1']-c['contour_f1'] if a['contour_f1'] is not None else None,
            LossF1_T=b['contour_f1']-d['contour_f1'] if b['contour_f1'] is not None else None))
    effects={k:bootstrap([r[k] for r in contributions if r[k] is not None]) for k in contributions[0] if k!='id'}
    gates={};deltas={}
    for arm,base in [('B','A'),('C','A'),('D','B'),('D_safe','B')]:
        delta={k:paired(records,arm,base,k) for k in KEYS};deltas[arm]=dict(base=base,metrics=delta)
        tail=[dict(id=i,delta=index[(arm,i)]['contour_f1']-index[(base,i)]['contour_f1']) for i in ids
            if index[(arm,i)]['contour_f1'] is not None and index[(arm,i)]['contour_f1']-index[(base,i)]['contour_f1']<-.15]
        checks=dict(f1=delta['contour_f1']['ci95'][0]>=-.020,iou=delta['iou']['ci95'][0]>=-.010,
            leak=delta['leak']['ci95'][1]<=.005,tail=not tail)
        gates[arm]=dict(base=base,checks=checks,safety_pass=all(checks.values()),extreme_events=tail,
            appearance_gain=-delta['lpips_gt']['mean'],appearance_ci_positive=delta['lpips_gt']['ci95'][1]<0,
            interpretation='INCONCLUSIVE' if arm=='D_safe' and not all(checks.values()) and not tail else None)
    # 开发止损按预注册方向判断；不扫层位/强度，也不把 CI 不确定叫作通过。
    useful=effects['TCPM_LPIPS']['mean']>0 and effects['Interaction_LPIPS']['mean']>0
    strong_safe=gates['D']['safety_pass'] and gates['D']['appearance_gain']>0
    safe_aux=gates['D_safe']['safety_pass'] and gates['D_safe']['appearance_gain']>0
    aux_delta=deltas['D_safe']['metrics']
    # 原方案允许唯一冻结 D-safe 在仅 CI 不确定时做一次风险确认，不称已通过。
    aux_inconclusive=(not gates['D_safe']['extreme_events'] and gates['D_safe']['appearance_gain']>0
        and aux_delta['contour_f1']['mean']>=-.020 and aux_delta['iou']['mean']>=-.010
        and aux_delta['leak']['mean']<=.005)
    allow_confirm=useful and (strong_safe or safe_aux or aux_inconclusive)
    reason='可解释 TCPM 条件增量且存在安全配置' if allow_confirm else 'TCPM 条件增量或结构折中未满足开发继续条件'
    write(OUT/'p2/means.json',means);write(OUT/'p2/valid_counts.json',counts)
    write(OUT/'p2/identity_effects.json',contributions);write(OUT/'p2/factorial_effects.json',effects)
    write(OUT/'p2/gates.json',gates);write(OUT/'p2/paired_deltas.json',deltas)
    write(OUT/'p2/subgroups.json',{name:{arm:{k:paired([r for r in records if select(r)],arm,'B',k) for k in KEYS}
        for arm in ['D','D_safe']} for name,select in [('old_bad4',lambda r:r['id'] in BAD),
            ('other24',lambda r:r['mask_valid'] and r['id'] not in BAD)]})
    decision(P2='pass_pending_visual' if allow_confirm else 'NO_GO',P3='not_run',
        continue_to_confirmation=allow_confirm,stop_reason=reason,
        positive_interaction_ci=effects['Interaction_LPIPS']['ci95'][0]>0,
        D_safe_confirmation_only_inconclusive=aux_inconclusive and not safe_aux,
        strict_unseen_generalization=False,paper_claim_supported=False)


def panels(rows,arms,folder,cell=128):
    folder.mkdir(parents=True,exist_ok=True);height=round(cell*4/3)
    per_page=4 if cell==128 else 1
    for start in range(0,len(rows),per_page):
        page=rows[start:start+per_page];canvas=Image.new('RGB',((3+len(arms))*cell,len(page)*(height+25)),'white')
        draw=ImageDraw.Draw(canvas)
        for y,row in enumerate(page):
            entries=[('Sketch',row['sketch']),('Reference',row['reference']),('GT',row['gt'])]+[(a,image_path(a,row['id'])) for a in arms]
            for x,(label,path) in enumerate(entries):
                draw.text((x*cell+2,y*(height+25)+2),row['id']+' '+label,fill='black')
                canvas.paste(Image.open(path).convert('RGB').resize((cell,height)),(x*cell,y*(height+25)+24))
        canvas.save(folder/('page%02d.png'%start))
