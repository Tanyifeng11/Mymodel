"""固定完整50步 RGB、参考反事实及配对门槛；无测试集调参。"""
import argparse
import csv
import gc
import hashlib
import numpy as np
import torch
from PIL import Image,ImageDraw
from tools.e44_protocol import *
from tools.e44_runtime import Runtime,sketch_masks
from tools.e44_metrics import contour_f1,highpass_error,calibration,paired


def generate(stage='p1',variant='B0'):
    assert read(OUT/'p0/smoke_report.json')['pass_p0']
    split='dev32' if stage=='p1' else 'confirm96'
    rows=read(OUT/'splits'/(split+'.json'));r=Runtime(variant)
    if variant!='B0':r.load(OUT/('p1' if stage=='p1' else 'p2')/variant/'checkpoints'/('step0800.pt' if stage=='p1' else 'step5000.pt'))
    try:
        for i,row in enumerate(rows):
            path=OUT/stage/'images'/variant/(row['id']+'.png')
            if path.with_suffix('.json').exists():assert read(path.with_suffix('.json'))['png_sha256']==sha(path)
            else:r.generate(row,path)
            print('E44 RGB',stage,variant,i+1,len(rows),flush=True)
    finally:r.close()


def counterfactual(variant):
    rows=read(OUT/'splits/dev32.json')[:8];r=Runtime(variant)
    if variant!='B0':r.load(OUT/'p1'/variant/'checkpoints/step0800.pt')
    records=[]
    try:
        for i,row in enumerate(rows):
            ref=Image.open(row['reference']).convert('RGB')
            hsv=np.array(ref.convert('HSV'));hsv[:,:,0]=(hsv[:,:,0].astype('int16')+85)%256
            color=Image.fromarray(hsv.astype('uint8'),'HSV').convert('RGB')
            donor=rows[(i+1)%len(rows)]
            variants=dict(Rplus=ref,R90=ref.transpose(Image.Transpose.ROTATE_90),
                Rminus=Image.open(donor['reference']).convert('RGB'),Rcolor=color,R0=ref)
            for name,image in variants.items():
                path=OUT/'p1/counterfactuals'/variant/row['id']/(name+'.png')
                if path.with_suffix('.json').exists():record=read(path.with_suffix('.json'))
                else:record,_,_=r.generate(row,path,reference=image,zero_reference=name=='R0')
                record.update(condition=name,donor_id=donor['id'] if name=='Rminus' else None,
                    different_motif_confirmed=None if name=='Rminus' else False,
                    r90_applicability='requires readable asymmetric motif; automatic selection not assumed',
                    rcolor_geometry_preservation='hue-only transform; gamut and brightness changes reported as limitation')
                records.append(record)
            write(OUT/'p1/counterfactuals'/variant/'manifest.json',records)
            print('E44 COUNTERFACTUAL',variant,i+1,len(rows),flush=True)
    finally:r.close()


@torch.inference_mode()
def evaluate(stage='p1'):
    from eval.eval_utils import prepare_evaluation_masks,estimate_foreground_mask,json_safe
    from eval.metrics import compute_structure_preservation,compute_texture_leakage,compute_texture_color_fidelity
    from transformers import CLIPConfig,CLIPModel,CLIPProcessor
    from tools.e33tmoc_appearance_eval import frozen_lpips
    rows=read(OUT/'splits'/('dev32.json' if stage=='p1' else 'confirm96.json'))
    folder=OUT/stage;records=[]
    metric,lpips_info=frozen_lpips();metric=metric.cuda()
    write(folder/'metric_calibration.json',calibration())
    clip_config=CLIPConfig.from_pretrained('models/clip/models/image_encoder',local_files_only=True)
    clip_model=CLIPModel.from_pretrained('models/clip',config=clip_config,local_files_only=True).cuda().eval().requires_grad_(False)
    clip_processor=CLIPProcessor.from_pretrained('models/clip/models/image_encoder',local_files_only=True)
    for variant in ['B0']+ARMS:
        group=[]
        for row in rows:
            path=folder/'images'/variant/(row['id']+'.png')
            a=Image.open(path).convert('RGB');b=Image.open(row['gt']).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR)
            tensor=lambda im:torch.from_numpy(np.asarray(im,np.float32).transpose(2,0,1)/127.5-1)[None].cuda()
            bundle=prepare_evaluation_masks(SIZE,sketch_path=row['sketch'],mask_policy='sketch_only')
            _,_,mask_info=sketch_masks(Image.open(row['sketch']).convert('RGB').resize(SIZE),'cpu')
            valid=mask_info['valid']
            rec=dict(id=row['id'],variant=variant,mask_valid=bool(valid),lpips_gt=float(metric(tensor(a),tensor(b)).mean()),
                iou=None,contour_f1=None,leak=None,lab=None,hp_gray=None)
            if valid:
                rec['iou']=json_safe(compute_structure_preservation(str(path),row['sketch'],mask_bundle=bundle)).get('struct_iou')
                foreground,_=estimate_foreground_mask(a,SIZE,return_info=True)
                rec['contour_f1']=contour_f1(foreground,bundle['garment']) if foreground is not None else None
                rec['leak']=json_safe(compute_texture_leakage(str(path),target_path=row['gt'],mask_bundle=bundle)).get('leak_colored_frac')
                rec['lab']=json_safe(compute_texture_color_fidelity(str(path),row['reference'],mask_bundle=bundle)).get('tcf_lab_delta')
                inner=np.asarray(bundle['garment']).astype('uint8')
                import cv2
                inner=cv2.erode(inner,np.ones((19,19),np.uint8))>0
                if inner.any():rec['hp_gray']=highpass_error(a,b,inner)
            ref=Image.open(row['reference']).convert('RGB')
            inputs=clip_processor(text=[row['caption']],images=[a,ref,b],return_tensors='pt',padding=True,truncation=True)
            inputs={k:v.to('cuda') for k,v in inputs.items()}
            text_f=clip_model.get_text_features(input_ids=inputs['input_ids'],attention_mask=inputs.get('attention_mask'))
            image_f=clip_model.get_image_features(pixel_values=inputs['pixel_values'])
            image_f=torch.nn.functional.normalize(image_f,dim=-1)
            text_f=torch.nn.functional.normalize(text_f,dim=-1)
            rec['clip_text']=float((text_f[0]*image_f[0]).sum())
            rec['clip_reference']=float((image_f[0]*image_f[1]).sum())
            rec['clip_real']=float((image_f[0]*image_f[2]).sum())
            group.append(rec)
        records.extend(group);write(folder/'metrics.json',records)
        print('E44 METRICS',variant,len(group),flush=True)
    write(folder/'metric_protocol.json',dict(lpips=lpips_info,fixed_n=len(rows),
        outer_contour=True,noncolor_metric='standardized gray highpass GT error; synthetic calibrated only',
        invalid_masks='NA regional metrics; retain full cohort and global LPIPS/CLIP',
        real_pattern_identity_validated=False,fid_kid='not computed at32/96'))
    with (folder/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    assess(stage,records);panels(stage,rows,['B0']+ARMS)
    base={r['id']:r for r in records if r['variant']=='B0'}
    by_id={r['id']:r for r in rows}
    for arm in ARMS:
        worst=sorted((r for r in records if r['variant']==arm),
            key=lambda r:r['lpips_gt']-base[r['id']]['lpips_gt'],reverse=True)[:10]
        write(folder/'worst_cases'/(arm+'.json'),worst)
        panels(stage,[by_id[r['id']] for r in worst],['B0',arm],subfolder='worst_cases/'+arm)


def assess(stage,records):
    keys=['lpips_gt','iou','contour_f1','leak','lab','hp_gray','clip_text','clip_reference','clip_real']
    deltas={a:{k:paired(records,a,'B0',k) for k in keys} for a in ARMS}
    counts={a:{k:sum(r['variant']==a and r.get(k) is not None for r in records) for k in keys} for a in ['B0']+ARMS}
    checks={};eligible=[];catastrophic={}
    base={r['id']:r for r in records if r['variant']=='B0'}
    for arm,d in deltas.items():
        def ci(key,index):return d[key]['ci95'][index] if d[key]['ci95'] is not None else None
        conditions=dict(iou=ci('iou',0) is not None and ci('iou',0)>=-.01,
            contour=ci('contour_f1',0) is not None and ci('contour_f1',0)>=-.02,
            leak=ci('leak',1) is not None and ci('leak',1)<=.005,
            lpips=d['lpips_gt']['mean'] is not None and d['lpips_gt']['mean']<=.01,
            text=d['clip_text']['mean'] is not None and d['clip_text']['mean']>=-.005,
            noncolor=ci('hp_gray',1) is not None and ci('hp_gray',1)<0,
            appearance=d['lpips_gt']['mean'] is not None and d['lpips_gt']['mean']<0)
        bad=[]
        for r in records:
            if r['variant']!=arm:continue
            for key,limit,sign in [('iou',.15,-1),('contour_f1',.15,-1),('leak',.10,1)]:
                if r.get(key) is not None and base[r['id']].get(key) is not None and sign*(r[key]-base[r['id']][key])>limit:
                    bad.append(dict(id=r['id'],metric=key,delta=r[key]-base[r['id']][key]))
        catastrophic[arm]=bad;conditions['no_catastrophic']=not bad;checks[arm]=conditions
        if all(conditions.values()):eligible.append(arm)
    combinations={a:{k:paired(records,'B3',a,k) for k in keys} for a in ['B1','B2']}
    independent=all(combinations[a][k]['mean'] is not None and combinations[a][k]['mean']<0 for a in ['B1','B2'] for k in ['lpips_gt','hp_gray'])
    allow_main='B3' in eligible and independent
    write(OUT/stage/'paired_deltas.json',deltas);write(OUT/stage/'valid_counts.json',counts)
    write(OUT/stage/'gates.json',dict(checks=checks,catastrophic=catastrophic,combination_comparisons=combinations))
    write(OUT/stage/'selection.json',dict(eligible=eligible,combination_advantage=independent,
        allow_main=allow_main,visual_review_required=True,real_pattern_semantics='not established by these proxies'))
    decision(**{stage.upper():'numerical_pass_pending_visual' if allow_main else 'NO_GO'},
        stop_reason=None if allow_main else 'no safe noncolor appearance gain with independent B3 advantage',
        paper_claim_supported=False)


def panels(stage,rows,arms,subfolder='contact_sheets'):
    folder=OUT/stage/subfolder;folder.mkdir(parents=True,exist_ok=True)
    for start in range(0,len(rows),4):
        canvas=Image.new('RGB',((3+len(arms))*128,4*194),'white');draw=ImageDraw.Draw(canvas)
        for i,row in enumerate(rows[start:start+4]):
            entries=[('Sketch',row['sketch']),('Reference',row['reference']),('GT',row['gt'])]
            entries +=[(a,str(OUT/stage/'images'/a/(row['id']+'.png'))) for a in arms]
            for j,(name,path) in enumerate(entries):
                draw.text((128*j+2,194*i+2),row['id']+' '+name,fill='black')
                canvas.paste(Image.open(path).convert('RGB').resize((128,171)),(128*j,194*i+20))
        canvas.save(folder/('page%02d.png'%(start//4)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['generate','evaluate','counterfactual'])
    p.add_argument('--variant',choices=['B0']+ARMS,default='B0');p.add_argument('--stage',default='p1');args=p.parse_args()
    if args.action=='generate':generate(args.stage,args.variant)
    elif args.action=='evaluate':evaluate(args.stage)
    else:counterfactual(args.variant)
