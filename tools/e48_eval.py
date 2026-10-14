"""e48 完整 RGB、公平能量对照和测量敏感性，不以 ε 代替终图结论。"""
import csv
import numpy as np
import cv2
import torch
from PIL import Image,ImageDraw
from tools.e48_protocol import *
from tools.e44_runtime import sketch_masks
from tools.e44_metrics import contour_f1,external_contour,highpass_error,paired
from tools.e45_eval import foreground_at
from tools.e35_freeu_protocol import bootstrap

KEYS=['lpips_gt','contour_f1','iou','leak','lab','hp_gray','clip_text','clip_reference','clip_real',
      'boundary_chamfer','boundary_hd95','area_relative_error']


def geometry(foreground,garment):
    a,b=external_contour(foreground),external_contour(garment)
    if not a.any() or not b.any():return dict(boundary_chamfer=None,boundary_hd95=None,area_relative_error=None)
    da=cv2.distanceTransform((~a).astype('uint8'),cv2.DIST_L2,5)
    db=cv2.distanceTransform((~b).astype('uint8'),cv2.DIST_L2,5)
    distances=np.concatenate([da[b],db[a]])
    return dict(boundary_chamfer=float(distances.mean()),boundary_hd95=float(np.percentile(distances,95)),
        area_relative_error=float((foreground.sum()-garment.sum())/max(garment.sum(),1)))


def panels(rows,arms,stage,folder,cell=128,overlays=False):
    folder.mkdir(parents=True,exist_ok=True);height=round(cell*4/3)
    for start in range(0,len(rows),4):
        page=rows[start:start+4];canvas=Image.new('RGB',((3+len(arms))*cell,len(page)*(height+25)),'white')
        draw=ImageDraw.Draw(canvas)
        for y,row in enumerate(page):
            entries=[('Sketch',row['sketch']),('Reference',row['reference']),('GT',row['gt'])]
            entries += [(a,OUT/stage/'masks'/row['id']/(a+'_overlay.png') if overlays
                else image_path(a,row['id'],stage)) for a in arms]
            for x,(label,path) in enumerate(entries):
                draw.text((x*cell+2,y*(height+25)+2),row['id']+' '+label,fill='black')
                canvas.paste(Image.open(path).convert('RGB').resize((cell,height)),(x*cell,y*(height+25)+24))
        canvas.save(folder/('page%02d.png'%start))


def assess(records,stage):
    arms=list(dict.fromkeys(r['variant'] for r in records));keys=KEYS
    means={a:{k:float(np.mean([r[k] for r in records if r['variant']==a and r[k] is not None]))
        for k in keys} for a in arms}
    counts={a:{k:sum(r['variant']==a and r[k] is not None for r in records) for k in keys} for a in arms}
    gates={};deltas={}
    base={r['id']:r for r in records if r['variant']=='B'}
    strong_gain=means['B']['lpips_gt']-means['D']['lpips_gt']
    for arm in arms:
        if arm=='B':continue
        delta={k:paired(records,arm,'B',k) for k in keys};deltas[arm]=delta
        extreme=[dict(id=r['id'],delta=r['contour_f1']-base[r['id']]['contour_f1']) for r in records
            if r['variant']==arm and r['contour_f1'] is not None and r['contour_f1']-base[r['id']]['contour_f1']<-.15]
        gain=means['B']['lpips_gt']-means[arm]['lpips_gt']
        checks=dict(f1=delta['contour_f1']['ci95'][0]>=-.020,iou=delta['iou']['ci95'][0]>=-.010,
            leak=delta['leak']['ci95'][1]<=.005,tail=not extreme,
            lpips=delta['lpips_gt']['ci95'][1]<0,retention=gain>=.50*strong_gain)
        gates[arm]=dict(checks=checks,passed=all(checks.values()),gain_retention=gain/strong_gain if strong_gain else None,
            extreme_events=extreme)
    comparisons={arm:{key:paired(records,arm,control,key) for key in keys}
        for arm,control in [('LATE','ENERGY_LATE'),('EPS_LOCK','ENERGY_EPS'),('LATENT_LOCK','SAFE')]
        if arm in arms and control in arms}
    write(OUT/stage/'means.json',means);write(OUT/stage/'valid_counts.json',counts)
    write(OUT/stage/'gates.json',gates);write(OUT/stage/'paired_deltas.json',deltas)
    write(OUT/stage/'control_comparisons.json',comparisons)
    promoted=[a for a in REPAIRS if a in gates and gates[a]['passed']]
    decision(**{stage:'complete'},candidate_passed=promoted,
        next='diagnosis-driven training or full dev32' if stage=='probe' else 'freeze one candidate only if all gates pass')


def mechanism_summary():
    groups={}
    for group,selected in [('known_bad',BAD),('controls',None)]:
        values=[]
        for row in read(OUT/'splits/probe.json'):
            if (row['id'] in BAD)!=(group=='known_bad'):continue
            item=read(image_path('D',row['id'],'audit').with_name(row['id']+'_intervention.json'))
            for s in item['steps']:
                if 'softmax' in s:values.append(s)
        groups[group]=dict(n_identities=7 if group=='known_bad' else 9,
            gate=float(np.mean([s['softmax']['gate'] for s in values])),
            p_ref_mean=float(np.mean([s['softmax']['p_ref_mean'] for s in values])),
            suppression_to_reference_rms=float(np.mean([s['softmax']['suppressed_rms']/max(s['softmax']['added_rms'],1e-12) for s in values])),
            component_cosine=float(np.mean([s['softmax']['component_cosine'] for s in values])),
            maximum_decomposition_error=max(s['softmax']['joint_fp32_max_error'] for s in values),
            boundary_epsilon_by_component={c:float(np.mean([s['component_counterfactuals'][c]['boundary']['rms'] for s in values]))
                for c in ['added','suppressed']})
    write(OUT/'mechanism_summary.json',dict(groups=groups,
        inference='fixed-weight same-state component intervention; does not prove unique cause',
        limitation='known bad identities were selected using e47; this is diagnostic development evidence'))


@torch.inference_mode()
def evaluate(stage,arms=None):
    from eval.eval_utils import prepare_evaluation_masks,estimate_foreground_mask,json_safe
    from eval.metrics import compute_texture_leakage,compute_texture_color_fidelity
    from transformers import CLIPConfig,CLIPModel,CLIPProcessor
    from tools.e33tmoc_appearance_eval import frozen_lpips
    from tools.e44_b1_mask import overlay
    arms=ARMS if arms is None else arms
    rows=read(OUT/'splits'/('probe.json' if stage=='probe' else 'dev32.json'))
    assert all(read(OUT/stage/(a+'_complete.json'))['n']==len(rows) for a in arms)
    metric,lpips_info=frozen_lpips();metric=metric.cuda()
    config=CLIPConfig.from_pretrained('models/clip/models/image_encoder',local_files_only=True)
    model=CLIPModel.from_pretrained('models/clip',config=config,local_files_only=True).cuda().eval().requires_grad_(False)
    processor=CLIPProcessor.from_pretrained('models/clip/models/image_encoder',local_files_only=True)
    tensor=lambda im:torch.from_numpy(np.asarray(im,np.float32).transpose(2,0,1)/127.5-1)[None].cuda()
    records=[];sensitivity=[];profiles=[]
    historic={(r['variant'],r['id']):r for s in ['p2','p3'] for r in read(PARENT/s/'metrics.json')}
    for row in rows:
        gt=Image.open(row['gt']).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR)
        ref=Image.open(row['reference']).convert('RGB')
        sketch=Image.open(row['sketch']).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR)
        raw,_,mask=sketch_masks(sketch,'cpu')
        bundle=prepare_evaluation_masks(SIZE,sketch_path=row['sketch'],mask_policy='sketch_only')
        garment=bundle['garment'];folder=OUT/stage/'masks'/row['id'];folder.mkdir(parents=True,exist_ok=True)
        if garment is not None:Image.fromarray(garment.astype('uint8')*255).save(folder/'sketch_mask.png')
        metadata=[read(image_path(a,row['id'],stage).with_suffix('.json')) for a in arms]
        assert len({m['initial_sha256'] for m in metadata})==1
        assert all(m['seed']==42 and not m['gt_used_in_inference'] for m in metadata)
        for arm in arms:
            path=image_path(arm,row['id'],stage);a=Image.open(path).convert('RGB')
            rec=dict(id=row['id'],variant=arm,mask_valid=bool(mask['valid']),**{k:None for k in KEYS})
            rec['lpips_gt']=float(metric(tensor(a),tensor(gt)).mean())
            fg=estimate_foreground_mask(a,SIZE)
            if fg is None:fg=np.zeros((SIZE[1],SIZE[0]),bool)
            Image.fromarray(fg.astype('uint8')*255).save(folder/(arm+'_foreground.png'))
            if garment is not None:overlay(a,fg,garment).save(folder/(arm+'_overlay.png'))
            if mask['valid']:
                rec['iou']=float((fg&garment).sum()/max((fg|garment).sum(),1))
                rec['contour_f1']=contour_f1(fg,garment)
                rec['leak']=json_safe(compute_texture_leakage(str(path),target_path=row['gt'],mask_bundle=bundle)).get('leak_colored_frac')
                rec['lab']=json_safe(compute_texture_color_fidelity(str(path),row['reference'],mask_bundle=bundle)).get('tcf_lab_delta')
                inner=cv2.erode(garment.astype('uint8'),np.ones((19,19),np.uint8))>0
                rec['hp_gray']=highpass_error(a,gt,inner) if inner.any() else None
                rec.update(geometry(fg,garment))
                profiles.append(dict(id=row['id'],variant=arm,
                    sketch_row_pixels=garment.sum(1).tolist(),foreground_row_pixels=fg.sum(1).tolist()))
                for distance in [12,18,24]:
                    for brightness in [230,235,240]:
                        f=foreground_at(a,distance,brightness)
                        if distance==18 and brightness==235:assert np.array_equal(f,fg)
                        sensitivity.append(dict(id=row['id'],variant=arm,distance=distance,brightness=brightness,
                            iou=float((f&garment).sum()/max((f|garment).sum(),1)),
                            **{'f1_%dpx'%t:contour_f1(f,garment,t) for t in [1,3,5,9]}))
            inputs=processor(text=[row['caption']],images=[a,ref,gt],return_tensors='pt',padding=True,truncation=True)
            inputs={k:v.cuda() for k,v in inputs.items()}
            text=F.normalize(model.get_text_features(input_ids=inputs['input_ids'],attention_mask=inputs.get('attention_mask')),dim=-1)
            features=F.normalize(model.get_image_features(pixel_values=inputs['pixel_values']),dim=-1)
            rec.update(clip_text=float((text[0]*features[0]).sum()),
                clip_reference=float((features[0]*features[1]).sum()),clip_real=float((features[0]*features[2]).sum()))
            if arm in ['B','D','SAFE']:
                old=historic[('D_safe' if arm=='SAFE' else arm,row['id'])]
                for k in KEYS[:9]:
                    assert (rec[k] is None)==(old[k] is None)
                    if rec[k] is not None:assert abs(rec[k]-old[k])<=2e-6,(arm,row['id'],k)
            records.append(rec)
        write(OUT/stage/'metrics.json',records)
        print('E48 METRICS',stage,row['id'],len(records),flush=True)
    with (OUT/stage/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    write(OUT/stage/'metric_protocol.json',dict(lpips=lpips_info,contour_tolerance_px=3,
        fixed_threshold_grid=dict(distance=[12,18,24],brightness=[230,235,240],tolerance=[1,3,5,9]),
        no_threshold_based_model_selection=True,motif_identity_validated=False))
    write(OUT/stage/'threshold_sensitivity.json',sensitivity);write(OUT/stage/'width_profiles.json',profiles)
    for label,group in [('time',['B','D','SAFE','EARLY','MIDDLE','LATE']),
        ('space',['B','D','SAFE','EPS_LOCK','ENERGY_EPS','LATENT_LOCK','ENERGY_LATE'])]:
        selected=[a for a in group if a in arms]
        panels(rows,selected,stage,OUT/stage/'panels'/label)
    base={r['id']:r for r in records if r['variant']=='B'};lookup={r['id']:r for r in rows}
    for arm in arms:
        if arm=='B':continue
        worst=sorted([r for r in records if r['variant']==arm and r['contour_f1'] is not None],
            key=lambda r:r['contour_f1']-base[r['id']]['contour_f1'])[:10]
        write(OUT/stage/'worst_cases'/(arm+'.json'),worst)
        panels([lookup[r['id']] for r in worst],['B','D',arm],stage,OUT/stage/'worst_cases'/arm)
        panels([lookup[r['id']] for r in worst],['B','D',arm],stage,OUT/stage/'overlays'/arm,overlays=True)
    assess(records,stage)
    if stage=='probe':mechanism_summary()
    write(OUT/stage/'evaluation_complete.json',dict(complete=True,n=len(rows),arms=arms,commit=commit()))


from torch.nn import functional as F
