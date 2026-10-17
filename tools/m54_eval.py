"""最终RGB指标与身份配对Bootstrap；辅助纹样身份不替代六门禁。"""
import csv
import cv2
from math import isfinite
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from tools.e44_metrics import contour_f1,external_contour,paired
from tools.e48_eval import geometry as boundary_geometry
from tools.m54_protocol import *

KEYS=['lpips_gt','contour_f1','iou','leak','boundary_chamfer','boundary_hd95','boundary_hausdorff',
      'boundary_displacement','texture_margin','texture_hp_margin','texture_chroma_margin','texture_vit_margin']


def crop(image,mask=None):
    a=np.asarray(image.convert('RGB'))
    if mask is None:mask=a.min(-1)<235
    yy,xx=np.where(mask)
    if len(xx):image=image.crop((int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1))
    return image.resize((224,224),Image.Resampling.BILINEAR)


def ncc(a,b):
    a=a.astype('float64').ravel();b=b.astype('float64').ravel();a-=a.mean();b-=b.mean()
    denominator=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/denominator) if denominator>1e-8 else 0.


def descriptor(image,model,processor):
    image=crop(image);a=np.asarray(image)
    gray=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY).astype('float32');hp=gray-cv2.GaussianBlur(gray,(0,0),2.)
    chroma=cv2.cvtColor(a,cv2.COLOR_RGB2LAB).astype('float32')[:,:,1:]-128
    inp=processor(images=image,return_tensors='pt')['pixel_values'].cuda()
    patches=model.vision_model(pixel_values=inp).last_hidden_state[:,1:].float()
    vector=F.normalize(F.normalize(patches,dim=-1).mean(1),dim=-1)[0].cpu().numpy()
    return hp,chroma,vector


def similarity(a,b):
    return [ncc(a[0],b[0]),ncc(a[1],b[1]),float(a[2]@b[2])]


def regional_metrics(image,path,gt_path,bundle,inference_mask_valid):
    from eval.eval_utils import estimate_foreground_mask,json_safe
    from eval.metrics import compute_texture_leakage
    garment=bundle['garment']
    assert garment is not None and garment.any(),'FIXED_SKETCH_EVALUATION_MASK_MISSING'
    fg=estimate_foreground_mask(image,SIZE)
    if fg is None:fg=np.zeros((SIZE[1],SIZE[0]),bool)
    # 推理增强的软mask门控不决定能否评分；所有身份用既定sketch_only评分区域。
    rec=dict(inference_mask_valid=bool(inference_mask_valid),evaluation_mask_source=bundle['stats']['mask_source'],
        evaluation_mask_pixels=int(garment.sum()),contour_f1=contour_f1(fg,garment),
        iou=float((fg&garment).sum()/max((fg|garment).sum(),1)),
        leak=json_safe(compute_texture_leakage(str(path),target_path=gt_path,mask_bundle=bundle)).get('leak_colored_frac'))
    rec.update({k:v for k,v in boundary_geometry(fg,garment).items() if k in KEYS})
    a,b=external_contour(fg),external_contour(garment)
    if a.any() and b.any():
        da=cv2.distanceTransform((~a).astype('uint8'),cv2.DIST_L2,5);db=cv2.distanceTransform((~b).astype('uint8'),cv2.DIST_L2,5)
        rec['boundary_hausdorff']=float(max(da[b].max(),db[a].max()))
        rec['boundary_displacement']=float(np.concatenate([da[b],db[a]]).mean())
    return rec


def assess(records,stage):
    arms=list(dict.fromkeys(r['variant'] for r in records));base={r['id']:r for r in records if r['variant']=='E5'}
    for arm in arms:
        group=[r for r in records if r['variant']==arm]
        assert len(group)==len(base) and {r['id'] for r in group}==set(base),'INCOMPLETE_EVALUATION_IDENTITIES'
        assert all(r.get(k) is not None and isfinite(r[k]) for r in group for k in ['lpips_gt','contour_f1','iou','leak']),('MISSING_OR_NONFINITE_CORE_METRIC',stage,arm)
    means={a:{k:float(np.mean([r[k] for r in records if r['variant']==a and r.get(k) is not None]))
        if any(r['variant']==a and r.get(k) is not None for r in records) else None for k in KEYS} for a in arms}
    counts={a:{k:sum(r['variant']==a and r.get(k) is not None for r in records) for k in KEYS} for a in arms}
    strong_gain=means['E5']['lpips_gt']-means['B1']['lpips_gt'];gates={};deltas={}
    for arm in arms:
        if arm=='E5':continue
        delta={k:paired(records,arm,'E5',k) for k in KEYS};deltas[arm]=delta
        versus_b1={k:paired(records,arm,'B1',k) for k in KEYS}
        extreme=[dict(id=r['id'],delta=r['contour_f1']-base[r['id']]['contour_f1']) for r in records
            if r['variant']==arm and r['contour_f1'] is not None and base[r['id']]['contour_f1'] is not None
            and r['contour_f1']-base[r['id']]['contour_f1']<-.15]
        gain=means['E5']['lpips_gt']-means[arm]['lpips_gt']
        complete=all(counts[a][k]==len(base) and delta[k]['n']==len(base)
            for a in [arm,'E5','B1'] for k in ['contour_f1','iou','leak','lpips_gt'])
        checks=dict(f1=complete and delta['contour_f1']['ci95'][0]>=-.020,
            iou=complete and delta['iou']['ci95'][0]>=-.010,leak=complete and delta['leak']['ci95'][1]<=.005,
            tail=complete and not extreme,lpips=complete and delta['lpips_gt']['ci95'][1]<0,
            retention=strong_gain>0 and gain>=.5*strong_gain)
        gates[arm]=dict(checks=checks,passed=all(checks.values()),gain_retention=gain/strong_gain if strong_gain>0 else None,
            paired_E5=delta,paired_B1=versus_b1,
            baseline_B1_gain=strong_gain,extreme_events=extreme,regional_metrics_complete=complete,
            pilot_clear_rejection=len(extreme)>=3,
            ambiguous_pilot_exception='nonpositive appearance/large structure drop needs preregistered numerical review; no post-hoc waiver')
    write(OUT/stage/'means.json',means);write(OUT/stage/'valid_counts.json',counts)
    write(OUT/stage/'gates.json',gates);write(OUT/stage/'paired_deltas.json',deltas)
    return gates


@torch.inference_mode()
def evaluate(stage):
    from eval.eval_utils import prepare_evaluation_masks
    from tools.e33tmoc_appearance_eval import frozen_lpips
    from transformers import CLIPConfig,CLIPModel,CLIPProcessor
    source=stage.split('_u')[0].split('_s')[0]
    rows=read(split_path(source))
    assert len({r['id'] for r in rows})==len(rows),'DUPLICATE_EVALUATION_ID'
    arms=[p.name[:-len('_complete.json')] for p in sorted((OUT/stage).glob('*_complete.json')) if p.name!='evaluation_complete.json']
    assert 'E5' in arms and 'B1' in arms
    for arm in arms:assert read(OUT/stage/(arm+'_complete.json'))['n']==len(rows)
    metric,lpips_info=frozen_lpips();metric=metric.cuda()
    config=CLIPConfig.from_pretrained('models/clip/models/image_encoder',local_files_only=True)
    model=CLIPModel.from_pretrained('models/clip',config=config,local_files_only=True).cuda().eval().requires_grad_(False)
    processor=CLIPProcessor.from_pretrained('models/clip/models/image_encoder',local_files_only=True)
    references={r['id']:descriptor(Image.open(r['reference']).convert('RGB'),model,processor) for r in rows}
    ordered=sorted(rows,key=lambda r:r['id'])
    wrong={row['id']:ordered[(i+1)%len(ordered)]['id'] for i,row in enumerate(ordered)}
    write(OUT/stage/'texture_metric_protocol.json',dict(crop='input foreground bounding box; unified224x224; no GT alignment',
        components=['grayscale HP sigma2 NCC','LAB chroma NCC','frozen CLIP ViT patch mean cosine'],
        aggregate='unweighted mean correct-minus-fixed-wrong similarity',wrong_reference_ids=wrong,
        selection='lexicographic cyclic offset1, fixed from inputs, not selected using generation',
        DINO_used=False,ViT_used=True,auxiliary_only=True,lpips=lpips_info))
    tensor=lambda im:torch.from_numpy(np.asarray(im,np.float32).transpose(2,0,1)/127.5-1)[None].cuda()
    records=[]
    profiles={a:read(OUT/stage/(a+'_profile.json')) for a in arms if (OUT/stage/(a+'_profile.json')).exists()}
    for row in rows:
        gt=Image.open(row['gt']).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR)
        bundle=prepare_evaluation_masks(SIZE,sketch_path=row['sketch'],mask_policy='sketch_only');garment=bundle['garment']
        metadata=[read(image_path(stage,a,row['id']).with_suffix('.json')) for a in arms]
        assert len({m['initial_sha256'] for m in metadata})==1
        for arm,meta in zip(arms,metadata):
            path=image_path(stage,arm,row['id']);assert sha(path)==meta['png_sha256']
            assert not meta['gt_used_in_inference'];image=Image.open(path).convert('RGB')
            rec=dict(id=row['id'],variant=arm,**{k:None for k in KEYS})
            rec.update(lpips_gt=float(metric(tensor(image),tensor(gt)).mean()),parameters=meta.get('parameters',0),
                delta_rms=meta.get('delta_rms',0.),epsilon_shift_rms=meta.get('epsilon_shift_rms',0.),
                reference_sensitivity_rms=meta.get('reference_sensitivity_rms'),diagnostic_latency_seconds=meta['seconds'],
                latency_seconds=float(np.median([x['seconds'] for x in profiles[arm]['records']])) if arm in profiles else None,
                peak_vram_gib=max(x['peak_memory_gib'] for x in profiles[arm]['records']) if arm in profiles else None,
                production_profile_n=profiles[arm]['n'] if arm in profiles else 0)
            effects=read(path.with_name(path.stem+'_effects.json'))
            for key in ['entropy','top1_mass','top5_mass','spatial_spread']:
                values=[x[key] for x in effects if key in x];rec[key]=float(np.mean(values)) if values else None
            rec.update(regional_metrics(image,path,row['gt'],bundle,meta['mask']['valid']))
            desc=descriptor(crop(image,garment) if garment is not None else image,model,processor)
            good=similarity(desc,references[row['id']]);bad=similarity(desc,references[wrong[row['id']]])
            margins=np.array(good)-np.array(bad)
            rec.update(texture_hp_margin=float(margins[0]),texture_chroma_margin=float(margins[1]),texture_vit_margin=float(margins[2]),texture_margin=float(margins.mean()))
            records.append(rec)
        write(OUT/stage/'per_image.json',records);print('M54 EVAL',stage,row['id'],flush=True)
    for filename in ['metrics.csv','per_image.csv']:
        with (OUT/stage/filename).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    gates=assess(records,stage)
    write(OUT/stage/'evaluation_complete.json',dict(complete=True,n=len(rows),arms=arms,passed=[a for a,g in gates.items() if g['passed']],commit=commit()))
    print('M54 EVALUATION COMPLETE',stage,flush=True)
