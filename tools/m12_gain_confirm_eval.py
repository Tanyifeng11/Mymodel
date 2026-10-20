"""六臂全部生成后统一评分；正式 GPU LPIPS 与预定 CPU 敏感性分开。"""
import platform
from math import isfinite
import numpy as np
from PIL import Image
from tools.m12_gain_confirm_protocol import *
from tools.m12_protocol import write_csv


def save_statistics(records,prefix='metrics'):
    from tools.m54_eval import KEYS
    identities=[r['id'] for r in rows()]
    result=statistics(records,identities,KEYS)
    root=RUN/prefix
    for name,key in [('per_identity_metrics','per_identity'),('means','means'),('paired_deltas','paired'),('gate_results','gates'),('per_seed_summary','per_seed')]:
        write(root/(name+'.json'),result[key])
    write_csv(root/'per_identity_metrics.csv',result['per_identity'])
    write(root/'valid_counts.json',{arm:{k:dict(valid=result['means'][arm][k]['n'],NA=96-result['means'][arm][k]['n']) for k in KEYS} for arm in ARMS})
    worst=[]
    indexed={(r['variant'],r['id']):r for r in result['per_identity']}
    for identity in identities:
        a,b=indexed['M12_GAIN',identity],indexed['E5',identity]
        worst.append(dict(id=identity,delta_f1=a['contour_f1']-b['contour_f1'] if a['contour_f1'] is not None and b['contour_f1'] is not None else None,
            delta_lpips=a['lpips_gt']-b['lpips_gt']))
    write_csv(root/'worst_cases.csv',sorted(worst,key=lambda r:r['delta_f1'] if r['delta_f1'] is not None else float('inf')))
    key=result['paired']['B1']['contour_f1'];main=result['gates']['M12_GAIN']
    incremental=result['paired']['M10_GAIN']
    safety=dict(f1=incremental['contour_f1']['n']==96 and incremental['contour_f1']['ci95'][0]>=-.020,
        iou=incremental['iou']['n']==96 and incremental['iou']['ci95'][0]>=-.010,
        leak=incremental['leak']['n']==96 and incremental['leak']['ci95'][1]<=.005)
    seed_directions={str(s):dict(lpips_vs_E5=result['per_seed'][str(s)]['M12_GAIN']['lpips_gt']['mean']-result['per_seed'][str(s)]['E5']['lpips_gt']['mean'],
        f1_vs_B1=result['per_seed'][str(s)]['M12_GAIN']['contour_f1']['mean']-result['per_seed'][str(s)]['B1']['contour_f1']['mean']) for s in SEEDS}
    robust=all(r['lpips_vs_E5']<0 and r['f1_vs_B1']>0 for r in seed_directions.values())
    b1=key['n']==96 and key['ci95'][0]>0
    benefits=main['checks']['lpips'] or b1
    outcome='GO' if main['passed'] and b1 and robust else 'CONDITIONAL' if benefits else 'NO_GO'
    differences=[indexed['M12_GAIN',i]['lpips_gt']-indexed['E5',i]['lpips_gt'] for i in identities]
    influence=dict(leave_one_out_lpips_mean_range=[min((sum(differences)-v)/95 for v in differences),max((sum(differences)-v)/95 for v in differences)],
        largest_five_absolute_identity_deltas=sorted(zip(identities,differences),key=lambda r:-abs(r[1]))[:5])
    write(root/'decision.json',dict(outcome=outcome,primary_six_gates=main['passed'],b1_structure_replicated=b1,
        all_seed_directions_consistent=robust,seed_directions=seed_directions,influence=influence,
        incremental_lpips_95_signal=incremental['lpips_gt']['ci95'][1]<0,incremental_descriptive_safety=safety,
        incremental_comparison_status='EXPLORATORY_NO_MULTIPLICITY_ADJUSTED_SUPERIORITY_CLAIM',
        true_pattern_ground_truth='NOT_RUN',blind_human_review='NOT_RUN',new_training=0))
    return result


def evaluate():
    import torch
    from tools.m12_gain_confirm import require,verify_outputs
    from tools.m54_eval import KEYS,descriptor,crop,similarity,regional_metrics
    from eval.eval_utils import prepare_evaluation_masks
    from tools.e33tmoc_appearance_eval import frozen_lpips
    from transformers import CLIPConfig,CLIPModel,CLIPProcessor
    require();verify_outputs()
    assert not (RUN/'metrics/complete.json').exists(),'EVALUATION_ALREADY_COMPLETE'
    metric,info=frozen_lpips();metric=metric.cuda()
    config=CLIPConfig.from_pretrained('models/clip/models/image_encoder',local_files_only=True)
    model=CLIPModel.from_pretrained('models/clip',config=config,local_files_only=True).cuda().eval().requires_grad_(False)
    processor=CLIPProcessor.from_pretrained('models/clip/models/image_encoder',local_files_only=True)
    selected=rows();ordered=sorted(selected,key=lambda r:r['id'])
    wrong={r['id']:ordered[(i+1)%len(ordered)]['id'] for i,r in enumerate(ordered)}
    tensor=lambda im:torch.from_numpy(np.asarray(im,np.float32).transpose(2,0,1)/127.5-1)[None].cuda()
    records=[];validity=[]
    with torch.inference_mode():
        references={r['id']:descriptor(Image.open(r['reference']).convert('RGB'),model,processor) for r in selected}
        for row in selected:
            gt=Image.open(row['gt']).convert('RGB').resize((384,512),Image.Resampling.BILINEAR)
            bundle=prepare_evaluation_masks((384,512),sketch_path=row['sketch'],mask_policy='sketch_only');garment=bundle['garment']
            for seed in SEEDS:
                for arm in ARMS:
                    path=image(arm,row['id'],seed);meta=read(path.with_suffix('.json'));rgb=Image.open(path).convert('RGB')
                    rec=dict(id=row['id'],seed=seed,variant=arm,**{key:None for key in KEYS})
                    rec.update(lpips_gt=float(metric(tensor(rgb),tensor(gt)).mean()),diagnostic_seconds=meta['seconds'],
                        inference_gate_valid=meta['mask']['valid'],na_reason=None)
                    if garment is not None and garment.any():
                        rec.update(regional_metrics(rgb,path,row['gt'],bundle,meta['mask']['valid']))
                    else:rec['na_reason']='original_sketch_only_evaluation_mask_unavailable'
                    desc=descriptor(crop(rgb,garment) if garment is not None else rgb,model,processor)
                    margins=np.asarray(similarity(desc,references[row['id']]))-similarity(desc,references[wrong[row['id']]])
                    rec.update(texture_hp_margin=float(margins[0]),texture_chroma_margin=float(margins[1]),texture_vit_margin=float(margins[2]),texture_margin=float(margins.mean()))
                    for key in KEYS:
                        if rec.get(key) is not None and not isfinite(rec[key]):rec[key]=None;rec['na_reason']='nonfinite_'+key
                    records.append(rec)
            validity.append(dict(id=row['id'],inference_gate_valid=meta['mask']['valid'],evaluation_mask_valid=bool(garment is not None and garment.any()),
                evaluation_mask_source=bundle['stats']['mask_source']))
            write(RUN/'metrics/per_image.json',records)
            print('CONFIRM_EVAL',row['id'],len(records),flush=True)
    assert len(records)==1728
    write_csv(RUN/'metrics/per_image_metrics.csv',records);write_csv(RUN/'metrics/mask_validity.csv',validity)
    write(RUN/'metrics/texture_metric_protocol.json',dict(auxiliary_only=True,ground_truth=False,wrong_reference_ids=wrong,
        components=['grayscale HP sigma2 NCC','LAB chroma NCC','frozen CLIP ViT patch mean cosine'],lpips=info))
    save_statistics(records)
    write(RUN/'metrics/complete.json',dict(complete=True,images=len(records),identities=96,seeds=SEEDS,commit=commit(),formal_lpips_device='GPU'))
    print('CONFIRM_EVALUATION_COMPLETE',flush=True)


def cpu_lpips():
    import torch
    from tools.e33tmoc_appearance_eval import frozen_lpips
    from tools.m12_gain_confirm import require
    require();assert read(RUN/'metrics/complete.json')['complete']
    metric,info=frozen_lpips();metric=metric.cpu();records=read(RUN/'metrics/per_image.json');index={(r['variant'],r['id'],r['seed']):r for r in records}
    results=[];sensitivity=[]
    tensor=lambda im:torch.from_numpy(np.asarray(im,np.float32).transpose(2,0,1)/127.5-1)[None]
    with torch.inference_mode():
        for row in rows():
            gt=Image.open(row['gt']).convert('RGB').resize((384,512),Image.Resampling.BILINEAR)
            resized_hash=hashlib.sha256(gt.tobytes()).hexdigest()
            for seed in SEEDS:
                for arm in ARMS:
                    rgb=Image.open(image(arm,row['id'],seed)).convert('RGB');value=float(metric(tensor(rgb),tensor(gt)).mean())
                    original=index[arm,row['id'],seed];diff=abs(value-original['lpips_gt'])
                    results.append(dict(id=row['id'],seed=seed,variant=arm,lpips_cpu=value,lpips_gpu=original['lpips_gt'],absolute_difference=diff,resized_gt_rgb_sha256=resized_hash))
                    sensitivity.append(dict(original,lpips_gt=value))
            print('CONFIRM_CPU_LPIPS',row['id'],len(results),flush=True)
    write(RUN/'precision/server_cpu_lpips.json',results)
    write(RUN/'precision/comparison.json',dict(atol=1e-6,maximum_absolute_difference=max(r['absolute_difference'] for r in results),
        strict_pass=all(r['absolute_difference']<=1e-6 for r in results),formal_values_replaced=False,
        torch=torch.__version__,python=platform.python_version(),lpips=info,threads=torch.get_num_threads()))
    save_statistics(sensitivity,'precision/cpu_sensitivity')
    write(RUN/'precision/complete.json',dict(complete=True,n=len(results),formal_device='GPU',sensitivity_device='CPU'))
    print('CONFIRM_CPU_LPIPS_COMPLETE',flush=True)
