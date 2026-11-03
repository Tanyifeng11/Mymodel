"""同输入八配置统一评价；主比较四个未校准模型，Lab单独作补充。"""
import csv
from pathlib import Path
import numpy as np
from PIL import Image
from tools.bf_t1_t2 import ROOT,PARENT,HISTORY,COLOR,NEW,RAW,ARMS,core,read,write,sha,output
from tools import t3_color_metrics as color
from tools.bf_t3_three_arm_report import paired,gray
from tools.bf_t3_ablation_report import holm,table

PRIMARY=['edge_f1','iou','edge_l1','leak','lpips','tcf_lab']
RESPONSE=['dists_margin','gray_margin','gray_dists']
CONDITIONS=['original','neutral_original','neutral_swapped']


def image(path):return np.asarray(Image.open(path).convert('RGB').resize((384,512),Image.Resampling.LANCZOS))


def audit_training():
    config=read(ROOT/'protocol.json')['training'];parent=read(ROOT/'audit/parent_training.json')
    lr0=[r['lr'] for r in csv.DictReader((HISTORY/'training/BF-SD/training_log.csv').open())]
    records=[]
    for arm in RAW:
        new=arm in NEW
        complete=read(ROOT/'training'/arm/'complete.json') if new else parent[arm]
        assert complete['updates']==28365 and complete['world']==2 and complete['initial_sha256']==config['initial_sha256']
        assert complete['stream_hashes']==parent['BF-SD']['stream_hashes']
        if new:
            rows=list(csv.DictReader((ROOT/'training'/arm/'training_log.csv').open()))
            assert len(rows)==len(lr0)==28365 and [r['lr'] for r in rows]==lr0
            assert [int(r['step']) for r in rows]==list(range(1,28366))
            assert all(np.isfinite(float(v)) for r in rows for v in r.values())
            assert all(abs(float(r['total'])-float(r['base'])-float(r['weighted_aux']))<1e-5 for r in rows)
            for rank in range(2):
                proof=read(ROOT/'calibration'/arm/('rank%d.json'%rank));assert proof['passed'] and proof['optimizer_updates']==0 and proof['base_exact']
        records.append(dict(arm=arm,new_training=new,trainable_parameters=complete['trainable_parameters'],
            initial_sha256=complete['initial_sha256'],checkpoint_sha256=complete['checkpoint_sha256'],
            updates=28365,seed=42,world=2,effective_batch=8,peak_lr=5e-5,warmup=500,schedule='cosine',
            cfg=7.,steps=50,width=384,height=512,sketch_scale=.6,texture_tokens=False,
            auxiliary=str(complete.get('coefficients',read(PARENT/'protocol.json')['coefficients'].get(arm))),
            eta=249 if new else None,training_seconds=complete['seconds']))
    table(ROOT/'metrics/model_parameters.csv',records)
    write(ROOT/'audit/final_comparability.json',dict(passed=True,same_initialization=True,same_random_stream=True,same_lr=True,
        calibration_without_updates=True,raw_models=RAW,parameters=records))


def process():
    from eval.eval_utils import prepare_evaluation_masks
    from eval.metrics import compute_structure_preservation,compute_texture_leakage,compute_texture_color_fidelity
    audit_training();rows=read(ROOT/'evaluation.json');masks_info=read(ROOT/'audit/masks.json');cache={};records=[]
    original={r['id']:r for r in rows if r['condition']=='neutral_original'}
    def ref(row):
        k=row['reference_sha256']
        if k not in cache:cache[k]=color.descriptor(np.asarray(Image.open(row['reference']).convert('RGB')))
        return cache[k]
    for row in rows:
        reference=np.asarray(Image.open(row['reference']).convert('RGB'));soft=np.load(ROOT/'masks'/(row['id']+'.npy'))
        valid=masks_info[row['id']]['valid']
        for arm in RAW:
            source=output(arm,row);rgb=image(source);edited=color.calibrate(rgb,reference,soft,valid)
            assert np.array_equal(edited[soft==0],rgb[soft==0])
            if not valid:assert np.array_equal(edited,rgb)
            dest=output(arm+'_LAB',row)
            if dest.exists():assert np.array_equal(image(dest),edited),'复用Lab结果不一致'
            else:
                dest.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(edited).save(dest)
                write(dest.with_suffix('.json'),dict(read(source.with_suffix('.json')),arm=arm+'_LAB',
                    output_sha256=sha(dest),calibrated_from_sha256=sha(source),lab_valid=valid,outside_exact=True))
        bundle=prepare_evaluation_masks((384,512),sketch_path=row['sketch'],mask_policy='sketch_only')
        inside,outside=color.regions(bundle['garment'])
        for arm in ARMS:
            path=output(arm,row);rgb=image(path);desc=color.descriptor(rgb,inside)
            kwargs=dict(gen_path=str(path),sketch_path=row['sketch'],target_path=row['gt'],mask_bundle=bundle)
            structure=compute_structure_preservation(**kwargs)
            rec=dict(arm=arm,id=row['id'],condition=row['condition'],category=row['category'],mask_valid=valid,
                color_pixels=int(inside.sum()),edge_f1=structure.get('edge_f1'),iou=structure.get('sketch_iou'),edge_l1=structure.get('edge_l1'),
                leak=compute_texture_leakage(**kwargs).get('leak_colored_frac'),
                tcf_lab=compute_texture_color_fidelity(texture_path=row['reference'],**kwargs).get('tcf_lab_delta'),
                **color.compare(desc,ref(row),'ref_'),**color.outside_metrics(rgb,outside),
                ref_ab_margin=None,ref_rgbuv_margin=None)
            if row['condition']=='neutral_swapped':
                old=color.compare(desc,ref(original[row['id']]),'old_')
                if rec['ref_ab_w1'] is not None:rec['ref_ab_margin']=old['old_ab_w1']-rec['ref_ab_w1'];rec['ref_rgbuv_margin']=old['old_rgbuv_h']-rec['ref_rgbuv_h']
            records.append(rec)
        print('PROCESSED',row['condition'],row['id'],flush=True)
    write(ROOT/'metrics/cpu_per_image.json',records)


def evaluate():
    import torch
    from eval.eval_utils import prepare_evaluation_masks
    from tools.e2a_attr_eval import frozen_lpips
    from tools import t3_response_run as response
    from tools.t3_response_common import patches
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    response.ROOT=ROOT;dists=response.load_dists('cuda');lpips,assets=frozen_lpips();write(ROOT/'audit/lpips_assets.json',assets)
    rows=read(ROOT/'evaluation.json');records=read(ROOT/'metrics/cpu_per_image.json')
    index={(r['arm'],r['condition'],r['id']):r for r in records}
    original={r['id']:r for r in rows if r['condition']=='neutral_original'};cache={};failures=[]
    def features(parts):return dists.forward_once(torch.from_numpy(np.stack(parts).transpose(0,3,1,2).copy()).float().cuda()/255)
    def ref(path,is_gray):
        k=(path,is_gray)
        if k not in cache:
            im=image(path);parts=[im[y:y+64,x:x+64] for y in np.linspace(0,448,4).round().astype(int) for x in np.linspace(0,320,4).round().astype(int)]
            cache[k]=features([gray(p) for p in parts] if is_gray else parts)
        return cache[k]
    def distance(a,b):
        m=response.distances(dists,a,b);return float(.5*(m.min(0).values.mean()+m.min(1).values.mean()))
    with torch.inference_mode():
        for row in rows:
            masks=prepare_evaluation_masks((384,512),sketch_path=row['sketch'],mask_policy='sketch_only');coords,candidates=patches(masks['garment']);latents=[]
            for arm in ARMS:
                path=output(arm,row);meta=read(path.with_suffix('.json'))
                assert sha(path)==meta['output_sha256'] and meta['seed']==row['seed'] and meta['caption']==row['caption']
                assert meta['cfg']==7 and meta['steps']==50 and not meta['GT_used_in_inference']
                for key in ['sketch','reference']:assert meta['input_'+key+'_sha256']==row[key+'_sha256']
                raw_arm=arm.removesuffix('_LAB')
                expected=read(ROOT/'training'/raw_arm/'complete.json')['checkpoint_sha256'] if raw_arm in NEW else read(ROOT/'audit/parent_training.json')[raw_arm]['checkpoint_sha256']
                assert meta['checkpoint_sha256']==expected;latents.append(meta['initial_latent_sha256'])
                rec=index[arm,row['condition'],row['id']];rec.update(lpips=None,gray_dists=None,gray_margin=None,dists_margin=None,patch_n=len(coords))
                if row['condition']=='original':rec['lpips']=float(lpips(core.tensor(path).cuda(),core.tensor(row['gt']).cuda()).mean())
                if row['condition']!='original' and len(coords)>=4:
                    im=image(path);parts=[im[y:y+64,x:x+64] for x,y in coords]
                    for is_gray in [False,True]:
                        generated=features([gray(p) for p in parts] if is_gray else parts);new=distance(generated,ref(row['reference'],is_gray))
                        if is_gray:rec['gray_dists']=new
                        if row['condition']=='neutral_swapped':rec['gray_margin' if is_gray else 'dists_margin']=distance(generated,ref(original[row['id']]['reference'],is_gray))-new
                elif row['condition']!='original':failures.append(dict(arm=arm,id=row['id'],condition=row['condition'],reason='fewer_than_four_fixed64px_patches'))
                if row['condition']=='neutral_swapped':
                    prior=read((ROOT/'outputs'/arm/'neutral_original'/(row['id']+'.json')))
                    assert prior['initial_latent_sha256']==meta['initial_latent_sha256']
            assert len(set(latents))==1
            print('EVALUATED',row['condition'],row['id'],flush=True)
    assert len(records)==1536
    write(ROOT/'metrics/per_image.json',records);table(ROOT/'metrics/per_image.csv',records);write(ROOT/'audit/metric_failures.json',failures)
    summarize(ROOT)
    write(ROOT/'evaluation_complete.json',dict(complete=True,outputs=1536,new_training=NEW,conditions=192,
        main_raw_cfg7=True,lab_secondary_same_algorithm=True,independent_confirmation=False))


def summarize(root):
    records=read(root/'metrics/per_image.json');index={(r['arm'],r['condition'],r['id']):r for r in records}
    metrics=[k for k in records[0] if k not in ['arm','id','condition','category','mask_valid','color_pixels','patch_n']]
    means=[]
    for c in CONDITIONS:
        for a in ARMS:
            part=[r for r in records if r['condition']==c and r['arm']==a]
            for k in metrics:
                values=[r[k] for r in part if r[k] is not None]
                means.append(dict(arm=a,condition=c,metric=k,n=len(values),planned_n=len(part),mean=float(np.mean(values)) if values else None))
    result=[]
    families=[('raw_primary12','original',PRIMARY,[(RAW[-1],a) for a in NEW]),
        ('raw_response6','neutral_swapped',RESPONSE,[(RAW[-1],a) for a in NEW]),
        ('lab_primary12','original',PRIMARY,[(RAW[-1]+'_LAB',a+'_LAB') for a in NEW]),
        ('lab_response6','neutral_swapped',RESPONSE,[(RAW[-1]+'_LAB',a+'_LAB') for a in NEW]),
        ('adapted_minus_base12','original',PRIMARY,[(a,RAW[0]) for a in NEW])]
    for c in CONDITIONS:
        families.append(('lab_color8_'+c,c,['ref_rgbuv_h','ref_ab_w1'],[(a+'_LAB',a) for a in RAW]))
    for family,c,keys,comparisons in families:
        part=[];ids=sorted({r['id'] for r in records if r['condition']==c})
        for a,b in comparisons:
            for k in keys:
                values=[index[a,c,i][k]-index[b,c,i][k] for i in ids if index[a,c,i][k] is not None and index[b,c,i][k] is not None]
                part.append(dict(comparison=a+' minus '+b,condition=c,metric=k,family=family,**paired(values)))
        holm(part);result+=part
    write(root/'metrics/summary.json',means);table(root/'metrics/summary.csv',means)
    write(root/'metrics/paired_statistics.json',result)
    table(root/'metrics/paired_statistics.csv',[dict(r,ci95=str(r['ci95'])) for r in result])
