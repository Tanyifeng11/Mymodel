"""同输入统一评价；颜色、感知参考响应和结构分别报告，不合成总分。"""
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from tools.bf_t3_three_arm import ROOT,PROJECT,ARMS,read,write,sha,tensor


def table(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def paired(values):
    x=np.asarray(values,float);x=x[np.isfinite(x)]
    if not len(x):return dict(n=0,mean=None,ci95=None,p=None)
    rng=np.random.default_rng(110142)
    bootstrap=x[rng.integers(len(x),size=(10000,len(x)))].mean(1)
    permutations=(x*rng.choice([-1,1],size=(10000,len(x)))).mean(1)
    return dict(n=len(x),mean=float(x.mean()),ci95=np.quantile(bootstrap,[.025,.975]).tolist(),
                p=float((1+(np.abs(permutations)>=abs(x.mean())).sum())/10001))


def gray(patch):
    a=np.asarray(patch,float)/255
    y=a@np.array([.2126,.7152,.0722])
    y=np.clip((y-y.mean())/max(y.std(),2/255)*.18+.5,0,1)
    return np.repeat(np.round(y[...,None]*255).astype('uint8'),3,axis=-1)


def evaluate():
    import torch
    from eval.eval_utils import prepare_evaluation_masks
    from eval.metrics import compute_structure_preservation,compute_texture_color_fidelity,compute_texture_leakage
    from tools.e2a_attr_eval import frozen_lpips
    from tools import t3_response_run as response
    from tools.t3_response_common import patches
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    response.ROOT=ROOT
    dists=response.load_dists('cuda')
    lpips,assets=frozen_lpips()
    rows=read(ROOT/'evaluation.json')
    original={r['id']:r for r in rows if r['condition']=='original'}
    reference_cache={}
    def features(parts):
        value=torch.from_numpy(np.stack(parts).transpose(0,3,1,2).copy()).float().cuda()/255
        return dists.forward_once(value)
    def ref(path,is_gray):
        key=(path,is_gray)
        if key not in reference_cache:
            image=np.asarray(Image.open(path).convert('RGB').resize((384,512),Image.Resampling.LANCZOS))
            parts=[image[y:y+64,x:x+64] for y in np.linspace(0,448,4).round().astype(int)
                   for x in np.linspace(0,320,4).round().astype(int)]
            reference_cache[key]=features([gray(p) for p in parts] if is_gray else parts)
        return reference_cache[key]
    def distance(a,b):
        matrix=response.distances(dists,a,b)
        return float(.5*(matrix.min(0).values.mean()+matrix.min(1).values.mean()))
    records=[]
    with torch.inference_mode():
        for row in rows:
            masks=prepare_evaluation_masks((384,512),sketch_path=row['sketch'],mask_policy='sketch_only')
            coords,_=patches(masks['garment'])
            latent=[]
            for arm in ARMS:
                path=ROOT/'outputs'/arm/row['condition']/(row['id']+'.png')
                meta=read(path.with_suffix('.json'))
                assert sha(path)==meta['output_sha256']
                if arm!='BF-Fashion': latent.append(meta['initial_latent_sha256'])
                kwargs=dict(gen_path=str(path),sketch_path=row['sketch'],target_path=row['gt'],mask_bundle=masks)
                s=compute_structure_preservation(**kwargs)
                record=dict(arm=arm,id=row['id'],split=row['split'],condition=row['condition'],
                    edge_f1=s.get('edge_f1'),iou=s.get('sketch_iou'),edge_l1=s.get('edge_l1'),
                    leak=compute_texture_leakage(**kwargs).get('leak_colored_frac'),
                    lpips=None,tcf_lab=None,dists_margin=None,gray_margin=None,gray_dists=None,patch_n=len(coords))
                if row['condition']=='original':
                    record['lpips']=float(lpips(tensor(path).cuda(),tensor(row['gt']).cuda()).mean())
                    record['tcf_lab']=compute_texture_color_fidelity(texture_path=row['reference'],**kwargs).get('tcf_lab_delta')
                if row['condition']=='swapped' and len(coords)>=4:
                    image=np.asarray(Image.open(path).convert('RGB'))
                    parts=[image[y:y+64,x:x+64] for x,y in coords]
                    for is_gray in [False,True]:
                        generated=features([gray(p) for p in parts] if is_gray else parts)
                        old_distance=distance(generated,ref(original[row['id']]['reference'],is_gray))
                        new_distance=distance(generated,ref(row['reference'],is_gray))
                        record['gray_margin' if is_gray else 'dists_margin']=old_distance-new_distance
                        if is_gray:record['gray_dists']=new_distance
                records.append(record)
            assert len(set(latent))==1
            print('EVALUATED',row['condition'],row['id'],flush=True)
    # 保存所有记录，包括无有效内部patch的身份。
    safe=[{k:(None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in r.items()} for r in records]
    write(ROOT/'metrics/per_image.json',safe);table(ROOT/'metrics/per_image.csv',safe)
    metrics=['edge_f1','iou','edge_l1','leak','lpips','tcf_lab','dists_margin','gray_margin','gray_dists']
    summaries=[]
    for arm in ARMS:
        for split,condition in [('heldout128','original'),('swap32','original'),('swap32','swapped')]:
            part=[r for r in safe if r['arm']==arm and r['split']==split and r['condition']==condition]
            for metric in metrics:
                values=[r[metric] for r in part if r[metric] is not None]
                summaries.append(dict(arm=arm,split=split,condition=condition,metric=metric,n=len(values),planned_n=len(part),
                                      mean=float(np.mean(values)) if values else None))
    write(ROOT/'metrics/summary.json',summaries);table(ROOT/'metrics/summary.csv',summaries)
    lookup={(r['arm'],r['id'],r['condition']):r for r in safe}
    comparisons=[]
    for other in ['BF-SD','BF-Fashion']:
        for metric in metrics:
            condition='swapped' if metric in ['dists_margin','gray_margin','gray_dists'] else 'original'
            selected=[r for r in rows if r['condition']==condition and (condition=='swapped' or r['split']=='heldout128')]
            values=[]
            for row in selected:
                a=lookup['BF-SD-T3',row['id'],condition][metric]
                b=lookup[other,row['id'],condition][metric]
                if a is not None and b is not None:values.append(a-b)
            comparisons.append(dict(comparison='BF-SD-T3 minus '+other,metric=metric,condition=condition,**paired(values)))
    for other in ['BF-SD','BF-Fashion']:
        family=sorted([r for r in comparisons if r['comparison'].endswith(' '+other) and r['p'] is not None],key=lambda r:r['p'])
        running=0.
        for index,row in enumerate(family):
            running=max(running,min(1.,row['p']*(len(family)-index)));row['p_holm']=running
    write(ROOT/'metrics/paired_statistics.json',comparisons)
    table(ROOT/'metrics/paired_statistics.csv',[dict(r,ci95=json.dumps(r['ci95']),p_holm=r.get('p_holm')) for r in comparisons])
    training={arm:read(ROOT/'training'/arm/'complete.json') for arm in ARMS[1:]}
    a,b=training.values()
    assert a['updates']==b['updates']==28365 and a['initial_sha256']==b['initial_sha256']
    assert a['stream_hashes']==b['stream_hashes'] and a['world']==b['world']
    write(ROOT/'audit/final_comparability.json',dict(passed=True,same_initialization=True,same_training_stream=True,
        updates=28365,world=a['world'],lpips_assets=assets,training=training,
        interpretation='T3 effect is only BF-SD-T3 versus BF-SD; native BF comparison remains descriptive'))
    chosen=sorted([r for r in rows if r['split']=='heldout128'],key=lambda r:sha(r['sketch']))[:4]
    chosen+=sorted([r for r in rows if r['condition']=='swapped'],key=lambda r:sha(r['sketch']))[:4]
    for page,start in enumerate(range(0,len(chosen),4),1):
        panel=Image.new('RGB',(1080,260*min(4,len(chosen)-start)),'white');draw=ImageDraw.Draw(panel)
        for j,row in enumerate(chosen[start:start+4]):
            paths=[row['sketch'],row['reference'],row['gt'] if row['condition']=='original' else None]
            paths += [ROOT/'outputs'/arm/row['condition']/(row['id']+'.png') for arm in ARMS]
            for col,(label,path) in enumerate(zip(['Sketch','Reference','GT (original only)']+ARMS,paths)):
                draw.text((col*180+3,j*260+2),label,fill='black')
                if path:
                    image=Image.open(path).convert('RGB');image.thumbnail((174,226),Image.Resampling.LANCZOS)
                    panel.paste(image,(col*180,j*260+22))
            draw.text((3,j*260+246),row['id']+' '+row['condition'],fill='black')
        (ROOT/'figures').mkdir(exist_ok=True);panel.save(ROOT/'figures'/('fixed_input_hash_%02d.png'%page))
    write(ROOT/'evaluation_complete.json',dict(outputs=len(safe),identities=160,
          training_updates_per_new_arm=28365,independent_confirmation=False,complete=True))


if __name__=='__main__':evaluate()
