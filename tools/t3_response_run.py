"""按冻结顺序复核开发证据、启用独立身份并生成参考替换结果。"""
import argparse
import csv
import datetime
import importlib.util
import math
import os
import shutil
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from tools.t3_response_common import *


def array(path):
    return np.asarray(Image.open(path).convert('RGB'))


def load_dists(device='cpu'):
    import torch
    source=SUP/'assets/DISTS/DISTS_pytorch/DISTS_pt.py'
    assert sha(source)==CONFIG['dists']['source_sha256']
    backbone=Path(torch.hub.get_dir())/'checkpoints/vgg16-397923af.pth'
    weight=source.with_name('weights.pt')
    assert sha(backbone)==CONFIG['dists']['backbone_sha256']
    assert sha(weight)==CONFIG['dists']['weights_sha256']
    spec=importlib.util.spec_from_file_location('t3_official_frozen_dists',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original=module.models.vgg16
    vgg=original(weights=None)
    vgg.load_state_dict(torch.load(backbone,map_location='cpu'),strict=True)
    module.models.vgg16=lambda **kwargs:vgg
    try:model=module.DISTS(load_weights=False)
    finally:module.models.vgg16=original
    weights=torch.load(weight,map_location='cpu')
    with torch.no_grad():model.alpha.copy_(weights['alpha']);model.beta.copy_(weights['beta'])
    write(ROOT/'audit/dists_assets.json',dict(source=str(source),source_sha256=sha(source),
        backbone=str(backbone),backbone_sha256=sha(backbone),weights=str(weight),weights_sha256=sha(weight)))
    return model.eval().requires_grad_(False).to(device)


def features(model,im,coords):
    import torch
    parts=np.stack([im[y:y+64,x:x+64] for x,y in coords])
    tensor=torch.from_numpy(parts.transpose(0,3,1,2).copy()).float().to(model.alpha.device)/255
    return model.forward_once(tensor)


def distances(model,fx,fy):
    """与既有逐通道官方 DISTS 矩阵公式完全相同，支持 CPU/GPU。"""
    import torch
    denominator=model.alpha.sum()+model.beta.sum()
    weights=(model.alpha/denominator).flatten().split(model.chns)
    weights2=(model.beta/denominator).flatten().split(model.chns)
    score=torch.zeros((len(fx[0]),len(fy[0])),device=model.alpha.device)
    for x,y,a,b in zip(fx,fy,weights,weights2):
        x=x.flatten(2);y=y.flatten(2);mx=x.mean(-1);my=y.mean(-1)
        xc=x-mx[:,:,None];yc=y-my[:,:,None]
        vx=xc.square().mean(-1);vy=yc.square().mean(-1)
        covariance=torch.bmm(xc.permute(1,0,2),yc.permute(1,2,0)).permute(1,2,0)/x.shape[-1]
        s1=(2*mx[:,None,:]*my[None,:,:]+1e-6)/(mx[:,None,:].square()+my[None,:,:].square()+1e-6)
        s2=(2*covariance+1e-6)/(vx[:,None,:]+vy[None,:,:]+1e-6)
        score+=(s1*a+s2*b).sum(-1)
    return 1-score


def patch_distance(model,first,second):
    matrix=distances(model,first,second)
    return float(.5*(matrix.min(1).values.mean()+matrix.min(0).values.mean()))


def reference_features(model,path):
    img=np.asarray(Image.open(path).convert('RGB').resize((384,512),Image.Resampling.LANCZOS))
    coords=[(int(x),int(y)) for y in np.linspace(0,448,4).round() for x in np.linspace(0,320,4).round()]
    return features(model,img,coords)


def sanity(model):
    import torch
    a=np.random.default_rng(271042).integers(0,256,(64,64,3),dtype=np.uint8)
    b=np.random.default_rng(271043).integers(0,256,(64,64,3),dtype=np.uint8)
    tensor=lambda im:torch.from_numpy(im.transpose(2,0,1).copy())[None].float().to(model.alpha.device)/255
    with torch.inference_mode():
        fa,fb=features(model,a,[(0,0)]),features(model,b,[(0,0)])
        value=patch_distance(model,fa,fb)
        official=float(model(tensor(a),tensor(b)))
        self_distance=patch_distance(model,fa,fa)
    assert abs(value-official)<2e-6 and abs(self_distance)<1e-5
    return dict(passed=True,official=official,matrix=value,self_distance=self_distance)


def statistics(values,seed=271042):
    values=np.asarray(values,float);values=values[np.isfinite(values)]
    if not len(values):return dict(n=0,mean=None,sd=None,ci95=None,p=None)
    rng=np.random.default_rng(seed)
    sample=values[rng.integers(len(values),size=(10000,len(values)))].mean(1)
    signs=rng.choice([-1,1],size=(10000,len(values)))
    p=(1+np.count_nonzero(np.abs((signs*values).mean(1))>=abs(values.mean())))/10001
    return dict(n=len(values),mean=float(values.mean()),sd=float(values.std(ddof=1)) if len(values)>1 else None,
                ci95=np.quantile(sample,[.025,.975]).tolist(),p=float(p))


def holm(records):
    ordered=sorted(records,key=lambda r:r['p']);running=0
    for i,row in enumerate(ordered):
        running=max(running,min(1.,row['p']*(len(ordered)-i)));row['p_holm']=running


def audit():
    """在服务器从原 PNG 重算三个相关旧臂，不拼接不同分母的均值。"""
    import torch
    complete=ROOT/'audit/development_replay_complete.json'
    if complete.exists():
        assert read(complete)['passed'];return
    model=load_dists();checks=sanity(model)
    original=read(MATCHED/'splits/P2b.json')
    swap=read(MATCHED/'splits/reference_swap.json')
    refcache={}
    def ref(path):
        if path not in refcache:refcache[path]=reference_features(model,path)
        return refcache[path]
    out=[]
    with torch.inference_mode():
        for stage,rows in [('P2b',original),('reference_swap',swap)]:
            for row in rows:
                mask=np.asarray(Image.open(SUP/f'masks/evaluation/{row["id"]}/garment.png').convert('L'))>127
                coords,count=patches(mask)
                for arm in ['C','T3','T3M']:
                    path=legacy(stage,arm)/'images'/(row['id']+'.png')
                    meta=read(path.with_suffix('.json'));assert sha(path)==meta['output_sha256']
                    rec=dict(identity=row['id'],stage=stage,arm=arm,output_sha256=sha(path),patch_n=len(coords),
                             candidate_patch_n=count,dists=None,dists_old=None,margin=None,
                             failure=None if len(coords)>=4 else 'insufficient_input_mask_patches')
                    key='reference' if stage=='P2b' else 'reference_alternative'
                    assert sha(row[key])==meta['inputs']['reference']
                    if len(coords)>=4:
                        f=features(model,array(path),coords)
                        rec['dists']=patch_distance(model,f,ref(row[key]))
                        if stage=='reference_swap':
                            rec['dists_old']=patch_distance(model,f,ref(row['reference']))
                            rec['margin']=rec['dists_old']-rec['dists']
                    out.append(rec)
            print('REPLAY',stage,len(out),flush=True)
    csv_write(ROOT/'audit/development_replay.csv',out)
    write(ROOT/'audit/development_replay.json',out)
    contrasts=[]
    lookup={(r['stage'],r['arm'],r['identity']):r for r in out}
    for arm in ['T3M','T3']:
        diffs=[lookup['reference_swap','C',r['id']]['margin']-lookup['reference_swap',arm,r['id']]['margin']
               for r in swap if lookup['reference_swap','C',r['id']]['margin'] is not None
               and lookup['reference_swap',arm,r['id']]['margin'] is not None]
        contrasts.append(dict(comparison='C-'+arm,**statistics(diffs,221042)))
    holm(contrasts)
    means={}
    for arm in ['C','T3','T3M']:
        means[arm]={stage:statistics([r['dists'] for r in out if r['arm']==arm and r['stage']==stage and r['dists'] is not None])
                    for stage in ['P2b','reference_swap']}
        means[arm]['margin']=statistics([r['margin'] for r in out if r['arm']==arm and r['margin'] is not None])
    diff=-next(r['mean'] for r in contrasts if r['comparison']=='C-T3')
    sigma=next(r['sd'] for r in contrasts if r['comparison']=='C-T3')
    assert abs(diff-.007089932759602864)<2e-6,('DEVELOPMENT_REPLAY_CHANGED',diff)
    assert means['C']['P2b']['n']==87 and means['C']['reference_swap']['n']==27
    assert all(r['n']==27 for r in contrasts)
    record=dict(passed=True,sanity=checks,means=means,contrasts=contrasts,planned_original=96,planned_swap=32,
       actual_margin_pairs=27,old_p_holm_reported=.0202,
       p_note='Monte Carlo draws reset per contrast; p may differ from old global RNG stream; old two-test Holm family retained',
       paired_sd=sigma,power_n_005=math.ceil(((1.959964+.841621)*sigma/.005)**2),
       power_missing_adjusted_n=math.ceil(((1.959964+.841621)*sigma/.005)**2/(27/32)),
       planned_confirmation_n=128,commit=commit(),no_new_training=True)
    write(complete,record);print('DEVELOPMENT_REPLAY_PASSED',diff,'N',27,flush=True)


def prepare():
    from scipy.optimize import linear_sum_assignment
    from eval.eval_utils import prepare_evaluation_masks
    from garment_mask_utils import mask_backend_info
    if (ROOT/'audit/preparation_complete.json').exists():return
    assert read(ROOT/'audit/development_replay_complete.json')['passed']
    # 先冻结协议，再打开清单；确认集输出不能参与任何参数或样本量选择。
    freeze(ROOT/'protocol/config.json',CONFIG)
    freeze(ROOT/'protocol/code_before_LOCK.json',dict(commit=commit(),files={p:sha(SOURCE/p) for p in
        ['tools/t3_response_common.py','tools/t3_response_run.py','eval/metrics.py','eval/eval_utils.py','garment_mask_utils.py']}))
    assert mask_backend_info()['mask_backend']=='opencv'
    previous=read(SUP/'audit/preparation_complete.json')
    assert previous['LOCK_generated_or_evaluated'] is False
    candidates=read(SUP/'splits/LOCK.json');assert len(candidates)==128
    ids={r['id'] for r in candidates};assert len(ids)==128
    p2={r['id'] for s in ['P2a','P2b'] for r in read(SUP/f'splits/{s}.json')}
    assert not ids&p2
    train=read(PROJECT/'data/train_bf_texture.json')
    train_ids={Path(r[k]).stem for r in train for k in ['cloth','sketch','texture']}
    assert not ids&train_ids
    # 审计原划分后新增实验的实际评价输出；只读未启用的其他 LOCK 清单不等于曝光。
    exposure=[]
    for manifest in sorted((PROJECT/'output_eval').glob('*/evaluation/**/generation_complete.json')):
        if ROOT in manifest.parents:continue
        actual={p.stem for p in (manifest.parent/'images').glob('*.png')}&ids
        if actual:exposure.append(dict(manifest=str(manifest),ids=sorted(actual)))
    assert not exposure,('LOCK_ALREADY_GENERATED',exposure)
    hold=PROJECT/'output_eval/e48_b1_structure_20261014/holdout_audit'
    complete=read(hold/'complete.json')
    signature_path=Path(complete['training_signatures']['path'])
    if not signature_path.is_absolute():signature_path=PROJECT/signature_path
    assert sha(signature_path)==complete['training_signatures']['sha256']
    train_sigs=read(signature_path)
    trainroot=PROJECT.parent/'datasets/BF/training'
    trainpaths=sorted({str(trainroot/r[k]) for r in train for k in ['cloth','sketch','texture']})
    assert set(trainpaths)<=set(train_sigs)
    training_exact={train_sigs[p][k] for p in trainpaths for k in ['file_sha256','pixel_sha256']}
    trainph=np.array([list(bytes.fromhex(train_sigs[p]['phash'])) for p in trainpaths if '/sketch/' not in p],np.uint8)
    vals=read(hold/'validation_signatures.json')
    assert sha(hold/'validation_signatures.json')==complete['validation_signatures_sha256']
    bits=np.array([bin(i).count('1') for i in range(256)],np.uint8)
    model_registry={}
    for arm,(path,expected) in MODELS.items():
        digest=sha(path);assert digest==expected,(arm,digest)
        model_registry[arm]=dict(path=str(path),sha256=digest)
    freeze(ROOT/'protocol/model_registry.json',model_registry)
    manifests=[];allph=[]
    for row in candidates:
        for key in ['gt','sketch','reference']:
            path=row[key];sig=vals[path]
            assert sha(path)==sig['file_sha256']
            pixel=sha_pixels(path);assert pixel==sig['pixel_sha256']
            assert sig['file_sha256'] not in training_exact and pixel not in training_exact
            if key in ['gt','reference']:
                probe=np.array(list(bytes.fromhex(sig['phash'])),np.uint8)
                nearest=int(bits[trainph^probe].sum(1).min());assert nearest>6
                allph.append((row['id'],probe))
            else:nearest=None
            manifests.append(dict(identity=row['id'],role=key,source_path=path,
               sha256=sig['file_sha256'],pixel_sha256=pixel,training_nearest_phash=nearest))
        bundle=prepare_evaluation_masks((384,512),sketch_path=row['sketch'],mask_policy='sketch_only')
        mask=bundle['garment'];coords,count=patches(mask)
        save=ROOT/'masks'/(row['id']+'.png');save.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(mask.astype('uint8')*255).save(save)
        row['metric_mask']=str(save);row['metric_mask_sha256']=sha(save)
        row['patches']=coords;row['candidate_patch_n']=count
        row['input_metric_invalid']=len(coords)<4
        row['mask_stats']=bundle['stats']
    # 身份间近重复亦复核，原始 GT/ref 的同身份相似不算泄漏。
    near=[]
    for i,(identity,probe) in enumerate(allph):
        for other,p in allph[:i]:
            if other!=identity and int(bits[probe^p].sum())<=6:near.append([identity,other])
    assert not near,('LOCK_CROSS_IDENTITY_NEAR_DUPLICATE',near)
    small=[np.asarray(Image.open(r['reference']).convert('RGB').resize((32,32)),float)/255 for r in candidates]
    cost=np.empty((128,128));differences=np.empty_like(cost)
    for i,row in enumerate(candidates):
        for j,donor in enumerate(candidates):
            diff=float(np.abs(small[i]-small[j]).mean());differences[i,j]=diff
            same=row['stratum']['category']==donor['stratum']['category']
            cost[i,j]=1e9 if i==j or diff<.05 else (0 if same else 1000)+int(stable(row['id']+'/'+donor['id'])[:8],16)/2**32
    ii,jj=linear_sum_assignment(cost)
    assert len(ii)==128 and max(cost[ii,jj])<1e9
    pairs=[]
    for i,j in zip(ii,jj):
        row=dict(candidates[i]);donor=candidates[j]
        row.update(reference_alternative=donor['reference'],alternative_id=donor['id'],
                   input_rgb_difference=float(differences[i,j]),
                   same_category=row['stratum']['category']==donor['stratum']['category'])
        pairs.append(row)
    freeze(ROOT/'splits/confirmation.json',pairs)
    freeze(ROOT/'splits/qualitative_fixed24.json',sorted(ids,key=stable)[:24])
    csv_write(ROOT/'splits/input_manifest.csv',manifests)
    csv_write(ROOT/'splits/split_manifest.csv',[dict(identity=r['id'],split='confirmation',
       category=r['stratum']['category'],input_metric_invalid=r['input_metric_invalid'],
       exclusion_reason='',alternative_id=r['alternative_id'],same_category=r['same_category'],
       reference_alternative_sha256=sha(r['reference_alternative'])) for r in pairs])
    freeze(ROOT/'audit/independence.json',dict(passed=True,train_manifest_sha256=sha(PROJECT/'data/train_bf_texture.json'),
       training_signature_sha256=sha(signature_path),training_signature_source_audited_in_SUP=True,
       training_ID_overlap=0,training_exact_or_phash_overlap=0,P2_overlap=0,
       previously_generated_LOCK_identities=exposure,cross_identity_near_duplicates=near,
       pretrained_exposure_unknown=True,selected_inputs_rehashed=len(manifests)))
    write(ROOT/'audit/preparation_complete.json',dict(passed=True,n=128,
       input_valid_n=sum(not r['input_metric_invalid'] for r in pairs),
       same_category_pairs=sum(r['same_category'] for r in pairs),
       protocol_sha256=sha(ROOT/'protocol/config.json'),split_sha256=sha(ROOT/'splits/confirmation.json'),
       LOCK_opened_under_new_protocol=True,commit=commit()))
    print('PREPARATION_PASSED',read(ROOT/'audit/preparation_complete.json'),flush=True)


def sha_pixels(path):
    import hashlib
    # 原签名采用解码后 RGB 原始像素，不加入尺寸头。
    return hashlib.sha256(np.asarray(Image.open(path).convert('RGB')).tobytes()).hexdigest()


def generate(arm):
    import torch
    from tools import e2a_attr_eval as evaluator
    assert read(ROOT/'audit/preparation_complete.json')['passed']
    assert read(ROOT/'protocol/config.json')==CONFIG
    path,expected=MODELS[arm];assert sha(path)==expected
    rows=read(ROOT/'splits/confirmation.json')
    evaluator.OUT=ROOT;evaluator.model_path=lambda _:path
    # 复用原生成函数；额外分支模型的检查点验证记录来自正式训练原目录。
    registry={str(p):dict(sha256=h) for p,h in MODELS.values()}
    freeze(ROOT/'checkpoint_manifest.json',registry)
    for stage in ['original','swapped']:
        variants=[dict(r,reference=r['reference_alternative'] if stage=='swapped' else r['reference']) for r in rows]
        evaluator.input_rows=lambda _,v=variants:v
        evaluator.generate(stage,arm)
    metric(arm)


def metric(arm):
    import torch
    from eval.eval_utils import prepare_evaluation_masks,json_safe
    from eval.metrics import compute_structure_preservation,compute_texture_leakage,compute_texture_color_fidelity
    done=ROOT/'metrics'/arm/'complete.json'
    if done.exists():return
    model=load_dists('cuda' if torch.cuda.is_available() else 'cpu')
    checks=sanity(model);rows=read(ROOT/'splits/confirmation.json');out=[]
    @lru_cache(maxsize=24)
    def ref(path):return reference_features(model,path)
    with torch.inference_mode():
        for row in rows:
            identity=row['id'];coords=row['patches'];binding=None
            assert sha(row['metric_mask'])==row['metric_mask_sha256']
            bundle=prepare_evaluation_masks((384,512),sketch_path=row['sketch'],mask_policy='sketch_only')
            assert np.array_equal(bundle['garment'],np.asarray(Image.open(row['metric_mask']))>127)
            for stage in ['original','swapped']:
                image=ROOT/f'evaluation/{stage}/{arm}/images/{identity}.png'
                meta=read(image.with_suffix('.json'));assert sha(image)==meta['output_sha256']
                bind=(meta['seed'],meta['initial_latent_sha256'],meta['inputs']['sketch'],meta['inputs']['gt'])
                if binding is None:binding=bind
                assert binding==bind and not meta['gt_used_in_inference']
                assert meta['inputs']['reference']==sha(row['reference_alternative' if stage=='swapped' else 'reference'])
                structure=compute_structure_preservation(gen_path=str(image),sketch_path=row['sketch'],mask_bundle=bundle)
                # 沿用历史固定衣外背景偏差定义；不把旧 GT 当新参考服装的反事实真值。
                leakage=compute_texture_leakage(gen_path=str(image),sketch_path=row['sketch'],target_path=row['gt'],mask_bundle=bundle)
                colors=[compute_texture_color_fidelity(gen_path=str(image),texture_path=row[k],mask_bundle=bundle)
                        for k in ['reference','reference_alternative']]
                rec=dict(identity=identity,arm=arm,stage=stage,output_sha256=sha(image),
                    seed=meta['seed'],latent_sha256=meta['initial_latent_sha256'],
                    iou=structure.get('sketch_iou'),edge_f1=structure.get('edge_f1'),edge_l1=structure.get('edge_l1'),
                    leak=leakage.get('leak_colored_frac'),color_old=colors[0].get('tcf_lab_delta'),color_new=colors[1].get('tcf_lab_delta'),
                    dists_old=None,dists_new=None,margin=None,patch_n=len(coords),
                    failure='insufficient_input_mask_patches' if len(coords)<4 else None,
                    metric_warnings=structure.get('metric_warnings',[])+leakage.get('metric_warnings',[]),
                    mask_stats=bundle['stats'],no_counterfactual_GT_metric=True)
                if rec['color_old'] is not None and rec['color_new'] is not None:rec['color_margin']=rec['color_old']-rec['color_new']
                else:rec['color_margin']=None
                if len(coords)>=4:
                    f=features(model,array(image),coords)
                    rec['dists_old']=patch_distance(model,f,ref(row['reference']))
                    rec['dists_new']=patch_distance(model,f,ref(row['reference_alternative']))
                    rec['margin']=rec['dists_old']-rec['dists_new']
                out.append(json_safe(rec))
            print('METRIC',arm,identity,len(out)//2,'/128',flush=True)
    write(ROOT/f'metrics/{arm}/per_identity.json',out);csv_write(ROOT/f'metrics/{arm}/per_identity.csv',out)
    write(done,dict(passed=True,arm=arm,n=128,rows=len(out),sanity=checks,commit=commit()))


def dispatch():
    assert read(ROOT/'audit/preparation_complete.json')['passed']
    jobs=read(ROOT/'jobs.json') if (ROOT/'jobs.json').exists() else []
    for arm in MODELS:
        if any(j['key']==arm for j in jobs):continue
        command=['sbatch','--parsable','-p','gpu','-N','1','-n','1','-c','4','--mem=32G','--gres=gpu:1',
            '-t','03:00:00','-J','T3R_'+arm,'-o',str(ROOT/f'logs/{arm}_%j.log'),
            str(SOURCE/'scripts/t3_response.sbatch'),'generate','--arm',arm]
        identifier=int(subprocess.check_output(command,text=True).strip().split(';')[0])
        jobs.append(dict(key=arm,id=identifier,gpus=1,command=command,commit=commit()))
        write(ROOT/'jobs.json',jobs);print('SUBMITTED',arm,identifier,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['audit','prepare','generate','metric','dispatch','analyse','package'])
    parser.add_argument('--arm',choices=list(MODELS));args=parser.parse_args()
    for name in ['logs','audit','protocol','splits','metrics','statistics']:(ROOT/name).mkdir(parents=True,exist_ok=True)
    import torch
    torch.set_num_threads(min(4,int(os.environ.get('SLURM_CPUS_PER_TASK','2'))))
    if args.action=='audit':audit()
    elif args.action=='prepare':
        audit();prepare()
    elif args.action=='generate':generate(args.arm)
    elif args.action=='metric':metric(args.arm)
    elif args.action=='dispatch':dispatch()
    elif args.action=='analyse':
        from tools.t3_response_report import analyse
        analyse()
    elif args.action=='package':
        from tools.t3_response_report import package
        package()


if __name__=='__main__':main()
