"""连续8000步课程；按最终阶段Gate停止，不在dev最佳step回滚。"""
import math
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.apacc_features import load_dino
from data.e33r_group_dataset import GroupDataset,encode_reference
from data.e33rc_real_pair_dataset import RealPairDataset
from tools.e33r_losses import group_loss
from tools.e33rc_losses import real_losses,retention_loss
from tools.e33rc_common import *
from tools.e33rc_evaluate import evaluate_dual

def forward_group(model,reference,structure):
    b,arms=reference.shape[:2]
    with torch.autocast('cuda',dtype=torch.bfloat16):
        pred=model(reference.flatten(0,1),structure[:,None].expand(-1,arms,-1,-1,-1).flatten(0,1))
    return {k:v.reshape(b,arms,*v.shape[1:]) for k,v in pred.items()}

@torch.no_grad()
def teacher_cache(seed,dino):
    path=OUT/'teachers'/('seed%d.pt'%seed)
    if path.exists():
        value=torch.load(path,map_location='cpu')
        assert value['source_sha256']==sha(source_checkpoint(seed))
        assert sha(path)==read(path.with_suffix('.json'))['cache_sha256'];return value
    teacher,_=load_control(seed);teacher.eval().requires_grad_(False)
    loader=DataLoader(GroupDataset(read(OUT/'teacher_probes.json')),batch_size=1,num_workers=0)
    values={k:[] for k in ['reference','structure','weight','orientation']}
    for i,case in enumerate(loader,1):
        ref=encode_reference(case['pixels'][:,:3],case['aux'][:,:3],dino)
        pred=forward_group(teacher,ref,case['structure'].cuda())
        values['reference'].append(ref.cpu());values['structure'].append(case['structure'])
        values['weight'].append(case['support'].float()*case['gt'][:,3])
        values['orientation'].append(pred['orientation'].cpu())
        if i%128==0:print('[E33RC fixed teacher probes]',seed,i,'/512',flush=True)
    cache={k:torch.cat(v) for k,v in values.items()};cache['source_sha256']=sha(source_checkpoint(seed))
    path.parent.mkdir(exist_ok=True);temporary=path.with_suffix('.tmp');torch.save(cache,temporary);temporary.replace(path)
    write(path.with_suffix('.json'),dict(source_sha256=cache['source_sha256'],cache_sha256=sha(path),
        probes=512,arms=['R0','R90','R180'],teacher_frozen=True,no_heldout_probes=True))
    del teacher;torch.cuda.empty_cache();return cache

def stream(loader):
    while True:
        for case in loader:yield case

def draw(iterator,count,seen,keys):
    rows=[]
    while len(rows)<count:
        row=next(iterator);sid=row['id'][0]
        if sid in seen:continue
        seen.add(sid);rows.append(row)
    return [{k:torch.cat([r[k] for r in rows[i:i+2]]) for k in keys} for i in range(0,count,2)]

def train_seed(seed,revision=False,variant='full'):
    folder=seed_folder(seed,revision,variant);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'run_status.json').exists():return read(folder/'run_status.json')
    assert not any(folder.glob('*/checkpoint_final.pt')),'异常中断后需核对阶段checkpoint，不能静默重复训练'
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    assert torch.cuda.is_bf16_supported()
    model,initial_sha=load_control(seed,variant=variant);model.train()
    # Geometry-only学生仍用完整输入教师；教师输入特征不能随消融改变。
    dino,_=load_dino('cuda',WEIGHTS)
    cache=teacher_cache(seed,dino) if variant not in ['B_no_retention','C_real_only'] else None
    # 缓存是否已存在不能影响各消融的dropout/nuisance随机序列。
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model.train();optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=5e-5,weight_decay=1e-4)
    controlled=read(OUT/'controlled_manifest.json');split=read(OUT/'split_manifest.json')
    cf_loader=DataLoader(GroupDataset(controlled['train'],training=True),batch_size=1,shuffle=True,num_workers=2,
        multiprocessing_context='spawn',pin_memory=True,drop_last=True,generator=torch.Generator().manual_seed(seed))
    real_loader=DataLoader(RealPairDataset(split['train'],read(OUT/'real_wrong_train.json')),batch_size=1,shuffle=True,num_workers=2,
        multiprocessing_context='spawn',pin_memory=True,drop_last=True,generator=torch.Generator().manual_seed(seed+1000))
    cf_iter=stream(cf_loader);real_iter=stream(real_loader)
    config=dict(seed=seed,variant=variant,revision=revision,initial_sha256=initial_sha,
        teacher_sha256=sha(source_checkpoint(seed)),optimizer_state_inherited=False,
        optimizer=PROTOCOL['optimizer'],loss=PROTOCOL['loss'],aggregation=PROTOCOL['aggregation'],
        no_real_rot90_in_training=True,GT_and_target_mask_supervision_only=True,
        effective_reconstruction_target_batch=8,auxiliary_teacher_probe_batch=2 if cache else 0)
    write(folder/'training_protocol.json',config)
    step=0;history=[];start=time.monotonic();probe_rng=np.random.default_rng(seed+330000);stage_decisions=[]
    for stage,stage_steps,ratio in STAGES:
        current=folder/stage;current.mkdir(exist_ok=True)
        if revision or variant=='D_fixed_50_50':ratio=.5
        if variant=='C_real_only':ratio=0
        ncf=int(8*ratio);nreal=8-ncf;retain_weight=1. if revision else .5
        if variant in ['B_no_retention','C_real_only']:retain_weight=0.
        rank_weight=0. if variant=='A_no_ranking' else .5
        write(current/'stage_protocol.json',dict(stage=stage,steps=stage_steps,controlled_identities=ncf,real_identities=nreal,
            retain_weight=retain_weight,rank_weight=rank_weight,initial_source=initial_sha,total_expected_steps=8000))
        for stage_step in range(1,stage_steps+1):
            step+=1;rate=min(step/300,1) if step<=300 else .5*(1+math.cos(math.pi*(step-300)/7700))
            for group in optimizer.param_groups:group['lr']=5e-5*rate
            optimizer.zero_grad(set_to_none=True);metrics=dict(controlled=0.,real=0.,rank=0.,retain=0.)
            seen=set()
            if ncf:
                for case in draw(cf_iter,ncf,seen,['pixels','aux','structure','gt','support']):
                    reference=encode_reference(case['pixels'],case['aux'],dino,variant_model(variant))
                    reference=torch.cat([reference,torch.zeros_like(reference[:,:1])],1)
                    pred=forward_group(model,reference,case['structure'].cuda())
                    loss,_=group_loss(pred,case['gt'].cuda(),case['support'].cuda())
                    scale=len(case['gt'])/8;(loss*scale).backward();metrics['controlled']+=float(loss.detach())*scale
            for case in draw(real_iter,nreal,seen,['reference','structure','gt','support']):
                pred=forward_group(model,case['reference'].cuda(),case['structure'].cuda())
                reconstruction,ranking=real_losses(pred,case['gt'].cuda(),case['support'].cuda())
                scale=len(case['gt'])/8;((reconstruction+rank_weight*ranking)*scale).backward()
                metrics['real']+=float(reconstruction.detach())*scale;metrics['rank']+=float(ranking.detach())*scale
            if retain_weight:
                idx=probe_rng.choice(512,2,replace=False)
                probe=forward_group(model,cache['reference'][idx].float().cuda(),cache['structure'][idx].float().cuda())
                retain=retention_loss(probe['orientation'],cache['orientation'][idx].cuda(),cache['weight'][idx].cuda())
                (retain_weight*retain).backward();metrics['retain']=float(retain.detach())
            gradients=[p.grad for p in model.parameters() if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
            assert not any(p.grad is not None for p in model.prior.parameters())
            assert not any(p.grad is not None for p in dino.parameters())
            if step==1:
                assert float(model.reference_projection[0].weight.grad.norm())>0
                assert float(model.q_head.weight.grad.norm())>0
                write(folder/'gradient_check.json',dict(prior_has_gradient=False,dino_has_gradient=False,
                    projection_gradient_norm=float(model.reference_projection[0].weight.grad.norm()),
                    Q_head_gradient_norm=float(model.q_head.weight.grad.norm()),
                    C_head_gradient_norm=float(model.confidence_head.weight.grad.norm()) if model.confidence_head.weight.grad is not None else None,
                    teacher_requires_grad=False,finite_gradients=True,distinct_reconstruction_target_identities=len(seen)))
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);optimizer.step()
            if step%50==0:
                history.append(dict(step=step,stage=stage,stage_step=stage_step,controlled_identities=ncf,
                    real_identities=nreal,elapsed_seconds=time.monotonic()-start,lr=5e-5*rate,**metrics))
                write(folder/'history.json',history);print('[E33RC train]',seed,variant,revision,history[-1],flush=True)
        checkpoint=current/'checkpoint_final.pt';temporary=current/'checkpoint_final.tmp'
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),steps=step,stage_steps=stage_steps,
            seed=seed,variant=variant,revision=revision,initial_sha256=initial_sha,git_commit=git_commit()),temporary);temporary.replace(checkpoint)
        prior=torch.load(E33R/'P0_prior/checkpoint_final.pt',map_location='cpu')['model']
        assert all(torch.equal(v.cpu(),prior[k]) for k,v in model.prior.state_dict().items())
        record=evaluate_dual(model,seed,current,dino,variant)
        if variant!='full':record['continue']=True;record['ablation_fixed_budget_no_gate_stop']=True;write(current/'decision_record.json',record)
        write(current/'phase_complete.json',dict(checkpoint_sha256=sha(checkpoint),steps=step,stage_steps=stage_steps,prior_unchanged=True))
        stage_decisions.append(record);finish_frozen()
        if not record['continue']:break
        model.train()
    status=dict(seed=seed,variant=variant,revision=revision,training_steps=step,completed_RC_C=step==8000,
        collapse=any(r['controlled_retention']['collapse'] for r in stage_decisions),stage_decisions=stage_decisions,
        stopped_at=stage_decisions[-1]['stage'] if step<8000 else None)
    write(folder/'run_status.json',status);del model,dino,cache;torch.cuda.empty_cache();return status
