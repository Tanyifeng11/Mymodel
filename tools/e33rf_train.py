"""固定endpoint训练；RF1诊断不影响RF2独立初始化与seed决策。"""
import math,random,time
import numpy as np,torch
from torch.utils.data import DataLoader
from data.e33rf_real_rotation_dataset import RealRotationDataset
from tools.e33rc_train_curriculum import forward_group,stream,draw
from tools.e33rf_losses import losses
from tools.e33rf_evaluate import evaluate
from tools.e33rf_common import *

def frozen_state(model):return {k:v.detach().cpu().clone() for k,v in model.backbone.state_dict().items()}
def verify_backbone(model,before):
    names={k for k,p in model.backbone.named_parameters() if p.requires_grad}
    checks={k:torch.equal(v.detach().cpu(),before[k]) for k,v in model.backbone.state_dict().items() if k not in names}
    assert all(checks.values());return dict(checked_tensors=len(checks),intentional_thaw=model.variant=='C_full_finetune',**{'pass':True})
def train(seed,phase,dino,variant='full'):
    dest=folder(seed,phase,variant);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'phase_complete.json').exists():
        done=read(dest/'phase_complete.json');assert done['checkpoint_sha256']==sha(dest/'checkpoint_final.pt')
        return read(dest/'decision_record.json')
    assert not (dest/'history.json').exists(),'已有未完成训练，需从已保存状态核对恢复，禁止静默重跑'
    checkpoint=folder(seed,'RF2')/'checkpoint_final.pt' if phase=='RF3' else None
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model,parameters=build(seed,variant,checkpoint);model.train();before=frozen_state(model)
    config=PROTOCOL['phases'][phase];steps=config['steps'];rate=config['lr'];warmup=config['warmup']
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=rate,weight_decay=1e-4)
    split=read(OUT/'split_manifest.json');wrong=read(OUT/'real_wrong_train.json')
    loader=DataLoader(RealRotationDataset(split['train'],phase,wrong),batch_size=1,shuffle=True,num_workers=2,
        multiprocessing_context='spawn',pin_memory=True,generator=torch.Generator().manual_seed(seed+1000))
    iterator=stream(loader);history=[];start=time.monotonic()
    write(dest/'training_protocol.json',dict(seed=seed,phase=phase,variant=variant,config=config,parameters=parameters,
        initial_checkpoint=str(checkpoint) if checkpoint else str(source_checkpoint(seed)),fresh_optimizer=True,backbone_eval=variant!='C_full_finetune'))
    for step in range(1,steps+1):
        factor=step/warmup if step<=warmup else .5*(1+math.cos(math.pi*(step-warmup)/(steps-warmup)))
        for group in optimizer.param_groups:group['lr']=rate*factor
        optimizer.zero_grad(set_to_none=True);seen=set();metrics={}
        microbatches=draw(iterator,8,seen,['reference','structure','gt','support'])
        counts=[int(((c['support'].float()*c['gt'][:,3]).flatten(1).sum(1)>0).sum()) for c in microbatches]
        valid_targets=sum(counts)
        for case,count in zip(microbatches,counts):
            pred=forward_group(model,case['reference'].cuda(),case['structure'].cuda())
            supervised_scale=count/max(valid_targets,1);regular_scale=len(case['gt'])/8
            total,part=losses(pred,case['gt'].cuda(),case['support'].cuda(),phase,variant,supervised_scale,regular_scale)
            total.backward()
            for key,value in part.items():
                scale=regular_scale if key=='adapter_id' else supervised_scale
                metrics[key]=metrics.get(key,0)+value*scale
        trainable=[p for p in model.parameters() if p.requires_grad]
        gradients=[p.grad for p in trainable if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        if step==1:
            write(dest/'gradient_check.json',dict(finite_gradients=True,distinct_identities=len(seen),valid_targets=valid_targets,
                frozen_gradients_absent=True,adapter_last_linear_gradient_norm=float(model.adapter.net[-1].weight.grad.norm()) if model.adapter else None))
        torch.nn.utils.clip_grad_norm_(trainable,1.);optimizer.step()
        if step%50==0:
            history.append(dict(step=step,lr=rate*factor,elapsed_seconds=time.monotonic()-start,**metrics))
            write(dest/'history.json',history);print('[E33RF train]',seed,phase,variant,history[-1],flush=True)
        if phase=='RF1' and step%500==0 and step<steps:
            diagnostic=dest/('diagnostic_step%d'%step);record=evaluate(model,seed,diagnostic,dino)
            record['diagnostic_only']=True;record['checkpoint_selection']=False
            write(diagnostic/'decision_record.json',record);model.train()
    target=dest/'checkpoint_final.pt';temporary=target.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),seed=seed,phase=phase,variant=variant,
        steps=steps,git_commit=git_commit(),parameters=parameters),temporary);temporary.replace(target)
    frozen=verify_backbone(model,before);write(dest/'backbone_frozen_check.json',frozen)
    record=evaluate(model,seed,dest,dino)
    if phase=='RF1':
        baseline=read(OUT/'RF0_reproduction'/('seed%d'%seed)/'real/summary.json')['rot90_success']['mean']
        diagnostics=[read(dest/('diagnostic_step%d'%s)/'decision_record.json') for s in [500,1000]]+[record]
        record['match_only_adapter_causes_real_causal_collapse']=any(x['real_rot90']<.4 and baseline-x['real_rot90']>.30 for x in diagnostics)
        write(dest/'decision_record.json',record)
    write(dest/'phase_complete.json',dict(seed=seed,phase=phase,variant=variant,training_steps=steps,
        checkpoint_sha256=sha(target),backbone_frozen_check=frozen));finish_frozen()
    del model,optimizer;torch.cuda.empty_cache();return record
