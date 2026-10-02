"""P0→sanity→full硬Gate；最终checkpoint评测，禁止held-out选择checkpoint。"""
import argparse,math,random,time
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.apacc_features import load_dino
from models.e33r_rotation_control import SketchPrior,RotationControl
from data.e33r_group_dataset import GroupDataset,encode_reference
from tools.e33r_common import *
from tools.e33r_losses import group_loss
from tools.e33r_evaluate import evaluate_records

def load_prior():
    path=OUT/'P0_prior/checkpoint_final.pt';info=read(OUT/'P0_prior/summary.json')
    assert info['pass'] and sha(path)==info['checkpoint_sha256']
    model=SketchPrior().cuda();model.load_state_dict(torch.load(path,map_location='cpu')['model'])
    return model.eval(),info['checkpoint_sha256']

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['sanity','full','ablation'],required=True)
    parser.add_argument('--seed',type=int,choices=[42,43,44],default=42)
    parser.add_argument('--variant',choices=['full']+PROTOCOL['ablations'],default='full')
    args=parser.parse_args();decision=read(OUT/'decision_summary.json')
    assert decision['prior_pass']
    if args.phase=='sanity':assert args.seed==42 and args.variant=='full'
    else:assert decision['sanity_pass']
    if args.phase=='ablation':
        assert args.seed==42 and args.variant!='full'
        full=read(OUT/'P1_controlled/seed42/dev/summary.json');assert full['gate_pass'] or full['near_gate']
    elif args.phase=='full':assert args.variant=='full'
    records=read(OUT/'sanity_manifest.json' if args.phase=='sanity' else OUT/'selected_manifest.json')
    folder=OUT/('R_sanity' if args.phase=='sanity' else
        'P1_controlled/seed%d'%args.seed if args.phase=='full' else 'ablations/'+args.variant)
    folder.mkdir(parents=True,exist_ok=True);assert not (folder/'checkpoint_final.pt').exists()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    assert torch.cuda.is_bf16_supported()
    prior,prior_sha=load_prior();model=RotationControl(prior,args.variant).cuda().train()
    dino,dino_sha=load_dino('cuda',WEIGHTS) if args.variant!='G_geometry_only' else (None,sha(WEIGHTS))
    optim=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
    loader=DataLoader(GroupDataset(records['train'],training=True),batch_size=2,shuffle=True,
        num_workers=2,pin_memory=True,drop_last=True,generator=torch.Generator().manual_seed(args.seed))
    steps=500 if args.phase=='sanity' else 8000;warmup=500
    r180=read(OUT/'R0_integrity/summary.json')['R180_training_enabled']
    configuration=dict(PROTOCOL['p1'])
    configuration.update(phase=args.phase,variant=args.variant,seed=args.seed,steps=steps,
        warmup=warmup,prior_sha256=prior_sha,dino_sha256=dino_sha,
        R180_loss_enabled=r180 and args.variant!='D_no_r180',prior_frozen=model.freeze_prior,
        reference_gradient=False,group_arms=['R0','R90','R180','N0','N90','N180','zero'],
        field_and_q='clean arms',nuisance='online resampled paired noisy arms consistency',
        train_cases=len(records['train']),device=torch.cuda.get_device_name())
    write(folder/'training_protocol.json',configuration)
    iterator=iter(loader);history=[];start=time.monotonic();gradient_check={}
    for step in range(1,steps+1):
        rate=min(step/warmup,1.) if step<=warmup else .5*(1+math.cos(math.pi*(step-warmup)/(steps-warmup)))
        for group in optim.param_groups:group['lr']=1e-4*rate
        optim.zero_grad(set_to_none=True);metrics={}
        for micro in range(4):
            try:case=next(iterator)
            except StopIteration:iterator=iter(loader);case=next(iterator)
            reference=encode_reference(case['pixels'],case['aux'],dino,args.variant)
            reference=torch.cat([reference,torch.zeros_like(reference[:,:1])],1)
            structure=case['structure'].cuda();gt=case['gt'].cuda();support=case['support'].cuda()
            b,arms=reference.shape[:2]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred=model(reference.flatten(0,1),structure[:,None].expand(-1,arms,-1,-1,-1).flatten(0,1))
            pred={k:v.reshape(b,arms,*v.shape[1:]) for k,v in pred.items()}
            loss,parts=group_loss(pred,gt,support,args.variant,r180)
            assert torch.isfinite(loss);(loss/4).backward()
            for name,value in dict(total=loss,**parts).items():metrics[name]=metrics.get(name,0)+float(value.detach())/4
            if step==1 and micro==0:
                gradient_check=dict(q_gradient_norm=float(model.q_head.weight.grad.norm()),
                    reference_projection_gradient_norm=float(model.reference_projection[0].weight.grad.norm()),
                    prior_has_gradient=any(p.grad is not None for p in model.prior.parameters()),
                    DINO_has_gradient=dino is not None and any(p.grad is not None for p in dino.parameters()),
                    finite_gradients=all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None),
                    geometry_support_cells=case['support'].flatten(1).sum(1).tolist())
                assert gradient_check['q_gradient_norm']>0 and gradient_check['finite_gradients']
                assert not gradient_check['DINO_has_gradient']
                if model.freeze_prior:assert not gradient_check['prior_has_gradient']
                write(folder/'gradient_check.json',gradient_check)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);optim.step()
        if step%50==0:
            history.append(dict(step=step,elapsed_seconds=time.monotonic()-start,**metrics));write(folder/'history.json',history)
            print('[E33R control]',args.phase,args.variant,args.seed,history[-1],flush=True)
    path=folder/'checkpoint_final.pt'
    torch.save(dict(model=model.state_dict(),steps=steps,seed=args.seed,variant=args.variant,prior_sha256=prior_sha,
                    git_commit=git_commit()),path)
    assert sha(OUT/'P0_prior/checkpoint_final.pt')==prior_sha
    baseline,_=load_prior();baseline.requires_grad_(False)
    write(folder/'checkpoint_integrity.json',dict(sha256=sha(path),prior_sha256=prior_sha,
        prior_checkpoint_unchanged=True,prior_model_unchanged=all(torch.equal(v,baseline.state_dict()[k])
           for k,v in model.prior.state_dict().items()),prior_freeze_exception=args.variant=='A_no_prior_freeze'))
    if model.freeze_prior:assert read(folder/'checkpoint_integrity.json')['prior_model_unchanged']
    if args.phase=='sanity':
        _,train=evaluate_records(model,records['train'],dino,folder/'train',args.variant,baseline)
        _,dev=evaluate_records(model,records['dev'],dino,folder/'dev',args.variant,baseline,True)
        checks={k:train[k]['mean']>=.95 for k in ['clean_r90_success','r180_identity_success','r0_identity_success']}
        checks['finite']=train['finite_prediction_rate']==1 and dev['finite_prediction_rate']==1
        passed=all(checks.values());write(folder/'gate.json',dict(checks=checks,**{'pass':passed},denominator=512))
        update_decision(sanity_pass=passed,next_route='P1_full' if passed else 'implementation_or_parameterization_debug')
        if not passed:
            write(folder/'implementation_review.json',dict(gradient_check=gradient_check,
                assertions='composition/sign/mask unit checks passed; complete7-arm groups; finite gradient; prior immutable',
                checkpoint_choice='fixed step500, no held-out selection',retrain=False,
                interpretation='sanity fail alone does not establish full8000-step architecture impossibility'))
        print('[E33R sanity Gate]',checks,flush=True)
    else:
        _,dev=evaluate_records(model,records['dev'],dino,folder/'dev',args.variant,baseline,True)
        strict_train=[r for r in records['train'] if r['strict']]
        evaluate_records(model,strict_train,dino,folder/'train_strict',args.variant,baseline)
        for group in ('causal_test','independent_confirmation'):
            evaluate_records(model,records[group],dino,folder/group,args.variant,baseline,True)
        if args.phase=='full':
            update_decision(**{'seed%d_pass'%args.seed:dev['gate_pass']})
            decision=read(OUT/'decision_summary.json');results=[decision['seed%d_pass'%s] for s in PROTOCOL['seeds']]
            if all(v is not None for v in results):
                passed=sum(bool(v) for v in results)>=2
                update_decision(controlled_rotation_causality_pass=passed,
                    next_route='E33_RC_real_rotation_curriculum' if passed else 'rotation_control_architecture_bottleneck')
        print('[E33R formal Gate]',args.variant,args.seed,dev['checks'],flush=True)
    frozen=read(OUT/'frozen_check.json');frozen['training_steps']=sum(torch.load(p,map_location='cpu')['steps'] for p in OUT.rglob('checkpoint_final.pt'))
    write(OUT/'frozen_check.json',frozen);finish_frozen(OUT)

if __name__=='__main__':main()
