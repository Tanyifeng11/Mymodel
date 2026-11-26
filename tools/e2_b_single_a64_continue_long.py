"""恢复 127619 A 的训练状态，单卡续训并记录同一批图的 FID 轨迹。"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from tools import e2_b_single_pants_inception_ab as source
from tools.e2_fft_experiment import SOURCE

cov, single, previous, init, tail, roi, ab = (
    source.cov, source.single, source.previous, source.init, source.tail, source.roi, source.ab)
RUN = 'a64_resume_256_fid'
PARENT_SHA = '897246cc75c29e11c124d952b3e4cbe49c8aeec90866e385c086809a99d0b0f7'
PARENT_STEP = 64
STEPS = 256
CHECKPOINTS = (64, 128, STEPS)


def optimizer_step(state):
    steps = {int(v['step']) for v in state['state'].values()}
    assert len(steps)==1
    return steps.pop()


def setup(prior,device):
    args,_ = source.setup(prior,device)
    parent = args.run
    assert tail.sha(parent/'review.zip')==PARENT_SHA
    manifest = tail.read(parent/'review_manifest.json')
    old = single.fusion.read_checked(parent/'protocol.json',manifest['protocol.json'])
    complete = single.fusion.read_checked(parent/'completion.json',manifest['completion.json'])
    assert complete['updates_per_arm']==PARENT_STEP and old['training_order']==[p['case_id'] for p in args.train_order]
    assert len(args.train_order)==64*single.BATCH
    step = parent/'validation'/f'step{PARENT_STEP}'
    result = single.fusion.read_checked(step/'result.json',manifest[f'validation/step{PARENT_STEP}/result.json'])
    assert result['pairs']==len(args.pairs['val'])==256 and result['protocol_sha256']==manifest['protocol.json']
    assert tail.sha(step/'pool3.npz')==result['pool3_sha256']
    rows = tail.read(step/'per_pair.json')
    assert [r['case_id'] for r in rows]==[p['case_id'] for p in args.pairs['val']]
    state = torch.load(parent/'resume.pt',map_location='cpu',weights_only=True)
    assert state['step']==PARENT_STEP and state['protocol_sha256']==manifest['protocol.json']
    args.start_head = step/'A.pt'
    params = torch.load(args.start_head,map_location='cpu',weights_only=True)
    assert params.keys()==state['heads']['A'].keys()
    assert all(torch.equal(v,state['heads']['A'][k]) for k,v in params.items())
    optim = state['optimizers']['A']
    assert optimizer_step(optim)==PARENT_STEP and len(optim['state'])==len(params)
    assert all(g['lr']==old['learning_rate']==single.LR for g in optim['param_groups'])
    args.start_digest = result['head_sha256']['A']
    args.start_renders = parent/'renders'/args.start_digest
    args.baseline_rows = rows
    args.run = parent/RUN
    args.run.mkdir(exist_ok=True,parents=True)
    args.spec = args.run/'protocol.json'
    baseline = init.metrics_from_group(result['groups']['all'],'A')
    baseline['pants_band_mae'] = result['groups']['pants']['texture']['A']['band_mae']
    protocol = dict(parent_job=127619,parent_arm='A',parent_step=PARENT_STEP,parent_review_sha256=PARENT_SHA,
        parent_resume_sha256=tail.sha(parent/'resume.pt'),parent_protocol_sha256=manifest['protocol.json'],
        initial_weights_sha256=tail.sha(args.start_head),initial_head_digest=args.start_digest,
        sampler=old['sampler'],model_sha256=old['model_sha256'],cohort_sha256=old['cohort_sha256'],
        keep_adamw_state=True,learning_rate=old['learning_rate'],effective_batch=single.BATCH,
        training_order=old['training_order'],repeats=STEPS//PARENT_STEP,additional_steps=STEPS,
        validation_pairs=256,validation_steps=list(CHECKPOINTS),baseline_metrics=baseline,
        thresholds=old['thresholds'],common_loss=old['common_loss'],pants_extra=old['A_extra'],
        gpus=1,code={n:tail.sha(SOURCE/n) for n in ('tools/e2_b_single_a64_continue_long.py',
            'scripts/e2_b_single_a64_continue_long.sbatch','tools/e2_b_single_pants_cov_ab.py')})
    return args,protocol,optim


def qualify(args,models,objective,optimizer_state):
    path=args.run/'qualification.json'
    if path.exists():
        assert tail.read(path)['protocol_sha256']==args.spec_sha
        return
    head,params=previous.start_head(args,models)
    opt=torch.optim.AdamW(params.values(),lr=single.LR)
    opt.load_state_dict(optimizer_state)
    assert optimizer_step(opt.state_dict())==PARENT_STEP
    pair=next(p for p in args.train_order if p['category']=='pants'
        and roi.regions(tail.case_mask(args,p),cov.CORE_RADIUS)[0].sum()>=cov.MIN_PIXELS)
    pred,gt,mask=previous.forward(args,models,head,pair)
    base,bterms=objective(pred,gt,mask,'A','pants')
    full,factual=objective(pred,gt,mask,'B','pants')
    for key in bterms.keys()-{'pants_covariance'}:
        torch.testing.assert_close(bterms[key],factual[key],atol=0,rtol=0)
    extra=factual['pants_covariance']
    torch.testing.assert_close(full-base,cov.COV_WEIGHT*extra,atol=5e-7,rtol=1e-4)
    grad=tail.backward(full,params)
    assert np.isfinite(grad) and grad>0 and float(extra.detach())>0
    models[2].adapter.clear()
    ab.audit_frozen(args,models,head,params)
    tail.write(path,dict(protocol_sha256=args.spec_sha,case=pair['case_id'],parent_head_match=True,
        optimizer_step=optimizer_step(opt.state_dict()),pant_covariance=float(extra.detach()),grad_norm=grad,
        common_h2_unchanged=True,optimizer_updates=0))
    print('A64_LONG_QUALIFIED',pair['case_id'],grad,flush=True)


def train(args,models,objective,metric_models,protocol,optimizer_state):
    head,params=previous.start_head(args,models)
    opt=torch.optim.AdamW(params.values(),lr=single.LR)
    opt.load_state_dict(optimizer_state)
    path=args.run/'resume.pt'
    step,trace,checks=0,[],{}
    if path.exists():
        own=torch.load(path,map_location='cpu',weights_only=True)
        assert own['protocol_sha256']==args.spec_sha
        step,trace,checks=(own[k] for k in ('step','trace','checks'))
        assert step==len(trace)
        single.rules.restore(params,own['head'])
        opt.load_state_dict(own['optimizer'])
    assert optimizer_step(opt.state_dict())==PARENT_STEP+step

    def weights():
        return {k:p.detach().cpu() for k,p in params.items()}

    def save():
        tail.save_torch(path,dict(protocol_sha256=args.spec_sha,step=step,trace=trace,checks=checks,
            head=weights(),optimizer=opt.state_dict()))

    def validate():
        dest=args.run/'validation'/f'step{step}'
        tail.save_torch(dest/'A.pt',weights())
        scored=previous.score(args,models,{'A':head},metric_models,'val',dest)['groups']
        metrics=init.metrics_from_group(scored['all'],'A')
        pants=scored['pants']['texture']['A']
        f=metrics['fid']
        gate=cov.validation_gate(metrics,args.start_metrics,args.zero_metrics,pants,args.start_pants)
        delta=f-protocol['baseline_metrics']['fid']
        protected=all(v for k,v in gate['checks'].items() if k!='fid_below_original_step0')
        checks[str(step)]=dict(metrics=metrics,pants=pants,fid_delta_vs_parent=delta,
            fid_terms=scored['all']['arms']['A']['fid'],improved_with_protection=delta<0 and protected,**gate)
        save()
        print('A64_LONG_VAL',step,f,delta,gate['checks'],flush=True)

    if step in CHECKPOINTS and str(step) not in checks:
        validate()
    for step in range(step+1,STEPS+1):
        offset=((step-1)%PARENT_STEP)*single.BATCH
        pairs=args.train_order[offset:offset+single.BATCH]
        trace.append(dict(step=step,**cov.update(args,models,objective,'B',head,params,opt,pairs)))
        save()
        if step%16==0:
            print('A64_LONG_TRAIN',step,STEPS,flush=True)
        if step in CHECKPOINTS:
            validate()
    ab.audit_frozen(args,models,head,params)
    assert optimizer_step(opt.state_dict())==PARENT_STEP+STEPS
    passing=[(int(s),r['metrics']['fid']) for s,r in checks.items() if r['improved_with_protection']]
    best=min(passing,key=lambda t:(t[1],t[0]))[0] if passing else 0
    formal=[int(s) for s,r in checks.items() if r['passed']]
    tail.write(args.run/'training.json',dict(protocol_sha256=args.spec_sha,trace=trace,validation=checks,
        frozen_unchanged=True,optimizer_start_step=PARENT_STEP,optimizer_end_step=PARENT_STEP+STEPS))
    tail.write(args.run/'completion.json',dict(additional_steps=STEPS,best_protected_step=best,
        original_five_gate_steps=formal,validation_pairs=256))
    init.fid.pack(args.run,['protocol.json','qualification.json','model.json','execution_environment.json',
        'training.json','completion.json',*(f'validation/step{s}/result.json' for s in CHECKPOINTS)])
    print('A64_LONG_DONE',best,formal,tail.sha(args.run/'review.zip'),flush=True)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('prior',type=Path)
    cli.add_argument('--check',action='store_true')
    opt=cli.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(4 if opt.check else 6)
    args,protocol,parent_optimizer=setup(opt.prior,'cpu' if opt.check else 'cuda:0')
    if opt.check:
        for row in args.baseline_rows:
            assert tail.sha(args.start_renders/(row['case_id']+'.png'))==row['pngs']['A']
        print('A64_LONG_PRECHECK',len(args.train_order),len(args.pairs['val']),
              optimizer_step(parent_optimizer),protocol['initial_weights_sha256'],flush=True)
        return
    assert torch.cuda.device_count()==1
    if args.spec.exists():
        assert tail.read(args.spec)==protocol
    else:
        tail.write(args.spec,protocol)
    args.spec_sha=tail.sha(args.spec)
    if (args.run/'completion.json').exists() and (args.run/'review.zip').exists():
        return
    models=init.load(args)
    assert args.original['frozen_model_hashes']==tail.read(args.parent_run/'model.json')['frozen_model_hashes']
    objective=cov.Objective(args)
    qualify(args,models,objective,parent_optimizer)
    metric_models=tail.metric_models('cpu',args.protocol['metric_assets'])
    train(args,models,objective,metric_models,protocol,parent_optimizer)


if __name__=='__main__':
    main()
