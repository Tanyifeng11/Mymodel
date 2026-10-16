"""GPU门禁、固定有效更新预算及真实RGB生成；失败记录保留。"""
import argparse
import json
import random
import time
import traceback
from datetime import datetime,timezone
import numpy as np
import torch
from torch.nn import functional as F
from tools.m54_protocol import *


class Inputs:
    def __init__(self,r):
        from tools import e44_train as module
        old=module.OUT;module.OUT=OUT
        try:self.data=module.TrainingInputs(r,'train1024')
        finally:module.OUT=old
        self.r=r;self.rows={x['id']:x for x in read(OUT/'splits/train1024.json')};self.count=0

    def next(self,disable_dropout=False):
        args,kw,noise,mask,record=self.data.next(disable_dropout)
        row=self.rows[record['id']];self.r.geometry(row['sketch'],row['reference']);self.count+=1
        # 历史image .05 + both .05已给出reference dropout .10，不再叠加为.19。
        if self.r.arm=='M5':record['outfitting_dropout']=record['dropout'] in ['image','both']
        return args,kw,noise,mask,record


def seed_all(seed=42):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


def loss_forward(r,args,kw,noise,mask):
    p=r.processor
    if r.arm=='M9':p.new_branch_active=False
    else:r.active=False
    try:
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):teacher=r.pipe.unet(*args,**kw).sample
    finally:r.active=True;p.new_branch_active=True
    p.candidate.atv=None
    with torch.autocast('cuda',dtype=torch.float16):prediction=r.pipe.unet(*args,**kw).sample
    diffusion=F.mse_loss(prediction.float(),noise.float());error=(prediction.float()-teacher.float()).square()
    outside=1-mask.float();keep=(error*outside).sum()/outside.sum().clamp_min(1e-8)
    loss=diffusion+.1*keep;boundary=diffusion.new_zeros(());atv=diffusion.new_zeros(())
    if r.arm=='M9':
        edge=F.interpolate(p.geometry[:,1:2],size=error.shape[-2:],mode='area')
        boundary=(error*edge).sum()/edge.sum().clamp_min(1e-8);loss=loss+.1*boundary
    if r.arm=='M1' and p.candidate.atv is not None:atv=p.candidate.atv;loss=loss+.001*atv
    return loss,dict(diffusion=float(diffusion.detach()),keep=float(keep.detach()),boundary=float(boundary.detach()),atv=float(atv.detach()))


def audit():
    from tools.m54_runtime import make_runtime,release
    from tools.e47_runtime import exact_trace
    assert read(OUT/'audit/prepare_complete.json')['passed']
    if (OUT/'audit/round0_complete.json').exists():assert read(OUT/'audit/round0_complete.json')['passed'];return
    rows=read(OUT/'splits/dev32.json')[:2];replay={}
    for arm,oldarm in [('E5','B'),('B1','D')]:
        r=make_runtime(arm,'audit/replay/'+arm)
        try:
            entries=[]
            for row in rows:
                path=image_path('audit/replay',arm,row['id']);r.generate(row,path)
                old=HISTORY/'dev32/images'/oldarm/(row['id']+'.png')
                check=exact_trace(path.with_suffix('.pt'),old.with_suffix('.pt'))
                assert check['bit_exact'] and sha(path)==sha(old),('REPLAY_FAILED',arm,row['id'],check)
                entries.append(dict(id=row['id'],trace=check,png_exact=True))
            replay[arm]=entries
        finally:release(r)
    results={}
    for arm in TRAIN_ARMS:
        seed_all();r=make_runtime(arm,'audit/modules/'+arm)
        try:
            data=Inputs(r);seed_all()
            for _ in range(32):
                args,kw,noise,mask,record=data.next(disable_dropout=True)
                if record['mask_valid']:break
            else:raise AssertionError('NO_VALID_TRAINING_MASK')
            p=r.processor
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                if arm=='M9':p.new_branch_active=False
                else:r.active=False
                baseline=r.pipe.unet(*args,**kw).sample
                r.active=True;p.new_branch_active=True
                initial=r.pipe.unet(*args,**kw).sample
            error=float((initial.float()-baseline.float()).abs().max());assert error<1e-6,(arm,error)
            loss,_=loss_forward(r,args,kw,noise,mask);loss.backward()
            gradients={n:float(v.grad.float().norm()) if v.grad is not None else None for n,v in p.candidate.named_parameters()}
            assert gradients['output.weight'] is not None and gradients['output.weight']>0,(arm,gradients)
            assert all(v is None or np.isfinite(v) for v in gradients.values())
            # 仅工程梯度诊断，非optimizer更新；随后销毁实例，科研训练重新零初始化。
            p.candidate.zero_grad(set_to_none=True)
            with torch.no_grad():
                p.candidate.output.weight.normal_(std=.001)
                if hasattr(p.candidate,'global_output'):p.candidate.global_output.weight.normal_(std=.001)
            loss,_=loss_forward(r,args,kw,noise,mask);loss.backward()
            live={n:float(v.grad.float().norm()) if v.grad is not None else None for n,v in p.candidate.named_parameters()}
            assert all(v is not None and np.isfinite(v) and v>0 for v in live.values()),('DEAD_PARAMETER_PATH',arm,live)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                r.active=False;off1=r.pipe.unet(*args,**kw).sample
                p.candidate.output.weight.mul_(2);off2=r.pipe.unet(*args,**kw).sample
                assert torch.equal(off1,off2)
            results[arm]=dict(initial_baseline='B1' if arm=='M9' else 'E5',initial_epsilon_fp32_max_abs_diff=error,
                zero_output_gradient=gradients,live_gradients=live,off_isolation=True,optimizer_updates=0,
                parameters=sum(v.numel() for v in r.parameters()),training_identity=record['id'])
        finally:release(r)
    write(OUT/'audit/round0_complete.json',dict(passed=True,replay=replay,modules=results,commit=commit(),training_updates=0))
    decision(R0='PASS')
    print('M54 ROUND0 PASS',flush=True)


def train(arm,total=800):
    from tools.m54_runtime import make_runtime,release
    assert arm in TRAIN_ARMS and total in [800,2400]
    assert read(OUT/'audit/round0_complete.json')['passed']
    folder=OUT/'training'/arm
    done=folder/('complete%04d.json'%total)
    if done.exists():assert read(done)['updates']==total;return
    seed_all();r=make_runtime(arm,'training/'+arm+'/runtime');data=Inputs(r)
    optimizer=torch.optim.AdamW(r.parameters(),lr=1e-4,weight_decay=.01);scaler=torch.amp.GradScaler('cuda')
    step=0;sequence=[];history=[];p=r.processor;p.candidate.train();started=time.perf_counter()
    if total==2400:
        state=r.load(folder/'checkpoints/step0800.pt');step=state['step'];optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        resume=state['resume'];data.data.order=resume['order'];data.data.rng.setstate(resume['data_rng']);data.count=resume['count']
        random.setstate(resume['python_rng']);np.random.set_state(resume['numpy_rng']);torch.set_rng_state(resume['torch_rng']);torch.cuda.set_rng_state_all(resume['cuda_rng'])
        sequence=read(folder/'random_sequence0800.json');history=read(folder/'history.json')
    else:
        assert not (folder/'checkpoints/step0000.pt').exists(),'PRESERVE_FAILED_ATTEMPT_BEFORE_RETRY'
        r.save(folder/'checkpoints/step0000.pt',0,optimizer,scaler);seed_all()
    windows={name:0 for name,_ in p.candidate.named_parameters()}
    try:
        while step<total:
            optimizer.zero_grad(set_to_none=True);items=[]
            for _ in range(8):
                args,kw,noise,mask,record=data.next();sequence.append(record)
                loss,item=loss_forward(r,args,kw,noise,mask);assert torch.isfinite(loss)
                if loss.requires_grad:scaler.scale(loss/8).backward()
                items.append(item)
            assert any(v.grad is not None for v in r.parameters()),'EMPTY_EFFECTIVE_UPDATE'
            scaler.unscale_(optimizer);norm=float(torch.nn.utils.clip_grad_norm_(r.parameters(),1.));assert np.isfinite(norm)
            for name,value in p.candidate.named_parameters():
                if value.grad is not None and float(value.grad.float().norm())>0:windows[name]+=1
            before=scaler.get_scale();scaler.step(optimizer);scaler.update();assert scaler.get_scale()>=before,'OPTIMIZER_UPDATE_SKIPPED'
            step+=1
            if step in [1,5,20] or step%25==0:
                item=dict(step=step,gradient_norm=norm,seconds=time.perf_counter()-started,
                    **{k:sum(x[k] for x in items)/8 for k in items[0]});history.append(item);write(folder/'history.json',history)
                print('M54 TRAIN',arm,step,item,flush=True)
            if step in [100,400,800,2400]:
                path=folder/'checkpoints'/('step%04d.pt'%step);r.save(path,step,optimizer,scaler)
                state=torch.load(path,map_location='cpu',weights_only=False)
                state['resume']=dict(order=data.data.order,data_rng=data.data.rng.getstate(),count=data.count,
                    python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
                torch.save(state,path);write(path.with_suffix('.json'),dict(step=step,sha256=sha(path),commit=commit()))
        assert all(windows.values()),('GRADIENT_COLLAPSE',arm,windows)
        write(folder/('random_sequence%04d.json'%total),sequence)
        write(done,dict(updates=step,additional_updates=800 if total==800 else 1600,
            parameters=sum(v.numel() for v in r.parameters()),nonzero_gradient_windows=windows,
            seconds=time.perf_counter()-started,peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,
            sequence_sha256=sha(folder/('random_sequence%04d.json'%total)),commit=commit()))
    finally:release(r)


def generate(arm,stage='pilot',updates=800,seed=42):
    from tools.m54_runtime import make_runtime,release
    assert read(OUT/'audit/round0_complete.json')['passed']
    basearm='M5' if arm.startswith('M5_S') else arm
    if basearm in TRAIN_ARMS:assert read(OUT/'training'/basearm/('complete%04d.json'%updates))['updates']==updates
    splitstage='pilot' if stage=='round1' else stage
    rows=read(OUT/'splits'/(splitstage+'.json'));folderstage=stage if updates==800 else stage+'_u'+str(updates)
    if seed!=42:folderstage+='_s'+str(seed)
    r=make_runtime(arm,folderstage+'/'+arm+'/runtime')
    if basearm in TRAIN_ARMS:r.load(OUT/'training'/basearm/'checkpoints'/('step%04d.pt'%updates));r.processor.candidate.eval().requires_grad_(False)
    try:
        for row in rows:
            path=image_path(folderstage,arm,row['id'])
            if path.exists() and path.with_suffix('.json').exists():assert sha(path)==read(path.with_suffix('.json'))['png_sha256'];continue
            record,_,_=r.generate(row,path,seed=seed);print('M54 RGB',arm,folderstage,row['id'],record['seconds'],flush=True)
    finally:release(r)
    write(OUT/folderstage/(arm+'_complete.json'),dict(n=len(rows),arm=arm,updates=updates if basearm in TRAIN_ARMS else 0,seed=seed,commit=commit()))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','audit','train','generate','evaluate','batch_generate','input_candidates','review_input'])
    p.add_argument('--arm');p.add_argument('--stage',default='pilot');p.add_argument('--updates',type=int,default=800);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--arms',nargs='+')
    p.add_argument('--diagnostic',action='store_true')
    p.add_argument('--review-result',choices=['PASS','FAIL']);p.add_argument('--review-note');p.add_argument('--reviewed-ids',nargs='+')
    args=p.parse_args()
    import os
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','2')) if args.action=='input_candidates' else 2)
    if os.environ.get('SLURM_JOB_ID'):
        write(OUT/'resources/code_entries'/(os.environ['SLURM_JOB_ID']+'.json'),dict(commit=commit(),arguments=vars(args),
            entry_utc=datetime.now(timezone.utc).isoformat(),source_files={str(p):sha(p) for pattern in ['tools/m54*.py','models/m54*.py','submit/m54*.sh'] for p in Path('.').glob(pattern)}))
    try:
        if args.action=='prepare':
            from tools.m54_prepare import prepare
            prepare()
        elif args.action=='audit':audit()
        elif args.action=='input_candidates':
            from tools.m54_input_candidates import build
            build()
        elif args.action=='review_input':
            assert args.review_result and args.review_note and args.reviewed_ids
            rows=read(OUT/'splits'/(args.stage+'.json'));ids={r['id'] for r in rows}
            assert set(args.reviewed_ids)<=ids
            passed=args.review_result=='PASS'
            if passed:assert set(args.reviewed_ids)==ids
            path=OUT/'audit/input_review'/(args.stage+'_approval.json' if passed else args.stage+'_review_r0.json')
            freeze(path,dict(passed=passed,reviewer='assistant',independent_human_review='NOT_RUN',
                actual_reviewed_ids=args.reviewed_ids,note=args.review_note,split_sha256=sha(OUT/'splits'/(args.stage+'.json')),commit=commit()))
        elif args.action=='train':train(args.arm,args.updates)
        elif args.action=='generate':generate(args.arm,args.stage,args.updates,args.seed)
        elif args.action=='batch_generate':
            for arm in args.arms:generate(arm,args.stage,args.updates,args.seed)
            if args.diagnostic:
                for arm in ['E5','B1','ORACLE']:generate(arm,'oracle_diagnostic')
        else:
            from tools.m54_eval import evaluate
            evaluate(args.stage)
            if args.stage=='round1':evaluate('oracle_diagnostic')
    except Exception:
        path=OUT/'failures'/(args.action+'_'+(args.arm or 'all')+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'.json')
        write(path,dict(arguments=vars(args),traceback=traceback.format_exc(),commit=commit()))
        raise


if __name__=='__main__':main()
