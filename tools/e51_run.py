"""先 GPU 精确回放/梯度验收，再运行两次固定800有效更新与完整生成。"""
import argparse
import gc
import time
import numpy as np
import torch
from torch.nn import functional as F
from tools.e51_protocol import *
from tools.e51_runtime import make_runtime,close
from tools.e34_sarr_protocol import seed_all
from tools.e44_runtime import digest


def release(r):
    close(r);gc.collect();torch.cuda.empty_cache()


class Inputs:
    def __init__(self,r):
        from tools import e44_train as module
        old=module.OUT;module.OUT=OUT
        try:self.data=module.TrainingInputs(r,'train1024')
        finally:module.OUT=old
        self.r=r;self.rows={row['id']:row for row in read(OUT/'splits/train1024.json')}

    def next(self,disable_dropout=False):
        args,kw,noise,mask,record=self.data.next(disable_dropout)
        self.r.geometry(self.rows[record['id']]['sketch'])
        return args,kw,noise,mask,record


def loss_forward(r,args,kw,noise,mask):
    r.active=False
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):teacher=r.pipe.unet(*args,**kw).sample
    r.active=True;r.processor.last_control=None
    with torch.autocast('cuda',dtype=torch.float16):pred=r.pipe.unet(*args,**kw).sample
    diffusion=F.mse_loss(pred.float(),noise.float());delta=(pred.float()-teacher.float()).square()
    outside=1-mask.float();edge=r.processor.geometry[:,1:2]
    keep=(delta*outside).sum()/outside.sum().clamp_min(1e-8)
    boundary=(delta*edge).sum()/edge.sum().clamp_min(1e-8)
    control=r.processor.last_control or dict(gate_mean=0.,applied_rms=0.,cap=1.)
    return diffusion+.1*keep+.1*boundary,dict(diffusion=float(diffusion.detach()),keep=float(keep.detach()),
        boundary=float(boundary.detach()),gate_mean=control['gate_mean'],applied_rms=control['applied_rms'],cap=control['cap'])


def controller_gradients(r):
    return {name:float(p.grad.float().norm()) if p.grad is not None else None for name,p in r.processor.controller.named_parameters()}


def audit():
    from tools.e47_runtime import exact_trace
    from tools.e44_runtime import digest
    prepare();assert not (OUT/'audit/complete.json').exists(),'禁止重复已完成验收'
    results={};initial={}
    for arm in TRAIN_ARMS:
        seed_all(42);r=make_runtime(arm,'audit/'+arm+'/runtime');p=r.processor
        try:
            initial[arm]={n:digest(v) for n,v in p.controller.state_dict().items()}
            exact=[]
            for row in read(OUT/'splits/probe.json')[:8]:
                path=OUT/'audit'/arm/'zero_output/images'/(row['id']+'.png')
                r.generate(row,path)
                old=SOURCE/'probe/images/B'/(row['id']+'.png');check=exact_trace(path.with_suffix('.pt'),old.with_suffix('.pt'))
                assert check['bit_exact'] and sha(path)==sha(old)
                exact.append(dict(id=row['id'],trace=check,png_exact=True))
            assert initial[arm]=={n:digest(v) for n,v in p.controller.state_dict().items()}
            # 局部诊断仅扰动输出头，不执行优化器更新或产生科研候选检查点。
            data=Inputs(r);seed_all(42)
            for _ in range(16):
                args,kw,noise,mask,record=data.next(disable_dropout=True)
                if record['mask_valid']:break
            else:raise AssertionError('没有有效草图供梯度验收')
            optimizer=torch.optim.AdamW(r.parameters(),lr=1e-4,weight_decay=.01)
            scaler=torch.amp.GradScaler('cuda');r.processor.controller.train()
            loss,_=loss_forward(r,args,kw,noise,mask);scaler.scale(loss).backward();scaler.unscale_(optimizer)
            zero_grads=controller_gradients(r)
            assert zero_grads['output.weight']>0 and all(v is None or np.isfinite(v) for v in zero_grads.values())
            r.pipe.unet.zero_grad(set_to_none=True)
            with torch.no_grad():p.controller.output.weight.fill_(.001)
            optimizer=torch.optim.AdamW(r.parameters(),lr=1e-4,weight_decay=.01);scaler=torch.amp.GradScaler('cuda')
            loss,_=loss_forward(r,args,kw,noise,mask);scaler.scale(loss).backward();scaler.unscale_(optimizer)
            live_grads=controller_gradients(r)
            tracked=['target.weight','reference.weight','mix.weight','output.weight','gating.weight']
            if arm=='TIME':tracked+=['time.0.weight','time.2.weight']
            assert all(live_grads.get(n) is not None and np.isfinite(live_grads[n]) and live_grads[n]>0 for n in tracked),live_grads
            p.controller.eval();r.pipe.unet.zero_grad(set_to_none=True)
            with torch.no_grad():
                r.active=False;off=r.pipe.unet(*args,**kw).sample
                r.active=True;p.residual_scale=0.;zero=r.pipe.unet(*args,**kw).sample
                assert torch.equal(off,zero)
                p.residual_scale=1.;live=r.pipe.unet(*args,**kw).sample
                assert torch.isfinite(live).all() and not torch.equal(live,off)
                raw=r.pipe.unet.forward
                p.enabled=False;a=raw(*args,**kw).sample
                p.enabled=False;p.controller.output.weight.mul_(2);b=raw(*args,**kw).sample
                assert torch.equal(a,b),'OFF 分支受控制器权重污染'
                p.controller.output.weight.fill_(.001)
            live_cases=[]
            for row in [read(OUT/'splits/probe.json')[0],read(OUT/'splits/probe.json')[-1]]:
                path=OUT/'audit'/arm/'live/images'/(row['id']+'.png');r.generate(row,path)
                value=read(path.with_name(path.stem+'_control.json'))
                assert all(s.get('mask_zero_max') in [None,0.] for s in value['steps'])
                assert all(s['controlled_rms']<=s['raw_b1_rms']+1e-6 for s in value['steps'] if not s.get('branch_off'))
                live_cases.append(row['id'])
            # 原始 B1 旁路完整回放，验证提取的是原公式的真实残差。
            p.raw_bypass=True;row=read(OUT/'splits/probe.json')[0]
            path=OUT/'audit'/arm/'raw_B1/images'/(row['id']+'.png');r.generate(row,path)
            old=SOURCE/'probe/images/D'/(row['id']+'.png');raw_check=exact_trace(path.with_suffix('.pt'),old.with_suffix('.pt'))
            assert raw_check['bit_exact'] and sha(path)==sha(old)
            results[arm]=dict(zero_output_cases=exact,zero_output_gradient=zero_grads,live_gradient=live_grads,
                scale0_same_state_exact=True,off_isolation=True,raw_B1_trace=raw_check,live_cases=live_cases,
                optimizer_updates=0,parameters=sum(v.numel() for v in r.parameters()))
        finally:release(r)
    assert initial['NOTIME']==initial['TIME']
    write(OUT/'audit/complete.json',dict(passed=True,arms=results,identical_initial_weights=True,
        initial_controller_hashes=initial['TIME'],commit=commit()))
    decision(audit='pass')


def train(arm):
    from tools.e22_4_generation import module_hashes
    prepare();assert read(OUT/'audit/complete.json')['passed']
    folder=OUT/'training'/arm
    assert not (folder/'checkpoints/step0000.pt').exists(),'先保存工程失败目录，禁止覆盖训练'
    seed_all(42);r=make_runtime(arm);data=Inputs(r);p=r.processor
    initial={n:digest(v) for n,v in p.controller.state_dict().items()}
    assert initial==read(OUT/'audit/complete.json')['initial_controller_hashes']
    p.controller.train();seed_all(42);torch.cuda.reset_peak_memory_stats()
    optimizer=torch.optim.AdamW(r.parameters(),lr=1e-4,weight_decay=.01);scaler=torch.amp.GradScaler('cuda')
    sequence=[];history=[];step=0;empty=0;started=time.perf_counter()
    tracked=['target.weight','reference.weight','mix.weight','output.weight','gating.weight']
    if arm=='TIME':tracked+=['time.0.weight','time.2.weight']
    windows={name:0 for name in tracked};named=dict(p.controller.named_parameters())
    r.save(folder/'checkpoints/step0000.pt',0,optimizer,scaler)
    try:
        while step<800:
            optimizer.zero_grad(set_to_none=True);items=[]
            for _ in range(8):
                args,kw,noise,mask,record=data.next();sequence.append(record)
                loss,item=loss_forward(r,args,kw,noise,mask);assert torch.isfinite(loss)
                if loss.requires_grad:scaler.scale(loss/8).backward()
                items.append(item)
            if not any(v.grad is not None for v in r.parameters()):empty+=1;continue
            scaler.unscale_(optimizer);norm=float(torch.nn.utils.clip_grad_norm_(r.parameters(),1.));assert np.isfinite(norm)
            for name in tracked:
                value=named[name].grad
                if value is not None and float(value.float().norm())>0:windows[name]+=1
            grads=controller_gradients(r) if step+1 in [1,5,20,800] else None
            if step==0:assert grads['output.weight']>0
            if step+1 in [20,800]:assert all(windows.values()),windows
            old=scaler.get_scale();scaler.step(optimizer);scaler.update();assert scaler.get_scale()>=old
            step+=1
            if step in [1,5,20] or step%25==0:
                item=dict(step=step,gradients=grads,gradient_windows=dict(windows),grad_norm=norm,
                    seconds=time.perf_counter()-started,**{k:sum(v[k] for v in items)/8 for k in items[0]})
                history.append(item);write(folder/'history.json',history)
                print('E51 TRAIN',arm,step,item['diffusion'],item['gate_mean'],flush=True)
            if step in [100,400,800]:r.save(folder/'checkpoints'/('step%04d.pt'%step),step,optimizer,scaler)
        write(folder/'random_sequence.json',sequence)
        expected=CONFIG['training']['expected_sequence_sha256']
        assert empty==0 and len(sequence)==6400 and sha(folder/'random_sequence.json')==expected
        target=dict(r.pipe.unet.named_modules())[LAYER[:-10]];target.set_processor(r.original)
        try:assert module_hashes(r.modules)==r.before
        finally:target.set_processor(p)
        assert r.b1_hashes()==r.frozen_b1_before
        write(folder/'complete.json',dict(updates=step,parameters=sum(v.numel() for v in r.parameters()),
            empty_windows=empty,sequence_sha256=expected,frozen_E5_and_B1=True,
            nonzero_gradient_windows=windows,seconds=time.perf_counter()-started,
            peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,commit=commit()))
    finally:release(r)


def generate(arm,stage):
    folder=OUT/'training'/arm;assert read(folder/'complete.json')['updates']==800
    r=make_runtime(arm,stage+'/'+arm+'/runtime');r.load(folder/'checkpoints/step0800.pt')
    r.processor.controller.eval().requires_grad_(False)
    rows=read(OUT/'splits'/(stage+'.json'))
    try:
        for row in rows:
            path=image_path(arm,row['id'],stage)
            if path.exists() and path.with_name(path.stem+'_control.json').exists():
                assert sha(path)==read(path.with_suffix('.json'))['png_sha256'];continue
            record,_,_=r.generate(row,path);print('E51 RGB',arm,stage,row['id'],record['seconds'],flush=True)
    finally:release(r)
    write(OUT/stage/(arm+'_complete.json'),dict(n=len(rows),arm=arm,checkpoint_sha256=sha(folder/'checkpoints/step0800.pt'),commit=commit()))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','audit','run','train','generate','evaluate'])
    p.add_argument('--arm',choices=TRAIN_ARMS);p.add_argument('--stage',choices=['probe','dev32'],default='probe')
    args=p.parse_args();torch.set_num_threads(2)
    if args.action=='prepare':prepare()
    elif args.action=='audit':audit()
    elif args.action=='train':train(args.arm)
    elif args.action=='generate':generate(args.arm,args.stage)
    elif args.action=='run':train(args.arm);generate(args.arm,'probe');generate(args.arm,'dev32')
    else:
        from tools.e51_eval import evaluate_all
        evaluate_all()


if __name__=='__main__':main()
