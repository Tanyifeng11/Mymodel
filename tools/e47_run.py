"""GPU P0 回放和 C 的一次固定训练；P0 未通过时拒绝训练。"""
import argparse
import gc
import time
from pathlib import Path

import torch
from PIL import Image
from tools.e34_sarr_protocol import seed_all
from tools.e44_runtime import sketch_masks
from tools.e44_train import loss_forward,gradients
from tools.e47_protocol import *
from tools.e47_runtime import runtime,TrainingInputs,exact_trace,close


def release(r):
    close(r)
    del r
    gc.collect();torch.cuda.empty_cache()


def smoke():
    comparison=read(OUT/'checkpoint_comparison.json')
    if not comparison['all_common_exact']:
        decision(P0='NO_CAUSAL_CLAIM',reason='共有权重改变，需要先预算 A_mirror',P1='not_run')
        return
    assert sha(E5)==E5_SHA and sha(B1)==B1_SHA
    rows=read(OUT/'splits/dev32.json')[:8]
    write(OUT/'manifests/e5_training_meta.json',read(OLD/'manifests/e5_training_meta.json'))
    results=[]
    seed_all(42);r=runtime('A',namespace='p0/A_runtime')
    try:
        for row in rows:r.generate(row,OUT/'p0/A'/(row['id']+'.png'))
    finally:release(r)
    seed_all(42);r=runtime('B',namespace='p0/off_runtime');r.pipe.use_tcpm_lite=False
    try:
        for row in rows:
            path=OUT/'p0/E5_off'/(row['id']+'.png');record,_,_=r.generate(row,path)
            original=OUT/'p0/A'/path.name
            trace=exact_trace(path.with_suffix('.pt'),original.with_suffix('.pt'))
            results.append(dict(id=row['id'],png_exact=sha(path)==sha(original),**trace))
    finally:release(r)
    write(OUT/'off_replay.json',dict(n=8,checks=results,passed=all(q['png_exact'] and q['bit_exact'] for q in results)))
    assert read(OUT/'off_replay.json')['passed'],'E5 完整 TCPM OFF 不能复现 E2b'
    replay=[];seed_all(42);r=runtime('B',True,'p0/D_runtime');r.load(B1)
    try:
        for row in rows:
            path=OUT/'p0/D'/(row['id']+'.png');rec,_,_=r.generate(row,path)
            historical=OLD/'p1/images/B1'/path.name
            trace=exact_trace(path.with_suffix('.pt'),historical.with_suffix('.pt'))
            replay.append(dict(id=row['id'],png_exact=sha(path)==sha(historical),mask=rec['mask'],**trace))
        write(OUT/'p0/b1_shapes.json',r.processor.trace)
        assert all(q['target_shape']==[1,768,640] and q['reference_shape'][1]==256 for q in r.processor.trace)
    finally:release(r)
    write(OUT/'p0/D_replay.json',dict(n=8,checks=replay,passed=all(q['png_exact'] and q['bit_exact'] for q in replay)))
    assert read(OUT/'p0/D_replay.json')['passed'],'历史 D 未能完整回放'
    seed_all(42);r=runtime('A',True,'p0/C_runtime')
    checks=[];cfg={}
    try:
        assert sum(p.numel() for p in r.parameters())==820481
        diagnostic=next(i for i,row in enumerate(rows) if sketch_masks(Image.open(row['sketch']).convert('RGB').resize(SIZE),'cpu')[2]['valid'])
        for i,row in enumerate(rows):
            r.active=False
            off,contexts,outputs=r.generate(row,OUT/'p0/C'/(row['id']+'_off.png'),capture=i==diagnostic)
            r.active=True
            zero,_,_=r.generate(row,OUT/'p0/C'/(row['id']+'_zero.png'))
            original=OUT/'p0/A'/(row['id']+'.png')
            values=dict(id=row['id'],off_png=off['png_sha256']==sha(original),zero_png=zero['png_sha256']==sha(original),
                off_trace=exact_trace(OUT/'p0/C'/(row['id']+'_off.pt'),original.with_suffix('.pt')),
                zero_trace=exact_trace(OUT/'p0/C'/(row['id']+'_zero.pt'),original.with_suffix('.pt')))
            assert values['off_png'] and values['zero_png'] and values['off_trace']['bit_exact'] and values['zero_trace']['bit_exact']
            checks.append(values)
            if i==diagnostic:
                with torch.no_grad():
                    r.processor.gate.fill_(.1)
                    for branch in ['conditional','unconditional']:
                        args,kw=contexts[branch];pred=r.pipe.unet(*args,**kw)[0]
                        delta=float((pred.float()-outputs[branch].float()).abs().mean())
                        assert delta>0 if branch=='conditional' else delta==0
                        cfg[branch]=delta
                    r.processor.gate.zero_()
        write(OUT/'p0/C_off_zero.json',checks);write(OUT/'p0/C_cfg.json',cfg)
        data=TrainingInputs(r);seed_all(42)
        optimizer=torch.optim.AdamW(r.parameters(),lr=1e-4,weight_decay=.01)
        scaler=torch.amp.GradScaler('cuda');reports=[]
        for step in range(1,21):
            while True:
                args,kw,noise,mask,record=data.next(disable_dropout=True)
                if r.mask_info['valid']:break
            optimizer.zero_grad(set_to_none=True)
            loss,stats=loss_forward(r,args,kw,noise,mask)
            assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(optimizer)
            grad=gradients(r);assert grad['gate'] is not None and grad['gate']>0
            if step>=5:assert grad['to_k_ref.weight']>0 and grad['to_v_ref.weight']>0
            if step in [1,5,20]:reports.append(dict(step=step,gradients=grad,input=record,**stats))
            torch.nn.utils.clip_grad_norm_(r.parameters(),1.);scaler.step(optimizer);scaler.update()
        write(OUT/'p0/C_gradient_audit.json',reports)
    finally:release(r)
    write(OUT/'p0/smoke_report.json',dict(passed=True,off_exact=True,D_replay_exact=True,
        C_off_zero_exact=True,conditional_isolation=True,gradient_pass=True,base_writer_unchanged=True))
    decision(P0='pass',baseline='E2b',causal_claim=True,P1='not_run')
    print('e47 P0 PASS',flush=True)


def train():
    assert read(OUT/'decision.json')['P0']=='pass'
    assert read(OUT/'p0/smoke_report.json')['passed']
    folder=OUT/'p1/C';assert not (folder/'complete.json').exists(),'禁止重复训练挑选结果'
    seed_all(42);r=runtime('A',True,'p1/C/runtime');data=TrainingInputs(r)
    seed_all(42);optimizer=torch.optim.AdamW(r.parameters(),lr=1e-4,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda');sequence=[];history=[];step=0;empty=0
    started=time.perf_counter();r.save(folder/'checkpoints/step0000.pt',0,optimizer,scaler)
    try:
        while step<800:
            optimizer.zero_grad(set_to_none=True);items=[]
            for micro in range(8):
                args,kw,noise,mask,record=data.next();sequence.append(record)
                loss,item=loss_forward(r,args,kw,noise,mask);assert torch.isfinite(loss)
                if loss.requires_grad:scaler.scale(loss/8).backward()
                items.append(item)
            if not any(p.grad is not None for p in r.parameters()):empty+=1;continue
            scaler.unscale_(optimizer);norm=float(torch.nn.utils.clip_grad_norm_(r.parameters(),1.))
            assert torch.isfinite(torch.tensor(norm))
            before=scaler.get_scale();scaler.step(optimizer);scaler.update()
            assert scaler.get_scale()>=before,'发生跳过更新，不能声称同预算'
            step+=1
            if step in [1,5,20] or step%25==0:
                item=dict(step=step,grad_norm=norm,gradients=gradients(r) if step in [1,5,20,800] else None,
                    **{k:sum(x[k] for x in items)/8 for k in items[0]},seconds=time.perf_counter()-started)
                history.append(item);write(folder/'history.json',history)
                print('e47 C TRAIN',step,item['diffusion'],item['keep'],flush=True)
            if step in [100,400,800]:r.save(folder/'checkpoints'/('step%04d.pt'%step),step,optimizer,scaler)
        write(folder/'random_sequence.json',sequence)
        historical=read(OLD/'p1/B1/complete.json')['sequence_sha256']
        current=sha(folder/'random_sequence.json');assert current==historical,'与历史 D 的噪声/样本/增强顺序不一致'
        write(folder/'complete.json',dict(updates=step,empty_windows=empty,parameters=820481,
            sequence_sha256=current,historical_sequence_sha256=historical,seconds=time.perf_counter()-started,
            peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,teacher='A with B1 OFF',commit=commit()))
    finally:release(r)
    decision(P1='complete',P2='not_run')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['smoke','train','generate','evaluate'])
    parser.add_argument('--arm',choices=['A','B','C','D','D_safe'],default='A');args=parser.parse_args()
    torch.set_num_threads(2)
    try:
        if args.action=='smoke':smoke()
        elif args.action=='train':train()
        else:
            from tools.e47_eval import generate,evaluate
            if args.action=='generate':generate(args.arm)
            else:evaluate()
    except Exception as exc:
        decision(engineering_error=repr(exc));raise


if __name__=='__main__':main()
