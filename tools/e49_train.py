"""先验收编码与 E5 精确回放，再进行两个固定 800 更新的实验。"""
import argparse
import gc
import time
import platform
import numpy as np
import torch
from tools.e49_protocol import *
from tools.e49_runtime import make_runtime, close
from tools.e34_sarr_protocol import seed_all
from tools.e44_train import TrainingInputs, loss_forward, gradients


def inputs(r):
    from tools import e44_train as module
    old=module.OUT; module.OUT=OUT
    try: return TrainingInputs(r,'train1024')
    finally: module.OUT=old


def zero_gate_audit(r, arm):
    from tools.e47_runtime import exact_trace
    from tools.e44_runtime import digest
    folder=OUT/'audit'/arm/'zero_gate'
    assert float(r.processor.gate)==0
    before={k:digest(v) for k,v in r.processor.trainable_state().items()}
    results=[]; r.processor.eval(); r.style_encoder.training=False
    for row in read(OUT/'splits/probe.json')[:8]:
        path=folder/'images'/(row['id']+'.png')
        record,_,_=r.generate(row,path)
        baseline=SOURCE/'probe/images/B'/(row['id']+'.png')
        trace=exact_trace(path.with_suffix('.pt'),baseline.with_suffix('.pt'))
        assert sha(path)==sha(baseline) and trace['bit_exact'],(arm,row['id'],trace)
        results.append(dict(id=row['id'],png_exact=True,trace=trace,encoder=r.style_encoder.last_audit,
            seconds=record['seconds']))
    assert before=={k:digest(v) for k,v in r.processor.trainable_state().items()}
    write(folder/'complete.json',dict(passed=True,n=len(results),results=results,commit=commit()))


def encoder_audit(r):
    from models.e49_style_encoder import FrozenStyleEncoder
    from tools.e44_runtime import digest
    v=r.pipe.image_encoder; e=v.vision_model.embeddings
    generator=torch.Generator(device='cpu').manual_seed(49001)
    pixels=torch.randn((3,3,v.config.image_size,v.config.image_size),generator=generator).to(
        device=r.pipe.device,dtype=e.patch_embedding.weight.dtype)
    normal=FrozenStyleEncoder(v,'CLIP'); shuffled=FrozenStyleEncoder(v,'SSVM')
    n=e.position_embedding.weight.shape[0]-1
    permutation=torch.stack([torch.randperm(n,generator=generator) for _ in range(3)])
    original_forward=v.vision_model.embeddings.forward
    with torch.no_grad():
        baseline=v(pixels,output_hidden_states=True).last_hidden_state
        ordinary=normal.encode_pixels(pixels)
        changed=shuffled.encode_pixels(pixels,permutation)
        ordered=shuffled.encode_pixels(pixels,torch.arange(n).repeat(3,1))
    assert torch.equal(baseline,ordinary),'普通编码必须完整重现原 CLIP forward'
    assert changed.shape==(3,n,v.config.hidden_size)
    assert float((changed-ordered).float().abs().max())>1e-5
    assert v.vision_model.embeddings.forward==original_forward
    # SSVM 所有编码改动均为局部计算，共享 E5 CLIP 的完整 forward 不能改变。
    with torch.no_grad(): assert torch.equal(baseline,v(pixels,output_hidden_states=True).last_hidden_state)
    cpu_before=torch.get_rng_state().clone(); cuda_before=torch.cuda.get_rng_state().clone()
    raw=pixels[:1].float().tanh(); normal(raw); a=dict(normal.last_audit)
    shuffled(raw); b=dict(shuffled.last_audit)
    assert a['crop_boxes']==b['crop_boxes'] and a['position_permutation_sha256']==b['position_permutation_sha256']
    assert torch.equal(cpu_before,torch.get_rng_state()) and torch.equal(cuda_before,torch.cuda.get_rng_state())
    # K/V 共同重排不应改变无位置读取，证明不是“编码后换个顺序”的伪消融。
    p=r.processor.style_embed
    f=changed.reshape(1,-1,changed.shape[-1]).float()
    index=torch.randperm(f.shape[1],generator=generator).to(f.device)
    with torch.no_grad():
        one=p(f); two=p(f[:,index])
    maxdelta=float((one-two).abs().max())
    assert torch.allclose(one,two,atol=2e-5,rtol=2e-5),maxdelta
    return dict(ordinary_clip_bit_exact=True,cls_removed=True,shuffled_before_encoder=True,
        actual_positional_change_max=float((changed-ordered).float().abs().max()),
        shared_forward_unchanged=True,global_rng_unchanged=True,crops_identical=True,
        post_encoding_permutation_max_delta=maxdelta,clip_width=v.config.hidden_size,
        patch_tokens=n,trainable_initial_sha256={k:digest(t) for k,t in r.processor.trainable_state().items()})


def audit():
    prepare()
    assert not (OUT/'audit/complete.json').exists(),'禁止重复完成的验收'
    initial={}; results={}
    for arm in TRAIN_ARMS:
        seed_all(42); r=make_runtime(arm,'audit/'+arm+'/runtime')
        try:
            result=encoder_audit(r); initial[arm]=result.pop('trainable_initial_sha256')
            zero_gate_audit(r,arm); results[arm]=result
            # 验收梯度流和 OFF 隔离；不保留任何候选 checkpoint。
            r.processor.train(); r.style_encoder.training=True; r.style_encoder.reset()
            data=inputs(r); seed_all(42)
            for _ in range(16):
                args,kw,noise,mask,record=data.next(disable_dropout=True)
                if record['mask_valid']: break
            else: raise AssertionError('没有可验收的有效 Sketch')
            r.processor.gate.data.fill_(.01)
            r.pipe.unet.zero_grad(set_to_none=True)
            loss,_=loss_forward(r,args,kw,noise,mask)
            loss.backward()
            names=['gate','to_k_ref.weight','to_v_ref.weight','style_embed.proj_in.weight','style_embed.latents']
            grads=gradients(r)
            assert all(grads.get(n) is not None and np.isfinite(grads[n]) and grads[n]>0 for n in names),grads
            results[arm]['gradient_flow']={n:grads[n] for n in names}
            r.active=True
            from tools.e49_gate_audit import off_isolation
            results[arm]['off_isolation']=off_isolation(r,args,kw)
        finally:
            close(r); del r; gc.collect(); torch.cuda.empty_cache()
    assert initial['CLIP']==initial['SSVM']
    write(OUT/'audit/complete.json',dict(passed=True,arms=results,identical_trainable_initialization=True,
        zero_gate_exact_cases_per_arm=8,environment=dict(python=platform.python_version(),torch=torch.__version__,
            cuda=torch.version.cuda,device=torch.cuda.get_device_name(0)),commit=commit()))
    decision(audit='pass')


def train(arm):
    from tools.e22_4_generation import module_hashes
    prepare(); assert read(OUT/'audit/complete.json')['passed']
    folder=OUT/'training'/arm
    assert not (folder/'checkpoints/step0000.pt').exists(),'重复训练须先记录并保留工程失败目录'
    seed_all(42); r=make_runtime(arm); data=inputs(r)
    r.processor.train(); r.style_encoder.training=True; r.style_encoder.reset()
    seed_all(42); torch.cuda.reset_peak_memory_stats()
    parameters=sum(p.numel() for p in r.parameters())
    optimizer=torch.optim.AdamW(r.parameters(),lr=1e-4,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda'); sequence=[]; history=[]; step=0; empty=0
    tracked=['to_k_ref.weight','to_v_ref.weight','style_embed.proj_in.weight','style_embed.latents']
    windows={n:0 for n in tracked}; named=dict(r.processor.named_parameters()); started=time.perf_counter()
    r.save(folder/'checkpoints/step0000.pt',0,optimizer,scaler)
    try:
        while step<800:
            optimizer.zero_grad(set_to_none=True); items=[]
            for _ in range(8):
                args,kw,noise,mask,record=data.next(); sequence.append(record)
                loss,item=loss_forward(r,args,kw,noise,mask)
                assert torch.isfinite(loss)
                if loss.requires_grad: scaler.scale(loss/8).backward()
                items.append(item)
            if not any(p.grad is not None for p in r.parameters()): empty+=1; continue
            scaler.unscale_(optimizer)
            norm=float(torch.nn.utils.clip_grad_norm_(r.parameters(),1.)); assert np.isfinite(norm)
            grad=gradients(r) if step+1 in [1,5,20,800] else None
            if step==0: assert grad['gate'] is not None and grad['gate']>0
            for n in tracked:
                value=named[n].grad
                if value is not None and float(value.float().norm())>0: windows[n]+=1
            if step+1 in [20,800]: assert all(windows.values()),windows
            scale=scaler.get_scale(); scaler.step(optimizer); scaler.update(); assert scaler.get_scale()>=scale
            step+=1
            if step in [1,5,20] or step%25==0:
                item=dict(step=step,grad_norm=norm,gradients=grad,gradient_windows=dict(windows),
                    seconds=time.perf_counter()-started,**{k:sum(x[k] for x in items)/8 for k in items[0]})
                history.append(item); write(folder/'history.json',history)
                print('E49 TRAIN',arm,step,item['diffusion'],item['keep'],flush=True)
            if step in [100,400,800]: r.save(folder/'checkpoints'/('step%04d.pt'%step),step,optimizer,scaler)
        write(folder/'random_sequence.json',sequence)
        expected=CONFIG['training']['expected_sequence_sha256']
        assert empty==0 and len(sequence)==6400 and sha(folder/'random_sequence.json')==expected
        target=dict(r.pipe.unet.named_modules())[LAYER[:-10]]; target.set_processor(r.original)
        try: assert module_hashes(r.modules)==r.before
        finally: target.set_processor(r.processor)
        write(folder/'complete.json',dict(arm=arm,updates=step,empty_windows=empty,parameters=parameters,
            sequence_sha256=expected,seconds=time.perf_counter()-started,
            peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,frozen_base_unchanged=True,
            nonzero_gradient_windows=windows,commit=commit()))
    finally:
        close(r); del r; gc.collect(); torch.cuda.empty_cache()


def generate(arm, stage):
    folder=OUT/'training'/arm
    assert read(folder/'complete.json')['updates']==800
    r=make_runtime(arm,stage+'/'+arm+'/runtime'); r.load(folder/'checkpoints/step0800.pt')
    r.processor.eval().requires_grad_(False); r.style_encoder.training=False
    rows=read(OUT/'splits'/('new96.json' if stage=='new96' else stage+'.json'))
    try:
        for row in rows:
            path=image_path(arm,row['id'],stage)
            if path.exists() and path.with_suffix('.json').exists():
                assert sha(path)==read(path.with_suffix('.json'))['png_sha256']; continue
            record,_,_=r.generate(row,path)
            write(path.with_name(path.stem+'_encoder.json'),r.style_encoder.last_audit if record['mask']['valid'] else
                dict(branch_off=True,reason='invalid Sketch mask'))
            print('E49 RGB',arm,stage,row['id'],record['seconds'],flush=True)
    finally:
        close(r); del r; gc.collect(); torch.cuda.empty_cache()
    write(OUT/stage/(arm+'_complete.json'),dict(arm=arm,n=len(rows),
        checkpoint_sha256=sha(folder/'checkpoints/step0800.pt'),commit=commit()))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['prepare','audit','run','train','generate','evaluate'])
    parser.add_argument('--arm',choices=TRAIN_ARMS); parser.add_argument('--stage',default='probe',choices=['probe','dev32','new96'])
    args=parser.parse_args(); torch.set_num_threads(2)
    if args.action=='prepare': prepare()
    elif args.action=='audit': audit()
    elif args.action=='train': train(args.arm)
    elif args.action=='generate': generate(args.arm,args.stage)
    elif args.action=='run': train(args.arm); generate(args.arm,'probe'); generate(args.arm,'dev32')
    else:
        from tools.e49_eval import evaluate_all
        evaluate_all()


if __name__=='__main__': main()
