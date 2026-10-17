"""后备真实GPU门禁/固定更新训练/生成；不改变旧轻量实验。"""
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from tools.m54_protocol import *
from tools.m54_run import Inputs,seed_all
from tools.m54_fallback_runtime import ARMS,REFERENCE_NOISE_SEED,make_runtime
from tools.m54_runtime import release


def require_protocol():
    protocol=read(OUT/'fallback/protocol.json');policy=read(OUT/'fallback/selection/pilot_policy.json')
    assert protocol==read('configs/m54_fallback_protocol_r1.json')
    assert policy['protocol_sha256']==sha(OUT/'fallback/protocol.json')
    assert policy['source_policy_sha256']==sha(OUT/'selection/pilot_policy.json')
    assert policy['split_sha256']==sha(split_path('pilot')) and policy['before_candidate_outputs']
    assert policy['arms']==ARMS and policy['maximum_additional_effective_updates']==4800
    assert policy['thresholds']==dict(mean_delta_lpips_strictly_less_than=0.,mean_delta_f1_greater_equal=-.05,tail_strictly_less_than=3)
    assert sha(E5)==E5_SHA and sha(B1)==B1_SHA,'FROZEN_SOURCE_CHANGED'
    return protocol


def loss_forward(r,args,kw,noise,mask):
    r.active=False
    try:
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):teacher=r.pipe.unet(*args,**kw).sample
    finally:r.active=True
    with torch.autocast('cuda',dtype=torch.float16):prediction=r.pipe.unet(*args,**kw).sample
    diffusion=F.mse_loss(prediction.float(),noise.float());outside=1-mask.float()
    keep=((prediction.float()-teacher.float()).square()*outside).sum()/outside.sum().clamp_min(1e-8)
    smooth=r.smoothness if r.arm=='M_L2' and r.smoothness is not None else diffusion.new_zeros(())
    loss=diffusion+.1*keep+.01*smooth
    return loss,dict(diffusion=float(diffusion.detach()),keep=float(keep.detach()),flow_smoothness=float(smooth.detach()))


def audit():
    from tools.m54_runtime import make_runtime as original_runtime
    from tools.e47_runtime import exact_trace
    require_protocol();path=OUT/'fallback/audit/round0_complete.json'
    if path.exists():assert read(path)['passed'];return
    freeze(OUT/'fallback/implementation.json',dict(reference_noise_seed=REFERENCE_NOISE_SEED,
        master_parameter_dtype='float32',branch_compute_dtype='float16 autocast',
        latent_condition='frozen VAE posterior mean',scientific_updates_at_record=0,
        implementation_sources={str(p):sha(p) for p in [Path('models/m54_fallback.py'),Path('tools/m54_fallback_runtime.py')]},
        protocol_sha256=sha(OUT/'fallback/protocol.json')))
    rows=read(split_path('dev32'))[:2];replay={};modules={}
    for arm in ['E5','B1']:
        r=original_runtime(arm,'fallback/audit/replay/'+arm)
        try:
            entries=[]
            for row in rows:
                target=image_path('fallback/audit/replay',arm,row['id']);r.generate(row,target)
                source=image_path('dev32_u2400',arm,row['id'])
                result=exact_trace(target.with_suffix('.pt'),source.with_suffix('.pt'))
                assert result['bit_exact'] and sha(target)==sha(source),('SOURCE_REPLAY_FAILED',arm,row['id'])
                entries.append(dict(id=row['id'],trace=result,png_exact=True))
            replay[arm]=entries
        finally:release(r)
    for arm in ARMS:
        seed_all();r=make_runtime(arm,'fallback/audit/initial/'+arm)
        try:
            initial=[];profiles=[]
            for row in rows:
                target=image_path('fallback/audit/initial',arm,row['id']);r.generate(row,target)
                source=image_path('fallback/audit/replay','E5',row['id'])
                result=exact_trace(target.with_suffix('.pt'),source.with_suffix('.pt'))
                assert result['bit_exact'] and sha(target)==sha(source),('FALLBACK_INITIAL_NOT_EXACT',arm,row['id'],result)
                initial.append(dict(id=row['id'],trace=result,png_exact=True))
                target2=image_path('fallback/audit/profile',arm,row['id']);profiles.append(r.profile(row,target2))
                assert sha(target2)==sha(target),'PRODUCTION_PROFILE_CHANGED_INITIAL_PNG'
        finally:release(r)
        # 独立实例：推理模式缓存不进入训练反向图；诊断不执行optimizer更新。
        seed_all();r=make_runtime(arm,'fallback/audit/gradients/'+arm)
        try:
            data=Inputs(r);seed_all();r.candidate.train()
            for _ in range(32):
                args,kw,noise,mask,record=data.next(disable_dropout=True)
                if record['mask_valid']:break
            else:raise AssertionError('NO_VALID_TRAINING_MASK')
            assert list(args[0].shape)==[1,4,64,48]
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                r.active=False;baseline=r.pipe.unet(*args,**kw).sample
                r.active=True;prediction=r.pipe.unet(*args,**kw).sample
            assert torch.equal(baseline,prediction),'INITIAL_TRAINING_EPSILON_NOT_EXACT'
            def gradients():
                loss,_=loss_forward(r,args,kw,noise,mask);(loss*1024).backward()
                return {n:float(p.grad.float().norm()/1024) if p.grad is not None else None for n,p in r.candidate.named_parameters()}
            zero=gradients();assert all(v is None or np.isfinite(v) for v in zero.values())
            if arm=='M_L1':assert all(v is not None and v>0 for v in zero.values()),('CAT_INITIAL_DEAD',zero)
            else:assert zero['output.weight']>0
            r.candidate.zero_grad(set_to_none=True)
            if arm=='M_L2':
                with torch.no_grad():
                    r.candidate.output.weight.normal_(std=.001)
                    for head in r.candidate.flow.heads:head[-1].weight.normal_(std=.001)
            live=gradients()
            assert all(v is not None and np.isfinite(v) and v>0 for v in live.values()),('FALLBACK_DEAD_PATH',arm,live)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                r.active=False;off=r.pipe.unet(*args,**kw).sample;assert torch.equal(off,baseline)
                r.active=True;r.allow_reference=False;reference_off=r.pipe.unet(*args,**kw).sample
                assert torch.equal(off,reference_off)
            modules[arm]=dict(initial=initial,production_profile=profiles,zero_output_gradients=zero,
                live_gradients=live,parameters=sum(p.numel() for p in r.parameters()),input_shape=list(args[0].shape),
                off_exact=True,reference_off_exact=True,optimizer_updates=0,training_identity=record['id'])
        finally:release(r)
        write(OUT/'fallback/audit'/(arm+'_round0.json'),modules[arm])
        print('FALLBACK GPU MODULE PASS',arm,flush=True)
    write(path,dict(passed=True,replay=replay,modules=modules,commit=commit(),training_updates=0,
        protocol_sha256=sha(OUT/'fallback/protocol.json'),policy_sha256=sha(OUT/'fallback/selection/pilot_policy.json'),
        reference_noise_seed=REFERENCE_NOISE_SEED))
    decision(fallback='ROUND0_PASS',fallback_effective_updates=0,complete=False)


def train(arm,total=800):
    require_protocol();assert arm in ARMS and total in [800,2400]
    audit=read(OUT/'fallback/audit/round0_complete.json');assert audit['passed']
    require_input_review('pilot')
    if total==2400:assert arm in read(OUT/'fallback/selection/pilot_selected.json')['selected']
    folder=OUT/'training'/arm;done=folder/('complete%04d.json'%total)
    if done.exists():assert read(done)['updates']==total;return
    seed_all();r=make_runtime(arm,'training/'+arm+'/runtime');data=Inputs(r)
    assert sum(p.numel() for p in r.parameters())==audit['modules'][arm]['parameters']
    optimizer=torch.optim.AdamW(r.parameters(),lr=1e-4,weight_decay=.01);scaler=torch.amp.GradScaler('cuda')
    step=0;sequence=[];history=[];r.candidate.train();started=time.perf_counter()
    try:
        if total==2400:
            state=r.load(folder/'checkpoints/step0800.pt');assert state['step']==800
            step=800;optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
            resume=state['resume'];data.data.order=resume['order'];data.data.rng.setstate(resume['data_rng']);data.count=resume['count']
            random.setstate(resume['python_rng']);np.random.set_state(resume['numpy_rng']);torch.set_rng_state(resume['torch_rng']);torch.cuda.set_rng_state_all(resume['cuda_rng'])
            sequence=read(folder/'random_sequence0800.json');history=read(folder/'history.json')
            assert data.count==6400 and len(sequence)==6400
        else:
            assert not (folder/'checkpoints/step0000.pt').exists(),'PRESERVE_FAILED_ATTEMPT_BEFORE_RETRY'
            r.save(folder/'checkpoints/step0000.pt',0,optimizer,scaler);seed_all()
        windows={n:0 for n,_ in r.candidate.named_parameters()}
        while step<total:
            optimizer.zero_grad(set_to_none=True);items=[]
            for _ in range(8):
                args,kw,noise,mask,record=data.next();sequence.append(record)
                loss,item=loss_forward(r,args,kw,noise,mask);assert torch.isfinite(loss)
                if loss.requires_grad:scaler.scale(loss/8).backward()
                items.append(item)
            assert any(p.grad is not None for p in r.parameters()),'EMPTY_EFFECTIVE_UPDATE'
            scaler.unscale_(optimizer);norm=float(torch.nn.utils.clip_grad_norm_(r.parameters(),1.));assert np.isfinite(norm)
            for name,value in r.candidate.named_parameters():
                if value.grad is not None and float(value.grad.float().norm())>0:windows[name]+=1
            before=scaler.get_scale();scaler.step(optimizer);scaler.update();assert scaler.get_scale()>=before,'OPTIMIZER_UPDATE_SKIPPED'
            step+=1
            if step in [1,5,20] or step%25==0:
                item=dict(step=step,gradient_norm=norm,seconds=time.perf_counter()-started,
                    **{k:sum(x[k] for x in items)/8 for k in items[0]})
                history.append(item);write(folder/'history.json',history);print('M54 FALLBACK TRAIN',arm,step,item,flush=True)
            if step in [100,400,800,2400]:
                path=folder/'checkpoints'/('step%04d.pt'%step);r.save(path,step,optimizer,scaler)
                state=torch.load(path,map_location='cpu',weights_only=False)
                state['resume']=dict(order=data.data.order,data_rng=data.data.rng.getstate(),count=data.count,
                    python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
                torch.save(state,path);write(path.with_suffix('.json'),dict(step=step,sha256=sha(path),commit=commit()))
        assert all(windows.values()),('GRADIENT_COLLAPSE',arm,windows)
        assert len(sequence)==8*total
        write(folder/('random_sequence%04d.json'%total),sequence)
        write(done,dict(updates=step,additional_updates=800 if total==800 else 1600,nonzero_gradient_windows=windows,
            parameters=sum(p.numel() for p in r.parameters()),sequence_sha256=sha(folder/('random_sequence%04d.json'%total)),
            seconds=time.perf_counter()-started,peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,commit=commit()))
    finally:release(r)


def generate(arm,stage='fb_pilot',updates=800,seed=42):
    require_protocol();assert arm in ARMS
    assert read(OUT/'fallback/audit/round0_complete.json')['passed']
    require_input_review(stage);assert read(OUT/'training'/arm/('complete%04d.json'%updates))['updates']==updates
    rows=read(split_path(stage));folderstage=result_stage(stage,updates,seed);done=OUT/folderstage/(arm+'_complete.json')
    if done.exists():assert read(done)['n']==len(rows) and read(done)['seed']==seed;return
    r=make_runtime(arm,folderstage+'/'+arm+'/runtime')
    try:
        assert r.load(OUT/'training'/arm/'checkpoints'/('step%04d.pt'%updates))['step']==updates
        r.candidate.eval().requires_grad_(False)
        for row in rows:
            path=image_path(folderstage,arm,row['id'])
            if path.exists() and path.with_suffix('.json').exists():assert sha(path)==read(path.with_suffix('.json'))['png_sha256'];continue
            r.generate(row,path,seed=seed);print('FALLBACK RGB',folderstage,arm,row['id'],flush=True)
        profiles=[]
        for row in rows[:2]:
            path=image_path('profiles/'+folderstage,arm,row['id']);profiles.append(r.profile(row,path,seed=seed))
            assert sha(path)==sha(image_path(folderstage,arm,row['id'])),'PRODUCTION_PROFILE_CHANGED_PNG'
        write(OUT/folderstage/(arm+'_profile.json'),dict(n=2,full50_steps=True,diagnostic_teacher_disabled=True,
            epsilon_trace_hooks_disabled=True,image_serialization_excluded=True,records=profiles))
    finally:release(r)
    write(done,dict(n=len(rows),arm=arm,updates=updates,components=[dict(arm=arm,updates=updates)],seed=seed,
        split_sha256=sha(split_path(stage)),commit=commit()))
