"""S2a/S2b：同初值、同退化、同预算配对训练；仅 dev Laplacian 选择检查点。"""
import argparse
import time
import cv2
import numpy as np
import torch
from models.e34_sarr_refiner import SARR
from tools.e34_sarr_protocol import *
from tools.e34_sarr_training import batch,dev_seed,lap_loss,masked,losses

@torch.inference_mode()
def evaluate(models,dev,dest,metric,dtype):
    records=[]
    for row in dev:
        (image,gt,mask,line,ref),info=batch([row],dev_seed(row['id']))
        base=float(lap_loss(image,gt,mask));scores={}
        for name,model in models.items():
            model.eval()
            with torch.autocast('cuda',dtype=dtype):pred=model(image,line,mask,ref)
            lap=float(lap_loss(pred['output'],gt,mask))
            identity=model(gt,line,mask,ref)['output']
            scores[name]=dict(lap=lap,identity_change=float(masked((identity-gt).abs(),mask)),
                outside_max=float(((pred['output']-image).abs()*(mask==0)).max()))
            assert scores[name]['outside_max']==0
            if name=='full':
                with torch.autocast('cuda',dtype=dtype):off=model(image,line,mask,ref,reference_off=True)['output']
                scores[name].update(reference_off_lap=float(lap_loss(off,gt,mask)),
                    on_off_MAE=float(masked((pred['output']-off).abs(),mask)))
        records.append(dict(id=row['id'],valid=info[0]['mask_valid'],degradation=info[0]['degradation'],base_lap=base,methods=scores))
    names=list(models);summary={name:dict(lap=bootstrap([r['methods'][name]['lap'] for r in records]),
        gain=bootstrap([(r['base_lap']-r['methods'][name]['lap'])/max(r['base_lap'],1e-8) if r['base_lap']>1e-8 else 0 for r in records]),
        identity_change=bootstrap([r['methods'][name]['identity_change'] for r in records])) for name in names}
    if 'full' in models and 'no_ref' in models:
        gain=[(r['methods']['no_ref']['lap']-r['methods']['full']['lap'])/max(r['methods']['no_ref']['lap'],1e-8)
            if r['methods']['no_ref']['lap']>1e-8 else 0 for r in records]
        summary['full_vs_no_ref']=bootstrap(gain)
        summary['reference_on_vs_off']=bootstrap([r['methods']['full']['reference_off_lap']-r['methods']['full']['lap'] for r in records])
    summary.update(n=128,valid_n=sum(r['valid'] for r in records),statistics_unit='identity; invalid/identity degradation no-improvement retained')
    write(dest/'identity_metrics.json',records);write(dest/'summary.json',summary)
    print('SYNTHETIC',str(dest),json.dumps(summary),flush=True)
    return summary

def run(phase):
    init();seed_all(42);cv2.setNumThreads(1);torch.set_num_threads(2)
    state=read(OUT/'final_decision.json')
    assert state['S1'] in ['pass','pass_user_exception']
    assert state['S0']=='pass', '须先完成真实 E5 固定24质量审计'
    if phase=='S2b':assert state['S2a']=='pass'
    train=read(OUT/'splits'/('pilot256.json' if phase=='S2a' else 'train.json'));dev=read(OUT/'splits/dev128.json')
    from tools.e33tmoc_appearance_eval import frozen_lpips,model_hash
    metric,info=frozen_lpips();metric=metric.cuda().eval();metric_hash=model_hash(metric)
    full=SARR().cuda();no_ref=SARR(True).cuda();no_ref.load_state_dict(full.state_dict())
    models=dict(full=full,no_ref=no_ref)
    assert sum(p.numel() for p in full.parameters())==sum(p.numel() for p in no_ref.parameters())
    optimizers={name:torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-4) for name,m in models.items()}
    updates=1500 if phase=='S2a' else 8000;interval=500 if phase=='S2a' else 2000
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scalers={name:torch.cuda.amp.GradScaler(enabled=dtype==torch.float16) for name in models}
    folder=OUT/phase;folder.mkdir(parents=True,exist_ok=True)
    write(folder/'implementation.json',dict(git_commit=commit(),updates=updates,train_n=len(train),
        dtype=str(dtype),microbatch=2,accumulation=4,effective_batch=8,lpips=info,lpips_state_hash=metric_hash,
        same_initialization=True,same_batch_order=True,same_degradation=True,no_ref_input='I_in resized128',
        selection='fixed1500 for S2a; minimum dev masked Laplacian at pre-fixed checkpoints for S2b',
        initial_from='fresh seed42 initialization, not pilot checkpoint',E5_loaded=False))
    rng=np.random.default_rng(42);history=[];gradient_audit={};best={name:(float('inf'),0) for name in models}
    def checkpoint(step):
        dest=OUT/'checkpoints'/phase;dest.mkdir(parents=True,exist_ok=True)
        path=dest/('step%d.pt'%step)
        torch.save(dict(step=step,phase=phase,models={k:m.state_dict() for k,m in models.items()},
            optimizers={k:o.state_dict() for k,o in optimizers.items()},git_commit=commit()),path)
        summary=evaluate(models,dev,OUT/'s2_synthetic_dev'/phase/('step%d'%step),metric,dtype)
        for name in models:
            score=summary[name]['lap']['mean']
            if score<best[name][0]:best[name]=(score,step)
        write(folder/'checkpoint_selection.json',{name:dict(dev_lap=v[0],step=v[1]) for name,v in best.items()})
        return summary
    checkpoint(0);start=time.perf_counter()
    for update in range(1,updates+1):
        if phase=='S2b':
            lr=1e-5+(1e-4-1e-5)*.5*(1+np.cos(np.pi*(update-1)/updates))
            for o in optimizers.values():
                for group in o.param_groups:group['lr']=float(lr)
        for model in models.values():model.train()
        averages={name:{} for name in models}
        for micro in range(4):
            selected=rng.choice(len(train),2,replace=False);seed=int(rng.integers(0,2**32-1))
            (image,gt,mask,line,ref),case_info=batch([train[i] for i in selected],seed)
            for name,model in models.items():
                with torch.autocast('cuda',dtype=dtype):pred=model(image,line,mask,ref)
                # 感知/频带/像素损失一律 FP32，保留 AMP 输出到 FP32 的梯度。
                with torch.autocast('cuda',enabled=False):loss,parts=losses(pred,gt,image,mask,metric)
                assert torch.isfinite(loss) and all(torch.isfinite(v) for v in parts.values())
                scalers[name].scale(loss/4).backward()
                for k,v in dict(parts,total=loss).items():averages[name][k]=averages[name].get(k,0)+float(v.detach())/4
        for name,model in models.items():
            scalers[name].unscale_(optimizers[name])
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            if update<=5:
                modules=dict(reference=model.reference,film=model.film1,delta=model.delta,gate=model.gate,refiner=model.first)
                gradient_audit.setdefault(name,[]).append(dict(update=update,**{k:float(sum(
                    p.grad.detach().float().abs().sum() for p in module.parameters() if p.grad is not None)) for k,module in modules.items()}))
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scalers[name].step(optimizers[name]);scalers[name].update();optimizers[name].zero_grad(set_to_none=True)
        if update==5:
            write(folder/'gradient_audit.json',gradient_audit)
            assert all(any(r[k]>0 for r in gradient_audit[name][1:]) for name in models for k in ['reference','film','delta','gate','refiner'])
        history.append(dict(update=update,seconds=time.perf_counter()-start,losses=averages))
        if update%25==0:
            print('TRAIN',phase,update,'/',updates,'seconds',round(time.perf_counter()-start,1),'losses',averages,flush=True)
            write(folder/'training_history.json',history)
        if update%interval==0:summary=checkpoint(update)
    write(folder/'training_history.json',history)
    selected={name:updates if phase=='S2a' else best[name][1] for name in models}
    for name,step in selected.items():
        ckpt=torch.load(OUT/'checkpoints'/phase/('step%d.pt'%step),map_location='cuda')
        models[name].load_state_dict(ckpt['models'][name])
    summary=evaluate(models,dev,OUT/'s2_synthetic_dev'/phase/'selected',metric,dtype)
    gain=summary['full']['gain']['mean'];extra=summary['full_vs_no_ref']
    passed=gain>=.08 and extra['mean']>=.03 and extra['ci95'][0]>0
    assert model_hash(metric)==metric_hash
    verify_sources()
    write(folder/'final.json',dict(pass_gate=passed,selected=selected,synthetic_gain=gain,
        additional_gain_vs_no_ref=extra,updates=updates,elapsed_seconds=time.perf_counter()-start,
        peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3,
        checkpoints={name:dict(path=str(OUT/'checkpoints'/phase/('step%d.pt'%step)),
            sha256=sha(OUT/'checkpoints'/phase/('step%d.pt'%step))) for name,step in selected.items()}))
    decision(**{phase:'pass' if passed else 'fail'},next_phase=('S2b' if phase=='S2a' else 'S3') if passed else 'stop_E34_SARR',
        stopped=not passed,reason=None if passed else 'S2 synthetic reference increment gate failed')
    print('S2_FINAL',phase,passed,gain,extra,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['S2a','S2b'],default='S2a');a=p.parse_args();run(a.phase)
