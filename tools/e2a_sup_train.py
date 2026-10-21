"""在冻结历史源码中只插入附加 loss hook；不改旧训练文件或旧实验根。"""
import argparse
import ast
import contextlib
import csv
import hashlib
import inspect
import os
import sys
import time
import numpy as np
import torch
from tools.e2a_sup_common import *
from tools.e2a_sup_losses import Supervision, weighted


class CalibrationDone(Exception): pass


def rms(value): return float(value.detach().float().square().mean().sqrt())


class Hook:
    def __init__(self,arm,phase,run):
        self.arm,self.phase,self.run=arm,phase,run
        self.impl=None; self.rows=[]; self.pending=[]; self.parameter_probe={}; self.checked=False
        self.coefficients=read(ROOT/'loss_registry.json')[arm]['coefficients'] if phase!='calibrate' else None

    def setup(self,env):
        self.acc=env['accelerator']; self.runtime=env['runtime']; self.rank=self.acc.process_index
        self.models=[env[k] for k in ['unet','ref_unet','bf','spatial_texture_encoder','spatial_injection']]
        self.optimizer=env['optimizer']
        assert self.acc.num_processes==2
        assert not any(p.requires_grad for p in env['vae'].parameters())
        assert env['noise_scheduler'].config.prediction_type=='epsilon'
        opt=self.optimizer.param_groups[0]
        freeze(self.run/('environment_rank%d.json'%self.rank),dict(torch=torch.__version__,
            optimizer=type(self.optimizer.optimizer).__name__,betas=list(opt['betas']),eps=opt['eps'],
            weight_decay=opt['weight_decay'],vae_scaling_factor=float(env['vae'].config.scaling_factor),
            scheduler=dict(env['noise_scheduler'].config),trainable_parameters=sum(p.numel() for p in env['trainable_params']),
            arm=self.arm,phase=self.phase,main_rng_seed=42+self.rank))
        # 记录实际 trainable tensors，后续 aux 单项梯度验收不会只检查一个游离 RGB 张量。
        params=env['trainable_params']; self.probes=[params[i] for i in sorted({0,len(params)//2,len(params)-1})]
        self.probe_before=[p.detach().clone() for p in self.probes] if self.phase=='calibrate' else []
        original=self.runtime.log_update
        def log_update(step,optimizer,grad_norm):
            original(step,optimizer,grad_norm)
            if self.pending:
                keys=list(self.pending[0]); vector=torch.tensor([np.mean([r[k] for r in self.pending]) for k in keys],device=self.acc.device)
                values=self.acc.reduce(vector,reduction='mean').cpu().tolist()
                if self.acc.is_main_process:
                    self.append(self.run/'auxiliary_log.csv',dict(update=step,**dict(zip(keys,values)),
                        peak_gpu_bytes=torch.cuda.max_memory_allocated(),job=os.environ.get('SLURM_JOB_ID')))
                self.pending.clear()
        self.runtime.log_update=log_update

    @contextlib.contextmanager
    def context(self):
        with contextlib.ExitStack() as stack:
            if self.phase=='calibrate':
                for m in self.models: stack.enter_context(self.acc.no_sync(m))
            yield

    @staticmethod
    def append(path,row):
        exists=path.exists()
        with path.open('a',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(row))
            if not exists: writer.writeheader()
            writer.writerow(row)

    def __call__(self,env,namespace):
        if self.impl is None:
            # 未来的特征先验初始化也必须在独立 RNG 环境里完成，不改主训练流。
            import random
            state=random.getstate(); npstate=np.random.get_state()
            try:
                with torch.random.fork_rng(devices=[self.acc.device.index]):
                    self.impl=Supervision(self.arm,dict(namespace,**env))
            finally: random.setstate(state); np.random.set_state(npstate)
        pred,gt,reference=env['decoded_loss'],env['target_loss'],env['texture_target_loss']
        started=time.time()
        with torch.cuda.amp.autocast(enabled=False):
            raw=self.impl.raw(pred,gt,reference,env['mask_loss'],env['timesteps'],env['texture_condition_weight'])
            if not self.checked:
                self.selfcheck(env)
                self.checked=True
        assert all(torch.isfinite(v).all().item() for v in raw.values())
        base=env['loss']; epsilon=env['noise_pred']
        if self.phase=='calibrate':
            base_rms=rms(torch.autograd.grad(base,epsilon,retain_graph=True)[0]); assert base_rms>0
            row=dict(microbatch=len(self.rows)+1,index=int(env['batch']['dataset_index'][0]),
                timestep=int(env['timesteps'][0]),base_rms=base_rms,rng_chain=self.runtime.rng_chain)
            for name,value in raw.items():
                grad=torch.autograd.grad(value,epsilon,retain_graph=True)[0]
                assert torch.isfinite(grad).all().item()
                row[name+'_rms']=rms(grad); row[name+'_ratio']=rms(grad)/base_rms
                if row[name+'_rms']>0 and name not in self.parameter_probe:
                    params=torch.autograd.grad(value,self.probes,allow_unused=True,retain_graph=True)
                    values=[rms(p) for p in params if p is not None]
                    assert values and max(values)>0 and all(np.isfinite(values))
                    self.parameter_probe[name]=dict(gradient_rms=values,actual_trainable_tensor_count=len(values))
            row.update(body_fraction=float(env['mask_loss'].mean()),
                decoded_out_of_range=float(((pred.detach()<-1)|(pred.detach()>1)).float().mean()),
                x0_out_of_range=float(((env['x0_hat'].detach()<-1)|(env['x0_hat'].detach()>1)).float().mean()),
                auxiliary_seconds=time.time()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated())
            self.rows.append(row)
            if len(self.rows)==256: self.finish_calibration()
            return base
        aux=weighted(raw,self.coefficients)
        self.pending.append(dict(**{k:float(v.detach()) for k,v in raw.items()},weighted_aux=float(aux.detach()),
            t_active=float((env['timesteps']<=CONFIG['eta']).float().mean()),
            decoded_out_of_range=float(((pred.detach()<-1)|(pred.detach()>1)).float().mean()),
            x0_out_of_range=float(((env['x0_hat'].detach()<-1)|(env['x0_hat'].detach()>1)).float().mean()),
            auxiliary_seconds=time.time()-started))
        return base+aux

    @torch.no_grad()
    def selfcheck(self,env):
        pred,gt,ref,mask=[env[k].detach() for k in ['decoded_loss','target_loss','texture_target_loss','mask_loss']]
        t=torch.zeros(pred.shape[0],device=pred.device,dtype=torch.long)
        weight=torch.ones(pred.shape[0],device=pred.device)
        a=self.impl.raw(pred,gt,ref,mask,t,weight)
        b=self.impl.raw(pred,gt,-ref,mask,t,weight)
        if self.arm=='T1': assert all(torch.equal(a[k],b[k]) for k in a)
        if self.arm=='T2':
            assert torch.equal(a['structure'],b['structure'])
            assert not torch.allclose(a['appearance'],b['appearance'],rtol=0,atol=1e-7)
        if self.arm=='T3':
            assert torch.equal(a['rest'],b['rest'])
            assert not torch.allclose(a['body_rgb'],b['body_rgb'],rtol=0,atol=1e-7)
        if self.arm!='T3':
            high=self.impl.raw(pred,gt,ref,mask,t+999,weight)
            assert all(v.item()==0 for v in high.values())
            mixed=self.impl.raw(pred.repeat(2,1,1,1),gt.repeat(2,1,1,1),ref.repeat(2,1,1,1),
                mask.repeat(2,1,1,1),torch.tensor([0,999],device=pred.device),weight.repeat(2))
            assert all(torch.allclose(mixed[k],a[k]/2,rtol=1e-5,atol=1e-6) for k in a)
        for fill in [0,1]:
            values=self.impl.raw(pred,gt,ref,torch.full_like(mask,fill),t,weight)
            assert all(torch.isfinite(v).item() for v in values.values())
        if self.impl.network is not None:
            assert not self.impl.network.training
            assert not any(p.requires_grad or p.grad is not None for p in self.impl.network.parameters())
        write(self.run/('loss_semantics_rank%d.json'%self.rank),dict(passed=True,reference_swap_correct=True,
            time_gate_checked=self.arm!='T3',mixed_batch_fixed_mean_checked=self.arm!='T3',
            empty_full_masks_finite=True,prior_frozen=True,target='GT for T1; Ref CLS/GT structure for T2; Ref body/GT rest for T3'))

    def finish_calibration(self):
        assert self.parameter_probe
        assert not self.optimizer.state,'Calibration performed optimizer updates'
        assert all(torch.equal(p,b) for p,b in zip(self.probes,self.probe_before))
        csv_write(self.run/('calibration_rank%d.csv'%self.rank),self.rows)
        write(self.run/('gradient_probe_rank%d.json'%self.rank),self.parameter_probe)
        self.acc.wait_for_everyone()
        if self.acc.is_main_process:
            rows=[]
            for rank in range(2):
                with (self.run/('calibration_rank%d.csv'%rank)).open() as f: rows.extend(csv.DictReader(f))
            assert all(read(self.run/('loss_semantics_rank%d.json'%r))['passed'] for r in range(2))
            with (OLD/'training/seed42/F000/training_log.csv').open() as f: baseline=list(csv.DictReader(f))[:64]
            local=rows[:256]
            assert all(local[4*i+3]['rng_chain']==baseline[i]['rng_rank0'] for i in range(64)), 'CALIBRATION_RNG_PREFIX_MISMATCH'
            names=[n[:-6] for n in rows[0] if n.endswith('_ratio')]
            assert all(set(read(self.run/('gradient_probe_rank%d.json'%rank)))==set(names) for rank in range(2))
            target=read(ROOT/'calibration_choice.json')['gradient_target']
            coefficients={}; evidence={}
            for name in names:
                ratios=np.array([float(r[name+'_ratio']) for r in rows]); positive=ratios[ratios>0]
                assert len(positive)>0
                coefficient=.1 if name=='body_rgb' else target/float(np.median(positive))
                coefficients[name]=coefficient
                evidence[name]=dict(median_positive_ratio=float(np.median(positive)),active_rate=float((ratios>0).mean()),
                    weighted_target=coefficient*float(np.median(positive)),minimum=float(ratios.min()),maximum=float(ratios.max()))
            registry=read(ROOT/'loss_registry.json') if (ROOT/'loss_registry.json').exists() else {}
            assert self.arm not in registry
            registry[self.arm]=dict(coefficients=coefficients,calibration=evidence,gradient_target=target,
                samples=512,optimizer_updates=0,coefficient_selection_used_P2=False,eta=CONFIG['eta'],source_commit=commit())
            write(ROOT/'loss_registry.json',registry)
            write(self.run/'calibration_complete.json',dict(passed=True,samples=512,effective_batches=64,
                optimizer_updates=0,actual_parameter_gradient_probes=True,coefficients=coefficients))
            print('CALIBRATION_PASSED',self.arm,coefficients,flush=True)
        self.acc.wait_for_everyone()
        raise CalibrationDone()


def train(arm,phase):
    assert read(ROOT/'audit/preparation_complete.json')['passed']
    run=folder(arm,phase);run.mkdir(parents=True,exist_ok=True)
    assert not (run/'training_complete.json').exists(),'COMPLETED_RUN_REQUIRES_NO_RETRAINING'
    original=(SOURCE/'train_GAM_texture_joint.py').read_text(encoding='utf-8')
    assert sha(SOURCE/'train_GAM_texture_joint.py')==read(ROOT/'audit/historical_source_sha256.json')['train_GAM_texture_joint.py']
    argv=list(read(OLD/'training/seed42/F000/config.json')['argv'])
    argv+=['--output_dir',str(run),'--attr_audit',str(run),'--attr_mask_cache',str(OLD/'masks/train'),
        '--start_global_step','0','--seed','42','--gradient_accumulation_steps','4',
        '--max_train_steps',str(800 if phase=='pilot' else 28365),
        '--lambda_region_texture','0','--lambda_boundary','0','--lambda_leak','0',
        '--report_to','none','--wandb_mode','disabled','--val_vis_steps','0','--vis_every_n_steps','0']
    # 只追加 hook；原 base/x0/VAE、数据、RNG、optimizer、AMP、checkpoint 源码保持字节不变。
    replacements={
        '    runtime = AuditRuntime(args, accelerator)':'    runtime = AuditRuntime(args, accelerator)\n    SUP_HOOK.setup(locals())',
        '            with accelerator.accumulate(unet):':'            with accelerator.accumulate(unet), SUP_HOOK.context():',
        '                if not torch.isfinite(loss.detach()).all():':
        '                loss = SUP_HOOK(locals(), globals())\n                if SUP_HOOK.phase == "calibrate":\n                    continue\n\n                if not torch.isfinite(loss.detach()).all():',
        '        num_training_steps=target_global_step,':'        num_training_steps=28365,',
    }
    modified=original
    for before,after in replacements.items():
        assert modified.count(before)==1,('HOOK_ANCHOR_CHANGED',before)
        modified=modified.replace(before,after)
    config=dict(protocol=CONFIG['protocol'],arm=arm,phase=phase,argv=argv,
        base_source_sha256=sha(SOURCE/'train_GAM_texture_joint.py'),
        transformed_source_sha256=hashlib.sha256(modified.encode()).hexdigest(),
        hooks=replacements,coefficient_registry=None if phase=='calibrate' else read(ROOT/'loss_registry.json')[arm],
        additions={p:sha(SOURCE/p) for p in ['tools/e2a_sup_train.py','tools/e2a_sup_losses.py']},
        world_size=2,effective_batch=8,updates=800 if phase=='pilot' else 28365,seed=42)
    if int(os.environ.get('RANK','0'))==0: freeze(run/'config.json',config)
    hook=Hook(arm,phase,run)
    # 原模块用于 main guards 和所有依赖；不向旧文件写入任何新代码。
    namespace=dict(__name__='__main__',__file__=str(SOURCE/'train_GAM_texture_joint.py'),SUP_HOOK=hook)
    sys.argv=[namespace['__file__']]+argv
    try: exec(compile(modified,namespace['__file__'],'exec'),namespace)
    except CalibrationDone: return
    if int(os.environ.get('RANK','0'))!=0:return
    if (run/'pauses'/('job_%s.json'%os.environ.get('SLURM_JOB_ID'))).exists():return
    count=read(run/'optimizer_update_count.json'); target=config['updates']
    assert count['actual_successful_updates']==target and count['skipped_updates']==0
    path=run/'checkpoint-final/joint_model.pt'
    weights=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    tensors=[v for group in weights.values() if isinstance(group,dict) for v in group.values() if torch.is_tensor(v)]
    assert all(torch.isfinite(t).all().item() for t in tensors)
    write(run/'checkpoint_verification.json',dict(passed=True,actual_updates=target,finite_tensors=len(tensors),
        path=str(path),sha256=sha(path),arm=arm,phase=phase))
    print('CHECKPOINT_VERIFIED',arm,phase,target,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--phase',choices=['calibrate','pilot','formal'],required=True)
    a=p.parse_args();train(a.arm,a.phase)
