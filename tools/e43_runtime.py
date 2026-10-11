"""包装原E5；共同前10步只计算一次，分支在各自z上求完整CFG。"""
import argparse
import copy
import os
import time
from types import MethodType
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from tools.e43_protocol import *
from tools.e43_metrics import masks,metric_record


def tensor_hash(x):
    return hashlib.sha256(x.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def combine(plus,zero,mask,alpha):
    # 不做零加法：alpha1必须直接返回原张量，保证float16舍入完全不变。
    if alpha==1.:return plus
    return plus+(alpha-1.)*mask*(plus-zero)


class Runner:
    def __init__(self):
        from models.e33tm_generation_wrapper import load_e5
        from tools.e22_4_generation import module_hashes
        self.p,self.modules,size,self.ns=load_e5();assert size==SIZE
        assert type(self.p.scheduler).__name__=='DDIMScheduler' and sha(E5)==EXPECTED_E5
        self.hash_modules=module_hashes;self.before=module_hashes(self.modules)
        assert all(not p.requires_grad for m in self.modules.values() for p in m.parameters())
        self.forward=self.p.unet.forward;self.step=self.p.scheduler.step;self.get=self.p.get_image_embeds
        self.calls=0;self.encoder_calls=0
        self.cuda=dict(device=torch.cuda.get_device_name(),slurm_job_id=os.environ.get('SLURM_JOB_ID'),code_commit=commit(),
            E5_sha256=sha(E5),scheduler=dict(self.p.scheduler.config),inference_args=vars(self.ns),frozen_modules=self.before)

    def call(self,x,t,kw):
        self.calls+=1
        return self.forward(x,t,**kw)[0]

    def zero_embeds(self,*args,**kw):
        counter=[0];hooks=[]
        def vision(m,a,k):
            if a:return (torch.zeros_like(a[0]),)+a[1:],k
            k=dict(k);k['pixel_values']=torch.zeros_like(k['pixel_values']);return a,k
        def bf(m,a,k):
            counter[0]+=1
            if counter[0]==1:
                k=dict(k)
                for n in ['clip_image_embeds','clip_vision_tokens','texture_images']:
                    if torch.is_tensor(k.get(n)):k[n]=torch.zeros_like(k[n])
            return a,k
        hooks.append(self.p.image_encoder.register_forward_pre_hook(vision,with_kwargs=True))
        hooks.append(self.p.bf_texture_conditioner.register_forward_pre_hook(bf,with_kwargs=True))
        try:return self.get(*args,**kw)
        finally:
            for h in hooks:h.remove()

    def guided(self,z,t,zero=False):
        x=self.p.scheduler.scale_model_input(z,t)
        c=dict(self.cond);u=dict(self.uncond)
        for kw in [c,u]:
            kw['cross_attention_kwargs']=dict(kw['cross_attention_kwargs'],balanced_gate_timestep=t.float().view(1)/1000)
        if zero:
            n=self.zero[0].shape[1]
            c['encoder_hidden_states']=torch.cat([c['encoder_hidden_states'][:,:-n],self.zero[0]],1)
            u['encoder_hidden_states']=torch.cat([u['encoder_hidden_states'][:,:-n],self.zero[1]],1)
        pc=self.call(x,t,c);pu=self.call(x,t,u)
        return pu+7.*(pc-pu)

    def response(self,index,delta,plus):
        d=delta.float();p=plus.float();m=self.mask
        rms=lambda x,w:float(((x.square()*w).sum()/(w.sum()*x.shape[1]).clamp_min(1e-12)).sqrt())
        self.responses[index]=[rms(d,m),float(d.square().mean().sqrt())]
        if index==9:
            cos=lambda w:float((d*p*w).sum()/((d.square()*w).sum().sqrt()*(p.square()*w).sum().sqrt()).clamp_min(1e-12))
            norm=d.square().sum(1).sqrt();self.last_stats=[cos(m),cos(torch.ones_like(m)),float(norm.std()/norm.mean().clamp_min(1e-12))]

    @torch.inference_mode()
    def baseline(self,row,seed,instrument=True):
        self.row=row;self.seed=seed;self.calls=0;self.responses={};self.trace=[];self.cond=None;self.uncond=None
        bundle,self.maskinfo=masks(row['sketch'])
        self.mask=torch.from_numpy(self.maskinfo['latent'])[None,None].to(self.p.device,self.p.unet.dtype)
        self.initial=torch.randn((1,4,64,48),device=self.p.device,dtype=self.p.unet.dtype,
            generator=torch.Generator(device=self.p.device).manual_seed(seed))*self.p.scheduler.init_noise_sigma
        old_prepare=self.p.prepare_latents
        def embeds(_p,*a,**kw):
            result=self.get(*a,**kw);self.tokens=result[0].detach().clone();self.encoder_calls+=1
            if instrument:
                self.zero=self.zero_embeds(*a,**kw);self.encoder_calls+=1
                self.zero_uncond_tokens_equal=bool(torch.equal(result[1],self.zero[1]))
            return result
        def forward(_u,*a,**kw):
            key='cond' if 'sa_hidden_states' in (kw.get('cross_attention_kwargs') or {}) else 'uncond'
            setattr(self,key,dict(kw));self.current_x=a[0];return (self.call(*a,kw),)
        def update(eps,t,z,**kw):
            index=len(self.trace)
            if instrument and index in [0,4,9]:
                self.response(index,eps-self.guided(z,t,zero=True),eps)
            result=self.step(eps,t,z,**kw);v=result[0]
            self.trace.append(tensor_hash(v))
            if index==9:
                self.prefix=v.detach().clone();torch.cuda.synchronize()
                self.prefix_seconds=time.perf_counter()-started;self.prefix_calls=self.calls
            self.final=v.detach().clone();return result
        sketch=Image.open(row['sketch']).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR)
        reference=Image.open(row['reference']).convert('RGB')
        self.p.prepare_latents=MethodType(lambda _p,*a,**kw:self.initial.clone(),self.p)
        self.p.get_image_embeds=MethodType(embeds,self.p);self.p.unet.forward=MethodType(forward,self.p.unet);self.p.scheduler.step=update
        torch.cuda.synchronize();started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        try:
            image=self.p(prompt=row['caption'],null_prompt='',negative_prompt=' worst quality, low quality',
                ref_image=to_tensor(sketch)[None]*2-1,texture_clip_image=reference,width=384,height=512,
                num_inference_steps=50,guidance_scale=7.,sketch_scale=.6,ipa_scale=1.,texture_mode=self.ns.texture_mode,
                texture_condition_mode='token',texture_preprocess_mode='plain_resize',texture_num_tokens=self.p.effective_texture_num_tokens,
                texture_scale=1.,spatial_mask=torch.from_numpy(bundle['garment'].astype('float32'))[None,None].to(self.p.device,torch.float16),
                generator=torch.Generator(device=self.p.device).manual_seed(seed))[0]
        finally:
            self.p.prepare_latents=old_prepare;self.p.get_image_embeds=self.get;self.p.unet.forward=self.forward;self.p.scheduler.step=self.step
        torch.cuda.synchronize()
        record=dict(seconds=time.perf_counter()-started,unet_calls=self.calls,reference_encoder_calls=2 if instrument else 1,
            prefix_seconds=self.prefix_seconds,prefix_unet_calls=self.prefix_calls,prefix_latent_hash=tensor_hash(self.prefix),
            peak_memory_bytes=torch.cuda.max_memory_allocated(),initial_hash=tensor_hash(self.initial),trajectory_hashes=list(self.trace))
        if instrument:
            self.features=[v for k in [0,4,9] for v in self.responses[k]]+self.last_stats+[
                self.maskinfo['area'],self.maskinfo['boundary'],float(np.asarray(reference.convert('L'),float).std()/255),
                float(self.tokens.float().norm()),float(self.maskinfo['valid'])]
            assert len(self.features)==14
            record['zero_uncond_tokens_equal']=self.zero_uncond_tokens_equal
        return image,record

    @torch.inference_mode()
    def suffix(self,alpha,start=10,unmasked=False,zero_delta=False,force_invalid=False):
        effective=alpha if self.maskinfo['valid'] and not force_invalid else 1.
        z=(self.prefix if start==10 else self.initial).clone();trace=[];before=self.calls
        branch_start_hash=tensor_hash(z)
        m=torch.ones_like(self.mask) if unmasked else self.mask
        self.p.scheduler.set_timesteps(50,device=self.p.device)
        torch.cuda.synchronize();started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        for t in self.p.scheduler.timesteps[start:]:
            plus=self.guided(z,t);zero=self.guided(z,t,zero=True) if effective!=1. and not zero_delta else plus
            eps=combine(plus,zero,m,effective)
            z=self.step(eps,t,z,eta=0.,return_dict=False)[0];trace.append(tensor_hash(z))
        image=self.p.vae.decode(z/self.p.vae.config.scaling_factor,return_dict=False)[0]
        image=(image/2+.5).clamp(0,1).cpu().permute(0,2,3,1).float().numpy();image=self.p.numpy_to_pil(image)[0]
        torch.cuda.synchronize()
        return image,dict(alpha=alpha,effective_alpha=effective,start_step=start+1,unmasked=unmasked,zero_delta=zero_delta,
            branch_start_hash=branch_start_hash,seconds=time.perf_counter()-started,unet_calls=self.calls-before,
            peak_memory_bytes=torch.cuda.max_memory_allocated(),trajectory_hashes=trace)

    def frozen(self):
        assert self.hash_modules(self.modules)==self.before,'生成模型参数改变'
        return dict(**self.cuda,parameters_unchanged=True,trainable_generation_parameters=0)


def precheck():
    path=OUT/'preflight.json'
    if path.exists():assert read(path)['passed'];return
    rows=read(OUT/'split_sources.json')['pilot'];run=Runner();folder=OUT/'precheck';folder.mkdir(exist_ok=True)
    checks=[]
    for index,row in enumerate(rows[:2]):
        seed=[42,123][index]
        original,a=run.baseline(row,seed,instrument=False);original.save(folder/(row['id']+'_original.png'))
        wrapped,b=run.baseline(row,seed);wrapped.save(folder/(row['id']+'_alpha1.png'))
        assert a['trajectory_hashes']==b['trajectory_hashes'] and np.array_equal(np.asarray(original),np.asarray(wrapped))
        assert sha(folder/(row['id']+'_original.png'))==sha(folder/(row['id']+'_alpha1.png'))
        tests=[]
        for alpha,start,zero,invalid in [(1.,0,False,False),(.65,10,True,False),(1.35,10,True,False),(.65,10,False,True)]:
            image,r=run.suffix(alpha,start=start,zero_delta=zero,force_invalid=invalid)
            assert r['trajectory_hashes']==b['trajectory_hashes'][start:]
            assert np.array_equal(np.asarray(image),np.asarray(wrapped))
            tests.append(dict(r,start=start,invalid_mask=invalid,pass_identical=True))
        checks.append(dict(id=row['id'],seed=seed,original=a,wrapped=b,tests=tests,full_trajectory_and_png_equal=True))
    from tools.e33tmoc_appearance_eval import frozen_lpips
    metric,info=frozen_lpips();metric.cuda();value=metric_record(folder/(rows[0]['id']+'_alpha1.png'),rows[0],metric)
    write(path,dict(passed=True,checks=checks,metric_smoke=value,lpips_initialization=info,frozen=run.frozen(),**verify_inputs()))
    print('E43 PRECHECK PASSED',flush=True)


def collect(phase,shard,shards):
    assert read(OUT/'preflight.json')['passed']
    if phase=='train':assert pilot_permission(),'Pilot未通过且无用户继续授权'
    if phase=='validation':assert (OUT/'policy.pkl').exists()
    if phase=='holdout':assert read(OUT/'validation/decision.json')['pass_stage']
    from tools.e33tmoc_appearance_eval import frozen_lpips
    run=Runner();metric,info=frozen_lpips();metric.cuda()
    rows=read(OUT/'split_sources.json')[phase];seeds=[42] if phase in ['pilot','train'] else CONFIG['seeds']
    for i,row in enumerate(rows):
        if i%shards!=shard:continue
        for seed in seeds:
            folder=OUT/phase/'samples'/row['id']/('seed%d'%seed)
            if (folder/'complete.json').exists():continue
            folder.mkdir(parents=True,exist_ok=True)
            baseline,record=run.baseline(row,seed);baseline.save(folder/'alpha1.png')
            np.savez_compressed(folder/'common_prefix.npz',initial=run.initial.cpu().numpy(),step10=run.prefix.cpu().numpy(),
                timesteps=run.p.scheduler.timesteps.cpu().numpy())
            outputs=[dict(arm='alpha1',alpha=1.,effective_alpha=1.,**record)]
            # 特征和mask在任何后缀分支执行前保存；不包含GT或后40步值。
            write(folder/'features.json',dict(values=run.features,mask_valid=run.maskinfo['valid'],feature_steps=[1,5,10],
                mask_stats=run.maskinfo['stats'],mask_confidence=run.maskinfo['confidence'],group_id=row['group_id']))
            np.save(folder/'reference_token_mean.npy',run.tokens.float().mean(1).cpu().numpy()[0])
            for alpha,arm in [(.65,'low'),(1.35,'high')]:
                image,r=run.suffix(alpha);image.save(folder/(arm+'.png'));outputs.append(dict(arm=arm,**r))
            if phase=='pilot' and i<4:
                for alpha,name in [(.65,'low'),(1.35,'high')]:
                    for kind,start,unmasked in [('full',0,False),('unmasked',10,True)]:
                        image,r=run.suffix(alpha,start=start,unmasked=unmasked);arm=kind+'_'+name
                        image.save(folder/(arm+'.png'));outputs.append(dict(arm=arm,**r))
            for r in outputs:
                path=folder/(r['arm']+'.png');r.update(id=row['id'],group_id=row['group_id'],seed=seed,output_png=str(path),
                    output_png_sha256=sha(path),metrics=metric_record(path,row,metric))
            write(folder/'complete.json',dict(id=row['id'],seed=seed,group_id=row['group_id'],outputs=outputs,slurm_job_id=os.environ.get('SLURM_JOB_ID')))
            print('E43 COLLECTED',phase,i,len(rows),row['id'],seed,flush=True)
    write(OUT/phase/('runtime_shard%d.json'%shard),dict(frozen=run.frozen(),**verify_inputs()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['precheck','pilot','train','validation','holdout'],required=True)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);args=p.parse_args()
    cv2.setNumThreads(1);torch.set_num_threads(2)
    if args.phase=='precheck':precheck()
    else:collect(args.phase,args.shard,args.shards)
