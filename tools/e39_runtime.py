"""只读钩子围绕原 E5 推理；同 latent 反事实和自由轨迹分开保存。"""
import copy
import hashlib
import os
import subprocess
import time
from types import MethodType
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor
from tools.e39_protocol import *


def digest(x):
    return hashlib.sha256(x.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def distance(a,b):
    a,b=a.float(),b.float();d=a-b
    return dict(l2=float(d.norm()), rms=float(d.square().mean().sqrt()),
                relative=float(d.norm()/a.norm().clamp_min(1e-12)), baseline_rms=float(a.square().mean().sqrt()))


class Runner:
    def __init__(self, shard):
        from models.e33tm_generation_wrapper import load_e5
        from tools.e22_4_generation import module_hashes
        self.p,self.modules,size,self.ns=load_e5()
        assert size == SIZE and type(self.p.scheduler).__name__=='DDIMScheduler'
        assert sha(E5)==EXPECTED_E5
        self.hash_modules=module_hashes;self.before=module_hashes(self.modules)
        assert all(not v.requires_grad for m in self.modules.values() for v in m.parameters())
        self.shard=shard;self.handles=[];self.active=False;self.observe=False;self.ctx=None
        self.old_forward=self.p.unet.forward;self.old_sdpa=F.scaled_dot_product_attention
        self.old_get=self.p.get_image_embeds;self.old_step=self.p.scheduler.step
        self.old_latents=self.p.prepare_latents
        self.initial=torch.randn((1,4,64,48),device=self.p.device,dtype=self.p.unet.dtype,
            generator=torch.Generator(device=self.p.device).manual_seed(42))*self.p.scheduler.init_noise_sigma
        self.p.unet.forward=MethodType(self.forward,self.p.unet)
        self.p.scheduler.step=self.step
        self.p.prepare_latents=MethodType(lambda _p,*a,**kw:self.initial.clone(),self.p)
        self.p.get_image_embeds=MethodType(self.get_embeds,self.p)
        F.scaled_dot_product_attention=self.sdpa
        self.install_hooks()
        write(OUT/'audit'/('runtime_shard%d.json'%shard),dict(checkpoint_sha256=sha(E5),
            git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            code_sha256={str(p):sha(p) for p in Path('tools').glob('e39_*.py')},
            slurm_job_id=os.environ.get('SLURM_JOB_ID'),node=os.environ.get('SLURMD_NODENAME'),
            scheduler=dict(self.p.scheduler.config),inference_args=vars(self.ns),
            frozen_modules=self.before,device=torch.cuda.get_device_name(),initial_sha256=digest(self.initial),
            processors={k:type(v).__name__ for k,v in self.p.unet.attn_processors.items()}))

    def install_hooks(self):
        bf=self.p.bf_texture_conditioner
        def bf_pre(m,args,kw):
            self.bf_call+=1
            if self.arm=='Rzero' and self.bf_call==1:
                kw=dict(kw)
                for k in ['clip_image_embeds','clip_vision_tokens','texture_images']:
                    if torch.is_tensor(kw.get(k)):kw[k]=torch.zeros_like(kw[k])
                return args,kw
        self.handles.append(bf.register_forward_pre_hook(bf_pre,with_kwargs=True))
        def vision_pre(m,args,kw):
            if self.arm=='Rzero':
                if args:return (torch.zeros_like(args[0]),)+args[1:],kw
                kw=dict(kw);kw['pixel_values']=torch.zeros_like(kw['pixel_values']);return args,kw
        self.handles.append(self.p.image_encoder.register_forward_pre_hook(vision_pre,with_kwargs=True))
        def vision_out(m,args,out):
            if self.capturing:
                self.features['clip_patches']=out.hidden_states[-1][:,1:].detach().cpu().numpy()
                self.features['clip_pool']=out.image_embeds.detach().cpu().numpy()
        self.handles.append(self.p.image_encoder.register_forward_hook(vision_out))
        for i in range(1,5):
            def stage(m,args,out,i=i):
                if self.bf_call==1:
                    self.features['cnn%d_pool8'%i]=F.adaptive_avg_pool2d(out,(8,8)).detach().cpu().numpy()
                    self.features['cnn%d_meanstd'%i]=torch.cat([out.float().mean((2,3)),out.float().std((2,3))],1).cpu().numpy()
                    if i==3:self.features['cnn3_native']=out.detach().cpu().numpy()
            self.handles.append(getattr(bf,'stage%d'%i).register_forward_hook(stage))
        def resampler(m,args,out):
            if self.bf_call==1:
                self.features['encoder_fused']=args[1].detach().cpu().numpy()
                self.features['token_resampler']=out[0].detach().cpu().numpy()
        self.handles.append(bf.resampler.register_forward_hook(resampler))
        def bf_out(m,args,out):
            if self.bf_call==1:self.features['token_bf']=out[0].detach().cpu().numpy()
        self.handles.append(bf.register_forward_hook(bf_out))
        for name,m in self.p.unet.named_modules():
            if not name.endswith('attn2') or not hasattr(m.processor,'to_k_ip'):continue
            proc=m.processor
            def before(mod,args,kw,name=name,proc=proc):
                self.ctx=dict(name=name,proc=proc,call=0,text_rms=None)
            def after(mod,args,out):self.ctx=None
            self.handles.extend([m.register_forward_pre_hook(before,with_kwargs=True),m.register_forward_hook(after)])
            def residual(x,name=name,proc=proc):
                if self.observe:
                    tex=float(x.float().square().mean().sqrt())
                    text=self.ctx.get('text_rms') if self.ctx else None
                    self.attn_rows.append(dict(t=self.probe_t,arm=self.arm,branch=self.branch,layer=name,
                        kind='post_gate_residual',rms=tex,text_rms=text,
                        residual_norm_fraction=tex/(tex+text+1e-12) if text is not None else None,
                        gate=float(torch.exp(proc.texture_gate_delta).clamp(proc.gate_min,proc.gate_max))
                             if proc.texture_gate_delta is not None and proc.use_texture_gate else 1.,
                        layer_group=proc.layer_group))
            proc.texture_probe_observer=residual

    def get_embeds(self,_p,*args,**kw):
        self.bf_call=0;self.capturing=True;self.features={}
        try:out=self.old_get(*args,**kw)
        finally:self.capturing=False
        self.features['token_tcpm']=out[0].detach().cpu().numpy()
        self.features['token_uncond']=out[1].detach().cpu().numpy()
        return out

    def forward(self,_unet,*args,**kw):
        self.branch='conditional' if 'sa_hidden_states' in (kw.get('cross_attention_kwargs') or {}) else 'unconditional'
        out=self.old_forward(*args,**kw)
        if self.active:
            eps=out[0] if isinstance(out,tuple) else out.sample
            self.eps[self.branch].append(eps.detach().cpu().numpy())
            if self.branch=='conditional':
                self.latents.append(args[0].detach().cpu().numpy())
                if self.condition is None:self.condition=copy.copy(kw)
            elif self.negative is None:self.negative=copy.copy(kw)
        return out

    def step(self,*args,**kw):
        out=self.old_step(*args,**kw)
        if self.active:
            self.steps.append(int(args[1]));self.guided.append(args[0].detach().cpu().numpy())
            self.last=out[0].detach().clone()
        return out

    def sdpa(self,q,k,v,*args,**kw):
        result=self.old_sdpa(q,k,v,*args,**kw)
        if not self.observe or self.ctx is None:return result
        ctx=self.ctx;kind='cross_text' if ctx['call']==0 else 'texture';ctx['call']+=1
        if kind=='cross_text':ctx['text_rms']=float(result.float().square().mean().sqrt())
        # 正确 softmax(QK^T/sqrt(d))。原 processor.attn_map 的括号不同，不能作概率使用。
        maps=[];entropy=[]
        for start in range(0,q.shape[-2],256):
            logits=(q[...,start:start+256,:].float()@k.float().transpose(-1,-2))*q.shape[-1]**-.5
            mask=kw.get('attn_mask',args[0] if args else None)
            if mask is not None:
                mask=mask[...,start:start+256,:] if mask.shape[-2]>1 else mask
                logits=logits.masked_fill(~mask,float('-inf')) if mask.dtype==torch.bool else logits+mask
            probabilities=logits.softmax(-1)
            maps.append(probabilities.mean((0,1)))
            entropy.append(-(probabilities*probabilities.clamp_min(1e-12).log()).sum(-1).mean())
        matrix=torch.cat(maps);spatial_h=round((len(matrix)*4/3)**.5);spatial_w=len(matrix)//spatial_h
        if spatial_h*spatial_w==len(matrix):
            compact=F.adaptive_avg_pool2d(matrix.T.reshape(1,matrix.shape[1],spatial_h,spatial_w),(16,12))[0]
        else:compact=matrix.T[:,None,:]
        key='%s_t%d_%s_%s_%s'%(self.arm,self.probe_t,self.branch,ctx['name'],kind)
        self.attn_maps[key]=compact.cpu().numpy().astype(np.float16)
        self.attn_rows.append(dict(t=self.probe_t,arm=self.arm,branch=self.branch,layer=ctx['name'],kind=kind,
            query_tokens=q.shape[-2],key_tokens=k.shape[-2],mean_entropy=float(torch.stack(entropy).mean()),
            token_mass=matrix.mean(0).cpu().tolist(),row_sum_error=float((matrix.sum(-1)-1).abs().max()),
            output_rms=float(result.float().square().mean().sqrt())))
        return result

    @torch.inference_mode()
    def generate(self,row,arm,reference,folder):
        from garment_mask_utils import build_sketch_garment_mask
        self.arm=arm;self.features={};self.bf_call=0;self.capturing=False
        sketch=Image.open(row['sketch']).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR)
        mask,info=build_sketch_garment_mask(sketch,*SIZE)
        self.active=True;self.eps={'conditional':[],'unconditional':[]};self.latents=[];self.guided=[];self.steps=[]
        self.condition=None;self.negative=None
        torch.cuda.synchronize();started=time.perf_counter()
        try:
            image=self.p(prompt=row['caption'],null_prompt='',negative_prompt=' worst quality, low quality',
                ref_image=to_tensor(sketch)[None]*2-1,texture_clip_image=reference,
                width=SIZE[0],height=SIZE[1],num_inference_steps=50,guidance_scale=7.,sketch_scale=.6,ipa_scale=1.,
                texture_mode=self.ns.texture_mode,texture_condition_mode='token',texture_preprocess_mode='plain_resize',
                texture_num_tokens=self.p.effective_texture_num_tokens,texture_scale=1.,
                spatial_mask=to_tensor(mask)[None].to(self.p.device,torch.float16),
                generator=torch.Generator(device=self.p.device).manual_seed(42))[0]
        finally:self.active=False
        assert len(self.steps)==50 and len(self.eps['conditional'])==50 and len(self.eps['unconditional'])==50
        guided=np.stack(self.guided);conditional=np.stack(self.eps['conditional']);unconditional=np.stack(self.eps['unconditional'])
        # 原 float16 CFG 舍入必须按相同 dtype 校验。
        assert np.max(np.abs(guided-(unconditional+np.float16(7)*(conditional-unconditional))))==0
        image.save(folder/(arm+'.png'));reference.save(folder/(arm+'_reference.png'))
        values=dict(eps_cond=conditional,eps_uncond=unconditional,eps_guided=guided,
                    final_latent=self.last.cpu().numpy(),timesteps=np.asarray(self.steps))
        if arm=='Rplus':values['latents']=np.stack(self.latents)
        np.savez_compressed(folder/(arm+'_trajectory.npz'),**values)
        np.savez_compressed(folder/(arm+'_features.npz'),**self.features)
        old=Path('output_eval/e36_dagf_guided_filter_20261010/p1/images/A0_E5_OFF')/(row['id']+'.png')
        record=dict(arm=arm,steps=self.steps,initial_sha256=digest(self.initial),png_sha256=sha(folder/(arm+'.png')),
            seconds=time.perf_counter()-started,mask_info=info,
            original_e5_png_equal=(sha(old)==sha(folder/(arm+'.png'))) if arm=='Rplus' and old.exists() else None)
        write(folder/(arm+'_generation.json'),record)
        return dict(condition=self.condition,negative=self.negative,eps_cond=conditional,eps_uncond=unconditional,
                    guided=guided,latent=self.last.clone(),latents=values.get('latents'),features=self.features,record=record)

    @torch.inference_mode()
    def probes(self,runs,folder):
        base=runs['Rplus'];metrics=[];arrays={};self.attn_rows=[];self.attn_maps={}
        # 50 步同状态因果差异：错误参考在正确参考的 latent 上重算，避免轨迹混淆。
        self.arm='Rminus'
        for i,t in enumerate(base['record']['steps']):
            x=torch.from_numpy(base['latents'][i]).to(self.p.device)
            kw=dict(base['condition']);kw['encoder_hidden_states']=runs['Rminus']['condition']['encoder_hidden_states']
            kw['cross_attention_kwargs']=dict(kw['cross_attention_kwargs'],balanced_gate_timestep=torch.tensor([t/1000],device=self.p.device))
            out=self.p.unet(x,torch.tensor(t,device=self.p.device),**kw)[0]
            matched=torch.from_numpy(base['eps_cond'][i]).to(self.p.device)
            unc=torch.from_numpy(base['eps_uncond'][i]).to(self.p.device)
            guided=unc+7*(out-unc);matched_guided=torch.from_numpy(base['guided'][i]).to(self.p.device)
            metrics.append(dict(step=i,t=t,conditional=distance(matched,out),unconditional=distance(unc,unc),
                                guided=distance(matched_guided,guided)))
            arrays['common_eps_t%d'%t]=out.cpu().numpy()
        exact=[]
        for t in TIMES:
            nearest=int(np.argmin(np.abs(np.asarray(base['record']['steps'])-t)))
            x=torch.from_numpy(base['latents'][nearest]).to(self.p.device);outputs={}
            self.observe=True;self.probe_t=t
            for arm,run in list(runs.items())+[('Rtoken0',base)]:
                self.arm=arm;kw=dict(base['condition']);kw['encoder_hidden_states']=run['condition']['encoder_hidden_states']
                if arm=='Rtoken0':
                    kw['encoder_hidden_states']=kw['encoder_hidden_states'].clone()
                    kw['encoder_hidden_states'][:,-self.p.effective_texture_num_tokens:]=0
                kw['cross_attention_kwargs']=dict(kw['cross_attention_kwargs'],balanced_gate_timestep=torch.tensor([t/1000],device=self.p.device))
                outputs[arm]=self.p.unet(x,torch.tensor(t,device=self.p.device),**kw)[0]
                arrays['exact_%s_t%d'%(arm,t)]=outputs[arm].cpu().numpy()
            self.arm='Rplus'
            neg=dict(base['negative']);neg['cross_attention_kwargs']=dict(neg['cross_attention_kwargs'],
                balanced_gate_timestep=torch.tensor([t/1000],device=self.p.device))
            unc=self.p.unet(x,torch.tensor(t,device=self.p.device),**neg)[0]
            self.observe=False
            arrays['exact_uncond_t%d'%t]=unc.cpu().numpy()
            for arm in outputs:
                if arm=='Rplus':continue
                exact.append(dict(t=t,arm=arm,latent_source_t=base['record']['steps'][nearest],
                    conditional=distance(outputs['Rplus'],outputs[arm]),
                    guided=distance(unc+7*(outputs['Rplus']-unc),unc+7*(outputs[arm]-unc)),
                    note='same z and same unconditional branch'))
        write(folder/'common_trajectory.json',metrics);write(folder/'exact_probes.json',exact)
        write(folder/'attention.json',self.attn_rows)
        np.savez_compressed(folder/'attention_maps.npz',**self.attn_maps)
        np.savez_compressed(folder/'common_predictions.npz',**arrays)
        free={arm:dict(final_latent=distance(base['latent'],r['latent']),
            eps_cond_rms_by_step=[float(np.sqrt(np.mean((base['eps_cond'][i].astype(np.float32)-r['eps_cond'][i].astype(np.float32))**2))) for i in range(50)],
            negative_embeds_identical=bool(torch.equal(base['negative']['encoder_hidden_states'],r['negative']['encoder_hidden_states'])))
            for arm,r in runs.items() if arm!='Rplus'}
        rgb={arm:(self.p.vae.decode(r['latent']/self.p.vae.config.scaling_factor,return_dict=False)[0].float()/2+.5)
             for arm,r in runs.items()}
        for arm in free:
            free[arm]['rgb_unclamped']=distance(rgb['Rplus'],rgb[arm])
            free[arm]['rgb_clamped']=distance(rgb['Rplus'].clamp(0,1),rgb[arm].clamp(0,1))
            free[arm]['rgb_saturated_fraction']=float(((rgb[arm]<0)|(rgb[arm]>1)).float().mean())
        write(folder/'free_trajectory_comparison.json',free)
        return metrics

    def close(self):
        self.p.unet.forward=self.old_forward;self.p.scheduler.step=self.old_step
        self.p.prepare_latents=self.old_latents;self.p.get_image_embeds=self.old_get
        F.scaled_dot_product_attention=self.old_sdpa
        for handle in self.handles:handle.remove()
        for proc in self.p.unet.attn_processors.values():
            if hasattr(proc,'texture_probe_observer'):del proc.texture_probe_observer
        after=self.hash_modules(self.modules);assert after==self.before
        write(OUT/'audit'/('frozen_shard%d.json'%self.shard),dict(before=self.before,after=after,unchanged=True))


def run(shard,shards,limit=None):
    prepare();pairs=read(OUT/'pairs.json');selected=pairs[shard::shards]
    if limit is not None:selected=selected[:limit]
    g=Runner(shard)
    started=time.perf_counter()
    try:
        for i,row in enumerate(selected):
            folder=OUT/'cases'/row['id'];folder.mkdir(parents=True,exist_ok=True)
            if (folder/'complete.json').exists():
                from tools.e39_semantics import evaluate
                g.arm='analysis';g.capturing=False
                if not (folder/'semantics.json').exists():evaluate(g.p,row,folder)
                continue
            image=Image.open(row['reference']).convert('RGB')
            references=dict(Rplus=image,Rminus=Image.open(row['wrong']['reference']).convert('RGB'),
                Rzero=Image.new('RGB',image.size),R90=image.transpose(Image.Transpose.ROTATE_90))
            if row['color_valid']:references['Rcolor']=Image.open(row['color']['reference']).convert('RGB')
            runs={arm:g.generate(row,arm,ref,folder) for arm,ref in references.items()}
            assert len({r['record']['initial_sha256'] for r in runs.values()})==1
            assert all(torch.equal(runs['Rplus']['negative']['encoder_hidden_states'],r['negative']['encoder_hidden_states']) for r in runs.values())
            g.probes(runs,folder)
            write(folder/'complete.json',dict(id=row['id'],arms=list(runs),training_updates=0,shard=shard))
            from tools.e39_semantics import evaluate
            g.arm='analysis';g.capturing=False;evaluate(g.p,row,folder)
            print('E39 CASE COMPLETE',shard,i+1,len(selected),row['id'],flush=True)
            del runs;torch.cuda.empty_cache()
    finally:g.close()
    write(OUT/('shard%d_complete.json'%shard),dict(shard=shard,identities=[r['id'] for r in selected],
        seconds=time.perf_counter()-started,peak_memory_gib=torch.cuda.max_memory_allocated()/2**30))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--limit',type=int);a=p.parse_args();run(a.shard,a.shards,a.limit)
