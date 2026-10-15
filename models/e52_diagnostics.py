"""保持原 B1 前向；旁路读数与对应干预只在明确指定的条件分支工作。"""
import math
import torch
from torch.nn import functional as F
from tools.e44_b1_mask import region_masks, weights_at, regional_stats
from models.e48_interventions import decomposition, projection, split, digest
from tools.e52_protocol import STEPS,LAYER,SIZE,CONFIG

class Diagnostic:
    def __init__(self,r,row,arm,teacher=None,mapping=None,patch=False):
        from tools.e44_runtime import sketch_masks
        from PIL import Image
        self.r,self.p,self.arm=r,r.processor,arm
        raw,_,info=sketch_masks(Image.open(row['sketch']).convert('RGB').resize(SIZE),'cpu')
        self.valid=info['valid'];self.weights=weights_at(region_masks(raw),(64,48),r.pipe.device)
        self.feature_weights=weights_at(region_masks(raw),(32,24),r.pipe.device)
        self.teacher=teacher;self.mapping=mapping;self.patch=patch
        self.target_query_count=None
        if mapping is not None:
            import cv2
            import numpy as np
            labels=np.load(row['geometry'])['target_labels']
            self.target_query_count=int((cv2.resize((labels>0).astype('float32'),(24,32),interpolation=cv2.INTER_AREA)>=.5).sum())
        self.steps=[];self.latents=[];self.attention_records=[];self.handles=[];self.cache={};self.phase=None
        self.raw=None;self.patch_site=None;self.activations={};self.topology=[]
        self.sites=['up_blocks.2.attentions.0.transformer_blocks.0.attn1',
            'up_blocks.2.resnets.1','up_blocks.2.attentions.1.transformer_blocks.0.attn1']
        if teacher is not None:
            from diffusers import DDIMScheduler
            self.teacher_scheduler=DDIMScheduler.from_config(r.pipe.scheduler.config)
            self.teacher_scheduler.set_timesteps(50,device=r.pipe.device)
            self.teacher_latent=teacher['initial_latent'].to(r.pipe.device)

    def capture(self,name):
        def hook(module,args,out):
            if self.phase in ['candidate','teacher_probe']:self.cache[name]=out.detach()
        return hook

    def observer(self,base,raw,applied):
        if self.phase in ['candidate','teacher_probe']:
            self.raw=raw;self.base=base

    def site_hook(self,name):
        def hook(module,args,out):
            if torch.is_tensor(out):
                if self.phase=='current_e5' and self.patch:
                    self.activations[name]=out.detach()
                    if len(self.steps)==0:self.topology.append(dict(name=name,shape=list(out.shape)))
                if self.phase=='patch' and self.patch_site==name:return self.activations[name]
        return hook

    def output(self,module,args,result):
        if self.phase not in ['candidate','teacher_probe'] or not self.p.enabled:return result
        index=self.index;raw=self.raw
        multiplier=1.
        if self.arm in ['G1_EARLY','G0_EARLY','B1_EARLY','SHRINK_EARLY']:
            multiplier=float(index<15)
        if self.arm=='SHRINK_EARLY':multiplier*=CONFIG['shrink']
        correction=None
        if self.arm in ['ORACLE','WRONG','RANDOM','ORACLE_ALPHA0']:
            assert self.mapping is not None
            q=split(self.cache['q'],module.heads);k=split(self.cache['k'],module.heads)
            kr=split(self.cache['kr'].to(q),module.heads);vr=split(self.cache['vr'].to(q),module.heads)
            self_logits=q.float()@k.float().transpose(-1,-2)/math.sqrt(q.shape[-1])
            logits=q.float()@kr.float().transpose(-1,-2)/math.sqrt(q.shape[-1])
            pref=torch.sigmoid(torch.logsumexp(logits,-1)-torch.logsumexp(self_logits,-1))
            original=torch.softmax(logits,-1)
            oracle=self.mapping.to(q.device).float()[None,None].expand_as(original)
            # 背景/无对应 query 完全沿用原注意力；不把零行解释为关闭注入。
            valid=oracle.sum(-1)>0
            oracle=torch.where(valid[...,None],oracle,original)
            alpha=0. if self.arm=='ORACLE_ALPHA0' else CONFIG['alpha']
            blend=(1-alpha)*original+alpha*oracle
            delta=(pref[...,None]*(blend-original))@vr.float()
            correction=projection(module,(self.p.gate*delta).to(q.dtype))
            self.current['routing']=dict(p_ref_max_error=float((pref*blend.sum(-1)-pref).abs().max()),
                row_sum_error=float((blend.sum(-1)-1).abs().max()),coverage_all_queries=float(valid.float().mean()),
                coverage_foreground=float(valid[0,0].sum())/max(self.target_query_count,1),
                entropy_original=float((-(original*original.clamp_min(1e-20).log()).sum(-1)).mean()),
                entropy_new=float((-(blend*blend.clamp_min(1e-20).log()).sum(-1)).mean()),
                original_ref_value_rms=float((original@vr.float()).square().mean().sqrt()),
                new_ref_value_rms=float((blend@vr.float()).square().mean().sqrt()))
            assert self.current['routing']['p_ref_max_error']<2e-6
        applied=raw if correction is None else raw+correction
        applied=applied*torch.as_tensor(multiplier,device=raw.device,dtype=raw.dtype)
        self.current['injection']=dict(raw_rms=float(raw.float().square().mean().sqrt()),
            applied_rms=float(applied.float().square().mean().sqrt()),multiplier=multiplier,
            cosine=float(F.cosine_similarity(raw.float().flatten(),applied.float().flatten(),dim=0)),
            regions=regional_stats(applied,self.feature_weights))
        if index in STEPS and self.phase=='candidate':
            components,stats=decomposition(module,self.p,self.cache)
            q=split(self.cache['q'],module.heads)
            kr=split(self.cache['kr'].to(q),module.heads)
            logits=q.float()@kr.float().transpose(-1,-2)/math.sqrt(q.shape[-1])
            attention=torch.softmax(logits,-1)
            entropy=-(attention*attention.clamp_min(1e-20).log()).sum(-1)
            stats['reference_entropy_by_head']=entropy.mean(-1).flatten().tolist()
            ks=split(self.cache['k'].to(q),module.heads)
            self_logits=q.float()@ks.float().transpose(-1,-2)/math.sqrt(q.shape[-1])
            pref=torch.sigmoid(torch.logsumexp(logits,-1)-torch.logsumexp(self_logits,-1))
            stats['p_ref_by_head']=pref.mean(-1).flatten().tolist()
            queries=torch.linspace(0,q.shape[2]-1,24,device=q.device).long()
            self.attention_records.append(dict(index=index,p_ref=pref.cpu(),reference_entropy=entropy.cpu(),
                reference_attention_samples=attention[:,:,queries].to(torch.float16).cpu(),
                query_indices=queries.cpu(),head_query_shape=list(pref.shape)))
            stats['writer_token_norm_mean']=float(self.p.features.float().norm(dim=-1).mean())
            stats['writer_token_norm_std']=float(self.p.features.float().norm(dim=-1).std())
            stats['component_sum_relative_error']=float((components['added'].float()+components['suppressed'].float()
                -raw.float()).norm()/raw.float().norm().clamp_min(1e-8))
            self.current['softmax']=stats
            self.current['reference_shape']=list(self.p.features.shape)
            self.current['target_shape']=list(raw.shape)
        if correction is None and multiplier==1.:return result
        if multiplier==0.:return self.base
        return self.base+applied

    def forward(self,*args,**kwargs):
        conditional='sa_hidden_states' in (kwargs.get('cross_attention_kwargs') or {})
        if not conditional:
            self.phase=None;assert not self.p.enabled
            return self.old_forward(*args,**kwargs)
        latent=args[0] if args else kwargs['sample'];t=args[1] if len(args)>1 else kwargs['timestep']
        self.index=len(self.steps);self.current=dict(index=self.index,timestep=int(t),
            x_t_source='free_'+self.arm,dtype=str(latent.dtype),backend='cuda',valid_mask=self.valid)
        self.cache={};enabled=self.p.enabled
        self.latents.append(latent.detach().cpu())
        try:
            self.p.enabled=False;self.phase='current_e5'
            own=self.old_forward(*args,**kwargs)
            self.p.enabled=enabled;self.phase='candidate'
            candidate=self.old_forward(*args,**kwargs) if enabled else own
            direct=candidate[0].float()-own[0].float()
            self.current['direct']=regional_stats(direct,self.weights) if self.valid else None
            if self.teacher is not None:
                assert int(t)==int(self.teacher_scheduler.timesteps[self.index])
                teacher_eps=self.teacher['conditional_epsilon'][self.index].to(latent.device)
                state=own[0].float()-teacher_eps.float();total=candidate[0].float()-teacher_eps.float()
                self.current['state']=regional_stats(state,self.weights) if self.valid else None
                self.current['total']=regional_stats(total,self.weights) if self.valid else None
                self.current['decomposition_max_error']=float((total-direct-state).abs().max())
                assert self.current['decomposition_max_error']<2e-6
                self.current['latent_rms']=float((latent.float()-self.teacher_latent.float()).square().mean().sqrt())
                if self.index in STEPS:
                    newargs=list(args);newkw=dict(kwargs)
                    if newargs:newargs[0]=self.teacher_latent
                    else:newkw['sample']=self.teacher_latent
                    self.p.enabled=False;self.phase='teacher_e5'
                    baseline=self.old_forward(*newargs,**newkw)
                    assert torch.equal(baseline[0],teacher_eps),'独立 E5 教师轨迹回放失败'
                    saved_injection=self.current.get('injection');saved_routing=self.current.get('routing')
                    self.p.enabled=enabled;self.phase='teacher_probe';self.cache={}
                    probe=self.old_forward(*newargs,**newkw) if enabled else baseline
                    self.current['teacher_state_injection']=self.current.get('injection')
                    self.current['injection']=saved_injection
                    if saved_routing is not None:self.current['routing']=saved_routing
                    self.current['teacher_state_effect']=regional_stats(probe[0].float()-baseline[0].float(),self.weights) if self.valid else None
                    self.current['teacher_state_replay_exact']=True
                u=self.teacher['unconditional_epsilon'][self.index].to(latent.device)
                self.teacher_latent=self.teacher_scheduler.step(u+7.*(teacher_eps-u),t,
                    self.teacher_latent,return_dict=False)[0]
            if self.patch and self.valid and enabled and self.index in STEPS:
                self.current['patches']={}
                for site in self.sites:
                    assert site in self.activations
                    self.p.enabled=enabled;self.phase='patch';self.patch_site=site
                    patched=self.old_forward(*args,**kwargs)
                    self.current['patches'][site]=regional_stats(patched[0].float()-own[0].float(),self.weights)
                self.patch_site=None
            self.steps.append(self.current)
            return candidate
        finally:self.p.enabled=enabled;self.phase=None;self.patch_site=None

    def install(self):
        self.old_forward=self.r.pipe.unet.forward;self.r.pipe.unet.forward=self.forward
        self.p.diagnostic_observer=self.observer
        module=dict(self.r.pipe.unet.named_modules())[LAYER[:-10]]
        self.handles.append(module.register_forward_hook(self.output))
        for key in ['q','k','v']:
            self.handles.append(getattr(module,'to_'+key).register_forward_hook(self.capture(key)))
        self.handles.append(self.p.to_k_ref.register_forward_hook(self.capture('kr')))
        self.handles.append(self.p.to_v_ref.register_forward_hook(self.capture('vr')))
        if self.patch:
            modules=dict(self.r.pipe.unet.named_modules())
            for site in self.sites:self.handles.append(modules[site].register_forward_hook(self.site_hook(site)))

    def close(self):
        self.r.pipe.unet.forward=self.old_forward;self.p.diagnostic_observer=None
        for h in self.handles:h.remove()
