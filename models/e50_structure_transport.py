"""保留原算子，只替换 self 项；独立轨迹不受当前学生 latent 污染。"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def heads(x,n):
    b,t,c=x.shape
    return x.reshape(b,t,n,c//n).transpose(1,2)


def attend(q,k,v,n):
    y=F.scaled_dot_product_attention(heads(q,n),heads(k,n),heads(v,n))
    return y.transpose(1,2).reshape(q.shape[0],q.shape[1],-1)


def project(attn,delta):
    # 原输出偏置/残差/Sketch 已在原处理器中应用，增量不可重复加偏置。
    return attn.to_out[1](F.linear(delta,attn.to_out[0].weight,None))/attn.rescale_output_factor


def correspondence(target,reference,mask,hw,cosine_min=.2,margin_min=.02,cycle_max=.05):
    """双向近邻和空间回环检查；不假定重复纹理能建立一一对应。"""
    target=F.normalize(target.float(),dim=-1);reference=F.normalize(reference.float(),dim=-1)
    scores=target@reference.transpose(-1,-2)
    top,indices=scores.topk(2,dim=-1);forward=indices[...,0]
    # 反向只在目标服装内部搜索，不能把背景匹配当作外观证据。
    valid=mask[...,0]>0
    reverse=scores.masked_fill(~valid[...,None],-float('inf')).argmax(dim=1)
    returned=reverse.gather(1,forward)
    h,w=hw;positions=torch.arange(h*w,device=target.device)[None]
    error=(((returned//w-positions//w).float()).square()+
           ((returned%w-positions%w).float()).square()).sqrt()/math.hypot(h,w)
    confidence=valid & (top[...,0]>=cosine_min) & (top[...,0]-top[...,1]>=margin_min) & (error<=cycle_max)
    return forward,confidence,dict(score=top[...,0],margin=top[...,0]-top[...,1],cycle=error)


class ControlledAttention(nn.Module):
    def __init__(self,original,name,controller):
        super().__init__();self.original=original;self.name=name;self.controller=controller

    def __call__(self,attn,hidden_states,encoder_hidden_states=None,attention_mask=None,temb=None,
            scale=1.,cond_hidden_states=None,sa_hidden_states=None,balanced_gate_timestep=None,
            tcpm_garment_mask=None,local_detail_tokens=None,local_detail_mask=None,local_detail_spatial_shape=None):
        ctl=self.controller
        kw=dict(encoder_hidden_states=encoder_hidden_states,attention_mask=attention_mask,temb=temb,
            scale=scale,cond_hidden_states=cond_hidden_states,sa_hidden_states=sa_hidden_states,
            balanced_gate_timestep=balanced_gate_timestep,tcpm_garment_mask=tcpm_garment_mask,
            local_detail_tokens=local_detail_tokens,local_detail_mask=local_detail_mask,
            local_detail_spatial_shape=local_detail_spatial_shape)
        base=self.original(attn,hidden_states,**kw)
        if ctl.phase not in ['teacher','candidate'] or ctl.disabled or not ctl.valid:return base
        assert hidden_states.ndim==3 and attn.spatial_norm is None and attn.group_norm is None
        assert encoder_hidden_states is None and attention_mask is None
        q,k,v=attn.to_q(hidden_states),attn.to_k(hidden_states),attn.to_v(hidden_states)
        if ctl.phase=='teacher':
            ctl.cache[self.name]=dict(q=q.detach(),k=k.detach(),h=hidden_states.detach())
            return base
        stored=ctl.cache[self.name]
        anchored=attend(stored['q'],stored['k'],v,attn.heads)
        own=attend(q,k,v,attn.heads)
        if ctl.kind=='MATCH' and self.name==ctl.match_layer:
            assert not ctl.p.enabled,'MATCH 必须替换原 B1 外观分支，不能叠加两次'
            ref=ctl.p.norm(ctl.p.features.float())
            h,w=ctl.target_hw
            mask=F.interpolate(ctl.soft,size=(h,w),mode='area').flatten(2).transpose(1,2).to(v)
            forward,conf,stats=correspondence(stored['h'],ref,mask,(h,w))
            rv=ctl.p.to_v_ref(ref).to(v)
            warped=rv.gather(1,forward[...,None].expand(-1,-1,rv.shape[-1]))
            mix=(ctl.p.gate.to(v)*mask*conf[...,None].to(v))*(warped-v)
            applied=attend(stored['q'],stored['k'],mix,attn.heads)
            anchored=anchored+applied
            valid=mask[...,0]>0;denom=max(int(valid.sum()),1)
            ctl.current['matching']=dict(coverage=float(conf.sum())/denom,
                matched_cells=int(conf.sum()),foreground_cells=int(valid.sum()),
                mean_cosine=float(stats['score'][valid].mean()) if valid.any() else None,
                mean_margin=float(stats['margin'][valid].mean()) if valid.any() else None,
                mean_cycle=float(stats['cycle'][valid].mean()) if valid.any() else None,
                local_value_delta_rms=float(mix.float().square().mean().sqrt()),
                self_output_delta_rms=float(applied.float().square().mean().sqrt()))
            ctl.maps.append(dict(index=ctl.index,forward=forward.cpu(),confidence=conf.cpu(),
                mask=mask.cpu(),**{k:t.cpu() for k,t in stats.items()}))
        delta=project(attn,anchored-own)
        ctl.current.setdefault('attention_delta_rms',{})[self.name]=float(delta.detach().float().square().mean().sqrt())
        return base+delta


class StructureTransport:
    def __init__(self,r,row,arm,baseline,disabled=False,audit=False):
        from PIL import Image
        from tools.e44_runtime import sketch_masks
        from tools.e50_protocol import SIZE,LAYER,FEATURE
        self.r,self.p=r,r.processor;self.arm=arm;self.kind=arm.replace('_ENERGY','')
        self.disabled=disabled;self.audit=audit;self.phase=None;self.index=0
        _,self.soft,info=sketch_masks(Image.open(row['sketch']).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR),r.pipe.device)
        self.valid=info['valid'];self.match_layer=LAYER[:-10];self.feature_layer=FEATURE
        self.target_hw=(32,24);self.cache={};self.steps=[];self.maps=[];self.handles=[];self.originals={}
        self.trace=baseline;self.latents=self.replay(baseline)
        self.unconditional=0

    def replay(self,trace):
        from diffusers import DDIMScheduler
        scheduler=DDIMScheduler.from_config(self.r.pipe.scheduler.config)
        scheduler.set_timesteps(50,device=self.r.pipe.device)
        latent=trace['initial_latent'].to(self.r.pipe.device);values=[]
        for i,t in enumerate(scheduler.timesteps):
            values.append(latent.detach().cpu())
            ec,eu=trace['conditional_epsilon'][i].to(latent),trace['unconditional_epsilon'][i].to(latent)
            latent=scheduler.step(eu+7.*(ec-eu),t,latent,return_dict=False)[0]
        assert torch.equal(latent.cpu(),trace['final_latent']),'独立基线完整 scheduler 回放失败'
        return values

    def feature(self,module,args,result):
        if self.disabled or not self.valid:return result
        if self.phase=='teacher':self.cache['feature']=result.detach()
        elif self.phase=='candidate' and self.index<30:
            target=self.cache['feature'];assert result.shape==target.shape
            self.current['feature_delta_rms']=float((result.float()-target.float()).square().mean().sqrt())
            return target.clone()
        return result

    def at_latent(self,args,kwargs,latent):
        if args:return (latent,)+args[1:],kwargs
        kw=dict(kwargs);kw['sample']=latent;return args,kw

    def forward(self,*args,**kwargs):
        conditional='sa_hidden_states' in (kwargs.get('cross_attention_kwargs') or {})
        if not conditional:
            self.phase=None;self.unconditional+=1
            out=self.old_forward(*args,**kwargs)
            if self.audit and self.unconditional in [1,50]:
                assert torch.equal(out[0],self.old_forward(*args,**kwargs)[0])
            return out
        enabled=self.p.enabled;latent=args[0] if args else kwargs['sample']
        self.index=len(self.steps);self.current=dict(index=self.index,enabled=enabled,independent_epsilon_exact=False)
        self.cache={}
        try:
            self.p.enabled=False;self.phase='teacher'
            ta,tk=self.at_latent(args,kwargs,self.latents[self.index].to(latent))
            teacher=self.old_forward(*ta,**tk)
            assert torch.equal(teacher[0].cpu(),self.trace['conditional_epsilon'][self.index]),'独立 E5 ε 不是历史精确轨迹'
            self.current['independent_epsilon_exact']=True
            self.phase=None;self.p.enabled=False
            base=self.old_forward(*args,**kwargs)
            self.p.enabled=enabled
            raw=self.old_forward(*args,**kwargs) if enabled else base
            self.phase='candidate';self.p.enabled=enabled and self.kind!='MATCH'
            candidate=self.old_forward(*args,**kwargs) if self.valid and not self.disabled else (raw if self.kind=='STRUCT' else base)
            effect=candidate[0].float()-base[0].float();original=raw[0].float()-base[0].float()
            norm=effect.square().sum();rawnorm=original.square().sum()
            self.current.update(candidate_epsilon_energy=float(norm),global_B1_epsilon_energy=float(rawnorm))
            if self.arm.endswith('_ENERGY'):
                assert rawnorm>0 or norm==0,'非零候选不可用零参考增量做能量对照'
                amplitude=(norm/rawnorm).sqrt() if rawnorm>0 else norm
                applied=original*amplitude
                eps=(base[0].float()+applied).to(base[0].dtype)
                candidate=(eps,)+base[1:]
                actual=(eps.float()-base[0].float()).square().sum()
                self.current.update(amplitude=float(amplitude),control_ideal_energy=float(applied.square().sum()),
                    control_actual_energy=float(actual),rounding_relative_error=float((actual-norm).abs()/norm.clamp_min(1e-12)))
            assert torch.isfinite(candidate[0]).all()
            self.steps.append(self.current);return candidate
        finally:self.phase=None;self.p.enabled=enabled

    def install(self):
        from tools.e50_protocol import ATTENTIONS
        modules=dict(self.r.pipe.unet.named_modules())
        for name in ATTENTIONS:
            attn=modules[name];self.originals[name]=attn.processor
            attn.set_processor(ControlledAttention(attn.processor,name,self))
        self.handles.append(modules[self.feature_layer].register_forward_hook(self.feature))
        self.old_forward=self.r.pipe.unet.forward;self.r.pipe.unet.forward=self.forward

    def close(self):
        self.r.pipe.unet.forward=self.old_forward
        for h in self.handles:h.remove()
        modules=dict(self.r.pipe.unet.named_modules())
        for name,original in self.originals.items():modules[name].set_processor(original)
