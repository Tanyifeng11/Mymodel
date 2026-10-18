"""M54机制适配；保留原E5算子，只在条件分支增加新计算路径。"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from models.reference_attention_variants import ReferenceAttentionProcessor


def high_frequency(x):
    low=F.avg_pool1d(x.transpose(1,2),5,stride=2,padding=2,count_include_pad=False)
    low=F.interpolate(low,size=x.shape[1],mode='linear',align_corners=False).transpose(1,2)
    return x-low


def split_heads(x,heads):return x.reshape(x.shape[0],-1,heads,x.shape[-1]//heads).transpose(1,2)


def attention_summary(attention):
    a=attention.detach();yy,xx=torch.meshgrid(torch.linspace(0,1,16,device=a.device),torch.linspace(0,1,16,device=a.device),indexing='ij')
    coords=torch.stack([xx.flatten(),yy.flatten()],-1);center=a@coords
    spread=((a@coords.square().sum(-1))-center.square().sum(-1)).clamp_min(0).sqrt().mean()
    return dict(entropy=float(-(a*a.clamp_min(1e-30).log()).sum(-1).mean()),
        top1_mass=float(a.max(-1).values.mean()),top5_mass=float(a.topk(5,dim=-1).values.sum(-1).mean()),spatial_spread=float(spread))


def m10_statistic_residual(target, reference, mode, mask=None):
    """M10前置消融：不增加参数，均值/标准差沿原token维计算。"""
    x, v = target.float(), reference.float()
    centered = x - x.mean(1, keepdim=True)
    mean_delta = v.mean(1, keepdim=True) - x.mean(1, keepdim=True)
    std_delta = centered / x.std(1, keepdim=True, unbiased=False).clamp_min(1e-6)
    std_delta = std_delta * v.std(1, keepdim=True, unbiased=False) - centered
    if mode == 'M10_MEAN':
        return mean_delta.expand_as(x)
    if mode == 'M10_STD':
        return std_delta
    if mode == 'M10_MASK':
        assert mask is not None
        # 保持原M10运算顺序，仅在残差后乘既有softmask。
        stat = centered / x.std(1, keepdim=True, unbiased=False).clamp_min(1e-6)
        stat = stat * v.std(1, keepdim=True, unbiased=False) + v.mean(1, keepdim=True)
        return (stat - x) * mask.to(x).flatten(2).transpose(1, 2)
    raise ValueError(mode)


class Candidate(nn.Module):
    def __init__(self,kind,width=640):
        super().__init__();self.kind=kind;self.width=width;self.tau=1.;self.last={};self.atv=None
        if kind=='M9':
            self.input=nn.Conv2d(width+4,32,3,padding=1)
            self.residual=nn.Sequential(nn.GroupNorm(4,32),nn.SiLU(),nn.Conv2d(32,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,32,3,padding=1))
            self.output=nn.Conv2d(32,width,1)
            self.alpha=nn.Parameter(torch.ones(()))
        else:
            # M11不允许归一化偏置在零高频输入上凭空生成参考内容。
            self.ref_norm=nn.LayerNorm(width,bias=(kind!='M11'))
            self.key=nn.Linear(width,width,bias=False);self.value=nn.Linear(width,width,bias=False)
            if kind in ['M1','M6']:
                self.target_norm=nn.LayerNorm(width);self.query=nn.Linear(width,width,bias=False)
            if kind=='M3':
                self.global_key=nn.Linear(width,width,bias=False);self.global_value=nn.Linear(width,width,bias=False)
                self.global_output=nn.Linear(width,width,bias=False)
                nn.init.zeros_(self.global_output.weight)
                self.a=nn.Parameter(torch.zeros(()));self.b=nn.Parameter(torch.zeros(()))
            if kind=='M4':
                self.local=nn.Sequential(nn.Conv2d(width,width,3,padding=1,groups=width),nn.GELU(),nn.Conv2d(width,width,1))
                self.gating=nn.Sequential(nn.Conv2d(width+4,32,1),nn.SiLU(),nn.Conv2d(32,1,1))
            if kind=='M5':
                self.fusion_qkv=nn.Linear(width,3*160,bias=False)
                self.fusion_out=nn.Linear(160,width,bias=False)
            self.output=nn.Linear(width,width,bias=False)
        nn.init.zeros_(self.output.weight)
        if self.output.bias is not None:nn.init.zeros_(self.output.bias)

    def forward(self,x,reference,frozen_q,heads,geometry,timestep,spatial_prior=None):
        b,n,c=x.shape;h,w=geometry.shape[-2:]
        time=torch.as_tensor(timestep,device=x.device,dtype=torch.float32).reshape(-1,1,1,1)/1000
        time=time.expand(b,1,h,w);maps=torch.cat([geometry.float(),time],1)
        self.atv=None
        if self.kind=='M9':
            z=self.input(torch.cat([x.float().transpose(1,2).reshape(b,c,h,w),maps],1))
            delta=self.alpha*self.output(z+self.residual(z))
            return delta.flatten(2).transpose(1,2)
        f=reference.float()
        if self.kind=='M11':f=high_frequency(f)
        f=self.ref_norm(f)
        if self.kind=='M4':f=self.local(f.transpose(1,2).reshape(b,c,16,16)).flatten(2).transpose(1,2)
        q=self.query(self.target_norm(x.float())) if hasattr(self,'query') else frozen_q.float()
        k=self.key(f);v=self.value(f)
        qh,kh,vh=(split_heads(z,heads) for z in [q,k,v])
        if self.kind=='M6':
            logits=F.normalize(qh,dim=-1)@F.normalize(kh,dim=-1).transpose(-1,-2)/.07
            assert spatial_prior is not None
            logits=logits+spatial_prior.clamp_min(1e-30).log()[:,None]
        else:logits=(qh@kh.transpose(-1,-2))/(math.sqrt(c//heads)*self.tau)
        attention=logits.softmax(-1);aligned=(attention@vh).transpose(1,2).reshape(b,n,c)
        self.last=attention_summary(attention)
        if self.kind=='M1':
            yy,xx=torch.meshgrid(torch.linspace(0,1,16,device=x.device),torch.linspace(0,1,16,device=x.device),indexing='ij')
            center=attention.mean(1)@torch.stack([xx.flatten(),yy.flatten()],-1)
            field=center.transpose(1,2).reshape(b,2,h,w)*geometry[:,:1]
            self.atv=(field[:,:,1:]-field[:,:,:-1]).abs().mean()+(field[:,:,:,1:]-field[:,:,:,:-1]).abs().mean()
        if self.kind=='M3':
            global_f=F.adaptive_avg_pool2d(f.transpose(1,2).reshape(b,c,16,16),(2,2)).flatten(2).transpose(1,2)
            gk,gv=split_heads(self.global_key(global_f),heads),split_heads(self.global_value(global_f),heads)
            g=F.scaled_dot_product_attention(qh,gk,gv).transpose(1,2).reshape(b,n,c)
            return self.a.sigmoid()*self.global_output(g)+self.b.sigmoid()*self.output(aligned)
        if self.kind=='M5':
            joint=torch.cat([x.float(),aligned],1);qf,kf,vf=self.fusion_qkv(joint).chunk(3,-1)
            fused=F.scaled_dot_product_attention(*(split_heads(z,4) for z in [qf,kf,vf])).transpose(1,2).reshape(b,2*n,160)
            aligned=self.fusion_out(fused[:,:n])-x.float()
        delta=self.output(aligned)
        if self.kind=='M4':
            gate=self.gating(torch.cat([x.float().transpose(1,2).reshape(b,c,h,w),maps],1)).sigmoid()
            delta=delta*gate.flatten(2).transpose(1,2)
        return delta


class CandidateCombination(nn.Module):
    """两条独立训练、冻结的残差相加；不新增可学习参数或组合训练。"""
    def __init__(self,components,scales):
        super().__init__();self.components=nn.ModuleList(components);self.scales=scales;self.last={}

    def forward(self,*args):
        outputs=[component(*args)*scale for component,scale in zip(self.components,self.scales)]
        self.last={str(i)+'_'+key:value for i,component in enumerate(self.components) for key,value in component.last.items()}
        return outputs[0]+outputs[1]


class M54Processor(ReferenceAttentionProcessor):
    """固定B1臂沿用原始参数；独立训练臂只更新candidate。"""
    def __call__(self,attn,hidden_states,encoder_hidden_states=None,attention_mask=None,temb=None,scale=1.,
                 cond_hidden_states=None,sa_hidden_states=None,balanced_gate_timestep=None,
                 tcpm_garment_mask=None,local_detail_tokens=None,local_detail_mask=None,local_detail_spatial_shape=None):
        # Diffusers按显式签名过滤cross_attention_kwargs；仅写**kwargs会丢失Sketch条件。
        kwargs=dict(cond_hidden_states=cond_hidden_states,sa_hidden_states=sa_hidden_states,
            balanced_gate_timestep=balanced_gate_timestep,tcpm_garment_mask=tcpm_garment_mask,
            local_detail_tokens=local_detail_tokens,local_detail_mask=local_detail_mask,
            local_detail_spatial_shape=local_detail_spatial_shape)
        # M7与tau1使用原融合SDPA，要求精确回放而不是近似重写。
        if self.mode in ['B1','SAFE'] or self.mode.startswith('M7') or self.mode=='M8_T100':
            result=super().__call__(attn,hidden_states,encoder_hidden_states,attention_mask,temb,scale,**kwargs)
            if self.mode=='M8_T100' and self.enabled and self.features is not None:
                x=hidden_states
                if attn.spatial_norm is not None:x=attn.spatial_norm(x,temb)
                if x.ndim==4:x=x.flatten(2).transpose(1,2)
                if attn.group_norm is not None:x=attn.group_norm(x.transpose(1,2)).transpose(1,2)
                q=split_heads(attn.to_q(x),attn.heads).float()
                k=split_heads(self.to_k_ref(self.norm(self.features.float())).to(x),attn.heads).float()
                logits=q@k.transpose(-1,-2)/math.sqrt(q.shape[-1])
                self.last_delta.update(attention_summary(logits.softmax(-1)))
            return result
        base=self.original(attn,hidden_states,encoder_hidden_states=encoder_hidden_states,attention_mask=attention_mask,temb=temb,scale=scale,**kwargs)
        if not self.enabled or kwargs.get('sa_hidden_states') is None or self.features is None:return base
        x=hidden_states
        if attn.spatial_norm is not None:x=attn.spatial_norm(x,temb)
        shape=x.shape
        if x.ndim==4:x=x.flatten(2).transpose(1,2)
        if attn.group_norm is not None:x=attn.group_norm(x.transpose(1,2)).transpose(1,2)
        b,n,c=x.shape;heads=attn.heads;q=attn.to_q(x)
        if self.mode=='M9':
            b1=super().__call__(attn,hidden_states,encoder_hidden_states,attention_mask,temb,scale,**kwargs)
            if not self.new_branch_active:return b1
            with torch.autocast(device_type=x.device.type,enabled=False):
                delta=self.candidate(x,self.features,q,heads,self.geometry,self.timestep)
            delta=delta.to(b1.dtype)/attn.rescale_output_factor
            if len(shape)==4:delta=delta.transpose(1,2).reshape(shape)
            result=b1+delta
            self.last_delta=dict(rms=float((result-base).detach().float().square().mean().sqrt()),
                structure_rms=float(delta.detach().float().square().mean().sqrt()))
            return result
        if hasattr(self,'candidate') and self.mode!='M9':
            with torch.autocast(device_type=x.device.type,enabled=False):
                delta=self.candidate(x,self.features,q,heads,self.geometry,self.timestep,self.spatial_prior)
            # 输出层已学习目标feature残差，不再叠加原B1 reference joint branch。
        else:
            f=self.norm(self.features.float());k=self.to_k_ref(f).to(q);v=self.to_v_ref(f).to(q)
            if self.mode=='HF_ONLY':v=high_frequency(v)
            qh=split_heads(q,heads);ks=split_heads(attn.to_k(x),heads);vs=split_heads(attn.to_v(x),heads)
            kr,vr=split_heads(k,heads),split_heads(v,heads)
            if self.mode in ['M10','M10_GAIN']:
                xf=x.float();vf=v.float();stat=(xf-xf.mean(1,keepdim=True))/xf.std(1,keepdim=True,unbiased=False).clamp_min(1e-6)
                stat=stat*vf.std(1,keepdim=True,unbiased=False)+vf.mean(1,keepdim=True)
                delta=self.gate*(stat-xf)
                if self.mode=='M10_GAIN':delta=delta*self.m10_gain(self.timestep)
            elif self.mode in ['M10_MEAN','M10_STD','M10_MASK']:
                delta=self.gate*m10_statistic_residual(x,v,self.mode,self.mask)
            else:
                logits_s=qh.float()@ks.float().transpose(-1,-2)/math.sqrt(c//heads)
                logits_r=qh.float()@kr.float().transpose(-1,-2)/math.sqrt(c//heads)
                p_ref=(logits_r.logsumexp(-1)-torch.logaddexp(logits_s.logsumexp(-1),logits_r.logsumexp(-1))).exp()[...,None]
                self_value=F.scaled_dot_product_attention(qh,ks,vs).float()
                tau={'M8_T085':.85,'M8_T070':.70}.get(self.mode,1.)
                a=(logits_r/tau).softmax(-1)
                if self.mode=='ORACLE':
                    m=self.oracle.to(a);valid=m.sum(-1)>.99
                    a=torch.where(valid[None,None,:,None],m[None,None],a)
                ref=a@vr.float();delta=(self.gate*p_ref*(ref-self_value)).transpose(1,2).reshape(b,n,c)
                self.last_attention=dict(**attention_summary(a),p_ref=float(p_ref.detach().mean()))
            delta=F.linear(delta.to(q),attn.to_out[0].weight,None);delta=attn.to_out[1](delta)
        delta=(delta*getattr(self,'effect_scale',1.)).to(base.dtype)/attn.rescale_output_factor
        if len(shape)==4:delta=delta.transpose(1,2).reshape(shape)
        result=base+delta
        self.last_delta=dict(rms=float((result-base).detach().float().square().mean().sqrt()),**getattr(self,'last_attention',{}))
        if hasattr(self,'candidate'):self.last_delta.update(self.candidate.last)
        return result
