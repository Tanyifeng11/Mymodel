"""冻结 B1 的真实投影残差，学习草图条件控制；不重复原输出偏置。"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from models.reference_attention_variants import ReferenceAttentionProcessor


def time_embedding(t, width=64):
    t=t.reshape(-1).float()
    frequencies=torch.exp(-math.log(10000)*torch.arange(width//2,device=t.device)/(width//2))
    phases=t[:,None]*frequencies[None]
    return torch.cat([phases.cos(),phases.sin()],dim=-1)


class ConditionalResidualController(nn.Module):
    def __init__(self, width=640, bottleneck=32):
        super().__init__()
        self.target=nn.Conv2d(width,bottleneck,1)
        self.reference=nn.Conv2d(width,bottleneck,1)
        self.mix=nn.Conv2d(2*bottleneck+3,bottleneck,3,padding=1)
        self.norm=nn.GroupNorm(4,bottleneck)
        self.time=nn.Sequential(nn.Linear(64,64),nn.SiLU(),nn.Linear(64,2*bottleneck))
        self.gating=nn.Conv2d(bottleneck,1,1)
        self.output=nn.Conv2d(bottleneck,width,1)
        nn.init.zeros_(self.gating.weight);nn.init.zeros_(self.gating.bias)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self, target, residual, sketch, timestep, timed):
        # 新模块保留 FP32；主干和冻结 B1 保持历史半精度算术。
        with torch.autocast(device_type=target.device.type,enabled=False):
            x=F.silu(self.target(target.float()))
            r=F.silu(self.reference(residual.float()))
            z=self.norm(self.mix(torch.cat([x,r,sketch.float()],dim=1)))
            e=time_embedding(timestep).to(z)
            if not timed:e=torch.zeros_like(e)
            scale,shift=self.time(e).chunk(2,dim=-1)
            z=F.silu(z*(1+scale[:,:,None,None])+shift[:,:,None,None])
            gate=self.gating(z).sigmoid()
            proposed=self.output(z)
            source_rms=residual.float().square().mean((1,2,3),keepdim=True).sqrt()
            proposed_rms=proposed.square().mean((1,2,3),keepdim=True).clamp_min(1e-16).sqrt()
            # 只缩小：投影后的整张特征残差 RMS 不超过当前原 B1 RMS。
            # 这不保证局部像素或下游 epsilon/RGB 的幅度、轮廓保持不变。
            cap=(source_rms/proposed_rms.clamp_min(1e-8)).clamp(max=1)
            return proposed*cap,gate,cap


class ControlledB1Processor(ReferenceAttentionProcessor):
    def __call__(self,attn,hidden_states,encoder_hidden_states=None,attention_mask=None,temb=None,
            scale=1.,cond_hidden_states=None,sa_hidden_states=None,balanced_gate_timestep=None,
            tcpm_garment_mask=None,local_detail_tokens=None,local_detail_mask=None,local_detail_spatial_shape=None):
        # Diffusers 会按显式签名过滤 cross_attention_kwargs；必须保留原条件参数。
        kwargs=dict(encoder_hidden_states=encoder_hidden_states,attention_mask=attention_mask,temb=temb,
            scale=scale,cond_hidden_states=cond_hidden_states,sa_hidden_states=sa_hidden_states,
            balanced_gate_timestep=balanced_gate_timestep,tcpm_garment_mask=tcpm_garment_mask,
            local_detail_tokens=local_detail_tokens,local_detail_mask=local_detail_mask,
            local_detail_spatial_shape=local_detail_spatial_shape)
        if not self.enabled or kwargs.get('sa_hidden_states') is None or self.features is None:
            return super().__call__(attn,hidden_states,**kwargs)
        if self.raw_bypass:return super().__call__(attn,hidden_states,**kwargs)
        if self.residual_scale==0:
            enabled=self.enabled;self.enabled=False
            try:return super().__call__(attn,hidden_states,**kwargs)
            finally:self.enabled=enabled
        values={};observer=self.diagnostic_observer
        self.diagnostic_observer=lambda base,raw,applied:values.update(base=base,raw=raw)
        try:
            with torch.no_grad():super().__call__(attn,hidden_states,**kwargs)
        finally:self.diagnostic_observer=observer
        base,raw=values['base'],values['raw']
        assert hidden_states.ndim==3 and hidden_states.shape[1:]==(768,640)
        b,n,c=hidden_states.shape;h,w=32,24
        target=hidden_states.detach().transpose(1,2).reshape(b,c,h,w)
        residual=raw.detach().transpose(1,2).reshape(b,c,h,w)
        sketch=F.interpolate(self.geometry,size=(h,w),mode='area')
        control,gate,cap=self.controller(target,residual,sketch,self.timestep,self.timed)
        mask=self.mask.float()
        applied=self.residual_scale*mask*gate*control
        result=base+applied.flatten(2).transpose(1,2).to(base)
        def rms(v):return float(v.detach().float().square().mean().sqrt())
        self.last_control=dict(timestep=int(self.timestep),gate_mean=float(gate.detach().mean()),
            gate_min=float(gate.detach().min()),gate_max=float(gate.detach().max()),
            raw_b1_rms=rms(residual),controlled_rms=rms(control),applied_rms=rms(applied),
            cap=float(cap.detach().mean()),mask_zero_max=float(applied.detach()[mask.expand_as(applied)==0].abs().max())
                if (mask==0).any() else None)
        if self.capture_maps:self.control_maps.append(gate.detach().cpu())
        if observer is not None:observer(base,raw,applied.flatten(2).transpose(1,2))
        return result
