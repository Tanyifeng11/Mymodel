"""独立参考注意力：沿用 B1 的 Writer、参数与零 gate，移除共享 softmax 竞争。"""
import torch
from torch.nn import functional as F
from models.reference_attention_variants import ReferenceAttentionProcessor


class IndependentReferenceProcessor(ReferenceAttentionProcessor):
    def __call__(self,attn,hidden_states,encoder_hidden_states=None,attention_mask=None,
                 temb=None,scale=1.,cond_hidden_states=None,sa_hidden_states=None,
                 balanced_gate_timestep=None,tcpm_garment_mask=None,local_detail_tokens=None,
                 local_detail_mask=None,local_detail_spatial_shape=None):
        base=self.original(attn,hidden_states,encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,temb=temb,scale=scale,cond_hidden_states=cond_hidden_states,
            sa_hidden_states=sa_hidden_states,balanced_gate_timestep=balanced_gate_timestep,
            tcpm_garment_mask=tcpm_garment_mask,local_detail_tokens=local_detail_tokens,
            local_detail_mask=local_detail_mask,local_detail_spatial_shape=local_detail_spatial_shape)
        if not self.enabled or sa_hidden_states is None or self.features is None:return base
        x=hidden_states
        if attn.spatial_norm is not None:x=attn.spatial_norm(x,temb)
        shape=x.shape
        if x.ndim==4:x=x.flatten(2).transpose(1,2)
        if attn.group_norm is not None:x=attn.group_norm(x.transpose(1,2)).transpose(1,2)
        b,n,c=x.shape;heads=attn.heads
        q=attn.to_q(x);f=self.norm(self.features.float())
        k,v=self.to_k_ref(f).to(q.dtype),self.to_v_ref(f).to(q.dtype)
        def split(y):return y.reshape(b,-1,heads,c//heads).transpose(1,2)
        reference=F.scaled_dot_product_attention(split(q),split(k),split(v))
        delta=(self.gate*reference).transpose(1,2).reshape(b,n,c)
        delta=F.linear(delta,attn.to_out[0].weight,None);delta=attn.to_out[1](delta)
        if len(shape)==4:delta=delta.transpose(1,2).reshape(shape)
        delta=delta/attn.rescale_output_factor
        if self.diagnostic_observer is not None:self.diagnostic_observer(base,delta,delta)
        return base+delta


def configure(r,arm):
    if arm=='DECOUPLED':
        # 替换行为而不重新初始化：K/V/Norm 与同 seed 的原 B1 初值逐值相同。
        r.processor.__class__=IndependentReferenceProcessor
    else:assert arm=='BOUNDARY'
    return r
