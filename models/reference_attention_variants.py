"""保留原 E5（包括草图 K/V），只在一个 attn1 上增加参考分支。"""
import torch
from torch import nn
from torch.nn import functional as F


class ReferenceAttentionProcessor(nn.Module):
    def __init__(self, original, name, hidden_size, reference_size, variant):
        super().__init__()
        self.original = original
        self.name, self.variant = name, variant
        self.norm = nn.LayerNorm(reference_size)
        self.to_k_ref = nn.Linear(reference_size, hidden_size, bias=False)
        self.to_v_ref = nn.Linear(reference_size, hidden_size, bias=False)
        if variant == 'B1':
            self.gate = nn.Parameter(torch.zeros(()))
        else:
            self.output = nn.Linear(hidden_size, hidden_size, bias=False)
            nn.init.zeros_(self.output.weight)
        self.features = self.mask = None
        self.enabled = False
        self.branch = 'unconditional'
        self.trace = []

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                attention_mask=None, temb=None, scale=1.0, cond_hidden_states=None,
                sa_hidden_states=None, balanced_gate_timestep=None,
                tcpm_garment_mask=None, local_detail_tokens=None,
                local_detail_mask=None, local_detail_spatial_shape=None):
        # OFF 直接调用原算子，避免重写原 E5 的归一化、偏置和残差约定。
        base = self.original(attn, hidden_states,
            encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask,
            temb=temb, scale=scale, cond_hidden_states=cond_hidden_states,
            sa_hidden_states=sa_hidden_states, balanced_gate_timestep=balanced_gate_timestep,
            tcpm_garment_mask=tcpm_garment_mask, local_detail_tokens=local_detail_tokens,
            local_detail_mask=local_detail_mask,
            local_detail_spatial_shape=local_detail_spatial_shape)
        if not self.enabled or sa_hidden_states is None or self.features is None:
            return base
        x = hidden_states
        if attn.spatial_norm is not None:
            x = attn.spatial_norm(x, temb)
        shape = x.shape
        if x.ndim == 4:
            x = x.flatten(2).transpose(1, 2)
        if attn.group_norm is not None:
            x = attn.group_norm(x.transpose(1, 2)).transpose(1, 2)
        b, n, c = x.shape
        q = attn.to_q(x)
        f = self.norm(self.features.float())
        k, v = self.to_k_ref(f).to(q.dtype), self.to_v_ref(f).to(q.dtype)
        heads = attn.heads
        def split(y):
            return y.reshape(b, -1, heads, c // heads).transpose(1, 2)
        qh = split(q)
        if self.variant == 'B1':
            # E5 的 LogoRef 把草图作为独立注意力相加；base 已完整保留它。
            # B1 只对原目标 self-attention 作联合插值，草图项保持不变。
            ks, vs = attn.to_k(x), attn.to_v(x)
            original = F.scaled_dot_product_attention(qh, split(ks), split(vs))
            joint = F.scaled_dot_product_attention(qh,
                split(torch.cat([ks, k], dim=1)), split(torch.cat([vs, v], dim=1)))
            delta = self.gate * (joint - original)
            delta = delta.transpose(1, 2).reshape(b, n, c)
        else:
            ref = F.scaled_dot_product_attention(qh, split(k), split(v))
            ref = ref.transpose(1, 2).reshape(b, n, c)
            delta = .1 * self.output(ref.float()).to(q.dtype)
            delta = delta * self.mask.to(delta).flatten(2).transpose(1, 2)
        # 对增量只应用原线性层的权重，偏置已经包含在 base 中。
        delta = F.linear(delta.to(q.dtype), attn.to_out[0].weight, None)
        delta = attn.to_out[1](delta)
        if len(shape) == 4:
            delta = delta.transpose(1, 2).reshape(shape)
        delta = delta / attn.rescale_output_factor
        if len(self.trace) < 8:
            self.trace.append(dict(branch=self.branch, target_shape=list(shape),
                reference_shape=list(self.features.shape), delta_rms=float(delta.detach().float().square().mean().sqrt()),
                mask_outside_max=float((delta.detach().flatten(2).transpose(1, 2)
                    if delta.ndim == 4 else delta.detach())[self.mask.flatten(2).transpose(1, 2).squeeze(-1) == 0].abs().max())
                    if self.variant != 'B1' and (self.mask == 0).any() else None))
        return base + delta

    def trainable_state(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith('original.')}
