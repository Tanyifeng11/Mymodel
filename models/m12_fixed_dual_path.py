"""冻结原生 B1/M10，从同一目标状态计算两个投影后的增量。"""
import copy
import torch
from torch import nn
from torch.nn import functional as F


class FrozenPath(nn.Module):
    def __init__(self, source, attention):
        super().__init__()
        self.norm = copy.deepcopy(source.norm)
        self.key = copy.deepcopy(source.to_k_ref)
        self.value = copy.deepcopy(source.to_v_ref)
        self.gain = nn.Parameter(source.gate.detach().clone(), requires_grad=False)
        self.projection = copy.deepcopy(attention.to_out[0])
        self.dropout = copy.deepcopy(attention.to_out[1])
        self.eval().requires_grad_(False)

    def reference(self, features, query):
        features = self.norm(features.float())
        return self.key(features).to(query), self.value(features).to(query)

    def project(self, delta, query):
        return self.dropout(F.linear(delta.to(query), self.projection.weight, None))


class FixedDualPathProcessor(nn.Module):
    def __init__(self, source, attention, alpha_b, alpha_m):
        super().__init__()
        self.original = source.original
        self.b1 = FrozenPath(source, attention)
        self.m10 = FrozenPath(source, attention)
        self.alpha_b, self.alpha_m = float(alpha_b), float(alpha_m)
        self.enabled = False
        self.features = self.mask = self.geometry = None
        self.timestep = torch.tensor([0])
        self.branch = 'unconditional'
        self.records = []
        self.last_delta = {}
        self.steps = {}
        self.record_enabled = True
        self.eval().requires_grad_(False)

    def reset(self):
        self.records = []
        self.steps = {}

    def observe(self, delta_b, delta_m, combined, active):
        branch = self.branch
        step = self.steps.get(branch, 0)
        self.steps[branch] = step + 1
        if not self.record_enabled:
            return
        shape = delta_b.shape
        if delta_b.ndim == 4:
            delta_b = delta_b.flatten(2).transpose(1, 2)
            delta_m = delta_m.flatten(2).transpose(1, 2)
            combined = combined.flatten(2).transpose(1, 2)
        delta_b, delta_m, combined = (value.detach().float() for value in [delta_b, delta_m, combined])
        assert all(torch.isfinite(value).all() for value in [delta_b, delta_m, combined])
        garment = self.geometry[:, :1].flatten(2).transpose(1, 2) >= .5
        boundary = self.geometry[:, 1:2].flatten(2).transpose(1, 2) > 0
        regions = dict(all=torch.ones_like(garment), garment=garment, boundary=boundary, outside=~garment)
        for name, mask in regions.items():
            expanded = mask.expand_as(combined)
            count = int(expanded.sum())
            selected_b, selected_m, selected_combo = [value[expanded] for value in [delta_b, delta_m, combined]]
            energy = [float(value.square().sum()) for value in [selected_b, selected_m, selected_combo]]
            rms = [(value / count) ** .5 if count else 0. for value in energy]
            dot = float((selected_b * selected_m).sum())
            norm = (energy[0] * energy[1]) ** .5
            self.records.append(dict(step=step, timestep=int(self.timestep.flatten()[0]), branch=branch,
                region=name, enabled=bool(active), alpha_b=self.alpha_b, alpha_m=self.alpha_m,
                elements=count, b1_energy=energy[0], m10_energy=energy[1], fusion_energy=energy[2],
                b1_rms=rms[0], m10_rms=rms[1], fusion_rms=rms[2], inner_product=dot,
                cosine=dot / norm if norm else None, target_shape=list(shape),
                reference_shape=list(self.features.shape) if self.features is not None and active else None))
        self.last_delta = dict(rms=self.records[-4]['fusion_rms'])

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, scale=1.,
                 cond_hidden_states=None, sa_hidden_states=None, balanced_gate_timestep=None,
                 tcpm_garment_mask=None, local_detail_tokens=None, local_detail_mask=None, local_detail_spatial_shape=None):
        base = self.original(attn, hidden_states, encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask, temb=temb, scale=scale, cond_hidden_states=cond_hidden_states,
            sa_hidden_states=sa_hidden_states, balanced_gate_timestep=balanced_gate_timestep,
            tcpm_garment_mask=tcpm_garment_mask, local_detail_tokens=local_detail_tokens,
            local_detail_mask=local_detail_mask, local_detail_spatial_shape=local_detail_spatial_shape)
        active = self.enabled and sa_hidden_states is not None and self.features is not None
        if not active:
            zero = torch.zeros_like(base)
            self.observe(zero, zero, zero, False)
            return base
        target = hidden_states
        if attn.spatial_norm is not None:
            target = attn.spatial_norm(target, temb)
        shape = target.shape
        if target.ndim == 4:
            target = target.flatten(2).transpose(1, 2)
        if attn.group_norm is not None:
            target = attn.group_norm(target.transpose(1, 2)).transpose(1, 2)
        batch, tokens, channels = target.shape
        query = attn.to_q(target)

        def heads(value):
            return value.reshape(batch, -1, attn.heads, channels // attn.heads).transpose(1, 2)

        key_b, value_b = self.b1.reference(self.features, query)
        key_self, value_self = attn.to_k(target), attn.to_v(target)
        original = F.scaled_dot_product_attention(heads(query), heads(key_self), heads(value_self))
        joint = F.scaled_dot_product_attention(heads(query), heads(torch.cat([key_self, key_b], dim=1)),
            heads(torch.cat([value_self, value_b], dim=1)))
        delta_b = (self.b1.gain * (joint - original)).transpose(1, 2).reshape(batch, tokens, channels)
        delta_b = self.b1.project(delta_b, query)
        if len(shape) == 4:
            delta_b = delta_b.transpose(1, 2).reshape(shape)
        delta_b = delta_b / attn.rescale_output_factor

        _, value_m = self.m10.reference(self.features, query)
        target_float, reference_float = target.float(), value_m.float()
        statistic = (target_float - target_float.mean(1, keepdim=True)) / target_float.std(1, keepdim=True, unbiased=False).clamp_min(1e-6)
        statistic = statistic * reference_float.std(1, keepdim=True, unbiased=False) + reference_float.mean(1, keepdim=True)
        delta_m = self.m10.project(self.m10.gain * (statistic - target_float), query)
        delta_m = (delta_m * 1.).to(base.dtype) / attn.rescale_output_factor
        if len(shape) == 4:
            delta_m = delta_m.transpose(1, 2).reshape(shape)
        if self.alpha_b == 0 and self.alpha_m == 0:
            combined = torch.zeros_like(base)
            result = base
        elif self.alpha_m == 0:
            combined = delta_b if self.alpha_b == 1 else self.alpha_b * delta_b
            result = base + combined
        elif self.alpha_b == 0:
            combined = delta_m if self.alpha_m == 1 else self.alpha_m * delta_m
            result = base + combined
        else:
            combined = self.alpha_b * delta_b + self.alpha_m * delta_m
            result = base + combined
        self.observe(delta_b, delta_m, combined, True)
        return result
