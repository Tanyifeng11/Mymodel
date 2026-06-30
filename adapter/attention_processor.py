from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from diffusers.utils import USE_PEFT_BACKEND
from diffusers.models.lora import LoRALinearLayer
from einops import rearrange, repeat

class AttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        cond_hidden_states=None,
        sa_hidden_states=None,
        balanced_gate_timestep=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class SAttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, name, hidden_size, cross_attention_dim=None):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        super().__init__()

        self.name = name
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

    def __call__(
            self,
            attn,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            temb: Optional[torch.FloatTensor] = None,
            scale: float = 1.0,
            cond_hidden_states=None,
            sa_hidden_states=None,
            balanced_gate_timestep=None,
    ) -> torch.FloatTensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            # for reference adapter
            if sa_hidden_states is not None:
                ref_hidden_states = sa_hidden_states[self.name]
                encoder_hidden_states = torch.cat([hidden_states, ref_hidden_states], dim=1)
            else:
                encoder_hidden_states = hidden_states

        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class CAttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, name, hidden_size, cross_attention_dim=None):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        super().__init__()

        self.name = name
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim



    def __call__(
            self,
            attn,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            temb: Optional[torch.FloatTensor] = None,
            scale: float = 1.0,
            cond_hidden_states=None,
            sa_hidden_states=None,
            balanced_gate_timestep=None,
    ) -> torch.FloatTensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)


        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

        
class IPAttnProcessor2_0(torch.nn.Module):
    r"""
    Attention processor for IP-Adapater for PyTorch 2.0.

    Supports Ti-MGD-style layer-grouped frequency separation:
      - layer_group="semantic": text-only cross-attention (no IP adapter).
        Used in low-resolution UNet layers (deep down_blocks, mid_block, shallow up_blocks).
      - layer_group="detail": texture-only IP adapter, text cross-attention uses
        zeroed-out encoder_hidden_states (prevent text from diluting texture).
        Used in high-resolution UNet layers (shallow down_blocks, deep up_blocks).
      - layer_group="all": legacy behavior — both text and texture participate.

    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
        num_tokens (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
            The context length of the image features.
        layer_group (`str`, defaults to "all"):
            "semantic" / "detail" / "all" — which frequency group this layer belongs to.
        detail_text_scale (`float`, defaults to 0.1):
            When layer_group="detail", text cross-attention is kept but scaled down
            by this factor (instead of being fully zeroed). 0.0 = fully zeroed.
    """

    def __init__(
        self,
        hidden_size,
        cross_attention_dim=None,
        scale=1.0,
        num_tokens=4,
        layer_group: str = "all",
        detail_text_scale: float = 0.1,
        use_texture_gate: bool = False,
        gate_type: str = "layer",
        gate_init: str = "identity",
        gate_reg_weight: float = 0.0,
        gate_min: float = 0.7,
        gate_max: float = 1.3,
        use_palette_tokens: bool = False,
        num_palette_tokens: int = 4,
        palette_branch_scale_init: float = 0.0,
        use_balanced_fusion_gate: bool = False,
        balanced_gate_hidden_dim: int = 64,
        balanced_gate_scale: float = 0.2,
        balanced_gate_min: float = 0.8,
        balanced_gate_max: float = 1.2,
    ):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens
        self.use_ip_adapter = True
        self.layer_group = layer_group
        self.detail_text_scale = detail_text_scale
        self.use_texture_gate = bool(use_texture_gate)
        self.gate_type = gate_type
        self.gate_init = gate_init
        self.gate_reg_weight = gate_reg_weight
        self.gate_min = float(gate_min)
        self.gate_max = float(gate_max)
        self.use_palette_tokens = bool(use_palette_tokens)
        self.num_palette_tokens = int(num_palette_tokens) if self.use_palette_tokens else 0
        self.use_balanced_fusion_gate = bool(use_balanced_fusion_gate)
        self.balanced_gate_scale = float(balanced_gate_scale)
        self.balanced_gate_min = float(balanced_gate_min)
        self.balanced_gate_max = float(balanced_gate_max)

        if self.gate_type != "layer":
            raise NotImplementedError("E2b-lite 目前只支持 gate_type='layer'.")

        if self.use_texture_gate:
            if self.gate_init != "identity":
                raise NotImplementedError("E2b-lite 目前只支持 gate_init='identity'.")
            self.texture_gate_delta = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("texture_gate_delta", None)

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_k_palette = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_palette = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.palette_branch_scale = nn.Parameter(torch.tensor(float(palette_branch_scale_init)))

        gate_dim = cross_attention_dim or hidden_size
        group_map = {"all": 0, "semantic": 1, "detail": 2}
        self.register_buffer(
            "balanced_gate_group_id",
            torch.tensor(group_map.get(layer_group, 0), dtype=torch.long),
            persistent=False,
        )
        if self.use_balanced_fusion_gate:
            gate_input_dim = gate_dim * 4 + 4
            gate_hidden_dim = int(balanced_gate_hidden_dim)
            self.balanced_gate_norm = nn.LayerNorm(gate_input_dim)
            self.balanced_gate_mlp = nn.Sequential(
                nn.Linear(gate_input_dim, gate_hidden_dim),
                nn.SiLU(),
                nn.Linear(gate_hidden_dim, 2),
            )
            nn.init.zeros_(self.balanced_gate_mlp[-1].weight)
            nn.init.zeros_(self.balanced_gate_mlp[-1].bias)
        else:
            self.balanced_gate_norm = None
            self.balanced_gate_mlp = None

    def _balanced_branch_gates(
        self,
        text_states,
        ip_hidden_states,
        palette_hidden_states,
        timestep,
        hidden_dtype,
        hidden_device,
    ):
        if (
            not self.use_balanced_fusion_gate
            or self.balanced_gate_mlp is None
            or ip_hidden_states is None
            or palette_hidden_states is None
        ):
            return None, None

        batch_size = text_states.shape[0]
        text_pool = text_states.float().mean(dim=1)
        texture_pool = ip_hidden_states.float().mean(dim=1)
        palette_pool = palette_hidden_states.float().mean(dim=1)
        consistency = (text_pool - palette_pool).abs()

        if timestep is None:
            t = torch.zeros(batch_size, 1, device=text_pool.device, dtype=text_pool.dtype)
        else:
            t = timestep
            if not torch.is_tensor(t):
                t = torch.tensor(t, device=text_pool.device)
            t = t.to(device=text_pool.device, dtype=text_pool.dtype)
            if t.ndim == 0:
                t = t.view(1).expand(batch_size)
            if t.ndim > 1:
                t = t.view(batch_size, -1)[:, 0]
            t = t.view(batch_size, 1)

        group = F.one_hot(
            self.balanced_gate_group_id.to(device=text_pool.device).expand(batch_size),
            num_classes=3,
        ).to(dtype=text_pool.dtype)
        gate_input = torch.cat([text_pool, texture_pool, palette_pool, consistency, t, group], dim=1)
        gate_input = gate_input.to(dtype=self.balanced_gate_norm.weight.dtype)
        delta = torch.tanh(self.balanced_gate_mlp(self.balanced_gate_norm(gate_input)))
        gates = 1.0 + self.balanced_gate_scale * delta
        gates = torch.clamp(gates, min=self.balanced_gate_min, max=self.balanced_gate_max)
        gates = gates.to(device=hidden_device, dtype=hidden_dtype)

        self.last_balanced_texture_gate = gates[:, 0].detach().float().mean()
        self.last_balanced_palette_gate = gates[:, 1].detach().float().mean()
        return gates[:, 0].view(batch_size, 1, 1), gates[:, 1].view(batch_size, 1, 1)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        cond_hidden_states=None,
        sa_hidden_states=None,
        balanced_gate_timestep=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if cond_hidden_states is not None:
            encoder_hidden_states = cond_hidden_states

        use_ip_adapter = self.use_ip_adapter and self.num_tokens > 0
        use_palette_adapter = self.use_palette_tokens and self.num_palette_tokens > 0
        ip_hidden_states = None
        palette_hidden_states = None
        gate_text_states = None
        gate_texture_states = None

        if encoder_hidden_states is None:
            raise ValueError("IPAttnProcessor2_0 expects encoder_hidden_states with appended texture tokens.")
        elif use_ip_adapter or use_palette_adapter:
            condition_tokens = self.num_tokens + self.num_palette_tokens
            if encoder_hidden_states.shape[1] < condition_tokens:
                raise ValueError(
                    f"encoder_hidden_states has {encoder_hidden_states.shape[1]} tokens, expected at least {condition_tokens}."
                )
            end_pos = encoder_hidden_states.shape[1] - condition_tokens
            text_token_count = end_pos
            texture_start = end_pos
            texture_end = texture_start + self.num_tokens
            palette_start = texture_end
            if kwargs.get("debug_texture_tokens", False):
                print(
                    f"[IPAttnProcessor2_0] encoder_hidden_states={tuple(encoder_hidden_states.shape)}, "
                    f"text_tokens={text_token_count}, texture_tokens={self.num_tokens}, "
                    f"palette_tokens={self.num_palette_tokens}, layer_group={self.layer_group}"
                )
            text_states = encoder_hidden_states[:, :end_pos, :].contiguous()
            if use_ip_adapter:
                ip_hidden_states = encoder_hidden_states[:, texture_start:texture_end, :].contiguous()
                gate_texture_states = ip_hidden_states
            if use_palette_adapter:
                palette_hidden_states = encoder_hidden_states[:, palette_start:, :].contiguous()
            encoder_hidden_states = text_states
            gate_text_states = text_states

            # === Phase 1: Ti-MGD layer-grouped frequency routing ===
            if self.layer_group == "semantic":
                # Low-resolution layers: text-only, disable IP adapter
                use_ip_adapter = False
            elif self.layer_group == "detail":
                # High-resolution layers: texture-dominant
                # Scale down text cross-attention to prevent text from diluting texture
                # but keep a small amount for structural guidance
                if self.detail_text_scale < 1e-6:
                    encoder_hidden_states = torch.zeros_like(encoder_hidden_states)
                else:
                    encoder_hidden_states = (encoder_hidden_states * self.detail_text_scale).contiguous()
            # "all" — legacy behavior, no change
            # === end layer-grouped routing ===

            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if use_ip_adapter:
            ip_key = self.to_k_ip(ip_hidden_states)
            ip_value = self.to_v_ip(ip_hidden_states)

            ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            ip_hidden_states = F.scaled_dot_product_attention(
                query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            with torch.no_grad():
                self.attn_map = query @ ip_key.transpose(-2, -1).softmax(dim=-1)
                #print(self.attn_map.shape)

            ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            ip_hidden_states = ip_hidden_states.to(query.dtype)

            gate = 1.0
            balanced_texture_gate, balanced_palette_gate = self._balanced_branch_gates(
                gate_text_states,
                gate_texture_states,
                palette_hidden_states,
                balanced_gate_timestep,
                hidden_states.dtype,
                hidden_states.device,
            )
            if self.use_texture_gate and self.texture_gate_delta is not None:
                gate_raw = torch.exp(self.texture_gate_delta).to(dtype=hidden_states.dtype, device=hidden_states.device)
                gate = torch.clamp(gate_raw, min=self.gate_min, max=self.gate_max)
            if balanced_texture_gate is not None:
                gate = gate * balanced_texture_gate
            hidden_states = hidden_states + self.scale * gate * ip_hidden_states

        if use_palette_adapter and palette_hidden_states is not None:
            if "balanced_palette_gate" not in locals():
                _, balanced_palette_gate = self._balanced_branch_gates(
                    gate_text_states,
                    gate_texture_states,
                    palette_hidden_states,
                    balanced_gate_timestep,
                    hidden_states.dtype,
                    hidden_states.device,
                )
            palette_key = self.to_k_palette(palette_hidden_states)
            palette_value = self.to_v_palette(palette_hidden_states)
            palette_key = palette_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            palette_value = palette_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            palette_out = F.scaled_dot_product_attention(
                query, palette_key, palette_value, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            palette_out = palette_out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            palette_out = palette_out.to(query.dtype)
            palette_scale = self.palette_branch_scale.to(
                dtype=hidden_states.dtype, device=hidden_states.device
            )
            if balanced_palette_gate is not None:
                palette_scale = palette_scale * balanced_palette_gate
            hidden_states = hidden_states + palette_scale * palette_out

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states



    
class LogoRefSAttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, name, hidden_size, cross_attention_dim=None, scale=1.0 ):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        super().__init__()

        self.name = name
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.to_k_ref = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ref = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)  
        self.scale = scale

    def __call__(
            self,
            attn,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            temb: Optional[torch.FloatTensor] = None,
            scale: float = 1.0,
            num_images_per_prompt=1,
            cond_hidden_states=None,
            sa_hidden_states=None,
            balanced_gate_timestep=None,

    ) -> torch.FloatTensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        # for ref adapter
        if sa_hidden_states is not None:
            ref_hidden_states = sa_hidden_states[self.name]
            
            # for ref sketch
            ref_key = self.to_k_ref(ref_hidden_states)
            ref_value = self.to_v_ref(ref_hidden_states)
            ref_key = ref_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ref_value = ref_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            ref_hidden_states = F.scaled_dot_product_attention(
                query, ref_key, ref_value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            ref_hidden_states = ref_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            ref_hidden_states = ref_hidden_states.to(query.dtype)
            
            
           
            hidden_states = hidden_states + ref_hidden_states * self.scale 

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class LogoCacheSAttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, name, hidden_size, cross_attention_dim=None,scale=1.0):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        super().__init__()

        self.name = name
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.cache = {}  # cache hidden states
        
    def __call__(
            self,
            attn,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            temb: Optional[torch.FloatTensor] = None,
            scale: float = 1.0,
    ) -> torch.FloatTensor:
        self.cache["hidden_states"] = hidden_states  # cache hidden states
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        

        
        #sketch
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)


        
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        
        return hidden_states

class LogoCacheCAttnProcessor2_0(torch.nn.Module):
    
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, name, hidden_size, cross_attention_dim=None):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        super().__init__()

        self.name = name
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim


    def __call__(
            self,
            attn,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            temb: Optional[torch.FloatTensor] = None,
            scale: float = 1.0,
            cond_hidden_states=None,
            sa_hidden_states=None,
            balanced_gate_timestep=None,
    ) -> torch.FloatTensor:
        

        return hidden_states

class SkipAttnProcessor(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        return hidden_states
