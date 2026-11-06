"""固定官方计算路线：Sketch reference attention + 4-token pooled CLIP adapter。"""
import torch
from torch import nn
from diffusers import UNet2DConditionModel, DDIMScheduler, DDPMScheduler
from .common import ROOT
from .vendor.adapter__attention_processor import (
    AttnProcessor2_0, CAttnProcessor2_0, IPAttnProcessor2_0,
    LogoRefSAttnProcessor2_0, LogoCacheSAttnProcessor2_0, LogoCacheCAttnProcessor2_0,
)


class FabricAttnProcessor(IPAttnProcessor2_0):
    # diffusers 0.31 按显式签名筛选 kwargs；参考特征仅由 self-attention 读取。
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
                 temb=None, sa_hidden_states=None, **kwargs):
        return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, temb)


class ImageProjModel(nn.Module):
    # 与官方 adapter/color_adapter.py 的 ImageProjModel 同构。
    def __init__(self, clip_dim=1024, cross_dim=768):
        super().__init__()
        self.proj = nn.Linear(clip_dim, 4 * cross_dim)
        self.norm = nn.LayerNorm(cross_dim)

    def forward(self, embedding):
        return self.norm(self.proj(embedding).reshape(-1, 4, self.norm.normalized_shape[0]))


def hidden_size(unet, name):
    if name.startswith('mid_block'):
        return unet.config.block_out_channels[-1]
    block = int(name.split('.')[1])
    channels = list(unet.config.block_out_channels)
    return (list(reversed(channels)) if name.startswith('up_blocks') else channels)[block]


class GarmentModel(nn.Module):
    def __init__(self, stage, device, training=True):
        super().__init__()
        assert stage in {'color', 'gam', 'combined'}
        self.stage = stage
        base = ROOT / 'initialization/sd15'
        self.unet = UNet2DConditionModel.from_pretrained(base, subfolder='unet', local_files_only=True)
        ref_initial = {}
        if stage != 'color':
            for name in self.unet.attn_processors:
                if name.endswith('attn1.processor'):
                    layer = self.unet.get_submodule(name[:-len('.processor')])
                    ref_initial[name] = {'to_k_ref.weight': layer.to_k.weight.detach().float().clone(),
                                         'to_v_ref.weight': layer.to_v.weight.detach().float().clone()}
        self.unet.requires_grad_(False).to(device=device, dtype=torch.bfloat16)
        processors = {}
        for name in self.unet.attn_processors:
            size = hidden_size(self.unet, name)
            if name.endswith('attn1.processor'):
                proc = LogoRefSAttnProcessor2_0(name, size) if stage != 'color' else AttnProcessor2_0()
                if stage != 'color':
                    # 草图 K/V 从公共 float32 原值初始化，不从已量化的冻结骨干复制。
                    proc.load_state_dict(ref_initial.pop(name), strict=True)
            elif stage == 'gam':
                proc = CAttnProcessor2_0(name, size, cross_attention_dim=self.unet.config.cross_attention_dim)
            else:
                proc = FabricAttnProcessor(size, self.unet.config.cross_attention_dim, num_tokens=4)
            processors[name] = proc.to(device=device)
        self.unet.set_attn_processor(processors)
        if stage != 'color':
            self.reference_unet = UNet2DConditionModel.from_pretrained(base, subfolder='unet', local_files_only=True)
            self.reference_unet.to(device=device, dtype=torch.float32)
            reference = {}
            for name in self.reference_unet.attn_processors:
                size = hidden_size(self.reference_unet, name)
                reference[name] = (LogoCacheSAttnProcessor2_0(name, size) if name.endswith('attn1.processor')
                                   else LogoCacheCAttnProcessor2_0(name, size, cross_attention_dim=size)).to(device)
            self.reference_unet.set_attn_processor(reference)
        if stage != 'gam':
            self.image_proj = ImageProjModel().to(device)
            public = torch.load(ROOT / 'initialization/ip-adapter_sd15.bin', map_location='cpu', weights_only=True)
            self.image_proj.load_state_dict(public['image_proj'], strict=True)
            expected = set()
            for index, proc in enumerate(self.unet.attn_processors.values()):
                if isinstance(proc, IPAttnProcessor2_0):
                    mapping = {key: public['ip_adapter'][f'{index}.{key}'] for key in ['to_k_ip.weight', 'to_v_ip.weight']}
                    proc.load_state_dict(mapping, strict=True)
                    expected.update(f'{index}.{key}' for key in mapping)
            assert expected == set(public['ip_adapter']), '公共IP权重有未解释的参数'
        if training:
            self.unet.enable_gradient_checkpointing()
            if stage != 'color':
                self.reference_unet.enable_gradient_checkpointing()
            self.train()
        else:
            self.eval().requires_grad_(False)
        self.assert_architecture()

    def assert_architecture(self):
        forbidden = ['bf_texture', 'bf_fusion', 'bFMFusion', 'tcpm', 'body_loss', 'rest_loss', 'film', 'nexus']
        assert not any(any(word.lower() in name.lower() for word in forbidden) for name, _ in self.named_parameters())
        if self.stage != 'gam':
            assert self.image_proj.proj.out_features == 4 * 768
        return True

    def stage_parameters(self, stage):
        if stage == 'color':
            return {name: p for name, p in self.named_parameters()
                    if name.startswith('image_proj.') or '.to_k_ip.' in name or '.to_v_ip.' in name}
        return {name: p for name, p in self.named_parameters()
                if name.startswith('reference_unet.') or '.to_k_ref.' in name or '.to_v_ref.' in name}

    def trainable(self):
        return {name: p for name, p in self.named_parameters() if p.requires_grad}

    def save_parameters(self, stage):
        return {name: p.detach().cpu().clone() for name, p in self.stage_parameters(stage).items()}

    def load_parameters(self, stage, values):
        expected = self.stage_parameters(stage)
        assert set(values) == set(expected), dict(missing=sorted(set(expected) - set(values)),
                                                  extra=sorted(set(values) - set(expected)))
        with torch.no_grad():
            for name, p in expected.items():
                assert p.shape == values[name].shape, name
                p.copy_(values[name])

    def clear_cache(self):
        if hasattr(self, 'reference_unet'):
            for proc in self.reference_unet.attn_processors.values():
                if hasattr(proc, 'cache'):
                    proc.cache.clear()

    def sketch_states(self, sketch_latent):
        self.clear_cache()
        t = torch.zeros(len(sketch_latent), device=sketch_latent.device, dtype=torch.long)
        self.reference_unet(sketch_latent, t, None, return_dict=False)
        return {name: proc.cache['hidden_states'] for name, proc in self.reference_unet.attn_processors.items()
                if name.endswith('attn1.processor')}

    def forward(self, noisy, timestep, text, sketch_latent=None, fabric_embedding=None):
        kwargs = {}
        if self.stage == 'gam':
            kwargs['sa_hidden_states'] = self.sketch_states(sketch_latent)
        if self.stage == 'color':
            text = torch.cat([text, self.image_proj(fabric_embedding)], dim=1)
        return self.unet(noisy, timestep, encoder_hidden_states=text, cross_attention_kwargs=kwargs).sample


def scheduler(stage):
    if stage == 'color':
        return DDPMScheduler.from_pretrained(ROOT / 'initialization/sd15', subfolder='scheduler', local_files_only=True)
    if stage == 'gam':
        # 官方 GAM 训练使用 zero-terminal-SNR；官方采样默认不启用，分别记录。
        return DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear',
                             num_train_timesteps=1000, rescale_betas_zero_snr=True,
                             timestep_spacing='trailing', prediction_type='epsilon', clip_sample=False)
    return DDIMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
                         beta_schedule='scaled_linear', clip_sample=False, set_alpha_to_one=False, steps_offset=1)
