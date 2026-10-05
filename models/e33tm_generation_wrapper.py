"""冻结 RF2 输入保持原协议；文本不会参与方向场计算。"""
from torch import nn

class TriModalField(nn.Module):
    def __init__(self, rf2):
        super().__init__()
        self.rf2 = rf2.eval().requires_grad_(False)

    @property
    def prior(self):
        return self.rf2.prior

    def forward(self, reference, structure, text=None):
        return self.rf2(reference, structure)


def spatial_carrier(reference, orientation, mask):
    """把RF2轴向场接到原E26 RGB remapper，随后沿用E25 latent载体。

    仅用reference RGB、预测场和sketch mask；不读取target RGB/GT/旋转标签。
    固定以衣身中心为旋转原点，不新增周期、相位或身份预测器。
    """
    import cv2
    import numpy as np
    from models.pattern_coordinate_field import build_affine_pattern_field, sample_reference_with_field
    from models.pattern_geometry import estimate_orientation_structure_tensor
    source_angle, _ = estimate_orientation_structure_tensor(reference)
    field = build_affine_pattern_field(mask, source_angle, 1., 0., source_size=reference.size)
    width, height = mask.size
    x, y = field.uv[...,0].copy(), field.uv[...,1].copy()
    # 先插值cos2theta/sin2theta，避免轴向角在0/180处不连续。
    dense = cv2.resize(np.moveaxis(orientation,0,-1), (width,height), interpolation=cv2.INTER_LINEAR)
    theta = np.arctan2(dense[...,1],dense[...,0])/2
    angle = (theta-np.deg2rad(source_angle)+np.pi/2)%np.pi-np.pi/2
    cx, cy = (reference.width-1)/2, (reference.height-1)/2
    # 逆变换把reference的原方向映射为RF预测方向。
    field.uv[...,0] = (x-cx)*np.cos(angle)+(y-cy)*np.sin(angle)+cx
    field.uv[...,1] = -(x-cx)*np.sin(angle)+(y-cy)*np.cos(angle)+cy
    return sample_reference_with_field(reference, field, mask), field


def load_e5():
    import argparse
    from pathlib import Path
    from tools.e15_d5_generation import build_inference_args, load_inference_module
    from tools.e33tm_protocol import E5
    root = Path.cwd()
    args = argparse.Namespace(checkpoint=str(root/E5), texture_ckpt=str(root/E5),
        base_model_path=str(root/'models/stable-diffusion-v1-5'),
        vae_model_path=str(root/'models/stable-diffusion-v1-5/vae'),
        clip_model=str(root/'models/clip'), device='cuda', seed=42, steps=50)
    ns = build_inference_args(args)
    pipe, _ = load_inference_module().prepare(ns)
    modules = {key: getattr(pipe,key) for key in ('unet','reference_unet','bf_texture_conditioner',
                                               'tcpm_lite','vae','text_encoder','image_encoder')}
    modules = {key: module for key,module in modules.items() if module is not None}
    for module in modules.values():
        module.eval().requires_grad_(False)
    pipe.set_progress_bar_config(disable=True)
    return pipe, modules, (ns.width,ns.height), ns


def generate(pipe, size, ns, text, sketch, reference, mask, diffusion_seed, scaffold=None, no_sketch=False):
    import torch
    from types import MethodType
    from torchvision.transforms.functional import to_tensor
    from tools.e25_spatial_diagnosis import timestep_for, sha
    generator = torch.Generator(device=pipe.device).manual_seed(diffusion_seed)
    shape = (1,pipe.unet.config.in_channels,size[1]//8,size[0]//8)
    noise = torch.randn(shape,device=pipe.device,dtype=pipe.unet.dtype,generator=generator)
    old_latents, old_timesteps = pipe.prepare_latents, pipe.scheduler.set_timesteps
    start = None
    if scaffold is not None:
        start = timestep_for(pipe.scheduler,.15)
        pixels = to_tensor(scaffold)[None].to(pipe.device,pipe.vae.dtype)*2-1
        latent = pipe.vae.encode(pixels).latent_dist.mean*pipe.vae.config.scaling_factor
        timestep = torch.tensor([start['timestep']],device=pipe.device,dtype=torch.long)
        initial = pipe.scheduler.add_noise(latent,noise,timestep)
        def shortened(steps,*args,**kwargs):
            old_timesteps(steps,*args,**kwargs)
            pipe.scheduler.timesteps = pipe.scheduler.timesteps[start['scheduler_index']:]
        pipe.scheduler.set_timesteps = shortened
    else:
        initial = noise*pipe.scheduler.init_noise_sigma
    pipe.prepare_latents = MethodType(lambda self,*args,**kwargs: initial.clone(),pipe)
    try:
        image = pipe(prompt=text, null_prompt='', negative_prompt=' worst quality, low quality',
            ref_image=to_tensor(sketch)[None]*2-1, texture_clip_image=reference,
            width=size[0],height=size[1],num_inference_steps=50,guidance_scale=7.,
            sketch_scale=0. if no_sketch else .6,ipa_scale=1.,
            texture_mode=ns.texture_mode,texture_condition_mode='token',
            texture_preprocess_mode='plain_resize',texture_num_tokens=pipe.effective_texture_num_tokens,
            texture_scale=1., spatial_mask=to_tensor(mask)[None].to(pipe.device,torch.float16),
            generator=torch.Generator(device=pipe.device).manual_seed(diffusion_seed))[0]
    finally:
        pipe.prepare_latents,pipe.scheduler.set_timesteps = old_latents,old_timesteps
    return image,dict(noise_sha256=sha(noise.cpu().numpy().tobytes()),
                      initial_latent_sha256=sha(initial.cpu().numpy().tobytes()),
                      refinement_start=start, text_used=text, texture_enabled=reference is not None,
                      sketch_enabled=not no_sketch)
