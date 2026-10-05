"""只观察scheduler的实际返回，不额外调用step或改变噪声。"""
from types import MethodType
import hashlib
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from tools.e25_spatial_diagnosis import timestep_for

def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def tensor_summary(value):
    x=value.detach().float()
    return dict(shape=list(value.shape),dtype=str(value.dtype),norm=float(x.norm()),
                mean=float(x.mean()),std=float(x.std()),min=float(x.min()),max=float(x.max()),sha256=tensor_sha(value))

@torch.inference_mode()
def decode(pipe,latent):
    rgb=pipe.vae.decode(latent.to(pipe.vae.dtype)/pipe.vae.config.scaling_factor,return_dict=False)[0]
    return Image.fromarray(((rgb/2+.5).clamp(0,1)[0].permute(1,2,0).float().cpu().numpy()*255).round().astype('uint8'))

@torch.inference_mode()
def refine(pipe,size,ns,text,sketch,reference,mask,carrier,strength=.15,conditions=(True,True,True),
           diffusion_seed=42,observer=None,save_tensors=None):
    noise=torch.randn((1,pipe.unet.config.in_channels,size[1]//8,size[0]//8),device=pipe.device,
        dtype=pipe.unet.dtype,generator=torch.Generator(device=pipe.device).manual_seed(diffusion_seed))
    start=timestep_for(pipe.scheduler,strength)
    pixels=to_tensor(carrier)[None].to(pipe.device,pipe.vae.dtype)*2-1
    latent=pipe.vae.encode(pixels).latent_dist.mean*pipe.vae.config.scaling_factor
    timestep=torch.tensor([start['timestep']],device=pipe.device,dtype=torch.long)
    initial=pipe.scheduler.add_noise(latent,noise,timestep)
    if save_tensors is not None:
        save_tensors.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(save_tensors,posterior_mean=latent.cpu().numpy(),noise=noise.cpu().numpy(),start_latent=initial.cpu().numpy())
    if observer: observer('S4',decode(pipe,latent),None)
    original_latents=pipe.prepare_latents;original_times=pipe.scheduler.set_timesteps;original_step=pipe.scheduler.step
    records=[]
    def shortened(steps,*args,**kwargs):
        assert steps==50
        original_times(steps,*args,**kwargs)
        pipe.scheduler.timesteps=pipe.scheduler.timesteps[start['scheduler_index']:]
    def observed_step(model_output,t,sample,*args,**kwargs):
        result=original_step(model_output,t,sample,*args,**kwargs)
        predicted=result.pred_original_sample if hasattr(result,'pred_original_sample') else result[1]
        info=dict(step_index=len(records),timestep=int(t),latent=tensor_summary(sample),
                  predicted_noise=tensor_summary(model_output),predicted_x0=tensor_summary(predicted))
        records.append(info)
        if observer: observer('x0_%02d'%len(records),decode(pipe,predicted),info)
        if save_tensors is not None and observer:
            np.savez_compressed(save_tensors.parent/('step_%02d.npz'%len(records)),
                predicted_noise=model_output.detach().cpu().numpy(),predicted_x0=predicted.detach().cpu().numpy())
        return result
    pipe.prepare_latents=MethodType(lambda self,*args,**kwargs:initial.clone(),pipe)
    pipe.scheduler.set_timesteps=shortened
    pipe.scheduler.step=observed_step
    try:
        t_on,s_on,r_on=conditions
        image=pipe(prompt=text if t_on else '',null_prompt='',negative_prompt=' worst quality, low quality',
            ref_image=to_tensor(sketch)[None]*2-1,texture_clip_image=reference if r_on else None,
            width=size[0],height=size[1],num_inference_steps=50,guidance_scale=7.,
            sketch_scale=.6 if s_on else 0.,ipa_scale=1.,texture_mode=ns.texture_mode,
            texture_condition_mode='token',texture_preprocess_mode='plain_resize',
            texture_num_tokens=pipe.effective_texture_num_tokens,texture_scale=1.,
            spatial_mask=to_tensor(mask)[None].to(pipe.device,torch.float16),
            generator=torch.Generator(device=pipe.device).manual_seed(diffusion_seed))[0]
    finally:
        pipe.prepare_latents=original_latents;pipe.scheduler.set_timesteps=original_times;pipe.scheduler.step=original_step
    assert len(records)==start['remaining_steps']
    if observer: observer('S9',image,None)
    return image,dict(start=start,scheduler_class=type(pipe.scheduler).__name__,scheduler_config=dict(pipe.scheduler.config),
        actual_timesteps=[r['timestep'] for r in records],effective_steps=len(records),
        posterior_mean=tensor_summary(latent),initial_latent=tensor_summary(initial),noise_sha256=tensor_sha(noise),
        carrier_sha256=hashlib.sha256(np.asarray(carrier).tobytes()).hexdigest(),conditions=list(conditions),steps=records)
