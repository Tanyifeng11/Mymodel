import gc
import hashlib
import time
import numpy as np
import torch
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, DDIMScheduler
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from cn_ipa.common import *

NEGATIVE=' worst quality, low quality'  # 与历史 M12 实际调用逐字相同。


def digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def load(cn_path=CN, ip_path=None, device='cuda'):
    assert __import__('diffusers').__version__=='0.31.0'
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    control=ControlNetModel.from_pretrained(str(cn_path),torch_dtype=torch.float16,local_files_only=True)
    encoder=CLIPVisionModelWithProjection.from_pretrained(str(ENCODER),torch_dtype=torch.float16,local_files_only=True)
    assert encoder.config.projection_dim==1024 and encoder.config.hidden_size==1280
    pipe=StableDiffusionControlNetPipeline.from_pretrained(str(SD),controlnet=control,
        image_encoder=encoder,feature_extractor=CLIPImageProcessor(),torch_dtype=torch.float16,
        safety_checker=None,requires_safety_checker=False,local_files_only=True)
    path=Path(ip_path) if ip_path else IPA/'ip-adapter_sd15.bin'
    pipe.load_ip_adapter(str(path.parent),weight_name=path.name,image_encoder_folder=None,local_files_only=True)
    pipe.scheduler=DDIMScheduler(num_train_timesteps=1000,beta_start=.00085,beta_end=.012,
        beta_schedule='scaled_linear',clip_sample=False,set_alpha_to_one=False,steps_offset=1,
        prediction_type='epsilon',timestep_spacing='leading')
    pipe.to(device);pipe.set_progress_bar_config(disable=True)
    for module in [pipe.unet,pipe.vae,pipe.controlnet,pipe.text_encoder,pipe.image_encoder]:
        module.eval().requires_grad_(False)
    return pipe


def trainable_parameters(unet):
    return {name:p for name,p in unet.named_parameters()
        if name.startswith('encoder_hid_proj.') or '.to_k_ip.' in name or '.to_v_ip.' in name}


def frozen_hashes(pipe):
    return dict(unet=state_hash(pipe.unet,('encoder_hid_proj.','.to_k_ip.','.to_v_ip.')),
        vae=state_hash(pipe.vae),controlnet=state_hash(pipe.controlnet),
        text_encoder=state_hash(pipe.text_encoder),image_encoder=state_hash(pipe.image_encoder))


def adapter_state(unet):
    projection=unet.encoder_hid_proj.image_projection_layers[0].state_dict()
    assert set(projection)=={'image_embeds.weight','image_embeds.bias','norm.weight','norm.bias'}
    image_proj={k.replace('image_embeds.','proj.'):v.detach().cpu().clone() for k,v in projection.items()}
    adapters={}
    for index,processor in enumerate(unet.attn_processors.values()):
        if hasattr(processor,'to_k_ip'):
            assert len(processor.to_k_ip)==len(processor.to_v_ip)==1
            adapters[f'{index}.to_k_ip.weight']=processor.to_k_ip[0].weight.detach().cpu().clone()
            adapters[f'{index}.to_v_ip.weight']=processor.to_v_ip[0].weight.detach().cpu().clone()
    return dict(image_proj=image_proj,ip_adapter=adapters)


def save_adapter(unet,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    state=adapter_state(unet);torch.save(state,path)
    write(path.with_suffix('.shapes.json'),{group:{k:list(v.shape) for k,v in values.items()} for group,values in state.items()})
    return sha(path)


def forward(pipe,noisy,t,text,reference,sketch,cn_scale=1.):
    # CN只看文本与线稿；参考信息只进入IPA，不向冻结CN偷偷传播。
    with torch.no_grad():
        down,mid=pipe.controlnet(noisy,t,encoder_hidden_states=text,controlnet_cond=sketch,
            conditioning_scale=cn_scale,return_dict=False)
    return pipe.unet(noisy,t,encoder_hidden_states=text,
        added_cond_kwargs={'image_embeds':[reference]},
        down_block_additional_residuals=down,mid_block_additional_residual=mid).sample


def inputs(row):
    return (Image.open(row['sketch']).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR),
        Image.open(row['reference']).convert('RGB'))


@torch.inference_mode()
def generate(pipe,row,path,seed=42,cn_scale=1.,ip_scale=.65,reference=None,sketch=None,prompt=None,capture=False):
    path=Path(path)
    if path.exists():
        meta=read(path.with_suffix('.json'))
        assert meta['png_sha256']==sha(path)
        assert meta['seed']==seed and meta['cn_scale']==cn_scale and meta['ip_scale']==ip_scale
        return meta
    original_sketch,original_ref=inputs(row)
    sketch=original_sketch if sketch is None else sketch
    reference=original_ref if reference is None else reference
    pipe.set_ip_adapter_scale(ip_scale)
    latents=torch.randn((1,4,64,48),device='cuda',dtype=pipe.unet.dtype,
        generator=torch.Generator(device='cuda').manual_seed(seed))
    initial=digest(latents*pipe.scheduler.init_noise_sigma)
    calls=[]
    def hook(_module,args,kwargs,out):
        if capture and not calls:
            calls.append(dict(noisy=args[0].detach().clone(),t=args[1].detach().clone(),
                text=kwargs['encoder_hidden_states'].detach().clone(),
                reference=kwargs['added_cond_kwargs']['image_embeds'][0].detach().clone(),
                prediction=out[0].detach().clone()))
    handle=pipe.unet.register_forward_hook(hook,with_kwargs=True) if capture else None
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    try:
        rgb=pipe(prompt=prompt if prompt is not None else row['caption'],negative_prompt=NEGATIVE,
            image=sketch,ip_adapter_image=reference,width=384,height=512,num_inference_steps=50,
            guidance_scale=7.,controlnet_conditioning_scale=cn_scale,latents=latents,
            generator=torch.Generator(device='cuda').manual_seed(seed)).images[0]
    finally:
        if handle:handle.remove()
    torch.cuda.synchronize();seconds=time.perf_counter()-start
    assert rgb.size==SIZE
    path.parent.mkdir(parents=True,exist_ok=True);rgb.save(path)
    meta=dict(id=row['id'],seed=seed,cn_scale=cn_scale,ip_scale=ip_scale,png_sha256=sha(path),
        initial_sha256=initial,gt_used_in_inference=False,prompt=prompt if prompt is not None else row['caption'],
        negative_prompt=NEGATIVE,seconds=seconds,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,gpu=torch.cuda.get_device_name(),
        scheduler=dict(pipe.scheduler.config),code_commit=commit(),
        sketch_sha256=sha(row['sketch']),reference_sha256=sha(row['reference']))
    write(path.with_suffix('.json'),meta)
    if capture:return meta,calls[0]
    return meta


def release(pipe):
    pipe.to('cpu');gc.collect();torch.cuda.empty_cache()
