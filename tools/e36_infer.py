"""复用 E35 的逐项原始 E5 生成协议；新增模块仅改指定 Skip。"""
import copy
import hashlib
import time
from pathlib import Path
from types import MethodType
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from tools.e36_protocol import *
from tools.e36_runtime import SkipBridge
from tools.e35_freeu_run import digest


class Generator:
    def __init__(self):
        from models.e33tm_generation_wrapper import load_e5
        from tools.e22_4_generation import module_hashes
        self.pipe,self.modules,size,self.ns=load_e5()
        assert size==SIZE and type(self.pipe.scheduler).__name__=='DDIMScheduler'
        assert all(not p.requires_grad for m in self.modules.values() for p in m.parameters())
        historical=read(PREVIOUS/'jobs/s0_shard0/implementation.json')
        assert sha(E5)==historical['e5_checkpoint_sha256']
        assert vars(self.ns)==historical['inference_args'], '原始E5推理配置发生变化'
        actual=dict(self.pipe.scheduler.config);old=historical['scheduler_config']
        assert {k:v for k,v in actual.items() if not k.startswith('_')}=={k:v for k,v in old.items() if not k.startswith('_')}
        self.before=module_hashes(self.modules)
        self.bridge=SkipBridge(self.pipe.unet)
        self.structure=[dict(stage=i,type=type(b).__name__,resnet_channels=[dict(input=r.in_channels,output=r.out_channels) for r in b.resnets]) for i,b in enumerate(self.pipe.unet.up_blocks)]
        # 使用VAE实际下采样比例，不凭固定8猜latent尺寸。
        factor=self.pipe.vae_scale_factor
        shape=(1,self.pipe.unet.config.in_channels,SIZE[1]//factor,SIZE[0]//factor)
        self.noise=torch.randn(shape,device=self.pipe.device,dtype=self.pipe.unet.dtype,
            generator=torch.Generator(device=self.pipe.device).manual_seed(42))
        self.initial=self.noise*self.pipe.scheduler.init_noise_sigma
        assert shape==(1,4,64,48), 'E34基座分辨率发生变化'
        write(OUT/'audit/model.json',dict(git_commit=commit(),checkpoint_sha256=sha(E5),
            unet_type=type(self.pipe.unet).__module__+'.'+type(self.pipe.unet).__name__,
            structure=self.structure,adapter_policy=CONFIG['cfg_adapter_policy'],
            scheduler=dict(self.pipe.scheduler.config),inference_args=vars(self.ns),
            dtype=str(self.pipe.unet.dtype),vae_dtype=str(self.pipe.vae.dtype),
            vae_scale_factor=factor,latent_shape=list(shape),noise_sha256=digest(self.noise),
            initial_sha256=digest(self.initial),frozen_modules=self.before,
            device=torch.cuda.get_device_name(),attention_processors={k:type(v).__name__ for k,v in self.pipe.unet.attn_processors.items()}))
        folder=OUT/'p0';folder.mkdir(parents=True,exist_ok=True)
        (folder/'unet_structure.txt').write_text(str(self.pipe.unet),encoding='utf-8')
        (folder/'pre_concat_source.py.txt').write_text(self.bridge.source,encoding='utf-8')
        torch.save(self.initial.detach().cpu(),OUT/'audit/initial_latent.pt')

    @torch.inference_mode()
    def generate(self,row,arm,path,capture_first=False):
        from garment_mask_utils import build_sketch_garment_mask
        pipe=self.pipe;self.bridge.adapter=self.adapters.get(arm) if hasattr(self,'adapters') else None
        sketch=Image.open(row['sketch']).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR)
        self.bridge.sketch=to_tensor(sketch)[None].to(pipe.device,torch.float32)*2-1
        reference=Image.open(row['reference']).convert('RGB')
        mask,_=build_sketch_garment_mask(sketch,*SIZE)
        old_prepare=pipe.prepare_latents;old_step=pipe.scheduler.step
        timesteps=[];last=[];captured={};outputs={};handles=[]
        pipe.prepare_latents=MethodType(lambda _self,*a,**kw:self.initial.clone(),pipe)
        def step(*args,**kwargs):
            result=old_step(*args,**kwargs);timesteps.append(int(args[1]));last[:]=[result[0]];return result
        pipe.scheduler.step=step
        if capture_first:
            def save(_m,args,kwargs):
                branch='conditional' if 'sa_hidden_states' in (kwargs.get('cross_attention_kwargs') or {}) else 'unconditional'
                if branch not in captured:captured[branch]=(args,copy.copy(kwargs))
            def save_output(_m,args,kwargs,result):
                branch='conditional' if 'sa_hidden_states' in (kwargs.get('cross_attention_kwargs') or {}) else 'unconditional'
                if branch not in outputs:outputs[branch]=result[0].clone()
            handles=[pipe.unet.register_forward_pre_hook(save,with_kwargs=True),
                     pipe.unet.register_forward_hook(save_output,with_kwargs=True)]
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        try:
            image=pipe(prompt=row['caption'],null_prompt='',negative_prompt=' worst quality, low quality',
                ref_image=to_tensor(sketch)[None]*2-1,texture_clip_image=reference,
                width=SIZE[0],height=SIZE[1],num_inference_steps=50,guidance_scale=7.,
                sketch_scale=.6,ipa_scale=1.,texture_mode=self.ns.texture_mode,texture_condition_mode='token',
                texture_preprocess_mode='plain_resize',texture_num_tokens=pipe.effective_texture_num_tokens,
                texture_scale=1.,spatial_mask=to_tensor(mask)[None].to(pipe.device,torch.float16),
                generator=torch.Generator(device=pipe.device).manual_seed(42))[0]
        finally:
            pipe.prepare_latents=old_prepare;pipe.scheduler.step=old_step
            for handle in handles:handle.remove()
        torch.cuda.synchronize();seconds=time.perf_counter()-started
        assert len(timesteps)==50 and all(torch.isfinite(t).all() for t in last)
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);image.save(path)
        record=dict(id=row['id'],arm=arm,seed=42,seconds=seconds,
            peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,steps=timesteps,
            noise_sha256=digest(self.noise),initial_latent_sha256=digest(self.initial),
            final_latent_sha256=digest(last[0]),png_sha256=sha(path),
            inputs={k:sha(row[k]) for k in ['sketch','reference','gt']},
            caption_sha256=hashlib.sha256(row['caption'].encode()).hexdigest(),gt_used_in_inference=False)
        write(path.with_suffix('.json'),record)
        return record,captured,outputs

    def close(self):
        from tools.e22_4_generation import module_hashes
        self.bridge.close();after=module_hashes(self.modules)
        assert after==self.before
        write(OUT/'audit/frozen_modules.json',dict(before=self.before,after=after,pass_unchanged=True))
        verify_sources()

