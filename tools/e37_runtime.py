"""无 E36 写入副作用的原 E5 运行器，以及固定真实训练噪声输入。"""
import random
import torch
from diffusers import DDIMScheduler
from tools.e36_infer import Generator as OriginalGenerator,digest
from tools.e36_runtime import SkipBridge
from tools.e37_protocol import *


class Generator(OriginalGenerator):
    # 仅复用 generate 的原始推理调用，禁止调用其写 E36 目录的初始化/收尾。
    def __init__(self):
        from models.e33tm_generation_wrapper import load_e5
        from tools.e22_4_generation import module_hashes
        from tools.e35_freeu_protocol import PREVIOUS
        self.pipe,self.modules,size,self.ns=load_e5()
        assert size==SIZE and type(self.pipe.scheduler).__name__=='DDIMScheduler'
        historical=read(PREVIOUS/'jobs/s0_shard0/implementation.json')
        assert vars(self.ns)==historical['inference_args']
        actual=dict(self.pipe.scheduler.config);old=historical['scheduler_config']
        assert {k:v for k,v in actual.items() if not k.startswith('_')}=={k:v for k,v in old.items() if not k.startswith('_')}
        assert all(not p.requires_grad for m in self.modules.values() for p in m.parameters())
        self.before=module_hashes(self.modules);self.bridge=SkipBridge(self.pipe.unet)
        self.noise=torch.randn((1,4,64,48),device=self.pipe.device,dtype=self.pipe.unet.dtype,
            generator=torch.Generator(device=self.pipe.device).manual_seed(42))
        self.initial=self.noise*self.pipe.scheduler.init_noise_sigma
        self.adapters={}
        write(OUT/'audit/model.json',dict(commit=commit(),checkpoint_sha256=sha(E5),
            scheduler=actual,inference_args=vars(self.ns),frozen_modules=self.before,
            vae_scale_factor=self.pipe.vae_scale_factor,vae_scaling_factor=self.pipe.vae.config.scaling_factor,
            latent_shape=list(self.initial.shape),initial_latent_sha256=digest(self.initial),
            adapter_policy=CONFIG['adapter_policy'],device=torch.cuda.get_device_name()))

    def load_adapter(self,step):
        from models.e36_guided_kernel_filter import GuidedKernelFilter
        assert sha(checkpoint(step))==WEIGHTS[step]
        torch.manual_seed(42)
        adapter=GuidedKernelFilter(640,1280,'B1_CONV').cuda()
        adapter.load_state_dict(torch.load(checkpoint(step),map_location='cpu',weights_only=False)['state_dict'])
        adapter.eval();assert sum(p.numel() for p in adapter.parameters())==143166
        self.adapters={'B1_CONV':adapter};self.bridge.adapter=adapter
        write(OUT/'audit/parameter_manifest.json',[dict(name=n,shape=list(p.shape),count=p.numel(),dtype=str(p.dtype))
            for n,p in adapter.named_parameters()])
        return adapter

    def close(self):
        from tools.e22_4_generation import module_hashes
        self.bridge.close();after=module_hashes(self.modules);assert after==self.before
        write(OUT/'audit/frozen_modules.json',dict(before=self.before,after=after,pass_unchanged=True))


class FixedInputs:
    def __init__(self,g):
        from train_GAM_texture_joint import JointTextureDataset
        self.g=g;self.pipe=g.pipe
        self.dataset=JointTextureDataset(str(OUT/'splits/probe_original_records.json'),g.pipe.tokenizer,
            '/share/home/u2515283058/datasets/BF/training',width=384,height=512,texture_preprocess_mode='plain_resize')
        self.scheduler=DDIMScheduler(beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear',
            num_train_timesteps=1000,prediction_type='epsilon')
        self.pipe.set_scale(.6)

    @torch.no_grad()
    def sample(self,record):
        from models.text_guided_queries import text_content_mask
        from garment_mask_utils import build_region_masks
        # 每条记录独立 seed；400/800 和所有损失复用相同 zt，原后验 sample 不改成 mean。
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(record['noise_seed']);torch.cuda.manual_seed_all(record['noise_seed'])
            s=self.dataset[record['index']]
            batch={k:s[k][None].cuda() for k in
                ['vae_cloth','vae_sketch','clip_texture','texture_image','garment_mask','input_ids']}
            p=self.pipe;dtype=p.unet.dtype
            tex=batch['texture_image'].to(dtype);clip=batch['clip_texture'].to(dtype);ids=batch['input_ids']
            latents=p.vae.encode(batch['vae_cloth'].to(dtype)).latent_dist.sample()*p.vae.config.scaling_factor
            sketch=batch['vae_sketch'].to(dtype)
            ref=p.vae.encode(sketch).latent_dist.sample()*p.vae.config.scaling_factor
            text=p.text_encoder(ids)[0];vision=p.image_encoder(clip,output_hidden_states=True)
            tokens=p.bf_texture_conditioner(clip_image_embeds=vision.image_embeds,texture_images=tex,
                clip_vision_tokens=vision.hidden_states[-1][:,1:],texture_mode='patch_resampled',
                text_embeds=text,text_mask=text_content_mask(ids,p.tokenizer.eos_token_id),
                local_detail_source='off',local_detail_grid=4)[0]
            encoder=torch.cat([text,p.tcpm_lite(tokens,text)],1)
            noise=torch.randn_like(latents);t=torch.tensor([record['t']],device='cuda',dtype=torch.long)
            noisy=self.scheduler.add_noise(latents,noise,t)
            p.reference_unet(ref,torch.zeros_like(t),None,return_dict=False)
            sa={n:proc.cache['hidden_states'] for n,proc in p.reference_unet.attn_processors.items()
                if 'attn1' in n and hasattr(proc,'cache')}
            # 原 E5 条件 mask 保留原数据管线；新增代理 mask 另从严格 sketch_only 构造。
            mask=build_region_masks(batch['garment_mask'].float(),kernel_size=9)[0]
            kw=dict(encoder_hidden_states=encoder,cross_attention_kwargs=dict(sa_hidden_states=sa,tcpm_garment_mask=mask))
            audit=dict(**record,noise_sha256=digest(noise),latent_sha256=digest(latents),
                noisy_sha256=digest(noisy),sketch_latent_sha256=digest(ref),dropout='none; fixed diagnostics only')
            return (noisy,t),kw,noise,sketch,(batch['vae_cloth'].float()+1)/2,(tex.float()+1)/2,audit


def predict(g,args,kw,sketch):
    g.bridge.sketch=sketch;g.bridge.adapter=g.adapters['B1_CONV']
    with torch.autocast('cuda',dtype=torch.float16):return g.pipe.unet(*args,**kw).sample


def decode_x0(g,data,args,prediction):
    noisy,t=args;alpha=data.scheduler.alphas_cumprod.to(noisy.device)[t].float()[:,None,None,None]
    x0=(noisy.float()-(1-alpha).sqrt()*prediction.float())/alpha.sqrt()
    # VAE 权重冻结，decode 本身不能 no_grad/inference_mode。
    decoded=g.pipe.vae.decode((x0/g.pipe.vae.config.scaling_factor).to(g.pipe.vae.dtype)).sample
    return (decoded.float()+1)/2
