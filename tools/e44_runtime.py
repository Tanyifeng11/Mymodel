"""真实 E5 实例的参考读取、完整 RGB 生成与冻结核验。"""
import copy
import time
import hashlib
from pathlib import Path
from types import MethodType
import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms.functional import to_tensor
from models.reference_attention_variants import ReferenceAttentionProcessor
from models.reference_feature_writer import ReferenceFeatureWriter, spatial_tokens
from tools.e44_protocol import OUT, LAYER, CONFIG, SIZE, E5, E5_SHA, read, write, sha, commit
from tools.e35_freeu_run import digest


def sketch_masks(sketch, device):
    from garment_mask_utils import build_sketch_garment_mask
    raw, info = build_sketch_garment_mask(sketch, *SIZE)
    garment = np.asarray(raw)>127
    distance=cv2.distanceTransform(garment.astype('uint8'),cv2.DIST_L2,5)
    soft=np.clip((distance-9)/8,0,1).astype('float32')
    valid=bool(info['mask_source']=='sketch_flood_fill' and not info['mask_low_confidence']
        and soft.mean()>=.03)
    if not valid: soft.fill(0)
    tensor=torch.from_numpy(soft)[None,None].to(device)
    return raw, tensor, dict(info,valid=valid,soft_coverage=float(soft.mean()),gt_fallback=False)


class Runtime:
    def __init__(self, variant='B0'):
        from models.e33tm_generation_wrapper import load_e5
        from tools.e22_4_generation import module_hashes
        assert sha(E5)==E5_SHA
        self.pipe,self.modules,size,self.ns=load_e5()
        assert size==SIZE and self.pipe.scheduler.config.prediction_type=='epsilon'
        self.before=module_hashes(self.modules)
        self.pipe.set_scale(.6)
        self.variant=variant;self.processor=None;self.writer=None
        self.bf_feature=None;self.pixel_reference=None;self.active=True
        self.mask=None;self.mask_info=None;self.allow_reference=True
        self.original=self.pipe.unet.attn_processors[LAYER]
        target=dict(self.pipe.unet.named_modules())[LAYER[:-10]]
        self.width=target.to_q.out_features
        self.capture_positive=True
        self.bf_pre_handle=self.pipe.bf_texture_conditioner.register_forward_pre_hook(self.before_bf,with_kwargs=True)
        self.bf_handle=self.pipe.bf_texture_conditioner.stage3.register_forward_hook(self.capture_bf)
        self.unet_handle=self.pipe.unet.register_forward_pre_hook(self.before_unet,with_kwargs=True)
        self.empty_text=self.pipe.text_encoder(self.pipe.tokenizer('',padding='max_length',truncation=True,
            max_length=self.pipe.tokenizer.model_max_length,return_tensors='pt').input_ids.to(self.pipe.device))[0].detach()
        if variant!='B0':
            if variant in ['B1','B3']:
                self.writer=ReferenceFeatureWriter(self.pipe,LAYER,'models/stable-diffusion-v1-5')
                ref_size=self.width
                self.writer_before=module_hashes({'writer':self.writer.unet})
            else:
                ref_size=self.pipe.bf_texture_conditioner.stage3[0].out_channels
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42)
                self.processor=ReferenceAttentionProcessor(self.original,LAYER,self.width,ref_size,variant).to(self.pipe.device)
            target.set_processor(self.processor)
        write(OUT/'manifests'/('model_'+variant+'.json'),dict(commit=commit(),e5_sha256=sha(E5),
            original_module_hashes=self.before,layer=LAYER,original_processor=type(self.original).__name__,
            hidden_size=self.width,inference_args=vars(self.ns),scheduler=dict(self.pipe.scheduler.config),
            attention_processors={n:type(p).__name__ for n,p in self.pipe.unet.attn_processors.items()},
            writer_initialization=CONFIG['writer'] if self.writer else None,
            writer_hashes=getattr(self,'writer_before',None),
            trainable_parameters=sum(p.numel() for p in self.parameters()),
            original_frozen_parameters=sum(p.numel() for m in self.modules.values() for p in m.parameters() if not p.requires_grad)))

    def parameters(self):
        return [p for n,p in self.processor.named_parameters() if not n.startswith('original.')] if self.processor else []

    def capture_bf(self,module,args,result):
        if self.capture_positive:self.bf_feature=result.detach()

    def before_bf(self,module,args,kwargs):
        # get_image_embeds 先算正条件再算零输入负条件，不能让负条件覆盖空间缓存。
        self.capture_positive=bool(kwargs['texture_images'].count_nonzero())

    def before_unet(self,module,args,kwargs):
        if self.processor is None:return
        branch='conditional' if 'sa_hidden_states' in (kwargs.get('cross_attention_kwargs') or {}) else 'unconditional'
        p=self.processor;p.branch=branch
        p.enabled=bool(self.active and branch=='conditional' and self.allow_reference and self.mask_info['valid'])
        if not p.enabled:return
        latent=args[0] if args else kwargs['sample']
        timestep=args[1] if len(args)>1 else kwargs['timestep']
        p.mask=F.interpolate(self.mask,size=(latent.shape[-2]//2,latent.shape[-1]//2),mode='area')
        if self.writer is not None:
            p.features=self.writer.write(timestep)
        else:
            p.features=spatial_tokens(self.bf_feature,self.bf_feature.shape[-2:])

    @torch.no_grad()
    def set_reference(self,pixels,mask,mask_info,allow=True):
        self.pixel_reference=pixels;self.mask=mask;self.mask_info=mask_info;self.allow_reference=allow
        if self.writer is not None and allow and mask_info['valid']:
            self.writer.set_reference(pixels,self.empty_text)

    @torch.inference_mode()
    def generate(self,row,path,seed=42,reference=None,zero_reference=False,capture=False):
        pipe=self.pipe
        sketch=Image.open(row['sketch']).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR)
        ref=reference if reference is not None else Image.open(row['reference']).convert('RGB')
        raw,mask,info=sketch_masks(sketch,pipe.device)
        pixels=to_tensor(ref.resize(SIZE,Image.Resampling.BILINEAR))[None].to(pipe.device)*2-1
        self.set_reference(pixels,mask,info,allow=not zero_reference)
        factor=pipe.vae_scale_factor
        shape=(1,4,SIZE[1]//factor,SIZE[0]//factor)
        noise=torch.randn(shape,device=pipe.device,dtype=pipe.unet.dtype,
            generator=torch.Generator(device=pipe.device).manual_seed(seed))
        initial=noise*pipe.scheduler.init_noise_sigma
        old_latents,old_step=pipe.prepare_latents,pipe.scheduler.step
        pipe.prepare_latents=MethodType(lambda _self,*a,**kw:initial.clone(),pipe)
        steps=[];final=[];contexts={};outputs={};handles=[]
        def scheduler_step(*a,**kw):
            result=old_step(*a,**kw);steps.append(int(a[1]));final[:]=[result[0]];return result
        pipe.scheduler.step=scheduler_step
        if capture:
            def pre(_m,a,kw):
                branch='conditional' if 'sa_hidden_states' in kw['cross_attention_kwargs'] else 'unconditional'
                if branch not in contexts:contexts[branch]=(a,copy.copy(kw))
            def post(_m,a,kw,out):
                branch='conditional' if 'sa_hidden_states' in kw['cross_attention_kwargs'] else 'unconditional'
                if branch not in outputs:outputs[branch]=out[0].clone()
            handles=[pipe.unet.register_forward_pre_hook(pre,with_kwargs=True),pipe.unet.register_forward_hook(post,with_kwargs=True)]
        old_get=pipe.get_image_embeds
        if zero_reference:
            # BF 的 R0 使用既有零输入条件嵌入，保留原负条件；Writer 分支 OFF。
            def zero_get(_self,*a,**kw):
                positive,negative=old_get(*a,**kw)
                return negative.clone(),negative
            pipe.get_image_embeds=MethodType(zero_get,pipe)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        try:
            image=pipe(prompt=row['caption'],null_prompt='',negative_prompt=' worst quality, low quality',
                ref_image=to_tensor(sketch)[None]*2-1,texture_clip_image=ref,width=384,height=512,
                num_inference_steps=50,guidance_scale=7.,sketch_scale=.6,ipa_scale=1.,
                texture_mode=self.ns.texture_mode,texture_condition_mode='token',texture_preprocess_mode='plain_resize',
                texture_num_tokens=pipe.effective_texture_num_tokens,texture_scale=1.,
                spatial_mask=to_tensor(raw)[None].to(pipe.device,torch.float16),
                generator=torch.Generator(device=pipe.device).manual_seed(seed))[0]
        finally:
            pipe.prepare_latents,pipe.scheduler.step,pipe.get_image_embeds=old_latents,old_step,old_get
            for h in handles:h.remove()
        torch.cuda.synchronize();assert len(steps)==50 and torch.isfinite(final[0]).all()
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);image.save(path)
        record=dict(id=row['id'],variant=self.variant,seed=seed,mask=info,seconds=time.perf_counter()-started,
            peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,steps=steps,
            initial_sha256=digest(initial),final_sha256=digest(final[0]),png_sha256=sha(path),
            reference_sha256=hashlib.sha256(ref.tobytes()).hexdigest(),gt_used_in_inference=False)
        write(path.with_suffix('.json'),record)
        return record,contexts,outputs

    def save(self,path,step,optimizer=None,scaler=None):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(variant=self.variant,step=step,state_dict=self.processor.trainable_state(),config=CONFIG,
            optimizer=optimizer.state_dict() if optimizer else None,scaler=scaler.state_dict() if scaler else None),path)
        write(path.with_suffix('.json'),dict(step=step,sha256=sha(path),commit=commit()))

    def load(self,path):
        state=torch.load(path,map_location='cpu',weights_only=False)
        assert state['config']==CONFIG and state['variant']==self.variant
        self.processor.load_state_dict(state['state_dict'],strict=False)
        return state

    def close(self):
        from tools.e22_4_generation import module_hashes
        dict(self.pipe.unet.named_modules())[LAYER[:-10]].set_processor(self.original)
        self.unet_handle.remove();self.bf_handle.remove();self.bf_pre_handle.remove()
        after=module_hashes(self.modules);assert self.before==after
        writer_check=None
        if self.writer:
            writer_check=module_hashes({'writer':self.writer.unet})
            assert writer_check==self.writer_before;self.writer.close()
        write(OUT/'manifests'/('frozen_'+self.variant+'.json'),dict(unchanged=True,before=self.before,after=after,writer=writer_check))
