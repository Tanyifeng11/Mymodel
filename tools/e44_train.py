"""三臂共用数据序列、噪声、timestep、dropout 和背景教师保持项。"""
import argparse
import random
import time
import torch
from diffusers import DDIMScheduler
from PIL import Image
from torch.nn import functional as F
from models.text_guided_queries import text_content_mask
from tools.e34_sarr_protocol import seed_all
from tools.e44_protocol import *
from tools.e44_runtime import Runtime, sketch_masks, digest


class TrainingInputs:
    def __init__(self,runtime,split='train1024'):
        from train_GAM_texture_joint import JointTextureDataset
        self.r=runtime;self.p=runtime.pipe
        self.meta=read(OUT/'manifests/e5_training_meta.json')
        self.rows=read(OUT/'splits'/(split+'.json'))
        self.dataset=JointTextureDataset(str(OUT/'splits'/(split+'_records.json')),
            self.p.tokenizer,'/share/home/u2515283058/datasets/BF/training',
            width=384,height=512,texture_preprocess_mode='plain_resize')
        self.scheduler=DDIMScheduler(beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear',
            num_train_timesteps=1000,prediction_type='epsilon')
        self.rng=random.Random(42);self.order=[]
        self.null=self.p.tokenizer('',padding='max_length',truncation=True,
            max_length=self.p.tokenizer.model_max_length,return_tensors='pt').input_ids.to(self.p.device)
        self.p.set_scale(.6)

    @torch.no_grad()
    def next(self,disable_dropout=False):
        if not self.order:
            self.order=list(range(len(self.dataset)));self.rng.shuffle(self.order)
        index=self.order.pop();sample=self.dataset[index];row=self.rows[index]
        p=self.p;dtype=p.unet.dtype
        batch={k:sample[k][None].to(p.device) for k in
            ['vae_cloth','vae_sketch','clip_texture','texture_image','garment_mask','input_ids']}
        tex=batch['texture_image'].to(dtype);clip=batch['clip_texture'].to(dtype);ids=batch['input_ids']
        value=self.rng.random();drop='none'
        if not disable_dropout:
            if value<.05:tex.zero_();clip.zero_();drop='image'
            elif value<.25:ids[:]=self.null;drop='text'
            elif value<.30:tex.zero_();clip.zero_();ids[:]=self.null;drop='both'
        latent=p.vae.encode(batch['vae_cloth'].to(dtype)).latent_dist.sample()*p.vae.config.scaling_factor
        sketch_latent=p.vae.encode(batch['vae_sketch'].to(dtype)).latent_dist.sample()*p.vae.config.scaling_factor
        text=p.text_encoder(ids)[0];vision=p.image_encoder(clip,output_hidden_states=True)
        tokens=p.bf_texture_conditioner(clip_image_embeds=vision.image_embeds,texture_images=tex,
            clip_vision_tokens=vision.hidden_states[int(self.meta['clip_hidden_layer'])][:,1:],texture_mode='patch_resampled',
            text_embeds=text,text_mask=text_content_mask(ids,p.tokenizer.eos_token_id),
            local_detail_source='off',local_detail_grid=4)[0]
        tokens=p.tcpm_lite(tokens,text)
        noise=torch.randn_like(latent);t=torch.randint(0,1000,(1,),device=p.device)
        noisy=self.scheduler.add_noise(latent,noise,t)
        p.reference_unet(sketch_latent,torch.zeros_like(t),None,return_dict=False)
        sa={n:proc.cache['hidden_states'].detach() for n,proc in p.reference_unet.attn_processors.items() if 'attn1' in n}
        # 原 E5 条件 mask 不改；新分支的 mask 单独从 Sketch 计算，禁止 GT 回退。
        from garment_mask_utils import build_region_masks
        old_mask=batch['garment_mask'].float()
        if self.meta['tcpm_mask_inner_only']:
            old_mask=build_region_masks(old_mask,kernel_size=int(self.meta['region_kernel_size']))[0]
        sketch=Image.open(row['sketch']).convert('RGB')
        _,mask,info=sketch_masks(sketch,p.device)
        self.r.set_reference(tex,mask,info,allow=drop not in ['image','both'])
        kw=dict(encoder_hidden_states=torch.cat([text,tokens],1),
            cross_attention_kwargs=dict(sa_hidden_states=sa,tcpm_garment_mask=old_mask))
        record=dict(id=row['id'],t=int(t),dropout=drop,mask_valid=info['valid'],
            noise_sha256=digest(noise),latent_sha256=digest(latent),sketch_latent_sha256=digest(sketch_latent))
        return (noisy,t),kw,noise,F.interpolate(mask,size=latent.shape[-2:],mode='area'),record


def loss_forward(runtime,args,kw,noise,mask):
    runtime.active=False
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        teacher=runtime.pipe.unet(*args,**kw).sample
    runtime.active=True
    with torch.autocast('cuda',dtype=torch.float16):
        prediction=runtime.pipe.unet(*args,**kw).sample
    diffusion=F.mse_loss(prediction.float(),noise.float())
    outside=1-mask.float()
    keep=((prediction.float()-teacher.float()).square()*outside).sum()/outside.sum().clamp_min(1e-8)
    return diffusion+.1*keep,dict(diffusion=float(diffusion.detach()),keep=float(keep.detach()),
        conditional_eps_mae=float((prediction.detach().float()-teacher.float()).abs().mean()))


def gradients(runtime):
    return {n:float(p.grad.float().norm()) if p.grad is not None else None
        for n,p in runtime.processor.named_parameters() if not n.startswith('original.')}


def train(variant,stage='pilot'):
    assert read(OUT/'p0/smoke_report.json')['pass_p0']
    if stage=='main':assert read(OUT/'p1/selection.json')['allow_main']
    seed_all(42);runtime=Runtime(variant);data=TrainingInputs(runtime,'train1024' if stage=='pilot' else 'train4096')
    seed_all(42);optimizer=torch.optim.AdamW(runtime.parameters(),lr=1e-4,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda');updates=800 if stage=='pilot' else 5000
    folder=OUT/('p1' if stage=='pilot' else 'p2')/variant
    runtime.save(folder/'checkpoints/step0000.pt',0,optimizer,scaler)
    started=time.perf_counter();history=[];sequence=[]
    try:
        step=0;empty_windows=0
        while step<updates:
            optimizer.zero_grad(set_to_none=True);items=[]
            for micro in range(8):
                args,kw,noise,mask,record=data.next();sequence.append(record)
                loss,item=loss_forward(runtime,args,kw,noise,mask)
                assert torch.isfinite(loss)
                # 无效mask和参考dropout时仍保留样本；其冻结主路损失没有新参数梯度。
                if loss.requires_grad:scaler.scale(loss/8).backward()
                items.append(item)
            if any(p.grad is not None for p in runtime.parameters()):
                step+=1
                scaler.unscale_(optimizer)
                grad=gradients(runtime) if step in [1,5,20,updates] else None
                norm=float(torch.nn.utils.clip_grad_norm_(runtime.parameters(),1.))
                assert np_isfinite(norm)
                scale_before=scaler.get_scale()
                scaler.step(optimizer);scaler.update()
                assert scaler.get_scale()>=scale_before,'梯度溢出导致 optimizer update 跳过；停止公平预算实验'
            else:
                empty_windows+=1
                continue
            if step in [1,5,20] or step%25==0 or step==updates:
                item=dict(step=step,**{k:sum(q[k] for q in items)/8 for k in items[0]},grad_norm=norm,
                    gradients=grad,seconds=time.perf_counter()-started,
                    peak_memory_gib=torch.cuda.max_memory_allocated()/2**30)
                history.append(item);write(folder/'history.json',history)
                print('E44 TRAIN',variant,step,item['diffusion'],item['keep'],flush=True)
            if step in CONFIG['checkpoints_'+stage]:
                runtime.save(folder/'checkpoints'/('step%04d.pt'%step),step,optimizer,scaler)
        write(folder/'random_sequence.json',sequence)
        write(folder/'complete.json',dict(updates=updates,sequence_sha256=sha(folder/'random_sequence.json'),
            seconds=time.perf_counter()-started,peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,
            parameters=sum(p.numel() for p in runtime.parameters()),empty_windows=empty_windows,commit=commit()))
    finally:runtime.close()


def np_isfinite(value):
    import math
    return math.isfinite(value)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=ARMS,required=True)
    parser.add_argument('--stage',choices=['pilot','main'],default='pilot');args=parser.parse_args()
    train(args.variant,args.stage)
