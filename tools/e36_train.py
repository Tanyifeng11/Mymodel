"""冻结 E5，仅 adapter 通过原标准 epsilon MSE 更新。"""
import argparse
import csv
import random
import time
import torch
from torch.nn import functional as F
from diffusers import DDIMScheduler
from models.e36_guided_kernel_filter import GuidedKernelFilter
from tools.e36_protocol import *
from tools.e36_infer import Generator,digest


class TrainingInputs:
    def __init__(self,g):
        from train_GAM_texture_joint import JointTextureDataset
        self.g=g;self.pipe=g.pipe
        self.dataset=JointTextureDataset(str(OUT/'splits/train_original_records.json'),g.pipe.tokenizer,
            '/share/home/u2515283058/datasets/BF/training',width=384,height=512,texture_preprocess_mode='plain_resize')
        self.scheduler=DDIMScheduler(beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear',
            num_train_timesteps=1000,prediction_type='epsilon')
        self.null=g.pipe.tokenizer('',padding='max_length',truncation=True,
            max_length=g.pipe.tokenizer.model_max_length,return_tensors='pt').input_ids.cuda()
        self.order=[];self.rng=random.Random(42)
        self.ids=read(OUT/'splits/train1024.json')
        g.pipe.set_scale(.6)

    @torch.no_grad()
    def next(self,batch_size=1):
        from models.text_guided_queries import text_content_mask
        from garment_mask_utils import build_region_masks
        samples=[];ids=[]
        for _ in range(batch_size):
            if not self.order:
                self.order=list(range(len(self.dataset)));self.rng.shuffle(self.order)
            i=self.order.pop();samples.append(self.dataset[i]);ids.append(self.ids[i]['id'])
        batch={k:torch.stack([s[k] for s in samples]).cuda() for k in
            ['vae_cloth','vae_sketch','clip_texture','texture_image','garment_mask','input_ids']}
        p=self.pipe;dtype=p.unet.dtype
        input_ids=batch['input_ids'];tex=batch['texture_image'].to(dtype);clip=batch['clip_texture'].to(dtype)
        drops=[]
        for i in range(batch_size):
            r=self.rng.random();drop='none'
            if r<.05:tex[i]=0;clip[i]=0;drop='image'
            elif r<.25:input_ids[i]=self.null[0];drop='text'
            elif r<.30:tex[i]=0;clip[i]=0;input_ids[i]=self.null[0];drop='both'
            drops.append(drop)
        latents=p.vae.encode(batch['vae_cloth'].to(dtype)).latent_dist.sample()*p.vae.config.scaling_factor
        sketch=batch['vae_sketch'].to(dtype)
        ref=p.vae.encode(sketch).latent_dist.sample()*p.vae.config.scaling_factor
        text=p.text_encoder(input_ids)[0];vision=p.image_encoder(clip,output_hidden_states=True)
        tokens=p.bf_texture_conditioner(clip_image_embeds=vision.image_embeds,texture_images=tex,
            clip_vision_tokens=vision.hidden_states[-1][:,1:],texture_mode='patch_resampled',
            text_embeds=text,text_mask=text_content_mask(input_ids,p.tokenizer.eos_token_id),
            local_detail_source='off',local_detail_grid=4)[0]
        tokens=p.tcpm_lite(tokens,text);encoder=torch.cat([text,tokens],1)
        noise=torch.randn_like(latents);t=torch.randint(0,1000,(batch_size,),device='cuda')
        noisy=self.scheduler.add_noise(latents,noise,t)
        p.reference_unet(ref,torch.zeros_like(t),None,return_dict=False)
        sa={n:proc.cache['hidden_states'] for n,proc in p.reference_unet.attn_processors.items()
            if 'attn1' in n and hasattr(proc,'cache')}
        mask=build_region_masks(batch['garment_mask'].float(),kernel_size=9)[0]
        kw=dict(encoder_hidden_states=encoder,cross_attention_kwargs=dict(sa_hidden_states=sa,tcpm_garment_mask=mask))
        return (noisy,t),kw,noise,sketch,dict(ids=ids,t=t.tolist(),dropout=drops,
            noise_sha256=digest(noise),latent_sha256=digest(latents),sketch_latent_sha256=digest(ref))


def gradients(adapter):
    return {n:float(p.grad.float().norm()) if p.grad is not None else None for n,p in adapter.named_parameters()}


def build_adapters(g):
    trace=read(OUT/'p0/hook_trace.json')['trace']
    first=next(r for r in trace if r['index']==0)
    assert first['shape'][-2:]==[32,24]
    c=first['shape'][1];time_c=first['time_shape'][1]
    adapters={}
    for arm in ARMS:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            adapters[arm]=GuidedKernelFilter(c,time_c,arm).cuda()
    counts={arm:sum(p.numel() for p in m.parameters()) for arm,m in adapters.items()}
    assert max(counts.values())/min(counts.values())<=1.05,counts
    return adapters,counts


def save(adapter,arm,step,extra):
    path=OUT/'checkpoints'/arm/('step%04d.pt'%step);path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(arm=arm,step=step,state_dict=adapter.state_dict(),config=CONFIG,**extra),path)
    write(path.with_suffix('.json'),dict(arm=arm,step=step,sha256=sha(path),commit=commit(),**extra))


def train(arm):
    assert read(OUT/'p0/smoke_report.json')['pass_p0']
    from tools.e34_sarr_protocol import seed_all
    seed_all(42);g=Generator();adapters,counts=build_adapters(g);adapter=adapters[arm]
    del adapters;g.adapters={arm:adapter};g.bridge.adapter=adapter
    data=TrainingInputs(g);seed_all(42)
    optimizer=torch.optim.AdamW(adapter.parameters(),lr=.0001,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda');history=[];sequence=[];started=time.perf_counter()
    save(adapter,arm,0,dict(parameters=counts[arm]))
    try:
        for step in range(1,801):
            optimizer.zero_grad(set_to_none=True);losses=[];base_losses=[];eps_mae=[]
            diagnostic=step in (1,5,100,800) or step%50==0
            adapter.collect=diagnostic
            for micro in range(8):
                args,kw,noise,sketch,record=data.next();sequence.append(record)
                g.bridge.sketch=sketch;g.bridge.adapter=adapter
                with torch.autocast('cuda',dtype=torch.float16):prediction=g.pipe.unet(*args,**kw).sample
                loss=F.mse_loss(prediction.float(),noise.float());assert torch.isfinite(loss)
                losses.append(float(loss.detach()));scaler.scale(loss/8).backward()
                if diagnostic and micro==0:
                    with torch.no_grad():
                        g.bridge.adapter=None
                        with torch.autocast('cuda',dtype=torch.float16):base=g.pipe.unet(*args,**kw).sample
                        g.bridge.adapter=adapter
                        base_losses.append(float(F.mse_loss(base.float(),noise.float())))
                        eps_mae.append(float((prediction.float()-base.float()).abs().mean()))
            scaler.unscale_(optimizer)
            grads=gradients(adapter) if step in (1,5,100,800) else None
            norm=float(torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.))
            assert torch.isfinite(torch.tensor(norm))
            scaler.step(optimizer);scaler.update()
            if diagnostic:
                item=dict(update=step,loss=sum(losses)/8,e5_loss_same_batch=base_losses[0],
                    conditional_eps_mae=eps_mae[0],unconditional_eps_mae=0.,unconditional_policy='OFF by design',
                    grad_norm=norm,gradients=grads,seconds=time.perf_counter()-started,
                    peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,**adapter.stats)
                history.append(item);write(OUT/'train'/arm/'history.json',history)
                print(arm,step,item['loss'],item['conditional_eps_mae'],flush=True)
            if step in (100,400,800):save(adapter,arm,step,dict(parameters=counts[arm]))
        write(OUT/'train'/arm/'random_sequence.json',sequence)
        write(OUT/'train'/arm/'complete.json',dict(updates=800,parameters=counts[arm],
            sequence_sha256=sha(OUT/'train'/arm/'random_sequence.json'),seconds=time.perf_counter()-started,
            peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,commit=commit()))
        with (OUT/'train'/arm/'train_history.csv').open('w',newline='') as f:
            keys=['update','loss','e5_loss_same_batch','conditional_eps_mae','unconditional_eps_mae','grad_norm','seconds','peak_memory_gib']
            writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(history)
    finally:g.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=ARMS,required=True)
    train(parser.parse_args().arm)
