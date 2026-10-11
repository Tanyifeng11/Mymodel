"""E41 冻结推理、同状态参考干预和七点预测x0；复用E39图像不重新生成。"""
import argparse
import os
import subprocess
import time
import copy
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor
import tools.e39_runtime as previous
from tools.e41_protocol import *
from tools.e41_attributes import extract,latent_readout,render
from tools.e40_runtime import eval_mask


def compact_features(features):
    result={}
    for layer in LAYERS:
        a=torch.as_tensor(features[layer]).float()
        if a.ndim==4:a=a.flatten(2).transpose(1,2)
        elif a.ndim==2:a=a[:,None,:]
        pooled=torch.cat([a.mean(1),a.std(1,unbiased=False)],1).ravel()
        spatial=F.adaptive_avg_pool1d(a.transpose(1,2),16).ravel()
        result[layer+'_meanstd']=pooled.numpy().astype(np.float32)
        result[layer+'_spatial']=spatial.numpy().astype(np.float32)
    return result


class Runner(previous.Runner):
    def __init__(self,shard):
        previous.OUT=OUT
        super().__init__(shard)
        self.observe=False

    @torch.inference_mode()
    def reference_features(self,image,text='a cloth'):
        self.arm='reference'
        positive,negative=self.p.encode_prompt(text,self.p.device,1,True,' worst quality, low quality')[:2]
        self.p.get_image_embeds(pil_image=image,width=384,height=512,texture_mode=self.ns.texture_mode,
            text_embeds=positive,negative_text_embeds=negative)
        return compact_features(self.features)

    @torch.inference_mode()
    def make_condition(self,row,reference):
        self.arm='reference'
        self.p.set_texture_token_enabled(True);self.p.set_scale(.6);self.p.set_ipa_scale(1.)
        pos,neg=self.p.encode_prompt(row['caption'],self.p.device,1,True,' worst quality, low quality')[:2]
        tokens,negative=self.p.get_image_embeds(pil_image=reference,width=384,height=512,
            texture_mode=self.ns.texture_mode,text_embeds=pos,negative_text_embeds=neg)
        sketch=Image.open(row['sketch']).convert('RGB').resize((384,512),Image.Resampling.BILINEAR)
        lat=self.p.vae.encode((to_tensor(sketch)[None]*2-1).to(self.p.device,self.p.vae.dtype)).latent_dist.mean*.18215
        self.p.reference_unet(lat,torch.tensor(0,device=self.p.device),encoder_hidden_states=None,return_dict=False)
        states={name:proc.cache['hidden_states'] for name,proc in self.p.reference_unet.attn_processors.items() if 'attn1' in name}
        common=dict(timestep_cond=None,added_cond_kwargs=None,return_dict=False)
        return dict(**common,encoder_hidden_states=torch.cat([pos,tokens],1),
            cross_attention_kwargs=dict(sa_hidden_states=states)),dict(**common,encoder_hidden_states=torch.cat([neg,negative],1),cross_attention_kwargs={})

    @torch.inference_mode()
    def decode(self,latent):
        rgb=self.p.vae.decode(latent.to(self.p.device,self.p.vae.dtype)/self.p.vae.config.scaling_factor,return_dict=False)[0]
        unclamped=rgb.float()/2+.5
        a=(unclamped.clamp(0,1)[0].permute(1,2,0).cpu().numpy()*255).round().astype('uint8')
        return Image.fromarray(a),float(((unclamped<0)|(unclamped>1)).float().mean())

    @torch.inference_mode()
    def trajectory(self,row,folder,archive):
        mask,valid,mask_info=eval_mask(row,(384,512));records=[];arrays={}
        canonical_mask=cv2.resize(mask.astype('uint8'),(256,256),interpolation=cv2.INTER_NEAREST)>0 if valid else None
        if mask is not None:Image.fromarray((mask*255).astype('uint8')).save(folder/'evaluation_mask.png')
        for t in TIMES:
            idx=int(np.where(archive['timesteps']==t)[0][0])
            z=torch.from_numpy(archive['latents'][idx]).to(self.p.device).float()
            eps=torch.from_numpy(archive['eps_guided'][idx]).to(self.p.device).float()
            alpha=self.p.scheduler.alphas_cumprod[t].to(self.p.device)
            x0=(z-(1-alpha).sqrt()*eps)/alpha.sqrt()
            image,saturation=self.decode(x0)
            path=folder/('pred_x0_t%d.png'%t);image.save(path)
            metrics=extract(image.resize((256,256),Image.Resampling.BILINEAR),canonical_mask) if valid else {a:None for a in ATTRS}
            lm=cv2.resize(mask.astype('uint8'),(48,64),interpolation=cv2.INTER_NEAREST)>0 if valid else None
            latent=x0.detach().cpu().numpy()
            if lm is not None:latent=latent*lm[None,None]
            arrays['x0_t%d'%t]=x0.cpu().numpy().astype(np.float16)
            arrays['z_t%d'%t]=z.cpu().numpy().astype(np.float16)
            arrays['readout_t%d'%t]=latent_readout(latent)
            records.append(dict(t=t,attributes=metrics,saturated_fraction=saturation,path=str(path),mask_valid=valid))
        final=extract(Image.open(folder/'Rplus.png').resize((256,256),Image.Resampling.BILINEAR),canonical_mask) if valid else {a:None for a in ATTRS}
        write(folder/'attributes.json',dict(reference=extract(Image.open(row['reference'])),trajectory=records,final=final,
            mask_valid=valid,mask_stats=mask_info,coordinate='all RGB measurements mapped to 256x256; image coordinates only, no physical fabric scale',labels='generated attributes are uncalibrated-domain proxies'))
        np.savez_compressed(folder/'seven_latents.npz',**arrays)

    @torch.inference_mode()
    def sensitivity(self,rows,folder):
        base_row=next(r for r in rows if r['condition']=='base');basefolder=OUT/'generated'/base_row['id']
        with np.load(basefolder/'Rplus_trajectory.npz') as f:base={k:f[k].copy() for k in f.files}
        ref=Image.open(base_row['reference']).convert('RGB');positive,negative=self.make_condition(base_row,ref)
        tokens={}
        for r in rows:
            kwargs,neg=self.make_condition(r,Image.open(r['reference']).convert('RGB'))
            assert torch.equal(negative['encoder_hidden_states'],neg['encoder_hidden_states'])
            assert torch.equal(positive['encoder_hidden_states'][:,:-16],kwargs['encoder_hidden_states'][:,:-16])
            tokens[r['condition']]=kwargs['encoder_hidden_states']
        records=[]
        for t in TIMES:
            idx=int(np.where(base['timesteps']==t)[0][0]);z=torch.from_numpy(base['latents'][idx]).to(self.p.device)
            kw=dict(positive);kw['cross_attention_kwargs']=dict(positive['cross_attention_kwargs'],balanced_gate_timestep=torch.tensor([t/1000],device=self.p.device))
            self.arm='probe'
            original=self.p.unet(z,torch.tensor(t,device=self.p.device),**kw)[0]
            saved=torch.from_numpy(base['eps_cond'][idx]).to(self.p.device)
            check=previous.distance(saved,original)
            assert check['relative']<.003,check
            sham=self.p.unet(z,torch.tensor(t,device=self.p.device),**kw)[0]
            null=previous.distance(original,sham)
            for r in rows:
                if r['attribute'] is None:continue
                new=dict(kw,encoder_hidden_states=tokens[r['condition']])
                altered=self.p.unet(z,torch.tensor(t,device=self.p.device),**new)[0]
                delta=previous.distance(original,altered)
                a=r['attribute'];change=error(r['params'][a],BASE[a],a)
                records.append(dict(carrier=r['carrier'],condition=r['condition'],attribute=a,t=t,
                    epsilon=delta,sham=null,baseline_replay=check,input_parameter_difference=change,
                    sensitivity_per_parameter_unit=delta['relative']/change,
                    note='same Rplus z_t, same timestep/text/sketch/unconditional; epsilon response is not semantic correctness'))
        write(folder/'sensitivity.json',records)

    @torch.inference_mode()
    def vae_standard(self,row,path):
        image=render(row['params'],row['phase'],row['color'])
        input_image=image.resize((384,512),Image.Resampling.BILINEAR)
        pixels=to_tensor(input_image)[None].to(self.p.device,self.p.vae.dtype)*2-1
        latent=self.p.vae.encode(pixels).latent_dist.mean*self.p.vae.config.scaling_factor
        decoded,saturation=self.decode(latent)
        np.savez_compressed(path,readout=latent_readout(latent.cpu().numpy()))
        write(path.with_suffix('.json'),dict(id=row['id'],params=row['params'],family=row['family'],fold=row['fold'],
            reference=extract(image),roundtrip=extract(decoded.resize((256,256),Image.Resampling.BILINEAR)),saturation=saturation,
            scope='VAE encoded full-field procedural standard; generated garment latents have domain shift'))


def run(shard,shards):
    prepare();g=Runner(shard);started=time.perf_counter()
    try:
        refs=read(OUT/'references.json')
        for i,r in enumerate(refs[shard::shards]):
            dest=OUT/'features'/(r['id']+'.npz');dest.parent.mkdir(exist_ok=True)
            if not dest.exists():np.savez_compressed(dest,**g.reference_features(Image.open(r['path']).convert('RGB')))
            if r['domain']=='controlled':
                vp=OUT/'vae_standards'/(r['id']+'.npz');vp.parent.mkdir(exist_ok=True)
                if not vp.exists():g.vae_standard(r,vp)
            if i%30==0:print('E41 REFERENCE',shard,i,len(refs[shard::shards]),flush=True)
        generations=read(OUT/'generation.json');carriers=sorted(set(r['carrier'] for r in generations))
        for ci,carrier in enumerate(carriers[shard::shards]):
            rows=[r for r in generations if r['carrier']==carrier]
            complete=OUT/'carriers'/carrier/'complete.json'
            if complete.exists():continue
            for r in rows:
                folder=OUT/'generated'/r['id'];folder.mkdir(parents=True,exist_ok=True)
                if not (folder/'Rplus.png').exists():g.generate(r,'Rplus',Image.open(r['reference']).convert('RGB'),folder)
                if not (folder/'attributes.json').exists():
                    with np.load(folder/'Rplus_trajectory.npz') as f:archive={k:f[k].copy() for k in f.files}
                    g.trajectory(r,folder,archive)
                print('E41 GENERATED',shard,r['id'],flush=True)
            dest=OUT/'carriers'/carrier;dest.mkdir(parents=True,exist_ok=True)
            g.sensitivity(rows,dest);write(complete,dict(carrier=carrier,images=10,complete=True))
            print('E41 CARRIER COMPLETE',shard,ci+1,len(carriers[shard::shards]),flush=True)
        oldrows=read(E39/'pairs.json')
        for row in oldrows[shard::shards]:
            source=E39/'cases'/row['id'];folder=OUT/'real_reuse'/row['id'];folder.mkdir(parents=True,exist_ok=True)
            before={arm:sha(source/(arm+'.png')) for arm in read(source/'complete.json')['arms']}
            if not (folder/'attributes.json').exists():
                # 新目录仅复制复核图，原始图/latent只读。
                Image.open(source/'Rplus.png').save(folder/'Rplus.png')
                with np.load(source/'Rplus_trajectory.npz') as f:archive={k:f[k].copy() for k in f.files}
                g.trajectory(row,folder,archive)
            assert before=={arm:sha(source/(arm+'.png')) for arm in before}
            write(folder/'audit.json',dict(original_png_hashes=before,unchanged=True,new_generations=0))
    finally:g.close()
    write(OUT/('shard%d_complete.json'%shard),dict(complete=True,seconds=time.perf_counter()-started,
        job=os.environ.get('SLURM_JOB_ID'),node=os.environ.get('SLURMD_NODENAME'),peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30))
    print('E41 SHARD COMPLETE',shard,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=2)
    a=p.parse_args();run(a.shard,a.shards)
