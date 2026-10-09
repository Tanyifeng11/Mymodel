"""2训练身份×3臂：no-op exact、实际last8/VAE梯度、RF常量对照与资源profiling。"""
import time
import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms.functional import to_tensor
from data.e33gc_renderer import construct
from data.e33r_group_dataset import source_inputs,encode_reference
from data.e32_target_pseudogt import structure_input
from models.apacc_features import load_dino
from models.e33tm_generation_wrapper import load_e5
from models.e33gc_adapter import CausalResidual,TextureInjection,SITE
from tools.e33gc_sampling import prefix,final8,tensor_sha
from tools.e33gc_protocol import *
from tools.e33rf_common import build,WEIGHTS
from tools.e33tmoc_appearance_eval import model_hash
from tools.e22_4_generation import module_hashes
from tools.e33tm_weight_audit import effective_hashes

def orientation(rgb):
    gray=(rgb*rgb.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
    k=gray.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]])[None,None]/8
    gx=F.conv2d(gray,k,padding=1);gy=F.conv2d(gray,k.transpose(-1,-2),padding=1)
    return F.normalize(-torch.cat([F.avg_pool2d(gx.square()-gy.square(),8),F.avg_pool2d(2*gx*gy,8)],1),dim=1,eps=1e-8)

def loss(images,targets,mask):
    ori=[orientation(image) for image in images];m=F.interpolate(mask,ori[0].shape[-2:],mode='nearest')
    rec=torch.stack([((image-target).square()+1e-6).sqrt().mul(mask).sum()/mask.expand_as(image).sum().clamp_min(1)
        for image,target in zip(images,targets)]).mean()
    pair=((1+(ori[0]*ori[1]).sum(1,keepdim=True))*m).sum()/m.sum().clamp_min(1)
    identity=((1-(ori[0]*ori[2]).sum(1,keepdim=True))*m).sum()/m.sum().clamp_min(1)
    return rec+.5*(pair+.5*identity)

@torch.no_grad()
def fields(rendered,rf,dino):
    arrays=[source_inputs(np.asarray(ref.crop((128,192,256,320))),np.ones((128,128),bool)) for ref in rendered['references']]
    pixels=torch.from_numpy(np.stack([v[0] for v in arrays]))[None]
    aux=torch.from_numpy(np.stack([v[1] for v in arrays]))[None]
    enc=encode_reference(pixels,aux,dino)
    st=cv2.resize(structure_input(rendered['sketch'],rendered['mask'].astype(bool)),(96,128),interpolation=cv2.INTER_AREA).transpose(2,0,1)
    structure=torch.from_numpy(st.copy())[None].cuda().expand(3,-1,-1,-1)
    with torch.autocast('cuda',dtype=torch.bfloat16):pred=rf(enc.flatten(0,1),structure)
    geom=torch.cat([pred['orientation'].float(),pred['confidence_logits'].float().sigmoid()],1)
    geom=F.interpolate(geom,(512,384),mode='bilinear',align_corners=False);geom[:,:2]=F.normalize(geom[:,:2],dim=1)
    mask=torch.from_numpy(rendered['mask'].astype(np.float32))[None,None].cuda().expand(3,-1,-1,-1)
    return torch.cat([geom,mask],1)

def png(rgb):return Image.fromarray(np.uint8(np.round(rgb[0].detach().cpu().numpy().transpose(1,2,0)*255)))

def run():
    prepare();torch.manual_seed(42);torch.set_num_threads(2);cv2.setNumThreads(1)
    rows=read(OUT/'G2_autograd_smoke/train2_rows.json');assert len(rows)==2
    pipe,modules,size,ns=load_e5();assert size==(384,512) and type(pipe.scheduler).__name__=='DDIMScheduler'
    rf,_=build(42,checkpoint=RF/'seed42/RF2/checkpoint_final.pt');rf.eval().requires_grad_(False)
    dino,_=load_dino('cuda',WEIGHTS);dino.eval().requires_grad_(False)
    rf_before=model_hash(rf);dino_before=model_hash(dino)
    before=module_hashes(modules);active=effective_hashes(modules)
    records=[];torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    for row in rows:
        adapter=CausalResidual().cuda();injection=TextureInjection(pipe,adapter);rendered=construct(row)
        geom=fields(rendered,rf,dino);mask=torch.from_numpy(rendered['inner'].astype(np.float32))[None,None].cuda()
        pipeline_mask=torch.from_numpy(rendered['mask'].astype(np.float32))[None,None].cuda()
        falloff=torch.from_numpy(rendered['falloff'])[None,None].cuda()
        dest=OUT/'G2_autograd_smoke'/row['id'];dest.mkdir(parents=True,exist_ok=True)
        prefix_states=[];images=[];trace=[];replay=[]
        for i,arm in enumerate(['R0','R90','R180']):
            injection.geometry=geom[i:i+1];injection.falloff=falloff
            cached,baseline=prefix(pipe,ns,row['caption'],rendered['sketch'],rendered['references'][i],pipeline_mask,injection)
            torch.save(cached,dest/(arm+'_prefix.pt'));prefix_states.append(cached);baseline.save(dest/(arm+'_B0.png'))
            with torch.no_grad():rgb,scores=final8(pipe,ns,row['caption'],rendered['sketch'],rendered['references'][i],pipeline_mask,cached,injection,checkpointed=False,trace=True)
            image=png(rgb);image.save(dest/(arm+'_B1.png'));exact=np.array_equal(np.asarray(image),np.asarray(baseline));assert exact,'B1/replay must exact match B0 PNG pixels'
            images.append(rgb);trace.extend(scores);replay.append(dict(arm=arm,exact=exact,B0_sha256=sha(dest/(arm+'_B0.png')),B1_sha256=sha(dest/(arm+'_B1.png')),
                prefix_latent_sha256=tensor_sha(cached['latent']),noise_sha256=cached['generation']['noise_sha256']))
        assert len({v['noise_sha256'] for v in replay})==1
        targets=[to_tensor(v)[None].cuda() for v in rendered['targets']]
        def gradient(constant=False):
            adapter.zero_grad(set_to_none=True);generated=[]
            for i in range(3):
                injection.geometry=geom[0:1] if constant else geom[i:i+1];injection.falloff=falloff
                rgb,_=final8(pipe,ns,row['caption'],rendered['sketch'],rendered['references'][i],pipeline_mask,prefix_states[i],injection,checkpointed=True)
                assert rgb.requires_grad;generated.append(rgb)
                assert np.array_equal(np.asarray(png(rgb)),np.asarray(png(images[i]))),'checkpointed no-op must preserve PNG pixels'
            value=loss(generated,targets,mask);value.backward()
            grads=[p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in adapter.parameters()]
            assert all(torch.isfinite(g).all() for g in grads) and sum(float(g.norm()) for g in grads)>0
            assert all(p.grad is None for m in list(modules.values())+[rf,dino] for p in m.parameters())
            return generated,grads,float(value.detach())
        t=time.monotonic();generated,grads,value=gradient();gradient_seconds=time.monotonic()-t
        _,constant_grads,constant_loss=gradient(True)
        gradient_delta=sum(float((a-b).square().sum()) for a,b in zip(grads,constant_grads))**.5
        # 在强零初始化时输出相同，但projection梯度会因RF不同改变；只据数值记录，不虚构成功。
        for p,g in zip(adapter.parameters(),grads):p.grad=g
        optimizer=torch.optim.AdamW(adapter.parameters(),lr=1e-4,weight_decay=1e-4);optimizer.step()
        injection.geometry=geom[:1];injection.falloff=falloff
        with torch.no_grad():updated,_=final8(pipe,ns,row['caption'],rendered['sketch'],rendered['references'][0],pipeline_mask,prefix_states[0],injection,checkpointed=False)
        changed=float((updated-images[0]).abs().max());assert changed>0
        record=dict(id=row['id'],replay=replay,final8_autograd=True,gradient_finite=True,frozen_gradients_absent=True,
            projection_gradient=float(grads[-2].norm()),constant_RF_gradient_delta=gradient_delta,constant_RF_loss=constant_loss,
            loss=value,one_optimizer_step_RGB_max_change=changed,checkpointed_3arm_backward_seconds=gradient_seconds,
            peak_GPU_bytes=torch.cuda.max_memory_allocated(),actual_feature_shape=injection.last_shape,
            site=SITE,DDIM_timesteps=prefix_states[0]['timesteps'].tolist(),vae_scaling_factor=prefix_states[0]['vae_scale'],
            forward_GT=False,training_loss='smoke Charbonnier + paired Sobel orientation only; not final training objective')
        write(dest/'audit.json',record);write(dest/'CFG_trace.json',trace);png(updated).save(dest/'R0_after_one_update.png')
        injection.close();records.append(record);print('[GC Smoke]',row['id'],record,flush=True)
    assert before==module_hashes(modules) and active==effective_hashes(modules)
    assert rf_before==model_hash(rf) and dino_before==model_hash(dino)
    write(OUT/'protocol/layer_mapping.json',dict(site=SITE,feature_shape=records[0]['actual_feature_shape'],scale='1/8',
        branch='conditional IP texture residual before attention output projection',adapter_parameters=196064,
        source='actual original E5 texture attention output, not remapped reference RGB'))
    write(OUT/'G2_autograd_smoke/smoke_audit.json',dict(records=records,elapsed_seconds=time.monotonic()-start,
        frozen_before=before,frozen_after=module_hashes(modules),effective_before=active,effective_after=effective_hashes(modules),
        G2_pass=True,only_smoke_no_formal_training=True))
    decision(adapter_noop_exact=True,final8_autograd_pass=True,next_route='await_G0_G1_and_split_resolution')
    frozen_check();bundle('smoke')

if __name__=='__main__':run()
