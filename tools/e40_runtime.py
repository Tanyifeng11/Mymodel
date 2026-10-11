"""冻结 E5：提取三种纹样表示，生成128张受控结果，重评 E39 原输出。"""
import argparse
import subprocess
import os
import time
import cv2
import numpy as np
import torch
from PIL import Image
from tools.e40_protocol import OUT, OLD, read, write, sha, E5, EXPECTED_E5


def patches(image,mask=None):
    a=np.asarray(image.convert('RGB'));h,w=a.shape[:2];items=[]
    for y in range(0,h-63,32):
        for x in range(0,w-63,32):
            if mask is None or mask[y:y+64,x:x+64].mean()>=.95:
                items.append(a[y:y+64,x:x+64])
    if len(items)>5:items=[items[i] for i in np.linspace(0,len(items)-1,5).astype(int)]
    return items


def spectrum(items):
    if not items:return np.zeros(304,np.float32),False
    window=np.outer(np.hanning(64),np.hanning(64));powers=[]
    for a in items:
        p=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY).astype(float)/255
        powers.append(abs(np.fft.fft2((p-p.mean())*window))**2)
    power=np.mean(powers,0);power[0,0]=0
    fy,fx=np.meshgrid(np.fft.fftfreq(64),np.fft.fftfreq(64),indexing='ij')
    r=np.hypot(fx,fy);theta=np.arctan2(fy,fx)%np.pi
    valid=(r>=2/64)&(r<=.4)
    angular=np.array([power[valid&(theta>=i*np.pi/18)&(theta<(i+1)*np.pi/18)].sum() for i in range(18)])
    radial=np.array([power[(r>=i/64)&(r<(i+1)/64)].sum() for i in range(2,32)])
    mag=cv2.resize(np.log1p(np.fft.fftshift(power)),(16,16),interpolation=cv2.INTER_AREA).ravel()
    def norm(x):return x/max(float(np.linalg.norm(x)),1e-12)
    return np.concatenate([norm(mag),norm(angular),norm(radial)]).astype(np.float32),True


@torch.inference_mode()
def features(pipe,image,mask=None,mask_valid=True):
    image=image.convert('RGB');a=np.asarray(image)
    lab=cv2.cvtColor(a.astype(np.float32)/255,cv2.COLOR_RGB2LAB)
    selected=lab if mask is None else lab[mask]
    if not len(selected):selected=lab.reshape(-1,3)
    selected=selected.reshape(-1,3)
    hist=np.histogramdd(a.reshape(-1,3)/255,bins=4,range=((0,1),)*3)[0].ravel()
    color=np.concatenate([selected.mean(0)/100,selected.std(0)/100,hist/hist.sum()]).astype(np.float32)
    items=patches(image,mask) if mask_valid else []
    fft,valid=spectrum(items)
    local=[Image.fromarray(v).convert('L').convert('RGB') for v in items]
    imgs=[image]+local;emb=[]
    for start in range(0,len(imgs),4):
        pixels=pipe.clip_image_processor(images=imgs[start:start+4],return_tensors='pt').pixel_values.to(pipe.device,pipe.image_encoder.dtype)
        v=pipe.image_encoder(pixels).image_embeds.float();v=v/v.norm(dim=1,keepdim=True)
        emb.extend(v.cpu().numpy())
    dim=len(emb[0]);local_emb=np.stack(emb[1:]) if local else np.zeros((1,dim),np.float32)
    return dict(clip=emb[0].astype(np.float32),local=np.concatenate([local_emb.mean(0),local_emb.std(0)]).astype(np.float32),
        fft=fft,color=color,lab_mean=selected.mean(0),local_valid=np.asarray(bool(local)),fft_valid=np.asarray(valid),
        patch_count=np.asarray(len(items)),mask_valid=np.asarray(mask_valid))


def save_features(pipe,image,path,mask=None,valid=True):
    if path.exists():return
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**features(pipe,image,mask,valid))


def eval_mask(row,size):
    from eval.eval_utils import prepare_evaluation_masks
    b=prepare_evaluation_masks(size,sketch_path=row['sketch'],mask_policy='sketch_only')
    s=b['stats'];valid=s.get('mask_source')=='sketch_flood_fill' and not s.get('mask_low_confidence',True)
    mask=cv2.erode(b['garment'].astype('uint8'),np.ones((9,9),np.uint8))>0 if valid else None
    return mask,valid,s


def run(shard,shards):
    from models.e33tm_generation_wrapper import load_e5, generate
    from tools.e22_4_generation import module_hashes
    from garment_mask_utils import build_sketch_garment_mask
    assert sha(E5)==EXPECTED_E5
    pipe,modules,size,ns=load_e5();before=module_hashes(modules);started=time.perf_counter()
    write(OUT/('audit/runtime_%d.json'%shard),dict(job=os.environ.get('SLURM_JOB_ID'),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        code_sha256={str(p):sha(p) for p in __import__('pathlib').Path('tools').glob('e40_*.py')},
        device=torch.cuda.get_device_name(),E5_sha256=sha(E5),before=before))
    refs=read(OUT/'references.json')
    for i,r in enumerate(refs[shard::shards]):
        save_features(pipe,Image.open(r['path']),OUT/'features'/('ref_'+r['id']+'_'+r['variant']+'.npz'))
        if i%50==0:print('E40 REF FEATURES',shard,i,flush=True)
    generations=read(OUT/'generation.json')
    for i,r in enumerate(generations[shard::shards]):
        folder=OUT/'generated'/r['id'];folder.mkdir(parents=True,exist_ok=True)
        png=folder/'image.png'
        if not png.exists():
            sketch=Image.open(r['sketch']).convert('RGB').resize(size,Image.Resampling.BILINEAR)
            reference=Image.open(r['reference']).convert('RGB')
            mask,info=build_sketch_garment_mask(sketch,*size)
            with torch.inference_mode():image,record=generate(pipe,size,ns,r['caption'],sketch,reference,mask,42)
            image.save(png);write(folder/'generation.json',dict(**r,record=record,png_sha256=sha(png),training_updates=0))
        mask,valid,info=eval_mask(r,size)
        save_features(pipe,Image.open(png),folder/'features.npz',mask,valid)
        refpath=OUT/'features'/('genref_'+r['pattern']+'_c%d.npz'%r['color'])
        # 两个 shard 可能同时请求同一参考；每个只写自己的文件以避免并发覆盖。
        refpath=refpath.with_name(refpath.stem+'_s%d.npz'%shard)
        save_features(pipe,Image.open(r['reference']),refpath)
        write(folder/'evaluation.json',dict(mask_valid=valid,mask_stats=info,reference_features=str(refpath)))
        print('E40 GENERATION',shard,i+1,len(generations[shard::shards]),r['id'],flush=True)
    old=read(OLD/'pairs.json')
    for row in old[shard::shards]:
        src=OLD/'cases'/row['id'];dst=OUT/'e39_reuse'/row['id'];dst.mkdir(parents=True,exist_ok=True)
        mask,valid,info=eval_mask(row,size)
        for arm in read(src/'complete.json')['arms']:
            image=src/(arm+'.png');reference=src/(arm+'_reference.png')
            save_features(pipe,Image.open(image),dst/(arm+'_generated.npz'),mask,valid)
            if arm!='Rzero':save_features(pipe,Image.open(reference),dst/(arm+'_reference.npz'))
        write(dst/'audit.json',dict(id=row['id'],mask_valid=valid,mask_stats=info,
            original_png_sha256={arm:sha(src/(arm+'.png')) for arm in read(src/'complete.json')['arms']},
            mode='read-only reuse, no regeneration'))
    after=module_hashes(modules);assert before==after
    write(OUT/('audit/frozen_%d.json'%shard),dict(before=before,after=after,unchanged=True,
        seconds=time.perf_counter()-started,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30))
    write(OUT/('shard%d_complete.json'%shard),dict(shard=shard,complete=True))
    print('E40 SHARD COMPLETE',shard,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=2)
    a=p.parse_args();run(a.shard,a.shards)
