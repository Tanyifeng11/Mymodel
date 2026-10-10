"""S0：真实原始 E5 完整 DDIM50，固定128身份；不训练、不接入方向场。"""
import argparse
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image,ImageDraw
from garment_mask_utils import build_sketch_garment_mask
from tools.e34_sarr_protocol import *
from tools.e34_sarr_metrics import Descriptor,patch_boxes,components,distance,distribution,gray_high

def rgb(path,size=SIZE,resample=Image.Resampling.BILINEAR):
    return np.asarray(Image.open(path).convert('RGB').resize(size,resample),np.float32)/255

def run(shard=0,shards=1):
    init();seed_all(42);torch.set_num_threads(2);cv2.setNumThreads(1)
    rows=read(OUT/'splits/dev128.json')[shard::shards]
    from models.e33tm_generation_wrapper import load_e5,generate
    from tools.e22_4_generation import module_hashes
    from tools.e33tm_weight_audit import effective_hashes
    pipe,modules,size,ns=load_e5()
    assert size==SIZE and type(pipe.scheduler).__name__=='DDIMScheduler'
    assert all(not p.requires_grad for m in modules.values() for p in m.parameters())
    before=module_hashes(modules);effective=effective_hashes(modules)
    metric=Descriptor('cuda')
    job=OUT/'jobs'/('s0_shard%d'%shard)
    write(job/'implementation.json',dict(git_commit=commit(),shard=shard,shards=shards,
        e5_checkpoint_sha256=sha(E5),scheduler_class=type(pipe.scheduler).__name__,
        scheduler_config=dict(pipe.scheduler.config),inference_args=vars(ns),
        modules_before=before,effective_before=effective,lpips=metric.info,lpips_state_sha256=metric.hash,
        files={str(p):sha(p) for p in list(Path('tools').glob('e34_sarr*.py'))+list(Path('models').glob('e34_sarr*.py'))},
        precision=str(pipe.unet.dtype),vae_precision=str(pipe.vae.dtype),device=torch.cuda.get_device_name()))
    for index,row in enumerate(rows):
        dest=OUT/'s0_e5_dev'/row['id'];done=dest/'audit.json'
        if done.exists():
            old=read(done)
            assert all(sha(dest/name)==h for name,h in old['files'].items())
            continue
        dest.mkdir(parents=True,exist_ok=True)
        sketch=Image.open(row['sketch']).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR)
        reference=Image.open(row['reference']).convert('RGB')
        mask,_=build_sketch_garment_mask(sketch,*SIZE)
        with np.load(OUT/'masks'/row['id']/'mask.npz') as z: safe=z['safe'].copy()
        captured=[];timesteps=[];decode=pipe.vae.decode;step=pipe.scheduler.step
        def capture(*args,**kwargs):
            result=decode(*args,**kwargs)
            captured.append((result[0]/2+.5).clamp(0,1).float().detach().cpu()[0])
            return result
        def count(*args,**kwargs):
            timesteps.append(int(args[1]));return step(*args,**kwargs)
        pipe.vae.decode=capture;pipe.scheduler.step=count
        start=time.perf_counter()
        try:
            with torch.inference_mode():image,info=generate(pipe,SIZE,ns,row['caption'],sketch,reference,mask,42)
        finally:pipe.vae.decode=decode;pipe.scheduler.step=step
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        assert len(captured)==1 and len(timesteps)==50
        raw=captured[0].permute(1,2,0).numpy()
        assert np.array_equal(np.round(raw*255).astype(np.uint8),np.asarray(image)), '保存量化必须与原E5一致'
        replay=None
        if index==0:
            with torch.inference_mode():again,again_info=generate(pipe,SIZE,ns,row['caption'],sketch,reference,mask,42)
            replay=bool(np.array_equal(np.asarray(again),np.asarray(image)))
            assert replay and again_info==info, 'E5 seed42 无法精确重放'
        image.save(dest/'I0.png')
        torch.save(captured[0].contiguous(),dest/'I0_float.pt')
        gt=rgb(row['gt']);ref=rgb(row['reference'],SIZE,Image.Resampling.BICUBIC)
        boxes=patch_boxes(safe,row['id'])
        ref_boxes=patch_boxes(np.ones((512,384),np.float32),row['id']+'/reference')
        desc={name:metric.extract(pix,bs) for name,pix,bs in [('I0',raw,boxes),('GT',gt,boxes),('R',ref,ref_boxes)]}
        a,b=desc['I0'],desc['GT']
        energy_ratio=a['HF_energy']/max(b['HF_energy'],1e-12) if a and b else None
        color=None
        if (safe>.5).any():
            inside=safe>.5
            lp=[cv2.GaussianBlur(p,(0,0),8) for p in [raw,gt,ref]]
            lab=[cv2.cvtColor(p,cv2.COLOR_RGB2Lab) for p in lp]
            color=dict(I0_GT=float(np.linalg.norm(lab[0][inside].mean(0)-lab[1][inside].mean(0))),
                I0_R=float(np.linalg.norm(lab[0][inside].mean(0)-lab[2].reshape(-1,3).mean(0))))
        write(dest/'audit.json',dict(id=row['id'],seed=42,seconds=seconds,timesteps=timesteps,
            generation=info,first_identity_exact_replay=replay,mask_source='sketch_only',
            gt_in_generation=False,boxes=boxes,reference_boxes=ref_boxes,descriptors=desc,
            HF_energy_ratio=energy_ratio,low_frequency_Lab=color,
            files={name:sha(dest/name) for name in ['I0.png','I0_float.pt']}))
        print('S0',shard,index+1,'/',len(rows),row['id'],'seconds',round(seconds,2),
            'patches',len(boxes),'HF_ratio',energy_ratio,flush=True)
    after=module_hashes(modules);active_after=effective_hashes(modules)
    assert before==after and effective==active_after
    assert all(p.grad is None for m in modules.values() for p in m.parameters())
    write(job/'frozen_modules.json',dict(before=before,after=after,effective_before=effective,
        effective_after=active_after,pass_unchanged=True,all_grad_none=True,
        peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3))
    verify_sources()
    if shards==1:summarize()

def panels(rows):
    folder=OUT/'visual_audit/s0_fixed24';folder.mkdir(parents=True,exist_ok=True)
    for page in range(0,24,4):
        sheet=Image.new('RGB',(960,4*294),'white');draw=ImageDraw.Draw(sheet)
        for j,row in enumerate(rows[page:page+4]):
            sid=row['id'];top=j*294
            base=rgb(OUT/'s0_e5_dev'/sid/'I0.png');gt=rgb(row['gt'])
            with np.load(OUT/'masks'/sid/'mask.npz') as z:mask=z['safe']
            error=np.abs(gray_high(base)-gray_high(gt))*mask
            heat=cv2.applyColorMap(np.round(np.clip(error/.15,0,1)*255).astype(np.uint8),cv2.COLORMAP_INFERNO)
            heat=Image.fromarray(cv2.cvtColor(heat,cv2.COLOR_BGR2RGB))
            items=[('Sketch',Image.open(row['sketch'])),('Reference',Image.open(row['reference'])),
                ('GT',Image.open(row['gt'])),('E5 RGB',Image.open(OUT/'s0_e5_dev'/sid/'I0.png')),('HF error',heat)]
            draw.text((3,top+2),sid+' | '+row['caption'][:135],fill='black')
            large=Image.new('RGB',(5*384,550),'white');ld=ImageDraw.Draw(large)
            ld.text((3,3),sid+' | '+row['caption'],fill='black')
            for k,(label,im) in enumerate(items):
                draw.text((k*192+2,top+20),label,fill='black')
                sheet.paste(im.convert('RGB').resize((192,256),Image.Resampling.BILINEAR),(k*192,top+37))
                ld.text((k*384+2,22),label,fill='black')
                large.paste(im.convert('RGB').resize(SIZE,Image.Resampling.BILINEAR),(k*384,38))
            large.save(folder/(sid+'.png'))
        sheet.save(folder/('page%02d.png'%(page//4)))

def summarize():
    rows=read(OUT/'splits/dev128.json')
    records=[read(OUT/'s0_e5_dev'/r['id']/'audit.json') for r in rows]
    valid=[r for r in records if r['descriptors']['GT'] is not None]
    assert len(records)==128
    scales=np.maximum(np.median([components(r['descriptors']['GT'],r['descriptors']['R']) for r in valid],axis=0),1e-6)
    write(OUT/'metric_calibration.json',dict(scales=scales.tolist(),weights=[.35,.35,.30],n=len(valid),
        fixed_before_training=True,source='train-only dev GT/R; never official validation'))
    for r in records:
        d=r['descriptors'];r['D_tex_E5_R']=distance(d['I0'],d['R'],scales)
        r['D_tex_GT_R']=distance(d['GT'],d['R'],scales)
    write(OUT/'s0_summary.json',dict(n=128,valid_patch_identities=len(valid),
        HF_ratio=distribution([r['HF_energy_ratio'] for r in records]),
        HF_ratio_below_one_n=sum(r['HF_energy_ratio'] is not None and r['HF_energy_ratio']<1 for r in records),
        D_tex_E5_R=distribution([r['D_tex_E5_R'] for r in records]),
        D_tex_GT_R=distribution([r['D_tex_GT_R'] for r in records]),
        generation_seconds=distribution([r['seconds'] for r in records]),
        raw_metrics=[dict(id=r['id'],HF_ratio=r['HF_energy_ratio'],D_tex_E5_R=r['D_tex_E5_R'],
            D_tex_GT_R=r['D_tex_GT_R'],low_frequency_Lab=r['low_frequency_Lab']) for r in records],
        AI_visual_review='pending; energy alone cannot decide task match',human_review=False))
    panels(rows);verify_sources()
    decision(S0='pending_AI_visual_review',next_phase='fixed24_visual_review')
    print('SUMMARY',json.dumps(read(OUT/'s0_summary.json'))[:1300],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--summarize',action='store_true');a=p.parse_args()
    if a.summarize:summarize()
    else:run(a.shard,a.shards)
