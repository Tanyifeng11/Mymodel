"""最终 RGB 的参考偏好；现有 TPF patch 是颜色代理，另报去色 FFT。"""
import cv2
import numpy as np
import torch
from PIL import Image
from tools.e39_protocol import *


def fft(image, mask=None):
    a=np.asarray(image.convert('L').resize(SIZE),np.float64)/255
    keep=np.ones(a.shape,bool) if mask is None else mask
    powers=[];window=np.outer(np.hanning(64),np.hanning(64))
    for y in range(0,a.shape[0]-63,32):
        for x in range(0,a.shape[1]-63,32):
            if keep[y:y+64,x:x+64].mean()<.95:continue
            p=a[y:y+64,x:x+64]
            if p.std()<.01:continue
            powers.append(abs(np.fft.fft2((p-p.mean())*window))**2)
    if not powers:return dict(angle=None,confidence=None,patches=0,angular=None,radial=None)
    power=np.mean(powers,0)
    fy,fx=np.meshgrid(np.fft.fftfreq(64),np.fft.fftfreq(64),indexing='ij')
    radius=np.hypot(fx,fy);theta=np.arctan2(fy,fx)
    valid=(radius>=2/64)&(radius<=.4);total=power[valid].sum()
    vec=np.sum(power[valid]*np.exp(2j*theta[valid]))/max(total,1e-12)
    angular=np.array([power[valid&((theta%np.pi)>=i*np.pi/18)&((theta%np.pi)<(i+1)*np.pi/18)].sum() for i in range(18)])
    radial=np.array([power[(radius>=i/64)&(radius<(i+1)/64)].sum() for i in range(2,26)])
    angular/=max(angular.sum(),1e-12);radial/=max(radial.sum(),1e-12)
    return dict(angle=float((np.angle(vec)/2*180/np.pi+90)%180),confidence=float(abs(vec)),
                patches=len(powers),angular=angular.tolist(),radial=radial.tolist())


def axial(a,b):
    return abs((a-b+90)%180-90)


@torch.inference_mode()
def evaluate(p,row,folder):
    from eval.eval_utils import prepare_evaluation_masks
    from eval.metrics import patch_texture_similarity
    arms=read(folder/'complete.json')['arms']
    images={arm:Image.open(folder/(arm+'.png')).convert('RGB') for arm in arms}
    refs={arm:Image.open(folder/(arm+'_reference.png')).convert('RGB') for arm in arms if arm!='Rzero'}
    bundle=prepare_evaluation_masks(SIZE,sketch_path=row['sketch'],mask_policy='sketch_only')
    stats=bundle['stats'];valid=bool(stats.get('mask_source')=='sketch_flood_fill' and not stats.get('mask_low_confidence',True))
    mask=bundle['garment'].astype(bool) if valid else None
    inner=cv2.erode(mask.astype(np.uint8),np.ones((9,9),np.uint8))>0 if valid else None
    names=list(images)+['ref_'+k for k in refs]
    all_images=list(images.values())+list(refs.values())
    embeds=[]
    for i in range(0,len(all_images),4):
        pixels=p.clip_image_processor(images=all_images[i:i+4],return_tensors='pt').pixel_values.to(p.device,p.image_encoder.dtype)
        v=p.image_encoder(pixels).image_embeds.float();embeds.extend((v/v.norm(dim=1,keepdim=True)).cpu().numpy())
    em=dict(zip(names,embeds));spectrum={k:fft(v,inner) if valid else fft(v,np.zeros((SIZE[1],SIZE[0]),bool)) for k,v in images.items()}
    ref_fft={k:fft(v) for k,v in refs.items()}
    records=[]
    for arm,image in images.items():
        for ref,reference in refs.items():
            rec=dict(arm=arm,reference=ref,clip_texture=float(em[arm]@em['ref_'+ref]),
                     tpf_patch=None,lab_delta=None,fft_orientation_error=None,fft_angular_l1=None,fft_radial_l1=None)
            if valid:
                rec['tpf_patch']=float(patch_texture_similarity(image,reference,Image.fromarray(inner.astype(np.uint8)*255),patch=8))
                ilab=cv2.cvtColor(np.asarray(image,np.float32)/255,cv2.COLOR_RGB2LAB)
                rlab=cv2.cvtColor(np.asarray(reference,np.float32)/255,cv2.COLOR_RGB2LAB)
                rec['lab_delta']=float(np.linalg.norm(ilab[inner].mean(0)-rlab.reshape(-1,3).mean(0)))
                a,b=spectrum[arm],ref_fft[ref]
                if a['angular'] is not None and b['angular'] is not None:
                    rec['fft_angular_l1']=float(np.abs(np.array(a['angular'])-b['angular']).sum())
                    rec['fft_radial_l1']=float(np.abs(np.array(a['radial'])-b['radial']).sum())
                    if a['confidence']>=.15 and b['confidence']>=.15:
                        rec['fft_orientation_error']=axial(a['angle'],b['angle'])
            records.append(rec)
    lookup={(r['arm'],r['reference']):r for r in records};effects={}
    for arm in refs:
        if arm=='Rplus':continue
        a,b,c,d=[lookup[key] for key in [(arm,arm),(arm,'Rplus'),('Rplus',arm),('Rplus','Rplus')]]
        effects[arm]={}
        for key,direction in [('clip_texture',1),('tpf_patch',1),('lab_delta',-1),('fft_angular_l1',-1),('fft_radial_l1',-1),('fft_orientation_error',-1)]:
            values=[r[key] for r in [a,b,c,d]]
            effects[arm][key]=dict(donor_gain=direction*(a[key]-c[key]),
                donor_vs_original_preference_shift=direction*((a[key]-b[key])-(c[key]-d[key]))) if all(v is not None for v in values) else None
    write(folder/'semantics.json',dict(id=row['id'],mask_valid=valid,mask_stats=stats,metrics=records,
        fft_generated=spectrum,fft_reference=ref_fft,reference_preference_effects=effects,
        limits=['CLIP is global appearance; not a pattern correctness oracle',
                'existing TPF patch compares chunk RGB means and is color sensitive',
                'FFT measures repeated direction/frequency, not motif identity; confidence>=.15 for axial angle',
                'no GT used; sketch-only masks; invalid masks retain NA']))


def run():
    from models.e33tm_generation_wrapper import load_e5
    p,modules,size,ns=load_e5()
    for row in read(OUT/'pairs.json'):
        folder=OUT/'cases'/row['id']
        if not (folder/'semantics.json').exists():evaluate(p,row,folder)
        print('SEMANTICS',row['id'],flush=True)


if __name__=='__main__':run()
