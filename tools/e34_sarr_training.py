"""原始 BF 在线退化与固定五项损失；训练输入和 GT 监督接口分开。"""
import hashlib
import math
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from models.e34_sarr_refiner import safe_mask

def load_case(row):
    gt=np.asarray(Image.open(row['gt']).convert('RGB').resize((384,512),Image.Resampling.BILINEAR),np.float32)/255
    sketch=Image.open(row['sketch']).convert('RGB')
    _,mask,line,info=safe_mask(sketch)
    reference=np.asarray(Image.open(row['reference']).convert('RGB').resize((128,128),Image.Resampling.BILINEAR),np.float32)/255
    return gt,mask,line,reference,info

def degrade(gt,rng):
    u=rng.random()
    if u<.45:
        sigma=float(rng.uniform(.8,2.4));return cv2.GaussianBlur(gt,(0,0),sigma),dict(kind='blur',sigma=sigma)
    if u<.75:
        scale=float(rng.uniform(.4,.75));h,w=gt.shape[:2]
        low=cv2.resize(gt,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_AREA)
        return np.clip(cv2.resize(low,(w,h),interpolation=cv2.INTER_CUBIC),0,1),dict(kind='resize',scale=scale)
    if u<.90:
        sigma=float(rng.uniform(.8,2.4));weight=float(rng.uniform(.5,.9))
        return weight*cv2.GaussianBlur(gt,(0,0),sigma)+(1-weight)*gt,dict(kind='lowpass_mix',sigma=sigma,weight=weight)
    return gt.copy(),dict(kind='identity')

def batch(rows,seed,device='cuda'):
    rng=np.random.default_rng(seed);data=[];infos=[]
    for row in rows:
        gt,mask,line,ref,info=load_case(row);blur,deg=degrade(gt,rng)
        image=gt+mask[:,:,None]*(blur-gt)
        data.append((image.transpose(2,0,1),gt.transpose(2,0,1),mask[None],line[None],ref.transpose(2,0,1)))
        infos.append(dict(id=row['id'],mask_valid=info['valid'],degradation=deg))
    return [torch.from_numpy(np.stack([r[k] for r in data])).to(device) for k in range(5)],infos

def masked(value,mask):
    return (value*mask).sum()/((mask.sum()*value.shape[1]).clamp_min(1e-8))

def blur(x,sigma=1.):
    radius=int(math.ceil(3*sigma));t=torch.arange(-radius,radius+1,device=x.device,dtype=x.dtype)
    kernel=torch.exp(-t.square()/(2*sigma*sigma));kernel=kernel/kernel.sum()
    channels=x.shape[1]
    x=F.conv2d(F.pad(x,(radius,radius,0,0),mode='reflect'),kernel.view(1,1,1,-1).expand(channels,1,1,-1),groups=channels)
    return F.conv2d(F.pad(x,(0,0,radius,radius),mode='reflect'),kernel.view(1,1,-1,1).expand(channels,1,-1,1),groups=channels)

def lap_loss(a,b,mask):
    result=a.new_tensor(0.)
    for _ in range(3):
        aa,bb=blur(a),blur(b)
        result=result+masked(((a-aa)-(b-bb)).abs(),mask)
        a,b=F.avg_pool2d(aa,2),F.avg_pool2d(bb,2);mask=F.avg_pool2d(mask,2)
    return result/3

def perceptual(a,b,mask,metric):
    import lpips
    aa=metric.net.forward(metric.scaling_layer(a*2-1));bb=metric.net.forward(metric.scaling_layer(b*2-1))
    result=a.new_tensor(0.)
    for k,(x,y) in enumerate(zip(aa,bb)):
        error=(lpips.normalize_tensor(x)-lpips.normalize_tensor(y)).square()
        value=metric.lins[k](error)
        m=F.interpolate(mask,size=value.shape[-2:],mode='area')
        result=result+masked(value,m)
    return result

def losses(pred,gt,image,mask,metric,detail=True):
    out=pred['output']
    values=dict(rec=masked(((out-gt).square()+1e-6).sqrt(),mask),
        lap=lap_loss(out,gt,mask),per=perceptual(out,gt,mask,metric),
        low=masked((blur(out,4)-blur(image,4)).abs(),mask),gate=masked(pred['confidence'],mask))
    total=values['rec']+(.5 if detail else 0)*values['lap']+.1*values['per']+.2*values['low']+.005*values['gate']
    return total,values

def dev_seed(sid):
    return int(hashlib.sha256(('E34/dev-degrade/42/'+sid).encode()).hexdigest()[:8],16)
