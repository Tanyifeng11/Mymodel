"""真实裁片的空间Lab目标与损失；低频亮度误差相对完整面料目标计算。"""
import cv2
import numpy as np
import torch
from torch.nn import functional as F

from models.e2_lab_anchor import rgb_to_lab
from tools.e2_b_repair_losses import lab_to_rgb


def feather(mask):
    return np.clip((cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,5)-8)/8,0,1)


def material_target(anchor,reference,mask):
    """真实裁片平面映射加弱照明场；不是实拍的反事实换布真值。"""
    rgb=np.asarray(anchor,np.float32)/255
    ref=np.asarray(reference,np.float32)/255
    source=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)
    ref_tensor=torch.from_numpy(ref.transpose(2,0,1).copy())[None].double()
    lab=rgb_to_lab(ref_tensor)[0].permute(1,2,0).numpy()
    weight=feather(mask);inside=mask.astype(np.float32)
    smooth=cv2.GaussianBlur(source[...,0]*inside,(0,0),64)/cv2.GaussianBlur(inside,(0,0),64).clip(.001)
    shade=np.clip(smooth-(smooth*weight).sum()/weight.sum(),-5,5)
    # 每像素只缩小越出RGB色域的L偏移，空间a、b始终来自裁片。
    desired=lab.copy();desired[...,0]+=shade
    trial=lab_to_rgb(desired)
    valid=((trial>=-1e-5)&(trial<=1.00001)).all(-1)
    lower=np.where(valid,1.,0.);upper=np.ones_like(lower)
    for _ in range(12):
        middle=(lower+upper)/2;trial_lab=lab.copy();trial_lab[...,0]+=middle*shade
        trial=lab_to_rgb(trial_lab);ok=((trial>=-1e-5)&(trial<=1.00001)).all(-1)
        lower=np.where(ok,middle,lower);upper=np.where(ok,upper,middle)
    desired[...,0]=lab[...,0]+lower*shade
    material=lab_to_rgb(desired)
    assert material[weight>0].min()>=-1e-5 and material[weight>0].max()<=1.00001
    result=rgb*(1-weight[...,None])+material.clip(0,1)*weight[...,None]
    image=np.rint(result*255).astype(np.uint8)
    output_lab=cv2.cvtColor(image.astype(np.float32)/255,cv2.COLOR_RGB2LAB)
    interior=weight>=.999
    reference_lab=cv2.cvtColor(ref,cv2.COLOR_RGB2LAB)
    audit=dict(shade_sigma=64.,shade_limit=5.,constant_ab=False,
        chroma_quantization_error=float(np.linalg.norm(output_lab[interior,1:]-reference_lab[interior,1:],axis=1).mean()),
        gamut_restricted_fraction=float((lower[interior]<.999).mean()),
        outside_unchanged=bool(np.array_equal(image[~mask],np.asarray(anchor)[~mask])))
    return image,audit


def highpass(value):
    return value-F.avg_pool2d(F.pad(value,(8,8,8,8),mode='reflect'),17,1)


def weighted(value,weight):
    return (value*weight).sum()/(weight.sum()*value.shape[1]).clamp_min(1e-8)


def loss_terms(rgb,goal,base,weight,old_template,coords):
    current,target=rgb_to_lab(rgb)/100,rgb_to_lab(goal)/100
    error=current-target
    high=highpass(error)
    patches=torch.stack([high[0,:,y:y+64,x:x+64] for x,y in coords])
    energy=old_template.square().mean((1,2,3))
    valid=energy>2.5e-5
    projection=(patches*old_template).mean((1,2,3))/energy.clamp_min(1e-8)
    old=(projection.abs()*valid).sum()/valid.sum().clamp_min(1)
    positions=(weight[0,0]>=.999).flatten().nonzero().flatten()
    if len(positions)>2048:
        positions=positions[torch.linspace(0,len(positions)-1,2048,device=positions.device).round().long()]
    angle=torch.arange(16,device=rgb.device,dtype=torch.float32)*torch.pi/16
    direction=torch.stack((angle.cos(),angle.sin()))
    samples=lambda value:value[0,1:].reshape(2,-1)[:,positions].T@direction
    palette=(samples(current).sort(0).values-samples(target).sort(0).values).abs().mean()
    # 减去完整目标后再做低通；目标的大圆点/宽色带不会被当成亮度漂移。
    low=F.avg_pool2d(error[:,:1]*weight,64,16)/F.avg_pool2d(weight,64,16).clamp_min(1e-6)
    low_weight=F.avg_pool2d(weight,64,16)
    return dict(rgb=weighted((rgb-goal).abs(),weight),pattern=weighted(high.abs(),weight),
        chroma=weighted(error[:,1:].abs(),weight),palette=palette,
        light=weighted(low.abs(),low_weight),old=old,protection=weighted((rgb-base).abs(),1-weight))


def old_template(anchor,positive,coords):
    to_lab=lambda image:rgb_to_lab(image)/100
    new=highpass(to_lab(positive))
    old=highpass(to_lab(anchor))-new
    patches=lambda field:torch.stack([field[0,:,y:y+64,x:x+64] for x,y in coords])
    old,new=patches(old),patches(new)
    shared=(old*new).sum((1,2,3),keepdim=True)/new.square().sum((1,2,3),keepdim=True).clamp_min(1e-8)
    return (old-shared*new).detach()
