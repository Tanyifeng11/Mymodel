"""Sketch 安全区与轻量 RGB 残差细化；GT 不属于前向接口。"""
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from garment_mask_utils import build_sketch_garment_mask, mask_backend_info

def safe_mask(sketch):
    assert mask_backend_info()['mask_backend']=='opencv', 'E34 禁止非 OpenCV 回退'
    width,height=384,512
    raw,info=build_sketch_garment_mask(sketch,width,height)
    m0=np.asarray(raw)>127
    gray=np.asarray(sketch.convert('L').resize((width,height),resample=2))
    line=gray<245
    protected=cv2.dilate(line.astype(np.uint8),np.ones((3,3),np.uint8))>0
    distance=cv2.distanceTransform(m0.astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    interior=(distance>17)&~protected
    feather=cv2.distanceTransform(interior.astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    safe=np.clip(feather/8,0,1).astype(np.float32)
    # 不可信凸包也只来自 sketch，但不作为本实验的可靠可编辑区。
    reliable=info['mask_source']=='sketch_flood_fill' and not info['mask_low_confidence']
    coverage=float(safe.sum()/safe.size)
    fraction=float(safe.sum()/max(m0.sum(),1))
    valid=bool(reliable and coverage>.03 and fraction>.20)
    if not valid: safe.fill(0)
    assert not safe[~m0].any() and not safe[distance<=17].any() and not safe[protected].any()
    return m0,safe,line.astype(np.float32),dict(info,valid=valid,
        proposed_image_coverage=coverage,proposed_garment_coverage=fraction,
        image_coverage=float(safe.mean()),garment_coverage=float(safe.sum()/max(m0.sum(),1)),
        gt_calls=0,erosion_distance_px=17)

def conv(a,b,stride=1):
    return nn.Conv2d(a,b,3,stride=stride,padding=1,padding_mode='reflect')

class Block(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.layers=nn.Sequential(nn.GroupNorm(8,c),nn.SiLU(),conv(c,c),
            nn.GroupNorm(8,c),nn.SiLU(),conv(c,c))
    def forward(self,x): return x+self.layers(x)

class ReferenceEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stages=nn.ModuleList([nn.Sequential(conv(a,b,s),nn.GroupNorm(8,b),nn.SiLU())
            for a,b,s in [(3,32,1),(32,64,2),(64,128,2)]])
        self.mlp=nn.Sequential(nn.Linear(448,128),nn.SiLU(),nn.Linear(128,128))
    def forward(self,x):
        stats=[]
        for layer in self.stages:
            x=layer(x); stats.extend([x.mean((2,3)),x.var((2,3),unbiased=False).add(1e-6).sqrt()])
        return self.mlp(torch.cat(stats,1))

class FiLM(nn.Module):
    def __init__(self,c):
        super().__init__(); self.affine=nn.Linear(128,2*c)
        nn.init.zeros_(self.affine.weight);nn.init.zeros_(self.affine.bias)
    def forward(self,x,q):
        gamma,beta=self.affine(q).chunk(2,1)
        return x*(1+gamma[:,:,None,None])+beta[:,:,None,None]

class SARR(nn.Module):
    def __init__(self,no_reference=False,anchored=True,confidence=True):
        super().__init__()
        self.no_reference=no_reference;self.anchored=anchored;self.use_confidence=confidence
        self.reference=ReferenceEncoder()
        self.first=nn.Sequential(conv(5,32),Block(32),Block(32))
        self.down1=nn.Sequential(conv(32,64,2),Block(64),Block(64))
        self.down2=nn.Sequential(conv(64,128,2),Block(128),Block(128))
        self.film1=FiLM(64);self.film2=FiLM(128)
        self.up1=nn.Sequential(conv(128+64,64),Block(64))
        self.up2=nn.Sequential(conv(64+32,32),Block(32))
        self.delta=conv(32,3);self.gate=conv(32,1)
        nn.init.zeros_(self.delta.weight);nn.init.zeros_(self.delta.bias)
    def forward(self,image,line,mask,reference,reference_off=False):
        source=image if self.no_reference else reference
        source=F.interpolate(source,size=(128,128),mode='bilinear',align_corners=False)
        q=self.reference(source)
        if reference_off:q=torch.zeros_like(q)
        a=self.first(torch.cat([image,line,mask],1))
        b=self.film1(self.down1(a),q);c=self.film2(self.down2(b),q)
        d=self.up1(torch.cat([F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        d=self.up2(torch.cat([F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1))
        delta=torch.tanh(self.delta(d)).float()
        confidence=torch.sigmoid(self.gate(d)).float() if self.use_confidence else torch.ones_like(mask)
        edit=mask if self.anchored else torch.ones_like(mask)
        output=(image.float()+.35*edit.float()*confidence*delta).clamp(0,1)
        return dict(output=output,delta=delta,confidence=confidence,q=q)
