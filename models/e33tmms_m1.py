"""固定三轮方向约束PatchMatch；真实exemplar，四邻域连续性。"""
import hashlib
from functools import lru_cache
import cv2
import numpy as np
from PIL import Image
from garment_mask_utils import estimate_cloth_foreground_mask
from models.local_pattern_field import patch_geometry
from tools.e33tmms_protocol import PROTOCOL

CONFIG=PROTOCOL['M1']

def tensor_direction(rgb):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    # 梯度法向转换为纹理切向；只用内部，避免裁片边缘。
    a=float((gx*gx-gy*gy)[4:-4,4:-4].mean());b=float((2*gx*gy)[4:-4,4:-4].mean())
    return (np.arctan2(b,a)/2+np.pi/2)%np.pi,float(np.hypot(gx,gy).mean())

def bank_for(reference):
    rgb=np.asarray(reference);fg=np.asarray(estimate_cloth_foreground_mask(reference,*reference.size)[0])>127
    values=rgb[fg] if fg.any() else rgb.reshape(-1,3)
    fallback=values.mean(0).astype(np.float32);bank=[];valid=np.zeros(fg.shape,bool)
    source_lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB).astype(np.float32)/255
    proto=np.concatenate([source_lab[fg].mean(0),source_lab[fg].std(0)]) if fg.any() else np.zeros(6)
    _,gradient=tensor_direction(rgb)
    for y in range(0,reference.height-63,16):
        for x in range(0,reference.width-63,16):
            patch=reference.crop((x,y,x+64,y+64));ratio=float(fg[y:y+64,x:x+64].mean())
            if ratio<.95:continue
            variance=float(np.asarray(patch.convert('L'),float).var()/255**2)
            if variance<.0003:continue
            info=patch_geometry(patch)
            if info['confidence']<.20:continue
            # 64px是正式source patch；128px真实上下文仅防旋转裁切，不加入新源样本。
            xx,yy=np.meshgrid(np.arange(x-32,x+96,dtype=np.float32),np.arange(y-32,y+96,dtype=np.float32))
            halo=cv2.remap(rgb,xx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=tuple(map(float,fallback)))
            support=cv2.remap(fg.astype(np.uint8),xx,yy,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
            rotation=cv2.getRotationMatrix2D((63.5,63.5),info['orientation'],1.)
            canonical=cv2.warpAffine(halo,rotation,(128,128),flags=cv2.INTER_LINEAR,borderValue=tuple(map(float,fallback)))[16:112,16:112]
            canonical_support=cv2.warpAffine(support,rotation,(128,128),flags=cv2.INTER_NEAREST)[16:112,16:112]
            patch_rgb=np.asarray(patch)
            gray=cv2.cvtColor(patch_rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
            grad=np.hypot(cv2.Sobel(gray,cv2.CV_32F,1,0),cv2.Sobel(gray,cv2.CV_32F,0,1))
            bank.append(dict(x=x+31.5,y=y+31.5,box=[x,y,x+64,y+64],orientation=info['orientation'],
                confidence=info['confidence'],foreground_ratio=ratio,variance=variance,
                rgb=patch_rgb,Lab=cv2.cvtColor(patch_rgb,cv2.COLOR_RGB2LAB),gradient=grad,
                canonical=canonical,canonical_support=canonical_support))
            valid[y:y+64,x:x+64]=True
    return bank,fg,valid,fallback,proto,gradient

def construct(reference,orientation,confidence,mask,sid):
    bank,fg,source_valid,fallback,proto,gradient=bank_for(reference)
    width,height=mask.size;inside=np.asarray(mask)>0
    dense=cv2.resize(np.moveaxis(orientation,0,-1),(width,height),interpolation=cv2.INTER_LINEAR)
    theta=np.arctan2(dense[...,1],dense[...,0])/2
    ys=list(range(-32,height,16));xs=list(range(-32,width,16))
    target_theta=np.array([[theta[np.clip(y+32,0,height-1),np.clip(x+32,0,width-1)] for x in xs] for y in ys])
    rng=np.random.default_rng(int(hashlib.sha256(('MS/M1/'+sid).encode()).hexdigest()[:16],16))
    labels=rng.integers(0,max(len(bank),1),target_theta.shape);history=[]
    xy=np.array([[p['x'],p['y']] for p in bank],np.float32)

    @lru_cache(maxsize=1024)
    def rendered(index,angle):
        p=bank[index];m=cv2.getRotationMatrix2D((47.5,47.5),-np.degrees(angle),1.)
        pixels=cv2.warpAffine(p['canonical'],m,(96,96),flags=cv2.INTER_LINEAR,borderValue=tuple(map(float,fallback)))[16:80,16:80].copy()
        support=cv2.warpAffine(p['canonical_support'],m,(96,96),flags=cv2.INTER_NEAREST)[16:80,16:80]>0
        pixels[~support]=np.uint8(np.round(fallback))
        lab=cv2.cvtColor(pixels,cv2.COLOR_RGB2LAB).astype(np.float32)/255
        measured,mag=tensor_direction(pixels)
        return pixels,cv2.resize(lab,(16,16)),np.concatenate([lab.mean((0,1)),lab.std((0,1))]),measured,mag,support

    def neighbors(i,j):
        return [(ni,nj) for ni,nj in ((i,j-1),(i-1,j),(i,j+1),(i+1,j)) if 0<=ni<len(ys) and 0<=nj<len(xs)]

    def expected(index,delta,angle):
        phi=np.deg2rad(bank[index]['orientation'])-angle;c,s=np.cos(phi),np.sin(phi)
        return xy[index]+np.array([c*delta[0]-s*delta[1],s*delta[0]+c*delta[1]])

    def energy(index,i,j,neis):
        angle=round(float(target_theta[i,j]),5);pixels,lab,stats,measured,mag,support=rendered(index,angle)
        app=float(np.abs(stats-proto).mean()+abs(mag-gradient)/max(1.,gradient))
        ori=float(1-np.cos(2*(measured-angle)))
        nbr=[];seam=[]
        for ni,nj in neis:
            other=int(labels[ni,nj]);delta=np.array([xs[j]-xs[nj],ys[i]-ys[ni]])
            proposal=expected(other,delta,float(target_theta[ni,nj]))
            nbr.append(min(2.,float(np.linalg.norm(xy[index]-proposal)/64)))
            old=rendered(other,round(float(target_theta[ni,nj]),5))[1]
            dx,dy=(int(delta[0]/4),int(delta[1]/4))
            x0,y0=max(0,-dx),max(0,-dy);x1,y1=min(16,16-dx),min(16,16-dy)
            seam.append(float(np.abs(lab[y0:y1,x0:x1]-old[y0+dy:y1+dy,x0+dx:x1+dx]).mean()))
        return app+.5*(np.mean(nbr) if nbr else 0)+ori+.25*(np.mean(seam) if seam else 0)

    if bank:
        for iteration in range(3):
            total=0.;sequence=[(i,j) for i in range(len(ys)) for j in range(len(xs))]
            if iteration%2:sequence.reverse()
            for i,j in sequence:
                neis=neighbors(i,j);current=int(labels[i,j]);proposals=[current]
                for ni,nj in neis:
                    other=int(labels[ni,nj]);delta=np.array([xs[j]-xs[nj],ys[i]-ys[ni]])
                    point=expected(other,delta,float(target_theta[ni,nj]))
                    proposals.append(int(np.argmin(((xy-point)**2).sum(1))))
                    proposals.append(other)
                best=current;score=energy(best,i,j,neis)
                for index in dict.fromkeys(proposals):
                    value=energy(index,i,j,neis)
                    if value<score:best,score=index,value
                radius=float(max(reference.size))
                while radius>=16:
                    point=xy[best]+rng.uniform(-radius,radius,2)
                    index=int(np.argmin(((xy-point)**2).sum(1)));value=energy(index,i,j,neis)
                    if value<score:best,score=index,value
                    radius*=.5
                labels[i,j]=best;total+=score
            history.append(dict(iteration=iteration+1,mean_energy=float(total/len(sequence))))
    accum=np.zeros((height,width,3),np.float32);norm=np.zeros((height,width),np.float32)
    valid_acc=np.zeros_like(norm);window=np.outer(np.hanning(64),np.hanning(64)).astype(np.float32);selected=[]
    if bank:
        for i,y in enumerate(ys):
            for j,x in enumerate(xs):
                index=int(labels[i,j]);patch,_,_,_,_,support=rendered(index,round(float(target_theta[i,j]),5))
                x0,y0=max(0,x),max(0,y);x1,y1=min(width,x+64),min(height,y+64)
                w=window[y0-y:y1-y,x0-x:x1-x];piece=patch[y0-y:y1-y,x0-x:x1-x]
                accum[y0:y1,x0:x1]+=piece*w[...,None];norm[y0:y1,x0:x1]+=w
                valid_acc[y0:y1,x0:x1]+=support[y0-y:y1-y,x0-x:x1-x]*w
                selected.append(dict(target_box=[x,y,x+64,y+64],source_index=index,
                    source_xy=xy[index].tolist(),target_tangent_degrees=float(np.degrees(target_theta[i,j])%180)))
    raw=np.where((norm>1e-8)[...,None],accum/np.maximum(norm[...,None],1e-8),fallback)
    S1=Image.fromarray(np.uint8(np.clip(np.round(raw),0,255)));pixels=np.asarray(S1).copy();pixels[~inside]=255
    state=dict(source_valid_support=source_valid,source_foreground=fg,target_support=inside,
        RF_orientation=orientation,RF_confidence=confidence,target_theta=target_theta,
        selected_indices=labels,source_valid_blend=valid_acc/np.maximum(norm,1e-8))
    metadata=dict(config=CONFIG,empty_bank=not bool(bank),bank_size=len(bank),foreground_mean_RGB=fallback.tolist(),
        bank=[{k:v for k,v in p.items() if k not in ('rgb','Lab','gradient','canonical','canonical_support')} for p in bank],
        selected_patches=selected,iterations=history)
    return S1,Image.fromarray(pixels),state,metadata,bank
