"""与E5/RF2/可微结构张量独立的解析纹样rasterizer；GT仅供低频明暗监督。"""
import hashlib
import cv2
import numpy as np
from PIL import Image
from data.e32_target_pseudogt import image_at,masks
from data.e33rf_real_rotation_dataset import rotate
from tools.e33gc_protocol import CONFIG,DATASET

def support(sketch):
    mask=masks(sketch)[0].astype(np.uint8)
    inner=cv2.erode(mask,np.ones((17,17),np.uint8))
    falloff=np.clip(cv2.distanceTransform(inner,cv2.DIST_L2,5)/8,0,1)
    return mask,inner.astype(bool),falloff.astype(np.float32)

def parameters(sid):
    seed=int(hashlib.sha256((CONFIG['renderer']['seed_prefix']+sid).encode()).hexdigest()[:16],16)
    rng=np.random.default_rng(seed)
    return dict(seed=seed,period=float(rng.uniform(16,48)),angle=float(rng.choice(CONFIG['renderer']['angle_degrees'])),
        phase=float(rng.uniform(0,2*np.pi)),color=rng.uniform(40,215,3).tolist(),contrast=float(rng.uniform(35,75)),
        waveform=str(rng.choice(CONFIG['renderer']['waveform'])))

def raster(meta,arm):
    y,x=np.mgrid[:512,:384];x=x-191.5;y=y-255.5
    theta=np.deg2rad(meta['angle']-arm)
    # 纹样切向theta；normal=(-sin(theta),cos(theta))。图像y向下，np.rot90是视觉逆时针。
    normal=-x*np.sin(theta)+y*np.cos(theta);along=x*np.cos(theta)+y*np.sin(theta)
    phase=2*np.pi*normal/meta['period']+meta['phase'];wave=np.cos(phase)
    if meta['waveform']=='tanh_cosine':wave=np.tanh(2.5*wave)
    if meta['waveform']=='elongated_dashes':wave=wave*(.65+.35*np.cos(2*np.pi*along/(4*meta['period'])))
    rgb=np.clip(np.asarray(meta['color'])[None,None]+meta['contrast']*wave[...,None],0,255)
    return rgb.astype(np.float32)

def construct(row):
    sid=row['id'];meta=parameters(sid);sketch=image_at(DATASET/row['sketch'])
    mask,inner,falloff=support(sketch)
    target=np.asarray(image_at(DATASET/row['target']),np.float32)/255
    low=cv2.GaussianBlur(target,(0,0),16)
    gray=low.mean(-1);scale=np.clip(gray/max(float(gray[inner].mean()) if inner.any() else .5,.1),.85,1.15)
    references=[];targets=[];source=Image.fromarray(np.uint8(np.round(raster(meta,0))))
    for arm in [0,90,180]:
        references.append(source if arm==0 else rotate(source,arm))
        pattern=np.clip(raster(meta,arm)/255*scale[...,None],0,1)
        # 背景也只有lowpass target，无GT高频纹样。GT从不进入reference/field/adapter。
        rgb=low*(1-falloff[...,None])+pattern*falloff[...,None]
        targets.append(Image.fromarray(np.uint8(np.clip(np.round(rgb*255),0,255))))
    meta.update(id=sid,analytic_tangent_degrees={str(arm):float((meta['angle']-arm)%180) for arm in [0,90,180]},
        mask_pixels=int(mask.sum()),inner_pixels=int(inner.sum()),target_pattern_coverage=float((falloff>=.99).sum()/max(inner.sum(),1)))
    return dict(id=sid,references=references,targets=targets,sketch=sketch,mask=mask,inner=inner,falloff=falloff,meta=meta)

def fft_tangent(image,box=(128,192,256,320)):
    # 独立频域读出，不使用训练结构张量或RF2标签；矩形DFT，Hann减弱边界频率。
    gray=np.asarray(image.crop(box).convert('L'),float);h,w=gray.shape
    spectrum=np.abs(np.fft.fftshift(np.fft.fft2((gray-gray.mean())*np.outer(np.hanning(h),np.hanning(w)))))**2
    fy,fx=np.mgrid[:h,:w];fy=(fy-h//2)/h;fx=(fx-w//2)/w
    spectrum[np.hypot(fx,fy)<1/64]=0
    iy,ix=np.unravel_index(spectrum.argmax(),spectrum.shape)
    tangent=(np.degrees(np.arctan2(fy[iy,ix],fx[iy,ix]))+90)%180
    return float(tangent)
