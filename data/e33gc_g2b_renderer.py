"""真实参考裁片是 reference 与 target 的唯一纹样来源；GT 仅供低通监督。"""
import cv2
import numpy as np
from PIL import Image
from data.e32_target_pseudogt import image_at
from data.e33gc_renderer import support
from tools.e33gc_g2b_protocol import CONFIG, DATASET, OUT, ARMS, sha, read

def input_dir(row):
    return OUT/('G1a_synthetic_reference_control' if row.get('route')=='B' else 'G1a_controlled_targets')/row['id']

def construct(row, targets=True):
    sketch=image_at(DATASET/row['sketch']); source=image_at(DATASET/row['reference'])
    mask,inner,falloff=support(sketch)
    crop=source.crop(CONFIG['crop']); patch=np.asarray(crop)
    if row.get('route')=='B':
        import hashlib
        seed=int(hashlib.sha256(('E33GC-G2b/synthetic/'+row['id']).encode()).hexdigest()[:16],16)
        rng=np.random.default_rng(seed);yy,xx=np.mgrid[:128,:128]
        period=int(rng.choice([16,32]));axis=int(rng.choice([0,1]));phase=float(rng.uniform(0,2*np.pi))
        color=rng.uniform(65,190,3);contrast=float(rng.uniform(35,55))
        wave=np.cos(2*np.pi*(xx if axis==0 else yy)/period+phase)
        patch=np.uint8(np.clip(np.round(color+contrast*wave[...,None]),0,255));crop=Image.fromarray(patch)
    y,x=np.mgrid[:512,:384]
    # 固定相位，以中心128裁片原生像素为局部纹样坐标；三臂不随机重采样。
    tiled=patch[(y-192)%128,(x-128)%128]
    refs=[]; patterns=[]
    source_mask=mask.copy()  # 新受控参考：同一 sketch 衣身形状、背景，只有纹样坐标旋转。
    for angle in [0,90,180]:
        p=np.rot90(patch,angle//90).copy(); pattern=p[(y-192)%128,(x-128)%128]
        ref=np.full_like(tiled,255); ref[source_mask>0]=pattern[source_mask>0]
        # RF2 固定中心裁片必须可见，受控参考以固定中心裁片嵌入，非 GT 信息。
        ref[192:320,128:256]=p
        refs.append(Image.fromarray(ref)); patterns.append(pattern.astype(np.float32)/255)
    result=dict(references=refs,sketch=sketch,mask=mask,inner=inner,falloff=falloff,source_crop=crop)
    if targets:
        gt=np.asarray(image_at(DATASET/row['target']),np.float32)/255
        low=cv2.GaussianBlur(gt,(0,0),16); gray=low.mean(-1)
        scale=np.clip(gray/max(float(gray[inner].mean()),.1),.85,1.15)
        result['targets']=[Image.fromarray(np.uint8(np.clip(np.round(255*(
            low*(1-falloff[...,None])+np.clip(p*scale[...,None],0,1)*falloff[...,None])),0,255))) for p in patterns]
    return result

def load(row,targets=False):
    # 模型前向只读预冻结 reference/sketch/mask；独立训练目标显式传 targets=True 才加载。
    d=input_dir(row)
    sketch=Image.open(d/'sketch.png').convert('RGB'); mask,inner,falloff=support(sketch)
    result=dict(sketch=sketch,mask=mask,inner=inner,falloff=falloff,
        references=[Image.open(d/(a+'_reference.png')).convert('RGB') for a in ARMS])
    if targets: result['targets']=[Image.open(d/(a+'_target.png')).convert('RGB') for a in ARMS]
    return result

