"""目标坐标下的候选分区；缓存误差图允许后续人工修mask而无需重跑模型。"""
import numpy as np
from PIL import Image


def partition(mask, latent_hw):
    import torch
    import torch.nn.functional as F
    a=torch.tensor(np.asarray(mask,dtype=np.float32)/255)[None,None]
    foreground=F.interpolate(a,size=latent_hw,mode='area')>=.5
    # 一格latent（约8像素）内缩/外扩；边界单独统计，三部分完整互斥。
    grow=F.max_pool2d(foreground.float(),3,1,1)>0
    shrink=1-F.max_pool2d(F.pad(1-foreground.float(),(1,1,1,1),value=1),3,1)
    inside=shrink[0,0].numpy().astype(bool)
    background=(~grow)[0,0].numpy()
    return dict(interior=inside,boundary=~(inside|background),background=background)


def prepare(row, data_root, out, sample_index, latent_hw):
    from pathlib import Path
    from garment_mask_utils import build_sketch_garment_mask, estimate_cloth_foreground_mask
    base=Path(data_root)
    with Image.open(base/row['cloth']) as im:cloth=im.convert('RGB').resize((384,512),Image.BILINEAR)
    sketch=base/row.get('sketch','__missing__')
    if sketch.is_file():
        with Image.open(sketch) as im:mask,info=build_sketch_garment_mask(im,384,512)
    else:
        mask,info=estimate_cloth_foreground_mask(cloth,384,512)
    regions=partition(mask,latent_hw)
    folder=Path(out)/'regions';folder.mkdir(exist_ok=True)
    mask.save(folder/('%05d_mask.png'%sample_index))
    np.savez_compressed(folder/('%05d_masks.npz'%sample_index),**regions)
    overlay=np.asarray(cloth).astype(float)
    for name,color in [('interior',[0,180,0]),('boundary',[255,180,0]),('background',[0,80,255])]:
        m=np.asarray(Image.fromarray(regions[name]).resize(cloth.size,Image.NEAREST)).astype(bool)
        overlay[m]=overlay[m]*.65+np.array(color)*.35
    Image.fromarray(overlay.astype('uint8')).save(folder/('%05d_overlay.png'%sample_index))
    info['latent_region_pixels']={k:int(v.sum()) for k,v in regions.items()}
    info['manual_review']='pending'
    return regions,info


def regional_summary(records):
    from tools.e14_matched_denoising import summary
    result={}
    for region in ['interior','boundary','background']:
        valid=[dict(r,losses=r['region_losses'][region]) for r in records
               if all(v is not None for v in r['region_losses'][region].values())]
        result[region]=summary(valid) if valid else None
    return result
