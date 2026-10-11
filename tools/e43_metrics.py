"""真实GT对齐LPIPS及既有结构指标，阈值只用固定分数量纲。"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@torch.inference_mode()
def cloth_lpips(metric,a,b,mask):
    import lpips
    aa=metric.net(metric.scaling_layer(a));bb=metric.net(metric.scaling_layer(b));total=0.
    for k,(x,y) in enumerate(zip(aa,bb)):
        v=metric.lins[k]((lpips.normalize_tensor(x)-lpips.normalize_tensor(y)).square())
        m=F.interpolate(mask.to(v),size=v.shape[-2:],mode='area')
        total+=float((v*m).sum()/m.sum().clamp_min(1e-12))
    return total


def masks(sketch):
    from eval.eval_utils import prepare_evaluation_masks
    from tools.e39_protocol import SIZE
    bundle=prepare_evaluation_masks(SIZE,sketch_path=sketch,mask_policy='sketch_only');s=bundle['stats']
    garment=bundle['garment'].astype(bool) if bundle['garment'] is not None else np.zeros((512,384),bool)
    # 原E5仍接受自动草图mask；测量失败的样本保留并精确回退alpha1。
    bundle['garment']=garment
    valid=s.get('mask_source')=='sketch_flood_fill' and not s.get('mask_low_confidence',True)
    dist=cv2.distanceTransform(garment.astype('uint8'),cv2.DIST_L2,5)
    interior=(dist>9).astype('float32');soft=np.clip((dist-9)/8,0,1).astype('float32')
    valid=bool(valid and interior.mean()>=.03)
    latent=cv2.resize(soft,(48,64),interpolation=cv2.INTER_AREA)
    boundary=cv2.morphologyEx(garment.astype('uint8'),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))
    return bundle,dict(valid=valid,stats=s,area=float(garment.mean()),boundary=float(boundary.mean()),
        confidence=float(s.get('mask_confidence',0)),latent=latent)


def metric_record(path,row,metric):
    from eval.metrics import compute_structure_preservation,compute_texture_leakage
    from eval.eval_utils import json_safe
    from tools.e39_protocol import SIZE
    a=Image.open(path).convert('RGB');b=Image.open(row['gt']).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR)
    bundle,mask=masks(row['sketch']);result=dict(mask_valid=mask['valid'],LPIPS_cloth=None,HP_gray=None,LAB=None,
        IoU=None,EdgeF1=None,Leak=None)
    struct=json_safe(compute_structure_preservation(str(path),row['sketch'],mask_bundle=bundle))
    result['EdgeF1']=struct.get('struct_edge_f1')
    if mask['valid']:
        device=next(metric.parameters()).device
        tensor=lambda im:torch.from_numpy(np.asarray(im,np.float32).transpose(2,0,1)/127.5-1)[None].to(device)
        m=bundle['garment'];mt=torch.from_numpy(m.astype('float32'))[None,None].to(device)
        result['LPIPS_cloth']=cloth_lpips(metric,tensor(a),tensor(b),mt)
        ga=np.asarray(a.convert('L'),np.float32)/255;gb=np.asarray(b.convert('L'),np.float32)/255
        hp=lambda x:x-cv2.GaussianBlur(x,(0,0),3)
        result['HP_gray']=float(np.abs(hp(ga)-hp(gb))[m].mean())
        lab=lambda im:cv2.cvtColor(np.asarray(im,np.float32)/255,cv2.COLOR_RGB2LAB)[m].mean(0)
        result['LAB']=float(np.linalg.norm(lab(a)-lab(b)))
        result['IoU']=struct.get('struct_iou')
        result['Leak']=json_safe(compute_texture_leakage(str(path),target_path=row['gt'],mask_bundle=bundle)).get('leak_colored_frac')
    return result


def safe_gain(base,other):
    if any(base[k] is None or other[k] is None for k in ['LPIPS_cloth','IoU','EdgeF1','Leak']):return None
    return dict(gain=base['LPIPS_cloth']-other['LPIPS_cloth'],
        utility=base['LPIPS_cloth']-other['LPIPS_cloth']-2*max(0,base['IoU']-other['IoU']-.005)
            -.5*max(0,base['EdgeF1']-other['EdgeF1']-.01)-4*max(0,other['Leak']-base['Leak']-.002),
        safe=other['IoU']-base['IoU']>=-.005 and other['EdgeF1']-base['EdgeF1']>=-.01 and other['Leak']-base['Leak']<=.002,
        noncolor=other['HP_gray']<base['HP_gray'],
        catastrophic=other['IoU']-base['IoU']<-.15 or other['EdgeF1']-base['EdgeF1']<-.15 or other['Leak']-base['Leak']>.10)
