"""原生正方形参考的解析干预；不拉伸长宽，不把补边当真实支持。"""
from io import BytesIO
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from models.local_pattern_field import patch_geometry
from models.pattern_geometry import axial_distance
from tools.e33_protocol import PROTOCOL


def transform(rgb, arm):
    n=len(rgb)
    assert rgb.shape==(n,n,3)
    if arm=='rot90':
        return np.rot90(rgb).copy(),np.ones((n,n),bool)
    s={'identity':1.,'scale_up':1.25,'scale_down':.80}[arm]
    yy,xx=np.indices((n,n),np.float32);c=(n-1)/2
    sx=(xx-c)/s+c;sy=(yy-c)/s+c
    valid=(sx>=0)&(sy>=0)&(sx<=n-1)&(sy<=n-1)
    pixels=cv2.remap(rgb,sx,sy,cv2.INTER_CUBIC,borderMode=cv2.BORDER_REFLECT_101)
    return pixels,valid


def nuisance(rgb,valid,seed):
    rng=np.random.default_rng(seed)
    im=ImageEnhance.Brightness(Image.fromarray(rgb)).enhance(rng.uniform(.97,1.03))
    im=ImageEnhance.Color(im).enhance(rng.uniform(.97,1.03))
    sigma=rng.uniform(0,.25)
    arr=cv2.GaussianBlur(np.asarray(im),(3,3),sigma) if sigma>.05 else np.asarray(im)
    out=BytesIO();Image.fromarray(arr).save(out,format='JPEG',quality=int(rng.integers(90,99)))
    arr=np.asarray(Image.open(BytesIO(out.getvalue())).convert('RGB'))
    dx,dy=rng.uniform(-.25,.25,2);n=len(arr)
    matrix=np.float32([[1,0,dx],[0,1,dy]])
    arr=cv2.warpAffine(arr,matrix,(n,n),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    # 保守支持：插值核与JPEG/blur边缘邻域不进入测量。
    support=cv2.erode(valid.astype(np.uint8),np.ones((5,5),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
    support=cv2.warpAffine(support,matrix,(n,n),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)>0
    return arr,support


def central_valid_box(support):
    n=len(support);center=(n-1)/2
    for width in range(n,15,-1):
        lo=int(np.floor(center-(width-1)/2));hi=lo+width
        if lo>=0 and hi<=n and support[lo:hi,lo:hi].all():return lo,hi
    return None


def audit_intervention(rgb,arm,seed=None):
    changed,support=transform(rgb,arm)
    base=np.asarray(rgb)
    if seed is not None:
        changed,support=nuisance(changed,support,seed)
        base,_=nuisance(base,np.ones(support.shape,bool),seed)
    box=central_valid_box(support)
    if box is None:return dict(valid=False,reason='no valid support')
    lo,hi=box
    # 相同native像素窗口，排除缩放补边；冻结同一个E26估计器。
    before=patch_geometry(Image.fromarray(base[lo:hi,lo:hi]))
    after=patch_geometry(Image.fromarray(changed[lo:hi,lo:hi]))
    shift=90 if arm=='rot90' else 0
    s={'rot90':1.,'scale_up':1.25,'scale_down':.80}[arm]
    ori=axial_distance(after['orientation'],before['orientation']+shift)
    delta=float(np.log2(max(after['frequency'],1e-6)/max(before['frequency'],1e-6)))
    per=abs(delta+np.log2(s));cfg=PROTOCOL['integrity']
    valid=before['confidence']>=cfg['estimator_confidence_min'] and after['confidence']>=cfg['estimator_confidence_min'] and ori<=15 and per<=.10
    return dict(valid=bool(valid),orientation_response_error_deg=float(ori),period_response_error_log2=float(per),
                period_response_log2=delta,expected_log2=float(-np.log2(s)),before=before,after=after,
                support_box=[lo,lo,hi,hi],support_fraction=float(support.mean()))
