"""固定灰度高频统计代理；只作评测，绝不将 GT 特征交给细化器。"""
import hashlib
import cv2
import numpy as np
import torch
from tools.e34_sarr_protocol import CONFIG,bootstrap

def patch_boxes(mask,sid):
    h,w=mask.shape;boxes=[]
    for y in range(0,h-63,32):
        for x in range(0,w-63,32):
            if (mask[y:y+64,x:x+64]>.5).mean()>=.95:
                key=hashlib.sha256(('E34/patch/%s/%d/%d'%(sid,y,x)).encode()).hexdigest()
                boxes.append((key,[x,y,x+64,y+64]))
    return [b for _,b in sorted(boxes)[:16]]

def gray_high(rgb):
    g=cv2.cvtColor(rgb.astype(np.float32),cv2.COLOR_RGB2GRAY)
    return g-cv2.GaussianBlur(g,(0,0),4)

class Descriptor:
    def __init__(self,device='cpu'):
        from tools.e33tmoc_appearance_eval import frozen_lpips,model_hash
        self.lpips,self.info=frozen_lpips()
        self.hash=model_hash(self.lpips)
        self.lpips=self.lpips.to(device)
        self.device=device

    @torch.inference_mode()
    def extract(self,rgb,boxes):
        if not boxes:return None
        patches=np.stack([rgb[y0:y1,x0:x1] for x0,y0,x1,y1 in boxes])
        highs=np.stack([gray_high(p) for p in patches]);lap=[];fft=[]
        fy,fx=np.meshgrid(np.fft.fftfreq(64),np.fft.fftfreq(64),indexing='ij')
        radius=np.hypot(fx,fy);edges=np.linspace(1/64,.25,17)
        window=np.outer(np.hanning(64),np.hanning(64))
        for g in highs:
            blurred=[cv2.GaussianBlur(g,(0,0),s) for s in [.8,1.6,3.2]]
            lap.append([np.sqrt(np.mean(a*a)) for a in [g-blurred[0],blurred[0]-blurred[1],blurred[1]-blurred[2]]])
            power=abs(np.fft.fft2(g*window))**2
            bins=np.array([power[(radius>=a)&(radius<b)].mean() for a,b in zip(edges[:-1],edges[1:])])
            fft.append(bins/max(bins.sum(),1e-12))
        # 去低频和颜色后固定映射到 LPIPS [-1,1]，不为每个方法调整对比度。
        inp=torch.from_numpy(np.repeat(np.clip(highs[:,None]/.25,-1,1),3,axis=1)).float().to(self.device)
        outputs=self.lpips.net.forward(inp);features=[]
        for layer in outputs[:2]:
            v=layer[:,:32].flatten(2)
            gram=v@v.transpose(1,2)/v.shape[-1]
            i,j=torch.triu_indices(32,32,device=v.device)
            features.extend([gram[:,i,j],v.var(2,unbiased=False)])
        perceptual=torch.cat(features,1).mean(0).cpu().numpy()
        return dict(lap=np.mean(lap,0).tolist(),fft=np.mean(fft,0).tolist(),
            gram=perceptual.tolist(),patch_n=len(boxes),HF_energy=float(np.mean(highs**2)),
            laplacian_energy=float(np.mean([cv2.Laplacian(g,cv2.CV_32F).var() for g in highs])))

def components(a,b):
    if a is None or b is None:return None
    return [float(np.mean(np.abs(np.asarray(a[k])-np.asarray(b[k])))) for k in ['lap','fft','gram']]

def distance(a,b,scales):
    c=components(a,b)
    return None if c is None else float(np.dot(np.array(c)/np.array(scales),[.35,.35,.30]))

def distribution(values):
    a=np.array([v for v in values if v is not None],float)
    if not len(a):return dict(n=0,mean=None,quantiles=None)
    return dict(n=len(a),mean=float(a.mean()),quantiles=np.quantile(a,[0,.1,.25,.5,.75,.9,1]).tolist())
