"""固定的可微 RGB 代理。GT 只用于监督；区域一律从草图获取。"""
import math
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from tools.e37_protocol import SIZE,CONFIG


def tensor_image(path,device='cuda',bicubic=False):
    im=Image.open(path).convert('RGB').resize(SIZE,Image.Resampling.BICUBIC if bicubic else Image.Resampling.BILINEAR)
    return torch.from_numpy(np.array(im,dtype=np.float32).transpose(2,0,1)/255.)[None].to(device)


def masks(row,device='cuda'):
    from eval.eval_utils import prepare_evaluation_masks
    from tools.e35_freeu_run import digest
    bundle=prepare_evaluation_masks(SIZE,sketch_path=row['sketch'],mask_policy='sketch_only')
    stats=bundle['stats'];valid=stats['mask_source']=='sketch_flood_fill' and not stats.get('mask_low_confidence',True)
    m=bundle['garment'].astype(np.uint8)
    kernel=np.ones((7,7),np.uint8)
    inner=cv2.erode(m,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)
    dilated=cv2.dilate(m,kernel)
    fields=dict(garment=m,inner=inner,band=dilated-inner,outside=1-dilated)
    result={k:torch.from_numpy(v.astype(np.float32))[None,None].to(device) for k,v in fields.items()}
    valid=valid and all(int(result[k].sum())>0 for k in ['inner','band','outside'])
    audit=dict(id=row['id'],valid=bool(valid),stats=stats,pixels={k:int(v.sum()) for k,v in result.items()},
        hashes={k:digest(v) for k,v in result.items()},gt_fallback=False)
    return result,audit


def mean_roi(x,m):return (x*m).sum((-2,-1))/m.sum((-2,-1))


def blur(x,sigma,size):
    a=torch.arange(size,device=x.device,dtype=x.dtype)-(size-1)/2
    k=torch.exp(-a.square()/(2*sigma*sigma));k=k/k.sum();k=k[:,None]*k[None,:]
    return F.conv2d(F.pad(x,(size//2,)*4,mode='replicate'),k[None,None].expand(x.shape[1],1,size,size),groups=x.shape[1])


def sobel(x):
    kernels=x.new_tensor([[[-1,0,1],[-2,0,2],[-1,0,1]],[[-1,-2,-1],[0,0,0],[1,2,1]]])/8
    b,c,h,w=x.shape
    out=F.conv2d(F.pad(x.reshape(b*c,1,h,w),(1,1,1,1),mode='replicate'),kernels[:,None])
    return (out.square().sum(1)+1e-12).sqrt().reshape(b,c,h,w)


def edge(x):return sobel(blur((x*x.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True),1.,5))


def features(x,sigma,size):
    sm=blur(x,sigma,size)
    gray=(sm*sm.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
    return torch.cat([sm,sobel(sm),sobel(gray)],1)


def gram(x,m):
    b,c,h,w=x.shape;f=(x*m.sqrt()).reshape(b,c,-1)
    return torch.bmm(f,f.transpose(1,2))/(m.sum((-2,-1))[:,:,None]*c)


def moments(x,m):
    mu=mean_roi(x,m);var=mean_roi((x-mu[:,:,None,None]).square(),m)
    return torch.cat([mu,(var+1e-8).sqrt()],1)


def components(image,gt,reference,region):
    # 全部 float32；图像不 clamp，以免人为截断 VAE→B1 梯度。
    x=image.float();g=gt.float();r=reference.float();m=region['inner']
    bg=mean_roi((x-g).abs(),region['outside']).mean()
    bd=mean_roi((edge(x)-edge(g)).abs(),region['band']).mean()
    color=(moments(x,m)-moments(g,m)).abs().mean()
    mom=(moments(x,m)-moments(r,torch.ones_like(m))).abs().mean()
    tex=[];ref=[]
    for sigma,size in zip(CONFIG['proxy']['gaussian_sigmas'],CONFIG['proxy']['gaussian_sizes']):
        fx=features(x,sigma,size);fg=features(g,sigma,size);fr=features(r,sigma,size)
        tex.append((gram(fx,m)-gram(fg,m)).abs().mean())
        ref.append((gram(fx,m)-gram(fr,torch.ones_like(m))).abs().mean())
    return dict(bg=bg,bd=bd,color=color,tex=torch.stack(tex).mean(),mom=mom,refgram=torch.stack(ref).mean())


def combined(c,weights):
    return dict(LS=c['bg']+weights['lambda_bd']*c['bd'],
        LAGT=c['color']+weights['lambda_tex']*c['tex'],
        LAR=c['mom']+weights['lambda_R']*c['refgram'])


def gradient(loss,adapter,retain_graph=True):
    params=list(adapter.named_parameters());scale=CONFIG['proxy']['gradient_loss_scale']
    grads=torch.autograd.grad(loss*scale,[p for _,p in params],retain_graph=retain_graph,allow_unused=True)
    missing=[n for (n,_),v in zip(params,grads) if v is None]
    assert not missing,('unused parameters',missing)
    flat=torch.cat([v.detach().float().flatten()/scale for v in grads])
    assert flat.numel()==143166 and torch.isfinite(flat).all()
    return flat


def association(x,y):
    from scipy.stats import rankdata
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    def rho(a,b):
        ar=rankdata(a);br=rankdata(b)
        if np.std(ar)==0 or np.std(br)==0:return None
        return float(np.corrcoef(ar,br)[0,1])
    point=rho(x,y);nonzero=(x!=0)&(y!=0)
    sign=float(np.mean(np.sign(x[nonzero])==np.sign(y[nonzero]))) if nonzero.any() else None
    rng=np.random.default_rng(CONFIG['seed_bootstrap']);boots=[];sign_boots=[]
    for _ in range(10000):
        ix=rng.integers(0,len(x),len(x));v=rho(x[ix],y[ix])
        if v is not None:boots.append(v)
        keep=(x[ix]!=0)&(y[ix]!=0)
        if keep.any():sign_boots.append(float(np.mean(np.sign(x[ix][keep])==np.sign(y[ix][keep]))))
    ci=np.quantile(boots,[.025,.975]).tolist() if boots else None
    inconclusive=point is None or ci is None or ci[1]-ci[0]>1 or point<0 or sign is None or sign<.5
    passed=not inconclusive and point>=.25 and sign>=.65
    return dict(rho=point,rho_ci95=ci,sign_agreement=sign,
        sign_ci95=np.quantile(sign_boots,[.025,.975]).tolist() if sign_boots else None,
        valid_n=len(x),nonzero_n=int(nonzero.sum()),
        bootstrap_valid_n=len(boots),bootstrap_reps=10000,pass_gate=bool(passed),inconclusive=bool(inconclusive))
