"""真实渲染UV到实际32×24/16×16网格；不把特征格坐标称作语义纯净来源。"""
import cv2
import numpy as np
from tools.e53_protocol import CONFIG,identity_seed

def grid(uv,labels,writer=False):
    # Target按实际特征格中心采样；Writer使用与spatial_tokens相同的adaptive pooling区间。
    h,w=32,24;ys=((np.arange(h)+.5)*uv.shape[0]/h).astype(int);xs=((np.arange(w)+.5)*uv.shape[1]/w).astype(int)
    values=uv[np.ix_(ys,xs)];parts=labels[np.ix_(ys,xs)];valid=parts>0
    if not writer:return values.reshape(-1,2),parts.ravel(),valid.ravel(),np.ones(h*w)
    out=[];labs=[];ok=[];purity=[]
    for y in range(16):
        ya=int(np.floor(y*h/16));yb=int(np.ceil((y+1)*h/16))
        for x in range(16):
            xa=int(np.floor(x*w/16));xb=int(np.ceil((x+1)*w/16))
            pp=parts[ya:yb,xa:xb].ravel();vv=values[ya:yb,xa:xb].reshape(-1,2)
            counts=np.bincount(pp);label=int(np.argmax(counts));fraction=float((pp==label).mean())
            out.append(vv[pp==label].mean(0));labs.append(label);purity.append(fraction)
            ok.append(label>0 and fraction>=CONFIG['part_purity_min'] and float((pp>0).mean())>=CONFIG['token_foreground_min'])
    return np.asarray(out),np.asarray(labs),np.asarray(ok),np.asarray(purity)

def correspondence(target,reference,mode='ORACLE',sigma=.03,seed=0):
    tu,tl,tv,_=target;ru,rl,rv,_=reference;tu=tu.copy()
    allowed=(tl[:,None]==rl[None])&tv[:,None]&rv[None]
    if mode in ['SHIFT','FLIP']:
        # 错误映射在同一部位UV区间内构造，避免改变garment support。
        for part in np.unique(tl[tv]):
            q=(tl==part)&tv;r=(rl==part)&rv
            if not r.any():continue
            lo=float(ru[r,0].min());hi=float(ru[r,0].max());span=max(hi-lo,1e-6)
            u=(tu[q,0]-lo)/span
            tu[q,0]=lo+((u+CONFIG['wrong_shift'])%1 if mode=='SHIFT' else 1-u)*span
    d2=((tu[:,None]-ru[None])**2).sum(-1)
    logits=-d2/(2*sigma*sigma);logits[~allowed]=-1e30
    logits-=logits.max(1,keepdims=True)
    m=np.exp(np.maximum(logits,-70))*allowed
    m/=np.maximum(m.sum(1,keepdims=True),1e-30)
    if mode=='RANDOM':
        rng=np.random.default_rng(seed)
        for part in np.unique(tl[tv]):
            q=np.where((tl==part)&tv)[0];cols=np.where((rl==part)&rv)[0]
            m[np.ix_(q,cols)]=m[np.ix_(q,rng.permutation(cols))]
    return m.astype('float32'),allowed

def matrices(row,geometry='N',mode='ORACLE',sigma=None):
    d=np.load(row['geometry']);t=grid(d['target_uv'],d['target_labels']);r=grid(d['uv'+geometry],d['labels'+geometry],True)
    m,allowed=correspondence(t,r,mode,sigma or CONFIG['sigma_uv'],identity_seed(row['id']))
    return m,allowed,t,r

def align(row,source='1',destination='0'):
    d=np.load(row['geometry']);a=grid(d['uv'+destination],d['labels'+destination],True);b=grid(d['uv'+source],d['labels'+source],True)
    return correspondence(a,b,sigma=CONFIG['sigma_uv'])[0]
