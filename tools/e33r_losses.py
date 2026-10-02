"""固定组内clean三arm、nuisance三arm、zero；监督只在M_cf。"""
import torch
from torch.nn import functional as F

def group_mean(value,weight):
    # 每个target先按confidence加权cells，再等权平均完整target组。
    return ((value*weight).flatten(1).sum(1)/weight.flatten(1).sum(1).clamp_min(1e-8)).mean()

def group_loss(pred,gt,support,variant='full',r180=True):
    ori,q,conf=pred['orientation'],pred['q'],pred['confidence_logits']
    weight=support.float()*gt[:,3];use=[0,1,2] if r180 and variant!='D_no_r180' else [0,1]
    cosine=lambda a,b:1-(a*b).sum(1)
    field=torch.stack([group_mean(cosine(ori[:,i],gt[:,:2]*(-1 if i==1 else 1)),weight) for i in use]).mean()
    qloss=torch.stack([group_mean(1-q[:,i,0]*(-1 if i==1 else 1),weight) for i in use]).mean()
    pairs=[group_mean(1+(ori[:,1]*ori[:,0]).sum(1),weight)]
    if 2 in use:pairs.append(group_mean(cosine(ori[:,2],ori[:,0]),weight))
    pair=torch.stack(pairs).mean() if variant!='C_no_pair' else ori.sum()*0
    zero=group_mean(1-q[:,6,0],weight)
    nuis=torch.stack([group_mean(cosine(ori[:,i],ori[:,i+3]),weight) for i in use]).mean()
    if variant=='E_no_nuisance':nuis=ori.sum()*0
    active=use+[i+3 for i in use]+[6]
    confidence=torch.stack([group_mean(F.binary_cross_entropy_with_logits(conf[:,i,0],
        torch.full_like(conf[:,i,0],0 if i==6 else 1),reduction='none'),weight) for i in active]).mean()
    losses=dict(field=field,q=qloss,pair=pair,zero=zero,nuisance=nuis,confidence=confidence)
    total=field+.5*qloss+.5*pair+.25*zero+.25*nuis+.1*confidence
    return total,losses
