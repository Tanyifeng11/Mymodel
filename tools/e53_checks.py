"""对会混淆因果结论的数值契约作离线核验。"""
import numpy as np
import torch
from torch.nn import functional as F
from models.e53_oracle import attention,match_energy
from tools.e53_mapping import matrices
from tools.e53_protocol import *
from tools.e53_run import targets

def main():
    torch.manual_seed(53)
    q=torch.randn(1,2,13,8);ks=torch.randn(1,2,13,8);vs=torch.randn_like(ks);kr=torch.randn(1,2,7,8);vr=torch.randn_like(kr)
    a,p=attention(q,kr,ks);own=F.scaled_dot_product_attention(q,ks,vs);joint=F.scaled_dot_product_attention(q,torch.cat([ks,kr],2),torch.cat([vs,vr],2))
    reconstructed=(1-p[...,None])*own+p[...,None]*(a@vr)
    assert torch.max(torch.abs(reconstructed-joint))<1e-6
    x=torch.randn(1,768,640).half()
    for target in [0,.01,.1,1,5]:
        _,r=match_energy(x,target);assert r['relative_error']<1e-4
    # 全局gain接近1时，FP16平台会让旧比例迭代振荡；保留固定门限核验fallback。
    source=float(x.float().square().mean().sqrt());fallbacks=0
    for offset in [.00011,.00019,.00030,-.00019]:
        y,r=match_energy(x,source*(1+offset));assert r['relative_error']<1e-4
        assert abs(float(y.float().square().mean().sqrt())/r['target']-1)<1e-4
        fallbacks+=r['method']=='bisection_after_rescale'
    print('E53 FP16 NEAR-UNIT GAIN CHECK',fallbacks,'bisection fallbacks',flush=True)
    base=np.linspace(.1,.9,50);total=np.dot(base,base)
    for window in ['ALL','EARLY','MIDDLE','LATE']+[[i] for i in KEY_STEPS]:
        t=np.asarray(targets(base,window));assert abs(np.dot(t,t)/total-1)<1e-12
    path=OUT/'splits/pilot.json'
    if path.exists():
        for row in read(path):
            for g in ['0','1','N']:
                original,support,_,_=matrices(row,g);valid=original.sum(1)>.99
                assert original.shape==(768,256) and valid.any()
                for mode in ['ORACLE','SHIFT','FLIP','RANDOM']:
                    m,allowed,_,_=matrices(row,g,mode)
                    assert np.array_equal(allowed,support) and np.max(np.abs(m[valid].sum(1)-1))<2e-6 and not m[~allowed].any()
    print('E53 OFFLINE CONTRACTS PASS',flush=True)

if __name__=='__main__':main()
