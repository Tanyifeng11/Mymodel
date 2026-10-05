"""同一target同一GT支持上计算match与两种轴向equivariance。"""
import torch
from tools.e33rc_losses import case_mean

def losses(pred,gt,support,phase,variant='full',supervised_scale=1.,regular_scale=1.):
    ori=pred['orientation'];weight=support.float()*gt[:,3];valid=weight.flatten(1).sum(1)>0
    avg=lambda x:case_mean(x,weight)[valid].mean() if valid.any() else x.sum()*0
    match=avg(1-(ori[:,0]*gt[:,:2]).sum(1))
    eq=avg(1+(ori[:,1]*ori[:,0]).sum(1)) if phase!='RF1' else match*0
    identity=avg(1-(ori[:,2]*ori[:,0]).sum(1)) if phase!='RF1' else match*0
    arms=1 if phase=='RF1' else 3
    regular=pred['adapter_delta'][:,:arms].float().square().flatten(2).mean(2).mean()
    # E/F仅统计被adapter处理的通道，避免零填充稀释正则。
    if variant=='E_geometry_only':regular=regular*394/3
    if variant=='F_dino_only':regular=regular*394/384
    rank=match*0
    if phase=='RF3':
        # 各case先算hinge再均值，而不是先平均error再hinge。
        e=torch.stack([case_mean(torch.rad2deg(torch.acos((ori[:,i]*gt[:,:2]).sum(1).clamp(-1+1e-6,1-1e-6)))/2,weight) for i in [0,3,4]],1)
        rank=(.5*torch.relu(5+e[:,0]-e[:,1])+.25*torch.relu(7.5+e[:,0]-e[:,2]))[valid].mean() if valid.any() else ori.sum()*0
    leq=0 if phase=='RF1' or variant=='A_no_equivariance' else 1
    l180=0 if phase=='RF1' or variant in ['A_no_equivariance','B_no_r180'] else .5
    total=supervised_scale*(match+leq*eq+l180*identity+rank)+regular_scale*.05*regular
    return total,dict(match=float(match.detach()),eq90=float(eq.detach()),eq180=float(identity.detach()),adapter_id=float(regular.detach()),rank=float(rank.detach()))
