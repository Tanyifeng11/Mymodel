"""真实ranking使用同target、同GT支持、case内confidence权重的轴向误差（度）。"""
import torch
from tools.e33r_losses import group_mean

def case_mean(value,weight):
    return (value*weight).flatten(1).sum(1)/weight.flatten(1).sum(1).clamp_min(1e-8)

def real_losses(pred,gt,support):
    ori=pred['orientation'];weight=support.float()*gt[:,3]
    valid=weight.flatten(1).sum(1)>0
    cosine=(ori*gt[:,None,:2]).sum(2).clamp(-1,1)
    reconstruct=case_mean(1-cosine[:,0],weight)
    # 训练时防止acos端点无穷梯度；评测仍使用原E32精确clamp[-1,1]。
    errors=torch.stack([case_mean(torch.rad2deg(torch.acos(cosine[:,i].clamp(-1+1e-6,1-1e-6)))/2,weight)
                        for i in range(3)],1)
    ranking=torch.relu(5+errors[:,0]-errors[:,1])+.5*torch.relu(7.5+errors[:,0]-errors[:,2])
    if not valid.any():return ori.sum()*0,ori.sum()*0
    return reconstruct[valid].mean(),ranking[valid].mean()

def retention_loss(student,teacher,weight):
    return torch.stack([group_mean(1-(student[:,i]*teacher[:,i]).sum(1),weight) for i in range(3)]).mean()
