"""仅改变reference特征；不接收target RGB、GT、干预标签。"""
import torch
from torch import nn

class RealAdapter(nn.Module):
    def __init__(self, width=32, variant='full'):
        super().__init__(); self.variant=variant; self.alpha=.1
        self.start,self.stop={'E_geometry_only':(384,387),'F_dino_only':(0,384)}.get(variant,(0,394))
        d=self.stop-self.start
        self.net=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,width),nn.GELU(),nn.Linear(width,d))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,reference):
        raw=self.net(reference[...,self.start:self.stop])
        delta=torch.cat([torch.zeros_like(reference[...,:self.start]),raw,
                         torch.zeros_like(reference[...,self.stop:])],-1)
        return reference+self.alpha*delta,delta

class FrozenCausalAdapter(nn.Module):
    def __init__(self,backbone,width=32,variant='full'):
        super().__init__();self.backbone=backbone;self.variant=variant
        backbone.requires_grad_(False);backbone.eval()
        self.adapter=None if variant=='C_full_finetune' else RealAdapter(width,variant)
        if self.adapter is None:
            backbone.requires_grad_(True);backbone.prior.requires_grad_(False)
    @property
    def prior(self):return self.backbone.prior
    def train(self,mode=True):
        super().train(mode)
        if self.variant!='C_full_finetune':self.backbone.eval()
        self.prior.eval();return self
    def forward(self,reference,structure):
        adapted,delta=self.adapter(reference) if self.adapter else (reference,torch.zeros_like(reference))
        # 冻结参数不等于no_grad：必须允许梯度经过backbone返回adapter。
        result=self.backbone(adapted,structure)
        result['adapter_delta']=delta;result['adapter_features']=adapted
        return result
