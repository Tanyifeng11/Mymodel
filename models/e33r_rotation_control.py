"""方向专用prior与复数旋转控制；forward不接收target RGB/GT。"""
import torch
from torch import nn
from torch.nn import functional as F
from models.e32_field import Residual,CrossFieldBlock

def compose(prior,q):
    x,y=prior[:,0:1],prior[:,1:2];c,s=q[:,0:1],q[:,1:2]
    return F.normalize(torch.cat([c*x-s*y,s*x+c*y],1),dim=1,eps=1e-6)

class SketchPrior(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(7,64,3,padding=1),Residual(64),
            nn.Conv2d(64,96,3,stride=2,padding=1),Residual(96),nn.Conv2d(96,128,3,padding=1),Residual(128))
        self.post=Residual(128);self.orientation=nn.Conv2d(128,2,1)
    def forward(self,structure):
        hidden=self.post(self.encoder(structure))
        return dict(orientation=F.normalize(self.orientation(hidden).float(),dim=1,eps=1e-6),hidden=hidden)

class RotationControl(nn.Module):
    def __init__(self,prior,variant='full'):
        super().__init__();self.prior=prior;self.variant=variant
        self.freeze_prior=variant!='A_no_prior_freeze';prior.requires_grad_(not self.freeze_prior)
        self.reference_projection=nn.Sequential(nn.Linear(394,128),nn.LayerNorm(128),nn.GELU())
        self.coordinate=nn.Linear(2,128);self.blocks=nn.ModuleList([CrossFieldBlock() for _ in range(3)])
        self.post=Residual(128);self.q_head=nn.Conv2d(128,2,1);self.confidence_head=nn.Conv2d(128,1,1)
        nn.init.normal_(self.q_head.weight,std=.001)
        with torch.no_grad():self.q_head.bias.copy_(torch.tensor([0.,0.] if variant=='B_additive' else [1.,0.]))
    def train(self,mode=True):
        super().train(mode)
        if self.freeze_prior:self.prior.eval()
        return self
    def forward(self,reference,structure):
        if self.variant=='F_dino_only':reference=torch.cat([reference[...,:384],torch.zeros_like(reference[...,384:])],-1)
        if self.variant=='G_geometry_only':
            r=torch.zeros_like(reference);r[...,384:387]=reference[...,384:387];reference=r
        base=self.prior(structure);hidden=base['hidden'];n,_,h,w=hidden.shape
        yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=hidden.device),torch.linspace(-1,1,w,device=hidden.device),indexing='ij')
        coords=torch.stack([xx,yy],-1).reshape(1,h*w,2).to(hidden.dtype)
        queries=hidden.flatten(2).transpose(1,2)+self.coordinate(coords)
        ref=self.reference_projection(reference.flatten(1,2))
        for block in self.blocks:queries=block(queries,ref)
        hidden=self.post(queries.transpose(1,2).reshape(n,128,h,w));raw=self.q_head(hidden).float()
        if self.variant=='B_additive':
            orientation=F.normalize(base['orientation']+raw,dim=1,eps=1e-6)
            p=base['orientation'];q=torch.cat([(orientation*p).sum(1,keepdim=True),
                orientation[:,1:2]*p[:,0:1]-orientation[:,0:1]*p[:,1:2]],1)
        else:q=F.normalize(raw,dim=1,eps=1e-6);orientation=compose(base['orientation'],q)
        return dict(orientation=orientation,q=q,confidence_logits=self.confidence_head(hidden).float(),prior=base['orientation'])
