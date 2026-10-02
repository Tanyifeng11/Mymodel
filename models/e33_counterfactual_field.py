"""固定sketch prior与reference旋转/频率残差；forward无target RGB。"""
import torch
from torch import nn
from torch.nn import functional as F
from models.e32_field import Residual,CrossFieldBlock


class SketchPrior(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(7,64,3,padding=1),Residual(64),
            nn.Conv2d(64,96,3,stride=2,padding=1),Residual(96),nn.Conv2d(96,128,3,padding=1),Residual(128))
        self.post=Residual(128)
        self.orientation=nn.Conv2d(128,2,1)
        self.period=nn.Conv2d(128,1,1)
        self.confidence=nn.Conv2d(128,2,1)
        nn.init.constant_(self.period.bias,-3.)

    def forward(self,structure):
        hidden=self.post(self.encoder(structure))
        return dict(orientation=F.normalize(self.orientation(hidden).float(),dim=1,eps=1e-6),
                    log_frequency=self.period(hidden).float(),confidence=self.confidence(hidden).float().sigmoid(),hidden=hidden)


class CounterfactualField(nn.Module):
    def __init__(self,prior,freeze_prior=True):
        super().__init__();self.prior=prior;self.freeze_prior=freeze_prior
        self.prior.requires_grad_(not freeze_prior)
        self.reference_projection=nn.Sequential(nn.Linear(395,128),nn.LayerNorm(128),nn.GELU())
        self.coordinate=nn.Linear(2,128)
        self.blocks=nn.ModuleList([CrossFieldBlock() for _ in range(3)])
        self.post=Residual(128)
        self.delta_orientation=nn.Conv2d(128,2,1)
        self.delta_period=nn.Conv2d(128,1,1)
        self.reference_confidence=nn.Conv2d(128,2,1)
        nn.init.zeros_(self.delta_orientation.weight)
        with torch.no_grad():self.delta_orientation.bias.copy_(torch.tensor([1.,0.]))
        nn.init.zeros_(self.delta_period.weight);nn.init.zeros_(self.delta_period.bias)

    def train(self,mode=True):
        super().train(mode)
        if self.freeze_prior:self.prior.eval()
        return self

    def forward(self,reference,structure):
        prior=self.prior(structure)
        hidden=prior['hidden'];n,_,h,w=hidden.shape
        yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=hidden.device),
                             torch.linspace(-1,1,w,device=hidden.device),indexing='ij')
        coordinates=torch.stack([xx,yy],-1).reshape(1,h*w,2).to(hidden.dtype)
        q=hidden.flatten(2).transpose(1,2)+self.coordinate(coordinates)
        r=self.reference_projection(reference.flatten(1,2))
        for block in self.blocks:q=block(q,r)
        hidden=self.post(q.transpose(1,2).reshape(n,128,h,w))
        delta=F.normalize(self.delta_orientation(hidden).float(),dim=1,eps=1e-6)
        ori=prior['orientation'];c,s=delta[:,0:1],delta[:,1:2]
        output=torch.cat([c*ori[:,0:1]-s*ori[:,1:2],s*ori[:,0:1]+c*ori[:,1:2]],1)
        dp=self.delta_period(hidden).float()
        return dict(orientation=output,log_frequency=prior['log_frequency']+dp,
                    confidence=self.reference_confidence(hidden).float().sigmoid(),
                    delta_orientation=delta,delta_period=dp,
                    prior_orientation=ori,prior_log_frequency=prior['log_frequency'])
