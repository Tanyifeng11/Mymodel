"""E32 dense reference-to-target field；forward不接受任何target RGB或pseudo-GT。"""

import torch
from torch import nn
from torch.nn import functional as F


class Residual(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net=nn.Sequential(nn.GroupNorm(8,width),nn.GELU(),nn.Conv2d(width,width,3,padding=1),
                               nn.GroupNorm(8,width),nn.GELU(),nn.Conv2d(width,width,3,padding=1))

    def forward(self,x):
        return x+self.net(x)


class CrossFieldBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.qnorm=nn.LayerNorm(128)
        self.rnorm=nn.LayerNorm(128)
        self.attention=nn.MultiheadAttention(128,4,dropout=.1,batch_first=True)
        self.ffnorm=nn.LayerNorm(128)
        self.ffn=nn.Sequential(nn.Linear(128,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,128))
        self.dropout=nn.Dropout(.1)

    def forward(self,q,r):
        normalized=self.rnorm(r)
        attended=self.attention(self.qnorm(q),normalized,normalized,need_weights=False)[0]
        q=q+self.dropout(attended)
        return q+self.dropout(self.ffn(self.ffnorm(q)))


class ExplicitPatternField(nn.Module):
    def __init__(self, reference_dim=395):
        super().__init__()
        self.reference_projection=nn.Sequential(nn.Linear(reference_dim,128),nn.LayerNorm(128),nn.GELU())
        self.structure_encoder=nn.Sequential(nn.Conv2d(7,64,3,padding=1),Residual(64),
            nn.Conv2d(64,96,3,stride=2,padding=1),Residual(96),nn.Conv2d(96,128,3,padding=1),Residual(128))
        self.coordinate=nn.Linear(2,128)
        self.blocks=nn.ModuleList([CrossFieldBlock() for _ in range(3)])
        self.post=Residual(128)
        self.orientation=nn.Conv2d(128,2,1)
        self.period=nn.Conv2d(128,1,1)
        self.geometry_confidence=nn.Conv2d(128,2,1)
        self.appearance=nn.Conv2d(128,64,1)
        self.appearance_norm=nn.LayerNorm(64)
        self.appearance_confidence=nn.Conv2d(128,1,1)
        nn.init.constant_(self.period.bias,-3.)
        self.set_appearance_trainable(False)

    def set_appearance_trainable(self,enabled):
        for module in (self.appearance,self.appearance_norm,self.appearance_confidence):
            module.requires_grad_(enabled)

    def forward(self,reference,structure):
        assert reference.shape[1:3]==(16,12) and structure.shape[1:]==(7,128,96)
        r=self.reference_projection(reference.flatten(1,2))
        spatial=self.structure_encoder(structure)
        yy,xx=torch.meshgrid(torch.linspace(-1,1,64,device=spatial.device),
                             torch.linspace(-1,1,48,device=spatial.device),indexing='ij')
        coordinates=torch.stack([xx,yy],-1).reshape(1,3072,2).to(spatial.dtype)
        q=spatial.flatten(2).transpose(1,2)+self.coordinate(coordinates)
        for block in self.blocks:
            q=block(q,r)
        spatial=self.post(q.transpose(1,2).reshape(-1,128,64,48))
        appearance=self.appearance_norm(self.appearance(spatial).permute(0,2,3,1)).permute(0,3,1,2)
        return dict(orientation=F.normalize(self.orientation(spatial).float(),dim=1,eps=1e-6),
                    log_frequency=self.period(spatial).float(),
                    confidence=torch.cat([self.geometry_confidence(spatial),self.appearance_confidence(spatial)],1).float().sigmoid(),
                    appearance=appearance.float())
