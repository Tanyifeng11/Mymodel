"""M2-A：独立外观编码与4级几何解码；前向只接收reference和RF几何。"""
import torch
from torch import nn
from torch.nn import functional as F

def block(a,b):
    return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(8,b),nn.SiLU(),
        nn.Conv2d(b,b,3,padding=1),nn.GroupNorm(8,b),nn.SiLU())

class AppearanceEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn=nn.Sequential(nn.Conv2d(3,32,5,stride=2,padding=2),nn.GroupNorm(8,32),nn.SiLU(),
            nn.Conv2d(32,64,3,stride=2,padding=1),nn.GroupNorm(8,64),nn.SiLU(),
            nn.Conv2d(64,64,3,stride=2,padding=1),nn.GroupNorm(8,64),nn.SiLU())
        self.global_code=nn.Linear(64,128)
    def forward(self,rgb):
        f=self.cnn(rgb)
        return self.global_code(f.mean((2,3))),F.adaptive_avg_pool2d(f,(8,8)).flatten(2).transpose(1,2)

class AppearanceGeometryDecoder(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=AppearanceEncoder()
        widths=[32,64,96,128,160]
        self.start=block(132,32)
        self.down=nn.ModuleList(nn.Sequential(nn.Conv2d(a,b,3,stride=2,padding=1),block(b,b)) for a,b in zip(widths,widths[1:]))
        self.attention=nn.MultiheadAttention(160,4,kdim=64,vdim=64,batch_first=True)
        self.up=nn.ModuleList(block(a+b,b) for a,b in zip(widths[:0:-1],widths[-2::-1]))
        self.rgb=nn.Conv2d(32,3,1)
        assert sum(p.numel() for p in self.parameters())<=5000000
    def appearance(self,rgb):return self.encoder(rgb)
    def decode(self,appearance,geometry):
        code,tokens=appearance;h,w=geometry.shape[-2:]
        f=self.start(torch.cat([geometry,code[:,:,None,None].expand(-1,-1,h,w)],1));skips=[]
        for layer in self.down:skips.append(f);f=layer(f)
        q=f.flatten(2).transpose(1,2)
        attended,_=self.attention(q,tokens,tokens,need_weights=False)
        f=f+attended.transpose(1,2).reshape_as(f)
        for layer,skip in zip(self.up,skips[::-1]):
            f=layer(torch.cat([F.interpolate(f,skip.shape[-2:],mode='bilinear',align_corners=False),skip],1))
        return self.rgb(f).sigmoid()
    def forward(self,reference,geometry):return self.decode(self.appearance(reference),geometry)
