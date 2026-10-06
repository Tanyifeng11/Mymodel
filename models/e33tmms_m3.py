"""固定 RF2 的3×3方向采样，单个1/8层的微型残差投影。"""
import torch
from torch import nn
from torch.nn import functional as F

class OrientedFeatureTransport(nn.Module):
    site='up_blocks.3.resnets.2'
    def __init__(self):
        super().__init__();self.weights=nn.Parameter(torch.zeros(9))
        self.projection=nn.Sequential(nn.Conv2d(128,16,1),nn.SiLU(),nn.Conv2d(16,320,1))
        nn.init.zeros_(self.projection[-1].weight);nn.init.zeros_(self.projection[-1].bias)
    def sample(self,feature,geometry):
        b,_,h,w=feature.shape
        field=F.interpolate(geometry.detach(),(h,w),mode='bilinear',align_corners=False)
        theta=.5*torch.atan2(field[:,1],field[:,0]);c,s=theta.cos(),theta.sin()
        y,x=torch.meshgrid(torch.arange(h,device=feature.device),torch.arange(w,device=feature.device),indexing='ij')
        samples=[]
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                # 先在feature格点坐标旋转，再分别归一化x/y，避免非方形画布扭曲角度。
                gx=(x[None]+dx*c-dy*s)*2/(w-1)-1
                gy=(y[None]+dx*s+dy*c)*2/(h-1)-1
                samples.append(F.grid_sample(feature,torch.stack([gx,gy],-1),mode='bilinear',padding_mode='border',align_corners=True))
        result=(torch.stack(samples,1)*self.weights.softmax(0)[None,:,None,None,None]).sum(1)
        return result,field[:,3:4]*(field[:,2:3]>=.25)
    def forward(self,feature,geometry):
        sampled,mask=self.sample(feature,geometry)
        return .1*self.projection(sampled)*mask

class SingleSiteInjection:
    def __init__(self,pipe,transport):
        self.pipe,self.transport=pipe,transport;self.residual=None;self.enabled=True;self.conditional=False
        self.count=0;self.last_shape=None
        site=pipe.unet.get_submodule(transport.site)
        self.handles=[pipe.unet.register_forward_pre_hook(self.before,with_kwargs=True),site.register_forward_hook(self.inject)]
    def before(self,module,args,kwargs):
        self.conditional='sa_hidden_states' in (kwargs.get('cross_attention_kwargs') or {})
    def inject(self,module,args,output):
        if not self.enabled or not self.conditional:return output
        assert self.residual is not None and output.shape==self.residual.shape
        assert output.shape[1:]==(320,64,48)
        self.count+=1;self.last_shape=list(output.shape)
        return output+self.residual.to(output.dtype)
    def set(self,feature,geometry):self.residual=self.transport(feature,geometry)
    def close(self):
        for handle in self.handles:handle.remove()
