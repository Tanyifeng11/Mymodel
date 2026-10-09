"""原IP纹理注意力消费位置的小残差；仅conditional最后8步开启。"""
import torch
from torch import nn
from torch.nn import functional as F

SITE = 'up_blocks.3.attentions.0.transformer_blocks.0.attn2'

class CausalResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(644,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,320,1))
        nn.init.zeros_(self.body[-1].weight);nn.init.zeros_(self.body[-1].bias)
    def forward(self,h,texture,geometry,falloff):
        # h和texture均是原E5在同一target query坐标上的特征，不旋转/重绘RGB。
        shape=(64,48)
        def spatial(x):
            assert x.shape[1:]==(3072,320)
            return F.layer_norm(x.float(),(320,)).transpose(1,2).reshape(-1,320,*shape)
        g=F.interpolate(geometry.detach().float(),shape,mode='bilinear',align_corners=False)
        g[:,:2]=F.normalize(g[:,:2],dim=1)
        mask=F.interpolate(falloff.detach().float(),shape,mode='bilinear',align_corners=False)
        value=self.body(torch.cat([spatial(h),spatial(texture),g],1)).tanh()*mask*.1
        return value.flatten(2).transpose(1,2)

class TextureInjection:
    def __init__(self,pipe,adapter):
        self.pipe,self.adapter=pipe,adapter
        self.enabled=False;self.conditional=False;self.index=0;self.count=0
        self.geometry=None;self.falloff=None;self.hidden=None;self.last_shape=None
        self.site=pipe.unet.get_submodule(SITE);self.processor=self.site.processor
        assert hasattr(self.processor,'to_k_ip') and hasattr(self.processor,'to_v_ip')
        assert getattr(self.processor,'texture_probe_transform',None) is None
        self.handle=self.site.register_forward_pre_hook(self.capture,with_kwargs=True)
        self.processor.texture_probe_transform=self.transform
    def capture(self,module,args,kwargs):
        self.hidden=args[0] if args else kwargs['hidden_states']
    def transform(self,texture):
        if not self.enabled or not self.conditional or self.index<42:return texture
        assert self.geometry is not None and self.falloff is not None
        residual=self.adapter(self.hidden,texture,self.geometry,self.falloff)
        self.count+=1;self.last_shape=list(texture.shape)
        return texture+residual.to(texture.dtype)
    def close(self):
        self.handle.remove();self.processor.texture_probe_transform=None
