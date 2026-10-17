"""已确认的空间拼接/显式形变后备；不包含GT、旧B1注入或可学习全局gain。"""
import hashlib
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call


def frozen_hash(module, excluded=()):
    digest=hashlib.sha256()
    for name,value in module.state_dict().items():
        if name in excluded:continue
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def concat_inputs(noisy, noisy_reference, sketch, reference, mask):
    assert noisy.shape==noisy_reference.shape==sketch.shape==reference.shape
    assert noisy.shape[1]==4 and mask.shape==(noisy.shape[0],1,*noisy.shape[-2:])
    return torch.cat([torch.cat([noisy,noisy_reference],-2),
        torch.cat([mask,torch.zeros_like(mask)],-2),torch.cat([sketch,reference],-2)],1)


class CatBranch(nn.Module):
    """完整9通道U-Net路径；只存独立QKV/out/首卷积参数，冻结权重不写入ckpt。

    initial对照仅在no_grad下functional_call。学生前向使用真实参数绑定，
    因此activation checkpoint在backward重算时不会读到已退出的临时参数。
    """
    def __init__(self,backbone):
        super().__init__();backbone.requires_grad_(False)
        old=backbone.conv_in
        conv=nn.Conv2d(9,old.out_channels,old.kernel_size,old.stride,old.padding,
            dilation=old.dilation,bias=old.bias is not None,device=old.weight.device,dtype=torch.float32)
        with torch.no_grad():
            conv.weight.zero_();conv.weight[:,:4].copy_(old.weight.float())
            if conv.bias is not None:conv.bias.copy_(old.bias.float())
        backbone.conv_in=conv
        if hasattr(backbone,'register_to_config'):backbone.register_to_config(in_channels=9)
        names=[];parameters=[]
        for name,value in list(backbone.named_parameters()):
            selected=name.startswith('conv_in.') or any('.attn1.'+key+'.' in name for key in ['to_q','to_k','to_v','to_out.0'])
            if not selected:continue
            parent,leaf=name.rsplit('.',1)
            replacement=nn.Parameter(value.detach().float().clone())
            backbone.get_submodule(parent).register_parameter(leaf,replacement)
            names.append(name);parameters.append(replacement)
        assert names and any('.attn1.to_q.' in n for n in names)
        self.parameter_names=names;self.trainable=nn.ParameterList(parameters)
        for i,p in enumerate(parameters):self.register_buffer('initial_'+str(i),p.detach().clone(),persistent=False)
        object.__setattr__(self,'backbone',backbone)  # 冻结骨干不重复注册/保存。
        if hasattr(backbone,'enable_gradient_checkpointing'):backbone.enable_gradient_checkpointing()
        self.initial_frozen_hash=frozen_hash(backbone,set(names))

    def train(self,mode=True):
        super().train(mode);self.backbone.train(mode);return self

    def forward(self,noisy,noisy_reference,sketch,reference,mask,timestep,encoder_hidden_states,cross_attention_kwargs):
        sample=concat_inputs(noisy,noisy_reference,sketch,reference,mask)
        cross=dict(cross_attention_kwargs or {});cross.pop('sa_hidden_states',None)
        if 'tcpm_garment_mask' in cross:
            cross['tcpm_garment_mask']=torch.cat([mask,torch.zeros_like(mask)],-2)
        kwargs=dict(encoder_hidden_states=encoder_hidden_states,cross_attention_kwargs=cross,return_dict=False)
        previous=self.backbone.training;self.backbone.eval()
        initial={name:getattr(self,'initial_'+str(i)) for i,name in enumerate(self.parameter_names)}
        with torch.no_grad():baseline=functional_call(self.backbone,initial,(sample,timestep),kwargs)[0]
        self.backbone.train(previous)
        # 兼容旧Diffusers的reentrant checkpoint；输入梯度仅用于重算，不作为参数更新。
        if torch.is_grad_enabled():sample=sample.detach().requires_grad_(True)
        prediction=self.backbone(sample,timestep,**kwargs)[0]
        return (prediction-baseline)[...,:noisy.shape[-2],:]

    def verify_frozen(self):
        assert frozen_hash(self.backbone,set(self.parameter_names))==self.initial_frozen_hash,'CAT_FROZEN_BACKBONE_CHANGED'


class ConvBlock(nn.Sequential):
    def __init__(self,cin,cout,stride=1):
        super().__init__(nn.Conv2d(cin,cout,3,stride,1),nn.GroupNorm(8,cout),nn.SiLU(),
            nn.Conv2d(cout,cout,3,1,1),nn.GroupNorm(8,cout),nn.SiLU())


def identity_grid(batch,height,width,device):
    y,x=torch.meshgrid((torch.arange(height,device=device)+.5)*2/height-1,
        (torch.arange(width,device=device)+.5)*2/width-1,indexing='ij')
    return torch.stack([x,y],-1)[None].expand(batch,-1,-1,-1)


def warp(source,flow):
    # grid_sample的flow为归一化坐标；上采样flow不额外乘像素尺度。
    grid=identity_grid(source.shape[0],*source.shape[-2:],source.device)+flow.float().permute(0,2,3,1)
    return F.grid_sample(source.float(),grid,mode='bilinear',padding_mode='zeros',align_corners=False).to(source.dtype)


class FlowPyramid(nn.Module):
    def __init__(self):
        super().__init__()
        self.target=nn.ModuleList([ConvBlock(4,32),ConvBlock(32,64,2),ConvBlock(64,128,2)])
        self.reference=nn.ModuleList([ConvBlock(4,32),ConvBlock(32,64,2),ConvBlock(64,128,2)])
        self.heads=nn.ModuleList([nn.Sequential(ConvBlock(2*c,c),nn.Conv2d(c,2,3,1,1)) for c in [32,64,128]])
        for head in self.heads:nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)

    def forward(self,sketch,reference):
        targets=[];references=[];a=sketch;b=reference
        for ta,rb in zip(self.target,self.reference):
            a=ta(a);b=rb(b);targets.append(a);references.append(b)
        flow=None;flows=[]
        for i in [2,1,0]:
            target,source=targets[i],references[i]
            flow=source.new_zeros((source.shape[0],2,*source.shape[-2:])) if flow is None else F.interpolate(flow,size=source.shape[-2:],mode='bilinear',align_corners=False)
            flow=flow+self.heads[i](torch.cat([target,warp(source,flow)],1));flows.append(flow)
        return warp(reference,flow),flows


def time_embedding(timestep,batch,device):
    t=torch.as_tensor(timestep,device=device).float().reshape(-1)
    if len(t)==1:t=t.expand(batch)
    frequencies=torch.exp(-math.log(10000)*torch.arange(16,device=device)/16)
    angle=t[:,None]*frequencies[None]
    return torch.cat([angle.cos(),angle.sin()],1)


class WarpBranch(nn.Module):
    def __init__(self):
        super().__init__();self.flow=FlowPyramid()
        self.time=nn.Sequential(nn.Linear(32,128),nn.SiLU(),nn.Linear(128,32))
        self.encoder=nn.ModuleList([ConvBlock(13,32),ConvBlock(32,64,2),ConvBlock(64,128,2),ConvBlock(128,256,2)])
        self.decoder=nn.ModuleList([ConvBlock(256+128,128),ConvBlock(128+64,64),ConvBlock(64+32,32)])
        self.output=nn.Conv2d(32,4,3,1,1);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        self.smoothness=None;self.last_flow=None

    def forward(self,noisy,sketch,reference,mask,timestep):
        warped,flows=self.flow(sketch,reference);self.last_flow=flows[-1]
        self.smoothness=torch.stack([((f[...,1:]-f[...,:-1]).abs().mean()+
            (f[:,:,1:]-f[:,:,:-1]).abs().mean())/2 for f in flows]).mean()
        hidden=torch.cat([noisy,sketch,warped,mask],1);skips=[]
        for i,block in enumerate(self.encoder):
            hidden=block(hidden)
            if i==0:hidden=hidden+self.time(time_embedding(timestep,len(noisy),noisy.device)).to(hidden.dtype)[:,:,None,None]
            skips.append(hidden)
        for block,skip in zip(self.decoder,reversed(skips[:-1])):
            hidden=block(torch.cat([F.interpolate(hidden,size=skip.shape[-2:],mode='bilinear',align_corners=False),skip],1))
        return self.output(hidden)
