"""E36：已发表 DAGF 双核思想的单尺度迁移，以及冲突代理对照。"""
import torch
from torch import nn
from torch.nn import functional as F


def gradient(x):
    kernels=x.new_tensor([[[-1,0,1],[-2,0,2],[-1,0,1]],
                          [[-1,-2,-1],[0,0,0],[1,2,1]]])/8
    b,c,h,w=x.shape
    out=F.conv2d(F.pad(x.reshape(b*c,1,h,w),(1,1,1,1),mode='replicate'),kernels[:,None])
    return (out.square().sum(1)+1e-12).sqrt().reshape(b,c,h,w)


def normalize(x):
    lo=x.amin((-2,-1),keepdim=True);hi=x.amax((-2,-1),keepdim=True)
    return (x-lo)/(hi-lo).clamp_min(1e-6)


def local_filter(z,k):
    b,c,h,w=z.shape
    neighbors=F.unfold(F.pad(z.float(),(1,1,1,1),mode='replicate'),3).reshape(b,c,9,h,w)
    return (neighbors*k.float()[:,None]).sum(2)


class GuidedKernelFilter(nn.Module):
    def __init__(self,channels,time_channels,arm,projection=64):
        super().__init__();self.arm=arm;self.stats={};self.collect=False
        self.norm=nn.GroupNorm(32 if channels%32==0 else 1,channels)
        self.project=nn.Conv2d(channels,projection,1)
        self.guide=nn.Sequential(nn.Conv2d(2,projection,3,padding=1),nn.SiLU(),
                                 nn.Conv2d(projection,projection,3,padding=1),nn.SiLU())
        self.time=nn.Linear(time_channels,8)
        self.head=nn.Conv2d(projection,channels,1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        if arm=='B1_CONV':
            self.conv=nn.Sequential(nn.Conv2d(2*projection+8,6,3,padding=1),nn.SiLU(),
                                    nn.Conv2d(6,projection,3,padding=1))
        else:
            self.ks=nn.Conv2d(projection,9,3,padding=1)
            self.kf=nn.Conv2d(projection,9,3,padding=1)
            self.mix=nn.Conv2d(2*projection+8+(arm=='B3_CONFLICT'),1,3,padding=1)
            if arm=='B3_CONFLICT':self.edge=nn.Conv2d(projection,1,1)

    def forward(self,feature,sketch,temb):
        # E5 保持 fp16；新增核、投影、softmax 和求和都使用 fp32。
        with torch.autocast(device_type=feature.device.type,enabled=False):
            f=feature.float();z=self.project(self.norm(f));h,w=z.shape[-2:]
            s=sketch.float().mean(1,keepdim=True)
            s=F.interpolate(s,size=(h,w),mode='bilinear',align_corners=False)
            es=gradient(s);g=self.guide(torch.cat([s,es],1))
            t=self.time(temb.float())[:,:,None,None].expand(-1,-1,h,w)
            inputs=[g,z,t];stats={}
            if self.arm=='B1_CONV':residual=self.conv(torch.cat(inputs,1))
            else:
                if self.arm=='B3_CONFLICT':
                    d=(normalize(es)-normalize(self.edge(gradient(z)))).abs()
                    inputs.append(d)
                ks=self.ks(g).softmax(1);kf=self.kf(z).softmax(1)
                a=self.mix(torch.cat(inputs,1)).sigmoid();k=a*ks+(1-a)*kf
                residual=local_filter(z,k)-z
                if self.collect:
                    stats.update(kernel_sum_max_error=float((k.sum(1)-1).abs().max()),
                        entropy_ks=float(-(ks*ks.clamp_min(1e-12).log()).sum(1).mean()),
                        entropy_kf=float(-(kf*kf.clamp_min(1e-12).log()).sum(1).mean()),
                        a_quantiles=torch.quantile(a.detach().flatten(),a.new_tensor([.1,.5,.9])).tolist())
                    if self.arm=='B3_CONFLICT':
                        median=d.detach().median();stats.update(d_mean=float(d.mean()),
                            a_high_d=float(a[d>=median].mean()),a_low_d=float(a[d<=median].mean()))
            delta=.1*self.head(residual)
            if self.collect:
                stats.update(feature_rms=float(f.square().mean().sqrt()),
                    delta_feature_rms=float(delta.square().mean().sqrt()),
                    delta_feature_relative=float(delta.square().mean().sqrt()/f.square().mean().sqrt().clamp_min(1e-12)))
                self.stats=stats
            return feature+delta.to(feature.dtype)
