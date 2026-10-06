"""可微训练代理；正式Gate仍使用冻结E26和原TextureSim。"""
import torch
from torch import nn
from torch.nn import functional as F
from models.bf_texture_module import BFTextureConditioner
from tools.e33tm_protocol import E5
from tools.e33tmoc_appearance_eval import frozen_lpips,model_hash

def frozen_texture():
    state=torch.load(E5,map_location='cpu')['bf_texture_conditioner']
    model=BFTextureConditioner(clip_embeddings_dim=state['token_source_proj.0.1.weight'].shape[1],
        cross_attention_dim=state['resampler_queries'].shape[-1],num_tokens=state['resampler_queries'].shape[1],
        stage_channels=tuple(state['stage%d.0.weight'%i].shape[0] for i in range(1,5)))
    missing,extra=model.load_state_dict(state,strict=False)
    assert not extra and all(k.startswith('pattern_head.') for k in missing)
    # 只提取E5已训练的前三个CNN层；不使用未训练辅助头。
    return nn.Sequential(model.stage1,model.stage2,model.stage3).cuda().eval().requires_grad_(False)

def mean_mask(x,mask):return (x*mask).sum()/mask.expand_as(x).sum().clamp_min(1)

def orientation(rgb):
    gray=(rgb*rgb.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
    k=gray.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]])[None,None]/8
    gx=F.conv2d(gray,k,padding=1);gy=F.conv2d(gray,k.transpose(-1,-2),padding=1)
    # 纹理切向是梯度法向的轴向反向，输出仍为cos2θ/sin2θ。
    a=F.avg_pool2d(gx.square()-gy.square(),8);b=F.avg_pool2d(2*gx*gy,8)
    return F.normalize(-torch.cat([a,b],1),dim=1,eps=1e-6)

def lab(rgb):
    linear=torch.where(rgb>.04045,((rgb+.055)/1.055).pow(2.4),rgb/12.92)
    m=rgb.new_tensor([[.4124564,.3575761,.1804375],[.2126729,.7151522,.0721750],[.0193339,.1191920,.9503041]])
    xyz=torch.einsum('ij,bjhw->bihw',m,linear)/rgb.new_tensor([.95047,1,1.08883])[None,:,None,None]
    f=torch.where(xyz>.008856,xyz.clamp_min(1e-8).pow(1/3),7.787*xyz+16/116)
    return torch.stack([(116*f[:,1]-16)/100,(500*(f[:,0]-f[:,1])/128+1)/2,(200*(f[:,1]-f[:,2])/128+1)/2],1)

def statistics(rgb,mask):
    n=mask.sum((2,3)).clamp_min(1)
    mu=(rgb*mask).sum((2,3))/n
    std=((rgb.square()*mask).sum((2,3))/n-mu.square()).clamp_min(1e-7).sqrt()
    local=F.avg_pool2d(rgb*mask,16)/F.avg_pool2d(mask,16).clamp_min(.01)
    local_mask=(F.avg_pool2d(mask,16)>=.95).float()
    local_mu=(local*local_mask).sum((2,3))/local_mask.sum((2,3)).clamp_min(1)
    local_std=((local.square()*local_mask).sum((2,3))/local_mask.sum((2,3)).clamp_min(1)-local_mu.square()).clamp_min(1e-7).sqrt()
    return torch.cat([mu,std,local_mu,local_std],1)

def soft_hist(rgb,mask):
    values=lab(F.interpolate(rgb,(64,48),mode='bilinear',align_corners=False))
    weights=F.interpolate(mask,values.shape[-2:],mode='nearest')
    centers=torch.linspace(0,1,8,device=rgb.device)
    bins=torch.exp(-.5*((values[:,:,None]-centers[None,None,:,None,None])/.10).square())
    bins=bins/bins.sum(2,keepdim=True).clamp_min(1e-6)
    return (bins*weights[:,:,None]).sum((3,4))/weights.sum((2,3))[:,:,None].clamp_min(1)

class LearnedLoss:
    def __init__(self):
        self.lpips,self.initialization=frozen_lpips();self.lpips=self.lpips.cuda()
        self.texture=frozen_texture();self.before={'lpips':model_hash(self.lpips),'texture':model_hash(self.texture)}
    def embedding(self,rgb,mask):
        masked=rgb*mask+(1-mask)
        feature=self.texture(F.interpolate(masked,(256,192),mode='bilinear',align_corners=False)*2-1)
        return torch.cat([feature.mean((2,3)),feature.std((2,3))],1)
    def __call__(self,output,references,targets,geom,support,rec_arms,appearance,source_mask):
        mask=geom[:,3:4];valid=F.interpolate(support[None,None].float(),output.shape[-2:],mode='nearest')
        valid=valid*rec_arms[:,None,None,None]
        rec=mean_mask((output-targets).abs(),valid)
        active=rec_arms>0
        pred_masked=output[active]*valid[active]+1-valid[active]
        target_masked=targets[active]*valid[active]+1-valid[active]
        rec=rec+.1*self.lpips(F.interpolate(pred_masked,(256,192),mode='bilinear',align_corners=False)*2-1,
            F.interpolate(target_masked,(256,192),mode='bilinear',align_corners=False)*2-1).mean()
        # 参考前景在数据端固定，模型输出不会参与源patch选择。
        app=(soft_hist(output,mask)-soft_hist(references,source_mask)).abs().mean()
        app=app+(statistics(output,mask)-statistics(references,source_mask)).abs().mean()
        app=app+(self.embedding(output,mask)-self.embedding(references,source_mask)).square().mean()
        out_ori=orientation(output);rf=F.interpolate(geom[:,:2],out_ori.shape[-2:],mode='bilinear',align_corners=False)
        rf=F.normalize(rf,dim=1);q=F.interpolate(geom[:,2:3],out_ori.shape[-2:],mode='nearest')
        valid_ori=(q>=.25)*F.interpolate(mask,out_ori.shape[-2:],mode='nearest')
        ori=mean_mask(1-(out_ori*rf).sum(1,keepdim=True),valid_ori)
        pair_mask=valid_ori[:1]*valid_ori[1:]
        expected=torch.cat([-out_ori[:1],out_ori[:1]],0)
        cf=mean_mask(1-(out_ori[1:]*expected).sum(1,keepdim=True),pair_mask)
        code,tokens=appearance
        inv=(code[1:]-code[:1]).square().mean()+(tokens[1:]-tokens[:1]).square().mean()
        losses=dict(rec=rec,app=app,ori=ori,inv=inv,cf=cf)
        return rec+.5*app+ori+.2*inv+cf,{k:float(v.detach()) for k,v in losses.items()}
    def verify(self):
        after={'lpips':model_hash(self.lpips),'texture':model_hash(self.texture)}
        assert self.before==after and all(p.grad is None for m in [self.lpips,self.texture] for p in m.parameters())
        return dict(before=self.before,after=after,**{'pass':True})
