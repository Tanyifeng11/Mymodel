"""附加监督；历史 base、x0 解码、训练冻结与更新逻辑由原训练脚本提供。"""
import importlib.util
import sys
import numpy as np
import torch
import torch.nn.functional as F
from tools.e2a_sup_common import CONFIG, ROOT, read, sha


class Supervision:
    def __init__(self, arm, helpers):
        # 只保存静态算子；helpers 含当前训练 locals，保留它会锁住首批完整计算图。
        self.arm=arm
        self.h={name:helpers[name] for name in ['build_region_masks','texture_color_stat_loss','masked_l1_loss']}
        self.network=None
        if arm not in ['T1','T2']: return
        assets=ROOT/'assets'; registry=read(assets/'registry.json')[arm]
        for path,digest in registry['files'].items(): assert sha(path)==digest
        if arm=='T1':
            path=assets/'DISTS/DISTS_pytorch/DISTS_pt.py'
            spec=importlib.util.spec_from_file_location('sup_official_dists',path)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            self.network=module.DISTS(load_weights=False)
            weights=torch.load(registry['alpha_beta'],map_location='cpu',weights_only=False)
            with torch.no_grad():
                self.network.alpha.copy_(weights['alpha']);self.network.beta.copy_(weights['beta'])
        else:
            # 官方 vision_transformer 从顶级 utils 导入，必须隔离仓库同名 utils。
            repo=assets/'dino'
            modules={}
            for name in ['utils','vision_transformer']:
                spec=importlib.util.spec_from_file_location('sup_official_dino_'+name,repo/(name+'.py'))
                module=importlib.util.module_from_spec(spec)
                previous=sys.modules.get('utils')
                try:
                    if name=='vision_transformer':sys.modules['utils']=modules['utils']
                    spec.loader.exec_module(module)
                finally:
                    if previous is None:sys.modules.pop('utils',None)
                    else:sys.modules['utils']=previous
                modules[name]=module
            self.network=modules['vision_transformer'].vit_small(patch_size=16,num_classes=0)
            weights=torch.load(registry['weights'],map_location='cpu',weights_only=False)
            self.network.load_state_dict(weights,strict=True)
        self.network.to(helpers['accelerator'].device,dtype=torch.float32).eval().requires_grad_(False)

    @staticmethod
    def roi(image,mask):
        outputs=[]
        for im,m in zip(image,mask):
            indices=torch.nonzero(m[0]>0,as_tuple=False)
            if indices.numel():
                lo=indices.min(0).values;hi=indices.max(0).values+1
                im=im[:,lo[0]:hi[0],lo[1]:hi[1]]
            h,w=im.shape[-2:]; side=max(h,w)
            im=F.pad(im,((side-w)//2,side-w-(side-w)//2,(side-h)//2,side-h-(side-h)//2),value=1)
            outputs.append(F.interpolate(im[None],size=(224,224),mode='bilinear',align_corners=False))
        return torch.cat(outputs)

    def dino(self,image):
        captured=[]
        layer=self.network.blocks[CONFIG['dino_key_block_zero_based']].attn.qkv
        hook=layer.register_forward_hook(lambda module,args,output:captured.append(output))
        mean=image.new_tensor([.485,.456,.406])[None,:,None,None]
        std=image.new_tensor([.229,.224,.225])[None,:,None,None]
        try: cls=self.network((image-mean)/std)
        finally: hook.remove()
        # 官方 qkv: [B,N,3*384]，keys 去 CLS，连接 6 个 head 后计算 token 余弦矩阵。
        qkv=captured[0]; keys=qkv.reshape(qkv.shape[0],qkv.shape[1],3,6,64)[:,:,1].flatten(2)[:,1:]
        keys=F.normalize(keys,dim=-1)
        return F.normalize(cls,dim=-1), keys@keys.transpose(-1,-2)

    def raw(self, pred, gt, reference, mask, timestep, texture_weight):
        body,boundary,outside=self.h['build_region_masks'](mask,9)
        assert (body+boundary+outside-1).abs().max().item()<1e-5
        if self.arm=='T3':
            bodyloss=pred.sum()*0
            if texture_weight.sum().item()>0:
                bodyloss=self.h['texture_color_stat_loss'](pred,reference,body,sample_weight=texture_weight)
            return dict(body_rgb=bodyloss,rest=self.h['masked_l1_loss'](pred,gt,1-body))
        active=timestep<=CONFIG['eta']; batch=pred.shape[0]
        zero=pred.flatten(1).sum(1)*0
        names=['perceptual','high_frequency'] if self.arm=='T1' else ['appearance','structure']
        result={name:zero.clone() for name in names}
        if not active.any():return {k:v.mean() for k,v in result.items()}
        p=((pred[active]+1)/2).clamp(0,1);g=((gt[active]+1)/2).clamp(0,1)
        garment=(body+boundary)[active]
        if self.arm=='T1':
            import cv2
            perc=self.network(p*garment,g*garment,require_grad=True,batch_average=False).reshape(-1)
            edges=[]
            for image in g:
                rgb=(image.detach().permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
                edge=cv2.Canny(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),100,200,apertureSize=3,L2gradient=False)
                edges.append(torch.from_numpy(edge.astype(np.float32)/255))
            edge=torch.stack(edges).to(p.device)[:,None]*garment
            hf=torch.linalg.vector_norm(((p-g)*edge).flatten(1),dim=1)/(3*edge.flatten(1).sum(1).clamp_min(1)).sqrt()
            result['perceptual'][active]=perc;result['high_frequency'][active]=hf
        else:
            p=self.roi(p,garment);g=self.roi(g,garment)
            ref=F.interpolate(((reference[active]+1)/2).clamp(0,1),size=(224,224),mode='bilinear',align_corners=False)
            cls_p,sim_p=self.dino(p)
            with torch.no_grad(): cls_r,_=self.dino(ref);_,sim_g=self.dino(g)
            result['appearance'][active]=torch.linalg.vector_norm(cls_p-cls_r,dim=-1)
            result['structure'][active]=(sim_p-sim_g).square().mean(dim=(-1,-2))
        # 固定完整 batch 的 mean；不按 active 样本个数重新放大。
        return {k:v.mean() for k,v in result.items()}


def weighted(raw, coefficients):
    return sum(raw[k]*v for k,v in coefficients.items())
