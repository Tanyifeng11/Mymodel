"""审计真正的 decoder 汇合点；原生不可用时移植官方 spatial-backbone + FFT。"""
import ast
import collections
import inspect
import textwrap
from types import MethodType
import torch


class FreeUBridge:
    def __init__(self, unet):
        self.unet=unet;self.cfg=None;self.branch='unknown';self.calls=collections.Counter();self.details=[]
        self.limit=2;self.globals=[];self.original={};self.patched={};self.handles=[]
        self.stages={id(b):i for i,b in enumerate(unet.up_blocks)}
        assert len(unet.up_blocks)>=2
        self.sources={str(i):inspect.getsource(type(b).forward) for i,b in enumerate(unet.up_blocks)}
        self.native=hasattr(unet,'enable_freeu') and hasattr(unet,'disable_freeu')
        for b in unet.up_blocks[:2]:
            ns=type(b).forward.__globals__
            if 'apply_freeu' not in ns or 'apply_freeu' not in inspect.getsource(type(b).forward): self.native=False
        if self.native:
            for b in unet.up_blocks[:2]:
                ns=type(b).forward.__globals__
                if any(ns is x[0] for x in self.globals):continue
                original=ns['apply_freeu'];self.globals.append((ns,original))
                def audited(stage,h,skip,*args,_original=original,**kwargs):
                    return self.apply(stage,h,skip,lambda a,c:_original(stage,a,c,*args,**kwargs))
                ns['apply_freeu']=audited
            self.mode='native_diffusers'
            self.operation_source=inspect.getsource(self.globals[0][1])
        else:
            self.mode='ported_official_spatial_freeu'
            for stage,block in enumerate(unet.up_blocks[:2]):
                # 只在实际 torch.cat([hidden_states,res_hidden_states]) 之前插入；保留整个原 forward。
                tree=ast.parse(textwrap.dedent(self.sources[str(stage)]));count=[0]
                class Insert(ast.NodeTransformer):
                    pass
                # Python3.8 无 ast.unparse：使用节点名称判断而非字符串替换。
                def matches(node):
                    return isinstance(node,(ast.List,ast.Tuple)) and [getattr(n,'id',None) for n in node.elts]==['hidden_states','res_hidden_states']
                def visit_assign(this,node):
                    this.generic_visit(node)
                    if (isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute)
                        and node.value.func.attr=='cat' and node.value.args and matches(node.value.args[0])):
                        count[0]+=1
                        return [ast.parse('hidden_states, res_hidden_states = _e35_apply(self, hidden_states, res_hidden_states)').body[0],node]
                    return node
                Insert.visit_Assign=visit_assign
                tree=Insert().visit(tree);ast.fix_missing_locations(tree)
                assert count[0]==1, '无法唯一确定 concat，拒绝接入'
                ns=dict(type(block).forward.__globals__)
                ns['_e35_apply']=lambda b,h,s:self.port(self.stages[id(b)],h,s)
                exec(compile(tree,'<E35 audited pre-concat port>','exec'),ns)
                self.original[stage]=block.forward
                self.patched[stage]=MethodType(ns['forward'],block)
            self.operation_source=inspect.getsource(self.port)
        def branch(_module,args,kwargs):
            self.branch='conditional' if 'sa_hidden_states' in (kwargs.get('cross_attention_kwargs') or {}) else 'unconditional'
        self.handles.append(unet.register_forward_pre_hook(branch,with_kwargs=True))

    def reset(self,limit=2):
        self.calls.clear();self.details=[];self.limit=limit

    def set(self,cfg):
        self.cfg=cfg
        if self.native:
            self.unet.disable_freeu()
            if cfg is not None:self.unet.enable_freeu(**cfg)
        else:
            for i in self.original:
                self.unet.up_blocks[i].forward=self.original[i] if cfg is None else self.patched[i]

    def apply(self,stage,h,skip,operation):
        if stage not in (0,1):return operation(h,skip)
        key='%d/%s'%(stage,self.branch);self.calls[key]+=1
        sample=sum(d['key']==key for d in self.details)<self.limit
        if sample:old_h=h.clone();old_skip=skip.clone()
        result=operation(h,skip)
        if sample:
            a,b=result
            self.details.append(dict(key=key,stage=stage,branch=self.branch,
                block_path='up_blocks.%d'%stage,h_shape=list(h.shape),skip_shape=list(skip.shape),
                h_mae=float((a.float()-old_h.float()).abs().mean()),
                skip_mae=float((b.float()-old_skip.float()).abs().mean()),
                finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())))
        return result

    def port(self,stage,h,skip):
        cfg=self.cfg
        if cfg is None:return h,skip
        def operation(h,skip):
            # 官方实现按每张图的 spatial mean min/max 增强前半 backbone 通道。
            mean=h.mean(1,keepdim=True);lo=mean.flatten(1).min(1)[0][:,None,None,None]
            hi=mean.flatten(1).max(1)[0][:,None,None,None]
            weight=(mean-lo)/(hi-lo).clamp_min(torch.finfo(mean.dtype).eps)
            h=h.clone();half=h.shape[1]//2
            h[:,:half]=h[:,:half]*((cfg['b%d'%(stage+1)]-1)*weight+1)
            # 384x512 latent 的非2幂尺寸要求局部FP32 FFT，再回到原dtype。
            spectrum=torch.fft.fftshift(torch.fft.fftn(skip.float(),dim=(-2,-1)),dim=(-2,-1))
            y,x=skip.shape[-2]//2,skip.shape[-1]//2
            spectrum[...,y-1:y+1,x-1:x+1]*=cfg['s%d'%(stage+1)]
            out=torch.fft.ifftn(torch.fft.ifftshift(spectrum,dim=(-2,-1)),dim=(-2,-1)).real
            return h,out.to(skip.dtype)
        return self.apply(stage,h,skip,operation)

    def structure(self):
        return [dict(stage=i,path='up_blocks.%d'%i,type=type(b).__module__+'.'+type(b).__name__,
            resolution_idx=getattr(b,'resolution_idx',None),resnet_channels=[
                dict(input=r.in_channels,output=r.out_channels) for r in b.resnets],
            selected=i in (0,1),gradient_checkpointing=getattr(b,'gradient_checkpointing',None))
            for i,b in enumerate(self.unet.up_blocks)]

    def report(self):return dict(mode=self.mode,operation_count=dict(self.calls),operations=self.details)

    def close(self):
        self.set(None)
        for ns,original in self.globals:ns['apply_freeu']=original
        for handle in self.handles:handle.remove()
