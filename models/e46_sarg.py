"""Sketch 外轮廓软带；只缩放既有 B1 完成投影后的增量，不新增参数。"""
import cv2
import numpy as np
import torch


def boundary_band(raw, sigma=16.):
    garment=np.asarray(raw)>127
    contours,_=cv2.findContours(garment.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    minimum=max(64.,max((cv2.contourArea(c) for c in contours),default=0.)*.01)
    selected=[c for c in contours if cv2.contourArea(c)>=minimum]
    edge=np.zeros(garment.shape,np.uint8)
    cv2.drawContours(edge,selected,-1,1,1)
    distance=cv2.distanceTransform(1-edge,cv2.DIST_L2,cv2.DIST_MASK_PRECISE) if selected else np.full(garment.shape,1e6,np.float32)
    band=np.exp(-distance**2/(2*sigma**2)).astype('float32')
    small=cv2.resize(band,(24,32),interpolation=cv2.INTER_AREA).clip(0,1)
    return dict(garment=garment,edge=edge,distance=distance,band=band,small=small,
        info=dict(contour_count=len(selected),rejected_count=len(contours)-len(selected),
            minimum_area=minimum,edge_pixels=int(edge.sum()),band_mean=float(band.mean()),
            original_mask_pixels=int(garment.sum()),rgb_shape=list(garment.shape),token_shape=[32,24]))


def gate_from_band(band,alpha=.65,beta=.60):
    return alpha*(1-beta*band)


def scale_residual(base,delta,gate):
    if isinstance(gate,(int,float)) and gate==0:
        return base
    value=torch.as_tensor(gate,device=delta.device,dtype=delta.dtype)
    if value.ndim==2:
        assert delta.shape[1]==value.numel()
        value=value.reshape(1,-1,1)
    return base+delta*value


class ResidualGate:
    def __init__(self,runtime,mode,maps,constant=None,layout_audit=False):
        from tools.e44_b1_mask import region_masks,weights_at
        from PIL import Image
        self.r,self.p=runtime,runtime.processor
        self.mode,self.constant=mode,constant
        self.maps,self.layout_audit=maps,layout_audit
        self.gate=torch.from_numpy(gate_from_band(maps['small'])).to(runtime.pipe.device)
        self.weights=weights_at(region_masks(Image.fromarray(maps['garment'].astype('uint8')*255)),(32,24),runtime.pipe.device)
        self.steps=[];self.handles=[];self.current=None;self.projection=None
        self.layout=[];self.unconditional_calls=0

    def before(self,module,args,kwargs):
        conditional='sa_hidden_states' in (kwargs.get('cross_attention_kwargs') or {})
        if not conditional:
            assert not self.p.enabled
            self.current=None;self.unconditional_calls+=1
            return
        t=args[1] if len(args)>1 else kwargs['timestep']
        self.current=dict(index=len(self.steps),timestep=int(t),enabled=self.p.enabled)
        self.steps.append(self.current)

    def observe(self,base,raw,applied):
        assert self.current is not None and self.p.enabled
        assert raw.shape==(1,768,640) and torch.equal(raw,applied)
        self.base,self.raw=base,raw

    def output(self,module,args,result):
        from tools.e44_b1_mask import regional_stats
        if self.current is None or not self.p.enabled:
            return result
        if self.mode in ['identity','calibration']:
            multiplier=1.
        elif self.mode=='G4':multiplier=self.gate
        elif self.mode=='beta0':multiplier=gate_from_band(self.gate*0,alpha=.65,beta=0.)
        elif self.mode=='zero':multiplier=0.
        else:multiplier=self.constant
        base,raw=self.base,self.raw
        tensor=torch.as_tensor(multiplier,device=raw.device,dtype=raw.dtype)
        if tensor.ndim==2:tensor=tensor.reshape(1,768,1)
        applied=raw*tensor
        ideal=raw.float()*self.gate.float().reshape(1,768,1)
        self.current.update(raw=regional_stats(raw,self.weights),applied=regional_stats(applied,self.weights),
            raw_energy=float(raw.float().square().sum()),applied_energy=float(applied.float().square().sum()),
            hypothetical_sarg_energy=float(ideal.square().sum()),
            gate_min=float(tensor.min()),gate_max=float(tensor.max()))
        assert torch.isfinite(applied).all()
        if self.mode=='G4':assert .259<=float(tensor.min())<=float(tensor.max())<=.651
        self.base=self.raw=None
        if self.mode in ['identity','calibration']:return result
        if self.mode=='zero':return base
        # base 从 observer 直接取得；禁止从已经舍入的 output 中减 delta 重建 base。
        return base+applied

    def capture_projection(self,module,args,result):
        self.projection=result.detach()

    def check_layout(self,module,args):
        tokens=args[0]
        projected=self.projection
        assert projected is not None and projected.shape==(1,640,32,24)
        assert torch.equal(projected.flatten(2).transpose(1,2),tokens), '真实 Transformer 空间/token 顺序不符'
        if len(self.layout)<2:
            y,x=np.unravel_index(int(self.maps['small'].argmax()),(32,24))
            points=[(0,0),(0,23),(31,0),(int(y),int(x))]
            encoded=torch.arange(768,device=tokens.device).reshape(1,1,32,24)
            flat=encoded.flatten(2).transpose(1,2)
            for yy,xx in points:assert int(flat[0,yy*24+xx,0])==yy*24+xx
            self.layout.append(dict(projection_shape=list(projected.shape),token_shape=list(tokens.shape),
                exact_projection_to_tokens=True,coordinate_points=[dict(y=yy,x=xx,token=yy*24+xx) for yy,xx in points]))
        self.projection=None

    def install(self):
        from tools.e44_protocol import LAYER
        modules=dict(self.r.pipe.unet.named_modules())
        self.r.active=True
        self.p.spatial_intervention='global'
        self.p.diagnostic_observer=self.observe
        self.handles.append(self.r.pipe.unet.register_forward_pre_hook(self.before,with_kwargs=True))
        self.handles.append(modules[LAYER[:-10]].register_forward_hook(self.output))
        if self.layout_audit:
            self.handles.append(modules['up_blocks.2.attentions.0.proj_in'].register_forward_hook(self.capture_projection))
            self.handles.append(modules['up_blocks.2.attentions.0.transformer_blocks.0.norm1'].register_forward_pre_hook(self.check_layout))

    def close(self):
        self.p.diagnostic_observer=None;self.r.active=True
        self.base=self.raw=self.projection=None
        for h in self.handles:h.remove()
