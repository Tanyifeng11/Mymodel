"""复用 E44 Writer/原 B1 权重，新增控制器是唯一训练对象。"""
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from models.e51_residual_controller import ControlledB1Processor,ConditionalResidualController
from tools.e44_runtime import Runtime,sketch_masks,digest
from tools.e51_protocol import *


class ControlRuntime(Runtime):
    def __init__(self,arm):
        super().__init__('B1');Runtime.load(self,B1)
        assert self.width==640 and arm in TRAIN_ARMS
        self.variant=arm;self.geometry_cache={}
        self.processor.requires_grad_(False)
        self.processor.__class__=ControlledB1Processor
        p=self.processor
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42);p.controller=ConditionalResidualController().to(self.pipe.device)
        p.timed=arm=='TIME';p.residual_scale=1.;p.raw_bypass=False;p.geometry=None
        p.last_control=None;p.capture_maps=False;p.control_maps=[]
        self.frozen_b1_before=self.b1_hashes()
        write(OUT/'manifests'/('model_'+arm+'.json'),dict(arm=arm,parameters=sum(p.numel() for p in self.parameters()),
            frozen_B1=self.frozen_b1_before,config=CONFIG,commit=commit()))

    def parameters(self):
        if self.processor is not None and hasattr(self.processor,'controller'):return list(self.processor.controller.parameters())
        return []

    def b1_hashes(self):
        return {n:digest(p) for n,p in self.processor.state_dict().items() if not n.startswith(('original.','controller.'))}

    def geometry(self,path):
        path=str(path)
        if path not in self.geometry_cache:
            raw,_,info=sketch_masks(Image.open(path).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR),'cpu')
            foreground=(np.asarray(raw)>127).astype('uint8');kernel=np.ones((19,19),'uint8')
            edge=(cv2.dilate(foreground,kernel)>0)&~(cv2.erode(foreground,kernel)>0)
            distance=np.clip(cv2.distanceTransform(foreground,cv2.DIST_L2,5)/32,0,1)
            maps=np.stack([foreground.astype('float32'),edge.astype('float32'),distance.astype('float32')])
            self.geometry_cache[path]=F.interpolate(torch.from_numpy(maps)[None],size=(64,48),mode='area').to(self.pipe.device)
        self.processor.geometry=self.geometry_cache[path]

    def before_unet(self,module,args,kwargs):
        super().before_unet(module,args,kwargs)
        if self.processor is not None:
            self.processor.timestep=torch.as_tensor(args[1] if len(args)>1 else kwargs['timestep'],device=self.pipe.device).reshape(-1)

    def generate(self,row,path,**kwargs):
        self.geometry(row['sketch']);self.processor.trace=[]
        self.processor.capture_maps=True;self.processor.control_maps=[]
        steps=[];old_forward=self.pipe.unet.forward;p=self.processor
        def forward(*args,**kw):
            conditional='sa_hidden_states' in (kw.get('cross_attention_kwargs') or {})
            p.last_control=None
            if not conditional:return old_forward(*args,**kw)
            enabled=p.enabled
            try:
                p.enabled=False
                with torch.no_grad():teacher=old_forward(*args,**kw)
            finally:p.enabled=enabled
            output=old_forward(*args,**kw)
            value=dict(p.last_control) if p.last_control else dict(timestep=int(p.timestep),branch_off=True)
            effect=output[0].float()-teacher[0].float()
            edge=F.interpolate(p.geometry[:,1:2],size=effect.shape[-2:],mode='area')
            outside=1-F.interpolate(self.mask.float(),size=effect.shape[-2:],mode='area')
            value.update(epsilon_rms=float(effect.square().mean().sqrt()),
                epsilon_boundary_rms=float(((effect.square()*edge).sum()/(edge.sum()*effect.shape[1]).clamp_min(1e-8)).sqrt()),
                epsilon_outside_rms=float(((effect.square()*outside).sum()/(outside.sum()*effect.shape[1]).clamp_min(1e-8)).sqrt()))
            steps.append(value);return output
        self.pipe.unet.forward=forward
        try:result=super().generate(row,path,**kwargs)
        finally:self.pipe.unet.forward=old_forward;p.capture_maps=False
        assert len(steps)==50
        path=Path(path)
        write(path.with_name(path.stem+'_control.json'),dict(arm=self.variant,steps=steps,scale=p.residual_scale,raw_bypass=p.raw_bypass))
        for index in [0,5,15,25,49]:
            if index<len(p.control_maps):
                image=(p.control_maps[index][0,0].numpy()*255).round().clip(0,255).astype('uint8')
                Image.fromarray(image).save(path.with_name(path.stem+'_gate%02d.png'%index))
        return result

    def save(self,path,step,optimizer=None,scaler=None):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(variant=self.variant,step=step,config=CONFIG,state_dict=self.processor.controller.state_dict(),
            optimizer=optimizer.state_dict() if optimizer else None,scaler=scaler.state_dict() if scaler else None),path)
        write(path.with_suffix('.json'),dict(step=step,sha256=sha(path),commit=commit()))

    def load(self,path):
        state=torch.load(path,map_location='cpu',weights_only=False)
        assert state['config']==CONFIG and state['variant']==self.variant
        self.processor.controller.load_state_dict(state['state_dict']);return state

    def close(self):
        assert self.b1_hashes()==self.frozen_b1_before
        write(OUT/'manifests'/('frozen_B1_'+self.variant+'.json'),dict(unchanged=True,before=self.frozen_b1_before,after=self.b1_hashes()))
        super().close()


def make_runtime(arm,namespace=None):
    from tools import e44_runtime as module
    from tools.e47_runtime import load_base
    from models import e33tm_generation_wrapper as loader
    old_loader,old_out=loader.load_e5,module.OUT
    module.OUT=OUT/(namespace or ('training/'+arm+'/runtime'));loader.load_e5=lambda:load_base('B')
    try:r=ControlRuntime(arm)
    finally:loader.load_e5=old_loader
    r.e47_original_out=old_out;r.pipe.use_tcpm_lite=True
    return r


def close(r):
    from tools.e47_runtime import close as release
    release(r)
