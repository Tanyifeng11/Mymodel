"""复用冻结E5/Writer；新模块和产物仅落M54目录。"""
import gc
import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms.functional import to_tensor
from models.m54_candidates import Candidate,M54Processor
from tools.e44_runtime import Runtime,sketch_masks,digest
from tools.m54_protocol import *


class M54Runtime(Runtime):
    def __init__(self,arm):
        recipe=arm
        arm='M5' if arm.startswith('M5_S') else arm
        super().__init__('B0' if arm=='E5' else 'B1')
        self.arm=arm;self.recipe=recipe;self.variant=recipe;self.geometry_cache={};self.prior_cache={};self.audit_steps=[]
        self.effects_enabled=True;self.counter_latent=None;self.counter_id=None
        if self.processor is None:return
        p=self.processor
        if arm not in TRAIN_ARMS or arm=='M9':Runtime.load(self,B1)
        p.requires_grad_(False);p.__class__=M54Processor;p.mode=arm
        p.geometry=None;p.timestep=torch.tensor([0],device=self.pipe.device);p.spatial_prior=None
        p.last_delta={};p.last_attention={}
        p.new_branch_active=True
        p.effect_scale={'M5_S075':.75,'M5_S100':1.,'M5_S125':1.25}.get(recipe,1.)
        if arm=='SAFE':p.gate.data.mul_(SAFE_SCALE)
        if arm in TRAIN_ARMS:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42);p.candidate=Candidate(arm,self.width).to(self.pipe.device)
            choice=OUT/'round1/selection.json'
            if arm=='M1' and choice.exists():p.candidate.tau=read(choice)['M1_temperature']
        p.diagnostic_observer=lambda base,raw,delta:setattr(p,'last_delta',dict(rms=float(((base+delta)-base).detach().float().square().mean().sqrt())))

    def parameters(self):
        if self.processor is not None and hasattr(self.processor,'candidate'):return list(self.processor.candidate.parameters())
        return []

    def geometry(self,sketch_path,reference_path=None):
        if self.processor is None:return
        key=str(sketch_path)
        if key not in self.geometry_cache:
            raw,_,_=sketch_masks(Image.open(sketch_path).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR),'cpu')
            fg=(np.asarray(raw)>127).astype('uint8');kernel=np.ones((19,19),'uint8')
            edge=(cv2.dilate(fg,kernel)>0)&~(cv2.erode(fg,kernel)>0)
            distance=np.clip(cv2.distanceTransform(fg,cv2.DIST_L2,5)/32,0,1)
            maps=np.stack([fg,edge,distance]).astype('float32')
            self.geometry_cache[key]=F.interpolate(torch.from_numpy(maps)[None],size=(32,24),mode='area').to(self.pipe.device)
        self.processor.geometry=self.geometry_cache[key]
        if self.arm=='M6' and reference_path is not None:
            key=(str(sketch_path),str(reference_path))
            if key not in self.prior_cache:
                mask=self.processor.geometry[0,0].cpu().numpy()>.1
                # 固定白底前景规则，仅使用参考输入，不读取GT或目标生成图。
                a=np.asarray(Image.open(reference_path).convert('RGB').resize(SIZE),np.float32)
                ref=(a.min(-1)<235)
                def coords(h,w,support):
                    yy,xx=np.where(support)
                    if len(xx)==0:return np.stack(np.meshgrid(np.linspace(0,1,w),np.linspace(0,1,h)),axis=-1).reshape(-1,2)
                    y,x=np.meshgrid((np.arange(h)+.5)*support.shape[0]/h,(np.arange(w)+.5)*support.shape[1]/w,indexing='ij')
                    return np.stack([(x-xx.min())/max(xx.max()-xx.min(),1),(y-yy.min())/max(yy.max()-yy.min(),1)],-1).reshape(-1,2)
                t=coords(32,24,mask);r=coords(16,16,ref)
                prior=np.exp(-((t[:,None]-r[None])**2).sum(-1)/(2*.15**2)).astype('float32')
                self.prior_cache[key]=torch.from_numpy(prior)[None].to(self.pipe.device)
            self.processor.spatial_prior=self.prior_cache[key]

    def before_unet(self,module,args,kwargs):
        super().before_unet(module,args,kwargs)
        if self.processor is not None:
            self.processor.timestep=torch.as_tensor(args[1] if len(args)>1 else kwargs['timestep'],device=self.pipe.device).reshape(-1)

    @torch.no_grad()
    def counter_reference(self,row):
        pixels=to_tensor(Image.open(row['reference']).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR))[None].to(self.pipe.device)*2-1
        self.counter_latent=self.pipe.vae.encode(pixels.to(self.pipe.vae.dtype)).latent_dist.mean*self.pipe.vae.config.scaling_factor
        self.counter_id=row['id']

    def generate(self,row,path,**kwargs):
        self.geometry(row['sketch'],row['reference'])
        if self.arm=='ORACLE':
            from tools.e53_mapping import matrices
            assert row.get('geometry') and row.get('synthetic')
            # 本诊断Writer读取G0T0，矩阵必须匹配G0，不能误用E53默认GN。
            self.processor.oracle=torch.from_numpy(matrices(row,geometry='0')[0]).to(self.pipe.device)
        old=self.pipe.unet.forward;steps=[];p=self.processor
        def forward(*args,**kw):
            conditional='sa_hidden_states' in (kw.get('cross_attention_kwargs') or {})
            if not conditional:return old(*args,**kw)
            if not self.effects_enabled:
                output=old(*args,**kw);steps.append(dict(step=len(steps),epsilon_shift_rms=None));return output
            enabled=p.enabled if p is not None else False
            if p is not None:p.enabled=False
            try:
                with torch.no_grad():teacher=old(*args,**kw)
            finally:
                if p is not None:p.enabled=enabled
            if p is not None:p.last_delta={}
            output=old(*args,**kw)
            effect=output[0].float()-teacher[0].float()
            item=dict(step=len(steps),epsilon_shift_rms=float(effect.square().mean().sqrt()),
                **(dict(p.last_delta) if p is not None else dict(rms=0.)))
            if self.counter_latent is not None and len(steps) in [0,24,49]:
                features=p.features;latent=self.writer.latent
                try:
                    self.writer.latent=self.counter_latent;p.features=self.writer.write(p.timestep)
                    counter=old(*args,**kw)[0].float()
                    item['reference_sensitivity_rms']=float((counter-output[0].float()).square().mean().sqrt())
                finally:p.features=features;self.writer.latent=latent
            steps.append(item)
            return output
        self.pipe.unet.forward=forward
        try:record,contexts,outputs=super().generate(row,path,**kwargs)
        finally:self.pipe.unet.forward=old
        assert len(steps)==50
        shifts=[s['epsilon_shift_rms'] for s in steps if s['epsilon_shift_rms'] is not None]
        sensitivity=[s['reference_sensitivity_rms'] for s in steps if 'reference_sensitivity_rms' in s]
        record.update(module=self.recipe,effect_scale=getattr(self.processor,'effect_scale',1.),production_profile=not self.effects_enabled,
            parameters=sum(v.numel() for v in self.parameters()),
            delta_rms=float(np.mean([s.get('rms',0.) for s in steps])),
            epsilon_shift_rms=float(np.mean(shifts)) if shifts else None,
            reference_sensitivity_rms=float(np.mean(sensitivity)) if sensitivity else None,counter_reference_id=self.counter_id)
        write(Path(path).with_suffix('.json'),record)
        write(Path(path).with_name(Path(path).stem+'_effects.json'),steps)
        return record,contexts,outputs

    def save(self,path,step,optimizer=None,scaler=None):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(arm=self.arm,step=step,config=CONFIG,state_dict=self.processor.candidate.state_dict(),
            attention_temperature=self.processor.candidate.tau,
            optimizer=optimizer.state_dict() if optimizer else None,scaler=scaler.state_dict() if scaler else None),path)
        write(path.with_suffix('.json'),dict(step=step,sha256=sha(path),commit=commit()))

    def load(self,path):
        state=torch.load(path,map_location='cpu',weights_only=False)
        assert state['config']==CONFIG and state['arm']==self.arm
        assert state.get('attention_temperature',1.)==self.processor.candidate.tau
        self.processor.candidate.load_state_dict(state['state_dict']);return state


def make_runtime(arm,namespace):
    from tools import e44_runtime as module
    from models import e33tm_generation_wrapper as loader
    from tools.e47_runtime import load_base
    old_loader,old_out,old_layer=loader.load_e5,module.OUT,module.LAYER
    layer=LAYER_ARMS.get(arm,LAYER)
    module.OUT=OUT/namespace;module.LAYER=layer;loader.load_e5=lambda:load_base('B')
    try:r=M54Runtime(arm)
    finally:loader.load_e5=old_loader;module.OUT=old_out;module.LAYER=old_layer
    r.m54_runtime_out=OUT/namespace;r.m54_layer=layer
    r.pipe.use_tcpm_lite=True
    return r


def release(r):
    from tools import e44_runtime as module
    old_out,old_layer=module.OUT,module.LAYER
    module.OUT=r.m54_runtime_out;module.LAYER=r.m54_layer
    try:r.close()
    finally:module.OUT=old_out;module.LAYER=old_layer
    gc.collect();torch.cuda.empty_cache()
