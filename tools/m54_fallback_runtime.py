"""已批准的两条后备分支；冻结E5主路，检查点只保存新增参数。"""
import copy
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms.functional import to_tensor
from models.m54_fallback import CatBranch,WarpBranch,reference_add_noise
from tools.e44_runtime import Runtime
from tools.m54_runtime import M54Runtime
from tools.m54_protocol import *

ARMS=['M_L1','M_L2']
REFERENCE_NOISE_SEED=42001


def clone_without_hooks(module):
    # 原运行时的hook持有管线闭包，只复制模型权重/处理器，不能把源管线带进新分支。
    saved=[]
    for child in module.modules():
        for name in ['_forward_hooks','_forward_pre_hooks']:
            value=getattr(child,name);saved.append((child,name,value));setattr(child,name,type(value)())
    try:return copy.deepcopy(module)
    finally:
        for child,name,value in saved:setattr(child,name,value)


class FallbackRuntime(M54Runtime):
    def __init__(self,arm):
        assert arm in ARMS
        super().__init__('E5');self.arm=arm;self.recipe=arm;self.variant=arm
        self.sketch_latent=None;self.reference_latent=None;self.record_effects=False;self.effects=[]
        self.smoothness=None
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(42);torch.cuda.manual_seed_all(42)
            self.candidate=CatBranch(clone_without_hooks(self.pipe.unet)) if arm=='M_L1' else WarpBranch().to(self.pipe.device)
        self.candidate.eval();self.original_forward=self.pipe.unet.forward
        self.original_alpha=self.pipe.scheduler.alphas_cumprod.detach().clone()
        self.pipe.unet.forward=self.enhanced_forward

    def parameters(self):
        return list(self.candidate.parameters()) if hasattr(self,'candidate') else []

    @torch.no_grad()
    def encode(self,pixels):
        return self.pipe.vae.encode(pixels.to(self.pipe.vae.dtype)).latent_dist.mean*self.pipe.vae.config.scaling_factor

    @torch.no_grad()
    def geometry(self,sketch_path,reference_path=None):
        key=str(sketch_path)
        if key not in self.geometry_cache:
            image=Image.open(sketch_path).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR)
            pixels=to_tensor(image)[None].to(self.pipe.device)*2-1
            self.geometry_cache[key]=self.encode(pixels)
        self.sketch_latent=self.geometry_cache[key]

    @torch.no_grad()
    def set_reference(self,pixels,mask,mask_info,allow=True):
        Runtime.set_reference(self,pixels,mask,mask_info,allow)
        self.reference_latent=self.encode(pixels) if allow and mask_info['valid'] else None
        if self.reference_latent is not None:
            self.reference_noise=torch.randn(self.reference_latent.shape,device=self.pipe.device,
                dtype=self.reference_latent.dtype,generator=torch.Generator(device=self.pipe.device).manual_seed(REFERENCE_NOISE_SEED))

    def enhanced_forward(self,*args,**kwargs):
        self.smoothness=None
        with torch.no_grad():output=self.original_forward(*args,**kwargs)
        conditional='sa_hidden_states' in (kwargs.get('cross_attention_kwargs') or {})
        if not conditional:return output
        enabled=self.active and self.allow_reference and self.mask_info['valid']
        delta=torch.zeros_like(output[0])
        if enabled:
            noisy=args[0] if args else kwargs['sample'];t=args[1] if len(args)>1 else kwargs['timestep']
            mask=F.interpolate(self.mask,size=noisy.shape[-2:],mode='area').to(noisy.dtype)
            assert self.sketch_latent is not None and self.reference_latent is not None
            # 主参数保留FP32；推理入口不保证已有autocast，分支自行使用同一FP16计算路径。
            with torch.autocast('cuda',dtype=torch.float16):
                if self.arm=='M_L1':
                    noisy_ref=reference_add_noise(self.reference_latent,self.reference_noise,t,self.pipe.scheduler.alphas_cumprod)
                    delta=self.candidate(noisy,noisy_ref,self.sketch_latent,self.reference_latent,mask,t,
                        kwargs['encoder_hidden_states'],kwargs.get('cross_attention_kwargs'))*mask
                else:
                    delta=self.candidate(noisy,self.sketch_latent,self.reference_latent,mask,t)*mask
                    self.smoothness=self.candidate.smoothness
        prediction=output[0]+delta.to(output[0].dtype)
        if self.record_effects:
            shift=prediction.detach().float()-output[0].float()
            self.effects.append(dict(step=len(self.effects),enabled=bool(enabled),
                epsilon_shift_rms=float(shift.square().mean().sqrt()),rms=float(delta.detach().float().square().mean().sqrt())))
        return (prediction,)+output[1:] if isinstance(output,tuple) else type(output)(sample=prediction)

    def generate(self,row,path,**kwargs):
        self.geometry(row['sketch'],row['reference']);self.effects=[];self.record_effects=True
        try:record,contexts,outputs=Runtime.generate(self,row,path,**kwargs)
        finally:self.record_effects=False
        assert len(self.effects)==50
        record.update(module=self.arm,parameters=sum(p.numel() for p in self.parameters()),
            epsilon_shift_rms=sum(x['epsilon_shift_rms'] for x in self.effects)/50,
            delta_rms=sum(x['rms'] for x in self.effects)/50,reference_noise_seed=REFERENCE_NOISE_SEED,
            writer_used=False,b1joint=False,fallback_protocol_sha256=sha(OUT/'fallback/protocol.json'))
        write(Path(path).with_suffix('.json'),record)
        write(Path(path).with_name(Path(path).stem+'_effects.json'),self.effects)
        return record,contexts,outputs

    def save(self,path,step,optimizer=None,scaler=None):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(arm=self.arm,step=step,config=read(OUT/'fallback/protocol.json'),
            state_dict=self.candidate.state_dict(),parameter_names=[n for n,_ in self.candidate.named_parameters()],
            cat_parameter_names=getattr(self.candidate,'parameter_names',None),reference_noise_seed=REFERENCE_NOISE_SEED,
            optimizer=optimizer.state_dict() if optimizer else None,scaler=scaler.state_dict() if scaler else None),path)
        write(path.with_suffix('.json'),dict(step=step,sha256=sha(path),commit=commit()))

    def load(self,path):
        path=Path(path);assert sha(path)==read(path.with_suffix('.json'))['sha256']
        state=torch.load(path,map_location='cpu',weights_only=False)
        assert state['arm']==self.arm and state['config']==read(OUT/'fallback/protocol.json')
        assert state['cat_parameter_names']==getattr(self.candidate,'parameter_names',None)
        assert state['reference_noise_seed']==REFERENCE_NOISE_SEED
        self.candidate.load_state_dict(state['state_dict']);return state

    def close(self):
        self.pipe.unet.forward=self.original_forward
        alpha=self.pipe.scheduler.alphas_cumprod
        assert alpha.device==self.original_alpha.device and alpha.dtype==self.original_alpha.dtype
        assert torch.equal(alpha,self.original_alpha),'FROZEN_E5_SCHEDULER_ALPHA_CHANGED'
        if self.arm=='M_L1':self.candidate.verify_frozen()
        super().close()


def make_runtime(arm,namespace):
    from tools import e44_runtime as module
    from models import e33tm_generation_wrapper as loader
    from tools.e47_runtime import load_base
    old_loader,old_out,old_layer=loader.load_e5,module.OUT,module.LAYER
    module.OUT=OUT/namespace;module.LAYER=LAYER;loader.load_e5=lambda:load_base('B')
    try:r=FallbackRuntime(arm)
    finally:loader.load_e5=old_loader;module.OUT=old_out;module.LAYER=old_layer
    r.m54_runtime_out=OUT/namespace;r.m54_layer=LAYER;r.pipe.use_tcpm_lite=True
    return r
