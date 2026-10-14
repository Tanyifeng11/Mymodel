"""e48 同状态因果干预、softmax 分解和独立基线 latent 融合。"""
import hashlib
import math
import torch
from torch.nn import functional as F
from tools.e48_protocol import *
from tools.e44_b1_mask import region_masks,weights_at,regional_stats
def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def projection(attn,value):
    b,h,n,d=value.shape
    value=value.transpose(1,2).reshape(b,n,h*d)
    value=F.linear(value,attn.to_out[0].weight,None)
    return attn.to_out[1](value)/attn.rescale_output_factor


def split(value,heads):
    b,n,c=value.shape
    return value.reshape(b,n,heads,c//heads).transpose(1,2)


@torch.no_grad()
def decomposition(attn,p,cache):
    q=cache['q'];dtype=q.dtype
    qh=split(q,attn.heads)
    ks,vs=split(cache['k'],attn.heads),split(cache['v'],attn.heads)
    kr,vr=split(cache['kr'].to(dtype),attn.heads),split(cache['vr'].to(dtype),attn.heads)
    scale=math.sqrt(qh.shape[-1])
    self_score=qh.float()@ks.float().transpose(-1,-2)/scale
    ref_score=qh.float()@kr.float().transpose(-1,-2)/scale
    probability=torch.sigmoid(torch.logsumexp(ref_score,-1)-torch.logsumexp(self_score,-1))
    own=F.scaled_dot_product_attention(qh,ks,vs)
    reference=F.scaled_dot_product_attention(qh,kr,vr)
    joint=F.scaled_dot_product_attention(qh,torch.cat([ks,kr],2),torch.cat([vs,vr],2))
    reconstructed=(1-probability[...,None])*own.float()+probability[...,None]*reference.float()
    relative=float((reconstructed-joint.float()).norm()/joint.float().norm().clamp_min(1e-8))
    # FP16 fused attention has independent rounding; analytical identity is checked in FP32 as well.
    own32=torch.softmax(self_score,-1)@vs.float()
    ref32=torch.softmax(ref_score,-1)@vr.float()
    joint32=torch.softmax(torch.cat([self_score,ref_score],-1),-1)@torch.cat([vs.float(),vr.float()],2)
    exact32=(1-probability[...,None])*own32+probability[...,None]*ref32
    error32=float((joint32-exact32).abs().max())
    assert error32<2e-4 and relative<.01,(error32,relative)
    coefficient=(probability.to(dtype)*p.gate).unsqueeze(-1)
    added=projection(attn,coefficient*reference)
    suppressed=projection(attn,-coefficient*own)
    cosine=float(F.cosine_similarity(added.float().flatten(),suppressed.float().flatten(),dim=0))
    return dict(added=added,suppressed=suppressed),dict(
        gate=float(p.gate),p_ref_mean=float(probability.mean()),
        p_ref_quantiles=torch.quantile(probability.flatten(),torch.tensor([.05,.5,.95],device=q.device)).tolist(),
        joint_fp32_max_error=error32,joint_fp16_relative_error=relative,
        added_rms=float(added.float().square().mean().sqrt()),
        suppressed_rms=float(suppressed.float().square().mean().sqrt()),component_cosine=cosine)


class Intervention:
    def __init__(self,r,row,arm,audit=False,diagnose=False):
        from tools.e44_runtime import sketch_masks
        from PIL import Image
        self.r,self.p,self.arm=r,r.processor,arm
        raw,soft,info=sketch_masks(Image.open(row['sketch']).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR),'cpu')
        self.valid=info['valid'];self.mask_info=info
        self.weights=weights_at(region_masks(raw),(64,48),r.pipe.device)
        self.token_weights=weights_at(region_masks(raw),(32,24),r.pipe.device)
        self.soft=F.interpolate(soft.to(r.pipe.device),size=(64,48),mode='area')
        self.audit,self.diagnose=audit,diagnose
        self.steps=[];self.scheduler_steps=[];self.handles=[];self.phase=None;self.component=None
        self.unconditional_calls=self.teacher_calls=self.student_calls=0
        self.teacher_module_calls=self.student_module_calls=0
        self.trajectory=[];self.capture={}
        self.base=self.raw=None
        self.baseline=None
        if arm=='LATENT_LOCK':
            from diffusers import DDIMScheduler
            source=history('B',row['id']).with_suffix('.pt')
            self.baseline=torch.load(source,map_location='cpu',weights_only=False)
            self.baseline_scheduler=DDIMScheduler.from_config(r.pipe.scheduler.config)
            self.baseline_scheduler.set_timesteps(50,device=r.pipe.device)
            self.baseline_latent=self.baseline['initial_latent'].to(r.pipe.device)
            self.baseline_index=0

    def capture_projection(self,name):
        def hook(module,args,result):
            if self.phase=='student':self.capture[name]=result.detach()
        return hook

    def observe(self,base,raw,applied):
        if self.phase=='student':self.base,self.raw=base,raw

    def output(self,module,args,result):
        if self.phase!='student' or not self.p.enabled:return result
        index=self.current['index']
        if self.component is not None:
            return self.base+self.components[self.component]
        raw=self.raw
        multiplier=1.
        if self.arm=='SAFE':multiplier=SAFE_SCALE
        elif self.arm in CONFIG['windows']:
            low,high=CONFIG['windows'][self.arm];multiplier=float(low<=index<high)
        elif self.arm=='ENERGY_LATE':multiplier=read(OUT/'audit/calibration.json')['late_lambda']
        # 历史 G3 先将系数舍入至残差 dtype；Python scalar 乘法的舍入顺序不同。
        applied=raw*torch.as_tensor(multiplier,device=raw.device,dtype=raw.dtype)
        self.current.update(raw_energy=float(raw.float().square().sum()),
            applied_energy=float(applied.float().square().sum()),multiplier=multiplier,
            injection_raw=regional_stats(raw,self.token_weights),
            injection_applied=regional_stats(applied,self.token_weights))
        if (self.diagnose and index in SELECTED_STEPS) or self.arm in COMPONENT_ARMS:
            self.components,self.current['softmax']=decomposition(module,self.p,self.capture)
            self.current['component_sum_relative_error']=float(
                (self.components['added'].float()+self.components['suppressed'].float()-raw.float()).norm()
                /raw.float().norm().clamp_min(1e-8))
            self.current['component_added']=regional_stats(self.components['added'],self.token_weights)
            self.current['component_suppressed']=regional_stats(self.components['suppressed'],self.token_weights)
        if self.arm in COMPONENT_ARMS:
            applied=self.components['added' if self.arm=='COMP_ADD' else 'suppressed']
            self.current.update(applied_energy=float(applied.float().square().sum()),
                injection_applied=regional_stats(applied,self.token_weights))
            return self.base+applied
        if multiplier==1.:return result
        if multiplier==0.:return self.base
        return self.base+applied

    def forward(self,*args,**kwargs):
        conditional='sa_hidden_states' in (kwargs.get('cross_attention_kwargs') or {})
        if not conditional:
            self.phase=None;self.unconditional_calls+=1
            output=self.old_forward(*args,**kwargs)
            assert not self.p.enabled
            if self.audit and self.unconditional_calls in [1,50]:
                assert torch.equal(output[0],self.old_forward(*args,**kwargs)[0])
            return output
        latent=args[0] if args else kwargs['sample']
        t=args[1] if len(args)>1 else kwargs['timestep']
        enabled=self.p.enabled;index=len(self.steps)
        self.current=dict(index=index,timestep=int(t),enabled=enabled,
            same_state_sha256=digest(latent),raw_energy=0.,applied_energy=0.)
        self.capture={}
        try:
            self.p.enabled=False;self.phase='teacher'
            teacher=self.old_forward(*args,**kwargs);self.teacher_calls+=1;self.teacher_module_calls+=1
            self.p.enabled=enabled;self.phase='student'
            student=self.old_forward(*args,**kwargs) if enabled else teacher
            self.student_calls+=1;self.student_module_calls+=int(enabled)
            effect=student[0].float()-teacher[0].float()
            self.current['matched_epsilon_raw']=regional_stats(effect,self.weights)
            if self.arm in ['EPS_LOCK','ENERGY_EPS'] and enabled:
                masked=effect*self.soft.float()
                numerator=masked.square().sum();denominator=effect.square().sum()
                amplitude=float((numerator/denominator).sqrt()) if denominator else 0.
                correction=masked if self.arm=='EPS_LOCK' else effect*amplitude
                result=(teacher[0].float()+correction).to(student[0].dtype)
                student=(result,)+student[1:]
                self.current['epsilon_matching']=dict(amplitude=amplitude,raw_energy=float(denominator),
                    masked_energy=float(numerator),applied_energy=float(correction.square().sum()))
                if self.arm=='EPS_LOCK':
                    zero=self.soft==0
                    assert torch.equal(result[zero.expand_as(result)],teacher[0][zero.expand_as(result)])
            final=student[0].float()-teacher[0].float()
            self.current['matched_epsilon_applied']=regional_stats(final,self.weights)
            self.current['matched_cfg_applied']=regional_stats(final*7.,self.weights)
            # Fixed-weight component interventions share exactly the current latent/conditions.
            if self.diagnose and enabled and index in SELECTED_STEPS:
                self.current['component_counterfactuals']={}
                for name in ['added','suppressed']:
                    self.component=name
                    candidate=self.old_forward(*args,**kwargs)[0]
                    delta=candidate.float()-teacher[0].float()
                    self.current['component_counterfactuals'][name]=regional_stats(delta,self.weights)
                self.component=None
            assert digest(latent)==self.current['same_state_sha256']
            self.steps.append(self.current)
            return student
        finally:
            self.component=None;self.phase=None;self.p.enabled=enabled;self.base=self.raw=None

    def scheduler_step(self,*args,**kwargs):
        result=self.old_step(*args,**kwargs)
        index=len(self.scheduler_steps)
        latent=result[0]
        if self.baseline is not None:
            assert int(args[1])==int(self.baseline_scheduler.timesteps[index])
            eps_c=self.baseline['conditional_epsilon'][index].to(latent.device)
            eps_u=self.baseline['unconditional_epsilon'][index].to(latent.device)
            guidance=eps_u+7.*(eps_c-eps_u)
            self.baseline_latent=self.baseline_scheduler.step(
                guidance,args[1],self.baseline_latent,return_dict=False)[0]
            if index==49:
                assert torch.equal(self.baseline_latent,self.baseline['final_latent'].to(latent.device)), 'E5 independent trajectory replay failed'
            if self.valid:
                latent=(self.soft.to(latent)*latent+(1-self.soft.to(latent))*self.baseline_latent)
                result=(latent,)+result[1:]
                zero=self.soft==0
                assert torch.equal(latent[zero.expand_as(latent)],self.baseline_latent[zero.expand_as(latent)])
        self.scheduler_steps.append(dict(index=index,timestep=int(args[1]),
            mean_abs_change=float((latent-args[2]).float().abs().mean()),
            outside_baseline_exact=self.baseline is not None and self.valid))
        self.trajectory.append(latent.detach().cpu())
        return result

    def install(self):
        self.r.active=self.arm!='B';self.p.spatial_intervention='global'
        self.old_forward=self.r.pipe.unet.forward;self.r.pipe.unet.forward=self.forward
        self.old_step=self.r.pipe.scheduler.step;self.r.pipe.scheduler.step=self.scheduler_step
        self.p.diagnostic_observer=self.observe
        module=dict(self.r.pipe.unet.named_modules())[LAYER[:-10]]
        self.handles.append(module.register_forward_hook(self.output))
        if self.diagnose or self.arm in COMPONENT_ARMS:
            for name in ['q','k','v']:
                self.handles.append(getattr(module,'to_'+name).register_forward_hook(self.capture_projection(name)))
            self.handles.append(self.p.to_k_ref.register_forward_hook(self.capture_projection('kr')))
            self.handles.append(self.p.to_v_ref.register_forward_hook(self.capture_projection('vr')))

    def close(self):
        self.r.pipe.unet.forward=self.old_forward;self.r.pipe.scheduler.step=self.old_step
        self.p.diagnostic_observer=None;self.r.active=True
        for handle in self.handles:handle.remove()
        self.handles=[]
