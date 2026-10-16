"""原B1上的参考分支修正；不改原self/草图路径，不新增参数。"""
import math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from models.e48_interventions import split,projection
from tools.e53_protocol import CONFIG,KEY_STEPS,OUT
from tools.e53_mapping import matrices,align
from tools.e44_b1_mask import region_masks,weights_at,regional_stats

def rms(x):return x.float().square().mean().sqrt()

def match_energy(value,target,tolerance=1e-4,context=None):
    """匹配应用增量的投影后RMS，复核cast后的实际值；0预算严格返回0。"""
    target=float(target);source=float(rms(value))
    if target==0:return torch.zeros_like(value),dict(before=source,after=0.,target=0.,gain=0.,relative_error=0.)
    assert source>1e-12,'ZERO_RESIDUAL_CANNOT_MATCH_POSITIVE_BUDGET'
    factor=target/source
    for _ in range(6):
        applied=(value.float()*factor).to(value.dtype);actual=float(rms(applied))
        error=abs(actual/target-1)
        if error<tolerance:break
        factor*=target/max(actual,1e-30)
    legacy_error=error;method='iterative_rescale'
    if error>=tolerance:
        # FP16已量化的向量在gain≈1处有平台，比例迭代可能在平台两侧振荡。
        # cast后RMS对正gain单调；二分搜索同一个全局标量，不改变方向或放宽门限。
        method='bisection_after_rescale';low=0.;high=2*max(factor,target/source)
        for _ in range(32):
            factor=(low+high)/2;applied=(value.float()*factor).to(value.dtype);actual=float(rms(applied));error=abs(actual/target-1)
            if error<tolerance:break
            if actual<target:low=factor
            else:high=factor
    if not torch.isfinite(applied).all() or error>=tolerance:
        from datetime import datetime,timezone
        path=OUT/'failures'/('energy_tensor_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.pt');path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(value=value.detach().cpu(),target=target,source=source,gain=factor,relative_error=error,legacy_relative_error=legacy_error,context=context),path)
        raise AssertionError(('ENERGY_MATCH_FAIL',error,str(path)))
    return applied,dict(before=source,after=actual,target=target,gain=factor,relative_error=error,method=method,legacy_relative_error=legacy_error)

def attention(q,k,ks):
    logits=q.float()@k.float().transpose(-1,-2)/math.sqrt(q.shape[-1])
    own=q.float()@ks.float().transpose(-1,-2)/math.sqrt(q.shape[-1])
    return torch.softmax(logits,-1),torch.sigmoid(torch.logsumexp(logits,-1)-torch.logsumexp(own,-1))

class OracleDiagnostic:
    def __init__(self,r,row,arm,spec,energy_targets=None,full=False):
        self.r,self.p,self.row,self.arm,self.spec=r,r.processor,row,arm,spec
        self.energy_targets=energy_targets;self.full=full;self.cache={};self.handles=[];self.steps=[];self.latents=[];self.phase=None
        self.m={};self.allowed={};self.uvdist={};self.valid={}
        if row.get('geometry'):
            for g in ['0','1','N']:
                m,allowed,t,ref=matrices(row,g);self.allowed[g]=torch.from_numpy(allowed).to(r.pipe.device)
                self.m[g]={mode:torch.from_numpy(matrices(row,g,mode)[0]).to(r.pipe.device) for mode in ['ORACLE','SHIFT','FLIP','RANDOM']}
                self.uvdist[g]=torch.from_numpy(np.linalg.norm(t[0][:,None]-ref[0][None],axis=-1)).to(r.pipe.device)
                self.valid[g]=torch.from_numpy(m.sum(1)>.99).to(r.pipe.device)
            self.transport=torch.from_numpy(align(row)).to(r.pipe.device)
            self.restore_transport=torch.from_numpy(align(row,source='0',destination='1')).to(r.pipe.device)
            labels=np.load(row['geometry'])['target_labels'];mask=labels>0
        else:
            from tools.e44_runtime import sketch_masks
            from PIL import Image
            mask=np.asarray(sketch_masks(Image.open(row['sketch']),'cpu')[0])>127
        self.weights=weights_at(region_masks(mask),(64,48),r.pipe.device)
        self.raw_dir=OUT/'traces'/spec.get('stage','audit')/arm/row['id']

    def capture(self,name):
        def hook(module,args,result):
            if self.phase=='candidate':self.cache[name]=result.detach()
        return hook

    def observer(self,base,raw,applied):
        if self.phase=='candidate':self.base,self.raw=base,raw

    def kv(self,geometry,texture=0):
        features=self.r.features_for(geometry,texture,self.timestep)
        normalized=self.p.norm(features.float())
        return split(self.p.to_k_ref(normalized).to(self.cache['q'].dtype),self.module.heads),split(self.p.to_v_ref(normalized).to(self.cache['q'].dtype),self.module.heads)

    def output(self,module,args,result):
        if self.phase!='candidate' or not self.p.enabled:return result
        q=split(self.cache['q'],module.heads);ks=split(self.cache['k'],module.heads);vs=split(self.cache['v'],module.heads)
        kr=split(self.cache['kr'].to(q),module.heads);vr=split(self.cache['vr'].to(q),module.heads)
        original,p_original=attention(q,kr,ks);own=torch.softmax(q.float()@ks.float().transpose(-1,-2)/math.sqrt(q.shape[-1]),-1)@vs.float()
        mode=self.spec['routing'];g=self.spec['geometry'];used=original;values=vr;pref=p_original
        if mode not in ['ORIGINAL','REPLAY']:
            # 固定同一当前Q的reference mass；防止几何改变K后同时改变增量门控质量。
            base_g='0' if mode in ['V_SWAP','K_SWAP'] else 'N'
            kbase,vbase=self.kv(base_g);abase,pref=attention(q,kbase,ks)
            if mode in ['V_SWAP','K_SWAP']:
                k1,v1=self.kv('1');tr=self.transport
                k1=torch.einsum('qr,bhrd->bhqd',tr,k1.float()).to(k1.dtype)
                v1=torch.einsum('qr,bhrd->bhqd',tr,v1.float()).to(v1.dtype)
                good=tr.sum(-1)>0
                k1=torch.where(good[None,None,:,None],k1,kbase);v1=torch.where(good[None,None,:,None],v1,vbase)
                used=abase if mode=='V_SWAP' else attention(q,k1,ks)[0]
                values=v1 if mode=='V_SWAP' else vbase
            else:
                valid=self.valid[g]
                if mode=='SUPPORT':
                    used=original*self.allowed[g][None,None];used=used/used.sum(-1,keepdim=True).clamp_min(1e-30)
                else:used=self.m[g]['ORACLE' if mode=='RESTORE_V0' else mode][None,None].expand_as(original)
                if mode=='RESTORE_V0':
                    _,v0=self.kv('0');tr=self.restore_transport
                    restored=torch.einsum('qr,bhrd->bhqd',tr,v0.float())
                    values=torch.where((tr.sum(-1)>0)[None,None,:,None],restored,vr.float())
                used=torch.where(valid[None,None,:,None],used,original)
                # 背景/无法映射query严格保留原分支，避免把NA当OFF。
                pref=torch.where(valid[None,None],pref,p_original)
        ref=used@values.float();ref_original=original@vr.float()
        correction=pref[...,None]*(ref-own)-p_original[...,None]*(ref_original-own)
        projected=projection(module,(self.p.gate*correction).to(q.dtype))
        value=self.raw.float()+projected.float();value=value.to(self.raw.dtype)
        if mode in ['ORIGINAL','REPLAY']:
            assert torch.count_nonzero(projected)==0,'ORIGINAL_SUBSTITUTION_NOT_ZERO'
            assert torch.equal(value,self.raw),'REPLAY_DELTA_CHANGED'
        target=self.energy_targets[self.index] if self.energy_targets is not None else float(rms(value))
        applied,energy=match_energy(value,target,CONFIG['energy']['tolerance'],dict(stage=self.spec['stage'],identity=self.row['id'],arm=self.arm,step=self.index))
        self.current.update(energy=energy,unconditional_energy=0.,raw_original_rms=float(rms(self.raw)),reference_rms=float(rms(ref)),self_rms=float(rms(own)),delta_cosine_original=float(F.cosine_similarity(applied.float().flatten(),self.raw.float().flatten(),dim=0)),shape=list(applied.shape),reference_shape=list(self.p.features.shape))
        if self.energy_targets is None and mode in ['ORIGINAL','REPLAY']:
            assert torch.equal(applied,self.raw),'MATCH_OFF_CHANGED_DELTA'
            final=result
        else:final=self.base+applied
        self.current['joint_exact']=bool(torch.equal(final,result))
        if self.row.get('geometry') and self.index in KEY_STEPS:
            # 在同一Q上将K/V先按真实UV对齐再做factorial；方向投影到target残差空间后才算cosine。
            k00,v00=self.kv('0',0);k01,v01=self.kv('1',0);k10,v10=self.kv('0',1);k11,v11=self.kv('1',1)
            tr=self.transport;good=tr.sum(-1)>0
            def transport(x,fallback):
                value=torch.einsum('qr,bhrd->bhqd',tr,x.float());return torch.where(good[None,None,:,None],value,fallback.float())
            k01,v01,k11,v11=transport(k01,k00),transport(v01,v00),transport(k11,k10),transport(v11,v10)
            self.current['factorial_interaction']={name:float((x11-x10.float()-x01+x00.float())[:,:,good].norm()/torch.stack([x00.float().norm(),x01.norm(),x10.float().norm(),x11.norm()]).mean().clamp_min(1e-12)) for name,x00,x01,x10,x11 in [('K',k00,k01,k10,k11),('V',v00,v01,v10,v11)]}
            m0=self.m['0']['ORACLE'][None,None];dg=m0@(v01-v00.float());dt=m0@(v10.float()-v00.float())
            geometry_direction=projection(module,(self.p.gate*pref[...,None]*dg).to(q.dtype));texture_direction=projection(module,(self.p.gate*pref[...,None]*dt).to(q.dtype))
            self.current['direction_cosine']={name:float(F.cosine_similarity(applied.float().flatten(),value.float().flatten(),dim=0)) for name,value in [('geometry',geometry_direction),('texture',texture_direction)]}
        if row_geometry:=self.row.get('geometry'):
            valid=self.valid[g];m=self.m[g]['ORACLE'];dist=self.uvdist[g];a=original[:,:,valid];mm=m[valid][None,None]
            quantities=dict(KL_oracle_to_original=(mm*(mm.clamp_min(1e-30).log()-a.clamp_min(1e-30).log())).sum(-1),KL_original_to_oracle=(a*(a.clamp_min(1e-30).log()-mm.clamp_min(1e-30).log())).sum(-1),mass_uv_radius=(a*(dist[valid][None,None]<=.05)).sum(-1),top1_uv=(torch.gather(dist[valid][None,None].expand_as(a),-1,a.argmax(-1,keepdim=True)).squeeze(-1)<=.05).float(),entropy=-(a*a.clamp_min(1e-30).log()).sum(-1),p_ref=pref[:,:,valid])
            self.current['attention']={name:dict(mean=float(x.mean()),median=float(x.median()),p90=float(torch.quantile(x.float().flatten(),.9))) for name,x in quantities.items()}
            self.current['p_ref_used_row_sum_error']=float((used.sum(-1)*pref-pref).abs().max())
            assert self.current['p_ref_used_row_sum_error']<2e-6
        if self.full or self.index in KEY_STEPS:
            raw=dict(identity=self.row['id'],condition=self.arm,step=self.index,timestep=int(self.timestep),CFG_branch='conditional',p_ref_original=p_original.cpu(),p_ref_used=pref.cpu(),A_ref_normalized=original.cpu(),A_used_normalized=used.cpu(),A_ref_storage='factorized: p_ref_original[...,None] * A_ref_normalized',M_used=None if mode in ['ORIGINAL','REPLAY','K_SWAP','V_SWAP'] else self.m[g].get(mode,torch.empty(0)).cpu(),K_ref=kr.cpu(),V_ref=vr.cpu(),V_oracle=values.cpu(),self_out=own.cpu(),reference_out=ref.cpu(),delta_before_energy_match=value.cpu(),delta_after_energy_match=applied.cpu(),latent_before=self.current_latent.cpu(),energy=energy)
            self.pending_raw=raw
        return final

    def forward(self,*args,**kwargs):
        conditional='sa_hidden_states' in (kwargs.get('cross_attention_kwargs') or {})
        if not conditional:return self.old_forward(*args,**kwargs)
        self.index=len(self.steps);self.timestep=args[1] if len(args)>1 else kwargs['timestep'];self.current_latent=(args[0] if args else kwargs['sample']).detach()
        self.current=dict(index=self.index,timestep=int(self.timestep),condition=self.arm,CFG_branch='conditional',unconditional_energy=0.)
        self.latents.append(self.current_latent.cpu());enabled=self.p.enabled;self.pending_raw=None
        try:
            self.p.enabled=False;self.phase='e5';baseline=self.old_forward(*args,**kwargs)
            self.p.enabled=enabled;self.phase='candidate';self.cache={};candidate=self.old_forward(*args,**kwargs) if enabled else baseline
            self.current['direct_epsilon']=regional_stats(candidate[0].float()-baseline[0].float(),self.weights)
            if not enabled:self.current.update(energy=dict(before=0.,after=0.,target=0.,gain=0.,relative_error=0.),bypassed=True)
            self.steps.append(self.current)
            return candidate
        finally:self.p.enabled=enabled;self.phase=None

    def scheduler_step(self,*args,**kwargs):
        result=self.old_step(*args,**kwargs)
        if self.pending_raw is not None:
            self.pending_raw['latent_after']=result[0].detach().cpu();self.raw_dir.mkdir(parents=True,exist_ok=True)
            torch.save(self.pending_raw,self.raw_dir/('%02d.pt'%self.index));self.pending_raw=None
        return result

    def install(self):
        self.module=dict(self.r.pipe.unet.named_modules())[CONFIG['layer'][:-10]]
        self.old_forward=self.r.pipe.unet.forward;self.r.pipe.unet.forward=self.forward
        self.old_step=self.r.pipe.scheduler.step;self.r.pipe.scheduler.step=self.scheduler_step
        self.p.diagnostic_observer=self.observer
        self.handles.append(self.module.register_forward_hook(self.output))
        for k in ['q','k','v']:self.handles.append(getattr(self.module,'to_'+k).register_forward_hook(self.capture(k)))
        self.handles.append(self.p.to_k_ref.register_forward_hook(self.capture('kr')));self.handles.append(self.p.to_v_ref.register_forward_hook(self.capture('vr')))

    def close(self):
        self.r.pipe.unet.forward=self.old_forward;self.r.pipe.scheduler.step=self.old_step;self.p.diagnostic_observer=None
        for handle in self.handles:handle.remove()
