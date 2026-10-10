"""P0 与阶段A：纯噪声完整50步，显式逐臂克隆同一个初始latent。"""
import argparse
import copy
import hashlib
import inspect
import time
from types import MethodType
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from tools.e35_freeu_protocol import *
from tools.e35_freeu_runtime import FreeUBridge


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class Generator:
    def __init__(self):
        from models.e33tm_generation_wrapper import load_e5
        from tools.e22_4_generation import module_hashes
        self.pipe,self.modules,size,self.ns=load_e5()
        assert size==SIZE and type(self.pipe.scheduler).__name__=='DDIMScheduler'
        assert all(not p.requires_grad for m in self.modules.values() for p in m.parameters())
        self.before=module_hashes(self.modules)
        self.bridge=FreeUBridge(self.pipe.unet)
        self.structure=self.bridge.structure()
        # 使用VAE实际下采样比例，不凭固定8猜latent尺寸。
        factor=self.pipe.vae_scale_factor
        shape=(1,self.pipe.unet.config.in_channels,SIZE[1]//factor,SIZE[0]//factor)
        self.noise=torch.randn(shape,device=self.pipe.device,dtype=self.pipe.unet.dtype,
            generator=torch.Generator(device=self.pipe.device).manual_seed(42))
        self.initial=self.noise*self.pipe.scheduler.init_noise_sigma
        assert shape==(1,4,64,48), 'E34基座分辨率发生变化'
        write(OUT/'audit/model.json',dict(git_commit=commit(),checkpoint_sha256=sha(E5),
            unet_type=type(self.pipe.unet).__module__+'.'+type(self.pipe.unet).__name__,
            structure=self.structure,freeu_mode=self.bridge.mode,
            scheduler=dict(self.pipe.scheduler.config),inference_args=vars(self.ns),
            dtype=str(self.pipe.unet.dtype),vae_dtype=str(self.pipe.vae.dtype),
            vae_scale_factor=factor,latent_shape=list(shape),noise_sha256=digest(self.noise),
            initial_sha256=digest(self.initial),frozen_modules=self.before,
            device=torch.cuda.get_device_name(),attention_processors={k:type(v).__name__ for k,v in self.pipe.unet.attn_processors.items()}))
        folder=OUT/'p0';folder.mkdir(parents=True,exist_ok=True)
        (folder/'unet_structure.txt').write_text(str(self.pipe.unet),encoding='utf-8')
        (folder/'freeu_implementation.py.txt').write_text(self.bridge.operation_source,encoding='utf-8')
        for stage,source in self.bridge.sources.items():
            (folder/('upblock_stage%s.py.txt'%stage)).write_text(source,encoding='utf-8')
        torch.save(self.initial.detach().cpu(),OUT/'audit/initial_latent.pt')

    @torch.inference_mode()
    def generate(self,row,arm,path,capture_first=False):
        from garment_mask_utils import build_sketch_garment_mask
        pipe=self.pipe;self.bridge.set(ARMS[arm] if isinstance(arm,str) else arm)
        sketch=Image.open(row['sketch']).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR)
        reference=Image.open(row['reference']).convert('RGB')
        mask,_=build_sketch_garment_mask(sketch,*SIZE)
        old_prepare=pipe.prepare_latents;old_step=pipe.scheduler.step
        timesteps=[];last=[];captured={};outputs={};handles=[]
        pipe.prepare_latents=MethodType(lambda _self,*a,**kw:self.initial.clone(),pipe)
        def step(*args,**kwargs):
            result=old_step(*args,**kwargs);timesteps.append(int(args[1]));last[:]=[result[0]];return result
        pipe.scheduler.step=step
        if capture_first:
            def save(_m,args,kwargs):
                branch='conditional' if 'sa_hidden_states' in (kwargs.get('cross_attention_kwargs') or {}) else 'unconditional'
                if branch not in captured:captured[branch]=(args,copy.copy(kwargs))
            def save_output(_m,args,kwargs,result):
                branch='conditional' if 'sa_hidden_states' in (kwargs.get('cross_attention_kwargs') or {}) else 'unconditional'
                if branch not in outputs:outputs[branch]=result[0].clone()
            handles=[pipe.unet.register_forward_pre_hook(save,with_kwargs=True),
                     pipe.unet.register_forward_hook(save_output,with_kwargs=True)]
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        try:
            image=pipe(prompt=row['caption'],null_prompt='',negative_prompt=' worst quality, low quality',
                ref_image=to_tensor(sketch)[None]*2-1,texture_clip_image=reference,
                width=SIZE[0],height=SIZE[1],num_inference_steps=50,guidance_scale=7.,
                sketch_scale=.6,ipa_scale=1.,texture_mode=self.ns.texture_mode,texture_condition_mode='token',
                texture_preprocess_mode='plain_resize',texture_num_tokens=pipe.effective_texture_num_tokens,
                texture_scale=1.,spatial_mask=to_tensor(mask)[None].to(pipe.device,torch.float16),
                generator=torch.Generator(device=pipe.device).manual_seed(42))[0]
        finally:
            pipe.prepare_latents=old_prepare;pipe.scheduler.step=old_step
            for handle in handles:handle.remove()
        torch.cuda.synchronize();seconds=time.perf_counter()-started
        assert len(timesteps)==50 and all(torch.isfinite(t).all() for t in last)
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);image.save(path)
        record=dict(id=row['id'],arm=arm,seed=42,seconds=seconds,
            peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,steps=timesteps,
            noise_sha256=digest(self.noise),initial_latent_sha256=digest(self.initial),
            final_latent_sha256=digest(last[0]),png_sha256=sha(path),
            inputs={k:sha(row[k]) for k in ['sketch','reference','gt']},
            caption_sha256=hashlib.sha256(row['caption'].encode()).hexdigest(),gt_used_in_inference=False)
        write(path.with_suffix('.json'),record)
        return record,captured,outputs

    def close(self):
        from tools.e22_4_generation import module_hashes
        self.bridge.close();after=module_hashes(self.modules)
        assert after==self.before
        write(OUT/'audit/frozen_modules.json',dict(before=self.before,after=after,pass_unchanged=True))
        verify_sources()


@torch.inference_mode()
def p0(g):
    bridge=g.bridge;checks=[];single={};neutral=dict(b1=1.,b2=1.,s1=1.,s2=1.)
    rows=read(OUT/'splits/dev32.json')[:8]
    for i,row in enumerate(rows):
        folder=OUT/'p0/replay'/row['id']
        off,args,outputs=g.generate(row,'A0_E5_OFF',folder/'OFF1.png',capture_first=i==0)
        off2,_,_=g.generate(row,'A0_E5_OFF',folder/'OFF2.png')
        archived=PREVIOUS/'s0_e5_dev'/row['id']/'I0.png'
        old=read(archived.parent/'audit.json')
        assert sha(archived)==old['files']['I0.png']
        replay=off['png_sha256']==off2['png_sha256']==sha(archived)
        latent_replay=off['final_latent_sha256']==off2['final_latent_sha256']
        assert replay and latent_replay, 'E5未精确复现历史RGB或最终latent'
        # 同一单步latent和同一condition，分别验证真实CFG两次调用。
        if i==0:
            for branch in ['conditional','unconditional']:
                a,kw=args[branch];predictions={};reports={}
                for name,cfg in [('OFF',None),('ON',ARMS['A2_MEDIUM']),('OFF_AFTER',None),('NEUTRAL',neutral)]:
                    bridge.set(cfg);bridge.reset()
                    predictions[name]=g.pipe.unet(*a,**kw)[0].clone();reports[name]=bridge.report()
                exact=torch.equal(predictions['OFF'],outputs[branch]) and torch.equal(predictions['OFF'],predictions['OFF_AFTER'])
                active=float((predictions['ON'].float()-predictions['OFF'].float()).abs().mean())
                error=float((predictions['NEUTRAL'].float()-predictions['OFF'].float()).abs().mean())
                details=reports['ON']['operations']
                applied=all(any(d['stage']==stage and d['h_mae']>0 and d['skip_mae']>0 and d['finite'] for d in details) for stage in (0,1))
                finite=all(torch.isfinite(v).all().item() for v in predictions.values())
                assert exact and finite and applied and active>0 and error<active*.05
                single[branch]=dict(off_exact=exact,on_eps_mae=active,neutral_eps_mae=error,
                    empirical_neutral_ratio=error/active,numerical_tolerance='neutral mean error <5% of active mean error',
                    actuation=reports)
            write(OUT/'p0/hook_map.json',single)
        bridge.reset(limit=1)
        g.generate(row,'A1_MILD',folder/'ON.png')
        on_counts=dict(bridge.calls)
        bridge.reset()
        restored,_,_=g.generate(row,'A0_E5_OFF',folder/'OFF_AFTER.png')
        assert off['png_sha256']==restored['png_sha256'] and off['final_latent_sha256']==restored['final_latent_sha256']
        assert not bridge.calls and all(on_counts.get('%d/%s'%(s,b),0)==150 for s in (0,1) for b in ['conditional','unconditional'])
        checks.append(dict(id=row['id'],off_replay=True,archived_e5_equal=True,
            off_after_on_equal=True,final_latent_replay=True,counts=on_counts,
            archived_sha256=sha(archived)))
        write(OUT/'p0/replay_checks.json',checks)
        print('P0',i+1,'/8',row['id'],'exact',flush=True)
    write(OUT/'p0/smoke_report.json',dict(pass_p0=True,n=8,checks=checks,single_step=single,
        cfg_mode='original two separate U-Net calls; both audited',training_steps=0,additional_parameters=0))
    decision(p0_pass=True,e5_replay_pass=True,disabled_replay_equal_e5=True,
        freeu_native_or_ported=bridge.mode,freeu_stage_actuation_count=checks[0]['counts'])


def generate_split(g,split,arms):
    assert read(OUT/'p0/smoke_report.json')['pass_p0']
    rows=read(OUT/'splits'/('%s.json'%split))
    folder=OUT/('a_screen' if split=='dev32' else 'a_confirm')
    write(folder/'params.json',{name:ARMS[name] for name in arms})
    write(folder/'manifest.json',rows)
    for i,row in enumerate(rows):
        for arm in arms:
            path=folder/'images'/arm/(row['id']+'.png')
            if path.with_suffix('.json').exists():
                old=read(path.with_suffix('.json'));assert old['png_sha256']==sha(path);continue
            g.bridge.reset(limit=1 if i==0 else 0)
            record,_,_=g.generate(row,arm,path)
            if i==0:write(folder/('actuation_%s.json'%arm),g.bridge.report())
            assert record['initial_latent_sha256']==digest(g.initial)
        if i in (0,1):
            check,_,_=g.generate(row,'A0_E5_OFF',folder/'off_restore'/(row['id']+'.png'))
            assert check['png_sha256']==sha(folder/'images/A0_E5_OFF'/(row['id']+'.png'))
        if i%8==0 or i+1==len(rows):print('GENERATED',split,i+1,'/',len(rows),flush=True)


def main(phase):
    from tools.e34_sarr_protocol import seed_all
    torch.set_num_threads(2);cv2.setNumThreads(1);seed_all(42)
    if not (OUT/'protocol.json').exists():prepare()
    assert read(OUT/'protocol.json')==CONFIG
    g=Generator()
    try:
        if phase in ['p0','all']:p0(g)
        if phase in ['screen','all']:
            generate_split(g,'dev32',list(ARMS))
            from tools.e35_freeu_stats import evaluate,select
            evaluate('dev32');selected=select()
        else:selected=read(OUT/'a_screen/selected_candidate.json') if (OUT/'a_screen/selected_candidate.json').exists() else None
        if phase in ['confirm','all'] and selected and selected['selected']:
            generate_split(g,'confirm96',['A0_E5_OFF',selected['selected']])
            from tools.e35_freeu_stats import evaluate,confirm
            evaluate('confirm96');confirm()
    finally:g.close()
    print('DECISION',json.dumps(read(OUT/'final_decision.json')),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['p0','screen','confirm','all'],default='all')
    args=parser.parse_args()
    try:main(args.phase)
    except Exception as exc:
        decision(error=repr(exc),a_result='engineering_blocked')
        raise
