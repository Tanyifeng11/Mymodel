"""复用原E5条件与DDIM；缓存前42步，真实最后8步及VAE保持autograd。"""
from types import MethodType
import hashlib
import torch
from torch.utils.checkpoint import checkpoint
from torchvision.transforms.functional import to_tensor
from models.e33tm_generation_wrapper import generate

def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def prefix(pipe,ns,caption,sketch,reference,mask,injection=None):
    states={};before=pipe.scheduler.step;forward=pipe.unet.forward;index=0
    if injection:injection.enabled=False
    def step(*args,**kwargs):
        nonlocal index
        result=before(*args,**kwargs)
        if index==41:states['latent']=result[0].detach().cpu().clone()
        index+=1;return result
    def observe(*args,**kwargs):
        branch='conditional' if 'sa_hidden_states' in kwargs.get('cross_attention_kwargs',{}) else 'unconditional'
        if branch not in states:
            embeds=kwargs['encoder_hidden_states'].detach().cpu().clone()
            states[branch+'_embeddings']=embeds;states[branch]=tensor_sha(embeds)
            if branch=='conditional':
                states['sketch_features']={k:v.detach().cpu().clone() for k,v in kwargs['cross_attention_kwargs']['sa_hidden_states'].items()}
        return forward(*args,**kwargs)
    pipe.scheduler.step=step;pipe.unet.forward=observe
    try:
        with torch.no_grad():image,info=generate(pipe,(384,512),ns,caption,sketch,reference,mask,42)
        assert index==50
        states.update(generation=info,timesteps=pipe.scheduler.timesteps.detach().cpu().clone(),
            scheduler_config=dict(pipe.scheduler.config),vae_scale=float(pipe.vae.config.scaling_factor))
        return states,image
    finally:pipe.scheduler.step=before;pipe.unet.forward=forward

def final8(pipe,ns,caption,sketch,reference,mask,cached,injection,checkpointed=True,trace=False):
    old_prepare,old_timesteps,old_step=pipe.prepare_latents,pipe.scheduler.set_timesteps,pipe.scheduler.step
    old_decode,old_forward=pipe.vae.decode,pipe.unet.forward
    outputs=[];scores=[];checked=set();index=42;start_count=injection.count
    geometry,falloff=injection.geometry,injection.falloff
    def timesteps(steps,*args,**kwargs):
        old_timesteps(steps,*args,**kwargs)
        assert torch.equal(pipe.scheduler.timesteps.cpu(),cached['timesteps'])
        pipe.scheduler.timesteps=pipe.scheduler.timesteps[42:]
    def forward(*args,**kwargs):
        branch='conditional' if 'sa_hidden_states' in kwargs.get('cross_attention_kwargs',{}) else 'unconditional'
        if branch not in checked:
            assert tensor_sha(kwargs['encoder_hidden_states'])==cached[branch], 'cached condition mismatch'
            if branch=='conditional':
                features=kwargs['cross_attention_kwargs']['sa_hidden_states']
                assert set(features)==set(cached['sketch_features'])
                assert all(tensor_sha(features[k])==tensor_sha(v) for k,v in cached['sketch_features'].items())
            checked.add(branch)
        def invoke(*inner_args,**inner_kwargs):
            # checkpoint重算时必须重新确定CFG分支，不能使用最后一次uncond的状态。
            injection.conditional='sa_hidden_states' in inner_kwargs.get('cross_attention_kwargs',{})
            injection.index=42
            # 三臂联合backward时，每臂重算仍使用该臂的field，而非最后一臂field。
            injection.geometry,injection.falloff=geometry,falloff
            return old_forward(*inner_args,**inner_kwargs)
        value=checkpoint(invoke,*args,use_reentrant=False,**kwargs) if checkpointed and torch.is_grad_enabled() else invoke(*args,**kwargs)
        if trace:
            conditional='sa_hidden_states' in kwargs.get('cross_attention_kwargs',{})
            prediction=value[0].detach().float()
            m=torch.nn.functional.interpolate(mask.to(prediction),prediction.shape[-2:],mode='nearest')
            scores.append(dict(step=index,branch='conditional' if conditional else 'unconditional',
                rms=float(prediction.square().mean().sqrt()),
                inner_rms=float((prediction.square()*m).sum().div((m.sum()*prediction.shape[1]).clamp_min(1)).sqrt()),
                outer_rms=float((prediction.square()*(1-m)).sum().div(((1-m).sum()*prediction.shape[1]).clamp_min(1)).sqrt())))
        return value
    def step(*args,**kwargs):
        nonlocal index
        result=old_step(*args,**kwargs)
        if trace:scores.append(dict(step=index,branch='guided',rms=float(args[0].detach().float().square().mean().sqrt())))
        index+=1;return result
    def decode(latents,*args,**kwargs):
        # 捕获真实最后latent的decode；仅交给原PIL后处理的副本detach。
        result=old_decode(latents,*args,**kwargs)
        # 与原E5保持同一fp16归一化/clip顺序；随后float32损失仍保留梯度。
        outputs.append((result[0]/2+.5).clamp(0,1).float())
        return (result[0].detach(),)
    pipe.prepare_latents=MethodType(lambda self,*args,**kwargs:cached['latent'].to(self.device,self.unet.dtype).clone(),pipe)
    pipe.scheduler.set_timesteps,pipe.scheduler.step=timesteps,step
    pipe.vae.decode,pipe.unet.forward=decode,forward
    injection.enabled=True
    try:
        pipe.__call__.__wrapped__(pipe,prompt=caption,null_prompt='',negative_prompt=' worst quality, low quality',
            ref_image=to_tensor(sketch)[None]*2-1,texture_clip_image=reference,width=384,height=512,
            num_inference_steps=50,guidance_scale=7.,sketch_scale=.6,ipa_scale=1.,
            texture_mode=ns.texture_mode,texture_condition_mode='token',texture_preprocess_mode='plain_resize',
            texture_num_tokens=pipe.effective_texture_num_tokens,texture_scale=1.,spatial_mask=mask.to(pipe.device,torch.float16),
            generator=torch.Generator(device=pipe.device).manual_seed(42))
        assert index==50 and len(outputs)==1 and injection.count-start_count==8
        return outputs[0],scores
    finally:
        pipe.prepare_latents,pipe.scheduler.set_timesteps,pipe.scheduler.step=old_prepare,old_timesteps,old_step
        pipe.vae.decode,pipe.unet.forward=old_decode,old_forward
