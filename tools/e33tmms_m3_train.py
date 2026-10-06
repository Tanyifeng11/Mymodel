"""M3 单步x0训练代理；正式Gate另用完整原E5采样与E26读出。"""
import math,random,time
from types import MethodType
import numpy as np,torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor
from data.e33tmms_learned_data import CounterfactualRGBDataset
from data.e32_target_pseudogt import image_at
from models.e33tmms_m3 import OrientedFeatureTransport,SingleSiteInjection
from models.e33tm_generation_wrapper import load_e5
from tools.e33tmms_m2_train import FrozenField,geometry,stream
from tools.e33tmms_learned_losses import LearnedLoss,mean_mask
from tools.e33tmms_protocol import *
from tools.e33tm_protocol import E5
from tools.e33rf_common import OUT as RF
from tools.e33tm_metrics import Evaluator
from tools.e22_4_generation import module_hashes
from tools.e33tm_weight_audit import effective_hashes

class CaptionedDataset(CounterfactualRGBDataset):
    def __getitem__(self,index):
        result=super().__getitem__(index);result['caption']=self.rows[index]['caption'];return result

def data_loader(rows,controlled,seed):
    return DataLoader(CaptionedDataset(rows,controlled),batch_size=1,shuffle=True,num_workers=2,
        multiprocessing_context='spawn',pin_memory=True,generator=torch.Generator().manual_seed(seed))

def predict_one(pipe,ns,caption,sketch,reference,mask,initial,index):
    """复用原pipeline条件支路；只保留一个DDIM步的真实pred_original_sample计算图。"""
    old_prepare,old_timesteps,old_step,old_decode=pipe.prepare_latents,pipe.scheduler.set_timesteps,pipe.scheduler.step,pipe.vae.decode
    captured=[]
    def timesteps(steps,*args,**kwargs):
        old_timesteps(steps,*args,**kwargs);pipe.scheduler.timesteps=pipe.scheduler.timesteps[index:index+1]
    def step(*args,**kwargs):
        value=old_step(*args,**kwargs);captured.append(value[1]);return value
    def unused_decode(*args,**kwargs):
        # 原pipeline末尾强制numpy/PIL；此输出不用作训练，避免它切断所捕获x0的计算图。
        with torch.no_grad():return old_decode(*args,**kwargs)
    pipe.prepare_latents=MethodType(lambda self,*args,**kwargs:initial.detach().clone(),pipe)
    pipe.scheduler.set_timesteps,pipe.scheduler.step,pipe.vae.decode=timesteps,step,unused_decode
    try:
        pipe.__call__.__wrapped__(pipe,prompt=caption,null_prompt='',negative_prompt=' worst quality, low quality',
            ref_image=sketch,width=384,height=512,num_inference_steps=50,guidance_scale=7.,
            sketch_scale=.6,ipa_scale=1.,texture_clip_image=reference,texture_mode=ns.texture_mode,
            texture_condition_mode='token',texture_preprocess_mode='plain_resize',
            texture_num_tokens=pipe.effective_texture_num_tokens,texture_scale=1.,spatial_mask=mask,
            generator=torch.Generator(device=pipe.device).manual_seed(42))
        assert len(captured)==1
        return (old_decode(captured[0]/pipe.vae.config.scaling_factor,return_dict=False)[0].float()/2+.5).clamp(0,1)
    finally:
        pipe.prepare_latents,pipe.scheduler.set_timesteps,pipe.scheduler.step,pipe.vae.decode=old_prepare,old_timesteps,old_step,old_decode

def text_embedding(model,rgb):
    pixels=F.interpolate(rgb,(224,224),mode='bilinear',align_corners=False)
    pixels=(pixels-rgb.new_tensor([.48145466,.4578275,.40821073])[None,:,None,None])/rgb.new_tensor([.26862954,.26130258,.27577711])[None,:,None,None]
    return F.normalize(model.get_image_features(pixel_values=pixels),dim=1)

def setup():
    prepare();d=read(OUT/'decision_summary.json')
    assert d['M1_carrier_pass'] is False
    assert d['M2_run'] and any(d.get('M2_'+k) is False for k in ['controlled_pass','real_carrier_pass','confirmation_pass','e5_pass'])
    cf=read(RF/'controlled_manifest.json');real=read(RF/'split_manifest.json')
    captions={Path(r['cloth']).stem:r['caption'] for r in read('data/train_bf_texture.json')}
    held=set(sum([read(OUT/'splits'/(s+'.json')) for s in ['diagnostic64','confirmation64','remaining126']],[]))
    for rows in (cf['train'],real['train']):
        assert set(r['id'] for r in rows).isdisjoint(held)
        for row in rows:row['caption']=captions[Path(row['target']).stem]
    manifest=E5.parent.parent/'experiment_manifest.json';original=read(manifest)
    assert original['freeze_for_tcpm_lite']==1 and original['use_tcpm_lite']==1 and not original['train_spatial_only']
    pipe,modules,size,ns=load_e5();assert size==(384,512)
    denominator=sum(p.numel() for p in modules['tcpm_lite'].parameters());assert denominator==889537
    transport=OrientedFeatureTransport().cuda();count=sum(p.numel() for p in transport.parameters())
    assert count==7513 and count<denominator*.05
    evaluator=Evaluator();modules['CLIP_loss']=evaluator.model
    proof=dict(git_commit=commit(),plan_sha256=PROTOCOL['plan_sha256'],seed=42,site=transport.site,
        channels=320,resolution=[64,48],alpha=.1,sampling='fixed3x3 feature-cell offsets rotated by half atan2 RF2; border;9 global softmax weights',
        trainable=count,original_E5_trainable=denominator,parameter_ratio=count/denominator,
        original_training_manifest_sha256=sha(manifest),original_training_commit=original['git_commit'],
        E5_sha256=sha(E5),caption_source_sha256=sha('data/train_bf_texture.json'),
        train_counts=dict(controlled=len(cf['train']),real=len(real['train'])),heldout_count=len(held),
        train_heldout_disjoint=True,data_source_sha256=sha('data/e33tmms_learned_data.py'),
        training_proxy='one differentiable DDIM x0 step; uniform original50 timestep; noisy GT-R0 latent identical across arms; rotations only in reference and frozen RF2; rotated GT only reconstruction supervision',
        inference='no target RGB or GT input; formal initialization recorded separately',
        loss=dict(rec=1.,appearance=.5,orientation=1.,counterfactual90_and180=1.,structure_noninferiority=1.,text_noninferiority=1.),
        noninferiority_proxy='background/boundary RGB L1 vs same frozen original E5 one-step output; CLIP cosine hinge allowing10% relative drop; bilinear224 CLIP proxy',
        effective_batch=8,microbatch=1,lr=1e-4,weight_decay=1e-4,warmup=200,precision='bf16 autocast, FP32 new parameters',
        controlled_steps=4000,real_mixed_steps=2000,checkpoint_selection='last scheduled endpoint only')
    write(OUT/'protocol/M3_implementation.json',proof)
    return cf,real,pipe,modules,ns,transport,evaluator

def train():
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(2)
    cf,real,pipe,modules,ns,transport,evaluator=setup()
    cf_lookup={r['id']:r for r in cf['train']};real_lookup={r['id']:r for r in real['train']}
    field=FrozenField();criterion=LearnedLoss();before=module_hashes(modules);active=effective_hashes(modules)
    injection=SingleSiteInjection(pipe,transport)
    decision(M3_run=True,next_route='M3_training')
    for phase,steps in [('controlled',4000),('real_mixed',2000)]:
        dest=OUT/'M3'/phase;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'training_complete.json').exists():
            checkpoint=dest/'checkpoint_final.pt';assert sha(checkpoint)==read(dest/'training_complete.json')['checkpoint_sha256']
            transport.load_state_dict(torch.load(checkpoint,map_location='cpu')['model']);continue
        optimizer=torch.optim.AdamW(transport.parameters(),lr=1e-4,weight_decay=1e-4)
        control=stream(data_loader(cf['train'],True,42));actual=stream(data_loader(real['train'],False,1042)) if phase=='real_mixed' else None
        history=[];control_seen=real_seen=0;first=1;resume=dest/'checkpoint_resume.pt'
        if resume.exists():
            saved=torch.load(resume,map_location='cpu');transport.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);first=saved['step']+1
            control_seen,real_seen=saved['controlled_seen'],saved['real_seen'];history=read(dest/'history.json')
            for _ in range(control_seen):next(control)
            if actual:
                for _ in range(real_seen):next(actual)
            random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        start=time.monotonic()
        for step in range(first,steps+1):
            factor=step/200 if step<=200 else .5*(1+math.cos(math.pi*(step-200)/(steps-200)))
            for group in optimizer.param_groups:group['lr']=1e-4*factor
            optimizer.zero_grad(set_to_none=True);seen=set();metrics={}
            for j in range(8):
                controlled=phase=='controlled' or j<4
                while True:
                    case=next(control if controlled else actual)
                    if controlled:control_seen+=1
                    else:real_seen+=1
                    sid=case['id'][0]
                    if sid not in seen:break
                seen.add(sid);refs=case['references'][0].cuda();targets=case['targets'][0].cuda()
                geom=geometry(case,field,'controlled_train' if controlled else 'real_train')
                # mask与structure之外，原sketch RGB从同身份原清单读取，保持原E5支路。
                row=(cf_lookup if controlled else real_lookup)[sid]
                sketch=(to_tensor(image_at(DATASET/row['sketch']))[None]*2-1)
                mask=case['mask'].cuda().to(pipe.unet.dtype);caption=case['caption'][0]
                references=[Image.fromarray(np.uint8(np.round(r.cpu().numpy().transpose(1,2,0)*255))) for r in refs]
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    features=criterion.texture(refs*2-1).float().detach()
                    latent=pipe.vae.encode(targets[:1].to(pipe.vae.dtype)*2-1).latent_dist.mean*pipe.vae.config.scaling_factor
                    pipe.scheduler.set_timesteps(50,device=pipe.device);index=random.randrange(50);t=pipe.scheduler.timesteps[index]
                    noise=torch.randn_like(latent);initial=pipe.scheduler.add_noise(latent,noise,t[None]).detach()
                    injection.enabled=False
                    baseline=torch.cat([predict_one(pipe,ns,caption,sketch,references[i],mask,initial,index) for i in range(3)])
                    text_args=evaluator.processor(text=[caption],return_tensors='pt',padding=True,truncation=True)
                    text=F.normalize(evaluator.model.get_text_features(**{k:v.cuda() for k,v in text_args.items()}),dim=1)
                    base_text=(text_embedding(evaluator.model,baseline)*text).sum(1)
                injection.enabled=True;pred=[]
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    for i in range(3):
                        injection.set(features[i:i+1],geom[i:i+1]);pred.append(predict_one(pipe,ns,caption,sketch,references[i],mask,initial,index))
                output=torch.cat(pred).float();dummy=(output.new_zeros(3,1),output.new_zeros(3,1,1))
                loss,parts=criterion(output,refs,targets,geom,case['support'][0].cuda(),case['rec_arms'][0].cuda(),dummy,case['source_masks'][0].cuda())
                m=geom[:,3:4];boundary=(F.max_pool2d(m,11,1,5)-(-F.max_pool2d(-m,11,1,5))).clamp(0,1)
                structure=mean_mask((output-baseline).abs(),(1-m+boundary).clamp(0,1))
                similarity=(text_embedding(evaluator.model,output)*text).sum(1)
                text_loss=F.relu(.9*base_text-similarity).mean();total=loss+structure+text_loss
                (total/8).backward()
                for k,v in dict(total=float(total.detach()),structure=float(structure.detach()),text=float(text_loss.detach()),**parts).items():metrics[k]=metrics.get(k,0)+v/8
            assert all(torch.isfinite(p.grad).all() for p in transport.parameters() if p.grad is not None)
            if step==first:
                frozen=[m for m in modules.values()]+[field.model,field.dino,criterion.texture,criterion.lpips]
                write(dest/'gradient_check.json',dict(distinct_identities=len(seen),finite=True,
                    projection_gradient=float(transport.projection[-1].weight.grad.norm()),
                    frozen_gradients_absent=all(p.grad is None for m in frozen for p in m.parameters()),
                    site_shape=injection.last_shape,one_conditional_site=True))
            torch.nn.utils.clip_grad_norm_(transport.parameters(),1.);optimizer.step()
            if step%25==0 or step==first:
                history.append(dict(step=step,elapsed_seconds=time.monotonic()-start,lr=1e-4*factor,**metrics));write(dest/'history.json',history);print('[MS M3 train]',phase,history[-1],flush=True)
            if step%250==0:
                temporary=resume.with_suffix('.tmp');torch.save(dict(step=step,model=transport.state_dict(),optimizer=optimizer.state_dict(),
                    controlled_seen=control_seen,real_seen=real_seen,python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),temporary);temporary.replace(resume)
        checkpoint=dest/'checkpoint_final.pt';torch.save(dict(model=transport.state_dict(),phase=phase,steps=steps,seed=42,git_commit=commit()),checkpoint)
        write(dest/'training_complete.json',dict(complete=True,steps=steps,checkpoint_sha256=sha(checkpoint)))
    injection.close();after=module_hashes(modules);effective_after=effective_hashes(modules)
    assert before==after and active==effective_after
    write(OUT/'M3/frozen_modules.json',dict(before=before,after=after,effective_before=active,effective_after=effective_after,field=field.verify(),loss=criterion.verify(),**{'pass':True}))
    frozen_check();decision(M3_run=True,next_route='M3_diagnostic64')
    from tools.e33tmms_benchmark import bundle
    bundle('M3_training')

if __name__=='__main__':train()
