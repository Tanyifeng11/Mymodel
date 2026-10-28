"""仅推理：E0/E1、已有BF与CNIPA权重；复用冻结输入与逐身份噪声。"""
import argparse
import importlib.util
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
from tools.t3_p0p1 import PROJECT,ROOT,SOURCE,read,write,sha


def digest(value):
    return __import__('hashlib').sha256(value.detach().cpu().numpy().tobytes()).hexdigest()


def selected_rows(arm, smoke, shard=0, shards=1):
    rows=read(ROOT/'protocol/input_manifest.json')
    if arm in ['E0','E1']:
        rows=[r for r in rows if r['condition']=='original']
    if smoke:
        # 选四个固定身份，替换组优先，以同时验证参考分支。
        candidates={r['id'] for r in rows if r['condition']=='swapped'} or {r['id'] for r in rows}
        ids=sorted(candidates,key=lambda i:__import__('hashlib').sha256(('T3PP_SMOKE/'+i).encode()).hexdigest())[:4]
        rows=[r for r in rows if r['id'] in ids]
    return [r for index,r in enumerate(rows) if index%shards==shard]


def internal(arm):
    path=Path(read(ROOT/'audit/server_inventory.json')['models'][arm]['path'])
    spec=importlib.util.spec_from_file_location('t3pp_frozen_internal',SOURCE/'inference_IMAGGarment-1.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sd=PROJECT/'models/stable-diffusion-v1-5'
    args=SimpleNamespace(GAM_model_ckpt=str(path),texture_ckpt=str(path),base_model_path=str(sd),
        vae_model_path=str(sd/'vae'),image_encoder_path=str(PROJECT/'models/clip'),device='cuda:0',
        width=384,height=512,texture_num_tokens=16,force_texture_num_tokens_override=False,
        texture_condition_mode='token',alpha1=2.,alpha2=2.,alpha3=1.5,alpha4=1.)
    pipe,_=module.prepare(args)
    for model in [pipe.unet,pipe.reference_unet,pipe.vae,pipe.text_encoder,pipe.image_encoder,pipe.bf_texture_conditioner]:
        model.eval().requires_grad_(False)
    pipe.set_layer_group_enabled(arm!='E0');pipe._setup_layer_groups();pipe.set_progress_bar_config(disable=True)
    actual={};original=pipe.prepare_latents
    def prepare_latents(*args,**kwargs):
        value=original(*args,**kwargs);actual['latent']=digest(value);return value
    pipe.prepare_latents=prepare_latents
    def run(row):
        seed=row['seed'];torch.manual_seed(seed)
        latent=torch.randn((1,4,64,48),generator=torch.Generator(device='cuda').manual_seed(seed),device='cuda',dtype=torch.float16)
        assert digest(latent)==row['historical_initial_latent_sha256']
        sketch=Image.open(row['sketch']).convert('RGB').resize((384,512),Image.Resampling.BILINEAR)
        sketch=torch.from_numpy(np.asarray(sketch,np.float32).transpose(2,0,1)/127.5-1)[None]
        reference=Image.open(row['reference']).convert('RGB')
        image=pipe(ref_image=sketch,prompt=row['caption'],texture_clip_image=reference,texture_embeds=None,
            null_prompt='',negative_prompt=' worst quality, low quality',width=384,height=512,
            num_images_per_prompt=1,guidance_scale=7.,sketch_scale=.6,ipa_scale=1.,
            generator=torch.Generator(device='cuda').manual_seed(seed),latents=latent,num_inference_steps=50,
            texture_mode='patch_resampled',texture_num_tokens=16,texture_scale=1.,texture_condition_mode='token',
            fusion_type='minimal',texture_preprocess_mode='plain_resize',spatial_mask=None,debug_spatial=False,
            force_texture_num_tokens_override=False)[0]
        assert actual['latent']==digest(latent)
        return image,dict(initial_latent_sha256=actual['latent'],text_used=True,
            layer_group_enabled=arm!='E0',checkpoint=str(path),steps=50,sampler='DDIM',cfg=7.)
    return run


def cnipa():
    base=PROJECT.parent/'cn_ipa_repro_20261022'
    sys.path.insert(0,str(base/'repo'))
    from cn_ipa import runtime as runtime
    choice=read(base/'selection/frozen_final_choice.json')
    # 先前独立开发阶段冻结的C2配置；本次不根据P2b结果重新选权重或scale。
    config=choice.get('C2',choice.get('models',{}).get('C2'))
    if config is None:
        config=read(ROOT/'protocol/cnipa_choice.json')
    assert sha(config['ip_path'])==config['ip_sha256']
    pipe=runtime.load(config['cn_path'],config['ip_path'])
    pipe.set_ip_adapter_scale(config['ip_scale'])
    def run(row):
        latent=torch.randn((1,4,64,48),generator=torch.Generator(device='cuda').manual_seed(row['seed']),device='cuda',dtype=torch.float16)
        assert digest(latent)==row['historical_initial_latent_sha256']
        image=pipe(prompt=row['caption'],negative_prompt=runtime.NEGATIVE,
            image=Image.open(row['sketch']).convert('RGB').resize((384,512),Image.Resampling.BILINEAR),
            ip_adapter_image=Image.open(row['reference']).convert('RGB'),width=384,height=512,
            num_inference_steps=50,guidance_scale=7.,controlnet_conditioning_scale=config['cn_scale'],
            latents=latent,generator=torch.Generator(device='cuda').manual_seed(row['seed'])).images[0]
        return image,dict(initial_latent_sha256=digest(latent),text_used=True,steps=50,cfg=7.,
            cn_scale=config['cn_scale'],ip_scale=config['ip_scale'],checkpoint=config)
    return run


def bf():
    from omegaconf import OmegaConf
    repo=PROJECT.parent/'BF-Fashion-'
    sys.path.insert(0,str(repo));sys.path.insert(0,str(repo/'src/taming-transformers'))
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.ddim import DDIMSampler
    asset=read(ROOT/'audit/BF_provenance.json')
    assert asset['training_overlap_ids']==[]
    config=OmegaConf.load(repo/'configs/latent-diffusion/mydata_ldm-vq-f8_test.yaml')
    config.model.params.first_stage_config.params.ckpt_path=asset['vq_path']
    model=instantiate_from_config(config.model)
    state=torch.load(asset['checkpoint_path'],map_location='cpu')
    missing,unexpected=model.load_state_dict(state['state_dict'],strict=False)
    assert not any(k.startswith(('model.diffusion_model.','cond_stage_model.','first_stage_model.')) for k in missing)
    del state
    model.cuda().eval().requires_grad_(False);sampler=DDIMSampler(model)
    write(ROOT/'audit'/('BF_loading_'+os.environ.get('SLURM_JOB_ID','local')+'.json'),
          dict(missing=list(missing),unexpected=list(unexpected),EMA=False,parameters=sum(p.numel() for p in model.parameters())))
    runtime=ROOT/'runtime'/('bf_'+os.environ.get('SLURM_JOB_ID','local'));runtime.mkdir(parents=True,exist_ok=True);os.chdir(runtime)
    def tensor(image):
        value=(np.asarray(image).astype('uint8')/127.5-1.).astype('float32')
        if value.ndim==2:value=value[:,:,None]
        return torch.from_numpy(value.transpose(2,0,1)[None]).cuda()
    def run(row):
        sketch=tensor(Image.open(row['sketch']).convert('L').resize((256,256),Image.Resampling.BILINEAR))
        reference=tensor(Image.open(row['reference']).convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
        seed=row['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        condition={'cond':model.get_learned_conditioning({'sketch':sketch,'texture':reference})}
        uncondition={'two_zero':model.get_learned_conditioning({'sketch':torch.zeros_like(sketch),'texture':torch.zeros_like(reference)}),
                     'one_zero':model.get_learned_conditioning({'sketch':torch.zeros_like(sketch),'texture':reference})}
        # 使用独立生成器，保证同身份新旧参考共享噪声；条件编码不影响latent RNG。
        latent=torch.randn((1,4,32,32),device='cuda',generator=torch.Generator(device='cuda').manual_seed(seed))
        initial=digest(latent)
        samples,_=sampler.sample(S=200,conditioning=condition,batch_size=1,shape=[4,32,32],verbose=False,
            unconditional_guidance_scale={'scale_1':1.1,'scale_2':1.5},unconditional_conditioning=uncondition,eta=0.,x_T=latent)
        decoded=torch.clamp((model.decode_first_stage(samples)+1)/2,0,1)
        array=decoded[0].detach().permute(1,2,0).mul(255).add(.5).clamp(0,255).to(torch.uint8).cpu().numpy()
        native=Image.fromarray(array);native_path=ROOT/'native/BF-Fashion'/row['condition']/(row['id']+'.png')
        native_path.parent.mkdir(parents=True,exist_ok=True);native.save(native_path)
        return native.resize((384,512),Image.Resampling.BILINEAR),dict(initial_latent_sha256=initial,text_used=False,
            steps=200,sampler='DDIM',eta=0.,scale_1=1.1,scale_2=1.5,native_size=[256,256],
            native_path=str(native_path),native_sha256=sha(native_path),checkpoint=asset['checkpoint_path'],EMA=False)
    return run


def generate(arm,smoke=False,shard=0,shards=1):
    rows=selected_rows(arm,smoke,shard,shards)
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    with torch.no_grad():
        runner=internal(arm) if arm in ['E0','E1'] else cnipa() if arm=='ControlNet+IP-Adapter' else bf()
        results=[]
        for row in rows:
            path=ROOT/'outputs'/arm/row['condition']/(row['id']+'.png')
            if path.exists():
                meta=read(path.with_suffix('.json'));assert meta['output_sha256']==sha(path)
                assert meta['input_reference_sha256']==row['reference_sha256'] and meta['seed']==row['seed']
                results.append(meta);continue
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();begin=time.perf_counter()
            image,meta=runner(row)
            torch.cuda.synchronize()
            assert image.size==(384,512)
            path.parent.mkdir(parents=True,exist_ok=True);image.save(path)
            meta.update(id=row['id'],arm=arm,condition=row['condition'],seed=row['seed'],
                prompt=row['caption'] if meta['text_used'] else None,GT_used_in_inference=False,
                output_sha256=sha(path),input_sketch_sha256=row['sketch_sha256'],input_reference_sha256=row['reference_sha256'],
                seconds=time.perf_counter()-begin,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                code_commit=__import__('subprocess').check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip())
            write(path.with_suffix('.json'),meta);results.append(meta)
            print('OUTPUT',arm,row['id'],row['condition'],len(results),'/',len(rows),flush=True)
    write(ROOT/'generation'/f'{arm}_{"smoke" if smoke else "formal"}_shard{shard}.json',
          dict(arm=arm,smoke=smoke,n=len(results),shard=shard,shards=shards,complete=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('arm',choices=['E0','E1','BF-Fashion','ControlNet+IP-Adapter'])
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=1)
    args=parser.parse_args();generate(args.arm,args.smoke,args.shard,args.shards)
