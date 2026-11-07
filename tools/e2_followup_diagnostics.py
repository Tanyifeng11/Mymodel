"""重放既有采样，不更新参数；区分梯度、patch有效性、图像尺度与VAE瓶颈。"""
from argparse import Namespace
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from models.e2_geometry import InteriorPatchSampler, build_geometry, extract_patches
from models.e2_local_fft import LocalFFTLoss, PowerSpectrumDescriptor, gray
from tools.e2_followup_data import SIZE, read, write, sha, rgb, spectral_proxy


def reference_tensor(path):
    value = torch.from_numpy(np.array(rgb(path))).permute(2,0,1).float()/255.
    return ((value-.5)/.5+1)*.5


def replay(args):
    parent, out = Path(args.parent), Path(args.out)
    rows = read(args.manifest)
    history = [__import__('json').loads(s) for s in (parent/'training/E2_FFT/training.jsonl').read_text().splitlines() if s]
    sampler = InteriorPatchSampler(110642)
    reference_mask = torch.ones(SIZE[1], SIZE[0])
    counts, scales, records = Counter(), Counter(), []
    for update in history:
        for token in update['tokens']:
            counts['samples'] += 1
            if token['t'] > 249:
                counts['high_t'] += 1
                continue
            if .2 <= token['drop'] < .3:
                counts['reference_dropout'] += 1
                continue
            index = token['index']
            path = Path(args.mask_root)/('%03d'%(index//1000))/('%05d_mask.png'%index)
            mask = torch.from_numpy(np.array(Image.open(path).convert('L').resize(SIZE, Image.Resampling.NEAREST))>.5*255)[None,None].float()
            geometry = build_geometry(mask, details=False)
            if not bool(geometry.valid[0]):
                counts['invalid_mask'] += 1
                continue
            counts['active'] += 1
            row = rows[index]
            ref = reference_tensor(Path(args.data)/row['texture'])
            lines = np.asarray(rgb(Path(args.data)/row['sketch']).convert('L')) < 245
            details = []
            for size in [32,64]:
                yc, n = sampler.coordinates(geometry.body[0], size, 4)
                rc, _ = sampler.coordinates(reference_mask, size, 4)
                stds = gray(extract_patches(ref,rc,size)).std((-2,-1),unbiased=False) if rc else torch.zeros(0)
                valid = int((stds>=.02).sum())
                reason = 'valid' if yc and valid else ('no_interior_patch' if not yc else 'low_texture_reference')
                scales['%s/%s'%(size,reason)] += 1
                details.append(dict(size=size,candidates=n,predicted_coordinates=yc,reference_coordinates=rc,
                    reference_stds=stds.tolist(),reference_valid_n=valid,reason=reason,
                    input_line_fraction=[float(lines[y:y+size,x:x+size].mean()) for x,y in yc]))
            valid_scales = sum(x['reason']=='valid' for x in details)
            counts['valid'] += int(valid_scales>0)
            counts['active_without_term'] += int(valid_scales==0)
            records.append(dict(update=update['step'],index=index,t=token['t'],scales=details,valid_scales=valid_scales))
            if counts['active']%250==0:print('REPLAY',dict(counts),flush=True)
    checkpoint = torch.load(parent/'training/E2_FFT/latest.pt',map_location='cpu',weights_only=False)
    expected_rng = checkpoint['states'][0]['patch']['rng']
    rng_exact = torch.equal(expected_rng,sampler.state_dict()['rng'])
    del checkpoint
    expected = {k:sum(x['patch_counts'][key] for x in history) for k,key in [('active','active_n'),('valid','valid_n')]}
    passed = counts['active']==expected['active'] and counts['valid']==expected['valid'] and rng_exact
    write(out/'replay/result.json',dict(passed=passed,counts=dict(counts),expected=expected,scale_reasons=dict(scales),
        final_sampling_rng_exact=rng_exact,history_sha256=sha(parent/'training/E2_FFT/training.jsonl'),
        numeric_scope='CPU重放原坐标与参考std过滤，不计算预测模型，不以零预测模拟训练损失',
        input_line_fraction_scope='输入草图线条混入代理；包含真实花纹，不能当作缝线语义标签',records=records))
    assert passed, '重放未与原训练计数或最终patch RNG一致，保留失败结果'
    print('REPLAY_PASS',dict(counts),'FINAL_RNG_EXACT',rng_exact,flush=True)


def fixed_prediction(pipe, network, core, batch, seed, timestep):
    from diffusers import DDIMScheduler
    device=batch['vae_cloth'].device
    rng=torch.Generator(device=device).manual_seed(seed)
    scheduler=DDIMScheduler(beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear',num_train_timesteps=1000)
    with torch.no_grad():
        posterior=pipe.vae.encode(batch['vae_cloth']).latent_dist
        latent=(posterior.mean+posterior.std*torch.randn(posterior.mean.shape,device=device,generator=rng))*pipe.vae.config.scaling_factor
        sl=pipe.vae.encode(batch['vae_sketch']).latent_dist.mean*pipe.vae.config.scaling_factor
        text=pipe.text_encoder(batch['input_ids'])[0]
        noise=torch.randn(latent.shape,device=device,generator=rng)
        t=torch.full((1,),timestep,device=device,dtype=torch.long)
        sa=core.sketch_states(pipe,sl.to(pipe.reference_unet.dtype),t)
    noisy=scheduler.add_noise(latent,noise,t)
    epsilon=network(noisy,t,text,batch['vae_sketch'],batch['texture_image'],sa)
    alpha=scheduler.alphas_cumprod.to(device)[t].reshape(-1,1,1,1)
    x0=(noisy.float()-(1-alpha).sqrt()*epsilon.float())/alpha.sqrt()
    vae_input=x0/pipe.vae.config.scaling_factor
    decoded=pipe.vae.decode(vae_input).sample.float()
    return dict(epsilon=epsilon,x0=x0,vae_input=vae_input,decoded=decoded,t=t,noise_sha256=core.digest(noise))


def probe(args):
    from tools.e2_fft_experiment import setup, load, environment
    from tools.e2_fft_training import dataset
    from models.e2_compat import load_trainable_strict
    parent,out=Path(args.parent),Path(args.out)
    # 与已通过的梯度/续训探针共用确定性反传设置，不更改历史推理默认后端。
    probe_args=Namespace(**vars(args));probe_args.action='calibrate'
    core,_=setup(probe_args)
    pipe,network,_=load(probe_args,core)
    data=dataset(pipe,core)
    descriptor=PowerSpectrumDescriptor()
    vae_rows=[]
    for record in read(out/'inputs/dev64.json')['records']:
        index=record['index'];item=data[index]
        mask=item['garment_mask'][None].to(args.device)
        geometry=build_geometry(mask,details=False)
        sampler=InteriorPatchSampler(110803+index)
        row=dict(index=index,scales=[])
        with torch.no_grad():
            source={k:item[v][None].to(args.device) for k,v in [('gt','vae_cloth'),('ref','texture_image')]}
            reconstructed={k:pipe.vae.decode(pipe.vae.encode(v).latent_dist.mean).sample.float() for k,v in source.items()}
            for size in [32,64]:
                yc,_=sampler.coordinates(geometry.body[0],size,4)
                rc,_=sampler.coordinates(torch.ones(SIZE[1],SIZE[0]),size,4)
                part=dict(size=size,gt_coordinates=yc,reference_coordinates=rc)
                patches={}
                for k,coords in [('gt',yc),('ref',rc)]:
                    if not coords:continue
                    original=extract_patches((source[k][0]+1)*.5,coords,size)
                    rec=extract_patches((reconstructed[k][0]+1)*.5,coords,size)
                    a,b=descriptor(original),descriptor(rec)
                    part[k+'_vae_spectrum_l1']=float((a-b).abs().mean())
                    part[k+'_original_proxy']=[spectral_proxy(x.permute(1,2,0).float().cpu().numpy()) for x in original]
                    part[k+'_reconstructed_proxy']=[spectral_proxy(x.permute(1,2,0).float().cpu().numpy()) for x in rec]
                    patches[k]=a.mean(0)
                if len(patches)==2:part['gt_reference_spectrum_l1']=float((patches['gt']-patches['ref']).abs().mean())
                row['scales'].append(part)
            if index in [x['index'] for x in read(out/'inputs/dev64.json')['records'][:16]]:
                panel=Image.new('RGB',(1536,512),'white')
                for col,(name,value) in enumerate([('GT',source['gt']),('VAE_GT',reconstructed['gt']),('reference',source['ref']),('VAE_reference',reconstructed['ref'])]):
                    arr=((value[0].cpu().float()+1)*127.5).clamp(0,255).round().byte().permute(1,2,0).numpy()
                    im=Image.fromarray(arr);ImageDraw.Draw(im).text((5,5),name,fill='red');panel.paste(im,(col*384,0))
                path=out/('vae/%s.png'%index);path.parent.mkdir(exist_ok=True);panel.save(path)
        vae_rows.append(row)
        print('VAE_DIAGNOSTIC',index,flush=True)
    write(out/'vae/result.json',dict(records=vae_rows,environment=environment(),posterior='mean',
        physical_scale_known=False,optimizer_updates=0))
    choices=read(parent/'split.json')['smoke']
    gradients=[]
    network.train()
    for arm in ['mature_E2','E2','E2_FFT']:
        checkpoint=None
        if arm!='mature_E2':
            checkpoint=parent/'training'/arm/'latest.pt'
            assert sha(checkpoint)==read(checkpoint.with_name('complete.json'))['checkpoint_sha256']
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            load_trainable_strict(network,state['network']);del state
        for choice in choices:
            index=choice['index'];item=data[index]
            batch={k:v[None].to(args.device) for k,v in item.items()}
            for timestep in [100,249]:
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    captured=fixed_prediction(pipe,network,core,batch,110800+index,timestep)
                fft=LocalFFTLoss(seed=110900+index)
                loss,stats=fft(captured['decoded'],batch['texture_image'],batch['garment_mask'],captured['t'],batch['dataset_index'])
                targets=[captured[k] for k in ['epsilon','x0','vae_input']]+[network.adapter.projections[0].weight,
                    network.adapter.fusion.downconv1[0].weight,next(p for p in network.unet.parameters() if p.requires_grad)]
                values=torch.autograd.grad(loss,targets,allow_unused=True)
                norms={k:float(v.float().norm()) if v is not None else 0. for k,v in zip(['epsilon','x0','vae_input','projection0','fusion_down1','GAM'],values)}
                assert all(np.isfinite(v) for v in norms.values())
                if stats['valid_n']:assert all(v>0 for v in norms.values())
                gradients.append(dict(arm=arm,index=index,t=timestep,loss=float(loss),gradients=norms,
                    patch_stats=stats,noise_sha256=captured['noise_sha256']))
                network.adapter.clear();del captured,loss,values,targets
                print('GRADIENT',arm,index,timestep,norms['epsilon'],flush=True)
    write(out/'gradients/result.json',dict(passed=True,records=gradients,environment=environment(),optimizer_updates=0,
        scope='固定原校准候选、相同噪声、t100/t249；检查梯度接线，不作为独立质量实验'))
