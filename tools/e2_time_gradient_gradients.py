"""原配对训练路径上的分项梯度竞争；不运行optimizer.step。"""
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from models.e2_compat import load_trainable_strict, e2_objective, observe_prediction
from models.e2_geometry import build_geometry
from models.e2_local_fft import LocalFFTLoss
from models.e2_local_texture import GaborDescriptor, ReferenceGramDescriptor, DescriptorLocalLoss
from tools.e2_fft_experiment import read, write, sha
from tools.e2_time_gradient import initialize, code_environment
from tools.e2_time_gradient_common import (TRAIN_TIMES, NOISE_SEEDS, OLD_WEIGHTS, digest_network,
    predicted_x0, gradient_pair, gradient_summary, same_patch_objective)


class FixedCoordinates:
    def __init__(self, record):self.record=record
    def coordinates(self,mask,size,count=4):
        part=self.record[str(size)];kind='reference' if bool((mask==1).all()) else 'body'
        result=[tuple(x) for x in part[kind]]
        return result,part['body_candidates'] if kind=='body' else (mask.shape[-1]-size+1)*(mask.shape[-2]-size+1)


def load(args,core):
    from train_GAM_texture_joint import JointTextureDataset,VGGGramStyleLoss
    pipe,network=core.load_network(args.device)
    data=JointTextureDataset(str(Path(args.out)/'inputs/training_manifest.json'),pipe.tokenizer,
        str(Path(args.out)/'inputs/train'),width=384,height=512,texture_preprocess_mode='plain_resize',
        mask_cache=str(Path(args.out)/'inputs/train_masks'))
    style=VGGGramStyleLoss().to(args.device).eval().requires_grad_(False)
    return pipe,network,data,style


def load_checkpoint(network,record):
    assert sha(record['path'])==record['sha256']
    state=torch.load(record['path'],map_location='cpu',weights_only=False)
    load_trainable_strict(network,state['network']);del state
    network.train()


def old_terms(pipe,captured,batch,style):
    from train_GAM_texture_joint import masked_edge_l1,texture_color_stat_loss,build_region_masks,masked_l1_loss
    decoded=captured['decoded'];mask=batch['garment_mask'];reference=batch['texture_image']
    mse=F.mse_loss(captured['epsilon'].float(),captured['noise'].float())
    gram=style(decoded,batch['vae_cloth'],mask=mask)
    edge=masked_edge_l1(decoded,batch['vae_sketch'],mask=mask)
    color=texture_color_stat_loss(decoded,reference,mask)
    body,_,_=build_region_masks(mask,9)
    body_rgb=texture_color_stat_loss(decoded,reference,body)
    rest=masked_l1_loss(decoded,batch['vae_cloth'],1-body)
    return dict(base=mse+gram+.05*edge+.1*color,body=body_rgb,rest=rest,mse=mse,gram=gram,edge=edge,color=color)


def fixed_capture(pipe,network,core,batch,seed,timestep,empty_text=False):
    from diffusers import DDIMScheduler
    rng=torch.Generator(device=batch['vae_cloth'].device).manual_seed(seed)
    scheduler=DDIMScheduler(beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear',num_train_timesteps=1000)
    device=batch['vae_cloth'].device
    with torch.no_grad():
        posterior=pipe.vae.encode(batch['vae_cloth']).latent_dist
        latent=(posterior.mean+posterior.std*torch.randn(posterior.mean.shape,device=device,generator=rng))*pipe.vae.config.scaling_factor
        sketch_latent=pipe.vae.encode(batch['vae_sketch']).latent_dist.mean*pipe.vae.config.scaling_factor
        drawn_drop=float(torch.rand((),device=device,generator=rng))
        text_ids=batch['input_ids']
        if empty_text:
            text_ids=pipe.tokenizer('',padding='max_length',truncation=True,max_length=77,return_tensors='pt').input_ids.to(device)
        text=pipe.text_encoder(text_ids)[0]
        noise=torch.randn(latent.shape,device=device,dtype=latent.dtype,generator=rng)
        drawn_t=torch.randint(0,1000,(len(latent),),device=device,generator=rng);t=drawn_t.clone().fill_(timestep)
        sa=core.sketch_states(pipe,sketch_latent.to(pipe.reference_unet.dtype),t)
        noisy=scheduler.add_noise(latent,noise,t)
    epsilon=network(noisy,t,text,batch['vae_sketch'],batch['texture_image'],sa)
    alpha=scheduler.alphas_cumprod.to(device)[t].reshape(-1,1,1,1)
    x0=(noisy.float()-(1-alpha).sqrt()*epsilon.float())/alpha.sqrt()
    decoded=pipe.vae.decode(x0/pipe.vae.config.scaling_factor).sample.float()
    return dict(epsilon=epsilon,decoded=decoded,x0=x0,noisy=noisy,alpha=alpha,noise=noise,t=t,
        features=list(network.adapter.features),noise_sha256=core.digest(noise),latent_sha256=core.digest(latent),
        drawn_drop=drawn_drop,drawn_t=int(drawn_t[0]),policy='diagnostic text/reference kept;original RNG draws consumed')


def targets_for(network,captured):
    names=['epsilon','decoded']+['feature%d'%i for i in range(4)];values=[captured['epsilon'],captured['decoded']]+captured['features']
    selected=[('projection%d'%i,p.weight) for i,p in enumerate(network.adapter.projections)]
    fusion=network.adapter.fusion
    selected += [('sketch_conv1',fusion.downconv1[0].weight),('texture_conv1',fusion.G_downconv1[0].weight),
        ('texture_conv2',fusion.G_downconv2[1].weight)]
    for label in ['G_bottleneck_alpha_2','G_bottleneck_beta_2','G_bottleneck_alpha_5','G_bottleneck_beta_5']:
        selected.append((label,next(getattr(fusion,label).parameters())))
    gam_name,gam=next((n,p) for n,p in network.unet.named_parameters() if p.requires_grad)
    selected.append(('GAM_representative',gam))
    full_names={id(p):n for n,p in network.named_parameters()}
    metadata={label:dict(parameter_name=full_names[id(value)],shape=list(value.shape)) for label,value in selected}
    for label,value in selected:names.append(label);values.append(value)
    metadata.update({n:dict(tensor_shape=list(v.shape)) for n,v in zip(names[:6],values[:6])})
    return names,values,metadata


def objectives(style,choice):
    terms={'fft':LocalFFTLoss(t_max=999),'gabor':DescriptorLocalLoss(GaborDescriptor(),t_max=999),
        'gram':DescriptorLocalLoss(ReferenceGramDescriptor(style),t_max=999)}
    for term in terms.values():term.sampler=FixedCoordinates(choice['coordinates'])
    return terms


def component_gradients(loss,targets):
    return [None if x is None else x.detach().float().cpu() for x in
        torch.autograd.grad(loss,targets,retain_graph=True,allow_unused=True)]


def measure(args,core,pipe,network,batch,style,choice,seed,timestep,weights):
    started=time.monotonic()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        captured=fixed_capture(pipe,network,core,batch,seed,timestep)
        raw=old_terms(pipe,captured,batch,style)
        terms=objectives(style,choice)
        extra={};stats={}
        for name,module in terms.items():
            value,current=module(captured['decoded'],batch['texture_image'],batch['garment_mask'],captured['t'],batch['dataset_index'],reference_keep=True)
            extra[name]=weights[name]*value;stats[name]=current
        valid_sizes={str(x['size']) for x in stats['gram']['samples'][0]['scales'] if x.get('reference_valid_n',0)>0 and x['predicted_n']>0}
        matched_coordinates={k:[tuple(x) for x in v['body']] for k,v in choice['coordinates'].items() if k in valid_sizes}
        matched=weights['gram']*same_patch_objective(captured['decoded'],batch['vae_cloth'],
            matched_coordinates,terms['gram'].descriptor)
        losses={'old_total':e2_objective(raw),**{'old_'+k:OLD_WEIGHTS[k]*raw[k] for k in OLD_WEIGHTS},
            'GT_localGram_diagnostic':matched,**{'counterfactual_ref_'+k:v for k,v in extra.items()}}
        active={name:timestep<=249 and stats[name]['valid_n']>0 for name in extra}
        for name in extra:
            if active[name]:losses['actual_joint_'+name]=losses['old_total']+extra[name]
    assert all(torch.isfinite(value) for value in losses.values())
    names,targets,metadata=targets_for(network,captured);gradients={};summaries={}
    for name,loss in losses.items():
        gradients[name]=component_gradients(loss,targets)
        summaries[name]={target:gradient_summary(g) for target,g in zip(names,gradients[name])}
    repeated_gradient_exact=None
    if args.action=='pilot-gradient':
        for name in ['old_total','counterfactual_ref_fft']:
            repeated=component_gradients(losses[name],targets)
            assert all((a is None and b is None) or (a is not None and b is not None and torch.equal(a,b)) for a,b in zip(repeated,gradients[name])),name+'重复梯度不一致'
        repeated_gradient_exact=True
    geometry=build_geometry(batch['garment_mask'],details=False)
    regions=dict(body=geometry.body.cpu(),boundary=geometry.inner_band.cpu(),background=geometry.background.cpu())
    comparisons=[]
    for aux in extra:
        ref=gradients['counterfactual_ref_'+aux]
        for old in ['old_total']+['old_'+k for k in OLD_WEIGHTS]+['GT_localGram_diagnostic']:
            for index,target in enumerate(names):
                record=dict(old_loss=old,reference_loss=aux,target=target,real_active=active[aux],
                    **gradient_pair(gradients[old][index],ref[index]))
                if target in ['epsilon','decoded']:
                    record['regions']={region:gradient_pair(gradients[old][index],ref[index],mask) for region,mask in regions.items()}
                comparisons.append(record)
        joint=gradients.get('actual_joint_'+aux,gradients['old_total'])
        for index,target in enumerate(names):
            comparisons.append(dict(old_loss='actual_joint_'+aux,reference_loss=aux,target=target,real_active=active[aux],
                **gradient_pair(joint[index],ref[index] if active[aux] else None)))
    record=dict(index=choice['index'],identity=choice['identity'],pattern_candidate=choice['pattern_candidate'],seed=seed,t=timestep,
        reference_kept=True,forced_condition_diagnostic=True,drawn_drop=captured['drawn_drop'],drawn_t=captured['drawn_t'],
        noise_sha256=captured['noise_sha256'],latent_sha256=captured['latent_sha256'],
        alpha=float(captured['alpha'].flatten()[0]),decoded_outside_range_fraction=float((captured['decoded'].abs()>1).float().mean()),
        losses={k:float(v.detach()) for k,v in losses.items()},old_raw={k:float(v.detach()) for k,v in raw.items()},
        weights=dict(old=OLD_WEIGHTS,reference=weights),targets=metadata,gradients=summaries,comparisons=comparisons,
        actual_reference_active=active,patch_stats=stats,matched_GT_local_sizes=sorted(valid_sizes),
        repeated_gradient_exact=repeated_gradient_exact,
        actual_ref_epsilon_norm={name:summaries['counterfactual_ref_'+name]['epsilon']['norm'] if active[name] else 0. for name in extra},
        directional_scope='shared epsilon/parameter representative gradients,not actual AdamW step',generator_updates=0,seconds=time.monotonic()-started)
    frame=dict(epsilon=captured['epsilon'].detach().float(),noisy=captured['noisy'].detach().float(),alpha=captured['alpha'].detach().float())
    network.adapter.clear()
    return record,frame


def directional_check(pipe,batch,style,choice,frame,weights):
    """FP32共同epsilon的数值方向导数；不改变任何模型参数或模拟成真实AdamW训练。"""
    coordinates={k:[tuple(x) for x in v['body']] for k,v in choice['coordinates'].items()}
    terms=objectives(style,choice);results=[]
    def losses(eps):
        x0=predicted_x0(frame['noisy'],eps,frame['alpha'])
        decoded=pipe.vae.decode(x0/pipe.vae.config.scaling_factor).sample.float()
        gram=style(decoded,batch['vae_cloth'],mask=batch['garment_mask'])
        values={}
        for name in ['fft','gram']:
            loss,_=terms[name](decoded,batch['texture_image'],batch['garment_mask'],
                torch.full((1,),100,device=eps.device,dtype=torch.long),batch['dataset_index'],reference_keep=True)
            values[name]=weights[name]*loss
        return gram,values
    with torch.autocast('cuda',enabled=False):
        eps=frame['epsilon'].clone().requires_grad_(True);gt,reference=losses(eps)
        gt_gradient=torch.autograd.grad(gt,eps,retain_graph=True)[0]
        for name in ['fft','gram']:
            gradient=torch.autograd.grad(reference[name],eps,retain_graph=True)[0]
            norm=float(gradient.double().norm())
            if not norm:
                results.append(dict(reference_loss=name,valid=False,reason='zero_reference_gradient'));continue
            direction=-gradient/gradient.square().mean().sqrt()
            expected=float((gt_gradient.double()*direction.double()).sum())
            steps=[]
            for h in [.001,.002]:
                with torch.no_grad():plus,_=losses(eps.detach()+h*direction);minus,_=losses(eps.detach()-h*direction)
                measured=float((plus-minus)/(2*h))
                steps.append(dict(h=h,gt_gram_directional_derivative=measured,
                    absolute_error=abs(measured-expected),relative_error=abs(measured-expected)/max(abs(expected),1e-8)))
            results.append(dict(reference_loss=name,valid=True,fp32_autograd_derivative=expected,
                gt_gram_increases_when_following_reference=expected>0,steps=steps,
                scope='FP32 downstream epsilon perturbation;L1 kinks/curvature reported;not optimizer causality'))
    return results


def legacy_parity(core,pipe,network,batch,style,seed):
    # 此处仅比较数值，避免同时保留两份SD计算图；实际梯度另在measure中重复验证。
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        with observe_prediction(pipe,network) as old:
            raw,token=core.losses(pipe,network,batch,torch.Generator(device=batch['vae_cloth'].device).manual_seed(seed),style,True)
        # 原audit_mode强制drop=0，按旧代码语义代表空文本、保留参考。
        # 这里只匹配该回归条件；正式诊断仍使用原配对文本和参考。
        current=fixed_capture(pipe,network,core,batch,seed,100,empty_text=True);new_raw=old_terms(pipe,current,batch,style)
    errors={k:float((raw[k]-new_raw[k]).abs()) for k in raw}
    errors.update(epsilon=float((old['epsilon']-current['epsilon']).abs().max()),
        decoded=float((old['decoded']-current['decoded']).abs().max()))
    assert token['noise_sha256']==current['noise_sha256'] and max(errors.values())==0
    network.adapter.clear()
    return dict(passed=True,max_error=max(errors.values()),errors=errors,noise_equal=True,
        scope='original audit_mode:t100,empty text,reference kept;primary diagnosis uses paired text')


def pilot(args):
    core,out=initialize(args,True);pipe,network,data,style=load(args,core);spec=read(out/'protocol.json')
    probes=read(out/'inputs/training_probes.json')['records']
    choices=[probes[0],next(x for x in probes if x['pattern_candidate']=='low_texture')];records=[];parity={}
    for checkpoint in ['mature','B0']:
        load_checkpoint(network,spec['checkpoints'][checkpoint]);before=digest_network(network)
        for choice in choices:
            batch={k:v[None].to(args.device) for k,v in data[choice['index']].items()}
            parity[checkpoint+'_'+str(choice['index'])]=legacy_parity(core,pipe,network,batch,style,NOISE_SEEDS[0]+choice['index'])
            for timestep in [100,250]:
                record,frame=measure(args,core,pipe,network,batch,style,choice,NOISE_SEEDS[0]+choice['index'],timestep,spec['auxiliary_weights'])
                if timestep==250:assert all(v==0 for v in record['actual_ref_epsilon_norm'].values())
                if choice['pattern_candidate']=='low_texture':assert not any(record['actual_reference_active'].values())
                if timestep==100 and choice['pattern_candidate']!='low_texture':
                    record['directional_check']=directional_check(pipe,batch,style,choice,frame,spec['auxiliary_weights'])
                records.append(record);del frame;torch.cuda.empty_cache()
        after=digest_network(network);assert before==after
        print('GRADIENT_PILOT_CHECKPOINT_PASS',checkpoint,flush=True)
    write(out/'audit/pilot_gradient.json',dict(passed=True,legacy_parity=parity,records=records,
        network_parameters_and_buffers_unchanged=True,generator_updates=0,environment=code_environment(),
        peak_gib=torch.cuda.max_memory_allocated()/2**30))
    print('GRADIENT_PILOT_PASS',flush=True)


def gradient(args):
    core,out=initialize(args,True);assert read(out/'audit/pilot_gradient.json')['passed']
    spec=read(out/'protocol.json');assert spec['environment']['commit']==code_environment()['commit']
    pipe,network,data,style=load(args,core);probes=read(out/'inputs/training_probes.json')['records'];audit=[]
    for checkpoint in ['mature','B0']:
        load_checkpoint(network,spec['checkpoints'][checkpoint]);before=digest_network(network);completed=[]
        for pos,choice in enumerate(probes):
            if pos%args.shards!=args.shard:continue
            folder=out/'gradient'/checkpoint/str(choice['index']);folder.mkdir(parents=True,exist_ok=True)
            if (folder/'complete.json').exists():
                previous=read(folder/'complete.json');assert previous['passed'] and sha(folder/'points.json')==previous['points_sha256']
                assert previous['source_commit']==spec['environment']['commit']
                completed.append(choice['index']);continue
            batch={k:v[None].to(args.device) for k,v in data[choice['index']].items()};records=[];started=time.monotonic()
            for seed in NOISE_SEEDS:
                for timestep in TRAIN_TIMES:
                    actual_seed=seed+choice['index']
                    record,frame=measure(args,core,pipe,network,batch,style,choice,actual_seed,timestep,spec['auxiliary_weights'])
                    if timestep==100 and seed==NOISE_SEEDS[0]:
                        record['directional_check']=directional_check(pipe,batch,style,choice,frame,spec['auxiliary_weights'])
                    records.append(record);del frame
                    write(folder/'progress.json',dict(records=records,checkpoint=checkpoint))
            write(folder/'points.json',records)
            write(folder/'complete.json',dict(passed=True,index=choice['index'],identity=choice['identity'],points=len(records),
                points_sha256=sha(folder/'points.json'),source_commit=spec['environment']['commit'],generator_updates=0,seconds=time.monotonic()-started))
            completed.append(choice['index']);torch.cuda.empty_cache()
            print('GRADIENT_CASE_PASS',checkpoint,choice['index'],len(completed),flush=True)
        after=digest_network(network);assert before==after
        audit.append(dict(checkpoint=checkpoint,cases=completed,network_state_before=before,network_state_after=after))
    write(out/'audit'/('gradient_shard_%d.json'%args.shard),dict(passed=True,checkpoints=audit,generator_updates=0,
        environment=code_environment(),peak_gib=torch.cuda.max_memory_allocated()/2**30))
    print('GRADIENT_SHARD_PASS',args.shard,flush=True)
