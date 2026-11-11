"""原DDIM50轨迹、同状态反事实和同描述子的VAE阳性对照。"""
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from models.e2_compat import load_trainable_strict
from models.e2_geometry import build_geometry, extract_patches
from models.e2_local_fft import PowerSpectrumDescriptor
from models.e2_texture_diagnostics import PATTERNS, activation_vectors, region_change
from tools.e2_fft_experiment import read, write, sha
from tools.e2_texture_bottleneck import condition, image, replay_image, forward_at, fft_vectors
from tools.e2_time_gradient import initialize, code_environment
from tools.e2_time_gradient_common import (FullTrace, predicted_x0, schedule_record,
    scheduler_counterfactual, digest_network, SCORE_STEPS)

CONTROL_VARIANTS=PATTERNS+tuple(x+'_shift' for x in PATTERNS)


def load(args,core):
    spec=read(Path(args.out)/'protocol.json')['checkpoints']['B0']
    assert sha(spec['path'])==spec['sha256']
    pipe,network=core.load_network(args.device)
    state=torch.load(spec['path'],map_location='cpu',weights_only=False)
    load_trainable_strict(network,state['network']);del state
    network.eval().requires_grad_(False)
    pipe.unet.half();pipe.vae.half();network.adapter.half()
    pipe.image_encoder.to(args.device);pipe.bf_texture_conditioner.to(args.device)
    return pipe,network


def masks_for(args,pair):
    from eval.eval_utils import prepare_evaluation_masks
    a=prepare_evaluation_masks((384,512),sketch_path=str(Path(args.data)/pair['sketch']),mask_policy='sketch_only')['garment']
    geometry=build_geometry(torch.from_numpy(a.astype(np.float32))[None,None].to(args.device),details=False)
    return dict(body=geometry.body,boundary=geometry.inner_band,background=geometry.background)


def decode(pipe,x0):return pipe.vae.decode((x0/pipe.vae.config.scaling_factor).to(pipe.vae.dtype)).sample.float()


def save_rgb(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    a=((value[0].float().cpu().permute(1,2,0).numpy()/2+.5).clip(0,1)*255).round().astype(np.uint8)
    Image.fromarray(a).save(path)


class TextureMetrics:
    """v3纯公式，独立记录原始x0曲线，不调用训练审计或冒充最终性能表。"""
    def __init__(self,args,out):
        from tools import t3_response_run as response
        self.args,self.response,self.cache=args,response,{}
        response.ROOT=out;self.model=response.load_dists(args.device)
        self.fft=PowerSpectrumDescriptor()

    def feature(self,parts):
        a=torch.from_numpy(np.stack(parts).transpose(0,3,1,2).copy()).float().to(self.args.device)/255
        return self.model.forward_once(a)

    def refs(self,path,scale,is_gray):
        from tools.e2_followup_data import rgb
        from tools.bf_t3_three_arm_report import gray
        key=(str(path),scale,is_gray)
        if key not in self.cache:
            size=(round(384*scale),round(512*scale));a=np.asarray(rgb(path,size))
            parts=[a[y:y+64,x:x+64] for y in np.linspace(0,size[1]-64,4).round().astype(int)
                for x in np.linspace(0,size[0]-64,4).round().astype(int)]
            self.cache[key]=self.feature([gray(p) for p in parts] if is_gray else parts)
        return self.cache[key]

    def score(self,pair,decoded):
        from tools.bf_t3_three_arm_report import gray
        a=((decoded[0].float().permute(1,2,0).cpu().numpy()/2+.5).clip(0,1)*255).round().astype(np.uint8)
        parts=[a[y:y+64,x:x+64] for x,y in pair['patch_coordinates']];result={}
        for mode in ['gray','rgb']:
            is_gray=mode=='gray';f=self.feature([gray(p) for p in parts] if is_gray else parts)
            distances={}
            for name,key in [('old','original_reference'),('new','swapped_reference')]:
                distances[name]={str(s):float(self.response.distances(self.model,f,self.refs(Path(self.args.data)/pair[key],s,is_gray)).mean()) for s in [.75,1.,1.33]}
            old,new=distances['old'],distances['new']
            result[mode]=dict(**distances,strict_margin=old['1.0']-new['1.0'],
                tolerant_margin=min(old.values())-min(new.values()))
        return result


def pilot(args):
    core,out=initialize(args);pipe,network=load(args,core);before=digest_network(network)
    pairs=read(out/'strict_pairs.json')['pairs'];records=[]
    chosen=[next(p for p in pairs if p['qualification']['old_pattern']==name) for name in ['dots','stripes','checks']]
    with torch.inference_mode():
        for pair in chosen:
            old=condition(args,core,network,pair,'old');new=condition(args,core,network,pair,'new')
            for name,maps in [('old',old),('new',new)]:
                with FullTrace(pipe,network.adapter) as tap:value,latent=image(args,core,pipe,network,pair,maps)
                records.append(replay_image(out,pair,name,value,latent))
                assert len(tap.states)==50 and tap.scheduler_calls==tap.condition_calls==50
                for step,state in tap.states.items():
                    repeated=forward_at(pipe,network,state,maps)
                    assert torch.equal(repeated,state['conditional']['epsilon'])
                    assert torch.equal(scheduler_counterfactual(pipe.scheduler,state['guided'],state),state['next_sample'])
                    if step<49:assert torch.equal(state['next_sample'],tap.states[step+1]['sample'])
            print('TIME_TRACE_PILOT_CASE',pair['case_id'],flush=True)
    after=digest_network(network);assert before==after
    write(out/'audit/pilot_trace.json',dict(passed=True,image_replays=records,all50_repeat_and_next_exact=True,
        network_state_before=before,network_state_after=after,generator_updates=0,environment=code_environment()))
    print('TIME_TRACE_PILOT_PASS',flush=True)


def trace(args):
    core,out=initialize(args);assert read(out/'audit/pilot_trace.json')['passed']
    spec=read(out/'protocol.json');assert spec['environment']['commit']==code_environment()['commit']
    pipe,network=load(args,core);before=digest_network(network);metric=TextureMetrics(args,out)
    pairs=read(out/'strict_pairs.json')['pairs'];completed=[]
    with torch.inference_mode():
        for pos,pair in enumerate(pairs):
            if pos%args.shards!=args.shard:continue
            folder=out/'trace'/pair['case_id'];folder.mkdir(parents=True,exist_ok=True)
            if (folder/'complete.json').exists():
                previous=read(folder/'complete.json')
                assert previous['passed'] and previous['vectors_sha256']==sha(folder/'vectors.npz')
                assert previous['positive_sha256']==sha(folder/'positive.npz') and previous['source_commit']==spec['environment']['commit']
                completed.append(pair['case_id']);continue
            started=time.monotonic();old=condition(args,core,network,pair,'old');new=condition(args,core,network,pair,'new')
            trajectories={};replays=[]
            for name,maps in [('old',old),('new',new)]:
                with FullTrace(pipe,network.adapter) as tap:value,latent=image(args,core,pipe,network,pair,maps)
                replays.append(replay_image(out,pair,name,value,latent));trajectories[name]=tap.states
            regions=masks_for(args,pair);real=[];schedules=[];coordinates=pair['patch_coordinates']
            references={}
            for name in ['old','new']:
                a=core.tensor(out/'inputs/controls'/pair['case_id']/name/'raw.png').to(args.device)
                references[name]=fft_vectors(metric.fft,(a+1)*.5,coordinates)
            raw_x0={}
            for step in range(50):
                a,b=trajectories['old'][step],trajectories['new'][step];sr=schedule_record(pipe.scheduler,a['timestep']);schedules.append(sr)
                assert int(a['timestep'])==int(b['timestep'])
                base=a['guided'];repeat=forward_at(pipe,network,a,old)
                assert torch.equal(repeat,a['conditional']['epsilon'])
                unc=forward_at(pipe,network,a,new,branch='unconditional');assert torch.equal(unc,a['unconditional']['epsilon'])
                new_eps=unc+7*(forward_at(pipe,network,a,new)-unc)
                old_x0=predicted_x0(a['sample'],base,sr['alpha']);new_x0=predicted_x0(a['sample'],new_eps,sr['alpha'])
                raw_x0[step]=old_x0
                next_new=scheduler_counterfactual(pipe.scheduler,new_eps,a)
                identity_error=float(((new_x0-old_x0)+sr['epsilon_to_x0']*(new_eps.float()-base.float())).abs().max())
                row=dict(step=step,t=sr['t'],schedule=sr,
                    common_sample_sha256=core.digest(a['sample']),
                    repeat_epsilon_exact=True,unconditional_independent=True,
                    own_epsilon=region_change(b['guided'],a['guided'],regions),
                    own_conditional=region_change(b['conditional']['epsilon'],a['conditional']['epsilon'],regions),
                    own_unconditional=region_change(b['unconditional']['epsilon'],a['unconditional']['epsilon'],regions),
                    own_latent=region_change(b['sample'],a['sample'],regions),
                    common_epsilon=region_change(new_eps,base,regions),common_x0=region_change(new_x0,old_x0,regions),
                    common_next=region_change(next_new,a['next_sample'],regions),x0_identity_max_error=identity_error)
                row['own_decoded']={}
                for name,state in [('old',a),('new',b)]:
                    x0=old_x0 if name=='old' else predicted_x0(state['sample'],state['guided'],sr['alpha'])
                    value=decode(pipe,x0);f=fft_vectors(metric.fft,(value+1)*.5,coordinates)
                    distances={k:float((f-v).abs().mean()) for k,v in references.items()}
                    data=dict(fft_distances=distances,fft_margin=distances['old']-distances['new'],
                        high_noise_extrapolation=sr['t']>249,decoded_outside_range_fraction=float((value.abs()>1).float().mean()),
                        raw_x0_rms=float(x0.square().mean().sqrt()))
                    if step in SCORE_STEPS:
                        data['v3_scores']=metric.score(pair,value)
                        if pair['case_id'] in spec['short_cases']:save_rgb(folder/'images'/('%02d_%s_x0.png'%(step,name)),value)
                    row['own_decoded'][name]=data
                real.append(row)
            write(folder/'real_trajectory.json',real);write(folder/'schedule.json',schedules)
            vectors={};positive={};control_rows=[]
            for variant in CONTROL_VARIANTS:
                maps=condition(args,core,network,pair,'old',variant);row_vectors={}
                source=core.tensor(out/'inputs/controls'/pair['case_id']/'old'/(variant+'.png')).to(args.device)
                encoded=pipe.vae.encode(source.to(pipe.vae.dtype)).latent_dist.mean
                reconstruction=pipe.vae.decode(encoded).sample.float()
                for name,a in [('input_fft',fft_vectors(metric.fft,(source+1)*.5,coordinates)),
                               ('vae_roundtrip_fft',fft_vectors(metric.fft,(reconstruction+1)*.5,coordinates))]:
                    positive.setdefault(name,[]).append(a.cpu().numpy())
                for kind,array in activation_vectors(encoded).items():positive.setdefault('vae_reference_latent__'+kind,[]).append(array)
                if pair['case_id'] in spec['short_cases']:save_rgb(folder/'images'/('vae_'+variant+'.png'),reconstruction)
                quality=[]
                for step,state in trajectories['old'].items():
                    sr=schedules[step];unc=state['unconditional']['epsilon'];eps=unc+7*(forward_at(pipe,network,state,maps)-unc)
                    x0=predicted_x0(state['sample'],eps,sr['alpha']);base=state['guided'];value=decode(pipe,x0)
                    next_sample=scheduler_counterfactual(pipe.scheduler,eps,state)
                    items={'guided_epsilon':eps,'epsilon_delta':eps.float()-base.float(),'x0_latent':x0,
                        'x0_delta_scaled':(x0-raw_x0[step])/sr['epsilon_to_x0'],'next_latent':next_sample}
                    for name,a in items.items():
                        for kind,array in activation_vectors(a).items():row_vectors.setdefault(name+'__'+kind,[]).append(array)
                    row_vectors.setdefault('decoded_fft',[]).append(fft_vectors(metric.fft,(value+1)*.5,coordinates).cpu().numpy())
                    quality.append(dict(step=step,t=sr['t'],decoded_outside_range_fraction=float((value.abs()>1).float().mean()),
                        raw_x0_rms=float(x0.square().mean().sqrt())))
                for name,values in row_vectors.items():vectors.setdefault(name,[]).append(np.stack(values))
                control_rows.append(dict(variant=variant,label=PATTERNS.index(variant.replace('_shift','')),
                    phase='shifted' if variant.endswith('_shift') else 'original',identity_group=pair['identity_group'],quality=quality))
            np.savez_compressed(folder/'vectors.npz',**{k:np.stack(v) for k,v in vectors.items()})
            np.savez_compressed(folder/'positive.npz',**{k:np.stack(v) for k,v in positive.items()})
            write(folder/'control_rows.json',control_rows)
            write(folder/'complete.json',dict(passed=True,case_id=pair['case_id'],steps=50,replays=replays,
                vectors_sha256=sha(folder/'vectors.npz'),positive_sha256=sha(folder/'positive.npz'),
                source_commit=spec['environment']['commit'],
                real_trajectory_sha256=sha(folder/'real_trajectory.json'),generator_updates=0,seconds=time.monotonic()-started))
            completed.append(pair['case_id']);metric.cache.clear();network.adapter.clear()
            del trajectories,tap,maps,old,new,raw_x0;torch.cuda.empty_cache()
            print('TIME_TRACE_CASE_PASS',pair['case_id'],len(completed),flush=True)
    after=digest_network(network);assert before==after
    write(out/'audit'/('trace_shard_%d.json'%args.shard),dict(passed=True,cases=completed,
        network_state_before=before,network_state_after=after,generator_updates=0,environment=code_environment()))
    print('TIME_TRACE_SHARD_PASS',args.shard,flush=True)
