"""已授权的冻结E2纹理瓶颈诊断；不训练生成器、不增加纹理编码器。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
import torch

from models.e2_compat import load_trainable_strict
from models.e2_geometry import build_geometry, extract_patches
from models.e2_local_fft import PowerSpectrumDescriptor, gray
from models.e2_texture_diagnostics import (STEPS, VARIANTS, PATTERNS, reference_control,
    color_signature, spectrum_signature, FeatureTap, TrajectoryTap, feature_mix,
    sketch_strength, region_change)
from tools.e2_fft_experiment import SOURCE, setup, environment, read, write, sha

BASELINE_SHA = '0f6d220fb5371f7222caeee94166067977102a505252576cdb910f199ad02a3a'
PAIRS_SHA = 'c36330795b0702ea2153647e427986f1a32627757198e2168e4eb5593d00d2a0'
SHORT_GROUPS = ('baseline', 'delta2', 'gam03', 'gam0')


def diagnostic_environment():
    value = environment()
    for name in ['models/e2_texture_diagnostics.py', 'tools/e2_texture_bottleneck.py',
                 'tools/e2_texture_bottleneck_eval.py', 'tests/test_e2_texture_diagnostics.py',
                 'scripts/e2_texture_bottleneck.sbatch', 'pipelines/IMAGGarment_pipeline.py']:
        value['source'][name] = sha(SOURCE / name)
    value.update(generator_updates=0, encoder_added=False, texture_tokens=False, A=False, B=False, C=False)
    return value


def initialize(args):
    core, out = setup(args)
    args.data = str(out / 'inputs/frozen')
    return core, out


def prepare(args):
    core, out = initialize(args); parent = Path(args.parent)
    assert sha(parent / 'strict_pairs.json') == PAIRS_SHA
    complete = read(parent / 'training/B0/complete.json')
    assert complete['checkpoint_sha256'] == BASELINE_SHA and sha(parent / 'training/B0/latest.pt') == BASELINE_SHA
    assert read(parent / 'audit/training_inputs_final.json')['passed']
    for name in ['models/bf_original_fusion.py', 'models/bf_sd_skip.py', 'models/e2_compat.py',
                 'tools/bf_t3_frozen_core.py', 'train_GAM_texture_joint.py']:
        assert sha(SOURCE / name) == complete['environment']['source'][name]
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(parent / 'strict_pairs.json', out / 'strict_pairs.json')
    pairs = read(out / 'strict_pairs.json')['pairs']; assert len(pairs) == 32
    groups = [p['identity_group'] for p in pairs]; assert len(set(groups)) == 32
    order = np.random.default_rng(110111).permutation(32)
    folds = {groups[int(i)]: int(j % 4) for j, i in enumerate(order)}
    short = set()
    for pattern in ['dots', 'stripes', 'checks']:
        short.update(p['case_id'] for p in pairs if p['case_id'] in
                     [x['case_id'] for x in pairs if x['qualification']['old_pattern']==pattern][:2])
    inputs, controls, preprocessing = {}, [], []
    for pair in pairs:
        for key in ['sketch', 'gt', 'original_reference', 'swapped_reference']:
            src = Path(args.frozen_inputs) / pair[key]; assert sha(src) == pair['hashes'][key]
            dst = out / 'inputs/frozen' / pair[key]; dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst); inputs[pair[key]] = dict(bytes=dst.stat().st_size, sha256=sha(dst))
        for condition, key in [('old','original_reference'), ('new','swapped_reference')]:
            native=Image.open(out/'inputs/frozen'/pair[key]).convert('RGB')
            resized=native.resize((384,512),Image.Resampling.BILINEAR);base=np.asarray(resized)
            restored=np.asarray(resized.resize(native.size,Image.Resampling.BILINEAR),np.float32)
            preprocessing.append(dict(case_id=pair['case_id'],condition=condition,native_size=list(native.size),
                model_size=[384,512],scale_xy=[384/native.width,512/native.height],
                anisotropic_resize=384/native.width!=512/native.height,
                bilinear_roundtrip_rgb_mae=float(np.abs(restored-np.asarray(native,np.float32)).mean()/255),
                input_tensor_sha256=core.digest(core.tensor(out/'inputs/frozen'/pair[key])),
                normalization='RGB uint8 /127.5-1;BILINEAR;no reference segmentation;controls applied after resize',
                roundtrip_interpretation='diagnostic interpolation change,not reconstruction quality or proof of lost pattern'))
            gray_std = float((base.astype(np.float64) @ np.array([.299,.587,.114])/255).std())
            signatures = [spectrum_signature(reference_control(base,p,pair['seed'])) for p in PATTERNS]
            eligible = gray_std >= .02 and all(int(s.argmax()) == i and float(s[i]) >= .1 for i,s in enumerate(signatures))
            for variant in VARIANTS:
                value = reference_control(base, variant, 110111 + pair['index'])
                assert np.array_equal(color_signature(base), color_signature(value))
                path = out / 'inputs/controls' / pair['case_id'] / condition / (variant+'.png')
                path.parent.mkdir(parents=True, exist_ok=True); Image.fromarray(value).save(path)
                if variant=='raw':
                    assert torch.equal(core.tensor(path),core.tensor(out/'inputs/frozen'/pair[key]))
                controls.append(dict(case_id=pair['case_id'],identity_group=pair['identity_group'],condition=condition,
                    variant=variant,file=path.relative_to(out).as_posix(),sha256=sha(path),color=color_signature(value).tolist(),
                    spectrum=spectrum_signature(value).tolist(),gray_std=gray_std,readout_eligible=eligible))
            src = parent / 'outputs/B0' / pair['case_id'] / (condition+'.png'); meta=read(src.with_suffix('.json'))
            assert sha(src)==meta['output_sha256'] and meta['checkpoint_sha256']==BASELINE_SHA
            dst = out / 'outputs/baseline' / pair['case_id'] / src.name; dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(src,dst)
            meta.update(group='baseline',reused=True,parent_metadata_sha256=sha(src.with_suffix('.json')),parent_output=str(src),generator_updates=0)
            write(dst.with_suffix('.json'),meta)
            if condition=='old' and pair['case_id'] in short:
                target=out/'outputs/delta2'/pair['case_id']/src.name;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(src,target);write(target.with_suffix('.json'),dict(meta,group='delta2',feature_gain=0.))
    followup = Path(args.followup)
    reused = {}
    for name in ['gradients/result.json','vae/result.json']:
        src=followup/name;target=out/'reused_evidence'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,target);reused[name]=dict(source=str(src),sha256=sha(src))
    write(out/'audit/input_files.json',inputs);write(out/'inputs/controls.json',controls)
    write(out/'audit/reference_preprocessing.json',preprocessing)
    write(out/'audit/reused_evidence.json',reused)
    write(out/'protocol.json',dict(version='E2_texture_bottleneck_v1_frozen_no_generator_training',
        checkpoint=str(parent/'training/B0/latest.pt'),checkpoint_sha256=BASELINE_SHA,manifest_sha256=PAIRS_SHA,
        input_source=str(args.frozen_inputs),input_identity='previous A frozen archive;all128 original B pair file SHA required equal',
        folds=folds,readout='4fold identity-grouped ridge penalty1;global moments and moments+32channels pooled4x4;original/shifted heldout tests',
        readout_labels='four histogram-preserving controlled gratings:vertical/horizontal period16/32;not real-pattern classification',
        readout_eligibility='input graystd>=.02 and all four intended spectral peaks dominant>=.1;no generated output selection',
        controls=list(VARIANTS),steps=list(STEPS),short_cases=[p['case_id'] for p in pairs if p['case_id'] in short],
        controlled_downstream='old-reference four gratings+phase shifts;common old latent,t,text;steps25/40/49 guided epsilon readout;low-t decoded fixedpatch FFT readout;not full synthetic generation',
        short_groups=list(SHORT_GROUPS),new_final_outputs=30,reused_final_outputs=70,
        sampling='unchanged384x512,DDIM50,CFG7,caption/negative/seed;GAM sketchscale=.6/.3/0 diagnostic groups only',
        delta2='F(sketch,old)+2*(F(sketch,new)-F(sketch,old));not pure texture stream;GAM intervention retains original BF sketch input',
        regions='body=original square9 erosion;boundary=foreground-body;background=1-foreground;area pooled',
        x0='actual guided epsilon,actual scheduler alpha;decode only t<=249;fixed eight body patches;FFT margin exploratory,not v3',
        reuse='baseline64 original PNG and metadata;old VAE/gradient evidence;no replay of training',
        statistics='diagnostic grouped bootstrap CI only;no new confirmatory p-values;original performance tables unchanged',
        limitations='synthetic grating readout is not texture fidelity;readout failure is not information-theoretic absence;32previouslyused unbalanced pairs;upstream exposure unknown',
        generator_updates=0,environment=diagnostic_environment()))
    print('TEXTURE_DIAG_PREPARE_PASS',len(controls),len(short),flush=True)


def weights_digest(network):
    h=hashlib.sha256()
    for name,value in network.named_parameters():
        h.update(name.encode());h.update(str((list(value.shape),str(value.dtype))).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_frozen(args,core):
    out=Path(args.out);spec=read(out/'protocol.json');assert sha(spec['checkpoint'])==BASELINE_SHA
    pipe,network=core.load_network(args.device)
    state=torch.load(spec['checkpoint'],map_location='cpu',weights_only=False)
    load_trainable_strict(network,state['network']);del state
    network.eval().requires_grad_(False)
    pipe.unet.half();pipe.vae.half();network.adapter.half()
    pipe.image_encoder.to(args.device);pipe.bf_texture_conditioner.to(args.device)
    return pipe,network


def image(args,core,pipe,network,pair,features,scale=.6):
    network.adapter.features=features
    latent=torch.randn((1,4,64,48),device=args.device,dtype=torch.float16,
        generator=torch.Generator(device=args.device).manual_seed(pair['seed']))
    digest=core.digest(latent)
    value=pipe(ref_image=core.tensor(Path(args.data)/pair['sketch']).to(args.device).half(),
        prompt=pair['caption'],texture_clip_image=None,texture_embeds=None,null_prompt='',negative_prompt=' worst quality, low quality',
        width=384,height=512,num_images_per_prompt=1,guidance_scale=7.,sketch_scale=scale,ipa_scale=0.,latents=latent,
        generator=torch.Generator(device=args.device).manual_seed(pair['seed']),num_inference_steps=50)[0]
    network.adapter.clear()
    return value,digest


def condition(args,core,network,pair,condition_name,variant='raw'):
    out=Path(args.out);reference=out/'inputs/controls'/pair['case_id']/condition_name/(variant+'.png')
    sketch=core.tensor(Path(args.data)/pair['sketch']).to(args.device).half()
    network.adapter.condition(sketch,core.tensor(reference).to(args.device).half())
    return list(network.adapter.features)


def replay_image(out,pair,condition_name,value,latent_sha):
    original=out/'outputs/baseline'/pair['case_id']/(condition_name+'.png')
    error=int(np.abs(np.asarray(value).astype(int)-np.asarray(Image.open(original).convert('RGB')).astype(int)).max())
    assert error==0 and latent_sha==read(original.with_suffix('.json'))['initial_latent_sha256']
    return dict(case_id=pair['case_id'],condition=condition_name,max_pixel_error=error,latent_equal=True)


def forward_at(pipe,network,state,features,scale=.6,branch='conditional'):
    network.adapter.features=features
    with sketch_strength(pipe,scale):
        value=pipe.unet(*state[branch]['args'],**state[branch]['kwargs'])[0]
    expected=[3,2,2,3] if branch=='conditional' else [0]*4
    assert network.adapter.injection_counts==expected
    return value


def pilot(args):
    core,out=initialize(args);pipe,network=load_frozen(args,core)
    before=weights_digest(network);spec=read(out/'protocol.json');records=[]
    pairs=read(out/'strict_pairs.json')['pairs']
    chosen=[next(p for p in pairs if p['qualification']['old_pattern']==x) for x in ['dots','stripes','checks']]
    with torch.inference_mode():
        for pair in chosen:
            old=condition(args,core,network,pair,'old')
            with FeatureTap(network.adapter) as tap: observed=condition(args,core,network,pair,'old')
            assert all(torch.equal(a,b) for a,b in zip(old,observed))
            with TrajectoryTap(pipe,network.adapter) as trajectory:
                value,latent=image(args,core,pipe,network,pair,feature_mix(old,observed,0))
            records.append(replay_image(out,pair,'old',value,latent))
            new=condition(args,core,network,pair,'new')
            value,latent=image(args,core,pipe,network,pair,feature_mix(old,new,1))
            records.append(replay_image(out,pair,'new',value,latent))
            assert trajectory.condition_calls==trajectory.scheduler_calls==50 and set(trajectory.states)==set(STEPS)
            for step,state in trajectory.states.items():
                repeat=forward_at(pipe,network,state,old)
                unc=forward_at(pipe,network,state,new,branch='unconditional')
                assert torch.equal(repeat,state['conditional']['epsilon']) and torch.equal(unc,state['unconditional']['epsilon'])
                assert torch.equal(state['guided'],unc+7*(repeat-unc))
                assert torch.equal(state['sample'],state['conditional']['args'][0])
            print('TEXTURE_DIAG_PILOT',pair['case_id'],flush=True)
    after=weights_digest(network);assert before==after
    write(out/'audit/pilot.json',dict(passed=True,image_replays=records,observed_forward_exact=True,
        common_state_repeats_exact=True,unconditional_reference_independent=True,cfg_formula_exact=True,
        injection_counts_conditional=[3,2,2,3],injection_counts_unconditional=[0]*4,
        weights_before=before,weights_after=after,generator_updates=0,environment=diagnostic_environment()))
    print('TEXTURE_DIAG_PILOT_PASS',flush=True)


def save_output(args,core,pair,group,condition_name,value,latent,scale,gain):
    out=Path(args.out);path=out/'outputs'/group/pair['case_id']/(condition_name+'.png')
    path.parent.mkdir(parents=True,exist_ok=True);value.save(path)
    write(path.with_suffix('.json'),dict(group=group,condition=condition_name,case_id=pair['case_id'],index=pair['index'],
        seed=pair['seed'],caption=pair['caption'],initial_latent_sha256=latent,output_sha256=sha(path),
        checkpoint_sha256=BASELINE_SHA,manifest_sha256=PAIRS_SHA,input_sketch_sha256=pair['hashes']['sketch'],
        input_references={k:pair['hashes'][k] for k in ['original_reference','swapped_reference']},
        sketch_scale=scale,feature_gain=gain,reused=False,generator_updates=0,GT_used_in_inference=False))


def fft_vectors(descriptor,image,coords):
    parts=extract_patches(image[0],coords,64)
    return descriptor(parts).mean(0)


def common_states(args,core,pipe,network,pair,features,states):
    from eval.eval_utils import prepare_evaluation_masks
    mask=prepare_evaluation_masks((384,512),sketch_path=str(Path(args.data)/pair['sketch']),mask_policy='sketch_only')['garment']
    mask=torch.from_numpy(mask.astype(np.float32))[None,None].to(args.device)
    geo=build_geometry(mask,details=False);masks=dict(body=geo.body,boundary=geo.inner_band,background=geo.background)
    old,new=features['old_raw'],features['new_raw'];delta=feature_mix(old,new,2.)
    candidates=[('new',new,.6),('shift',features['old_shift'],.6),('shuffle',features['old_shuffle'],.6),('delta2',delta,.6)]
    for level in range(4):
        replacement=list(old);replacement[level]=new[level];candidates.append(('level%d'%level,replacement,.6))
    for name,scale in [('gam03',.3),('gam0',0.)]:
        candidates.extend([(name+'_old',old,scale),(name+'_new',new,scale)])
    descriptor=PowerSpectrumDescriptor();refs={};ref_n={}
    coords=[(int(x),int(y)) for y in np.linspace(0,512-64,4).round() for x in np.linspace(0,384-64,4).round()]
    for name in ['old','new']:
        value=core.tensor(Path(args.out)/'inputs/controls'/pair['case_id']/name/'raw.png').to(args.device)
        patches=extract_patches((value[0]+1)*.5,coords,64)
        valid=gray(patches).std((-2,-1),unbiased=False)>=.02;ref_n[name]=int(valid.sum())
        refs[name]=descriptor(patches[valid]).mean(0) if valid.any() else None
    rows=[];short=pair['case_id'] in read(Path(args.out)/'protocol.json')['short_cases']
    for step,state in states.items():
        base=state['conditional']['epsilon'];unc=state['unconditional']['epsilon'];guided=state['guided']
        repeat=forward_at(pipe,network,state,old)
        assert torch.equal(repeat,base)
        unc_new=forward_at(pipe,network,state,new,branch='unconditional');assert torch.equal(unc,unc_new)
        rows.append(dict(step=step,t=state['timestep'],variant='repeat',conditional=region_change(repeat,base,masks)))
        for name,maps,scale in [('old',old,.6)]+candidates:
            eps=base if name=='old' else forward_at(pipe,network,state,maps,scale)
            guided_new=unc+7*(eps-unc)
            row=dict(step=step,t=state['timestep'],variant=name,
                conditional=region_change(eps,base,masks),guided=region_change(guided_new,guided,masks))
            if state['timestep']<=249 and name in ['old','new','delta2']:
                alpha=pipe.scheduler.alphas_cumprod[state['timestep']].to(device=args.device,dtype=torch.float32)
                x0=(state['sample'].float()-(1-alpha).sqrt()*guided_new.float())/alpha.sqrt()
                decoded=pipe.vae.decode((x0/pipe.vae.config.scaling_factor).to(pipe.vae.dtype)).sample
                f=fft_vectors(descriptor,(decoded.float()+1)*.5,pair['patch_coordinates'])
                distances={k:float((f-v).abs().mean()) if v is not None else None for k,v in refs.items()}
                row['decoded_fft']=dict(distances=distances,reference_patch_n=ref_n,
                    margin=distances['old']-distances['new'] if all(v is not None for v in distances.values()) else None,
                    metric='exploratory fixedpatch FFT;not old v3 DISTS')
                if short:
                    path=Path(args.out)/'x0'/pair['case_id']/('%02d_%s.png'%(step,name));path.parent.mkdir(parents=True,exist_ok=True)
                    Image.fromarray(((decoded[0].float().cpu().clamp(-1,1).permute(1,2,0).numpy()+1)*127.5).round().astype(np.uint8)).save(path)
            rows.append(row)
    return rows


def controlled_downstream(args,core,pipe,network,pair,states):
    """同一实际轨迹状态的受控纹样读出，避免把响应幅度当成可读性。"""
    from models.e2_texture_diagnostics import activation_vectors
    vectors={};rows=[];descriptor=PowerSpectrumDescriptor()
    for variant in list(PATTERNS)+[p+'_shift' for p in PATTERNS]:
        maps=condition(args,core,network,pair,'old',variant)
        rows.append(dict(case_id=pair['case_id'],identity_group=pair['identity_group'],condition='old',variant=variant))
        for step in [25,40,49]:
            state=states[step];unc=state['unconditional']['epsilon']
            eps=forward_at(pipe,network,state,maps);guided=unc+7*(eps-unc)
            for kind,array in activation_vectors(guided).items():
                vectors.setdefault('epsilon_step%d__%s'%(step,kind),[]).append(array)
            if state['timestep']<=249:
                alpha=pipe.scheduler.alphas_cumprod[state['timestep']].to(device=args.device,dtype=torch.float32)
                x0=(state['sample'].float()-(1-alpha).sqrt()*guided.float())/alpha.sqrt()
                decoded=pipe.vae.decode((x0/pipe.vae.config.scaling_factor).to(pipe.vae.dtype)).sample
                array=fft_vectors(descriptor,(decoded.float()+1)*.5,pair['patch_coordinates']).cpu().numpy()
                vectors.setdefault('x0_fft_step%d__moments'%step,[]).append(array)
    return rows,{k:np.stack(v) for k,v in vectors.items()}


def diagnose(args):
    core,out=initialize(args);assert read(out/'audit/pilot.json')['passed']
    assert read(out/'protocol.json')['environment']['commit']==environment()['commit']
    pipe,network=load_frozen(args,core);before=weights_digest(network);spec=read(out/'protocol.json')
    controls=read(out/'inputs/controls.json');pairs=read(out/'strict_pairs.json')['pairs'];completed=[]
    with torch.inference_mode():
        for pos,pair in enumerate(pairs):
            if pos%args.shards!=args.shard:continue
            folder=out/'cases'/pair['case_id'];folder.mkdir(parents=True,exist_ok=True)
            if (folder/'result.json').exists():
                previous=read(folder/'result.json');assert previous['passed'] and sha(folder/'features.npz')==previous['features_sha256']
                completed.append(pair['case_id']);continue
            started=time.monotonic();vectors={};feature_rows=[];features={}
            for condition_name in ['old','new']:
                for variant in VARIANTS:
                    meta=next(c for c in controls if (c['case_id'],c['condition'],c['variant'])==(pair['case_id'],condition_name,variant))
                    path=out/meta['file'];assert sha(path)==meta['sha256']
                    with FeatureTap(network.adapter) as tap:maps=condition(args,core,network,pair,condition_name,variant)
                    for stage,value in tap.values.items():
                        for kind,array in value.items():vectors.setdefault(stage+'__'+kind,[]).append(array)
                    for stage,array in [('input_color',np.asarray(meta['color'],np.float32)),('input_spectrum',np.asarray(meta['spectrum'],np.float32))]:
                        for kind in ['moments','spatial']:vectors.setdefault(stage+'__'+kind,[]).append(array)
                    feature_rows.append(meta)
                    if variant=='raw' or condition_name=='old' and variant in ['shift','shuffle']:
                        features[condition_name+'_'+variant]=maps
            np.savez_compressed(folder/'features.npz',**{k:np.stack(v) for k,v in vectors.items()})
            write(folder/'feature_rows.json',feature_rows)
            with TrajectoryTap(pipe,network.adapter) as trajectory:
                value,latent=image(args,core,pipe,network,pair,features['old_raw'])
            replay=replay_image(out,pair,'old',value,latent)
            rows=common_states(args,core,pipe,network,pair,features,trajectory.states)
            write(folder/'common_states.json',rows)
            downstream_rows,downstream=controlled_downstream(args,core,pipe,network,pair,trajectory.states)
            np.savez_compressed(folder/'downstream.npz',**downstream)
            write(folder/'downstream_rows.json',downstream_rows)
            if pair['case_id'] in spec['short_cases']:
                for group in ['delta2','gam03','gam0']:
                    for condition_name in ['old','new']:
                        if group=='delta2' and condition_name=='old':continue
                        gain=2. if group=='delta2' else (0. if condition_name=='old' else 1.)
                        scale={'delta2':.6,'gam03':.3,'gam0':0.}[group]
                        maps=feature_mix(features['old_raw'],features['new_raw'],gain)
                        value,latent=image(args,core,pipe,network,pair,maps,scale)
                        save_output(args,core,pair,group,condition_name,value,latent,scale,gain)
            write(folder/'result.json',dict(passed=True,case_id=pair['case_id'],feature_rows=len(feature_rows),
                features_sha256=sha(folder/'features.npz'),common_state_rows=len(rows),baseline_replay=replay,
                downstream_sha256=sha(folder/'downstream.npz'),
                generator_updates=0,seconds=time.monotonic()-started))
            completed.append(pair['case_id']);network.adapter.clear()
            del trajectory,features,maps,tap;torch.cuda.empty_cache()
            print('TEXTURE_DIAG_CASE_PASS',pair['case_id'],len(completed),flush=True)
    after=weights_digest(network);assert before==after
    write(out/'audit'/('shard_%d.json'%args.shard),dict(passed=True,cases=completed,weights_before=before,weights_after=after,
        checkpoint_sha256=sha(spec['checkpoint']),generator_updates=0,environment=diagnostic_environment()))
    print('TEXTURE_DIAG_SHARD_PASS',args.shard,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','pilot','diagnose','readout','evaluate','bundle'])
    p.add_argument('--project',default='/share/home/u2515283058/Mymodel')
    p.add_argument('--parent',default='/share/home/u2515283058/Mymodel/output_eval/e2_b_screen_20261108')
    p.add_argument('--followup',default='/share/home/u2515283058/Mymodel/output_eval/e2_fft_20261106/followup_20261108')
    p.add_argument('--frozen-inputs',default='/share/home/u2515283058/Mymodel/output_eval/e2_a_minimal_20261109/inputs/frozen')
    p.add_argument('--out',default='/share/home/u2515283058/Mymodel/output_eval/e2_texture_bottleneck_20261111')
    p.add_argument('--device',default='cuda:0');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=2)
    args=p.parse_args()
    if args.action in ['readout','evaluate','bundle']:
        from tools import e2_texture_bottleneck_eval
        getattr(e2_texture_bottleneck_eval,args.action)(args)
    else:globals()[args.action](args)


if __name__=='__main__':main()
