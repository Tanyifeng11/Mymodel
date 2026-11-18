"""仅当5388局部半径不足时，诊断全层B零偏移；不训练或采用。"""
import argparse
import copy
import inspect
from pathlib import Path
import subprocess
import time
import zipfile

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.checkpoint import checkpoint

from models.e2_terminal_full_b_offsets import TerminalFullBOffsets,terminal_inputs,PARAMETERS
from tools import e2_reference_hybrid_radius_diagnostic as parent
from tools import e2_state_b_pair_evaluation as evaluation
from tools.e2_state_b_amplitude_response import read,write,sha,pixels
from tools.e2_state_b_backward_repro import tensor_sha
from tools.e2_state_b_scaled_retry import scaled_scalar
from tools.e2_oracle_texture import weights_digest


SOURCE=Path(__file__).resolve().parents[1]
FACTORS=(0.,.25,.5,1.)
EXTRA=('models/e2_terminal_full_b_offsets.py','tools/e2_terminal_full_b_radius_diagnostic.py','scripts/e2_terminal_full_b_radius_diagnostic.sbatch')


def verify(args):
    spec,_=parent.verify(args);protocol=read(args.parent/'protocol.json');result=read(args.parent/'result.json')
    assert result['status']=='complete' and result['frozen_unchanged'] and result['source_final_verification']
    assert result['fresh_VJP']==16 and result['repeat_groups']==8 and result['native_anchor_RGB_PNG_controls']==48
    assert all(sha(args.parent/n)==h for n,h in read(args.parent/'manifest.json').items())
    assert all(sha(SOURCE/n)==h for n,h in protocol['source_sha256'].items())
    assert all(sha(Path(n))==h for n,h in protocol['input_sha256'].items())
    return spec,protocol,result


def run(args):
    from tools import e2_state_b_pilot as pilot
    started=time.monotonic();torch.set_num_threads(2);cv2.setNumThreads(1)
    spec,previous,prior=verify(args);assert not args.out.exists(),'不覆盖全层B半径诊断';args.out.mkdir()
    finite_means={str(f):float(np.mean([r['continuous']['A'] for r in prior['rows'] if r['factor_of_radius']==f])) for f in FACTORS}
    guard=prior['first_order']['A_linear_optimistic_ceiling']<.75 and max(finite_means.values())<.75
    files=[args.parent/n for n in ['protocol.json','result.json','manifest.json','diagnostic.pt']]
    protocol=dict(kind='conditional full-network B weight offsets,official A/g radius diagnosis',parameters=PARAMETERS,
        parameter_family='all24 B weight/bias tensors at zero offsets;f68 originals read-only',
        input='zt and raw CFG epsilon from fixed final S0 cache;FP32 x0 reconstructed in original SamplerB order',
        anchor='recomputed frozen B zero output;cached GPU B parity separately recorded,not forced',
        forward='native_f68+raw32(z_f68+kappa*(B_with_offsets-B_recomputed_zero))-raw32(z_f68)',
        conditional_gate=guard,parent_linear_A_ceiling=prior['first_order']['A_linear_optimistic_ceiling'],parent_finite_mean_A=finite_means,
        conditional_rule='both first-order optimistic mean-A ceiling and all predefined finite mean-A below.75',
        original_loss_step_gates_unchanged=True,original_S0_code_modified=False,functional_budget_matched_to_5388=False,
        metric='same official precise-L/PNG-positive/inside mask as parent;FP64 box matrices',
        scope='four seen pastel alpha1 only;not full48 candidate qualification or global capacity proof',
        radius=spec['training']['displacement_cap'],factors=list(FACTORS),planned_VJP=16,planned_repeat_groups=8,
        optimizer_steps=0,new_S0=0,UNet_sampling_calls=0,holdout_read=0,adopted_checkpoints=[],
        source_sha256=dict(previous['source_sha256'],**{n:sha(SOURCE/n) for n in EXTRA}),
        input_sha256={str(p.resolve()):sha(p) for p in files},
        execution_commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip())
    write(args.out/'protocol.json',protocol)
    if not guard:
        write(args.out/'result.json',dict(status='skipped_by_predeclared_parent_gate',guard=guard,optimizer_steps=0,new_S0=0,holdout_read=0))
        write(args.out/'manifest.json',{n:sha(args.out/n) for n in ['protocol.json','result.json']})
        with zipfile.ZipFile(args.out/'review.zip','w',zipfile.ZIP_DEFLATED) as z:
            for n in ['protocol.json','result.json','manifest.json']:z.write(args.out/n,n)
        print('FULL_B_RADIUS_SKIPPED',flush=True);return
    branch=parent.replay.proposal.fresh.direction;source=branch.delta.post.q8.r16.source
    runtime=copy.copy(args);runtime.action='train';runtime.out=args.source_run
    _,pipe,net=pilot.load_core(runtime);net.eval().requires_grad_(False);pipe.vae.float().eval().requires_grad_(False)
    original=source.load_head(args,args.device);head=TerminalFullBOffsets(original.base).to(args.device).eval()
    parameters=head.increment_parameters();hashes=[weights_digest(v) for v in [head.base,net,pipe.vae]]
    baseline=read(args.source_run/'evaluation/f68/replay/result.json');base_rows={(r['case'],r['sample_condition']):r for r in baseline['rows']}
    overlay=read(args.source_run/'one_step_overlay.json');old=torch.load(args.parent/'diagnostic.pt',map_location='cpu',weights_only=False)
    anchors={v['case']+'/'+v['sample_condition']:v for v in prior['native_anchor_controls']}
    selected=[v for v in spec['inputs'] if v['group']=='pastel' and v['arm']=='a100'];assert len(selected)==4
    entries,_,_=evaluation.replay_inputs(runtime,spec);records=[];saved=dict(points={});zero_checks=[];counter=[0]
    def count(module,inputs):counter[0]+=1
    hook=pipe.vae.decoder.register_forward_pre_hook(count)
    for index,pair in enumerate(spec['train']):
        cache,_=source.load_cache(args,index,overlay);meta=read(args.basis/('cache/shard%d.json'%index))
        assert hashes==[cache['b_hash'],cache['net_hash'],meta['vae_hash_fp32']]
        assert sha(Path(inspect.getsourcefile(type(pipe.scheduler))))==meta['scheduler']['source_sha256']
        assert source.preflight.scheduler_configuration(pipe.scheduler.config,meta['scheduler']['config'])['normalized_equal']
        pipe.scheduler.set_timesteps(50,device=args.device)
        with torch.no_grad():
            for cond,record in cache['records'].items():
                zt,x0,condition=terminal_inputs(record,args.device)
                frozen=head.base(zt,x0,**condition);zero=head(zt,x0,**condition)
                assert tensor_sha(frozen)==tensor_sha(zero)
                cached=record['base_delta'].to(args.device)
                zero_checks.append(dict(case=pair['case_id'],sample_condition=cond,zero_functional_byte_equal=tensor_sha(frozen)==tensor_sha(zero),
                    cached_GPU_output_byte_equal=tensor_sha(frozen)==tensor_sha(cached),cached_relative_L2=float((frozen-cached).norm()/cached.norm()),
                    cached_max_abs=float((frozen-cached).abs().max())))
        item=next(v for v in selected if v['case']==pair['case_id']);cond=item['sample_condition'];key=pair['case_id']+'/'+cond;record=cache['records'][cond]
        zt,x0,condition=terminal_inputs(record,args.device);zbase=record['final'].to(args.device)
        native=old['native_anchors'][key].to(args.device).float();assert tensor_sha(native)==anchors[key]['native_RGB_sha256']
        kappa=branch.delta.post.coefficient(torch.tensor(record['alpha'],device=args.device,dtype=torch.float32),pipe.scheduler.final_alpha_cumprod.to(args.device).float())
        with torch.no_grad():zero=head.base(zt,x0,**condition);reference=branch.delta.decode_raw32(pipe.vae,zbase.float())
        entry=next(v for v in entries if (v['case'],v['sample_condition'])==(pair['case_id'],cond))
        positive=pixels(entry['folder']/(cond+'_positive.png'));cloth=pixels(entry['folder']/'mask.png')[...,0]>0;metric=parent.OfficialFields(positive,cloth,args.device)
        def forward(use_checkpoint=True):
            difference=head(zt,x0,**condition)-zero;z=zbase.float()+kappa*difference
            decode=lambda value:branch.delta.decode_raw32(pipe.vae,value)
            current=checkpoint(decode,z,use_reentrant=False) if use_checkpoint else decode(z)
            return z,branch.delta.anchored_rgb(native,reference,current),difference
        gradients=[];numbers=[]
        for repeat in range(2):
            z,rgb,_=forward();assert torch.equal(rgb.detach(),native)
            values=metric(rgb);numbers.append({n:float(v.detach()) for n,v in values.items()})
            gradients.append(dict(A=scaled_scalar(values['A'],parameters,retain_graph=True).cpu(),g=scaled_scalar(values['g'],parameters).cpu()))
        assert numbers[0]==numbers[1]
        reference_fields=next(v for v in prior['records'] if v['case']==pair['case_id'])['zero']
        assert max(abs(numbers[0][n]-reference_fields[n]) for n in numbers[0])<1e-10
        repeated={n:branch.delta.post.q8.r16.repeat_report(gradients[1][n],gradients[0][n]) for n in ['A','g']}
        assert all(v['passed'] for v in repeated.values())
        records.append(dict(item,seed=pair['seed'],noise_digest=record['noise_digest'],zero=numbers[0],repeat=repeated))
        saved['points'][key]=dict(gradients=gradients,zt=zt.cpu(),x0=x0.cpu(),condition={n:v.cpu() for n,v in condition.items()},
            zero_B=zero.cpu(),base_latent=zbase.cpu(),native_RGB=native.cpu(),kappa=float(kappa),seed=pair['seed'],noise_digest=record['noise_digest'],probes={})
        print('FULL_B_RADIUS_FRESH',index,key,flush=True)
    mean={n:torch.stack([p['gradients'][0][n] for p in saved['points'].values()]).double().mean(0) for n in ['A','g']}
    direction=(mean['A']*(protocol['radius']/float(mean['A'].norm()))).float();direction*=min(1.,protocol['radius']/float(direction.norm()));saved['direction']=direction
    rows=[]
    for pair in spec['train']:
        item=next(v for v in selected if v['case']==pair['case_id']);cond=item['sample_condition'];key=pair['case_id']+'/'+cond;point=saved['points'][key]
        zt,x0,zero,zbase,native=[point[n].to(args.device) for n in ['zt','x0','zero_B','base_latent','native_RGB']]
        condition={n:v.to(args.device) for n,v in point['condition'].items()};kappa=torch.tensor(point['kappa'],device=args.device)
        with torch.no_grad():reference=branch.delta.decode_raw32(pipe.vae,zbase.float())
        entry=next(v for v in entries if (v['case'],v['sample_condition'])==(pair['case_id'],cond))
        positive=pixels(entry['folder']/(cond+'_positive.png'));cloth=pixels(entry['folder']/'mask.png')[...,0]>0;metric=parent.OfficialFields(positive,cloth,args.device)
        old_fields=next(v for v in records if v['case']==pair['case_id'])['zero']
        for factor in FACTORS:
            theta=direction*factor;source.set_parameters(parameters,theta.to(args.device))
            with torch.no_grad():z,rgb,difference=forward(False);values={n:float(v) for n,v in metric(rgb).items()};image=branch.delta.quantize(rgb)
            if factor==0:assert tensor_sha(torch.from_numpy(image))==anchors[key]['pixel_array_sha256']
            file=Path('images')/('%s_%.2f.png'%(pair['case_id'],factor));(args.out/file).parent.mkdir(exist_ok=True);Image.fromarray(image).save(args.out/file)
            rows.append(dict(item,seed=pair['seed'],noise_digest=point['noise_digest'],factor_of_radius=factor,parameter_norm=float(theta.norm()),
                image=file.as_posix(),png_sha256=sha(args.out/file),continuous=values,
                prediction={n:old_fields[n]+float(point['gradients'][0][n].double()@theta.double()) for n in ['A','g']},
                **evaluation.score_entry(entry,image,'replay')))
            point['probes'][str(factor)]=dict(rgb=rgb.cpu(),latent=z.cpu(),difference=difference.cpu())
        print('FULL_B_RADIUS_REPLAY',pair['case_id'],flush=True)
    source.set_parameters(parameters,torch.zeros_like(direction).to(args.device));hook.remove()
    assert not torch.count_nonzero(source.flatten(parameters)) and not any(p.grad is not None for p in parameters)
    assert [weights_digest(v) for v in [head.base,net,pipe.vae]]==hashes
    verify(args);assert all(sha(Path(n))==h for n,h in protocol['input_sha256'].items());assert all(sha(SOURCE/n)==h for n,h in protocol['source_sha256'].items())
    means=lambda n:float(np.mean([v['zero'][n] for v in records]))
    result=dict(status='complete',records=records,rows=rows,zero_B_controls=zero_checks,zero_PNG_controls=4,
        fresh_VJP=16,repeat_groups=8,cache_observations=16,decoder_forward_entries=counter[0],
        first_order={n+'_linear_optimistic_ceiling':means(n)+protocol['radius']*float(mean[n].norm()) for n in ['A','g']},
        frozen_unchanged=True,source_final_verification=True,optimizer_steps=0,new_S0=0,UNet_sampling_calls=0,holdout_read=0,
        adopted_checkpoints=[],eligibility='diagnostic only;no full48 quality or global nonlinear certificate',seconds=time.monotonic()-started)
    torch.save(saved,args.out/'diagnostic.pt');write(args.out/'result.json',result)
    names=['protocol.json','result.json','diagnostic.pt']+[r['image'] for r in rows];write(args.out/'manifest.json',{n:sha(args.out/n) for n in names})
    with zipfile.ZipFile(args.out/'review.zip','w',zipfile.ZIP_DEFLATED) as z:
        for n in names+['manifest.json']:z.write(args.out/n,n)
    print('FULL_B_RADIUS_REVIEW',sha(args.out/'review.zip'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prior',type=Path,required=True)
    parser.add_argument('--project',type=Path,default=Path('/share/home/u2515283058/Mymodel'));parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();args.prior=args.prior.resolve();args.project=args.project.resolve();auto=args.prior/'autonomous_repair_20261116'
    for n,value in dict(fixed='scaled_projection_retry',preflight='terminal_head_preflight',basis='reference_terminal_preflight_v2',
        source_run='reference_terminal_one_step',global_source='reference_terminal_global_amplitude',r16='reference_terminal_r16_diagnostic',
        q8='reference_terminal_q8_direction_diagnostic',post='reference_terminal_post_latent_probe',fp32='reference_terminal_fp32_delta_probe',
        direction='reference_terminal_fp32_direction',gradients='reference_terminal_hybrid_gradients',proposals='reference_terminal_hybrid_proposals',
        replay='reference_terminal_hybrid_replay',parent='reference_terminal_hybrid_radius_diagnostic',out='terminal_full_B_radius_diagnostic').items():setattr(args,n,auto/value)
    args.data=args.project.parent/'datasets/BF';args.checkpoint=args.project/'output_eval/e2_state_b_pastel_followup_20261115/trust_retry/training/step1.pt'
    run(args)
