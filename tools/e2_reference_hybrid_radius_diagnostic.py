"""固定f68零5388的官方连续幅度/增益导数及原半径射线；不训练。"""
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

from models.e2_reference_terminal_hybrid import ReferenceTerminalHybrid,PARAMETERS
from tools import e2_reference_terminal_hybrid_replay as replay
from tools import e2_state_b_pair_evaluation as evaluation
from tools.e2_b_last_step_metrics import precise_l,highpass
from tools.e2_state_b_amplitude_response import read,write,sha,pixels
from tools.e2_state_b_backward_repro import tensor_sha
from tools.e2_state_b_scaled_retry import scaled_scalar
from tools.e2_oracle_texture import weights_digest


SOURCE=Path(__file__).resolve().parents[1]
FACTORS=(0.,.25,.5,1.)
EXTRA=('tools/e2_reference_hybrid_radius_diagnostic.py','scripts/e2_reference_hybrid_radius_diagnostic.sbatch')


def box_matrix(size,device):
    """实测复现CV默认17格边界；矩阵乘法避免引入额外padding反传路径。"""
    indices=np.pad(np.arange(size),(8,8),mode='reflect');matrix=np.zeros((size,size),np.float64)
    for i in range(size):np.add.at(matrix[i],indices[i:i+17],1/17)
    return torch.from_numpy(matrix).to(device)


def precise_l_tensor(rgb):
    value=rgb.double()[0].permute(1,2,0)
    linear=torch.where(value>.04045,((value+.055)/1.055).pow(2.4),value/12.92)
    y=(linear*torch.tensor([.2126729,.7151522,.0721750],device=rgb.device,dtype=torch.float64)).sum(-1)
    f=torch.where(y>.008856,y.clamp_min(.008856).pow(1/3),7.787037*y+16/116)
    return (116*f-16)/100


class OfficialFields:
    def __init__(self,positive,mask,device):
        self.left=box_matrix(mask.shape[0],device);self.right=box_matrix(mask.shape[1],device)
        self.inside=torch.from_numpy(cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,5)>=16).to(device)
        self.p=torch.from_numpy(highpass(precise_l(positive))).to(device)[self.inside]
        self.energy=self.p.square().mean();assert self.energy>1e-8

    def __call__(self,rgb):
        field=precise_l_tensor(rgb);h=(field-self.left@field@self.right.T)[self.inside]
        amplitude=(h.square().mean()/self.energy).sqrt()
        gain=(h*self.p).mean()/self.energy
        return dict(A=amplitude,g=gain,c=gain/amplitude)


def numpy_fields(rgb,positive,mask):
    inside=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,5)>=16
    h=highpass(precise_l(rgb[0].permute(1,2,0).double().cpu().numpy()*255))[inside]
    p=highpass(precise_l(positive))[inside];a=float(np.sqrt(np.mean(h*h)/np.mean(p*p)))
    g=float(np.mean(h*p)/np.mean(p*p));return dict(A=a,g=g,c=g/a)


def verify(args):
    spec,_,_=replay.inputs(args)
    protocol=read(args.replay/'protocol.json');result=read(args.replay/'result.json')
    assert result['status']=='complete' and result['frozen_unchanged'] and result['source_final_verification']
    assert all(sha(args.replay/n)==h for n,h in read(args.replay/'manifest.json').items())
    assert all(sha(Path(n))==h for n,h in protocol['input_sha256'].items())
    assert all(sha(SOURCE/n)==h for n,h in protocol['source_sha256'].items())
    return spec,protocol


def run(args):
    from tools import e2_state_b_pilot as pilot
    started=time.monotonic();cv2.setNumThreads(1);torch.set_num_threads(2)
    spec,previous=verify(args);assert not args.out.exists(),'不覆盖半径诊断'
    args.out.mkdir();branch=replay.proposal.fresh.direction;source=branch.delta.post.q8.r16.source
    selected=[v for v in spec['inputs'] if v['group']=='pastel' and v['arm']=='a100'];assert len(selected)==4
    files=[args.replay/n for n in ['protocol.json','result.json','manifest.json','diagnostic.pt']]
    baseline_file=args.source_run/'evaluation/f68/replay/result.json'
    files += [baseline_file]+[args.source_run/v['image'] for v in read(baseline_file)['rows']]
    for item in selected:
        folder=args.source_run/'cases'/item['case'];files += [folder/'mask.png',folder/(item['sample_condition']+'_positive.png')]
    protocol=dict(kind='fresh official A/g derivative and original-radius best-first-order mean-A ray',
        scope='four seen pastel alpha1 rays plus48 frozen native anchors;no full48 candidate or holdout qualification',parameters=PARAMETERS,
        original_S0_code_modified=False,original_loss_step_and_gates_unchanged=True,
        metric='precise-L FP64;CV17-box equivalent matrices;official PNG positive and inside mask',
        point='f68 plus zero5388;no old A/g gradient reused',planned_VJP=16,planned_repeat_groups=8,planned_native_anchor_controls=48,
        factors=list(FACTORS),radius=spec['training']['displacement_cap'],
        ray='positive mean official-continuous A gradient,normalized to original parameter cap;diagnostic,not optimizer step',
        bounds='unconstrained Cauchy ceiling of first-order model only;not nonlinear/global infeasibility proof',
        optimizer_steps=0,new_S0=0,UNet_sampling_calls=0,holdout_read=0,adopted_checkpoints=[],
        source_sha256=dict(previous['source_sha256'],**{n:sha(SOURCE/n) for n in EXTRA}),
        input_sha256={str(p.resolve()):sha(p) for p in files},
        execution_commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip())
    write(args.out/'protocol.json',protocol)
    runtime=copy.copy(args);runtime.action='train';runtime.out=args.source_run
    _,pipe,net=pilot.load_core(runtime);net.eval().requires_grad_(False)
    original=source.load_head(args,args.device);head=ReferenceTerminalHybrid(original).to(args.device).eval()
    parameters=head.increment_parameters();assert sum(v.numel() for v in parameters)==PARAMETERS
    pipe.vae.float().eval().requires_grad_(False);hashes=[weights_digest(v) for v in [head.head.base,net,pipe.vae]]
    counter=[0]
    def count_decoder(module,inputs):counter[0]+=1
    hook=pipe.vae.decoder.register_forward_pre_hook(count_decoder)
    overlay=read(args.source_run/'one_step_overlay.json');baseline=read(args.source_run/'evaluation/f68/replay/result.json')
    base_rows={(v['case'],v['sample_condition']):v for v in baseline['rows']};entries,_,_=evaluation.replay_inputs(runtime,spec)
    saved=dict(points={},native_anchors={});records=[];anchor_controls=[]
    for index,pair in enumerate(spec['train']):
        item=next(v for v in selected if v['case']==pair['case_id']);cond=item['sample_condition'];key=item['case']+'/'+cond
        cache,_=source.load_cache(args,index,overlay);record=cache['records'][cond]
        meta=read(args.basis/('cache/shard%d.json'%index));assert hashes==[cache['b_hash'],cache['net_hash'],meta['vae_hash_fp32']]
        assert sha(Path(inspect.getsourcefile(type(pipe.scheduler))))==meta['scheduler']['source_sha256']
        audit=source.preflight.scheduler_configuration(pipe.scheduler.config,meta['scheduler']['config']);assert audit['normalized_equal']
        pipe.scheduler.set_timesteps(50,device=args.device)
        zbase=record['final'].to(args.device);feature,mask,base_b=[record[n].to(args.device) for n in ['feature','mask','base_delta']]
        conditioning=record['target_record'][3]['reference'].to(args.device);phi=original.statistics(conditioning)['phi']
        kappa=branch.delta.post.coefficient(torch.tensor(record['alpha'],device=args.device,dtype=torch.float32),pipe.scheduler.final_alpha_cumprod.to(args.device).float())
        pipe.vae.half()
        try:native=branch.delta.post.q8.r16.native_rgb_half(pipe.vae,zbase).float()
        finally:pipe.vae.float()
        with torch.no_grad():reference=branch.delta.decode_raw32(pipe.vae,zbase.float())
        old=read(args.gradients/('gradients/shard%d.json'%index));control=next(v for v in old['values'] if v['sample_condition']==cond)
        assert tensor_sha(native)==control['native_RGB_sha256']
        entry=next(v for v in entries if (v['case'],v['sample_condition'])==(item['case'],cond))
        positive=pixels(entry['folder']/(cond+'_positive.png'));cloth=pixels(entry['folder']/'mask.png')[...,0]>0
        metric=OfficialFields(positive,cloth,args.device)
        def forward(use_checkpoint=True):
            difference=head.difference(feature,mask,phi,conditioning,base_b);z=zbase.float()+kappa*difference
            decode=lambda value:branch.delta.decode_raw32(pipe.vae,value)
            current=checkpoint(decode,z,use_reentrant=False) if use_checkpoint else decode(z)
            return z,branch.delta.anchored_rgb(native,reference,current),difference
        grads=[];numbers=[]
        for repeat in range(2):
            z,rgb,_=forward();assert torch.equal(rgb.detach(),native)
            values=metric(rgb);numbers.append({n:float(v.detach()) for n,v in values.items()})
            grads.append(dict(A=scaled_scalar(values['A'],parameters,retain_graph=True).cpu(),g=scaled_scalar(values['g'],parameters).cpu()))
        assert numbers[0]==numbers[1]
        cpu=numpy_fields(native.cpu(),positive,cloth);assert max(abs(cpu[n]-numbers[0][n]) for n in cpu)<1e-10
        repeats={n:branch.delta.post.q8.r16.repeat_report(grads[1][n],grads[0][n]) for n in ['A','g']}
        assert all(v['passed'] for v in repeats.values())
        records.append(dict(item,seed=pair['seed'],noise_digest=record['noise_digest'],zero=numbers[0],CPU_zero=cpu,repeat=repeats))
        saved['points'][key]=dict(gradients=grads,native_RGB=native.cpu(),base_latent=zbase.cpu(),feature=feature.cpu(),mask=mask.cpu(),
            base_B=base_b.cpu(),reference=conditioning.cpu(),phi=phi.cpu(),kappa=float(kappa),seed=pair['seed'],noise_digest=record['noise_digest'],probes={})
        # 补齐全部固定f68原生RGB，后续直接与已存96候选浮点图比较，避免再解码。
        by_condition={v['sample_condition']:v for v in old['values']}
        pipe.vae.half()
        try:
            with torch.no_grad():
                for anchor_item in [v for v in spec['inputs'] if v['case']==pair['case_id']]:
                    anchor_cond=anchor_item['sample_condition'];anchor_record=cache['records'][anchor_cond]
                    anchor_rgb=native.half() if anchor_cond==cond else branch.delta.post.q8.r16.native_rgb_half(pipe.vae,anchor_record['final'].to(args.device))
                    assert anchor_rgb.dtype==torch.float16
                    assert tensor_sha(anchor_rgb.float())==by_condition[anchor_cond]['native_RGB_sha256']
                    image=branch.delta.quantize(anchor_rgb.float());original_image=args.source_run/base_rows[pair['case_id'],anchor_cond]['image']
                    assert sha(original_image)==base_rows[pair['case_id'],anchor_cond]['png_sha256']
                    assert np.array_equal(image,pixels(original_image))
                    saved['native_anchors'][pair['case_id']+'/'+anchor_cond]=anchor_rgb.cpu()
                    anchor_controls.append(dict(anchor_item,seed=pair['seed'],noise_digest=anchor_record['noise_digest'],
                        native_RGB_sha256=tensor_sha(anchor_rgb.float()),pixel_array_sha256=tensor_sha(torch.from_numpy(image))))
        finally:pipe.vae.float()
        print('HYBRID_RADIUS_FRESH',index,key,flush=True)
    mean_A=torch.stack([v['gradients'][0]['A'] for v in saved['points'].values()]).double().mean(0)
    mean_g=torch.stack([v['gradients'][0]['g'] for v in saved['points'].values()]).double().mean(0)
    direction=(mean_A*(protocol['radius']/float(mean_A.norm()))).float()
    direction=direction*min(1.,protocol['radius']/float(direction.norm()))
    saved['direction']=direction;rows=[]
    for index,pair in enumerate(spec['train']):
        item=next(v for v in selected if v['case']==pair['case_id']);cond=item['sample_condition'];key=item['case']+'/'+cond
        point=saved['points'][key];feature,mask,base_b,conditioning,phi=[point[n].to(args.device) for n in ['feature','mask','base_B','reference','phi']]
        zbase=point['base_latent'].to(args.device);native=point['native_RGB'].to(args.device);kappa=torch.tensor(point['kappa'],device=args.device)
        with torch.no_grad():reference=branch.delta.decode_raw32(pipe.vae,zbase.float())
        entry=next(v for v in entries if (v['case'],v['sample_condition'])==(item['case'],cond))
        positive=pixels(entry['folder']/(cond+'_positive.png'));cloth=pixels(entry['folder']/'mask.png')[...,0]>0;metric=OfficialFields(positive,cloth,args.device)
        for factor in FACTORS:
            theta=direction*factor;source.set_parameters(parameters,theta.to(args.device))
            with torch.no_grad():
                z,rgb,difference=forward(False);values={n:float(v) for n,v in metric(rgb).items()};image=branch.delta.quantize(rgb)
            if factor==0:
                assert torch.equal(rgb,native) and tensor_sha(torch.from_numpy(image))==base_rows[item['case'],cond]['pixel_array_sha256']
            file=Path('images')/('%s_%.2f.png'%(item['case'],factor));(args.out/file).parent.mkdir(exist_ok=True);Image.fromarray(image).save(args.out/file)
            old=next(v for v in records if v['case']==item['case'])['zero']
            row=dict(item,seed=pair['seed'],noise_digest=point['noise_digest'],factor_of_radius=factor,image=file.as_posix(),png_sha256=sha(args.out/file),continuous=values,
                prediction={n:old[n]+float(point['gradients'][0][n].double()@theta.double()) for n in ['A','g']},
                parameter_norm=float(theta.norm()),**evaluation.score_entry(entry,image,'replay'))
            rows.append(row);point['probes'][str(factor)]=dict(rgb=rgb.cpu(),latent=z.cpu(),difference=difference.cpu())
        print('HYBRID_RADIUS_REPLAY',index,key,flush=True)
    source.set_parameters(parameters,torch.zeros_like(direction).to(args.device));hook.remove()
    assert not torch.count_nonzero(source.flatten(parameters)) and not any(p.grad is not None for p in parameters)
    assert [weights_digest(v) for v in [head.head.base,net,pipe.vae]]==hashes
    verify(args);assert all(sha(Path(n))==h for n,h in protocol['input_sha256'].items())
    assert all(sha(SOURCE/n)==h for n,h in protocol['source_sha256'].items())
    means=lambda n:float(np.mean([v['zero'][n] for v in records]))
    result=dict(status='complete',records=records,rows=rows,frozen_unchanged=True,source_final_verification=True,
        fresh_VJP=16,repeat_groups=8,cache_observations=16,zero_PNG_controls=4,native_anchor_controls=anchor_controls,
        native_anchor_RGB_PNG_controls=len(anchor_controls),decoder_forward_entries=counter[0],
        first_order=dict(mean_A=means('A'),mean_g=means('g'),gradient_mean_A_norm=float(mean_A.norm()),gradient_mean_g_norm=float(mean_g.norm()),
            A_linear_optimistic_ceiling=means('A')+protocol['radius']*float(mean_A.norm()),
            g_linear_optimistic_ceiling=means('g')+protocol['radius']*float(mean_g.norm()),
            required_mean_A_if_all4_quality=.75,necessary_mean_g_if_all4_quality=.375),
        optimizer_steps=0,new_S0=0,UNet_sampling_calls=0,holdout_read=0,adopted_checkpoints=[],
        eligibility='diagnostic subset,not candidate;linear ceiling is not global nonlinear certificate',seconds=time.monotonic()-started)
    torch.save(saved,args.out/'diagnostic.pt');write(args.out/'result.json',result)
    names=['protocol.json','result.json','diagnostic.pt']+[v['image'] for v in rows]
    write(args.out/'manifest.json',{n:sha(args.out/n) for n in names})
    with zipfile.ZipFile(args.out/'review.zip','w',zipfile.ZIP_DEFLATED) as z:
        for n in names+['manifest.json']:z.write(args.out/n,n)
    print('HYBRID_RADIUS_REVIEW',sha(args.out/'review.zip'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prior',type=Path,required=True)
    parser.add_argument('--project',type=Path,default=Path('/share/home/u2515283058/Mymodel'));parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();args.prior=args.prior.resolve();args.project=args.project.resolve();auto=args.prior/'autonomous_repair_20261116'
    for n,value in dict(fixed='scaled_projection_retry',preflight='terminal_head_preflight',basis='reference_terminal_preflight_v2',
        source_run='reference_terminal_one_step',global_source='reference_terminal_global_amplitude',r16='reference_terminal_r16_diagnostic',
        q8='reference_terminal_q8_direction_diagnostic',post='reference_terminal_post_latent_probe',fp32='reference_terminal_fp32_delta_probe',
        direction='reference_terminal_fp32_direction',gradients='reference_terminal_hybrid_gradients',proposals='reference_terminal_hybrid_proposals',
        replay='reference_terminal_hybrid_replay',out='reference_terminal_hybrid_radius_diagnostic').items():setattr(args,n,auto/value)
    args.data=args.project.parent/'datasets/BF';args.checkpoint=args.project/'output_eval/e2_state_b_pastel_followup_20261115/trust_retry/training/step1.pt'
    run(args)
