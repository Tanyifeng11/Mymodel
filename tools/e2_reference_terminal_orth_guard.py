"""保留C1原始更新，仅新增四条目标正交能量约束；fresh8 VJP后评测单个C2。"""
import copy
from pathlib import Path
import subprocess
import time
import zipfile

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.checkpoint import checkpoint

from tools import e2_reference_terminal_fp32_propose as proposal
from tools import e2_state_b_pair_evaluation as evaluation
from tools import e2_state_b_pair_summary as summary
from tools.e2_b_pair_finetune_losses import projection
from tools.e2_b_scaled_replay_review import replay_eligibility
from tools.e2_reference_terminal_fp32_replay_summary import repaired_models
from tools.e2_state_b_amplitude_response import read,write,sha
from tools.e2_state_b_backward_repro import tensor_sha
from tools.e2_oracle_texture import weights_digest


SOURCE=Path(__file__).resolve().parents[1]
EXTRA=('tools/e2_reference_terminal_orth_guard.py','scripts/e2_reference_terminal_orth_guard.sbatch',
       'tools/e2_reference_terminal_fp32_replay_summary.py')


def controls(args):
    old=copy.copy(args);old.out=args.proposals
    spec,_,shards=proposal.inputs(old);previous=proposal.verify(old)
    assert all(sha(args.proposals/n)==h for n,h in read(args.proposals/'manifest.json').items())
    for folder in (args.replay,args.stages):
        assert all(sha(folder/n)==h for n,h in read(folder/'manifest.json').items())
    assert read(args.stages/'result.json')['passed']
    return spec,previous,shards


def run(args):
    from tools import e2_state_b_pilot as pilot
    from tools.e2_fft_experiment import environment
    fresh=proposal.fresh;branch=fresh.direction;source=branch.delta.post.q8.r16.source
    spec,previous,shards=controls(args);started=time.monotonic()
    raw=torch.load(args.proposals/'raw.pt',map_location='cpu',weights_only=False)
    old_c1=torch.load(args.proposals/'C1.pt',map_location='cpu',weights_only=False)
    old_vector=torch.cat([old_c1['increments'][n].flatten() for n in source.INCREMENTS])
    saved=torch.load(args.stages/'diagnostic.pt',map_location='cpu',weights_only=False)['points']
    old_points=torch.load(args.replay/'diagnostic.pt',map_location='cpu',weights_only=False)['points']
    baseline=read(args.source_run/'evaluation/f68/replay/result.json')
    historical=read(args.replay/'result.json')
    assert historical['status']=='failed' and historical['completed_cache_observations']==96
    assert not args.out.exists(),'不覆盖C2正交保护实验';args.out.mkdir(parents=True)
    files=[args.proposals/n for n in read(args.proposals/'manifest.json')]+[args.proposals/'manifest.json']
    files += [folder/n for folder in (args.replay,args.stages) for n in ('protocol.json','result.json','manifest.json','diagnostic.pt')]
    files += [args.source_run/'evaluation/f68/replay/result.json']
    protocol=dict(kind='C2:same C1 raw and objective,only4 extra orthogonal-energy QP constraints',
        parameter_point='f68 zero4616',raw_C1_sha256=tensor_sha(raw['raws']['C1']['raw']),
        original_constraints=336,extra_constraints=4,extra_margin=spec['training']['default_margin'],
        extra_selection='four seen pastel alpha1,not all12 or holdout',extra_actual_tolerance=1e-8,
        original_loss_step_and_gates_unchanged=True,new_constraint_is_stricter=True,
        fresh_parameter_VJP=8,repeat_groups=4,old_G_reuse='same unchanged native FP32 branch,f68 zero4616 only',
        original_training=spec['training'],original_acceptance=spec['acceptance'],
        new_S0=0,UNet_sampling_calls=0,optimizer_steps=0,planned_QP_solves=1,holdout_read=0,adopted_checkpoints=[],
        input_sha256={str(p.resolve()):sha(p) for p in files},
        source_sha256=dict(previous['source_sha256'],**{n:sha(SOURCE/n) for n in EXTRA}),
        execution_commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip())
    write(args.out/'protocol.json',protocol)
    result=dict(status='incomplete',passed=False,new_parameter_VJP=0,repeat_groups=0,completed_cache_observations=0,
        zero_PNG_and_seven_task_controls=0,new_S0=0,UNet_sampling_calls=0,optimizer_steps=0,QP_solves=0,
        model_VAE_decode_calls=0,holdout_read=0,formal_adoption_eligibility=False,adopted_checkpoints=[],orthogonal_controls=[])
    tensors=dict(orth_gradients=[],points={});error=None
    try:
        runtime=copy.copy(args);runtime.action='train';runtime.out=args.source_run
        _,pipe,net=pilot.load_core(runtime);net.eval().requires_grad_(False)
        pipe.vae.float().eval().requires_grad_(False);head=source.load_head(args,args.device)
        parameters=head.increment_parameters();zero=torch.zeros_like(old_vector)
        hashes=dict(base=weights_digest(head.base),net=weights_digest(net),vae=weights_digest(pipe.vae))
        assert hashes['net']==historical['frozen_hashes']['net'] and hashes['vae']==historical['frozen_hashes']['vae']
        result.update(environment=environment(),frozen_hashes=hashes)
        def decode(z):
            result['model_VAE_decode_calls']+=1
            return branch.delta.decode_raw32(pipe.vae,z)
        def native_decode(z):
            pipe.vae.half()
            try:
                result['model_VAE_decode_calls']+=1
                return branch.delta.post.q8.r16.native_rgb_half(pipe.vae,z).float()
            finally:pipe.vae.float()
        overlay=read(args.source_run/'one_step_overlay.json');zero_cache={};fresh_rows=[]
        for index,pair in enumerate(spec['train']):
            cache,_=source.load_cache(args,index,overlay)
            item=next(r for r in spec['inputs'] if r['case']==pair['case_id'] and r['group']=='pastel' and r['arm']=='a100')
            cond=item['sample_condition'];key=item['case']+'/'+cond;record=cache['records'][cond]
            positive,weight=[record['target_record'][i].to(args.device) for i in (1,4)]
            feature,mask,base_b=[record[n].to(args.device) for n in ('feature','mask','base_delta')]
            zbase=record['final'].to(args.device);assert tensor_sha(zbase)==tensor_sha(old_points[key]['base_latent'])
            with torch.no_grad():
                native=native_decode(zbase);reference=decode(zbase.float())
                assert torch.equal(native.cpu(),saved[key]['native_rgb_half'].float())
                assert torch.equal(head.from_features(feature,mask,head.statistics(record['target_record'][3]['reference'].to(args.device))['phi']),base_b)
                phi=head.statistics(record['target_record'][3]['reference'].to(args.device))['phi']
                kappa=torch.tensor(old_points[key]['kappa'],device=args.device,dtype=torch.float32)
                zero_cache[key]=(native,reference)
            gradients=[];numbers=[]
            for repeat in range(2):
                z=branch.continuous_latent(zbase,kappa,head.from_features(feature,mask,phi),base_b)
                rgb=branch.delta.anchored_rgb(native,reference,checkpoint(decode,z,use_reentrant=False))
                assert torch.equal(rgb.detach(),native)
                fields=projection(rgb,positive,positive,weight)
                measured={n:float(v.detach()) for n,v in fields.items()}
                assert measured==shards[index]['late']['a100']['features']
                gradient=fresh.scaled_scalar(fields['orth'],parameters).detach().cpu()
                gradients.append(gradient);numbers.append(measured);result['new_parameter_VJP']+=1
            # 单条4616维梯度必须整体比较，不能调用按七任务矩阵取行的接口。
            repeated=branch.delta.post.q8.r16.repeat_report(gradients[1],gradients[0]);assert repeated['passed'] and numbers[0]==numbers[1]
            old_orth=float(projection(saved[key]['outputs']['C1'].to(args.device),positive,positive,weight)['orth'])
            predicted=float(gradients[0].double()@old_vector.double());actual=old_orth-numbers[0]['orth']
            assert np.sign(predicted)==np.sign(actual),'新增正交梯度方向与真实旧C1端点符号不一致'
            control=dict(case=item['case'],sample_condition=cond,baseline=numbers[0]['orth'],
                old_C1_predicted_delta=predicted,old_C1_actual_delta=actual,
                old_C1_relative_linear_remainder=abs(predicted-actual)/max(abs(predicted),1e-12),repeat=repeated)
            result['orthogonal_controls'].append(control);result['repeat_groups']+=1
            tensors['orth_gradients'].append(dict(control,gradients=torch.stack(gradients)))
            fresh_rows.append(gradients[0]);print('ORTH_FRESH',index,result['new_parameter_VJP'],flush=True)
        assert result['new_parameter_VJP']==8 and result['repeat_groups']==4
        constraints=torch.cat((raw['constraints'],torch.stack(fresh_rows)))
        margins=torch.cat((raw['margins'],torch.full((4,),spec['training']['default_margin'])))
        norms,active,normalized=source.qr.normalization(constraints);assert bool(active.all())
        rows=normalized.double().numpy();q,r,qr_report=source.qr.factorize(rows)
        diagnostic,numeric=source.qr.solve_arm(rows,q,raw['raws']['C1']['raw'],raw['raws']['C1']['raw_norm'],margins.double().numpy(),qr_report)
        result['QP_solves']=1;result['solver']=dict(qr=qr_report,arm=diagnostic)
        assert qr_report['passed'] and diagnostic['numerical_qualification']
        candidate,step=source.original_step(raw['raws']['C1']['raw'],numeric['delta32'],constraints,spec['training'])
        assert step['parameter_updates']==1 and step['minimum_cosine']>=-1e-5
        report=dict(step,arm='C2',base_b_hash=hashes['base'],adopted=False,
            projection=dict(feasible=True,method='same QR/SLSQP with4 extra rows',actual_slack_min=diagnostic['fp32_slack_min']),
            protocol_sha256=sha(args.out/'protocol.json'))
        fresh.legacy.set_parameters(parameters,candidate.to(args.device))
        torch.save(dict(increments={n:getattr(head,n).detach().cpu() for n in source.INCREMENTS},calibration=source.preflight.LOCAL_CALIBRATION,report=report),args.out/'C2.pt')
        torch.save(dict(constraints=constraints,margins=margins,norms=norms,active=active,N32=normalized,
            Q=torch.from_numpy(q),R=torch.from_numpy(r),arm=numeric,original_raw_C1=raw['raws']['C1']['raw'],candidate=candidate),args.out/'numerics.pt')
        result['proposal']=report;candidate_hash=tensor_sha(fresh.legacy.flatten(parameters))
        entries,_,_=evaluation.replay_inputs(runtime,spec)
        baseline_tasks={(v['case'],v['sample_condition']):v for shard in shards for v in shard['values']}
        candidate_rows=[];late={}
        with torch.no_grad():
            for index,pair in enumerate(spec['train']):
                cache,_=source.load_cache(args,index,overlay)
                objective=source.objective_from_cache(runtime,pipe,head,pair,cache)
                meta=read(args.basis/('cache/shard%d.json'%index))
                assert source.preflight.scheduler_configuration(pipe.scheduler.config,meta['scheduler']['config'])['normalized_equal']
                assert [hashes['base'],hashes['net'],hashes['vae']]==[cache['b_hash'],cache['net_hash'],meta['vae_hash_fp32']]
                pipe.scheduler.set_timesteps(50,device=args.device)
                for entry in [v for v in entries if v['case']==pair['case_id']]:
                    case,cond=entry['case'],entry['sample_condition'];key=case+'/'+cond;record=cache['records'][cond]
                    original=baseline_tasks[case,cond];zbase=record['final'].to(args.device)
                    if key in zero_cache:native,reference=zero_cache[key]
                    else:native=native_decode(zbase);reference=decode(zbase.float())
                    assert tensor_sha(native)==original['native_RGB_sha256']
                    assert tensor_sha(torch.from_numpy(branch.delta.quantize(native)))==original['pixel_array_sha256']
                    no_b=pipe.scheduler.step(record['raw_epsilon'].to(args.device),pipe.scheduler.timesteps[-1],record['zt'].to(args.device),return_dict=False)[0].float()
                    result['model_VAE_decode_calls']+=1;no_b_rgb=objective.rgb(no_b).detach()
                    zero_tasks=fresh.task_values(objective,dict(pair=pair,condition=cond),zbase.float(),native,no_b_rgb)
                    assert {n:float(v) for n,v in zero_tasks.items()}==original['tasks']
                    result['zero_PNG_and_seven_task_controls']+=1
                    feature,mask,base_b=[record[n].to(args.device) for n in ('feature','mask','base_delta')]
                    phi=head.statistics(record['target_record'][3]['reference'].to(args.device))['phi'];assert float(phi)==original['phi']
                    kappa=branch.delta.post.coefficient(torch.tensor(record['alpha'],device=args.device,dtype=torch.float32),pipe.scheduler.final_alpha_cumprod.to(args.device).float())
                    current_b=head.from_features(feature,mask,phi);z=branch.continuous_latent(zbase,kappa,current_b,base_b)
                    rgb=branch.delta.anchored_rgb(native,reference,decode(z))
                    tasks=fresh.task_values(objective,dict(pair=pair,condition=cond),z,rgb,no_b_rgb);numbers={n:float(v) for n,v in tasks.items()}
                    image=branch.delta.quantize(rgb);target=args.out/'C2'/(case+'_'+cond+'.png');target.parent.mkdir(exist_ok=True);Image.fromarray(image).save(target)
                    row=dict(case=case,condition=entry['condition'],sample_condition=cond,group=entry['group'],arm=entry['arm'],factor=entry['factor'],model_arm='C2',
                        seed=pair['seed'],noise_digest=record['noise_digest'],prefix_metadata_inherited=True,image=target.relative_to(args.out).as_posix(),
                        png_sha256=sha(target),pixel_array_sha256=tensor_sha(torch.from_numpy(image)),rgb_float32_sha256=tensor_sha(rgb),latent_continuous_sha256=tensor_sha(z),
                        training_tasks=numbers,training_task_comparison=evaluation.training_comparison(numbers,original['tasks']),cached_frozen_prefix=True,new_S0_calls=0,
                        **evaluation.score_entry(entry,image,'replay'))
                    candidate_rows.append(row)
                    pt=dict(latent=z.cpu(),delta_B_difference=(current_b-base_b).cpu(),kappa=float(kappa),phi=float(phi))
                    item=next(v for v in spec['inputs'] if (v['case'],v['sample_condition'])==(case,cond))
                    if source.preflight.old.key(item) in fresh.SENTINELS:pt['rgb']=rgb.cpu()
                    tensors['points'][key]=pt
                    if item['group']=='pastel':
                        positive,weight=[record['target_record'][i].to(args.device) for i in (1,4)]
                        base_positive=cache['records'][item['condition']+'_a100']['target_record'][1].to(args.device)
                        late[key]={n:float(v) for n,v in projection(rgb,positive,base_positive,weight).items()}
                    result['completed_cache_observations']+=1
                    print('ORTH_C2_REPLAY',result['completed_cache_observations'],flush=True)
        assert result['completed_cache_observations']==result['zero_PNG_and_seven_task_controls']==48
        failures=[dict(case=v['case'],sample_condition=v['sample_condition'],failures=v['training_task_comparison']['failures']) for v in candidate_rows if v['training_task_comparison']['failures']]
        replay=dict(arm='C2',split='replay',rows=candidate_rows,**evaluation.aggregate(candidate_rows,'replay'),
            training_task_protection=dict(passed=not failures,failures=failures,conditions=48,tasks=336),
            strict_replay_protection=evaluation.strict_replay_protection(candidate_rows,{(v['case'],v['condition'],v['arm']):v for v in baseline['rows']}))
        replay['original_replay_checks']=replay_eligibility(report,replay,baseline)
        replay['pastel_quality_pass']={a:sum(v['scores']['l_quality_pass'] for v in candidate_rows if v['group']=='pastel' and v['arm']==a) for a in ('a075','a100','a125')}
        result.update(replay=replay,late_features=late,extra_guard_actual=[dict(case=v['case'],sample_condition=v['sample_condition'],
            delta=late[v['case']+'/'+v['sample_condition']]['orth']-v['baseline']) for v in result['orthogonal_controls']])
        assert [weights_digest(head.base),weights_digest(net),weights_digest(pipe.vae)]==[hashes['base'],hashes['net'],hashes['vae']]
        assert tensor_sha(fresh.legacy.flatten(parameters))==candidate_hash
        assert not any(p.grad is not None for model in (head,net,pipe.vae) for p in model.parameters())
        result['frozen_unchanged']=True
        result['extra_guard_actual_passed']=all(v['delta']<=1e-8 for v in result['extra_guard_actual'])
        # 历史125693缺失seed；只在内存修复同源元数据，保留原失败档案与冻结后检事实。
        old_models,_=repaired_models(historical,baseline,read(args.proposals/'proposals.json'))
        result['comparisons']={}
        for name,reference in [('f68',baseline),('C0',old_models['C0']),('C1',old_models['C1'])]:
            changes=summary.paired_changes(reference,replay,'replay');response=summary.response_changes(reference,replay)
            gains=changes['pastel']['a100']['metrics'];slope=response['pastel']['center_slope']['mean_error_delta']
            result['comparisons'][name]=dict(paired=changes,response=response,
                innovation=dict(slope_error_delta=slope,gain_error_delta=gains['l_target_gain']['mean_error_delta'],correlation_delta=gains['l_correlation']['mean_delta'],
                    passed=slope < -1e-6 and gains['l_target_gain']['mean_error_delta'] < -1e-6 and gains['l_correlation']['mean_delta']>=-1e-5))
        result.update(status='complete',passed=True,frozen_unchanged=True,meaning_of_passed='execution controls only;inspect all original gates,quality and extra guard')
    except Exception as caught:
        error=caught;result.update(status='failed',passed=False,error_type=type(caught).__name__,message=str(caught))
    try:
        controls(args)
        assert all(sha(Path(n))==h for n,h in protocol['input_sha256'].items())
        assert all(sha(SOURCE/n)==h for n,h in protocol['source_sha256'].items())
        result['source_final_verification']=True
    except Exception as caught:
        error=error or caught;result.update(status='failed',passed=False,source_final_verification=False,source_error=str(caught))
    result['seconds']=time.monotonic()-started;torch.save(tensors,args.out/'diagnostic.pt');result['tensor_file_sha256']=sha(args.out/'diagnostic.pt')
    write(args.out/'result.json',result);names=[str(p.relative_to(args.out).as_posix()) for p in args.out.rglob('*') if p.is_file()]
    write(args.out/'manifest.json',{n:sha(args.out/n) for n in names})
    with zipfile.ZipFile(args.out/'review.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for n in names+['manifest.json']:archive.write(args.out/n,n)
    print('ORTH_GUARD_REVIEW',result['passed'],sha(args.out/'review.zip'),flush=True)
    if error is not None:raise error


if __name__=='__main__':
    args=proposal.arguments();args.proposals=args.out
    auto=args.prior/'autonomous_repair_20261116'
    args.replay=auto/'reference_terminal_fp32_replay';args.stages=auto/'reference_terminal_fp32_pair_metric_stages'
    args.out=auto/'reference_terminal_orth_guard';args.device='cuda:0';torch.set_num_threads(2);cv2.setNumThreads(2);run(args)
