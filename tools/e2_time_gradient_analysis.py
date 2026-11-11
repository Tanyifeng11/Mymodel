"""独立身份聚合、全时间读出、诊断证据汇总与Slurm成功依赖调度。"""
from collections import defaultdict
import json
import math
from pathlib import Path
import subprocess
import time
import zipfile

import numpy as np

from models.e2_texture_diagnostics import grouped_ridge,group_accuracy
from tools.e2_fft_experiment import read,write,sha,SOURCE
from tools.e2_time_gradient import initialize,code_environment
from tools.e2_time_gradient_common import descriptive,TRAIN_TIMES,NOISE_SEEDS


def aggregate_trace(out,spec,pairs):
    audits=[read(out/'audit'/('trace_shard_%d.json'%i)) for i in range(2)]
    for a in audits:assert a['passed'] and a['network_state_before']==a['network_state_after']
    seen=[c for a in audits for c in a['cases']]
    assert len(seen)==32 and set(seen)=={p['case_id'] for p in pairs}
    records=[];arrays=defaultdict(list);positive=defaultdict(list);rows=[];common=defaultdict(list)
    for p in pairs:
        folder=out/'trace'/p['case_id'];complete=read(folder/'complete.json')
        assert complete['passed'] and complete['steps']==50 and all(r['max_pixel_error']==0 for r in complete['replays'])
        assert sha(folder/'vectors.npz')==complete['vectors_sha256'] and sha(folder/'positive.npz')==complete['positive_sha256']
        assert sha(folder/'real_trajectory.json')==complete['real_trajectory_sha256']
        rows.extend(read(folder/'control_rows.json'))
        with np.load(folder/'vectors.npz') as z:
            for k in z.files:arrays[k].append(z[k])
        with np.load(folder/'positive.npz') as z:
            for k in z.files:positive[k].append(z[k])
        real=read(folder/'real_trajectory.json');assert len(real)==50
        for r in real:
            assert r['repeat_epsilon_exact'] and r['unconditional_independent']
            for metric in ['own_epsilon','own_conditional','own_unconditional','own_latent','common_epsilon','common_x0','common_next']:
                for region in ['body','boundary','background']:
                    value=r[metric][region]['relative_rms']
                    common[r['step'],r['t'],metric,region].append(value)
            a,b=r['own_decoded']['old'],r['own_decoded']['new']
            common[r['step'],r['t'],'own_fft_response','body'].append(b['fft_margin']-a['fft_margin'])
            if 'v3_scores' in a:
                for mode in ['gray','rgb']:
                    for scale in ['strict','tolerant']:
                        key=scale+'_margin';common[r['step'],r['t'],'v3_'+mode+'_'+scale,'body'].append(b['v3_scores'][mode][key]-a['v3_scores'][mode][key])
            common[r['step'],r['t'],'x0_identity_max_error','all'].append(r['x0_identity_max_error'])
    groups=np.array([r['identity_group'] for r in rows]);labels=np.array([r['label'] for r in rows])
    phases=np.array([r['phase'] for r in rows]);predictions=[]
    def probe(stage,value,step=None):
        prediction,_=grouped_ridge(value,value,labels,groups,spec['folds'],4)
        for phase in ['all','original','shifted']:
            selected=np.ones(len(labels),bool) if phase=='all' else phases==phase
            records.append(dict(stage=stage,step=step,phase=phase,**group_accuracy(prediction[selected],labels[selected],groups[selected])))
        predictions.extend(dict(stage=stage,step=step,identity_group=g,variant=r['variant'],phase=r['phase'],
            label=int(y),prediction=int(a)) for r,g,y,a in zip(rows,groups,labels,prediction))
    for k,parts in positive.items():probe(k,np.concatenate(parts))
    for k,parts in arrays.items():
        values=np.concatenate(parts)
        assert values.shape[:2]==(256,50)
        for step in range(50):probe(k,values[:,step],step)
        print('TIME_READOUT_STAGE',k,flush=True)
    write(out/'analysis/time_readouts.json',records);write(out/'analysis/time_readout_predictions.json',predictions)
    summaries=[dict(step=s,t=t,metric=m,region=r,**descriptive(v)) for (s,t,m,r),v in sorted(common.items())]
    write(out/'analysis/time_signal.json',summaries)
    write(out/'analysis/time_interpretation_contract.json',dict(
        within_representation_time_comparison=True,cross_representation_accuracy_not_information_rank=True,
        schedules=read(out/'trace'/pairs[0]['case_id']/'schedule.json'),
        positive='samebodypatchFFT of input and VAEroundtrip;mean latent72-dimensional control',
        normalized_residual='x0_delta_scaled=-epsilon_delta up toFP32rounding;distinguishknownattenuationfrominformationabsence',
        v3_steps=spec['real_dists_score_steps'],high_t_decoded_is_extrapolation=True,
        temporal_bands='descriptive lowt<=249/mid250..749/high750..999;no confirmatory changepoint test'))
    return records,summaries


def aggregate_gradients(out,spec,probes):
    point_groups=defaultdict(lambda:defaultdict(list));norms=defaultdict(lambda:defaultdict(list))
    contexts={};point_count=0;directional=[];case_times=[]
    for shard in range(4):
        a=read(out/'audit'/('gradient_shard_%d.json'%shard));assert a['passed'] and a['generator_updates']==0
        for checkpoint in a['checkpoints']:assert checkpoint['network_state_before']==checkpoint['network_state_after']
    for checkpoint in ['mature','B0']:
        for p in probes:
            folder=out/'gradient'/checkpoint/str(p['index']);complete=read(folder/'complete.json')
            assert complete['passed'] and complete['points']==20 and complete['points_sha256']==sha(folder/'points.json')
            rows=read(folder/'points.json');assert len(rows)==20
            assert {(r['seed']-p['index'],r['t']) for r in rows}=={(s,t) for s in NOISE_SEEDS for t in TRAIN_TIMES}
            case_times.append(dict(checkpoint=checkpoint,index=p['index'],seconds=complete['seconds']))
            for r in rows:
                point_count+=1;key=(r['index'],r['seed'])
                value=(r['noise_sha256'],r['latent_sha256'],r['drawn_drop'],r['drawn_t'])
                if key in contexts:assert contexts[key]==value,'不同t或权重点的训练噪声/latent不一致'
                contexts[key]=value
                band='low' if r['t']<=249 else 'mid' if r['t']<=749 else 'high'
                for name,n in r['gradients'].items():
                    for target,g in n.items():
                        for when in [str(r['t']),band]:
                            norms[checkpoint,when,name,target][r['identity']].append(g['norm'])
                for c in r['comparisons']:
                    activity='actual_active' if c['real_active'] else 'counterfactual_inactive'
                    for when in [str(r['t']),band]:
                        base=(checkpoint,when,c['reference_loss'],c['old_loss'],c['target'],activity)
                        for metric in ['cosine','first_norm','second_norm']:
                            point_groups[base+(metric,)][r['identity']].append(c[metric])
                        if c['second_norm'] and c['dot'] is not None:
                            opposition=-c['dot']/(c['second_norm']**2)
                            point_groups[base+('opposition_ratio',)][r['identity']].append(opposition)
                            point_groups[base+('negative_fraction',)][r['identity']].append(float(c['dot']<0))
                        for region,part in c.get('regions',{}).items():
                            point_groups[base+('cosine_'+region,)][r['identity']].append(part['cosine'])
                if r.get('directional_check'):
                    directional.append(dict(checkpoint=checkpoint,index=r['index'],identity=r['identity'],t=r['t'],checks=r['directional_check']))
    assert point_count==1280
    result=[]
    for (checkpoint,when,ref,old,target,activity,metric),identities in point_groups.items():
        means=[float(np.mean([v for v in values if v is not None])) for values in identities.values() if any(v is not None for v in values)]
        result.append(dict(checkpoint=checkpoint,time=when,reference_loss=ref,old_loss=old,target=target,activity=activity,metric=metric,**descriptive(means)))
    norm_result=[]
    for (checkpoint,when,name,target),identities in norms.items():
        means=[float(np.mean([v for v in values if v is not None])) for values in identities.values() if any(v is not None for v in values)]
        norm_result.append(dict(checkpoint=checkpoint,time=when,loss=name,target=target,**descriptive(means)))
    write(out/'analysis/gradient_competition.json',result);write(out/'analysis/gradient_norms.json',norm_result)
    write(out/'analysis/directional_checks.json',directional);write(out/'analysis/gradient_resources.json',case_times)
    write(out/'analysis/gradient_interpretation_contract.json',dict(points=point_count,identities=32,checkpoints=2,noise_replicates=2,
        population='inputstratified diagnostic,24texture/8lowtexture;not unbiased training population',
        primary='real_active low noise original pairedtraining only;inactive actualreferencecosine undefined',
        counterfactual='t250+ openedgate gradients separately labelled,not original supervision',
        matched_control='GTlocalGram shares descriptor/preprocess/usedscales/bodycoords;not old maskedGram replacement',
        interpretation='negativecosine diagnostic,not actualAdamWcausality;representative parameter targets explicitly named'))
    return result


def plots(out,readouts,time_rows,gradient_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(10,8))
    for stage in ['guided_epsilon__spatial','epsilon_delta__spatial','x0_latent__spatial','x0_delta_scaled__spatial','decoded_fft']:
        a=sorted([r for r in readouts if r['stage']==stage and r['step'] is not None and r['phase']=='all'],key=lambda r:r['step'])
        axes[0].plot([r['step'] for r in a],[r['accuracy'] for r in a],label=stage)
    axes[0].axhline(.25,color='gray',ls='--');axes[0].set_ylim(0,1.03);axes[0].legend(fontsize=8)
    axes[0].set_ylabel('Identity grouped readout accuracy');axes[0].set_title('DDIM50 temporal diagnostics;covered-phase readout,not texture fidelity')
    for metric in ['common_epsilon','common_x0','common_next','own_latent']:
        a=sorted([r for r in time_rows if r['metric']==metric and r['region']=='body'],key=lambda r:r['step'])
        axes[1].plot([r['step'] for r in a],[r['mean'] for r in a],label=metric)
    axes[1].set_xlabel('Denoising step (t decreases)');axes[1].set_ylabel('Body relative RMS');axes[1].legend(fontsize=8)
    fig.tight_layout();(out/'figures').mkdir(exist_ok=True);fig.savefig(out/'figures/time_trace.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
    for ax,checkpoint in zip(axes,['mature','B0']):
        for ref in ['fft','gabor','gram']:
            a=sorted([r for r in gradient_rows if r['checkpoint']==checkpoint and r['old_loss']=='old_gram' and r['target']=='epsilon'
                and r['reference_loss']==ref and r['metric']=='cosine' and r['time'].isdigit() and r['n']],key=lambda r:int(r['time']))
            ax.plot([int(r['time']) for r in a],[r['mean'] for r in a],marker='.',label=ref)
        ax.axhline(0,color='gray',ls='--');ax.axvline(249,color='gray',ls=':');ax.set_title(checkpoint);ax.legend()
        ax.set_xlabel('Training t (250+ counterfactual gate only)')
    axes[0].set_ylabel('GT Gram vs weighted reference cosine');fig.tight_layout();fig.savefig(out/'figures/gradient_competition.png',dpi=150);plt.close(fig)


def aggregate(args):
    _,out=initialize(args);spec=read(out/'protocol.json');pairs=read(out/'strict_pairs.json')['pairs']
    assert read(out/'audit/pilot_trace.json')['passed'] and read(out/'audit/pilot_gradient.json')['passed']
    assert sha(out/'strict_pairs.json')==spec['pairs_sha256']
    for checkpoint in spec['checkpoints'].values():assert sha(checkpoint['path'])==checkpoint['sha256']
    readouts,signals=aggregate_trace(out,spec,pairs)
    gradients=aggregate_gradients(out,spec,read(out/'inputs/training_probes.json')['records'])
    plots(out,readouts,signals,gradients)
    evidence=[r for r in gradients if r['checkpoint']=='B0' and r['time']=='low' and r['activity']=='actual_active'
        and r['old_loss'] in ['old_gram','old_total','GT_localGram_diagnostic']
        and r['target'] in ['epsilon','decoded','projection0','projection1','projection2','projection3']]
    write(out/'analysis/B_direction_evidence.json',dict(
        decision_scope='choose one B hypothesis after inspecting time curves,weighted competition,matchedGT control and directionalchecks;no training claim',
        evidence=evidence,formal_followup_if_GTGram_changed='G0oldGram/noB;G1oldGram+B;G2modifiedGram/noB;G3modifiedGram+B;matchedupdatebudget',
        all_generator_updates=0,new_encoder=False,selection_pending_scientific_review=True))
    write(out/'audit/aggregate_complete.json',dict(passed=True,trace_identities=32,gradient_points=1280,
        control_readout_statistics=len(readouts),generator_updates=0,environment=code_environment()))
    print('TIME_GRADIENT_AGGREGATE_PASS',len(readouts),flush=True)


def bundle(args):
    out=Path(args.out);assert read(out/'audit/aggregate_complete.json')['passed']
    prefixes=['inputs','outputs','trace','gradient','audit','analysis','figures','logs','tests']
    files=sorted({p for prefix in prefixes for p in (out/prefix).rglob('*') if p.is_file()
        and p.suffix in ['.json','.jsonl','.txt','.png','.jpg','.jpeg','.npz','.out','.err']})+[out/'protocol.json',out/'strict_pairs.json']
    write(out/'bundle_manifest.json',{p.relative_to(out).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files})
    with zipfile.ZipFile(out/'review_results.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in files+[out/'bundle_manifest.json']:archive.write(p,p.relative_to(out).as_posix())
    write(out/'bundle_info.json',dict(bytes=(out/'review_results.zip').stat().st_size,sha256=sha(out/'review_results.zip'),files=len(files)))
    print('TIME_GRADIENT_BUNDLE_PASS',sha(out/'review_results.zip'),flush=True)


def dispatch(args):
    out=Path(args.out);spec=read(out/'protocol.json');stage=args.stage
    record_path=out/'audit'/('dispatch_'+stage+'.json')
    jobs=read(record_path) if record_path.exists() else {}
    def submit(key,arguments):
        if key in jobs:return jobs[key]
        started=time.monotonic()
        while True:
            result=subprocess.run(['sbatch','--parsable']+arguments,capture_output=True,text=True)
            if result.returncode==0:
                identifier=result.stdout.strip().split(';')[0];jobs[key]=identifier;write(record_path,jobs)
                print('TIME_DIAGNOSTIC_SUBMITTED',key,identifier,flush=True);return identifier
            write(out/'audit'/('submit_retry_'+stage+'.json'),dict(key=key,stdout=result.stdout,stderr=result.stderr))
            if not any(x in result.stderr for x in ['QOS','job submit limit','maximum number of jobs']):raise RuntimeError(result.stderr)
            assert time.monotonic()-started<6500,'GPU提交配额持续占用，保留已提交作业，不重复提交'
            time.sleep(15)
    script=str(SOURCE/'scripts/e2_time_gradient.sbatch')
    if stage=='trace':
        assert read(out/'audit/pilot_trace.json')['passed'] and read(out/'audit/pilot_gradient.json')['passed']
        ids=[]
        for index in range(2):
            ids.append(submit('trace%d'%index,['-p','gpu','--gres=gpu:A30:1','-c','2','--mem=32G','-t','04:00:00',
                '-J','E2TG-trace%d'%index,'-o',str(out/'logs'/('trace%d_%%j.out'%index)),'-e',str(out/'logs'/('trace%d_%%j.err'%index)),
                script,'trace','--shard',str(index),'--shards','2']))
        submit('gradient_dispatch',['-p','cpu','-c','1','--mem=4G','-t','02:00:00','-J','E2TG-dispatchG',
            '--dependency=afterok:'+':'.join(ids),'-o',str(out/'logs/dispatchG_%j.out'),'-e',str(out/'logs/dispatchG_%j.err'),
            script,'dispatch','--stage','gradient'])
    else:
        for i in range(2):assert read(out/'audit'/('trace_shard_%d.json'%i))['passed']
        pilot=read(out/'audit/pilot_gradient.json');seconds=[r.get('seconds',10) for r in pilot['records']]
        hours=min(8,max(2,int(math.ceil((float(np.median(seconds))*320*1.8+300)/3600))))
        ids=[]
        for index in range(4):
            ids.append(submit('gradient%d'%index,['-p','gpu','--gres=gpu:A30:1','-c','2','--mem=32G','-t','%02d:00:00'%hours,
                '-J','E2TG-grad%d'%index,'-o',str(out/'logs'/('gradient%d_%%j.out'%index)),'-e',str(out/'logs'/('gradient%d_%%j.err'%index)),
                script,'gradient','--shard',str(index),'--shards','4']))
        submit('aggregate',['-p','cpu','-c','4','--mem=32G','-t','02:00:00','-J','E2TG-aggregate',
            '--dependency=afterok:'+':'.join(ids),'-o',str(out/'logs/aggregate_%j.out'),'-e',str(out/'logs/aggregate_%j.err'),script,'finish'])
    print('TIME_GRADIENT_DISPATCH_PASS',stage,flush=True)
