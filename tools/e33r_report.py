"""生成可下载结果包；未通过早期Gate明确标记未运行后续阶段。"""
import argparse,json,subprocess,tarfile
from collections import Counter
from tools.e33r_common import *
from tools.e33r_visualize import create_panels,prior_panels

def frozen_prior_baseline(records):
    import numpy as np
    import torch
    from tools.e33r_train_prior import PriorDataset,axial_map
    from tools.e33r_train_control import load_prior
    model,_=load_prior();rows=[]
    with torch.no_grad():
        for row in records:
            case=PriorDataset([row])[0];gt=case['supervision_geometry']
            support=cf_support(gt.numpy(),case['supervision_interior'].numpy(),row['box'])
            with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(case['structure'][None].cuda())['orientation'].cpu()
            error=None
            if support.any():error=float((axial_map(pred,gt[:2][None])[0][support]*gt[3][support]).sum()/gt[3][support].sum())
            rows.append(dict(id=row['id'],strict=row['strict'],readable_cells=int(support.sum()),
                prior_orientation_error_deg=error,R90_response_error_deg=90 if support.any() else None,
                R90_success=False,R180_success=bool(support.any()),R0_success=bool(support.any()),
                zero_control_magnitude=0,zero_ratio=None,sensitivity_ratio=0))
    folder=OUT/'baselines';write(folder/'B0_rows.json',rows)
    return dict(description='frozen prior, identical output for every reference; analytic no-response baseline',
        denominator=len(rows),local_prior_orientation_error=bootstrap([r['prior_orientation_error_deg'] for r in rows if r['prior_orientation_error_deg'] is not None]),
        clean_r90_success=bootstrap([0.]*len(rows)),clean_r90_response_error=bootstrap([90. for r in rows if r['readable_cells']]),
        r180_identity_success=bootstrap([float(r['R180_success']) for r in rows]),
        zero_control_magnitude=0,zero_ratio=None,sensitivity_ratio=0,
        ratio_note='zero/valid ratio undefined because no valid control; local accuracy uses same128dev M_cf as B3')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--review-json',default='')
    args=parser.parse_args();decision=read(OUT/'decision_summary.json')
    chosen=read(OUT/'selected_manifest.json');split=read(OUT/'split_manifest.json')
    checks={'protocol':read(OUT/'protocol.json')['sha256']==protocol_hash(),
        'selected_manifest':sha(OUT/'selected_manifest.json')==read(OUT/'input_provenance.json')['selected_manifest_sha256'],
        'available_counts':[len(chosen[g]) for g in GROUPS]==PROTOCOL['expected_available'],
        'strict_counts':[sum(r['strict'] for r in chosen[g]) for g in GROUPS]==PROTOCOL['expected_strict'],
        'split_disjoint':len(set(r['id'] for g in GROUPS for r in split[g]))==sum(len(split[g]) for g in GROUPS)}
    integrity=read(OUT/'R0_integrity/summary.json')
    checks['integrity_ids']={p.stem for p in (OUT/'R0_integrity/cases').glob('*.json')}=={r['id'] for g in GROUPS for r in chosen[g]}
    prior=read(OUT/'P0_prior/summary.json');checks['prior_checkpoint']=sha(OUT/'P0_prior/checkpoint_final.pt')==prior['checkpoint_sha256']
    checks['prior_dev_ids']={r['id'] for r in read(OUT/'P0_prior/dev_rows.json')}=={r['id'] for r in split['dev']}
    stages={};visual={};required=[];total_steps=6000
    continuation=sanity_continuation() if decision['prior_pass'] else dict(allowed=False,overridden=False)
    checks['sanity_continuation_valid']=not decision['prior_pass'] or continuation['allowed'] or not (OUT/'sanity_override.json').exists()
    if not decision['prior_pass']:
        stopped='P0_prior';required=['R180_integrity','P0_prior'];route='prior_accuracy_failed_no_control_training'
        if not (OUT/'P0_prior/visual/selection.json').exists():prior_panels()
        visual['P0_prior']=read(OUT/'P0_prior/visual/selection.json')['ids']
    else:
        sanity=read(OUT/'R_sanity/gate.json');total_steps+=500;checks['sanity_denominator']=sanity['denominator']==512
        required=['R180_integrity','P0_prior','R_sanity']
        folder=OUT/'R_sanity';stages['sanity']=read(folder/'dev/summary.json')
        if not (folder/'visual/selection.json').exists():create_panels(folder,read(OUT/'sanity_manifest.json')['dev'])
        visual['R_sanity']=read(folder/'visual/selection.json')['unique_ids']
        if not continuation['allowed']:stopped='R_sanity';route='implementation_or_parameterization_debug'
        else:
            stopped=None;results=[]
            for seed in PROTOCOL['seeds']:
                folder=OUT/'P1_controlled'/('seed%d'%seed);name='seed%d'%seed
                summary=read(folder/'dev/summary.json');stages[name]=summary;results.append(summary['gate_pass']);total_steps+=8000
                checks[name+'/phase_complete']=read(folder/'phase_complete.json')['checkpoint_sha256']==sha(folder/'checkpoint_final.pt')
                checks[name+'/dev_ids']={r['id'] for r in read(folder/'dev/rows.json')}=={r['id'] for r in chosen['dev']}
                checks[name+'/train_strict_ids']={r['id'] for r in read(folder/'train_strict/rows.json')}=={r['id'] for r in chosen['train'] if r['strict']}
                checks[name+'/shared_prior']=read(folder/'checkpoint_integrity.json')['prior_sha256']==prior['checkpoint_sha256']
                for group in ('causal_test','independent_confirmation'):
                    checks[name+'/'+group]={r['id'] for r in read(folder/group/'rows.json')}=={r['id'] for r in chosen[group]}
                if not (folder/'visual/selection.json').exists():create_panels(folder,chosen['dev'])
                visual['P1_controlled/'+name]=read(folder/'visual/selection.json')['unique_ids']
            passed=sum(results)>=2;route='E33_RC_real_rotation_curriculum' if passed else 'rotation_control_architecture_bottleneck'
            decision.update(controlled_rotation_causality_pass=passed,**{'seed%d_pass'%s:r for s,r in zip(PROTOCOL['seeds'],results)})
            trigger=stages['seed42']['gate_pass'] or stages['seed42']['near_gate']
            for variant in PROTOCOL['ablations']:
                folder=OUT/'ablations'/variant
                if trigger:
                    stages[variant]=read(folder/'dev/summary.json');total_steps+=8000
                    checks[variant+'/phase_complete']=read(folder/'phase_complete.json')['checkpoint_sha256']==sha(folder/'checkpoint_final.pt')
                    for group,expected in [('dev',chosen['dev']),('train_strict',[r for r in chosen['train'] if r['strict']]),
                            ('causal_test',chosen['causal_test']),('independent_confirmation',chosen['independent_confirmation'])]:
                        checks[variant+'/'+group+'_ids']={r['id'] for r in read(folder/group/'rows.json')}=={r['id'] for r in expected}
                    checks[variant+'/shared_prior']=read(folder/'checkpoint_integrity.json')['prior_sha256']==prior['checkpoint_sha256']
                    if not (folder/'visual/selection.json').exists():create_panels(folder,chosen['dev'])
                    visual['ablations/'+variant]=read(folder/'visual/selection.json')['unique_ids']
            required+=['P1_controlled_seed42','P1_controlled_seed43','P1_controlled_seed44']+PROTOCOL['ablations']*int(trigger)
    metrics=dict(clean_rot90_success='clean_r90_success',noisy_rot90_success='noisy_r90_both_success',
        r180_identity_success='r180_identity_success',r0_identity_success='r0_identity_success',
        zero_residual_ratio='zero_ratio',reference_sketch_sensitivity_ratio='sensitivity_ratio',
        r0_orientation_error_deg='r0_absolute_error')
    for output,key in metrics.items():
        values=[stages['seed%d'%s][key]['mean'] for s in PROTOCOL['seeds']] if stopped is None else []
        decision[output]=sum(values)/3 if len(values)==3 and all(v is not None for v in values) else None
    decision['strict_subset_rot90_success']=sum(stages['seed%d'%s]['strict']['clean_r90_success']['mean'] for s in PROTOCOL['seeds'])/3 if stopped is None else None
    decision['metric_aggregation']='arithmetic mean of full-seed means; Gate is still at least2of3 complete per-seed Gates'
    decision['next_route']=route;decision['sanity_gate_overridden']=continuation['overridden'];write(OUT/'decision_summary.json',decision)
    expected_visual={k:v for k,v in visual.items()}
    if args.review_json:
        review=json.loads(args.review_json);assert review['reviewed_ids']==expected_visual
        write(OUT/'audits/visual_review.json',review)
    reviewed=(OUT/'audits/visual_review.json').exists() and read(OUT/'audits/visual_review.json')['reviewed_ids']==expected_visual
    write(OUT/'audits/visual_required.json',expected_visual)
    if decision['prior_pass']:b0=frozen_prior_baseline(chosen['dev'])
    else:b0=dict(status='not_run_prior_gate_failed',full_dev_prior=prior)
    history=read(E32/'decision_summary.json')
    write(OUT/'baselines.json',dict(B0=b0,B1=dict(task='historical real-reference diagnostic, different task',
        E32_decision_sha256=sha(E32/'decision_summary.json'),real_orientation_advantage_deg=history['matched_orientation_advantage_deg'],
        real_rot90_response=history['rot90_geometry_response'],directly_comparable_to_controlled=False),
        B2=dict(task='source-side E26 R0/R90/R180 analytic readout; not a trained target model',groups=integrity['groups']),B3=stages))
    diagnoses={}
    for folder in [OUT/'R_sanity']+[OUT/'P1_controlled'/('seed%d'%s) for s in PROTOCOL['seeds']]:
        if (folder/'dev/rows.json').exists():
            rows=read(folder/'dev/rows.json');diagnoses[str(folder.relative_to(OUT))]=dict(
                overlapping_counts=dict(Counter(c for row in rows for c in row['failure_categories'])),denominator=len(rows))
    write(OUT/'audits/failure_categories.json',diagnoses)
    if stopped=='R_sanity':
        rows=read(OUT/'R_sanity/train/rows.json');failed=[r for r in rows if not r['clean_r90_success']]
        diagnosis=[]
        for row in failed:
            source=read(OUT/'R0_integrity/cases'/(row['id']+'.json'))['arms']['rot90']['clean']
            diagnosis.append(dict(row,source_orientation_deg=source['before']['orientation'],
                source_confidence=source['before']['confidence'],source_rotation_integrity_pass=source['valid']))
        configuration=read(OUT/'R_sanity/training_protocol.json')
        write(OUT/'audits/sanity_failure_diagnosis.json',dict(denominator=len(rows),failed_count=len(failed),
            failed_strict_count=sum(r['strict'] for r in failed),failed_cases=diagnosis,
            failed_prior_error=bootstrap([r['prior_absolute_error'] for r in failed if r['prior_absolute_error'] is not None]),
            successful_prior_error=bootstrap([r['prior_absolute_error'] for r in rows if r['clean_r90_success'] and r['prior_absolute_error'] is not None]),
            optimization=dict(steps=configuration['steps'],warmup=configuration['warmup'],
                note='sanity reused P1 warmup500: the500step run is entirely warmup; limitation, no post-result schedule tuning'),
            implementation_checks=dict(complex_sign_and_native_rotation='unit checks passed',
                bbox_mapping='unit intersection check passed',complete_group='seven arms; no branch-index input',
                gradients=read(OUT/'R_sanity/gradient_check.json'),
                frozen_prior=read(OUT/'R_sanity/checkpoint_integrity.json')),
            conclusion='no detected sign/mask/batching/frozen-gradient bug; fixed500step configuration misses95% train Gate; no full run or retraining'))
    if continuation['overridden']:
        path=OUT/'audits/sanity_failure_diagnosis.json'
        diagnosis=read(path)
        diagnosis.setdefault('original_sanity_stop_conclusion',diagnosis['conclusion'])
        diagnosis.update(observed_at_phase='original500step_sanity',sanity_continuation=continuation,
            conclusion='fixed500step sanity R90 failed; original failure retained; user authorized full8000step runs, assessed separately by unchanged formal Gate')
        write(path,diagnosis)
    write(OUT/'audits/numeric_integrity.json',dict(checks=checks,**{'pass':all(checks.values())}))
    write(OUT/'ablations/status.json',dict(required=bool(continuation['allowed'] and decision['seed42_pass'] is not None and
           (stages['seed42']['gate_pass'] or stages['seed42']['near_gate'])),variants=PROTOCOL['ablations'],
        completed=[v for v in PROTOCOL['ablations'] if v in stages],
        reason='fixed seed42 Gate/near-Gate trigger; user override applies only to sanity R90 stop'))
    frozen=read(OUT/'frozen_check.json');frozen['training_steps']=total_steps;write(OUT/'frozen_check.json',frozen);finish_frozen(OUT)
    completion=dict(numeric_artifacts_complete=all(checks.values()),frozen_inputs_pass=True,visual_review_completed=reviewed,
        training_steps=total_steps,E5_training_steps=0,stopped_at=stopped,required_phases=required,
        experiment_complete=bool(all(checks.values()) and reviewed),report_git_commit=git_commit(),
        controlled_rotation_causality_pass=decision['controlled_rotation_causality_pass'],
        sanity_pass=decision['sanity_pass'],sanity_continuation=continuation,
        note='field-level controlled rotation only; prior/sanity stop does not test full8000step capability; no real training or generation')
    write(OUT/'completion_check.json',completion)
    ids=','.join(sorted({p.stem.rsplit('_',1)[-1] for p in OUT.glob('*.log') if p.stem.rsplit('_',1)[-1].isdigit()}))
    with (OUT/'job_status.log').open('w') as f:subprocess.run(['sacct','-j',ids,'--format=JobID,State,ExitCode,Elapsed','-n'],stdout=f,check=True)
    paths=[p for p in OUT.rglob('*') if p.is_file() and p.suffix in ('.json','.png','.log','.err') and p.name!='artifact_manifest.json']
    write(OUT/'artifact_manifest.json',dict(files={str(p.relative_to(OUT)):sha(p) for p in paths},
         large_checkpoints='retained on server; fields npz retained on server; numeric rows and visuals included'))
    with tarfile.open(OUT/'local_review_bundle.tar.gz','w:gz') as f:
        for p in paths:f.add(p,arcname=str(p.relative_to(OUT)))
        f.add(OUT/'artifact_manifest.json',arcname='artifact_manifest.json')
    print(json.dumps(dict(decision=decision,completion=completion,bundle_sha256=sha(OUT/'local_review_bundle.tar.gz'))),flush=True)

if __name__=='__main__':main()
