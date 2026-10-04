"""完整端点、消融、冻结和数值验收；人工逐图复核前不标记完成。"""
import argparse,json,tarfile
import numpy as np
from tools.e33rf_common import *
from tools.e33rf_visualize import create_panels
from tools.e33r_evaluate import summarize

def failure_route(real):
    r=real['rot90_success']['mean'];m=real['errors']['matched']['mean'];n=real['near_advantage']['mean']
    if r<.2:return 'real_domain_adapter_bottleneck'
    if r>=.5 and m<=7 and n<5:return 'real_pair_identifiability_bottleneck'
    if r>=.5 and n>=5 and m>7:return 'target_alignment_revision'
    return 'unclassified_gate_failure'

def report(review_json=''):
    reproduced=read(OUT/'rf0_status.json')['pass'];table=[];numeric={};visual={};probes={}
    split=read(OUT/'split_manifest.json');controlled=read(OUT/'controlled_manifest.json')['dev'];endpoints=[]
    def audit(stage,seed):
        name=str(stage.relative_to(OUT));real=read(stage/'real/summary.json');cf=read(stage/'controlled/summary.json')
        rr=read(stage/'real/rows.json');cr=read(stage/'controlled/rows.json')
        numeric[name+'/real_ids']=len(rr)==256 and {r['id'] for r in rr}=={r['id'] for r in split['dev']}
        numeric[name+'/controlled_ids']=len(cr)==128 and {r['id'] for r in cr}=={r['id'] for r in controlled}
        numeric[name+'/finite']=all(r['finite'] for r in rr+cr)
        numeric[name+'/readable']=sum(r['readable_cells']>0 for r in rr)==real['readable_cases']==217
        numeric[name+'/strict']=cf['strict']['denominator']==45
        recomputed=summarize(cr)
        numeric[name+'/controlled_statistics']=all(cf[k]==v for k,v in recomputed.items())
        numeric[name+'/controlled_gate']=cf['preservation_gate']==controlled_gate(cf)
        numeric[name+'/real_gate']=real['gate']==real_gate(real)
        numeric[name+'/pilot']=read(stage/'decision_record.json')['pilot']==pilot_gate(cf,real)
        keys=dict(near_advantage='near_advantage',random_advantage='random_advantage',rot90_success='rot90_response_success',
            rot90_error='rot90_response_error',r180_identity='r180_identity_success',r180_error='r180_response_error',adapter_residual_norm='adapter_residual_norm')
        for result,key in keys.items():numeric[name+'/'+result]=real[result]==bootstrap([float(r[key]) for r in rr if r[key] is not None])
        for arm in ['matched','rot90','rot180','color_near','random','zero']:
            numeric[name+'/'+arm]=real['errors'][arm]==bootstrap([float(r[arm+'_error']) for r in rr if r[arm+'_error'] is not None])
        for group in ['causal_test','independent_confirmation']:
            rows=read(stage/group/'rows.json');numeric[name+'/'+group]=len(rows)==len(split[group]) and {r['id'] for r in rows}=={r['id'] for r in split[group]} and all(r['finite'] for r in rows)
        drift=read(stage/'feature_drift/rows.json')
        numeric[name+'/drift_ids']=len(drift)==64 and {r['id'] for r in drift}==set(read(OUT/'feature_drift_ids.json'))
        numeric[name+'/drift_finite']=all(np.isfinite([r[k] for k in ['cosine_drift','raw_residual_norm','scaled_residual_norm','rotation_relative_cosine_change']]).all() for r in drift)
        done=stage/'phase_complete.json'
        if done.exists():
            value=read(done);phase=value['phase'];numeric[name+'/steps']=value['training_steps']==PROTOCOL['phases'][phase]['steps']
            numeric[name+'/checkpoint']=value['checkpoint_sha256']==sha(stage/'checkpoint_final.pt')
            numeric[name+'/frozen_backbone']=read(stage/'backbone_frozen_check.json')['pass']
        ids,probe=create_panels(stage,seed);visual[name]=ids;probes[name]=probe
        table.append(dict(model=name,controlled=cf,real=real,drift=drift));return cf,real
    routes={};passed=[];cf_pass=[];rf1collapse={};completed_ablations=[]
    for seed in SEEDS:
        baseline=OUT/'RF0_reproduction'/('seed%d'%seed);audit(baseline,seed)
        numeric['RF0_seed%d'%seed]=read(baseline/'reproduction_gate.json')['pass']==all(read(baseline/'reproduction_gate.json')['checks'].values())
        if not reproduced:continue
        for phase in ['RF1','RF2']:
            stage=folder(seed,phase);assert (stage/'phase_complete.json').exists()
            if phase=='RF1':
                for step in [500,1000]:audit(stage/('diagnostic_step%d'%step),seed)
            cf,real=audit(stage,seed)
        rf1collapse[str(seed)]=read(folder(seed,'RF1')/'decision_record.json')['match_only_adapter_causes_real_causal_collapse']
        pilot=pilot_gate(cf,real)['pass'];rf3=folder(seed,'RF3')
        numeric['seed%d/RF3_schedule'%seed]=pilot==(rf3/'phase_complete.json').exists()
        if pilot:cf,real=audit(rf3,seed);endpoint=rf3
        else:assert (folder(seed,'RF2')/'RF3_not_run.json').exists();endpoint=folder(seed,'RF2')
        ok=controlled_gate(cf)['pass'] and real_gate(real)['pass'];passed.append(ok);cf_pass.append(controlled_gate(cf)['pass'])
        routes[str(seed)]='E33_I_identity_causality' if ok else failure_route(real)
        endpoints.append(dict(seed=seed,endpoint=str(endpoint.relative_to(OUT)),controlled=cf,real=real,dual_gate_pass=ok))
    if reproduced:
        for variant in ABLATIONS:
            stage=folder(42,'RF2',variant);assert (stage/'phase_complete.json').exists();audit(stage,42);completed_ablations.append(variant)
    success=reproduced and sum(passed)>=2
    route='reproduction_mismatch' if not reproduced else 'E33_I_identity_causality' if success else next(iter(routes.values())) if len(set(routes.values()))==1 else 'mixed_seed_failures'
    means=lambda scope,key:float(np.mean([e[scope][key]['mean'] for e in endpoints])) if endpoints else None
    decision=dict(adapter_input_channels=394,adapter_parameters=26430,adapter_alpha=.1,adapter_width=32,
        rf0_reproduction_pass=reproduced,real_domain_equivariance_adapter_pass=success,
        reproduced_all_seeds=reproduced,real_rotation_causality_pass=success,controlled_preservation_pass=reproduced and sum(cf_pass)>=2,
        seed_routes=routes,next_route=route,endpoints=endpoints,match_only_collapse=rf1collapse,
        real_matched_orientation_error_deg=float(np.mean([e['real']['errors']['matched']['mean'] for e in endpoints])) if endpoints else None,
        real_near_orientation_advantage_deg=means('real','near_advantage'),real_random_orientation_advantage_deg=means('real','random_advantage'),
        real_rot90_response=means('real','rot90_success'),real_r180_identity=means('real','r180_identity'),
        controlled_clean_rot90=means('controlled','clean_r90_success'),controlled_noisy_rot90=means('controlled','noisy_r90_both_success'),
        controlled_r180_identity=means('controlled','r180_identity_success'),E5_training_steps=0,
        destructive_interference_supported=None,real_pair_identifiability_bottleneck=any(r=='real_pair_identifiability_bottleneck' for r in routes.values()),
        real_domain_adapter_bottleneck=any(r=='real_domain_adapter_bottleneck' for r in routes.values()),
        inference_note='Real R90/R180 are explicitly optimized in RF2/RF3; response alone is not independent generalization evidence. Means descriptive, Gate evaluated per seed.',
        historical_RC=read(RC/'decision_summary.json'),averaging='per-seed endpoints; no intermediate checkpoint selection')
    decision['matched_orientation_error_deg']=decision['real_matched_orientation_error_deg']
    decision['near_orientation_advantage_deg']=decision['real_near_orientation_advantage_deg']
    decision['random_orientation_advantage_deg']=decision['real_random_orientation_advantage_deg']
    decision['adapter_feature_drift']={str(e['seed']):read(OUT/e['endpoint']/'feature_drift/rows.json') for e in endpoints}
    for seed in SEEDS:
        decision['seed%d_rf2_real_rot90'%seed]=read(folder(seed,'RF2')/'real/summary.json')['rot90_success']['mean'] if reproduced else None
        decision['seed%d_real_gate'%seed]=next((e['real']['gate'] for e in endpoints if e['seed']==seed),None)
    if reproduced:
        formal=read(folder(42,'RF2')/'decision_record.json');finetune=read(folder(42,'RF2','C_full_finetune')/'decision_record.json')
        decision['destructive_interference_supported']=bool(formal['controlled']['pass'] and formal['real_rot90']>=.5 and finetune['real_rot90']<.2)
        decision['destructive_interference_interpretation']='Limited seed42 RF2 contrast; RF1 is a separate matched-only diagnostic and prior RC used a different curriculum. Not proof of a unique mechanism.'
    write(OUT/'decision_summary.json',decision);write(OUT/'result_table.json',table)
    write(OUT/'audits/numeric_integrity.json',dict(checks=numeric,**{'pass':all(numeric.values())}));assert all(numeric.values())
    write(OUT/'visual_audit/required.json',visual)
    if review_json:
        review=json.loads(review_json);assert review['reviewed_ids']==visual
        write(OUT/'visual_audit/review.json',review)
    reviewpath=OUT/'visual_audit/review.json';reviewed=reviewpath.exists() and read(reviewpath)['reviewed_ids']==visual
    if reviewed:decision['visual_observations']=read(reviewpath).get('observations',[]);write(OUT/'decision_summary.json',decision)
    finish_frozen();steps=sum(read(p)['training_steps'] for p in OUT.rglob('phase_complete.json'))
    completion=dict(numeric_artifacts_complete=True,frozen_inputs_pass=True,visual_review_completed=reviewed,experiment_complete=reviewed,
        training_steps=steps,E5_training_steps=0,full_seeds_completed=SEEDS if reproduced else [],ablations_completed=completed_ablations,
        rf0_gate_failed=not reproduced,scientific_success=success,report_git_commit=git_commit())
    write(OUT/'completion_check.json',completion)
    paths=[p for p in OUT.rglob('*') if p.is_file() and p.suffix in ['.json','.png'] and p.name!='artifact_manifest.json']
    # 活跃日志只截取一次，避免manifest哈希与归档读取时刻不一致。
    for p in list(OUT.glob('*.log'))+list(OUT.glob('*.err')):
        dest=OUT/'log_snapshots'/p.name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(p.read_bytes());paths.append(dest)
    for name,ids in visual.items():
        stage=OUT/name
        paths += [stage/'real/fields'/(sid+'.npz') for sid in ids]
        paths += [stage/'controlled/fields'/(sid+'.npz') for sid in sorted(({sid for sid in ids if sid in {r['id'] for r in controlled}})|{probes[name]})]
        paths += list((stage/'feature_drift').glob('*.npz'))
    paths=sorted(set(paths));write(OUT/'artifact_manifest.json',dict(files={str(p.relative_to(OUT)):sha(p) for p in paths},checkpoints='full checkpoints and all fields retained on server'))
    with tarfile.open(OUT/'local_review_bundle.tar.gz','w:gz') as archive:
        for p in paths:archive.add(p,arcname=str(p.relative_to(OUT)))
        archive.add(OUT/'artifact_manifest.json',arcname='artifact_manifest.json')
    print(json.dumps(dict(decision=decision,completion=completion,bundle_sha256=sha(OUT/'local_review_bundle.tar.gz'))),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--review-json',default='');report(parser.parse_args().review_json)
