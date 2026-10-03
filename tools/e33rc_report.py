"""数值验收与可下载包；真实科学失败仍可完整完成实验，人工复核前不标完成。"""
import argparse
import json
import subprocess
import tarfile
import numpy as np
import torch
from tools.e33rc_common import *
from tools.e33rc_visualize import create_panels

def report(review_json=''):
    selected=read(OUT/'selected_full_runs.json');revision=selected['revision'];seeds=selected['seeds']
    passed=[];controlled_pass=[];full=[];table=[];numeric={};visual={}
    def audit_stage(folder,diagnostic_name):
        real=read(folder/'real/summary.json');cf=read(folder/'controlled/summary.json')
        cfrows=read(folder/'controlled/rows.json');realrows=read(folder/'real/rows.json')
        expected=read(OUT/'split_manifest.json')
        numeric[diagnostic_name+'/controlled_ids']={r['id'] for r in cfrows}=={r['id'] for r in read(OUT/'controlled_manifest.json')['dev']}
        numeric[diagnostic_name+'/real_ids']={r['id'] for r in realrows}=={r['id'] for r in expected['dev']}
        numeric[diagnostic_name+'/finite']=all(r['finite'] for r in cfrows+realrows)
        numeric[diagnostic_name+'/real_readability']=real['readable_cases']==217
        numeric[diagnostic_name+'/strict']=cf['strict']['denominator']==45
        numeric[diagnostic_name+'/retention_gate']=cf['retention_gate']==retention_gate(cf)
        numeric[diagnostic_name+'/real_gate']=real['gate']==real_gate(real)
        for g in ['causal_test','independent_confirmation']:
            numeric[diagnostic_name+'/'+g]={r['id'] for r in read(folder/g/'rows.json')}=={r['id'] for r in expected[g]}
        table.append(dict(model=diagnostic_name,controlled=cf,real=real))
        if not (folder/'visual_audit/selection.json').exists():create_panels(folder)
        visual[str(folder.relative_to(OUT))]=read(folder/'visual_audit/selection.json')['unique_ids']
        return cf,real
    for seed in [42,43,44]:
        audit_stage(OUT/'RC0_zero_shot'/('seed%d'%seed),'RC0_seed%d'%seed)
        status=seeds[str(seed)];folder=seed_folder(seed,revision)
        for record in status['stage_decisions']:
            cf,real=audit_stage(folder/record['stage'],'seed%d/'%seed+record['stage'])
            done=read(folder/record['stage']/'phase_complete.json')
            numeric['seed%d/'%seed+record['stage']+'/checkpoint']=done['checkpoint_sha256']==sha(folder/record['stage']/'checkpoint_final.pt')
        retained=bool(status['completed_RC_C'] and cf['retention_gate']['pass'])
        successful=bool(retained and real['gate']['pass'])
        controlled_pass.append(retained);passed.append(successful);full.append(real)
    ablation_status=read(OUT/'ablations/status.json');completed=[]
    if ablation_status['required']:
        for name in ABLATIONS:
            folder=seed_folder(42,revision,name);status=read(folder/'run_status.json');assert status['training_steps']==8000
            audit_stage(folder/STAGES[-1][0],name);completed.append(name)
    ablation_status['completed']=completed;write(OUT/'ablations/status.json',ablation_status)
    success=sum(passed)>=2;retention=sum(controlled_pass)>=2
    route='E33_I_identity_causality' if success else 'real_pair_identifiability_bottleneck' if retention else 'stop_real_curriculum'
    avg=lambda key:float(np.mean([f[key]['mean'] for f in full])) if all(f[key]['mean'] is not None for f in full) else None
    decision=dict(zero_shot_real_transfer='RC0_zero_shot: paired baseline only',controlled_retention_pass=retention,
        real_rotation_causality_pass=success,real_matched_orientation_error_deg=float(np.mean([f['errors']['matched']['mean'] for f in full])),
        real_near_orientation_advantage_deg=avg('near_advantage'),real_random_orientation_advantage_deg=avg('random_advantage'),
        real_rot90_response=avg('rot90_success'),curriculum_causal_collapse=revision or any(s['collapse'] for s in seeds.values()),
        retention_revision_used=revision,next_route=route,color_shortcut_detected=None,sketch_shortcut_resurgence=None,
        shortcut_note='requires qualitative visual assessment; approximate equality is not established by nonsignificant differences',
        averaging='description only: final-or-stopped seed endpoint means; 2of3 success requires actual RC-C and complete dual Gate')
    for seed,retained,ok in zip([42,43,44],controlled_pass,passed):
        decision['seed%d_controlled_retention'%seed]=retained;decision['seed%d_real_pass'%seed]=bool(seeds[str(seed)]['completed_RC_C'] and full[[42,43,44].index(seed)]['gate']['pass'])
    for output,key in [('controlled_clean_rot90','clean_r90_success'),('controlled_noisy_rot90','noisy_r90_both_success'),('controlled_r180_identity','r180_identity_success')]:
        endpoints=[read(seed_folder(s,revision)/seeds[str(s)]['stage_decisions'][-1]['stage']/'controlled/summary.json')[key]['mean'] for s in [42,43,44]]
        decision[output]=float(np.mean(endpoints))
    write(OUT/'decision_summary.json',decision);write(OUT/'result_table.json',table)
    write(OUT/'audits/numeric_integrity.json',dict(checks=numeric,**{'pass':all(numeric.values())}));assert all(numeric.values())
    write(OUT/'visual_audit/required.json',visual)
    if review_json:
        review=json.loads(review_json);assert review['reviewed_ids']==visual
        write(OUT/'visual_audit/review.json',review)
        for key in ['color_shortcut_detected','sketch_shortcut_resurgence']:
            if key in review:decision[key]=review[key]
        decision['shortcut_visual_observations']=review.get('observations',[]);write(OUT/'decision_summary.json',decision)
    review_path=OUT/'visual_audit/review.json';reviewed=review_path.exists() and read(review_path)['reviewed_ids']==visual
    steps=sum(read(p)['training_steps'] for p in OUT.rglob('run_status.json'))
    frozen=read(OUT/'frozen_check.json');frozen['training_steps']=steps;write(OUT/'frozen_check.json',frozen);finish_frozen()
    completion=dict(numeric_artifacts_complete=True,frozen_inputs_pass=True,visual_review_completed=reviewed,
        experiment_complete=reviewed,training_steps=steps,E5_training_steps=0,
        real_rotation_causality_pass=success,controlled_retention_pass=retention,revision_used=revision,
        completed_full_seeds=[s for s in [42,43,44] if seeds[str(s)]['completed_RC_C']],
        ablations_required=ablation_status['required'],ablations_completed=completed,report_git_commit=git_commit())
    write(OUT/'completion_check.json',completion)
    paths=[p for p in OUT.rglob('*') if p.is_file() and p.suffix in ['.json','.png','.log','.err'] and p.name!='artifact_manifest.json']
    write(OUT/'artifact_manifest.json',dict(files={str(p.relative_to(OUT)):sha(p) for p in paths},
        checkpoints_fields_teacher_cache='retained on server; numeric rows, audits, logs and selected PNGs in local review bundle'))
    with tarfile.open(OUT/'local_review_bundle.tar.gz','w:gz') as archive:
        for p in paths:archive.add(p,arcname=str(p.relative_to(OUT)))
        archive.add(OUT/'artifact_manifest.json',arcname='artifact_manifest.json')
    print(json.dumps(dict(decision=decision,completion=completion,bundle_sha256=sha(OUT/'local_review_bundle.tar.gz'))),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--review-json',default='')
    report(parser.parse_args().review_json)
