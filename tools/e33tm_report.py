"""只汇总实际完成的实验；硬停止后的阶段标记未运行。"""
import argparse
import tarfile
import numpy as np
from tools.e33tm_protocol import *
from tools.e33tm_metrics import bootstrap, summarize, compare

def run():
    d=read(OUT/'decision_summary.json')
    for key in ('seed42_image_r90','seed43_image_r90','seed44_image_r90',
                'seed42_image_r180','seed43_image_r180','seed44_image_r180',
                'no_text_textscore_drop','no_sketch_structure_drop','no_texture_r90_drop',
                'text_texture_conflict_detected'):
        d.setdefault(key,None)
    if d.get('field_precheck_pass') is False:
        d.update(hard_stop=True,next_route='trimodal_wrapper_integration_bug')
    cohort=read(OUT/'manifests/cohorts.json')
    # 校正早期摘要把数据集大小标成条件成功率分母的元数据；不重算/改变field分数。
    field_path=OUT/'field_precheck/summary.json'
    field_report=read(field_path)
    for item in field_report['seeds']:
        rows=read(OUT/'field_precheck'/('seed%d'%item['seed'])/'real/rows.json')
        item['real_evaluated_cases']=len(rows)
        item['real_denominator']=sum(r['rot90_response_success'] is not None for r in rows)
        write(OUT/'field_precheck'/('seed%d'%item['seed'])/'summary.json',item)
    write(field_path,field_report)
    interventions=read(OUT/'manifests/intervention_data.json') if (OUT/'manifests/intervention_data.json').exists() else None
    table=[]
    base_path=OUT/'baseline_e5/rows.json'
    baseline=read(base_path) if base_path.exists() else None
    primary_ids={r['id'] for r in cohort['primary']}
    checks={}
    passing=[]
    for name in ('baseline_e5','rf2_seed42','rf2_seed43','rf2_seed44'):
        path=OUT/name/'rows.json'
        if not path.exists(): continue
        rows=read(path)
        complete={r['id'] for r in rows}==primary_ids and len(rows)==len(primary_ids)
        checks[name+'/complete_primary']=complete
        table.append(dict(model=name,complete=complete,**summarize(rows)))
        if complete and name!='baseline_e5' and baseline and {r['id'] for r in baseline}==primary_ids:
            result=compare(rows,baseline);write(OUT/name/'comparison.json',result);passing.append(result['pass'])
    main_complete=len(passing)==3
    if main_complete: d['trimodal_reference_causality_pass']=sum(passing)>=2
    elif d.get('hard_stop'): d['trimodal_reference_causality_pass']=False
    roles=[]
    ablations=[]
    full_path=OUT/'rf2_seed42/rows.json'
    full=read(full_path) if full_path.exists() else None
    if full:
        lookup={r['id']:r for r in full}
        for setting in ('no_text','shuffled_text','no_sketch','wrong_sketch','no_texture','wrong_texture'):
            path=OUT/'ablations'/setting/'rows.json'
            if path.exists():
                values=read(path)
                expected=primary_ids if setting in ('no_text','no_sketch','no_texture') else {
                    sid for sid in primary_ids if interventions['donors'][sid] is not None}
                checks['ablation/'+setting]={r['id'] for r in values}==expected and len(values)==len(expected)
                ablations.append(dict(setting=setting,**summarize(values),requested_primary=len(primary_ids),
                                      unavailable_donor_cases=len(primary_ids)-len(expected)))
        for setting,metric in [('no_text','text_score'),('no_sketch','contour_f1'),('no_texture','r90_success')]:
            path=OUT/'ablations'/setting/'rows.json'
            if not path.exists(): continue
            rows=read(path);assert {r['id'] for r in rows}==primary_ids
            drop=bootstrap([lookup[r['id']][metric]-r[metric] for r in rows])
            item=dict(setting=setting,metric=metric,drop=drop,**summarize(rows),significant=drop['ci95'][0]>0)
            if setting=='no_text':
                item['r90_change']=bootstrap([lookup[r['id']]['r90_success']-r['r90_success'] for r in rows])
                item['r90_preserved_within_10pp']=abs(item['r90_change']['mean'])<=.1
                d['no_text_textscore_drop']=drop
            if setting=='no_sketch': d['no_sketch_structure_drop']=drop
            if setting=='no_texture': d['no_texture_r90_drop']=drop
            roles.append(item)
        if len(roles)==3: d['modality_role_disentanglement_pass']=all(r['significant'] for r in roles)
    conflict=[]
    for setting in ('C0','C1','C2'):
        path=OUT/'conflict_test'/setting/'rows.json'
        if path.exists(): conflict.append(dict(caption=setting,**summarize(read(path))))
    if len(conflict)==3:
        lookup={r['caption']:r for r in conflict}
        rates=[lookup[k]['statistics']['r90_success']['mean'] for k in ('C0','C1','C2')]
        d['text_texture_conflict_detected']=bool(rates[1]>rates[0] and rates[2]>rates[0])
    near_path=OUT/'text_compatible_near/rows.json'
    near=None
    if near_path.exists():
        rows=read(near_path);pairs=[(r,lookup_full) for r in rows for lookup_full in full if r['id']==lookup_full['id']]
        expected={sid for sid in primary_ids if interventions['text_compatible_near'][sid]['available']}
        checks['near/eligible_ids']={r['id'] for r in rows}==expected and len(rows)==len(expected)
        readable=[(r,b) for r,b in pairs if r['matched_target_error'] is not None and b['matched_target_error'] is not None]
        near=dict(available_cases=len(rows),requested_primary=len(primary_ids),diagnostic_only=True,
            matched_error=bootstrap([b['matched_target_error'] for r,b in readable]),
            near_error=bootstrap([r['matched_target_error'] for r,b in readable]),
            near_advantage=bootstrap([r['matched_target_error']-b['matched_target_error'] for r,b in readable]))
    if main_complete and len(roles)==3 and not d.get('hard_stop'):
        if not d['trimodal_reference_causality_pass']:
            gains=[read(OUT/('rf2_seed%d'%s)/'comparison.json') for s in SEEDS]
            if sum(not r['checks']['r90'] or not r['checks']['gain'] for r in gains)>=2:
                d['next_route']='field_to_generation_interface_bottleneck'
            elif sum(not r['checks']['contour'] for r in gains)>=2: d['next_route']='structure_safe_injection_revision'
            else: d['next_route']='multimodal_condition_conflict'
        elif not d['modality_role_disentanglement_pass']: d['next_route']='modality_role_calibration'
        else: d['next_route']='trimodal_causal_loop_validated'
    stage_names=('baseline_e5','rf2_seed42','rf2_seed43','rf2_seed44',
        'robustness/baseline_e5','robustness/rf2_seed42',
        'ablations/no_text','ablations/shuffled_text','ablations/no_sketch','ablations/wrong_sketch',
        'ablations/no_texture','ablations/wrong_texture','text_compatible_near',
        'conflict_test/baseline_e5','conflict_test/C0','conflict_test/C1','conflict_test/C2')
    stages={name:dict(completed=(OUT/name/'rows.json').exists(),required=(
        d.get('field_precheck_pass') is True if name in ('baseline_e5','rf2_seed42') else
        not d.get('hard_stop',False) and (not name.startswith('conflict_test/') or bool(cohort['conflict']))))
        for name in stage_names}
    d['not_run_reason']='protocol_hard_stop' if d.get('hard_stop') else None
    write(OUT/'decision_summary.json',d)
    robust=[]
    for name in ('baseline_e5','rf2_seed42','rf2_seed43','rf2_seed44'):
        path=OUT/'robustness'/name/'rows.json'
        if path.exists():
            values=read(path)
            checks['robust/'+name]=len(values)==len(cohort['robust64'])*4 and {
                (r['id'],r['diffusion_seed']) for r in values}=={(sid,seed) for sid in cohort['robust64'] for seed in (42,43,44,45)}
            robust.append(dict(model=name,diagnostic_only=True,**summarize(values)))
    error_categories={}
    if baseline:
        original={r['id']:r for r in baseline}
        for seed in SEEDS:
            path=OUT/('rf2_seed%d'%seed)/'rows.json'
            if not path.exists(): continue
            field_rows={r['id']:r for r in read(OUT/'field_precheck'/('seed%d'%seed)/'real/rows.json')}
            errors=[]
            for row in read(path):
                base=original[row['id']];tags=[]
                if not row['r90_success']:
                    tags.append('TM4_texture_R90_ignored')
                    if field_rows[row['id']]['rot90_response_success'] is True:
                        tags.append('TM1_field_causality_lost_after_E5')
                if row['contour_f1']<base['contour_f1']-.02: tags.append('TM2_structure_degradation')
                if row['text_score']<base['text_score']*.98: tags.append('TM3_text_semantic_degradation')
                if not row['r180_success']: tags.append('TM5_R180_false_response')
                if row['r90_coverage']<.5 or row['fixed_support_cells']<16: tags.append('TM10_small_readable_orientation_support')
                errors.append(dict(id=row['id'],categories=tags))
            error_categories[str(seed)]=errors
    write(OUT/'error_categories.json',error_categories)
    write(OUT/'result_table.json',dict(generation=table,modality_roles=roles,ablations=ablations,
                                     robustness=robust,conflict=conflict,text_compatible_near=near))
    frozen=freeze_check()
    complete=bool((d.get('hard_stop') or (main_complete and len(roles)==3 and len(ablations)==6 and len(robust)==2 and near is not None and
                  (len(conflict)==3 or not cohort['conflict']))) and all(checks.values()))
    write(OUT/'completion_check.json',dict(experiment_execution_complete=complete,scientific_success=bool(
        d.get('trimodal_reference_causality_pass') and d.get('modality_role_disentanglement_pass')),
        hard_stop=d.get('hard_stop',False),stages=stages,numeric_checks=checks,frozen_pass=frozen['pass']))
    files={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and
           p.name not in ('artifact_manifest.json','final_review_bundle.tar.gz','smoke_review.zip') and
           not p.name.endswith(('.log','.err'))}
    write(OUT/'artifact_manifest.json',dict(files=files,git_commit=commit(),training_steps=0))
    with tarfile.open(OUT/'final_review_bundle.tar.gz','w:gz') as archive:
        for path in OUT.rglob('*'):
            if not path.is_file() or path.name=='final_review_bundle.tar.gz': continue
            if path.suffix=='.json' or (path.suffix=='.png' and ('visual_audit' in path.parts or 'smoke_audit' in path.parts)):
                archive.add(path,arcname=str(path.relative_to(OUT)))
    print('[E33TM final]',d,flush=True)
    return d

if __name__=='__main__': run()
