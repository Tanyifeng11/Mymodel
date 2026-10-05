"""按固定分母和共同可读身份汇总；选择候选后不再改变confirmation口径。"""
import argparse
import numpy as np
from tools.e33tmif_protocol import *
from tools.e33tmif_metrics import summarize,compare,bootstrap,pair
from tools.e33tmif_run import case_path,field_path,complete

STAGES=['S0','S1','S2','S4']+['x0_%02d'%i for i in range(1,9)]+['S9']

def records(group,seed,ids,stage):
    rows=[]
    for sid in ids:
        path=case_path(group,seed,sid)
        assert complete(path/'case.json'), '缺失或指纹错误：'+str(path)
        rows.append(read(path/stage/'pair.json'))
    return rows

def stage_summary(seed,ids,stages,group='stage_survival'):
    return {s:summarize(records(group,seed,ids,s)) for s in stages}

def reproduce(cohort):
    results=[]
    primary=[r['id'] for r in cohort['primary']]
    for seed in SEEDS:
        field_rows=[]
        for row in cohort['dev']:
            with np.load(field_path(seed)/(row['id']+'.npz')) as z:
                geo=dict(zip(ARMS,z['orientation'][:3]))
                m=pair(geo,z['support'].astype(bool),z['gt'][3],field=True)
            field_rows.append(dict(id=row['id'],**m))
        readable=[r for r in field_rows if r['r90_readable']]
        field90=float(np.mean([r['r90_success'] for r in readable]))
        images=records('stage_survival',seed,primary,'S9')
        image90=float(np.mean([r['r90_success'] for r in images]))
        original_field=read(TM/('field_precheck/seed%d/summary.json'%seed))['real_r90']
        original_image=read(TM/('rf2_seed%d/summary.json'%seed))['statistics']['r90_success']['mean']
        diffs=dict(field=abs(field90-original_field),image=abs(image90-original_image))
        counts={arm:0 for arm in ARMS};noise_equal=True
        for sid in primary:
            dest=case_path('stage_survival',seed,sid)
            case=read(dest/'case.json')
            for arm in ARMS:
                old=read(TM/('rf2_seed%d'%seed)/sid/'d42'/(arm+'.json'))
                counts[arm]+=sha(dest/'S9'/(arm+'.png'))==old['output_sha256']
                noise_equal &= case['schedules'][arm]['noise_sha256']==old['noise_sha256']
        record=dict(seed=seed,field_r90=field90,field_readable_N=len(readable),field_total_N=len(field_rows),
            image_r90=image90,image_N=len(images),original_field=original_field,original_image=original_image,
            differences=diffs,pixel_identical_count=counts,paired_noise_reproduced=noise_equal,
            **{'pass':all(v<=.005+1e-12 for v in diffs.values()) and noise_equal})
        results.append(record);write(OUT/'reproduction'/('seed%d'%seed)/'summary.json',record)
    passed=all(r['pass'] for r in results)
    write(OUT/'reproduction/summary.json',dict(seeds=results,**{'pass':passed}))
    if not passed: write(OUT/'decision_summary.json',dict(reproduction_pass=False,next_route='reproduction_mismatch'))
    assert passed,'复现容差失败，停止定位'
    return results

def localize(cohort,splits):
    primary=[r['id'] for r in cohort['primary']];diag=splits['diagnostic64']
    full=stage_summary(42,primary,['S0','S1','S2','S4','S9'])
    diagnostic=stage_summary(42,diag,STAGES)
    adjacent={}
    for a,b in zip(STAGES,STAGES[1:]):
        c=compare(records('stage_survival',42,diag,a),records('stage_survival',42,diag,b))
        adjacent[a+'->'+b]=c
    primary_pre={a+'->'+b:compare(records('stage_survival',42,primary,a),records('stage_survival',42,primary,b))
                 for a,b in [('S0','S1'),('S1','S2'),('S2','S4')]}
    candidates=[(c['r90_success']['mean'],edge) for edge,c in adjacent.items() if c['significant_drop']]
    chosen=max(candidates)[1] if candidates else 'cumulative_small_losses'
    field_edges=['S0->S1','S1->S2']
    output=dict(primary_bottleneck_stage=chosen,stage_pair=chosen.split('->') if candidates else None,
        selection_scope='same diagnostic64 at every stage; pre-diffusion contrasts separately verified on all254',
        full254=full,diagnostic64=diagnostic,adjacent_diagnostic64=adjacent,adjacent_full254=primary_pre,
        field_to_rgb_bottleneck=any(primary_pre[e]['significant_drop'] for e in field_edges),
        field_to_rgb_readability_loss=any(primary_pre[e]['significant_drop'] and primary_pre[e]['readability_loss'] for e in field_edges),
        vae_orientation_bottleneck=primary_pre['S2->S4']['significant_drop'],
        initialization_or_first_denoise_bottleneck=adjacent['S4->x0_01']['significant_drop'],
        first_to_last=compare(records('stage_survival',42,diag,'x0_01'),records('stage_survival',42,diag,'x0_08')),
        readability_loss_stage=[edge for edge,c in adjacent.items() if c['significant_drop'] and c['readability_loss']])
    write(OUT/'stage_survival/localization.json',output)
    return output

def conditions(splits):
    ids=splits['diagnostic64']
    rows={name:records('condition_factorization/'+name,42,ids,'S9') for name in CONDITIONS}
    groups={name:summarize(r) for name,r in rows.items()}
    comparisons={a+'->'+b:compare(rows[a],rows[b],.15) for a,b in
        [('C0','C1'),('C0','C2'),('C0','C3'),('C4','C7'),('C1','C4'),('C3','C6'),('C5','C7'),('C2','C4'),('C3','C5'),('C6','C7')]}
    # 所有组的初始carrier、VAE mean、噪声与initial latent必须完全相同。
    for sid in ids:
        cases=[read(case_path('condition_factorization/'+n,42,sid)/'case.json') for n in CONDITIONS]
        for arm in ARMS:
            for key in ('noise_sha256','carrier_sha256'):
                assert len({c['schedules'][arm][key] for c in cases})==1
            for key in ('posterior_mean','initial_latent'):
                assert len({c['schedules'][arm][key]['sha256'] for c in cases})==1
    full_original=records('stage_survival',42,ids,'S9')
    assert all(a['r90_success']==b['r90_success'] and a['r90_error']==b['r90_error']
               for a,b in zip(full_original,rows['C7']))
    write(OUT/'condition_factorization/summary.json',dict(groups=groups,comparisons=comparisons,
        carrier_noise_initial_latent_equal=True,full_reproduced=True,
        text_competition=comparisons['C0->C1']['significant_drop'],
        sketch_competition=comparisons['C0->C2']['significant_drop'],
        appearance_texture_competition=comparisons['C0->C3']['significant_drop'] or comparisons['C4->C7']['significant_drop']))

def candidate_checks(rows,baseline):
    a,b=summarize(rows)['statistics'],summarize(baseline)['statistics']
    return dict(r90=a['r90_success']['mean']>=b['r90_success']['mean']+.15-1e-12,
        contour=a['contour_f1']['mean']>=b['contour_f1']['mean']-.05-1e-12,
        text=a['text_score']['mean']>=b['text_score']['mean']*.95-1e-12)

def strengths(splits,confirmation=False):
    if confirmation:
        candidate=read(OUT/'strength_grid/candidate.json')
        values=sorted({.15,candidate['strength']});ids=splits['confirmation64'];group='strength_confirmation'
    else: values=STRENGTHS;ids=splits['diagnostic64'];group='strength_grid'
    rows={str(s):records(group+'/%.2f'%s,42,ids,'S9') for s in values}
    baseline=rows[str(.15)]
    groups={name:summarize(value) for name,value in rows.items()}
    checks={name:candidate_checks(value,baseline) for name,value in rows.items()}
    actual={str(s):read(case_path(group+'/%.2f'%s,42,ids[0])/'case.json')['schedules']['R0']['start'] for s in values}
    result=dict(groups=groups,checks=checks,actual_sampling=actual,
        comparisons={name:compare(baseline,value,.15) for name,value in rows.items()})
    if confirmation:
        result['selected_strength']=candidate['strength'];result['pass']=all(checks[str(candidate['strength'])].values())
    else:
        available=[s for s in values if all(checks[str(s)].values())]
        strength=sorted(available,key=lambda s:(-groups[str(s)]['statistics']['r90_success']['mean'],
                         -groups[str(s)]['statistics']['text_score']['mean'],s))[0] if available else None
        frozen=dict(strength=strength,diagnostic_ids_sha256=sha(OUT/'splits/diagnostic64.json'),
            source_commit=commit(),selection='highest R90,then TextScore,then lowest strength; before any confirmation generation')
        path=OUT/'strength_grid/candidate.json'
        if path.exists(): assert read(path)['strength']==strength
        else: write(path,frozen)
        result['candidate']=frozen
    write(OUT/group/'summary.json',result)

def purity(splits):
    ids=splits['diagnostic64'];groups={};contrasts={}
    for kind in ('P0','P1','P2','P3'):
        groups[kind]=stage_summary(42,ids,STAGES,group='carrier_purity/'+kind)
    for kind in ('P1','P2','P3'):
        contrasts[kind]={stage:compare(records('carrier_purity/'+kind,42,ids,stage),
            records('carrier_purity/P0',42,ids,stage),.15) for stage in ('S2','S4','S9')}
    # compare的before-after在这里表示纯化载体相对P0的增益。
    contamination=all(contrasts[k]['S2']['significant_drop'] for k in ('P1','P2','P3'))
    write(OUT/'carrier_purity/summary.json',dict(groups=groups,improvement_over_P0=contrasts,
        carrier_content_contamination=contamination,
        qualifier='P3仅定位；masked/normalized carrier不是本轮新方法；无需改变原主Gate'))

def seed_confirmation(cohort,splits):
    loc=read(OUT/'stage_survival/localization.json');stage_pair=loc['stage_pair']
    reports=[]
    for seed in (43,44):
        if stage_pair and all(s in ('S0','S1','S2','S4') for s in stage_pair):
            ids=[r['id'] for r in cohort['primary']];group='stage_survival'
        else: ids=splits['diagnostic64'];group='seed_confirmation'
        if stage_pair:
            contrast=compare(records(group,seed,ids,stage_pair[0]),records(group,seed,ids,stage_pair[1]))
            passed=contrast['significant_drop']
        else:
            # 累积小损失必须重现：无单段20pp且S0->S9累计>=20pp。
            contrasts={a+'->'+b:compare(records(group,seed,ids,a),records(group,seed,ids,b)) for a,b in zip(STAGES,STAGES[1:])}
            contrast=compare(records(group,seed,ids,'S0'),records(group,seed,ids,'S9'))
            passed=not any(c['significant_drop'] for c in contrasts.values()) and contrast['significant_drop']
        reports.append(dict(seed=seed,stage_pair=stage_pair,comparison_N=len(ids),contrast=contrast,**{'pass':passed}))
    write(OUT/'stage_survival/seed_confirmation.json',reports)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('reproduce','localize','conditions','strengths','confirmation','purity','seed_confirmation'))
    args=parser.parse_args();cohort,splits=prepare()
    if args.action in ('localize','seed_confirmation'): globals()[args.action](cohort,splits)
    elif args.action=='reproduce': reproduce(cohort)
    elif args.action=='confirmation': strengths(splits,True)
    else: globals()[args.action](splits)
    freeze_check();print('[IF aggregate]',args.action,'complete',flush=True)

if __name__=='__main__': main()
