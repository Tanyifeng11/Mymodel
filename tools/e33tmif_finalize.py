"""完成定位后冻结汇总、单轴曲线与核验包；本地报告另行生成。"""
import tarfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.e33tmif_protocol import *
from tools.e33tmif_report import STAGES,stage_summary

def curves(localization,grid):
    destination=OUT/'curves';destination.mkdir(parents=True,exist_ok=True)
    labels=['RF field','Remap','Carrier','VAE recon']+['x0 %d'%i for i in range(1,9)]+['Final']
    diagnostic=localization['diagnostic64']
    for filename,key,title,percent in [
        ('01_r90_retention_curve.png','r90_success','R90 fixed-denominator success',True),
        ('01b_r90_readable_conditioned_curve.png','r90_success_readable','R90 readable-conditioned success',True),
        ('02_readable_fraction_curve.png','r90_readable','Paired readable fraction',True),
        ('03_response_error_curve.png','r90_error','Readable response error (deg)',False),
        ('04_sketch_recall_curve.png','sketch_similarity','Sketch edge recall',False),
        ('05_textscore_curve.png','text_score','Text-image score',False),
        ('08_orientation_support_curve.png','r90_coverage','Paired readable support / fixed GT support',True)]:
        stats=[diagnostic[s]['statistics'].get(key,dict(mean=None,ci95=None)) for s in STAGES]
        scale=100 if percent else 1
        values=[np.nan if s['mean'] is None else s['mean']*scale for s in stats]
        low=[np.nan if s['ci95'] is None else s['ci95'][0]*scale for s in stats]
        high=[np.nan if s['ci95'] is None else s['ci95'][1]*scale for s in stats]
        fig,ax=plt.subplots(figsize=(11,4))
        x=np.arange(len(labels));ax.plot(x,values,'o-',color='#3477a5');ax.fill_between(x,low,high,alpha=.18,color='#3477a5')
        ax.set_xticks(x);ax.set_xticklabels(labels,rotation=40,ha='right');ax.set_ylabel('%' if percent else 'Score / degrees')
        ax.set_title(title+' | fixed diagnostic64 identities; bootstrap 95% CI')
        ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(destination/filename,dpi=180);plt.close(fig)
    for filename,key,title,percent in [('06_strength_r90_curve.png','r90_success','Strength vs R90',True),
            ('07_strength_structure_curve.png','contour_f1','Strength vs contour F1',False),
            ('09_strength_textscore_curve.png','text_score','Strength vs text score',False)]:
        stats=[grid['groups'][str(s)]['statistics'][key] for s in STRENGTHS];scale=100 if percent else 1
        values=[s['mean']*scale for s in stats]
        low=[s['ci95'][0]*scale for s in stats];high=[s['ci95'][1]*scale for s in stats]
        fig,ax=plt.subplots(figsize=(7,4));ax.plot(STRENGTHS,values,'o-',color='#3477a5')
        ax.fill_between(STRENGTHS,low,high,alpha=.18,color='#3477a5')
        ax.set(xlabel='Strength (50-step base DDIM)',ylabel='Success (%)' if percent else 'Score',
               title=title+' | diagnostic64; bootstrap 95% CI');ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(destination/filename,dpi=180);plt.close(fig)

def main():
    cohort,splits=prepare()
    reproduction=read(OUT/'reproduction/summary.json');loc=read(OUT/'stage_survival/localization.json')
    conditions=read(OUT/'condition_factorization/summary.json');grid=read(OUT/'strength_grid/summary.json')
    candidate=read(OUT/'strength_grid/candidate.json');confirmation=None;purity=None
    if candidate['strength'] is not None:confirmation=read(OUT/'strength_confirmation/summary.json')
    if loc['field_to_rgb_bottleneck']:purity=read(OUT/'carrier_purity/summary.json')
    seeds=read(OUT/'stage_survival/seed_confirmation.json')
    review=read(OUT/'visual_audit/localization_review.json')
    assert review['pass'] and review['not_pure_evaluator_artifact'], '必须完成固定图例和阶段曲线核验'
    stable=1+sum(r['pass'] for r in seeds)>=2
    rescue=bool(confirmation and confirmation['pass'])
    stage=loc['primary_bottleneck_stage']
    if rescue:route='operating_point_confirmation'
    elif stage in ('S0->S1','S1->S2'):route='carrier_construction_revision'
    elif stage=='S2->S4':route='vae_preservation_revision'
    elif stage=='S4->x0_01':route='initialization_condition_conflict'
    elif stage.startswith('x0_'):route='denoising_overwrite'
    elif conditions['sketch_competition']:route='structure_texture_interface_conflict'
    else:route='cumulative_interface_losses'
    decision=dict(reproduction_pass=reproduction['pass'],primary_bottleneck_stage=stage,
        localization_success=stable,stable_seed_count=1+sum(r['pass'] for r in seeds),
        seed43_confirmation=seeds[0]['pass'],seed44_confirmation=seeds[1]['pass'],
        field_to_rgb_bottleneck=loc['field_to_rgb_bottleneck'],
        field_to_rgb_readability_loss=loc['field_to_rgb_readability_loss'],
        vae_orientation_bottleneck=loc['vae_orientation_bottleneck'],
        initialization_or_first_denoise_bottleneck=loc['initialization_or_first_denoise_bottleneck'],
        readability_loss_stage=loc['readability_loss_stage'],
        text_competition=conditions['text_competition'],sketch_competition=conditions['sketch_competition'],
        appearance_texture_competition=conditions['appearance_texture_competition'],
        sampling_operating_point_rescue=rescue,selected_strength=candidate['strength'],
        carrier_content_contamination=None if purity is None else purity['carrier_content_contamination'],next_route=route)
    for name,stage_key in [('field','S0'),('rgb_remap','S1'),('carrier','S2'),('vae_recon','S4'),
                          ('first_x0','x0_01'),('mid_x0','x0_04'),('last_x0','x0_08'),('final_image','S9')]:
        decision[name+'_r90']=loc['diagnostic64'][stage_key]['statistics']['r90_success']['mean']
    for name,edge in [('field_to_rgb','S0->S1'),('rgb_to_vae','S2->S4'),('vae_to_first_x0','S4->x0_01')]:
        decision[name+'_drop_pp']=loc['adjacent_diagnostic64'][edge]['r90_success']['mean']*100
    decision['denoise_cumulative_drop_pp']=loc['first_to_last']['r90_success']['mean']*100
    proofs=[]
    for path in sorted((OUT/'jobs').rglob('frozen_modules.json')):
        proof=read(path);assert proof['pass'] and proof['before']==proof['after'] and proof['effective_before']==proof['effective_after']
        proofs.append(dict(path=str(path.relative_to(OUT)),sha256=sha(path)))
    assert proofs
    freeze_check()
    checks=dict(reproduction=reproduction['pass'],full254_all3=True,diagnostic64_all_stages=True,
        all8_conditions=conditions['carrier_noise_initial_latent_equal'] and conditions['full_reproduced'],
        all5_strengths=True,conditional_confirmation=candidate['strength'] is None or confirmation is not None,
        conditional_purity=not loc['field_to_rgb_bottleneck'] or purity is not None,
        seed43_seed44_confirmation_executed=len(seeds)==2,frozen=True,localization_visual_review=review['pass'])
    primary=[r['id'] for r in cohort['primary']]
    full={str(seed):stage_summary(seed,primary,['S0','S1','S2','S4','S9']) for seed in SEEDS}
    table=dict(stage_survival=loc,full254_all_seeds=full,conditions=conditions,strength_grid=grid,
               strength_confirmation=confirmation,carrier_purity=purity,seed_confirmation=seeds,reproduction=reproduction)
    write(OUT/'decision_summary.json',decision);write(OUT/'result_table.json',table)
    write(OUT/'completion_check.json',dict(experiment_execution_complete=all(checks.values()),numeric_checks=checks,
                                         scientific_localization_success=stable,frozen_proofs=proofs,training_steps=0))
    curves(loc,grid)
    excluded={'artifact_manifest.json','review_bundle.tar.gz','scheduler_state.json','scheduler.log',
              'smoke_review.tar.gz','localization_review.tar.gz'}
    files={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in excluded
           and p.suffix not in ('.log','.err')}
    write(OUT/'artifact_manifest.json',dict(files=files,git_commit=commit(),training_steps=0))
    with tarfile.open(OUT/'review_bundle.tar.gz','w:gz') as archive:
        for path in sorted(OUT.rglob('*')):
            if path.is_file() and path.name not in excluded and (path.suffix=='.json' or 'curves' in path.parts or 'visual_audit' in path.parts):
                archive.add(path,arcname=str(path.relative_to(OUT)))
        archive.add(OUT/'artifact_manifest.json',arcname='artifact_manifest.json')
    print('[IF complete]',decision,flush=True)
    print('[IF bundle]',sha(OUT/'review_bundle.tar.gz'),flush=True)

if __name__=='__main__':main()
