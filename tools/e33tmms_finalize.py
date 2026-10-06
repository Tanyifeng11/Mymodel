"""只在序贯搜索到达终点后归档；执行完成与科学成功分别记录。"""
from tools.e33tmms_protocol import *
from tools.e33tmms_benchmark import folder,bundle

def finish():
    prepare();d=read(OUT/'decision_summary.json');selected=d['selected_method']
    success=selected in ('M1','M2','M3')
    if success:
        assert d[selected+'_e5_pass'] and d[selected+'_confirmation_pass']
        assert d['next_route']==selected+'_full_validation'
    else:
        assert d['M3_run'] and d['M3_e5_pass'] is False
        assert d['next_route']=='representation_search_failed'
    frozen_check();assert read(OUT/'frozen_check.json')['pass']
    table=read(OUT/'result_table.json')
    table['reproduced_baselines']={name:read(folder(name,'diagnostic64')/'summary.json')
        for name in ('C0','C1','C2')}
    controlled=OUT/'M2/controlled/dev128/summary.json'
    if controlled.exists():table['M2_controlled_dev128']=dict(summary=read(controlled),
        gate=read(OUT/'M2/controlled/gate.json'),space='RGB',training_steps=4000)
    write(OUT/'result_table.json',table)
    skipped=[]
    if not d['M1_carrier_pass']:skipped+=['M1 confirmation64','M1 E5 mini']
    elif not d['M1_confirmation_pass']:skipped+=['M1 E5 mini']
    if not d['M2_run']:skipped+=['M2 training and evaluation']
    elif not d['M2_controlled_pass']:skipped+=['M2 real mixed training','M2 real diagnostic64','M2 confirmation64','M2 E5 mini']
    elif not d['M2_real_carrier_pass']:skipped+=['M2 confirmation64','M2 E5 mini']
    elif not d.get('M2_confirmation_pass'):skipped+=['M2 E5 mini']
    if not d['M3_run']:skipped+=['M3 training and evaluation']
    elif not d['M3_feature_pass']:skipped+=['M3 confirmation64']
    completion=dict(experiment_execution_complete=True,scientific_success=success,
        selected_method=selected,next_route=d['next_route'],skipped_due_to_prespecified_gates=skipped,
        original_split_reused=True,seed42_only=True,identity_bootstrap_draws=2000,
        full254='separate future validation stage; method search stops at first confirmation success',
        frozen_previous_experiments=True,git_commit=commit())
    write(OUT/'completion_check.json',completion)
    # 包不纳入manifest，避免打包文件与清单之间相互引用；服务器保留全部原始产物。
    files={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*'))
        if p.is_file() and p.name!='artifact_manifest.json' and not p.name.endswith('_review.tar.gz')
        and not (p.name.startswith('job_') and p.suffix in ('.log','.err'))}
    write(OUT/'artifact_manifest.json',dict(files=files,file_count=len(files),git_commit=commit(),
        excluded='review archives and appendable Slurm stdout/stderr; raw images/features/checkpoints included; logs retained on server'))
    bundle('final');print('[MS completion]',completion,flush=True)

if __name__=='__main__':finish()
