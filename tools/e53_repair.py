"""只复现工程失败条件；保留旧作业、旧日志和未完成张量，不按科学效果选样。"""
import os
import subprocess
from datetime import datetime,timezone
import numpy as np
from tools.e53_protocol import *


def energy_repair_gate():
    from tools.e53_runtime import runtime,case,clear,release
    from tools.e53_run import baseline_energy,targets
    prepare();jobs=read(OUT/'resources/jobs.json');old={x['name']:x for x in jobs}
    failed={name:old[name]['id'] for name in ['PILOT0','PILOT1','PILOT2','PILOT3']}
    assert all('ENERGY_MATCH_FAIL' in (OUT/'resources'/('gpu_%d.err'%job)).read_text() for job in failed.values())
    rows=read(OUT/'splits/pilot.json');checks=[];r=runtime('energy_repair_R1/runtime')
    try:
        for shard in range(4):
            pending=[(row,arm) for i,row in enumerate(rows) if i%4==shard for arm in PILOT_ARMS
                     if not image_path('pilot',arm,row['id']).with_name(row['id']+'_trace.json').exists()]
            assert pending,'FAILED_SHARD_WITH_NO_UNFINISHED_CONDITION'
            row,arm=pending[0];assert arm!='E5'
            clear(r);base=baseline_energy(r,row,'pilot');spec=ARMS[arm]
            value=case(r,row,arm,'pilot',spec,targets(base,spec['window']),full=True)
            total_error=abs(value['total_injection_energy']/float(np.dot(base,base))-1)
            fallback=[dict(step=s['index'],**s['energy']) for s in value['steps'] if s['energy'].get('method')=='bisection_after_rescale']
            assert fallback and total_error<1e-4 and all(s['energy']['relative_error']<1e-4 for s in value['steps'])
            checks.append(dict(shard=shard,original_job=failed['PILOT%d'%shard],id=row['id'],arm=arm,
                total_relative_error=total_error,max_step_error=max(s['energy']['relative_error'] for s in value['steps']),
                reproduced_legacy_failures=fallback,selection='first unfinished fixed-order condition in engineering-failed shard'))
    finally:release(r)
    write(OUT/'audit/energy_repair_R1.json',dict(passed=True,checks=checks,failed_jobs=failed,
        repair_job=int(os.environ['SLURM_JOB_ID']),commit=commit(),utc=datetime.now(timezone.utc).isoformat(),
        scope='same global scalar energy match, exact same 1e-4 threshold; no scientific identities, arms, or budgets changed'))
    decision(engineering_energy_repair='PASS_R1');print('E53 ENERGY REPAIR R1 PASS',len(checks),flush=True)


def storage_repair_gate():
    from tools.e53_runtime import runtime,case,clear,release
    from tools.e53_run import baseline_energy,targets
    from tools.e53_storage import verify_case,load_raw
    assert read(OUT/'audit/storage_compression_R1.json')['passed']
    plan=read(OUT/'resources/storage_repair_plan_R1.json');rows=read(OUT/'splits/pilot.json');checks=[];r=runtime('storage_repair_R1/runtime')
    try:
        for name,identity in sorted(plan['failed_jobs'].items()):
            shard=int(name[5:])
            pending=[(row,arm) for i,row in enumerate(rows) if i%4==shard for arm in PILOT_ARMS
                     if not image_path('pilot',arm,row['id']).with_name(row['id']+'_trace.json').exists()]
            row,arm=pending[0];assert arm!='E5'
            clear(r);base=baseline_energy(r,row,'pilot');spec=ARMS[arm]
            value=case(r,row,arm,'pilot',spec,targets(base,spec['window']),full=True)
            error=abs(value['total_injection_energy']/float(np.dot(base,base))-1)
            assert error<1e-4 and all(s['energy']['relative_error']<1e-4 for s in value['steps'])
            raw=verify_case('pilot',arm,row['id'],True)
            for step in [0,49]:
                frame=load_raw(OUT/'traces/pilot'/arm/row['id']/('%02d.pt'%step))
                assert frame['identity']==row['id'] and frame['step']==step
                assert all(k in frame for k in ['latent_before','latent_after','A_ref_normalized','A_used_normalized','V_ref','delta_after_energy_match'])
            checks.append(dict(original_job=identity,shard=shard,id=row['id'],arm=arm,total_relative_error=error,
                max_step_error=max(s['energy']['relative_error'] for s in value['steps']),raw=raw,
                selection='first unfinished fixed-order condition in storage-failed shard'))
    finally:release(r)
    write(OUT/'audit/storage_repair_R1.json',dict(passed=True,checks=checks,failed_jobs=plan['failed_jobs'],
        repair_job=int(os.environ['SLURM_JOB_ID']),commit=commit(),utc=datetime.now(timezone.utc).isoformat(),
        scope='same full50 raw tensors stored losslessly; no dtype, value, scientific arms, identities or thresholds changed'))
    decision(engineering_storage_repair='PASS_R1');print('E53 STORAGE GPU RECHECK R1 PASS',len(checks),flush=True)


def retry_pilot():
    from tools.e53_dispatch import states,submit
    storage=(OUT/'audit/storage_repair_R1.json').exists()
    path=OUT/'audit'/('storage_repair_R1.json' if storage else 'energy_repair_R1.json')
    if not path.exists():return False
    proof=read(path);latest,st=states()
    if not proof['passed'] or st.get(proof['repair_job'])!='COMPLETED':return False
    allowed=set(proof['failed_jobs'].values())
    failed=[j for j in latest.values() if st.get(j['id']) in ['FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL']]
    if not failed or any(j['id'] not in allowed for j in failed):return False
    replacements={}
    for name,old_id in sorted(proof['failed_jobs'].items()):
        shard=int(name[5:])
        identity=submit(name+('_R2' if storage else '_R1'),'generate',True,extra=['--stage','pilot','--shard',str(shard),'--shards','4'],
                        logical_name=name,supersedes=old_id);replacements[name]=identity
    ids=[replacements.get('PILOT%d'%shard,latest['PILOT%d'%shard]['id']) for shard in range(4)]
    evaluator=latest['PILOT_EVAL']['id']
    cmd=['scontrol','update','JobId=%d'%evaluator,'Dependency=afterok:'+':'.join(map(str,ids))]
    result=subprocess.run(cmd,text=True,capture_output=True);assert result.returncode==0,result.stderr
    jobs=read(OUT/'resources/jobs.json')
    for job in jobs:
        if job['id']==evaluator:
            job['original_dependencies']=job.get('original_dependencies',job['dependencies']);job['dependencies']=ids
    write(OUT/'resources/jobs.json',jobs)
    write(OUT/'resources'/('storage_repair_R1_resubmission.json' if storage else 'energy_repair_R1_resubmission.json'),dict(original_failed_jobs=proof['failed_jobs'],replacement_jobs=ids,
        evaluator=evaluator,dependency_command=cmd,stdout=result.stdout,stderr=result.stderr,repair_gate_sha256=sha(path),commit=commit(),utc=datetime.now(timezone.utc).isoformat()))
    print('E53 PILOT RETRY',('R2' if storage else 'R1'),'SUBMITTED',ids,'EVAL_DEPENDENCY_REPAIRED',evaluator,flush=True)
    return True
