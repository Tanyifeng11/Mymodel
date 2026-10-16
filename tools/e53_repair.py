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


def retry_pilot():
    from tools.e53_dispatch import states,submit
    path=OUT/'audit/energy_repair_R1.json'
    if not path.exists():return False
    proof=read(path);latest,st=states()
    if not proof['passed'] or st.get(proof['repair_job'])!='COMPLETED':return False
    allowed=set(proof['failed_jobs'].values())
    failed=[j for j in latest.values() if st.get(j['id']) in ['FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL']]
    if not failed or any(j['id'] not in allowed for j in failed):return False
    ids=[]
    for shard in range(4):
        name='PILOT%d'%shard
        identity=submit(name+'_R1','generate',True,extra=['--stage','pilot','--shard',str(shard),'--shards','4'],
                        logical_name=name,supersedes=proof['failed_jobs'][name]);ids.append(identity)
    evaluator=latest['PILOT_EVAL']['id']
    cmd=['scontrol','update','JobId=%d'%evaluator,'Dependency=afterok:'+':'.join(map(str,ids))]
    result=subprocess.run(cmd,text=True,capture_output=True);assert result.returncode==0,result.stderr
    jobs=read(OUT/'resources/jobs.json')
    for job in jobs:
        if job['id']==evaluator:
            job['original_dependencies']=job.get('original_dependencies',job['dependencies']);job['dependencies']=ids
    write(OUT/'resources/jobs.json',jobs)
    write(OUT/'resources/energy_repair_R1_resubmission.json',dict(original_failed_jobs=proof['failed_jobs'],replacement_jobs=ids,
        evaluator=evaluator,dependency_command=cmd,stdout=result.stdout,stderr=result.stderr,repair_gate_sha256=sha(path),commit=commit(),utc=datetime.now(timezone.utc).isoformat()))
    print('E53 PILOT RETRY R1 SUBMITTED',ids,'EVAL_DEPENDENCY_REPAIRED',evaluator,flush=True)
    return True
