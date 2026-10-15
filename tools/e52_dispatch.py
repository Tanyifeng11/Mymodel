"""一次提交有阶段门禁的作业图；以 Slurm 依赖控制顺序，不在登录节点跑模型。"""
import argparse
import datetime
import os
import re
import subprocess
from tools.e52_protocol import *

def resources():
    value=subprocess.check_output(['scontrol','show','nodes','-o'],text=True)
    total=allocated=0
    for line in value.splitlines():
        fields=dict(item.split('=',1) for item in line.split() if '=' in item)
        a=re.search(r'(?:^|,)gres/gpu=(\d+)',fields.get('CfgTRES',''))
        b=re.search(r'(?:^|,)gres/gpu=(\d+)',fields.get('AllocTRES',''))
        if a:total+=int(a.group(1));allocated+=int(b.group(1)) if b else 0
    return dict(total=total,allocated=allocated,free=max(0,total-allocated),
        raw=value,at=datetime.datetime.now(datetime.timezone.utc).isoformat())

def submit(name,action,gpu=True,depends=None,minutes=120,extra=None):
    command=['sbatch','--parsable','--job-name=E52_'+name,'--time=%02d:%02d:00'%(minutes//60,minutes%60)]
    if depends:command+=['--dependency=afterok:'+':'.join(str(x) for x in depends)]
    command+=['submit/e52_gpu.sh' if gpu else 'submit/e52_cpu.sh',action]+(extra or [])
    response=subprocess.run(command,text=True,capture_output=True)
    if response.returncode:
        path=OUT/'resources/submission_failures.json';failures=read(path) if path.exists() else []
        failures.append(dict(name=name,command=command,error=response.stderr,at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        write(path,failures);raise RuntimeError(response.stderr.strip())
    output=response.stdout.strip();job=int(output.split(';')[0])
    record=dict(name=name,id=job,gpu=gpu,cpus=2 if gpu else 8,minutes=minutes,depends=depends or [],
        command=command,submitted=datetime.datetime.now(datetime.timezone.utc).isoformat(),commit=commit())
    path=OUT/'resources/jobs.json';jobs=read(path) if path.exists() else []
    jobs.append(record);write(path,jobs);print('E52 SUBMIT',name,job,flush=True)
    return job

def launch(p0):
    snapshot=resources()
    if not (OUT/'resources/initial_cluster.json').exists():write(OUT/'resources/initial_cluster.json',snapshot)
    jobs_path=OUT/'resources/jobs.json'
    if not jobs_path.exists():write(jobs_path,[dict(name='P0',id=p0,gpu=True,cpus=2,minutes=40,depends=[],
        command=['sbatch','submit/e52_gpu.sh','audit'],submitted='previously submitted through portal',commit='aa44b26')])
    plan_path=OUT/'resources/dispatch_plan.json'
    if not plan_path.exists():
        prior=read(jobs_path);pilot=next((r for r in prior if r['name'].startswith('PILOT_') and '--shards' in r['command']),None)
        workers=int(pilot['command'][pilot['command'].index('--shards')+1]) if pilot else min(8,snapshot['free']) if snapshot['free'] else 4
        write(plan_path,dict(workers=workers,max_active_jobs=5,max_gpu_active_jobs=4,incremental=True,
            reason='submit only ready stages, preserve accepted jobs, respect cluster submit limits',commit=commit()))
    advance()

def advance():
    path=OUT/'resources/dispatch_plan.json'
    if not path.exists():return
    plan=read(path);workers=plan['workers'];jobs=read(OUT/'resources/jobs.json')
    lookup={r['name']:r for r in jobs};ids=','.join(str(r['id']) for r in jobs)
    output=subprocess.check_output(['sacct','-X','-j',ids,'--format=JobIDRaw,State','-P','-n'],text=True)
    states={int(line.split('|')[0]):line.split('|')[1] for line in output.splitlines() if line.split('|')[0].isdigit()}
    queue=subprocess.check_output(['squeue','-h','-u','u2515283058','-o','%P'],text=True)
    slots=max(0,plan['max_active_jobs']-len(queue.splitlines()))
    gpu_slots=max(0,plan.get('max_gpu_active_jobs',4)-sum(line.strip()=='gpu' for line in queue.splitlines()))
    current_job=os.environ.get('SLURM_JOB_ID')
    def done(name,marker=None):
        if name not in lookup:return False
        marker_ready=marker is None or (OUT/marker).exists()
        # postamble 此时父作业仍是 RUNNING；产物完成后可提交 afterok 子作业，避免额外等一次心跳。
        current_ready=marker is not None and str(lookup[name]['id'])==current_job and marker_ready
        return marker_ready and (states.get(lookup[name]['id'])=='COMPLETED' or current_ready)
    def ensure(name,action,gpu=True,parents=None,minutes=120,extra=None):
        nonlocal slots,gpu_slots
        if name in lookup:return lookup[name]['id']
        if slots<=0 or gpu and gpu_slots<=0:return None
        try:identity=submit(name,action,gpu,[lookup[x]['id'] for x in parents or []],minutes,extra)
        except RuntimeError as error:
            print('E52 SUBMIT DEFERRED',name,str(error),flush=True);slots=0;return None
        slots-=1;gpu_slots-=int(gpu);lookup[name]=dict(id=identity,name=name);return identity
    ensure('DATA','data',False,minutes=20)
    if done('P0','audit/p0_complete.json') and done('DATA','splits/natural.json'):
        ensure('TOKEN','provenance',parents=['P0','DATA'],minutes=20)
    if done('TOKEN','audit/token_provenance.json'):
        ensure('PROBE','generate',parents=['TOKEN'],minutes=30,extra=['--stage','probe','--arm','all'])
        for i in range(workers):ensure('PILOT_%d'%i,'generate',parents=['TOKEN'],minutes=40 if workers>=4 else 100,
            extra=['--stage','pilot','--arm','all','--shard',str(i),'--shards',str(workers)])
    pilot=['PILOT_%d'%i for i in range(workers)]
    if all(done(name,'resources/pilot_shard%d_complete.json'%i) for i,name in enumerate(pilot)):
        ensure('PILOT_EVAL','evaluate',parents=pilot,minutes=25,extra=['--stage','pilot'])
    if done('PILOT_EVAL','metrics/pilot/summary.json') and done('PROBE','resources/probe_shard0_complete.json'):
        ensure('FREEZE','freeze',False,parents=['PILOT_EVAL','PROBE'],minutes=15)
    if done('FREEZE','manifest/confirmation.json'):
        for i in range(workers):ensure('LOCK_%d'%i,'generate',parents=['FREEZE'],minutes=60 if workers>=4 else 120,
            extra=['--stage','lock','--arm','all','--shard',str(i),'--shards',str(workers)])
    locked=['LOCK_%d'%i for i in range(workers)]
    if all(done(name,'resources/lock_shard%d_complete.json'%i) for i,name in enumerate(locked)):
        ensure('LOCK_EVAL','evaluate',parents=locked,minutes=30,extra=['--stage','lock'])
    if done('LOCK_EVAL','metrics/lock/summary.json'):
        for i in range(workers):ensure('N96_%d'%i,'generate',parents=['LOCK_EVAL'],minutes=40 if workers>=4 else 120,
            extra=['--stage','natural','--arm','all','--shard',str(i),'--shards',str(workers)])
    natural=['N96_%d'%i for i in range(workers)]
    if all(done(name,'resources/natural_shard%d_complete.json'%i) for i,name in enumerate(natural)):
        ensure('N96_EVAL','evaluate',parents=natural,minutes=30,extra=['--stage','natural'])
    if done('N96_EVAL','metrics/natural/summary.json'):
        ensure('FINAL','finish',False,parents=['N96_EVAL'],minutes=20)
    write(OUT/'resources/dispatch_complete.json',dict(submitted=True,incremental=True,workers=workers,
        final_job=lookup.get('FINAL',{}).get('id'),commit=commit(),slots_remaining=slots,
        scientific_training_updates=0,dependency_failure='repair failed engineering stage before advancing; science failure is never an automatic retry'))

def status():
    jobs=read(OUT/'resources/jobs.json') if (OUT/'resources/jobs.json').exists() else []
    ids=','.join(str(x['id']) for x in jobs) or '117877'
    print(subprocess.check_output(['sacct','-X','-j',ids,'--format=JobID,JobName%20,State,ExitCode,Elapsed,AllocTRES%45','-P'],text=True))
    for name in ['decision.json','audit/p0_complete.json','audit/token_provenance.json','resources/dispatch_complete.json']:
        p=OUT/name
        if p.exists():
            value=read(p);print(name,{k:v for k,v in value.items() if k in ['P0','P1','P2','P3','P4','P5','H_G','H_C','H_D','MODEL','passed','oracle_valid','valid_fraction','workers','final_job']})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['launch','advance','status']);parser.add_argument('--p0',type=int,default=117877)
    args=parser.parse_args()
    if args.action=='launch':launch(args.p0)
    elif args.action=='advance':advance()
    else:status()
