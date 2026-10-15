"""一次提交有阶段门禁的作业图；以 Slurm 依赖控制顺序，不在登录节点跑模型。"""
import argparse
import datetime
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
    output=subprocess.check_output(command,text=True).strip();job=int(output.split(';')[0])
    record=dict(name=name,id=job,gpu=gpu,cpus=2 if gpu else 8,minutes=minutes,depends=depends or [],
        command=command,submitted=datetime.datetime.now(datetime.timezone.utc).isoformat(),commit=commit())
    path=OUT/'resources/jobs.json';jobs=read(path) if path.exists() else []
    jobs.append(record);write(path,jobs);print('E52 SUBMIT',name,job,flush=True)
    return job

def launch(p0):
    path=OUT/'resources/dispatch_complete.json';assert not path.exists(),'已有作业图，不能重复提交'
    snapshot=resources();write(OUT/'resources/initial_cluster.json',snapshot)
    workers=min(8,max(1,snapshot['free']))
    jobs_path=OUT/'resources/jobs.json'
    if not jobs_path.exists():write(jobs_path,[dict(name='P0',id=p0,gpu=True,cpus=2,minutes=40,depends=[],
        command=['sbatch','submit/e52_gpu.sh','audit'],submitted='previously submitted through portal',commit='aa44b26')])
    data=submit('DATA','data',False,minutes=20)
    provenance=submit('TOKEN','provenance',depends=[p0,data],minutes=20)
    probe=submit('PROBE','generate',depends=[provenance],minutes=30,extra=['--stage','probe','--arm','all'])
    pilot=[submit('PILOT_%d'%i,'generate',depends=[provenance],minutes=100,
        extra=['--stage','pilot','--arm','all','--shard',str(i),'--shards',str(workers)]) for i in range(workers)]
    pilot_eval=submit('PILOT_EVAL','evaluate',depends=pilot,minutes=25,extra=['--stage','pilot'])
    frozen=submit('FREEZE','freeze',False,depends=[pilot_eval,probe],minutes=15)
    locked=[submit('LOCK_%d'%i,'generate',depends=[frozen],minutes=120,
        extra=['--stage','lock','--arm','all','--shard',str(i),'--shards',str(workers)]) for i in range(workers)]
    locked_eval=submit('LOCK_EVAL','evaluate',depends=locked,minutes=30,extra=['--stage','lock'])
    natural=[submit('N96_%d'%i,'generate',depends=[locked_eval],minutes=120,
        extra=['--stage','natural','--arm','all','--shard',str(i),'--shards',str(workers)]) for i in range(workers)]
    natural_eval=submit('N96_EVAL','evaluate',depends=natural,minutes=30,extra=['--stage','natural'])
    final=submit('FINAL','finish',False,depends=[natural_eval],minutes=20)
    write(path,dict(submitted=True,workers=workers,final_job=final,commit=commit(),
        scientific_training_updates=0,dependency_failure='downstream not_run until failed stage repaired; never infer science failure'))

def status():
    jobs=read(OUT/'resources/jobs.json') if (OUT/'resources/jobs.json').exists() else []
    ids=','.join(str(x['id']) for x in jobs) or '117877'
    print(subprocess.check_output(['sacct','-X','-j',ids,'--format=JobID,JobName%20,State,ExitCode,Elapsed,AllocTRES%45','-P'],text=True))
    for name in ['decision.json','audit/p0_complete.json','audit/token_provenance.json','resources/dispatch_complete.json']:
        p=OUT/name
        if p.exists():
            value=read(p);print(name,{k:v for k,v in value.items() if k in ['P0','P1','P2','P3','P4','P5','H_G','H_C','H_D','MODEL','passed','oracle_valid','valid_fraction','workers','final_job']})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['launch','status']);parser.add_argument('--p0',type=int,default=117877)
    args=parser.parse_args()
    if args.action=='launch':launch(args.p0)
    else:status()
