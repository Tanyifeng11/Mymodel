"""提交数受限时分阶段续接；全部GPU预约与真实用量共同约束100小时硬上限。"""
import csv
import os
import re
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.bf_t1_t2 import ROOT,read,write

ROOT.joinpath('logs').mkdir(parents=True,exist_ok=True)
jobs=read(ROOT/'submitted_jobs.json') if (ROOT/'submitted_jobs.json').exists() else {}
reservations=read(ROOT/'job_reservations.json') if (ROOT/'job_reservations.json').exists() else {}


def committed_gpu_seconds():
    if not jobs:return 0
    raw=subprocess.check_output(['sacct','-X','-j',','.join(map(str,jobs.values())),'-P',
        '--format=JobID,State,ElapsedRaw,AllocTRES'],text=True)
    (ROOT/'audit/budget_latest_sacct.psv').write_text(raw)
    actual={r['JobID']:r for r in csv.DictReader(raw.splitlines(),delimiter='|')};total=0
    for key,job in jobs.items():
        reserve=reservations.get(key,dict(gpu=0,wall_seconds=0));record=actual.get(str(job));gpu=reserve['gpu']
        if record:
            allocated=re.search(r'(?:^|,)gres/gpu=([0-9]+)',record['AllocTRES'])
            elapsed=int(record['ElapsedRaw'])*(int(allocated[1]) if allocated else 0)
            terminal=record['State'].split()[0].rstrip('+') in ['COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE']
            total+=elapsed if terminal else max(elapsed,gpu*reserve['wall_seconds'])
        else:total+=gpu*reserve['wall_seconds']
    return total


def submit(key,mode,dependency=(),arm=None,gpu=0,minutes=60):
    if key in jobs:return jobs[key]
    committed=committed_gpu_seconds();new=gpu*minutes*60
    assert committed+new<=360000,('GPU_BUDGET_REFUSED',committed,new)
    args=['sbatch','--parsable','-p','gpu' if gpu else 'cpu','-J','bft12_'+key,'-N1','-n1','-c','8' if gpu==2 else '4',
        '--mem','96G' if gpu==2 else '48G','-t',str(minutes),'-o',str(ROOT/'logs'/(key+'_%j.log'))]
    if gpu:args+=['--gres=gpu:'+str(gpu)]
    if dependency:args+=['--dependency=afterok:'+':'.join(map(str,dependency))]
    args+=[str(ROOT/'source/scripts/bf_t1_t2.sbatch'),mode]+([arm] if arm else [])
    job=subprocess.check_output(args,text=True).strip();assert job.isdigit(),job
    jobs[key]=job;reservations[key]=dict(gpu=gpu,wall_seconds=minutes*60,mode=mode,arm=arm)
    write(ROOT/'submitted_jobs.json',jobs);write(ROOT/'job_reservations.json',reservations)
    write(ROOT/'audit/budget_at_submit.json',dict(committed_before=committed,new_reservation=new,limit=360000,jobs=jobs))
    print(key,job,flush=True);return job


mode=sys.argv[1] if len(sys.argv)>1 else 'initial'
if mode=='initial':
    if not (ROOT/'protocol.json').exists():
        from tools.bf_t1_t2 import prepare
        prepare()
    calibration=submit('calibration','calibration',gpu=2,minutes=60)
    t1=submit('T1','train',[calibration],'BF-SD-T1',2,1320)
    t2=submit('T2','train',[calibration],'BF-SD-T2',2,1320)
    submit('process','process',[t1,t2],minutes=180)
elif mode=='evaluate':
    assert (ROOT/'metrics/cpu_per_image.json').exists()
    submit('evaluate','evaluate',gpu=1,minutes=30)
elif mode=='package':
    assert read(ROOT/'evaluation_complete.json')['complete'];submit('package','package',minutes=30)
else:raise ValueError(mode)
