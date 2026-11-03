"""登记后不重提；两项单GPU作业预留合计3600GPU秒。"""
import sys
import subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.t3_cfg_lab import ROOT,read,write

ROOT.joinpath('logs').mkdir(parents=True,exist_ok=True)
path=ROOT/'submitted_jobs.json'
jobs=read(path) if path.exists() else {}


def submit(mode,dependency=None,minutes=60,gpu=False):
    if mode in jobs:return jobs[mode]
    args=['sbatch','--parsable','-p','gpu' if gpu else 'cpu','-J','t3cl_'+mode,'-N1','-n1','-c','4',
          '--mem','48G','-t',str(minutes),'-o',str(ROOT/'logs'/(mode+'_%j.log'))]
    if gpu:args+=['--gres=gpu:1']
    if dependency:args+=['--dependency=afterok:'+str(dependency)]
    args+=[str(ROOT/'source/scripts/t3_cfg_lab.sbatch'),mode]
    job=subprocess.check_output(args,text=True).strip();assert job.isdigit(),job
    jobs[mode]=job;write(path,jobs);print(mode,job,flush=True)
    return job


if len(sys.argv)>1 and sys.argv[1]=='package':
    submit('package')
else:
    prepare=submit('prepare')
    generate=submit('generate',prepare,40,True)
    process=submit('process',generate,120)
    submit('neural',process,17,True)
