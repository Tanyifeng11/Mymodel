"""四作业依赖图，逐项持久化；重复调用默认只显示既有账本。"""
import argparse
import subprocess
from tools.e51_protocol import *


def dispatch(resume_missing=False):
    path=OUT/'job_ids.json'
    if path.exists() and not resume_missing:print('EXISTING E51 JOBS',read(path),flush=True);return
    prepare();jobs=read(path) if path.exists() else {};write(path,jobs)
    def submit(name,args,minutes,deps=()):
        if name in jobs:return jobs[name]
        command=['sbatch','--parsable','--job-name=e51_'+name,'--time=%02d:%02d:00'%(minutes//60,minutes%60)]
        if deps:command+=['--dependency=afterok:'+':'.join(map(str,deps))]
        job=int(subprocess.check_output(command+['submit/e51_gpu.sh']+args,text=True).strip().split(';')[0])
        jobs[name]=job;write(path,jobs);decision(submitted_jobs=jobs);print('E51 SUBMITTED',name,job,flush=True);return job
    try:
        audit=submit('audit',['audit'],25)
        training=[submit(a,['run','--arm',a],75,[audit]) for a in TRAIN_ARMS]
        submit('evaluate',['evaluate'],15,training)
    except subprocess.CalledProcessError:
        print('E51 SUBMISSION WAIT: retain ledger; inspect Slurm before --resume-missing',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resume-missing',action='store_true')
    dispatch(p.parse_args().resume_missing)
