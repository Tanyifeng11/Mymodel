"""只通过Slurm执行；每次提交/失败留痕，先门禁后科研训练。"""
import argparse
import subprocess
from datetime import datetime,timezone
from tools.m54_protocol import *


def submit(name,action,gpu,dependencies=(),extra=()):
    path=OUT/'resources/jobs.json';jobs=read(path) if path.exists() else []
    existing=next((x for x in jobs if x['name']==name),None)
    if existing:return existing['id']
    cmd=['sbatch','--parsable','--job-name=M54_'+name]
    if dependencies:cmd+=['--dependency=afterok:'+':'.join(map(str,dependencies))]
    cmd+=['submit/m54_'+('gpu' if gpu else 'cpu')+'.sh',action]+list(extra)
    result=subprocess.run(cmd,text=True,capture_output=True)
    if result.returncode:
        p=OUT/'resources/submission_failures.json';failures=read(p) if p.exists() else []
        failures.append(dict(command=cmd,stdout=result.stdout,stderr=result.stderr,utc=datetime.now(timezone.utc).isoformat()));write(p,failures)
        raise RuntimeError(result.stderr)
    identity=int(result.stdout.strip().split(';')[0]);jobs.append(dict(name=name,id=identity,action=action,gpu=gpu,
        dependencies=list(dependencies),extra=list(extra),commit=commit(),utc=datetime.now(timezone.utc).isoformat()))
    write(path,jobs);print('M54 SUBMITTED',name,identity,flush=True);return identity


def status():
    path=OUT/'resources/jobs.json';jobs=read(path) if path.exists() else []
    if not jobs:print('NO_M54_JOBS');return
    output=subprocess.check_output(['sacct','-X','-P','-j',','.join(str(j['id']) for j in jobs),
        '--format=JobIDRaw,JobName,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES,Start,End'],text=True)
    (OUT/'resources/accounting_current.psv').write_text(output,encoding='utf-8');print(output,flush=True)
    if (OUT/'decision.json').exists():print('DECISION',read(OUT/'decision.json'),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['gates','status']);args=p.parse_args()
    (OUT/'resources').mkdir(parents=True,exist_ok=True)
    if args.action=='gates':
        prepare=submit('PREPARE','prepare',False)
        submit('ROUND0','audit',True,[prepare])
    status()


if __name__=='__main__':main()
