"""保留提交/拒绝记录；第一批只提交数据、精确回放和能量门禁。"""
import argparse
import subprocess
from datetime import datetime,timezone
from tools.e53_protocol import *

def submit(name,action,gpu,dependencies=(),extra=()):
    path=OUT/'resources/jobs.json';jobs=read(path) if path.exists() else []
    if any(x['name']==name for x in jobs):return next(x['id'] for x in jobs if x['name']==name)
    cmd=['sbatch','--parsable','--job-name=E53_'+name]
    if dependencies:cmd+=['--dependency=afterok:'+':'.join(str(x) for x in dependencies)]
    cmd+=['submit/e53_'+('gpu' if gpu else 'cpu')+'.sh',action]+list(extra)
    result=subprocess.run(cmd,text=True,capture_output=True)
    if result.returncode:
        p=OUT/'resources/submission_failures.json';failures=read(p) if p.exists() else []
        failures.append(dict(utc=datetime.now(timezone.utc).isoformat(),command=cmd,stdout=result.stdout,stderr=result.stderr,returncode=result.returncode));write(p,failures)
        raise RuntimeError(result.stderr)
    identity=int(result.stdout.strip().split(';')[0]);jobs.append(dict(name=name,id=identity,gpu=gpu,action=action,dependencies=list(dependencies),extra=list(extra),commit=commit(),utc=datetime.now(timezone.utc).isoformat()));write(path,jobs)
    print('SUBMITTED',name,identity,flush=True);return identity

def status():
    p=OUT/'resources/jobs.json';jobs=read(p) if p.exists() else []
    if not jobs:print('NO_E53_JOBS');return
    result=subprocess.check_output(['sacct','-X','-P','-j',','.join(str(x['id']) for x in jobs),'--format=JobIDRaw,JobName,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES,Start,End'],text=True)
    (OUT/'resources/accounting_current.psv').write_text(result,encoding='utf-8');print(result,flush=True)
    d=OUT/'decision.json'
    if d.exists():print('DECISION',read(d),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['gates','status']);args=p.parse_args();(OUT/'resources').mkdir(parents=True,exist_ok=True)
    if args.action=='status':status();return
    data=submit('DATA','data',False);audit=submit('REPLAY','audit',True,[data]);submit('ENERGY','energy',True,[audit]);status()

if __name__=='__main__':main()
