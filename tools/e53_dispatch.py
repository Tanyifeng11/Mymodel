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

def states():
    path=OUT/'resources/jobs.json';jobs=read(path) if path.exists() else []
    if not jobs:return {},{}
    output=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(str(j['id']) for j in jobs),'--format=JobIDRaw,State'],text=True)
    byid={int(line.split('|')[0]):line.split('|')[1].split()[0] for line in output.splitlines() if line.split('|')[0].isdigit()}
    return {j['name']:j for j in jobs},byid

def advance():
    import fcntl
    (OUT/'resources').mkdir(parents=True,exist_ok=True)
    with (OUT/'resources/dispatch.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        jobs,st=states()
        if any(st.get(j['id']) in ['FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL'] for j in jobs.values()):print('ENGINEERING_REPAIR_REQUIRED');return
        if not (OUT/'audit/energy_gate.json').exists() or not read(OUT/'audit/energy_gate.json')['passed']:return
        approval=OUT/'audit/human_mapping_review.json'
        if not approval.exists() or not read(approval)['approved']:print('HUMAN_MAPPING_REVIEW_REQUIRED');return
        active=sum(st.get(j['id'],'PENDING')!='COMPLETED' for j in jobs.values());slots=5-active
        def add(name,action,gpu,dependencies=(),extra=()):
            nonlocal slots
            if name in jobs:return jobs[name]['id']
            if slots<=0:return None
            identity=submit(name,action,gpu,dependencies,extra);jobs[name]=dict(id=identity,name=name);slots-=1;return identity
        for stage in ['pilot','lock']:
            gate=OUT/'configs/lock.json' if stage=='lock' else approval
            if not gate.exists():break
            ids=[]
            for shard in range(4):
                identity=add(stage.upper()+str(shard),'generate',True,extra=['--stage',stage,'--shard',str(shard),'--shards','4'])
                if identity is not None:ids.append(identity)
            if len(ids)<4:break
            evaluator=add(stage.upper()+'_EVAL','evaluate',False,ids,['--stage',stage])
            summary=OUT/'metrics'/stage/'summary.json'
            if not evaluator or not summary.exists() or st.get(evaluator)!='COMPLETED':break
            if stage=='pilot':
                freezer=add('FREEZE','freeze',False,[evaluator])
                if not freezer or st.get(freezer)!='COMPLETED':break
            else:add('FINAL','finish',False,[evaluator])
        status()

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['gates','status','advance']);args=p.parse_args();(OUT/'resources').mkdir(parents=True,exist_ok=True)
    if args.action=='status':status();return
    if args.action=='advance':advance();return
    data=submit('DATA','data',False);audit=submit('REPLAY','audit',True,[data]);submit('ENERGY','energy',True,[audit]);status()

if __name__=='__main__':main()
