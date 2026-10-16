"""只通过Slurm执行；每次提交/失败留痕，先门禁后科研训练。"""
import argparse
import subprocess
from datetime import datetime,timezone
from tools.m54_protocol import *


def submit(name,action,gpu,dependencies=(),extra=(),slurm_options=(),logical_name=None):
    path=OUT/'resources/jobs.json';jobs=read(path) if path.exists() else []
    existing=next((x for x in jobs if x['name']==name),None)
    if existing:return existing['id']
    cmd=['sbatch','--parsable','--job-name=M54_'+name]
    cmd+=list(slurm_options)
    if dependencies:cmd+=['--dependency=afterok:'+':'.join(map(str,dependencies))]
    cmd+=['submit/m54_'+('gpu' if gpu else 'cpu')+'.sh',action]+list(extra)
    result=subprocess.run(cmd,text=True,capture_output=True)
    if result.returncode:
        p=OUT/'resources/submission_failures.json';failures=read(p) if p.exists() else []
        failures.append(dict(command=cmd,stdout=result.stdout,stderr=result.stderr,utc=datetime.now(timezone.utc).isoformat()));write(p,failures)
        raise RuntimeError(result.stderr)
    identity=int(result.stdout.strip().split(';')[0]);record=dict(name=name,id=identity,action=action,gpu=gpu,
        dependencies=list(dependencies),extra=list(extra),slurm_options=list(slurm_options),commit=commit(),utc=datetime.now(timezone.utc).isoformat())
    if logical_name:record['logical_name']=logical_name
    jobs.append(record)
    write(path,jobs);print('M54 SUBMITTED',name,identity,flush=True);return identity


def status():
    path=OUT/'resources/jobs.json';jobs=read(path) if path.exists() else []
    if not jobs:print('NO_M54_JOBS');return
    output=subprocess.check_output(['sacct','-X','-P','-j',','.join(str(j['id']) for j in jobs),
        '--format=JobIDRaw,JobName,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES,Start,End'],text=True)
    (OUT/'resources/accounting_current.psv').write_text(output,encoding='utf-8');print(output,flush=True)
    if (OUT/'decision.json').exists():print('DECISION',read(OUT/'decision.json'),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['gates','status','advance','input_candidates','recheck_modules','repair_round0']);args=p.parse_args()
    (OUT/'resources').mkdir(parents=True,exist_ok=True)
    if args.action=='gates':
        prepare=submit('PREPARE','prepare',False)
        submit('ROUND0','audit',True,[prepare])
    elif args.action=='advance':advance()
    elif args.action=='input_candidates':submit('INPUT_CANDIDATES_R1','input_candidates',False,
        slurm_options=['--cpus-per-task=16','--mem=32G','--time=04:00:00'])
    elif args.action=='recheck_modules':
        original=next(x for x in read(OUT/'resources/jobs.json') if x['name']=='ROUND0')
        protocol=submit('PROTOCOL_R1','prepare',False,[original['id']])
        m11=submit('ROUND0_M11_R1','audit',True,[protocol],['--arm','M11'],logical_name='ROUND0')
        submit('ROUND0_M9_R1','audit',True,[m11],['--arm','M9'],logical_name='ROUND0')
    elif args.action=='repair_round0':
        latest={j.get('logical_name',j['name']):j for j in read(OUT/'resources/jobs.json')}
        failed=latest['ROUND0']['id']
        protocol=submit('PREPARE_REPAIR_R1','prepare_retry',False,extra=['--failed-job',str(failed)])
        submit('ROUND0_R1','audit',True,[protocol],logical_name='ROUND0')
    status()


def advance():
    import fcntl
    with (OUT/'resources/dispatch.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        jobs=read(OUT/'resources/jobs.json')
        raw=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(str(j['id']) for j in jobs),'--format=JobIDRaw,State'],text=True)
        states={int(s.split('|')[0]):s.split('|')[1].split()[0] for s in raw.splitlines() if s.split('|')[0].isdigit()}
        latest={j.get('logical_name',j['name']):j for j in jobs}
        failed=[j for j in latest.values() if states.get(j['id']) in ['FAILED','TIMEOUT','OUT_OF_MEMORY','CANCELLED','NODE_FAIL']]
        if failed:print('ENGINEERING_REPAIR_REQUIRED',[(j['name'],j['id']) for j in failed]);return
        if not (OUT/'audit/round0_complete.json').exists():return
        assert read(OUT/'audit/round0_complete.json')['passed']
        if states.get(latest['ROUND0']['id'])!='COMPLETED':return
        review=OUT/'audit/input_review/pilot_approval.json'
        if not review.exists():print('ASSISTANT_PILOT_INPUT_REVIEW_PENDING');return
        assert read(review)['passed']
        assert read(review)['split_sha256']==sha(split_path('pilot')),'INPUT_REVIEW_SPLIT_MISMATCH'
        if not (OUT/'round1/evaluation_complete.json').exists():
            base=submit('R1_BASELINES','batch_generate',True,extra=['--stage','round1','--arms','E5','B1','SAFE'])
            shards=[]
            for i,arms in enumerate([list(LAYER_ARMS),['M8_T100','M8_T085','M8_T070'],['M10','HF_ONLY']]):
                extra=['--stage','round1','--arms']+arms
                if i==2:extra+=['--diagnostic']
                shards.append(submit('R1_ZERO'+str(i),'batch_generate',True,[base],extra))
            submit('R1_EVAL','evaluate',False,[base]+shards,['--stage','round1']);return
        choice=OUT/'round1/selection.json'
        if not choice.exists():
            gates=read(OUT/'round1/gates.json');means=read(OUT/'round1/means.json')
            possible=[a for a in ['M8_T100','M8_T085','M8_T070'] if gates[a]['passed']]
            winner=min(possible,key=lambda a:(means[a]['lpips_gt'],a)) if possible else 'M8_T100'
            freeze(choice,dict(M1_temperature={'M8_T100':1.,'M8_T085':.85,'M8_T070':.70}[winner],
                M8_selected=winner,selection='minimum LPIPS among six-gate passers; otherwise tau1',before_training=True))
        active=sum(states.get(j['id'],'PENDING') not in ['COMPLETED'] for j in latest.values())
        slots=max(0,4-active)
        for arm in TRAIN_ARMS:
            if 'TRAIN_'+arm not in latest:
                if slots<=0:break
                submit('TRAIN_'+arm,'train',True,extra=['--arm',arm]);slots-=1
        # 后续Pilot生成以实际训练完成检查点为准，绝不将工程失败当作科研淘汰。
        for arm in TRAIN_ARMS:
            job=latest.get('TRAIN_'+arm)
            if job and states.get(job['id'])=='COMPLETED' and 'PILOT_'+arm not in latest and slots>0:
                recipes=['M5_S075','M5_S100','M5_S125'] if arm=='M5' else [arm]
                submit('PILOT_'+arm,'batch_generate',True,extra=['--stage','pilot','--arms']+recipes);slots-=1
        decision(R1='COMPLETE',R2='IN_PROGRESS')


if __name__=='__main__':main()
