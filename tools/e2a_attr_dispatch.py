"""Slurm 阶段编排；仅在上阶段产物完整且门槛允许时推进。"""
import argparse
import re
import subprocess
import tarfile
from tools.e2a_attr_common import *


def jobs(): return read(OUT/'jobs.json') if (OUT/'jobs.json').exists() else []


def job_state(job):
    value=subprocess.check_output(['sacct','-n','-X','-j',str(job['id']),'--format=State','--parsable2'],text=True).strip()
    return value.splitlines()[0].split('|')[0].split()[0] if value else 'PENDING'


def layout():
    path=OUT/'training_layout.json'
    if path.exists(): return read(path)
    output=subprocess.check_output(['scontrol','show','node','--oneliner'],text=True)
    available=[]
    for line in output.splitlines():
        if 'Gres=gpu:' not in line: continue
        total=re.search(r'Gres=gpu:[^ :]+:(\d+)',line)
        used=re.search(r'AllocTRES=([^ ]*)',line)
        allocated=re.search(r'gres/gpu=(\d+)',used.group(1)) if used else None
        free=int(total.group(1))-(int(allocated.group(1)) if allocated else 0)
        available.append(free)
    largest=max(available or [0]); world=max([g for g in [1,2,4] if g<=max(1,largest)])
    return freeze(path,dict(world_size=world,effective_batch=8,accumulation=8//world,
        observed_node_free_gpus=available,selection='largest up to four GPUs currently free on one node; same layout for all training seeds/arms'))


def submit(key,action,args=(),gpus=0,minutes=None,dependency=None,retry=False):
    history=jobs(); previous=[j for j in history if j['key']==key]
    if previous:
        current=previous[-1]; status=job_state(current)
        if status in ['PENDING','RUNNING','COMPLETING','CONFIGURING']: return current
        if status=='COMPLETED': raise RuntimeError('COMPLETED_WITHOUT_EXPECTED_ARTIFACT '+key)
        if not retry: raise RuntimeError('FAILED_JOB_REQUIRES_REVIEW %s %s'%(key,status))
    jobname='E2A_'+key[-100:]; log=OUT/'logs'/(key+'_%j.log')
    cmd=['sbatch','--parsable','--partition='+('gpu' if gpus else 'cpu'),'--nodes=1','--ntasks=1',
         '--cpus-per-task='+str(4*gpus if gpus else 4),'--mem='+str(max(32,24*gpus) if gpus else 8)+'G',
         '--time='+('00:40:00' if action=='smoke' else '1-00:00:00' if action=='train' else '03:00:00'),
         '--job-name='+jobname,'--output='+str(log),'--export=ALL,E2A_GPUS='+str(max(1,gpus))]
    if dependency: cmd+=['--dependency=afterany:'+':'.join(str(j['id']) for j in dependency)]
    cmd += [str(SOURCE/'scripts/e2a_attr.sbatch'),action,*args]
    identifier=subprocess.check_output(cmd,text=True).strip().split(';')[0]
    value=dict(key=key,id=int(identifier),action=action,args=list(args),gpus=gpus,
        command=cmd,execution_commit=commit(),attempt=len(previous)+1)
    history.append(value); write(OUT/'jobs.json',history)
    print('SUBMITTED',key,identifier,flush=True); return value


def all_done(paths): return all(Path(p).exists() for p in paths)


def watch(batch):
    if not batch: return
    key='advance_'+ '_'.join(str(j['id']) for j in batch)
    submit(key,'advance',dependency=batch)


def accounting():
    history=jobs()
    if not history: return
    ids=','.join(str(j['id']) for j in history)
    fields=['JobIDRaw','JobName','State','ExitCode','ElapsedRaw','AllocTRES','TotalCPU','MaxRSS','Start','End']
    output=subprocess.check_output(['sacct','-n','-X','-j',ids,'--parsable2','--format='+','.join(fields)],text=True)
    rows=[]
    for line in output.splitlines():
        if not line.strip(): continue
        row=dict(zip(fields,line.split('|')))
        match=re.search(r'gres/gpu=(\d+)',row.get('AllocTRES',''))
        row['allocated_gpu_hours']=float(row.get('ElapsedRaw') or 0)*int(match.group(1))/3600 if match else 0.
        row['ledger_scope']='actual Slurm allocation elapsed; includes startup/checkpoint; not pure GPU kernel time'
        rows.append(row)
    csv_write(OUT/'runtime_resource_ledger.csv',rows)
    for folder in (OUT/'training').glob('seed*/F*'):
        path=folder/'runtime_resource_ledger.csv'
        if not path.exists(): path.symlink_to(OUT/'runtime_resource_ledger.csv')


def portable():
    accounting()
    with tarfile.open(OUT/'review_results.tar.gz','w:gz') as archive:
        for name in ['config.json','checkpoint_manifest.json','training_layout.json','jobs.json','status.json',
                     'runtime_resource_ledger.csv','final_result.json','audit','statistics','selection','splits']:
            path=OUT/name
            if path.exists(): archive.add(path,arcname=name)
        for folder in (OUT/'evaluation').glob('*/*'):
            if folder.is_dir():
                for path in folder.iterdir():
                    if path.is_file(): archive.add(path,arcname=str(path.relative_to(OUT)))
        for folder in (OUT/'training').glob('seed*/F*'):
            for name in ['config.json','loss_factor_manifest.json','optimizer_update_count.json',
                         'seed_and_rng_manifest.json','checkpoint_verification.json','training_log.csv']:
                path=folder/name
                if path.exists(): archive.add(path,arcname=str(path.relative_to(OUT)))
    (OUT/'review_results.tar.gz.sha256').write_text(sha(OUT/'review_results.tar.gz')+'  review_results.tar.gz\n')


def advance(retry=False):
    from tools.e2a_attr_stats import rng_audit,stage_a,factorial,lock_plan,final_lock
    assert read(OUT/'audit/masks_complete.json')['passed']
    if not (OUT/'audit/mask_visual_review.json').exists():
        state(preparation='MASK_VISUAL_REVIEW_PENDING'); return
    if not (OUT/'audit/loss_switch_selfcheck.json').exists():
        from tools.e2a_attr_selfcheck import selfcheck
        selfcheck()
    world=layout()['world_size']
    smoke_folders=[train_folder(42,f,True) for f in ['000','111']]
    if not all_done([p/'checkpoint_verification.json' for p in smoke_folders]):
        batch=[submit('smoke_s42_F'+f,'smoke',['--seed','42','--factor',f],world,retry=retry)
               for f,p in zip(['000','111'],smoke_folders) if not (p/'checkpoint_verification.json').exists()]
        watch(batch); state(preparation='SMOKE_RUNNING'); return
    write(OUT/'audit/smoke_rng_audit.json',rng_audit(42,['000','111'],True))
    state(preparation='PASSED')
    a_paths=[checkpoint(42,f).parent.parent/'checkpoint_verification.json' for f in ['000','111']]
    if not all_done(a_paths):
        batch=[submit('train_s42_F'+f,'train',['--seed','42','--factor',f],world,retry=retry)
               for f,p in zip(['000','111'],a_paths) if not p.exists()]
        watch(batch); state(A='TRAINING'); return
    a_arms=['E1','s42_F000','s42_F111']
    if not all_done([OUT/'evaluation/P1'/a/'evaluation_complete.json' for a in a_arms]):
        batch=[submit('eval_P1_'+a,'eval',['--stage','P1','--arm',a],1,retry=retry)
               for a in a_arms if not (OUT/'evaluation/P1'/a/'evaluation_complete.json').exists()]
        watch(batch); state(A='EVALUATING'); return
    decision=read(OUT/'statistics/P1/decision.json') if (OUT/'statistics/P1/decision.json').exists() else stage_a()
    if not decision['continue_to_B']:
        state(A='COMPLETE',B='STOPPED_BY_PROTOCOL',complete=True,
              conclusion='No overall benefit under the preregistered stage-A rule; no factorial/LOCK claim')
        portable(); return
    b_paths=[checkpoint(43,f).parent.parent/'checkpoint_verification.json' for f in FACTORS]
    if not all_done(b_paths):
        batch=[submit('train_s43_F'+f,'train',['--seed','43','--factor',f],world,retry=retry)
               for f,p in zip(FACTORS,b_paths) if not p.exists()]
        watch(batch); state(B='TRAINING'); return
    b_arms=['s43_F'+f for f in FACTORS]
    if not all_done([OUT/'evaluation/P2'/a/'evaluation_complete.json' for a in b_arms]):
        batch=[submit('eval_P2_'+a,'eval',['--stage','P2','--arm',a],1,retry=retry)
               for a in b_arms if not (OUT/'evaluation/P2'/a/'evaluation_complete.json').exists()]
        watch(batch); state(B='EVALUATING'); return
    selection=read(OUT/'selection/confirmation.json') if (OUT/'selection/confirmation.json').exists() else factorial()
    if not selection['any_favourable_direction']:
        state(confirmation='STOPPED_NO_FAVOURABLE_FACTOR',LOCK='NOT_OPENED',complete=True,
              conclusion='All factorial candidates lack favourable directions; preserve exploratory factorial evidence')
        portable(); return
    missing=[(s,f) for s in [42,43,44] for f in selection['arms'] if not (checkpoint(s,f).parent.parent/'checkpoint_verification.json').exists()]
    if missing:
        batch=[submit('train_s%d_F%s'%(s,f),'train',['--seed',str(s),'--factor',f],world,retry=retry) for s,f in missing]
        watch(batch); state(confirmation='TRAINING'); return
    arms=['s%d_F%s'%(s,f) for s in [42,43,44] for f in selection['arms']]
    missing=[a for a in arms if not (OUT/'evaluation/P2'/a/'evaluation_complete.json').exists()]
    if missing:
        batch=[submit('eval_P2_'+a,'eval',['--stage','P2','--arm',a],1,retry=retry) for a in missing]
        watch(batch); state(confirmation='EVALUATING'); return
    lock=read(OUT/'selection/LOCK_protocol.json') if (OUT/'selection/LOCK_protocol.json').exists() else lock_plan()
    missing=[a for a in lock['arms'] if not (OUT/'evaluation/LOCK'/a/'evaluation_complete.json').exists()]
    if missing:
        batch=[submit('eval_LOCK_'+a,'eval',['--stage','LOCK','--arm',a],1,retry=retry) for a in missing]
        watch(batch); state(LOCK='EVALUATING'); return
    if not (OUT/'final_result.json').exists(): final_lock()
    portable()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--retry-failed',action='store_true')
    parser.add_argument('--package-only',action='store_true');args=parser.parse_args()
    try:
        if args.package_only: portable()
        else: advance(args.retry_failed); accounting()
    except Exception as error:
        state(engineering_failure=repr(error),complete=False); accounting(); raise
