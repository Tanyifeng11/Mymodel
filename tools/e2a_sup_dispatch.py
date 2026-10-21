"""独立编排 T3→T1→T2；工程失败停下审查，不缩预算、不进入旧八组。"""
import argparse
import csv
import fcntl
import re
import shutil
import subprocess
from tools.e2a_sup_common import *


def history(): return read(ROOT/'jobs.json') if (ROOT/'jobs.json').exists() else []


def job_state(job):
    value=subprocess.check_output(['sacct','-n','-X','-j',str(job['id']),'--format=State','--parsable2'],text=True).strip()
    return value.splitlines()[0].split('|')[0].split()[0] if value else 'PENDING'


def submit(key,action,args=(),gpus=0,dependency=None):
    jobs=history();prior=[j for j in jobs if j['key']==key]
    if prior:
        latest=prior[-1];state=job_state(latest)
        if state in ['RUNNING','PENDING','CONFIGURING','COMPLETING']:return latest
        resume=False
        if action=='train':
            run=folder(args[1],args[3])
            resume=state=='COMPLETED' and (run/'pauses'/('job_%d.json'%latest['id'])).exists()
        release=ROOT/'audit'/('retry_release_%d.json'%latest['id'])
        if not resume and release.exists():
            review=read(release)
            resume=(review['passed'] and review['failed_job']==latest['id'] and
                review['key']==key and review['source_commit']==commit())
        if not resume:raise RuntimeError('REVIEW_REQUIRED %s %s'%(key,state))
    command=['sbatch','--parsable','--partition='+('gpu' if gpus else 'cpu'),'--nodes=1','--ntasks=1',
        '--cpus-per-task='+str(4*gpus if gpus else 8),'--mem='+str(48 if gpus==2 else 32 if gpus else 16)+'G',
        '--time='+('04:00:00' if action=='train' else '03:00:00'),
        '--job-name=E2AS_'+key,'--output='+str(ROOT/'logs'/(key+'_%j.log'))]
    if gpus:command+=['--gres=gpu:'+str(gpus)]
    if dependency:command+=['--dependency=afterany:'+':'.join(str(j['id']) for j in dependency)]
    command +=[str(SOURCE/'scripts/e2a_sup.sbatch'),action,*args]
    identifier=int(subprocess.check_output(command,text=True).strip().split(';')[0])
    value=dict(id=identifier,key=key,action=action,args=list(args),gpus=gpus,command=command,
        commit=commit(),attempt=len(prior)+1)
    if prior and (ROOT/'audit'/('retry_release_%d.json'%prior[-1]['id'])).exists():
        value['retry_of']=prior[-1]['id']
        prior[-1]['superseded_by']=identifier
    jobs.append(value);write(ROOT/'jobs.json',jobs);print('SUBMITTED',key,identifier,flush=True);return value


def watch(jobs):
    submit('advance_'+'_'.join(str(j['id']) for j in jobs),'advance',dependency=jobs)


def review_calibration_retry(identifier,reason):
    """仅放行已审查、尚无科学产物的失败校准；归档整次尝试后重新固定 512 样本。"""
    assert reason.strip()
    with (ROOT/'dispatch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        jobs=history(); job=next(j for j in jobs if j['id']==identifier)
        assert job['id']==[j for j in jobs if j['key']==job['key']][-1]['id']
        assert job['action']=='train' and job['args'][2:]==['--phase','calibrate']
        assert job_state(job) in ['FAILED','OUT_OF_MEMORY','TIMEOUT','CANCELLED']
        assert not any(job_state(j) in ['RUNNING','PENDING','CONFIGURING','COMPLETING'] for j in jobs)
        arm=job['args'][1]; run=folder(arm,'calibrate')
        assert not (run/'calibration_complete.json').exists()
        assert arm not in (read(ROOT/'loss_registry.json') if (ROOT/'loss_registry.json').exists() else {})
        archive=ROOT/'superseded/engineering_failures'/('%s_job_%d'%(job['key'],identifier))
        assert not archive.exists()
        log=ROOT/'logs'/('%s_%d.log'%(job['key'],identifier))
        evidence=dict(passed=True,failed_job=identifier,key=job['key'],state=job_state(job),reason=reason,
            source_commit=commit(),failed_log_sha256=sha(log),archive=str(archive),
            preserved_files={str(p.relative_to(run)):sha(p) for p in run.rglob('*') if p.is_file()},
            calibration_restarts_from_E1=True,optimizer_updates_authorized=0,
            target_samples=512,effective_batches=64,scientific_budget_changed=False)
        archive.parent.mkdir(parents=True,exist_ok=True)
        shutil.move(str(run),str(archive))
        freeze(ROOT/'audit'/('retry_release_%d.json'%identifier),evidence)
        accounting()


def accounting():
    jobs=history()
    if not jobs:return
    fields=['JobIDRaw','JobName','State','ExitCode','ElapsedRaw','AllocTRES','TotalCPU','MaxRSS','Start','End']
    output=subprocess.check_output(['sacct','-n','-X','-j',','.join(str(j['id']) for j in jobs),
        '--parsable2','--format='+','.join(fields)],text=True)
    rows=[]
    for line in output.splitlines():
        if not line.strip():continue
        row=dict(zip(fields,line.split('|')));match=re.search(r'gres/gpu=(\d+)',row.get('AllocTRES',''))
        row['gpu_hours']=float(row.get('ElapsedRaw') or 0)*(int(match.group(1)) if match else 0)/3600
        row['includes']='all allocations, calibration, pilot, formal, pauses, evaluation and failures'
        rows.append(row)
    csv_write(ROOT/'resource_ledger.csv',rows)


def training_audit(arm,phase):
    run=folder(arm,phase);count=read(run/'optimizer_update_count.json');n=800 if phase=='pilot' else 28365
    assert read(run/'checkpoint_verification.json')['passed']
    assert count['actual_successful_updates']==n and count['skipped_updates']==0
    with (run/'training_log.csv').open() as f: rows=list(csv.DictReader(f))
    with (OLD/'training/seed42/F000/training_log.csv').open() as f: baseline=list(csv.DictReader(f))[:n]
    assert [int(r['update']) for r in rows]==list(range(1,n+1))
    for key in ['update','lr','microbatches_per_rank','rng_rank0']:
        assert all(a[key]==b[key] for a,b in zip(rows,baseline)),('TRACE_MISMATCH',arm,phase,key)
    streams=read(run/'seed_and_rng_manifest.json')
    assert len(streams['streams'])==2
    if phase=='formal':
        assert streams==read(OLD/'training/seed42/F000/seed_and_rng_manifest.json'),'ALL_RANK_RNG_MISMATCH'
    write(ROOT/'audit'/(phase+'_'+arm+'.json'),dict(passed=True,actual_updates=n,skipped=0,
        rank0_full_update_trace_equal_B=True,all_rank_final_streams_equal_B=phase=='formal',
        checkpoint_sha256=read(run/'checkpoint_verification.json')['sha256']))


def advance():
    ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'logs').mkdir(exist_ok=True)
    lock=(ROOT/'dispatch.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    accounting()
    if not (ROOT/'audit/preparation_complete.json').exists():
        job=submit('prepare','prepare');watch([job]);return
    assert read(ROOT/'audit/preparation_complete.json')['passed']
    assert (ROOT/'calibration_choice.json').exists(),'CALIBRATION_TARGET_NOT_FROZEN'
    for arm in ARMS:
        if arm in ['T1','T2'] and not (ROOT/'assets/registry.json').exists():
            job=submit('assets_'+arm,'assets',[arm]);watch([job]);return
        if arm in ['T1','T2'] and arm not in read(ROOT/'assets/registry.json'):
            job=submit('assets_'+arm,'assets',[arm]);watch([job]);return
        run=folder(arm,'calibrate')
        if not (run/'calibration_complete.json').exists():
            job=submit('calibrate_'+arm,'train',['--arm',arm,'--phase','calibrate'],gpus=2)
            status(stage='CALIBRATING_'+arm,engineering_failure=None);watch([job]);return
        for phase in ['pilot','formal']:
            run=folder(arm,phase)
            if not (run/'checkpoint_verification.json').exists():
                job=submit(phase+'_'+arm,'train',['--arm',arm,'--phase',phase],gpus=2)
                status(stage=phase.upper()+'_'+arm,engineering_failure=None);watch([job]);return
            training_audit(arm,phase)
            # pilot 不用于选系数。先由可审查的工程/图像诊断确认未损坏，才允许正式预算。
            if phase=='pilot' and not (ROOT/'audit'/('pilot_release_'+arm+'.json')).exists():
                if not (ROOT/'evaluation/pilot_P2a'/arm/'evaluation_complete.json').exists():
                    job=submit('pilot_eval_'+arm,'eval',['--arm',arm,'--stage','P2a','--phase','pilot'],gpus=1)
                    status(stage='PILOT_DIAGNOSTICS_'+arm);watch([job]);return
                status(stage='PILOT_REVIEW_REQUIRED_'+arm,engineering_failure=None)
                print('PILOT_REVIEW_REQUIRED',arm,flush=True);return
        if not (ROOT/'evaluation/P2a'/arm/'evaluation_complete.json').exists():
            job=submit('P2a_'+arm,'eval',['--arm',arm,'--stage','P2a'],gpus=1);watch([job]);return
    # 大规模比较之前，reference-swap 的合法同类参考必须先人工核对，避免按生成选参考。
    if not (ROOT/'audit/reference_swap_release.json').exists():
        status(stage='REFERENCE_SWAP_REVIEW_REQUIRED');return
    jobs=[]
    for arm in ['B','C','T1','T2','T3']:
        if not (ROOT/'evaluation/P2b'/arm/'evaluation_complete.json').exists():
            jobs.append(submit('P2b_'+arm,'eval',['--arm',arm,'--stage','P2b'],gpus=1))
    if jobs:watch(jobs);return
    for arm in ['B','C']+ARMS:
        if not (ROOT/'evaluation/reference_swap'/arm/'evaluation_complete.json').exists():
            jobs.append(submit('reference_swap_'+arm,'eval',['--arm',arm,'--stage','reference_swap'],gpus=1))
    if jobs:watch(jobs);return
    from tools.e2a_sup_stats import summarize
    summarize()
    print('Five-arm statistics complete; swap, blind/tail review and local report remain',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--retry-job',type=int)
    parser.add_argument('--reason')
    options=parser.parse_args()
    try:
        if options.retry_job:review_calibration_retry(options.retry_job,options.reason or '')
        advance()
    except Exception as error:
        status(engineering_failure=repr(error));raise
