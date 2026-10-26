"""第一阶段单臂顺序编排；工程图审和科学开发门槛均保留。"""
import argparse
import fcntl
import subprocess
from tools.e2a_matched_common import *
from tools import e2a_sup_dispatch as reused

ACTIVE=['RUNNING','PENDING','CONFIGURING','COMPLETING']
reused.ROOT=ROOT;reused.folder=folder
history=reused.history
job_state=reused.job_state
accounting=reused.accounting


def submit(key,action,args=(),gpus=0,dependency=None):
    jobs=history();prior=[j for j in jobs if j['key']==key]
    if prior:
        latest=prior[-1];state=job_state(latest)
        if state in ACTIVE:return latest
        release=ROOT/'audit'/('retry_release_%d.json'%latest['id'])
        assert release.exists(),('REVIEW_REQUIRED',key,state)
        review=read(release)
        assert review['passed'] and review['failed_job']==latest['id'] and review['key']==key
        assert review['source_commit']==commit()
    command=['sbatch','--parsable','--partition='+('gpu' if gpus else 'cpu'),
        '--nodes=1','--ntasks=1','--cpus-per-task='+str(4*gpus if gpus else 8),
        '--mem='+str(48 if gpus==2 else 32 if gpus else 16)+'G',
        '--time='+('7-00:00:00' if action=='train' else '03:00:00'),
        '--job-name=E2AM_'+key,'--output='+str(ROOT/'logs'/(key+'_%j.log'))]
    if gpus:command+=['--gres=gpu:'+str(gpus)]
    if dependency:command+=['--dependency=afterany:'+':'.join(str(j['id']) for j in dependency)]
    command +=[str(SOURCE/'scripts/e2a_matched.sbatch'),action,*args]
    identifier=int(subprocess.check_output(command,text=True).strip().split(';')[0])
    job=dict(id=identifier,key=key,action=action,args=list(args),gpus=gpus,
        command=command,commit=commit(),attempt=len(prior)+1)
    if prior:job['retry_of']=prior[-1]['id'];prior[-1]['superseded_by']=identifier
    jobs.append(job);write(ROOT/'jobs.json',jobs)
    print('SUBMITTED',key,identifier,flush=True);return job


def watch(job):
    submit('advance_'+str(job['id']),'advance',dependency=[job])


def training_audit(phase):
    reused.training_audit('T3M',phase)
    run=folder('T3M',phase)
    assert not (run/'overflow_failure.json').exists()
    verification=read(run/'checkpoint_verification.json')
    assert sha(run/'checkpoint-final/joint_model.pt')==verification['sha256']
    with (run/'auxiliary_log.csv').open() as f:
        import csv,math
        rows=list(csv.DictReader(f))
    n=800 if phase=='pilot' else 28365
    assert [int(r['update']) for r in rows]==list(range(1,n+1))
    assert all(math.isfinite(float(v)) for r in rows for k,v in r.items() if k!='job')
    with (run/'training_log.csv').open() as f:training=list(csv.DictReader(f))
    assert all(math.isfinite(float(v)) for r in training for k,v in r.items() if k not in ['rng_rank0','job'])
    assert all(read(run/('matched_binding_rank%d.json'%rank))['gamma']==read(ROOT/'gamma_calibration.json')['gamma'] for rank in range(2))


def release_pilot(review):
    assert review.strip(),'Describe the actual eight-image engineering inspection'
    training_audit('pilot')
    evaluated=ROOT/'evaluation/pilot_P2a/T3M'
    complete=read(evaluated/'evaluation_complete.json')
    assert complete['complete'] and complete['n']==8
    freeze(ROOT/'audit/pilot_release_T3M.json',dict(passed=True,
        reviewer='AI engineering visual inspection; not human blind assessment',review=review,
        output_grid_sha256=sha(evaluated/'output_grid.png'),
        evaluation_sha256=sha(evaluated/'evaluation_complete.json'),
        pilot_audit_sha256=sha(ROOT/'audit/pilot_T3M.json'),
        registry_sha256=sha(ROOT/'loss_registry.json'),
        human_blind_scores=None,coefficients_changed_after_P2=False))


def advance():
    ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'logs').mkdir(exist_ok=True)
    with (ROOT/'dispatch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        accounting()
        if not (ROOT/'audit/preparation_complete.json').exists():
            job=submit('prepare','prepare');watch(job);return
        if not (folder('T3M','calibrate')/'calibration_complete.json').exists():
            job=submit('calibrate_T3M','train',['--arm','T3M','--phase','calibrate'],2)
            status(stage='CALIBRATING_T3M',engineering_failure=None);watch(job);return
        assert read(folder('T3M','calibrate')/'calibration_complete.json')['passed']
        assert read(ROOT/'gamma_calibration.json')['used_P2_or_LOCK'] is False
        for phase in ['pilot','formal']:
            if not (folder('T3M',phase)/'checkpoint_verification.json').exists():
                job=submit(phase+'_T3M','train',['--arm','T3M','--phase',phase],2)
                status(stage=phase.upper()+'_T3M',engineering_failure=None);watch(job);return
            training_audit(phase)
            release=ROOT/'audit/pilot_release_T3M.json'
            if release.exists():assert read(release)['passed']
            if phase=='pilot' and not (ROOT/'audit/pilot_release_T3M.json').exists():
                if not (ROOT/'evaluation/pilot_P2a/T3M/evaluation_complete.json').exists():
                    job=submit('pilot_eval_T3M','eval',['--arm','T3M','--stage','P2a','--phase','pilot'],1)
                    status(stage='PILOT_DIAGNOSTICS_T3M',engineering_failure=None);watch(job);return
                status(stage='PILOT_REVIEW_REQUIRED_T3M',engineering_failure=None);return
        for stage in ['P2a','P2b','reference_swap']:
            if not (ROOT/'evaluation'/stage/'T3M/evaluation_complete.json').exists():
                job=submit(stage+'_T3M','eval',['--arm','T3M','--stage',stage],1)
                status(stage=stage.upper()+'_T3M',engineering_failure=None);watch(job);return
        from tools.e2a_matched_stats import summarize
        summarize()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--release-pilot',action='store_true')
    parser.add_argument('--review',default='');options=parser.parse_args()
    try:
        if options.release_pilot:release_pilot(options.review)
        advance()
    except Exception as error:
        status(engineering_failure=repr(error));raise
