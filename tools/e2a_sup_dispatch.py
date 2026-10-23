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


def review_environment_resume(identifier,reason):
    """仅恢复在环境检查处失败、未开始 load_state 的训练，不重置已有科学更新。"""
    assert reason.strip()
    with (ROOT/'dispatch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        jobs=history();job=next(j for j in jobs if j['id']==identifier)
        assert job['id']==[j for j in jobs if j['key']==job['key']][-1]['id']
        assert job['action']=='train' and job['args'][2:]==['--phase','formal']
        assert job_state(job)=='FAILED'
        assert not any(job_state(j) in ['RUNNING','PENDING','CONFIGURING','COMPLETING'] for j in jobs)
        run=folder(job['args'][1],'formal');state=run/'state-latest'
        assert not (run/'training_complete.json').exists() and not (run/'overflow_failure.json').exists()
        assert not (run/('recovery_discarded_%d.json'%identifier)).exists()
        log=ROOT/'logs'/('%s_%d.log'%(job['key'],identifier));text=log.read_text()
        assert 'FROZEN_CHANGED' in text and 'environment_rank' in text
        config=read(run/'config.json')
        assert all(sha(SOURCE/name)==digest for name,digest in config['additions'].items())
        assert sha(SOURCE/'train_GAM_texture_joint.py')==config['base_source_sha256']
        # 使用同一原始参数重建 CPU scheduler，只允许默认名称列表排列不同。
        from diffusers import DDIMScheduler
        scheduler=dict(DDIMScheduler(beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear',
            num_train_timesteps=1000,prediction_type='epsilon').config)
        for rank in range(2):
            saved=read(run/('environment_rank%d.json'%rank));current=dict(saved,scheduler=scheduler)
            assert canonical_environment(saved)==canonical_environment(current)
        progress=read(state/'progress.json');n=progress['updates']
        with (run/'training_log.csv').open() as f:rows=list(csv.DictReader(f))
        with (OLD/'training/seed42/F000/training_log.csv').open() as f:baseline=list(csv.DictReader(f))[:n]
        assert [int(r['update']) for r in rows]==list(range(1,n+1))
        assert all(a[k]==b[k] for a,b in zip(rows,baseline)
            for k in ['update','lr','microbatches_per_rank','rng_rank0'])
        assert progress['world_size']==2 and progress['seed']==42
        assert read(state/'stream_rank0.json')['rng_chain']==rows[-1]['rng_rank0']
        names=['pytorch_model.bin']+['pytorch_model_%d.bin'%i for i in range(1,5)]+[
            'optimizer.bin','custom_checkpoint_0.pkl','scaler.pt','random_states_0.pkl','random_states_1.pkl',
            'stream_rank0.json','stream_rank1.json','progress.json']
        assert all((state/name).is_file() and (state/name).stat().st_size>0 for name in names)
        archive=ROOT/'audit/engineering_failures'/('job_%d'%identifier)
        assert not archive.exists();archive.mkdir(parents=True)
        files=[log,run/'config.json',run/'training_log.csv',run/'auxiliary_log.csv',
            run/'environment_rank0.json',run/'environment_rank1.json',state/'progress.json',
            state/'stream_rank0.json',state/'stream_rank1.json']
        evidence={}
        for path in files:
            target=archive/path.name;shutil.copy2(str(path),str(target));evidence[path.name]=sha(target)
        review=dict(passed=True,failed_job=identifier,key=job['key'],reason=reason,source_commit=commit(),
            resume_update=n,checkpoint_unchanged=True,scientific_budget_changed=False,
            preserved_files=evidence,state_file_sizes={name:(state/name).stat().st_size for name in names},
            normalization_only='scheduler._use_default_values order; all actual parameters remain strict')
        freeze(ROOT/'audit'/('retry_release_%d.json'%identifier),review)
        accounting()


def review_assets_retry(identifier,reason):
    """仅放行已审查的空仓库/权重直连失败，先保留原件；不得重置已注册依赖或训练。"""
    assert reason.strip()
    with (ROOT/'dispatch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        jobs=history();job=next(j for j in jobs if j['id']==identifier)
        assert job['id']==[j for j in jobs if j['key']==job['key']][-1]['id']
        assert job['action']=='assets' and job_state(job)=='FAILED'
        assert not any(job_state(j) in ['RUNNING','PENDING','CONFIGURING','COMPLETING'] for j in jobs)
        arm=job['args'][0];assert arm in ['T1','T2']
        registry=ROOT/'assets/registry.json'
        assert arm not in (read(registry) if registry.exists() else {})
        assert not any(folder(arm,phase).exists() for phase in ['calibrate','pilot','formal'])
        name,url=('DISTS','https://github.com/dingkeyan93/DISTS.git') if arm=='T1' else ('dino','https://github.com/facebookresearch/dino.git')
        repo=ROOT/'assets'/name
        assert subprocess.check_output(['git','-C',str(repo),'remote','get-url','origin'],text=True).strip()==url
        log=ROOT/'logs'/('%s_%d.log'%(job['key'],identifier));text=log.read_text()
        head=subprocess.run(['git','-C',str(repo),'rev-parse','--verify','HEAD'],capture_output=True,text=True)
        if head.returncode:
            assert "ambiguous argument 'HEAD'" in text and "'rev-parse', 'HEAD'" in text
            failure_kind='empty_git_HEAD'
        else:
            assert arm=='T1' and '[Errno 101] Network is unreachable' in text
            assert 'https://download.pytorch.org/models/vgg16-397923af.pth' in text
            assert 'vgg16(weights=VGG16_Weights.IMAGENET1K_V1)' in text
            assert 'HEAD is now at '+head.stdout.strip()[:7] in text
            failure_kind='official_VGG_direct_network_unreachable'
        archive=ROOT/'audit/engineering_failures'/('job_%d'%identifier)
        assert not archive.exists();archive.mkdir(parents=True)
        for path in [log,ROOT/'jobs.json',ROOT/'status.json']:
            shutil.copy2(str(path),str(archive/path.name))
        shutil.copytree(str(repo/'.git'),str(archive/(name+'_git')))
        saved={str(path.relative_to(archive)):sha(path) for path in archive.rglob('*') if path.is_file()}
        assert sha(archive/log.name)==sha(log)
        freeze(ROOT/'audit'/('retry_release_%d.json'%identifier),dict(passed=True,failed_job=identifier,
            key=job['key'],reason=reason,source_commit=commit(),preserved_files=saved,
            failed_source_commit=job['commit'],failure_kind=failure_kind,
            dependency_commit=head.stdout.strip() if not head.returncode else None,
            official_origin=url,registry_not_created=True,training_not_started=True,
            scientific_budget_changed=False))
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
            job=submit('assets_'+arm,'assets',[arm]);status(stage='PREPARING_ASSETS_'+arm,engineering_failure=None);watch([job]);return
        if arm in ['T1','T2'] and arm not in read(ROOT/'assets/registry.json'):
            job=submit('assets_'+arm,'assets',[arm]);status(stage='PREPARING_ASSETS_'+arm,engineering_failure=None);watch([job]);return
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
    from tools.e2a_sup_reference import approved_reference_manifest
    approved_reference_manifest()
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
    from tools.e2a_sup_bundle import build
    build()
    print('Five-arm statistics complete; swap, blind/tail review and local report remain',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--retry-job',type=int)
    parser.add_argument('--resume-reviewed-job',type=int)
    parser.add_argument('--retry-assets-job',type=int)
    parser.add_argument('--reason')
    options=parser.parse_args()
    try:
        if options.retry_job:review_calibration_retry(options.retry_job,options.reason or '')
        if options.resume_reviewed_job:review_environment_resume(options.resume_reviewed_job,options.reason or '')
        if options.retry_assets_job:review_assets_retry(options.retry_assets_job,options.reason or '')
        advance()
    except Exception as error:
        status(engineering_failure=repr(error));raise
