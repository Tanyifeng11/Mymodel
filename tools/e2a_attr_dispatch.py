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
    history=jobs(); attempts=[j for j in history if j['key']==key]
    previous=[j for j in attempts if not j.get('superseded_by')]
    if previous:
        current=previous[-1]; status=job_state(current)
        if status in ['PENDING','RUNNING','COMPLETING','CONFIGURING']: return current
        resumable=False
        if action=='train':
            seed=int(args[args.index('--seed')+1]); factor=args[args.index('--factor')+1]
            folder=train_folder(seed,factor)
            resumable=(status=='COMPLETED' and (folder/'pauses'/('job_%s.json'%current['id'])).exists()
                       and (folder/'state-latest/progress.json').exists())
        if status=='COMPLETED' and not resumable: raise RuntimeError('COMPLETED_WITHOUT_EXPECTED_ARTIFACT '+key)
        if not (retry or resumable): raise RuntimeError('FAILED_JOB_REQUIRES_REVIEW %s %s'%(key,status))
    jobname='E2A_'+key[-100:]; log=OUT/'logs'/(key+'_%j.log')
    cmd=['sbatch','--parsable','--partition='+('gpu' if gpus else 'cpu'),'--nodes=1','--ntasks=1',
         '--cpus-per-task='+str(4*gpus if gpus else 4),'--mem='+str(max(32,24*gpus) if gpus else 8)+'G',
         '--time='+('00:40:00' if action=='smoke' else '04:00:00' if action=='train' else '03:00:00'),
         '--job-name='+jobname,'--output='+str(log),'--export=ALL,E2A_GPUS='+str(max(1,gpus))]
    if gpus: cmd+=['--gres=gpu:'+str(gpus)]
    if dependency: cmd+=['--dependency=afterany:'+':'.join(str(j['id']) for j in dependency)]
    cmd += [str(SOURCE/'scripts/e2a_attr.sbatch'),action,*args]
    identifier=subprocess.check_output(cmd,text=True).strip().split(';')[0]
    value=dict(key=key,id=int(identifier),action=action,args=list(args),gpus=gpus,
        command=cmd,execution_commit=commit(),attempt=len(attempts)+1)
    history.append(value); write(OUT/'jobs.json',history)
    print('SUBMITTED',key,identifier,flush=True); return value


def all_done(paths): return all(Path(p).exists() for p in paths)


def watch(batch):
    if not batch: return
    key='advance_'+ '_'.join(str(j['id']) for j in batch)
    submit(key,'advance',dependency=batch)


def accounting():
    history=jobs()
    identifiers={j['id'] for j in history}
    identifiers.update(int(match.group(1)) for path in (OUT/'logs').glob('*.log')
                       for match in [re.search(r'_(\d+)\.log$',path.name)] if match)
    if not identifiers: return
    ids=','.join(map(str,sorted(identifiers)))
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
                     'runtime_resource_ledger.csv','final_result.json','controlled_result.json','evaluation/metric_protocol.json',
                     'audit','statistics','selection','splits']:
            path=OUT/name
            if path.exists(): archive.add(path,arcname=name)
        for folder in (OUT/'evaluation').glob('*/*'):
            if folder.is_dir():
                for path in folder.iterdir():
                    if path.is_file(): archive.add(path,arcname=str(path.relative_to(OUT)))
                for path in (folder/'images').glob('*.json'):
                    archive.add(path,arcname=str(path.relative_to(OUT)))
        for folder in (OUT/'training').glob('seed*/F*'):
            for name in ['config.json','loss_factor_manifest.json','optimizer_update_count.json',
                         'seed_and_rng_manifest.json','checkpoint_verification.json','training_log.csv']:
                path=folder/name
                if path.exists(): archive.add(path,arcname=str(path.relative_to(OUT)))
            for pattern in ['pauses/*.json','recovery_discarded_*.json','state-latest/*.json']:
                for path in folder.glob(pattern): archive.add(path,arcname=str(path.relative_to(OUT)))
        # 被用户资源变更替换的训练仍留在服务器；审查包收录其日志与元数据。
        for path in (OUT/'superseded').rglob('*'):
            if path.is_file() and not path.is_symlink() and path.suffix in ['.json','.csv','.txt']:
                archive.add(path,arcname=str(path.relative_to(OUT)))
    (OUT/'review_results.tar.gz.sha256').write_text(sha(OUT/'review_results.tar.gz')+'  review_results.tar.gz\n')


def switch_to_two_gpus():
    """用户要求由单卡改为双卡：保留旧运行，两臂重新从 E1 初始化。"""
    record_path=OUT/'audit/resource_change_two_gpu.json'
    if record_path.exists():
        assert layout()['world_size']==2
        return read(record_path)
    previous_layout=layout(); assert previous_layout['world_size']==1
    assert not list((OUT/'training').glob('seed*/F*/checkpoint_verification.json')), '已有正式完成结果，需另行审查资源变更'
    assert not (OUT/'statistics/P1/decision.json').exists(), '不得看过归因结果后重新配置'
    history=jobs(); affected=[j for j in history if j['action'] in ['smoke','train','advance']]
    states={str(j['id']):job_state(j) for j in affected}
    active=['PENDING','RUNNING','COMPLETING','CONFIGURING','SUSPENDED']
    assert not any(s in active for s in states.values()), ('先停止旧作业及其依赖',states)
    destination=OUT/'superseded/layout_1gpu_20261019'
    assert not destination.exists(), '旧运行已部分归档，请检查后再继续'
    destination.mkdir(parents=True)
    partial=[]
    for folder in (OUT/'training').glob('seed*/F*'):
        path=folder/'training_log.csv'
        if path.exists():
            import csv
            with path.open(newline='') as handle: rows=list(csv.DictReader(handle))
            partial.append(dict(run=str(folder.relative_to(OUT)),logged_successful_updates=int(rows[-1]['update']) if rows else 0,
                                log_sha256=sha(path),retained_in_new_budget=False))
    write(destination/'status_before.json',read(OUT/'status.json'))
    write(destination/'jobs_before.json',history)
    moved=[]
    for name in ['training','smoke','training_layout.json','audit/smoke_rng_audit.json']:
        source=OUT/name
        if source.exists():
            target=destination/name; target.parent.mkdir(parents=True,exist_ok=True)
            source.rename(target); moved.append(name)
    new_layout=dict(world_size=2,effective_batch=8,accumulation=4,
        selection='user explicitly requested two GPUs per training job; all arms restart from the same E1 checkpoint',
        previous_layout_archive=str(destination/'training_layout.json'))
    freeze(OUT/'training_layout.json',new_layout)
    for job in affected: job['superseded_by']=str(record_path)
    write(OUT/'jobs.json',history)
    record=dict(reason='user requested pause and two GPUs per training job',execution_commit=commit(),
        previous_layout=previous_layout,new_layout=new_layout,old_job_states=states,
        archive=str(destination),moved=moved,partial_training=partial,
        initialization='same frozen E1; restart both arms at update zero',
        targets_and_data_unchanged=True,updates_per_arm=CONFIG['updates'],
        previous_results_used_for_selection=False)
    freeze(record_path,record)
    state(preparation='REVALIDATING_TWO_GPUS',A='NOT_RUN',B='NOT_RUN',confirmation='NOT_RUN',LOCK='NOT_RUN',
          complete=False,resource_change=str(record_path),engineering_failure=None)
    return record


def advance(retry=False):
    # 最新用户范围优先；完成 B/C 后等待决定，不能落入自动八组分支。
    if (OUT/'audit/controlled_resume_scope.json').exists():
        from tools.e2a_attr_controlled import advance_controlled
        return advance_controlled(retry)
    # 用户修订范围后，所有旧 afterany 入口也必须遵守只训练 B 的新范围。
    if (OUT/'audit/historical_reuse_scope.json').exists():
        from tools.e2a_attr_reuse import advance_reuse
        return advance_reuse(retry)
    from tools.e2a_attr_stats import rng_audit,stage_a,factorial,lock_plan,final_lock
    assert read(OUT/'config.json')==CONFIG,'冻结协议发生变化'
    assert read(OUT/'audit/masks_complete.json')['passed']
    review_path=SOURCE/'tools/e2a_attr_mask_review.json'
    if not (OUT/'audit/mask_visual_review.json').exists() and review_path.exists():
        review=read(review_path)
        assert review['reviewed']==100 and review['review_indices']==read(OUT/'audit/masks_complete.json')['review_indices']
        assert review['archive_sha256']==sha(OUT/'mask_review.tar.gz')
        assert all(sha(OUT/'audit/mask_review'/name)==digest for name,digest in review['page_sha256'].items())
        freeze(OUT/'audit/mask_visual_review.json',review)
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
    parser.add_argument('--switch-to-two-gpus',action='store_true')
    parser.add_argument('--reuse-historical-e2a',action='store_true')
    parser.add_argument('--resume-controlled-c',action='store_true')
    parser.add_argument('--package-only',action='store_true');args=parser.parse_args()
    try:
        if args.switch_to_two_gpus: switch_to_two_gpus()
        if args.reuse_historical_e2a:
            from tools.e2a_attr_reuse import activate
            activate()
        if args.resume_controlled_c:
            from tools.e2a_attr_controlled import activate
            activate()
        if args.package_only: portable()
        else: advance(args.retry_failed); accounting()
    except Exception as error:
        state(engineering_failure=repr(error),complete=False); accounting(); raise
