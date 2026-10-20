"""用户恢复 C：保留完整状态续跑，只重测 B/C，八组等待新的决定。"""
import shutil
from tools.e2a_attr_common import *

SCOPE = OUT/'audit/controlled_resume_scope.json'
STAGE = 'P1_controlled'
ARMS = ['s42_F000','s42_F111']
ACTIVE = ['PENDING','RUNNING','COMPLETING','CONFIGURING','SUSPENDED']
LOG_KEYS = ['update','lr','microbatches_per_rank','rng_rank0']


def log_rows(folder):
    with (folder/'training_log.csv').open(newline='') as handle:
        return list(csv.DictReader(handle))


def audit_pair_prefix(limit):
    rows=[log_rows(train_folder(42,f)) for f in ['000','111']]
    assert all(len(r)>=limit for r in rows)
    for index in range(limit):
        assert all(int(r[index]['update'])==index+1 for r in rows)
        assert all(rows[0][index][k]==rows[1][index][k] for k in LOG_KEYS), ('PAIRED_LOG_MISMATCH',index+1)
    return dict(passed=True,updates=limit,equal_fields=LOG_KEYS)


def activate():
    from tools.e2a_attr_dispatch import jobs,job_state,layout
    from tools.e2a_attr_reuse import audit_b
    if SCOPE.exists(): return read(SCOPE)
    assert read(OUT/'config.json')==CONFIG and layout()['world_size']==2
    history=jobs(); current=int(os.environ.get('SLURM_JOB_ID','0'))
    blocked=[j for j in history if j['id']!=current and not j.get('superseded_by')
             and (j['action'] in ['train','advance','eval']) and job_state(j) in ACTIVE]
    assert not blocked,('ACTIVE_EXPERIMENT_JOBS',blocked)
    assert not any(j['key'].startswith('train_s43_') for j in history)
    audit_b()
    b,c=[train_folder(42,f) for f in ['000','111']]
    configs=[read(p/'config.json') for p in [b,c]]
    for key in ['seed','world_size','effective_batch','accumulation','updates','scheduler','amp','source_sha256']:
        assert configs[0][key]==configs[1][key],('CONFIG_MISMATCH',key)
    assert all(sha(SOURCE/p)==h for p,h in configs[1]['source_sha256'].items())
    for p,h in read(OUT/'checkpoint_manifest.json').items(): assert sha(p)==h['sha256']
    assert not (c/'training_complete.json').exists()
    progress=read(c/'state-latest/progress.json'); step=progress['updates']
    assert progress['seed']==42 and progress['world_size']==2 and progress['microbatches']==step*4
    assert 0<step<CONFIG['updates']
    files=['progress.json','optimizer.bin','custom_checkpoint_0.pkl','scaler.pt']
    files+=['pytorch_model%s.bin'%('' if i==0 else '_'+str(i)) for i in range(5)]
    files+=['random_states_%d.pkl'%i for i in range(2)]+['stream_rank%d.json'%i for i in range(2)]
    assert all((c/'state-latest'/name).stat().st_size>0 for name in files)
    rows=log_rows(c); prefix=audit_pair_prefix(step)
    assert rows[step-1]['rng_rank0']==read(c/'state-latest/stream_rank0.json')['rng_chain']
    previous=[j for j in history if j['key']=='train_s42_F111' and not j.get('superseded_by')][-1]
    assert job_state(previous).startswith('CANCELLED'), ('EXPECTED_USER_CANCELLED_C',previous)
    destination=OUT/'audit/controlled_resume'; destination.mkdir(parents=True,exist_ok=True)
    # 取消后未完整保存的 377 次更新需重放；旧日志完整留档，而非静默丢弃。
    backup=destination/'C_training_log_before_resume.csv'
    if not backup.exists(): shutil.copy2(c/'training_log.csv',backup)
    assert sha(backup)==sha(c/'training_log.csv')
    freeze(destination/'before.json',dict(status=read(OUT/'status.json'),jobs=history,
        progress=progress,stream_states=[read(c/'state-latest'/('stream_rank%d.json'%i)) for i in range(2)]))
    old_archive=OUT/'superseded/historical_reuse_20261020'; old_archive.mkdir(parents=True,exist_ok=True)
    for name in ['review_results.tar.gz','review_results.tar.gz.sha256','final_result.json']:
        if (OUT/name).exists() and not (old_archive/name).exists(): shutil.copy2(OUT/name,old_archive/name)
    record=dict(amendment='user_20261020_resume_C_retest_B_C_before_factorial',execution_commit=commit(),
        user_request='resume C; freshly evaluate B and C; decide whether to enter eight-arm training afterwards',
        B_checkpoint=read(b/'checkpoint_verification.json'),C_resume=progress,
        C_previous_job=previous['id'],C_previous_job_state=job_state(previous),
        C_last_logged_update=int(rows[-1]['update']),C_replay_updates=len(rows)-step,
        C_log_backup=str(backup),C_log_backup_sha256=sha(backup),paired_prefix_audit=prefix,
        resume_components=files,original_config_preserved=True,
        stage=STAGE,split='P1',split_sha256=sha(OUT/'splits/P1.json'),arms=ARMS,
        prior_historical_screening_seen=True,selection_bias_note='Same P1 already inspected in historical screening; this is not an untouched final holdout.',
        factorial_requires_new_user_decision=True,automatic_factorial_authorized=False,LOCK_opened=False)
    freeze(SCOPE,record)
    state(scope=record['amendment'],A='TRAINING_CONTROLLED_C',C='RESUMING',B='AWAIT_USER_DECISION',
          confirmation='NOT_REQUESTED',LOCK='NOT_OPENED',complete=False,engineering_failure=None)
    return record


def training_audit():
    from tools.e2a_attr_stats import rng_audit
    from tools.e2a_attr_reuse import audit_b
    audit_b(); c=train_folder(42,'111')
    assert read(c/'checkpoint_verification.json')['passed']
    result=dict(passed=True,rng_audit=rng_audit(42,['000','111']),
                continuous_paired_logs=audit_pair_prefix(CONFIG['updates']))
    write(OUT/'audit/controlled_training_pair.json',result)
    return result


def screening():
    from tools.e2a_attr_stats import paired,summaries,records,tails,holm
    scope=read(SCOPE); audit=training_audit()
    values=summaries(STAGE,ARMS); main=paired(STAGE,ARMS[1],ARMS[0]); holm(list(main.items()))
    assert all(v['n']==CONFIG['p1_n'] for v in main.values())
    no_benefit=all(DIRECTION[k]*main[k]['mean']<=0 for k in PRIMARY+APPEARANCE)
    adverse=lambda k:max(DIRECTION[k]*v for v in main[k]['ci95'])<0
    clearly_worse=sum(adverse(k) for k in PRIMARY)>=3 and all(adverse(k) for k in APPEARANCE)
    old,new=[records(s,ARMS[0]) for s in ['P1',STAGE]]
    repeat=[]
    for identity in sorted(new):
        metas=[read(OUT/'evaluation'/s/ARMS[0]/'images'/(identity+'.json')) for s in ['P1',STAGE]]
        for k in ['seed','initial_latent_sha256','inputs']: assert metas[0][k]==metas[1][k]
        repeat.append(dict(id=identity,output_sha256_equal=metas[0]['output_sha256']==metas[1]['output_sha256'],
            metric_delta={k:new[identity][k]-old[identity][k] for k in PRIMARY+APPEARANCE}))
    write(OUT/'audit/B_fresh_evaluation_repeatability.json',repeat)
    result=dict(scope=scope['amendment'],complete=True,stage=STAGE,comparisons={'C-B':main},summaries=values,
        training_audit=audit,no_favourable_direction=no_benefit,clearly_worse=clearly_worse,
        review_factorial_investment=not(no_benefit or clearly_worse),recommendation_only=True,
        automatic_factorial_authorized=False,training_seed_count=1,training_rng_matched=True,
        statistical_unit='24 paired P1 identities; one matched training seed; no multi-seed confirmation',
        thresholds=CONFIG['noninferiority'],prior_historical_screening_seen=True,
        selection_bias_note=scope['selection_bias_note'],motif_identity_proven=False)
    tails(STAGE,ARMS[1],ARMS[0])
    write(OUT/'statistics'/STAGE/'decision.json',result); write(OUT/'controlled_result.json',result)
    state(A='COMPLETE_CONTROLLED',C='COMPLETE',B='AWAIT_USER_DECISION',complete=True,engineering_failure=None)
    return result


def advance_controlled(retry=False):
    from tools.e2a_attr_dispatch import submit,watch,portable,jobs,layout
    scope=read(SCOPE)
    assert scope['factorial_requires_new_user_decision'] and not scope['automatic_factorial_authorized']
    assert read(OUT/'config.json')==CONFIG and layout()['world_size']==2
    assert sha(OUT/'splits/P1.json')==scope['split_sha256']
    c=train_folder(42,'111')
    if not (c/'checkpoint_verification.json').exists():
        attempts=[j for j in jobs() if j['key']=='train_s42_F111' and not j.get('superseded_by')]
        # 仅首次恢复用户主动取消的作业；后续工程失败仍需审查。
        first_resume=bool(attempts and attempts[-1]['id']==scope['C_previous_job'])
        job=submit('train_s42_F111','train',['--seed','42','--factor','111'],2,retry=retry or first_resume)
        watch([job]); state(A='TRAINING_CONTROLLED_C',C='TRAINING',complete=False); return
    training_audit()
    missing=[a for a in ARMS if not (OUT/'evaluation'/STAGE/a/'evaluation_complete.json').exists()]
    if missing:
        batch=[submit('eval_'+STAGE+'_'+a,'eval',['--stage',STAGE,'--arm',a],1,retry=retry) for a in missing]
        watch(batch); state(A='EVALUATING_CONTROLLED',C='TRAINED',complete=False); return
    if not (OUT/'controlled_result.json').exists(): screening()
    portable()
