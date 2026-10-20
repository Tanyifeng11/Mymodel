"""用户修订：只续训 B，复用历史 E2a 权重做同集筛查，不自动进入八组。"""
import csv
import os
import re
from tools.e2a_attr_common import *

SCOPE = OUT/'audit/historical_reuse_scope.json'
HISTORICAL = PROJECT/'output/phase1_e2a_region_e5/checkpoint-final/joint_model.pt'
LIMITATION = ('Historical E2a and B do not share audited training RNG or hardware layout. '
              'This is a historical-checkpoint screening comparison, not a strict paired '
              'training ablation or proof of causal independent benefit.')


def activate():
    from tools.e2a_attr_dispatch import jobs,job_state,layout
    if SCOPE.exists(): return read(SCOPE)
    assert read(OUT/'config.json')==CONFIG
    assert layout()['world_size']==2
    history=jobs(); current=int(os.environ.get('SLURM_JOB_ID','0'))
    blocked=[j for j in history if j['id']!=current and not j.get('superseded_by')
             and (j['key']=='train_s42_F111' or j['action']=='advance')
             and job_state(j) in ['PENDING','RUNNING','COMPLETING','CONFIGURING']]
    assert not blocked,('STOP_OLD_C_AND_ADVANCE_FIRST',blocked)
    assert not (OUT/'statistics/P1/decision.json').exists()
    assert not any(j['key'].startswith('train_s43_') for j in history)
    c=train_folder(42,'111'); b=train_folder(42,'000')
    progress=read(b/'state-latest/progress.json')
    assert progress['world_size']==2 and progress['seed']==42
    assert read(OUT/'audit/mask_visual_review.json')['passed']
    assert read(OUT/'audit/smoke_rng_audit.json')['passed']
    # mmap 只读取旧模型的元数据，不修改、复制或重新保存权重。
    import torch
    payload=torch.load(HISTORICAL,map_location='cpu',mmap=True,weights_only=False)
    meta=payload['meta']
    for key,value in dict(lambda_region_texture=.1,lambda_boundary=.05,lambda_leak=.1,
                          layer_group_enabled=1,texture_condition_mode='token',
                          texture_num_tokens=16,width=384,height=512).items():
        assert meta[key]==value,('WRONG_HISTORICAL_E2A',key,meta.get(key))
    assert all(k in payload for k in ['unet','ref_unet','texture_adapter','bf_texture_conditioner'])
    del payload
    manifest=PROJECT/'output/phase1_e2a_region_e5/experiment_manifest.json'
    wandb=PROJECT/'wandb/run-20260617_225130-ee7e5n00/files/wandb-metadata.json'
    metadata=read(wandb); args=metadata['args']
    assert args==read(OUT/'manifests/historical_command.json')
    assert Path(args[args.index('--gam_init_ckpt')+1]).resolve()==E1.resolve()
    provenance=dict(experiment_manifest=read(manifest),wandb_metadata=metadata,
                    source_sha256={str(p):sha(p) for p in [manifest,wandb]},checkpoint_meta=meta)
    freeze(OUT/'audit/historical_E2a_provenance.json',provenance)
    steps=re.fullmatch(r'checkpoint-(\d+)',HISTORICAL.resolve().parent.name)
    with (c/'training_log.csv').open(newline='') as handle: c_rows=list(csv.DictReader(handle))
    freeze(OUT/'audit/scope_before_historical_reuse.json',dict(status=read(OUT/'status.json'),jobs=history))
    record=dict(amendment='user_20261020_reuse_historical_E2a_only_train_B',execution_commit=commit(),
        user_request='pause new C; reuse existing E1/E2a; train only B; decide on eight arms after screening',
        original_config_preserved=True,stage='P1',arms=['E1','s42_F000','historical_E2a'],
        B_resume=progress,B_target_updates=28365,
        C_paused=dict(last_logged_update=int(c_rows[-1]['update']),
                      complete_state=read(c/'state-latest/progress.json'),files_preserved=True,
                      jobs=[dict(id=j['id'],state=job_state(j)) for j in history
                            if j['key']=='train_s42_F111' and not j.get('superseded_by')]),
        historical_checkpoint=dict(path=str(HISTORICAL),resolved=str(HISTORICAL.resolve()),
            bytes=HISTORICAL.stat().st_size,sha256=sha(HISTORICAL),
            nominal_updates_from_directory=int(steps.group(1)) if steps else None,
            effective_successful_updates_audited=False),
        training_rng_matched=False,causal_independent_benefit_confirmable=False,limitation=LIMITATION,
        screening_rule='original stage-A directional stop rule; recommendation only; no automatic factorial',
        factorial_requires_new_user_decision=True,LOCK_opened=False)
    freeze(SCOPE,record)
    state(scope=record['amendment'],A='TRAINING_B_ONLY',C='PAUSED_BY_USER',
          B='AWAIT_USER_DECISION',confirmation='NOT_REQUESTED',LOCK='NOT_OPENED',
          complete=False,engineering_failure=None)
    return record


def audit_b():
    folder=train_folder(42,'000'); config=read(folder/'config.json')
    count=read(folder/'optimizer_update_count.json')
    assert read(folder/'checkpoint_verification.json')['passed']
    assert count['actual_successful_updates']==28365 and count['skipped_updates']==0
    assert config['world_size']==2 and config['effective_batch']==8 and config['accumulation']==4
    streams=read(folder/'seed_and_rng_manifest.json')['streams']
    assert sorted(s['rank'] for s in streams)==[0,1]
    assert all(s['actual_updates']==28365 and s['microbatches']==28365*4 for s in streams)
    with (folder/'training_log.csv').open(newline='') as handle: rows=list(csv.DictReader(handle))
    assert len(rows)==28365 and all(int(r['update'])==i+1 for i,r in enumerate(rows))
    assert rows[-1]['rng_rank0']==next(s['rng_chain'] for s in streams if s['rank']==0)
    result=dict(passed=True,actual_updates=28365,skipped_updates=0,world_size=2,
        all_rank_streams_recorded=True,continuous_update_log=True,
        paired_historical_training_rng_available=False)
    write(OUT/'audit/B_historical_reuse_training.json',result)
    return result


def screening():
    from tools.e2a_attr_stats import paired,summaries,tails,holm
    scope=read(SCOPE); arms=scope['arms']; audit=audit_b()
    values=summaries('P1',arms)
    deltas={name:paired('P1',a,b) for name,a,b in [
        ('B-E1',arms[1],arms[0]),('historical_E2a-B',arms[2],arms[1]),
        ('historical_E2a-E1',arms[2],arms[0])]}
    main=deltas['historical_E2a-B']; holm(list(main.items()))
    no_benefit=all(DIRECTION[k]*main[k]['mean']<=0 for k in PRIMARY+APPEARANCE)
    adverse=lambda k:max(DIRECTION[k]*v for v in main[k]['ci95'])<0
    clearly_worse=sum(adverse(k) for k in PRIMARY)>=3 and all(adverse(k) for k in APPEARANCE)
    result=dict(scope=scope['amendment'],complete=True,B_training_audit=audit,
        comparisons=deltas,summaries=values,no_favourable_direction=no_benefit,clearly_worse=clearly_worse,
        review_factorial_investment=not(no_benefit or clearly_worse),
        recommendation_only=True,automatic_factorial_authorized=False,
        historical_training_rng_matched=False,independent_causal_benefit_proven=False,
        limitation=LIMITATION,motif_identity_proven=False,
        statistical_unit='24 paired held-out identities; one checkpoint per arm; not training-seed replication',
        thresholds=CONFIG['noninferiority'],screening_rule=scope['screening_rule'])
    write(OUT/'statistics/historical_reuse/paired_bootstrap.json',deltas)
    write(OUT/'statistics/historical_reuse/decision.json',result)
    tails('P1',arms[2],arms[1]); write(OUT/'final_result.json',result)
    state(A='COMPLETE_HISTORICAL_REUSE',B='AWAIT_USER_DECISION',C='PAUSED_BY_USER',
          confirmation='NOT_REQUESTED',LOCK='NOT_OPENED',complete=True)
    return result


def advance_reuse(retry=False):
    from tools.e2a_attr_dispatch import submit,watch,portable,layout
    scope=read(SCOPE); assert scope['factorial_requires_new_user_decision']
    assert read(OUT/'config.json')==CONFIG and layout()['world_size']==2
    if not (train_folder(42,'000')/'checkpoint_verification.json').exists():
        batch=[submit('train_s42_F000','train',['--seed','42','--factor','000'],2,retry=retry)]
        watch(batch); state(A='TRAINING_B_ONLY',C='PAUSED_BY_USER'); return
    audit_b()
    missing=[a for a in scope['arms'] if not (OUT/'evaluation/P1'/a/'evaluation_complete.json').exists()]
    if missing:
        batch=[submit('eval_P1_'+a,'eval',['--stage','P1','--arm',a],1,retry=retry) for a in missing]
        watch(batch); state(A='EVALUATING_HISTORICAL_REUSE'); return
    if not (OUT/'statistics/historical_reuse/decision.json').exists(): screening()
    portable()
