"""以身份配对的 bootstrap、随机符号检验、Holm 校正及预注册阶段门槛。"""
import argparse
import itertools
import math
from statistics import NormalDist
import numpy as np
from tools.e2a_attr_common import *

KEYS=PRIMARY+APPEARANCE


def effect(values,seed=219042):
    values=np.asarray(values,float); values=values[np.isfinite(values)]; n=len(values)
    if not n: return dict(n=0,mean=None,sd=None,median=None,ci95=None,p=None)
    rng=np.random.default_rng(seed); count=CONFIG['bootstrap_samples']
    means=values[rng.integers(0,n,(count,n))].mean(1)
    null=(values*rng.choice([-1,1],(count,n))).mean(1)
    observed=float(values.mean())
    return dict(n=n,mean=observed,sd=float(values.std(ddof=1)) if n>1 else 0.,median=float(np.median(values)),
        ci95=np.quantile(means,[.025,.975]).tolist(),
        p=float((1+np.count_nonzero(np.abs(null)>=abs(observed)-1e-15))/(count+1)))


def records(stage,arm):
    assert read(OUT/'evaluation'/stage/arm/'evaluation_complete.json')['complete']
    return {r['id']:r for r in read(OUT/'evaluation'/stage/arm/'per_identity_metrics.json')}


def paired(stage,arm,control):
    a,b=records(stage,arm),records(stage,control); assert set(a)==set(b)
    # 每个身份的初始 latent、生成 seed 和输入文件须逐臂一致。
    for identity in a:
        metas=[read(OUT/'evaluation'/stage/name/'images'/(identity+'.json')) for name in [arm,control]]
        for key in ['seed','initial_latent_sha256','inputs']: assert metas[0][key]==metas[1][key]
    result={}
    for key in KEYS:
        values=[a[i][key]-b[i][key] for i in sorted(a) if a[i].get(key) is not None and b[i].get(key) is not None]
        result[key]=effect(values)
        assert result[key]['n']>0,('NO_VALID_PAIRED_METRIC',stage,arm,control,key)
    return result


def rng_audit(seed,factors,smoke=False):
    streams=[read(train_folder(seed,f,smoke)/'seed_and_rng_manifest.json') for f in factors]
    assert all(value==streams[0] for value in streams[1:]),'PAIRED_TRAINING_RNG_MISMATCH'
    counts=[read(train_folder(seed,f,smoke)/'optimizer_update_count.json') for f in factors]
    assert all(value==counts[0] for value in counts[1:])
    assert counts[0]['actual_successful_updates']==(8 if smoke else 28365)
    return dict(passed=True,seed=seed,factors=factors,actual_updates=counts[0]['actual_successful_updates'],
        world_size=counts[0]['world_size'],all_rank_streams_equal=True)


def holm(tests):
    ordered=sorted(tests,key=lambda item:item[1]['p'] if item[1]['p'] is not None else 1.)
    previous=0.; total=len(ordered)
    for index,(_,value) in enumerate(ordered):
        previous=max(previous,min(1.,(total-index)*(value['p'] if value['p'] is not None else 1.)))
        value['p_holm']=previous


def summaries(stage,arms):
    values={arm:{key:effect([r[key] for r in records(stage,arm).values() if r.get(key) is not None])
                       for key in KEYS} for arm in arms}
    write(OUT/'statistics'/stage/'summaries.json',values)
    return values


def tails(stage,arm,control):
    a,b=records(stage,arm),records(stage,control); result=[]
    for identity in sorted(a):
        row=dict(id=identity,arm=arm,control=control)
        for key in KEYS:
            row[key+'_delta']=a[identity][key]-b[identity][key] if a[identity].get(key) is not None and b[identity].get(key) is not None else None
        row['primary_adverse']=any(row[key+'_delta'] is not None and DIRECTION[key]*row[key+'_delta'] < -CONFIG['noninferiority'][key] for key in PRIMARY)
        result.append(row)
    result.sort(key=lambda r:DIRECTION['edge_f1']*r['edge_f1_delta'] if r['edge_f1_delta'] is not None else -999)
    csv_write(OUT/'evaluation'/stage/arm/'tail_cases.csv',result)
    return result


def stage_a():
    audit=rng_audit(42,['000','111'])
    arms=['E1','s42_F000','s42_F111']; summaries('P1',arms)
    deltas={name:paired('P1',a,b) for name,a,b in [('B-A',arms[1],arms[0]),('C-B',arms[2],arms[1]),('C-A',arms[2],arms[0])]}
    main=deltas['C-B']
    no_benefit=all(DIRECTION[k]*main[k]['mean']<=0 for k in KEYS)
    adverse=lambda key: max(DIRECTION[key]*v for v in main[key]['ci95'])<0
    clearly_worse=sum(adverse(k) for k in PRIMARY)>=3 and all(adverse(k) for k in APPEARANCE)
    proceed=not(no_benefit or clearly_worse)
    decision=dict(continue_to_B=proceed,no_favourable_direction=no_benefit,clearly_worse=clearly_worse,
        rationale='mixed or favourable paired directions: factorial attribution warranted' if proceed else 'preregistered stop; engineering audit required before closure',
        rng_audit=audit,comparisons=deltas,rule_frozen_before_outputs=CONFIG)
    write(OUT/'statistics/P1/paired_bootstrap.json',deltas); write(OUT/'statistics/P1/decision.json',decision)
    for arm in arms:
        write(OUT/'evaluation/P1'/arm/'paired_bootstrap.json',deltas)
        write(OUT/'evaluation/P1'/arm/'factorial_effects.json',dict(applicable=False,reason='stage A; not an eight-arm factorial'))
    tails('P1',arms[2],arms[1]); state(A='COMPLETE',B='READY' if proceed else 'STOPPED_BY_PROTOCOL')
    return decision


def factorial():
    rng=rng_audit(43,FACTORS)
    arms=['s43_F'+f for f in FACTORS]; summaries('P2',arms)
    table={f:records('P2','s43_F'+f) for f in FACTORS}
    identities=sorted(table['000']); assert all(set(t)==set(identities) for t in table.values())
    result={}; tests=[]
    for subset in [(0,),(1,),(2,),(0,1),(0,2),(1,2),(0,1,2)]:
        name='x'.join(['body','boundary','outside'][i] for i in subset); result[name]={}
        coefficients={f:float(np.prod([2*int(f[i])-1 for i in subset]))/4. for f in FACTORS}
        for key in KEYS:
            values=[sum(coefficients[f]*table[f][identity][key] for f in FACTORS)
                for identity in identities if all(table[f][identity].get(key) is not None for f in FACTORS)]
            value=effect(values); value['coefficients']=coefficients
            result[name][key]=value; tests.append((name+'/'+key,value))
    holm(tests)
    comparisons={f:paired('P2','s43_F'+f,'s43_F000') for f in FACTORS if f!='000'}
    score=lambda f:sum(DIRECTION[k]*comparisons[f][k]['mean']/CONFIG['meaningful_gain'][k] for k in KEYS)/len(KEYS)
    best_single=max(['100','010','001'],key=score)
    supported_pairs=[]
    for name,factor in [('bodyxboundary','110'),('bodyxoutside','101'),('boundaryxoutside','011')]:
        if any(value['p_holm']<.05 for value in result[name].values()): supported_pairs.append(factor)
    selected=sorted(set(['000','111',best_single]+supported_pairs))
    candidates=[f for f in FACTORS if f!='000' and any(DIRECTION[k]*comparisons[f][k]['mean']>0 for k in KEYS)]
    # 只将待确认的组带入后续候选，避免为未重复训练的组宣称稳定性。
    candidate=max([f for f in selected if f in candidates],key=score) if any(f in candidates for f in selected) else '111'
    selection=dict(arms=selected,best_single=best_single,supported_interaction_pairs=supported_pairs,
        provisional_candidate=candidate,seeds=[42,43,44],rng_audit=rng,
        score='mean oriented delta / prespecified meaningful gain, all six metrics',
        any_favourable_direction=bool(candidates),single_seed_exploratory=True)
    write(OUT/'statistics/P2/factorial_effects.json',result); write(OUT/'statistics/P2/paired_bootstrap.json',comparisons)
    freeze(OUT/'selection/confirmation.json',selection)
    for arm in arms:
        write(OUT/'evaluation/P2'/arm/'factorial_effects.json',result)
        write(OUT/'evaluation/P2'/arm/'paired_bootstrap.json',comparisons)
        if arm!='s43_F000': tails('P2',arm,'s43_F000')
    state(B='COMPLETE',confirmation='READY'); return selection


def lock_plan():
    selection=read(OUT/'selection/confirmation.json'); factors=selection['arms']
    for seed in [42,43,44]: rng_audit(seed,factors)
    deltas={f:{k:[] for k in KEYS} for f in factors if f!='000'}
    seed_effects={}
    for f in deltas:
        seed_effects[f]={}
        for seed in [42,43,44]: seed_effects[f][str(seed)]=paired('P2','s%d_F%s'%(seed,f),'s%d_F000'%seed)
        for key in KEYS:
            tables=[(records('P2','s%d_F%s'%(seed,f)),records('P2','s%d_F000'%seed)) for seed in [42,43,44]]
            for identity in sorted(tables[0][0]):
                if all(a[identity].get(key) is not None and b[identity].get(key) is not None for a,b in tables):
                    deltas[f][key].append(float(np.mean([a[identity][key]-b[identity][key] for a,b in tables])))
    summaries_={f:{k:effect(v) for k,v in values.items()} for f,values in deltas.items()}
    score=lambda f:sum(DIRECTION[k]*summaries_[f][k]['mean']/CONFIG['meaningful_gain'][k] for k in KEYS)/len(KEYS)
    safe=[f for f in summaries_ if all(DIRECTION[k]*summaries_[f][k]['mean']>=-CONFIG['noninferiority'][k] for k in PRIMARY)]
    candidate=max(safe or list(summaries_),key=score)
    positive=[k for k in PRIMARY if DIRECTION[k]*summaries_[candidate][k]['mean']>0]
    primary=max(positive or APPEARANCE,key=lambda k:DIRECTION[k]*summaries_[candidate][k]['mean']/CONFIG['meaningful_gain'][k])
    sd=summaries_[candidate][primary]['sd']; delta=CONFIG['meaningful_gain'][primary]
    z=NormalDist().inv_cdf(1-CONFIG['family_alpha']/len(PRIMARY)/2)+NormalDist().inv_cdf(CONFIG['power'])
    required=max(96,math.ceil((z*sd/delta)**2))
    pool=read(OUT/'splits/LOCK_pool.json'); n=min(required,len(pool))
    arms=['E1']+['s%d_F%s'%(seed,f) for seed in [42,43,44] for f in ['000',candidate]]
    protocol=dict(candidate=candidate,primary_metric=primary,meaningful_gain=delta,
        requested_power=CONFIG['power'],required_n=required,actual_n=n,power_limited=n<required,
        planning='normal approximation from paired identity variance averaged over three training seeds',
        rows=pool[:n],arms=list(dict.fromkeys(arms)),seed_effects=seed_effects,
        identity_averaged_effects=summaries_,candidate_frozen_before_LOCK=True)
    freeze(OUT/'selection/LOCK_protocol.json',protocol); state(confirmation='COMPLETE',LOCK='READY')
    return protocol


def final_lock():
    protocol=read(OUT/'selection/LOCK_protocol.json'); candidate=protocol['candidate']
    summaries('LOCK',protocol['arms']); deltas={}
    for seed in [42,43,44]: deltas[str(seed)]=paired('LOCK','s%d_F%s'%(seed,candidate),'s%d_F000'%seed)
    aggregate={}
    for key in KEYS:
        pairs=[(records('LOCK','s%d_F%s'%(seed,candidate)),records('LOCK','s%d_F000'%seed)) for seed in [42,43,44]]
        values=[np.mean([a[i][key]-b[i][key] for a,b in pairs]) for i in sorted(pairs[0][0])
                if all(a[i].get(key) is not None and b[i].get(key) is not None for a,b in pairs)]
        aggregate[key]=effect(values)
    holm(list(aggregate.items()))
    key=protocol['primary_metric']; value=aggregate[key]
    oriented_ci=sorted(DIRECTION[key]*v for v in value['ci95'])
    safety={k:min(DIRECTION[k]*v for v in aggregate[k]['ci95'])>=-CONFIG['noninferiority'][k] for k in PRIMARY}
    replicated=all(min(DIRECTION[key]*v for v in deltas[str(seed)][key]['ci95'])>0 for seed in [42,43,44])
    confirmed=oriented_ci[0]>0 and value['p_holm']<.05 and all(safety.values()) and replicated
    result=dict(complete=True,candidate=candidate,primary_metric=key,paired_by_training_seed=deltas,
        identity_averaged_effects=aggregate,structural_noninferiority=safety,improvement_confirmed=confirmed,
        training_seeds=3,primary_replicated_in_all_three_seeds=replicated,
        meaningful_gain_ci_supported=oriented_ci[0]>=protocol['meaningful_gain'],
        independent_identities=protocol['actual_n'],power_limited=protocol['power_limited'],
        motif_identity_proven=False,pretrained_exposure_unknown=True)
    write(OUT/'statistics/LOCK/paired_bootstrap.json',result); write(OUT/'final_result.json',result)
    for arm in protocol['arms']:
        write(OUT/'evaluation/LOCK'/arm/'paired_bootstrap.json',result)
        write(OUT/'evaluation/LOCK'/arm/'factorial_effects.json',read(OUT/'statistics/P2/factorial_effects.json'))
    state(LOCK='COMPLETE',complete=True,improvement_confirmed=confirmed); return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['A','B','lock-plan','LOCK','smoke'])
    args=parser.parse_args()
    actions={'A':stage_a,'B':factorial,'lock-plan':lock_plan,'LOCK':final_lock,
             'smoke':lambda:write(OUT/'audit/smoke_rng_audit.json',rng_audit(42,['000','111'],True))}
    print(actions[args.action](),flush=True)
