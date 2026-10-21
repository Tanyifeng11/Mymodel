"""身份单位统计；主检验仅三项 IoU 的 Holm 族，其余六指标强制完整报告。"""
import math
import numpy as np
from tools.e2a_sup_common import *
from tools.e2a_attr_common import PRIMARY, APPEARANCE, DIRECTION, CONFIG as OLD_CONFIG
from tools.e2a_attr_stats import holm

KEYS=PRIMARY+APPEARANCE


def effect(values):
    values=np.asarray(values,dtype=float);values=values[np.isfinite(values)];n=len(values)
    if not n:return dict(n=0,mean=None,ci95=None,p=None)
    rng=np.random.default_rng(CONFIG['statistics_seed']);count=10000
    means=values[rng.integers(0,n,(count,n))].mean(1)
    null=(values*rng.choice([-1,1],size=(count,n))).mean(1)
    return dict(n=n,mean=float(values.mean()),median=float(np.median(values)),
        ci95=np.quantile(means,[.025,.975]).tolist(),
        p=float((1+(np.abs(null)>=abs(values.mean())-1e-15).sum())/(count+1)))


def records(arm):
    path=ROOT/'evaluation/P2b'/arm
    assert read(path/'evaluation_complete.json')['complete']
    return {row['id']:row for row in read(path/'per_identity_metrics.json')}


def compare(arm,control):
    a,b=records(arm),records(control);assert set(a)==set(b) and len(a)==96
    for identity in a:
        m=[read(ROOT/'evaluation/P2b'/name/'images'/(identity+'.json')) for name in [arm,control]]
        assert all(m[0][k]==m[1][k] for k in ['seed','initial_latent_sha256','inputs'])
    result={};tails=[]
    for key in KEYS:
        ids=[i for i in sorted(a) if a[i].get(key) is not None and b[i].get(key) is not None]
        delta=np.array([a[i][key]-b[i][key] for i in ids])
        value=effect(delta);value['valid_denominator']=len(ids);value['missing']=96-len(ids)
        oriented=delta*DIRECTION[key];n=max(1,len(ids));margin=OLD_CONFIG['noninferiority'][key]
        value['win_fraction']=float((oriented>0).sum()/n);value['tie_fraction']=float((oriented==0).sum()/n)
        value['loss_fraction']=float((oriented<0).sum()/n)
        if value['ci95']:
            bounds=sorted(DIRECTION[key]*v for v in value['ci95'])
            value['noninferiority']=dict(margin=margin,passed=bounds[0]>=-margin,clearly_failed=bounds[1]<-margin)
        worst=np.argsort(oriented)[:max(1,math.ceil(len(ids)*.1))]
        value['worst10pct_mean_oriented_delta']=float(oriented[worst].mean()) if len(ids) else None
        for j in worst: tails.append(dict(arm=arm,control=control,metric=key,id=ids[j],delta=float(delta[j]),
            beyond_noninferiority=bool(oriented[j]<-margin)))
        result[key]=value
    csv_write(ROOT/'statistics/P2b'/('%s_minus_%s_tails.csv'%(arm,control)),tails)
    return result


def summarize():
    comparisons={arm:compare('C',arm) for arm in ARMS}
    holm([(arm,comparisons[arm]['iou']) for arm in ARMS])
    summaries={arm:{key:effect([r[key] for r in records(arm).values() if r.get(key) is not None])
                       for key in KEYS} for arm in ['B','C']+ARMS}
    limitations=['single training seed42; P2b method comparison, not final LOCK',
        'T1 is AnyDressing-GTL adapted, T2 is DINO training adaptation, neither is full-paper replication',
        'C and T3 differ in both boundary separation and rest weighting; unique boundary mechanism not identified',
        'DISTS/DINO/SD pretraining exposure unknown; metrics do not prove motif identity']
    decision=dict(comparisons=comparisons,summaries=summaries,primary_family=CONFIG['primary_family'],
        exploratory_C_minus_B=compare('C','B'),limitations=limitations,
        additional_training_started=False,LOCK_opened=False,
        recommendation_requires_all_safety_metrics_and_swap_review=True)
    write(ROOT/'statistics/P2b/decision.json',decision)
    print('PRIMARY_IOU_HOLM',[(a,comparisons[a]['iou']['mean'],comparisons[a]['iou']['p_holm']) for a in ARMS])
    # 人工盲评、swap 对应与资源账本/尾例审查未结束前不宣布整个协议完成。
    status(stage='STATISTICS_COMPLETE_QUALITATIVE_REVIEW_PENDING',complete=False)


if __name__=='__main__':summarize()
