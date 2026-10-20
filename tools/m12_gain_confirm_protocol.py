"""独立确认的固定协议与身份级统计；不依赖 GPU 或历史结果目录。"""
import hashlib
from pathlib import Path
import numpy as np
from tools.m54_protocol import read, write, sha, commit

PROJECT = Path('/share/home/u2515283058/Mymodel')
RUN = PROJECT/'output_eval/m12_gain_independent_confirm_20261020'
OLD = PROJECT/'output_eval/m12_gain_dev32_20261020'
M12 = PROJECT/'output_eval/m12_fixed_dual_path_20261018'
M54 = PROJECT/'output_eval/m54_b1_screen_20261017'
GAIN_CHECKPOINT = M54/'m10_gain_r1/training/checkpoints/step2400.pt'
ARMS = ['E5', 'B1', 'M10', 'M10_GAIN', 'M12', 'M12_GAIN']
SEEDS = [42, 43, 44]
CORE = ['lpips_gt', 'contour_f1', 'iou', 'leak']
PLAN_SHA = 'ce2f17b7d5d8278c1d2ad0293478de047b35e0beed8b4bf09cfdeeac0cdd4524'
SMOKE_IDS = ['174397224', '193488637', '53887902', '53328647']
SETTINGS = dict(identities=96, seeds=SEEDS, arms=ARMS, main_candidate='M12_GAIN',
    scientific_images=1728, new_training_updates=0, alpha_B=.25, alpha_M=.5,
    inference=dict(size=[384,512],steps=50,cfg=7.,sampler='DDIM',sketch=.6,texture=1.,ipa=1.),
    bootstrap=dict(unit='identity',seed=32042,draws=10000,aggregation='mean of all three seeds; any missing seed metric => identity metric NA'),
    extreme='any seed delta F1 < -0.15 counts identity; three-seed mean extreme also reported',
    margins=dict(f1=-.020,iou=-.010,leak=.005,extreme=-.15,retention=.5),
    primary='M12_GAIN vs E5: all six gates required; no waiver',
    key_secondary='M12_GAIN vs B1 F1: 95% lower > 0; interpreted after primary gate success',
    other_comparisons='preregistered secondary/exploratory 95% intervals; no multiplicity-adjusted superiority claims',
    minimum_effects=dict(lpips_vs_E5=.005,lpips_vs_M10_GAIN=.002,f1_vs_B1=.015),
    sample_decision='96 identities approved before generation; small incremental LPIPS comparison exploratory; no result-dependent expansion',
    local_verification=dict(lpips_cpu_gpu_atol=1e-6,regional_atol=1e-12,
        formal_device='server GPU',sensitivity='recompute every LPIPS on server CPU and local CPU; preserve strict failure, compare CPU path, recalculate LPIPS gates/retention and comparisons without replacing formal GPU values'),
    profiling=dict(smoke_identity_indices=[0,1,2,3],warmup=1,measurements=4,seed=42,
        scope='native production pipeline including conditional Writer; excludes input/reference setup and PNG serialization',
        process='one fresh process per arm, diagnostics disabled, no preceding other-arm model'),
    pattern_ground_truth='NOT_RUN',independent_blind_human_review='NOT_RUN',writer_pretraining_overlap='UNKNOWN',
    decision='GO requires engineering, primary all-six, B1 F1 and consistent seed directions; otherwise CONDITIONAL if a prespecified benefit replicated, NO_GO if none; extra value vs M10_GAIN reported separately',
    stop='no training, alpha search, added masks, new recipe, result-dependent sample additions or exclusions')


def identity_order(identity):
    return hashlib.sha256(('M12_GAIN_CONFIRM_20261020_96042_'+identity).encode()).hexdigest()


def non_clothing_caption(caption):
    # 旧关键词“top”会误收 rolltop/drawstring top 背包；服装资格不变。
    return any(word in caption.lower() for word in ['briefcase','backpack'])


def rows():
    return read(RUN/'protocol/identities.json')


def image(arm, identity, seed):
    return RUN/'outputs'/arm/identity/('seed%d.png'%seed)


def bootstrap(values):
    a = np.asarray(values,np.float64)
    if not len(a):
        return dict(n=0,mean=None,ci95=[None,None])
    rng = np.random.default_rng(32042)
    samples = np.concatenate([a[rng.integers(0,len(a),(500,len(a)))].mean(1) for _ in range(20)])
    return dict(n=len(a),mean=float(a.mean()),ci95=np.percentile(samples,[2.5,97.5]).tolist())


def aggregate(records, identities, keys=CORE):
    index = {(r['variant'],r['id'],r['seed']):r for r in records}
    assert len(index)==len(records)==len(identities)*len(ARMS)*len(SEEDS),'DUPLICATE_OR_MISSING_IMAGE_RECORD'
    result=[]
    for arm in ARMS:
        for identity in identities:
            entries=[index[arm,identity,seed] for seed in SEEDS]
            item=dict(id=identity,variant=arm,seeds=SEEDS)
            for key in keys:
                values=[r.get(key) for r in entries]
                item[key]=float(np.mean(values)) if all(v is not None and np.isfinite(v) for v in values) else None
            result.append(item)
    return result


def statistics(records, identities, keys=CORE):
    agg=aggregate(records,identities,keys)
    index={(r['variant'],r['id']):r for r in agg}
    raw={(r['variant'],r['id'],r['seed']):r for r in records}
    means={arm:{key:bootstrap([index[arm,i][key] for i in identities if index[arm,i][key] is not None]) for key in keys} for arm in ARMS}
    def paired(arm,base,key):
        return bootstrap([index[arm,i][key]-index[base,i][key] for i in identities if index[arm,i][key] is not None and index[base,i][key] is not None])
    comparisons={base:{key:paired('M12_GAIN',base,key) for key in keys} for base in ARMS[:-1]}
    gates={}
    denominator=means['E5']['lpips_gt']['mean']-means['B1']['lpips_gt']['mean']
    for arm in ARMS[1:]:
        delta={key:paired(arm,'E5',key) for key in CORE}
        extreme=[];averaged=[]
        for identity in identities:
            events=[dict(seed=s,delta=raw[arm,identity,s]['contour_f1']-raw['E5',identity,s]['contour_f1']) for s in SEEDS
                if raw[arm,identity,s]['contour_f1'] is not None and raw['E5',identity,s]['contour_f1'] is not None
                and raw[arm,identity,s]['contour_f1']-raw['E5',identity,s]['contour_f1']<-.15]
            if events:extreme.append(dict(id=identity,events=events))
            if index[arm,identity]['contour_f1'] is not None and index['E5',identity]['contour_f1'] is not None and index[arm,identity]['contour_f1']-index['E5',identity]['contour_f1']<-.15:averaged.append(identity)
        complete=all(delta[k]['n']==len(identities) for k in CORE)
        retention=(means['E5']['lpips_gt']['mean']-means[arm]['lpips_gt']['mean'])/denominator if denominator>0 else None
        checks=dict(f1=complete and delta['contour_f1']['ci95'][0]>=-.020,
            iou=complete and delta['iou']['ci95'][0]>=-.010,leak=complete and delta['leak']['ci95'][1]<=.005,
            extreme=complete and not extreme,lpips=complete and delta['lpips_gt']['ci95'][1]<0,
            retention=complete and retention is not None and retention>=.5)
        gates[arm]=dict(checks=checks,passed=all(checks.values()),paired_E5=delta,retention=retention,
            retention_denominator=denominator,extreme_identity_events=extreme,mean_seed_extreme_ids=averaged,complete_core=complete)
    per_seed={str(seed):{arm:{key:bootstrap([raw[arm,i,seed][key] for i in identities if raw[arm,i,seed].get(key) is not None]) for key in keys} for arm in ARMS} for seed in SEEDS}
    return dict(per_identity=agg,means=means,paired=comparisons,gates=gates,per_seed=per_seed)
