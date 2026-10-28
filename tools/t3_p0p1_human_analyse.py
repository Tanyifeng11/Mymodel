"""只分析真实答卷：试评描述；正式协议冻结后考虑评价者和案例的交叉重复。"""
import argparse
import csv
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from tools.t3_p0p1 import read,write,sha,csv_write
from tools.t3_p0p1_review import holm


def analyse(project,response_dir):
    root=project/'output_eval/t3_p0_p1_20261028/human_eval'
    files=sorted(response_dir.glob('*.csv')) if response_dir.is_dir() else []
    raw=[]
    for path in files:
        with path.open(encoding='utf-8-sig',newline='') as stream:
            raw.extend(dict(row,source_file=path.name,source_sha256=sha(path)) for row in csv.DictReader(stream))
    if not raw:
        write(root/'analysis_status.json',dict(status='AWAITING_REAL_RESPONSES',human_reviewers=0,raw_rows=0,statistics_run=False))
        print('NO_HUMAN_RESPONSES; no scores or win rates generated')
        return
    csv_write(root/'responses_raw.csv',raw)
    assignments={}
    with (root/'private/randomization_manifest.csv').open(encoding='utf-8-sig',newline='') as stream:
        assignments={(r['reviewer'],r['trial_id']):r for r in csv.DictReader(stream)}
    duplicates=Counter((r['reviewer'],r['phase'],r['trial_id']) for r in raw)
    issues=[dict(reviewer=k[0],phase=k[1],trial_id=k[2],count=v) for k,v in duplicates.items() if v>1]
    if issues:
        write(root/'analysis_status.json',dict(status='DUPLICATE_RESPONSES_REQUIRE_RESOLUTION',duplicates=issues,
                                               raw_preserved=True,statistics_run=False))
        return
    clean=[]
    for row in raw:
        key=(row['reviewer'],row['trial_id'])
        assert key in assignments,('UNKNOWN_REVIEWER_OR_TRIAL',key)
        assignment=assignments[key]
        baseline=assignment['A'] if assignment['B']=='T3' else assignment['B']
        for question in ['q1','q2','q3']:
            answer=row.get(question,'')
            assert answer in ['A','B','tie','cannot',''],('INVALID_ANSWER',key,answer)
            score=None if answer in ['cannot',''] else 0 if answer=='tie' else 1 if assignment[answer]=='T3' else -1
            clean.append(dict(reviewer=row['reviewer'],phase=row['phase'],id=assignment['id'],trial_id=row['trial_id'],
                baseline=baseline,question=question,answer=answer,score=score,
                status='cannot_judge' if answer=='cannot' else 'missing' if not answer else 'answered',
                seconds=float(row.get('seconds') or 0),background=row.get('background','')))
    csv_write(root/'responses_clean.csv',clean)
    descriptive=[]
    for phase in sorted({r['phase'] for r in clean}):
        for baseline in ['B','T1','T2','C']:
            for question in ['q1','q2','q3']:
                group=[r for r in clean if r['phase']==phase and r['baseline']==baseline and r['question']==question]
                valid=[r for r in group if r['score'] is not None]
                descriptive.append(dict(phase=phase,baseline=baseline,question=question,n_ratings=len(valid),
                    n_reviewers=len({r['reviewer'] for r in valid}),n_cases=len({r['id'] for r in valid}),
                    signed_preference_mean=float(np.mean([r['score'] for r in valid])) if valid else None,
                    T3_preferred=sum(r['score']==1 for r in valid),baseline_preferred=sum(r['score']==-1 for r in valid),
                    ties=sum(r['score']==0 for r in valid),cannot_judge=sum(r['status']=='cannot_judge' for r in group),
                    missing=sum(r['status']=='missing' for r in group)))
    csv_write(root/'human_eval_descriptive.csv',descriptive)
    freeze=root/'formal_freeze.json'
    if not freeze.is_file():
        write(root/'analysis_status.json',dict(status='REAL_PILOT_DESCRIPTIVE_ONLY_FORMAL_NOT_FROZEN',
            human_reviewers=len({r['reviewer'] for r in raw}),raw_rows=len(raw),formal_statistics_run=False))
        return
    protocol=read(freeze)
    formal=[r for r in clean if r['phase']=='formal_draft' and r['score'] is not None]
    assert len({r['reviewer'] for r in formal})>=protocol['minimum_reviewers'], 'formal sample not yet complete'
    assert protocol['analysis']=='crossed_rater_case_random_intercept_signed_preference'
    import pandas as pd
    import statsmodels.api as sm
    results=[]
    for baseline in ['B','T1','T2','C']:
        for question in ['q1','q2','q3']:
            group=[r for r in formal if r['baseline']==baseline and r['question']==question]
            frame=pd.DataFrame(group);frame['all']=1
            model=sm.MixedLM.from_formula('score ~ 1',groups='all',re_formula='0',
                vc_formula={'reviewer':'0+C(reviewer)','case':'0+C(id)'},data=frame)
            fit=model.fit(reml=True,method='lbfgs')
            assert fit.converged, ('MIXED_MODEL_DID_NOT_CONVERGE',baseline,question)
            low,high=fit.conf_int().loc['Intercept']
            # 交叉簇bootstrap仅作敏感性区间，不把逐题回答当独立样本。
            raters=sorted(frame.reviewer.unique());cases=sorted(frame.id.unique())
            ri=np.array([raters.index(v) for v in frame.reviewer]);ci=np.array([cases.index(v) for v in frame.id])
            rng=np.random.default_rng(281042)
            rw=rng.multinomial(len(raters),np.ones(len(raters))/len(raters),size=10000)
            cw=rng.multinomial(len(cases),np.ones(len(cases))/len(cases),size=10000)
            weights=rw[:,ri]*cw[:,ci];denominator=weights.sum(1);valid=denominator>0
            boots=(weights[valid]*frame.score.to_numpy()).sum(1)/denominator[valid]
            blow,bhigh=np.quantile(boots,[.025,.975])
            results.append(dict(baseline=baseline,question=question,n_ratings=len(group),
                n_reviewers=frame.reviewer.nunique(),n_cases=frame.id.nunique(),effect=float(fit.fe_params['Intercept']),
                ci_low=float(low),ci_high=float(high),p_raw=float(fit.pvalues['Intercept']),
                two_way_bootstrap_low=float(blow),two_way_bootstrap_high=float(bhigh),two_way_bootstrap_valid_draws=int(valid.sum()),
                method='crossed reviewer/case random intercept linear mixed model; signed preference with ties=0; REML/Wald',
                formal_protocol_sha256=sha(freeze)))
    families=[ [r for r in results if r['question']=='q1' and r['baseline']!='C'],
               [r for r in results if r['question']!='q1' and r['baseline']!='C'],
               [r for r in results if r['baseline']=='C'] ]
    for family in families:holm(family)
    csv_write(root/'human_eval_statistics.csv',results)
    write(root/'analysis_status.json',dict(status='FORMAL_REAL_RESPONSES_ANALYSED',human_reviewers=len({r['reviewer'] for r in formal}),
        raw_rows=len(raw),case_and_reviewer_dependence_modelled=True,AI_ratings=0,
        limitation='signed preference is a bounded ordinal summary; small-case and Wald approximation limits must be reported'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);parser.add_argument('--responses',type=Path,required=True)
    args=parser.parse_args();analyse(args.project,args.responses)
