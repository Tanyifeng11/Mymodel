"""E33审计复核与下载包；完整性失败属于完成的停止实验，不属于架构失败。"""
import argparse
import json
import subprocess
import tarfile
from collections import Counter
from tools.e32_common import read,write,sha,finish_frozen,bootstrap,git_commit
from tools.e33_protocol import OUT,GROUPS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-json',default='')
    args=parser.parse_args();split=read(OUT/'split_manifest.json')
    protocol=read(OUT/'protocol.json');decision=read(OUT/'decision_summary.json')
    summary=read(OUT/'transform_integrity/summary.json');mode=summary['selected_mode']
    rows=[read(p) for p in sorted((OUT/'transform_integrity/cases').glob('*.json'))]
    checks={};failures={}
    for group in GROUPS:
        selected=[r for r in rows if r['group']==group]
        expected={r['id'] for r in split[group]}
        checks[group+'/ids']=len(selected)==len(expected) and {r['id'] for r in selected}==expected
        checks[group+'/protocol']=all(r['protocol_sha256']==protocol['protocol_sha256'] for r in selected)
        available=[r for r in selected if r[mode]['controlled_available']]
        checks[group+'/counts']=len(available)==summary['coverage'][group][mode]['available_cases']
        failures[group]={}
        for arm in ('rot90','scale_up','scale_down'):
            measurements=[r['interventions'][mode][arm]['clean'] for r in available]
            reasons=Counter()
            for r in measurements:
                if not r['valid']:
                    if r.get('before',{}).get('confidence',0)<.25:reasons['baseline_E26_unreadable']+=1
                    if r.get('after',{}).get('confidence',0)<.25:reasons['transformed_E26_unreadable']+=1
                    if r.get('orientation_response_error_deg',999)>15:reasons['orientation_response']+=1
                    if r.get('period_response_error_log2',999)>.10:reasons['period_response']+=1
            failures[group][arm]=dict(denominator=len(measurements),valid=sum(r['valid'] for r in measurements),
                 overlapping_failure_reasons=dict(reasons),
                 baseline_confidence=bootstrap([r['before']['confidence'] for r in measurements if 'before' in r]),
                 transformed_confidence=bootstrap([r['after']['confidence'] for r in measurements if 'after' in r]))
    checks['total_count']=len(rows)==45400
    checks['summary_matches_decision']=summary['pass']==decision['transform_integrity_pass']
    write(OUT/'audits/numeric_integrity.json',dict(checks=checks,**{'pass':all(checks.values())}))
    write(OUT/'audits/intervention_failure_diagnosis.json',dict(groups=failures,
          note='failure reasons overlap; compares original and transformed E26 in common native valid support; no reference reselection'))
    ids=read(OUT/'audits/visual_selection.json')['ids']
    if args.review_json:
        review=json.loads(args.review_json)
        assert set(review['reviewed_ids'])==set(ids)
        write(OUT/'audits/visual_review.json',review)
    reviewed=(OUT/'audits/visual_review.json').exists() and set(read(OUT/'audits/visual_review.json')['reviewed_ids'])==set(ids)
    finish_frozen(OUT);frozen=read(OUT/'frozen_check.json')
    stopped=not decision['transform_integrity_pass']
    if stopped:
        assert not list(OUT.rglob('checkpoint*.pt')),'完整性失败却存在训练checkpoint'
        write(OUT/'ablations/status.json',dict(status='not_run_transform_integrity_failed',
              completed_audit_comparator='G_no_structural_exclusion_candidate_availability (not model training)',
              not_run=['A_no_prior_freeze','B_no_counterfactual_loss','C_no_wrong_ranking','D_no_retention',
                       'E_rot90_only','F_scale_only','G_no_structural_exclusion_model'],
              reason='no P0/P1/P2 training allowed before intervention integrity Gate'))
    completion=dict(numeric_artifacts_complete=all(checks.values()),frozen_inputs_pass=frozen['pass'],
        transform_integrity_pass=decision['transform_integrity_pass'],visual_review_completed=reviewed,
        training_steps=0 if stopped else None,stopped_at='Step0_transform_integrity' if stopped else None,
        experiment_complete=bool(stopped and all(checks.values()) and frozen['pass'] and reviewed),
        report_git_commit=git_commit(),
        note='integrity failure diagnoses synthetic intervention / estimator observability; controlled/real model capability untested')
    write(OUT/'completion_check.json',completion)
    jobids=','.join(sorted({p.stem.rsplit('_',1)[-1] for p in OUT.glob('*.log') if p.stem.rsplit('_',1)[-1].isdigit()}))
    with (OUT/'job_status.log').open('w') as log:
        subprocess.run(['sacct','-j',jobids,'--format=JobID,State,ExitCode,Elapsed','-n'],stdout=log,check=True)
    paths=[p for p in OUT.rglob('*') if p.is_file() and p.suffix in ('.json','.png','.log','.err')
           and p.name!='artifact_manifest.json']
    manifest={str(p.relative_to(OUT)):sha(p) for p in paths}
    write(OUT/'artifact_manifest.json',dict(files=manifest,large_checkpoints='retained on server if later phases allowed'))
    with tarfile.open(OUT/'local_review_bundle.tar.gz','w:gz') as archive:
        for p in paths:archive.add(p,arcname=str(p.relative_to(OUT)))
        archive.add(OUT/'artifact_manifest.json',arcname='artifact_manifest.json')
    print(json.dumps(dict(decision=decision,completion=completion,bundle_sha256=sha(OUT/'local_review_bundle.tar.gz'))),flush=True)


if __name__=='__main__':main()
