"""E32 结果核验与本地下载包；停止时不运行后续阶段。"""

import argparse
import json
import math
import subprocess
import tarfile
from pathlib import Path

import numpy as np

from tools.e32_common import OUT,DATASET,read,write,sha,finish_frozen,git_commit


def finite(value):
    if isinstance(value,float):
        return math.isfinite(value)
    if isinstance(value,list):
        return all(finite(v) for v in value)
    if isinstance(value,dict):
        return all(finite(v) for v in value.values())
    return True


def check_stage0(out):
    split=read(out/'split_manifest.json')
    checks={}
    for name,records in [('train_audit',split['stage0_train_audit']),('dev',split['dev']),
                         ('validation_diagnostic',split['confirmation_all'])]:
        rows=read(out/'stage0_pair_audit'/name/'rows.json')
        summary=read(out/'stage0_pair_audit'/name/'summary.json')
        checks[name+'/ids']={r['id'] for r in rows}=={r['id'] for r in records} and len(rows)==len(records)
        checks[name+'/finite']=finite(rows) and finite(summary)
        wrong=read(out/'stage0_pair_audit'/name/'wrong_manifest.json')
        checks[name+'/wrong_distinct']=all(sid!=v['color_near'] and sid!=v['random'] and v['color_near']!=v['random'] for sid,v in wrong.items())
    return checks


def check_geometry(out):
    checks={}
    from tools.e32_a_geometry_train import VARIANTS,variant_folder
    runs=[('full',seed) for seed in (42,43,44)]+[(variant,42) for variant in VARIANTS[1:]]
    for variant,seed in runs:
        folder=variant_folder(out,seed,variant)
        key='%s/seed%d'%(variant,seed)
        checks[key+'/steps']=read(folder/'training_protocol.json')['steps']==8000
        history=read(folder/'training_history.json')
        checks[key+'/history']=len(history)==80 and history[-1]['step']==8000 and finite(history)
        for name in ('dev','causal_test','independent_confirmation'):
            rows=read(folder/name/'rows.json')
            checks[key+'/'+name+'/count']=len(rows)=={'dev':256,'causal_test':8,'independent_confirmation':10}[name]
            checks[key+'/'+name+'/finite']=finite(rows) and finite(read(folder/name/'summary.json'))
        checks[key+'/fields_count']=len(list((folder/'fields').glob('*/predictions.npz')))==256
        for path in (folder/'fields').glob('*/predictions.npz'):
            with np.load(path) as d:
                checks[key+'/field/'+path.parent.name]=all(np.isfinite(d[k]).all() for k in d.files)
                for arm in ('matched','color_near','random','zero','rot90'):
                    checks[key+'/confidence/'+path.parent.name+'/'+arm]=bool((d[arm+'_confidence']>=0).all() and (d[arm+'_confidence']<=1).all())
    return checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--review-json')
    args=parser.parse_args();out=args.out
    decision=read(out/'decision_summary.json')
    checks=check_stage0(out)
    stopped=None
    if not decision['pair_dependence_pass']:
        stopped='Stage0'
        for name in ('A_geometry','B_appearance','C_scaffold','D_generation','E_confirmation','ablations'):
            write(out/name/'status.json',dict(status='not_run_stage0_gate_failed',reason='dataset_pairing_diagnosis'))
    elif decision['geometry_field_pass'] is False:
        stopped='StageA'
        checks.update(check_geometry(out))
        from tools.e32_visualize import geometry_plots
        geometry_plots(out,DATASET)
    if args.review_json:
        review=json.loads(args.review_json)
        selected=read(out/'audits/stage0_visual_selection.json')['selected_ids']
        assert set(review['reviewed_stage0_ids'])==set(selected)
        if stopped=='StageA':
            selected_A=read(out/'audits/geometry_visual_selection.json')['selected_ids']
            assert set(review['reviewed_geometry_ids'])==set(selected_A)
        write(out/'audits/visual_review.json',review)
    reviewed=False
    if (out/'audits/visual_review.json').exists():
        review=read(out/'audits/visual_review.json')
        reviewed=set(review.get('reviewed_stage0_ids',[]))==set(read(out/'audits/stage0_visual_selection.json')['selected_ids'])
        if stopped=='StageA':
            reviewed=reviewed and set(review.get('reviewed_geometry_ids',[]))==set(read(out/'audits/geometry_visual_selection.json')['selected_ids'])
    finish_frozen(out)
    frozen=read(out/'frozen_check.json')
    protocols=list((out/'A_geometry').glob('seed*/training_protocol.json'))+list((out/'ablations').glob('*/seed*/training_protocol.json'))
    frozen['training_steps']=sum(read(path)['steps'] for path in protocols)
    frozen['E5_training_steps']=0
    frozen['note']='DINO/E26 frozen; E5/VAE/U-Net/BF/TCPM not loaded by Stage0/StageA; training_steps counts completed field runs'
    write(out/'frozen_check.json',frozen)
    write(out/'audits/numeric_integrity.json',dict(checks=checks,**{'pass':all(checks.values())}))
    subprocess.run(['sacct','-j',args_job_ids(out),'--format=JobID,State,ExitCode,Elapsed','-n'],
                   stdout=(out/'job_status.log').open('w'),check=True)
    completion=dict(numeric_artifacts_complete=all(checks.values()),frozen_inputs_pass=read(out/'frozen_check.json')['pass'],
        stage0_pass=decision['pair_dependence_pass'],stage_A_pass=decision['geometry_field_pass'],
        stopped_at=stopped,visual_review_completed=reviewed,
        experiment_complete=bool(stopped and all(checks.values()) and reviewed and read(out/'frozen_check.json')['pass']),report_git_commit=git_commit(),
        note='completed stopped experiment is not a scientific Gate pass; downstream stages only allowed after prior Gate pass')
    write(out/'completion_check.json',completion)
    paths=[p for p in out.rglob('*') if p.is_file() and p.suffix in ('.json','.png','.log','.err')
           and 'cache' not in p.parts and 'training_mask_diagnostics' not in p.parts]
    manifest={str(p.relative_to(out)):sha(p) for p in paths if p.name!='artifact_manifest.json'}
    write(out/'artifact_manifest.json',dict(files=manifest,large_arrays_and_checkpoints='retained on server; excluded from local review bundle'))
    with tarfile.open(out/'local_review_bundle.tar.gz','w:gz') as archive:
        for path in paths:
            if path.name!='artifact_manifest.json':archive.add(path,arcname=str(path.relative_to(out)))
        archive.add(out/'artifact_manifest.json',arcname='artifact_manifest.json')
    print(json.dumps(dict(decision=decision,completion=completion,bundle_sha256=sha(out/'local_review_bundle.tar.gz'))),flush=True)


def args_job_ids(out):
    ids=[]
    for prefix in ('stage0_','A_','report_'):
        for p in out.glob(prefix+'*.log'):
            sid=p.stem[len(prefix):]
            if sid.isdigit():ids.append(sid)
    return ','.join(ids)


if __name__=='__main__':main()
