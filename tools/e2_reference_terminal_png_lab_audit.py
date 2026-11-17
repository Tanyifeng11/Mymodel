"""同144张历史PNG的Lab精度旁路审计；保留原验收，不训练或采用。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import time
import zipfile

import cv2
import numpy as np

from tools.e2_b_amplitude_response_metrics import to_lab
from tools.e2_b_last_step_metrics import lab, highpass, palette, residue_components
from tools import e2_state_b_pair_evaluation as evaluation
from tools.e2_state_b_amplitude_response import read, write, sha, pixels


SOURCE=Path(__file__).resolve().parents[1]
EXTRA=('tools/e2_reference_terminal_png_lab_audit.py','scripts/e2_reference_terminal_png_lab_audit.sbatch')
STAGES=('original_CV32','CV_Lab_F64_math','analytic_Lab_F64_math')


def lab_scores(fields,mask):
    """两个旁路都用同一F64指标算术，仅Lab输入路径不同；L幅度/响应沿用原值。"""
    current,target,goal,anchor=fields
    assert all(v.dtype==np.float64 for v in fields)
    inside=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,5)>=16
    den=cv2.GaussianBlur(mask.astype(np.float64),(0,0),24).clip(.001)
    def low(reference):
        delta=cv2.GaussianBlur((current[...,0]-reference[...,0])*mask,(0,0),24)/den
        return float(np.abs(delta[inside]).mean())
    error=(current-target)/100
    dx=np.pad(np.diff(error,axis=1),((0,0),(0,1),(0,0)))
    dy=np.pad(np.diff(error,axis=0),((0,1),(0,0),(0,0)))
    values=dict(palette_swd=float(np.abs(palette(current[inside,1:])-palette(goal[inside,1:])).mean()),
        spatial_ab_mae_positive=float(np.abs(current[inside,1:]-target[inside,1:]).mean()),
        spatial_ab_mae_goal=float(np.abs(current[inside,1:]-goal[inside,1:]).mean()),
        low_l=low(goal),low_l_positive=low(target),
        edge=float(np.abs(np.concatenate((dx,dy),axis=2))[inside].mean()),
        l_edge=float((np.abs(dx[...,0][inside]).mean()+np.abs(dy[...,0][inside]).mean())/2))
    values.update(residue_components(*[highpass(v)[inside] for v in (current,target,anchor)]))
    return values


def run(args):
    cv2.setNumThreads(1);started=time.monotonic()
    native=args.auto/'reference_terminal_native_metric_audit_v2';replay=args.auto/'reference_terminal_fp32_replay'
    source_run=args.auto/'reference_terminal_one_step';previous=read(native/'protocol.json')
    assert read(native/'result.json')['status']=='complete'
    assert all(sha(native/n)==h for n,h in read(native/'manifest.json').items())
    assert all(sha(SOURCE/n)==h for n,h in previous['source_sha256'].items())
    baseline=read(source_run/'evaluation/f68/replay/result.json');old=read(replay/'result.json')
    assert old['completed_cache_observations']==96 and old['status']=='failed' and old['message']=="'seed'"
    reports=dict(f68=baseline,**old['models']);assert all(len(v['rows'])==48 for v in reports.values())
    inputs=[native/n for n in ('protocol.json','result.json','manifest.json')]
    inputs += [replay/n for n in ('protocol.json','result.json','manifest.json')]+[source_run/'evaluation/f68/replay/result.json']
    jobs=[]
    for model,report in reports.items():
        for row in report['rows']:
            folder=source_run/'cases'/row['case'];image=(source_run if model=='f68' else replay)/row['image']
            assert sha(image)==row['png_sha256']
            for name,h in row['assets'].items():assert sha(folder/name)==h
            inputs += [image]+[folder/n for n in row['assets']]
            jobs.append((model,row,folder,image))
    inputs=list(dict.fromkeys(inputs));assert not args.out.exists(),'不覆盖全PNG Lab审计';args.out.mkdir(parents=True)
    protocol=dict(kind='same144 historical PNG Lab-path shadow audit',stages=list(STAGES),
        decomposition='original CV32 vs same CV Lab values with F64 math vs analytic Lab with same F64 math',
        original_formal_acceptance_unchanged=True,shadow_tolerances='original numeric tolerances,diagnostic only;not replacement acceptance',
        amplitude_correlation_response='original precise-L fields retained unchanged in all stages',
        scope='complete48 seen conditions per model;not holdout or new S0',opencv=cv2.__version__,numpy=np.__version__,
        new_S0=0,model_decodes=0,reverse_passes=0,optimizer_steps=0,holdout_read=0,adopted_checkpoints=[],
        input_sha256={str(p.resolve()):sha(p) for p in inputs},
        source_sha256=dict(previous['source_sha256'],**{n:sha(SOURCE/n) for n in EXTRA}),
        execution_commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip())
    write(args.out/'protocol.json',protocol)
    def score(job):
        model,row,folder,image=job;picture=pixels(image)
        mask=pixels(folder/'mask.png');mask=(mask[...,0] if mask.ndim==3 else mask)>0
        targets=[pixels(folder/n) for n in (row['sample_condition']+'_positive.png',row['sample_condition']+'_goal.png','train_anchor.png')]
        values={}
        for stage,transform in [('CV_Lab_F64_math',lambda x:lab(x).astype(np.float64)),('analytic_Lab_F64_math',to_lab)]:
            updated=lab_scores([transform(v) for v in [picture]+targets],mask)
            values[stage]=dict(row,scores=dict(row['scores'],**updated))
        return model,row,values
    staged={stage:{m:[] for m in reports} for stage in STAGES}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,(model,row,values) in enumerate(pool.map(score,jobs),1):
            staged['original_CV32'][model].append(row)
            for stage,v in values.items():staged[stage][model].append(v)
            if i%12==0:print('PNG_LAB_AUDIT',i,flush=True)
    result=dict(audit_completed=True,source_controls_passed=False,stages={},completed_original_PNGs=144,
        actual_new_shadow_scores=288,formal_adoption_eligibility=False,original_runtime_status_preserved=old['status'],
        new_S0=0,model_decodes=0,reverse_passes=0,optimizer_steps=0,holdout_read=0,adopted_checkpoints=[])
    for stage,models in staged.items():
        base={(r['case'],r['condition'],r['arm']):r for r in models['f68']};output={}
        for model,rows in models.items():
            report=dict(rows=rows,aggregate=evaluation.aggregate(rows,'replay'))
            if model!='f68':
                report['original_tolerance_shadow_diagnostic']=evaluation.strict_replay_protection(rows,base)
                deltas=[r['scores']['old_projection']-base[r['case'],r['condition'],r['arm']]['scores']['old_projection'] for r in rows if r['scores']['old_projection'] is not None]
                report['all_mean_residue_delta']=float(np.mean(deltas))
                report['strict_mean_residue_decrease']=report['all_mean_residue_delta'] < -1e-8
            output[model]=report
        result['stages'][stage]=output
    assert all(sha(Path(n))==h for n,h in protocol['input_sha256'].items())
    assert all(sha(SOURCE/n)==h for n,h in protocol['source_sha256'].items())
    result.update(source_controls_passed=True,source_final_verification=True,seconds=time.monotonic()-started)
    write(args.out/'result.json',result);names=['protocol.json','result.json']
    write(args.out/'manifest.json',{n:sha(args.out/n) for n in names})
    with zipfile.ZipFile(args.out/'review.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for n in names+['manifest.json']:archive.write(args.out/n,n)
    print('PNG_LAB_AUDIT_REVIEW',sha(args.out/'review.zip'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prior',type=Path,required=True)
    args=parser.parse_args();args.auto=args.prior.resolve()/'autonomous_repair_20261116'
    args.out=args.auto/'reference_terminal_png_lab_audit';run(args)
