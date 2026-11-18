"""固定f68/C1 PNG的Lab变化分量干预；不改变模型、S0或原验收。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import time
import zipfile

import cv2
import numpy as np
from PIL import Image

from tools.e2_b_amplitude_response_metrics import to_lab,from_lab
from tools import e2_state_b_pair_evaluation as evaluation
from tools.e2_state_b_amplitude_response import read,write,sha,pixels


SOURCE=Path(__file__).resolve().parents[1]
VARIANTS=('L_full','L_high','L_low','ab_only')
EXTRA=('tools/e2_reference_terminal_lab_components.py','scripts/e2_reference_terminal_lab_components.sbatch')


def quantize(rgb):
    return np.rint(np.clip(rgb,0,1).astype(np.float32)*255).astype(np.uint8)


def components(base,current):
    """只读两个模型输出，构造时不读取目标/参考；不添加新mask以免混入第二种干预。"""
    baseline=to_lab(base);delta=to_lab(current)-baseline
    low=cv2.GaussianBlur(delta[...,0],(0,0),24);high=delta[...,0]-low
    fields={name:np.zeros_like(delta) for name in VARIANTS}
    fields['L_full'][...,0]=delta[...,0]
    fields['L_high'][...,0]=high;fields['L_low'][...,0]=low
    fields['ab_only'][...,1:]=delta[...,1:]
    assert np.max(np.abs(fields['L_high']+fields['L_low']+fields['ab_only']-delta))<1e-12
    origin=base.astype(np.float64)/255;inverse=from_lab(baseline)
    def render(change):
        value=origin+(from_lab(baseline+change)-inverse)
        return quantize(value),float(((value<0)|(value>1)).mean())
    assert np.array_equal(render(np.zeros_like(delta))[0],base)
    assert np.array_equal(render(delta)[0],current)
    return {name:dict(zip(('image','gamut_clipped_channel_fraction'),render(change))) for name,change in fields.items()}


def run(args):
    cv2.setNumThreads(1);started=time.monotonic();source=args.auto/'reference_terminal_one_step';replay=args.auto/'reference_terminal_fp32_replay'
    previous_dir=args.auto/'reference_terminal_png_lab_audit';previous=read(previous_dir/'protocol.json')
    assert read(previous_dir/'result.json')['audit_completed']
    assert all(sha(previous_dir/n)==h for n,h in read(previous_dir/'manifest.json').items())
    assert all(sha(SOURCE/n)==h for n,h in previous['source_sha256'].items())
    baseline=read(source/'evaluation/f68/replay/result.json');old=read(replay/'result.json')
    assert old['status']=='failed' and old['completed_cache_observations']==96
    rows=old['models']['C1']['rows'];base={(r['case'],r['sample_condition']):r for r in baseline['rows']}
    spec=read(source/'protocol.json');pairs={v['case_id']:v for v in spec['train']}
    inputs=[previous_dir/n for n in ('protocol.json','result.json','manifest.json')]
    inputs += [source/'evaluation/f68/replay/result.json',source/'protocol.json',replay/'result.json',replay/'manifest.json']
    jobs=[]
    for row in rows:
        case,cond=row['case'],row['sample_condition'];b=base[case,cond];folder=source/'cases'/case
        before,after=source/b['image'],replay/row['image'];assert sha(before)==b['png_sha256'] and sha(after)==row['png_sha256']
        for n,h in row['assets'].items():assert sha(folder/n)==h
        inputs += [before,after]+[folder/n for n in row['assets']]
        jobs.append((row,b,folder,before,after))
    assert len(jobs)==48 and not args.out.exists(),'不覆盖Lab分量诊断';args.out.mkdir(parents=True)
    inputs=list(dict.fromkeys(inputs))
    protocol=dict(kind='fixed C1-minus-f68 PNG Lab component interventions;not model candidates',
        variants=list(VARIANTS),luminance_split='global Gaussian sigma24,not exact masked official lowpass projection',
        mask_intervention=False,construction_inputs='only two generated PNGs;no goal/reference used to construct outputs',
        color_space='repository analytic double Lab,anchored inverse residual;original F32 PNG quantizer',
        limits='gamut clipping and Q8 can reintroduce chroma/lowL;no exact preservation claimed',
        original_loss_and_gates_unchanged=True,original_336_training_task_gates_evaluable=False,
        scope='all48 seen conditions;not holdout or fresh S0',new_S0=0,model_decodes=0,parameter_VJP=0,optimizer_steps=0,holdout_read=0,adopted_checkpoints=[],
        input_sha256={str(p.resolve()):sha(p) for p in inputs},source_sha256=dict(previous['source_sha256'],**{n:sha(SOURCE/n) for n in EXTRA}),
        execution_commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip())
    write(args.out/'protocol.json',protocol)
    def score(job):
        row,b,folder,before,after=job;constructed=components(pixels(before),pixels(after));output={}
        entry=dict(folder=folder,condition=row['condition'],sample_condition=row['sample_condition'],pair=pairs[row['case']],assets=row['assets'],source_image=before)
        # 反事实无新的latent/参数路径，不继承原C1的training_tasks或模型资格字段。
        metadata={n:b[n] for n in ('case','condition','sample_condition','group','arm','factor','seed','noise_digest')}
        for name,item in constructed.items():
            picture=item['image'];file=Path(name)/(row['case']+'_'+row['sample_condition']+'.png');(args.out/file).parent.mkdir(exist_ok=True)
            Image.fromarray(picture).save(args.out/file)
            output[name]=dict(metadata,variant=name,image=file.as_posix(),png_sha256=sha(args.out/file),
                gamut_clipped_channel_fraction=item['gamut_clipped_channel_fraction'],**evaluation.score_entry(entry,picture,'replay'))
        return output
    collected={n:[] for n in VARIANTS}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,value in enumerate(pool.map(score,jobs),1):
            for n,row in value.items():collected[n].append(row)
            if i%4==0:print('LAB_COMPONENTS',i,flush=True)
    base_map={(r['case'],r['condition'],r['arm']):r for r in baseline['rows']}
    result=dict(audit_completed=True,zero_PNG_controls=48,full_delta_PNG_controls=48,completed_interventions=192,variants={},
        original_C1_reference=old['models']['C1'],formal_adoption_eligibility=False,original_336_training_task_gates_evaluated=False,
        new_S0=0,model_decodes=0,parameter_VJP=0,optimizer_steps=0,holdout_read=0,adopted_checkpoints=[])
    for name,rr in collected.items():
        report=dict(rows=rr,**evaluation.aggregate(rr,'replay'),strict_PNG_protection=evaluation.strict_replay_protection(rr,base_map))
        report['all_mean_residue_delta']=float(np.mean([v['scores']['old_projection']-base_map[v['case'],v['condition'],v['arm']]['scores']['old_projection'] for v in rr if v['scores']['old_projection'] is not None]))
        report['pastel_quality_pass']={a:sum(v['scores']['l_quality_pass'] for v in rr if v['group']=='pastel' and v['arm']==a) for a in ('a075','a100','a125')}
        result['variants'][name]=report
    assert all(sha(Path(n))==h for n,h in protocol['input_sha256'].items())
    assert all(sha(SOURCE/n)==h for n,h in protocol['source_sha256'].items())
    result.update(source_final_verification=True,seconds=time.monotonic()-started)
    write(args.out/'result.json',result);names=['protocol.json','result.json']+[str(p.relative_to(args.out).as_posix()) for p in args.out.rglob('*.png')]
    write(args.out/'manifest.json',{n:sha(args.out/n) for n in names})
    with zipfile.ZipFile(args.out/'review.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for n in names+['manifest.json']:archive.write(args.out/n,n)
    print('LAB_COMPONENTS_REVIEW',sha(args.out/'review.zip'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prior',type=Path,required=True)
    args=parser.parse_args();args.auto=args.prior.resolve()/'autonomous_repair_20261116';args.out=args.auto/'reference_terminal_lab_components';run(args)
