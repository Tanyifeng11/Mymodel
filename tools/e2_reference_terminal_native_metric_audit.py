"""只读已有16条件RGB：分开训练/官方亮度口径、Lab近似与PNG量化影响。"""
import argparse
from pathlib import Path
import subprocess
import time
import zipfile

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from models.e2_lab_anchor import rgb_to_lab
from tools.e2_b_amplitude_response_metrics import to_lab
from tools.e2_b_last_step_metrics import highpass, residue_components
from tools import e2_state_b_pair_evaluation as evaluation
from tools.e2_state_b_amplitude_response import read, write, sha, pixels
from tools.e2_state_b_backward_repro import tensor_sha


SOURCE=Path(__file__).resolve().parents[1]
EXTRA=('tools/e2_reference_terminal_native_metric_audit.py',
       'scripts/e2_reference_terminal_native_metric_audit.sbatch')


def box_light(rgb,positive,weight,dtype):
    current,target=[rgb_to_lab(v.to(dtype))/100 for v in (rgb,positive)]
    w=weight.to(dtype); sw=F.avg_pool2d(w,64,16)
    low=F.avg_pool2d((current-target)[:,:1]*w,64,16)/sw.clamp_min(1e-6)
    return float((low.abs()*sw).sum()/sw.sum().clamp_min(1e-8))


def gaussian_light(current,target,mask):
    inside=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,5)>=16
    den=cv2.GaussianBlur(mask.astype(np.float64),(0,0),24).clip(.001)
    low=cv2.GaussianBlur((current-target)*mask,(0,0),24)/den
    return float(np.abs(low[inside]).mean())


def run(args):
    torch.set_num_threads(2); cv2.setNumThreads(2); started=time.monotonic()
    stages=args.auto/'reference_terminal_fp32_pair_metric_stages'
    replay=args.auto/'reference_terminal_fp32_replay'
    r16=args.auto/'reference_terminal_r16_diagnostic'
    previous=read(stages/'protocol.json'); checked=read(stages/'result.json')
    assert checked['status']=='complete' and checked['passed'] and checked['source_final_verification']
    assert all(sha(stages/n)==h for n,h in read(stages/'manifest.json').items())
    assert all(sha(SOURCE/n)==h for n,h in previous['source_sha256'].items())
    for folder in (replay,r16):
        assert all(sha(folder/n)==h for n,h in read(folder/'manifest.json').items())
    paired=torch.load(stages/'diagnostic.pt',map_location='cpu',weights_only=False)['points']
    sentinels=torch.load(r16/'diagnostic.pt',map_location='cpu',weights_only=False)['sentinels']
    old=torch.load(replay/'diagnostic.pt',map_location='cpu',weights_only=False)['points']
    old_result=read(replay/'result.json'); source_run=args.auto/'reference_terminal_one_step'
    base=read(source_run/'evaluation/f68/replay/result.json'); spec=read(source_run/'protocol.json')
    pairs={v['case_id']:v for v in spec['train']}
    rows={('f68',r['case'],r['sample_condition']):r for r in base['rows']}
    for model,v in old_result['models'].items(): rows.update({(model,r['case'],r['sample_condition']):r for r in v['rows']})
    points={k:dict(native_rgb_half=v['native_rgb_half'],positive=v['positive'],weight=v['weight'],outputs=v['outputs']) for k,v in paired.items()}
    overlapping=0
    for key,v in old.items():
        if 'rgb' not in v['candidates']['C1']: continue
        sentinel=sentinels[key]
        if key in points:
            assert all(torch.equal(points[key][n],sentinel[n]) for n in ('native_rgb_half','positive','weight'))
            assert all(torch.equal(points[key]['outputs'][m],v['candidates'][m]['rgb']) for m in ('C0','C1'))
            overlapping+=1
        else: points[key]=dict(native_rgb_half=sentinel['native_rgb_half'],positive=sentinel['positive'],weight=sentinel['weight'],outputs={m:v['candidates'][m]['rgb'] for m in ('C0','C1')})
    assert len(points)==16 and overlapping==4
    gradient_rows={}
    inputs=[stages/n for n in ('protocol.json','manifest.json','result.json','diagnostic.pt')]
    inputs += [folder/n for folder in (replay,r16) for n in ('protocol.json','manifest.json','result.json','diagnostic.pt')]
    inputs += [source_run/'protocol.json',source_run/'evaluation/f68/replay/result.json']
    for i in range(4):
        f=args.auto/('reference_terminal_fp32_gradients/gradients/shard%d.json'%i); inputs.append(f)
        for row in read(f)['values']: gradient_rows[row['case'],row['sample_condition']]=row
    for key in points:
        case,cond=key.split('/'); row=rows['f68',case,cond]; folder=source_run/'cases'/case
        inputs += [folder/n for n in row['assets']]
        inputs += [source_run/row['image']]
        for model in ('C0','C1'): inputs.append(replay/rows[model,case,cond]['image'])
    inputs=list(dict.fromkeys(inputs)); assert not args.out.exists(),'不覆盖连续指标审计';args.out.mkdir(parents=True)
    protocol=dict(kind='native continuous display metric audit,read existing RGB only',
        selection='12 seen pastel plus8 fixed sentinels with4 overlaps;16 unique;no synthetic',
        interpretation='single-condition contrasts except complete12 pastel;not full48/336 or causal gradient qualification',
        original_losses_gates_step_and_S0_unchanged=True,new_S0=0,model_decodes=0,reverse_passes=0,optimizer_steps=0,holdout_read=0,adopted_checkpoints=[],
        paths=['train_F32_box64_positive_float','train_F64_box64_positive_float','preciseL_Gaussian24_positive_float','preciseL_Gaussian24_positive_PNG','preciseL_Gaussian24_goal_PNG','official_CV_continuous_RGB','official_PNG'],
        input_sha256={str(p.resolve()):sha(p) for p in inputs},source_sha256=dict(previous['source_sha256'],**{n:sha(SOURCE/n) for n in EXTRA}),
        execution_commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip())
    write(args.out/'protocol.json',protocol)
    result=dict(status='incomplete',passed=False,rows=[],RGB_identity_controls=0,PNG_identity_controls=0,overlapping_actual_RGB_controls=overlapping,
        new_S0=0,model_decodes=0,reverse_passes=0,optimizer_steps=0,holdout_read=0,formal_adoption_eligibility=False,adopted_checkpoints=[])
    for key,pt in points.items():
        case,cond=key.split('/'); item=rows['f68',case,cond]; folder=source_run/'cases'/case
        entry=dict(folder=folder,condition=item['condition'],sample_condition=cond,pair=pairs[case],assets=item['assets'],source_image=source_run/item['image'])
        mask=pixels(folder/'mask.png');mask=(mask[...,0] if mask.ndim==3 else mask)>0
        inside=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,5)>=16
        pos,goal,anchor=[pixels(folder/n) for n in (cond+'_positive.png',cond+'_goal.png','train_anchor.png')]
        floatpixels=lambda v:v[0].permute(1,2,0).double().numpy()*255
        positive_float=floatpixels(pt['positive'])
        p_lab,g_lab,a_lab=[to_lab(v) for v in (pos,goal,anchor)]
        p_float_lab=to_lab(positive_float)
        for model in ('f68','C0','C1'):
            rgb=pt['native_rgb_half'].float() if model=='f68' else pt['outputs'][model]
            prior=rows[model,case,cond]
            digest=tensor_sha(rgb); expected=gradient_rows[case,cond]['native_RGB_sha256'] if model=='f68' else prior['rgb_float32_sha256']
            assert digest==expected;result['RGB_identity_controls']+=1
            image=np.rint(rgb[0].permute(1,2,0).numpy()*255).astype(np.uint8)
            historical=source_run/prior['image'] if model=='f68' else replay/prior['image']
            assert np.array_equal(image,pixels(historical));result['PNG_identity_controls']+=1
            floating=floatpixels(rgb); current_lab=to_lab(floating)
            measured=evaluation.score_entry(entry,floating,'replay')
            exact_residue=residue_components(highpass(current_lab)[inside],highpass(p_lab)[inside],highpass(a_lab)[inside])
            train=gradient_rows[case,cond]['tasks']['light'] if model=='f68' else prior['training_tasks']['light']
            result['rows'].append(dict(case=case,sample_condition=cond,condition=item['condition'],group=item['group'],arm=item['arm'],model=model,
                rgb_float32_sha256=digest,pixel_array_sha256=tensor_sha(torch.from_numpy(image)),
                historical_training_light=train,train_F32_box64=box_light(rgb,pt['positive'],pt['weight'],torch.float32),
                train_F64_box64=box_light(rgb,pt['positive'],pt['weight'],torch.float64),
                preciseL_Gaussian24_positive_float=gaussian_light(current_lab[...,0],p_float_lab[...,0],mask),
                preciseL_Gaussian24_positive_PNG=gaussian_light(current_lab[...,0],p_lab[...,0],mask),
                preciseL_Gaussian24_goal_PNG=gaussian_light(current_lab[...,0],g_lab[...,0],mask),
                continuous_preciseLab_residue=exact_residue,continuous_official_CV=measured,png_official=dict(scores=prior['scores'],projection=prior['projection'])))
        print('NATIVE_METRIC_AUDIT',len(result['rows']),flush=True)
    assert result['RGB_identity_controls']==result['PNG_identity_controls']==48
    assert all(sha(Path(n))==h for n,h in protocol['input_sha256'].items())
    assert all(sha(SOURCE/n)==h for n,h in protocol['source_sha256'].items())
    result.update(status='complete',passed=True,source_final_verification=True,seconds=time.monotonic()-started,
        meaning_of_passed='source and historical RGB/PNG identity only;not image quality or gradient qualification')
    write(args.out/'result.json',result);names=['protocol.json','result.json']
    write(args.out/'manifest.json',{n:sha(args.out/n) for n in names})
    with zipfile.ZipFile(args.out/'review.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for n in names+['manifest.json']:archive.write(args.out/n,n)
    print('NATIVE_METRIC_AUDIT_REVIEW',sha(args.out/'review.zip'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prior',type=Path,required=True)
    args=parser.parse_args();args.auto=args.prior.resolve()/'autonomous_repair_20261116'
    args.out=args.auto/'reference_terminal_native_metric_audit';run(args)
