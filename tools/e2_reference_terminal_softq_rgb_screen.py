"""固定55份已存RGB筛查软量化；只算RGB导数，不算B参数G或训练。"""
import argparse
import math
from pathlib import Path
import subprocess
import time
import zipfile

import numpy as np
import torch
from models.e2_lab_anchor import rgb_to_lab
from tools.e2_b_endpoint_losses import luminance
from tools.e2_state_b_amplitude_response import read,write,sha


SOURCE=Path(__file__).resolve().parents[1]
SHARPNESS=(2.,8.,32.)
FACTORS=(-1.,-.5,0.,.5,1.)
EXTRA=('tools/e2_reference_terminal_softq_rgb_screen.py','scripts/e2_reference_terminal_softq_rgb_screen.sbatch')


def soft_q8(rgb,beta):
    """DSQ式固定tanh松弛；不学习阈值，不复现原论文训练，也不替换真实PNG量化器。"""
    units=rgb.clamp(0,1)*255
    cell=units.floor()
    fraction=units-cell
    return (cell+.5+.5*torch.tanh(beta*(fraction-.5))/math.tanh(beta/2))/255


def hard_q8(rgb):
    # 原CUDA除常数的倒数乘口径；不使用STE或反传。
    return (rgb*255).round().clamp(0,255)*(1./255)


def screen(saved,source,history):
    result=dict(rows=[],RGB_VJP=0,parameter_VJP=0,model_decodes=0,new_S0=0,optimizer_steps=0,holdout_read=0,
        formal_adoption_eligibility=False,source_control_scalar_max_difference=0.)
    old={(v['case'],v['sample_condition']):v for v in history['rows']}
    for key,pt in saved['points'].items():
        case,cond=key.split('/');record=source['points'][key]
        positive,weight,coeff=record['positive'],record['weight'],record['residue_coefficients']
        def losses(rgb):
            return dict(global_amplitude=luminance(rgb,positive,weight,[])[0]['amplitude'],
                residue=((rgb_to_lab(rgb)[0]*coeff).sum()-record['residue_offset']).abs())
        modes=dict(continuous=lambda x:x,hard_Q8=hard_q8,
            STE_RGB=lambda x:hard_q8(x).detach()+(x-x.detach()))
        modes.update({'soft_'+str(int(k)):lambda x,k=k:soft_q8(x,k) for k in SHARPNESS})
        pictures={h:pt['probes'][str(h)]['rgb'] for h in FACTORS}
        values={}
        with torch.no_grad():
            for mode,transform in modes.items():
                values[mode]={str(h):{n:float(v) for n,v in losses(transform(pictures[h])).items()} for h in FACTORS}
        for mode,old_mode in [('continuous','continuous_RGB'),('hard_Q8','Q8_STE')]:
            for probe in old[case,cond]['probes']:
                for task,value in probe['values'][old_mode].items():
                    result['source_control_scalar_max_difference']=max(result['source_control_scalar_max_difference'],abs(value-values[mode][str(probe['factor'])][task]))
        tangents={str(h):(pictures[h].double()-pictures[-h].double())/(2*h) for h in (.5,1.)}
        predicted={}
        for mode,transform in modes.items():
            if mode=='hard_Q8':continue
            rgb=pictures[0.].clone().requires_grad_(True);terms=losses(transform(rgb));predicted[mode]={}
            for i,task in enumerate(('global_amplitude','residue')):
                gradient=torch.autograd.grad(terms[task],rgb,retain_graph=i==0)[0]
                assert torch.isfinite(gradient).all();result['RGB_VJP']+=1
                predicted[mode][task]={h:float((gradient.double()*tangent).sum()) for h,tangent in tangents.items()}
        result['rows'].append(dict(case=case,sample_condition=cond,group=old[case,cond]['group'],
            values=values,RGB_tangent_predictions=predicted,
            interpretation='RGB AD contracted with stored symmetric RGB secants;not fresh B-parameter derivative'))
        print('SOFTQ_RGB_SCREEN',len(result['rows']),result['RGB_VJP'],flush=True)
    assert len(result['rows'])==11 and result['RGB_VJP']==110
    summary={}
    for mode in ('continuous','STE_RGB')+tuple('soft_'+str(int(k)) for k in SHARPNESS):
        summary[mode]={}
        for task in ('global_amplitude','residue'):
            summary[mode][task]={}
            for h in (.5,1.):
                rel=[];same=0;own_rel=[]
                for row in result['rows']:
                    prediction=row['RGB_tangent_predictions'][mode][task][str(h)]
                    hard=(row['values']['hard_Q8'][str(h)][task]-row['values']['hard_Q8'][str(-h)][task])/(2*h)
                    own=(row['values'][mode][str(h)][task]-row['values'][mode][str(-h)][task])/(2*h)
                    same+=int(np.sign(prediction)==np.sign(hard))
                    rel.append(abs(prediction-hard)/max(abs(hard),1e-12));own_rel.append(abs(prediction-own)/max(abs(own),1e-12))
                summary[mode][task][str(h)]=dict(hard_Q8_sign_agreement=same,total=11,
                    hard_Q8_median_relative_error=float(np.median(rel)),own_soft_median_relative_error=float(np.median(own_rel)))
    result['summary']=summary
    return result


def run(args):
    torch.set_num_threads(2);started=time.monotonic();direction=args.auto/'reference_terminal_fp32_direction';q8=args.auto/'reference_terminal_q8_direction_diagnostic'
    previous=read(direction/'protocol.json')
    for folder in (direction,q8):assert all(sha(folder/n)==h for n,h in read(folder/'manifest.json').items())
    assert read(direction/'result.json')['passed'] and all(sha(SOURCE/n)==h for n,h in previous['source_sha256'].items())
    inputs=[folder/n for folder in (direction,q8) for n in ('protocol.json','result.json','manifest.json','diagnostic.pt')]
    assert not args.out.exists(),'不覆盖软量化RGB筛查';args.out.mkdir(parents=True)
    protocol=dict(kind='stored RGB secant screen of fixed soft Q8;not B gradient qualification',
        reference='https://arxiv.org/abs/1908.05033',adaptation='fixed beta in8bit image RGB;not learned DSQ weights/activations training',
        sharpness=list(SHARPNESS),selection='same11 seen G1 controls;55 stored RGB',
        loss_formula_unchanged=True,training_forward_counterfactual=True,formal_PNG_quantizer_unchanged=True,
        old_parameter_G_reusable=False,new_S0=0,model_decodes=0,parameter_VJP=0,RGB_VJP=110,optimizer_steps=0,holdout_read=0,adopted_checkpoints=[],
        input_sha256={str(p.resolve()):sha(p) for p in inputs},source_sha256=dict(previous['source_sha256'],**{n:sha(SOURCE/n) for n in EXTRA}),
        execution_commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip())
    write(args.out/'protocol.json',protocol)
    result=screen(torch.load(direction/'diagnostic.pt',map_location='cpu',weights_only=False),torch.load(q8/'diagnostic.pt',map_location='cpu',weights_only=False),read(direction/'result.json'))
    assert all(sha(Path(n))==h for n,h in protocol['input_sha256'].items())
    assert all(sha(SOURCE/n)==h for n,h in protocol['source_sha256'].items())
    result.update(audit_completed=True,source_final_verification=True,seconds=time.monotonic()-started)
    write(args.out/'result.json',result);names=['protocol.json','result.json'];write(args.out/'manifest.json',{n:sha(args.out/n) for n in names})
    with zipfile.ZipFile(args.out/'review.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for n in names+['manifest.json']:archive.write(args.out/n,n)
    print('SOFTQ_RGB_SCREEN_REVIEW',sha(args.out/'review.zip'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prior',type=Path,required=True)
    args=parser.parse_args();args.auto=args.prior.resolve()/'autonomous_repair_20261116';args.out=args.auto/'reference_terminal_softq_rgb_screen';run(args)
