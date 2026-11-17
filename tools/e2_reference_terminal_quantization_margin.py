"""固定已存浅彩RGB，检查量化边界留距；不采样、不解码、不训练。"""
import argparse
from pathlib import Path
import subprocess
import time
import zipfile

import cv2
import numpy as np
from PIL import Image
import torch

from tools import e2_state_b_pair_evaluation as evaluation
from tools.e2_state_b_amplitude_response import read, write, sha, pixels
from tools.e2_state_b_backward_repro import tensor_sha


SOURCE = Path(__file__).resolve().parents[1]
# 单位为一个8位量化级距；预先固定，不按结果搜索或挑选。
MARGINS = (0., 1/64, 1/16, 1/4, 1/2)
EXTRA = ('tools/e2_reference_terminal_quantization_margin.py',
         'scripts/e2_reference_terminal_quantization_margin.sbatch')


def quantize(rgb):
    return np.rint(rgb[0].transpose(1, 2, 0)*255).astype(np.uint8)


def reanchor(native, current, margin):
    """重锚已观测且clamp后的RGB残差；不是重跑原始未裁剪解码差。"""
    assert native.dtype == current.dtype == np.float32 and 0 <= margin <= .5
    if margin == 0:
        return current.copy(), native.copy()
    q0 = np.rint(native*255).astype(np.float64)
    units = np.clip(native.astype(np.float64)*255, q0-.5+margin, q0+.5-margin)
    anchor = (units/255).clip(0, 1).astype(np.float32)
    return (anchor+(current-native)).clip(0, 1), anchor


def run(args):
    torch.set_num_threads(2); cv2.setNumThreads(2)
    started = time.monotonic()
    stages = args.auto/'reference_terminal_fp32_pair_metric_stages'
    prior_protocol, prior = read(stages/'protocol.json'), read(stages/'result.json')
    assert prior['status'] == 'complete' and prior['passed'] and prior['source_final_verification']
    assert all(sha(stages/n) == h for n,h in read(stages/'manifest.json').items())
    assert all(sha(Path(n)) == h for n,h in prior_protocol['input_sha256'].items())
    assert all(sha(SOURCE/n) == h for n,h in prior_protocol['source_sha256'].items())
    data = torch.load(stages/'diagnostic.pt',map_location='cpu',weights_only=False)['points']
    source_run = args.auto/'reference_terminal_one_step'
    baseline = read(source_run/'evaluation/f68/replay/result.json')
    spec = read(source_run/'protocol.json'); pairs={v['case_id']:v for v in spec['train']}
    selected = [v for v in prior['rows'] if v['model']=='f68']; assert len(selected)==12
    baselines={(r['case'],r['sample_condition']):r for r in baseline['rows']}
    inputs=[stages/n for n in ('protocol.json','manifest.json','result.json','diagnostic.pt')]
    inputs += [source_run/'protocol.json',source_run/'evaluation/f68/replay/result.json']
    for row in selected:
        folder=source_run/'cases'/row['case']; old=baselines[row['case'],row['sample_condition']]
        inputs += [folder/n for n in old['assets']]
        inputs += [folder/(row['condition']+'_positive.png'),source_run/old['image']]
    inputs=list(dict.fromkeys(inputs))
    assert not args.out.exists(), '不覆盖量化留距诊断'
    args.out.mkdir(parents=True)
    protocol=dict(kind='bounded observed RGB residual reanchoring counterfactual',
        selection='12 already-seen pastel conditions,not complete48 or holdout',
        margins_in_Q8_bin_units=list(MARGINS),production_code_modified=False,
        losses_and_gates_unchanged=True,forward_changed=True,
        gradient_cache_reusable_for_changed_anchor=False,
        counterfactual_limit='current RGB already clamped;not original raw decoder difference',
        new_S0=0,model_decodes=0,reverse_passes=0,optimizer_steps=0,holdout_read=0,adopted_checkpoints=[],
        input_sha256={str(p.resolve()):sha(p) for p in inputs},
        source_sha256=dict(prior_protocol['source_sha256'],**{n:sha(SOURCE/n) for n in EXTRA}),
        execution_commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip())
    write(args.out/'protocol.json',protocol)
    result=dict(status='incomplete',passed=False,baseline_rows=[],margins={},
        zero_PNG_controls=0,original_candidate_PNG_controls=0,completed_observations=0,
        new_S0=0,model_decodes=0,reverse_passes=0,optimizer_steps=0,holdout_read=0,
        formal_adoption_eligibility=False,adopted_checkpoints=[])
    entries={}
    for row in selected:
        case,cond=row['case'],row['sample_condition']; old=baselines[case,cond]
        entry=dict(folder=source_run/'cases'/case,condition=row['condition'],sample_condition=cond,
            pair=pairs[case],assets=old['assets'],source_image=source_run/old['image'])
        entries[case,cond]=entry
        image=quantize(data[case+'/'+cond]['native_rgb_half'].float().numpy())
        assert np.array_equal(image,pixels(entry['source_image']))
        measured=evaluation.score_entry(entry,image,'replay')
        result['baseline_rows'].append(dict(row,**measured))
    baseline_map={(r['case'],r['condition'],r['arm']):r for r in result['baseline_rows']}
    for margin in MARGINS:
        name='m'+format(margin,'.6f'); arms={}
        for model in ('C0','C1'):
            rows=[]
            for row in selected:
                case,cond=row['case'],row['sample_condition']; point=data[case+'/'+cond]
                native=point['native_rgb_half'].float().numpy(); current=point['outputs'][model].numpy()
                rgb,anchor=reanchor(native,current,margin); image=quantize(rgb)
                assert np.array_equal(quantize(anchor),quantize(native)); result['zero_PNG_controls']+=1
                if margin==0:
                    assert np.array_equal(rgb,current)
                    old=next(r for r in prior['rows'] if (r['model'],r['case'],r['sample_condition'])==(model,case,cond))
                    assert tensor_sha(torch.from_numpy(image))==old['pixel_array_sha256']
                    result['original_candidate_PNG_controls']+=1
                filename=Path(name)/model/(case+'_'+cond+'.png'); (args.out/filename).parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray(image).save(args.out/filename)
                measured=evaluation.score_entry(entries[case,cond],image,'replay')
                rows.append(dict(row,model=model,margin=margin,**measured,
                    image=str(filename.as_posix()),png_sha256=sha(args.out/filename),
                    pixel_array_sha256=tensor_sha(torch.from_numpy(image)),
                    anchor_changed_channels=int(np.count_nonzero(anchor!=native)),
                    native_exact_half_bin_channels=int(np.count_nonzero(np.mod(native*255,1)==.5)),
                    native_to_anchor_max=float(np.abs(anchor-native).max()),
                    changed_channels_from_f68=int(np.count_nonzero(image!=quantize(native)))))
                result['completed_observations']+=1
            arms[model]=dict(rows=rows,aggregate=evaluation.aggregate(rows,'replay'),
                protection=evaluation.strict_replay_protection(rows,baseline_map))
        # 记录原定义的逐身份中心斜率及alpha1增益误差，不以新门槛替代原门槛。
        response={}
        for model,v in arms.items():
            rr=v['aggregate']['responses']; mid=[r['scores'] for r in v['rows'] if r['arm']=='a100']
            response[model]=dict(slope_absolute_error=float(np.mean([abs(r['response']['center_slope']-1) for r in rr])),
                gain_absolute_error=float(np.mean([r['l_target_gain_error'] for r in mid])),
                correlation=float(np.mean([r['l_correlation'] for r in mid])))
        result['margins'][name]=dict(margin=margin,models=arms,response=response,
            C1_minus_C0={k:response['C1'][k]-response['C0'][k] for k in response['C1']})
        print('QUANTIZATION_MARGIN',margin,result['completed_observations'],flush=True)
    assert result['completed_observations']==120 and result['zero_PNG_controls']==120 and result['original_candidate_PNG_controls']==24
    assert all(sha(Path(n))==h for n,h in protocol['input_sha256'].items())
    assert all(sha(SOURCE/n)==h for n,h in protocol['source_sha256'].items())
    result.update(status='complete',passed=True,source_final_verification=True,seconds=time.monotonic()-started,
        meaning_of_passed='source and zero controls only;not quality,training qualification or complete protection')
    write(args.out/'result.json',result)
    names=['protocol.json','result.json']+[str(p.relative_to(args.out).as_posix()) for p in args.out.rglob('*.png')]
    write(args.out/'manifest.json',{n:sha(args.out/n) for n in names})
    with zipfile.ZipFile(args.out/'review.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for n in names+['manifest.json']: archive.write(args.out/n,n)
    print('QUANTIZATION_MARGIN_REVIEW',sha(args.out/'review.zip'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--prior',type=Path,required=True)
    args=parser.parse_args(); args.auto=args.prior.resolve()/'autonomous_repair_20261116'
    args.out=args.auto/'reference_terminal_quantization_margin'
    run(args)
