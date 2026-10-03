"""同一真实target的五arm配对测量；heldout不用于checkpoint/参数选择。"""
import numpy as np
import torch
from data.e33rc_real_pair_dataset import cached_real,orientation_reference
from data.e32_field_dataset import cache_path
from tools.e33r_evaluate import axial,evaluate_records
from tools.e33rc_common import *

def summarize_real(rows):
    stat=lambda key:bootstrap([float(r[key]) for r in rows if r[key] is not None])
    summary=dict(case_count=len(rows),readable_cases=sum(r['readable_cells']>0 for r in rows),
        finite_prediction_rate=sum(r['finite'] for r in rows)/len(rows),
        errors={arm:stat(arm+'_error') for arm in ['matched','color_near','random','rot90','zero']},
        near_advantage=stat('near_advantage'),random_advantage=stat('random_advantage'),
        zero_advantage=stat('zero_advantage'),rot90_error=stat('rot90_response_error'),rot90_success=stat('rot90_response_success'))
    summary['gate']=real_gate(summary)
    return summary

@torch.no_grad()
def evaluate_real(model,records,folder,save_fields=True,diagnostic=False):
    model.eval();wrong=read(OUT/'real_wrong_eval.json');rows=[];folder.mkdir(parents=True,exist_ok=True)
    for index,row in enumerate(records,1):
        case=cached_real(row,True);refs=[case['reference']]
        for name in ['color_near','random']:
            with np.load(cache_path(E32,wrong[row['id']][name])) as z:refs.append(orientation_reference(z['reference']))
        refs += [case['reference_rot90'],np.zeros_like(case['reference'])]
        reference=torch.as_tensor(np.stack(refs),device='cuda');structure=torch.as_tensor(case['structure'],device='cuda')
        values=[]
        for start in range(0,5,2):
            with torch.autocast('cuda',dtype=torch.bfloat16):p=model(reference[start:start+2],structure[None].expand(min(2,5-start),-1,-1,-1))
            values.append({k:v.float().cpu().numpy() for k,v in p.items()})
        pred={k:np.concatenate([v[k] for v in values]) for k in values[0]}
        gt=case['supervision_geometry'];support=(case['supervision_interior']>=.95)&(gt[3]>=.25);w=gt[3][support]
        finite=all(np.isfinite(v).all() for v in pred.values())
        mean=lambda v:float(np.average(v[support],weights=w)) if support.any() and finite else None
        errors={name+'_error':mean(axial(pred['orientation'][i],gt[:2])) for i,name in enumerate(['matched','color_near','random','rot90','zero'])}
        response=mean(axial(pred['orientation'][3],-pred['orientation'][0]));refmotion=mean(axial(pred['orientation'][3],pred['orientation'][0]))
        near=errors['color_near_error']-errors['matched_error'] if support.any() and finite else None
        random=errors['random_error']-errors['matched_error'] if support.any() and finite else None
        zero=errors['zero_error']-errors['matched_error'] if support.any() and finite else None
        cats=[]
        if near is not None and near<=0:cats.append('RC2_wrong_beats_matched')
        if response is not None and response>15:cats.append('RC3_no_response' if refmotion<75 else 'RC4_wrong_axis')
        if errors['matched_error'] is not None and errors['matched_error']>7:cats.append('RC6_absolute_drift')
        # 差值与置信区间原样保留；模糊的“≈”只作图像诊断，不新增成功阈值。
        rows.append(dict(id=row['id'],case_id=row.get('case_id'),finite=finite,readable_cells=int(support.sum()),
            wrong_references=wrong[row['id']],near_advantage=near,random_advantage=random,zero_advantage=zero,
            rot90_response_error=response,rot90_response_success=response<=15 if response is not None else None,
            failure_categories=cats,**errors))
        if save_fields:
            dest=folder/'fields';dest.mkdir(exist_ok=True)
            np.savez_compressed(dest/(row['id']+'.npz'),**pred,gt=gt,support=support,interior=case['supervision_interior'])
        if index%128==0:print('[E33RC real eval]',folder,index,'/',len(records),flush=True)
    write(folder/'rows.json',rows)
    summary=summarize_real(rows) if not diagnostic else dict(case_count=len(rows),diagnostic_only=True,used_for_gate=False)
    write(folder/'summary.json',summary);return summary

def evaluate_dual(model,seed,folder,dino,variant='full'):
    controlled=read(OUT/'controlled_manifest.json');split=read(OUT/'split_manifest.json')
    _,cf=evaluate_records(model,controlled['dev'],dino,folder/'controlled',variant_model(variant),model.prior,True)
    cf['retention_gate']=retention_gate(cf);write(folder/'controlled/summary.json',cf)
    real=evaluate_real(model,split['dev'],folder/'real')
    for name in ['causal_test','independent_confirmation']:
        evaluate_real(model,split[name],folder/name,False,True)
    record=dict(stage=folder.name,seed=seed,controlled_retention=cf['retention_gate'],real_gate=real['gate'],
        controlled_clean_r90=cf['clean_r90_success']['mean'],controlled_noisy_r90=cf['noisy_r90_both_success']['mean'],
        real_match_error=real['errors']['matched']['mean'],real_near_error=real['errors']['color_near']['mean'],
        real_near_advantage=real['near_advantage']['mean'],real_rot90=real['rot90_success']['mean'],
        **{'continue':cf['retention_gate']['pass']})
    write(folder/'decision_record.json',record)
    print('[E33RC Dual Gate]',record,flush=True);return record
