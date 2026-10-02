"""按case统计和paired bootstrap；held-out不用于训练选择。"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from data.e33r_group_dataset import GroupDataset,encode_reference,jitter_structure
from tools.e33r_common import OUT,DATASET,bootstrap,write

def axial(a,b):return np.degrees(np.arccos(np.clip((a*b).sum(0),-1,1)))/2

def ratio_bootstrap(numerator,denominator):
    a=np.asarray(numerator,float);b=np.asarray(denominator,float)
    if not len(a) or b.mean()<=1e-12:return dict(mean=None,ci95=None,n=len(a))
    rng=np.random.default_rng(32042);draws=[]
    for _ in range(2000):
        index=rng.integers(len(a),size=len(a));d=b[index].mean()
        if d>1e-12:draws.append(a[index].mean()/d)
    return dict(mean=float(a.mean()/b.mean()),ci95=np.percentile(draws,[2.5,97.5]).tolist() if draws else None,n=len(a))

def summarize(rows):
    readable=[r for r in rows if r['readable_cells']>0 and r['finite']]
    names=['clean_r90_success','noisy_r90_both_success','r180_identity_success','r0_identity_success',
           'r90_response_error','r180_response_error','r0_response_error','r0_absolute_error',
           'r90_absolute_error','r180_absolute_error','prior_absolute_error','prior_delta_deg',
           'zero_identity_l1','s_ref','s_sketch','d_zero','d_valid']
    values={name:bootstrap([float(r[name]) for r in rows if r[name] is not None]) for name in names}
    values['zero_ratio']=ratio_bootstrap([r['d_zero'] for r in readable],[r['d_valid'] for r in readable])
    values['sensitivity_ratio']=ratio_bootstrap([r['s_ref'] for r in readable],[r['s_sketch'] for r in readable])
    values.update(denominator=len(rows),readable_cases=len(readable),empty_support_cases=len(rows)-len(readable),
                  finite_prediction_rate=sum(r['finite'] for r in rows)/max(len(rows),1))
    thresholds={'clean_r90_success':(.90,True),'noisy_r90_both_success':(.85,True),
         'r180_identity_success':(.90,True),'r0_identity_success':(.90,True),
         'zero_ratio':(.10,False),'sensitivity_ratio':(2.,True),'prior_delta_deg':(1.,False)}
    checks={};deviations={}
    for key,(threshold,larger) in thresholds.items():
        v=values[key]['mean'];checks[key]=v is not None and (v>=threshold if larger else v<=threshold)
        if not checks[key]:deviations[key]=None if v is None else ((threshold-v) if larger else (v-threshold))/threshold
    values.update(checks=checks,gate_pass=all(checks.values()) and values['finite_prediction_rate']==1,
          near_gate=len(deviations)==1 and list(deviations.values())[0] is not None and list(deviations.values())[0]<=.10
                    and values['finite_prediction_rate']==1,failed_threshold_relative_deviations=deviations)
    return values

def case_metrics(row,pred,gt,support,jitter_meta):
    ori,q=pred['orientation'],pred['q'];prior=pred['frozen_prior']
    weight=gt[3]*support;valid=support.astype(bool);w=weight[valid]
    finite=all(np.isfinite(v).all() for v in pred.values())
    mean=lambda v:float(np.average(v[valid],weights=w)) if valid.any() and finite else None
    errors=dict(r90_response_error=mean(axial(ori[1],-ori[0])),r180_response_error=mean(axial(ori[2],ori[0])),
         r0_response_error=mean(axial(ori[0],prior)),r0_absolute_error=mean(axial(ori[0],gt[:2])),
         r90_absolute_error=mean(axial(ori[1],-gt[:2])),r180_absolute_error=mean(axial(ori[2],gt[:2])),
         prior_absolute_error=mean(axial(prior,gt[:2])),zero_identity_l1=mean(abs(q[9,0]-1)+abs(q[9,1])),
         s_ref=mean(axial(ori[0],ori[1])),s_sketch=mean(axial(ori[0],ori[10])),
         d_zero=mean(1-q[9,0]),d_valid=mean(1-q[1,0]))
    errors['prior_delta_deg']=errors['r0_absolute_error']-errors['prior_absolute_error'] if errors['r0_absolute_error'] is not None else None
    noisy=[mean(axial(ori[k+1],-ori[k])) for k in (3,6)]
    success=lambda value:value is not None and value<=15
    errors.update(clean_r90_success=success(errors['r90_response_error']),r180_identity_success=success(errors['r180_response_error']),
          r0_identity_success=success(errors['r0_response_error']),noisy_r90_both_success=all(success(v) for v in noisy))
    categories=[]
    if errors['s_ref'] is not None and errors['s_ref']<15:categories+=['R1_no_reference_shortcut','R2_prior_domination']
    if not errors['clean_r90_success']:
        categories.append('R3_R90_under_response' if errors['s_ref'] is not None and errors['s_ref']<75 else 'R4_R90_wrong_axis_response')
    if not errors['r180_identity_success']:categories.append('R5_R180_false_response')
    if errors['d_zero'] is not None and errors['d_zero']>.1*errors['d_valid']:categories.append('R6_zero_nonzero_control')
    if not errors['noisy_r90_both_success']:categories.append('R7_nuisance_sensitive')
    if errors['prior_delta_deg'] is not None and errors['prior_delta_deg']>1:categories.append('R8_absolute_drift')
    if not valid.any():categories.append('R9_empty_readable_support')
    return dict(id=row['id'],strict=bool(row['strict']),readable_cells=int(valid.sum()),finite=finite,
          noisy_r90_errors=noisy,jitter=jitter_meta,failure_categories=categories,**errors)

@torch.no_grad()
def evaluate_records(model,records,dino,folder,variant='full',frozen_prior=None,save_fields=False,diagnostic_only=False):
    folder.mkdir(parents=True,exist_ok=True);model.eval();frozen_prior=frozen_prior or model.prior
    # 连续评测会在CUDA/OpenCV线程已启动后再次fork，可能在首个case阻塞。
    # 评测使用主进程读取；两次nuisance仍由固定case seed确定，指标口径不变。
    loader=DataLoader(GroupDataset(records),batch_size=1,shuffle=False,num_workers=0,pin_memory=True)
    rows=[]
    for index,(row,case) in enumerate(zip(records,loader),1):
        reference=encode_reference(case['pixels'],case['aux'],dino,variant)
        # 0..8 clean/noise;9zero;10jitter，保持eval与train的zero下标明确分离。
        reference=torch.cat([reference,torch.zeros_like(reference[:,:1]),reference[:,:1]],1).flatten(0,1)
        structure=case['structure'].cuda();jitter,meta=jitter_structure(row)
        structures=torch.cat([structure.expand(10,-1,-1,-1),jitter[None].cuda()],0)
        values=[]
        for start in range(0,11,2):
            with torch.autocast('cuda',dtype=torch.bfloat16):p=model(reference[start:start+2],structures[start:start+2])
            values.append({k:v.float().cpu().numpy() for k,v in p.items()})
        pred={k:np.concatenate([v[k] for v in values]) for k in values[0]}
        with torch.autocast('cuda',dtype=torch.bfloat16):baseline=frozen_prior(structure)['orientation']
        pred['frozen_prior']=baseline[0].float().cpu().numpy()
        gt=case['gt'][0].numpy();support=case['support'][0].numpy()
        rows.append(case_metrics(row,pred,gt,support,meta))
        if save_fields:
            path=folder/'fields'/(row['id']+'.npz');path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path,orientation=pred['orientation'],q=pred['q'],
                 confidence=1/(1+np.exp(-np.clip(pred['confidence_logits'],-60,60))),
                 prior=pred['frozen_prior'],gt=gt,support=support,
                 source_geometry=case['aux'][0,:3,:,:,:3].numpy())
        if index%128==0:print('[E33R eval]',folder,index,'/',len(records),flush=True)
    write(folder/'rows.json',rows);summary=summarize(rows)
    summary['strict']=summarize([r for r in rows if r['strict']])
    if diagnostic_only:summary=dict(denominator=len(rows),diagnostic_only=True,raw_case_table='rows.json',used_for_gate=False)
    write(folder/'summary.json',summary)
    return rows,summary
