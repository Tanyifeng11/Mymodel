"""真实6arm和原controlled完整评测；每个target是统计单位。"""
import numpy as np,torch
from data.e33rf_real_rotation_dataset import reference_group
from data.e32_field_dataset import cache_path
from data.e33rc_real_pair_dataset import orientation_reference
from tools.e33r_evaluate import axial,evaluate_records
from tools.e33rf_common import *

def evaluate_real(model,records,destination,diagnostic=False):
    destination.mkdir(parents=True,exist_ok=True);model.eval();wrong=read(OUT/'real_wrong_eval.json');rows=[]
    drift_ids=set(read(OUT/'feature_drift_ids.json'));drift=[]
    for row in records:
        case=reference_group(row,'RF2');refs=list(case['reference'].numpy())
        for arm in ['color_near','random']:
            with np.load(cache_path(E32,wrong[row['id']][arm])) as z:refs.append(orientation_reference(z['reference']))
        refs.append(np.zeros_like(refs[0]));reference=torch.as_tensor(np.stack(refs),device='cuda')
        structure=case['structure'].cuda();values=[]
        with torch.no_grad():
            for i in range(0,6,2):
                with torch.autocast('cuda',dtype=torch.bfloat16):p=model(reference[i:i+2],structure[None].expand(2,-1,-1,-1))
                values.append({k:v.float().cpu().numpy() for k,v in p.items()})
        pred={k:np.concatenate([v[k] for v in values]) for k in values[0]}
        gt=case['gt'].numpy();support=case['support'].numpy();w=gt[3][support]
        finite=all(np.isfinite(v).all() for v in pred.values())
        avg=lambda v:float(np.average(v[support],weights=w)) if support.any() and finite else None
        names=['matched','rot90','rot180','color_near','random','zero']
        errors={arm+'_error':avg(axial(pred['orientation'][i],gt[:2])) for i,arm in enumerate(names)}
        response=avg(axial(pred['orientation'][1],-pred['orientation'][0]));r180=avg(axial(pred['orientation'][2],pred['orientation'][0]))
        mean_delta=float(np.linalg.norm(.1*pred['adapter_delta'][0],axis=-1).mean())
        rows.append(dict(id=row['id'],finite=finite,readable_cells=int(support.sum()),wrong_references=wrong[row['id']],
            near_advantage=errors['color_near_error']-errors['matched_error'] if support.any() and finite else None,
            random_advantage=errors['random_error']-errors['matched_error'] if support.any() and finite else None,
            rot90_response_error=response,rot90_response_success=response<=15 if response is not None else None,
            r180_response_error=r180,r180_identity_success=r180<=15 if r180 is not None else None,
            adapter_residual_norm=mean_delta,**errors))
        if not diagnostic:
            field=destination/'fields';field.mkdir(exist_ok=True)
            np.savez_compressed(field/(row['id']+'.npz'),orientation=pred['orientation'],q=pred['q'],
                confidence_logits=pred['confidence_logits'],gt=gt,support=support,adapter_residual=pred['adapter_delta'])
            if row['id'] in drift_ids:
                before=reference.cpu().numpy();after=pred['adapter_features']
                cos=lambda a,b:(a*b).sum(-1)/np.maximum(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1),1e-8)
                raw=pred['adapter_delta']
                drift.append(dict(id=row['id'],cosine_drift=float((1-cos(before[0],after[0])).mean()),
                    raw_residual_norm=float(np.linalg.norm(raw[0],axis=-1).mean()),scaled_residual_norm=mean_delta,
                    rotation_relative_cosine_change=float(abs(cos(after[0],after[1])-cos(before[0],before[1])).mean())))
                d=destination.parent/'feature_drift';d.mkdir(exist_ok=True)
                np.savez_compressed(d/(row['id']+'.npz'),before=before[:3],after=after[:3],raw_residual=raw[:3])
    write(destination/'rows.json',rows)
    if diagnostic:summary=dict(case_count=len(rows),used_for_gate=False,diagnostic_only=True)
    else:
        stat=lambda key:bootstrap([float(r[key]) for r in rows if r[key] is not None])
        summary=dict(case_count=len(rows),readable_cases=sum(r['readable_cells']>0 for r in rows),finite_prediction_rate=sum(r['finite'] for r in rows)/len(rows),
            errors={arm:stat(arm+'_error') for arm in names},near_advantage=stat('near_advantage'),random_advantage=stat('random_advantage'),
            rot90_error=stat('rot90_response_error'),rot90_success=stat('rot90_response_success'),r180_error=stat('r180_response_error'),
            r180_identity=stat('r180_identity_success'),adapter_residual_norm=stat('adapter_residual_norm'))
        summary['gate']=real_gate(summary);write(destination.parent/'feature_drift/rows.json',drift)
    write(destination/'summary.json',summary);return summary
def evaluate(model,seed,destination,dino):
    split=read(OUT/'split_manifest.json');cf=read(OUT/'controlled_manifest.json');model.eval()
    _,controlled=evaluate_records(model,cf['dev'],dino,destination/'controlled','full',model.prior,True)
    controlled['preservation_gate']=controlled_gate(controlled);write(destination/'controlled/summary.json',controlled)
    real=evaluate_real(model,split['dev'],destination/'real')
    for key in ['causal_test','independent_confirmation']:evaluate_real(model,split[key],destination/key,True)
    record=dict(seed=seed,phase=destination.name,controlled=controlled['preservation_gate'],real=real['gate'],
        pilot=pilot_gate(controlled,real),controlled_clean=controlled['clean_r90_success']['mean'],
        matched_error=real['errors']['matched']['mean'],near_advantage=real['near_advantage']['mean'],real_rot90=real['rot90_success']['mean'])
    write(destination/'decision_record.json',record);print('[E33RF Gate]',record,flush=True);return record
