"""E32 StageA：完整BF划分、固定8000步、三seed，失败即停。"""

import argparse
import math
import random
from pathlib import Path

import numpy as np
import cv2
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data.e32_field_dataset import FieldDataset,prepare,cache_path
from data.e32_target_pseudogt import TAU
from data.e32_wrong_reference_sampler import wrong_references
from models.e32_field import ExplicitPatternField
from tools.e32_common import OUT,DATASET,WEIGHTS,read,write,sha,bootstrap,finish_frozen,git_commit

VARIANTS=('full','A_no_target_supervision','E_no_interior','G_no_E26','G_dino_only')


def variant_folder(out,seed,variant):
    return out/'A_geometry'/('seed%d'%seed) if variant=='full' else out/'ablations'/variant/('seed%d'%seed)


def variant_reference(reference,variant):
    result=reference.clone() if isinstance(reference,torch.Tensor) else reference.copy()
    if variant=='G_no_E26':result[...,384:388]=0
    if variant=='G_dino_only':result[...,384:]=0
    return result


def variant_batch(batch,variant):
    if variant=='E_no_interior':
        batch['supervision_geometry']=batch['supervision_unmasked_geometry']
        batch['supervision_interior']=batch['supervision_foreground']
    elif variant=='A_no_target_supervision':
        # 明确的proxy训练对照：reference可读geometry均值广播到sketch mask；没有target标签或pair权重。
        source=batch['reference'][...,384:388]
        conf=source[...,3]*(source[...,3]>=TAU)
        denom=conf.sum((1,2)).clamp_min(1e-8)
        ori=(source[...,:2]*conf[...,None]).sum((1,2))/denom[:,None]
        ori=F.normalize(ori,dim=-1,eps=1e-6)
        freq=(source[...,2]*conf).sum((1,2))/denom
        confidence=(source[...,3]*conf).sum((1,2))/denom
        proxy=torch.cat([ori,freq[:,None],confidence[:,None]],-1)[:,:,None,None].expand(-1,-1,64,48)
        sketch_mask=F.interpolate(batch['structure'][:,3:4],(64,48),mode='area')[:,0]
        batch['supervision_geometry']=proxy
        batch['supervision_interior']=(sketch_mask>=.95).float()
        batch['pair_weight']=torch.ones_like(batch['pair_weight'])
    return batch


def weighted_mean(values,weights):
    return (values*weights).sum()/weights.sum().clamp_min(1e-8)


def geometry_loss(prediction,batch,zero):
    gt=batch['supervision_geometry']
    mask=(batch['supervision_interior']>=.95).float()[:,None]
    weight=batch['pair_weight'].reshape(-1,1,1,1)
    matched=(~zero).float().reshape(-1,1,1,1)
    confidence=gt[:,3:4]
    valid=mask*(confidence>=TAU)*confidence*weight*matched
    orientation=weighted_mean(1-(prediction['orientation']*gt[:,:2]).sum(1,keepdim=True),valid)
    period=weighted_mean(F.smooth_l1_loss(prediction['log_frequency'],gt[:,2:3],reduction='none'),valid)
    target_confidence=confidence.repeat(1,2,1,1)*matched
    calibration=weighted_mean(F.binary_cross_entropy(prediction['confidence'][:,:2].clamp(1e-5,1-1e-5),
                                      target_confidence,reduction='none'),(mask*weight).expand(-1,2,-1,-1))
    smooth=orientation*0
    for axis in (2,3):
        left=[slice(None)]*4;right=[slice(None)]*4
        left[axis]=slice(None,-1);right[axis]=slice(1,None)
        left,right=tuple(left),tuple(right)
        edge=(1-(gt[:,:2][left]*gt[:,:2][right]).sum(1,keepdim=True)).clamp_min(0)
        edge=edge+abs(gt[:,2:3][left]-gt[:,2:3][right])
        weights=mask[left]*mask[right]*weight*matched*torch.exp(-5*edge)
        difference=(prediction['orientation'][left]-prediction['orientation'][right]).abs().mean(1,keepdim=True)
        difference=difference+abs(prediction['log_frequency'][left]-prediction['log_frequency'][right])
        smooth=smooth+weighted_mean(difference,weights)
    loss=orientation+period+.1*calibration+.05*smooth
    return loss,dict(orientation=float(orientation.detach()),period=float(period.detach()),
                     confidence=float(calibration.detach()),smoothness=float(smooth.detach()))


def train(args,seed):
    assert read(args.out/'decision_summary.json')['pair_dependence_pass']
    split=read(args.out/'split_manifest.json')
    folder=variant_folder(args.out,seed,args.variant)
    if (folder/'checkpoint_final.pt').exists():
        return
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    dataset=FieldDataset(args.out,split['train'])
    loader=DataLoader(dataset,batch_size=8,shuffle=True,num_workers=2,pin_memory=True,drop_last=True,
                       generator=torch.Generator().manual_seed(seed))
    model=ExplicitPatternField().cuda()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
    bf16=torch.cuda.is_bf16_supported()
    dtype=torch.bfloat16 if bf16 else torch.float16
    scaler=torch.cuda.amp.GradScaler(enabled=not bf16)
    folder.mkdir(parents=True,exist_ok=True)
    history=[];iterator=iter(loader)
    model.train()
    for step in range(1,8001):
        try:batch=next(iterator)
        except StopIteration:iterator=iter(loader);batch=next(iterator)
        batch={k:v.cuda(non_blocking=True) for k,v in batch.items() if isinstance(v,torch.Tensor)}
        batch=variant_batch(batch,args.variant)
        reference=variant_reference(batch['reference'],args.variant)
        zero=torch.rand(len(reference),device='cuda')<.1
        reference[zero]=0
        scale=min(step/500,1.) if step<=500 else .5*(1+math.cos(math.pi*(step-500)/7500))
        for group in optimizer.param_groups:group['lr']=1e-4*scale
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda',dtype=dtype):
            prediction=model(reference,batch['structure'])
            loss,parts=geometry_loss(prediction,batch,zero)
        assert torch.isfinite(loss), (seed,step,parts)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        scaler.step(optimizer);scaler.update()
        if step%100==0:
            record=dict(step=step,loss=float(loss.detach()),**parts)
            history.append(record)
            write(folder/'training_history.json',history)
            print('[E32 A train]',seed,step,record,flush=True)
    torch.save(dict(model=model.state_dict(),seed=seed,steps=8000,input_dim=395,
                    git_commit=git_commit(),variant=args.variant,appearance_trained=False),folder/'checkpoint_final.pt')
    write(folder/'training_protocol.json',dict(seed=seed,steps=8000,batch=8,train_cases=len(dataset),
          device=torch.cuda.get_device_name(),mixed_precision=str(dtype),reference_dropout=.1,
          loss_weights=dict(orientation=1.,period=1.,confidence=.1,smoothness=.05),
          GT_discontinuity_weight='exp(-5*(1-dot(O_i,O_j)+abs(logf_i-logf_j)))',
          trainable=['reference_projection','structure_encoder','cross_field_decoder','geometry_heads'],
          frozen=['DINO','E26','E5','U-Net','VAE','BF','TCPM','appearance_heads'],
          zero_reference='confidence-only loss; no geometry reconstruction'))


@torch.inference_mode()
def prediction(model,reference,structure):
    values=model(torch.as_tensor(reference[None],device='cuda',dtype=torch.float32),
                 torch.as_tensor(structure[None],device='cuda',dtype=torch.float32))
    return {k:v[0].cpu().numpy() for k,v in values.items()}


def errors(pred,gt,interior):
    readable=(interior>=.95)&(gt[3]>=TAU)
    weights=gt[3][readable]
    if not readable.any():
        return dict(orientation_error=None,period_log2_error=None,valid_geometry_coverage=float((pred['confidence'][:2].min(0)[interior>=.95]>=TAU).mean())
                    if (interior>=.95).any() else 0.,readable_cells=0)
    dot=np.clip((pred['orientation']*gt[:2]).sum(0),-1,1)
    ori=float(np.average(np.degrees(np.arccos(dot[readable]))/2,weights=weights))
    period=float(np.average(abs(pred['log_frequency'][0][readable]-gt[2][readable])/np.log(2),weights=weights))
    return dict(orientation_error=ori,period_log2_error=period,
                valid_geometry_coverage=float((pred['confidence'][:2].min(0)[readable]>=TAU).mean()),
                readable_cells=int(readable.sum()))


def evaluate(args,seed):
    split=read(args.out/'split_manifest.json');folder=variant_folder(args.out,seed,args.variant)
    checkpoint=torch.load(folder/'checkpoint_final.pt',map_location='cpu')
    model=ExplicitPatternField().cuda().eval();model.load_state_dict(checkpoint['model'])
    hashes=read(args.out/'audits/stage0_input_hashes.json')
    groups=dict(dev=split['dev'],causal_test=split['causal_test'],independent_confirmation=split['independent_confirmation'])
    summaries={}
    for name,records in groups.items():
        # held-out donor pool dev+18；选择只看reference histogram。
        pool=split['dev']+split['confirmation_all']
        cached={}
        features={}
        for row in pool:
            with np.load(cache_path(args.out,row['id'])) as data:
                cached[row['id']]={k:np.asarray(data[k],np.float32) for k in data.files}
                features[row['id']]=dict(histogram=cached[row['id']]['reference_histogram'])
        wrong=wrong_references(pool,features,hashes)
        rows=[]
        for row in records:
            sid=row['id'];case=cached[sid]
            refs=dict(matched=case['reference'],color_near=cached[wrong[sid]['color_near']]['reference'],
                      random=cached[wrong[sid]['random']]['reference'],zero=np.zeros_like(case['reference']),
                      rot90=case['reference_rot90'])
            predictions={arm:prediction(model,variant_reference(r,args.variant),case['structure']) for arm,r in refs.items()}
            measured={arm:errors(p,case['supervision_geometry'],case['supervision_interior']) for arm,p in predictions.items()}
            gt=case['supervision_geometry'];valid=(case['supervision_interior']>=.95)&(gt[3]>=TAU)
            sketch_gray=case['structure'][:3].mean(0)
            edges=cv2.resize((sketch_gray<.95).astype(np.float32),(48,64),interpolation=cv2.INTER_AREA)>.05
            edges=cv2.dilate(edges.astype(np.uint8),np.ones((3,3),np.uint8))>0
            structure_region=edges&(case['supervision_foreground']>=.95)
            pattern_region=valid&~edges
            confidence=predictions['matched']['confidence'][:2].mean(0)
            structure_ratio=float(confidence[structure_region].mean()/max(float(confidence[pattern_region].mean()),1e-3)) \
                            if structure_region.any() and pattern_region.any() else None
            if valid.any():
                dot=np.clip(-(predictions['matched']['orientation']*predictions['rot90']['orientation']).sum(0),-1,1)
                response=float(np.average(np.degrees(np.arccos(dot[valid]))/2,weights=gt[3][valid]))
            else:response=None
            rows.append(dict(id=sid,case_id=row.get('case_id'),arms=measured,rot90_response_error=response,
                             rot90_response_success=bool(response<=15) if response is not None else None,
                             structural_edge_response_ratio=structure_ratio,
                             wrong_references=wrong[sid]))
            if name=='dev':
                path=folder/'fields'/sid;path.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path/'predictions.npz',**{arm+'_'+key:value for arm,p in predictions.items()
                                   for key,value in p.items() if key!='appearance'},
                                   gt_geometry=gt,interior=case['supervision_interior'])
        summary=dict(case_count=len(rows),arms={},advantages={},rotation={})
        for arm in ('matched','color_near','random','zero','rot90'):
            summary['arms'][arm]={metric:bootstrap([r['arms'][arm][metric] for r in rows if r['arms'][arm][metric] is not None])
                                 for metric in ('orientation_error','period_log2_error','valid_geometry_coverage')}
        for arm in ('color_near','random','zero'):
            summary['advantages'][arm]={metric:bootstrap([r['arms'][arm][metric]-r['arms']['matched'][metric] for r in rows
                if r['arms']['matched'][metric] is not None]) for metric in ('orientation_error','period_log2_error')}
        summary['rotation']['error']=bootstrap([r['rot90_response_error'] for r in rows if r['rot90_response_error'] is not None])
        summary['rotation']['success']=bootstrap([float(r['rot90_response_success']) for r in rows if r['rot90_response_success'] is not None])
        summary['structural_edge_response_ratio']=bootstrap([r['structural_edge_response_ratio'] for r in rows if r['structural_edge_response_ratio'] is not None])
        summary['structure_diagnostic_definition']='sketch-line band / readable non-line interior confidence; automatic proxy, no collar/placket ground-truth labels'
        ori=summary['advantages']['color_near']['orientation_error']
        period=summary['advantages']['color_near']['period_log2_error']
        zero_ora=summary['advantages']['zero']['orientation_error']
        zero_per=summary['advantages']['zero']['period_log2_error']
        positive=lambda v:v['n']>=8 and v['ci95'][0]>0
        summary['gate_components']=dict(orientation_advantage=positive(ori) and ori['mean']>=10,
            period_advantage=positive(period) and period['mean']>=.15,
            rot90_response=summary['rotation']['success']['mean'] is not None and summary['rotation']['success']['mean']>=.7,
            zero_reference_degradation=positive(zero_ora) and positive(zero_per))
        summary['pass']=all(summary['gate_components'].values())
        write(folder/name/'rows.json',rows);write(folder/name/'summary.json',summary)
        summaries[name]=summary
        print('[E32 A eval]',seed,name,summary['gate_components'],flush=True)
    write(folder/'summary.json',summaries)
    return summaries


def decide(args):
    summaries={str(s):read(args.out/'A_geometry'/('seed%d'%s)/'summary.json')['dev'] for s in (42,43,44)}
    passing=[s for s,v in summaries.items() if v['pass']]
    decision=read(args.out/'decision_summary.json')
    decision['geometry_field_pass']=len(passing)>=2
    decision['matched_orientation_advantage_deg']=float(np.mean([v['advantages']['color_near']['orientation_error']['mean'] for v in summaries.values()]))
    decision['matched_period_advantage_log2']=float(np.mean([v['advantages']['color_near']['period_log2_error']['mean'] for v in summaries.values()]))
    decision['rot90_geometry_response']=float(np.mean([v['rotation']['success']['mean'] for v in summaries.values()]))
    decision['reference_ablation_pass']=len(passing)>=2
    decision['next_route']='appearance_field_training' if decision['geometry_field_pass'] else 'stop_geometry_field_reference_causality_failed'
    write(args.out/'decision_summary.json',decision)
    write(args.out/'A_geometry/three_seed_gate.json',dict(seeds=summaries,passing_seeds=passing,
          **{'pass':decision['geometry_field_pass']},bootstrap_unit='case per seed; repeated arms paired within case; seeds not independent cases'))
    write(args.out/'A_geometry/status.json',dict(status='complete',training_steps_per_seed=8000,passing_seeds=passing))
    if not decision['geometry_field_pass']:
        for name in ('B_appearance','C_scaffold','D_generation','E_confirmation'):
            write(args.out/name/'status.json',dict(status='not_run_stage_A_gate_failed',
                  reason='conditional downstream-stage training and corresponding ablations disallowed by hard stop'))
        write(args.out/'ablations/status.json',dict(status='stage_A_ablations_complete; downstream_ablations_not_run_stage_A_gate_failed',
              completed=list(VARIANTS[1:]),not_run=['B_no_geometry_heads','C_no_ranking','D_no_rot90_identity','F_no_color_near',
              'StageB_full_2x2_target_supervision_x_ranking'],reason='appearance-stage variants require StageA pass'))
    finish_frozen(args.out)
    print('[E32 A decision]',decision,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','train','eval','decide'])
    parser.add_argument('--seed',type=int,default=42,choices=[42,43,44])
    parser.add_argument('--variant',choices=VARIANTS,default='full')
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--dataset',type=Path,default=DATASET)
    parser.add_argument('--weights',type=Path,default=WEIGHTS)
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.out,args.dataset,args.weights)
    elif args.action=='train':train(args,args.seed)
    elif args.action=='eval':evaluate(args,args.seed)
    else:decide(args)


if __name__=='__main__':main()
