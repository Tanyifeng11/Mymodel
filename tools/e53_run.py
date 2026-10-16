"""阶段入口：回放/数值门禁、人工核验、pilot、冻结lock各自留痕。"""
import argparse
import json
import traceback
from datetime import datetime,timezone
import numpy as np
import torch
from tools.e53_protocol import *

def targets(baseline,window='ALL'):
    base=np.asarray(baseline,np.float64);active=WINDOWS[window] if isinstance(window,str) else window
    out=np.zeros(50,np.float64);energy=float(np.dot(base,base));part=float(np.dot(base[active],base[active]))
    assert part>0 and len(base)==50,'INVALID_ENERGY_BASELINE'
    out[active]=base[active]*np.sqrt(energy/part)
    assert abs(float(np.dot(out,out))/energy-1)<1e-12
    return out.tolist()

def baseline_energy(r,row,stage):
    from tools.e53_runtime import case
    value=case(r,row,'BUDGET_BASELINE',stage,dict(ARMS['ORIGINAL']),None,full=stage=='audit_energy')
    assert value['generation']['mask']['valid'],'SYNTHETIC_SKETCH_MASK_INVALID'
    return [s['energy']['after'] for s in value['steps']]

def audit():
    from tools.e53_runtime import runtime,case,clear,release
    from tools.e47_runtime import exact_trace
    prepare();rows=read(HISTORY/'splits/dev32.json');r=runtime('audit/runtime');checks=[]
    try:
        for row in rows:
            clear(r)
            for arm,mode in [('ORIGINAL','ORIGINAL'),('REPLAY','REPLAY')]:
                current=case(r,row,arm,'audit_replay',dict(geometry='N',texture=0,routing=mode,window='ALL'),None,full=True)
                path=image_path('audit_replay',arm,row['id']);source=HISTORY/'images/p0/B1'/(row['id']+'.png')
                result=exact_trace(path.with_suffix('.pt'),source.with_suffix('.pt'))
                assert result['bit_exact'] and sha(path)==sha(source),('BASELINE_REPLAY_FAIL',row['id'],arm,result)
                original_states=torch.load(source.with_name(source.stem+'_states.pt'),map_location='cpu',weights_only=False)['latents']
                states=torch.load(path.with_name(path.stem+'_latents.pt'),map_location='cpu',weights_only=False)['latents_before']
                assert torch.equal(states,original_states),'LATENT_STEP_REPLAY_FAIL'
                assert all(s.get('joint_exact',True) for s in current['steps']),'JOINT_HIDDEN_REPLAY_FAIL'
                checks.append(dict(id=row['id'],arm=arm,epsilon_latent_exact=result,all_step_latents_exact=True,png_exact=True,joint_exact=True,mask_valid=current['generation']['mask']['valid']))
        row=rows[0];clear(r);case(r,row,'E5','audit_replay',ARMS['E5'],None,full=False)
        new=image_path('audit_replay','E5',row['id']);old=HISTORY/'images/p0/E5'/(row['id']+'.png')
        assert exact_trace(new.with_suffix('.pt'),old.with_suffix('.pt'))['bit_exact'] and sha(new)==sha(old),'E5_REPLAY_FAIL'
    finally:release(r)
    write(OUT/'audit/exact_replay.json',dict(passed=True,n=32,comparisons=checks,commit=commit()))
    decision(P0='PASS');print('E53 EXACT REPLAY PASS',flush=True)

def energy_gate():
    from tools.e53_runtime import runtime,case,clear,release
    assert read(OUT/'audit/exact_replay.json')['passed']
    rows=read(OUT/'splits/pilot.json');r=runtime('audit_energy/runtime');checks=[]
    try:
        for row in [rows[0],rows[3]]:
            clear(r);base=baseline_energy(r,row,'audit_energy');total=float(np.dot(base,base))
            for arm in ['ORIGINAL','ORACLE','SHIFT','SUPPORT','G0','G1','EARLY','MIDDLE','LATE','STEP00','STEP48']:
                spec=ARMS[arm];budget=targets(base,spec['window']);v=case(r,row,arm,'audit_energy',spec,budget,full=True)
                error=abs(v['total_injection_energy']/total-1)
                assert error<1e-4,('TOTAL_ENERGY_MATCH_FAIL',row['id'],arm,error)
                checks.append(dict(id=row['id'],arm=arm,total_energy=v['total_injection_energy'],baseline_total_energy=total,total_relative_error=error,max_step_error=max(s['energy']['relative_error'] for s in v['steps']),max_gain=max(s['energy']['gain'] for s in v['steps']),unconditional_energy=0.))
    finally:release(r)
    write(OUT/'audit/energy_gate.json',dict(passed=True,checks=checks,commit=commit()));decision(energy='PASS');print('E53 ENERGY GATE PASS',flush=True)

def generate(stage,shard=0,shards=1):
    from tools.e53_runtime import runtime,case,clear,release
    assert read(OUT/'audit/exact_replay.json')['passed'] and read(OUT/'audit/energy_gate.json')['passed']
    approval=read(OUT/'audit/human_mapping_review.json');assert approval['approved'] and approval['reviewer']=='user','MANUAL_MAPPING_REVIEW_REQUIRED'
    if stage=='lock':assert read(OUT/'configs/lock.json')['frozen']
    rows=read(OUT/'splits'/(stage+'.json'));arms=PILOT_ARMS if stage=='pilot' else LOCK_ARMS;r=runtime(stage+'/runtime_%d'%shard)
    try:
        for i,row in enumerate(rows):
            if i%shards!=shard:continue
            clear(r);base=baseline_energy(r,row,stage);total=float(np.dot(base,base))
            for arm in arms:
                spec=ARMS[arm];energy=None if arm=='E5' else targets(base,spec['window']);v=case(r,row,arm,stage,spec,energy,full=stage=='pilot')
                if arm!='E5':assert abs(v['total_injection_energy']/total-1)<1e-4,'TOTAL_ENERGY_MATCH_FAIL'
            print('E53 IDENTITY COMPLETE',stage,i,len(rows),flush=True)
    finally:release(r)
    write(OUT/'resources'/('%s_shard%d_complete.json'%(stage,shard)),dict(stage=stage,shard=shard,shards=shards,n=sum(i%shards==shard for i in range(len(rows))),arms=arms,commit=commit()))

def freeze_lock():
    from tools.e53_data import synthesize,mapping_review
    assert read(OUT/'metrics/pilot/summary.json')['engineering_complete']
    assert read(OUT/'audit/texture_calibration.json')['passed']
    assert read(OUT/'audit/human_mapping_review.json')['approved']
    # 不按pilot效果删除对照：固定矩阵早于任何pilot模型输出。
    rows=synthesize('lock',64);mapping_review(rows,'lock')
    freeze(OUT/'splits/lock.json',rows)
    freeze(OUT/'configs/lock.json',dict(frozen=True,arms=LOCK_ARMS,sigma_uv=CONFIG['sigma_uv'],wrong_shift=CONFIG['wrong_shift'],thresholds=CONFIG['practical'],identity_ids=[x['id'] for x in rows],predeclared_matrix=True,commit=commit()))
    decision(pilot='COMPLETE',lock='FROZEN_NOT_RUN')

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','data','audit','energy','generate','evaluate','freeze','accept_mapping'])
    p.add_argument('--stage',default='pilot',choices=['pilot','lock']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--review-note');a=p.parse_args();torch.set_num_threads(2)
    try:
        if a.action=='prepare':prepare()
        elif a.action=='data':
            from tools.e53_data import main as data
            data()
        elif a.action=='audit':audit()
        elif a.action=='energy':energy_gate()
        elif a.action=='generate':generate(a.stage,a.shard,a.shards)
        elif a.action=='freeze':freeze_lock()
        elif a.action=='evaluate':
            from tools.e53_eval import evaluate
            evaluate(a.stage)
        else:
            assert a.review_note,'必须提供已收到的真人核验说明，不能由模型冒充'
            write(OUT/'audit/human_mapping_review.json',dict(approved=True,reviewer='user',note=a.review_note,utc=datetime.now(timezone.utc).isoformat(),scope='pilot mapping review images; Lock shares same frozen renderer/mapping rule'))
            decision(human_mapping_review='APPROVED_BY_USER')
    except Exception:
        from pathlib import Path
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
        write(OUT/'failures'/(a.action+'_'+stamp+'.json'),dict(action=vars(a),commit=commit(),traceback=traceback.format_exc(),classification='ENGINEERING_FAILURE_PENDING_REVIEW'))
        raise

if __name__=='__main__':main()
