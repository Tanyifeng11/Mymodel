"""冻结 GPU 评价与直接时间消融比较；不把开发结果称为独立确认。"""
import numpy as np
from tools.e51_protocol import *


def evaluate(stage):
    from tools import e48_eval as module
    from tools.e44_metrics import paired
    updates=dict(OUT=OUT,ARMS=ARMS,REPAIRS=TRAIN_ARMS,COMPONENT_ARMS=[],
        image_path=image_path,decision=decision,mechanism_summary=lambda:None)
    old={k:getattr(module,k) for k in updates}
    try:
        for k,v in updates.items():setattr(module,k,v)
        module.evaluate(stage,arms=ARMS)
        rows=read(OUT/'splits'/(stage+'.json'));module.panels(rows,ARMS,stage,OUT/stage/'panels/controller')
        metrics=read(OUT/stage/'metrics.json')
        write(OUT/stage/'time_comparison.json',{k:paired(metrics,'TIME','NOTIME',k) for k in module.KEYS})
        summaries={}
        for arm in TRAIN_ARMS:
            cases={row['id']:read(image_path(arm,row['id'],stage).with_name(row['id']+'_control.json'))['steps'] for row in rows}
            steps=[s for value in cases.values() for s in value if not s.get('branch_off')]
            summaries[arm]=dict(valid_identities=sum(any(not s.get('branch_off') for s in value) for value in cases.values()),
                steps=len(steps),by_timestep_band={name:{key:float(np.mean([s[key] for s in steps if lo<=s['timestep']<hi]))
                    for key in ['gate_mean','raw_b1_rms','applied_rms','cap','epsilon_rms','epsilon_boundary_rms','epsilon_outside_rms']}
                    for name,lo,hi in [('early_high_noise',700,1000),('middle',300,700),('late_low_noise',0,300)]},
                cap_active_steps=sum(s['cap']<1 for s in steps),
                all_zero_injection_steps=sum(s['applied_rms']==0 for s in steps),
                per_identity={i:{key:float(np.mean([s[key] for s in value if not s.get('branch_off')]))
                    for key in ['gate_mean','applied_rms','epsilon_rms']} for i,value in cases.items() if any(not s.get('branch_off') for s in value)})
        write(OUT/stage/'controller_summary.json',summaries)
    finally:
        for k,v in old.items():setattr(module,k,v)


def evaluate_all():
    completed={a:read(OUT/'training'/a/'complete.json') for a in TRAIN_ARMS}
    assert all(v['updates']==800 for v in completed.values())
    assert len({v['parameters'] for v in completed.values()})==1
    assert len({v['sequence_sha256'] for v in completed.values()})==1
    decision(training='complete',training_updates={a:800 for a in TRAIN_ARMS})
    for stage in ['probe','dev32']:evaluate(stage)
    passed=[a for a in TRAIN_ARMS if all(read(OUT/s/'gates.json')[a]['passed'] for s in ['probe','dev32'])]
    decision(candidate_passed=passed,new96='not_run',
        final_decision='PROMISING_REQUIRES_CONFIRMATION' if passed else 'NO_GO',
        next='download and independently verify; inspect all boards, worst10, motif identity and gating maps',
        motif_identity_validated=False,visual_review='pending_local')


if __name__=='__main__':evaluate_all()
