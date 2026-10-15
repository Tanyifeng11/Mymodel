"""复用原 GPU 指标，增加机制对照；筛选集结果不称独立验证。"""
import numpy as np
from tools.e50_protocol import *


def evaluate():
    from tools import e48_eval as module
    from tools.e44_metrics import paired
    update=dict(OUT=OUT,ARMS=ARMS,REPAIRS=CANDIDATES,COMPONENT_ARMS=[],
        image_path=image_path,decision=decision,mechanism_summary=lambda:None)
    old={k:getattr(module,k) for k in update}
    try:
        for k,v in update.items():setattr(module,k,v)
        module.evaluate('probe',arms=ARMS)
        rows=read(OUT/'splits/probe.json')
        module.panels(rows,ARMS,'probe',OUT/'probe/panels/all')
    finally:
        for k,v in old.items():setattr(module,k,v)
    metrics=read(OUT/'probe/metrics.json')
    comparison={a:{k:paired(metrics,a,a+'_ENERGY',k) for k in module.KEYS} for a in CANDIDATES}
    write(OUT/'probe/mechanism_comparisons.json',comparison)
    mechanism={}
    for arm in ARMS[2:]:
        steps=[s for row in rows for s in read(image_path(arm,row['id']).with_name(row['id']+'_intervention.json'))['steps']]
        match=[s['matching'] for s in steps if 'matching' in s]
        mechanism[arm]=dict(independent_E5_exact_steps=sum(s['independent_epsilon_exact'] for s in steps),
            steps=len(steps),candidate_epsilon_energy=float(sum(s['candidate_epsilon_energy'] for s in steps)),
            original_epsilon_energy=float(sum(s['global_B1_epsilon_energy'] for s in steps)),
            coverage_mean=float(np.mean([s['coverage'] for s in match])) if match else None,
            coverage_zero_steps=sum(s['matched_cells']==0 for s in match) if match else None,
            value_delta_rms_mean=float(np.mean([s['local_value_delta_rms'] for s in match])) if match else None,
            energy_amplitude_range=[min(s['amplitude'] for s in steps),max(s['amplitude'] for s in steps)] if arm.endswith('_ENERGY') else None,
            energy_rounding_max=max(s.get('rounding_relative_error',0.) for s in steps))
    write(OUT/'probe/intervention_summary.json',mechanism)
    gates=read(OUT/'probe/gates.json');passed=[a for a in CANDIDATES if gates[a]['passed']]
    # 图板仍需本地逐张核对；局部匹配全无覆盖不能算有纹样迁移证据。
    decision(probe='complete',candidate_passed=passed,training='not_run',dev32='not_run',new96='not_run',
        final_decision='PROMISING_REQUIRES_DEV32' if passed else 'NO_GO',
        motif_identity_validated=False,visual_review='pending_local',
        next='download, independently recompute statistics, inspect all boards and worst10')


if __name__=='__main__':evaluate()
