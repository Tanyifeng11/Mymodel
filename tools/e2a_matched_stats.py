"""开发集两项主结构检验；结果只通向开发审查，不自动揭开 LOCK。"""
from tools.e2a_matched_common import *
from tools import e2a_sup_stats as backend
from tools.e2a_attr_common import DIRECTION
from tools.e2a_attr_stats import holm


def summarize():
    backend.ROOT=ROOT;backend.CONFIG=CONFIG
    result=backend.compare('C','T3M')
    holm([(key,result[key]) for key in ['iou','leak']])
    keys=backend.KEYS
    summaries={arm:{key:backend.effect([r[key] for r in backend.records(arm).values() if r.get(key) is not None])
        for key in keys} for arm in ['B','C','T3','T3M']}
    meaningful=any(DIRECTION[k]*result[k]['mean']>=CONFIG['meaningful_gain'][k] for k in ['iou','leak'])
    safe=all(not result[k]['noninferiority']['clearly_failed'] for k in ['iou','leak','edge_f1','edge_l1'])
    write(ROOT/'statistics/P2b/decision.json',dict(comparison=result,summaries=summaries,
        primary_family=['C minus T3M IoU','C minus T3M Leak'],two_sided_Holm=True,
        meaningful_structure_signal=meaningful,no_clear_structural_degradation=safe,
        structural_development_gate=meaningful and safe,texture_review_required=True,
        gamma_unchanged=True,LOCK_opened=False,seed43_started=False,
        stage='P2B_DEVELOPMENT_REVIEW_REQUIRED',P2b_is_development_only=True))
    status(stage='P2B_DEVELOPMENT_REVIEW_REQUIRED',complete=False,engineering_failure=None)
    print('DEVELOPMENT_STRUCTURE_GATE',meaningful and safe,flush=True)


if __name__=='__main__':summarize()
