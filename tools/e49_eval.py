"""复用已验证的 GPU 指标与冻结门槛；单独写 e49 结果，不改 e48。"""
from tools.e49_protocol import *


def evaluate(stage, arms=ARMS):
    from tools import e48_eval as module
    # 旧评测函数的全局绑定必须一并替换，特别是 image_path 的原协议闭包。
    updates=dict(OUT=OUT,ARMS=arms,REPAIRS=TRAIN_ARMS,COMPONENT_ARMS=[],
        image_path=image_path,decision=decision,mechanism_summary=lambda:None)
    saved={k:getattr(module,k) for k in updates}
    try:
        for k,v in updates.items(): setattr(module,k,v)
        module.evaluate(stage,arms=arms)
        rows=read(OUT/'splits'/(stage+'.json'))
        module.panels(rows,arms,stage,OUT/stage/'panels/encoder')
        if all(a in arms for a in TRAIN_ARMS):
            records=read(OUT/stage/'metrics.json')
            from tools.e44_metrics import paired
            write(OUT/stage/'encoder_comparison.json',{k:paired(records,'SSVM','CLIP',k) for k in module.KEYS})
    finally:
        for k,v in saved.items(): setattr(module,k,v)


def evaluate_all():
    completed={a:read(OUT/'training'/a/'complete.json') for a in TRAIN_ARMS}
    assert all(v['updates']==800 for v in completed.values())
    assert len({v['parameters'] for v in completed.values()})==1
    assert len({v['sequence_sha256'] for v in completed.values()})==1
    decision(training='complete',training_updates={a:800 for a in TRAIN_ARMS})
    for stage in ['probe','dev32']: evaluate(stage)
    possible=[a for a in TRAIN_ARMS if all(read(OUT/s/'gates.json')[a]['passed'] for s in ['probe','dev32'])]
    decision(candidate_passed=possible,new96='not_run' if not possible else 'pending_freeze',
        final_decision='NO_GO' if not possible else 'pending_confirmation',
        next='local archive verification and report' if not possible else 'audit history and freeze one candidate before new96')


if __name__=='__main__': evaluate_all()
