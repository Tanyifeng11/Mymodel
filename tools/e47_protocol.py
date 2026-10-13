"""e47 固定析因协议；历史产物只读，未执行阶段使用 not_run。"""
from pathlib import Path
from tools.e44_protocol import E5, E5_SHA, SIZE, LAYER, read, write, sha, commit

OUT = Path('output_eval/e47_tcpm_b1_factorial_20261013')
OLD = Path('output_eval/e44_mimicbrush_drra_20261012')
E46 = Path('output_eval/e46_sarg_20261013')
E2B = Path('output/phase1_e2b_color_safe_gate_e3/checkpoint-final/joint_model.pt')
B1 = OLD/'p1/B1/checkpoints/step0800.pt'
B1_SHA = 'd107dc331c5d8b15079040dc351ec737b0644f9af745834379876e3e524ab84e'
SAFE_SCALE = 0.5595459363700961
PLAN_SHA = 'fcb170a76a1e8af0aaf77d7ff531c39666608650028451e2b16f199627114373'
CONFIG = dict(experiment='e47', version=1, source_document_title='E48',
    source_plan_sha256=PLAN_SHA,
    arms=dict(A='audited No-TCPM base', B='E5', C='A plus newly trained B1', D='historical E5 plus B1'),
    auxiliary=dict(D_safe=SAFE_SCALE, main_interaction=False),
    size=list(SIZE), steps=50, scheduler='DDIM', cfg=7., seed=42,
    scales=dict(sketch=.6, texture=1., ipa=1.), layer=LAYER,
    training=dict(seed=42, identities=1024, updates=800, microbatch=1,
        accumulation=8, lr=1e-4, weight_decay=.01, grad_clip=1., keep_weight=.1,
        teacher='same frozen A with newly added B1 OFF', checkpoint='step800 only'),
    bootstrap=dict(unit='garment identity', draws=10000, seed=32042),
    safety=dict(f1_lower=-.020, iou_lower=-.010, leak_upper=.005, tail_f1=-.15),
    promotion='P0 pass, useful P2 appearance and acceptable structure, then frozen confirmation; no dev tuning',
    strict_unseen='requires training identity/file/source/near-duplicate audit',
    reference_metrics='CLIP-I Reference and LAB are proxies, not motif identity/direction proof')


def decision(**values):
    path = OUT/'decision.json'
    state = read(path) if path.exists() else dict(experiment='e47', P0='not_run',
        P1='not_run', P2='not_run', P3='not_run', causal_claim=False,
        paper_claim_supported=False)
    state.update(values)
    write(path, state)
    return state


def freeze_protocol():
    path = OUT/'protocol.json'
    if path.exists():
        assert read(path) == CONFIG, '禁止按结果调整协议'
    else:
        write(path, CONFIG)
    return CONFIG
