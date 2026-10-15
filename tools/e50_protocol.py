"""e50 无训练、独立结构轨迹与对应 V 迁移；结果不反调配置。"""
from pathlib import Path
from tools.e49_protocol import copy_exact
from tools.e47_protocol import E5, E5_SHA, B1, B1_SHA, SIZE, LAYER, read, write, sha, commit

OUT = Path('output_eval/e50_structure_transport_20261015')
SOURCE = Path('output_eval/e48_b1_structure_20261014')
ARMS = ['B', 'D', 'STRUCT', 'STRUCT_ENERGY', 'MATCH', 'MATCH_ENERGY']
CANDIDATES = ['STRUCT', 'MATCH']
ATTENTIONS = [f'up_blocks.{b}.attentions.{a}.transformer_blocks.0.attn1'
              for b in [1, 2, 3] for a in range(3)]
FEATURE = 'up_blocks.0.resnets.1'
CONFIG = dict(experiment='e50', version=1, training_updates=0, new_parameters=0,
    base_sha256=E5_SHA, b1_sha256=B1_SHA, seed=42, steps=50, cfg=7., size=list(SIZE),
    scales=dict(sketch=.6, texture=1., ipa=1.), arms=ARMS,
    known_bad=['172600892','180633937','193488637','204446309','150553679','79583392','163690164'],
    cohort='same preselected e48 probe16; 7 known failures plus 9 fixed controls; development only',
    structure=dict(source='independent historical E5 same-seed full trajectory; exact epsilon replay each step',
        attention_layers=ATTENTIONS, attention='replace self affinity with independent E5 Q/K; retain student V and original Sketch term',
        attention_steps=[0,50], feature_layer=FEATURE, feature_steps=[0,30],
        feature='full E5 ResNet output after skip addition; no latent or final RGB blending'),
    matching=dict(layer=LAYER[:-10], source='same frozen B1 Writer and trained norm/to_v_ref, no new encoder',
        descriptors='L2 normalized target E5 attention input and normalized reference Writer spatial tokens',
        reference_tokens=256, target_tokens=768, metric='cosine argmax in both directions',
        cosine_min=.20, margin_min=.02, cycle_max_diagonal_fraction=.05,
        mask='existing Sketch 9px erosion plus8px feather; invalid OFF; no GT',
        formula='E5_affinity @ (student_V + gate*mask*confidence*(warped_reference_V-student_V))',
        fallback='zero-confidence location keeps student V; repeated patterns may defeat matching',
        strength='same frozen B1 scalar gate; not calibrated on probe RGB',
        interpretation='local matching adaptation, not original DreamMatcher semantic PCA or official reproduction'),
    energy_control='same current latent: E5 + global original B1 epsilon delta scaled to candidate epsilon delta L2; may amplify; log amplitude and norm error',
    mechanism='compare each candidate with its per-step epsilon-energy control; do not infer mechanism from safety alone',
    audit='8 disabled-intervention full RGB/50-step exact replays; live STRUCT and MATCH finite/isolation checks; all independent E5 steps exact',
    statistics=dict(unit='identity', bootstrap_draws=10000, seed=32042),
    safety=dict(f1_lower=-.020, iou_lower=-.010, leak_upper=.005, tail=-.15),
    appearance=dict(lpips_ci_upper=0., minimum_gain_retention=.50),
    interpretation='selected diagnostic cohort, not general failure rate; passing => promising_requires_dev32, no automatic new training or confirmation',
    references=dict(CtrlX='https://arxiv.org/html/2406.07540v1',
        DreamMatcher='https://arxiv.org/html/2402.09812v2'),
    limitation='SD1.5/E5 adaptation; original Ctrl-X SDXL and DreamMatcher full-subject correspondence differ')


def decision(**updates):
    path=OUT/'decision.json'
    value=read(path) if path.exists() else dict(experiment='e50',audit='not_run',probe='not_run',
        dev32='not_run',training='not_run',new96='not_run',unseen_generalization=False)
    value.update(updates);write(path,value);return value


def image_path(arm,identity,stage='probe'):
    return OUT/stage/'images'/arm/(identity+'.png')


def prepare():
    assert read(SOURCE/'decision.json')['final_decision']=='NO_GO'
    assert sha(E5)==E5_SHA and sha(B1)==B1_SHA
    path=OUT/'protocol.json'
    if path.exists():assert read(path)==CONFIG,'协议已冻结'
    else:write(path,CONFIG)
    for name in ['splits/probe.json','splits/dev32.json','manifests/input_hashes.json']:
        copy_exact(SOURCE/name,OUT/name)
    for name in ['protocol.json','decision.json','probe/gates.json','mechanism_summary.json']:
        copy_exact(SOURCE/name,OUT/'source_diagnosis'/name)
    rows=read(OUT/'splits/probe.json')
    hashes=read(OUT/'manifests/input_hashes.json')
    assert all(sha(p)==h for p,h in hashes.items())
    for arm in ['B','D']:
        for row in rows:
            src=SOURCE/'probe/images'/arm/(row['id']+'.png')
            for suffix in ['.png','.json']:copy_exact(src.with_suffix(suffix),image_path(arm,row['id']).with_suffix(suffix))
        write(OUT/'probe'/(arm+'_complete.json'),dict(n=16,reused_from=str(SOURCE),commit=commit()))
    prereg=OUT/'manifests/pre_registration.json'
    if not prereg.exists():
        from datetime import datetime,timezone
        write(prereg,dict(commit=commit(),utc=datetime.now(timezone.utc).isoformat(),
            protocol_sha256=sha(path),before_new_RGB=True))
    decision(prepared=True)
    return rows
