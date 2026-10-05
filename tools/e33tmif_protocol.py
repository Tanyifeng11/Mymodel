"""E33-TM-IF 预先冻结口径；读取旧实验，不修改其产物。"""
from pathlib import Path
from tools.e33tm_protocol import read, write, sha, commit, order, E5, RF, DATASET, SEEDS
from tools.e33tm_protocol import OUT as TM

OUT = Path('output_eval/e33_tm_if_interface_localization_20261006')
ARMS = ('R0', 'R90', 'R180')
STRENGTHS = (.05, .15, .30, .50, .70)
CONDITIONS = {'C0': (False, False, False), 'C1': (True, False, False),
    'C2': (False, True, False), 'C3': (False, False, True),
    'C4': (True, True, False), 'C5': (True, False, True),
    'C6': (False, True, True), 'C7': (True, True, True)}
PROTOCOL = dict(experiment='E33-TM-IF', training_steps=0, architecture_changes=0,
    plan_sha256='380e1697ebae904e039d07d02d3862135b35b64cc3f7750475e511a92d1efa3d',
    old_experiment=str(TM), rf_seeds=SEEDS, diffusion_seed=42,
    field_reproduction='all original dev256; original readable-GT denominator217 retained',
    image_reproduction='fresh full generation for all primary254 and all3 RF seeds; no cached image reuse',
    reproduction_tolerance=.005, stage_fixed_denominator=254,
    readable_threshold=.25, success_error_degrees=15, bootstrap_draws=2000, bootstrap_seed=32042,
    stage0='same GT support and weights as E33-TM; RF confidence auxiliary, not a new field gate',
    stage1='raw cv2.remap before confidence fallback and garment masking; evaluate on fixed original GT support',
    stage2='exact old carrier: remap interior + appearance fallback boundary + original garment mask',
    stage3='VAE posterior mean scaled latent; no RGB orientation score',
    stage5='noisy start latent; no formal RGB orientation score',
    predicted_x0='scheduler.step output pred_original_sample; one scheduler step only; never noisy latent decode',
    cfg=7., negative_prompt=' worst quality, low quality', sketch_scale=.6, texture_scale=1.,
    base_steps=50, current_strength=.15, conditions=CONDITIONS, strengths=STRENGTHS,
    condition_controls='same precomputed full RF carrier/mask/latent/noise; disable refinement branches only; common CFG/negative prompt',
    split='SHA256 E33TMIF/split/id ordering: first64 diagnostic,next64 confirmation,last126 remaining',
    visual='field-success identities in diagnostic64, fixed E33TMIF/visual hash first16; not final-output selection',
    adjacent_drop=.20, competition_drop=.15,
    candidate=dict(r90_gain=.15,contour_drop=.05,relative_text_drop=.05,
                   tie_break='highest R90, then highest TextScore, then lowest strength'),
    purity=dict(scope='diagnostic64, only if field->RGB Gate passes', improvement=.15,
        P1='warp source-valid mask (nonwhite RGB<245 in any channel AND source E26 confidence>=.25); invalid garment pixels neutral220',
        P2='grayscale subtract Gaussian sigma8 low-frequency, fixed multiplier2, center128, clip32..224; outside garment255',
        P3='RF axial field -> local oriented cosine, wavelength16px, phase0, grayscale128+80*cos; outside garment255'),
    readability_attribution='paired successful->failure transitions: count newly-unreadable vs still-readable incorrect; majority newly-unreadable => readability_loss; evaluator limits reported separately',
    routing='confirmed operating point first; else largest significant adjacent stage drop; condition/purity interactions reported alongside stage',
    confirmation='no heldout grid; selected strength and .15 baseline once on confirmation64; RF43/44 only selected stage pair on same64 if denoise localization',
    user_confirmation='开始实行')

def prepare():
    cohort=read(TM/'manifests/cohorts.json')
    assert len(cohort['primary'])==254
    ranked=order(cohort['primary'],'E33TMIF/split')
    splits={name:[r['id'] for r in rows] for name,rows in
        [('diagnostic64',ranked[:64]),('confirmation64',ranked[64:128]),('remaining126',ranked[128:])]}
    field={r['id']:r for r in read(TM/'field_precheck/seed42/real/rows.json')}
    visual=order([r for r in ranked[:64] if field[r['id']]['rot90_response_success'] is True], 'E33TMIF/visual')[:16]
    assert len(visual)==16
    splits['field_success_fixed_hash16']=[r['id'] for r in visual]
    for name,ids in splits.items():
        path=OUT/'splits'/(name+'.json')
        if path.exists(): assert read(path)==ids
        else: write(path,ids)
    path=OUT/'protocol/protocol.json'
    if path.exists(): assert read(path)==read_normalized(PROTOCOL)
    else: write(path,PROTOCOL)
    locked=[E5,TM/'manifests/cohorts.json',TM/'weight_audit.json',TM/'completion_check.json',
        Path('data/train_bf_texture.json')]
    locked += [RF/('seed%d'%s)/'RF2/checkpoint_final.pt' for s in SEEDS]
    old=read(TM/'frozen_check.json')
    locked += [Path(p) for p in old['before']]
    locked += [Path(p) for p in ('models/e33tm_generation_wrapper.py','tools/e33tm_metrics.py',
        'models/pattern_coordinate_field.py','tools/e25_spatial_diagnosis.py')]
    sources={str(p):sha(p) for p in locked}
    freeze=OUT/'frozen_check.json'
    if freeze.exists(): assert read(freeze)['before']==sources
    else: write(freeze,dict(before=sources,git_commit=commit(),training_steps=0))
    manifest=OUT/'protocol/source_manifest.json'
    if not manifest.exists():
        write(manifest,dict(cohort_sha256=sha(TM/'manifests/cohorts.json'),
              splits_sha256={n:sha(OUT/'splits'/(n+'.json')) for n in splits},git_commit=commit()))
    return cohort,splits

def read_normalized(value):
    import json
    return json.loads(json.dumps(value))

def freeze_check():
    value=read(OUT/'frozen_check.json')
    value['after']={p:sha(p) for p in value['before']}
    value['pass']=value['before']==value['after']
    assert value['pass'], '冻结输入或原实现变动'
    write(OUT/'frozen_check.json',value)
    return value

if __name__=='__main__': prepare()
