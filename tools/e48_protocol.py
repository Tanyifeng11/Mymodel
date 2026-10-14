"""e48 结构机制诊断与有限修复预算；评测结果不反向修改协议。"""
import random
import shutil
from pathlib import Path
from tools.e47_protocol import E5,E5_SHA,B1,B1_SHA,SAFE_SCALE,SIZE,LAYER,read,write,sha,commit

OUT=Path('output_eval/e48_b1_structure_20261014')
PARENT=Path('output_eval/e47_tcpm_b1_factorial_20261013')
BAD=['172600892','180633937','193488637','204446309','150553679','79583392','163690164']
ARMS=['B','D','SAFE','EARLY','MIDDLE','LATE','EPS_LOCK','ENERGY_EPS','LATENT_LOCK','ENERGY_LATE']
REPAIRS=['EARLY','MIDDLE','LATE','EPS_LOCK','LATENT_LOCK']
SELECTED_STEPS=[0,5,14,15,24,34,35,49]
CONFIG=dict(experiment='e48',version=1,seed=42,steps=50,cfg=7.,size=list(SIZE),
    scales=dict(sketch=.6,texture=1.,ipa=1.),base='E5',checkpoint_sha=dict(E5=E5_SHA,B1=B1_SHA),
    known_bad=BAD,probe_n=16,probe_controls='9 valid dev32 identities, deterministic random seed48014',
    arms=ARMS,windows=dict(EARLY=[0,15],MIDDLE=[15,35],LATE=[35,50]),
    safe_lambda=SAFE_SCALE,epsilon_protection='teacher same latent; original 9px erosion and 8px feather mask',
    latent_protection='independent E5 same seed trajectory; mix outside original soft mask each DDIM step',
    energy_control='epsilon energy matched per same-state; late-window energy calibrated on fixed train8',
    decomposition='per-head joint=(1-p_ref)*self+p_ref*ref; exact q/k/v audit at selected steps',
    decomposition_steps=SELECTED_STEPS,mask_invalid='retain; all new branches OFF; regional metrics NA',
    metrics='same e47 backend plus boundary distance, area, threshold sensitivity; no GT at inference',
    bootstrap=dict(draws=10000,seed=32042,unit='garment identity'),
    safety=dict(f1_lower=-.020,iou_lower=-.010,leak_upper=.005,tail=-.15),
    appearance=dict(lpips_ci_upper=0.,minimum_gain_retention=.50),
    training=dict(max_runs=2,updates=800,seed=42,microbatch=1,accumulation=8,lr=1e-4,
        weight_decay=.01,keep=.1,boundary_keep=.1,initialization='fresh zero gate, no warm start',
        options=['independent reference attention','B1 with boundary-weighted epsilon distillation']),
    validation='new96 only after candidate freeze and identity/exact/near-duplicate audit; reused confirm96 is development',
    references=dict(IP_Adapter='https://arxiv.org/abs/2308.06721',
        Blended_Diffusion='https://arxiv.org/abs/2111.14818',StyleID='https://arxiv.org/abs/2312.09008'),
    scope='diagnostic adaptations, not exact paper reproduction; no full backbone retraining')


def decision(**changes):
    path=OUT/'decision.json'
    value=read(path) if path.exists() else dict(experiment='e48',audit='not_run',probe='not_run',
        dev32='not_run',training='not_run',new96='not_run',unseen_generalization=False)
    value.update(changes);write(path,value);return value


def history(arm,identity):
    folder='p2' if any(r['id']==identity for r in read(PARENT/'splits/dev32.json')) else 'p3'
    source='D_safe' if arm=='SAFE' else arm
    return PARENT/folder/'images'/source/(identity+'.png')


def image_path(arm,identity,stage='probe'):
    return OUT/stage/'images'/arm/(identity+'.png')


def prepare():
    assert read(PARENT/'decision.json')['P0']=='pass'
    assert sha(E5)==E5_SHA and sha(B1)==B1_SHA
    path=OUT/'protocol.json'
    if path.exists():assert read(path)==CONFIG
    else:write(path,CONFIG)
    dev=read(PARENT/'splits/dev32.json');confirm=read(PARENT/'splits/confirm96.json')
    lookup={r['id']:r for r in dev+confirm}
    valid={r['id'] for r in read(PARENT/'p2/metrics.json') if r['variant']=='B' and r['mask_valid']}
    controls=sorted([r for r in dev if r['id'] in valid and r['id'] not in BAD],key=lambda r:r['id'])
    random.Random(48014).shuffle(controls)
    probe=[lookup[i] for i in BAD]+controls[:9]
    assert len(probe)==16 and len({r['id'] for r in probe})==16
    calibration_ids=['100063292','100084474','100166343','100170818','100264769','100293865','100298252','100328348']
    pool=read('output_eval/e44_mimicbrush_drra_20261012/splits/train4096.json')
    calibration=[next(r for r in pool if r['id']==i) for i in calibration_ids]
    for name,rows in [('probe',probe),('dev32',dev),('calibration',calibration)]:
        target=OUT/'splits'/(name+'.json')
        if target.exists():assert read(target)==rows
        else:write(target,rows)
    for name in ['train1024.json','train1024_records.json']:
        target=OUT/'splits'/name;source=Path('output_eval/e44_mimicbrush_drra_20261012/splits')/name
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert sha(source)==sha(target)
        else:shutil.copy2(source,target)
    write(OUT/'manifests/e5_training_meta.json',read(PARENT/'manifests/e5_training_meta.json'))
    paths={r[k] for r in probe+dev+calibration for k in ['gt','sketch','reference']}
    hashes={p:sha(p) for p in sorted(paths)};hashes.update({str(E5):E5_SHA,str(B1):B1_SHA})
    frozen=OUT/'manifests/input_hashes.json'
    if frozen.exists():assert read(frozen)==hashes
    else:write(frozen,hashes)
    return probe
