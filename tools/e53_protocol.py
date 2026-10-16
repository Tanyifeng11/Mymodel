"""E53 可执行契约：先数值/测量门禁，再人工映射核验，最后冻结确认。"""
import hashlib
from pathlib import Path
from tools.e47_protocol import E5,E5_SHA,B1,B1_SHA,SIZE,LAYER,read,write,sha,commit

OUT=Path('output_eval/e53_b1_oracle_20261016')
HISTORY=Path('output_eval/e52_b1_rca_20261015')
KEY_STEPS=[0,4,8,12,16,24,32,40,48]
WINDOWS=dict(ALL=list(range(50)),EARLY=list(range(15)),MIDDLE=list(range(15,35)),LATE=list(range(35,50)))
ARMS={
    'E5':dict(geometry='N',texture=0,routing='OFF',window='ALL'),
    'ORIGINAL':dict(geometry='N',texture=0,routing='ORIGINAL',window='ALL'),
    'SUPPORT':dict(geometry='N',texture=0,routing='SUPPORT',window='ALL'),
    'ORACLE':dict(geometry='N',texture=0,routing='ORACLE',window='ALL'),
    'SHIFT':dict(geometry='N',texture=0,routing='SHIFT',window='ALL'),
    'FLIP':dict(geometry='N',texture=0,routing='FLIP',window='ALL'),
    'RANDOM':dict(geometry='N',texture=0,routing='RANDOM',window='ALL'),
    'G0':dict(geometry='0',texture=0,routing='ORACLE',window='ALL'),
    'G1':dict(geometry='1',texture=0,routing='ORACLE',window='ALL'),
    'G1_V0':dict(geometry='1',texture=0,routing='RESTORE_V0',window='ALL'),
    'WRONG_TEXTURE':dict(geometry='N',texture=1,routing='ORACLE',window='ALL'),
    'G0_ORIGINAL':dict(geometry='0',texture=0,routing='ORIGINAL',window='ALL'),
    'V_SWAP':dict(geometry='0',texture=0,routing='V_SWAP',window='ALL'),
    'K_SWAP':dict(geometry='0',texture=0,routing='K_SWAP',window='ALL'),
    'EARLY':dict(geometry='N',texture=0,routing='ORACLE',window='EARLY'),
    'MIDDLE':dict(geometry='N',texture=0,routing='ORACLE',window='MIDDLE'),
    'LATE':dict(geometry='N',texture=0,routing='ORACLE',window='LATE'),
}
for i in KEY_STEPS:ARMS['STEP%02d'%i]=dict(geometry='N',texture=0,routing='ORACLE',window=[i])
PILOT_ARMS=list(ARMS)
LOCK_ARMS=[a for a in ARMS if a not in ['FLIP','RANDOM','MIDDLE'] and not a.startswith('STEP')]
CONFIG=dict(experiment='E53_B1_ORACLE',version=1,source_plan_sha256='c2c674b4b872b35e6653c66c01978bc0ab88b3a206c7357f87b06a772df9722b',training_updates=0,new_parameters=0,
    e5_sha256=E5_SHA,b1_sha256=B1_SHA,seed=42,steps=50,cfg=7.,size=list(SIZE),
    scales=dict(sketch=.6,texture=1.,ipa=1.),layer=LAYER,target_hw=[32,24],writer_hw=[16,16],
    sigma_uv=.03,sigma_sensitivity=[.02,.03,.05],wrong_shift=.25,token_foreground_min=.5,part_purity_min=.75,
    mapping_semantics='renderer geometry UV correspondence at feature-grid locations; semantic purity of mixed Writer features is not assumed',
    neutral_geometry='fixed within garment category; shirt and trousers retain consistent part charts',
    reference_mass='fixed from geometry-N texture0 original at the same current query; K/V swaps use G0 mass',
    energy=dict(tolerance=1e-4,measure='post-projection applied delta after model-dtype cast',
        baseline='per-identity ORIGINAL GN full free trajectory recorded before interventions',
        content='fixed baseline RMS per step',time='same baseline total sum RMS^2; baseline step-energy weights restricted and renormalized to active window',
        unconditional='B1 disabled, energy exactly zero',amplification='allowed as specified in E53; all gains recorded; single-step concentration does not represent normal B1 operation'),
    statistics=dict(unit='identity',bootstrap_samples=10000,seed=32042,primary_family_size=3,
        G='reference-following boundary rho G1-G0',C='ORACLE-ORIGINAL F1 with support-only, texture-rescue and wrong-map conjunctive checks',D='EARLY-LATE F1 at equal actual total energy'),
    practical=dict(f1=.02,texture_accuracy=.05,f1_safety_lower=-.02,iou_safety_lower=-.01,leak_upper=.005),
    texture=dict(library_n=8,space='known target UV, fixed support, normalized highpass NCC',blank_is_failure=True),
    pilot_n=24,lock_n=64,pilot_arms=PILOT_ARMS,lock_arms=LOCK_ARMS,
    human_mapping_review='REQUIRED before pilot; not substituted by model QC',natural='only after synthetic mechanism support',
    precision='original fused B1 preserved; FP32 reference correction; energy match in FP32 then audit cast')

def freeze(path,value):
    path=Path(path)
    if path.exists():assert read(path)==value,('FROZEN_FILE_CHANGED',str(path))
    else:write(path,value)
    return value

def decision(**changes):
    p=OUT/'decision.json';v=read(p) if p.exists() else dict(experiment='E53',P0='NOT_RUN',energy='NOT_RUN',mapping='NOT_RUN',pilot='NOT_RUN',lock='NOT_RUN',H_G='NOT_RUN',H_C='NOT_RUN',H_D='NOT_RUN',human_mapping_review='NOT_RUN',training_updates=0)
    v.update(changes);write(p,v);return v

def prepare():
    freeze(OUT/'configs/protocol.json',CONFIG)
    assert sha(E5)==E5_SHA and sha(B1)==B1_SHA
    files=read(HISTORY/'manifest/model_file_sha256.json')
    assert all(sha(p)==h for p,h in files.items()),'MODEL_FILE_CHANGED'
    freeze(OUT/'manifest/model_file_sha256.json',files)
    write(OUT/'manifest/code.json',dict(commit=commit(),files={str(p):sha(p) for pattern in ['tools/e53*.py','models/e53*.py','submit/e53*.sh'] for p in Path('.').glob(pattern)}))
    decision()

def image_path(stage,arm,identity):return OUT/'images'/stage/arm/(identity+'.png')
def identity_seed(identity):return int(hashlib.sha256(('E53_20261016_'+identity).encode()).hexdigest()[:8],16)
