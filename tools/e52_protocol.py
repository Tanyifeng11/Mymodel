"""E52 固定契约；任何确认集结果都不能反向修改配置。"""
import hashlib
from pathlib import Path
from tools.e47_protocol import E5, E5_SHA, B1, B1_SHA, SIZE, LAYER, read, write, sha, commit

OUT = Path('output_eval/e52_b1_rca_20261015')
SOURCE = Path('output_eval/e48_b1_structure_20261014')
PLAN_SHA = 'de3bf200202a6b7814dbe4c332bf602b634441664b62eb69ddc7d1aeb5c135e7'
STEPS = [0, 4, 9, 14, 19, 29, 39, 49]
A_ARMS = ['E5', 'G0T0', 'G1T0', 'G0T1', 'G1T1', 'SHAM']
B_ARMS = ['ORACLE', 'WRONG', 'RANDOM']
C_ARMS = ['G1_EARLY', 'G0_EARLY', 'SHRINK_EARLY']
CONFIG = dict(experiment='E52_B1_RCA', version=1, plan_sha256=PLAN_SHA,
    training_updates=0, extra_parameters=0, seed=42, steps=50, cfg=7., size=list(SIZE),
    scales=dict(sketch=.6, texture=1., ipa=1.), e5_sha256=E5_SHA, b1_sha256=B1_SHA,
    layer=LAYER, conditional_only=True, base_reference='fixed per identity',
    writer_reference='independent experimental input', pilot_n=24, locked_n=64, new_n=96,
    selected_steps=STEPS, early=[0,15], late_monitor=[39,49],
    alpha=.5, shrink=.5595459363700961, gain_max=1.,
    provenance=dict(min_valid_fraction=.70, tolerance_cells=2., min_local_mass=.15,
        timesteps=[999,499,19], perturbation_grid=4, valid_token_foreground_fraction=.50),
    bootstrap=dict(repeats=10000,seed=32042,unit='identity',family='Holm HG HC HD'),
    geometry_fix='G0T0 vs G1T0; frozen before outcomes; no pilot best-arm selection',
    natural_arms=['E5','B1','B1_EARLY','SHRINK_EARLY'],
    natural_selection_seed='E52_B1_RCA_20261015',
    mask_invalid='retain image, branch OFF, regional metrics NA',
    human_review='not_run until independent human reviewers actually participate',
    model_thresholds=dict(f1_lower=-.020,iou_lower=-.010,leak_upper=.005,tail=-.15,
        lpips_upper=0.,retention=.50),
    synthetic_model_go='NOT_APPLICABLE: no natural GT',
    strict_unseen=False)

def freeze(path,value):
    path=Path(path)
    if path.exists():assert read(path)==value, ('冻结文件发生变化',str(path))
    else:write(path,value)
    return value

def decision(**changes):
    path=OUT/'decision.json'
    value=read(path) if path.exists() else dict(experiment='E52',P0='not_run',P1='not_run',
        P2='not_run',P3='not_run',P4='not_run',P5='not_run',H_G='NOT_RUN',H_C='NOT_RUN',
        H_D='NOT_RUN',MODEL='NOT_EVALUATED',training_updates=0)
    value.update(changes);write(path,value);return value

def prepare():
    freeze(OUT/'manifest/protocol.json',CONFIG)
    assert sha(E5)==E5_SHA and sha(B1)==B1_SHA,'原始权重校验失败'
    sources={name:SOURCE/'splits'/(name+'.json') for name in ['dev32','probe','calibration']}
    for name,path in sources.items():freeze(OUT/'splits'/(name+'.json'),read(path))
    hashes={str(E5):E5_SHA,str(B1):B1_SHA}
    for path in sources.values():hashes[str(path)]=sha(path)
    for rows in [read(p) for p in sources.values()]:
        for row in rows:
            for key in ['gt','sketch','reference']:hashes[row[key]]=sha(row[key])
    freeze(OUT/'manifest/input_sha256.json',hashes)
    model_files=read('output_eval/e44_mimicbrush_drra_20261012/manifests/model_file_hashes.json')
    # 旧清单不只包含 Writer；逐文件核验后保存本轮副本。
    assert all(sha(p)==h for p,h in model_files.items()),'冻结模型文件发生变化'
    freeze(OUT/'manifest/model_file_sha256.json',model_files)
    write(OUT/'manifest/code.json',dict(commit=commit(),source_commit='b57b804',
        files={str(p):sha(p) for pattern in ['tools/e52*.py','models/e52*.py','submit/e52*.sh'] for p in Path('.').glob(pattern)}))
    decision();print('E52 PREPARE PASS',commit(),flush=True)

def image_path(stage,arm,identity):return OUT/'images'/stage/arm/(identity+'.png')

def order(identity):return hashlib.sha256((CONFIG['natural_selection_seed']+identity).encode()).hexdigest()
