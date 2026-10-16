"""M54冻结协议：本地确认的五项修订优先于原文冲突。"""
from pathlib import Path
from tools.e47_protocol import E5,E5_SHA,B1,B1_SHA,SIZE,LAYER,SAFE_SCALE,read,write,sha,commit

OUT=Path('output_eval/m54_b1_screen_20261017')
HISTORY=Path('output_eval/e48_b1_structure_20261014')
TRAIN_ARMS=['M1','M11','M9','M4','M3','M5','M2','M6']
LAYER_ARMS={f'M7_A{i}':f'up_blocks.2.attentions.{i}.transformer_blocks.0.attn1.processor' for i in range(3)}
ZERO_ARMS=list(LAYER_ARMS)+['M8_T100','M8_T085','M8_T070','M10','HF_ONLY']
CONFIG=dict(experiment='M54',version=2,source_plan_sha256='8fdb4b2e4f847694c09dfc02ecce3f65b9cf66cd708dfc098568e3428dbf12cd',
    user_authorization='开始实施；确认前一轮展示的五项修订',e5_sha256=E5_SHA,b1_sha256=B1_SHA,
    size=list(SIZE),steps=50,seed=42,cfg=7.,scales=dict(sketch=.6,texture=1.,ipa=1.),layer=LAYER,
    target_hw=[32,24],reference_hw=[16,16],conditional_only=True,
    training=dict(identities=1024,resolution=list(SIZE),updates=800,deepening_total_updates=2400,
        microbatch=1,accumulation=8,lr=1e-4,weight_decay=.01,grad_clip=1.,seed=42,
        outside_keep=.1,ATV_M1=.001,boundary_keep_M9=.1,reference_dropout_M5=.10,
        teacher='same-state frozen E5 for all candidates; M9 initial identity baseline is B1',checkpoint_selection='fixed final; no dev selection'),
    revisions=dict(M9='only new structure branch trainable; frozen B1 exception; zero-conv output, alpha initial1; initial=B1',
        M10='fixed nonzero statistics residual through frozen original output projection and frozen B1 gain; no zero learned layer',
        M1='train target LayerNorm and Q/K/V, zero output; ATV .001',
        M2='frozen target Q; train image-only reference norm/K/V and zero output; original joint branch OFF',
        resolution='1024 training identities, not1024 pixels; historical384x512 retained'),
    reference_off='new reference enhancement OFF equals E5; M9 structure-only ablation separately reported; all new enhancement OFF equals E5',
    layer_search='all three same640 same32x24 up_blocks.2 attn1; incompatible-width layers excluded, no untrained cross-width projection',
    zero_arms=ZERO_ARMS,train_arms=TRAIN_ARMS,M8_temperatures=[1.,.85,.70],
    M6=dict(tau=.07,sigma_spatial=.15,prior='normalized Sketch/reference foreground bounding boxes; no GT'),
    M5=dict(fusion_width=160,heads=4,scales=[.75,1.,1.25]),M11=dict(pool_kernel=5,pool_stride=2),
    statistics=dict(unit='identity',bootstrap_samples=10000,seed=32042),
    safety=dict(f1_lower=-.020,iou_lower=-.010,leak_upper=.005,tail_f1=-.15,lpips_upper=0.,gain_retention=.5),
    ranking=['tail','F1 safety','IoU safety','LPIPS','parameters','latency','VRAM'],
    pilot_n=24,dev_n=32,lock_n=64,natural_n=96,top_n=4,combinations_max=4,natural_top_n=2,
    final_inference_seeds=[42,123,456],
    stop='engineering failures preserved and fixed; no threshold/sample tuning on Lock/Natural',
    fallback='only after all lightweight routes fail; CatVTON and explicit warping require separate concrete mechanism implementation, never silently count as run',
    literature_adaptation=True,independent_human_review='NOT_RUN unless actual independent reviewers participate')

PAPERS={
 'M1':'https://arxiv.org/abs/2312.01725','M2':'https://arxiv.org/abs/2308.06721',
 'M3':'https://arxiv.org/abs/2403.05139','M4':'https://arxiv.org/abs/2409.08258',
 'M5':'https://ojs.aaai.org/index.php/AAAI/article/view/32973','M6':'https://arxiv.org/abs/2402.09812',
 'M7':'https://arxiv.org/abs/2404.02733','M8':'https://arxiv.org/abs/2311.03335',
 'M9':'https://arxiv.org/abs/2302.08453','M10':None,'M11':None}

def freeze(path,value):
    path=Path(path)
    if path.exists():assert read(path)==value,('FROZEN_CHANGED',str(path))
    else:write(path,value)
    return value

def decision(**changes):
    path=OUT/'decision.json'
    result=read(path) if path.exists() else dict(experiment='M54',R0='NOT_RUN',R1='NOT_RUN',R2='NOT_RUN',
        R3='NOT_RUN',dev32='NOT_RUN',combinations='NOT_RUN',lock='NOT_RUN',natural='NOT_RUN',seeds='NOT_RUN')
    result.update(changes);write(path,result);return result

def image_path(stage,arm,identity):return OUT/stage/'images'/arm/(identity+'.png')

def split_path(stage):
    if stage=='round1':stage='pilot'
    active=OUT/'active_inputs.json'
    if active.exists() and stage in read(active)['splits']:return OUT/read(active)['splits'][stage]
    return OUT/'splits'/(stage+'.json')

def protocol_path():return OUT/'config_r1.json'
