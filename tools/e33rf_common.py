"""已确认的RF固定协议，训练和判定均不根据dev调整。"""
import hashlib,json
from pathlib import Path
import torch
from tools.e33rc_common import E33R,E32,DATASET,WEIGHTS,read,write,sha,bootstrap,git_commit,source_checkpoint,load_control
from tools.e33rc_common import OUT as RC
from tools.e33r_common import hash_order
from models.e33rf_real_adapter import FrozenCausalAdapter

OUT=Path('output_eval/e33_rf_frozen_causal_adapter_20261005')
SEEDS=[42,43,44]
ABLATIONS=['A_no_equivariance','B_no_r180','C_full_finetune','D_capacity_x2','E_geometry_only','F_dino_only']
PROTOCOL=dict(experiment='E33-RF',seeds=SEEDS,adapter=dict(input_dim=394,width=32,capacity_x2=64,alpha=.1,
    alpha_trainable=False,last_linear_zero=True,parameter_budget='main<5% of original745475; D capacity ablation exempt'),
    phases=dict(RF1=dict(steps=1500,lr=1e-4,warmup=100),RF2=dict(steps=3000,lr=5e-5,warmup=200),
                RF3=dict(steps=1500,lr=2e-5,warmup=100)),
    optimizer='fresh AdamW each phase;wd1e-4;cosine;bf16;effective8 distinct identities;micro2',
    initialization='RF1 and RF2 independent original E33-R per-seed weights plus zero adapter; RF3 from RF2final',
    loss=dict(match=1,eq90=1,eq180=.5,adapter_id=.05,rank_near=.5,rank_random=.25),
    adapter_id='unscaled A squared mean per token/channel, then case and R/R90/R180 arm mean; RF1 matched only',
    data='exact E32/RC immutable split,GT,donors; train45126/dev256/causal8/ind10; controlled128/strict45',
    rotations='source RGB fixed384x512 canvas native90/180 crop/pad, E32 frozen extractor; no GT rotation or intervention flag input',
    rf0_gate='per seed abs R90 delta<=.02,near delta<=.5deg,match delta<=.5deg,controlledclean>=.99; all3 required',
    pilot='per seed RF2 realR90>=.5,match<=8,controlledclean>=.95; RF3 iff all3 checks pass',
    final='per seed last scheduled endpoint; match<=7,near>=5,nearCIlo>0,R90>=.5,CFclean>=.95/noisy>=.90/R180>=.95;2of3',
    ablations='seed42 independently original full weights;3000steps RF2config; A removes both90/180; C only intentional backbone thaw, no adapter',
    controlled='adapter applied equally to all controlled arms, no domain label or controlled bypass; frozenbackbone eval disables dropout',
    r180='axial response error relative to matched<=15deg; diagnostic only, no added Gate',
    drift='fixed hash64 dev identities over all256;cosine_before_after,raw andscaled L2norm,relative R/R90 cosine-change',
    visual='extremes16/hash16 plus hash16 per RF0success->RFfail/success cohort, fixed4 tracked IDs; union; no manual reselection',
    route='success E33_I_identity_causality; otherwise report per-seed reasons, homogeneous failure routes only, mixed_seed_failures if mixed',
    no_new_data=True,no_new_annotations=True,E5_steps=0)

def protocol_sha():return hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()
def folder(seed,phase,variant='full'):
    return OUT/('seed%d'%seed if variant=='full' else 'ablations/'+variant)/phase
def build(seed,variant='full',checkpoint=None,device='cuda'):
    base,source=load_control(seed,device=device)
    original=sum(p.numel() for p in base.parameters() if p.requires_grad)
    model=FrozenCausalAdapter(base,64 if variant=='D_capacity_x2' else 32,variant).to(device)
    if checkpoint:
        value=torch.load(checkpoint,map_location='cpu');assert value['seed']==seed and value['variant']==variant
        model.load_state_dict(value['model'],strict=True)
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if variant=='full':assert trainable<.05*original
    return model,dict(source_sha256=source,original_trainable=original,trainable=trainable,ratio=trainable/original)
def finish_frozen():
    value=read(OUT/'frozen_check.json');value['after']={p:sha(p) for p in value['before']}
    value['pass']=value['before']==value['after'];assert value['pass'];write(OUT/'frozen_check.json',value)
def controlled_gate(cf):
    checks={k:cf[k]['mean'] is not None and cf[k]['mean']>=t for k,t in
        [('clean_r90_success',.95),('noisy_r90_both_success',.90),('r180_identity_success',.95)]}
    checks['finite']=cf['finite_prediction_rate']==1
    return dict(checks=checks,**{'pass':all(checks.values())})
def real_gate(real):
    near=real['near_advantage'];checks=dict(match=real['errors']['matched']['mean'] is not None and real['errors']['matched']['mean']<=7,
        near=near['mean'] is not None and near['mean']>=5,near_ci=near['ci95'] is not None and near['ci95'][0]>0,
        rot90=real['rot90_success']['mean'] is not None and real['rot90_success']['mean']>=.5,finite=real['finite_prediction_rate']==1)
    return dict(checks=checks,**{'pass':all(checks.values())})
def pilot_gate(cf,real):
    checks=dict(match=real['errors']['matched']['mean'] is not None and real['errors']['matched']['mean']<=8,
        rot90=real['rot90_success']['mean'] is not None and real['rot90_success']['mean']>=.5,
        controlled=cf['clean_r90_success']['mean'] is not None and cf['clean_r90_success']['mean']>=.95,
        finite=real['finite_prediction_rate']==cf['finite_prediction_rate']==1)
    return dict(checks=checks,**{'pass':all(checks.values())})
