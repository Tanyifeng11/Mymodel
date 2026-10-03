"""固定E33-RC协议；真实/controlled分母分别保持E32/E33-R原划分。"""
from pathlib import Path
import hashlib
import json
import torch
from tools.e32_common import read, write, sha, bootstrap, git_commit, DATASET, WEIGHTS
from tools.e33r_common import OUT as E33R, E32, hash_order
from models.e33r_rotation_control import SketchPrior, RotationControl

OUT = Path('output_eval/e33_rc_real_rotation_20261003')
STAGES = [('RCA_75_25', 2000, .75), ('RCB_50_50', 3000, .50), ('RCC_25_75', 3000, .25)]
ABLATIONS = ['A_no_ranking', 'B_no_retention', 'C_real_only', 'D_fixed_50_50',
             'E_geometry_only', 'F_dino_only']
PROTOCOL = dict(experiment='E33-RC', seeds=[42,43,44], stages=STAGES,
    plan_sha256='9026fbb8576cb2bd139efa4978f479e05d91faa006ac9411e426cd368a4a51d0',
    initialization='corresponding E33-R full step8000; fresh optimizer; no checkpoint selection',
    optimizer=dict(lr=5e-5, weight_decay=1e-4, warmup=300, steps=8000, precision='bf16',
                   effective_batch=8, microbatch=2, scheduler='continuous8000step cosine'),
    loss=dict(controlled=1., real=1., rank=.5, retain=.5, near_margin_deg=5., random_margin_deg=7.5),
    aggregation='per-target branch losses weighted by branch identity count/effective8; retention auxiliary probe mean separately',
    real_protocol='exact E32 split/cache/heldout donor pool/dev256/GT support and fixed-canvas90; drop only period input',
    controlled_protocol='unchanged E33-R group_loss, M_cf, two fixed eval nuisance arms, jitter and dev128/strict45',
    retention='fixed original sanity train512 probes; clean R0/R90/R180; teacher corresponding frozen full checkpoint',
    retention_probe_batch=2, retention_note='auxiliary probes separate from 8-identity reconstruction mixing ratio',
    bootstrap_unit='target identity; paired variants; 2000 draws; E32 fixed seed32042',
    controlled_gate=dict(clean=.90,noisy=.85,r180=.90,zero=.10,sensitivity=2.),
    real_gate=dict(near_advantage_deg=5.,near_ci_lower_positive=True,rot90=.50,matched_error_deg=7.),
    stage_stop='any retention Gate fail stops this seed; clean<.85 triggers one global revision',
    revision=dict(maximum=1,ratio=.5,retain=1.,restart='all3 original E33-R full checkpoints;8000steps each'),
    ablation_policy='run six fixed8000step seed42 variants iff selected full seed42 reaches RC-C; no revision/early collapse stop in ablations',
    real_only='L_controlled=0 and L_retain=0; only real reconstruction and ranking, no probe training',
    input_variants='E/F load same full seed42 weights; input masks follow E33-R G/F; full-input teacher fixed',
    shortcut_diagnostics='continuous paired errors/CI and visual audit; approximate statements do not change Gates',
    forbidden=['period loss','real rot90 training','new annotations','new external data','E5 training','heldout tuning'])

def protocol_sha():
    return hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()

def variant_model(variant):
    return {'E_geometry_only':'G_geometry_only','F_dino_only':'F_dino_only'}.get(variant,'full')

def seed_folder(seed,revision=False,variant='full'):
    if variant!='full': return OUT/'ablations'/variant
    return OUT/('retention_revision' if revision else '')/('seed%d'%seed)

def source_checkpoint(seed): return E33R/'P1_controlled'/('seed%d'%seed)/'checkpoint_final.pt'

def load_control(seed,checkpoint=None,variant='full',device='cuda'):
    source=source_checkpoint(seed); integrity=read(source.parent/'checkpoint_integrity.json')
    assert sha(source)==integrity['sha256']
    path=Path(checkpoint) if checkpoint else source
    checkpoint_data=torch.load(path,map_location='cpu')
    assert checkpoint_data['seed']==seed
    if path==source: assert checkpoint_data['steps']==8000 and checkpoint_data['variant']=='full'
    model=RotationControl(SketchPrior(),variant_model(variant)).to(device)
    model.load_state_dict(checkpoint_data['model'],strict=True)
    model.prior.requires_grad_(False);model.prior.eval()
    expected=torch.load(E33R/'P0_prior/checkpoint_final.pt',map_location='cpu')['model']
    assert all(torch.equal(v.cpu(),expected[k]) for k,v in model.prior.state_dict().items())
    return model,sha(path)

def finish_frozen():
    value=read(OUT/'frozen_check.json')
    value['after']={p:sha(p) for p in value['before']}
    value['pass']=value['before']==value['after'];assert value['pass']
    write(OUT/'frozen_check.json',value)

def selected_revision():
    return (OUT/'retention_revision/authorization.json').exists()

def retention_gate(summary):
    thresholds={'clean_r90_success':(.90,True),'noisy_r90_both_success':(.85,True),
                'r180_identity_success':(.90,True),'zero_ratio':(.10,False),'sensitivity_ratio':(2.,True)}
    checks={k:summary[k]['mean'] is not None and (summary[k]['mean']>=t if up else summary[k]['mean']<=t)
            for k,(t,up) in thresholds.items()}
    checks['finite']=summary['finite_prediction_rate']==1
    clean=summary['clean_r90_success']['mean']
    return dict(checks=checks,**{'pass':all(checks.values())},collapse=clean is not None and clean<.85)

def real_gate(summary):
    advantage=summary['near_advantage'];error=summary['errors']['matched'];rotation=summary['rot90_success']
    checks=dict(near_advantage=advantage['mean'] is not None and advantage['mean']>=5,
        near_ci_positive=advantage['ci95'] is not None and advantage['ci95'][0]>0,
        rot90=rotation['mean'] is not None and rotation['mean']>=.5,
        matched_error=error['mean'] is not None and error['mean']<=7,
        finite=summary['finite_prediction_rate']==1)
    return dict(checks=checks,**{'pass':all(checks.values())})
