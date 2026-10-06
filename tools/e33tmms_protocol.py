"""MS固定协议；旧实验只读，所有新产物写入独立目录。"""
from tools.e33tmoc_protocol import IF, TM, OUT as OC
from tools.e33tm_protocol import read, write, sha, commit, DATASET, order
from pathlib import Path

OUT = Path('output_eval/e33_tm_ms_representation_search_20261007')
PROTOCOL = dict(experiment='E33-TM-MS',version=1,
    plan_sha256='dfe20ee1cb34a94aaf094f02880cfe84cf86f037ad10a531d74732254d67a524',
    development_seed=42,diffusion_seed=42,bootstrap_draws=2000,bootstrap_seed=32042,
    statistical_unit='target identity; rotations repeated measures; unreadable counts failure',
    search='MS0 -> M1 -> M2 only if M1 fails -> M3 only if M2 fails; stop at first final/confirmation success',
    reproduction=dict(r90_tolerance=.02,texture_relative_tolerance=.05),
    M1=dict(patch=64,stride=16,iterations=3,random_search_decay=.5,
        foreground_min=.95,q_min=.20,variance_min=.0003,
        weights=dict(appearance=1.,neighborhood=.5,orientation=1.,seam=.25),
        canonical_guard=128,canonical_canvas=96,energy_grid=16,
        appearance='normalized Lab mean/std plus gradient magnitude statistics of source foreground; no target RGB',
        neighborhood='source coordinate distance to inverse-steered neighbor proposals, normalized by64px, clipped2',
        orientation='differentiable-style structure tensor axial mismatch of actual steered patch to target RF; E26 only formal evaluation',
        seam='Lab overlap L1 with four currently assigned neighbors, normalized255',
        blend='single Hann64; no seam/patch/stride/mask sweep',
        rotation_support='source64 filtered first; surrounding real128 pixels guard; invalid/background transformed pixels replaced by single foreground-mean fallback',
        fallback='single reference foreground mean RGB; no synthetic stripes'),
    carrier=dict(S1_r90=.45,S2_r90=.40,S2_gain=.15,paired_ci_lower=0.,r180=.60,
        readable=.55,texture=.19,Lab_C0_ratio=1.20),
    M1_hard_stop=dict(S2_r90=.35,texture=.16),
    confirmation=dict(S2_r90=.35,S2_gain=.12,texture=.17,paired_ci_lower=0.),
    mini=dict(r90=.30,gain=.15,r180=.60,contour_drop=.05,text_relative_drop=.10),
    M2=dict(encoder='small CNN global128/local8x8x64',decoder='4-level UNet width32 with local token attention',
        output='RGB first',parameter_max=5000000,controlled_steps=4000,real_mixed_steps=2000,
        controlled_fraction=.5,effective_batch=8,lr=1e-4,weight_decay=1e-4,warmup=200,
        loss=dict(rec=1.,app=.5,ori=1.,inv=.2,cf=1.),controlled_gate=dict(r90=.70,r180=.80,texture=.18),
        confirmation='same as M1, user confirmed',train='original E33 train only; no diagnostic/confirmation/remaining identity',
        real_gate='S1>=.45 S2>=.40 Texture>=.19 Lab<=C0*1.2; hard stop realR90<.30 orTexture<.15'),
    M3=dict(grid='3x3',geometry_learnable=False,alpha=.1,site='one E5 1/8-resolution feature layer',
        controlled_steps=4000,real_mixed_steps=2000,controlled_fraction=.5,
        confirmation='repeat diagnostic gate plus paired R90 gain CI lower>0; user confirmed',
        gate=dict(r90=.35,gain=.20,r180=.65,contour_drop=.05,text_relative_drop=.10,texture_ratio=.9),
        parameter_budget='less than5% original E5 trainable parameter count before freezing backbone'),
    refinement=dict(strength=.15,steps=50,cfg=7.,sketch_scale=.6,texture_scale=1.),
    success=dict(r90=.30,texture_old_full_ratio=.9,confirmation_required=True,
        text_contour='paired CI does not show significant decline; method-specific non-inferiority gates also required'),
    user_confirmation='开始实施; approved M2 confirmation and M3 schedule/confirmation supplements')

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'protocol/protocol.json'
    if path.exists():assert read(path)==PROTOCOL
    else:write(path,PROTOCOL)
    assert read(OC/'completion_check.json')['experiment_execution_complete']
    cohort=read(TM/'manifests/cohorts.json')['primary'];assert len(cohort)==254
    for name in ('diagnostic64','confirmation64','remaining126','field_success_fixed_hash16'):
        src=IF/'splits'/(name+'.json');dst=OUT/'splits'/(name+'.json')
        if dst.exists():assert sha(dst)==sha(src)
        else:write(dst,read(src))
        assert sha(src)==sha(dst)
    before=dict(read(OC/'frozen_check.json')['before'])
    for rel in ('artifact_manifest.json','completion_check.json','decision_summary.json','protocol/appearance_carrier.json'):
        before[str(OC/rel)]=sha(OC/rel)
    f=OUT/'frozen_check.json'
    if f.exists():assert read(f)['before']==before
    else:write(f,dict(before=before,git_commit=commit()))
    write(OUT/'protocol/implementation.json',dict(git_commit=commit(),files={str(p):sha(p) for p in
        sorted(Path('tools').glob('e33tmms*.py'))+sorted(Path('models').glob('e33tmms*.py'))+
        sorted(Path('submit').glob('e33tmms*.sh'))}))
    if not (OUT/'decision_summary.json').exists():
        keys=['reproduction_pass','M1_run','M1_carrier_pass','M1_confirmation_pass','M1_e5_pass',
              'M2_run','M2_controlled_pass','M2_real_carrier_pass','M2_e5_pass','M3_run','M3_feature_pass','M3_e5_pass',
              'selected_method','selected_space','selected_image_r90','selected_texture_sim','selected_text_delta','selected_contour_delta']
        write(OUT/'decision_summary.json',dict({k:None for k in keys},next_route='MS0'))
    return cohort

def frozen_check():
    p=OUT/'frozen_check.json';r=read(p);r['after']={n:sha(n) for n in r['before']}
    r['pass']=r['before']==r['after'];assert r['pass'];write(p,r)

def decision(**updates):
    p=OUT/'decision_summary.json';d=read(p);d.update(updates);write(p,d);return d

def freeze_inputs(rows,ids,split):
    from data.e32_field_dataset import cache_path
    from tools.e33rf_common import E32
    manifest=read(IF/'artifact_manifest.json')['files'];paths=[]
    for row in rows:
        if row['id'] not in ids:continue
        f=IF/'reproduction/seed42/fields'/(row['id']+'.npz')
        assert sha(f)==manifest[str(f.relative_to(IF))]
        paths.extend([f,cache_path(E32,row['id']),DATASET/row['reference'],DATASET/row['sketch']])
    value={str(p):sha(p) for p in paths};dest=OUT/'protocol'/('inputs_%s_N%d.json'%(split,len(ids)))
    if dest.exists():assert read(dest)==value
    else:write(dest,value)
    return value
