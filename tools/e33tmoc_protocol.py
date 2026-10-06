"""OC逐级Gate，原IF split与冻结输入只读。"""
from pathlib import Path
from tools.e33tm_protocol import read, write, sha, commit, DATASET, order
from tools.e33tmif_protocol import OUT as IF, TM

OUT = Path('output_eval/e33_tm_oc_carrier_construction_20261006')
PLAN_SHA = 'cef6c359de1c6f88db83101f4f8b3803e47a987ac899f20e3f0c934f5049b72b'
PROTOCOL = dict(experiment='E33-TM-OC', plan_sha256=PLAN_SHA, training_steps=0,
    rf_seeds=[42,43,44], development_seed=42, diffusion_seed=42,
    original_split=str(IF/'splits'), bootstrap_draws=2000, bootstrap_seed=32042,
    angle='image x right,y down; theta is texture tangent; analytic normal=(-sin theta,cos theta)',
    synthetic=dict(angles=[0,22.5,45,67.5,90,112.5,135,157.5],
        source='neutral procedural horizontal cosine stripes, period16,128+80*cos; no real reference appearance',
        full_canvas='identity full-white mask only to satisfy old remapper API; formal output is raw S1 before mask/fallback',
        sizes=[[384,512]], source_size=[384,512], field_hw=[64,48],
        formal_evaluator='frozen E26 native 16x12 patches; all192 cells, unreadable retained',
        gate_error_degrees=3, gate_readable=.95,
        controls='source and direct analytic renderer, plus unbounded procedural sampling of exact old UV; controls never replace formal gate',
        local_boundary='report all cells and cells whose64px footprints stay entirely within a single known region'),
    candidate=dict(S1_r90=.55,S2_r90=.50,S1_readable=.65,S2_readable=.60,
        S1_r180=.70,S2_r180=.70,S1_S2_drop=.10),
    freeze='at most1 main and1 fallback; S2 R90, R180,readability,appearance; no heldout tuning',
    confirmation=dict(S2_r90=.45,gain=.20,paired_ci_lower=0),
    appearance=dict(texture_ratio=.8,color='paired Lab-distance increase not significantly positive; CI lower<=0'),
    vae=dict(r90_drop=.05), mini=dict(r90=.35,gain=.20,r180=.65,contour_drop=.05,text_drop=.10),
    full=dict(S1_r90=.50,S2_r90=.45,r90=.30,gain=.20,text_drop=.05,
        contour='no significant paired decrease'),
    final=dict(min_seeds=2,S1_r90=.50,S2_r90=.45,r90=.30,gain=.20,r180=.65,
        text_drop=.05,contour_drop=.05),
    refinement=dict(strength=.15,steps=50,cfg=7,sketch_scale=.6,texture_scale=1),
    user_confirmation='开始实施')

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/'protocol/protocol.json'
    if p.exists(): assert read(p)==PROTOCOL
    else: write(p,PROTOCOL)
    cohort=read(TM/'manifests/cohorts.json')['primary']
    assert len(cohort)==254
    for name in ('diagnostic64','confirmation64','remaining126','field_success_fixed_hash16'):
        source=IF/'splits'/(name+'.json'); dest=OUT/'splits'/(name+'.json')
        if dest.exists(): assert read(dest)==read(source)
        else: write(dest,read(source))
        assert sha(source)==sha(dest)
    locked=list(read(IF/'frozen_check.json')['before'])
    locked += [str(IF/p) for p in ('completion_check.json','decision_summary.json','protocol/protocol.json')]
    locked += [str(IF/'splits'/(n+'.json')) for n in ('diagnostic64','confirmation64','remaining126','field_success_fixed_hash16')]
    locked += ['models/local_pattern_field.py','models/pattern_geometry.py','data/e32_target_pseudogt.py']
    fingerprints={p:sha(p) for p in sorted(set(locked))}
    p=OUT/'frozen_check.json'
    if p.exists(): assert read(p)['before']==fingerprints
    else: write(p,dict(before=fingerprints,git_commit=commit(),training_steps=0))
    write(OUT/'protocol/implementation.json',dict(git_commit=commit(),
        source_files={str(p):sha(p) for p in sorted(Path('tools').glob('e33tmoc_*.py'))+
            sorted(Path('models').glob('e33tmoc_*.py'))+sorted(Path('submit').glob('e33tmoc*.sh'))},
        split_sha256={p.stem:sha(p) for p in (OUT/'splits').glob('*.json')}))
    return cohort

def frozen_check():
    p=OUT/'frozen_check.json'; record=read(p)
    record['after']={name:sha(name) for name in record['before']}
    record['pass']=record['before']==record['after']; assert record['pass']
    write(p,record)

def freeze_carrier_inputs(cohort,ids,seed=42):
    """RF缓存对照IF最终指纹，新增原图/GT逐文件冻结。"""
    from data.e32_field_dataset import cache_path
    from tools.e33rf_common import E32
    previous=read(IF/'artifact_manifest.json')['files'];paths=[]
    for row in cohort:
        if row['id'] not in ids:continue
        sid=row['id'];field=IF/'reproduction'/('seed%d'%seed)/'fields'/(sid+'.npz')
        assert sha(field)==previous[str(field.relative_to(IF))], 'RF field缓存与IF归档不一致'
        baseline=IF/'stage_survival'/('seed%d'%seed)/sid/'case.json'
        assert sha(baseline)==previous[str(baseline.relative_to(IF))]
        paths += [field,baseline,cache_path(E32,sid),DATASET/row['reference'],DATASET/row['sketch']]
    fingerprints={str(p):sha(p) for p in paths}
    path=OUT/'protocol'/('carrier_inputs_seed%d_N%d.json'%(seed,len(ids)))
    if path.exists():assert read(path)['files']==fingerprints
    else:write(path,dict(files=fingerprints,IF_field_manifest_verified=True,case_count=len(ids),seed=seed))
    return fingerprints

