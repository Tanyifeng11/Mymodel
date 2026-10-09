"""E33-GC首版预算、采样、统计协议；旧实验只读。"""
from pathlib import Path
from tools.e33tm_protocol import read,write,sha,commit,DATASET,E5
from tools.e33tmoc_protocol import IF,TM

OUT=Path('output_eval/e33_gc_generation_causality_20261009')
RF=Path('output_eval/e33_rf_frozen_causal_adapter_20261005')
MS=Path('output_eval/e33_tm_ms_representation_search_20261007')
CONFIG=dict(experiment='E33-GC',version=1,
    plan_sha256='1047fb0227ace6e3374349c8883a72cf811d995eefcc84a1c5907126b6347ac3',
    size=[384,512],sampler='DDIM',steps=50,prefix=42,unroll=8,noise_seed=42,training_seed=42,
    cfg=7.,sketch_scale=.6,texture_scale=1.,negative_prompt=' worst quality, low quality',
    bootstrap_draws=2000,bootstrap_seed=32042,unit='identity; rotations repeated; unreadable failure',
    adapter=dict(site='up_blocks.3.attentions.0.transformer_blocks.0.attn2',width=32,alpha=.1,
        parameters=196064,parameter_max=1000000,conditional_only=True,zero_last_projection=True,
        appearance='actual frozen E5 spatial IP-attention output at same target query coordinates'),
    mask=dict(source='original sketch-only build_sketch_garment_mask',erosion=17,falloff_pixels=8),
    renderer=dict(seed_prefix='E33GC/renderer/v1/',period_range=[16.,48.],angle_degrees=[0,22.5,45,67.5,90,112.5,135,157.5],
        colors='independent train-only hash RGB uniform40..215; contrast scalar35..75',
        waveform=['cosine','tanh_cosine','elongated_dashes'],phase_range=[0.,6.283185307179586],
        shading_sigma=16.,shading_range=[.85,1.15],background='same lowpass train target; no high-frequency RGB target copied'),
    loss=dict(rgb=1.,appearance=.20,orientation_pair=.50,structure=.25,text=.10,identity=.05),
    G3=dict(train_max=512,updates=400,lr=1e-4,weight_decay=1e-4,effective_batch=4,
        selection='step400 only; step0/100/200 diagnostic only',r90=.30,gain=.20,r180=.70,readable=.60,
        human_visible_min=12,human_denominator=16,contour_drop=.03,contour_ci_lower=-.05,
        sketch_drop=.03,background_MAE_increase=.02),
    G4=dict(train_max=512,updates=300,lr=5e-5,controlled_fraction=.5,effective_batch=4,
        r90=.20,gain=.15,r180=.65,texture=.060,texture_gain=.015,contour_drop=.03,
        contour_ci_lower=-.05,sketch_drop=.03,text_ratio=.95,readable=.55,eligible_min=24),
    sham='same-budget same-module; RF2 R0 field fixed for all reference arms; other inputs and loss weights unchanged',
    G5='only after G4 jointPASS; same gate confirmation64 once; then remaining126 and declared primary254',
    storage='all server raw outputs in independent OUT; final narrative report local docs only')

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/'protocol/frozen_config.json'
    if p.exists():assert read(p)==CONFIG
    else:write(p,CONFIG)
    for name in ['diagnostic64','confirmation64','remaining126','field_success_fixed_hash16']:
        src=MS/'splits'/(name+'.json');dst=OUT/'splits'/(name+'.json')
        if not dst.exists():write(dst,read(src))
        assert sha(src)==sha(dst)
    paths=[E5,RF/'seed42/RF2/checkpoint_final.pt']
    for root in [IF,MS,TM]:
        for name in ['artifact_manifest.json','completion_check.json','decision_summary.json']:
            path=root/name
            if path.exists():paths.append(path)
    before={str(p):sha(p) for p in paths};f=OUT/'frozen_check.json'
    if f.exists():assert read(f)['before']==before
    else:write(f,dict(before=before))
    if not (OUT/'decision_summary.json').exists():
        keys=['reproduction_pass','intervention_validity_pass','controlled_target_pass','adapter_noop_exact',
            'final8_autograd_pass','input_audited_eligible_n','g3_controlled_image_r90_fixed','g3_controlled_r180_fixed',
            'g3_gate_pass','g4_real_image_r90_fixed','g4_real_image_r90_gain_pp_vs_matched_e5',
            'g4_real_r90_ci_lower_pp','g4_real_texture_sim','g4_real_contour_f1','g4_real_sketch_recall',
            'g4_real_textscore','reference_identity_discrimination_supported','g4_gate_pass','confirmation_pass','feasibility_pass']
        write(OUT/'decision_summary.json',dict({k:None for k in keys},paper_method_sufficient=False,next_route='G0_G1_G2_pending'))
    write(OUT/'protocol/code_manifest.json',dict(git_commit=commit(),files={str(p):sha(p) for p in
        sorted(Path('tools').glob('e33gc*.py'))+sorted(Path('models').glob('e33gc*.py'))+sorted(Path('data').glob('e33gc*.py'))+
        sorted(Path('submit').glob('e33gc*.sh'))}))

def decision(**updates):
    p=OUT/'decision_summary.json';d=read(p);d.update(updates);write(p,d);return d

def frozen_check():
    p=OUT/'frozen_check.json';r=read(p);r['after']={name:sha(name) for name in r['before']}
    r['pass']=r['before']==r['after'];write(p,r);assert r['pass']

def bundle(label):
    import tarfile
    path=OUT/(label+'_review.tar.gz')
    with tarfile.open(path,'w:gz') as tar:
        for f in sorted(OUT.rglob('*')):
            if f.is_file() and (f.suffix=='.json' or 'visual_audit' in f.parts):tar.add(f,arcname=str(f.relative_to(OUT)))
    print('[GC bundle]',path,'SHA256',sha(path),flush=True)
