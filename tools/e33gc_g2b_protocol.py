"""G2b v1.1：只写新目录；用户授权双 AI 审核，人工字段仍为 null。"""
from pathlib import Path
import hashlib
import json
from tools.e33tm_protocol import read, write, sha, commit, DATASET, E5, RF
from tools.e33gc_protocol import OUT as OLD
from tools.e33tmoc_protocol import IF, TM

OUT = Path('output_eval/e33_gc_g2b_20261009')
ARMS = ('R0', 'R90', 'R180')
CONFIG = dict(experiment='E33-GC-G2b-v1.1', plan_sha256='5c0656104dda8436e5e1483802dd5f0214d200db8fdf65d31e114a986914fec5',
    size=[384,512], steps=50, prefix=42, unroll=8,
    cfg=7., sketch_scale=.6, texture_scale=1., negative_prompt=' worst quality, low quality',
    train_seed=42, noise_seed=42, updates=160, diagnostics=[0,40,80,160], endpoint=160,
    optimizer='AdamW', lr=1e-4, weight_decay=1e-4, clip_grad_norm=1., effective_identities=4,
    microbatch='one identity with three paired arms', site='up_blocks.3.attentions.0.transformer_blocks.0.attn2',
    parameters=196064, alpha_max=.1, loss=dict(rgb=1.,orientation_pair=.5,outside=.25),
    gpu_hours_cap=6., upper_updates=80, upper_fit_labels=['fit_01','fit_02'],
    review=dict(type='two_independent_AI_agents', user_override='双人独立审核由你完成。开始实现相关实验',
        human_certified=False, scientific_human_fields=None),
    route_A='real RGB source crop repeated with fixed local coordinates; only patch rotates inside reference support',
    crop=[128,192,256,320], crop_rule='fixed center128 before output; no output-dependent crop search',
    input_rule=dict(confidence=.35,gray_std=.025,extreme_fraction_max=.20),
    dft=dict(block=64,stride=64,minimum_blocks=3,min_std=.015,min_axis_coherence=.35,
        low_frequency=1.5/64,high_frequency=.20,maximum_error=15),
    new_dev_independence='GC adapter only; may have been seen by historical E5/RF2',
    G3='not_authorized',G4='not_authorized',G5='not_authorized')

def decision(**values):
    p=OUT/'decision_summary.json'; d=read(p) if p.exists() else {}
    d.update(values); write(p,d); return d

def init():
    OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/'protocol/frozen_config.json'
    if p.exists(): assert read(p)==CONFIG
    else: write(p,CONFIG)
    if not (OUT/'decision_summary.json').exists():
        names=['gc_controlled_dev_new_frozen','real_input_human_eligible_n','real_input_eligibility_gate_24of64',
            'g1a_reference_target_consistency_pass','g1a_human_direction_gate_pass','g1a_text_compatibility_pass',
            'g2b_route','g2b_train_identity_n','g2b_step0_exact_match','g2b_loss_autograd_pass','g2b_peak_memory_gib',
            'g2b_fit_final_image_r90_n_of_4','g2b_fit_r180_n_of_4','g2b_fit_readable_n_of_4',
            'g2b_fit_visible_motif_n_of_4','g2b_fit_masked_charbonnier_reduction_pct',
            'g2b_fit_structure_and_appearance_gate','g2b_fit_pass','g2b_free_residual_upper_pass',
            'g2c_six_loss_contract_pass','feasibility_pass']
        decision(**{n:None for n in names},g0_original_reproduction_unchanged=True,
            g2_original_autograd_smoke_unchanged=True,old_controlled_dev128_collision_confirmed=True,
            g3_authorized=False,g4_authorized=False,g5_authorized=False,paper_method_sufficient=False,
            review_type=CONFIG['review']['type'],next_route='preparation')
    p=OUT/'protocol/original_g0_g2_hashes.json'
    if not p.exists():
        paths=[OLD/'decision_summary.json',OLD/'completion_check.json',OLD/'frozen_check.json',
            OLD/'protocol/frozen_config.json',E5,RF/'seed42/RF2/checkpoint_final.pt']
        write(p,{str(x):sha(x) for x in paths})
    verify_frozen()

def verify_frozen():
    before=read(OUT/'protocol/original_g0_g2_hashes.json')
    after={p:sha(p) for p in before}
    write(OUT/'frozen_check.json',dict(before=before,after=after,pass_unchanged=before==after))
    assert before==after

def bundle(label):
    import tarfile
    path=OUT/(label+'_review.tar.gz')
    with tarfile.open(path,'w:gz') as tar:
        for f in sorted(OUT.rglob('*')):
            if f.is_file() and (f.suffix in ('.json','.csv','.txt') or
                ('visual_audit' in f.parts and f.suffix=='.png')):
                tar.add(f,arcname=str(f.relative_to(OUT)))
    print('BUNDLE',str(path),sha(path),flush=True)

