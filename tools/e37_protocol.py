"""E37 事前合同：只新增产物，不覆盖 E36；后续阶段必须通过前置门槛。"""
import random
from pathlib import Path
from tools.e35_freeu_protocol import read, write, sha, commit, E5, SIZE

OUT = Path('output_eval/e37_scrao_feasibility_20261010')
E36 = Path('output_eval/e36_dagf_guided_filter_20261010')
WEIGHTS = {400: 'e7717ef30f865f2f66d792b9b82e0a62ffe817c3af24ccbc06fee1ea851cb9e5',
           800: '224ae2939806d63b51715a3164e0dac230880eb8f83cc5d633ccdfc153e74189'}
CONFIG = dict(experiment='E37_P0_SCRAO', version=1, seed_probe=37010, seed_t=37011,
    seed_train512=37012, seed_noise=42, seed_bootstrap=37042, bootstrap_reps=10000,
    parameters=143166, checkpoints=[400,800], train_probe_n=96, dev_n=32,
    s0_identities='first two frozen dev32; gradient first valid frozen train-probe identity at low-t record',
    prohibited_splits=['confirm96','validation500'], conditioning='original E5 unchanged',
    adapter_policy='conditional only; unconditional OFF',
    proxy_mask='strict sketch_only; low confidence NA; no GT fallback',
    reference_roi='entire original plain_resize texture; no reference mask or new crop',
    proxy=dict(t_ranges=[[0,150],[151,400]], boundary_radius=3, inner_erosion=3,
        gaussian_sigmas=[1.,2.], gaussian_sizes=[5,9], sobel_divisor=8,
        gram_normalization='ROI pixel count * feature channels',
        rgb='unclamped differentiable (VAE decode+1)/2; clamp only saved previews',
        calibration='B1-800 train-probe96 x two fixed timesteps, no dropout; no dev',
        lambda_bd='median norm(g_bg) / median norm(g_bd)',
        lambda_tex='median Lcolor / median Ltex',
        lambda_R='median Dmom / median Dgram', gradient_loss_scale=1024.,
        severe_saturation='more than 50% of RGB pixels outside [0,1]',
        x0_fail='nonfinite or more than 50% samples severely saturated'),
    proxy_gate=dict(rho_min=.25, sign_min=.65, max_ci_width=1., mask_coverage_min=.8,
        comparisons=['LS vs -IoU','LS vs Leak','LAGT vs LPIPS','LAR vs TCF','LAR vs TPF_Gram'],
        policy='at least one structure comparison and one reference comparison plus GT pass; no direction contradiction or wide CI',
        sign='exclude exact zero on either side; retain fixed and valid denominators'),
    s2=dict(cosine_threshold=-.1, finite_nonzero_min=.9, conflict_min=.25, ci_lower_min=.1,
        identity_aggregation='mean over two timesteps before bootstrap',
        trim='remove ceil(5% identities) ranked by mean max(norm gS,norm gAR); same gates after trim',
        checkpoint_policy='all gates at both 400 and 800'),
    s3=dict(checkpoint=800, minibatches=8, micro_batch=1, accumulated_batch=4,
        train_external='next 32 fixed probe identities after first32 gradient identities',
        displacement_norm=1e-3, delta=1e-12, matched_norm=True,
        structure_gate='OP-U1 delta LS mean<0 and identity bootstrap upper<0',
        appearance_gate='OP-U1 delta LAR <= 1% of external baseline mean LAR',
        repeated_direction_min=.65),
    p1=dict(arms=['P1_W','P1_PC','P1_OP'], updates=160, micro_batch=1, accumulation=4,
        lr=2e-5, weight_decay=.01, clip_grad=1., seed=42, train_n=512,
        checkpoints=[0,80,160], evaluation_checkpoint=160,
        train_t='original uniform 0..999; RGB auxiliary only t<=400',
        weights='train-probe gradient median norms relative to epsilon, lock before training',
        max_gpu_hours=3., iou_vs_b1_min=.004, iou_ci_lower_min=0.,
        iou_vs_e5_min=-.005, tcf_vs_e5_max=-1.5, lpips_vs_e5_max=-.008,
        edge_min=-.01, clip_min=-.005, leak_max=.005,
        catastrophe=dict(iou=.15, edge=.15, leak=.10)),
    diagnostics_max_gpu_hours=2., new_method_proposed=False,
    new_method_supported=False, paper_novelty_claim=False)


def checkpoint(step):
    return E36/'checkpoints/B1_CONV'/('step%04d.pt'%step)


def decision(**updates):
    path=OUT/'decision.json'
    state=read(path) if path.exists() else dict(experiment=CONFIG['experiment'],
        s0_pass=None,s1_pass=None,s2_pass=None,s3_pass=None,p1_pass=None,
        phase='not_run',stop_reason=None,new_method_proposed=False,
        new_method_supported=False,paper_novelty_claim=False)
    state.update(updates);write(path,state);return state


def prepare():
    import yaml
    if (OUT/'protocol_locked.yaml').exists():
        assert yaml.safe_load((OUT/'protocol_locked.yaml').read_text())==CONFIG
        return
    assert sha(E5)=='aea56b2d7b02492f0e58fc2abc45b6a5a996b9fd962f31e30aa26ff5b0ed28f0'
    for step,h in WEIGHTS.items():assert sha(checkpoint(step))==h
    rows=read(E36/'splits/train1024.json')
    probe=random.Random(37010).sample(sorted(rows,key=lambda r:r['id']),96)
    dev=read(E36/'splits/dev32.json')
    assert not {r['id'] for r in probe}&{r['id'] for r in dev}
    write(OUT/'splits/train_probe96.json',probe);write(OUT/'splits/dev32.json',dev)
    index={r['id']:i for i,r in enumerate(rows)}
    original=read(E36/'splits/train_original_records.json')
    write(OUT/'splits/probe_original_records.json',[original[index[r['id']]] for r in probe])
    rng=random.Random(37011)
    records=[dict(id=r['id'],index=i,range=k,t=rng.randint(lo,hi),noise_seed=37011000+i*2+k)
        for i,r in enumerate(probe) for k,(lo,hi) in enumerate(CONFIG['proxy']['t_ranges'])]
    write(OUT/'splits/fixed_noisy_records.json',records)
    frozen={str(p):sha(p) for p in [E5,*[checkpoint(s) for s in WEIGHTS],
        E36/'p1/metrics.json',E36/'p1/generation_manifest.json',E36/'splits/train1024.json']}
    for r in probe+dev:
        for k in ['gt','sketch','reference']:frozen[r[k]]=sha(r[k])
    for p in (OUT/'splits').glob('*.json'):frozen[str(p)]=sha(p)
    write(OUT/'audit/input_hashes.json',frozen)
    write(OUT/'audit/source_hashes.json',{str(p):sha(p) for p in sorted(Path('.').rglob('*.py'))
        if not p.name.startswith('e37_') and not any(x in p.parts for x in ['.git','output_eval','output','tmp','wandb'])})
    (OUT/'protocol_locked.yaml').write_text(yaml.safe_dump(CONFIG,allow_unicode=True,sort_keys=False),encoding='utf-8')
    write(OUT/'audit/manifest.json',dict(commit=commit(),weights=WEIGHTS,
        dataset_modified=False,caption_modified=False,probe_seen_by_B1=True,
        dev_used_by_E35_E36=True,confirm_or_validation_accessed=False,
        original_training=45384,original_validation=500,dev128=128,remaining_train=44756))
    decision(phase='S0')


def verify_inputs():
    frozen=read(OUT/'audit/input_hashes.json')
    assert all(sha(p)==h for p,h in frozen.items())
    sources=read(OUT/'audit/source_hashes.json')
    assert all(sha(p)==h for p,h in sources.items())
    write(OUT/'audit/inputs_unchanged.json',dict(pass_unchanged=True,files=len(frozen),sources=len(sources)))
