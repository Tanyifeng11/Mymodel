"""E34 固定实验合同；所有产物写入独立目录，原始数据与 E5 只读。"""
from pathlib import Path
import hashlib
import json
import os
import random
import subprocess

OUT = Path('output_eval/e34_sarr_20261010')
BF = Path('/share/home/u2515283058/datasets/BF/training')
TRAIN = Path('data/train_bf_texture.json')
VALIDATION = Path('eval/benchmarks/phase1_bf_val_split.json')
E5 = Path('output/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt')
SIZE = (384, 512)
CONFIG = dict(experiment='E34-SARR', version=1,
    plan_sha256='135a1d11a7f8b7aaa8655accccc522c685f13f4d6bc866661a081f0b36a6ebba',
    original_data_modified=False, new_annotations=False, new_reference_pairs=False,
    e5_frozen=True, size=list(SIZE), generation_seed=42, training_seed=42,
    sampler='DDIMScheduler', steps=50, cfg=7., sketch_scale=.6, texture_scale=1., ipa_scale=1.,
    negative_prompt=' worst quality, low quality', null_prompt='', texture_mode='patch_resampled',
    texture_condition_mode='token', texture_preprocess_mode='plain_resize',
    gt_resize='PIL bilinear (existing MyDataset)', sketch_resize='PIL bilinear (existing generate_one)',
    reference_resize='original RGB to original E5; new encoder only bilinear128',
    split_hash_prefix='E34-SARR/dev/v1/', dev_n=128, pilot_n=256,
    mask=dict(source='sketch_only', backend='opencv', erosion_distance_px=17,
        ambiguity_resolution='Euclidean distance>17, not 17x17 kernel (which only erodes8px)',
        line_threshold=245, line_dilation_kernel=3, feather_px=8,
        min_image_coverage=.03, min_garment_coverage=.20, min_valid_rate=.80,
        invalid='exact bypass; retain fixed denominator; never GT fallback'),
    descriptor=dict(patch=64, stride=32, max_patches=16, min_occupancy=.95,
        highpass_sigma=4., lap_sigmas=[.8,1.6,3.2], fft_bins=16, fft_range=[1/64,.25],
        features='official frozen AlexNet first2 ReLU maps, first32 channels Gram upper triangle and variance',
        weights=[.35,.35,.30], calibration='median GT-to-reference component distances on dev128; floor1e-6',
        missing='no patch: no improvement in fixed denominator; diagnostic values null'),
    s0_gate=dict(visual='fixed hash-first24 AI review; stop if majority motif identity/layout/content errors',
        energy='descriptive only; no energy-only success gate'),
    network=dict(channels=[32,64,128], q_dim=128, residual_alpha=.35,
        no_ref='same entire architecture; reference encoder consumes bilinear128 I_in, never R',
        init='zero delta head and zero FiLM; exact identity'),
    train=dict(microbatch=2, accumulation=4, lr=1e-4, final_lr=1e-5,
        pilot_updates=1500, full_updates=8000, pilot_checkpoints=[0,500,1000,1500],
        full_checkpoints=[0,2000,4000,6000,8000], selection='dev masked Laplacian error only',
        losses=dict(rec=1.,lap=.5,per=.1,low=.2,gate=.005)),
    s2_gate=dict(lap_gain=.08, additional_gain_vs_no_ref=.03),
    s3_gate=dict(tex_gain=.05, additional_gain_vs_no_ref=.03, iou_drop=.01,
        contour_drop=.01, text_drop=.005, outside_MAE=0.),
    s3b=dict(enabled=False, eligibility='only S2 pass and real E5 S3 fail', max_updates=1000),
    formal_seeds=[42,123,2026], bootstrap_draws=10000, bootstrap_seed=32042,
    statistical_unit='original garment identity; seeds aggregate within identity')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.%d.tmp'%os.getpid())
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def commit():
    return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()

def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def init():
    path=OUT/'protocol.json'
    if path.exists(): assert read(path)==CONFIG, '已冻结协议不允许悄悄改变'
    else: write(path,CONFIG)
    env=OUT/'environment.json'
    if env.exists():
        import cv2
        import PIL
        import torch
        locked=read(env)
        assert cv2.__version__==locked['opencv'] and PIL.__version__==locked['pillow']
        assert torch.__version__==locked['torch'], '实验期间依赖版本不得静默变化'

def decision(**values):
    path=OUT/'final_decision.json'
    d=read(path) if path.exists() else dict(S0='not_run',S1='not_run',S2a='not_run',
        S2b='not_run',S3='not_run',S3b='not_run',S4='not_run',paper_method_sufficient=False)
    d.update(values); write(path,d); return d

def apply_amendments():
    amendment=read('tools/e34_sarr_amendments.json')
    actual=read(OUT/'masks_audit.json')
    assert actual['valid_n']==96 and actual['n']==128 and actual['valid_rate']==.75
    write(OUT/'protocol_amendments.json',amendment)
    decision(S1='pass_user_exception',S1_original_numerical='fail',
        S1_observed_valid_rate=.75,S1_original_threshold=.80,
        S1_user_override=True,stopped=False,next_phase='S0_then_conditional_S2',
        note='用户观察结果后明确豁免 S1；原始75%保留，掩码和名单不变')
    return amendment

def bootstrap(values):
    import numpy as np
    a=np.asarray(values,np.float64)
    if not len(a): return dict(n=0,mean=None,ci95=None)
    rng=np.random.default_rng(CONFIG['bootstrap_seed']);means=[]
    for start in range(0,CONFIG['bootstrap_draws'],500):
        means.extend(a[rng.integers(0,len(a),(min(500,CONFIG['bootstrap_draws']-start),len(a)))].mean(1))
    return dict(n=len(a),mean=float(a.mean()),ci95=np.percentile(means,[2.5,97.5]).tolist())

def verify_sources():
    before=read(OUT/'frozen_hashes.json')
    after={p:sha(p) for p in before}
    assert before==after, '原始输入或冻结权重发生变化'
    write(OUT/'frozen_check.json',dict(before=before,after=after,pass_unchanged=True))
