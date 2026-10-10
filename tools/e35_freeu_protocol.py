"""E35 固定合同：继承 E34 身份和原始 E5，所有新产物集中保存。"""
import hashlib
import json
import os
import subprocess
from pathlib import Path

OUT = Path('output_eval/e35_freeu_skip_20261010')
PREVIOUS = Path('output_eval/e34_sarr_20261010')
E5 = Path('output/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt')
SIZE = (384, 512)
ARMS = {
    'A0_E5_OFF': None,
    'A1_MILD': dict(b1=1.10, b2=1.20, s1=.95, s2=.80),
    'A2_MEDIUM': dict(b1=1.20, b2=1.40, s1=.90, s2=.50),
    'A3_SD15_OFFICIAL': dict(b1=1.50, b2=1.60, s1=.90, s2=.20),
}
METRICS = ['struct_iou', 'struct_edge_f1', 'lpips_gt', 'clip_texture',
           'clip_real', 'tcf_lab_delta', 'leak_colored_frac', 'tpf_gram_l1',
           'tpf_patch_sim', 'leak_edge_density']
MASK_METRICS = ['struct_iou', 'tcf_lab_delta', 'leak_colored_frac',
                'tpf_gram_l1', 'tpf_patch_sim', 'leak_edge_density']
CONFIG = dict(experiment='E35', version=1, training_steps=0, extra_parameters=0,
    width=384, height=512, steps=50, cfg=7., sketch_scale=.6, texture_scale=1., ipa_scale=1.,
    seed=42, caption_policy='original_unchanged', dataset_modified=False,
    split='E34 dev128; SHA256(identity) ascending first32 / remaining96',
    freeu_candidates=ARMS, bootstrap_reps=10000, bootstrap_seed=32042,
    mask_source='sketch_only; low-confidence/fallback invalid; no GT fallback',
    thresholds=dict(iou_gain=.005, iou_ci_lower=0., edge_drop=.010, lab_rise=.50,
        clip_drop=.005, leak_rise=.005, lpips_rise=.010),
    lpips_threshold_note='pre-registered absolute +0.010 to operationalize no obvious LPIPS degradation',
    selection='max paired mean IoU; paired candidate gap CI spanning0 => LPIPS then mildness',
    screen_safety='apply confirmation safety limits, require all core metrics measurable; no positive-IoU gate on dev32',
    catastrophic_proxy=dict(iou_drop=.15,edge_drop=.15,leak_rise=.10,
        policy='objective screen flag, requires visual audit; reject candidate before confirm if any flag'),
    missing='report valid/invalid per metric and fixed cohort denominator; no silent deletion',
    official_seeds=[42,123,2026], b_authorized_by_evidence=False)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.%d.tmp'%os.getpid())
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def commit():
    return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()

def decision(**kwargs):
    path = OUT/'final_decision.json'
    state = read(path) if path.exists() else dict(experiment='E35',p0_pass=None,
        a_result='not_run',b_result='not_run',b_authorized=False,
        official_validation_result='not_run',novel_method_claim_supported=False,
        dataset_modified=False,caption_modified=False,training_steps=0)
    state.update(kwargs);write(path,state);return state

def bootstrap(values):
    import numpy as np
    a = np.asarray(values, np.float64)
    if not len(a): return dict(n=0,mean=None,ci95=None)
    assert np.isfinite(a).all()
    rng=np.random.default_rng(32042);means=[]
    for _ in range(20): means.extend(a[rng.integers(0,len(a),(500,len(a)))].mean(1))
    return dict(n=len(a),mean=float(a.mean()),ci95=np.percentile(means,[2.5,97.5]).tolist())

def verify_sources():
    hashes=read(OUT/'audit/frozen_hashes.json')
    assert all(sha(p)==h for p,h in hashes.items()), '原始权重或数据发生变化'
    write(OUT/'audit/frozen_check.json',dict(pass_unchanged=True,files=len(hashes)))

def prepare():
    import platform
    import cv2
    import diffusers
    import numpy as np
    import PIL
    import torch
    from garment_mask_utils import mask_backend_info
    if (OUT/'protocol.json').exists(): assert read(OUT/'protocol.json')==CONFIG
    write(OUT/'protocol.json',CONFIG)
    audit=read(PREVIOUS/'dataset_audit.json')
    assert (audit['original_unique_n'],audit['validation_n'],audit['train_n'])==(45384,500,44756)
    old=read(PREVIOUS/'frozen_hashes.json')
    assert all(sha(p)==old[p] for p in [str(E5),'data/train_bf_texture.json','eval/benchmarks/phase1_bf_val_split.json'])
    dev=read(PREVIOUS/'splits/dev128.json');train=read(PREVIOUS/'splits/train.json')
    val=read(PREVIOUS/'splits/validation_original.json')
    original=read('data/train_bf_texture.json')
    index={Path(r['cloth']).stem:r for r in original}
    held={Path(r['target']).stem for r in val};ids={r['id'] for r in dev}
    assert len(ids)==128 and len(held)==500
    assert not ids&held and not ids&{r['id'] for r in train}
    frozen={}
    for r in dev:
        assert r['caption']==index[r['id']]['caption']
        for key,source_key in [('gt','cloth'),('sketch','sketch'),('reference','texture')]:
            source=index[r['id']].get(source_key,index[r['id']].get('color'))
            assert Path(r[key])==Path('/share/home/u2515283058/datasets/BF/training')/source
            assert sha(r[key])==old[r[key]];frozen[r[key]]=old[r[key]]
    dev=sorted(dev,key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest())
    write(OUT/'splits/dev32.json',dev[:32]);write(OUT/'splits/confirm96.json',dev[32:])
    for p in [E5,Path('data/train_bf_texture.json'),Path('eval/benchmarks/phase1_bf_val_split.json'),
              PREVIOUS/'splits/dev128.json',PREVIOUS/'splits/train.json',PREVIOUS/'splits/validation_original.json']:
        frozen[str(p)]=sha(p)
    write(OUT/'audit/frozen_hashes.json',frozen)
    write(OUT/'audit/dataset_audit.json',dict(original_n=len(original),validation_n=len(val),dev_n=128,
        screen_n=32,confirm_n=96,train_n=len(train),disjoint=True,caption_unchanged=True,
        split_hashes={p.name:sha(p) for p in (OUT/'splits').glob('*.json')},
        identity_limitation='train-only; E5 may have trained on these identities'))
    write(OUT/'audit/environment.json',dict(python=platform.python_version(),torch=torch.__version__,
        cuda=torch.version.cuda,diffusers=diffusers.__version__,opencv=cv2.__version__,
        pillow=PIL.__version__,numpy=np.__version__,git_commit=commit(),mask_backend=mask_backend_info()))
    diff=subprocess.check_output(['git','diff','d0c5b8b','HEAD'],text=True)
    (OUT/'audit/git_diff.patch').write_text(diff,encoding='utf-8')
    print('PREPARED dev32=32 confirm96=96 train=44756',flush=True)

if __name__=='__main__': prepare()
