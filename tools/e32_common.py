"""E32 的固定划分、记录、bootstrap 和决策链。"""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


OUT = Path('output_eval/e32_target_supervised_pattern_field_20261001')
DATASET = Path('/share/home/u2515283058/datasets/BF')
WEIGHTS = Path('output_eval/e30_apacc_20260930/dinov2_vits14_pretrain.pth')
CAUSAL_IDS = (6, 7, 9, 10, 12, 13, 14, 17)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def bootstrap(values):
    values = np.asarray(values, float)
    if not len(values):
        return dict(mean=None, ci95=None, n=0)
    draws = np.random.default_rng(32042).choice(values, (2000, len(values)), replace=True).mean(1)
    return dict(mean=float(values.mean()), ci95=np.percentile(draws, [2.5, 97.5]).tolist(), n=len(values))


def initial_decision():
    names = ('pair_dependence_pass', 'geometry_field_pass', 'appearance_field_pass', 'scaffold_pass',
             'generation_pass', 'confirmation_pass', 'matched_orientation_advantage_deg',
             'matched_period_advantage_log2', 'matched_appearance_advantage', 'rot90_geometry_response',
             'rot90_identity_stability', 'reference_ablation_pass', 'structure_safe', 'background_safe',
             'oracle_gap_recovery_follow', 'oracle_gap_recovery_orientation', 'next_route')
    return dict.fromkeys(names)


def make_split(out, dataset):
    all_train = read('data/train_bf_texture.json')
    refs = read('data/e27_c3_cases.json')['references']
    banned = {Path(r['source']).stem for r in refs}
    eligible = [r for r in all_train if Path(r['cloth']).stem not in banned]
    eligible.sort(key=lambda r: hashlib.sha256(('E32/'+r['cloth']).encode()).hexdigest())
    def training_row(r):
        return dict(id=Path(r['cloth']).stem, reference='training/'+r['texture'],
                    target='training/'+r['cloth'], sketch='training/'+r['sketch'])
    dev = [training_row(r) for r in eligible[:256]]
    train = [training_row(r) for r in eligible[256:]]
    known = [dict(id=Path(r['source']).stem, case_id=r['id'],
                  reference='validation/texture/'+Path(r['source']).name,
                  target=r['source'], sketch='validation/sketch/'+Path(r['source']).name)
             for r in refs]
    causal = [r for r in known if r['case_id'] in CAUSAL_IDS]
    confirmation = [r for r in known if r['case_id'] not in CAUSAL_IDS]
    splits = dict(train=train, dev=dev, causal_test=causal, independent_confirmation=confirmation,
                  confirmation_all=known, stage0_train_audit=train[:512],
                  excluded_training_identities=sorted(banned), original_training_count=len(all_train),
                  selection='SHA256(E32/cloth) sort; first256 held-out dev; no evaluation-label selection',
                  original_E29_fixed_sketch='unpaired transfer diagnostic only; not paired target error GT')
    sets = [{r['id'] for r in splits[n]} for n in ('train', 'dev', 'causal_test', 'independent_confirmation')]
    for i, s in enumerate(sets):
        assert len(s) == len(splits[('train', 'dev', 'causal_test', 'independent_confirmation')[i]])
        for t in sets[i+1:]:
            assert not s & t
    missing = [str(dataset/r[k]) for name in ('train', 'dev', 'causal_test', 'independent_confirmation')
               for r in splits[name] for k in ('reference', 'target', 'sketch') if not (dataset/r[k]).is_file()]
    write(out/'audits/input_availability.json', dict(missing=missing, **{'pass':not missing}))
    assert not missing, missing[:10]
    write(out/'split_manifest.json', splits)
    return splits


def frozen_manifest(out, weights):
    sources = ['models/local_pattern_field.py', 'models/pattern_geometry.py', 'garment_mask_utils.py']
    checkpoints = [Path('output/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt')]
    sources += ['models/bf_texture_module.py', 'models/tcpm_lite.py',
                'inference_IMAGGarment-1.py']
    paths = [Path(p) for p in sources] + [Path(weights)]
    paths += [p for p in checkpoints if p.exists()]
    frozen = {str(p):sha(p) for p in paths}
    write(out/'frozen_check.json', dict(before=frozen, after=None, **{'pass':None},
                                      training_steps=0, E5_training_steps=0,
                                      note='DINO/E26 frozen; E5/VAE/U-Net/BF/TCPM not loaded by Stage0'))
    return frozen


def finish_frozen(out):
    value = read(out/'frozen_check.json')
    value['after'] = {p:sha(p) for p in value['before']}
    value['pass'] = value['before'] == value['after']
    write(out/'frozen_check.json', value)
    assert value['pass']


def git_commit():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
