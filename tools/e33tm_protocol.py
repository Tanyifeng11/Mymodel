"""E33-TM 固定协议；结果不能反向改变队列、注入参数或判定阈值。"""
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess

OUT = Path('output_eval/e33_tm_trimodal_validation_20261005')
RF = Path('output_eval/e33_rf_frozen_causal_adapter_20261005')
DATASET = Path('/share/home/u2515283058/datasets/BF')
E5 = Path('output/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt')
SEEDS = (42, 43, 44)
ARMS = ('R0', 'R90', 'R180', 'Rzero')
ORIENTATION = r'\b(vertical(?:ly)?|horizontal(?:ly)?|diagonal(?:ly)?|slanted|oblique)\b'
PATTERN = r'\b(striped?|stripes|plaid|checked|checkered|floral|polka[ -]dots?|printed|geometric|woven)\b'
SCALE = r'\b(small|large|tiny|fine|coarse|dense|sparse|wide|thin)\b'
PROTOCOL = dict(experiment='E33-TM', version=1, training_steps=0, rf_phase='RF2', rf_seeds=SEEDS,
    carrier='E25 VAE posterior mean + fixed strength .15 DDIM refinement; E26 reference RGB remap in sketch interior',
    baseline='original E5 checkpoint, original text/sketch/texture path, standard 50 DDIM steps',
    full='same original E5 weights and appearance path, RF2 dense orientation to spatial carrier, 8 remaining DDIM steps',
    strength=.15, steps=50, guidance_scale=7., sketch_scale=.6, texture_scale=1.,
    bootstrap_draws=2000, bootstrap_seed=32042, statistical_unit='target identity',
    primary='all original dev identities with geometry-neutral captions; no filtering by generated readability',
    missing_orientation='unreadable R0 or rotated output counts as failure; fixed denominator',
    robustness='fixed hash64 primary identities, diffusion seeds 42/43/44/45; separate diagnostic',
    ablations='all primary identities, RF2 seed42, diffusion seed42; same-category different-identity donors',
    text_evaluator='fixed pretrained CLIP text-image cosine; matched original caption even in text ablations',
    texture_evaluator='existing interior patch_texture_similarity plus CLIP image-image cosine',
    gate=dict(r90=.5, gain=.2, paired_ci_lower=0., r180=.8, contour_drop=.02, relative_text_drop=.02),
    hard_stop=dict(precheck_difference=.02, seed42_gain_max=.05, structure_drop=.05),
    text_no_text_r90_tolerance=.10, role_gate='paired case bootstrap 95% lower>0 for each corresponding score drop',
    rf_original_gate_unchanged=True, user_confirmation='开始实施',
    plan_sha256='34d90ab01701902ee864d13a6db6220839f219363c15a9d75ba264f98cbe6567')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.%d.tmp'%os.getpid())
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def order(rows, tag):
    return sorted(rows, key=lambda r: hashlib.sha256((tag+'/'+r['id']).encode()).hexdigest())

def category(caption):
    groups = [('dress', r'dress|gown'), ('jacket', r'jacket|blazer|coat'),
              ('trousers', r'pants|trousers|jeans|leggings'), ('skirt', r'skirt'),
              ('knit', r'sweater|cardigan|pullover'), ('shirt', r'shirt|blouse|top|tunic'),
              ('sweatshirt', r'sweatshirt|hoodie'), ('shorts', r'shorts')]
    for name, words in groups:
        if re.search(r'\b('+words+r')\b', caption, re.I):
            return name
    return 'other'

def pattern(caption):
    for name, words in [('stripe', 'striped|stripes|stripe'), ('plaid', 'plaid|checked|checkered'),
                        ('floral', 'floral'), ('dot', 'polka|dots'), ('print', 'printed|geometric')]:
        if re.search(r'\b('+words+r')\b', caption, re.I):
            return name
    return 'unspecified'

def neutral(caption):
    return re.sub(r'\s+', ' ', re.sub(ORIENTATION, '', caption, flags=re.I)).strip()

def counterfactual(caption):
    def replace(m):
        word = m.group().lower()
        if word.startswith('vertical'): return 'horizontal'
        if word.startswith('horizontal'): return 'vertical'
        return 'oppositely diagonal'
    return re.sub(ORIENTATION, replace, caption, flags=re.I)

def freeze_check():
    value = read(OUT/'frozen_check.json')
    value['after'] = {p: sha(p) for p in value['before']}
    value['pass'] = value['before'] == value['after']
    assert value['pass'], 'frozen source changed'
    write(OUT/'frozen_check.json', value)
    return value

def decision(**updates):
    path = OUT/'decision_summary.json'
    result = read(path) if path.exists() else {}
    result.update(updates)
    write(path, result)
    return result

def commit():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
