"""E39 固定输入、配对和审计；所有服务器写入仅发生在本轮 OUT。"""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path

OUT = Path('output_eval/e39_reference_trajectory_20261011')
E5 = Path('output/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt')
EXPECTED_E5 = 'aea56b2d7b02492f0e58fc2abc45b6a5a996b9fd962f31e30aa26ff5b0ed28f0'
SIZE = (384, 512)
ARMS = ['Rplus', 'Rminus', 'Rzero', 'R90', 'Rcolor']
TIMES = [999, 800, 600, 400, 200, 50, 0]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def pattern(text):
    for name, expr in [('stripe', r'strip(?:e|es|ed)'), ('plaid', r'plaid|checkered|checked|gingham|tartan'),
                       ('dots', r'polka|dotted|dots'), ('floral', r'floral|flowers?'),
                       ('graphic', r'graphic|logo|text|skull'), ('denim', r'denim|jeans')]:
        if re.search(r'\b(?:'+expr+r')\b', text, re.I): return name
    return 'unspecified'


def category(text):
    for name, expr in [('dress', r'dress'), ('bottom', r'pants|trousers|jeans|shorts|skirt'),
                       ('outer', r'jacket|coat|blazer|cardigan')]:
        if re.search(r'\b(?:'+expr+r')\b', text, re.I): return name
    return 'top'


def prepare():
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw
    if (OUT/'protocol.json').exists():
        assert sha(E5) == EXPECTED_E5
        return
    assert sha(E5) == EXPECTED_E5
    rows = read('output_eval/e36_dagf_guided_filter_20261010/splits/dev32.json')
    pool = read('output_eval/e36_dagf_guided_filter_20261010/splits/train1024.json')
    assert len(rows) == 32
    def appearance(r):
        a = np.asarray(Image.open(r['reference']).convert('RGB').resize((64,64)), np.float32)/255
        lab = cv2.cvtColor(a, cv2.COLOR_RGB2LAB).reshape(-1,3)
        return lab.mean(0), np.quantile(lab, [.1,.5,.9], axis=0).reshape(-1)
    stats = {r['id']: appearance(r) for r in pool+rows}
    hashes = {r['id']: sha(r['reference']) for r in pool+rows}
    pairs = []
    for i, r in enumerate(rows):
        candidates = [q for q in pool if category(q['caption']) == category(r['caption'])
                      and hashes[q['id']] != hashes[r['id']]]
        wrong = random.Random(39011+i).choice(sorted(candidates, key=lambda q:q['id']))
        other = [q for q in candidates if pattern(q['caption']) != pattern(r['caption'])
                 and pattern(q['caption']) != 'unspecified' and pattern(r['caption']) != 'unspecified']
        # 颜色和标签选配独立于模型输出；找不到合格配对保留 NA，不能把近邻冒充同色异纹。
        selected = min(other, key=lambda q:float(np.linalg.norm(stats[q['id']][1]-stats[r['id']][1]))) if other else None
        delta = float(np.linalg.norm(stats[selected['id']][0]-stats[r['id']][0])) if selected else None
        valid = selected is not None and delta <= 5.
        pair = dict(**r, category=category(r['caption']), pattern=pattern(r['caption']),
                    wrong=wrong, color=selected if valid else None, color_candidate=selected,
                    color_lab_delta=delta, color_valid=valid,
                    color_status='candidate_requires_visual_review' if valid else 'NA_no_color_matched_different_label',
                    labels='caption-derived weak labels; not human pattern ground truth')
        pairs.append(pair)
        folder=OUT/'inputs'/r['id'];folder.mkdir(parents=True,exist_ok=True)
        views=[('R+',r['reference']),('R-',wrong['reference']),('Rcolor candidate',selected['reference'] if selected else None)]
        canvas=Image.new('RGB',(4*160,240),'white');draw=ImageDraw.Draw(canvas)
        for j,(label,path) in enumerate(views):
            if path:canvas.paste(Image.open(path).convert('RGB').resize((160,208)),(j*160,32))
            draw.text((j*160+3,3),label,fill='black')
        draw.text((3*160+3,3),r['id'],fill='black')
        draw.text((3*160+3,26),'LAB %.2f'%delta if delta is not None else 'no donor',fill='black')
        canvas.save(folder/'pair_review.png')
        for key in ['sketch','reference']:
            Image.open(r[key]).convert('RGB').save(folder/(key+'.png'))
    frozen={str(E5):sha(E5)}
    for r in pairs:
        for q in [r,r['wrong'],r['color_candidate']]:
            if q:
                for key in ['reference','sketch']:
                    frozen[q[key]]=sha(q[key])
    config=dict(experiment='E39', seed=42, pairing_seed=39011, steps=50,cfg=7.,size=SIZE,
        training_updates=0, checkpoint_sha256=EXPECTED_E5, arms=ARMS, exact_probe_timesteps=TIMES,
        cohort='frozen E36 dev32; exploratory reused identities',
        zero='CLIP pixel_values, pooled/patch embeddings and normalized CNN input all zero; learned biases retained',
        swap='Rminus at every original DDIM timestep on Rplus latent; all arms at seven exact t on nearest Rplus latent',
        exact_probe='off-grid t uses nearest observed Rplus latent without changing DDIM50 scheduler; diagnostic only',
        attention='text and texture separate softmax; report distributions and post-gate residual norm fraction, not joint probability',
        color_matching='same garment category, different explicit caption pattern label, mean LAB delta<=5; visual review required',
        linear_probe='frozen representations only; caption weak labels grouped by reference identity; no generation weights updated',
        thresholds='no arbitrary bottleneck cutoff; paired effects, representation readouts and semantic success reported with uncertainty')
    write(OUT/'protocol.json',config);write(OUT/'pairs.json',pairs);write(OUT/'audit/input_hashes.json',frozen)
    print('PREPARED',len(pairs),'Rcolor valid candidates',sum(r['color_valid'] for r in pairs),flush=True)


if __name__=='__main__': prepare()
