"""T3 独立参考响应确认：已批准的参数在打开 LOCK 前冻结。"""
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
PROJECT = Path(os.environ.get('E2A_PROJECT', '/share/home/u2515283058/Mymodel'))
ROOT = PROJECT / 'output_eval/t3_reference_response_20261027'
SUP = PROJECT / 'output_eval/e2a_sup_comp_01_20261021'
ATTR = PROJECT / 'output_eval/e2a_attr_01_20261019'
MATCHED = PROJECT / 'output_eval/e2a_3d2d_tex_01_20261026'
MODELS = {
    'E1': (PROJECT / 'output/phase1_e1_grouped_e5/checkpoint-final/joint_model.pt',
           'e51d71dbf17c424fb639293ab99cfa83b9658451e127e6a61e8263e15bd5e39a'),
    'C': (ATTR / 'training/seed42/F111/checkpoint-final/joint_model.pt',
          '517be044acd7ca149c2b09c20d7b8971680e24516ad4f357927c6bb5d72ed7b6'),
    'T3': (SUP / 'training/formal/T3/checkpoint-final/joint_model.pt',
           '25f95844696ba5c07eb9223e3bdd4d68c78ba4f1a36d561569c35cc90186d259'),
}
CONFIG = dict(
    protocol='T3-REFERENCE-RESPONSE-20261027', approved_by_user='2026-10-27 开始实施',
    source_plan_sha256='342f8a65e7dd827df7d80222a1521041ee6ff3ef8092501ea57e67303e43d872',
    primary='mean identity-paired T3 minus C DISTS Margin on swapped output',
    margin='D(same swapped output, old reference) - D(same swapped output, new reference)',
    meaningful_difference=.005, alpha=.05, planned_n=128, minimum_valid_fraction=.8,
    primary_family=['T3 minus C DISTS Margin'],
    primary_test='two-sided paired mean sign-flip, plus-one Monte Carlo p; singleton family',
    bootstrap=10000, signflip=10000, statistics_seed=271042,
    CI='identity-paired percentile bootstrap 95%', missing='pairwise complete; no imputation or replacement; retain all failures',
    structure_margins=dict(iou=.010, edge_f1=.020, leak=.005),
    leak_definition='legacy saturation OR RGB deviation from old GT on fixed sketch-defined exterior; background proxy only, not counterfactual garment quality or pure geometry',
    structure_gate='for original AND swapped outputs, all three oriented paired CI lower bounds exceed negative margin; Bonferroni simultaneous one-sided 95% over six endpoints',
    secondary=['color Margin'], descriptive=['E1', 'old/new DISTS', 'edge_l1'],
    old_reference_GT_metrics='not used for counterfactual quality; no swapped LPIPS/FID',
    inference=dict(seed_base=42, steps=50, cfg=7., sketch_scale=.6, texture_scale=1.,
                   resolution=[384,512], sampler='DDIM', explicit_paired_latent=True),
    patch=dict(size=64,stride=32,erosion_kernel=11,occupancy_min=.9,minimum=4,maximum=8,reference_grid=4),
    dists=dict(source_commit='1267d8cb626c98706db3697422701c56a85ebf2e',
       source_sha256='69b63ebd4e1aece0f79a3b922ba803b5f6fec127babee94869e9eceaa42b4612',
       weights_sha256='f5e65c96230b7f6ca995691647d482237e4cab8a50c5c4a5784f219ef0748218',
       backbone_sha256='397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0',
       aggregation='0.5*(mean generated-to-nearest-reference + mean reference-to-nearest-generated)',
       RGB=True, range=[0,1], reference_resize='Pillow LANCZOS 384x512',generated_resize='none',
       reference_decoder='OpenCV IMREAD_COLOR, BGR2RGB; exact frozen Pillow12.1 reference pixels verified on all96 development references',
       numerical_precision='float32; CUDA matmul and cuDNN TF32 disabled; three historical case checks <=2e-6',
       caveat='derived patch-set perceptual response, not validated motif identity accuracy'),
    qualitative_selection='24 stable input-hash identities before generation; plus deterministic best/worst six by T3-C after analysis, clearly labelled',
    continuation='only if primary p<.05, CI lower>0, mean>=.005, valid>=80%, six structural safety gates pass',
    later_training=dict(arms=['T3-no-body','T3-no-rest','C-seed43','T3-seed43','C-seed44','T3-seed44'],
       updates=28365,effective_batch=8,initialization='same frozen E1',body=.1,rest=.03816244801608925),
    human_blind_scores=None, human_reviewers=0,
    human_policy='prepare anonymous material; actual ratings require human participants; AI review reported separately',
    pretraining_exposure='unknown', stop='no tuning/retesting on this confirmation set; report negative result and stop new training if gate fails',
)


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p, value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(p)


def freeze(p, value):
    p=Path(p)
    if p.exists(): assert read(p)==value, ('FROZEN_CHANGED',str(p))
    else: write(p,value)
    return value


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def stable(value):
    return hashlib.sha256(('T3_RESPONSE_271042/'+str(value)).encode()).hexdigest()


def commit():
    return subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()


def csv_write(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def patches(mask):
    import cv2
    import numpy as np
    interior=cv2.erode(mask.astype('uint8'),np.ones((11,11),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    coords=[(x,y) for y in range(0,449,32) for x in range(0,321,32)
            if interior[y:y+64,x:x+64].mean()>=.9]
    n=len(coords)
    if n>8:coords=[coords[i] for i in np.linspace(0,n-1,8).round().astype(int)]
    return coords,n


def legacy(stage,arm):
    base=SUP if arm in ['T1','T2'] else MATCHED
    return base/'evaluation'/stage/arm
