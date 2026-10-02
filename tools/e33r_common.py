"""E33-R固定协议、冻结候选与局部支持；不重新选择patch。"""
import hashlib
import json
from pathlib import Path
import numpy as np
from tools.e32_common import read,write,sha,bootstrap,git_commit,DATASET,WEIGHTS,finish_frozen,frozen_manifest
from tools.e33_protocol import GROUPS

OUT=Path('output_eval/e33r_rotation_causality_20261002')
E33=Path('output_eval/e33_counterfactual_reference_20261002')
E32=Path('output_eval/e32_target_supervised_pattern_field_20261001')
PROTOCOL=dict(experiment='E33-R',version=1,seeds=[42,43,44],
    selection='exact E33 selected-mode top1; no crop selector and no integrity-based reselection',
    expected_available=[25272,128,4,5],expected_strict=[8158,45,2,4],
    support='original bbox footprint at64x48 intersect E32 eroded interior>=.95 and orientation confidence>=.25',
    empty_support='retain identities; no geometric loss, success=False and absolute error unavailable; report denominator',
    p0=dict(train='selected available train25272',steps=6000,batch=8,warmup=400,seed=42,
            lr=1e-4,weight_decay=1e-4,gate='all256 dev predictions finite; readable case count equal E32; case mean orientation error<=15deg'),
    sanity=dict(train=512,dev=64,steps=500,seed=42,selection='fixed SHA256 sort, no outcome filtering',
                gate='clean train R90/R180/R0 success each>=.95'),
    p1=dict(steps=8000,warmup=500,effective_target_batch=8,microbatch=2,lr=1e-4,weight_decay=1e-4,
            mixed_precision='bf16',loss_weights=dict(field=1,q=.5,pair=.5,zero=.25,nuisance=.25,confidence=.1)),
    r180_integrity='clean train/dev>=.95 and both-noisy train/dev>=.90; confidence before/after>=.25, axial error<=15deg; if fail remove R180 losses only',
    q_targets=dict(identity=[1,0],rot90=[-1,0],rot180=[1,0],zero=[1,0]),
    initialization='Q head small random weights std.001, identity bias; avoid antipodal cosine stationary initialization',
    reference='448 square DINO-S/14 last4 mean to16x16,384dims; native-patch E26 orientation2+confidence1, Lab mean/std6, E32 detail/selfsim1; total394; no period channel',
    nuisance='E33 same paired parameters across R0/R90/R180; resample each training access; JPEG90..98,brightness/saturation.97..1.03,blur<=.25,translation<=.25px',
    evaluation_noise='two original E33 audit SHA256 seeds; no selecting successful noise',
    sketch_jitter='fixed hash +-1px translation plus1% internal ink dropout; recompute sketch mask/coords/distance; retain dropout only if mask components/holes preserved',
    metrics='confidence-weighted cell error then equal case average; response<=15deg; R0 identity relative to prior; prior preservation compared on same M_cf',
    zero_ratio='mean_case mean_support(1-Qzero.x) / mean_case mean_support(1-Q90.x)',
    sensitivity='mean_case axial_distance(R0,R90) / mean_case axial_distance(R0,jitter); paired case bootstrap2000',
    gate=dict(clean_rot90=.90,noisy_rot90=.85,r180=.90,r0=.90,zero_ratio=.10,sensitivity=2,prior_delta_deg=1),
    ablations=['A_no_prior_freeze','B_additive','C_no_pair','D_no_r180','E_no_nuisance','F_dino_only','G_geometry_only'],
    ablation_trigger='full seed42 passes or exactly1 failed Gate and relative threshold deviation<=10%; main Gate unchanged',
    forbidden=['scale interventions','period causal loss','real curriculum','identity/scaffold renderer','E5 generation','new data','new manual annotations'],
    limitations=['fixed Qidentity and absolute field GT are competing objectives when prior is imperfect',
                 'existing E33 candidates may contain folds/seams/background gaps; orientation control is not motif correspondence'])

def protocol_hash():return hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()

def hash_order(rows,tag):
    return sorted(rows,key=lambda r:hashlib.sha256((tag+'/'+r['id']).encode()).hexdigest())

def cf_support(gt,interior,box):
    x,y,x1,y1=box
    assert all(v%8==0 for v in box)
    footprint=np.zeros((64,48),bool);footprint[y//8:y1//8,x//8:x1//8]=True
    return footprint&(interior>=.95)&(gt[3]>=.25)

def prepare_manifest():
    OUT.mkdir(parents=True,exist_ok=True)
    assert read(E33/'completion_check.json')['experiment_complete']
    if (OUT/'protocol.json').exists():
        assert read(OUT/'protocol.json')['sha256']==protocol_hash()
        assert read(OUT/'input_provenance.json')['E33_split_sha256']==sha(E33/'split_manifest.json')
        return read(OUT/'selected_manifest.json')
    split=read(E33/'split_manifest.json');mode=read(E33/'transform_integrity/summary.json')['selected_mode']
    chosen={};strict=[];digest=hashlib.sha256()
    for group in GROUPS:
        rows=[]
        for row in split[group]:
            path=E33/'transform_integrity/cases'/(row['id']+'.json');source=read(path)
            assert source['id']==row['id'] and source['group']==group
            digest.update((row['id']+sha(path)).encode())
            candidate=source[mode]
            if candidate['controlled_available']:
                rows.append(dict(row,box=candidate['box'],strict=source['strict']['controlled_available'],
                    native_geometry=candidate['native_geometry'],E33_case_sha256=sha(path)))
        chosen[group]=rows;strict.append(sum(r['strict'] for r in rows))
    assert [len(chosen[g]) for g in GROUPS]==PROTOCOL['expected_available']
    assert strict==PROTOCOL['expected_strict']
    write(OUT/'split_manifest.json',split);write(OUT/'selected_manifest.json',chosen)
    write(OUT/'sanity_manifest.json',dict(train=hash_order(chosen['train'],'E33R/sanity/train')[:512],
         dev=hash_order(chosen['dev'],'E33R/sanity/dev')[:64]))
    write(OUT/'protocol.json',dict(protocol=PROTOCOL,sha256=protocol_hash(),git_commit=git_commit()))
    write(OUT/'input_provenance.json',dict(E33_split_sha256=sha(E33/'split_manifest.json'),
         E33_summary_sha256=sha(E33/'transform_integrity/summary.json'),E33_cases_combined_sha256=digest.hexdigest(),
         selected_manifest_sha256=sha(OUT/'selected_manifest.json'),selected_mode=mode,
         expected_available=PROTOCOL['expected_available'],strict_counts=strict))
    frozen_manifest(OUT,WEIGHTS)
    write(OUT/'decision_summary.json',dict(r180_integrity_pass=None,prior_pass=None,sanity_pass=None,
         seed42_pass=None,seed43_pass=None,seed44_pass=None,controlled_rotation_causality_pass=None,next_route='R180_integrity'))
    return chosen

def update_decision(**values):
    v=read(OUT/'decision_summary.json');v.update(values);write(OUT/'decision_summary.json',v)

def audit_seed(sid,j=0):return (int(hashlib.sha256(('E33/audit/'+sid).encode()).hexdigest()[:8],16)+j)%2**32
