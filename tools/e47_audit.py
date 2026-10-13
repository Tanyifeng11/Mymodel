"""P0 CPU 审计：逐张量比较、历史训练来源、精确重复及近重复候选。"""
import argparse
import csv
import gc
import hashlib
import itertools
import platform
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, __version__ as pillow_version
from tools.e47_protocol import *

BF = Path('/share/home/u2515283058/datasets/BF/training')


def tensors(value, prefix=''):
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            name = prefix+'.'+str(key) if prefix else str(key)
            if torch.is_tensor(item):
                result[name] = item
            elif isinstance(item, dict) and key not in ['meta', 'metadata']:
                result.update(tensors(item, name))
    return result


def tensor_sha(value):
    value = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(value.numpy()).hexdigest()


def max_delta(a, b):
    aa, bb = a.reshape(-1), b.reshape(-1)
    return max((float((aa[i:i+1000000].double()-bb[i:i+1000000].double()).abs().max())
        for i in range(0, aa.numel(), 1000000)), default=0.)


def checkpoint_audit():
    freeze_protocol()
    assert sha(E5) == E5_SHA and sha(B1) == B1_SHA, '历史 checkpoint 身份不符'
    write(OUT/'checkpoint_hashes.json', {str(p): dict(sha256=sha(p),
        resolved_path=str(p.resolve()), bytes=p.stat().st_size) for p in [E2B,E5,B1]})
    states = [torch.load(p, map_location='cpu', weights_only=False) for p in [E2B,E5]]
    a,b = [tensors(s) for s in states]
    rows=[]
    for i,key in enumerate(sorted(set(a)|set(b))):
        x,y=a.get(key),b.get(key)
        hx=tensor_sha(x) if x is not None else None
        hy=tensor_sha(y) if y is not None else None
        shared=x is not None and y is not None
        same_shape=shared and x.shape==y.shape
        equal=bool(same_shape and x.dtype==y.dtype and hx==hy)
        rows.append(dict(key=key, shared=shared, shape_e2b=list(x.shape) if x is not None else None,
            shape_e5=list(y.shape) if y is not None else None,
            dtype_e2b=str(x.dtype) if x is not None else None,
            dtype_e5=str(y.dtype) if y is not None else None,
            sha256_e2b=hx,sha256_e5=hy,identical=equal if shared else None,
            max_abs_delta=0. if equal else max_delta(x,y) if same_shape else None,
            historical_trainable=key.startswith('tcpm_lite.'),
            trainability_evidence='historical freeze flag and source; tensor comparison separately verifies values'))
        if i%500==0:print('e47 TENSOR',i,len(set(a)|set(b)),flush=True)
    with (OUT/'audit_trainables.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    changed=[r for r in rows if r['shared'] and not r['identical']]
    added=[r['key'] for r in rows if not r['shared'] and r['sha256_e5'] is not None]
    removed=[r['key'] for r in rows if not r['shared'] and r['sha256_e2b'] is not None]
    write(OUT/'checkpoint_comparison.json',dict(shared=sum(r['shared'] for r in rows),
        changed_common=changed,added=added,removed=removed,
        all_common_exact=not changed,new_keys_only_tcpm=all(k.startswith('tcpm_lite.') for k in added)))
    for label,state in zip(['e2b','e5'],states):write(OUT/'manifests'/(label+'_checkpoint_meta.json'),state.get('meta',{}))
    manifest_path=E5.parent.parent/'experiment_manifest.json'
    historical=read(manifest_path)
    write(OUT/'manifests/e5_experiment_manifest.json',historical)
    checkpoints=[]
    for path in sorted(E5.parent.parent.glob('checkpoint-*')):
        checkpoints.append(dict(path=str(path),resolved=str(path.resolve()),
            numeric_step=int(path.name.split('-')[-1]) if re.fullmatch(r'checkpoint-\d+',path.name) else None))
    code_commit=historical.get('git_commit')
    source_status='historical git commit unavailable'
    if code_commit:
        try:
            source=subprocess.check_output(['git','show',code_commit+':train_GAM_texture_joint.py'],text=True)
            (OUT/'manifests/e5_historical_training_source.py').write_text(source,encoding='utf-8')
            source_status='recovered exact archived git commit'
        except subprocess.CalledProcessError:source_status='archived git commit not available locally'
    tcpm=states[1].get('tcpm_lite',{})
    groups=[dict(lr=historical.get('tcpm_lr'),keys=[k for k in tcpm if k!='residual_scale']),
        dict(lr=historical.get('tcpm_scale_lr'),keys=['residual_scale'])]
    write(OUT/'historical_training_audit.json',dict(manifest_sha256=sha(manifest_path),
        manifest=historical,checkpoint_aliases=checkpoints,source_status=source_status,
        optimizer_groups_reconstructed=groups,optimizer_archive='not recovered; do not claim actual saved optimizer state',
        total_tcpm_parameters=sum(t.numel() for t in tcpm.values()),
        final_updates_from_checkpoint_alias=int(E5.parent.resolve().name.split('-')[-1])
            if re.fullmatch(r'checkpoint-\d+',E5.parent.resolve().name) else None,
        consumed_random_sequence=None,
        exact_frozen_shared_weights=not changed))
    decision(P0='cpu_audited_pending_gpu',baseline_provisional='E2b' if not changed else 'A_mirror_required',
        common_weight_changes=len(changed),causal_claim=False)
    del states,a,b;gc.collect()


def normalize(row):
    def path(key):
        p=Path(row[key]);return str(p if p.is_absolute() else BF/p)
    gt=next(k for k in ['gt','cloth','target'] if row.get(k))
    reference=next((k for k in ['reference','texture','color'] if row.get(k)),None)
    return dict(id=str(row.get('id',Path(row[gt]).stem)),gt=path(gt),
        sketch=path('sketch') if row.get('sketch') else None,
        reference=path(reference) if reference else None)


def image_signature(path):
    with Image.open(path) as im:
        rgb=im.convert('RGB');small=np.asarray(rgb.resize((32,32),Image.Resampling.BILINEAR))
        gray=np.asarray(rgb.convert('L').resize((32,32),Image.Resampling.LANCZOS),np.float32)
        dct=cv2.dct(gray)[:8,:8].copy();med=np.median(dct.reshape(-1)[1:])
        bits=(dct>med).reshape(-1);bits[0]=False
        return dict(path=path,file_sha256=sha(path),pixel_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            size=list(rgb.size),phash=np.packbits(bits).tobytes().hex(),thumbnail=small)


def split_audit(workers=8):
    sources=dict(historical_train=Path('data/train_bf_texture.json'),
        train1024=OLD/'splits/train1024.json',dev32=OLD/'splits/dev32.json',
        confirm96=OLD/'splits/confirm96.json',
        validation500=Path('output_eval/e34_sarr_20261010/splits/validation_original.json'))
    cohorts={name:[normalize(r) for r in read(path)] for name,path in sources.items()}
    for name in ['train1024','dev32','confirm96']:
        write(OUT/'splits'/(name+'.json'),read(sources[name]))
    write(OUT/'splits/train1024_records.json',read(OLD/'splits/train1024_records.json'))
    paths=sorted({row[k] for rows in cohorts.values() for row in rows for k in ['gt','sketch','reference'] if row[k]})
    cache={}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i,result in enumerate(pool.map(image_signature,paths)):
            cache[result['path']]=result
            if i%1000==0:print('e47 DATA HASH',i,len(paths),flush=True)
    intersections={}
    for x,y in itertools.combinations(cohorts,2):
        intersections[x+'__'+y]=dict(identity=sorted({r['id'] for r in cohorts[x]}&{r['id'] for r in cohorts[y]}),
            exact={kind:{key:sorted({cache[r[key]][kind] for r in cohorts[x] if r[key]}&
                {cache[r[key]][kind] for r in cohorts[y] if r[key]}) for key in ['gt','sketch','reference']}
                for kind in ['file_sha256','pixel_sha256']})
    # pHash 仅筛查候选；缩略图 RGB MAE 同时保留，不能把相似衣型认作同源。
    near=[];lookup=np.array([bin(i).count('1') for i in range(256)],np.uint8)
    for key in ['gt','reference']:
        train=sorted({r[key] for r in cohorts['historical_train'] if r[key]})
        matrix=np.array([list(bytes.fromhex(cache[p]['phash'])) for p in train],np.uint8)
        for cohort in ['dev32','confirm96','validation500']:
            for row in cohorts[cohort]:
                p=row[key]
                if p is None:continue
                probe=np.array(list(bytes.fromhex(cache[p]['phash'])),np.uint8)
                distance=lookup[matrix^probe].sum(1)
                for i in np.flatnonzero(distance<=6):
                    other=train[i]
                    if cache[p]['pixel_sha256']==cache[other]['pixel_sha256']:continue
                    mae=float(np.abs(cache[p]['thumbnail'].astype(np.float32)-cache[other]['thumbnail']).mean()/255)
                    near.append(dict(cohort=cohort,id=row['id'],kind=key,held_path=p,train_path=other,
                        phash_distance=int(distance[i]),thumbnail_rgb_mae=mae,
                        same_source_confirmed=None,interpretation='candidate only; no semantic/source labels'))
    write(OUT/'near_duplicate_candidates.json',near)
    write(OUT/'manifests/file_signatures.json',{p:{k:v for k,v in q.items() if k!='thumbnail'} for p,q in cache.items()})
    seen=intersections['historical_train__validation500']['identity']
    write(OUT/'split_audit.json',dict(sources={n:dict(path=str(p),sha256=sha(p),n=len(cohorts[n])) for n,p in sources.items()},
        pairwise=intersections,near_duplicate_candidate_count=len(near),
        near_duplicate_method='64bit grayscale pHash Hamming<=6 plus RGB thumbnail MAE; semantic/source match unconfirmed',
        historical_consumed_identity_trace=None,training_manifest_exposure='possible for any member; no exact consumed trace',
        writer_source='pretrained SD1.5 U-Net; original pretraining identities unavailable',
        writer_pretraining_overlap=None,validation_manifest_identity_overlap=len(seen),
        strict_unseen_generalization=False,
        independence='cannot certify unseen: consumed history and pretrained Writer provenance incomplete'))
    frozen={str(p):sha(p) for p in sources.values()}
    frozen.update({p:cache[p]['file_sha256'] for name in ['train1024','dev32','confirm96','validation500']
        for r in cohorts[name] for k in ['gt','sketch','reference'] for p in [r[k]] if p})
    write(OUT/'manifests/input_hashes.json',frozen)


def environment():
    import diffusers
    from garment_mask_utils import mask_backend_info
    info=dict(python=platform.python_version(),torch=torch.__version__,diffusers=diffusers.__version__,
        cuda=torch.version.cuda,opencv=cv2.__version__,pillow=pillow_version,
        mask=mask_backend_info(),size=list(SIZE),contour='RETR_EXTERNAL, tolerance3px DIST_L2 maskSize5',
        invalid='E44 sketch flood-fill + confidence + soft coverage >=0.03; retain IDs, regional NA',
        commit=commit(),code={str(p):sha(p) for p in [Path('garment_mask_utils.py'),
            Path('eval/eval_utils.py'),Path('tools/e44_metrics.py'),Path('tools/e44_runtime.py')]})
    assert info['mask']['mask_backend']=='opencv'
    write(OUT/'environment.json',info)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args();torch.set_num_threads(2)
    try:
        checkpoint_audit();environment();split_audit(args.workers)
        print('e47 CPU AUDIT COMPLETE',read(OUT/'decision.json'),flush=True)
    except Exception as exc:
        decision(P0='engineering_failure',error=repr(exc));raise


if __name__=='__main__':main()
