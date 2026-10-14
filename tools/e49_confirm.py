"""开发门槛通过后，审计历史暴露并只进行一次新 96 身份确认。"""
import argparse
import gc
import random
import numpy as np
import torch
from tools.e49_protocol import *


def candidate():
    assert read(OUT/'audit/complete.json')['passed']
    for stage in ['probe','dev32']: assert read(OUT/stage/'evaluation_complete.json')['complete']
    possible=[a for a in TRAIN_ARMS if all(read(OUT/s/'gates.json')[a]['passed'] for s in ['probe','dev32'])]
    if not possible:
        decision(new96='not_run',final_decision='NO_GO',stop_reason='no candidate passes all frozen gates on both development sets')
        return None
    means=read(OUT/'dev32/means.json'); arm=min(possible,key=lambda a:(means[a]['lpips_gt'],a))
    checkpoint=OUT/'training'/arm/'checkpoints/step0800.pt'
    assert read(OUT/'training'/arm/'complete.json')['updates']==800
    value=dict(candidate=arm,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        selection='minimum dev32 LPIPS among all-gate passers; name breaks ties; fixed step800 only',
        development_sha256={str(OUT/s/f):sha(OUT/s/f) for s in ['probe','dev32']
            for f in ['metrics.json','gates.json','evaluation_complete.json']})
    path=OUT/'new96/candidate.json'
    if path.exists(): assert read(path)==value
    else: write(path,value)
    return value


def audit(workers=4):
    assert candidate() is not None
    if (OUT/'holdout_audit/complete.json').exists(): return
    from tools import e48_holdout as module
    previous=module.OUT; module.OUT=OUT
    try: module.audit(workers)
    finally: module.OUT=previous


def freeze():
    value=candidate()
    if value is None: return None
    path=OUT/'new96/protocol.json'
    if path.exists():
        frozen=read(path)
        assert frozen['candidate']==value['candidate'] and frozen['checkpoint_sha256']==value['checkpoint_sha256']
        return frozen
    assert read(OUT/'holdout_audit/complete.json')['complete']
    from tools.e48_holdout import phash_matrix, ROOT
    signatures=read(OUT/'holdout_audit/validation_signatures.json')
    rows=sorted(read(OUT/'holdout_audit/eligible.json'),key=lambda r:r['id'])
    random.Random(48096).shuffle(rows)
    exact={k:set() for k in ['file_sha256','pixel_sha256']}; selected=[]; paths=[]; skipped=[]
    lookup=np.asarray([bin(i).count('1') for i in range(256)],np.uint8)
    for row in rows:
        reason=None
        for kind in ['gt','sketch','reference']:
            if any(signatures[row[kind]][k] in exact[k] for k in exact): reason='selected_exact_duplicate'
        if paths and reason is None:
            matrix=phash_matrix(paths,signatures)
            for kind in ['gt','reference']:
                probe=np.asarray(list(bytes.fromhex(signatures[row[kind]]['phash'])),np.uint8)
                if int(lookup[matrix^probe].sum(1).min())<=6: reason='selected_near_candidate'
        if reason: skipped.append(dict(id=row['id'],reason=reason)); continue
        selected.append(row); paths.extend(row[k] for k in ['gt','reference'])
        for kind in ['gt','sketch','reference']:
            for k in exact: exact[k].add(signatures[row[kind]][k])
        if len(selected)==96: break
    assert len(selected)==96
    write(OUT/'splits/new96.json',selected); write(OUT/'new96/within_split_exclusions.json',skipped)
    frozen_inputs={row[k]:sha(row[k]) for row in selected for k in ['gt','sketch','reference']}
    frozen_inputs.update({str(ROOT/'text'/(r['id']+'.txt')):sha(ROOT/'text'/(r['id']+'.txt')) for r in selected})
    write(OUT/'new96/input_hashes.json',frozen_inputs)
    frozen=dict(value,arms=['B','D',value['candidate']],selection_seed=48096,
        split_sha256=sha(OUT/'splits/new96.json'),input_sha256=sha(OUT/'new96/input_hashes.json'),
        audit_sha256=sha(OUT/'holdout_audit/complete.json'),one_confirmation_only=True,
        thresholds=CONFIG['safety'],appearance=CONFIG['appearance'],strict_unseen_generalization=False,
        clip_pretraining_overlap=None,writer_pretraining_overlap=None,commit=commit())
    write(path,frozen); decision(new96='frozen',candidate=value['candidate']); return frozen


def generate(arm):
    protocol=freeze(); assert protocol is not None and arm in protocol['arms']
    assert sha(protocol['checkpoint'])==protocol['checkpoint_sha256']
    assert sha(OUT/'splits/new96.json')==protocol['split_sha256']
    assert all(sha(p)==h for p,h in read(OUT/'new96/input_hashes.json').items())
    if arm in TRAIN_ARMS:
        from tools.e49_train import generate as trained
        trained(arm,'new96'); return
    from tools import e47_runtime as module
    previous=module.OUT; module.OUT=OUT
    try: r=module.runtime('B',arm=='D','new96/'+arm+'/runtime')
    finally: module.OUT=previous
    if arm=='D': r.load(B1); r.processor.eval().requires_grad_(False)
    try:
        for row in read(OUT/'splits/new96.json'):
            path=image_path(arm,row['id'],'new96')
            if path.exists() and path.with_suffix('.json').exists():
                assert sha(path)==read(path.with_suffix('.json'))['png_sha256']; continue
            r.generate(row,path)
    finally:
        module.close(r); del r; gc.collect(); torch.cuda.empty_cache()
    write(OUT/'new96'/(arm+'_complete.json'),dict(arm=arm,n=96,protocol_sha256=sha(OUT/'new96/protocol.json'),commit=commit()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['candidate','audit','freeze','generate','evaluate'])
    parser.add_argument('--arm'); parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args(); torch.set_num_threads(2)
    if args.action=='candidate': print(candidate(),flush=True)
    elif args.action=='audit': audit(args.workers)
    elif args.action=='freeze': print(freeze(),flush=True)
    elif args.action=='generate': generate(args.arm)
    else:
        from tools.e49_eval import evaluate
        protocol=freeze(); assert protocol is not None
        evaluate('new96',protocol['arms'])
