"""开发证据通过后，只冻结一个候选并进行一次新96确认。"""
import argparse
import random
import numpy as np
import torch
from tools.e48_protocol import *
from tools.e48_holdout import ROOT,phash_matrix


def freeze():
    path=OUT/'new96/protocol.json'
    if path.exists():return read(path)
    for stage in ['probe','dev32']:assert read(OUT/stage/'evaluation_complete.json')['complete']
    assert read(OUT/'holdout_audit/complete.json')['complete']
    trained={a:read(OUT/'training'/a/'complete.json') for a in ['DECOUPLED','BOUNDARY']}
    assert all(r['updates']==800 and r['frozen_base_unchanged'] for r in trained.values())
    # 最终汇总反映真实训练记录，不能沿用 prepare() 的 not_run 初值。
    decision(training='complete',training_updates={a:r['updates'] for a,r in trained.items()})
    probe=read(OUT/'probe/gates.json');dev=read(OUT/'dev32/gates.json')
    possible=[a for a in REPAIRS+['DECOUPLED','BOUNDARY'] if a in dev and a in probe
        and dev[a]['passed'] and probe[a]['passed']]
    if not possible:
        decision(new96='not_run',final_decision='NO_GO',
            stop_reason='no candidate passes the frozen development and failure-regression gates')
        return None
    # 选择规则在读取新96图像前冻结：通过门槛者中开发LPIPS最低，名称只打破平局。
    means=read(OUT/'dev32/means.json');candidate=min(possible,key=lambda a:(means[a]['lpips_gt'],a))
    rows=sorted(read(OUT/'holdout_audit/eligible.json'),key=lambda r:r['id'])
    random.Random(48096).shuffle(rows)
    signatures=read(OUT/'holdout_audit/validation_signatures.json')
    exact={k:set() for k in ['file_sha256','pixel_sha256']};selected=[];paths=[];skipped=[]
    lookup=np.asarray([bin(i).count('1') for i in range(256)],np.uint8)
    for row in rows:
        reason=None
        for kind in ['gt','sketch','reference']:
            if any(signatures[row[kind]][k] in exact[k] for k in exact):reason='selected_exact_duplicate'
        if paths and reason is None:
            matrix=phash_matrix(paths,signatures)
            for kind in ['gt','reference']:
                probe=np.asarray(list(bytes.fromhex(signatures[row[kind]]['phash'])),np.uint8)
                if int(lookup[matrix^probe].sum(1).min())<=6:reason='selected_near_candidate'
        if reason:skipped.append(dict(id=row['id'],reason=reason));continue
        selected.append(row);paths.extend(row[k] for k in ['gt','reference'])
        for kind in ['gt','sketch','reference']:
            for k in exact:exact[k].add(signatures[row[kind]][k])
        if len(selected)==96:break
    assert len(selected)==96
    checkpoint=OUT/'training'/candidate/'checkpoints/step0800.pt' if candidate in ['DECOUPLED','BOUNDARY'] else B1
    write(OUT/'splits/new96.json',selected)
    write(OUT/'new96/within_split_exclusions.json',skipped)
    frozen={p:sha(p) for row in selected for p in [row[k] for k in ['gt','sketch','reference']]}
    frozen.update({str(ROOT/'text'/(r['id']+'.txt')):sha(ROOT/'text'/(r['id']+'.txt')) for r in selected})
    write(OUT/'new96/input_hashes.json',frozen)
    value=dict(candidate=candidate,arms=['B','D','SAFE',candidate],selection_seed=48096,
        selection='minimum dev32 LPIPS among candidates passing both probe16 and dev32 frozen gates',
        split_sha256=sha(OUT/'splits/new96.json'),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        generation=dict(seed=42,steps=50,cfg=7.,size=list(SIZE),scales=CONFIG['scales']),
        input_sha256=sha(OUT/'new96/input_hashes.json'),audit_sha256=sha(OUT/'holdout_audit/complete.json'),
        development_sha256={str(OUT/s/f):sha(OUT/s/f) for s in ['probe','dev32']
            for f in ['metrics.json','gates.json','evaluation_complete.json']},
        one_confirmation_only=True,thresholds=CONFIG['safety'],appearance=CONFIG['appearance'],
        strict_unseen_generalization=False,writer_pretraining_overlap=None,commit=commit())
    write(path,value);decision(new96='frozen',candidate=candidate)
    return value


def generate(arm):
    protocol=freeze();assert protocol is not None and arm in protocol['arms']
    assert sha(protocol['checkpoint'])==protocol['checkpoint_sha256']
    assert sha(OUT/'splits/new96.json')==protocol['split_sha256']
    assert all(sha(p)==expected for p,expected in read(OUT/'new96/input_hashes.json').items())
    from tools.e48_run import make_runtime,run_case,release
    if arm in ['DECOUPLED','BOUNDARY']:
        from tools.e48_train import make_runtime as trained_runtime
        r=trained_runtime(arm);r.load(protocol['checkpoint']);r.processor.eval().requires_grad_(False)
    else:r=make_runtime('new96/'+arm+'/runtime')
    try:
        for row in read(OUT/'splits/new96.json'):run_case(r,row,arm,'new96',compare=False)
    finally:release(r)
    write(OUT/'new96'/(arm+'_complete.json'),dict(arm=arm,n=96,protocol_sha256=sha(OUT/'new96/protocol.json'),commit=commit()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','generate','evaluate'])
    parser.add_argument('--arm');args=parser.parse_args();torch.set_num_threads(2)
    if args.action=='freeze':print(freeze(),flush=True)
    elif args.action=='generate':generate(args.arm)
    else:
        protocol=freeze();assert protocol is not None
        from tools.e48_eval import evaluate
        evaluate('new96',protocol['arms'])
