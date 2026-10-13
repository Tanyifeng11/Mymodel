"""仅在 P2 值得继续时，冻结 D-safe 做一次 confirm96 风险确认。"""
import argparse
import numpy as np
import torch
from PIL import Image
from tools.e44_runtime import sketch_masks
from tools.e44_metrics import paired
from tools.e47_protocol import *
from tools.e47_runtime import runtime,close,exact_trace
from tools.e47_eval import KEYS,image_path,panels

ARMS=['B','D','D_safe']


def freeze():
    assert read(OUT/'decision.json')['continue_to_confirmation']
    rows=read(OUT/'splits/confirm96.json')
    protocol=dict(experiment='e47',stage='P3 confirm96',arms=ARMS,seed=42,size=list(SIZE),
        ddim_steps=50,cfg=7.,scales=dict(sketch=.6,texture=1.,ipa=1.),
        D_safe_scale=SAFE_SCALE,checkpoint_sha256=dict(E5=E5_SHA,B1=B1_SHA),
        split_sha256=sha(OUT/'splits/confirm96.json'),ids=[r['id'] for r in rows],
        thresholds=CONFIG['safety'],one_confirmation_only=True,new_training=0,
        label='development-domain supplemental risk confirmation; not independent unseen generalization',
        selection='none; fixed weights, fixed lambda; stop after safety failure',commit=commit())
    path=OUT/'p3/protocol.json'
    if path.exists():assert read(path)==protocol
    else:write(path,protocol)
    return rows


def generate(arm):
    rows=freeze();r=runtime('B',arm!='B','p3/'+arm+'_runtime')
    if arm!='B':r.load(B1);r.processor.eval().requires_grad_(False)
    from models.e46_sarg import boundary_band,ResidualGate
    def sample(row,path):
        ctl=None
        if arm=='D_safe':
            raw,_,_=sketch_masks(Image.open(row['sketch']).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR),'cpu')
            ctl=ResidualGate(r,'G3',boundary_band(raw),constant=SAFE_SCALE);ctl.install()
        try:result=r.generate(row,path)
        finally:
            if ctl:ctl.close()
        if ctl:
            assert len(ctl.steps)==ctl.unconditional_calls==50
            write(path.with_name(path.stem+'_scale.json'),dict(mode='G3',constant=SAFE_SCALE,
                steps=ctl.steps,unconditional_calls=ctl.unconditional_calls))
        return result
    try:
        if arm=='D_safe':
            probe=next(row for row in read(OUT/'splits/dev32.json') if sketch_masks(
                Image.open(row['sketch']).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR),'cpu')[2]['valid'])
            path=OUT/'p3/safe_replay'/(probe['id']+'.png')
            sample(probe,path);old=E46/'p1/images/G3'/path.name
            trace=exact_trace(path.with_suffix('.pt'),old.with_suffix('.pt'))
            assert sha(path)==sha(old) and trace['bit_exact']
            write(OUT/'p3/safe_replay/check.json',dict(png_exact=True,**trace))
        for i,row in enumerate(rows):
            path=image_path(arm,row['id'],'p3')
            if path.exists():assert read(path.with_suffix('.json'))['png_sha256']==sha(path)
            else:sample(row,path)
            print('e47 CONFIRM',arm,i+1,96,flush=True)
    finally:close(r)
    write(OUT/'p3'/(arm+'_complete.json'),dict(n=96,seed=42,protocol_sha256=sha(OUT/'p3/protocol.json')))


def assess_confirm(records):
    means={a:{k:float(np.mean([r[k] for r in records if r['variant']==a and r[k] is not None])) for k in KEYS} for a in ARMS}
    counts={a:{k:sum(r['variant']==a and r[k] is not None for r in records) for k in KEYS} for a in ARMS}
    assert all(counts[a]['lpips_gt']==96 for a in ARMS)
    base={r['id']:r for r in records if r['variant']=='B'};gates={};deltas={}
    for arm in ['D','D_safe']:
        delta={k:paired(records,arm,'B',k) for k in KEYS};deltas[arm]=delta
        tail=[dict(id=r['id'],delta=r['contour_f1']-base[r['id']]['contour_f1']) for r in records
            if r['variant']==arm and r['contour_f1'] is not None and r['contour_f1']-base[r['id']]['contour_f1']<-.15]
        checks=dict(f1=delta['contour_f1']['ci95'][0]>=-.020,iou=delta['iou']['ci95'][0]>=-.010,
            leak=delta['leak']['ci95'][1]<=.005,tail=not tail,lpips=delta['lpips_gt']['mean']<0)
        gates[arm]=dict(checks=checks,passed=all(checks.values()),extreme_events=tail)
        lookup={r['id']:r for r in read(OUT/'splits/confirm96.json')}
        worst=sorted([r for r in records if r['variant']==arm and r['contour_f1'] is not None],
            key=lambda r:r['contour_f1']-base[r['id']]['contour_f1'])[:10]
        write(OUT/'p3/worst_cases'/(arm+'.json'),worst)
        panels([lookup[r['id']] for r in worst],['B',arm],OUT/'p3/worst_cases'/arm,stage='p3')
    write(OUT/'p3/means.json',means);write(OUT/'p3/valid_counts.json',counts)
    write(OUT/'p3/paired_deltas.json',deltas);write(OUT/'p3/gates.json',gates)
    passed=gates['D_safe']['passed']
    decision(P3='development_confirmation_pass' if passed else 'NO_GO',
        final_decision='NO_INDEPENDENT_HOLDOUT' if passed else 'NO_GO',
        stop_reason='开发域确认通过，但原 validation500 无独立性证明；需审计真正 holdout' if passed else '固定 D-safe 在 confirm96 仍未通过预设结构/外观门槛',
        validation500='not_run',formal_multiseed='not_run',strict_unseen_generalization=False,paper_claim_supported=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','generate','evaluate'])
    parser.add_argument('--arm',choices=ARMS,default='D_safe');args=parser.parse_args();torch.set_num_threads(2)
    if args.action=='freeze':freeze()
    elif args.action=='generate':generate(args.arm)
    else:
        freeze()
        assert all(read(OUT/'p3'/(a+'_complete.json'))['n']==96 for a in ARMS)
        for row in read(OUT/'splits/confirm96.json'):
            metadata=[read(image_path(a,row['id'],'p3').with_suffix('.json')) for a in ARMS]
            assert len({m['initial_sha256'] for m in metadata})==1
            assert all(m['seed']==42 and not m['gt_used_in_inference'] for m in metadata)
        from tools.e47_eval import evaluate
        evaluate('p3')


if __name__=='__main__':main()
