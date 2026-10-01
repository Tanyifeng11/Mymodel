"""E31：完整逐图审计、数值一致性检查与下载包；不改变实验参数。"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tools.e31_a_anchor_mining import ARMS, ABLATIONS, read, write, bundle, sha


def audit(out, review=None):
    full=out/'A_anchor_mining/B4_soft_unknown'
    rows=read(full/'rows.json')
    checks={}
    for arm in ARMS+ABLATIONS:
        folder=out/('A_anchor_mining' if arm in ARMS else 'audits/ablations')/arm
        data=read(folder/'rows.json')
        checks[arm+'/reference_indices']={r['index'] for r in data}==set(range(64,96)) and len(data)==32
        for r in data:
            with np.load(folder/('%03d_support.npz'%r['index'])) as d:
                p,mask=d['probabilities'],d['foreground']
                observed=float((p[...,:-1].max(-1)[mask]>=.5).mean()) if p.shape[-1]>1 else 0.
                checks[arm+'/%03d'%r['index']]=bool(abs(observed-r['foreground_coverage'])<1e-8
                    and np.max(abs(p[mask].sum(-1)-1))<1e-5
                    and (not (~mask).any() or p[~mask].max()==0)
                    and p.shape[-1]==r['k']+1)
    write(out/'audits/probability_integrity.json',dict(checks=checks,**{'pass':all(checks.values())}))
    # 八个 reference 一页，全体 32 张都有原图、裁剪和软场/unknown。
    for start in range(64,96,8):
        sheet=Image.new('RGB',(896,8*182),'white')
        draw=ImageDraw.Draw(sheet)
        for y,index in enumerate(range(start,start+8)):
            row=next(r for r in rows if r['index']==index)
            folder=full/'visualizations'/('%03d'%index)
            pics=[('reference',Image.open(folder/'reference.png')),('anchors',Image.open(folder/'topk.png'))]
            for name in ('anchor_0.png','anchor_0_support.png','unknown.png','coverage.png'):
                pics.append((name,Image.open(folder/name) if (folder/name).exists() else Image.new('RGB',(256,256),'white')))
            draw.text((3,y*182+3),'%03d K%d cov %.2f unk %.2f IoU %.2f'%(index,row['k'],row['foreground_coverage'],row['unknown_mass'],row['support_soft_iou']),fill='black')
            for x,(name,pic) in enumerate(pics):
                sheet.paste(pic.convert('RGB').resize((144,144)),(x*149,y*182+34))
                draw.text((x*149+3,y*182+20),name[:22],fill='black')
        sheet.save(out/'audits'/('contact_%03d_%03d.png'%(start,start+7)))
    augmentation={}
    for name in ('brightness','color_jitter','translation','resize','rot90','pattern_rot90'):
        records=[]
        for index in range(64,96):
            record=read(full/('%03d.json'%index))
            match=next(a for a in record['augmentations'] if a['variant']==name)
            records.append(dict(index=index,split='dev' if index<80 else 'stress_dev',**match))
        augmentation[name]=dict(rows=records,reference_count=32,
            mean_soft_iou=float(np.mean([r['soft_iou'] for r in records])),
            mean_fixed_anchor_soft_iou=float(np.mean([r['fixed_anchor_soft_iou'] for r in records])))
    probes=[r['pattern_probe'] for r in augmentation['pattern_rot90']['rows'] if 'pattern_probe' in r]
    readable=[p for p in probes if p['readable']]
    augmentation['pattern_probe_summary']=dict(available_references=len(probes),readable_references=len(readable),
        mean_learned_identity=float(np.mean([p['learned_identity'] for p in probes])) if probes else None,
        orientation_median_error=float(np.median([p['orientation_error'] for p in readable])) if readable else None,
        orientation_success_fraction=float(np.mean([p['orientation_error']<=15 for p in readable])) if readable else None,
        period_mean_log_error=float(np.mean([p['period_log_error'] for p in readable])) if readable else None,
        local_support_soft_iou=float(np.mean([p['local_support_soft_iou'] for p in probes])) if probes else None,
        definition='actual 90-degree rotation of an inscribed square in selected anchor; no panel GT; not a whole-garment pattern rotation claim')
    write(out/'audits/augmentation_summary.json',augmentation)
    if review:
        value=json.loads(review)
        assert set(value['reviewed_indices'])==set(range(64,96))
        write(out/'audits/visual_review.json',value)
    completion=read(out/'completion_check.json')
    reviewed=read(out/'audits/visual_review.json') if (out/'audits/visual_review.json').exists() else {}
    completion['visual_review_completed']=set(reviewed.get('reviewed_indices',[]))==set(range(64,96))
    completion['probability_integrity_pass']=all(checks.values())
    completion['experiment_complete']=bool(completion['numeric_artifacts_complete'] and completion['frozen_inputs_pass']
        and completion['probability_integrity_pass'] and completion['visual_review_completed']
        and not completion['stage_A_pass'])
    completion['report_git_commit']=__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip()
    write(out/'completion_check.json',completion)
    paths=[str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() and p.suffix not in ('.gz','.log','.err')]
    write(out/'artifact_manifest.json',dict(files=paths,reference_unit='reference image',bootstrap_resamples=2000,
        review_bundle='local_review_bundle.tar.gz'))
    args=argparse.Namespace(out=out)
    bundle(args)
    print(json.dumps(dict(decision=read(out/'decision_summary.json'),completion=completion),ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('output_eval/e31_anchor_correspondence_20261001'))
    parser.add_argument('--review-json')
    args=parser.parse_args()
    audit(args.out,args.review_json)
