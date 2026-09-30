"""E29：固定 E28 推理协议，拆分 source panel 与 canonical crop 提议。"""

import argparse
import json
import math
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from models.panel_correspondence import build_panel_scaffold, source_panels, semantic_match
from tools.e20_utilization import case_stat
from tools.e27_correspondence import descriptor, file_sha, similarity
from tools.e27_experiment import stats, write
from tools.e28_diagnostics import recovery_stat
from tools.e28_experiment import CASE_IDS, SEEDS, VARIANTS, _boundary_f1, generate as generate_e28


ARMS = ('P0_manual_panel_manual_crop', 'P1_auto_panel_manual_crop',
        'P2_manual_panel_auto_crop', 'P3_auto_panel_auto_crop')
ANCHORS = {ARMS[0]: 'B1_oracle', ARMS[3]: 'B4_auto_panel'}


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def audit(root, e28, out):
    assert load(e28/'A_audit/report.json')['pass']
    assert load(e28/'completion_check.json')['pass']
    assert load(e28/'B_decomposition/component_check.json')['pass']
    cases = load(e28/'cases.json')
    assert tuple(r['id'] for r in cases['references']) == CASE_IDS
    assert set(cases['references'][0]['variants'][i]['variant'] for i in range(2)) == set(VARIANTS)
    for name in ('A_audit', 'expected', 'inputs'):
        shutil.copytree(e28/name, out/name, dirs_exist_ok=True)
    for name in ('cases.json',): shutil.copy2(e28/name, out/name)
    for arm, old in ANCHORS.items():
        src, dst = e28/'B_decomposition'/old, out/'B_causal'/arm
        shutil.copytree(src, dst, dirs_exist_ok=True)
        for path in dst.glob('c*_s*_*.json'):
            row = load(path)
            row['reused_from_e28'] = old
            row['arm'] = arm
            write(path, row)
    protocol = load(e28/'protocol.json')
    protocol.update(experiment='E29', source_experiment=str(e28), arms=ARMS,
                    panel_proposal='E28 analytic infer_regions', crop_selector='E28 select_crop',
                    assignment='E28 manual semantic/IoU label transfer',
                    warp='E28 oracle_local_uv', composition='E28 fixed_safe_oracle',
                    E5_training_steps=0, anchors_reused=ANCHORS)
    write(out/'protocol.json', protocol)
    write(out/'A_audit/report.json', {'pass': True, 'cases': list(CASE_IDS),
          'source_e28_audit_pass': True, 'source_e28_completion_pass': True,
          'manual_crop_available': True, 'target_fixed': True,
          'anchor_reuse': ANCHORS, 'training_steps': 0})
    print('[E29 A] audit complete', flush=True)


def scaffold(root, out):
    assert load(out/'A_audit/report.json')['pass']
    cases = load(out/'cases.json')
    groups = load(root/'data/e28_panel_cases.json')
    target_mask = np.asarray(Image.open(out/cases['sketches'][0]['mask']).convert('L')) > 0
    checks = []
    for case in cases['references']:
        cid = case['id']
        source_mask = np.asarray(Image.open(out/'A_audit/inputs'/f'c{cid:02d}_mask.png')) > 0
        for variant in VARIANTS:
            prefix = f'c{cid:02d}_{variant}'
            original = Image.open(out/'A_audit/inputs'/f'c{cid:02d}_original.png').convert('RGB')
            reference = Image.open(out/'A_audit/inputs'/f'{prefix}.png').convert('RGB')
            for anchor, e28_arm in ANCHORS.items():
                image, *_ = build_panel_scaffold(reference, target_mask, case['panels'],
                          source_mask, groups[str(cid)], e28_arm, original, variant)
                reused = Image.open(out/'B_causal'/anchor/f'{prefix}_scaffold.png').convert('RGB')
                error = np.abs(np.asarray(image).astype(np.int16)-np.asarray(reused).astype(np.int16))
                assert int(error.max()) == 0, (anchor, prefix, int(error.max()))
                checks.append({'arm': anchor, 'case': cid, 'variant': variant,
                               'max_rgb_error': int(error.max())})
            for arm in ARMS[1:3]:
                folder = out/'B_causal'/arm
                folder.mkdir(parents=True, exist_ok=True)
                path = folder/f'{prefix}_scaffold.png'
                image, result, sources, targets, matches, matrix, warps = build_panel_scaffold(
                    reference, target_mask, case['panels'], source_mask,
                    groups[str(cid)], arm, original, variant)
                image.save(path)
                details = {'case_id': cid, 'variant': variant, 'arm': arm,
                           'assignments': {m.target_panel_id: m.source_panel_id for m in matches},
                           'assignment_scores': matrix,
                           'source_panels': [p.panel_id for p in sources],
                           'source_crop_boxes': {p.panel_id: p.crop for p in sources},
                           'target_panels': [p.panel_id for p in targets],
                           'scaffold_sha256': file_sha(path),
                           'fixed_components': ['manual semantic assignment', 'oracle_local_uv',
                                                'fixed_safe_oracle', 'frozen E5']}
                write(path.with_suffix('.json'), details)
                np.savez_compressed(folder/f'{prefix}_fields.npz',
                    ownership_map=result.ownership_map,
                    confidence_map=result.confidence_map,
                    seam_map=result.seam_map,
                    source_index_map=result.source_index_map,
                    source_panel_ids=np.array([w.source_panel_id for w in warps]),
                    target_panel_ids=np.array([w.target_panel_id for w in warps]),
                    uv=np.stack([w.uv_field for w in warps]),
                    valid=np.stack([w.valid_mask for w in warps]))
                print('[E29 scaffold]', arm, prefix, flush=True)
    write(out/'B_causal/component_check.json', {
        'pass': True, 'anchor_scaffold_equivalence': checks,
        'P1_only_changed': 'source/target panel masks, panel coordinate support and derived semantic matching; manual crop copied unchanged after matching',
        'P2_only_changed': 'crop box/content selected with E28 select_crop inside unchanged manual source panel mask',
        'P3_equivalent_to': 'E28 B4_auto_panel',
        'P0_equivalent_to': 'E28 B1_oracle'})
    print('[E29 B] scaffolds complete', flush=True)


def generate(root, out):
    assert load(out/'B_causal/component_check.json')['pass']
    generate_e28(root, out, ARMS[1:3], 'B_causal')


def crop_metrics(out):
    cases = load(out/'cases.json')
    groups = load(Path(__file__).resolve().parents[1]/'data/e28_panel_cases.json')
    rows = []
    for case in cases['references']:
        cid = case['id']
        for variant in VARIANTS:
            prefix = f'c{cid:02d}_{variant}'
            reference = Image.open(out/'A_audit/inputs'/f'{prefix}.png').convert('RGB')
            source_mask = np.asarray(Image.open(out/'A_audit/inputs'/f'c{cid:02d}_mask.png')) > 0
            manual = source_panels(reference, case['panels'], 'manual', source_mask, groups[str(cid)])
            manual_by_name = {p.panel_id: p for p in manual}
            for arm in (ARMS[2], ARMS[3]):
                details = load(out/'B_causal'/arm/f'{prefix}_scaffold.json')
                if arm == ARMS[2]:
                    assignments = {n: n for n in manual_by_name}
                else:
                    auto = source_panels(reference, case['panels'], 'automatic', source_mask, {})
                    assignments = semantic_match(auto, manual)
                crops = details.get('source_crop_boxes')
                if crops is None:
                    auto = source_panels(reference, case['panels'], 'automatic', source_mask, {})
                    crops = {p.panel_id: p.crop for p in auto}
                for name, man in manual_by_name.items():
                    selected = [box for src, box in crops.items() if assignments[src] == name]
                    if not selected:
                        rows.append({'case': cid, 'variant': variant, 'arm': arm, 'panel': name,
                                     'missing': True, 'crop_iou': 0., 'center_error': 1.,
                                     'scale_error': None, 'crop_identity_score': 0.,
                                     'contamination_ratio': 1., 'crop_homogeneity': 0.})
                        continue
                    box = max(selected, key=lambda b: _box_iou(b, man.crop))
                    a, b = np.array(box, dtype=float), np.array(man.crop, dtype=float)
                    ai = max(0, min(a[2], b[2])-max(a[0], b[0]))*max(0, min(a[3], b[3])-max(a[1], b[1]))
                    area_a, area_b = (a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1])
                    panel_width = max(man.bbox[2]-man.bbox[0], 1)
                    panel_height = max(man.bbox[3]-man.bbox[1], 1)
                    center = np.linalg.norm(((a[:2]+a[2:])/2-(b[:2]+b[2:])/2)/[panel_width,panel_height])
                    patch = reference.crop(tuple(map(int, box)))
                    manual_patch = reference.crop(tuple(map(int, man.crop)))
                    # 非纹样代理：越出人工 panel 的像素和离人工 panel 边界 3px 内的像素。
                    interior = cv2.erode(man.mask.astype(np.uint8), np.ones((7,7), np.uint8)) > 0
                    region = interior[int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                    contamination = 1-float(region.mean()) if region.size else 1.
                    quadrants = [patch.crop((x*patch.width//2, y*patch.height//2,
                                            (x+1)*patch.width//2, (y+1)*patch.height//2))
                                 for y in range(2) for x in range(2)]
                    vec = [descriptor(q)[0] for q in quadrants]
                    homogeneity = float(np.clip(np.mean([max(0.,np.dot(vec[i],vec[j]) /
                            max(np.linalg.norm(vec[i])*np.linalg.norm(vec[j]), 1e-9))
                            for i in range(4) for j in range(i+1,4)]), 0, 1))
                    rows.append({'case': cid, 'variant': variant, 'arm': arm, 'panel': name,
                                 'missing': False, 'crop_iou': float(ai/max(area_a+area_b-ai,1)),
                                 'center_error': float(center),
                                 'scale_error': float(abs(math.log(max(area_a,1)/max(area_b,1)))),
                                 'crop_identity_score': similarity(patch, manual_patch)[0],
                                 'contamination_ratio': contamination,
                                 'crop_homogeneity': homogeneity,
                                 'manual_crop': man.crop, 'auto_crop': box})
    summary = {}
    for arm in (ARMS[2], ARMS[3]):
        selected = [r for r in rows if r['arm'] == arm]
        summary[arm] = {k: case_stat([(r['case'],r[k]) for r in selected if r[k] is not None])
                        for k in ('crop_iou','center_error','scale_error','crop_identity_score',
                                  'contamination_ratio','crop_homogeneity')}
        case_medians = [float(np.median([r['crop_iou'] for r in selected if r['case']==cid]))
                        for cid in CASE_IDS]
        summary[arm]['median_crop_iou'] = float(np.median(case_medians))
    return {'rows': rows, 'summary': summary,
            'metric_definition': 'IoU/center/scale against manual crop; descriptor identity uses E27 FFT+autocorrelation+RGB; contamination is outside or within 3px of manual panel boundary; homogeneity is pairwise quadrant descriptor similarity; missing manual panel counts zero.'}


def _box_iou(a,b):
    inter = max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    aa = (a[2]-a[0])*(a[3]-a[1]); bb = (b[2]-b[0])*(b[3]-b[1])
    return inter/max(aa+bb-inter,1)


def report(root, e28, out):
    assert load(out/'B_causal/component_check.json')['pass']
    assert load(out/'B_causal/frozen_check.json')['pass']
    rows, summary = {}, {}
    shared = {}
    expected = {(c,s,v) for c in CASE_IDS for s in SEEDS for v in VARIANTS}
    for arm in ARMS:
        folder = out/'B_causal'/arm
        rows[arm] = [load(p) for p in sorted(folder.glob('c*_s*_*.json'))]
        assert {(r['case'],r['seed'],r['variant']) for r in rows[arm]} == expected
        assert len(rows[arm]) == 32
        for row in rows[arm]:
            key = row['case'],row['seed'],row['variant']
            hashes = tuple(row[k] for k in ('noise_sha256','condition_sha256','sketch_sha256',
                                            'target_mask_sha256','checkpoint_sha256'))
            if key in shared: assert shared[key] == hashes
            else: shared[key] = hashes
            assert file_sha(folder/f"c{row['case']:02d}_s{row['seed']}_{row['variant']}.png") == row['output_sha256']
        summary[arm] = stats(rows[arm])
    old = load(e28/'B_decomposition/report.json')
    panel = {'panel_miou':old['panel_miou'], 'panel_coverage':old['panel_coverage'],
             'boundary_f1':old['panel_boundary_f1'], 'by_name':old['panel_miou_by_name'],
             'source':'E28 B4; same infer_regions source proposal and same 8 cases/2 variants'}
    crop = crop_metrics(out)
    write(out/'B_causal/panel_metrics.json', panel)
    write(out/'B_causal/crop_metrics.json', crop)
    contrasts, recoveries = {}, {}
    case_summary = {(arm,cid): stats([r for r in rows[arm] if r['case']==cid])
                    for arm in ARMS for cid in CASE_IDS}
    for arm in ARMS[1:]:
        contrasts[arm] = {}
        recoveries[arm] = {}
        for metric in ('follow','identity','theta_error','contour_f1','leakage'):
            pairs = []
            for cid in CASE_IDS:
                items = {a: case_summary[a,cid] for a in ARMS}
                values = [items[a][metric]['mean'] if items[a][metric] else None
                          for a in (arm, ARMS[3], ARMS[0])]
                if None not in values:
                    recoveries[arm].setdefault(metric,[]).append((cid,*values))
                p0, current = items[ARMS[0]][metric],items[arm][metric]
                if p0 and current:pairs.append((cid,current['mean']-p0['mean']))
            contrasts[arm][metric] = case_stat(pairs) if pairs else None
            recoveries[arm][metric] = recovery_stat(recoveries[arm][metric],
                                     metric=='theta_error')
    # 32 张图不是独立样本；所有路由判断使用 case 聚合均值。
    p0,p1,p2,p3 = (summary[a] for a in ARMS)
    mean = lambda r,k: r[k]['mean'] if r[k] else None
    panel_checks = [mean(p0,'follow')-mean(p1,'follow')>=.15,
                    mean(p1,'theta_error')-mean(p0,'theta_error')>=15,
                    mean(p0,'identity')-mean(p1,'identity')>=.05,
                    recoveries[ARMS[1]]['follow']['mean']<=.70 if recoveries[ARMS[1]]['follow']['positive_oracle_gap'] else False,
                    panel['panel_miou']['mean']<.80,
                    np.mean([panel['by_name'][n]['mean'] for n in ('body_left','body_right')])<.70,
                    panel['panel_coverage']['mean']<.95]
    cmetric = crop['summary'][ARMS[2]]
    crop_checks = [mean(p0,'follow')-mean(p2,'follow')>=.15,
                   mean(p2,'theta_error')-mean(p0,'theta_error')>=15,
                   mean(p0,'identity')-mean(p2,'identity')>=.05,
                   recoveries[ARMS[2]]['follow']['mean']<=.70 if recoveries[ARMS[2]]['follow']['positive_oracle_gap'] else False,
                   cmetric['crop_identity_score']['mean']<.85,
                   cmetric['contamination_ratio']['mean']>.10,
                   cmetric['crop_homogeneity']['mean']<.80]
    panel_flag = sum(panel_checks)>=3 and any(panel_checks[:2])
    crop_flag = sum(crop_checks)>=3 and any(crop_checks[:2])
    interaction_follow = min(mean(p1,'follow'),mean(p2,'follow'))-mean(p3,'follow')
    interaction_theta = mean(p3,'theta_error')-max(mean(p1,'theta_error'),mean(p2,'theta_error'))
    interaction_flag = not(panel_flag or crop_flag) and (interaction_follow>=.15 or interaction_theta>=15)
    if panel_flag and crop_flag: route='joint_panel_crop'
    elif panel_flag: route='panel_parser'
    elif crop_flag: route='canonical_crop_selector'
    elif interaction_flag: route='proposal_interaction'
    else: route='stop_no_clear_bottleneck'
    decision = {'experiment':'E29','causal_split_complete':True,
                'panel_bottleneck':bool(panel_flag),'crop_bottleneck':bool(crop_flag),
                'interaction_bottleneck':bool(interaction_flag),
                'joint_bottleneck':bool(panel_flag and crop_flag),
                'panel_checks':[bool(x) for x in panel_checks],
                'crop_checks':[bool(x) for x in crop_checks],
                'interaction_follow_drop':interaction_follow,
                'interaction_theta_extra':interaction_theta,
                'selected_route':route,'next_route':route,
                'panel_repair_pass':None,'crop_repair_pass':None,
                'pilot_pass':None,'confirmation_pass':None,'multi_target_pass':None}
    write(out/'B_causal/report.json', {'summary':summary, 'paired_delta_vs_P0':contrasts,
                                       'panel_metrics':'panel_metrics.json',
                                       'crop_metrics':'crop_metrics.json',
                                       'noise_shared':True,'conditions_shared':True,
                                       'model_frozen':True,'images_per_arm':32})
    write(out/'B_causal/recovery_report.json',recoveries)
    write(out/'B_causal/decision.json',decision)
    write(out/'decision_summary.json',decision)
    # 只按预先固定的 case ID 绘制，另从其余三例用 seed=42 抽两例。
    ids=[13,6,7,12,17]+random.Random(42).sample([9,10,14],2)
    review=out/'review_images';review.mkdir(exist_ok=True)
    for cid in ids:
        case=next(c for c in load(out/'cases.json')['references'] if c['id']==cid)
        reference=Image.open(out/'A_audit/inputs'/f'c{cid:02d}_original.png').convert('RGB')
        manual=reference.copy();d=ImageDraw.Draw(manual)
        for p in case['panels']:
            d.rectangle(p['box'],outline='red',width=2);d.rectangle(p['crop'],outline='lime',width=2)
        auto=reference.copy();d=ImageDraw.Draw(auto)
        smask=np.asarray(Image.open(out/'A_audit/inputs'/f'c{cid:02d}_mask.png'))>0
        for p in source_panels(reference,case['panels'],'automatic',smask,{}):
            d.rectangle(p.bbox,outline='red',width=2);d.rectangle(p.crop,outline='lime',width=2)
        tiles=[(reference,'reference'),(manual,'manual panels/crops'),(auto,'auto panels/crops')]
        for arm in ARMS:
            folder=out/'B_causal'/arm
            for suffix,label in (('original_scaffold','S0'),('original_vae','S1'),('s42_original','S2'),('s42_rot90','rot90')):
                tiles.append((Image.open(folder/f'c{cid:02d}_{suffix}.png').convert('RGB'),arm[:2]+' '+label))
        sheet=Image.new('RGB',(5*256,4*370),'white');draw=ImageDraw.Draw(sheet)
        for i,(im,label) in enumerate(tiles):
            im.thumbnail((250,340));x=(i%5)*256;y=(i//5)*370
            sheet.paste(im,(x,y));draw.text((x+3,y+344),label,fill='black')
        sheet.save(review/f'c{cid:02d}.jpg',quality=92)
    write(out/'artifact_manifest.json',{'experiment':'E29','audit':'A_audit/report.json',
          'report':'B_causal/report.json','panel':'B_causal/panel_metrics.json',
          'crop':'B_causal/crop_metrics.json','recovery':'B_causal/recovery_report.json',
          'decision':'decision_summary.json','review_images':[f'review_images/c{c:02d}.jpg' for c in ids]})
    print('[E29 B]',json.dumps({'route':route,'panel_checks':panel_checks,'crop_checks':crop_checks}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('audit','scaffold','generate','report'),required=True)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--e28',type=Path,default=Path('output_eval/e28_20260929'))
    parser.add_argument('--out',type=Path,default=Path('output_eval/e29_20260930'))
    args=parser.parse_args()
    root=args.root.resolve();e28=args.e28.resolve();out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if args.stage=='audit':audit(root,e28,out)
    elif args.stage=='scaffold':scaffold(root,out)
    elif args.stage=='generate':
        import torch
        with torch.inference_mode():generate(root,out)
    else:report(root,e28,out)


if __name__=='__main__':main()
