"""E29 联合瓶颈后：panel 标注可行性与固定规则 crop proposal 一次评估。"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from models.e29_crop_selector import select_crop_e29
from tools.e20_utilization import case_stat
from tools.e27_correspondence import similarity, source_labels
from tools.e27_experiment import write
from tools.e28_experiment import CASE_IDS, VARIANTS


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def measure(image, panel_mask, manual_box, selected_box, info, case, variant, panel):
    a, b = np.array(selected_box, float), np.array(manual_box, float)
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a, area_b = (a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1])
    ys, xs = np.where(panel_mask)
    panel_size = np.array([max(xs.max()-xs.min()+1,1), max(ys.max()-ys.min()+1,1)])
    center = np.linalg.norm(((a[:2]+a[2:])/2-(b[:2]+b[2:])/2)/panel_size)
    interior = cv2.erode(panel_mask.astype(np.uint8), np.ones((7,7),np.uint8)) > 0
    region = interior[int(a[1]):int(a[3]),int(a[0]):int(a[2])]
    return {'case':case,'variant':variant,'panel':panel,
            'manual_crop':manual_box,'selected_crop':selected_box,
            'crop_iou':float(intersection/max(area_a+area_b-intersection,1)),
            'center_error':float(center),
            'scale_error':float(abs(math.log(max(area_a,1)/max(area_b,1)))),
            'crop_identity_score':similarity(image.crop(tuple(map(int,selected_box))),
                                             image.crop(tuple(map(int,manual_box))))[0],
            'crop_homogeneity':info['homogeneity'],
            'contamination_ratio':1-float(region.mean()),
            'selector':info}


def summarize(rows):
    metrics = ('crop_iou','center_error','scale_error','crop_identity_score',
               'crop_homogeneity','contamination_ratio')
    summary = {key:case_stat([(r['case'],r[key]) for r in rows]) for key in metrics}
    summary['median_crop_iou'] = float(np.median([
        np.median([r['crop_iou'] for r in rows if r['case']==cid])
        for cid in sorted({r['case'] for r in rows})]))
    summary['cases'] = len({r['case'] for r in rows})
    summary['panels'] = len(rows)
    summary['fallbacks'] = sum(r['selector']['fallback'] for r in rows)
    checks = [summary['median_crop_iou'] >= .60,
              summary['crop_identity_score']['mean'] >= .85,
              summary['crop_homogeneity']['mean'] >= .80,
              summary['contamination_ratio']['mean'] <= .10]
    summary['proposal_gate_checks'] = [bool(v) for v in checks]
    summary['proposal_gate_pass'] = all(checks)
    return summary


def evaluate_case(case, image, source_mask, variant):
    labels = source_labels(source_mask, case['panels'])
    rows = []
    for index, panel in enumerate(case['panels']):
        mask = labels == index
        if not mask.any():
            continue
        box, info = select_crop_e29(image, mask)
        rows.append(measure(image, mask, panel['crop'], box, info,
                            case['id'], variant, panel['name']))
    return rows


def run(root, dataset, out):
    decision = load(out/'decision_summary.json')
    assert decision['selected_route'] == 'joint_panel_crop'
    cases = load(root/'data/e27_c3_cases.json')['references']
    non_test = [c for c in cases if c['id'] not in CASE_IDS]
    assert len(non_test) == 10 and len(cases) == 18
    panel = {'total_manual_annotated_references':len(cases),
             'held_out_causal_cases':list(CASE_IDS),
             'non_test_manual_annotated_references':len(non_test),
             'recommended_min_train':100,'recommended_min_validation':20,
             'training_ready':False,
             'action':'collect independent source panel annotations; do not train on E29 causal cases'}
    panel_path = out/'C_training/panel_parser/feasibility.json'
    write(panel_path, panel)
    dev_rows = []
    for case in non_test:
        image = Image.open(dataset/case['source']).convert('RGB')
        mask = np.asarray(Image.open(dataset/case['source_mask']).convert('L')) > 0
        dev_rows.extend(evaluate_case(case, image, mask, 'original'))
        print('[E29 crop dev]', case['id'], flush=True)
    dev_summary = summarize(dev_rows)
    test_rows = []
    for case in cases:
        if case['id'] not in CASE_IDS:
            continue
        cid = case['id']
        mask = np.asarray(Image.open(out/'A_audit/inputs'/f'c{cid:02d}_mask.png')) > 0
        for variant in VARIANTS:
            image = Image.open(out/'A_audit/inputs'/f'c{cid:02d}_{variant}.png').convert('RGB')
            test_rows.extend(evaluate_case(case, image, mask, variant))
        print('[E29 crop heldout]', cid, flush=True)
    test_summary = summarize(test_rows)
    crop = {'selector':'models.e29_crop_selector.select_crop_e29',
            'selection_inputs':'reference RGB and panel mask only; no case ID, manual crop or motif group',
            'candidate_sizes':[48,64,80,96], 'stride':16, 'minimum_occupancy':.95,
            'fixed_weights':{'homogeneity':.25,'geometry_confidence':.20,
                             'boundary_margin':.20,'panel_color_similarity':.20,
                             'crop_size':.10,'occupancy':.05},
            'dev_non_test':dev_summary,'heldout_causal':test_summary,
            'dev_rows':dev_rows,'heldout_rows':test_rows,
            'gate':'median IoU>=.60, descriptor identity>=.85, homogeneity>=.80, contamination<=.10; all required'}
    write(out/'C_training/crop_selector/probe.json', crop)
    decision.update(panel_training_ready=False,
                    panel_training_data='10 non-test manually annotated references; minimum 100 train + 20 validation',
                    crop_proposal_pass=bool(test_summary['proposal_gate_pass']),
                    crop_proposal_dev_pass=bool(dev_summary['proposal_gate_pass']),
                    next_route='collect_panel_annotations_and_revise_crop_selector'
                    if not test_summary['proposal_gate_pass'] else 'collect_panel_annotations_then_crop_pilot')
    write(out/'decision_summary.json', decision)
    write(out/'completion_check.json',
          {'pass':True,'causal_split_complete':True,'selected_route':'joint_panel_crop',
           'panel_training_ready':False,'crop_proposal_pass':bool(test_summary['proposal_gate_pass']),
           'pilot_run':False,'pilot_reason':'panel training set below minimum; crop proposal gate failed'
           if not test_summary['proposal_gate_pass'] else 'panel training set below minimum',
           'confirmation_run':False,'causal_cases_used_for_training':False,
           'all_four_arms_images':32,'frozen_e5':True})
    manifest = load(out/'artifact_manifest.json')
    manifest.update(panel_feasibility='C_training/panel_parser/feasibility.json',
                    crop_probe='C_training/crop_selector/probe.json',
                    completion='completion_check.json')
    write(out/'artifact_manifest.json',manifest)
    print('[E29 C]',json.dumps({'dev_gate':dev_summary['proposal_gate_pass'],
                                'heldout_gate':test_summary['proposal_gate_pass'],
                                'heldout_iou':test_summary['median_crop_iou']}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--dataset',type=Path,default=Path('/share/home/u2515283058/datasets/BF'))
    parser.add_argument('--out',type=Path,default=Path('output_eval/e29_20260930'))
    args = parser.parse_args()
    run(args.root.resolve(),args.dataset.resolve(),args.out.resolve())
