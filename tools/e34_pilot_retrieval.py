"""40参考的四组固定检索；只在前置门槛通过后运行，无优化器或训练。"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from models.apacc_features import load_dino, extract_dense_features, estimated_foreground
from models.apacc_affinity_adapter import PatternAffinityAdapter
from models.apacc_canonicality import crop_metrics
from models.local_pattern_field import patch_geometry
from models.pattern_geometry import axial_distance
from tools.e30_a2_affinity import extra_features, tensor_field, norm_at
from tools.e34_protocol import E30
from tools.e34_stage0 import read, write, sha
from tools.e34_evidence import cache_reference, nms_top, iou
from tools.e34_pilot_audit import masks, cosine
from tools.e34_pilot_prepare import PILOT, archive


DEFINITION = dict(
    ranking_inputs='RGB and estimated foreground only; no annotation, family, identity query or validation optimization',
    A0='exact E29 fixed score on the shared candidate set',
    A1='frozen DINO within-candidate cosine coherence',
    A2='frozen E30 affinity within-candidate cosine coherence',
    A3='0.40 affinity + 0.25 geometry + 0.20 homogeneity + 0.15 interior margin - 0.20 structural proxy',
    geometry='E26 confidence times average quadrant orientation and frequency consistency; no readable quadrant gives geometry 0',
    structure='frozen apacc_canonicality Canny/Hough long-line proxy; can penalize genuine stripe/plaid lines',
    coverage='all references with at least one valid candidate in the complete shared candidate pool',
    recall='all references with at least one valid candidate among NMS top K; invalid reference contributes zero',
    identity='maximum frozen E30 full-reference ROI mean cosine against non-invalid canonical evidence; evaluation only',
    contamination='visual structural mask or outside confirmed garment silhouette; unknown fine seams may remain',
    confidence='visual reference evidence must be high/medium; readable patches additionally require E26 >=0.25',
    readability='report overall selected E26 readable and subset with visually readable annotation separately; literal overall 0.80 gate preserved',
    significance='paired reference bootstrap of validation V@1(A3-A0), 2000 replicates seed34042, lower95% bound >0',
    human_GT=False, independent_test_used=False, learned_model=False)


def mean_descriptor(field, box, size):
    x, y, x1, y1 = box
    w, h = size
    return field[int(y*64/h):int(y1*64/h), int(x*64/w):int(x1*64/w)].mean((0, 1))


def evaluate_candidates(row, image, cache, dense, protocol):
    fg, support, structure = masks(row, image.size)
    evidence = [e for e in row['evidence'] if e['confidence']!='invalid']
    descriptors = [mean_descriptor(dense, e['box'], image.size) for e in evidence]
    results, scores = [], cache['baselines'].copy()
    thresholds = protocol['valid']
    for j, box_array in enumerate(cache['boxes']):
        box = box_array.tolist()
        x, y, x1, y1 = box
        patch = image.crop(box)
        geometry = patch_geometry(patch)
        # region argument receives RGB foreground, not evaluation support, and is not a ranking signal.
        extra = crop_metrics(image, cache['foreground'], cache['foreground'], box)
        quadrants = [patch_geometry(patch.crop((a*patch.width//2, b*patch.height//2,
            (a+1)*patch.width//2, (b+1)*patch.height//2))) for b in range(2) for a in range(2)]
        geo = geometry['confidence'] * (extra['orientation_consistency']+extra['period_consistency'])/2 if any(q['valid'] for q in quadrants) else 0.
        features = cache['features'][j]
        weights = protocol['fixed_A3']
        scores[j, 3] = weights['affinity']*features[129]+weights['geometry']*geo+weights['homogeneity']*features[130]+weights['boundary']*features[133]-weights['structure_penalty']*extra['structural_edge']
        purity = float(support[y:y1, x:x1].mean())
        contamination = float((structure | ~fg)[y:y1, x:x1].mean())
        candidate_descriptor = mean_descriptor(dense, box, image.size)
        identities = [cosine(candidate_descriptor, z) for z in descriptors]
        identity = max(identities or [0.])
        # 不用IoU选identity oracle；只用于判断候选能否匹配至少一个已标identity组。
        valid_matches = []
        for e, similarity in zip(evidence, identities):
            orient_ok = not e['readable'] or (geometry['valid'] and geometry['confidence'] >= thresholds['readable_confidence'] and axial_distance(geometry['orientation'], e['orientation_deg']) <= thresholds['readable_visual_orientation_error_max'])
            valid_matches.append(similarity >= thresholds['affinity_identity'] and orient_ok)
        valid = bool(any(valid_matches) and purity >= thresholds['support_purity'] and contamination <= thresholds['structural_contamination_max'] and features[130] >= thresholds['homogeneity'])
        results.append(dict(box=box, valid=valid, purity=purity, contamination=contamination,
            identity=identity, homogeneity=float(features[130]), E26_readable=bool(geometry['valid'] and geometry['confidence'] >= .25),
            E26_confidence=geometry['confidence'], orientation=geometry['orientation'], frequency=geometry['frequency'],
            geometry_consistency=geo, structural_proxy=extra['structural_edge'],
            diagnostic_box_iou=max([iou(box, e['box']) for e in evidence] or [0.])))
    return results, scores


def summary(rows):
    output = {}
    for group in ('train', 'validation'):
        output[group] = {}
        for arm in ('A0', 'A1', 'A2', 'A3'):
            selected = [r for r in rows if r['group']==group and r['arm']==arm]
            metrics = {key: float(np.mean([r[key] for r in selected])) for key in ('coverage', 'valid1', 'recall3', 'recall5', 'purity', 'contamination', 'identity', 'homogeneity', 'readable', 'diagnostic_box_iou')}
            gt_readable = [r for r in selected if r['GT_readable']]
            metrics.update(n=len(selected), readable_conditioned_n=len(gt_readable),
                readable_conditioned=float(np.mean([r['readable'] for r in gt_readable])) if gt_readable else None,
                invalid_selected=sum(r['invalid_reference'] and bool(r['selected']) for r in selected))
            output[group][arm] = metrics
    validation = {arm:{r['id']:r for r in rows if r['group']=='validation' and r['arm']==arm} for arm in ('A0', 'A3')}
    difference = np.array([validation['A3'][rid]['valid1']-validation['A0'][rid]['valid1'] for rid in sorted(validation['A0'])])
    rng = np.random.default_rng(34042)
    bootstrap = [float(rng.choice(difference, len(difference), replace=True).mean()) for _ in range(2000)]
    output['A3_minus_A0_valid1'] = dict(mean=float(difference.mean()), ci95=np.percentile(bootstrap, [2.5, 97.5]).tolist(), unit='reference')
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default=Path('/share/home/u2515283058/datasets/BF'))
    args = parser.parse_args()
    assert read(PILOT/'status.json')['A2_ready'], '前置门槛失败，不运行检索'
    protocol = read(PILOT/'protocol.json')
    document = read(Path('assets/e34_pilot_annotations_40.json'))
    records = document['records']
    assert {r['id'] for r in records} == {r['id'] for r in protocol['records']}
    assert sum(r['group']=='train' for r in records)==30 and sum(r['group']=='validation' for r in records)==10
    device = torch.device('cuda')
    torch.set_num_threads(4)
    weights = Path('output_eval/e30_apacc_20260930/dinov2_vits14_pretrain.pth')
    dino, dino_sha = load_dino(device, weights)
    checkpoint = E30/'A21_embedding/seed42/adapter.pt'
    state = torch.load(checkpoint, map_location=device)
    adapter = PatternAffinityAdapter(state['input_dim']-384, not state['no_geometry']).to(device)
    adapter.load_state_dict(state['model'])
    adapter.eval()
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    normalization = norm_at(E30)
    folder = PILOT/'retrieval'
    write(folder/'definitions.json', dict(DEFINITION, dino_sha256=dino_sha, affinity_sha256=sha(checkpoint),
        source_sha256=sha(Path('tools/e34_pilot_retrieval.py')), annotation_sha256=sha(Path('assets/e34_pilot_annotations_40.json'))))
    results = []
    recruitment = {r['id']:r for r in protocol['records']}
    for index, row in enumerate(records):
        assert sha(args.dataset/row['reference']) == row['reference_sha256'] == recruitment[row['id']]['reference_sha256']
        image = Image.open(args.dataset/row['reference']).convert('RGB')
        cache = cache_reference(row, image, dino, adapter, normalization, device)
        features = extra_features(image, cache['foreground'], extract_dense_features(image, cache['foreground'], dino, device))
        with torch.inference_mode():
            dense = adapter(torch.tensor(tensor_field(features, normalization), device=device)).cpu().numpy()
        candidates, scores = evaluate_candidates(row, image, cache, dense, protocol)
        write(folder/'candidates'/(row['id']+'.json'), dict(id=row['id'], group=row['group'], candidates=candidates))
        for column, arm in enumerate(('A0', 'A1', 'A2', 'A3')):
            chosen = nms_top(scores[:, column], cache['boxes']) if len(candidates) else []
            selected = [dict(candidates[j], score=float(scores[j, column])) for j in chosen]
            first = selected[0] if selected else dict(purity=0., contamination=1., identity=0., homogeneity=0., E26_readable=False, diagnostic_box_iou=0.)
            result = dict(id=row['id'], group=row['group'], family=row['family'], arm=arm, candidate_count=len(candidates),
                invalid_reference=all(e['confidence']=='invalid' for e in row['evidence']), GT_readable=any(e['readable'] and e['confidence']!='invalid' for e in row['evidence']),
                selected=selected, coverage=float(any(c['valid'] for c in candidates)), valid1=float(bool(selected) and selected[0]['valid']),
                recall3=float(any(c['valid'] for c in selected[:3])), recall5=float(any(c['valid'] for c in selected)),
                readable=float(first['E26_readable']), **{key:first[key] for key in ('purity', 'contamination', 'identity', 'homogeneity', 'diagnostic_box_iou')})
            results.append(result)
            if selected:
                overlay = image.copy()
                draw = ImageDraw.Draw(overlay)
                for evidence in row['evidence']:
                    draw.rectangle(evidence['box'], outline='lime', width=2)
                draw.rectangle(first['box'], outline='red', width=2)
                destination = folder/'selected'/arm/row['group']
                destination.mkdir(parents=True, exist_ok=True)
                overlay.save(destination/(row['id']+'.png'))
        print('fixed retrieval', index+1, '/40', row['id'], len(candidates), flush=True)
    write(folder/'rows.json', results)
    report = summary(results)
    g = protocol['A2_gate']
    metric = report['validation']['A3']
    report['gate_checks'] = {k: metric[k] >= g[k] for k in ('coverage', 'recall5', 'valid1', 'identity', 'homogeneity')}
    report['gate_checks'].update(contamination=metric['contamination']<=g['contamination_max'],
        selected_readable=metric['readable']>=g['selected_readable'],
        A3_significantly_better_A0=report['A3_minus_A0_valid1']['ci95'][0]>g['A3_minus_A0_valid1_ci95_lower_gt'])
    report.update(exploratory_gate_pass=all(report['gate_checks'].values()), original_human_gate_pass=False,
        generated_diffusion_images=0, optimizer_steps=0, expected_result_rows=160, actual_result_rows=len(results))
    write(folder/'summary.json', report)
    archive(PILOT, 'e34_pilot_retrieval_results.zip')
    print('retrieval Pilot gate', report['exploratory_gate_pass'], flush=True)


if __name__ == '__main__':
    main()
