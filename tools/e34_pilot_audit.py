"""同一助手两轮视觉复核及新增参考的旋转完整性。失败时保留证据并停止扩标。"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from data.e33_interventions import nuisance
from models.apacc_features import load_dino, extract_dense_features, estimated_foreground
from models.apacc_affinity_adapter import PatternAffinityAdapter
from models.local_pattern_field import patch_geometry
from models.pattern_geometry import axial_distance
from tools.e30_a2_affinity import extra_features, tensor_field, norm_at
from tools.e34_protocol import E30
from tools.e34_stage0 import read, write, sha
from tools.e34_evidence import iou
from tools.e34_pilot_prepare import PILOT, archive


def mask(row, kind, size):
    image = Image.new('1', size)
    draw = ImageDraw.Draw(image)
    for polygon in row[kind]:
        draw.polygon([tuple(p) for p in polygon], fill=1)
    return np.asarray(image, dtype=bool)


def masks(row, size):
    fg = mask(row, 'foreground_polygons', size)
    structure = mask(row, 'structure_polygons', size)
    support = mask(row, 'support_polygons', size) & fg & ~structure
    return fg, support, structure


def consistency(document, protocol, dataset):
    first = {r['id']: r for r in document['records']}
    results = []
    for second in document['second_review']:
        row = first[second['id']]
        size = Image.open(dataset / row['reference']).size
        _, a, _ = masks(row, size)
        _, b, _ = masks(second, size)
        results.append(dict(id=row['id'], support_iou=float((a & b).sum()/max((a | b).sum(), 1)),
            best_box_iou=max(iou(x['box'], y['box']) for x in row['evidence'] for y in second['evidence']),
            family=float(row['family']==second['family']),
            readability=float(row['evidence'][0]['readable']==second['evidence'][0]['readable']),
            confidence=float(row['evidence'][0]['confidence']==second['evidence'][0]['confidence'])))
    assert {r['id'] for r in results} == set(protocol['review_ids'])
    metrics = {key: float(np.mean([r[key] for r in results])) for key in protocol['annotation_gate']}
    passed = all(metrics[k] >= threshold for k, threshold in protocol['annotation_gate'].items())
    result = dict(rows=results, means=metrics, exploratory_gate_pass=passed,
        provenance='same assistant sequential review, original images shuffled without overlays; not independent annotators or genuinely blinded', human_inter_annotator_gate_pass=False)
    write(PILOT / 'consistency.json', result)
    return result


def cosine(a, b):
    return float(np.dot(a, b)/max(np.linalg.norm(a)*np.linalg.norm(b), 1e-8))


def descriptor(image, box, dino, adapter, normalization, device):
    # 同一完整RGB上下文的冻结E30描述子；不把几何代理当identity。
    fg = estimated_foreground(image)
    features = extra_features(image, fg, extract_dense_features(image, fg, dino, device))
    with torch.inference_mode():
        z = adapter(torch.tensor(tensor_field(features, normalization), device=device)).cpu().numpy()
    x, y, x1, y1 = box
    h, w = image.height, image.width
    return z[int(y*64/h):int(y1*64/h), int(x*64/w):int(x1*64/w)].mean((0, 1))


def run(document, protocol, dataset, device):
    recruitment = {r['id']:r for r in protocol['records'] if r['schema_pilot']}
    assert set(recruitment) == {r['id'] for r in document['records']}
    for row in document['records']:
        assert row['reference_sha256'] == recruitment[row['id']]['reference_sha256'] == sha(dataset/row['reference'])
        image = Image.open(dataset / row['reference']).convert('RGB')
        fg, support, structure = masks(row, image.size)
        folder = PILOT / 'annotations' / row['id']
        folder.mkdir(parents=True, exist_ok=True)
        for name, array in [('foreground', fg), ('support', support), ('structure', structure)]:
            Image.fromarray((array*255).astype(np.uint8)).save(folder / (name+'.png'))
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        for polygon in row['support_polygons']:
            draw.line([tuple(p) for p in polygon]+[tuple(polygon[0])], fill='lime', width=2)
        for polygon in row['structure_polygons']:
            draw.line([tuple(p) for p in polygon]+[tuple(polygon[0])], fill='red', width=2)
        for evidence in row['evidence']:
            draw.rectangle(evidence['box'], outline='cyan', width=2)
            x, y, x1, y1 = evidence['box']
            assert x1-x == y1-y
            if evidence['confidence'] != 'invalid':
                assert support[y:y1, x:x1].mean() >= .9
        overlay.save(folder/'overlay.png')
    write(PILOT/'annotations/source.json', document)
    review = consistency(document, protocol, dataset)
    checkpoint = E30/'A21_embedding/seed42/adapter.pt'
    weights = Path('output_eval/e30_apacc_20260930/dinov2_vits14_pretrain.pth')
    dino, dino_sha = load_dino(device, weights)
    state = torch.load(checkpoint, map_location=device)
    adapter = PatternAffinityAdapter(state['input_dim']-384, not state['no_geometry']).to(device)
    adapter.load_state_dict(state['model'])
    adapter.eval()
    for p in adapter.parameters():
        p.requires_grad_(False)
    normalization = norm_at(E30)
    measurements = []
    for index, row in enumerate(document['records']):
        e = row['evidence'][0]
        # 未读方向或invalid参考不做正式旋转；避免旋转已标出的结构/未知区域。
        formal = e['readable'] and e['confidence']=='high'
        if not formal:
            measurements.append(dict(id=row['id'], group=row['group'], family=row['family'], formal=False,
                intervention_run=False, reason='not high-confidence orientation-readable canonical evidence'))
            continue
        original_image = Image.open(dataset/row['reference']).convert('RGB')
        original = np.asarray(original_image)
        _, support, _ = masks(row, original_image.size)
        x, y, x1, y1 = e['box']
        patch = original[y:y1, x:x1].copy()
        rotated = np.rot90(patch).copy()
        clean = original.copy()
        clean[y:y1, x:x1] = rotated
        item = dict(id=row['id'], group=row['group'], family=row['family'],
            formal=formal, intervention_run=True,
            outside_support_unchanged=bool(np.array_equal(original[~support], clean[~support])),
            inverse_exact=bool(np.array_equal(np.rot90(rotated, -1), patch)))
        if item['formal']:
            noisy_base, _ = nuisance(patch, np.ones(patch.shape[:2], bool), 34042+index)
            noisy_rot, _ = nuisance(rotated, np.ones(patch.shape[:2], bool), 34042+index)
            b, r = [patch_geometry(Image.fromarray(a)) for a in (patch, rotated)]
            nb, nr = [patch_geometry(Image.fromarray(a[3:-3, 3:-3])) for a in (noisy_base, noisy_rot)]
            item.update(clean_error=float(axial_distance(r['orientation'], b['orientation']+90)),
                noisy_error=float(axial_distance(nr['orientation'], nb['orientation']+90)),
                clean_confidence=min(b['confidence'], r['confidence']), noisy_confidence=min(nb['confidence'], nr['confidence']),
                visual_error=float(axial_distance(b['orientation'], e['orientation_deg'])))
            item['clean_success'] = item['clean_error'] <= 15 and item['clean_confidence'] >= .25
            item['noisy_success'] = item['noisy_error'] <= 15 and item['noisy_confidence'] >= .25
            views = [original_image, Image.fromarray(clean)]
            for changed_patch in (noisy_base, noisy_rot):
                array = original.copy()
                array[y:y1, x:x1] = changed_patch
                views.append(Image.fromarray(array))
            z, rz, nz, nrz = [descriptor(view, e['box'], dino, adapter, normalization, device) for view in views]
            item.update(clean_identity=cosine(z, rz), noisy_identity=cosine(nz, nrz),
                nuisance_control_identity=cosine(z, nz))
            for name, view in zip(('original', 'rot90', 'nuisance', 'nuisance_rot90'), views):
                destination = PILOT / 'rot90' / row['id']
                destination.mkdir(parents=True, exist_ok=True)
                view.save(destination / (name+'.png'))
        measurements.append(item)
        print('rot90', index+1, item, flush=True)
    formal = [r for r in measurements if r['formal']]
    metric = lambda key: float(np.mean([r[key] for r in formal])) if formal else 0.
    summary = dict(formal_count=len(formal), total_count=len(measurements), clean_success=metric('clean_success'),
        noisy_success=metric('noisy_success'), clean_identity=metric('clean_identity'), noisy_identity=metric('noisy_identity'),
        nuisance_control_identity=metric('nuisance_control_identity'), outside_support_unchanged=bool(formal) and all(r['outside_support_unchanged'] for r in formal),
        inverse_exact=bool(formal) and all(r['inverse_exact'] for r in formal), rows=measurements)
    g = protocol['rot90_gate']
    summary['gate_pass'] = bool(formal and summary['clean_success'] >= g['clean'] and summary['noisy_success'] >= g['nuisance']
        and summary['outside_support_unchanged'] and min(summary['clean_identity'], summary['noisy_identity']) >= g['identity_cosine']
        and summary['nuisance_control_identity']-summary['noisy_identity'] <= g['identity_loss_vs_nuisance_control'])
    write(PILOT/'rot90_audit.json', summary)
    passed = review['exploratory_gate_pass'] and summary['gate_pass']
    write(PILOT/'status.json', dict(stage='E34-A0/A0.1/A1', completed=True, exploratory_preconditions_pass=passed,
        A2_ready=passed, learned_training_ready=False, diffusion_ready=False, human_annotation_gate_pass=False,
        source_sha256=sha(Path('assets/e34_pilot_annotations.json')), dino_sha256=dino_sha, adapter_sha256=sha(checkpoint),
        reason='preconditions passed' if passed else 'stop expansion: annotation or rot90 gate failed'))
    archive(PILOT, 'e34_pilot_preconditions_results.zip')
    print('E34 preconditions', passed, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default=Path('/share/home/u2515283058/datasets/BF'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(read(Path('assets/e34_pilot_annotations.json')), read(PILOT/'protocol.json'), args.dataset, torch.device('cuda'))
