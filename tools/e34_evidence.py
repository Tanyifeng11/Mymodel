"""E34-A五组真实证据发现对照；用户授权模型弱标注，原人工门槛仍未通过。"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image, ImageDraw

from models.apacc_features import extract_dense_features, estimated_foreground, load_dino
from models.apacc_affinity_adapter import PatternAffinityAdapter
from models.e29_crop_selector import _homogeneity
from models.local_pattern_field import patch_geometry
from tools.e27_correspondence import similarity
from tools.e30_a2_affinity import extra_features, tensor_field, norm_at
from tools.e34_stage0 import read, write, sha
from tools.e34_annotations import materialize, intervention_audit
from tools.e34_protocol import OUT, E30, PROTOCOL


# 在评估前锁定。所有组读相同RGB估计前景；标注只用于train loss及评估。
SETTINGS = dict(version=2, scope='E34-A exploratory with assistant visual weak annotations',
    authorization='你直接帮我标注，继续完成下一步', seeds=[42, 43, 44],
    frozen_affinity_seed=42, candidates=PROTOCOL['candidate'],
    A0='E29 exact .25 homogeneity + .20 E26 confidence + .20 margin + .20 color + .10 size/96 + .05 occupancy',
    A1='DINO within-crop mean cosine coherence', A2='frozen E30 seed42 within-crop cosine coherence',
    A3='.60 affinity coherence + .20 geometry confidence + .20 confidence-weighted orientation consistency',
    A4='64D mean affinity + 64D affinity std + 10 scalar context; Linear64 GELU Linear1 sigmoid',
    steps=1500, batch_references=64, lr=.0002, weight_decay=.0001,
    train_target='positive_purity * (.5 + .5 max GT IoU); soft BCE; unknown region weight .1',
    sampling='uniform train reference then uniform candidate; no val/test in optimization',
    normalization='train references only; frozen for val/test', nms_iou=.5, top_k=5,
    contamination_gate='conservative upper bound 1-positive-region purity, with lower bound 1-estimated foreground purity',
    coverage='valid references with top1 IoU>=.5 and positive-region purity>=.90',
    invalid='no reliable canonical evidence; included as train negatives; heldout invalid false acceptance reported separately',
    family_labels_used_in_model=False, annotation_foreground_used_in_model=False,
    original_human_gate_pass=False, full_reference_intervention_gate_pass=False,
    validation_only_gate=True, no_threshold_tuning=True,
    limitation='positive-region masks are partial, foreground is heuristic, identities are per-reference; no formal human-GT conclusion')


class EvidenceHead(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dimension, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.layers(x).squeeze(-1)


def iou(a, b):
    x, y = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x1 - x) * max(0, y1 - y)
    return intersection / max((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection, 1)


def nms_top(scores, boxes):
    order = np.argsort(-scores, kind='stable')
    chosen = []
    for index in order:
        if all(iou(boxes[index], boxes[j]) < SETTINGS['nms_iou'] for j in chosen):
            chosen.append(int(index))
            if len(chosen) == 5:
                break
    return chosen


def coherence(values):
    mean = values.mean(0)
    return float(np.linalg.norm(mean)), mean, values.std(0)


def cache_reference(row, image, dino, adapter, normalization, device):
    fg = estimated_foreground(image)
    features = extra_features(image, fg, extract_dense_features(image, fg, dino, device))
    appearance = features['appearance']
    with torch.inference_mode():
        z = adapter(torch.tensor(tensor_field(features, normalization), device=device)).cpu().numpy()
    distance = cv2.distanceTransform(fg.astype(np.uint8), cv2.DIST_L2, 5)
    rgb = np.asarray(image)
    color = np.median(rgb[fg], axis=0)
    ys, xs = np.where(fg)
    boxes, context, baseline = [], [], []
    for size in (48, 64, 80, 96):
        for y in range(max(0, int(ys.min())), min(image.height-size, int(ys.max())) + 1, 16):
            for x in range(max(0, int(xs.min())), min(image.width-size, int(xs.max())) + 1, 16):
                occupancy = float(fg[y:y+size, x:x+size].mean())
                if occupancy < .95:
                    continue
                box = [x, y, x+size, y+size]
                patch = image.crop(box)
                geometry = patch_geometry(patch)
                homogeneous = _homogeneity(patch)
                margin = float(np.clip(np.percentile(distance[y:y+size, x:x+size], 20) / (size/4), 0, 1))
                col = float(np.clip(1 - np.abs(rgb[y:y+size, x:x+size].mean((0, 1)) - color).mean() / 96, 0, 1))
                y0, y1, x0, x1 = y//4, (y+size)//4, x//4, (x+size)//4
                dino_coherence, _, _ = coherence(appearance[y0:y1, x0:x1].reshape(-1, 384))
                affinity_coherence, mean, std = coherence(z[y0:y1, x0:x1].reshape(-1, 64))
                geometry_field = features['geometry'][y0:y1, x0:x1].reshape(-1, 4)
                weight = geometry_field[:, 3]
                consistency = float(np.linalg.norm((geometry_field[:, :2] * weight[:, None]).sum(0)) / max(weight.sum(), 1e-8))
                context.append(np.concatenate([mean, std, [dino_coherence, affinity_coherence,
                    homogeneous, geometry['confidence'], consistency, margin, col, occupancy, size/96,
                    float(np.asarray(patch.convert('L')).std()/255)]]))
                a0 = .25*homogeneous + .20*geometry['confidence'] + .20*margin + .20*col + .10*size/96 + .05*occupancy
                a3 = .60*affinity_coherence + .20*geometry['confidence'] + .20*consistency
                baseline.append([a0, dino_coherence, affinity_coherence, a3])
                boxes.append(box)
    # 某些细窄参考没有满足前景纯度的候选；保留空集，不偷用标注生成候选。
    return dict(boxes=np.asarray(boxes, dtype=np.int32).reshape(-1, 4),
                features=np.asarray(context, dtype=np.float32).reshape(-1, 138),
                baselines=np.asarray(baseline, dtype=np.float32).reshape(-1, 4), foreground=fg)


def supervised_arrays(row, cache, out):
    support = np.asarray(Image.open(out/'visual_annotations'/row['pattern_support'])) > 0
    purity, overlaps = [], []
    for box in cache['boxes']:
        x, y, x1, y1 = box
        purity.append(float(support[y:y1, x:x1].mean()))
        overlaps.append(max([iou(box, e['box']) for e in row['evidence']] or [0.]))
    purity, overlaps = np.asarray(purity), np.asarray(overlaps)
    targets = purity * (.5 + .5*overlaps)
    # invalid reference由视觉明确无可靠evidence；valid参考未标区域只给低权重。
    weights = np.where((purity >= .90) | (not row['valid_evidence']), 1., .1)
    return targets.astype(np.float32), weights.astype(np.float32)


def train(rows, caches, out, seed, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    selected = [r for r in rows if r['group'] == 'train' and len(caches[r['id']]['boxes'])]
    fields = [caches[r['id']]['features'] for r in selected]
    # 每参考先求均值，再计算归一化尺度，避免大衣候选数多而占据统计量。
    mean = np.mean([x.mean(0) for x in fields], 0)
    std = np.maximum(np.sqrt(np.mean([((x-mean)**2).mean(0) for x in fields], 0)), .02)
    inputs = [torch.tensor(np.clip((x-mean)/std, -8, 8), device=device) for x in fields]
    truth = [supervised_arrays(r, caches[r['id']], out) for r in selected]
    targets = [torch.tensor(t[0], device=device) for t in truth]
    weights = [torch.tensor(t[1], device=device) for t in truth]
    head = EvidenceHead(len(mean)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=SETTINGS['lr'], weight_decay=SETTINGS['weight_decay'])
    losses = []
    for step in range(SETTINGS['steps']):
        ref_indices = rng.integers(0, len(selected), SETTINGS['batch_references'])
        choices = [(int(r), int(rng.integers(len(fields[r])))) for r in ref_indices]
        x = torch.stack([inputs[r][c] for r, c in choices])
        y = torch.stack([targets[r][c] for r, c in choices])
        w = torch.stack([weights[r][c] for r, c in choices])
        optimizer.zero_grad()
        loss = (F.binary_cross_entropy_with_logits(head(x), y, reduction='none')*w).sum()/w.sum()
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == SETTINGS['steps']-1:
            losses.append(dict(step=step+1, loss=float(loss.item())))
            print('[E34-A train]', seed, losses[-1], flush=True)
    folder = out/'A_evidence'/('seed%d' % seed)
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder/'head.pt'
    torch.save(dict(model=head.state_dict(), mean=mean, std=std, seed=seed, train_ids=[r['id'] for r in selected], settings=SETTINGS), checkpoint)
    write(folder/'training.json', dict(steps=SETTINGS['steps'], train_reference_count=len(selected),
        valid_train_reference_count=sum(r['valid_evidence'] for r in selected), heldout_used=False,
        losses=losses, checkpoint_sha256=sha(checkpoint)))
    return head.eval(), mean, std


def evaluate(row, image, cache, scores, out, arm, seed):
    gt = [e['box'] for e in row['evidence']]
    chosen = nms_top(scores, cache['boxes']) if len(scores) else []
    result = dict(id=row['id'], group=row['group'], seed=seed, arm=arm,
        family=row['pattern_family'], valid_evidence=row['valid_evidence'], candidate_count=len(scores),
        selected=[dict(box=cache['boxes'][j].tolist(), score=float(scores[j])) for j in chosen],
        top1_iou=0., top5_recall=0., coverage=0., contamination_upper=1., contamination_lower=1.,
        homogeneity=0., identity=0., affinity_identity=0., orientation_error_deg=None,
        oracle_candidate_iou=max([max([iou(b,g) for g in gt] or [0.]) for b in cache['boxes']] or [0.]))
    if not row['valid_evidence']:
        result['invalid_reference_selected'] = bool(chosen)
        result['invalid_reference_score_ge_half'] = bool(chosen and scores[chosen[0]] >= .5)
        return result
    if not chosen:
        return result
    j = chosen[0]
    box = cache['boxes'][j].tolist()
    x, y, x1, y1 = box
    best_gt = max(range(len(gt)), key=lambda k: iou(box, gt[k]))
    support = np.asarray(Image.open(out/'visual_annotations'/row['pattern_support'])) > 0
    purity = float(support[y:y1, x:x1].mean())
    patch = image.crop(box)
    identity = similarity(patch, image.crop(gt[best_gt]))[0]
    top1_iou = iou(box, gt[best_gt])
    result.update(top1_iou=top1_iou,
        top5_recall=float(np.mean([max(iou(cache['boxes'][c], g) for c in chosen) >= .5 for g in gt])),
        coverage=float(top1_iou >= .5 and purity >= .90), contamination_upper=1-purity,
        contamination_lower=1-float(cache['foreground'][y:y1,x:x1].mean()),
        homogeneity=_homogeneity(patch), identity=identity)
    # 对同参考中最接近GT的候选作额外冻结affinity诊断，不用它替代GT纹样。
    candidate_nearest_gt = max(range(len(scores)), key=lambda c: iou(cache['boxes'][c], gt[best_gt]))
    a, b = cache['features'][j,:64], cache['features'][candidate_nearest_gt,:64]
    result['affinity_identity'] = float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-8))
    if row['orientation_readable']:
        prediction = patch_geometry(patch)
        result['orientation_error_deg'] = float(abs((prediction['orientation']-row['evidence'][best_gt]['orientation_deg']+90)%180-90)) if prediction['valid'] else 90.
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for g in gt:
        draw.rectangle(g, outline='lime', width=2)
    draw.rectangle(box, outline='red', width=2)
    destination = out/'A_evidence'/('seed%d' % seed)/'selected'/arm/row['group']
    destination.mkdir(parents=True, exist_ok=True)
    overlay.save(destination/(row['id']+'.png'))
    patch.save(destination/(row['id']+'_crop.png'))
    return result


def bootstrap_difference(a4, baseline, field='top1_iou'):
    # seed先在reference内平均，再以独立reference配对抽样。
    identities = sorted(set(r['id'] for r in a4 if r['valid_evidence']))
    differences = []
    for identity in identities:
        av = np.mean([r[field] for r in a4 if r['id']==identity])
        bv = np.mean([r[field] for r in baseline if r['id']==identity])
        differences.append(float(av-bv))
    rng = np.random.default_rng(34042)
    samples = [float(np.mean(rng.choice(differences, len(differences), replace=True))) for _ in range(2000)]
    return dict(unit='reference; seeds averaged inside reference', n=len(identities),
                mean=float(np.mean(differences)), ci95=np.percentile(samples,[2.5,97.5]).tolist())


def summarize(results):
    summary = {}
    for group in ('validation', 'independent_test'):
        summary[group] = {}
        for arm in ('A0','A1','A2','A3','A4'):
            rows = [r for r in results if r['group']==group and r['arm']==arm]
            valid = [r for r in rows if r['valid_evidence']]
            per_ref = {}
            for identity in sorted(set(r['id'] for r in valid)):
                selected = [r for r in valid if r['id']==identity]
                per_ref[identity] = {metric:float(np.mean([r[metric] for r in selected])) for metric in
                    ('top1_iou','top5_recall','coverage','contamination_upper','contamination_lower','homogeneity','identity')}
            def values(k):
                return [v[k] for v in per_ref.values()]
            invalid = [r for r in rows if not r['valid_evidence']]
            summary[group][arm] = dict(evaluable_references=len(per_ref),
                recruited_references=len(set(r['id'] for r in rows)), median_crop_iou=float(np.median(values('top1_iou'))),
                top5_recall=float(np.mean(values('top5_recall'))), coverage=float(np.mean(values('coverage'))),
                contamination_upper=float(np.mean(values('contamination_upper'))),
                contamination_lower=float(np.mean(values('contamination_lower'))),
                homogeneity=float(np.mean(values('homogeneity'))),identity=float(np.mean(values('identity'))),
                invalid_selected_rate=float(np.mean([r['invalid_reference_selected'] for r in invalid])) if invalid else None,
                invalid_score_ge_half_rate=float(np.mean([r['invalid_reference_score_ge_half'] for r in invalid])) if invalid else None)
    validation = [r for r in results if r['group']=='validation']
    a4 = [r for r in validation if r['arm']=='A4']
    comparisons = {arm:bootstrap_difference(a4,[r for r in validation if r['arm']==arm]) for arm in ('A0','A3')}
    m = summary['validation']['A4']
    gate = PROTOCOL['A_gate']
    checks = {k: m[k]>=gate[k] for k in ('median_crop_iou','identity','homogeneity')}
    checks.update(contamination=m['contamination_upper']<=gate['contamination'],
        top5_recall=m['top5_recall']>=gate['top5_recall'], evidence_coverage=m['coverage']>=gate['evidence_coverage'],
        A4_better_than_A0=comparisons['A0']['ci95'][0]>0,
        A4_not_below_A3=comparisons['A3']['mean']>=0)
    stop = dict(evidence_coverage=m['coverage']<.60, contamination=m['contamination_upper']>.15,
                top5_recall=m['top5_recall']<.70)
    return dict(metrics=summary, paired_crop_iou_comparisons=comparisons, validation_checks=checks,
                exploratory_A_quality_gate_pass=all(checks.values()), A_hard_stop=any(stop.values()),
                stop_checks=stop, original_human_annotation_gate_pass=False,
                formal_E34_B_authorized_by_evidence=False, generated_diffusion_images=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('/share/home/u2515283058/datasets/BF'))
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    out = args.out
    (out/'A_evidence').mkdir(parents=True,exist_ok=True)
    stage0 = read(out/'decision_summary.json')
    for key, value in stage0['checks'].items():
        if key not in ('human_annotations_ready','new_reference_rot90_integrity_ready') and not value:
            raise ValueError('Inherited integrity check failed: '+key)
    code_sha = subprocess.check_output(['git','rev-parse','HEAD'],universal_newlines=True).strip()
    source = Path('assets/e34_visual_annotations.json')
    rows, annotations = materialize(source,args.dataset,out)
    localized = intervention_audit(rows,args.dataset,out)
    settings = dict(SETTINGS,code_sha=code_sha,annotation_sha256=sha(source),
                    inherited_original_protocol_sha256=sha(out/'protocol/protocol.json'),
                    source_plan_sha256=PROTOCOL['plan_sha256'], frozen_E26_sha256=sha('models/local_pattern_field.py'),
                    frozen_E5_sha256=sha('output/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt'))
    write(out/'protocol/visual_annotation_amendment.json', settings)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4)
    weights = Path('output_eval/e30_apacc_20260930/dinov2_vits14_pretrain.pth')
    dino,dino_sha = load_dino(device,weights)
    checkpoint = E30/'A21_embedding/seed42/adapter.pt'
    ckpt = torch.load(checkpoint,map_location=device)
    adapter = PatternAffinityAdapter(ckpt['input_dim']-384,not ckpt['no_geometry']).to(device).eval()
    adapter.load_state_dict(ckpt['model'])
    for p in adapter.parameters():
        p.requires_grad_(False)
    settings.update(frozen_dino_sha256=dino_sha,frozen_affinity_sha256=sha(checkpoint))
    write(out/'protocol/visual_annotation_amendment.json',settings)
    cache_dir = out/'A_evidence/cache'
    cache_dir.mkdir(parents=True,exist_ok=True)
    caches = {}
    normalization = norm_at(E30)
    for index,row in enumerate(rows):
        path = cache_dir/(row['id']+'.npz')
        signature = hashlib.sha256((row['reference_sha256']+dino_sha+sha(checkpoint)+sha('tools/e34_evidence.py')).encode()).hexdigest()
        if path.exists():
            with np.load(path) as data:
                valid_cache = str(data['signature'])==signature
                if valid_cache:
                    cache = {k:data[k] for k in ('boxes','features','baselines','foreground')}
        else:
            valid_cache = False
        if not valid_cache:
            image = Image.open(args.dataset/row['reference']).convert('RGB')
            cache = cache_reference(row,image,dino,adapter,normalization,device)
            np.savez_compressed(path,**cache,signature=signature)
        caches[row['id']] = cache
        print('[E34-A cache]',index+1,'/140',row['id'],len(cache['boxes']),flush=True)
    # candidate覆盖上界在学习前计算；仍运行固定训练以审计各对照，无结果驱动重招募。
    oracle = []
    for row in rows:
        boxes=caches[row['id']]['boxes'];gt=[e['box'] for e in row['evidence']]
        oracle.append(dict(id=row['id'],group=row['group'],valid=row['valid_evidence'],
            candidate_count=len(boxes),small_evidence_count=sum(g[2]-g[0]<48 for g in gt),
            top1_iou_upper=max([max([iou(b,g) for g in gt] or [0.]) for b in boxes] or [0.]),
            recall_upper=float(np.mean([max([iou(b,g) for b in boxes] or [0.])>=.5 for g in gt])) if gt else None))
    write(out/'A_evidence/candidate_oracle.json',oracle)
    results=[]
    for seed in (42,43,44):
        head,mean,std=train(rows,caches,out,seed,device)
        for row in rows:
            if row['group']=='train':
                continue
            image=Image.open(args.dataset/row['reference']).convert('RGB')
            cache=caches[row['id']]
            with torch.inference_mode():
                a4=torch.sigmoid(head(torch.tensor(np.clip((cache['features']-mean)/std,-8,8),device=device))).cpu().numpy()
            for column,arm in enumerate(('A0','A1','A2','A3','A4')):
                scores=a4 if arm=='A4' else cache['baselines'][:,column]
                results.append(evaluate(row,image,cache,scores,out,arm,seed))
        write(out/'A_evidence'/('seed%d'%seed)/'rows.json',[r for r in results if r['seed']==seed])
    summary=summarize(results)
    summary.update(code_sha=code_sha,annotation_audit=annotations,
        localized_intervention={k:v for k,v in localized.items() if k!='rows'},
        training_steps_per_seed=1500,total_training_steps=4500,reference_count=140,
        frozen_hashes_unchanged=sha(weights)==dino_sha and sha(checkpoint)==settings['frozen_affinity_sha256']
            and sha('models/local_pattern_field.py')==settings['frozen_E26_sha256']
            and sha('output/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt')==settings['frozen_E5_sha256'])
    write(out/'A_evidence/summary.json',summary)
    write(out/'A_evidence/status.json',dict(status='completed_exploratory',gate_pass=summary['exploratory_A_quality_gate_pass'],
        hard_stop=summary['A_hard_stop'],annotation_source='assistant_visual_review'))
    write(out/'A_evidence/completion_check.json',dict(completed=True,seed_count=3,arms=5,heldout_result_rows=len(results),
        expected_rows=40*5*3,all_rows_present=len(results)==40*5*3,generated_diffusion_images=0,
        original_human_gate_pass=False,full_reference_intervention_gate_pass=False))
    for stage in ('B_canonical','C_field_ablation','D_injection','E_end2end','F_causal','G_generalization'):
        write(out/stage/'status.json',dict(status='not_run',reason='A quality gate / full-reference intervention / original human GT requirement unmet'))
    write(out/'A_evidence/artifact_manifest.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*')
        if p.is_file() and (p.parent==out/'A_evidence' or 'A_evidence' in p.parts or 'visual_annotations' in p.parts)
        and p.name!='artifact_manifest.json' and p.suffix not in ('.npz','.pt','.zip','.log','.err')})
    with ZipFile(out/'e34_A_visual_review.zip','w',ZIP_DEFLATED) as archive:
        for folder in ('visual_annotations','A_evidence'):
            for path in sorted((out/folder).rglob('*')):
                if path.is_file() and path.suffix not in ('.npz','.pt'):
                    archive.write(path,str(path.relative_to(out)))
        archive.write(out/'protocol/visual_annotation_amendment.json','protocol/visual_annotation_amendment.json')
    print('[E34-A summary]',json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
