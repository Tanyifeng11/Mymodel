"""E31 Stage A：冻结 anchor mining、真实增强、六项消融和条件式 Gate。"""

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance
from scipy.optimize import linear_sum_assignment

from models.apacc_affinity_adapter import PatternAffinityAdapter
from models.apacc_features import extract_dense_features, load_dino
from models.e29_crop_selector import select_crop_e29
from models.e31_anchor_candidates import candidate_boxes, patch_field, unit
from models.e31_anchor_scoring import describe_candidate, eligible_candidates, WEIGHTS
from models.e31_anchor_selection import select_anchors
from models.e31_soft_support import support_fields, coverage
from models.local_pattern_field import patch_geometry
from tools.e30_a2_affinity import extra_features, tensor_field, load_feature, norm_at


ARMS = ('B0_rule', 'B1_top1', 'B2_topk', 'B3_soft', 'B4_soft_unknown')
ABLATIONS = ('A_no_diversity', 'B_no_geometry', 'C_no_spatial', 'D_no_unknown',
             'E_no_richness', 'F_dino_support')
AUGS = ('brightness', 'color_jitter', 'translation', 'resize')
VARIANTS = ('original',) + AUGS + ('rot90', 'pattern_rot90')
CONFIG = dict(Kmax=4, stride=16, foreground_occupancy=.95, min_boundary_distance=3,
              canonicality_min=.50, richness_min=.25, pattern_gray_std_min=.025,
              plain_foreground_highpass_max=.012, unknown_score=.65,
              temperatures=[.08, .10, .15], default_temperature=.10,
              lambda_d=.30, lambda_s=.10, redundancy_cosine=.95,
              coverage_gain_min=.05, support_coverage_threshold=.50,
              score_weights=WEIGHTS, support_weights=dict(embedding=.50, orientation=.15,
              frequency=.15, color=.05, spatial=.15), revision=0, training_steps=0)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def ci(values):
    if not values:
        return None
    values = np.asarray(values, float)
    draws = np.random.default_rng(42).choice(values, (2000, len(values)), replace=True).mean(1)
    return np.percentile(draws, [2.5, 97.5]).tolist()


def load_adapter(source, device):
    ckpt = torch.load(source/'A21_embedding/seed42/adapter.pt', map_location='cpu', weights_only=False)
    model = PatternAffinityAdapter(ckpt['input_dim']-384, not ckpt['no_geometry'])
    model.load_state_dict(ckpt['model'], strict=True)
    return model.to(device).eval().requires_grad_(False)


def embed(features, model, norm, device):
    values = tensor_field(features, norm)
    with torch.inference_mode():
        return model(torch.from_numpy(values).to(device)).cpu().numpy()


def augment(image, mask, name):
    rgb = np.asarray(image)
    identity = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
    matrix = identity.copy()
    if name == 'brightness':
        rgb = np.asarray(ImageEnhance.Brightness(image).enhance(1.08))
    elif name == 'color_jitter':
        rgb = np.clip(rgb.astype(float)*[1.03, .97, 1.01]+[1, -1, 0], 0, 255).astype(np.uint8)
    elif name == 'translation':
        matrix[:, 2] = [4, -3]
    elif name == 'resize':
        matrix[:, :2] *= .96
        matrix[:, 2] = np.asarray([image.width, image.height])*.02
    elif name == 'rot90':
        matrix = np.array([[0, 1, 0], [-1, 0, image.width-1]], np.float32)
    if name in ('translation', 'resize', 'rot90'):
        rgb = cv2.warpAffine(rgb, matrix, image.size, flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
        mask = cv2.warpAffine(mask.astype(np.uint8), matrix, image.size, flags=cv2.INTER_NEAREST) > 0
    return Image.fromarray(rgb), mask, matrix


def cache_dir(args, index, variant):
    return args.out/'audits/cache'/('%03d'%index)/variant


def load_cached(args, index, variant):
    folder = cache_dir(args, index, variant)
    with np.load(folder/'features.npz') as d:
        f = {k: d[k].astype(np.float32) for k in d.files if k != 'embedding'}
        z = d['embedding'].astype(np.float32)
    meta = read(folder/'candidates.json')
    return Image.open(folder/'reference.png').convert('RGB'), f['foreground'] > 0, f, z, meta


def make_cache(args, index, variant, image, mask, matrix, f, z, baseline_box=None, probe=None):
    folder = cache_dir(args, index, variant)
    folder.mkdir(parents=True, exist_ok=True)
    image.save(folder/'reference.png')
    boxes, distance = candidate_boxes(mask)
    candidates = [describe_candidate(image, mask, distance, f, z, box) for box in boxes]
    if baseline_box is None:
        baseline_box, baseline_info = select_crop_e29(image, mask)
    else:
        baseline_info = None
    baseline = next((c for c in candidates if c['bbox'] == list(baseline_box)), None)
    if baseline is None:
        baseline = describe_candidate(image, mask, distance, f, z, tuple(baseline_box))
    np.savez_compressed(folder/'features.npz', **f, embedding=z)
    write(folder/'candidates.json', dict(candidates=candidates, B0=baseline,
          B0_rule_info=baseline_info, forward_matrix=matrix.tolist(), pattern_probe=probe))


def prepare(args):
    args.out.mkdir(parents=True, exist_ok=True)
    source_protocol = read(args.source/'protocol.json')
    assert sha(Path('models/local_pattern_field.py')) == source_protocol['e26_sha256']
    splits = read(args.source/'split_manifest.json')
    assert len(splits['dev']) == len(splits['stress_dev']) == 16
    assert len(set(splits['train']+splits['dev']+splits['stress_dev'])) == 96
    assert read(args.source/'A21_embedding/summary.json')['seeds']['42']['gate_pass']
    write(args.out/'split_manifest.json', splits)
    write(args.out/'anchor_config.json', CONFIG)
    for folder in ('A_anchor_mining', 'B_target_scaffold', 'C_generation', 'D_confirmation', 'audits'):
        (args.out/folder).mkdir(exist_ok=True)
    names = splits['dev']+splits['stress_dev']
    paths = [args.source/'A21_embedding/seed42/adapter.pt', args.source/'train_normalization.npz',
             args.source/'split_manifest.json', args.weights]
    paths += [args.source/'feature_cache'/('%03d.npz'%i) for i in range(64, 96)]
    paths += [args.dataset/'training'/name for name in names]
    paths += [Path(p) for p in ('models/apacc_affinity_adapter.py', 'models/apacc_features.py',
               'models/apacc_canonicality.py', 'models/local_pattern_field.py', 'models/e29_crop_selector.py',
               'inference_IMAGGarment-1.py', 'models/IMAGGarment.py', 'models/bf_texture_module.py',
               'models/tcpm_module.py', 'output/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt')
               if Path(p).exists()]
    frozen_files = {str(p): sha(p) for p in paths}
    write(args.out/'input_manifest.json', dict(frozen_files=frozen_files, source_protocol=source_protocol,
          source=str(args.source), git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
          training_steps=0, manual_annotations_used=False, E29_causal_test_used=False,
          reference_unit='reference image', bootstrap_resamples=2000))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dino, dino_sha = load_dino(device, args.weights)
    model, norm = load_adapter(args.source, device), norm_at(args.source)
    write(args.out/'protocol.json', dict(experiment='E31', device=device, dino_sha256=dino_sha,
          config=CONFIG, reference_count=32, seed=42, training_steps=0,
          B0='E29 rule selector on entire estimated foreground, annotation-free adapted baseline',
          B0_B1_B2_support='standardized auxiliary support readout for selection controls',
          stability='full feature recomputation for each real image transform; fixed anchors and reselected anchors; Hungarian descriptor matching; anchor-only soft IoU, unknown separately',
          augmentation=dict(brightness=1.08, color_jitter='RGB gain [1.03,.97,1.01], bias [1,-1,0]',
                            translation=[4, -3], resize=.96, rot90='whole-image spatial equivariance',
                            pattern_rot90='rotate inscribed square in first selected interior anchor; garment/mask unchanged'),
          geometry_gate='only references with >=2 readable quadrants and readable anchor; unreadable excluded and counted',
          contamination='outside 3px-eroded estimated foreground proxy; structural edges reported separately',
          stress_dev='inherited deterministic unstratified split', temperature_selection='dev only, prespecified max min(coverage, end_to_end_soft_iou), tie prefer .10',
          hard_segmentation_used=False))
    for index, name in enumerate(names, 64):
        original = Image.open(args.dataset/'training'/name).convert('RGB').resize((256, 256))
        original_f = load_feature(args.source, index)
        original_mask = original_f['foreground'] > 0
        for variant in VARIANTS:
            folder = cache_dir(args, index, variant)
            if (folder/'candidates.json').exists():
                continue
            probe = None
            image, mask, matrix = augment(original, original_mask, variant)
            if variant == 'pattern_rot90':
                oi, om, of, oz, meta = load_cached(args, index, 'original')
                candidates, mode = eligible_candidates(meta['candidates'], oi, om)
                anchors, _ = select_anchors(candidates, of, oz, om, mode=mode)
                if anchors:
                    x0, y0, x1, y1 = anchors[0]['bbox']
                    size = min(x1-x0, y1-y0)
                    cx, cy = (x0+x1)//2, (y0+y1)//2
                    box = (cx-size//2, cy-size//2, cx+size//2, cy+size//2)
                    crop = oi.crop(box)
                    rotated = crop.transpose(Image.Transpose.ROTATE_90)
                    image = oi.copy()
                    image.paste(rotated, box[:2])
                    old, new = patch_geometry(crop), patch_geometry(rotated)
                    error = abs((new['orientation']-old['orientation']+0)%180-90)
                    probe = dict(box=list(box), anchor_bbox=anchors[0]['bbox'], readable=old['valid'] and new['valid'],
                                 orientation_error=float(error), period_log_error=float(abs(np.log(max(new['frequency'], 1e-4)/max(old['frequency'], 1e-4)))))
            if variant == 'original':
                f = {k: v for k, v in original_f.items() if k not in ('appearance_aug', 'appearance_rot')}
            else:
                f = extra_features(image, mask, extract_dense_features(image, mask, dino, device))
                f['foreground'] = mask.astype(np.float32)
            z = embed(f, model, norm, device)
            make_cache(args, index, variant, image, mask, matrix, f, z, probe=probe)
            print('[E31 prepare]', index, variant, flush=True)
    write(args.out/'audits/cache_manifest.json', dict(variants=list(VARIANTS), references=list(range(64, 96)),
          feature_sha256={str(p.relative_to(args.out)): sha(p) for p in (args.out/'audits/cache').rglob('features.npz')}))


def method(image, mask, features, z, meta, arm, temperature):
    no_richness = arm == 'E_no_richness'
    candidates, mode = eligible_candidates(meta['candidates'], image, mask, no_richness)
    options = dict(geometry=arm != 'B_no_geometry', spatial=arm != 'C_no_spatial',
                   dino=arm == 'F_dino_support', unknown=arm not in ('B3_soft', 'D_no_unknown'))
    # support 消融仅改传播，保持 B4 选出的 anchor，不混入 selection 改动。
    if arm == 'B0_rule':
        anchors, events = [meta['B0']], [dict(reason='E29_whole_foreground_rule')]
        mode = 'E29_rule'
    else:
        anchors, events = select_anchors(candidates, features, z, mask, mode=mode,
                 top1=arm == 'B1_top1', diversity=arm != 'A_no_diversity',
                 no_richness=no_richness, temperature=temperature)
    p, detail = support_fields(features, z, anchors, mask, temperature=temperature, **options)
    return anchors, events, mode, p, detail, options


def align_channels(original, augmented, p, q, mask, aug_mask, matrix, key='mean_embedding'):
    inverse = cv2.invertAffineTransform(np.asarray(matrix, np.float32))
    q = np.stack([cv2.warpAffine(q[..., c], inverse, (mask.shape[1], mask.shape[0])) for c in range(q.shape[-1])], -1)
    am = cv2.warpAffine(aug_mask.astype(np.uint8), inverse, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST) > 0
    n, m = len(original), len(augmented)
    order, used = {}, set()
    if n and m:
        sim = np.asarray([a[key] for a in original]) @ np.asarray([a[key] for a in augmented]).T
        a, b = linear_sum_assignment(-sim)
        order, used = dict(zip(a.tolist(), b.tolist())), set(b.tolist())
    extra = [j for j in range(m) if j not in used]
    pp = np.concatenate((p[..., :n], np.zeros((*mask.shape, len(extra))), p[..., -1:]), -1)
    qq = np.zeros_like(pp)
    for i, j in order.items():
        qq[..., i] = q[..., j]
    for k, j in enumerate(extra):
        qq[..., n+k] = q[..., j]
    qq[..., -1] = q[..., -1]
    # 被变换裁出视野的原前景计 unknown，不能从稳定性分母删去。
    qq[mask & ~am, -1] = 1
    return pp, qq, mask | am, float((mask & am).sum()/max(mask.sum(), 1)), order


def consistency(p, q, valid):
    a, b = p[valid].astype(float), q[valid].astype(float)
    anchor_a, anchor_b = a[:, :-1], b[:, :-1]
    union = np.maximum(anchor_a, anchor_b).sum()
    soft = float(np.minimum(anchor_a, anchor_b).sum()/union) if union > 1e-8 else 0.
    eps = 1e-8
    kl = float((a*np.log((a+eps)/(b+eps))).sum(-1).mean())
    return dict(soft_iou=soft, kl=kl, argmax_agreement=float((a.argmax(-1)==b.argmax(-1)).mean()),
                all_channel_soft_iou=float(np.minimum(a,b).sum()/max(np.maximum(a,b).sum(),1e-8)))


def audit_stability(args, index, original, arm, temperature):
    anchors, events, mode, p, detail, opts, image, mask, f, z, meta = original
    rows = []
    for name in AUGS+('rot90', 'pattern_rot90'):
        ai, am, af, az, metadata = load_cached(args, index, name)
        aa, _, _, ap, _, _ = method(ai, am, af, az, metadata, arm, temperature)
        pp, qp, valid, visible, order = align_channels(anchors, aa, p, ap, mask, am, metadata['forward_matrix'],
                                                      'mean_dino' if opts['dino'] else 'mean_embedding')
        stat = consistency(pp, qp, valid)
        # 固定 crop 几何位置，但 descriptor 重新从真实增强特征求出。
        transformed = []
        matrix = np.asarray(metadata['forward_matrix'], float)
        for anchor in anchors:
            c = dict(anchor)
            center = matrix @ np.r_[anchor['center'], 1]
            c['center'] = center.tolist()
            c['center_uv'] = (center/np.asarray(ai.size)).tolist()
            corners = matrix @ np.array([[anchor['bbox'][0], anchor['bbox'][1], 1],
                                         [anchor['bbox'][2]-1, anchor['bbox'][3]-1, 1]]).T
            x0,y0 = np.maximum(np.floor(corners.min(1)).astype(int), 0)
            x1,y1 = np.minimum(np.ceil(corners.max(1)).astype(int)+1, ai.size)
            c['bbox'] = [int(x0),int(y0),int(x1),int(y1)]
            if x1 <= x0 or y1 <= y0:
                continue
            c['mean_embedding'] = unit(patch_field(az, c['bbox'], ai.size).mean((0,1))).tolist()
            c['mean_dino'] = unit(patch_field(af['appearance'], c['bbox'], ai.size).mean((0,1))).tolist()
            c['lab'] = patch_field(af['color'][..., :3], c['bbox'], ai.size).mean((0,1)).tolist()
            c['metrics'] = dict(c['metrics'])
            geo = patch_geometry(ai.crop(tuple(c['bbox'])))
            c['metrics'].update(orientation_mean=geo['orientation'], frequency_mean=geo['frequency'],
                                geometry_confidence=geo['confidence'], geometry_readable=geo['valid'])
            transformed.append(c)
        fixedp, _ = support_fields(af, az, transformed, am, temperature=temperature, **opts)
        fp, fq, fv, _, _ = align_channels(anchors, transformed, p, fixedp, mask, am,
                                         metadata['forward_matrix'], 'mean_dino' if opts['dino'] else 'mean_embedding')
        fixed = consistency(fp, fq, fv)
        stat.update(variant=name, fixed_anchor_soft_iou=fixed['soft_iou'], visible_source_fraction=visible,
                    original_k=len(anchors), augmented_k=len(aa), channel_assignment=order)
        if name == 'pattern_rot90' and metadata['pattern_probe']:
            probe = metadata['pattern_probe']
            box = probe['box']
            before = unit(patch_field(z, box).mean((0,1)))
            after = unit(patch_field(az, box).mean((0,1)))
            dbefore = unit(patch_field(f['appearance'], box).mean((0,1)))
            dafter = unit(patch_field(af['appearance'], box).mean((0,1)))
            localmask = np.zeros_like(mask)
            localmask[box[1]:box[3],box[0]:box[2]] = True
            stat['pattern_probe'] = dict(probe, learned_identity=float(np.dot(before,after)),
                dino_identity=float(np.dot(dbefore,dafter)),
                local_support_soft_iou=consistency(pp,qp,localmask)['soft_iou'])
        rows.append(stat)
    return rows


def row_metrics(index, arm, anchors, mode, p, mask, candidates, stability):
    readable = [a['metrics'] for a in anchors if a['metrics']['geometry_readable']]
    # 每个 reference 内先聚合，缺失 reference 不能被排除出 validity/identity。
    metrics = dict(index=index, split='dev' if index < 80 else 'stress_dev', arm=arm,
                   k=len(anchors), mode=mode, candidate_count=len(candidates), valid_anchor=bool(anchors),
                   foreground_coverage=coverage(p,mask), ambiguity=1-coverage(p,mask),
                   unknown_mass=float(p[..., -1][mask].mean()),
                   unknown_majority=float((p[..., -1][mask]>.5).mean()),
                   dominance=float(p[..., :-1][mask].mean(0).max()) if anchors else 0.,
                   identity=float(np.mean([a['metrics']['identity'] for a in anchors])) if anchors else 0.,
                   contamination=float(np.mean([a['metrics']['contamination'] for a in anchors])) if anchors else 1.,
                   homogeneity=float(np.mean([a['metrics']['homogeneity'] for a in anchors])) if anchors else 0.,
                   canonicality=float(np.mean([a['metrics']['canonicality'] for a in anchors])) if anchors else 0.,
                   pattern_richness=float(np.mean([a['metrics']['pattern_richness'] for a in anchors])) if anchors else 0.,
                   structural_edge=float(np.mean([a['metrics']['structural_edge'] for a in anchors])) if anchors else 0.,
                   orientation_consistency=float(np.mean([a['orientation_consistency'] for a in readable])) if readable else None,
                   period_consistency=float(np.mean([a['period_consistency'] for a in readable])) if readable else None,
                   readable_anchor_count=len(readable))
    pairs = [float(np.dot(a['mean_embedding'],b['mean_embedding'])) for i,a in enumerate(anchors) for b in anchors[i+1:]]
    metrics.update(redundancy=float(np.mean(pairs)) if pairs else None,
                   pairwise_similarity_median=float(np.median(pairs)) if pairs else None)
    regular = [r for r in stability if r['variant'] in AUGS]
    metrics.update(support_soft_iou=float(np.mean([r['soft_iou'] for r in regular])) if regular else None,
                   support_soft_iou_worst=float(min(r['soft_iou'] for r in regular)) if regular else None,
                   fixed_anchor_soft_iou=float(np.mean([r['fixed_anchor_soft_iou'] for r in regular])) if regular else None,
                   support_kl=float(np.mean([r['kl'] for r in regular])) if regular else None,
                   argmax_agreement=float(np.mean([r['argmax_agreement'] for r in regular])) if regular else None,
                   probability_sum_max_error=float(abs(p[mask].sum(-1)-1).max()),
                   outside_probability_max=float(p[~mask].max()) if (~mask).any() else 0.)
    return metrics


def heat(field, mask):
    field = cv2.resize(np.asarray(field,np.float32), (mask.shape[1],mask.shape[0]))
    rgb = cv2.applyColorMap(np.uint8(np.clip(field,0,1)*255),cv2.COLORMAP_TURBO)[...,::-1].copy()
    rgb[~mask] = 245
    return Image.fromarray(rgb)


def visualize(folder, image, mask, candidates, anchors, p, detail, row):
    folder.mkdir(parents=True,exist_ok=True)
    image.save(folder/'reference.png')
    quality = np.zeros(mask.shape,np.float32)
    for c in candidates:
        x0,y0,x1,y1 = c['bbox']
        quality[y0:y1,x0:x1] = np.maximum(quality[y0:y1,x0:x1],c['metrics']['canonicality'])
    visuals = [('reference',image),('candidate canonicality',heat(quality,mask))]
    view = image.copy()
    draw = ImageDraw.Draw(view)
    palette = [(230,40,40),(30,140,250),(40,190,80),(200,50,230)]
    for i,a in enumerate(anchors):
        draw.rectangle(a['bbox'],outline=palette[i%4],width=3)
        draw.text(a['center'],str(i+1),fill=palette[i%4])
        crop = image.crop(tuple(a['bbox']))
        crop.save(folder/('anchor_%d.png'%i))
        visuals.append(('A%d Q=%.3f rich=%.3f'%(i+1,a['metrics']['canonicality'],a['metrics']['pattern_richness']),crop.resize((256,256))))
    view.save(folder/'topk.png')
    visuals.insert(2,('selected anchors K=%d'%len(anchors),view))
    for i,(_,d) in enumerate(detail):
        for name in ('embedding','orientation','frequency'):
            value = (d[name]+1)/2 if name=='embedding' else d[name]
            pic = heat(value,mask)
            pic.save(folder/('anchor_%d_%s.png'%(i,name)))
            visuals.append(('A%d %s'%(i+1,name),pic))
        pic = heat(p[...,i],mask)
        pic.save(folder/('anchor_%d_support.png'%i))
        visuals.append(('A%d soft support'%(i+1),pic))
    unknown = heat(p[...,-1],mask)
    unknown.save(folder/'unknown.png')
    visuals.append(('unknown',unknown))
    coverage_map = (p[...,:-1].max(-1)>=.5) if anchors else np.zeros_like(mask)
    covered = heat(coverage_map,mask)
    covered.save(folder/'coverage.png')
    visuals.append(('coverage %.1f%%'%(100*row['foreground_coverage']),covered))
    labels = p.argmax(-1)
    color = np.full((*mask.shape,3),245,np.uint8)
    for i in range(len(anchors)):
        color[(labels==i)&mask] = palette[i%4]
    color[(labels==len(anchors))&mask] = 100
    argmax = Image.fromarray(color)
    argmax.save(folder/'argmax_visualization_only.png')
    visuals.append(('argmax visualization only',argmax))
    cols, width, height = 5, 256, 282
    sheet = Image.new('RGB',(cols*width,((len(visuals)+cols-1)//cols)*height),'white')
    draw = ImageDraw.Draw(sheet)
    for i,(title,pic) in enumerate(visuals):
        x,y=(i%cols)*width,(i//cols)*height
        sheet.paste(pic.resize((width,width)),(x,y+26))
        draw.text((x+3,y+5),title,fill='black')
    sheet.save(folder/'audit.png')


def evaluate_reference(args,index,arm,temperature,save=True,stability=True):
    image,mask,f,z,meta = load_cached(args,index,'original')
    anchors,events,mode,p,detail,opts = method(image,mask,f,z,meta,arm,temperature)
    diagnostics = audit_stability(args,index,(anchors,events,mode,p,detail,opts,image,mask,f,z,meta),arm,temperature) if stability else []
    row = row_metrics(index,arm,anchors,mode,p,mask,meta['candidates'],diagnostics)
    if save:
        base = 'A_anchor_mining' if arm in ARMS else 'audits/ablations'
        folder = args.out/base/arm
        write(folder/('%03d.json'%index),dict(metrics=row,anchors=anchors,selection_events=events,augmentations=diagnostics))
        np.savez_compressed(folder/('%03d_support.npz'%index),probabilities=p,foreground=mask,
                            raw_scores=np.stack([s for s,_ in detail],-1) if detail else np.empty((64,64,0)))
        if arm=='B4_soft_unknown' or arm=='B0_rule':
            visualize(folder/'visualizations'/('%03d'%index),image,mask,meta['candidates'],anchors,p,detail,row)
    return row


def summarize(rows):
    report = {}
    keys = ['foreground_coverage','ambiguity','unknown_mass','unknown_majority','dominance','identity','contamination',
            'homogeneity','canonicality','pattern_richness','structural_edge','orientation_consistency','period_consistency',
            'redundancy','pairwise_similarity_median','support_soft_iou','support_soft_iou_worst','fixed_anchor_soft_iou',
            'support_kl','argmax_agreement']
    for split in ('dev','stress_dev','combined'):
        data = [r for r in rows if split=='combined' or r['split']==split]
        s = dict(reference_count=len(data),valid_anchor_reference_coverage=float(np.mean([r['valid_anchor'] for r in data])),
                 valid_anchor_count=sum(r['valid_anchor'] for r in data),mean_k=float(np.mean([r['k'] for r in data])),
                 k_histogram={str(k):sum(r['k']==k for r in data) for k in range(5)},
                 multi_anchor_references=sum(r['k']>1 for r in data),
                 fallback_references=[r['index'] for r in data if r['mode']=='low_frequency_fallback'])
        for k in keys:
            v = [r[k] for r in data if r[k] is not None]
            s[k] = float(np.mean(v)) if v else None
            s[k+'_ci95'] = ci(v)
            s[k+'_reference_count'] = len(v)
        pairs = [r['pairwise_similarity_median'] for r in data if r['pairwise_similarity_median'] is not None]
        s['median_pairwise_similarity'] = float(np.median(pairs)) if pairs else None
        report[split] = s
    return report


def choose_temperature(args):
    results={}
    for t in CONFIG['temperatures']:
        rows = [evaluate_reference(args,i,'B4_soft_unknown',t,save=False) for i in range(64,80)]
        results[str(t)] = dict(summary=summarize(rows)['dev'],rows=rows)
        print('[E31 temperature]',t,results[str(t)]['summary']['foreground_coverage'],results[str(t)]['summary']['support_soft_iou'],flush=True)
    selected=max(CONFIG['temperatures'],key=lambda t:(min(results[str(t)]['summary']['foreground_coverage'],
                        results[str(t)]['summary']['support_soft_iou']), -abs(t-.10)))
    write(args.out/'audits/temperature_selection.json',dict(selected=selected,dev_only=True,results=results,
          criterion='maximize min(dev coverage@.5, dev end-to-end anchor-only soft IoU), tie nearest .10'))
    return selected


def run(args):
    temperature = read(args.out/'audits/temperature_selection.json')['selected'] if (args.out/'audits/temperature_selection.json').exists() else choose_temperature(args)
    for arm in ARMS+ABLATIONS:
        rows=[]
        base='A_anchor_mining' if arm in ARMS else 'audits/ablations'
        for index in range(64,96):
            path=args.out/base/arm/('%03d.json'%index)
            row=read(path)['metrics'] if path.exists() else evaluate_reference(args,index,arm,temperature)
            rows.append(row)
            print('[E31 evaluate]',arm,index,'K',row['k'],'coverage',round(row['foreground_coverage'],4),'stability',round(row['support_soft_iou'],4),flush=True)
        write(args.out/base/arm/'rows.json',rows)
        write(args.out/base/arm/'summary.json',summarize(rows))
    report(args)


def gates(full,baseline):
    anchor_checks,support_checks={},{}
    hard=[]
    for split in ('dev','stress_dev'):
        s,b=full[split],baseline[split]
        anchor_checks[split+'/valid_reference_coverage'] = s['valid_anchor_reference_coverage']>=.80
        anchor_checks[split+'/redundancy'] = s['median_pairwise_similarity'] is None or s['median_pairwise_similarity']<=.90
        anchor_checks[split+'/identity'] = s['identity']>=b['identity']-.02
        anchor_checks[split+'/contamination'] = s['contamination']<=b['contamination']+1e-8
        for key in ('orientation_consistency','period_consistency'):
            anchor_checks[split+'/'+key] = s[key] is None or s[key]>=.88
        support_checks[split+'/coverage'] = s['foreground_coverage']>=.80
        support_checks[split+'/ambiguity'] = s['ambiguity']<=.20
        support_checks[split+'/stability'] = s['support_soft_iou']>=.80
        for key,value in (('valid_anchor_reference_coverage',s['valid_anchor_reference_coverage']<.80),
                          ('foreground_coverage',s['foreground_coverage']<.80),
                          ('ambiguity',s['ambiguity']>.30),('support_soft_iou',s['support_soft_iou']<.70)):
            if value: hard.append(split+'/'+key)
    return dict(anchor_checks=anchor_checks,support_checks=support_checks,
                anchor_pass=all(anchor_checks.values()),support_pass=all(support_checks.values()),
                **{'pass':all(anchor_checks.values()) and all(support_checks.values())},hard_stop_reasons=hard)


def report(args):
    rows = read(args.out/'A_anchor_mining/B4_soft_unknown/rows.json')
    full = summarize(rows)
    base = read(args.out/'A_anchor_mining/B0_rule/summary.json')
    gate = gates(full,base)
    # 硬停止优先，不能用无限次温度/阈值试验绕过。
    route = 'stop_annotation_free_local_correspondence' if gate['hard_stop_reasons'] else (
            'E31_B_target_correspondence' if gate['pass'] else
            'anchor_scoring_revision' if not gate['anchor_pass'] else 'support_metric_revision')
    s = full['combined']
    decision=dict(anchor_mining_pass=gate['anchor_pass'],support_pass=gate['support_pass'],stage_A_pass=gate['pass'],
        scaffold_pass=None,generation_pass=None,confirmation_pass=None,
        anchor_valid_reference_coverage=s['valid_anchor_reference_coverage'],foreground_support_coverage=s['foreground_coverage'],
        support_ambiguity=s['ambiguity'],anchor_redundancy=s['median_pairwise_similarity'],
        orientation_consistency=s['orientation_consistency'],period_consistency=s['period_consistency'],
        support_stability=s['support_soft_iou'],oracle_gap_recovery_follow=None,oracle_gap_recovery_orientation=None,
        next_route=route,hard_stop_reasons=gate['hard_stop_reasons'],training_steps=0,revision=0)
    write(args.out/'A_anchor_mining/gate.json',dict(gate,summary=full,baseline=base))
    write(args.out/'decision_summary.json',decision)
    for folder in ('B_target_scaffold','C_generation','D_confirmation'):
        write(args.out/folder/'status.json',dict(status='pending_gate_pass' if gate['pass'] else 'not_run_stage_A_gate_failed'))
    paired={}
    reference={r['index']:r for r in rows}
    for arm in ARMS+ABLATIONS:
        folder=args.out/('A_anchor_mining' if arm in ARMS else 'audits/ablations')/arm
        others=read(folder/'rows.json')
        paired[arm]={}
        for split in ('dev','stress_dev','combined'):
            selected=[r for r in others if split=='combined' or r['split']==split]
            paired[arm][split]={}
            for key in ('foreground_coverage','support_soft_iou','identity','contamination','pattern_richness','k'):
                delta=[r[key]-reference[r['index']][key] for r in selected]
                paired[arm][split][key]=dict(mean=float(np.mean(delta)),ci95=ci(delta))
    write(args.out/'audits/paired_comparisons.json',dict(reference_unit='reference image',bootstrap_resamples=2000,
          comparison='arm minus B4, repeated augmentation aggregated within reference',arms=paired))
    taxonomy={'A1':'no valid anchor','A2':'redundant anchors','A3':'misses major pattern mode',
              'A4':'prefers plain region','A5':'structural contamination','A6':'support over-expands',
              'A7':'support under-covers','A8':'support ambiguous','A9':'wrong geometry compatibility',
              'A10':'unknown overuse','A11':'unknown underuse','A12':'scaffold seam artifact'}
    tags={key:[] for key in taxonomy}
    for r in rows:
        tests=dict(A1=not r['valid_anchor'],A2=r['pairwise_similarity_median'] is not None and r['pairwise_similarity_median']>.90,
                   A4=r['valid_anchor'] and r['mode']=='pattern' and r['pattern_richness']<.25,
                   A5=r['structural_edge']>.5,A7=r['foreground_coverage']<.80,A8=r['ambiguity']>.20,
                   A9=r['orientation_consistency'] is not None and (r['orientation_consistency']<.88 or r['period_consistency']<.88),
                   A10=r['unknown_majority']>.20)
        for code,flag in tests.items():
            if flag:tags[code].append(r['index'])
    write(args.out/'audits/error_tags.json',dict(taxonomy=taxonomy,automatic_proxy_indices=tags,
          requires_independent_visual_evidence=['A3','A4','A5','A6','A9','A11'],A12='not_applicable_until_stage_B'))
    frozen=read(args.out/'input_manifest.json')['frozen_files']
    checks={name:Path(name).exists() and sha(Path(name))==digest for name,digest in frozen.items()}
    write(args.out/'frozen_check.json',dict(checks=checks,training_steps=0,**{'pass':all(checks.values())}))
    required=['decision_summary.json','frozen_check.json','anchor_config.json','protocol.json','input_manifest.json',
              'audits/temperature_selection.json','audits/cache_manifest.json','audits/paired_comparisons.json','audits/error_tags.json']
    for arm in ARMS+ABLATIONS:
        basepath=('A_anchor_mining/' if arm in ARMS else 'audits/ablations/')+arm
        required += [basepath+'/summary.json',basepath+'/rows.json']
        required += [basepath+'/%03d.json'%i for i in range(64,96)]
        required += [basepath+'/%03d_support.npz'%i for i in range(64,96)]
    required += ['A_anchor_mining/B4_soft_unknown/visualizations/%03d/audit.png'%i for i in range(64,96)]
    missing=[p for p in required if not (args.out/p).exists()]
    write(args.out/'completion_check.json',dict(numeric_artifacts_complete=not missing,required_missing=missing,
          frozen_inputs_pass=all(checks.values()),stopped_pilot=bool(gate['hard_stop_reasons']),
          stage_A_pass=gate['pass'],visual_review_completed=False,experiment_complete=False,
          note='visual review must be recorded before completion; completeness never implies method Gate passed'))
    artifacts=[str(p.relative_to(args.out)) for p in args.out.rglob('*') if p.is_file() and p.suffix not in ('.gz','.log','.err')]
    write(args.out/'artifact_manifest.json',dict(files=artifacts,reference_unit='reference image',bootstrap_resamples=2000))
    bundle(args)
    print('[E31 decision]',json.dumps(decision,ensure_ascii=False),flush=True)


def bundle(args):
    with tarfile.open(args.out/'local_review_bundle.tar.gz','w:gz') as archive:
        for p in args.out.rglob('*'):
            if not p.is_file() or '/audits/cache/' in p.as_posix():
                continue
            if p.suffix in ('.json','.png'):
                archive.add(p,arcname=str(p.relative_to(args.out)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','report','all'))
    parser.add_argument('--source',type=Path,default=Path('output_eval/e30_a2_affinity_20260930'))
    parser.add_argument('--out',type=Path,default=Path('output_eval/e31_anchor_correspondence_20261001'))
    parser.add_argument('--dataset',type=Path,default=Path('/share/home/u2515283058/datasets/BF'))
    parser.add_argument('--weights',type=Path,default=Path('output_eval/e30_apacc_20260930/dinov2_vits14_pretrain.pth'))
    args=parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    if args.action in ('prepare','all'):prepare(args)
    if args.action in ('run','all'):run(args)
    if args.action=='report':report(args)
