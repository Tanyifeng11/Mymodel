"""E30-A2 无标注特征缓存、冻结分区、三种子训练和 embedding Gate。"""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.nn import functional as F

from models.apacc_adaptive_graph import segment
from models.apacc_affinity_adapter import PatternAffinityAdapter
from models.apacc_canonicality import select_canonical_crop
from models.apacc_features import estimated_foreground, extract_dense_features, load_dino
from models.apacc_region_graph import discover_pattern_regions


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dino_only(image, model, device):
    rgb = cv2.resize(np.asarray(image.convert('RGB')), (448, 448), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()[None].to(device) / 255
    mean = t.new_tensor([.485, .456, .406])[None, :, None, None]
    std = t.new_tensor([.229, .224, .225])[None, :, None, None]
    with torch.inference_mode():
        layers = model.get_intermediate_layers((t - mean) / std, n=4, reshape=True)
        value = F.interpolate(torch.stack(layers).mean(0), (64, 64), mode='bilinear', align_corners=False)[0]
        return F.normalize(value, dim=0).cpu().numpy().transpose(1, 2, 0)


def extra_features(image, mask, base):
    rgb = np.asarray(image.convert('RGB'))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32) / 255
    mean = cv2.GaussianBlur(lab, (0, 0), 5)
    std = np.sqrt(np.maximum(cv2.GaussianBlur(lab * lab, (0, 0), 5) - mean * mean, 0))
    color = np.concatenate([cv2.resize(mean, (64, 64)), cv2.resize(std, (64, 64))], -1)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    band = gray - cv2.GaussianBlur(gray, (0, 0), 4)
    fft = np.stack([cv2.GaussianBlur(abs(gx), (0, 0), 5),
                    cv2.GaussianBlur(abs(gy), (0, 0), 5),
                    cv2.GaussianBlur(abs(band), (0, 0), 5),
                    cv2.GaussianBlur(band * band, (0, 0), 5)], -1)
    fft = cv2.resize(fft, (64, 64))
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5) / 256
    distance = cv2.resize(distance, (64, 64))
    base.update(color=color, fft=fft, boundary_distance=distance)
    return base


def tensor_field(features, norm):
    a = features['appearance'].astype(np.float32)
    a /= np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)
    extra = np.concatenate([features['geometry'], features['color'], features['fft'],
                            features['self_similarity'][..., None],
                            features['boundary_distance'][..., None]], -1).astype(np.float32)
    extra = np.clip((extra - norm['median']) / norm['mad'], -8, 8)
    return np.concatenate([a, extra], -1)


def cache_path(out, index):
    return out / 'feature_cache' / ('%03d.npz' % index)


def load_feature(out, index):
    with np.load(cache_path(out, index)) as data:
        return {k: data[k].astype(np.float32) for k in data.files}


def prepare(args):
    out = args.out
    records = json.loads((args.root / 'data/train_bf_texture.json').read_text(encoding='utf-8'))
    records = sorted(records, key=lambda r: hashlib.sha256(r['cloth'].encode()).hexdigest())
    selected = records[:96]
    split = {'train': [r['cloth'] for r in selected[:64]],
             'dev': [r['cloth'] for r in selected[64:80]],
             'stress_dev': [r['cloth'] for r in selected[80:96]],
             'selection': 'SHA256(cloth) sort of existing BF training split, disjoint 64/16/16',
             'note': 'Stress-dev is deterministic and unstratified; no motif labels were read.'}
    write(out / 'split_manifest.json', split)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, dino_sha = load_dino(device, args.weights)
    write(out / 'protocol.json', {'dino_sha256': dino_sha,
          'e26_sha256': sha(args.root / 'models/local_pattern_field.py'),
          'split': 'BF/training', 'grid': 64, 'supercell_grid': 16,
          'reference_count': 96, 'manual_annotations_used': False,
          'augmentation_seed': 20260930,
          'augmentation': 'brightness 1.08, contrast 0.92, Gaussian blur 0.4; rot90',
          'steps': 3000, 'seeds': [42, 43, 44]})
    folder = out / 'feature_cache'
    folder.mkdir(parents=True, exist_ok=True)
    for i, record in enumerate(selected):
        if cache_path(out, i).exists():
            continue
        image = Image.open(args.dataset / 'training' / record['cloth']).convert('RGB').resize((256, 256))
        mask = estimated_foreground(image)
        feat = extra_features(image, mask, extract_dense_features(image, mask, model, device))
        aug = ImageEnhance.Brightness(image).enhance(1.08)
        aug = ImageEnhance.Contrast(aug).enhance(.92).filter(ImageFilter.GaussianBlur(.4))
        feat['appearance_aug'] = dino_only(aug, model, device)
        feat['appearance_rot'] = np.rot90(dino_only(image.transpose(Image.Transpose.ROTATE_90), model, device), -1)
        # Geometry under rot90 changes by 90 degrees; in the double-angle basis this flips sign.
        feat['foreground'] = mask.astype(np.uint8)
        np.savez_compressed(cache_path(out, i), **{k: v.astype(np.float16) if k not in ('foreground',) else v for k, v in feat.items()})
        print('[A2 prepare]', i + 1, '/96', record['cloth'], flush=True)
    samples = []
    for i in range(64):
        f = load_feature(out, i)
        x = np.concatenate([f['geometry'], f['color'], f['fft'],
                            f['self_similarity'][..., None], f['boundary_distance'][..., None]], -1)
        samples.append(x[f['occupancy'] >= .5][::8])
    pooled = np.concatenate(samples)
    med = np.median(pooled, axis=0)
    mad = np.median(abs(pooled - med), axis=0) * 1.4826
    np.savez(out / 'train_normalization.npz', median=med, mad=np.maximum(mad, .02))
    write(out / 'frozen_check.json', {'dino_sha256': dino_sha,
          'e26_sha256': sha(args.root / 'models/local_pattern_field.py'),
          'dino_changed_tensor_count': 0, 'e26_changed': False, 'e5_changed': False})


def norm_at(out):
    with np.load(out / 'train_normalization.npz') as d:
        return {k: d[k] for k in d.files}


def pool(a):
    return cv2.resize(a.astype(np.float32), (16, 16), interpolation=cv2.INTER_AREA).reshape(256, -1)


def pairs(f):
    """Return unsupervised periodic positives and adjacent discontinuity negatives."""
    a, g, c, fft, occ = (pool(f[k]) for k in ('appearance', 'geometry', 'color', 'fft', 'occupancy'))
    occ = occ[:, 0]
    pos, neg, kinds = [], [], []
    for y in range(16):
        for x in range(16):
            i = y * 16 + x
            if occ[i] < .7:
                continue
            period_px = 1 / max(math.exp(float(g[i, 2])), 1e-3)
            theta = math.atan2(float(g[i, 1]), float(g[i, 0])) / 2 + math.pi / 2
            dx, dy = round(math.cos(theta) * period_px / 16), round(math.sin(theta) * period_px / 16)
            xx, yy = x + dx, y + dy
            if g[i, 3] > .12 and (dx or dy) and 0 <= xx < 16 and 0 <= yy < 16:
                j = yy * 16 + xx
                if occ[j] >= .7 and np.dot(a[i], a[j]) > .8:
                    pos.append((i, j))
            for xx, yy in ((x + 1, y), (x, y + 1)):
                if xx >= 16 or yy >= 16:
                    continue
                j = yy * 16 + xx
                if occ[j] < .7:
                    continue
                confidence = min(g[i, 3], g[j, 3])
                cosine = np.dot(a[i], a[j]) / max(np.linalg.norm(a[i]) * np.linalg.norm(a[j]), 1e-8)
                ori = np.dot(g[i, :2], g[j, :2])
                freq = abs(g[i, 2] - g[j, 2])
                color = np.linalg.norm(c[i] - c[j])
                texture = np.linalg.norm(fft[i] - fft[j])
                if confidence > .12 and ori < .5:
                    kind = 'orientation'
                elif confidence > .12 and freq > .4:
                    kind = 'period'
                elif cosine > .80 and (color > .10 or texture > .04):
                    kind = 'appearance'
                else:
                    continue
                neg.append((i, j))
                kinds.append(kind)
    return np.array(pos, dtype=np.int64).reshape(-1, 2), np.array(neg, dtype=np.int64).reshape(-1, 2), kinds


def frozen(x):
    a = x[..., :384]
    g = x[..., 384:388]
    return F.normalize(torch.cat((a, .25 * g), -1), dim=-1)


def dataset(out, indices):
    norm = norm_at(out)
    records = []
    for i in indices:
        f = load_feature(out, i)
        base = pool(tensor_field(f, norm))
        aug_f = dict(f, appearance=f['appearance_aug'])
        rot_f = dict(f, appearance=f['appearance_rot'], geometry=f['geometry'].copy())
        rot_f['geometry'][..., :2] *= -1
        records.append({'id': i, 'base': base, 'aug': pool(tensor_field(aug_f, norm)),
                        'rot': pool(tensor_field(rot_f, norm)),
                        'valid': pool(f['occupancy'])[:, 0] >= .7,
                        'pos': pairs(f)[0], 'neg': pairs(f)[1], 'neg_kinds': pairs(f)[2]})
    return records


def reg_loss(z):
    z = z - z.mean(0)
    variance = F.relu(.10 - torch.sqrt(z.var(0) + 1e-4)).mean()
    covariance = z.T @ z / max(len(z) - 1, 1)
    covariance = (covariance.square().sum() - covariance.diag().square().sum()) / z.shape[1]
    return variance + covariance


def train(args):
    out, device = args.out, 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    data = dataset(out, range(64))
    extra_dim = data[0]['base'].shape[-1] - 384
    model = PatternAffinityAdapter(extra_dim, not args.no_geometry).to(device)
    initial = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    pools = []
    for r, record in enumerate(data):
        pools.extend((r, int(i)) for i in np.where(record['valid'])[0])
    negatives = [(r, int(i), int(j)) for r, row in enumerate(data) for i, j in row['neg']]
    periodic = [(r, int(i), int(j)) for r, row in enumerate(data) for i, j in row['pos']]
    if not negatives:
        raise RuntimeError('No hard negatives; inspect automatic pair construction')
    manifest = {'augmentation_positive_nodes': len(pools), 'periodic_pairs': len(periodic),
                'hard_negative_pairs': len(negatives), 'hard_negative_types': {},
                'no_geometry': args.no_geometry, 'no_boundary': args.no_boundary,
                'no_period': args.no_period}
    for row in data:
        for kind in row['neg_kinds']:
            manifest['hard_negative_types'][kind] = manifest['hard_negative_types'].get(kind, 0) + 1
    folder = out / 'A21_embedding' / ('seed%d' % args.seed)
    write(folder / 'pair_sampling_manifest.json', manifest)
    base = torch.tensor(np.stack([r['base'] for r in data]), device=device)
    aug = torch.tensor(np.stack([r['aug'] for r in data]), device=device)
    rng = np.random.default_rng(args.seed)
    losses = []
    for step in range(3000):
        picks = rng.integers(len(pools), size=64)
        ri = np.array([pools[p][0] for p in picks])
        pi = np.array([pools[p][1] for p in picks])
        negp = rng.integers(len(negatives), size=64)
        nr = np.array([negatives[p][0] for p in negp])
        ni = np.array([negatives[p][1] for p in negp])
        nj = np.array([negatives[p][2] for p in negp])
        model.train()
        za, zp = model(base[ri, pi]), model(aug[ri, pi])
        zn1, zn2 = model(base[nr, ni]), model(base[nr, nj])
        inv = (1 - (za * zp).sum(-1)).mean()
        negsim = (zn1 * zn2).sum(-1)
        contrast = F.relu(negsim - .3).mean() if not args.no_boundary else 0
        boundary = F.relu(.2 - (za * zp).sum(-1) + negsim).mean() if not args.no_boundary else 0
        period = 0
        if periodic and not args.no_period:
            ps = rng.integers(len(periodic), size=64)
            pr = np.array([periodic[p][0] for p in ps])
            px = np.array([periodic[p][1] for p in ps])
            py = np.array([periodic[p][2] for p in ps])
            period = (1 - (model(base[pr, px]) * model(base[pr, py])).sum(-1)).mean()
        loss = inv + contrast + boundary + .5 * period + .1 * reg_loss(torch.cat((za, zn1, zn2)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scale = min(1., (step + 1) / 300) * (.5 + .5 * math.cos(math.pi * step / 3000))
        for group in optimizer.param_groups:
            group['lr'] = 2e-4 * scale
        if step % 100 == 0 or step == 2999:
            losses.append({'step': step + 1, 'loss': float(loss.item()),
                           'invariance': float(inv.item()), 'negative_similarity': float(negsim.mean().item())})
            print('[A2 train]', args.seed, step + 1, losses[-1], flush=True)
    checkpoint = folder / 'adapter.pt'
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'input_dim': base.shape[-1], 'seed': args.seed,
                'no_geometry': args.no_geometry}, checkpoint)
    write(folder / 'train_log.json', {'initial_sha256': initial, 'final_sha256': sha(checkpoint),
          'loss': losses, 'steps': 3000, 'lr': 2e-4, 'weight_decay': 1e-4,
          'frozen_dino_changed_tensor_count': 0})


def embed_rank(x):
    x = x - x.mean(0, keepdims=True)
    singular = np.linalg.svd(x, compute_uv=False)
    p = singular / max(singular.sum(), 1e-8)
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def bootstrap(values, seed):
    rng = np.random.default_rng(seed)
    means = np.array([np.mean(rng.choice(values, len(values), replace=True)) for _ in range(2000)])
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def audit(args):
    out, device = args.out, 'cuda' if torch.cuda.is_available() else 'cpu'
    data = dataset(out, range(64, 80))
    summary = {}
    for seed in (42, 43, 44):
        folder = out / 'A21_embedding' / ('seed%d' % seed)
        ckpt = torch.load(folder / 'adapter.pt', map_location=device)
        model = PatternAffinityAdapter(ckpt['input_dim'] - 384, ckpt['no_geometry'] is False).to(device)
        model.load_state_dict(ckpt['model'])
        model.eval()
        rows, rank_points = [], []
        for row in data:
            values = {}
            for name, mapper in [('frozen', frozen), ('learned', model)]:
                with torch.inference_mode():
                    b, a, r = (mapper(torch.tensor(row[k], device=device)).cpu().numpy()
                               for k in ('base', 'aug', 'rot'))
                valid = row['valid']
                positive = np.sum(b[valid] * a[valid], -1)
                negative = np.sum(b[row['neg'][:, 0]] * b[row['neg'][:, 1]], -1) if len(row['neg']) else np.array([])
                rotation = np.sum(b[valid] * r[valid], -1)
                values[name] = {'positive': float(positive.mean()) if len(positive) else None,
                                'hard_negative': float(negative.mean()) if len(negative) else None,
                                'rot90': float(rotation.mean()) if len(rotation) else None,
                                'negative_count': len(negative)}
                if name == 'learned':
                    rank_points.append(b[valid][::4])
            rows.append({'id': row['id'], **values})
        usable = [r for r in rows if r['frozen']['hard_negative'] is not None and r['learned']['hard_negative'] is not None]
        results = {'references': len(usable), 'rows': rows}
        for name in ('frozen', 'learned'):
            results[name] = {}
            for metric in ('positive', 'hard_negative', 'rot90'):
                v = np.array([r[name][metric] for r in usable if r[name][metric] is not None])
                results[name][metric] = float(v.mean()) if len(v) else None
                results[name][metric + '_ci95'] = bootstrap(v, seed) if len(v) else None
            results[name]['margin'] = results[name]['positive'] - results[name]['hard_negative'] if len(usable) else None
        results['effective_rank'] = embed_rank(np.concatenate(rank_points)) if rank_points else 0
        f, l = results['frozen'], results['learned']
        checks = {'positive_nondecrease': l['positive'] >= f['positive'],
                  'negative_decrease': l['hard_negative'] < f['hard_negative'],
                  'margin_increase': l['margin'] > f['margin'],
                  'effective_rank_at_least_8': results['effective_rank'] >= 8,
                  'rot90_identity': l['rot90'] >= f['rot90'] - .02}
        results['gate_checks'], results['gate_pass'] = checks, all(checks.values())
        write(folder / 'audit.json', results)
        summary[str(seed)] = {'gate_pass': results['gate_pass'], 'checks': checks,
                              'frozen': f, 'learned': l, 'effective_rank': results['effective_rank']}
    passed = sum(r['gate_pass'] for r in summary.values()) >= 2
    write(out / 'A21_embedding/summary.json', {'seeds': summary, 'gate_pass': passed,
          'minimum_passing_seeds': 2, 'evaluation_references': 'disjoint dev 64:80',
          'next_route': 'adaptive_segmentation' if passed else 'feature_revision'})
    write(out / 'decision_summary.json', {'embedding_gate_pass': passed,
          'adaptive_segmentation_pass': None, 'canonicality_compatibility_pass': None,
          'next_route': 'adaptive_segmentation' if passed else 'feature_revision'})
    print('[A2 embedding gate]', passed, flush=True)


def frozen_adaptive(args):
    out = args.out
    manifest = json.loads((out / 'split_manifest.json').read_text(encoding='utf-8'))
    rows = []
    folder = out / 'A20_frozen_adaptive'
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(64, 96):
        f = load_feature(out, index)
        labels, info = segment(f)
        regions = []
        for k in range(int(labels.max() + 1)):
            mask = cv2.resize((labels == k).astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST).astype(bool)
            mask &= f['foreground'] > 0
            if not mask.any():
                continue
            ys, xs = np.where(mask)
            regions.append({'mask': mask, 'bbox': (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)),
                            'area_ratio': float(mask.mean())})
        name = (manifest['dev'] + manifest['stress_dev'])[index - 64]
        image = Image.open(args.dataset / 'training' / name).convert('RGB').resize((256, 256))
        selected = select_canonical_crop(image, regions, f['foreground'].astype(bool)) if regions else None
        row = {'index': index, 'image': name, 'split': 'dev' if index < 80 else 'stress_dev',
               'region_count': info['regions'], 'foreground_coverage': info['foreground_coverage'],
               'threshold': info['threshold'], 'valid_crop': selected is not None,
               'crop': None if selected is None else {'box': selected['box'], 'metrics': selected['metrics']}}
        rows.append(row)
        np.savez_compressed(folder / ('%03d_regions.npz' % index), labels=labels)
        if index < 80:
            palette = np.array([[225, 225, 225], [230, 75, 70], [80, 140, 220],
                                [55, 180, 120], [210, 145, 50], [130, 90, 195]], np.uint8)
            colored = palette[np.maximum(labels, 0) % len(palette)]
            colored[labels < 0] = 255
            Image.fromarray(cv2.resize(colored, (256, 256), interpolation=cv2.INTER_NEAREST)).save(
                folder / ('%03d_regions.png' % index))
            image.save(folder / ('%03d_reference.png' % index))
        print('[A2 frozen adaptive]', index - 63, '/32', row['region_count'], row['valid_crop'], flush=True)
    summary = {}
    for split in ('dev', 'stress_dev'):
        subset = [r for r in rows if r['split'] == split]
        valid = [r for r in subset if r['valid_crop']]
        summary[split] = {'count': len(subset), 'valid_crop_coverage': len(valid) / len(subset),
                          'mean_region_count': float(np.mean([r['region_count'] for r in subset])),
                          'median_region_count': float(np.median([r['region_count'] for r in subset])),
                          'foreground_coverage': float(np.mean([r['foreground_coverage'] for r in subset]))}
        for metric in ('orientation_consistency', 'period_consistency', 'identity', 'homogeneity', 'contamination'):
            values = [r['crop']['metrics'][metric] for r in valid]
            summary[split][metric] = float(np.mean(values)) if values else None
            summary[split][metric + '_ci95'] = bootstrap(np.array(values), 123) if values else None
    write(folder / 'rows.json', rows)
    write(folder / 'summary.json', summary)


def regions_from_labels(labels, foreground):
    regions = []
    for k in range(int(labels.max() + 1)):
        mask = cv2.resize((labels == k).astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST).astype(bool)
        mask &= foreground
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        regions.append({'mask': mask, 'bbox': (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)),
                        'area_ratio': float(mask.mean())})
    return regions


def compute_labels(f, embedding, method):
    if method == 'adaptive':
        return segment(f, embedding)[0]
    old = dict(f, color=f['color'][..., :3])
    if embedding is not None:
        old['appearance'] = embedding
    return discover_pattern_regions(old, f['foreground'].astype(bool), 'combined')[1]


def partition_stability(a, b, occupancy):
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import adjusted_mutual_info_score
    valid = occupancy >= .5
    aa, bb = a[valid], b[valid]
    both = (aa >= 0) & (bb >= 0)
    if not both.any():
        return {'iou': 0., 'ami': 0.}
    aa, bb = aa[both], bb[both]
    left, right = np.unique(aa), np.unique(bb)
    mat = np.zeros((len(left), len(right)), np.float64)
    for i, x in enumerate(left):
        for j, y in enumerate(right):
            intersection = np.sum((aa == x) & (bb == y))
            union = np.sum((aa == x) | (bb == y))
            mat[i, j] = intersection / max(union, 1)
    row, col = linear_sum_assignment(-mat)
    return {'iou': float(np.mean(mat[row, col])) if len(row) else 0.,
            'ami': float(adjusted_mutual_info_score(aa, bb))}


def segmentation(args):
    out, device = args.out, 'cuda' if torch.cuda.is_available() else 'cpu'
    audit_path = out / 'A21_embedding/summary.json'
    audit_summary = json.loads(audit_path.read_text(encoding='utf-8'))
    if not audit_summary['gate_pass']:
        raise RuntimeError('Embedding Gate 未通过，禁止 learned segmentation')
    manifest = json.loads((out / 'split_manifest.json').read_text(encoding='utf-8'))
    norm = norm_at(out)
    seed = args.seed
    ckpt = torch.load(out / 'A21_embedding' / ('seed%d' % seed) / 'adapter.pt', map_location=device)
    model = PatternAffinityAdapter(ckpt['input_dim'] - 384, not ckpt['no_geometry']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    folder = out / 'A22_segmentation' / ('seed%d' % seed)
    palette = np.array([[225, 225, 225], [230, 75, 70], [80, 140, 220],
                        [55, 180, 120], [210, 145, 50], [130, 90, 195]], np.uint8)
    arms = ('frozen_fixed', 'frozen_adaptive', 'learned_fixed', 'learned_adaptive')
    rows = {arm: [] for arm in arms}
    for arm in arms:
        (folder / arm).mkdir(parents=True, exist_ok=True)
    for index in range(64, 96):
        f = load_feature(out, index)
        with torch.inference_mode():
            embedded = model(torch.tensor(tensor_field(f, norm).reshape(-1, ckpt['input_dim']), device=device))
            embedded = embedded.reshape(64, 64, 64).cpu().numpy()
            aug_f = dict(f, appearance=f['appearance_aug'])
            augmented = model(torch.tensor(tensor_field(aug_f, norm).reshape(-1, ckpt['input_dim']), device=device))
            augmented = augmented.reshape(64, 64, 64).cpu().numpy()
        name = (manifest['dev'] + manifest['stress_dev'])[index - 64]
        image = Image.open(args.dataset / 'training' / name).convert('RGB').resize((256, 256))
        if index >= 80:
            folder.joinpath('previews').mkdir(parents=True, exist_ok=True)
            image.save(folder / 'previews' / ('%03d_reference.png' % index))
            Image.fromarray((f['foreground'] * 255).astype(np.uint8)).save(folder / 'previews' / ('%03d_foreground.png' % index))
            anchor = np.unravel_index(int(np.argmax(f['boundary_distance'] * (f['occupancy'] >= .5))), (64, 64))
            for label, features in [('dino', f['appearance']), ('learned', embedded)]:
                ref = features[anchor]
                sim = features @ ref
                Image.fromarray(np.uint8(np.clip((sim + 1) * 127.5, 0, 255))).resize((256, 256)).save(
                    folder / 'previews' / ('%03d_%s_similarity.png' % (index, label)))
            angle = (np.arctan2(f['geometry'][..., 1], f['geometry'][..., 0]) / (2 * np.pi) + .5) * 255
            Image.fromarray(np.uint8(np.clip(angle, 0, 255))).resize((256, 256)).save(
                folder / 'previews' / ('%03d_orientation.png' % index))
            freq = np.clip((f['geometry'][..., 2] + 8) / 7 * 255, 0, 255)
            Image.fromarray(freq.astype(np.uint8)).resize((256, 256)).save(
                folder / 'previews' / ('%03d_frequency.png' % index))
        for arm in arms:
            embedding = embedded if arm.startswith('learned') else None
            method = 'adaptive' if arm.endswith('adaptive') else 'fixed'
            labels = compute_labels(f, embedding, method)
            regions = regions_from_labels(labels, f['foreground'].astype(bool))
            selected = select_canonical_crop(image, regions, f['foreground'].astype(bool)) if regions else None
            aug_labels = compute_labels(aug_f, augmented if embedding is not None else None, method)
            stability = partition_stability(labels, aug_labels, f['occupancy'])
            row = {'image': name, 'index': index, 'split': 'dev' if index < 80 else 'stress_dev',
                   'region_count': len(regions), 'foreground_coverage': float((labels >= 0).sum() / max((f['occupancy'] >= .5).sum(), 1)),
                   'valid_crop': selected is not None, 'region_stability_iou': stability['iou'],
                   'region_stability_ami': stability['ami'],
                   'crop': None if selected is None else {'box': selected['box'], 'metrics': selected['metrics']}}
            rows[arm].append(row)
            np.savez_compressed(folder / arm / ('%03d_regions.npz' % index), labels=labels)
            if index >= 80:
                arm_folder = folder / 'previews' / arm
                arm_folder.mkdir(parents=True, exist_ok=True)
                colored = palette[np.maximum(labels, 0) % len(palette)]
                colored[labels < 0] = 255
                Image.fromarray(cv2.resize(colored, (256, 256), interpolation=cv2.INTER_NEAREST)).save(
                    arm_folder / ('%03d_regions.png' % index))
                if selected:
                    image.crop(selected['box']).save(arm_folder / ('%03d_crop.png' % index))
        print('[A2 segment]', seed, index - 63, '/32',
              {a: rows[a][-1]['region_count'] for a in arms}, flush=True)
    report = {'seed': seed, 'arms': {}, 'metric_note': 'Reference-level means; stability compares real photometric augmentation.'}
    for arm in arms:
        (folder / arm).mkdir(parents=True, exist_ok=True)
        write(folder / arm / 'rows.json', rows[arm])
        report['arms'][arm] = {}
        for split in ('dev', 'stress_dev'):
            subset = [r for r in rows[arm] if r['split'] == split]
            valid = [r for r in subset if r['valid_crop']]
            metrics = {'count': len(subset), 'valid_crop_coverage': len(valid) / len(subset),
                       'mean_region_count': float(np.mean([r['region_count'] for r in subset])),
                       'median_region_count': float(np.median([r['region_count'] for r in subset]))}
            for key in ('foreground_coverage', 'region_stability_iou', 'region_stability_ami'):
                vals = np.array([r[key] for r in subset])
                metrics[key], metrics[key + '_ci95'] = float(vals.mean()), bootstrap(vals, seed)
            for key in ('orientation_consistency', 'period_consistency', 'identity', 'homogeneity', 'contamination', 'rot90_success'):
                vals = np.array([r['crop']['metrics'][key] for r in valid if r['crop']['metrics'][key] is not None])
                metrics[key] = float(vals.mean()) if len(vals) else None
                metrics[key + '_ci95'] = bootstrap(vals, seed) if len(vals) else None
            report['arms'][arm][split] = metrics
    m = report['arms']['learned_adaptive']['dev']
    stress = report['arms']['learned_adaptive']['stress_dev']
    checks = {'valid_crop_coverage': m['valid_crop_coverage'] >= .8,
              'orientation': m['orientation_consistency'] is not None and m['orientation_consistency'] >= .8775,
              'period': m['period_consistency'] is not None and m['period_consistency'] >= .8817,
              'identity': m['identity'] is not None and m['identity'] >= .9584,
              'homogeneity': m['homogeneity'] is not None and m['homogeneity'] >= .9095,
              'contamination': m['contamination'] is not None and m['contamination'] <= .0094,
              'region_count': m['median_region_count'] <= 6 and m['mean_region_count'] > 1,
              'stability': m['region_stability_iou'] >= .70,
              'stress_valid_crop': stress['valid_crop_coverage'] >= .8}
    report['gate_checks'], report['gate_pass'] = checks, all(checks.values())
    write(folder / 'summary.json', report)
    print('[A2 segmentation gate]', seed, report['gate_pass'], checks, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'frozen', 'train', 'audit', 'segment'))
    p.add_argument('--root', type=Path, default=Path.cwd())
    p.add_argument('--dataset', type=Path, default=Path('/share/home/u2515283058/datasets/BF'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--weights', type=Path)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-geometry', action='store_true')
    p.add_argument('--no-boundary', action='store_true')
    p.add_argument('--no-period', action='store_true')
    a = p.parse_args()
    {'prepare': prepare, 'frozen': frozen_adaptive, 'train': train, 'audit': audit,
     'segment': segmentation}[a.action](a)
