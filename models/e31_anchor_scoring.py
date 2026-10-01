"""E31 固定 canonicality 与 richness；不训练 scorer。"""

import cv2
import numpy as np
from PIL import Image

from models.apacc_canonicality import crop_metrics
from models.e31_anchor_candidates import patch_field, unit
from models.local_pattern_field import patch_geometry
from tools.e27_correspondence import descriptor


WEIGHTS = dict(identity=.18, homogeneity=.12, shift_consistency=.14,
               period_consistency=.14, rot90_equivariance=.08, tile_seam=.08,
               boundary_score=.08, pattern_richness=.14, structural_edge=.04)


def richness(patch):
    gray = np.asarray(patch.convert('L'), np.float32) / 255
    centered = gray - gray.mean()
    window = np.outer(np.hanning(gray.shape[0]), np.hanning(gray.shape[1]))
    power = abs(np.fft.fft2(centered * window)) ** 2
    fy, fx = np.meshgrid(np.fft.fftfreq(gray.shape[0]), np.fft.fftfreq(gray.shape[1]), indexing='ij')
    band = (np.hypot(fx, fy) >= .04) & (np.hypot(fx, fy) <= .45)
    # 绝对带通幅度防止微弱噪声凭高频比例被当成丰富纹样。
    energy = float(np.clip(np.sqrt(power[band].sum()) / max(gray.size, 1) / .045, 0, 1))
    autocorr = np.fft.ifft2(abs(np.fft.fft2(centered)) ** 2).real
    yy, xx = np.indices(gray.shape)
    dist = np.hypot(np.minimum(xx, gray.shape[1]-xx), np.minimum(yy, gray.shape[0]-yy))
    periodic = (dist >= 4) & (dist <= min(gray.shape)/2)
    peak = float(np.clip(autocorr[periodic].max() / max(float(autocorr[0, 0]), 1e-8), 0, 1))
    counts = np.histogram(gray, bins=64, range=(0, 1))[0].astype(float)
    prob = counts / counts.sum()
    entropy = float(-(prob * np.log2(prob + 1e-12)).sum() / 6)
    return dict(pattern_richness=.4*energy+.3*peak+.3*entropy,
                mid_high_frequency_energy=energy, autocorrelation_peak=peak,
                local_entropy=entropy, gray_std=float(gray.std()))


def score(metrics, no_richness=False):
    weights = dict(WEIGHTS)
    # E26 不可读时不奖励默认 0.5，也不淘汰 floral/mixed。
    if not metrics['geometry_readable']:
        weights['period_consistency'] = weights['rot90_equivariance'] = 0
    if no_richness:
        weights['pattern_richness'] = 0
    denom = sum(v for k, v in weights.items() if k != 'structural_edge')
    positive = sum(weights[k] * (metrics[k] or 0.) for k in weights if k != 'structural_edge')
    return float(positive / max(denom, 1e-8) - weights['structural_edge'] * metrics['structural_edge'])


def describe_candidate(image, mask, distance, features, embedding, box):
    patch = image.crop(box)
    m = crop_metrics(image, mask, mask, box)
    geo = patch_geometry(patch)
    rotated = patch_geometry(patch.transpose(Image.Transpose.ROTATE_90))
    m.update(richness(patch))
    z = patch_field(embedding, box, image.size).reshape(-1, embedding.shape[-1])
    g = patch_field(features['geometry'], box, image.size).reshape(-1, 4)
    x0, y0, x1, y1 = box
    confidence = float(geo['confidence'])
    m.update(geometry_readable=geo['valid'], geometry_confidence=confidence,
             orientation_mean=geo['orientation'], frequency_mean=geo['frequency'],
             orientation_var=float(1-np.linalg.norm(g[:, :2].mean(0))),
             frequency_std=float(g[:, 2].std()), embedding_var=float(z.var(0).mean()),
             boundary_distance=float(distance[y0:y1, x0:x1].min()),
             boundary_score=float(np.clip(distance[y0:y1, x0:x1].mean()/24, 0, 1)),
             rot90_equivariance=float(np.exp(-m['rotation_error']/15)) if geo['valid'] and rotated['valid'] else 0.)
    # 可读性限定在有效局部窗口，未读窗口不能用 0.5 参与 Gate。
    quadrants = [patch.crop((x*patch.width//2, y*patch.height//2,
                            (x+1)*patch.width//2, (y+1)*patch.height//2))
                 for y in range(2) for x in range(2)]
    readable = [patch_geometry(p) for p in quadrants]
    readable = [g for g in readable if g['valid']]
    m['readable_quadrants'] = len(readable)
    m['geometry_readable'] = bool(geo['valid'] and len(readable) >= 2)
    if not m['geometry_readable']:
        m['orientation_consistency'] = m['period_consistency'] = None
    else:
        angles = np.deg2rad([2*g['orientation'] for g in readable])
        m['orientation_consistency'] = float(abs(np.mean(np.exp(1j*angles))))
        m['period_consistency'] = float(np.exp(-np.std(np.log([max(g['frequency'], 1e-4) for g in readable]))))
    m['canonicality'] = score(m)
    m['canonicality_no_richness'] = score(m, True)
    return dict(bbox=list(box), center=[(x0+x1)/2, (y0+y1)/2],
                center_uv=[(x0+x1)/2/image.width, (y0+y1)/2/image.height],
                mean_embedding=unit(z.mean(0)).tolist(),
                mean_dino=unit(patch_field(features['appearance'], box, image.size).mean((0, 1))).tolist(),
                lab=patch_field(features['color'][..., :3], box, image.size).mean((0, 1)).tolist(),
                identity_descriptor=descriptor(patch)[0].tolist(), metrics=m)


def eligible_candidates(candidates, image, mask, no_richness=False):
    valid = [c for c in candidates if c['metrics']['canonicality_no_richness' if no_richness else 'canonicality'] >= .50]
    pattern = [c for c in valid if c['metrics']['pattern_richness'] >= .25 and c['metrics']['gray_std'] >= .025]
    if no_richness:
        return valid, 'no_richness'
    if pattern:
        return pattern, 'pattern'
    # 整衣近乎纯色才允许一个 appearance fallback，不能让普通候选全被素色替代。
    rgb = np.asarray(image.convert('RGB'), np.float32)/255
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    high = abs(gray-cv2.GaussianBlur(gray, (0, 0), 3))
    plain = float(high[mask].mean()) < .012
    return (valid if plain else []), ('low_frequency_fallback' if plain else 'no_valid_pattern')
