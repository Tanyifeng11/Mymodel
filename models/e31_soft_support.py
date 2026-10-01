"""E31 learned affinity + geometry + weak spatial prior 的前景概率场。"""

import cv2
import numpy as np


def anchor_support(features, embedding, anchor, geometry=True, spatial=True, dino=False):
    z = features['appearance'] if dino else embedding
    a = np.asarray(anchor['mean_dino' if dino else 'mean_embedding'], np.float32)
    z = z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-8)
    embed = np.einsum('hwd,d->hw', z, a)
    g, m = features['geometry'], anchor['metrics']
    angle = np.deg2rad(2*m['orientation_mean'])
    direction = g[..., :2]/np.maximum(np.linalg.norm(g[..., :2], axis=-1, keepdims=True), 1e-8)
    ori = (1+direction[..., 0]*np.cos(angle)+direction[..., 1]*np.sin(angle))/2
    freq = np.exp(-abs(g[..., 2]-np.log(max(m['frequency_mean'], 1e-4))))
    color = np.exp(-np.linalg.norm(features['color'][..., :3]-anchor['lab'], axis=-1)/.12)
    h, w = embed.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    u, v = (xx+.5)/w, (yy+.5)/h
    au, av = anchor['center_uv']
    prior = np.exp(-np.hypot(u-au, v-av)/.5)
    prior *= np.exp(-.25*((u-.5)*(au-.5)<0)-.25*((v-.5)*(av-.5)<0))
    reliability = np.clip(g[..., 3]/.25, 0, 1)*np.clip(m['geometry_confidence']/.25, 0, 1) if geometry else np.zeros_like(embed)
    if not m['geometry_readable']:
        reliability *= 0
    wo = wf = .15*reliability
    ws = .15 if spatial else 0.
    denom = .50+.05+ws+wo+wf
    result = (.50*embed+wo*ori+wf*freq+.05*color+ws*prior)/denom
    return result.astype(np.float32), dict(embedding=embed, orientation=ori, frequency=freq,
                                         color=color, spatial=prior, geometry_weight=reliability)


def probabilities(scores, mask, temperature=.10, unknown=True, unknown_score=.65):
    # 通道顺序：anchor 0..K-1，末通道始终是 unknown（禁用时为零）。
    if not len(scores):
        return mask.astype(np.float32)[..., None]
    h, w = scores[0].shape
    logits = np.stack(list(scores)+([np.full((h, w), unknown_score)] if unknown else []), -1)/temperature
    logits -= logits.max(-1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(-1, keepdims=True)
    if not unknown:
        p = np.concatenate((p, np.zeros((h, w, 1))), -1)
    mh, mw = mask.shape
    p = np.stack([cv2.resize(p[..., i], (mw, mh)) for i in range(p.shape[-1])], -1)
    p *= mask[..., None]
    # mask-normalized Gaussian，背景不会向衣内或跨外轮廓传递概率。
    denom = cv2.GaussianBlur(mask.astype(np.float32), (3, 3), .6)
    for i in range(p.shape[-1]):
        p[..., i] = cv2.GaussianBlur(p[..., i], (3, 3), .6)/np.maximum(denom, 1e-8)
    p *= mask[..., None]
    p /= np.maximum(p.sum(-1, keepdims=True), 1e-8)
    return p.astype(np.float32)


def coverage(p, mask):
    return float((p[..., :-1].max(-1)[mask] >= .50).mean()) if p.shape[-1] > 1 else 0.


def support_fields(features, embedding, anchors, mask, temperature=.10, unknown=True,
                   geometry=True, spatial=True, dino=False):
    outputs = [anchor_support(features, embedding, a, geometry, spatial, dino) for a in anchors]
    p = probabilities([s for s, _ in outputs], mask, temperature, unknown)
    return p, outputs
