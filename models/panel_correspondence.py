"""E28 的四个可替换环节。人工信息只在指定 semi-oracle 臂中使用。"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from models.confidence_local_correspondence import infer_regions, select_crop
from models.local_pattern_field import patch_geometry, rectify_reference
from tools.e27_correspondence import source_labels, target_panel_masks, target_parts, similarity


@dataclass
class PanelRegion:
    panel_id: str
    mask: np.ndarray
    bbox: tuple
    confidence: float
    parent_id: Optional[str]
    semantic_type: str
    motif_group: Optional[str] = None
    crop: Optional[tuple] = None
    solid: bool = False


@dataclass
class PanelMatch:
    target_panel_id: str
    source_panel_id: str
    score: float
    confidence: float
    assignment_mode: str


@dataclass
class PanelWarp:
    source_panel_id: str
    target_panel_id: str
    uv_field: np.ndarray
    valid_mask: np.ndarray
    confidence: np.ndarray
    warp_mode: str
    rgb: np.ndarray
    fallback_rgb: np.ndarray
    blend_confidence: float


@dataclass
class PanelTransferResult:
    scaffold: np.ndarray
    ownership_map: np.ndarray
    confidence_map: np.ndarray
    seam_map: np.ndarray
    source_index_map: np.ndarray


def _bbox(mask):
    y, x = np.where(mask)
    return (int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1))


def _semantic(name):
    if name.startswith('body') or name in ('upper', 'middle', 'lower'):
        return 'body'
    if 'sleeve' in name:
        return 'sleeve'
    if 'pocket' in name:
        return 'pocket'
    return name


def source_panels(image, panels, mode, source_mask, groups):
    if mode == 'manual':
        labels = source_labels(source_mask, panels)
        return [PanelRegion(p['name'], labels == i, tuple(p['box']), 1., None,
                            _semantic(p['name']), groups[p['name']], tuple(p['crop']), p.get('solid', False))
                for i, p in enumerate(panels) if (labels == i).any()]
    _, records, _ = infer_regions(image)
    return [PanelRegion(r['name'], r['area'], tuple(r['source_box']),
                        float(r['coverage']), None, _semantic(r['name']), None,
                        tuple(r['crop'])) for r in records]


def target_panels(mask, panels, mode, source_names=None):
    if mode == 'manual':
        areas = target_panel_masks(mask, panels)
        return [PanelRegion(p['name'], a, _bbox(a), 1., None, _semantic(p['name']))
                for p, a in zip(panels, areas) if a.any()]
    labels, u, v = target_parts(mask)
    names = source_names or ('body', 'left_sleeve', 'right_sleeve', 'collar')
    areas = []
    for name in names:
        if name == 'body_left': area = (labels == 'body') & (u < .5)
        elif name == 'body_right': area = (labels == 'body') & (u >= .5)
        elif name == 'body_upper': area = (labels == 'body') & (v < .5)
        elif name == 'body_lower': area = (labels == 'body') & (v >= .5)
        else: area = labels == name
        if name.startswith('body'):
            for extra in ('left_sleeve', 'right_sleeve', 'collar'):
                if extra not in names:
                    area = area | (labels == extra)
        areas.append((name, area))
    areas = [(name, a) for name, a in areas if a.sum() > 256]
    return [PanelRegion(name, a, _bbox(a), 1., None, _semantic(name)) for name, a in areas]


def semantic_match(auto, manual):
    """最大 IoU 标签到自动区域；B4 不让自动 assignment 混入。"""
    mapping = {}
    for a in auto:
        scores = [(float((a.mask & m.mask).sum() / max((a.mask | m.mask).sum(), 1)), m.panel_id)
                  for m in manual if a.semantic_type == m.semantic_type]
        if not scores:
            scores = [(float((a.mask & m.mask).sum() / max((a.mask | m.mask).sum(), 1)), m.panel_id)
                      for m in manual]
        mapping[a.panel_id] = max(scores)[1]
    return mapping


def assign_panels(sources, targets, mode, manual_sources=None, manual_targets=None):
    src_by_name = {p.panel_id: p for p in sources}
    src_label = semantic_match(sources, manual_sources) if manual_sources else None
    tgt_label = semantic_match(targets, manual_targets) if manual_targets else None
    matches, matrices = [], {}
    for target in targets:
        expected = tgt_label[target.panel_id] if tgt_label else target.panel_id
        scores = {}
        for source in sources:
            label = src_label[source.panel_id] if src_label else source.panel_id
            if mode == 'manual':
                score = float(label == expected)
            else:
                # E27 自动系统只使用几何/语义；不读取人工 source 名称或 motif group。
                sx = (source.bbox[0] + source.bbox[2]) / (2 * source.mask.shape[1])
                sy = (source.bbox[1] + source.bbox[3]) / (2 * source.mask.shape[0])
                tx = (target.bbox[0] + target.bbox[2]) / (2 * target.mask.shape[1])
                ty = (target.bbox[1] + target.bbox[3]) / (2 * target.mask.shape[0])
                score = float(source.semantic_type == target.semantic_type) - .6 * abs(sx-tx) - .3 * abs(sy-ty)
            scores[source.panel_id] = score
        if mode == 'manual':
            same = [s for s in sources if (src_label[s.panel_id] if src_label else s.panel_id) == expected]
            if not same:
                # 自动 proposal 无精确同名区域时，保留 IoU 标签中最接近的源区域。
                same = [max(sources, key=lambda s: float((s.mask & manual_sources[[p.panel_id for p in manual_sources].index(expected)].mask).sum()))]
            selected = same[0]
        else:
            selected = max(sources, key=lambda s: scores[s.panel_id])
        ordered = sorted(scores.values(), reverse=True)
        margin = ordered[0] - ordered[1] if len(ordered) > 1 else 1.
        matches.append(PanelMatch(target.panel_id, selected.panel_id, scores[selected.panel_id],
                                  float(np.clip(margin, 0, 1)), mode))
        matrices[target.panel_id] = scores
    return matches, matrices


def estimate_panel_warp(image, source, target, mode, variant='original'):
    crop = image.crop(source.crop)
    if variant == 'rot90' and not source.solid:
        crop = crop.transpose(Image.Transpose.ROTATE_90)
    box = source.crop
    geom = patch_geometry(crop)
    original = crop
    gy, gx = np.indices((crop.height, crop.width), dtype=np.float32)
    rect_uv = np.stack((gx, gy), -1)
    if geom['valid'] and min(crop.size) >= 32:
        crop, rect_uv, _ = rectify_reference(crop)
    preservation = similarity(original, crop)[0]
    if mode == 'automatic_local' and preservation < .85:
        # 保留 E27 的 identity rejection；人工 crop 固定，只有局部 warp 改变。
        crop = original
        rect_uv = np.stack((gx, gy), -1)
        preservation = 1.
    blend_conf = float(np.clip(preservation * source.confidence * (.4 + .6*geom['confidence']), 0, 1))
    conf = blend_conf if mode == 'automatic_local' else 1.
    yy, xx = np.indices(target.mask.shape, dtype=np.float32)
    x0, y0, x1, y1 = target.bbox
    sx0, sy0, sx1, sy1 = source.bbox
    uvx = np.mod((xx-x0)/max(x1-x0-1, 1) * (sx1-sx0-1), crop.width).astype(np.float32)
    uvy = np.mod((yy-y0)/max(y1-y0-1, 1) * (sy1-sy0-1), crop.height).astype(np.float32)
    rgb = cv2.remap(np.asarray(crop), uvx, uvy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP).astype(np.float32)
    low = cv2.GaussianBlur(np.asarray(original), (0, 0), 2.)
    fallback = cv2.remap(low, uvx, uvy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    uv = cv2.remap(rect_uv, uvx, uvy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    if variant == 'rot90' and not source.solid:
        uv = np.stack((box[2]-box[0]-1-uv[..., 1], uv[..., 0]), -1)
    uv[..., 0] += box[0]
    uv[..., 1] += box[1]
    confidence = np.where(target.mask, conf, 0).astype(np.float32)
    return PanelWarp(source.panel_id, target.panel_id, uv, target.mask, confidence, mode, rgb,
                     fallback, blend_conf)


def compose_panels(warps, target_mask, mode, source_ids, seam_owner=None):
    masks = np.stack([w.valid_mask for w in warps])
    owner = np.argmax(masks, axis=0)
    if mode == 'fixed_safe_oracle':
        weights = masks.astype(np.float32)
        weights /= np.maximum(weights.sum(0), 1)
    else:
        support = np.stack([cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.5) for m in masks])
        conf = np.stack([w.confidence.max() for w in warps])[:, None, None]
        score = np.log(np.maximum(support, 1e-12)) + 4 * conf
        weights = np.exp(score-score.max(0)); weights /= weights.sum(0)
    rgb = (weights[..., None] * np.stack([w.rgb for w in warps])).sum(0)
    # Oracle 从白背景计算 seam blur，避免把黑色未覆盖区混入外轮廓。
    rgb[~target_mask] = 255
    labels = owner.astype(np.uint8) if seam_owner is None else seam_owner.astype(np.uint8)
    seam = (cv2.dilate(labels, np.ones((3, 3), np.uint8)) !=
            cv2.erode(labels, np.ones((3, 3), np.uint8))) & target_mask
    if mode == 'fixed_safe_oracle':
        smooth = cv2.GaussianBlur(rgb, (5, 5), .8)
        rgb[seam] = .5*rgb[seam] + .5*smooth[seam]
    rgb[~target_mask] = 255
    confmap = (weights*np.stack([w.confidence for w in warps])).sum(0)
    confmap[~target_mask] = 0
    ids = np.array([source_ids.index(w.source_panel_id) for w in warps], np.uint8)
    return PanelTransferResult(np.clip(np.rint(rgb), 0, 255).astype(np.uint8), owner.astype(np.uint8),
                               confmap, seam, ids[owner])


ARMS = {
    'B1_oracle': ('manual', 'manual', 'manual', 'oracle_local_uv', 'fixed_safe_oracle'),
    'B2_auto_warp': ('manual', 'manual', 'manual', 'automatic_local', 'fixed_safe_oracle'),
    'B3_auto_ownership': ('manual', 'manual', 'automatic', 'oracle_local_uv', 'fixed_safe_oracle'),
    'B4_auto_panel': ('automatic', 'automatic', 'manual', 'oracle_local_uv', 'fixed_safe_oracle'),
    'B5_auto_composition': ('manual', 'manual', 'manual', 'oracle_local_uv', 'current_calpc'),
    'B6_full_auto': ('automatic', 'automatic', 'automatic', 'automatic_local', 'current_calpc'),
    'P1_auto_panel_manual_crop': ('automatic', 'automatic', 'manual', 'oracle_local_uv', 'fixed_safe_oracle'),
    'P2_manual_panel_auto_crop': ('manual', 'manual', 'manual', 'oracle_local_uv', 'fixed_safe_oracle'),
}


def build_panel_scaffold(reference_rgb, target_mask, panels, source_mask, groups, arm,
                         original_rgb=None, variant='original'):
    source_mode, target_mode, ownership_mode, warp_mode, composition_mode = ARMS[arm]
    manual_sources = source_panels(reference_rgb, panels, 'manual', source_mask, groups)
    manual_targets = target_panels(target_mask, panels, 'manual')
    sources = manual_sources if source_mode == 'manual' else source_panels(reference_rgb, panels, 'automatic', source_mask, groups)
    if arm == 'P1_auto_panel_manual_crop':
        labels = semantic_match(sources, manual_sources)
        manual_by_id = {p.panel_id: p for p in manual_sources}
        for source in sources:
            selected = manual_by_id[labels[source.panel_id]]
            source.crop, source.solid = selected.crop, selected.solid
    elif arm == 'P2_manual_panel_auto_crop':
        for source in sources:
            source.crop = tuple(select_crop(reference_rgb, source.mask)[0])
    targets = manual_targets if target_mode == 'manual' else target_panels(
        target_mask, panels, 'automatic', [p.panel_id for p in sources])
    matches, matrix = assign_panels(sources, targets, ownership_mode,
                                    manual_sources if source_mode == 'automatic' else None,
                                    manual_targets if target_mode == 'automatic' else None)
    by_id = {p.panel_id: p for p in sources}
    # 人工 crop 从原图整体旋转，严格复用 E27 oracle recipe。
    # 自动 proposal 的 crop 则来自实际 variant RGB，不使用人工 canonical crop。
    manual_crop = arm != 'P2_manual_panel_auto_crop' and (source_mode == 'manual' or arm == 'P1_auto_panel_manual_crop')
    warp_image = original_rgb if manual_crop and original_rgb is not None else reference_rgb
    warp_variant = variant if manual_crop else 'original'
    warps = [estimate_panel_warp(warp_image, by_id[m.source_panel_id], t, warp_mode, warp_variant)
             for t, m in zip(targets, matches)]
    if arm == 'B5_auto_composition':
        for warp in warps:
            c = warp.blend_confidence
            warp.rgb = c*warp.rgb + (1-c)*warp.fallback_rgb
            warp.confidence = np.where(warp.valid_mask, c, 0).astype(np.float32)
    seam_owner = np.argmax(np.stack(target_panel_masks(target_mask, panels)), axis=0) if target_mode == 'manual' else None
    result = compose_panels(warps, target_mask, composition_mode, [p.panel_id for p in sources], seam_owner)
    return Image.fromarray(result.scaffold), result, sources, targets, matches, matrix, warps
