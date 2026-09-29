"""E27 无训练基准：完整参考的固定轮廓反事实、人工 panel 上界及局部读出。"""

import hashlib
import math

import cv2
import numpy as np
from PIL import Image

from models.local_pattern_field import patch_geometry, rectify_reference


VERSION = 'e27_manual_panels_v1'


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_labels(mask, panels):
    """人工矩形划分仅用于 benchmark 反事实和 oracle；背景严格按原 mask。"""
    labels = np.zeros(mask.shape, np.uint8)
    # body 默认覆盖前景，后续小衣片覆盖 body。
    for i, panel in enumerate(panels):
        x0, y0, x1, y1 = panel['box']
        labels[y0:y1, x0:x1] = i
    labels[~mask] = 255
    return labels


def pattern_variant(image, mask, panels, name, scale=1.):
    if name == 'original' and scale == 1.:
        return image.copy()
    rgb = np.asarray(image).copy()
    labels = source_labels(mask, panels)
    yy, xx = np.indices(mask.shape, dtype=np.float32)
    for i, panel in enumerate(panels):
        if panel.get('solid', False):
            continue
        x0, y0, x1, y1 = panel['crop']
        crop = np.asarray(image.crop((x0, y0, x1, y1)))
        if name == 'rot90':
            crop = np.rot90(crop).copy()
        # canonical patch 平铺只作用于该源衣片；轮廓/背景/衣片归属均保持。
        uvx = np.mod((xx - panel['box'][0]) / scale, crop.shape[1]).astype(np.float32)
        uvy = np.mod((yy - panel['box'][1]) / scale, crop.shape[0]).astype(np.float32)
        sampled = cv2.remap(crop, uvx, uvy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        rgb[labels == i] = sampled[labels == i]
    return Image.fromarray(rgb)


def target_parts(mask):
    """固定长袖目标的几何衣片；无学习 parser，归一化 bbox 划分。"""
    yy, xx = np.where(mask)
    x0, x1, y0, y1 = xx.min(), xx.max(), yy.min(), yy.max()
    gy, gx = np.indices(mask.shape)
    u, v = (gx-x0)/max(x1-x0, 1), (gy-y0)/max(y1-y0, 1)
    labels = np.full(mask.shape, 'body', dtype='<U16')
    labels[(u < .23) & (v > .17)] = 'left_sleeve'
    labels[(u > .77) & (v > .17)] = 'right_sleeve'
    labels[v < .17] = 'collar'
    labels[~mask] = 'background'
    return labels, u, v


def target_panel_masks(mask, panels):
    labels, u, v = target_parts(mask)
    areas = {}
    for panel in panels:
        name = panel['name']
        if name == 'body':
            area = labels == 'body'
        elif name in ('left_sleeve', 'right_sleeve', 'collar'):
            area = labels == name
        elif name == 'body_left':
            area = (labels == 'body') & (u < .5)
        elif name == 'body_right':
            area = (labels == 'body') & (u >= .5)
        elif name == 'upper':
            area = (labels == 'body') & (v < .40)
        elif name == 'middle':
            area = (labels == 'body') & (v >= .40) & (v < .75)
        elif name == 'lower':
            area = (labels == 'body') & (v >= .75)
        elif name == 'left_pocket':
            area = (labels == 'body') & (u > .27) & (u < .45) & (v > .30) & (v < .43)
        elif name == 'right_pocket':
            area = (labels == 'body') & (u > .55) & (u < .73) & (v > .30) & (v < .43)
        else:
            raise ValueError(name)
        areas[name] = area
    # 各目标像素恰好一个 owner；未提供袖/领的 reference 使用 body fallback。
    owner = np.full(mask.shape, -1, int)
    default = next((i for i,p in enumerate(panels) if p['name'] == 'body'), 0)
    owner[mask] = default
    for i,p in enumerate(panels):
        owner[areas[p['name']]] = i
    return [owner == i for i in range(len(panels))]


def oracle_scaffold(image, mask, panels, variant, rectified=True):
    """独立人工 canonical panel recipe。评测 prototype 使用未矫正版。"""
    areas = target_panel_masks(mask, panels)
    yy, xx = np.indices(mask.shape, dtype=np.float32)
    rgb = np.full((*mask.shape, 3), 255., np.float32)
    maps, stats = [], []
    for panel, area in zip(panels, areas):
        if not area.any():
            continue
        crop = image.crop(tuple(panel['crop']))
        if variant == 'rot90' and not panel.get('solid', False):
            crop = crop.transpose(Image.Transpose.ROTATE_90)
        info = patch_geometry(crop)
        if rectified and info['valid'] and min(crop.size)>=32:
            crop, _, rect = rectify_reference(crop)
        else:
            rect = None
        ys, xs = np.where(area)
        # normalized panel UV 对应源 panel 的像素尺度；canonical patch 周期延拓。
        # crop 只提供纯纹样，不把小 crop 放大成整个衣片，避免人为改变周期。
        box = panel['box']
        uvx = np.mod((xx-xs.min()) / max(xs.max()-xs.min(), 1) * (box[2]-box[0]-1), crop.width)
        uvy = np.mod((yy-ys.min()) / max(ys.max()-ys.min(), 1) * (box[3]-box[1]-1), crop.height)
        sampled = cv2.remap(np.asarray(crop), uvx.astype(np.float32), uvy.astype(np.float32),
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        rgb[area] = sampled[area]
        maps.append(np.stack((uvx, uvy), -1))
        stats.append({'panel':panel['name'], 'geometry':info, 'rectification':rect,
                      'target_area':int(area.sum()), 'solid':panel.get('solid', False)})
    if rectified:
        # 仅在 3px 内部衣片缝带做 soft blend，不要求各衣片纹样相位连续。
        owner = np.argmax(np.stack(areas), axis=0).astype(np.uint8)
        seam = cv2.dilate(owner, np.ones((3,3),np.uint8)) != cv2.erode(owner,np.ones((3,3),np.uint8))
        smooth = cv2.GaussianBlur(rgb, (5,5), .8)
        rgb[seam & mask] = .5 * rgb[seam & mask] + .5 * smooth[seam & mask]
    rgb[~mask] = 255
    return Image.fromarray(np.clip(np.rint(rgb),0,255).astype(np.uint8)), areas, stats


def descriptor(image):
    """64px phase tolerant FFT + autocorrelation + RGB histogram，无外部权重。"""
    rgb = np.asarray(image.resize((64,64), Image.Resampling.BILINEAR), dtype=float) / 255
    gray = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2GRAY).astype(float)
    centered = gray-gray.mean()
    spectrum = np.log1p(abs(np.fft.fftshift(np.fft.fft2(centered)))**2)
    spectrum[30:35,30:35] = 0
    spec = cv2.resize(spectrum, (16,16), interpolation=cv2.INTER_AREA).ravel()
    spec /= max(np.linalg.norm(spec), 1e-9)
    auto = np.array([np.mean(centered*np.roll(centered, d, axis=a))/max(np.mean(centered**2),1e-9)
                     for a in (0,1) for d in (4,8,12,16,24)])
    auto /= max(np.linalg.norm(auto),1e-9)
    hist = np.concatenate([np.histogram(rgb[...,c],bins=8,range=(0,1))[0] for c in range(3)]).astype(float)
    hist /= max(np.linalg.norm(hist),1e-9)
    return np.concatenate((spec*.6, auto*.25, hist*.15)), auto


def similarity(a, b):
    da, aa = descriptor(a)
    db, ab = descriptor(b)
    score = float(np.dot(da,db)/max(np.linalg.norm(da)*np.linalg.norm(db),1e-9))
    auto = float(np.dot(aa,ab)/max(np.linalg.norm(aa)*np.linalg.norm(ab),1e-9))
    return score, auto


def local_readout(image, expected, areas, panels):
    """共同人工 transfer prototype；没有真实 dense UV GT，不按方法自身 scaffold 评分。"""
    rows = []
    for area, panel in zip(areas, panels):
        interior = cv2.erode(area.astype(np.uint8),np.ones((9,9),np.uint8)) > 0
        size = 32 if panel['name'] in ('left_sleeve','right_sleeve','collar','left_pocket','right_pocket') else 64
        for y in range(0,image.height-size+1,size//2):
            for x in range(0,image.width-size+1,size//2):
                if interior[y:y+size,x:x+size].mean() < .9:
                    continue
                a, b = image.crop((x,y,x+size,y+size)), expected.crop((x,y,x+size,y+size))
                gt, pred = patch_geometry(b), patch_geometry(a)
                sim, auto = similarity(a,b)
                orientable = bool(gt['valid'] and not panel.get('solid',False))
                theta = abs((pred['orientation']-gt['orientation']+90)%180-90)
                # 不可读的生成输出按 90deg 失败计，不删掉；同时报告 coverage。
                theta_error = theta if pred['valid'] else 90.
                period = abs(math.log2(max(pred['frequency'],1e-9)/max(gt['frequency'],1e-9)))
                rows.append({'panel':panel['name'],'roi':[x,y,size,size], 'identity':sim,'self_similarity':auto,
                    'expected_valid':orientable,'output_valid':pred['valid'], 'theta_error':theta_error,
                    'period_error':period if pred['valid'] else 4.,
                    'theta_expected':gt['orientation'], 'theta_output':pred['orientation'],
                    'frequency_expected':gt['frequency'],'frequency_output':pred['frequency']})
    geometry = [r for r in rows if r['expected_valid']]
    patterned = [r for r in rows if not next(p for p in panels if p['name']==r['panel']).get('solid',False)]
    def avg(key, selected, default):
        return float(np.mean([r[key] for r in selected])) if selected else default
    return {'theta_error':avg('theta_error',geometry,None),
            'period_error':avg('period_error',geometry,None),
            'local_geometry_follow':avg('theta_error',geometry,90.) <= 20 if geometry else None,
            'identity':avg('identity',patterned,0.), 'self_similarity':avg('self_similarity',patterned,0.),
            'geometry_patch_n':len(geometry), 'local_patch_n':len(rows),
            'output_geometry_coverage':avg('output_valid',geometry,None), 'patches':rows}
