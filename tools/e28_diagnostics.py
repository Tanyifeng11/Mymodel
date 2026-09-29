"""E28 的 case 聚合、衣片缝诊断与映射诊断；UV 仅相对人工 recipe。"""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from models.local_pattern_field import patch_geometry
from models.panel_correspondence import source_panels
from tools.e27_correspondence import target_panel_masks


def recovery_stat(case_values, lower_is_better=False):
    if not case_values:
        return None
    values = np.asarray([v[1:] for v in case_values], float)
    sign = -1 if lower_is_better else 1
    def ratio(v):
        a, b0, b1 = v.mean(0)
        gap = sign*(b1-b0)
        return float((a-b0)/(b1-b0)) if abs(gap) > 1e-8 else None, float(gap)
    value, gap = ratio(values)
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(2000):
        v, g = ratio(values[rng.integers(0, len(values), len(values))])
        if v is not None and g > 1e-8:
            samples.append(v)
    return {'mean': value, 'oracle_gap': gap, 'positive_oracle_gap': bool(gap > 1e-8),
            'ci95': np.percentile(samples, [2.5, 97.5]).tolist() if len(samples) >= 1000 else None,
            'case_n': len(values), 'bootstrap_valid_replicates': len(samples),
            'case_values': case_values,
            'definition': '先 case 内聚合，再对匹配 case 重采样，计算聚合均值之比；非正 Oracle gap 不作归因 Gate。'}


def seam_metrics(image, areas, panels, groups):
    rgb = np.asarray(image, dtype=np.float32)/255
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    grad = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    rows = []
    for i, a in enumerate(areas):
        if not a.any(): continue
        for j, b in enumerate(areas[i+1:], i+1):
            if not b.any(): continue
            ab = a & (cv2.dilate(b.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)
            ba = b & (cv2.dilate(a.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)
            if min(ab.sum(), ba.sum()) < 8: continue
            pa, pb = panels[i]['name'], panels[j]['name']
            jump = None
            if groups[pa] == groups[pb] and not panels[i].get('solid', False):
                # 只在同 motif group、两侧都可读且 patch 不跨片时约束方向。
                angles = []
                for area, band in ((a, ab), (b, ba)):
                    distance = cv2.distanceTransform(area.astype(np.uint8), cv2.DIST_L2, 5)
                    candidates = (distance >= 9) & (cv2.dilate(band.astype(np.uint8), np.ones((25,25), np.uint8)) > 0)
                    ys, xs = np.where(candidates)
                    found = []
                    for k in np.linspace(0, max(len(xs)-1, 0), min(len(xs), 8)).astype(int):
                        x, y = int(xs[k]), int(ys[k])
                        info = patch_geometry(image.crop((x-8, y-8, x+8, y+8)))
                        if info['valid']: found.append(info['orientation'])
                    if found:
                        angle = np.angle(np.mean(np.exp(2j*np.deg2rad(found))))/2
                        angles.append(float(np.rad2deg(angle) % 180))
                if len(angles) == 2: jump = float(abs((angles[0]-angles[1]+90) % 180-90))
            rows.append({'panel_a': pa, 'panel_b': pb,
                         'rgb_band_delta': float(np.abs(rgb[ab].mean(0)-rgb[ba].mean(0)).mean()),
                         'gradient_band_delta': float(abs(grad[ab].mean()-grad[ba].mean())),
                         'orientation_jump_same_group': jump,
                         'same_motif_group': groups[pa] == groups[pb]})
    return rows


def uv_diagnostics(fields):
    rows = []
    for name, uv, mask in zip(fields['target_panel_ids'], fields['uv'], fields['valid']):
        interior = cv2.erode(mask.astype(np.uint8), np.ones((3,3), np.uint8)) > 0
        dy, dx = np.gradient(uv.astype(float), axis=(0,1))
        # 排除周期平铺的坐标跳变，不将 wrap seam 误报为 fold。
        smooth = interior & (np.linalg.norm(dx, axis=-1) < 16) & (np.linalg.norm(dy, axis=-1) < 16)
        determinant = dx[...,0]*dy[...,1]-dx[...,1]*dy[...,0]
        matrix = np.stack((dx[smooth], dy[smooth]), -1)
        singular = np.linalg.svd(matrix, compute_uv=False) if len(matrix) else np.zeros((0,2))
        rows.append({'panel': str(name), 'smooth_pixel_n': int(smooth.sum()),
                     'wrapped_jump_excluded_fraction': float((interior & ~smooth).sum()/max(interior.sum(),1)),
                     'negative_jacobian_ratio': float((determinant[smooth] < -1e-6).mean()) if smooth.any() else None,
                     'median_jacobian_scale': float(np.median(np.sqrt(np.abs(determinant[smooth])))) if smooth.any() else None,
                     'median_anisotropy': float(np.median(singular[:,0]/np.maximum(singular[:,1],1e-6))) if len(singular) else None,
                     'extreme_scale_ratio': float(((singular[:,0] > 4) | (singular[:,1] < .25)).mean()) if len(singular) else None,
                     'out_of_bounds_sampling_ratio': float(((uv[...,0]<0)|(uv[...,0]>255)|(uv[...,1]<0)|(uv[...,1]>255))[mask].mean())})
    return rows


def intermediate_diagnostics(root, out, cases, groups, arm_rows):
    from tools.e20_utilization import case_stat
    mask = np.asarray(Image.open(out/cases['sketches'][0]['mask']).convert('L')) > 0
    seam_rows, mapping_rows = [], []
    overlays = out/'A_audit/auto_overlays'; overlays.mkdir(exist_ok=True)
    matrices = out/'B_decomposition/ownership_matrices'; matrices.mkdir(exist_ok=True)
    for case in cases['references']:
        cid = case['id']; panels = case['panels']; cg = groups[str(cid)]
        original = Image.open(out/'A_audit/inputs'/f'c{cid:02d}_original.png').convert('RGB')
        smask = np.asarray(Image.open(out/'A_audit/inputs'/f'c{cid:02d}_mask.png')) > 0
        sources = source_panels(original, panels, 'automatic', smask, cg)
        overlay = original.copy(); draw = ImageDraw.Draw(overlay)
        for source in sources:
            draw.rectangle(source.bbox, outline='red', width=2)
            draw.rectangle(source.crop, outline='lime', width=2)
            draw.text(source.bbox[:2], source.panel_id, fill='yellow')
        overlay.save(overlays/f'c{cid:02d}.png')
        detail = json.loads((out/'B_decomposition/B3_auto_ownership'/f'c{cid:02d}_original_scaffold.json').read_text())
        targets = list(detail['assignment_scores']); source_ids = list(next(iter(detail['assignment_scores'].values())))
        sheet = Image.new('RGB', (max(1,len(source_ids))*112+136, len(targets)*32+55), 'white')
        d = ImageDraw.Draw(sheet)
        for j, name in enumerate(source_ids): d.text((136+j*112, 5), name, fill='black')
        for i, target in enumerate(targets):
            d.text((2, 40+i*32), target, fill='black')
            for j, source in enumerate(source_ids):
                score = detail['assignment_scores'][target][source]
                intensity = int(255*np.clip((score+1)/2,0,1))
                x, y = 136+j*112, 35+i*32
                d.rectangle((x,y,x+110,y+30), fill=(255-intensity,255,255-intensity))
                d.text((x+3,y+7), f'{score:.2f}'+(' *' if detail['assignments'][target]==source else ''), fill='black')
        sheet.save(matrices/f'c{cid:02d}.png')
        areas = target_panel_masks(mask, panels)
        for arm, rows in arm_rows.items():
            for variant in ('original', 'rot90'):
                folder = out/'B_decomposition'/arm; prefix = f'c{cid:02d}_{variant}'
                files = [('S0', folder/f'{prefix}_scaffold.png'), ('S1', folder/f'{prefix}_vae.png')]
                files += [('S2_s'+str(seed), folder/f'c{cid:02d}_s{seed}_{variant}.png') for seed in (42,43)]
                for stage, path in files:
                    seams = seam_metrics(Image.open(path).convert('RGB'), areas, panels, cg)
                    for r in seams: seam_rows.append({'case':cid, 'variant':variant, 'arm':arm, 'stage':stage, **r})
                if arm == 'B2_auto_warp':
                    for r in uv_diagnostics(np.load(folder/f'{prefix}_fields.npz')):
                        mapping_rows.append({'case':cid, 'variant':variant, 'arm':arm, **r})
    seam_summary = {}
    for arm in arm_rows:
        seam_summary[arm] = {}
        for stage in ('S0','S1','S2'):
            records = [r for r in seam_rows if r['arm']==arm and r['stage'].startswith(stage)]
            seam_summary[arm][stage] = {metric: case_stat([(r['case'],r[metric]) for r in records if r[metric] is not None])
                 if any(r[metric] is not None for r in records) else None
                 for metric in ('rgb_band_delta','gradient_band_delta','orientation_jump_same_group')}
    payload = {'seam_summary': seam_summary, 'seam_rows': seam_rows, 'mapping_rows': mapping_rows,
               'seam_note': '3 px 各侧 band 均值差；RGB 差异是描述项。方向 jump 只统计同 motif group 且可读、不跨片的局部 patch。',
               'mapping_note': '周期 wrap 跳变排除后计算 Jacobian，非物理 UV GT。'}
    (out/'B_decomposition/intermediate_diagnostics.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return {'seam_summary': seam_summary, 'diagnostic_file':'B_decomposition/intermediate_diagnostics.json'}
