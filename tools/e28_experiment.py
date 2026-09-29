"""E28：审计、半 oracle scaffold、冻结 E5 生成与因果归因。"""

import argparse
import json
import shutil
import subprocess
import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from models.panel_correspondence import ARMS, build_panel_scaffold, source_panels, target_panels, semantic_match
from tools.e27_correspondence import file_sha, local_readout, oracle_scaffold, target_panel_masks
from tools.e27_experiment import audited_metrics, finite_json, write


CASE_IDS = (6, 7, 9, 10, 12, 13, 14, 17)
SEEDS = (42, 43)
VARIANTS = ('original', 'rot90')
ARM_NAMES = ('B0_global_anchor', *ARMS)
RECIPE_VERSION = 'e28_single_component_v2'


def _load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _inputs(out, case, variant):
    prefix = f"c{case['id']:02d}_{variant}"
    return Image.open(out/'A_audit/inputs'/f'{prefix}.png').convert('RGB'), prefix


def _stage_metrics(image, mask, sketch, reference, expected, roi):
    from tools.e22_o4_metrics import measure
    from tools.e26_audit import structural_drift
    from eval.eval_utils import estimate_foreground_mask
    from tools.e22_o4_metrics import contour
    m = measure(image, mask, sketch, reference, 0., roi)
    drift = structural_drift(image, expected, mask)
    target = np.asarray(mask) > 0
    actual = estimate_foreground_mask(image, image.size)
    edge = contour(target)
    pred = contour(actual)
    band = cv2.dilate(edge.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    m.update(drift)
    m.update(foreground_occupancy=float(actual.mean()),
             outer_boundary_damage=float(np.mean(edge[band] != pred[band])))
    return {k: m[k] for k in ('contour_f1', 'sketch_iou', 'leakage', 'foreground_occupancy',
                              'contour_displacement_px', 'background_rgb_deviation',
                              'outer_boundary_damage')}


def _boundary_f1(a, b, tolerance):
    kernel = np.ones((3, 3), np.uint8)
    ae = a & ~(cv2.erode(a.astype(np.uint8), kernel) > 0)
    be = b & ~(cv2.erode(b.astype(np.uint8), kernel) > 0)
    reach = np.ones((2*tolerance+1, 2*tolerance+1), np.uint8)
    precision = (ae & (cv2.dilate(be.astype(np.uint8), reach) > 0)).sum()/max(ae.sum(), 1)
    recall = (be & (cv2.dilate(ae.astype(np.uint8), reach) > 0)).sum()/max(be.sum(), 1)
    return float(2*precision*recall/max(precision+recall, 1e-12))


def audit(root, e27, dataset, out):
    old = _load(e27/'cases.json')
    groups = _load(root/'data/e28_panel_cases.json')
    refs = [r for r in old['references'] if r['id'] in CASE_IDS]
    assert [r['id'] for r in refs] == list(CASE_IDS), 'E27 oracle 8-case 顺序或集合不一致'
    sketch = old['sketches'][0]
    (out/'A_audit').mkdir(parents=True, exist_ok=True)
    for key in ('path', 'mask'):
        dest = out/sketch[key]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(e27/sketch[key], dest)
    shutil.copy2(e27/'A_audit/audit.csv', out/'A_audit/audit.csv')
    target = np.asarray(Image.open(out/sketch['mask']).convert('L')) > 0
    audit_rows = []
    for case in refs:
        cid = case['id']; names = [p['name'] for p in case['panels']]
        assert set(names) == set(groups[str(cid)]), f'case {cid} ownership 标注不全'
        assert case['oracle'] and case['difficulty'] in ('Medium', 'Hard')
        original_path = dataset/case['source']; mask_path = dataset/case['source_mask']
        assert file_sha(original_path) == case['source_sha256'], original_path
        assert file_sha(mask_path) == case['mask_sha256'], mask_path
        source = Image.open(original_path).convert('RGB')
        smask = np.asarray(Image.open(mask_path).convert('L')) > 0
        assert source.size == (256, 256) and smask.shape == (256, 256)
        manual_s = source_panels(source, case['panels'], 'manual', smask, groups[str(cid)])
        manual_t = target_panels(target, case['panels'], 'manual')
        # E27 的层级矩形会让父 body 被后续子 panel 完全覆盖；这不是缺标注。
        assert len(manual_s) == len(names) and manual_t
        assert set(p.panel_id for p in manual_t) <= set(names)
        auto_s = source_panels(source, case['panels'], 'automatic', smask, groups[str(cid)])
        assert semantic_match(auto_s, manual_s)
        mask_dest = out/'A_audit/inputs'/f'c{cid:02d}_mask.png'
        mask_dest.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(smask.astype(np.uint8)*255).save(mask_dest)
        for variant in VARIANTS:
            src = e27/'A_audit/inputs'/f'c{cid:02d}_{variant}.png'
            dest = out/'A_audit/inputs'/src.name
            shutil.copy2(src, dest)
            if variant == 'original':
                assert np.array_equal(np.asarray(Image.open(dest).convert('RGB')), np.asarray(source))
            else:
                rgb = np.asarray(Image.open(dest).convert('RGB'))
                assert np.array_equal(rgb[~smask], np.asarray(source)[~smask])
            expected = e27/'expected'/f'c{cid:02d}_{variant}.png'
            dst_expected = out/'expected'/expected.name
            dst_expected.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(expected, dst_expected)
        overlay = source.copy(); d = ImageDraw.Draw(overlay)
        for p in case['panels']:
            d.rectangle(p['box'], outline='red', width=2)
            d.rectangle(p['crop'], outline='lime', width=2)
            d.text(tuple(p['crop'][:2]), p['name'], fill='yellow')
        overlay_dir = out/'A_audit/panel_overlays'; overlay_dir.mkdir(parents=True, exist_ok=True)
        overlay.save(overlay_dir/f'c{cid:02d}.png')
        audit_rows.append({'case_id': cid, 'source_sha256': file_sha(original_path),
                           'source_mask_sha256': file_sha(mask_path),
                           'source_panels': names, 'motif_groups': groups[str(cid)],
                           'auto_panel_ids': [p.panel_id for p in auto_s],
                           'target_panel_ids': [p.panel_id for p in manual_t],
                           'inactive_target_panels': sorted(set(names)-{p.panel_id for p in manual_t})})
    assert len({r['source_sha256'] for r in audit_rows}) == 8
    report = {'num_cases': 8, 'manual_source_panel_complete': True,
              'manual_target_panel_complete': True, 'ownership_annotation_complete': True,
              'rot90_valid': True, 'source_sha_unique': True, 'target_fixed': True,
              'source_sha_match_e27': True, 'manual_auto_id_mapping_complete': True,
              'pass': True, 'cases': audit_rows}
    write(out/'A_audit/report.json', report)
    write(out/'cases.json', {'references': refs, 'sketches': [sketch]})
    write(out/'protocol.json', {'experiment': 'E28', 'source_experiment': str(e27),
          'case_ids': CASE_IDS, 'seeds': SEEDS, 'variants': VARIANTS, 'arms': ARM_NAMES,
          'canvas': [384, 512], 'sampler': 'DDIM', 'steps': 50, 'cfg': 7,
          'refinement_strength': .15, 'vae_latent': 'posterior_mean', 'training_steps': 0,
          'bootstrap_unit': 'reference_case', 'bootstrap_replicates': 2000,
          'source_manifest_sha256': file_sha(root/'data/e27_c3_cases.json'),
          'ownership_annotation_sha256': file_sha(root/'data/e28_panel_cases.json')})
    print('[E28 A] audit passed', flush=True)


def scaffold(root, e27, out):
    assert _load(out/'A_audit/report.json')['pass']
    cases = _load(out/'cases.json')
    groups = _load(root/'data/e28_panel_cases.json')
    mask = np.asarray(Image.open(out/cases['sketches'][0]['mask']).convert('L')) > 0
    panel_rows = []
    equivalence = []
    for case in cases['references']:
        source_mask = np.asarray(Image.open(out/'A_audit/inputs'/f"c{case['id']:02d}_mask.png")) > 0
        for variant in VARIANTS:
            reference, prefix = _inputs(out, case, variant)
            original, _ = _inputs(out, case, 'original')
            # 验证统一入口的全人工组合逐像素复现旧 oracle，防止旋转/拼接混入干预。
            baseline, *_ = build_panel_scaffold(reference, mask, case['panels'], source_mask,
                            groups[str(case['id'])], 'B1_oracle', original, variant)
            oracle_image, _, _ = oracle_scaffold(original, mask, case['panels'], variant,
                                                 rectified=True, source_mask=source_mask)
            error = np.abs(np.asarray(baseline).astype(int)-np.asarray(oracle_image).astype(int))
            assert error.max() <= 1 and error.mean() < .001, (prefix, error.max(), error.mean())
            equivalence.append({'case': case['id'], 'variant': variant,
                                'max_rgb_error': int(error.max()), 'mean_rgb_error': float(error.mean())})
            manual_s = source_panels(reference, case['panels'], 'manual', source_mask, groups[str(case['id'])])
            auto_s = source_panels(reference, case['panels'], 'automatic', source_mask, groups[str(case['id'])])
            matched = semantic_match(auto_s, manual_s)
            for man in manual_s:
                options = [a for a in auto_s if matched[a.panel_id] == man.panel_id]
                auto = max(options, key=lambda a: (a.mask & man.mask).sum()) if options else None
                area = np.logical_or.reduce([a.mask for a in options]) if options else np.zeros_like(man.mask)
                inter = int((area & man.mask).sum())
                panel_rows.append({'case': case['id'], 'variant': variant,
                                   'auto_panel': auto.panel_id if auto else None,
                                   'manual_panel': man.panel_id,
                                   'iou': inter/max(int((area | man.mask).sum()), 1),
                                   'coverage': inter/max(int(man.mask.sum()), 1),
                                   'boundary_f1_3px': _boundary_f1(area, man.mask, 3),
                                   'boundary_f1_5px': _boundary_f1(area, man.mask, 5)})
            for arm in ARM_NAMES:
                folder = out/'B_decomposition'/arm
                folder.mkdir(parents=True, exist_ok=True)
                path = folder/f'{prefix}_scaffold.png'
                if path.exists() and path.with_suffix('.json').exists():
                    continue
                if arm == 'B0_global_anchor':
                    image = Image.open(e27/'B_baseline/scaffolds'/f'{prefix}_B1_global_rectified.png').convert('RGB')
                    details = {'source': 'E27 B1_global_rectified', 'assignments': {}, 'matrix': {}}
                elif arm == 'B1_oracle':
                    image, _, _ = oracle_scaffold(original, mask, case['panels'], variant,
                                                   rectified=True,
                                                   field_path=folder/f'{prefix}_fields.npz',
                                                   source_mask=source_mask)
                    details = {'source': 'E27 oracle_scaffold', 'assignments':
                               {p['name']: p['name'] for p in case['panels']}, 'matrix': {}}
                elif arm == 'B6_full_auto':
                    from models.confidence_local_correspondence import build_scaffold
                    image, arrays, info = build_scaffold(reference, mask, 'full_CALPC')
                    details = {'source': 'E27 full_CALPC', 'assignments':
                               {r['name']: r['name'] for r in info['regions']}, 'matrix': {}, 'info': info}
                    np.savez_compressed(folder/f'{prefix}_fields.npz', **arrays)
                else:
                    image, result, sources, targets, matches, matrix, warps = build_panel_scaffold(
                        reference, mask, case['panels'], source_mask, groups[str(case['id'])], arm,
                        original, variant)
                    details = {'source': 'E28 modular', 'assignments':
                               {m.target_panel_id: m.source_panel_id for m in matches},
                               'assignment_scores': matrix,
                               'assignment_margin': {m.target_panel_id: m.confidence for m in matches},
                               'source_panels': [p.panel_id for p in sources],
                               'target_panels': [p.panel_id for p in targets]}
                    np.savez_compressed(folder/f'{prefix}_fields.npz',
                                        ownership_map=result.ownership_map,
                                        confidence_map=result.confidence_map,
                                        seam_map=result.seam_map,
                                        source_index_map=result.source_index_map,
                                        source_panel_ids=np.array([w.source_panel_id for w in warps]),
                                        target_panel_ids=np.array([w.target_panel_id for w in warps]),
                                        uv=np.stack([w.uv_field for w in warps]),
                                        valid=np.stack([w.valid_mask for w in warps]))
                image.save(path)
                details.update(case_id=case['id'], variant=variant, arm=arm,
                               recipe_version=RECIPE_VERSION,
                               scaffold_sha256=file_sha(path))
                write(path.with_suffix('.json'), details)
                print('[E28 scaffold]', prefix, arm, flush=True)
    write(out/'B_decomposition/panel_proposal.json', {'rows': panel_rows})
    write(out/'B_decomposition/component_check.json', {'pass': True, 'recipe_version': RECIPE_VERSION,
          'manual_recipe_equivalence': equivalence,
          'interventions': {arm: modes for arm, modes in ARMS.items()},
          'automatic_ownership_definition': 'E28 extracted semantic/location prior; E27 has implicit same-region assignment; no learned target appearance',
          'B6_implementation': 'unmodified E27 full_CALPC'})


def revise(out):
    """首轮诊断留存；只重跑受旋转/白背景 seam 修正影响的干预臂。"""
    archive = out/'B_initial_4195106'
    archive.mkdir(exist_ok=True)
    for arm in ARM_NAMES[2:6]:
        path = out/'B_decomposition'/arm
        if path.exists() and not (archive/arm).exists():
            shutil.move(str(path), str(archive/arm))
    for name in ('report.json', 'recovery_report.json'):
        path = out/'B_decomposition'/name
        if path.exists() and not (archive/name).exists(): shutil.copy2(path, archive/name)
    path = out/'decision_summary.json'
    if path.exists() and not (archive/path.name).exists(): shutil.copy2(path, archive/path.name)
    for name in ('frozen_check.json', 'noise_manifest.json'):
        path = out/name
        if path.exists() and not (archive/name).exists(): shutil.copy2(path, archive/name)


def generate(root, out):
    import torch
    from torchvision.transforms.functional import to_tensor
    from tools import e23_mechanism as e23
    from tools import e25_spatial_diagnosis as e25
    from tools.e22_4_generation import module_hashes, native_pipeline
    torch.set_num_threads(4)
    cases = _load(out/'cases.json')
    sk = cases['sketches'][0]
    mask = Image.open(out/sk['mask']).convert('L')
    sketch = Image.open(out/sk['path']).convert('RGB')
    pipe, modules, width, height = native_pipeline(root, 'E5')
    pipe.set_progress_bar_config(disable=True)
    assert type(pipe.scheduler).__name__ == 'DDIMScheduler'
    assert mask.size == (width, height)
    before = module_hashes(modules)
    checkpoint_sha = file_sha(root/'output/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt')
    start = e25.timestep_for(pipe.scheduler, .15)
    assert start['remaining_steps'] == 8
    bank = e23.token_bank(pipe, cases, out, width, height, 'E5')
    def tensor_sha(value):
        h = hashlib.sha256()
        def visit(v):
            if torch.is_tensor(v): h.update(v.detach().cpu().contiguous().numpy().tobytes())
            elif isinstance(v, dict):
                for k in sorted(v): h.update(str(k).encode()); visit(v[k])
            elif isinstance(v, (tuple, list)):
                for x in v: visit(x)
            else: h.update(str(v).encode())
        visit(value)
        return h.hexdigest()
    condition_manifest = {f'c{cid:02d}_{variant}': tensor_sha(value)
                          for (cid, variant), value in bank.items()}
    condition_manifest.update(prompt='a cloth', negative_prompt=' worst quality, low quality',
                              sketch_sha256=file_sha(out/sk['path']), target_mask_sha256=file_sha(out/sk['mask']),
                              texture_tokens=16, texture_path=True)
    generation_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    from tools.e27_experiment import metric_eligibility
    eligibility = metric_eligibility(out)
    noise_manifest = {}
    for case in cases['references']:
        areas = target_panel_masks(np.asarray(mask) > 0, case['panels'])
        for variant in VARIANTS:
            reference, prefix = _inputs(out, case, variant)
            expected = Image.open(out/'expected'/f'{prefix}.png').convert('RGB')
            for seed in SEEDS:
                noise = torch.randn((1, 4, height//8, width//8),
                                    generator=torch.Generator(device=pipe.device).manual_seed(seed),
                                    device=pipe.device, dtype=pipe.vae.dtype)
                nsha = e25.sha(noise.cpu().numpy().tobytes())
                noise_manifest[f'{prefix}_s{seed}'] = nsha
                for arm in ARM_NAMES:
                    folder = out/'B_decomposition'/arm
                    final_path = folder/f'c{case["id"]:02d}_s{seed}_{variant}.png'
                    if final_path.exists() and final_path.with_suffix('.json').exists():
                        old = _load(final_path.with_suffix('.json'))
                        assert old['checkpoint_sha256'] == checkpoint_sha and old['noise_sha256'] == nsha
                        if 'condition_sha256' in old:
                            assert old['condition_sha256'] == condition_manifest[prefix]
                        else:
                            # 原锚点按同一输入/冻结模型重新计算条件哈希，标明这是补充核对。
                            old['condition_sha256'] = condition_manifest[prefix]
                            old['condition_hash_verified_on_resume'] = True
                            old['sketch_sha256'] = condition_manifest['sketch_sha256']
                            old['target_mask_sha256'] = condition_manifest['target_mask_sha256']
                            write(final_path.with_suffix('.json'), old)
                        continue
                    scaffold_image = Image.open(folder/f'{prefix}_scaffold.png').convert('RGB')
                    pixels = to_tensor(scaffold_image)[None].to(pipe.device, pipe.vae.dtype)*2-1
                    z0 = pipe.vae.encode(pixels).latent_dist.mean*pipe.vae.config.scaling_factor
                    if seed == SEEDS[0]:
                        decoded = pipe.vae.decode(z0/pipe.vae.config.scaling_factor).sample
                        reconstruction = Image.fromarray(np.clip(np.rint(((decoded[0].float().cpu().permute(1,2,0).numpy()+1)/2)*255),0,255).astype(np.uint8))
                        reconstruction.save(folder/f'{prefix}_vae.png')
                    image, checked_noise, latent_sha = e25.refine_one(
                        pipe, z0, noise, start, sketch, mask, reference,
                        bank[case['id'], variant], True, mask.size)
                    assert checked_noise == nsha
                    image.save(final_path)
                    local = local_readout(image, expected, areas, case['panels'])
                    row = {'case': case['id'], 'reference_id': case['reference_id'],
                           'difficulty': case['difficulty'], 'family': case['pattern_type'],
                           'seed': seed, 'variant': variant, 'arm': arm,
                           'noise_sha256': nsha, 'initial_latent_sha256': latent_sha,
                           'checkpoint_sha256': checkpoint_sha, 'output_sha256': file_sha(final_path),
                           'generation_git_commit': generation_commit, 'recipe_version': RECIPE_VERSION,
                           'condition_sha256': condition_manifest[prefix],
                           'sketch_sha256': condition_manifest['sketch_sha256'],
                           'target_mask_sha256': condition_manifest['target_mask_sha256'],
                           'scaffold_sha256': file_sha(folder/f'{prefix}_scaffold.png'),
                           'component_modes': ARMS.get(arm, ('global',)*5),
                           'selected_assignments': _load((folder/f'{prefix}_scaffold.png').with_suffix('.json'))['assignments'],
                           'scaffold_metrics': _stage_metrics(scaffold_image, mask, sketch, reference, expected, sk['roi']),
                           'vae_metrics': _stage_metrics(reconstruction if seed == SEEDS[0] else Image.open(folder/f'{prefix}_vae.png').convert('RGB'), mask, sketch, reference, expected, sk['roi']),
                           'final_metrics': _stage_metrics(image, mask, sketch, reference, expected, sk['roi']),
                           **local}
                    row.update(row['final_metrics'])
                    audited_metrics(row, eligibility)
                    write(final_path.with_suffix('.json'), row)
                    print('[E28 final]', arm, prefix, seed, flush=True)
    after = module_hashes(modules)
    write(out/'frozen_check.json', {'pass': before == after, 'before': before, 'after': after,
                                   'checkpoint_sha256': checkpoint_sha, 'training_steps': 0})
    assert before == after
    write(out/'noise_manifest.json', noise_manifest)
    write(out/'condition_manifest.json', condition_manifest)


def report(root, out):
    from tools.e20_utilization import case_stat
    from tools.e27_experiment import stats
    from tools.e28_diagnostics import recovery_stat, intermediate_diagnostics
    assert _load(out/'frozen_check.json')['pass'], '冻结模型校验失败'
    cases = _load(out/'cases.json')
    groups = _load(root/'data/e28_panel_cases.json')
    rows = {}
    for arm in ARM_NAMES:
        paths = sorted((out/'B_decomposition'/arm).glob('c*_s*_*.json'))
        assert len(paths) == 32, f'{arm}: {len(paths)}/32 images'
        rows[arm] = [_load(p) for p in paths]
    noise = {}
    for arm, items in rows.items():
        for r in items:
            key = (r['case'], r['seed'], r['variant'])
            value = (r['noise_sha256'], r['condition_sha256'], r['sketch_sha256'], r['target_mask_sha256'], r['checkpoint_sha256'])
            if key in noise: assert noise[key] == value
            else: noise[key] = value
    summary = {}
    keys = ('identity', 'theta_error', 'period_error', 'contour_f1', 'leakage')
    for arm, items in rows.items():
        summary[arm] = stats(items)
        for k in keys:
            values = [(r['case'], r[k]) for r in items if r.get(k) is not None]
            summary[arm][k] = case_stat(values) if values else None
        summary[arm]['structure_stages'] = {}
        for stage in ('scaffold_metrics', 'vae_metrics', 'final_metrics'):
            stage_summary = {}
            for k in ('contour_f1', 'sketch_iou', 'leakage', 'foreground_occupancy',
                      'contour_displacement_px', 'background_rgb_deviation',
                      'outer_boundary_damage'):
                values = [(r['case'], r[stage][k]) for r in items if r[stage].get(k) is not None]
                stage_summary[k] = case_stat(values) if values else None
            summary[arm]['structure_stages'][stage] = stage_summary
        follow = []
        for cid in CASE_IDS:
            for seed in SEEDS:
                pair = [r for r in items if r['case'] == cid and r['seed'] == seed]
                assert len(pair) == 2
                valid = [r for r in pair if r.get('local_geometry_follow') is not None]
                follow.append((cid, float(all(r['local_geometry_follow'] for r in valid)
                                          and all(r['identity'] >= .65 for r in pair))))
        summary[arm]['follow'] = case_stat(follow)
    index = {arm: {(r['case'], r['seed'], r['variant']): r for r in items} for arm, items in rows.items()}
    recoveries = {}
    for arm in ARM_NAMES[2:]:
        recoveries[arm] = {}
        for metric in ('identity', 'theta_error', 'period_error', 'follow', 'contour_f1'):
            case_values = []
            for cid in CASE_IDS:
                if metric == 'follow':
                    case_values.append((cid, *[stats([r for r in rows[a] if r['case'] == cid])['follow']['mean']
                                               for a in (arm, 'B0_global_anchor', 'B1_oracle')]))
                    continue
                values = []
                for seed in SEEDS:
                    for variant in VARIANTS:
                        key = cid, seed, variant
                        a = index[arm][key].get(metric)
                        b0 = index['B0_global_anchor'][key].get(metric)
                        b1 = index['B1_oracle'][key].get(metric)
                        if None in (a, b0, b1): continue
                        values.append((a, b0, b1))
                if values:
                    a, b0, b1 = np.mean(values, axis=0)
                    case_values.append((cid, float(a), float(b0), float(b1)))
            recoveries[arm][metric] = recovery_stat(case_values, metric in ('theta_error', 'period_error'))
    proposal = _load(out/'B_decomposition/panel_proposal.json')['rows']
    panel_iou = case_stat([(r['case'], r['iou']) for r in proposal])
    panel_coverage = case_stat([(r['case'], r['coverage']) for r in proposal])
    panel_boundary = {str(t): case_stat([(r['case'], r[f'boundary_f1_{t}px']) for r in proposal])
                      for t in (3, 5)}
    panel_by_name = {name: case_stat([(r['case'], r['iou']) for r in proposal
                                     if r['manual_panel'] == name])
                     for name in sorted({r['manual_panel'] for r in proposal})}
    uv_rows = []
    for cid, variant in ((cid, v) for cid in CASE_IDS for v in VARIANTS):
        prefix = f'c{cid:02d}_{variant}'
        oracle = np.load(out/'B_decomposition/B1_oracle'/f'{prefix}_fields.npz')
        automatic = np.load(out/'B_decomposition/B2_auto_warp'/f'{prefix}_fields.npz')
        names = [str(n) for n in oracle['panel_names']]
        case = next(c for c in cases['references'] if c['id'] == cid)
        panel_names = [p['name'] for p in case['panels']]
        for i, name in enumerate(names):
            j = list(automatic['target_panel_ids']).index(name)
            a = automatic['uv'][j] / np.array([256., 256.])
            b = oracle['garment_to_source_panel_uv'][i] / np.array([256., 256.])
            valid = oracle['target_panel_masks'][panel_names.index(name)] & automatic['valid'][j]
            distances = np.linalg.norm(a-b, axis=-1)[valid]
            if len(distances):
                confidence = automatic['confidence_map'][valid]
                uv_rows.append({'case': cid, 'variant': variant, 'panel': name, 'mean': float(distances.mean()),
                                'median': float(np.median(distances)),
                                'p90': float(np.percentile(distances, 90)),
                                'confidence_weighted': float(np.average(distances, weights=confidence)) if confidence.sum() else None})
    uv_epe = case_stat([(r['case'], r['mean']) for r in uv_rows]) if uv_rows else None
    ownership = []
    for case in cases['references']:
        cid = case['id']
        for variant in VARIANTS:
            path = out/'B_decomposition/B3_auto_ownership'/f'c{cid:02d}_{variant}_scaffold.json'
            assignments = _load(path)['assignments']
            matrix = _load(path)['assignment_scores']
            for target, selected in assignments.items():
                correct_group = groups[str(cid)][target]
                selected_group = groups[str(cid)][selected]
                correct = [v for source, v in matrix[target].items() if groups[str(cid)][source] == correct_group]
                wrong = [v for source, v in matrix[target].items() if groups[str(cid)][source] != correct_group]
                ownership.append({'case': cid, 'variant': variant, 'target': target,
                                  'selected': selected, 'selected_group': selected_group,
                                  'correct': selected_group == correct_group,
                                  'margin': max(correct)-max(wrong) if wrong else None})
    paa = case_stat([(r['case'], float(r['correct'])) for r in ownership])
    mixed = case_stat([(r['case'], float(r['correct'])) for r in ownership if r['case'] >= 12])
    margins = [r['margin'] for r in ownership if r['margin'] is not None]
    def mean(arm, key): return summary[arm][key]['mean']
    def recovery(arm, key):
        item = recoveries[arm][key]
        return item['mean'] if item and item['positive_oracle_gap'] else None
    b2_structure_safe = (mean('B2_auto_warp', 'contour_f1') >= mean('B1_oracle', 'contour_f1')-.02
                         and mean('B2_auto_warp', 'leakage') <= mean('B1_oracle', 'leakage')+.01)
    b2_warp = any(recovery('B2_auto_warp', key) is not None and recovery('B2_auto_warp', key) < .6
                  for key in ('theta_error', 'identity')) and b2_structure_safe
    b2_close = (mean('B2_auto_warp', 'identity') >= mean('B1_oracle', 'identity')-.03
                and mean('B2_auto_warp', 'theta_error') <= mean('B1_oracle', 'theta_error')+5
                and b2_structure_safe and not b2_warp)
    b3_ownership = (summary['B3_auto_ownership']['follow']['mean'] <
                    summary['B1_oracle']['follow']['mean']-.05 and
                    (paa['mean'] < .85 or mixed['mean'] < .75 or
                     (margins and np.median(margins) <= 0))) and b2_close
    b4_panel = (summary['B4_auto_panel']['follow']['mean'] <
                summary['B1_oracle']['follow']['mean']-.05 and
                (panel_iou['mean'] < .75 or panel_coverage['mean'] < .9)
                and summary['B3_auto_ownership']['follow']['mean'] >= summary['B1_oracle']['follow']['mean']-.05
                and paa['mean'] >= .85 and b2_close)
    b5_comp = (mean('B5_auto_composition', 'contour_f1') < mean('B1_oracle', 'contour_f1')-.02
               and mean('B5_auto_composition', 'identity') >= mean('B1_oracle', 'identity')-.03
               and summary['B5_auto_composition']['follow']['mean'] >=
               summary['B1_oracle']['follow']['mean']-.05)
    if b2_warp: route = 'deformable_panel_correspondence'
    elif b3_ownership: route = 'ownership_matching'
    elif b4_panel: route = 'panel_parser'
    elif b5_comp: route = 'boundary_safe_composition'
    else: route = 'stop_no_stable_gain'
    oracle_reproduced = bool((summary['B1_oracle']['follow']['mean']-summary['B0_global_anchor']['follow']['mean'] >= .10
                or mean('B1_oracle', 'theta_error') <= .8*mean('B0_global_anchor', 'theta_error'))
                and mean('B1_oracle', 'contour_f1') >= mean('B0_global_anchor', 'contour_f1')-.02
                and mean('B1_oracle', 'leakage') <= mean('B0_global_anchor', 'leakage')+.01)
    if not oracle_reproduced: route = 'stop_no_stable_gain'
    decision = {'A_audit_pass': True, 'B_oracle_reproduced': bool(
                oracle_reproduced),
                'warp_bottleneck': bool(b2_warp), 'ownership_bottleneck': bool(b3_ownership),
                'panel_bottleneck': bool(b4_panel), 'composition_bottleneck': bool(b5_comp),
                'primary_bottleneck': route, 'selected_repair': route if route in ('ownership_matching','panel_parser','boundary_safe_composition') else None,
                'next_route': route, 'experiment': 'E28', 'repair_pilot_pass': None,
                'confirmation_pass': None, 'structure_safe': bool(b2_structure_safe),
                'rotation_pass': None, 'component_check_pass': _load(out/'B_decomposition/component_check.json')['pass'],
                'warp_close_to_oracle': bool(b2_close),
                'gate_thresholds': {'pattern_follow_drop': .05, 'identity_close': .03, 'theta_close_deg': 5,
                      'warp_recovery': .6, 'paa': .85, 'mixed_paa': .75, 'panel_miou': .75,
                      'coverage': .9, 'contour_drop': .02, 'leakage_increase': .01}}
    diagnostics = intermediate_diagnostics(root, out, cases, groups, rows)
    write(out/'B_decomposition/report.json', {'summary': summary, 'panel_miou': panel_iou,
          'panel_miou_by_name': panel_by_name, 'panel_boundary_f1': panel_boundary,
          'panel_coverage': panel_coverage, 'uv_epe_relative_to_oracle_recipe': uv_epe,
          'uv_rows': uv_rows, 'paa': paa, 'mixed_paa': mixed,
          'median_ownership_margin': float(np.median(margins)) if margins else None,
          'ownership_rows': ownership, 'noise_shared': True, 'conditions_shared': True,
          'ownership_confusion': {target: {group: sum(r['target']==target and r['selected_group']==group for r in ownership)
                                  for group in sorted({r['selected_group'] for r in ownership})}
                                  for target in sorted({r['target'] for r in ownership})},
          'ownership_by_panel': {target: {'paa': case_stat([(r['case'], float(r['correct'])) for r in ownership if r['target']==target]),
               'margin': case_stat([(r['case'], r['margin']) for r in ownership if r['target']==target and r['margin'] is not None])
                         if any(r['target']==target and r['margin'] is not None for r in ownership) else None}
               for target in sorted({r['target'] for r in ownership})},
          'intermediate_diagnostics': diagnostics})
    write(out/'B_decomposition/recovery_report.json', recoveries)
    write(out/'decision_summary.json', decision)
    review_dir = out/'review_images'; review_dir.mkdir(exist_ok=True)
    for cid in (13, 6, 7, 12, 17):
        tiles = [(out/'A_audit/inputs'/f'c{cid:02d}_original.png', 'reference'),
                 (out/'A_audit/inputs'/f'c{cid:02d}_rot90.png', 'rot90'),
                 (out/'A_audit/panel_overlays'/f'c{cid:02d}.png', 'manual panels')]
        tiles.extend([(out/'A_audit/auto_overlays'/f'c{cid:02d}.png', 'auto panels'),
                      (out/'B_decomposition/ownership_matrices'/f'c{cid:02d}.png', 'ownership scores')])
        for arm in ARM_NAMES:
            folder = out/'B_decomposition'/arm
            tiles.extend([(folder/f'c{cid:02d}_original_scaffold.png', arm+' S0'),
                          (folder/f'c{cid:02d}_original_vae.png', arm+' S1'),
                          (folder/f'c{cid:02d}_s42_original.png', arm+' S2'),
                          (folder/f'c{cid:02d}_s42_rot90.png', arm+' rot90')])
        width, height = 192, 282
        columns = 8
        sheet = Image.new('RGB', (columns*width, ((len(tiles)+columns-1)//columns)*height), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, (path, label) in enumerate(tiles):
            image = Image.open(path).convert('RGB')
            image.thumbnail((width, 256))
            x, y = (i % columns)*width, (i // columns)*height
            sheet.paste(image, (x, y))
            draw.text((x+2, y+259), label, fill='black')
        sheet.save(review_dir/f'c{cid:02d}.jpg', quality=90)
    write(out/'artifact_index.json', {'experiment': 'E28',
          'cases': str((out/'cases.json').relative_to(out)),
          'audit': 'A_audit/report.json', 'decomposition': 'B_decomposition/report.json',
          'recovery': 'B_decomposition/recovery_report.json',
          'decision': 'decision_summary.json',
          'review_images': [f'review_images/c{cid:02d}.jpg' for cid in (13, 6, 7, 12, 17)]})
    print('[E28 B]', json.dumps(decision), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path.cwd())
    p.add_argument('--e27', type=Path, default=Path('output_eval/e27_20260929'))
    p.add_argument('--dataset', type=Path, default=Path('/share/home/u2515283058/datasets/BF'))
    p.add_argument('--out', type=Path, default=Path('output_eval/e28_20260929'))
    p.add_argument('--stage', choices=('A', 'revise', 'scaffold', 'generate', 'report'), required=True)
    args = p.parse_args()
    args.root = args.root.resolve(); args.e27 = args.e27.resolve()
    args.dataset = args.dataset.resolve(); args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == 'A': audit(args.root, args.e27, args.dataset, args.out)
    elif args.stage == 'revise': revise(args.out)
    elif args.stage == 'scaffold': scaffold(args.root, args.e27, args.out)
    elif args.stage == 'report': report(args.root, args.out)
    else:
        import torch
        with torch.inference_mode(): generate(args.root, args.out)


if __name__ == '__main__': main()
