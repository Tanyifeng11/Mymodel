"""C两阶段评测：固定门控诊断单列；学习臂统一v3与三域指标，统计单位为32服装。"""
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image
import torch

from tools.e2_fft_experiment import setup, load, read, write, sha
from models.e2_region_router import attach_router, LEARNED
from models.e2_compat import load_trainable_strict


def checkpoint_record(out, arm):
    return read(out / 'training' / ('C0' if arm in ['C1', 'C2'] else arm) / 'complete.json')


def training_audit(out, fixed=False):
    assert read(out / 'regression/forward_result.json')['passed'] and read(out / 'regression/image_result.json')['passed']
    if not fixed:
        assert read(out / 'smoke/result.json')['passed'] and read(out / 'audit/baseline_replay.json')['passed']
        assert read(out / 'audit/training_inputs_final.json')['passed'] and read(out / 'audit/mask_inputs_final.json')['passed']
    baseline = None; records = {}; reuse = read(out / 'audit/baseline_reuse.json')
    for arm in (['C0'] if fixed else ['C0'] + list(LEARNED)):
        complete = checkpoint_record(out, arm)
        path = Path(reuse['checkpoint']) if arm == 'C0' else out / 'training' / arm / 'latest.pt'
        assert complete['status'] == 'completed' and complete['updates'] == 2000 and sha(path) == complete['checkpoint_sha256']
        logs = [json.loads(x) for x in (out / 'training' / arm / 'training.jsonl').read_text().splitlines()]
        assert [x['step'] for x in logs] == list(range(1, 2001))
        assert all(np.isfinite(x['grad_norm']) and all(np.isfinite(v) for v in x['losses'].values()) for x in logs)
        tokens = [x['tokens'] for x in logs]
        if baseline is None: baseline = tokens
        else:
            assert baseline == tokens
            assert all(len(x['routing']) == 8 and x['eta'] == min((x['step'] - 1) / 100., 1.) for x in logs)
            assert all(np.isfinite(z) for x in logs for micro in x['routing'] for scale in micro['scales']
                for key in ['mean', 'variance', 'saturation', 'app_gain_range'] for z in scale[key])
        records[arm] = dict(updates=2000, checkpoint_sha256=complete['checkpoint_sha256'],
            training_seconds=sum(x['seconds'] for x in logs), peak_gib=max(x['peak_gib'] for x in logs), reused=arm == 'C0')
    write(out / 'audit' / ('fixed_completion.json' if fixed else 'training_completion.json'),
        dict(passed=True, same_legacy_stream=True, arms=records, fixed_probe_only=fixed))


def generate(args):
    from tools.e2_c_screen import load_c, c_environment
    core, out = setup(args); spec = read(out / 'strict_pairs.json'); arm = args.arm
    if arm == 'C0':
        print('C_BASELINE_OUTPUTS_REUSED', flush=True); return
    fixed = arm in ['C1', 'C2']
    if not fixed:
        assert read(out / 'audit/training_inputs_final.json')['passed'] and read(out / 'audit/mask_inputs_final.json')['passed']
    complete = checkpoint_record(out, arm)
    checkpoint = Path(read(out / 'audit/baseline_reuse.json')['checkpoint']) if fixed else out / 'training' / arm / 'latest.pt'
    assert sha(checkpoint) == complete['checkpoint_sha256']
    pipe, network, _ = load(args, core)
    if fixed:
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        load_trainable_strict(network, state['network']); del state
        attach_router(network, arm)
    else:
        attach_router(network, arm)
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        load_trainable_strict(network, state['network']); del state
    network.eval().requires_grad_(False); pipe.unet.half(); pipe.vae.half(); network.adapter.half()
    pipe.image_encoder.to(args.device); pipe.bf_texture_conditioner.to(args.device)
    records = []
    with torch.inference_mode():
        for index, pair in enumerate(spec['pairs']):
            for condition in ['old', 'new']:
                key = 'original_reference' if condition == 'old' else 'swapped_reference'
                sketch, reference = [Path(args.data) / pair[k] for k in ['sketch', key]]
                assert sha(sketch) == pair['hashes']['sketch'] and sha(reference) == pair['hashes'][key]
                path = out / 'outputs' / arm / pair['case_id'] / (condition + '.png')
                gate_path = out / 'gates' / arm / pair['case_id'] / (condition + '.json')
                if path.exists():
                    meta = read(path.with_suffix('.json'))
                    assert meta['output_sha256'] == sha(path) and meta['checkpoint_sha256'] == complete['checkpoint_sha256']
                    assert meta['manifest_sha256'] == sha(out / 'strict_pairs.json') and gate_path.exists()
                    records.append(meta); continue
                adapter = network.adapter; adapter.records = []
                if adapter.router is not None: adapter.router.save_maps = index < 4
                adapter.condition(core.tensor(sketch).to(args.device).half(), core.tensor(reference).to(args.device).half())
                gate_record = adapter.records[-1]
                maps = adapter.fixed_maps if fixed else adapter.router.gate_maps
                if index < 4:
                    for level, value in enumerate(maps):
                        for channel in range(value.shape[1]):
                            dest = gate_path.parent / ('%s_l%d_g%d.png' % (condition, level, channel))
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            Image.fromarray((value[0, channel].numpy().clip(0, 1) * 255).round().astype('uint8')).save(dest)
                write(gate_path, gate_record)
                latent = torch.randn((1, 4, 64, 48), device=args.device, dtype=torch.float16,
                    generator=torch.Generator(device=args.device).manual_seed(pair['seed']))
                latent_hash = core.digest(latent)
                image = pipe(ref_image=core.tensor(sketch).to(args.device).half(), prompt=pair['caption'],
                    texture_clip_image=None, texture_embeds=None, null_prompt='', negative_prompt=' worst quality, low quality',
                    width=384, height=512, num_images_per_prompt=1, guidance_scale=7., sketch_scale=.6, ipa_scale=0.,
                    generator=torch.Generator(device=args.device).manual_seed(pair['seed']), latents=latent, num_inference_steps=50)[0]
                path.parent.mkdir(parents=True, exist_ok=True); image.save(path)
                meta = dict(arm=arm, condition=condition, case_id=pair['case_id'], index=pair['index'], seed=pair['seed'],
                    caption=pair['caption'], initial_latent_sha256=latent_hash, output_sha256=sha(path),
                    checkpoint_sha256=complete['checkpoint_sha256'], input_sketch_sha256=sha(sketch),
                    input_reference_sha256=sha(reference), manifest_sha256=sha(out / 'strict_pairs.json'),
                    GT_used_in_inference=False, new_training_updates=0 if fixed else 2000,
                    checkpoint_training_updates=2000, gate_record_sha256=sha(gate_path))
                write(path.with_suffix('.json'), meta); records.append(meta); adapter.clear()
                print('C_GENERATED', arm, pair['case_id'], condition, flush=True)
    write(out / 'generation' / (arm + '.json'), dict(status='completed', outputs=len(records), records=records,
        fixed_probe_only=fixed, environment=c_environment()))


def region_metrics(image, sketch, gt, old_reference, new_reference, mask):
    """只在输入决定的三域评分；新参考的边界颜色不与旧GT RGB作准确度比较。"""
    from scipy.ndimage import distance_transform_edt
    from garment_mask_utils import build_region_masks
    from eval.eval_utils import estimate_foreground_mask
    from eval.metrics import _binary_edges, _dilate_binary, _rgb_to_lab
    from tools.e2_a_evaluation import boundary_f1
    m = mask.astype(bool)
    body = build_region_masks(torch.from_numpy(m.astype('float32'))[None, None], 9)[0][0, 0].numpy() > .5
    band, background = m & ~body, ~m
    rgb = np.asarray(image.convert('RGB'), dtype=np.float64) / 255.
    target = np.asarray(gt.convert('RGB'), dtype=np.float64) / 255.
    gray = np.asarray(image.convert('L'), dtype=np.float32)
    sk = np.asarray(sketch.convert('L'), dtype=np.float32)
    ge = _binary_edges(gray, .08, .16); se = _binary_edges(sk, .04, .12)
    gen_n, target_n = int((ge & band).sum()), int((se & band).sum())
    if target_n >= 10:
        precision = float((ge & band & _dilate_binary(se, 5)).sum()) / gen_n if gen_n else 0.
        recall = float((se & band & _dilate_binary(ge, 5)).sum()) / target_n
        band_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    else: band_f1 = None
    d = distance_transform_edt(m) - distance_transform_edt(~m)
    gy, gx = np.gradient(gray / 255.); weight = np.hypot(gx, gy) * (np.abs(d) <= 8)
    center = float((weight * d).sum() / weight.sum()) if weight.sum() else None
    spread = float(np.sqrt((weight * (d - center) ** 2).sum() / weight.sum())) if center is not None else None
    lab = _rgb_to_lab((rgb * 255).astype('uint8')).astype(np.float64)
    refs = [_rgb_to_lab(np.asarray(ref.convert('RGB'), dtype='uint8')).astype(np.float64).reshape(-1, 3).mean(0)
            for ref in [old_reference, new_reference]]
    color = {}
    for name, region in [('body', body), ('band', band)]:
        for k, ref in zip(['old', 'new'], refs):
            color[name + '_color_' + k] = float(np.linalg.norm(lab[region].mean(0) - ref)) if region.any() else None
    pred = estimate_foreground_mask(image, image.size)
    return dict(boundary_f1_r2=boundary_f1(pred, m, 2), band_edge_f1=band_f1,
        band_target_edge_pixels=target_n, band_generated_edge_pixels=gen_n,
        band_gradient_spread_pixels=spread, band_gradient_center_pixels=center,
        background_mae=float(np.abs(rgb[background] - target[background]).mean()) if background.any() else None,
        body_pixels=int(body.sum()), band_pixels=int(band.sum()), background_pixels=int(background.sum()),
        generated_mask_area=float(pred.mean()), failed_generation=not(.02 <= float(pred.mean()) <= .95), **color)


def evaluate(args):
    from tools import e2_b_screen_evaluation as legacy
    from tools.e2_c_screen import c_environment
    from tools.e2_b_screen_training import assert_snapshot
    from tools.bf_t3_three_arm_report import paired
    from tools.bf_t3_ablation_report import holm
    from eval.eval_utils import prepare_evaluation_masks
    _, out = setup(args); fixed = args.phase == 'fixed'
    arms = ['C0', 'C1', 'C2'] if fixed else ['C0'] + list(LEARNED)
    if fixed:
        digest = assert_snapshot(args)
        write(out / 'audit/fixed_input_check.json', dict(passed=True, initial_full_audit_sha256=digest,
            current_size_mtime_equal=True, note='pre-training inference probe; not the final post-training audit'))
    with patch.object(legacy, 'ARMS', arms), patch.object(legacy, 'training_audit', lambda p: training_audit(p, fixed)), \
         patch.object(legacy, 'checkpoint_record', checkpoint_record):
        legacy.evaluate(args, input_audit_name='fixed_input_check.json' if fixed else 'training_inputs_final.json')
    pairs = read(out / 'strict_pairs.json')['pairs']; regions = []
    for p in pairs:
        sketch, gt, old, new = [Image.open(Path(args.data) / p[k]).convert('RGB').resize((384, 512), Image.Resampling.BILINEAR)
            for k in ['sketch', 'gt', 'original_reference', 'swapped_reference']]
        mask = prepare_evaluation_masks((384, 512), sketch_path=str(Path(args.data) / p['sketch']), mask_policy='sketch_only')['garment']
        for arm in arms:
            for condition in ['old', 'new']:
                image = Image.open(out / 'outputs' / arm / p['case_id'] / (condition + '.png')).convert('RGB')
                regions.append(dict(arm=arm, condition=condition, case_id=p['case_id'], index=p['index'],
                    identity_group=p['identity_group'], **region_metrics(image, sketch, gt, old, new, mask)))
    write(out / 'metrics/region_per_image.json', regions)
    region = {(x['arm'], x['case_id'], x['condition']): x for x in regions}
    oldrows = {(x['arm'], x['case_id'], x['condition']): x for x in read(out / 'metrics/per_image.json')}
    protections = []
    for arm in arms[1:]:
        for condition, key, table in [('old', 'boundary_f1_r2', region), ('old', 'iou', oldrows),
             ('old', 'lpips', oldrows), ('new', 'boundary_f1_r2', region), ('new', 'iou', oldrows),
             ('new', 'leak', oldrows), ('new', 'background_mae', region), ('new', 'band_edge_f1', region)]:
            delta = []
            for p in pairs:
                a, b = [table[a, p['case_id'], condition][key] for a in ['C0', arm]]
                if a is not None and b is not None: delta.append(b - a)
            protections.append(dict(arm=arm, comparison=arm + '_minus_C0', metric=condition + '_' + key,
                family='protection', direction='lower' if key in ['lpips', 'leak', 'background_mae'] else 'higher', **paired(delta)))
    holm(protections); write(out / 'metrics/protection_statistics.json', protections)
    version = read(out / 'metrics/version.json')
    version.update(protocol=read(out / 'protocol.json'), phase=args.phase, primary_comparisons=len(arms) - 1,
        secondary_comparisons=3 * (len(arms) - 1), protection_comparisons=len(protections), noninferiority_claim=False,
        region_metric_version='C_regions_v1_old9x9_bandCanny_bgRGBMAE_Labmean_gradspread8',
        band_width_note='weighted SD of gray gradient signed-distance in +/-8px; not semantic stroke width',
        color_note='mean Lab distance to original/swapped reference; no paired new-reference GT', environment=c_environment())
    write(out / 'metrics/version.json', version)
    if fixed:
        dest = out / 'metrics/fixed'; dest.mkdir(exist_ok=True)
        for path in (out / 'metrics').glob('*.json'): shutil.copy2(path, dest / path.name)
        write(out / 'diagnostics/fixed_complete.json', dict(passed=True, generated_arms=['C1', 'C2'],
            new_training_updates=0, outputs=192, metrics=str(dest), environment=c_environment()))
    print('C_EVALUATION_COMPLETED', args.phase, flush=True)


def bundle(args):
    out = Path(args.out)
    prefixes = ['inputs','audit','smoke','comparison','training','generation','outputs','metrics','logs','tests','regression','gates','diagnostics']
    files = sorted({p for n in prefixes for p in (out / n).rglob('*') if p.is_file() and
                    p.suffix in ['.json','.jsonl','.png','.jpg','.jpeg','.txt','.out','.err','.csv']})
    files += [out / n for n in ['strict_pairs.json', 'split.json', 'protocol.json']]
    write(out / 'bundle_manifest.json', {p.relative_to(out).as_posix():dict(bytes=p.stat().st_size, sha256=sha(p)) for p in files})
    with zipfile.ZipFile(out / 'review_results.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in files + [out / 'bundle_manifest.json']: z.write(p, p.relative_to(out).as_posix())
    write(out / 'bundle_info.json', dict(bytes=(out / 'review_results.zip').stat().st_size,
        sha256=sha(out / 'review_results.zip'), files=len(files)))
    print('C_BUNDLE_COMPLETED', sha(out / 'review_results.zip'), flush=True)
