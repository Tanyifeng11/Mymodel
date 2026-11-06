"""准备好的两臂原协议生成/评测入口；没有正式训练checkpoint时拒绝运行。"""
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import torch

from models.e2_compat import load_trainable_strict
from tools.e2_fft_experiment import environment, load, read, setup, sha, write


def evaluation_rows(args):
    return read(Path(args.project) / 'output_eval/bf_t3_ablation_20261102/evaluation.json')


def generate(args):
    core, out = setup(args)
    folder = out / 'training' / args.arm
    complete = read(folder / 'complete.json')
    assert complete['updates'] == complete['config']['updates'] == 2000
    assert sha(folder / 'latest.pt') == complete['checkpoint_sha256']
    pipe, network, _ = load(args, core)
    checkpoint = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=False)
    load_trainable_strict(network, checkpoint['network'])
    del checkpoint
    network.eval().requires_grad_(False)
    pipe.unet.half(); pipe.vae.half(); network.adapter.half()
    pipe.image_encoder.to(args.device); pipe.bf_texture_conditioner.to(args.device)
    with torch.inference_mode():
        for row in evaluation_rows(args):
            dest = out / 'outputs' / args.arm / row['condition'] / (row['id'] + '.png')
            if dest.exists():
                metadata = read(dest.with_suffix('.json'))
                assert metadata['checkpoint_sha256'] == complete['checkpoint_sha256']
                assert sha(dest) == metadata['output_sha256']
                continue
            sketch = core.tensor(row['sketch']).to(args.device).half()
            reference = core.tensor(row['reference']).to(args.device).half()
            network.adapter.condition(sketch, reference)
            latent = torch.randn((1, 4, 64, 48), device=args.device, dtype=torch.float16,
                                 generator=torch.Generator(device=args.device).manual_seed(row['seed']))
            latent_hash = core.digest(latent)
            image = pipe(ref_image=sketch, prompt=row['caption'], texture_clip_image=None, texture_embeds=None,
                null_prompt='', negative_prompt=' worst quality, low quality', width=384, height=512,
                num_images_per_prompt=1, guidance_scale=7., sketch_scale=.6, ipa_scale=0.,
                generator=torch.Generator(device=args.device).manual_seed(row['seed']), latents=latent,
                num_inference_steps=50)[0]
            dest.parent.mkdir(parents=True, exist_ok=True); image.save(dest)
            write(dest.with_suffix('.json'), dict(id=row['id'], seed=row['seed'], caption=row['caption'],
                checkpoint_sha256=complete['checkpoint_sha256'], initial_latent_sha256=latent_hash,
                output_sha256=sha(dest), GT_used_in_inference=False,
                input_sketch_sha256=sha(row['sketch']), input_reference_sha256=sha(row['reference'])))
            network.adapter.clear()
            print('GENERATED', args.arm, row['condition'], row['id'], flush=True)


def evaluate(args):
    from eval.eval_utils import prepare_evaluation_masks
    from eval.metrics import compute_structure_preservation, compute_texture_color_fidelity, compute_texture_leakage
    from tools.e2a_attr_eval import frozen_lpips
    from tools.bf_t3_three_arm_report import gray, paired, table
    from tools.bf_t3_ablation_report import holm
    from tools.t3_response_common import patches
    from tools import t3_response_run as response
    core, out = setup(args)
    completed = {arm: read(out / 'training' / arm / 'complete.json') for arm in ['E2', 'E2_FFT']}
    for arm, complete in completed.items():
        assert complete['status'] == 'completed'
        assert sha(out / 'training' / arm / 'latest.pt') == complete['checkpoint_sha256']
    response.ROOT = out
    dists = response.load_dists(args.device)
    lpips, lpips_assets = frozen_lpips()
    rows = evaluation_rows(args)
    original = {r['id']: r for r in rows if r['condition'] == 'neutral_original'}
    references = {}

    def features(parts):
        tensor = torch.from_numpy(np.stack(parts).transpose(0, 3, 1, 2).copy()).float().to(args.device) / 255
        return dists.forward_once(tensor)

    def ref(path, is_gray):
        key = (path, is_gray)
        if key not in references:
            image = np.asarray(Image.open(path).convert('RGB').resize((384, 512), Image.Resampling.LANCZOS))
            parts = [image[y:y + 64, x:x + 64] for y in np.linspace(0, 448, 4).round().astype(int)
                     for x in np.linspace(0, 320, 4).round().astype(int)]
            references[key] = features([gray(p) for p in parts] if is_gray else parts)
        return references[key]

    def distance(first, second):
        matrix = response.distances(dists, first, second)
        return float(.5 * (matrix.min(0).values.mean() + matrix.min(1).values.mean()))

    records = []
    with torch.inference_mode():
        for row in rows:
            masks = prepare_evaluation_masks((384, 512), sketch_path=row['sketch'], mask_policy='sketch_only')
            coords, candidates = patches(masks['garment'])
            hashes = []
            for arm in ['E2', 'E2_FFT']:
                path = out / 'outputs' / arm / row['condition'] / (row['id'] + '.png')
                metadata = read(path.with_suffix('.json'))
                assert metadata['checkpoint_sha256'] == completed[arm]['checkpoint_sha256']
                assert not metadata['GT_used_in_inference'] and sha(path) == metadata['output_sha256']
                assert metadata['input_sketch_sha256'] == row['sketch_sha256']
                assert metadata['input_reference_sha256'] == row['reference_sha256']
                assert metadata['seed'] == row['seed'] and metadata['caption'] == row['caption']
                hashes.append(metadata['initial_latent_sha256'])
                kwargs = dict(gen_path=str(path), sketch_path=row['sketch'], target_path=row['gt'], mask_bundle=masks)
                structure = compute_structure_preservation(**kwargs)
                value = dict(id=row['id'], arm=arm, condition=row['condition'], category=row['category'],
                    patch_n=len(coords), patch_candidates=candidates, edge_f1=structure['edge_f1'],
                    iou=structure['sketch_iou'], edge_l1=structure['edge_l1'],
                    leak=compute_texture_leakage(**kwargs)['leak_colored_frac'], lpips=None, tcf_lab=None,
                    dists_margin=None, gray_margin=None, gray_dists=None)
                if row['condition'] == 'original':
                    value['lpips'] = float(lpips(core.tensor(path).to(args.device), core.tensor(row['gt']).to(args.device)).mean())
                    value['tcf_lab'] = compute_texture_color_fidelity(texture_path=row['reference'], **kwargs)['tcf_lab_delta']
                if row['condition'].startswith('neutral') and len(coords) >= 4:
                    image = np.asarray(Image.open(path).convert('RGB'))
                    parts = [image[y:y + 64, x:x + 64] for x, y in coords]
                    for is_gray in [False, True]:
                        predicted = features([gray(p) for p in parts] if is_gray else parts)
                        new = distance(predicted, ref(row['reference'], is_gray))
                        if is_gray:
                            value['gray_dists'] = new
                        if row['condition'] == 'neutral_swapped':
                            old = distance(predicted, ref(original[row['id']]['reference'], is_gray))
                            value['gray_margin' if is_gray else 'dists_margin'] = old - new
                records.append({k: None if isinstance(v, float) and not np.isfinite(v) else v for k, v in value.items()})
            assert len(set(hashes)) == 1
    assert len(records) == 384
    lookup = {(r['arm'], r['id'], r['condition']): r for r in records}
    comparisons = []
    for condition, metrics in [('original', ['edge_f1', 'iou', 'edge_l1', 'leak', 'lpips', 'tcf_lab']),
                               ('neutral_swapped', ['dists_margin', 'gray_margin', 'gray_dists'])]:
        family = []
        for metric in metrics:
            differences = []
            for row in rows:
                if row['condition'] == condition:
                    a, b = [lookup[arm, row['id'], condition][metric] for arm in ['E2_FFT', 'E2']]
                    if a is not None and b is not None:
                        differences.append(a - b)
            family.append(dict(condition=condition, metric=metric, comparison='E2_FFT_minus_E2', **paired(differences)))
        holm(family); comparisons += family
    write(out / 'metrics/per_image.json', records); table(out / 'metrics/per_image.csv', records)
    write(out / 'metrics/paired_statistics.json', comparisons)
    write(out / 'metrics/version.json', dict(version='legacy_v1_unchanged_definitions',
        planned_identities=128, swap_identities=32, expected_outputs=384, actual_outputs=len(records),
        new_paired_families='Holm6_original_and_Holm3_response', historical_pvalues_not_reused=True,
        lpips_assets=lpips_assets, environment=environment(), same_color_pattern_test=False,
        limitation='历史32件换料不是严格同颜色异纹样配对；不能据此单独声称解决几何花型迁移'))


def bundle(args):
    _, out = setup(args)
    files = [p for p in out.rglob('*') if p.is_file()
             and p.parts[len(out.parts)] not in ['source', 'geometry_cache']
             and p.name != 'bundle_manifest.json'
             and p.suffix in ['.json', '.jsonl', '.png', '.txt', '.out', '.err', '.csv']]
    manifest = {str(p.relative_to(out)): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in files}
    write(out / 'bundle_manifest.json', manifest)
    with zipfile.ZipFile(out / 'review_results.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files + [out / 'bundle_manifest.json']:
            archive.write(path, str(path.relative_to(out)))
    print('BUNDLE', out / 'review_results.zip', (out / 'review_results.zip').stat().st_size,
          sha(out / 'review_results.zip'), flush=True)
