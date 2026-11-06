"""E2局部FFT实验的增量入口；历史核心/损失/推理/指标保持原样。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from models.e2_compat import (compare_snapshots, compute_losses, cpu_snapshot,
                             e2_objective, load_trainable_strict, observe_prediction)

SOURCE = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(part)
    return value.hexdigest()


def configure(args):
    from tools import bf_t3_frozen_core as core
    core.PROJECT = Path(args.project)
    core.E1 = core.PROJECT / 'output/phase1_e1_grouped_e5/checkpoint-final/joint_model.pt'
    core.ROOT = core.PROJECT / 'output_eval/bf_fusion_t3_three_arm_20261101'
    core.MASKS = core.PROJECT / 'output_eval/e2a_attr_01_20261019/masks/train'
    return core


def environment():
    import platform
    import diffusers
    return dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), diffusers=diffusers.__version__,
                gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                commit=subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip(),
                source={str(p.relative_to(SOURCE)): sha(p) for p in [
                    SOURCE / 'models/bf_original_fusion.py', SOURCE / 'models/bf_sd_skip.py',
                    SOURCE / 'tools/bf_t3_frozen_core.py', SOURCE / 'train_GAM_texture_joint.py',
                    SOURCE / 'models/e2_compat.py', Path(__file__)]})


def setup(args):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    core = configure(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    return core, out


def load(args, core):
    config = read(SOURCE / 'configs/e2_fft_regression.json')
    checkpoint = Path(args.project) / config['checkpoint']
    assert sha(checkpoint) == config['checkpoint_sha256'], '成熟E2权重哈希改变'
    pipe, network = core.load_network(args.device)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)['network']
    checked = load_trainable_strict(network, state)
    del state
    return pipe, network, checked


def evaluation_batch(pipe, row, core, index, device):
    from eval.eval_utils import prepare_evaluation_masks
    mask = prepare_evaluation_masks((384, 512), sketch_path=row['sketch'], mask_policy='sketch_only')['garment']
    return dict(vae_cloth=core.tensor(row['gt']).to(device), vae_sketch=core.tensor(row['sketch']).to(device),
                texture_image=core.tensor(row['reference']).to(device),
                garment_mask=torch.from_numpy(mask.astype(np.float32))[None, None].to(device),
                input_ids=pipe.tokenizer(row['caption'], padding='max_length', truncation=True,
                    max_length=77, return_tensors='pt').input_ids.to(device),
                dataset_index=torch.tensor([index], device=device))


def prepare(args):
    core, out = setup(args)
    config = read(SOURCE / 'configs/e2_fft_regression.json')
    manifest = Path(args.project) / config['evaluation_manifest']
    assert sha(manifest) == config['evaluation_sha256']
    rows = read(manifest)
    cases = []
    for selection in config['cases']:
        row = next(r for r in rows if r['id'] == selection['id'] and r['condition'] == 'original')
        assert row['seed'] == selection['seed']
        for key in ['sketch', 'reference', 'gt']:
            assert sha(row[key]) == row[key + '_sha256']
        cases.append(row)
    write(out / 'regression/input_manifest.json', cases)
    train = read(core.ROOT / 'training_manifest.json')
    write(out / 'audit/input_audit.json', dict(training_rows=len(train),
        train_sha256=sha(core.ROOT / 'training_manifest.json'), evaluation_sha256=sha(manifest),
        checkpoint_sha256=sha(Path(args.project) / config['checkpoint']),
        regression_cases=[r['id'] for r in cases], new_training_executed=False))
    print('PREPARED', out, flush=True)


def regression(args):
    from train_GAM_texture_joint import VGGGramStyleLoss
    core, out = setup(args)
    if not (out / 'regression/input_manifest.json').exists():
        prepare(args)
    pipe, network, checked = load(args, core)
    network.eval()
    style = VGGGramStyleLoss().to(args.device).eval().requires_grad_(False)
    records = []
    for index, row in enumerate(read(out / 'regression/input_manifest.json')):
        batch = evaluation_batch(pipe, row, core, index, args.device)
        generator = torch.Generator(device=args.device).manual_seed(42000 + index)
        initial = generator.get_state().clone()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            with observe_prediction(pipe, network) as observed:
                raw, token = core.losses(pipe, network, batch, generator, style)
            before = cpu_snapshot(raw, token, observed, e2_objective(raw), generator)
        network.adapter.clear()
        generator.set_state(initial)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            total, raw, token, observed, _ = compute_losses(pipe, network, batch, generator, style, core)
            after = cpu_snapshot(raw, token, observed, total, generator)
        errors = compare_snapshots(before, after)
        assert network.adapter.injection_counts == [3, 2, 2, 3]
        folder = out / 'regression' / row['id']
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(dict(before=before, after=after), folder / 'forward.pt')
        records.append(dict(id=row['id'], token=token, max_abs=errors,
                            feature_shapes=[list(v.shape) for v in before['features']]))
        network.adapter.clear()
        print('REGRESSION', row['id'], max(errors.values()), flush=True)
    # 保存/恢复：完整旧可训练键检查后，再做同批/同RNG前向。
    snapshot = out / 'regression/roundtrip.pt'
    torch.save(network.trainable_state(), snapshot)
    load_trainable_strict(network, torch.load(snapshot, map_location='cpu', weights_only=False))
    generator.set_state(initial)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        total, raw, token, observed, _ = compute_losses(pipe, network, batch, generator, style, core)
        restored = cpu_snapshot(raw, token, observed, total, generator)
    restored_errors = compare_snapshots(after, restored)
    write(out / 'regression/forward_result.json', dict(passed=True, checkpoint=checked,
        cases=records, restored_errors=restored_errors, environment=environment(),
        peak_gib=torch.cuda.max_memory_allocated() / 2**30, final_images_executed=False))
    print('FORWARD_REGRESSION_PASS', flush=True)


def generate_regression(args):
    from PIL import Image
    core, out = setup(args)
    assert read(out / 'regression/forward_result.json')['passed']
    pipe, network, checked = load(args, core)
    network.eval().requires_grad_(False)
    pipe.unet.half(); pipe.vae.half(); network.adapter.half()
    pipe.image_encoder.to(args.device); pipe.bf_texture_conditioner.to(args.device)
    records = []
    with torch.inference_mode():
        for row in read(out / 'regression/input_manifest.json'):
            images = []
            initial_hashes = []
            for name in ['legacy', 'all_off']:
                # 两个名字共用原推理入口，新B模块在推理中不注册网络、不增加CFG支路。
                network.adapter.condition(core.tensor(row['sketch']).to(args.device).half(),
                                          core.tensor(row['reference']).to(args.device).half())
                latent = torch.randn((1, 4, 64, 48), device=args.device, dtype=torch.float16,
                                     generator=torch.Generator(device=args.device).manual_seed(row['seed']))
                initial_hashes.append(core.digest(latent))
                image = pipe(ref_image=core.tensor(row['sketch']).to(args.device).half(), prompt=row['caption'],
                    texture_clip_image=None, texture_embeds=None, null_prompt='',
                    negative_prompt=' worst quality, low quality', width=384, height=512,
                    num_images_per_prompt=1, guidance_scale=7., sketch_scale=.6, ipa_scale=0.,
                    generator=torch.Generator(device=args.device).manual_seed(row['seed']), latents=latent,
                    num_inference_steps=50)[0]
                path = out / 'regression' / row['id'] / (name + '.png')
                image.save(path)
                images.append(np.asarray(image).copy())
                network.adapter.clear()
            assert initial_hashes[0] == initial_hashes[1]
            assert np.array_equal(images[0], images[1]), '全关推理像素发生变化'
            historical = Path(args.project) / 'output_eval/bf_t3_ablation_20261102/outputs/BF-SD-T3/original' / (row['id'] + '.png')
            meta = read(historical.with_suffix('.json'))
            assert sha(historical) == meta['output_sha256']
            old = np.asarray(Image.open(historical).convert('RGB'))
            records.append(dict(id=row['id'], seed=row['seed'], old_vs_off_pixel_max_abs=0,
                latent_sha256=initial_hashes[0], historical_latent_matches=initial_hashes[0] == meta['initial_latent_sha256'],
                historical_pixel_max_abs=int(np.abs(old.astype(int) - images[0].astype(int)).max()),
                historical_png_sha256=sha(historical)))
            print('IMAGE_REGRESSION', records[-1], flush=True)
    write(out / 'regression/image_result.json', dict(passed=True, cases=records, checkpoint=checked))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'regression', 'generate-regression',
        'prepare-protocol', 'calibrate', 'smoke', 'prepare-comparison', 'train'])
    parser.add_argument('--project', default='/share/home/u2515283058/Mymodel')
    parser.add_argument('--out', default='/share/home/u2515283058/Mymodel/output_eval/e2_fft_20261106')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--config')
    args = parser.parse_args()
    name = args.action.replace('-', '_')
    if name in ['prepare_protocol', 'calibrate', 'smoke', 'prepare_comparison', 'train']:
        from tools import e2_fft_training
        getattr(e2_fft_training, name)(args)
    else:
        globals()[name](args)


if __name__ == '__main__':
    main()
