"""只训练一个原始 SD 起点：联合更新 UNet、全新 BFM 和全新条件模块。"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

import numpy as np
from PIL import Image, ImageDraw

from tools import bf_t3_frozen_core as core

PROJECT, SOURCE = core.PROJECT, core.SOURCE
ROOT = PROJECT / 'output_eval/sd_bfm_fresh_20261113'
PARENT = PROJECT / 'output_eval/bf_fusion_t3_three_arm_20261101'
ARM = 'SD-BFM-T3-fresh'
ARMS = ['BF-Fashion', 'BF-SD-T3', ARM]
UPDATES = 28365
read, write, sha, digest = core.read, core.write, core.sha, core.digest


def prepare():
    if (ROOT / 'protocol.json').exists():
        assert read(ROOT / 'protocol.json')['source_sha256'] == sha(__file__)
        return
    rows = read(PARENT / 'evaluation.json')
    train = read(PARENT / 'training_manifest.json')
    assert len(rows) == 192 and len(train) == 45384
    train_ids = {Path(r['cloth']).stem.split('_')[0] for r in train}
    for row in rows:
        assert row['id'].split('_')[0] not in train_ids
        for key in ['sketch', 'reference', 'gt']:
            assert sha(row[key]) == row[key + '_sha256']
        for arm in ARMS[:-1]:
            old = PARENT / 'outputs' / arm / row['condition'] / (row['id'] + '.png')
            meta = read(old.with_suffix('.json'))
            assert sha(old) == meta['output_sha256']
            for key in ['sketch', 'reference']:
                assert meta['input_' + key + '_sha256'] == row[key + '_sha256']
            target = ROOT / 'outputs' / arm / row['condition'] / old.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old, target)
            write(target.with_suffix('.json'), dict(meta, reused_from=str(old)))
    write(ROOT / 'evaluation.json', rows)
    write(ROOT / 'training_manifest.json', train)
    sd = PROJECT / 'models/stable-diffusion-v1-5'
    files = [p for part in ['unet', 'vae', 'text_encoder', 'tokenizer', 'scheduler']
             for p in (sd / part).rglob('*') if p.is_file()]
    write(ROOT / 'audit/base_model_files.json', {
        str(p.relative_to(sd)): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in files})
    write(ROOT / 'protocol.json', dict(arm=ARM, seed=42, samples=len(train), updates=UPDATES,
        epochs=5, effective_batch=8, world=2, initialization='original SD1.5; fresh BFM, zero projections and fresh GAM',
        e1_loaded=False, historical_bfm_loaded=False, base_model=str(sd), source_sha256=sha(__file__),
        source_commit=subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip(),
        train_sha256=sha(ROOT / 'training_manifest.json'),
        trainable='generation UNet, original BFM, four projections, GAM; unused IP texture-token parameters excluded',
        frozen='SD VAE, text encoder and original-SD reference UNet',
        optimizer='bitsandbytes AdamW8bit 0.45.5; betas .9/.999, eps 1e-8, wd .01',
        unet_lr=1e-5, condition_lr=5e-5, warmup=500, schedule='cosine', precision='bf16 autocast/fp32 parameters',
        losses='epsilon MSE + GT masked Gram + .05 sketch edge + .1 garment reference RGB + .1 body reference RGB + .03816244801608925 rest GT L1',
        dropout=dict(text_only=.2, reference_only=.05, both=.05),
        inference=dict(size=[384, 512], steps=50, cfg=7., sketch_scale=.6, texture_tokens=False),
        evaluation=dict(primary128=128, swap32=32, rows=192, reused_baselines=ARMS[:-1]),
        limitations=['single training seed; previously inspected evaluation identities',
                     'initialization, historical BFM exposure and backbone trainability differ from E1 model',
                     'BF-Fashion historical native resolution and sampling budget differ',
                     'cannot attribute all differences solely to removing E1']))
    print('PREPARED', len(train), len(rows), flush=True)


def load_network(device, checkpoint=None, training=False):
    import torch
    from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
    from transformers import CLIPTokenizer, CLIPTextModel
    from adapter.attention_processor import (LogoRefSAttnProcessor2_0, LogoCacheSAttnProcessor2_0,
                                             LogoCacheCAttnProcessor2_0, IPAttnProcessor2_0)
    from pipelines.IMAGGarment_pipeline import IMAGGarment
    from models.bf_sd_skip import BFSkipFusion, BFSDNetwork

    class FreshPipeline(IMAGGarment):
        # 关闭的纹理 token 支路无需创建投影或读取任何历史 checkpoint。
        def init_proj(self):
            return None

        def load_texture_adapter(self):
            self.texture_meta = dict(initialization='SD1.5 only', e1_loaded=False)

    torch.manual_seed(42)
    sd = PROJECT / 'models/stable-diffusion-v1-5'
    unet = UNet2DConditionModel.from_pretrained(sd, subfolder='unet')
    base_probe = digest(unet.conv_in.weight)
    reference = UNet2DConditionModel.from_pretrained(sd, subfolder='unet')
    assert digest(reference.conv_in.weight) == base_probe
    processors, caches = {}, {}
    for name in unet.attn_processors:
        if name.startswith('mid_block'):
            channels = unet.config.block_out_channels[-1]
        elif name.startswith('up_blocks'):
            channels = list(reversed(unet.config.block_out_channels))[int(name.split('.')[1])]
        else:
            channels = unet.config.block_out_channels[int(name.split('.')[1])]
        if name.endswith('attn1.processor'):
            proc = LogoRefSAttnProcessor2_0(name, channels)
            attention = unet.get_submodule(name.rsplit('.processor', 1)[0])
            proc.to_k_ref.weight.data.copy_(attention.to_k.weight.data)
            # 新条件路径起初不扰动 SD，随后共同学习；不读取 E1 的投影。
            torch.nn.init.zeros_(proc.to_v_ref.weight)
            processors[name] = proc
            caches[name] = LogoCacheSAttnProcessor2_0(name, channels)
        else:
            processors[name] = IPAttnProcessor2_0(channels, unet.config.cross_attention_dim, num_tokens=16)
            caches[name] = LogoCacheCAttnProcessor2_0(name, channels, unet.config.cross_attention_dim)
    unet.set_attn_processor(processors)
    reference.set_attn_processor(caches)
    vae = AutoencoderKL.from_pretrained(sd, subfolder='vae').to(device=device, dtype=torch.float32)
    text = CLIPTextModel.from_pretrained(sd, subfolder='text_encoder').to(device=device, dtype=torch.float16)
    tokenizer = CLIPTokenizer.from_pretrained(sd, subfolder='tokenizer')
    reference.to(device=device, dtype=torch.float16).eval().requires_grad_(False)
    vae.eval().requires_grad_(False)
    text.eval().requires_grad_(False)
    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_start=.00085, beta_end=.012,
        beta_schedule='scaled_linear', clip_sample=False, set_alpha_to_one=False, steps_offset=1)
    pipe = FreshPipeline(vae=vae, reference_unet=reference, unet=unet.to(device), tokenizer=tokenizer,
        text_encoder=text, image_encoder=None, texture_ckpt=None, scheduler=scheduler,
        safety_checker=None, feature_extractor=None)
    pipe.set_texture_token_enabled(False)
    pipe.set_layer_group_enabled(True)
    pipe._setup_layer_groups()
    pipe.set_progress_bar_config(disable=True)
    adapter = BFSkipFusion().to(device)
    network = BFSDNetwork(unet, adapter)
    if checkpoint:
        value = torch.load(checkpoint, map_location='cpu', weights_only=False)
        network.load_state_dict(value['network'], strict=True)
    unet.requires_grad_(training)
    adapter.requires_grad_(training)
    for processor in unet.attn_processors.values():
        if isinstance(processor, IPAttnProcessor2_0):
            processor.requires_grad_(False)
    if training:
        unet.enable_gradient_checkpointing()
    pipe.base_probe_sha256 = base_probe
    return pipe, network


def train():
    import torch
    import bitsandbytes as bnb
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs
    from diffusers.optimization import get_scheduler
    from torch.utils.data import DataLoader, DistributedSampler
    from train_GAM_texture_joint import JointTextureDataset, VGGGramStyleLoss

    config = read(ROOT / 'protocol.json')
    assert config['source_sha256'] == sha(__file__)
    world = int(os.environ.get('WORLD_SIZE', '1'))
    assert world == config['world'] == 2
    accumulation = 8 // world
    acc = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=accumulation,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    torch.cuda.set_device(acc.local_process_index)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pipe, network = load_network(acc.device, training=True)
    params = [p for p in network.parameters() if p.requires_grad]
    backbone, condition = [], []
    for name, parameter in network.named_parameters():
        if parameter.requires_grad:
            (condition if name.startswith('adapter.') or '.processor.' in name else backbone).append(parameter)
    optimizer = bnb.optim.AdamW8bit([dict(params=backbone, lr=config['unet_lr']),
                                   dict(params=condition, lr=config['condition_lr'])],
                                  betas=(.9, .999), eps=1e-8, weight_decay=.01)
    scheduler = get_scheduler('cosine', optimizer=optimizer, num_warmup_steps=500, num_training_steps=UPDATES)
    data = JointTextureDataset(str(ROOT / 'training_manifest.json'), pipe.tokenizer,
        str(PROJECT.parent / 'datasets/BF/training'), width=384, height=512,
        texture_preprocess_mode='plain_resize', mask_cache=str(core.MASKS))
    sampler = DistributedSampler(data, num_replicas=world, rank=acc.process_index, seed=42, drop_last=False)
    loader = DataLoader(data, batch_size=1, sampler=sampler, num_workers=2, pin_memory=True)
    assert len(loader) % accumulation == 0
    style = VGGGramStyleLoss().to(acc.device).eval().requires_grad_(False)
    folder = ROOT / 'training' / ARM
    folder.mkdir(parents=True, exist_ok=True)
    rng = torch.Generator(device=acc.device).manual_seed(42000 + acc.process_index)
    step, stream = 0, '0' * 64
    resume = folder / 'latest.pt'
    if resume.exists():
        state = torch.load(resume, map_location='cpu', weights_only=False)
        assert state['source_sha256'] == sha(__file__) and state['world'] == world
        network.load_state_dict(state['network'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        rng.set_state(state['rngs'][acc.process_index])
        step, stream = state['step'], state['streams'][acc.process_index]
        del state
    if acc.is_main_process:
        write(ROOT / 'audit/initialization.json', dict(e1_loaded=False, historical_bfm_loaded=False,
            base_conv_in_sha256=pipe.base_probe_sha256, trainable_parameters=sum(p.numel() for p in params),
            backbone_parameters=sum(p.numel() for p in backbone), condition_parameters=sum(p.numel() for p in condition),
            optimizer=bnb.__version__, resumed_step=step, gpu=torch.cuda.get_device_name()))
    network, optimizer = acc.prepare(network, optimizer)
    network.train()
    epoch_steps = len(loader) // accumulation
    epoch, skip = step // epoch_steps, (step % epoch_steps) * accumulation
    started = time.monotonic()

    def save(final=False):
        states, streams = [None] * world, [None] * world
        torch.distributed.all_gather_object(states, rng.get_state().cpu())
        torch.distributed.all_gather_object(streams, stream)
        if acc.is_main_process:
            # 包含 BatchNorm running statistics；只保存 parameters 会遗漏这些缓冲区。
            weights = {k: v.detach().cpu() for k, v in acc.unwrap_model(network).state_dict().items()}
            value = dict(network=weights, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                rngs=states, streams=streams, step=step, world=world, source_sha256=sha(__file__))
            temporary = folder / 'latest.tmp'
            torch.save(value, temporary)
            temporary.replace(resume)
            if final:
                torch.save(dict(network=weights, step=step, source_sha256=sha(__file__)), folder / 'final.pt')
        acc.wait_for_everyone()

    while step < UPDATES:
        sampler.set_epoch(epoch)
        for micro, batch in enumerate(loader):
            if micro < skip:
                continue
            batch = {k: v.to(acc.device) for k, v in batch.items()}
            with acc.accumulate(network):
                with acc.autocast():
                    raw, token = core.losses(pipe, network, batch, rng, style)
                    loss = raw['base'] + .1 * raw['body'] + .03816244801608925 * raw['rest']
                assert torch.isfinite(loss).item(), token
                acc.backward(loss)
                if acc.sync_gradients:
                    if step < 3:
                        groups = dict(unet=backbone, bfm=list(acc.unwrap_model(network).adapter.fusion.parameters()),
                            projections=list(acc.unwrap_model(network).adapter.projections.parameters()),
                            gam=[p for n, p in acc.unwrap_model(network).unet.named_parameters() if '.processor.' in n and p.requires_grad])
                        norms = {name: float(torch.stack([p.grad.detach().float().square().sum()
                            for p in group if p.grad is not None]).sum().sqrt()) for name, group in groups.items()}
                        assert all(np.isfinite(v) for v in norms.values())
                        if step == 2:
                            assert all(v > 0 for v in norms.values()), norms
                        if acc.is_main_process:
                            write(ROOT / 'audit' / ('joint_gradients_step%d.json' % (step+1)), norms)
                    gradient = acc.clip_grad_norm_(params, 1.)
                    assert torch.isfinite(gradient).item()
                optimizer.step()
                if acc.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            stream = hashlib.sha256(stream.encode() + json.dumps(token, sort_keys=True).encode()).hexdigest()
            acc.unwrap_model(network).adapter.clear()
            if acc.sync_gradients:
                assert not acc.optimizer_step_was_skipped
                step += 1
                record = dict(step=step, seconds=time.monotonic() - started,
                    unet_lr=scheduler.get_last_lr()[0], condition_lr=scheduler.get_last_lr()[1],
                    total=float(loss.detach()), grad=float(gradient), peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                    **{k: float(v.detach()) for k, v in raw.items()})
                if acc.is_main_process:
                    log = folder / 'training_log.csv'
                    exists = log.exists()
                    with log.open('a', encoding='utf-8', newline='') as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(record))
                        if not exists:
                            writer.writeheader()
                        writer.writerow(record)
                    if step <= 3 or step % 25 == 0:
                        write(folder / 'progress.json', record)
                        print('TRAIN', json.dumps(record), flush=True)
                if step % 1000 == 0 or step == UPDATES:
                    save(step == UPDATES)
                if step >= UPDATES:
                    break
        skip, epoch = 0, epoch + 1
    if acc.is_main_process:
        write(folder / 'complete.json', dict(updates=step, world=world, effective_batch=8,
            checkpoint_sha256=sha(folder / 'final.pt'), seconds=time.monotonic() - started,
            e1_loaded=False, historical_bfm_loaded=False))


def generate():
    import torch
    checkpoint = ROOT / 'training' / ARM / 'final.pt'
    complete = read(checkpoint.parent / 'complete.json')
    assert complete['updates'] == UPDATES and sha(checkpoint) == complete['checkpoint_sha256']
    pipe, network = load_network('cuda', checkpoint)
    network.eval().requires_grad_(False)
    pipe.unet.half()
    pipe.vae.half()
    network.adapter.half()
    with torch.inference_mode():
        for row in read(ROOT / 'evaluation.json'):
            dest = ROOT / 'outputs' / ARM / row['condition'] / (row['id'] + '.png')
            if dest.exists():
                meta = read(dest.with_suffix('.json'))
                assert sha(dest) == meta['output_sha256'] and meta['checkpoint_sha256'] == complete['checkpoint_sha256']
                continue
            sketch = core.tensor(row['sketch']).cuda().half()
            reference = core.tensor(row['reference']).cuda().half()
            network.adapter.condition(sketch, reference)
            latent = torch.randn((1, 4, 64, 48), device='cuda', dtype=torch.float16,
                generator=torch.Generator(device='cuda').manual_seed(row['seed']))
            latent_sha = digest(latent)
            old = read((ROOT / 'outputs/BF-SD-T3' / row['condition'] / dest.name).with_suffix('.json'))
            assert latent_sha == old['initial_latent_sha256']
            result = pipe(ref_image=sketch, prompt=row['caption'], texture_clip_image=None, texture_embeds=None,
                null_prompt='', negative_prompt=' worst quality, low quality', width=384, height=512,
                num_images_per_prompt=1, guidance_scale=7., sketch_scale=.6, ipa_scale=0.,
                generator=torch.Generator(device='cuda').manual_seed(row['seed']), latents=latent,
                num_inference_steps=50)[0]
            dest.parent.mkdir(parents=True, exist_ok=True)
            result.save(dest)
            write(dest.with_suffix('.json'), dict(id=row['id'], arm=ARM, condition=row['condition'], split=row['split'],
                seed=row['seed'], initial_latent_sha256=latent_sha, output_sha256=sha(dest),
                checkpoint_sha256=complete['checkpoint_sha256'], GT_used_in_inference=False,
                input_sketch_sha256=row['sketch_sha256'], input_reference_sha256=row['reference_sha256'],
                steps=50, cfg=7., native_size=[384, 512]))
            network.adapter.clear()
            print('GENERATED', row['id'], row['condition'], flush=True)
    write(ROOT / 'generation_complete.json', dict(complete=True, outputs=192))


def evaluate():
    import torch
    from eval.eval_utils import prepare_evaluation_masks
    from eval.metrics import compute_structure_preservation, compute_texture_color_fidelity, compute_texture_leakage
    from tools.e2a_attr_eval import frozen_lpips
    from tools import t3_response_run as response
    from tools.t3_response_common import patches
    from tools.bf_t3_three_arm_report import gray, paired, table

    response.ROOT = ROOT
    dists = response.load_dists('cuda')
    lpips, assets = frozen_lpips()
    write(ROOT / 'audit/lpips_assets.json', assets)
    rows = read(ROOT / 'evaluation.json')
    original = {r['id']: r for r in rows if r['condition'] == 'original'}
    reference_cache = {}

    def features(parts):
        value = torch.from_numpy(np.stack(parts).transpose(0, 3, 1, 2).copy()).float().cuda() / 255
        return dists.forward_once(value)

    def ref(path, is_gray):
        key = (path, is_gray)
        if key not in reference_cache:
            image = np.asarray(Image.open(path).convert('RGB').resize((384, 512), Image.Resampling.LANCZOS))
            parts = [image[y:y+64, x:x+64] for y in np.linspace(0, 448, 4).round().astype(int)
                     for x in np.linspace(0, 320, 4).round().astype(int)]
            reference_cache[key] = features([gray(p) for p in parts] if is_gray else parts)
        return reference_cache[key]

    def distance(a, b):
        matrix = response.distances(dists, a, b)
        return float(.5 * (matrix.min(0).values.mean() + matrix.min(1).values.mean()))

    records = []
    with torch.inference_mode():
        for row in rows:
            masks = prepare_evaluation_masks((384, 512), sketch_path=row['sketch'], mask_policy='sketch_only')
            coords, _ = patches(masks['garment'])
            for arm in ARMS:
                path = ROOT / 'outputs' / arm / row['condition'] / (row['id'] + '.png')
                meta = read(path.with_suffix('.json'))
                assert sha(path) == meta['output_sha256']
                kwargs = dict(gen_path=str(path), sketch_path=row['sketch'], target_path=row['gt'], mask_bundle=masks)
                structure = compute_structure_preservation(**kwargs)
                record = dict(arm=arm, id=row['id'], split=row['split'], condition=row['condition'],
                    edge_f1=structure.get('edge_f1'), iou=structure.get('sketch_iou'), edge_l1=structure.get('edge_l1'),
                    leak=compute_texture_leakage(**kwargs).get('leak_colored_frac'), lpips=None, tcf_lab=None,
                    dists_margin=None, gray_margin=None, gray_dists=None, patch_n=len(coords))
                if row['condition'] == 'original':
                    record['lpips'] = float(lpips(core.tensor(path).cuda(), core.tensor(row['gt']).cuda()).mean())
                    record['tcf_lab'] = compute_texture_color_fidelity(texture_path=row['reference'], **kwargs).get('tcf_lab_delta')
                if row['condition'] == 'swapped' and len(coords) >= 4:
                    image = np.asarray(Image.open(path).convert('RGB'))
                    parts = [image[y:y+64, x:x+64] for x, y in coords]
                    for is_gray in [False, True]:
                        generated = features([gray(p) for p in parts] if is_gray else parts)
                        old_distance = distance(generated, ref(original[row['id']]['reference'], is_gray))
                        new_distance = distance(generated, ref(row['reference'], is_gray))
                        record['gray_margin' if is_gray else 'dists_margin'] = old_distance - new_distance
                        if is_gray:
                            record['gray_dists'] = new_distance
                records.append({k: None if isinstance(v, float) and not np.isfinite(v) else v for k, v in record.items()})
            print('EVALUATED', row['id'], row['condition'], flush=True)
    assert len(records) == 576
    write(ROOT / 'metrics/per_image.json', records)
    table(ROOT / 'metrics/per_image.csv', records)
    metrics = ['edge_f1', 'iou', 'edge_l1', 'leak', 'lpips', 'tcf_lab', 'dists_margin', 'gray_margin', 'gray_dists']
    summary, comparisons = [], []
    lookup = {(r['arm'], r['id'], r['condition']): r for r in records}
    for arm in ARMS:
        for split, condition in [('heldout128', 'original'), ('swap32', 'original'), ('swap32', 'swapped')]:
            for metric in metrics:
                selected = [r for r in records if r['arm'] == arm and r['split'] == split and r['condition'] == condition]
                values = [r[metric] for r in selected if r[metric] is not None]
                summary.append(dict(arm=arm, split=split, condition=condition, metric=metric,
                    n=len(values), planned_n=len(selected), mean=float(np.mean(values)) if values else None))
    for other in ARMS[:-1]:
        family = []
        for metric in metrics:
            condition = 'swapped' if metric in metrics[6:] else 'original'
            selected = [r for r in rows if r['condition'] == condition and (condition == 'swapped' or r['split'] == 'heldout128')]
            values = []
            for row in selected:
                a, b = lookup[ARM, row['id'], condition][metric], lookup[other, row['id'], condition][metric]
                if a is not None and b is not None:
                    values.append(a - b)
            family.append(dict(comparison=ARM + ' minus ' + other, metric=metric, **paired(values)))
        running = 0.
        ordered = sorted([r for r in family if r['p'] is not None], key=lambda r: r['p'])
        for i, value in enumerate(ordered):
            running = max(running, min(1., value['p'] * (len(ordered)-i)))
            value['p_holm'] = running
        comparisons.extend(family)
    write(ROOT / 'metrics/summary.json', summary)
    table(ROOT / 'metrics/summary.csv', summary)
    write(ROOT / 'metrics/paired_statistics.json', comparisons)
    table(ROOT / 'metrics/paired_statistics.csv', [dict(r, ci95=json.dumps(r['ci95'])) for r in comparisons])
    chosen = sorted([r for r in rows if r['split'] == 'heldout128'], key=lambda r: r['sketch_sha256'])[:4]
    chosen += sorted([r for r in rows if r['condition'] == 'swapped'], key=lambda r: r['sketch_sha256'])[:4]
    (ROOT / 'figures').mkdir(exist_ok=True)
    for page in range(2):
        panel = Image.new('RGB', (1080, 1040), 'white')
        draw = ImageDraw.Draw(panel)
        for j, row in enumerate(chosen[page*4:page*4+4]):
            paths = [row['sketch'], row['reference'], row['gt'] if row['condition'] == 'original' else None]
            paths += [ROOT / 'outputs' / a / row['condition'] / (row['id']+'.png') for a in ARMS]
            for col, (label, path) in enumerate(zip(['Sketch', 'Reference', 'GT original only']+ARMS, paths)):
                draw.text((col*180+3, j*260+2), label, fill='black')
                if path:
                    im = Image.open(path).convert('RGB')
                    im.thumbnail((174, 226), Image.Resampling.LANCZOS)
                    panel.paste(im, (col*180, j*260+22))
            draw.text((3, j*260+246), row['id']+' '+row['condition'], fill='black')
        panel.save(ROOT / 'figures' / ('fixed_%d.png' % page))
    write(ROOT / 'evaluation_complete.json', dict(complete=True, outputs=576, new_outputs=192,
        new_training_runs=1, updates=UPDATES, baselines_recomputed_under_same_metrics=True))


def package():
    assert read(ROOT / 'evaluation_complete.json')['complete']
    files = [p for folder in ['audit', 'training', 'metrics', 'figures', 'outputs']
             for p in (ROOT / folder).rglob('*') if p.is_file() and p.suffix in ['.json', '.csv', '.png', '.psv']]
    files += [ROOT / p for p in ['protocol.json', 'evaluation.json', 'generation_complete.json', 'evaluation_complete.json', 'submitted_jobs.json']]
    write(ROOT / 'bundle_manifest.json', {str(p.relative_to(ROOT)): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in files})
    with tarfile.open(ROOT / 'review_results.tar.gz', 'w:gz') as tar:
        for p in files + [ROOT / 'bundle_manifest.json']:
            tar.add(p, arcname=str(p.relative_to(ROOT)))
    write(ROOT / 'bundle_info.json', dict(bytes=(ROOT / 'review_results.tar.gz').stat().st_size,
        sha256=sha(ROOT / 'review_results.tar.gz')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'train', 'generate', 'evaluate', 'package'])
    globals()[parser.parse_args().stage]()
