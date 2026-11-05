"""TG-TFM 第一阶段训练与生成；保持完整 E2/T3 损失及原 CFG。"""
import argparse
import csv
import os
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
import torch

from models.tg_tfm import TGTFMAdapter, TGTFMNetwork


def configure_core(args):
    from tools import bf_t3_frozen_core as core
    if args.project_root:
        core.PROJECT = Path(args.project_root)
        core.E1 = core.PROJECT / 'output/phase1_e1_grouped_e5/checkpoint-final/joint_model.pt'
        core.ROOT = core.PROJECT / 'output_eval/bf_fusion_t3_three_arm_20261101'
        core.MASKS = core.PROJECT / 'output_eval/e2a_attr_01_20261019/masks/train'
    if args.e1_checkpoint:
        core.E1 = Path(args.e1_checkpoint)
    if args.initialization_root:
        core.ROOT = Path(args.initialization_root)
    return core


def load_network(args, core, checkpoint=None):
    state = torch.load(checkpoint, map_location='cpu', weights_only=False) if checkpoint else None
    config = state['fusion_config'] if state else dict(variant=args.variant or 'T3')
    if state and args.variant and config['variant'] != args.variant:
        raise ValueError('指定版本与检查点不一致')
    pipe, original = core.load_network(args.device)
    original.adapter.unbind()
    if not state:
        config.update(text_dim=int(pipe.text_encoder.config.hidden_size),
                      eos_token_id=int(pipe.tokenizer.eos_token_id))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        adapter = TGTFMAdapter(**config).to(args.device)
    old = original.adapter.fusion.state_dict()
    missing = adapter.fusion.load_state_dict(old, strict=False).missing_keys
    assert all(key.startswith('text_modulators.') for key in missing)
    network = TGTFMNetwork(pipe.unet, adapter)
    if state:
        network.load_checked(state['network'])
    elif args.start_checkpoint:
        old_state = torch.load(args.start_checkpoint, map_location='cpu', weights_only=False)
        report = network.load_checked(old_state['network'], legacy=True)
        if int(os.environ.get('RANK', '0')) == 0:
            core.write(Path(args.output_dir) / config['variant'] / 'legacy_load.json', report)
    return pipe, network, state


def compute_losses(pipe, network, batch, rng, style, core):
    import torch.nn.functional as F
    from diffusers import DDIMScheduler
    from train_GAM_texture_joint import masked_edge_l1, texture_color_stat_loss, build_region_masks, masked_l1_loss
    scheduler = DDIMScheduler(beta_start=.00085, beta_end=.012, beta_schedule='scaled_linear',
                              num_train_timesteps=1000)
    device = batch['vae_cloth'].device
    with torch.no_grad():
        posterior = pipe.vae.encode(batch['vae_cloth']).latent_dist
        latent = (posterior.mean + posterior.std * torch.randn(posterior.mean.shape,
                  device=device, generator=rng)) * pipe.vae.config.scaling_factor
        sketch_latent = pipe.vae.encode(batch['vae_sketch']).latent_dist.mean * pipe.vae.config.scaling_factor
        drop = float(torch.rand((), device=device, generator=rng))
        text_ids = batch['input_ids']
        if drop < .2 or .25 <= drop < .3:
            text_ids = pipe.tokenizer('', padding='max_length', truncation=True, max_length=77,
                                      return_tensors='pt').input_ids.to(device)
        text = pipe.text_encoder(text_ids)[0]
        noise = torch.randn(latent.shape, device=device, dtype=latent.dtype, generator=rng)
        timestep = torch.randint(0, 1000, (len(latent),), device=device, generator=rng)
        sa = core.sketch_states(pipe, sketch_latent.to(pipe.reference_unet.dtype), timestep)
    reference = batch['texture_image']
    keep = not (.2 <= drop < .3)
    conditioned = reference if keep else torch.zeros_like(reference)
    noisy = scheduler.add_noise(latent, noise, timestep)
    predicted = network(noisy, timestep, text, batch['vae_sketch'], conditioned, sa,
                        text_ids=text_ids)
    mse = F.mse_loss(predicted.float(), noise.float())
    alpha = scheduler.alphas_cumprod.to(device)[timestep].reshape(-1, 1, 1, 1)
    x0 = (noisy.float() - (1 - alpha).sqrt() * predicted.float()) / alpha.sqrt()
    decoded = pipe.vae.decode(x0 / pipe.vae.config.scaling_factor).sample.float()
    # 损失仍使用历史训练区域；新增前向区域只从草图取得。
    mask = batch['garment_mask']
    gram = style(decoded, batch['vae_cloth'], mask=mask)
    edge = masked_edge_l1(decoded, batch['vae_sketch'], mask=mask)
    color = texture_color_stat_loss(decoded, reference, mask) if keep else predicted.sum() * 0
    body, _, _ = build_region_masks(mask, 9)
    body_rgb = texture_color_stat_loss(decoded, reference, body) if keep else predicted.sum() * 0
    rest = masked_l1_loss(decoded, batch['vae_cloth'], 1 - body)
    total = mse + gram + .05 * edge + .1 * color + .1 * body_rgb + .03816244801608925 * rest
    token = dict(index=int(batch['dataset_index'][0]), timestep=int(timestep[0]), drop=drop,
                 noise_sha256=core.digest(noise))
    return dict(total=total, mse=mse, gram=gram, edge=edge, color=color, body=body_rgb, rest=rest), token


def train(args, core):
    import hashlib
    import json
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs
    from diffusers.optimization import get_scheduler
    from torch.utils.data import DataLoader, DistributedSampler
    from train_GAM_texture_joint import JointTextureDataset, VGGGramStyleLoss
    acc = Accelerator(mixed_precision=args.precision,
                      gradient_accumulation_steps=args.accumulation,
                      kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    args.device = str(acc.device)
    if acc.device.type == 'cuda':
        torch.cuda.set_device(acc.local_process_index)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    folder = Path(args.output_dir) / (args.variant or 'T3')
    folder.mkdir(parents=True, exist_ok=True)
    latest = folder / 'latest.pt'
    pipe, network, resumed = load_network(args, core, latest if latest.exists() else None)
    params = [p for p in network.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, betas=(.9, .999), eps=1e-8, weight_decay=.01)
    lr = get_scheduler('cosine', optimizer, num_warmup_steps=args.warmup, num_training_steps=args.max_steps)
    manifest = Path(args.train_manifest) if args.train_manifest else core.ROOT / 'training_manifest.json'
    data_root = Path(args.data_root) if args.data_root else core.PROJECT.parent / 'datasets/BF/training'
    dataset = JointTextureDataset(str(manifest), pipe.tokenizer, str(data_root),
        width=384, height=512, texture_preprocess_mode='plain_resize', mask_cache=str(core.MASKS))
    sampler = DistributedSampler(dataset, num_replicas=acc.num_processes, rank=acc.process_index,
                                 seed=42, drop_last=False)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=args.workers)
    if not len(loader) or len(loader) % args.accumulation:
        raise ValueError('每个进程的批次数必须非零且能被梯度累积数整除')
    style = VGGGramStyleLoss().to(acc.device).eval().requires_grad_(False)
    rng = torch.Generator(device=acc.device).manual_seed(42000 + acc.process_index)
    shared = hashlib.sha256()
    if not resumed:
        for name, parameter in network.named_parameters():
            if parameter.requires_grad and '.text_modulators.' not in name:
                shared.update(name.encode() + parameter.detach().cpu().contiguous().numpy().tobytes())
    initial_shared = resumed['run']['initial_shared_sha256'] if resumed else shared.hexdigest()
    run = dict(max_steps=args.max_steps, learning_rate=args.learning_rate, warmup=args.warmup,
               world=acc.num_processes, accumulation=args.accumulation, precision=args.precision,
               manifest_sha256=core.sha(manifest), e1_sha256=core.sha(core.E1),
               bf_initialization_sha256=core.sha(core.ROOT / 'initialization/bf_fusion.pt'),
               start_checkpoint_sha256=core.sha(args.start_checkpoint) if args.start_checkpoint else None,
               text_pooling='first tokenizer EOS hidden state; matching dropped text',
               cfg='original: fusion/GAM conditional only; negative branch has no fusion',
               loss='unchanged_full_T3', initial_shared_sha256=initial_shared)
    step, stream = 0, '0' * 64
    if resumed:
        if resumed['run'] != run:
            raise ValueError('续训的数据、初始化或训练配置与已保存配置不一致')
        optimizer.load_state_dict(resumed['optimizer'])
        lr.load_state_dict(resumed['scheduler'])
        rng.set_state(resumed['rngs'][acc.process_index])
        step, stream = resumed['step'], resumed['stream_hashes'][acc.process_index]
        log = folder / 'training_log.csv'
        if acc.is_main_process and log.exists():
            records = list(csv.DictReader(log.open(encoding='utf-8')))
            retained = [row for row in records if int(row['step']) <= step]
            if len(retained) != len(records):
                shutil.copy2(log, log.with_name('training_log_before_resume_%d.csv' % time.time_ns()))
                with log.open('w', newline='', encoding='utf-8') as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                    writer.writeheader()
                    writer.writerows(retained)
    if acc.is_main_process:
        core.write(folder / 'config.json', dict(run, fusion_config=network.adapter.config,
                   trainable_parameters=sum(p.numel() for p in params),
                   source_sha256={name: core.sha(core.SOURCE / name) for name in (
                       'models/tg_tfm.py', 'tools/train_tg_tfm.py',
                       'models/bf_original_fusion.py', 'models/bf_sd_skip.py',
                       'garment_mask_utils.py', 'train_GAM_texture_joint.py')}))
    network, optimizer = acc.prepare(network, optimizer)
    network.train()
    per_epoch = len(loader) // args.accumulation
    epoch, skip = step // per_epoch, (step % per_epoch) * args.accumulation
    saved_streams = resumed['stream_hashes'] if resumed else None

    def save():
        rngs, streams = [None] * acc.num_processes, [None] * acc.num_processes
        if acc.num_processes > 1:
            torch.distributed.all_gather_object(rngs, rng.get_state().cpu())
            torch.distributed.all_gather_object(streams, stream)
        else:
            rngs[0], streams[0] = rng.get_state().cpu(), stream
        if acc.is_main_process:
            state = dict(acc.unwrap_model(network).checkpoint(), optimizer=optimizer.state_dict(),
                         scheduler=lr.state_dict(), rngs=rngs, stream_hashes=streams, run=run, step=step)
            temporary = folder / 'latest.tmp'
            torch.save(state, temporary)
            temporary.replace(latest)
            torch.save(dict(acc.unwrap_model(network).checkpoint(), step=step, run=run),
                       folder / ('step%05d.pt' % step))
        return streams

    while step < args.max_steps:
        sampler.set_epoch(epoch)
        for micro, batch in enumerate(loader):
            if micro < skip:
                continue
            batch = {k: v.to(acc.device) for k, v in batch.items()}
            with acc.accumulate(network):
                with acc.autocast():
                    raw, token = compute_losses(pipe, network, batch, rng, style, core)
                stream = hashlib.sha256(stream.encode() + json.dumps(token, sort_keys=True).encode()).hexdigest()
                if not torch.isfinite(raw['total']):
                    raise RuntimeError('训练出现非有限损失：%s' % token)
                acc.backward(raw['total'])
                if acc.sync_gradients:
                    grad = acc.clip_grad_norm_(params, 1.)
                    if not torch.isfinite(grad):
                        raise RuntimeError('训练出现非有限梯度')
                    if args.gradient_audit and step == 2:
                        bare = acc.unwrap_model(network)
                        if bare.adapter.injection_counts != [3, 2, 2, 3]:
                            raise RuntimeError('实际四尺度注入次数不符')
                        names = ['adapter.fusion.downconv1.0.weight', 'adapter.fusion.G_downconv1.0.weight',
                                 'adapter.projections.0.weight']
                        names += [name for name, p in bare.named_parameters() if '.text_modulators.' in name and name.endswith('head.3.weight')]
                        gradients = {name: float(dict(bare.named_parameters())[name].grad.float().norm()) for name in names}
                        if not all(np.isfinite(value) for value in gradients.values()) or not all(gradients[name] > 0 for name in names[:3]):
                            raise RuntimeError('梯度验收失败：%s' % gradients)
                        if acc.is_main_process:
                            core.write(folder / 'gradients.json', gradients)
                optimizer.step()
                if acc.sync_gradients:
                    lr.step()
                optimizer.zero_grad(set_to_none=True)
            acc.unwrap_model(network).adapter.clear()
            if acc.sync_gradients:
                if acc.optimizer_step_was_skipped:
                    raise RuntimeError('优化器跳步，停止以保留成功更新计数')
                step += 1
                if acc.is_main_process:
                    record = dict(step=step, seconds=time.monotonic() - started,
                                  lr=lr.get_last_lr()[0], grad=float(grad),
                                  peak_gib=torch.cuda.max_memory_allocated()/2**30,
                                  **{key: float(value.detach()) for key, value in raw.items()})
                    path = folder / 'training_log.csv'
                    exists = path.exists()
                    with path.open('a', newline='', encoding='utf-8') as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(record))
                        if not exists:
                            writer.writeheader()
                        writer.writerow(record)
                    if step <= 3 or step % 25 == 0:
                        core.write(folder / 'progress.json', dict(record, film=acc.unwrap_model(network).adapter.fusion.stats()))
                        print('TRAIN', args.variant or 'T3', json.dumps(record), flush=True)
                if step % args.checkpoint_every == 0 or step == args.max_steps:
                    saved_streams = save()
                if step >= args.max_steps:
                    break
        epoch, skip = epoch + 1, 0
    acc.wait_for_everyone()
    if acc.is_main_process:
        final = folder / ('step%05d.pt' % step)
        core.write(folder / 'complete.json', dict(updates=step, run=run,
            trainable_parameters=sum(p.numel() for p in params), seconds=time.monotonic() - started,
            checkpoint_sha256=core.sha(final), variant=acc.unwrap_model(network).adapter.config['variant'],
            stream_hashes=saved_streams))
        print('COMPLETE', folder, step, flush=True)


def generate(args, core):
    if not args.checkpoint:
        raise ValueError('生成需要 --checkpoint')
    pipe, network, state = load_network(args, core, args.checkpoint)
    network.eval().requires_grad_(False)
    # 注册但未启用的 CPU 条件模块也会影响 Diffusers._execution_device。
    pipe.image_encoder.to(args.device)
    pipe.bf_texture_conditioner.to(args.device)
    dtype = torch.float16 if torch.device(args.device).type == 'cuda' else torch.float32
    # 与历史 BF-SD 推理一致，所有参与生成的模块及 latent 使用同一精度。
    for module in (pipe.unet, pipe.vae, pipe.reference_unet, pipe.text_encoder, network.adapter):
        module.to(dtype=dtype)
    rows = core.read(args.eval_manifest or core.PROJECT / 'output_eval/bf_t3_ablation_20261102/evaluation.json')
    if args.limit:
        rows = rows[:args.limit]
    folder = Path(args.output_dir) / state['fusion_config']['variant'] / 'outputs'
    checkpoint_sha = core.sha(args.checkpoint)
    with torch.inference_mode():
        for row in rows:
            destination = folder / row['condition'] / (row['id'] + '.png')
            if destination.exists():
                meta = core.read(destination.with_suffix('.json'))
                if (meta['checkpoint_sha256'] != checkpoint_sha or meta['output_sha256'] != core.sha(destination)
                        or meta['input_sketch_sha256'] != core.sha(row['sketch'])
                        or meta['input_reference_sha256'] != core.sha(row['reference'])
                        or meta['caption'] != row['caption'] or meta['seed'] != row['seed']
                        or meta['cfg'] != 7.):
                    raise ValueError('已有生成图与当前权重或输入不一致：%s' % destination)
                continue
            sketch, reference = core.tensor(row['sketch']), core.tensor(row['reference'])
            sketch, reference = sketch.to(device=args.device, dtype=dtype), reference.to(device=args.device, dtype=dtype)
            ids = pipe.tokenizer(row['caption'], padding='max_length', truncation=True,
                max_length=pipe.tokenizer.model_max_length, return_tensors='pt').input_ids.to(args.device)
            network.adapter.condition(sketch, reference, ids)
            calls_before = (network.adapter.conditional_calls, network.adapter.unconditional_calls)
            started = time.monotonic()
            latent = torch.randn((1, 4, 64, 48), device=args.device,
                                 dtype=dtype,
                                 generator=torch.Generator(device=args.device).manual_seed(row['seed']))
            initial_hash = core.digest(latent)
            try:
                image = pipe(ref_image=sketch, prompt=row['caption'], texture_clip_image=None,
                    texture_embeds=None, null_prompt='', negative_prompt=' worst quality, low quality',
                    width=384, height=512, num_images_per_prompt=1, guidance_scale=7., sketch_scale=.6,
                    ipa_scale=0., latents=latent, num_inference_steps=50,
                    generator=torch.Generator(device=args.device).manual_seed(row['seed']))[0]
                destination.parent.mkdir(parents=True, exist_ok=True)
                image.save(destination)
                core.write(destination.with_suffix('.json'), dict(id=row['id'], condition=row['condition'],
                    split=row['split'], caption=row['caption'], seed=row['seed'], fusion_config=network.adapter.config,
                    checkpoint_sha256=checkpoint_sha, output_sha256=core.sha(destination),
                    initial_latent_sha256=initial_hash, input_sketch_sha256=core.sha(row['sketch']),
                    input_reference_sha256=core.sha(row['reference']), GT_used_in_inference=False,
                    cfg_policy='original_conditional_only', seconds=time.monotonic()-started,
                    film=network.adapter.fusion.stats(),
                    cfg_calls=[network.adapter.conditional_calls-calls_before[0], network.adapter.unconditional_calls-calls_before[1]],
                    native_size=[384, 512], cfg=7., steps=50))
            finally:
                network.adapter.clear()
            print('GENERATED', destination, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['train', 'generate'])
    parser.add_argument('--variant', choices=['T0', 'T1', 'T2', 'T3'],
                        help='第一阶段：基线/草图/面料/双分支')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--project-root')
    parser.add_argument('--initialization-root', help='已有 BF-SD 实验根目录，包含 initialization/bf_fusion.pt')
    parser.add_argument('--e1-checkpoint')
    parser.add_argument('--start-checkpoint', help='短程筛选的共同完整 E2 权重；正式训练留空')
    parser.add_argument('--train-manifest')
    parser.add_argument('--data-root')
    parser.add_argument('--eval-manifest')
    parser.add_argument('--checkpoint')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-steps', type=int, default=28365)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--warmup', type=int, default=500)
    parser.add_argument('--accumulation', type=int, default=8,
                        help='单进程累积数；双卡使用4以保持有效 batch8')
    parser.add_argument('--precision', choices=['no', 'fp16', 'bf16'], default='bf16')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--checkpoint-every', type=int, default=1000)
    parser.add_argument('--gradient-audit', action='store_true', help='第3次更新检查真实四尺度及门控梯度')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.max_steps < 1 or args.accumulation < 1 or args.checkpoint_every < 1:
        parser.error('更新数、梯度累积数与保存间隔必须为正数')
    core = configure_core(args)
    globals()[args.action](args, core)


if __name__ == '__main__':
    main()
