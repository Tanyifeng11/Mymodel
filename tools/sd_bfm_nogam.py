"""单因素消融：原始 SD 起点，联合训练 BFM、四尺度投影与生成 UNet，移除 GAM。"""
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
ROOT = PROJECT / 'output_eval/sd_bfm_nogam_20261114'
PARENT = PROJECT / 'output_eval/sd_bfm_fresh_20261113'
ARM = 'SD-BFM-T3-noGAM'
ARMS = ['BF-Fashion', 'BF-SD-T3', 'SD-BFM-T3-fresh', ARM]
UPDATES = 28365
read, write, sha, digest = core.read, core.write, core.sha, core.digest


def prepare():
    if (ROOT / 'protocol.json').exists() and (ROOT / 'audit/shared_initialization.json').exists():
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
    assert sha(ROOT / 'training_manifest.json') == sha(PARENT / 'training_manifest.json')
    sd = PROJECT / 'models/stable-diffusion-v1-5'
    files = [p for part in ['unet', 'vae', 'text_encoder', 'tokenizer', 'scheduler']
             for p in (sd / part).rglob('*') if p.is_file()]
    write(ROOT / 'audit/base_model_files.json', {
        str(p.relative_to(sd)): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in files})
    assert read(ROOT / 'audit/base_model_files.json') == read(PARENT / 'audit/base_model_files.json')
    write(ROOT / 'protocol.json', dict(arm=ARM, seed=42, samples=len(train), updates=UPDATES,
        epochs=5, effective_batch=8, world=2, initialization='original SD1.5; identical shared initialization to GAM control; no GAM or reference UNet',
        e1_loaded=False, historical_bfm_loaded=False, base_model=str(sd), source_sha256=sha(__file__),
        source_commit=read(ROOT / 'deployment.json')['source_commit'],
        train_sha256=sha(ROOT / 'training_manifest.json'),
        trainable='generation UNet, original BFM, four projections; unused IP texture-token parameters excluded',
        frozen='SD VAE and text encoder; reference UNet absent',
        optimizer='bitsandbytes AdamW8bit 0.45.5; betas .9/.999, eps 1e-8, wd .01',
        unet_lr=1e-5, condition_lr=5e-5, warmup=500, schedule='cosine', precision='bf16 autocast/fp32 parameters',
        losses='epsilon MSE + GT masked Gram + .05 sketch edge + .1 garment reference RGB + .1 body reference RGB + .03816244801608925 rest GT L1',
        dropout=dict(text_only=.2, reference_only=.05, both=.05),
        inference=dict(size=[384, 512], steps=50, cfg=7., gam=False, texture_tokens=False, bfm_cond=True, bfm_uncond=False),
        evaluation=dict(primary128=128, swap32=32, rows=192, reused_baselines=ARMS[:-1]),
        control=str(PARENT), sole_change='remove reference UNet and GAM; BFM gate independent of GAM',
        limitations=['single training seed; previously inspected evaluation identities',
                     'inherited CFG training/inference mismatch and original captions for swapped references',
                     'BF-Fashion historical native resolution, losses and training budget differ']))
    pipe, network = load_network('cpu')
    write(ROOT / 'audit/shared_initialization.json', pipe.shared_initialization)
    assert pipe.reference_unet is None
    del pipe, network
    print('PREPARED', len(train), len(rows), flush=True)


def load_network(device, checkpoint=None, training=False):
    import gc
    import torch
    from adapter.attention_processor import AttnProcessor2_0, IPAttnProcessor2_0
    from models.bf_sd_nogam import NoGAMSkipFusion, NoGAMNetwork
    from tools import sd_bfm_fresh as control

    # 先按对照的原始初始化顺序构造；共享权重逐项相等，再丢弃 GAM/参考 UNet。
    # 临时参考 UNet 只用于复现初始化随机数消耗，不参与本实验的前向或训练。
    pipe, previous = control.load_network(device, training=False)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    before = previous.state_dict()
    shared = {k: v for k, v in before.items() if '.to_k_ref.' not in k and '.to_v_ref.' not in k}
    hashes = {k: digest(v) for k, v in shared.items()}
    previous.adapter.unbind()
    processors = {name: AttnProcessor2_0() if name.endswith('attn1.processor') else processor
                  for name, processor in pipe.unet.attn_processors.items()}
    pipe.unet.set_attn_processor(processors)
    pipe.register_modules(reference_unet=None)
    adapter = NoGAMSkipFusion().to(device)
    adapter.load_state_dict(previous.adapter.state_dict(), strict=True)
    network = NoGAMNetwork(pipe.unet, adapter)
    assert set(network.state_dict()) == set(shared)
    assert all(torch.equal(value, shared[name]) for name, value in network.state_dict().items())
    pipe.shared_initialization = dict(shared_tensors=len(shared), shared_elements=sum(v.numel() for v in shared.values()),
        aggregate_sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        all_shared_tensors_exactly_equal=True, gam_parameters=0, reference_unet_present=False)
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    del before, shared, previous
    gc.collect()
    if checkpoint:
        value = torch.load(checkpoint, map_location='cpu', weights_only=False)
        network.load_state_dict(value['network'], strict=True)
    pipe.unet.requires_grad_(training)
    adapter.requires_grad_(training)
    for processor in pipe.unet.attn_processors.values():
        if isinstance(processor, IPAttnProcessor2_0):
            processor.requires_grad_(False)
    if training:
        pipe.unet.enable_gradient_checkpointing()
    return pipe, network


def losses(pipe, network, batch, rng, style):
    import torch
    import torch.nn.functional as F
    from train_GAM_texture_joint import masked_edge_l1, texture_color_stat_loss, build_region_masks, masked_l1_loss
    from diffusers import DDIMScheduler
    scheduler = DDIMScheduler(beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear',num_train_timesteps=1000)
    device = batch['vae_cloth'].device
    with torch.no_grad():
        posterior = pipe.vae.encode(batch['vae_cloth']).latent_dist
        latent = (posterior.mean + posterior.std*torch.randn(posterior.mean.shape,device=device,generator=rng))*pipe.vae.config.scaling_factor
        drop = float(torch.rand((),device=device,generator=rng))
        text_ids = batch['input_ids']
        if drop < .2 or .25 <= drop < .3:
            text_ids = pipe.tokenizer('',padding='max_length',truncation=True,max_length=77,return_tensors='pt').input_ids.to(device)
        text = pipe.text_encoder(text_ids)[0]
        noise = torch.randn(latent.shape,device=device,dtype=latent.dtype,generator=rng)
        t = torch.randint(0,1000,(len(latent),),device=device,generator=rng)
    reference = batch['texture_image']
    keep = not (.2 <= drop < .3)
    conditioned_reference = reference if keep else torch.zeros_like(reference)
    noisy = scheduler.add_noise(latent,noise,t)
    predicted = network(noisy,t,text,batch['vae_sketch'],conditioned_reference)
    mse = F.mse_loss(predicted.float(),noise.float())
    alpha = scheduler.alphas_cumprod.to(device)[t].reshape(-1,1,1,1)
    x0 = (noisy.float()-(1-alpha).sqrt()*predicted.float())/alpha.sqrt()
    # VAE 冻结但保留输入梯度；两臂都走同样的解码与基础风格监督。
    decoded = pipe.vae.decode(x0/pipe.vae.config.scaling_factor).sample.float()
    mask = batch['garment_mask']
    gram = style(decoded,batch['vae_cloth'],mask=mask)
    edge = masked_edge_l1(decoded,batch['vae_sketch'],mask=mask)
    color = texture_color_stat_loss(decoded,reference,mask) if keep else predicted.sum()*0
    body,_,_ = build_region_masks(mask,9)
    body_rgb = texture_color_stat_loss(decoded,reference,body) if keep else predicted.sum()*0
    rest = masked_l1_loss(decoded,batch['vae_cloth'],1-body)
    return dict(base=mse+gram+.05*edge+.1*color,body=body_rgb,rest=rest,mse=mse,gram=gram,edge=edge,color=color), dict(
        index=int(batch['dataset_index'][0]),t=int(t[0]),drop=drop,noise_sha256=digest(noise))

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
            shared_initialization=pipe.shared_initialization,
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
                control = torch.load(PARENT / 'training/SD-BFM-T3-fresh/latest.pt', map_location='cpu', weights_only=False)
                assert control['step'] == step and control['streams'] == streams
                write(ROOT / 'audit/data_streams.json', dict(step=step, streams=streams, exactly_equal_to_gam_control=True))
                del control
        acc.wait_for_everyone()

    while step < UPDATES:
        sampler.set_epoch(epoch)
        for micro, batch in enumerate(loader):
            if micro < skip:
                continue
            batch = {k: v.to(acc.device) for k, v in batch.items()}
            with acc.accumulate(network):
                with acc.autocast():
                    raw, token = losses(pipe, network, batch, rng, style)
                    loss = raw['base'] + .1 * raw['body'] + .03816244801608925 * raw['rest']
                assert torch.isfinite(loss).item(), token
                acc.backward(loss)
                if acc.sync_gradients:
                    if step < 3:
                        groups = dict(unet=backbone, bfm=list(acc.unwrap_model(network).adapter.fusion.parameters()),
                            projections=list(acc.unwrap_model(network).adapter.projections.parameters()))
                        norms = {name: float(torch.stack([p.grad.detach().float().square().sum()
                            for p in group if p.grad is not None]).sum().sqrt()) for name, group in groups.items()}
                        assert all(np.isfinite(v) for v in norms.values())
                        if step == 2:
                            assert all(v > 0 for v in norms.values()), norms
                        if acc.is_main_process:
                            write(ROOT / 'audit' / ('joint_gradients_step%d.json' % (step+1)),
                                dict(norms, bfm_injection_counts=acc.unwrap_model(network).adapter.injection_counts,
                                     gam_parameters=0, reference_unet_present=pipe.reference_unet is not None))
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


def sample_without_gam(pipe, adapter, caption, latent, seed):
    import torch
    device = latent.device
    pipe._guidance_scale, pipe._cross_attention_kwargs, pipe._clip_skip = 7., None, None
    pipe.scheduler.set_timesteps(50, device=device)
    text, negative = pipe.encode_prompt(caption, device, 1, True, ' worst quality, low quality')
    latents = pipe.prepare_latents(1, 4, 384, 512, text.dtype, device,
        torch.Generator(device=device).manual_seed(seed), latents=latent)
    extra = pipe.prepare_extra_step_kwargs(torch.Generator(device=device).manual_seed(seed), 0.)
    for t in pipe.scheduler.timesteps:
        model_input = pipe.scheduler.scale_model_input(torch.cat([latents] * 2), t)
        adapter.enabled = True
        cond = pipe.unet(model_input[0:1], t, encoder_hidden_states=text, return_dict=False)[0]
        assert adapter.injection_counts == [3, 2, 2, 3]
        # 保持对照的 CFG 行为：无条件支路关闭 BFM，仅删除 GAM 这个因素。
        adapter.enabled = False
        uncond = pipe.unet(model_input[1:2], t, encoder_hidden_states=negative, return_dict=False)[0]
        assert adapter.injection_counts == [0, 0, 0, 0]
        predicted = uncond + 7. * (cond - uncond)
        latents = pipe.scheduler.step(predicted, t, latents, **extra, return_dict=False)[0]
    adapter.enabled = True
    image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    image = (image / 2 + .5).clamp(0, 1).cpu().permute(0, 2, 3, 1).float().numpy()
    return pipe.numpy_to_pil(image)[0]


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
            result = sample_without_gam(pipe, network.adapter, row['caption'], latent, row['seed'])
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
    assert len(records) == len(ARMS) * 192
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
        panel = Image.new('RGB', (180 * (3 + len(ARMS)), 1040), 'white')
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
    write(ROOT / 'evaluation_complete.json', dict(complete=True, outputs=192*len(ARMS), new_outputs=192,
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
