"""独立两阶段训练；只使用官方 noise prediction 目标。"""
import contextlib
import hashlib
import json
import math
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from .common import CONFIG, ROOT, HERE, read, write, output, sha
from .cache import CachedDataset
from .network import GarmentModel, scheduler


def parameter_hash(parameters):
    digest = hashlib.sha256()
    for name, value in parameters.items():
        digest.update(name.encode())
        digest.update(value.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save(folder, model, optimizer, step, rng, rank, world, info):
    rng_state = rng.get_state().cpu()
    states = [None] * world
    if world > 1:
        dist.all_gather_object(states, rng_state)
        optimizer.consolidate_state_dict(to=0)
    else:
        states = [rng_state]
    if rank == 0:
        values = model.save_parameters(info['stage'])
        snapshot = dict(stage=info['stage'], step=step, parameters=values, info=info)
        destination = output(folder / 'latest.pt')
        torch.save(dict(snapshot, optimizer=optimizer.state_dict(), rng_states=states), destination.with_suffix('.tmp'))
        destination.with_suffix('.tmp').replace(destination)
        torch.save(snapshot, output(folder / 'final.pt'))
        write(folder / 'progress.json', dict(step=step, checkpoint_sha256=sha(folder / 'final.pt'), **info))
        # 用严格参数集加载验证检查点，禁止静默部分加载。
        recovered = torch.load(folder / 'final.pt', map_location='cpu', weights_only=True, mmap=True)
        expected = model.stage_parameters(info['stage'])
        assert set(recovered['parameters']) == set(expected)
        assert all(torch.equal(value.detach().cpu(), recovered['parameters'][name]) for name, value in expected.items())
        del values, recovered
    if world > 1:
        dist.barrier()


def train(stage, device, rank, world, short=False):
    folder = ROOT / 'training' / ('preflight_' + stage if short else stage)
    complete = folder / 'complete.json'
    if complete.exists():
        print('TRAIN_ALREADY_COMPLETE', stage, short, flush=True)
        return
    assert 8 % world == 0
    torch.manual_seed(CONFIG['seed'])
    model = GarmentModel(stage, device, training=True)
    parameters = model.trainable()
    assert set(parameters) == set(model.stage_parameters(stage))
    initial = parameter_hash(parameters) if rank == 0 else None
    shared = [initial]
    if world > 1:
        dist.broadcast_object_list(shared, src=0)
    initial = shared[0]
    lr = CONFIG[stage + '_lr']
    if world > 1:
        from torch.distributed.optim import ZeroRedundancyOptimizer
        optimizer = ZeroRedundancyOptimizer(list(parameters.values()), optimizer_class=torch.optim.AdamW,
                    lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=CONFIG['weight_decay'], fused=True)
        wrapped = DistributedDataParallel(model, device_ids=[device.index], broadcast_buffers=False,
                                           find_unused_parameters=(stage == 'gam'))
    else:
        optimizer = torch.optim.AdamW(parameters.values(), lr=lr, weight_decay=CONFIG['weight_decay'], fused=True)
        wrapped = model
    cached = CachedDataset(short=short)
    noise_scheduler = scheduler(stage)
    count = 16 if short else 45384
    per_epoch = count // CONFIG['effective_batch']
    target_steps = 3 if short else per_epoch * CONFIG[stage + '_epochs']
    rng = torch.Generator(device=device).manual_seed(CONFIG['seed'] + 100003 * rank)
    step = 0
    previous_seconds = 0.
    if (folder / 'latest.pt').exists():
        state = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=True)
        assert state['stage'] == stage and state['info']['world'] == world
        model.load_parameters(stage, state['parameters'])
        optimizer.load_state_dict(state['optimizer'])
        rng.set_state(state['rng_states'][rank])
        step = state['step']
        previous_seconds = state['info']['seconds']
        initial = state['info']['initial_parameters_sha256']
        del state
    if rank == 0:
        print('TRAIN', stage, 'short', short, 'parameters', sum(p.numel() for p in parameters.values()),
              'world', world, 'updates', target_steps, 'start', step, flush=True)
    empty = cached.empty.to(device)
    started = time.time()
    epoch = None
    last_gradients = {}
    log = output(folder / ('steps_rank%d.jsonl' % rank)).open('a', encoding='utf-8')
    while step < target_steps:
        current_epoch = step // per_epoch
        if epoch != current_epoch:
            epoch = current_epoch
            permutation = torch.randperm(count, generator=torch.Generator().manual_seed(CONFIG['seed'] + epoch)).tolist()
        offset = (step % per_epoch) * 8
        indices = permutation[offset:offset + 8]
        local = indices[rank::world]
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for micro, index in enumerate(local):
            batch = cached.batch([index], device)
            target = batch['target_mean'] + batch['target_std'] * torch.randn(batch['target_std'].shape, generator=rng, device=device)
            target = target * 0.18215
            noise = torch.randn(target.shape, generator=rng, device=device)
            if stage == 'gam':
                # 官方 GAM 默认 noise_offset=0.05；Color Adapter 不使用该项。
                noise += 0.05 * torch.randn((1, 4, 1, 1), generator=rng, device=device)
            timestep = torch.randint(0, 1000, (1,), generator=rng, device=device)
            noisy = noise_scheduler.add_noise(target, noise, timestep)
            draw = float(torch.rand((), generator=rng, device=device))
            text = empty if .05 <= draw < .15 else batch['text']
            fabric = batch['fabric']
            if draw < .05 or .10 <= draw < .15:
                fabric = torch.zeros_like(fabric)
            sketch = (batch['sketch_mean'] + batch['sketch_std'] *
                      torch.randn(batch['sketch_std'].shape, generator=rng, device=device)) * 0.18215
            sync = contextlib.nullcontext() if world == 1 or micro == len(local) - 1 else wrapped.no_sync()
            with sync:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    prediction = wrapped(noisy, timestep, text, sketch_latent=sketch, fabric_embedding=fabric)
                    loss = F.mse_loss(prediction.float(), noise.float())
                assert torch.isfinite(loss), (stage, step, index)
                (loss / len(local)).backward()
            losses.append(float(loss.detach()))
            model.clear_cache()
        # 分模块审计只在短测/里程碑执行，避免每步逐参数同步 GPU。
        if short or step % 1000 == 0 or step + 1 == target_steps:
            groups = {'reference_unet': [], 'sketch_adapter': [], 'fabric_adapter': []}
            for name, value in parameters.items():
                if value.grad is not None:
                    group = ('reference_unet' if name.startswith('reference_unet.') else
                             'sketch_adapter' if '_ref.' in name else 'fabric_adapter')
                    groups[group].append(value.grad.detach().float().square().sum())
            last_gradients = {key: float(torch.stack(values).sum().sqrt()) if values else 0.
                              for key, values in groups.items()}
        norm = torch.nn.utils.clip_grad_norm_(list(parameters.values()), 1.0, foreach=True, error_if_nonfinite=True)
        optimizer.step()
        step += 1
        seconds = previous_seconds + time.time() - started
        row = dict(step=step, epoch=epoch, indices=indices, loss=float(np.mean(losses)), gradient_norm=float(norm),
                   gradients=last_gradients, seconds=seconds, peak_gib=torch.cuda.max_memory_allocated(device) / 2**30)
        log.write(json.dumps(row) + '\n')
        if step % 100 == 0 or short or step == target_steps:
            log.flush()
            if rank == 0:
                print('UPDATE', stage, step, '/', target_steps, 'loss', row['loss'], 'seconds', round(seconds),
                      'peak_GiB', round(row['peak_gib'], 2), flush=True)
        if step % per_epoch == 0 or step == target_steps:
            info = dict(stage=stage, world=world, effective_batch=8, target_updates=target_steps, short=short,
                        initial_parameters_sha256=initial, seconds=seconds,
                        trainable_parameters=sum(p.numel() for p in parameters.values()),
                        source_sha256={str(p.relative_to(HERE)): sha(p) for p in HERE.glob('*.py')},
                        last_gradients=last_gradients, peak_gib=row['peak_gib'],
                        noise_offset=.05 if stage == 'gam' else 0.,
                        zero_terminal_snr=(stage == 'gam'), objective='epsilon MSE only', optimizer='AdamW fused',
                        frozen_unet_storage_dtype='bfloat16', trainable_storage_dtype='float32')
            save(folder, model, optimizer, step, rng, rank, world, info)
    log.close()
    assert last_gradients['fabric_adapter'] > 0 if stage == 'color' else (
        last_gradients['reference_unet'] > 0 and last_gradients['sketch_adapter'] > 0)
    if rank == 0:
        info.update(updates=step, passed=True, no_BF_T3_parameters=True,
                    checkpoint_sha256=sha(folder / 'final.pt'), gpu_hours=info['seconds'] * world / 3600)
        write(complete, info)
    if world > 1:
        dist.barrier()
