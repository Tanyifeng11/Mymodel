"""承接 E1 联合训练终点，再训练第 6—10 轮，并逐轮保存和评测。"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tarfile
import time

import numpy as np

from tools import e1_bfm_joint as previous

core = previous.core
PROJECT = previous.PROJECT
PARENT = previous.ROOT
ROOT = PROJECT / 'output_eval/e1_bfm_continue_20261117'
PARENT_ARM = previous.ARM
PARENT_UPDATES = 28365
EPOCH_UPDATES = 5673
CONTINUATION_UPDATES = 5 * EPOCH_UPDATES
PARENT_SHA = '13221b1f332191ebb86f43de4cda347d477537fd57cd71fa81cb31bc773f5588'
LEARNING_RATES = (2e-6, 1e-5)
BASELINES = ['BF-Fashion', PARENT_ARM]
read, write, sha, digest = core.read, core.write, core.sha, core.digest


def epoch_root(epoch):
    return ROOT / 'epochs' / ('epoch_%02d' % epoch)


def epoch_arm(epoch):
    return 'E1-BFM-T3-epoch%02d' % epoch


def prepare():
    import torch

    parent_folder = PARENT / 'training' / PARENT_ARM
    assert sha(parent_folder / 'final.pt') == PARENT_SHA
    final = torch.load(parent_folder / 'final.pt', map_location='cpu', weights_only=False)
    latest = torch.load(parent_folder / 'latest.pt', map_location='cpu', weights_only=False)
    assert final['step'] == latest['step'] == PARENT_UPDATES and latest['world'] == 2
    assert set(final['network']) == set(latest['network'])
    assert all(torch.equal(value, latest['network'][name]) for name, value in final['network'].items())
    assert len(latest['optimizer']['param_groups']) == 2 and latest['optimizer']['state']
    parent_streams = latest['streams']
    del final, latest
    write(ROOT / 'audit/parent_checkpoint.json', dict(final_sha256=PARENT_SHA,
        final_equals_latest=True, parent_step=PARENT_UPDATES, optimizer_moments_present=True,
        private_rngs_present=True, streams=parent_streams))
    protocol = read(PARENT / 'protocol.json')
    protocol.update(arm='E1-BFM-T3-continued', initialization='parent E1 joint final; preserve optimizer moments and private RNG',
        parent=str(PARENT), parent_checkpoint_sha256=PARENT_SHA, parent_updates=PARENT_UPDATES,
        epochs=5, total_epochs=10, updates=CONTINUATION_UPDATES, total_updates=56730,
        unet_lr=LEARNING_RATES[0], condition_lr=LEARNING_RATES[1], warmup=0,
        schedule='one cosine decay across all five continuation epochs',
        source_sha256=sha(__file__), source_commit=read(ROOT / 'deployment.json')['source_commit'],
        evaluation=dict(rows=192, primary128=128, swap32=32, every_epoch=True,
            baselines=BASELINES, primary_final_epoch=10, intermediate_epochs='exploratory'))
    write(ROOT / 'protocol.json', protocol)
    shutil.copy2(PARENT / 'training_manifest.json', ROOT / 'training_manifest.json')
    assert sha(ROOT / 'training_manifest.json') == protocol['train_sha256']
    rows = read(PARENT / 'evaluation.json')
    assert len(rows) == 192
    for row in rows:
        for key in ['sketch', 'reference', 'gt']:
            assert sha(row[key]) == row[key + '_sha256']
    for epoch in range(6, 11):
        folder = epoch_root(epoch)
        write(folder / 'protocol.json', dict(protocol, arm=epoch_arm(epoch), epoch=epoch))
        write(folder / 'evaluation.json', rows)
        for row in rows:
            for arm in BASELINES:
                old = PARENT / 'outputs' / arm / row['condition'] / (row['id'] + '.png')
                meta = read(old.with_suffix('.json'))
                assert sha(old) == meta['output_sha256']
                for key in ['sketch', 'reference']:
                    assert meta['input_' + key + '_sha256'] == row[key + '_sha256']
                target = folder / 'outputs' / arm / row['condition'] / old.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(old, target)
                shutil.copy2(old.with_suffix('.json'), target.with_suffix('.json'))
            # 复用原生成函数的初始噪声核对，其固定路径只需要旧版 metadata。
            old = PARENT / 'outputs/BF-SD-T3' / row['condition'] / (row['id'] + '.json')
            target = folder / 'outputs/BF-SD-T3' / row['condition'] / old.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old, target)
    print('PREPARED continuation epochs 6-10', flush=True)


def trim_log(path, step):
    if not path.exists():
        return
    with path.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        records = [row for row in reader if int(row['step']) <= step]
    assert [int(row['step']) for row in records] == list(range(1, step + 1))
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def train(epoch):
    import torch
    import bitsandbytes as bnb
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs
    from diffusers.optimization import get_scheduler
    from torch.utils.data import DataLoader, DistributedSampler
    from train_GAM_texture_joint import JointTextureDataset, VGGGramStyleLoss

    config = read(ROOT / 'protocol.json')
    assert config['source_sha256'] == sha(__file__)
    assert int(os.environ.get('WORLD_SIZE', '1')) == 2
    target = (epoch - 5) * EPOCH_UPDATES
    folder = ROOT / 'training'
    folder.mkdir(parents=True, exist_ok=True)
    resume = folder / 'latest.pt'
    epoch_folder = epoch_root(epoch) / 'training' / epoch_arm(epoch)
    if (epoch_folder / 'complete.json').exists():
        complete = read(epoch_folder / 'complete.json')
        assert complete['updates'] == target and sha(epoch_folder / 'final.pt') == complete['checkpoint_sha256']
        return
    acc = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=4,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    torch.cuda.set_device(acc.local_process_index)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    parent_folder = PARENT / 'training' / PARENT_ARM
    checkpoint = resume if resume.exists() else parent_folder / 'final.pt'
    pipe, network = previous.load_network(acc.device, checkpoint=checkpoint, training=True)
    backbone, condition = [], []
    for name, parameter in network.named_parameters():
        if parameter.requires_grad:
            (condition if name.startswith('adapter.') or '.processor.' in name else backbone).append(parameter)
    parameters = backbone + condition
    optimizer = bnb.optim.AdamW8bit([dict(params=backbone, lr=LEARNING_RATES[0]),
        dict(params=condition, lr=LEARNING_RATES[1])], betas=(.9, .999), eps=1e-8, weight_decay=.01)
    # 先建立新调度器，再恢复 optimizer，避免调度器构造覆盖断点中的当前 lr。
    scheduler = get_scheduler('cosine', optimizer=optimizer, num_warmup_steps=0,
        num_training_steps=CONTINUATION_UPDATES)
    state = torch.load(resume if resume.exists() else parent_folder / 'latest.pt',
        map_location='cpu', weights_only=False)
    optimizer.load_state_dict(state['optimizer'])
    rng = torch.Generator(device=acc.device)
    rng.set_state(state['rngs'][acc.process_index])
    stream = state['streams'][acc.process_index]
    if resume.exists():
        assert state['source_sha256'] == sha(__file__) and state['world'] == 2
        step = state['step']
        scheduler.load_state_dict(state['scheduler'])
    else:
        assert state['step'] == PARENT_UPDATES and state['world'] == 2
        step = 0
        for group, lr in zip(optimizer.param_groups, LEARNING_RATES):
            group['lr'] = group['initial_lr'] = lr
    assert scheduler.last_epoch == step
    expected = [lr * .5 * (1 + math.cos(math.pi * step / CONTINUATION_UPDATES)) for lr in LEARNING_RATES]
    assert np.allclose([group['lr'] for group in optimizer.param_groups], expected, rtol=1e-10, atol=1e-15)
    assert target - EPOCH_UPDATES <= step <= target
    del state
    if acc.is_main_process:
        trim_log(folder / 'training_log.csv', step)
        write(ROOT / 'audit' / ('start_epoch%02d.json' % epoch), dict(continuation_step=step,
            total_updates=PARENT_UPDATES + step, optimizer_moments_restored=True,
            unet_lr=optimizer.param_groups[0]['lr'], condition_lr=optimizer.param_groups[1]['lr'],
            backbone_parameters=sum(p.numel() for p in backbone),
            condition_parameters=sum(p.numel() for p in condition), gpu=torch.cuda.get_device_name()))
    acc.wait_for_everyone()
    data = JointTextureDataset(str(ROOT / 'training_manifest.json'), pipe.tokenizer,
        str(PROJECT.parent / 'datasets/BF/training'), width=384, height=512,
        texture_preprocess_mode='plain_resize', mask_cache=str(core.MASKS))
    sampler = DistributedSampler(data, num_replicas=2, rank=acc.process_index, seed=42, drop_last=False)
    loader = DataLoader(data, batch_size=1, sampler=sampler, num_workers=2, pin_memory=True)
    assert len(loader) // 4 == EPOCH_UPDATES and len(loader) % 4 == 0
    sampler.set_epoch(5 + step // EPOCH_UPDATES)
    skip = (step % EPOCH_UPDATES) * 4
    style = VGGGramStyleLoss().to(acc.device).eval().requires_grad_(False)
    network, optimizer = acc.prepare(network, optimizer)
    network.train()
    started = time.monotonic()

    def save(boundary=False):
        rngs, streams = [None] * 2, [None] * 2
        torch.distributed.all_gather_object(rngs, rng.get_state().cpu())
        torch.distributed.all_gather_object(streams, stream)
        if acc.is_main_process:
            weights = {name: value.detach().cpu() for name, value in acc.unwrap_model(network).state_dict().items()}
            value = dict(network=weights, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                rngs=rngs, streams=streams, step=step, total_updates=PARENT_UPDATES + step,
                world=2, source_sha256=sha(__file__), parent_checkpoint_sha256=PARENT_SHA)
            temporary = folder / 'latest.tmp'
            torch.save(value, temporary)
            temporary.replace(resume)
            if boundary:
                epoch_folder.mkdir(parents=True, exist_ok=True)
                snapshot = epoch_folder / 'final.pt'
                if snapshot.exists():
                    saved = torch.load(snapshot, map_location='cpu', weights_only=False)
                    assert saved['step'] == step and saved['source_sha256'] == sha(__file__)
                    assert all(torch.equal(tensor, saved['network'][name]) for name, tensor in weights.items())
                    del saved
                else:
                    # 后续 latest 使用原子替换，该硬链接永久保留本轮完整状态。
                    os.link(resume, snapshot)
                write(epoch_folder / 'complete.json', dict(updates=step, total_updates=PARENT_UPDATES + step,
                    epoch=epoch, world=2, effective_batch=8, checkpoint_sha256=sha(snapshot),
                    parent_checkpoint_sha256=PARENT_SHA, e1_loaded=True, historical_bfm_loaded=True))
        acc.wait_for_everyone()

    for micro, batch in enumerate(loader):
        if micro < skip:
            continue
        if step >= target:
            break
        batch = {key: value.to(acc.device) for key, value in batch.items()}
        with acc.accumulate(network):
            with acc.autocast():
                raw, token = core.losses(pipe, network, batch, rng, style)
                loss = raw['base'] + .1 * raw['body'] + .03816244801608925 * raw['rest']
            assert torch.isfinite(loss).item(), token
            acc.backward(loss)
            if acc.sync_gradients:
                if step % EPOCH_UPDATES == 0:
                    model = acc.unwrap_model(network)
                    groups = dict(unet=backbone, bfm=list(model.adapter.fusion.parameters()),
                        projections=list(model.adapter.projections.parameters()),
                        gam=[p for name, p in model.unet.named_parameters() if '.processor.' in name and p.requires_grad])
                    norms = {name: float(torch.stack([p.grad.float().square().sum() for p in group
                        if p.grad is not None]).sum().sqrt()) for name, group in groups.items()}
                    assert all(np.isfinite(value) and value > 0 for value in norms.values()), norms
                    assert model.adapter.injection_counts == [3, 2, 2, 3]
                    if acc.is_main_process:
                        write(ROOT / 'audit' / ('joint_gradients_epoch%02d.json' % epoch), norms)
                gradient = acc.clip_grad_norm_(parameters, 1.)
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
            record = dict(step=step, total_updates=PARENT_UPDATES + step, epoch=epoch,
                epoch_job_seconds=time.monotonic() - started, unet_lr=scheduler.get_last_lr()[0],
                condition_lr=scheduler.get_last_lr()[1], total=float(loss.detach()), grad=float(gradient),
                peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                **{key: float(value.detach()) for key, value in raw.items()})
            if acc.is_main_process:
                path = folder / 'training_log.csv'
                exists = path.exists()
                with path.open('a', encoding='utf-8', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(record))
                    if not exists:
                        writer.writeheader()
                    writer.writerow(record)
                if step <= 3 or step % 25 == 0 or step == target:
                    write(folder / 'progress.json', record)
                    print('TRAIN', json.dumps(record), flush=True)
            if step % 1000 == 0 or step == target:
                save(step == target)
    assert step == target
    if not (epoch_folder / 'complete.json').exists():
        save(True)


def evaluate(epoch):
    previous.ROOT = epoch_root(epoch)
    previous.ARM = epoch_arm(epoch)
    previous.ARMS = BASELINES + [previous.ARM]
    previous.UPDATES = (epoch - 5) * EPOCH_UPDATES
    previous.generate()
    previous.evaluate()


def report():
    from tools.bf_t3_three_arm_report import table

    summary, comparisons = [], []
    parent = {(row['arm'], row['split'], row['condition'], row['metric']): row
        for row in read(PARENT / 'metrics/summary.json')}
    for epoch in range(6, 11):
        folder = epoch_root(epoch)
        assert read(folder / 'evaluation_complete.json')['complete']
        for row in read(folder / 'metrics/summary.json'):
            if row['arm'] in BASELINES and row['mean'] is not None:
                old = parent[row['arm'], row['split'], row['condition'], row['metric']]
                assert row['n'] == old['n'] and abs(row['mean'] - old['mean']) < 1e-6
            summary.append(dict(row, epoch=epoch))
        comparisons.extend(dict(row, epoch=epoch) for row in read(folder / 'metrics/paired_statistics.json'))
    write(ROOT / 'metrics/epoch_summary.json', summary)
    table(ROOT / 'metrics/epoch_summary.csv', summary)
    write(ROOT / 'metrics/epoch_paired_statistics.json', comparisons)
    table(ROOT / 'metrics/epoch_paired_statistics.csv', [dict(row, ci95=json.dumps(row['ci95'])) for row in comparisons])
    write(ROOT / 'complete.json', dict(complete=True, parent_epochs=5, additional_epochs=5, final_epoch=10,
        continuation_updates=CONTINUATION_UPDATES, total_updates=PARENT_UPDATES + CONTINUATION_UPDATES,
        new_outputs=960, per_epoch_evaluation=True, comparisons=BASELINES, primary_final_epoch=10))
    files = [path for path in ROOT.rglob('*') if path.is_file() and 'source' not in path.relative_to(ROOT).parts
        and path.suffix in ['.json', '.csv', '.png'] and path.name not in ['bundle_info.json', 'bundle_manifest.json']]
    write(ROOT / 'bundle_manifest.json', {str(path.relative_to(ROOT)): dict(bytes=path.stat().st_size,
        sha256=sha(path)) for path in files})
    with tarfile.open(ROOT / 'review_results.tar.gz', 'w:gz') as bundle:
        for path in files + [ROOT / 'bundle_manifest.json']:
            bundle.add(path, arcname=str(path.relative_to(ROOT)))
    write(ROOT / 'bundle_info.json', dict(bytes=(ROOT / 'review_results.tar.gz').stat().st_size,
        sha256=sha(ROOT / 'review_results.tar.gz')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'train', 'evaluate', 'report'])
    parser.add_argument('--epoch', type=int, choices=range(6, 11))
    args = parser.parse_args()
    if args.stage in ['train', 'evaluate']:
        assert args.epoch is not None
        globals()[args.stage](args.epoch)
    else:
        globals()[args.stage]()
