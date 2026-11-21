"""承接第10轮原始 T3 设置，联合续训第11—20轮并逐轮评测。"""
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
from tools import e1_bfm_reference_region as evaluation
from tools.e1_bfm_continue import trim_log

core = previous.core
PROJECT, SOURCE = core.PROJECT, Path(__file__).resolve().parents[1]
ROOT = PROJECT / 'output_eval/e1_bfm_continue20_20261121'
PARENT = PROJECT / 'output_eval/e1_bfm_continue_20261117/epochs/epoch_10'
REFERENCE = PROJECT / 'output_eval/e1_bfm_reference_region_20261119/final_comparison'
PARENT_ARM = 'E1-BFM-T3-epoch10'
PARENT_FILE = PARENT / 'training' / PARENT_ARM / 'final.pt'
PARENT_SHA = '40beb650de16ee648e6339318e7471f853ee076104490286eba80ea1b97dc213'
PARENT_UPDATES, EPOCH_UPDATES, CONTINUATION_UPDATES = 56730, 5673, 56730
LEARNING_RATES, LR_FLOOR_RATIO = (1e-6, 5e-6), .1
BASELINES = ['BF-Fashion', PARENT_ARM, 'E1-RR-control-e12', 'E1-RR-texture-e12']
SOURCE_FILES = ['tools/e1_bfm_continue20.py', 'tools/e1_bfm_continue.py',
    'tools/e1_bfm_joint.py', 'tools/bf_t3_frozen_core.py', 'tools/e1_bfm_reference_region.py',
    'models/bf_sd_skip.py', 'models/bf_original_fusion.py', 'train_GAM_texture_joint.py',
    'pipelines/IMAGGarment_pipeline.py', 'adapter/attention_processor.py', 'texture_preprocess.py']
read, write, sha, digest = core.read, core.write, core.sha, core.digest


def source_hashes():
    return {name: sha(SOURCE / name) for name in SOURCE_FILES}


def lr_factor(step):
    return LR_FLOOR_RATIO + (1-LR_FLOOR_RATIO)*.5*(1+math.cos(math.pi*step/CONTINUATION_UPDATES))


def epoch_root(epoch):
    return ROOT / 'epochs' / ('epoch_%02d' % epoch)


def epoch_arm(epoch):
    return 'E1-BFM-T3-epoch%02d' % epoch


def prepare():
    import torch
    assert sha(PARENT_FILE) == PARENT_SHA
    state = torch.load(PARENT_FILE, map_location='cpu', weights_only=False)
    assert state['total_updates'] == PARENT_UPDATES and state['step'] == 28365 and state['world'] == 2
    assert state['optimizer']['state'] and len(state['optimizer']['param_groups']) == 2
    assert len(state['rngs']) == len(state['streams']) == 2
    assert all(group['lr'] == 0 for group in state['optimizer']['param_groups'])
    write(ROOT/'audit/parent.json', dict(checkpoint=str(PARENT_FILE), sha256=PARENT_SHA,
        total_updates=PARENT_UPDATES, full_optimizer_and_private_rngs=True, streams=state['streams']))
    del state
    protocol = read(PARENT/'protocol.json')
    protocol.update(arm='E1-BFM-T3-continued20', parent=str(PARENT_FILE),
        parent_checkpoint_sha256=PARENT_SHA, parent_updates=PARENT_UPDATES,
        epochs=10, total_epochs=20, updates=CONTINUATION_UPDATES, total_updates=113460,
        unet_lr=LEARNING_RATES[0], condition_lr=LEARNING_RATES[1],
        final_unet_lr=1e-7, final_condition_lr=5e-7, lr_floor_ratio=LR_FLOOR_RATIO, warmup=0,
        schedule='one cosine across all ten continuation epochs, ending at 10% of starting LR',
        optimizer_resume='preserve parent AdamW8bit moments and both private RNGs; reset only LR/scheduler',
        sampler_epochs=list(range(10,20)), source_sha256=sha(__file__), source_hashes=source_hashes(),
        source_commit=read(ROOT/'deployment.json')['source_commit'],
        loss_implementation='unchanged core.losses and historical VGGGramStyleLoss from epoch10',
        evaluation=dict(rows=192, primary128=128, swap32=32, every_epoch=True,
            baselines=BASELINES, primary_final_epoch=20, intermediate_epochs='exploratory'))
    if (ROOT/'protocol.json').exists():
        assert read(ROOT/'protocol.json') == protocol
    else:
        write(ROOT/'protocol.json', protocol)
    shutil.copy2(PROJECT/'output_eval/e1_bfm_continue_20261117/training_manifest.json', ROOT/'training_manifest.json')
    assert sha(ROOT/'training_manifest.json') == protocol['train_sha256']
    rows = read(REFERENCE.parent/'evaluation.json')
    assert len(rows) == 192 and len({(r['id'],r['condition']) for r in rows}) == 192
    train_ids = {Path(r['cloth']).stem.split('_')[0] for r in read(ROOT/'training_manifest.json')}
    assert not train_ids & {r['id'].split('_')[0] for r in rows}
    for row in rows:
        for key in ['sketch','reference','gt']:
            assert sha(row[key]) == row[key+'_sha256']
        for arm in BASELINES:
            path = REFERENCE/'outputs'/arm/row['condition']/(row['id']+'.png')
            meta = read(path.with_suffix('.json'))
            assert sha(path) == meta['output_sha256']
            assert meta['input_sketch_sha256'] == row['sketch_sha256']
            assert meta['input_reference_sha256'] == row['reference_sha256']
            assert meta['GT_used_in_inference'] is False
            if arm != 'BF-Fashion':
                assert meta['initial_latent_sha256'] == row['initial_latent_sha256']
    write(ROOT/'evaluation.json', rows)
    # 固定历史输入、输出及指标；每轮重算后核对，防止评测口径漂移。
    shutil.copy2(REFERENCE/'metrics/per_image.json', ROOT/'audit/baseline_per_image.json')
    for epoch in range(11,21):
        where = epoch_root(epoch)
        write(where/'protocol.json', dict(protocol, arm=epoch_arm(epoch), epoch=epoch))
        write(where/'evaluation.json', rows)
        for arm in BASELINES:
            evaluation.copy_outputs(REFERENCE, where, arm, rows)
    write(ROOT/'audit/prepared.json', dict(passed=True, source_hashes=source_hashes(),
        parent_sha256=PARENT_SHA, baseline_arms=BASELINES, rows=192, train_test_identity_overlap=0))
    print('PREPARED epochs 11-20', flush=True)


def train(epoch, smoke=False):
    import torch
    import bitsandbytes as bnb
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs
    from torch.optim.lr_scheduler import LambdaLR
    from torch.utils.data import DataLoader, DistributedSampler
    from train_GAM_texture_joint import JointTextureDataset, VGGGramStyleLoss

    config = read(ROOT / 'protocol.json')
    assert config['source_sha256'] == sha(__file__) and config['source_hashes'] == source_hashes()
    if not smoke:
        assert read(ROOT/'audit/preflight_passed.json')['source_hashes'] == source_hashes()
    assert int(os.environ.get('WORLD_SIZE', '1')) == 2
    target = 3 if smoke else (epoch - 10) * EPOCH_UPDATES
    folder = ROOT / ('audit/preflight_training' if smoke else 'training')
    folder.mkdir(parents=True, exist_ok=True)
    resume = folder / 'latest.pt'
    epoch_folder = folder if smoke else epoch_root(epoch) / 'training' / epoch_arm(epoch)
    if not smoke and (epoch_folder / 'complete.json').exists():
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
    scheduler = LambdaLR(optimizer, lr_lambda=lr_factor)
    state = torch.load(resume if resume.exists() else PARENT_FILE,
        map_location='cpu', weights_only=False)
    optimizer.load_state_dict(state['optimizer'])
    assert len(optimizer.state) > 0 and len(optimizer.param_groups) == 2
    restored_optimizer_entries = len(optimizer.state)
    rng = torch.Generator(device=acc.device)
    rng.set_state(state['rngs'][acc.process_index])
    stream = state['streams'][acc.process_index]
    if resume.exists():
        assert state['source_sha256'] == sha(__file__) and state['world'] == 2
        assert state['total_updates'] == PARENT_UPDATES + state['step']
        step = state['step']
        scheduler.load_state_dict(state['scheduler'])
    else:
        assert state['total_updates'] == PARENT_UPDATES and state['step'] == 28365 and state['world'] == 2
        step = 0
        for group, lr in zip(optimizer.param_groups, LEARNING_RATES):
            group['lr'] = group['initial_lr'] = lr
    assert scheduler.last_epoch == step
    expected = [lr * lr_factor(step) for lr in LEARNING_RATES]
    assert np.allclose([group['lr'] for group in optimizer.param_groups], expected, rtol=1e-10, atol=1e-15)
    assert target - EPOCH_UPDATES <= step <= target
    del state
    if acc.is_main_process:
        trim_log(folder / 'training_log.csv', step)
        write(ROOT / 'audit' / ('smoke_start.json' if smoke else 'start_epoch%02d.json' % epoch), dict(continuation_step=step,
            total_updates=PARENT_UPDATES + step, optimizer_moments_restored=True,
            optimizer_state_entries=restored_optimizer_entries, private_rngs_restored=True, sampler_epoch=10+step//EPOCH_UPDATES,
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
    sampler.set_epoch(10 + step // EPOCH_UPDATES)
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
            if not smoke and (step % 1000 == 0 or step == target):
                save(step == target)
    assert step == target
    if smoke:
        if acc.is_main_process:
            write(ROOT/'audit/preflight_passed.json', dict(passed=True, smoke_steps=step,
                source_hashes=source_hashes(), parent_sha256=PARENT_SHA, joint_gradient_norms=norms,
                optimizer_state_entries=restored_optimizer_entries, private_rngs_restored=True,
                starting_lrs=list(LEARNING_RATES), next_lrs=scheduler.get_last_lr(),
                injection_counts=acc.unwrap_model(network).adapter.injection_counts))
        acc.wait_for_everyone()
    elif not (epoch_folder / 'complete.json').exists():
        save(True)


def evaluate(epoch):
    where, arm = epoch_root(epoch), epoch_arm(epoch)
    rows = read(ROOT/'evaluation.json')
    evaluation.generate(where, arm, where/'training'/arm/'final.pt', rows, (epoch-10)*EPOCH_UPDATES)
    evaluation.evaluate(where, BASELINES+[arm], rows, 'heldout128')
    frozen = {(r['arm'],r['id'],r['condition']):r for r in read(ROOT/'audit/baseline_per_image.json')}
    for row in read(where/'metrics/per_image.json'):
        if row['arm'] in BASELINES:
            old = frozen[row['arm'],row['id'],row['condition']]
            assert row.keys() == old.keys()
            for key,value in row.items():
                assert (abs(value-old[key])<1e-6 if isinstance(value,float) else value==old[key]), (key,row)
    marker = read(where/'evaluation_complete.json')
    write(where/'evaluation_complete.json', dict(marker, baselines_match_historical=True,
        epoch=epoch, new_outputs=192, primary_final_epoch=20))


def report():
    from tools.bf_t3_three_arm_report import table
    summary, comparisons, checkpoints = [], [], {}
    for epoch in range(11,21):
        where = epoch_root(epoch)
        assert read(where/'evaluation_complete.json')['baselines_match_historical']
        ckpt = where/'training'/epoch_arm(epoch)
        complete = read(ckpt/'complete.json')
        assert complete['updates'] == (epoch-10)*EPOCH_UPDATES
        assert complete['total_updates'] == epoch*EPOCH_UPDATES
        assert sha(ckpt/'final.pt') == complete['checkpoint_sha256']
        checkpoints[epoch_arm(epoch)] = complete['checkpoint_sha256']
        summary.extend(dict(row,epoch=epoch) for row in read(where/'metrics/summary.json'))
        comparisons.extend(dict(row,epoch=epoch) for row in read(where/'metrics/paired_statistics.json'))
    write(ROOT/'metrics/epoch_summary.json', summary)
    table(ROOT/'metrics/epoch_summary.csv', summary)
    write(ROOT/'metrics/epoch_paired_statistics.json', comparisons)
    table(ROOT/'metrics/epoch_paired_statistics.csv', [dict(row,ci95=json.dumps(row['ci95'])) for row in comparisons])
    write(ROOT/'complete.json', dict(complete=True, parent_epochs=10, additional_epochs=10,
        final_epoch=20, primary_final_epoch=20, continuation_updates=CONTINUATION_UPDATES,
        total_updates=PARENT_UPDATES+CONTINUATION_UPDATES, new_outputs=1920,
        per_epoch_checkpoint_and_evaluation=True, comparisons=BASELINES, checkpoint_sha256=checkpoints,
        interpretation='fixed epoch20 primary; other epochs exploratory; report regardless of metric direction'))
    files = [p for p in ROOT.rglob('*') if p.is_file() and 'source' not in p.relative_to(ROOT).parts
        and p.suffix in ['.json','.csv','.png'] and p.name not in ['bundle_info.json','bundle_manifest.json']]
    write(ROOT/'bundle_manifest.json',{str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files})
    with tarfile.open(ROOT/'review_results.tar.gz','w:gz') as archive:
        for path in files+[ROOT/'bundle_manifest.json']:
            archive.add(path,arcname=str(path.relative_to(ROOT)))
    write(ROOT/'bundle_info.json',dict(bytes=(ROOT/'review_results.tar.gz').stat().st_size,
        sha256=sha(ROOT/'review_results.tar.gz')))
    print('COMPLETED', json.dumps(read(ROOT/'complete.json')), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare','smoke','train','evaluate','report'])
    parser.add_argument('--epoch', type=int, choices=range(11,21))
    args = parser.parse_args()
    if args.stage in ['train','evaluate']:
        assert args.epoch is not None
        globals()[args.stage](args.epoch)
    elif args.stage == 'smoke':
        train(11, smoke=True)
    else:
        globals()[args.stage]()
