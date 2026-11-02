"""新BF-SD架构的2×2区域监督消融；只新增no-body、no-rest训练。"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
# 原实验打包函数后来修订过；另存训练时a821123版本，保持整个核心文件逐字节相同。
from tools import bf_t3_frozen_core as core

PROJECT, SOURCE = core.PROJECT, core.SOURCE
PARENT = PROJECT / 'output_eval/bf_fusion_t3_three_arm_20261101'
ROOT = PROJECT / 'output_eval/bf_t3_ablation_20261102'
ARMS = ['BF-SD', 'BF-SD-T3-no-body', 'BF-SD-T3-no-rest', 'BF-SD-T3']
NEW_ARMS = ARMS[1:3]
COEFFICIENTS = {
    'BF-SD': (0., 0.), 'BF-SD-T3-no-body': (0., .03816244801608925),
    'BF-SD-T3-no-rest': (.1, 0.), 'BF-SD-T3': (.1, .03816244801608925)}
read, write, sha, digest = core.read, core.write, core.sha, core.digest
core.ROOT = ROOT


def core_source_path(name):
    return SOURCE / ('tools/bf_t3_frozen_core.py' if name == 'tools/bf_t3_three_arm.py' else name)


def prepare():
    assert not (ROOT / 'protocol.json').exists(), '不覆盖已冻结协议'
    previous = read(PARENT / 'protocol.json')
    audit = read(PARENT / 'audit/final_comparability.json')
    assert audit['passed'] and audit['same_initialization'] and audit['same_training_stream']
    parent_training = {arm: read(PARENT / 'training' / arm / 'complete.json') for arm in [ARMS[0], ARMS[-1]]}
    for value in parent_training.values():
        assert value['updates'] == 28365 and value['world'] == 2
        for name, expected in value['source_sha256'].items():
            assert sha(core_source_path(name)) == expected, ('共享核心源码改变', name)
    assert sha(core.E1) == core.E1_SHA == previous['e1_sha256']
    (ROOT / 'initialization').mkdir(parents=True, exist_ok=True)
    source = PARENT / 'initialization/bf_fusion.pt'
    assert sha(source) == previous['fusion_sha256']
    shutil.copy2(source, ROOT / 'initialization/bf_fusion.pt')
    train = read(PARENT / 'training_manifest.json')
    assert len(train) == 45384 and read(PROJECT / 'data/train_bf_texture.json') == train
    assert sha(PROJECT / 'data/train_bf_texture.json') == previous['train_sha256']
    shutil.copy2(PARENT / 'training_manifest.json', ROOT / 'training_manifest.json')
    primary = [dict(r) for r in read(PARENT / 'evaluation.json') if r['split'] == 'heldout128']
    assert len(primary) == 128 and all(r['condition'] == 'original' for r in primary)
    # 仅依据输入哈希选择，每类8件；供体在同一类别循环移位，排除自参考。
    neutral_text = {'top': 'A product photo of a top.', 'outwear': 'A product photo of an outer garment.',
                    'pants': 'A product photo of a pair of trousers.', 'dress': 'A product photo of a dress.'}
    extra, assignments = [], []
    for category, caption in neutral_text.items():
        selected = sorted([r for r in primary if r['category'] == category],
                          key=lambda r: hashlib.sha256(('bf_t3_ablation_20261102:' + r['id']).encode()).hexdigest())[:8]
        assert len(selected) == 8
        for i, row in enumerate(selected):
            donor = selected[(i + 4) % 8]
            assert donor['id'] != row['id']
            old = dict(row, split='neutral32', condition='neutral_original', caption=caption,
                       original_caption=row['caption'])
            new = dict(old, condition='neutral_swapped', reference=donor['reference'],
                       reference_sha256=donor['reference_sha256'], reference_donor_id=donor['id'],
                       reference_source=donor.get('reference_source', donor['reference']))
            extra.extend([old, new])
            assignments.append(dict(id=row['id'], donor_id=donor['id'], category=category, caption=caption))
    rows = primary + extra
    train_ids = {Path(r['cloth']).stem.split('_')[0] for r in train}
    for row in rows:
        assert row['id'].split('_')[0] not in train_ids
        for key in ['sketch', 'reference', 'gt']:
            source = Path(row[key]); expected = row[key + '_sha256']
            assert sha(source) == expected
            target = ROOT / 'inputs' / expected
            target.parent.mkdir(exist_ok=True)
            if not target.exists(): shutil.copy2(source, target)
            row[key] = str(target)
    for arm in [ARMS[0], ARMS[-1]]:
        reused = ROOT / 'reused_training' / arm
        reused.mkdir(parents=True, exist_ok=True)
        for name in ['complete.json', 'training_log.csv']:
            shutil.copy2(PARENT / 'training' / arm / name, reused / name)
        for row in primary:
            source = PARENT / 'outputs' / arm / 'original' / (row['id'] + '.png')
            meta = read(source.with_suffix('.json'))
            assert sha(source) == meta['output_sha256']
            assert meta['checkpoint_sha256'] == parent_training[arm]['checkpoint_sha256']
            for key in ['sketch', 'reference']:
                assert meta['input_' + key + '_sha256'] == row[key + '_sha256']
            target = ROOT / 'outputs' / arm / 'original' / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            write(target.with_suffix('.json'), dict(meta, reused_from=str(source)))
    write(ROOT / 'evaluation.json', rows)
    write(ROOT / 'audit/parent_training.json', parent_training)
    write(ROOT / 'audit/neutral_reference_assignments.json', assignments)
    write(ROOT / 'protocol.json', dict(name=ROOT.name, arms=ARMS, new_training_arms=NEW_ARMS,
        seed=42, updates=28365, samples=len(train), effective_batch=8, world=2,
        learning_rate=5e-5, warmup=500, schedule='cosine', optimizer=previous['optimizer'],
        precision=previous['precision'], initialization=previous['initialization'],
        initial_sha256=parent_training[ARMS[0]]['initial_sha256'],
        parent_protocol_sha256=sha(PARENT / 'protocol.json'),
        training_manifest_sha256=sha(ROOT / 'training_manifest.json'),
        bf_fusion_sha256=previous['fusion_sha256'], core_source_sha256=parent_training[ARMS[0]]['source_sha256'],
        base_objective=previous['objectives']['base'], coefficients={a: dict(body=b, rest=r) for a, (b, r) in COEFFICIENTS.items()},
        inference=previous['inference'], evaluation=dict(primary_original=128, neutral_original=32,
            neutral_swapped=32, rows_per_arm=192, neutral_caption=True, donor_rule='input hash then within-category offset4 of8'),
        checkpoint_selection='fixed final successful update28365; no outcome-based selection',
        statistics=dict(bootstrap=10000, sign_flip=10000, seed=110142,
            primary_family='full minus base/no-body/no-rest: six primary metrics, Holm18',
            response_family='full minus base/no-body/no-rest: RGB margin/gray margin/gray DISTS, Holm9'),
        limitations=['one training seed; previously inspected128 identities; exploratory',
            'RGB moments measure appearance/color, not complete motif identity',
            '64x64 patches do not establish whole-repeat fidelity',
            'neutral caption response is separate from original-caption primary test'],
        source_commit=subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()))
    print('PREPARED', len(train), len(rows), flush=True)


def train(arm, preflight=False):
    import torch
    from torch.utils.data import DataLoader, DistributedSampler
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs
    from diffusers.optimization import get_scheduler
    from train_GAM_texture_joint import JointTextureDataset, VGGGramStyleLoss
    assert arm in NEW_ARMS
    config = read(ROOT / 'protocol.json')
    if not preflight: assert read(ROOT / 'audit/preflight.json')['passed']
    world = int(os.environ.get('WORLD_SIZE', '1'))
    assert world == (1 if preflight else 2), '正式训练保持与历史对照相同的双卡随机流'
    accumulation = 8 // world
    acc = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=accumulation,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    torch.cuda.set_device(acc.local_process_index)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print('LAUNCH_ALLOCATION', os.environ.get('SLURM_JOB_NODELIST'), world,
          acc.local_process_index, os.environ.get('CUDA_VISIBLE_DEVICES'), flush=True)
    pipe, network = core.load_network(acc.device)
    params = [p for p in network.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=5e-5, betas=(.9, .999), eps=1e-8, weight_decay=.01)
    lr = get_scheduler('cosine', optimizer=optimizer, num_warmup_steps=500, num_training_steps=config['updates'])
    dataset = JointTextureDataset(str(ROOT / 'training_manifest.json'), pipe.tokenizer,
        str(PROJECT.parent / 'datasets/BF/training'), width=384, height=512,
        texture_preprocess_mode='plain_resize', mask_cache=str(core.MASKS))
    sampler = DistributedSampler(dataset, num_replicas=world, rank=acc.process_index, seed=42, drop_last=False)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=2, pin_memory=True)
    assert len(loader) % accumulation == 0
    style = VGGGramStyleLoss().to(acc.device).eval().requires_grad_(False)
    folder = ROOT / ('audit/preflight/' + arm if preflight else 'training/' + arm)
    folder.mkdir(parents=True, exist_ok=True)
    source_hashes = {p: sha(core_source_path(p)) for p in config['core_source_sha256']}
    assert source_hashes == config['core_source_sha256']
    initial = hashlib.sha256()
    for name, value in network.trainable_state().items(): initial.update(name.encode() + value.numpy().tobytes())
    initial_sha = initial.hexdigest()
    assert initial_sha == config['initial_sha256']
    rng = torch.Generator(device=acc.device).manual_seed(42000 + acc.process_index)
    stream = '0' * 64
    step, start = 0, time.monotonic()
    resume = folder / 'latest.pt'
    if resume.exists() and not preflight:
        state = torch.load(resume, map_location='cpu')
        assert state['world'] == world and state['arm'] == arm and state['initial_sha256'] == initial_sha
        network.load_trainable(state['network']); optimizer.load_state_dict(state['optimizer'])
        lr.load_state_dict(state['scheduler']); rng.set_state(state['rngs'][acc.process_index])
        stream, step = state['stream_hashes'][acc.process_index], state['step']
    network, optimizer = acc.prepare(network, optimizer)
    network.train()
    total = 3 if preflight else config['updates']
    updates_per_epoch = len(loader) // accumulation
    epoch, skip_batches = step // updates_per_epoch, (step % updates_per_epoch) * accumulation
    body_weight, rest_weight = COEFFICIENTS[arm]

    def save(final=False):
        states, chains = [None] * world, [None] * world
        if world > 1:
            torch.distributed.all_gather_object(states, rng.get_state().cpu())
            torch.distributed.all_gather_object(chains, stream)
        else: states[0], chains[0] = rng.get_state().cpu(), stream
        if acc.is_main_process:
            value = dict(network=acc.unwrap_model(network).trainable_state(), optimizer=optimizer.state_dict(),
                scheduler=lr.state_dict(), rngs=states, stream_hashes=chains, step=step,
                world=world, arm=arm, initial_sha256=initial_sha)
            temporary = folder / 'latest.tmp'; torch.save(value, temporary); temporary.replace(resume)
            if final or step % updates_per_epoch == 0:
                torch.save(dict(network=value['network'], step=step, arm=arm, initial_sha256=initial_sha),
                           folder / ('step%05d.pt' % step))

    logs = []
    while step < total:
        sampler.set_epoch(epoch)
        for micro, batch in enumerate(loader):
            if micro < skip_batches: continue
            batch = {k: v.to(acc.device) for k, v in batch.items()}
            with acc.accumulate(network):
                with acc.autocast(): raw, token = core.losses(pipe, network, batch, rng, style, preflight)
                stream = hashlib.sha256(stream.encode() + json.dumps(token, sort_keys=True).encode()).hexdigest()
                loss = raw['base'] + body_weight * raw['body'] + rest_weight * raw['rest']
                assert torch.isfinite(loss).item(), token
                if preflight and step == 2 and micro % accumulation == 0:
                    bare = acc.unwrap_model(network)
                    assert bare.adapter.injection_counts == [3, 2, 2, 3]
                    probe = bare.adapter.projections[0].weight
                    gradients = {k: float(torch.autograd.grad(raw[k], probe, retain_graph=True)[0].float().norm())
                                 for k in ['body', 'rest']}
                    gradients['original_fusion'] = float(torch.autograd.grad(loss, bare.adapter.fusion.downconv1[0].weight,
                                                                           retain_graph=True)[0].float().norm())
                    assert all(np.isfinite(v) and v > 0 for v in gradients.values())
                    if acc.is_main_process: write(folder / 'gradients.json', gradients)
                acc.backward(loss)
                if acc.sync_gradients:
                    grad = acc.clip_grad_norm_(params, 1.)
                    assert torch.isfinite(grad).item()
                optimizer.step()
                if acc.sync_gradients: lr.step()
                optimizer.zero_grad(set_to_none=True)
            acc.unwrap_model(network).adapter.clear()
            if acc.sync_gradients:
                assert not acc.optimizer_step_was_skipped
                step += 1
                record = dict(step=step, seconds=time.monotonic() - start, lr=lr.get_last_lr()[0], grad=float(grad),
                    peak_gib=torch.cuda.max_memory_allocated() / 2**30, total=float(loss.detach()),
                    weighted_body=float(raw['body'].detach()) * body_weight,
                    weighted_rest=float(raw['rest'].detach()) * rest_weight,
                    **{k: float(v.detach()) for k, v in raw.items()})
                logs.append(record)
                if acc.is_main_process:
                    path = folder / 'training_log.csv'; exists = path.exists()
                    with path.open('a', encoding='utf-8', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=list(record))
                        if not exists: writer.writeheader()
                        writer.writerow(record)
                    if step <= 3 or step % 25 == 0:
                        write(folder / 'progress.json', record); print('TRAIN', arm, json.dumps(record), flush=True)
                if step % 1000 == 0 or step == total: save(step == total)
                if step >= total: break
        skip_batches = 0; epoch += 1
    acc.wait_for_everyone()
    streams = [None] * world
    if world > 1: torch.distributed.all_gather_object(streams, stream)
    else: streams[0] = stream
    if not preflight:
        assert streams == read(ROOT / 'audit/parent_training.json')['BF-SD']['stream_hashes']
    if acc.is_main_process:
        write(folder / 'complete.json', dict(arm=arm, updates=step, initial_sha256=initial_sha,
            trainable_parameters=sum(p.numel() for p in params), stream_hashes=streams,
            source_sha256=source_hashes, ablation_source_sha256=sha(Path(__file__)),
            seconds=time.monotonic() - start, world=world, effective_batch=8, logs=logs[-10:],
            coefficients=dict(body=body_weight, rest=rest_weight),
            log_scope='rank0 last microbatch per successful update; same as parent log',
            checkpoint_sha256=sha(folder / ('step%05d.pt' % step))))
    print('COMPLETE', arm, step, flush=True)


def preflight():
    for arm in NEW_ARMS:
        subprocess.run([sys.executable, '-m', 'tools.bf_t3_ablation', 'check', '--arm', arm], check=True)
    write(ROOT / 'audit/preflight.json', dict(passed=True, updates_per_arm=3, arms=NEW_ARMS))


def generate(arm):
    import torch
    checkpoint = (ROOT if arm in NEW_ARMS else PARENT) / 'training' / arm / 'step28365.pt'
    checkpoint_sha = sha(checkpoint)
    complete = read(checkpoint.parent / 'complete.json')
    assert complete['updates'] == 28365 and checkpoint_sha == complete['checkpoint_sha256']
    pipe, network = core.load_network('cuda', checkpoint)
    network.eval().requires_grad_(False)
    pipe.unet.half(); pipe.vae.half(); network.adapter.half()
    rows = read(ROOT / 'evaluation.json')
    with torch.inference_mode():
        for row in rows:
            dest = ROOT / 'outputs' / arm / row['condition'] / (row['id'] + '.png')
            if dest.exists():
                meta = read(dest.with_suffix('.json'))
                assert meta['checkpoint_sha256'] == checkpoint_sha and sha(dest) == meta['output_sha256']
                assert meta['seed'] == row['seed']
                for key in ['sketch', 'reference']: assert meta['input_' + key + '_sha256'] == row[key + '_sha256']
                continue
            sketch = core.tensor(row['sketch']).cuda().half(); reference = core.tensor(row['reference']).cuda().half()
            network.adapter.condition(sketch, reference)
            latent = torch.randn((1, 4, 64, 48), device='cuda', dtype=torch.float16,
                generator=torch.Generator(device='cuda').manual_seed(row['seed']))
            before = digest(latent)
            started = time.monotonic()
            image = pipe(ref_image=sketch, prompt=row['caption'], texture_clip_image=None, texture_embeds=None,
                null_prompt='', negative_prompt=' worst quality, low quality', width=384, height=512,
                num_images_per_prompt=1, guidance_scale=7., sketch_scale=.6, ipa_scale=0.,
                generator=torch.Generator(device='cuda').manual_seed(row['seed']), latents=latent,
                num_inference_steps=50)[0]
            dest.parent.mkdir(parents=True, exist_ok=True); image.save(dest)
            write(dest.with_suffix('.json'), dict(id=row['id'], arm=arm, condition=row['condition'], split=row['split'],
                seed=row['seed'], initial_latent_sha256=before, output_sha256=sha(dest),
                checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha, text_used=True, caption=row['caption'],
                GT_used_in_inference=False, input_sketch_sha256=row['sketch_sha256'],
                input_reference_sha256=row['reference_sha256'], steps=50, cfg=7., native_size=[384, 512],
                seconds=time.monotonic() - started))
            network.adapter.clear(); print('GENERATED', arm, row['condition'], row['id'], flush=True)
    write(ROOT / 'generation' / arm / 'complete.json', dict(n=len(rows), checkpoint_sha256=checkpoint_sha))


def package():
    import tarfile
    job_ids = ','.join(map(str, read(ROOT / 'submitted_jobs.json').values()))
    accounting = subprocess.check_output(['sacct', '-X', '-j', job_ids, '-P',
        '--format=JobID,JobName,State,ExitCode,ElapsedRaw,AllocTRES,AllocCPUS,NodeList'], text=True)
    (ROOT / 'audit/sacct_at_packaging.psv').write_text(accounting, encoding='utf-8')
    files = [p for folder in ['audit', 'training', 'reused_training', 'generation', 'metrics', 'figures', 'outputs', 'inputs']
             for p in (ROOT / folder).rglob('*') if p.is_file() and p.suffix not in {'.pt', '.tmp'}]
    files += [SOURCE / p for p in ['tools/bf_t3_ablation.py', 'tools/bf_t3_ablation_report.py',
                                  'tools/bf_t3_frozen_core.py', 'scripts/bf_t3_ablation.sbatch',
                                  'scripts/bf_t3_ablation_submit.py']]
    files += [ROOT / name for name in ['protocol.json', 'evaluation.json', 'submitted_jobs.json', 'evaluation_complete.json']]
    write(ROOT / 'bundle_manifest.json', {str(p.relative_to(ROOT)): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in files})
    with tarfile.open(ROOT / 'review_results.tar.gz', 'w:gz') as tar:
        for p in files + [ROOT / 'bundle_manifest.json']: tar.add(p, arcname=str(p.relative_to(ROOT)))
    write(ROOT / 'bundle_info.json', dict(bytes=(ROOT / 'review_results.tar.gz').stat().st_size,
        sha256=sha(ROOT / 'review_results.tar.gz'), members=len(files) + 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'preflight', 'check', 'train', 'generate', 'evaluate', 'package'])
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args()
    if args.action == 'train': train(args.arm)
    elif args.action == 'check': train(args.arm, True)
    elif args.action == 'generate': generate(args.arm)
    elif args.action == 'evaluate':
        from tools.bf_t3_ablation_report import evaluate
        evaluate()
    else: globals()[args.action]()
