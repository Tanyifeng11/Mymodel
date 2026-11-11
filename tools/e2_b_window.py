"""同预算验证局部FFT时间窗，复用审计通过的B0与原B2。"""
import argparse
import json
from pathlib import Path
import random
import shutil

import numpy as np
import torch

from models.e2_compat import compute_losses, load_trainable_strict
from models.e2_local_fft import LocalFFTLoss
from tools.e2_a_minimal import mask_snapshot
from tools.e2_b_screen_training import assert_snapshot, audit_training
from tools.e2_fft_experiment import SOURCE, environment, load, read, setup, sha, write
from tools.e2_fft_training import TrainingRun, dataset, frozen_hash, optimizer_hash, tensor_hash
from tools.e2_time_gradient_common import digest_network

ARM = 'B2_t100_249'
ARMS = ('B0', 'B2', ARM)
MIN_T = 100
CALIBRATION_UPDATES = 64
BATCH = 8
PAIRS_SHA = 'c36330795b0702ea2153647e427986f1a32627757198e2168e4eb5593d00d2a0'
CHECKPOINTS = {
    'B0': '0f6d220fb5371f7222caeee94166067977102a505252576cdb910f199ad02a3a',
    'B2': 'e19f7e726359cdc2b9f49483845d1028a143b020969226f08cc95ac79ac7608d',
}


def log_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def verified_copy(baseline, out, manifest, name):
    source = baseline / name
    assert sha(source) == manifest[name]['sha256'], name
    target = out / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def verify_checkpoints(baseline):
    for arm, expected in CHECKPOINTS.items():
        folder = baseline / 'training' / arm
        record = read(folder / 'complete.json')
        assert record['status'] == 'completed' and record['updates'] == 2000
        assert record['checkpoint_sha256'] == expected == sha(folder / 'latest.pt')


def prepare(args):
    core, out = setup(args)
    baseline = Path(args.baseline)
    verify_checkpoints(baseline)
    manifest = read(baseline / 'bundle_manifest.json')
    for arm in CHECKPOINTS:
        for name in ['complete.json', 'training.jsonl']:
            verified_copy(baseline, out, manifest, 'training/%s/%s' % (arm, name))
        verified_copy(baseline, out, manifest, 'comparison/%s.json' % arm)
    configs = [read(out / 'comparison' / (arm + '.json')) for arm in CHECKPOINTS]
    for config in configs:
        assert config['updates'] == 2000 and config['effective_batch'] == BATCH
        assert config['world_size'] == 1 and config['train_seed'] == 42
        assert config['initialization_sha256'] == read(SOURCE / 'configs/e2_fft_regression.json')['checkpoint_sha256']
        assert config['split_sha256'] == sha(baseline / 'split.json')
        assert config['training_input_map_sha256'] == sha(baseline / 'audit/training_inputs.json')
    assert {k: v for k, v in configs[0].items() if k not in {'arm','fft','fft_weight'}} == {
        k: v for k, v in configs[1].items() if k not in {'arm','fft','fft_weight'}}
    assert configs[1]['fft'] and configs[1]['fft_t_max'] == 249 and 'fft_t_min' not in configs[1]
    for name in ['models/bf_original_fusion.py', 'models/bf_sd_skip.py', 'models/e2_compat.py',
                 'models/e2_geometry.py', 'tools/bf_t3_frozen_core.py', 'train_GAM_texture_joint.py']:
        for arm in CHECKPOINTS:
            assert sha(SOURCE / name) == read(out / 'training' / arm / 'complete.json')['environment']['source'][name], name
    assert read(baseline / 'audit/training_inputs_final.json')['passed']
    audit_training(args)
    assert sha(out / 'audit/training_inputs.json') == sha(baseline / 'audit/training_inputs.json')
    mask_snapshot(args, core)
    old_mask = Path(args.mask_parent) / 'audit/mask_inputs.json'
    assert sha(out / 'audit/mask_inputs.json') == sha(old_mask)
    verified_copy(baseline, out, manifest, 'strict_pairs.json')
    assert sha(out / 'strict_pairs.json') == PAIRS_SHA
    pairs = read(out / 'strict_pairs.json')['pairs']
    assert len(pairs) == len({p['identity_group'] for p in pairs}) == 32
    for pair in pairs:
        for key in ['sketch', 'gt', 'original_reference', 'swapped_reference']:
            name = pair[key]
            source = Path(args.data) / name
            assert sha(source) == pair['hashes'][key]
            target = out / 'inputs/frozen' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        for arm in CHECKPOINTS:
            for condition in ['old', 'new']:
                prefix = 'outputs/%s/%s/%s' % (arm, pair['case_id'], condition)
                for suffix in ['.png', '.json']:
                    verified_copy(baseline, out, manifest, prefix + suffix)
                meta = read(out / (prefix + '.json'))
                assert meta['checkpoint_sha256'] == CHECKPOINTS[arm]
                assert meta['manifest_sha256'] == PAIRS_SHA and meta['output_sha256'] == sha(out / (prefix + '.png'))
    logs = [log_rows(out / 'training' / arm / 'training.jsonl') for arm in CHECKPOINTS]
    assert all([row['step'] for row in rows] == list(range(1, 2001)) for rows in logs)
    assert all(a['tokens'] == b['tokens'] and a['chain'] == b['chain'] for a, b in zip(*logs))
    rows = read(core.ROOT / 'training_manifest.json')
    identity = lambda index: Path(rows[index]['cloth']).stem.split('_')[0]
    excluded = {identity(t['index']) for row in logs[1] for t in row['tokens']}
    diagnostic = read(Path(args.diagnostic) / 'inputs/training_probes.json')['records']
    excluded.update(identity(r['index']) for r in diagnostic)
    split = read(out / 'split.json')
    excluded.update(identity(r['index']) for r in split['smoke'])
    excluded.update(identity(i) for i in split['validation_indices'])
    pool = sorted(split['train_indices'])
    random.Random(111102).shuffle(pool)
    indices = []
    for index in pool:
        key = identity(index)
        if key not in excluded:
            indices.append(index)
            excluded.add(key)
        if len(indices) == BATCH * CALIBRATION_UPDATES:
            break
    assert len(indices) == BATCH * CALIBRATION_UPDATES
    groups = [indices[i:i+BATCH] for i in range(0, len(indices), BATCH)]
    write(out / 'calibration/inputs.json', dict(groups=groups, seed=111102, noise_seed=111103,
        patch_seed=110642, excludes_formal_stream_and_prior_diagnostic_identities=True,
        split_sha256=sha(out / 'split.json'), generator_updates=0))
    write(out / 'audit/protocol.json', dict(arm=ARM, t_min=MIN_T, t_max=249,
        initialization_sha256=configs[1]['initialization_sha256'], updates=2000, effective_batch=BATCH,
        reuse_checkpoints=CHECKPOINTS, pairs_sha256=PAIRS_SHA, primary=ARM+'_minus_B2 gray_strict_response_change',
        secondary='other three candidate vs B2 texture metrics, Holm',
        protection='old/new edge_f1, IoU, edge_l1, leak, tcf_lab and old LPIPS candidate vs B2, Holm',
        descriptive='candidate vs B0 gray strict paired difference',
        source_commit=environment()['commit'], window_selected_before_generation=True))
    print('B_WINDOW_PREPARE_PASS', len(indices), flush=True)


def fft_for(config, minimum):
    return LocalFFTLoss(sizes=config['fft_sizes'], count=config['fft_count'], t_max=config['fft_t_max'],
                        std_min=config['fft_std_min'], t_min=minimum)


def same_patch_state(a, b):
    return a['seed'] == b['seed'] and torch.equal(a['rng'], b['rng'])


def auxiliary_gradient(module, batch, captured, token, weight, retain_graph):
    loss, stats = module(captured['decoded'], batch['texture_image'], batch['garment_mask'],
                         captured['timesteps'], batch['dataset_index'],
                         reference_keep=not .2 <= token['drop'] < .3)
    if not stats['valid_n']:
        return stats, np.zeros(5)
    targets = [captured['epsilon'], *captured['features']]
    gradients = torch.autograd.grad(weight * loss / BATCH, targets, retain_graph=retain_graph)
    norms = np.asarray([float(g.detach().float().norm()) for g in gradients])
    assert np.isfinite(norms).all()
    return stats, norms


def summarize_norms(values):
    values = np.asarray(values)
    return dict(mean=values.mean(0).tolist(), p50=np.quantile(values, .5, axis=0).tolist(),
                p90=np.quantile(values, .9, axis=0).tolist(), max=values.max(0).tolist())


def calibrate(args):
    from accelerate import Accelerator
    from train_GAM_texture_joint import VGGGramStyleLoss

    core, out = setup(args)
    config = read(out / 'comparison/B2.json')
    assert assert_snapshot(args) == config['training_input_map_sha256']
    acc = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=BATCH)
    assert acc.num_processes == 1
    args.device = str(acc.device)
    pipe, network, _ = load(args, core)
    data = dataset(pipe, core)
    style = VGGGramStyleLoss().to(args.device).eval().requires_grad_(False)
    network = acc.prepare_model(network)
    network.train()
    before = digest_network(network)
    protocol = read(out / 'calibration/inputs.json')
    rng = torch.Generator(device=acc.device).manual_seed(protocol['noise_seed'])
    old, candidate = fft_for(config, 0), fft_for(config, MIN_T)
    records = []
    for step, indices in enumerate(protocol['groups'], 1):
        squared = np.zeros((2, 5))
        counts = np.zeros((2, 2), dtype=int)
        slots = []
        for index in indices:
            batch = {k: v[None].to(acc.device) for k, v in data[index].items()}
            with acc.autocast():
                total, raw, token, captured, _ = compute_losses(pipe, network, batch, rng, style, core)
                old_stats, old_norm = auxiliary_gradient(old, batch, captured, token, config['fft_weight'], True)
                new_stats, new_norm = auxiliary_gradient(candidate, batch, captured, token, 1., False)
            assert same_patch_state(old.sampling_state(), candidate.sampling_state())
            if new_stats['active_n']:
                assert old_stats == new_stats
            squared += np.square([old_norm, new_norm])
            counts += [[s['active_n'], s['valid_n']] for s in [old_stats, new_stats]]
            slots.append(dict(token=token, old_active=old_stats['active_n'], old_valid=old_stats['valid_n'],
                candidate_active=new_stats['active_n'], candidate_valid=new_stats['valid_n'],
                old_weighted_gradient=old_norm.tolist(), candidate_unit_gradient=new_norm.tolist()))
            acc.unwrap_model(network).adapter.clear()
            del total, raw, captured, batch
        records.append(dict(update_group=step, old_weighted_norm=np.sqrt(squared[0]).tolist(),
            candidate_unit_norm=np.sqrt(squared[1]).tolist(), old_active=int(counts[0, 0]),
            old_valid=int(counts[0, 1]), candidate_active=int(counts[1, 0]), candidate_valid=int(counts[1, 1]),
            slots=slots))
        if step == 1 or step % 8 == 0:
            write(out / 'calibration/progress.json', records)
            print('B_WINDOW_CALIBRATION', step, 'of', len(protocol['groups']), flush=True)
    assert digest_network(network) == before
    assert all(p.grad is None for p in network.parameters())
    old_norms = np.asarray([r['old_weighted_norm'] for r in records])
    unit_norms = np.asarray([r['candidate_unit_norm'] for r in records])
    assert np.count_nonzero(unit_norms[:, 0]) >= 8
    weight = float(old_norms[:, 0].mean() / unit_norms[:, 0].mean())
    assert np.isfinite(weight) and weight > 0
    write(out / 'calibration/result.json', dict(passed=True, fft_weight=weight,
        original_fft_weight=config['fft_weight'], updates=0, groups=len(records), effective_batch=BATCH,
        coordinates_equal_when_both_active=True, patch_rng_equal=True, model_state_unchanged=True,
        norm_names=['epsilon', 'projection_feature0', 'projection_feature1', 'projection_feature2', 'projection_feature3'],
        norm_reduction='sqrt(sum of squared norms of per-slot gradients of loss/8), then mean across all groups',
        original=summarize_norms(old_norms), candidate_scaled=summarize_norms(unit_norms * weight),
        candidate_scaling='one fixed coefficient from unweighted candidate gradients; linear approximation before bf16 rounding',
        records=records, inputs_sha256=sha(out / 'calibration/inputs.json'), environment=environment()))
    config.update(arm=ARM, protocol='B_fft_window_v1', code_commit=environment()['commit'],
        split=str(out / 'split.json'), fft_t_min=MIN_T, fft_weight=weight,
        calibration_sha256=sha(out / 'calibration/result.json'))
    write(out / 'comparison' / (ARM + '.json'), config)
    print('B_WINDOW_CALIBRATION_PASS', weight, flush=True)


def smoke(args):
    from accelerate import Accelerator
    from train_GAM_texture_joint import VGGGramStyleLoss

    core, out = setup(args)
    acc = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=BATCH)
    assert acc.num_processes == 1
    args.device = str(acc.device)
    pipe, network, _ = load(args, core)
    initial = network.trainable_state()
    frozen = frozen_hash(pipe, network)
    style = VGGGramStyleLoss().to(args.device).eval().requires_grad_(False)
    data = dataset(pipe, core)
    indices = read(out / 'split.json')['train_indices']
    replay = {}
    for arm in CHECKPOINTS:
        load_trainable_strict(network, initial)
        config = read(out / 'comparison' / (arm + '.json'))
        run = TrainingRun(args, core, pipe, network, data, style, config, indices, acc)
        expected = log_rows(out / 'training' / arm / 'training.jsonl')[:4]
        observed = [run.update() for _ in expected]
        for a, b in zip(observed, expected):
            for key in ['tokens', 'chain', 'losses', 'grad_norm', 'patch_counts', 'lr']:
                assert a[key] == b[key], (arm, a['step'], key, a[key], b[key])
        replay[arm] = dict(passed=True, updates=4, tokens_equal=True, loss_grad_and_patch_equal=True)
        network = acc.unwrap_model(run.network, keep_fp32_wrapper=False)
        del run
        acc.free_memory()
    write(out / 'audit/baseline_replay.json', replay)
    load_trainable_strict(network, initial)
    config = read(out / 'comparison' / (ARM + '.json'))
    assert config['calibration_sha256'] == sha(out / 'calibration/result.json')
    run = TrainingRun(args, core, pipe, network, data, style, config, indices, acc)
    history = [run.update()]
    folder = out / 'smoke'
    folder.mkdir(exist_ok=True)
    run.save(folder / 'resume.pt')
    expected = run.update()
    expected_weights = tensor_hash(acc.unwrap_model(run.network).trainable_state().items())
    expected_optimizer = optimizer_hash(run.optimizer)
    expected_patch = run.fft.sampling_state()
    run.restore(folder / 'resume.pt')
    actual = run.update()
    assert tensor_hash(acc.unwrap_model(run.network).trainable_state().items()) == expected_weights
    assert optimizer_hash(run.optimizer) == expected_optimizer
    assert same_patch_state(expected_patch, run.fft.sampling_state())
    for key in ['tokens', 'chain', 'losses', 'grad_norm', 'patch_counts', 'lr']:
        assert actual[key] == expected[key], key
    history += [actual, run.update(), run.update()]
    prior = log_rows(out / 'training/B2/training.jsonl')[:4]
    assert all(a['tokens'] == b['tokens'] for a, b in zip(history, prior))
    assert sum(r['patch_counts']['valid_n'] for r in history) > 0
    assert frozen_hash(pipe, acc.unwrap_model(run.network)) == frozen
    write(out / 'smoke/result.json', dict(passed=True, exact_resume=True, updates=4,
        same_legacy_stream=True, records=history, reused_replay=replay,
        config_sha256=sha(out / 'comparison' / (ARM + '.json')), environment=environment()))
    print('B_WINDOW_SMOKE_PASS', flush=True)


def audit_final(args):
    core, out = setup(args)
    verify_checkpoints(Path(args.baseline))
    audit_training(args, final=True)
    mask_snapshot(args, core, final=True)
    assert sha(out / 'audit/mask_inputs.json') == sha(Path(args.mask_parent) / 'audit/mask_inputs.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'calibrate', 'smoke', 'train', 'audit-final', 'generate', 'evaluate', 'bundle'])
    parser.add_argument('--project', default='/share/home/u2515283058/Mymodel')
    parser.add_argument('--data', default='/share/home/u2515283058/datasets/BF')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out', required=True)
    parser.add_argument('--candidates', default=str(SOURCE / 'configs/e2_b_candidates.json'))
    args = parser.parse_args()
    root = Path(args.project) / 'output_eval'
    args.baseline = args.parent = str(root / 'e2_b_screen_20261108')
    args.diagnostic = str(root / 'e2_time_gradient_20261111')
    args.mask_parent = str(root / 'e2_a_minimal_20261109')
    args.arm = ARM
    out = Path(args.out)
    if args.action == 'train':
        from tools.e2_b_screen_training import train
        assert read(out / 'smoke/result.json')['config_sha256'] == sha(out / 'comparison' / (ARM + '.json'))
        assert read(out / 'comparison' / (ARM + '.json'))['calibration_sha256'] == sha(out / 'calibration/result.json')
        train(args)
    elif args.action in ['generate', 'evaluate', 'bundle']:
        from tools import e2_b_window_evaluation
        args.data = str(out / 'inputs/frozen')
        getattr(e2_b_window_evaluation, args.action)(args)
    else:
        globals()[args.action.replace('-', '_')](args)


if __name__ == '__main__':
    main()
