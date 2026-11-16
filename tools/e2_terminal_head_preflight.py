"""末步2308参数头的零更新资格：48缓存一致性、2固定扰动、8×7×2局部反传。"""
import argparse
import copy
import inspect
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
import zipfile

import cv2
import numpy as np
from PIL import Image
import torch

from models.e2_terminal_output_head import TerminalOutputHead
from models.e2_state_texture_b import StateTextureB
from tools.e2_b_endpoint_losses import TASKS, flatten, set_parameters
from tools.e2_b_pair_finetune_losses import projection, pair_coefficients
from tools.e2_state_b_amplitude_response import CHECKPOINT_SHA, read, write, sha, pixels
from tools.e2_state_b_backward_repro import tensor_sha, vector_summary
from tools.e2_state_b_scaled_retry import SENTINELS, scaled_tasks, replica_report, NumericalFailure
from tools.e2_state_b_pair_retry import PROTOCOL_SHA


SOURCE = Path(__file__).resolve().parents[1]
PARAMETERS = 2308
PERTURBATION_NORM = .007854645
NONZERO_PROBES = (('validation_870', 'r1033_native', 'a075'),
                  ('validation_1641', 'r5121_native', 'a100'))
EXTRA = ('models/e2_terminal_output_head.py', 'tools/e2_terminal_head_preflight.py',
         'scripts/e2_terminal_head_preflight.sbatch')


def key(item):
    return item['case'], item['condition'], item['arm']


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    return value


def fixed_perturbation(count=PARAMETERS):
    value = torch.sin(torch.arange(1, count+1, dtype=torch.float64))
    return (value*(PERTURBATION_NORM/value.norm())).float()


def terminal_replay(head, record, scheduler, device):
    """epsilon是未修正CFG原值；head已包含base权重，不能再叠加base_delta。"""
    assert record['epsilon_kind'] == 'raw_unmodified_CFG' and record['index'] == 49 and record['t'] == 1
    zt, raw = [record[name].to(device) for name in ('zt', 'raw_epsilon')]
    assert zt.dtype == raw.dtype == torch.float16
    t = scheduler.timesteps[-1]
    assert int(t) == record['t']
    alpha = scheduler.alphas_cumprod[int(t)].to(device).float()
    assert float(alpha) == record['alpha']
    delta = head.from_features(record['feature'].to(device), record['mask'].to(device))
    corrected = (raw.float()-(alpha/(1-alpha)).sqrt()*delta).to(raw.dtype)
    previous = scheduler.step(corrected, t, zt, return_dict=False)[0]
    with torch.no_grad():
        base = scheduler.step(raw, t, zt, return_dict=False)[0].float()
    return previous, base, delta, corrected


def fixed_args(args):
    result = copy.copy(args)
    result.out = args.fixed
    return result


def prepare(args):
    from tools import e2_b_scaled_projection_retry as retry
    retry.verify(fixed_args(args))
    assert not (args.out/'preflight_protocol.json').exists(), '不覆盖已有资格实验'
    spec = read(args.fixed/'protocol.json')
    assert sha(args.fixed/'protocol.json') == PROTOCOL_SHA
    args.out.mkdir(parents=True, exist_ok=True)
    inputs, copied = {}, {}
    def duplicate(source, relative, expected=None):
        digest = sha(source)
        if expected is not None:
            assert digest == expected
        dest = args.out/relative
        assert not dest.exists(), str(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        assert sha(dest) == digest
        inputs[source.resolve().as_posix()] = copied[relative] = digest
    duplicate(args.fixed/'protocol.json', 'protocol.json', PROTOCOL_SHA)
    for name, digest in spec['assets'].items():
        duplicate(args.fixed/name, name, digest)
    duplicate(args.fixed/'evaluation/f68/replay/result.json', 'historical_f68.json')
    historical = read(args.out/'historical_f68.json')
    assert len(historical['rows']) == 48 and historical['checkpoint_sha256'] == CHECKPOINT_SHA
    for row in historical['rows']:
        duplicate(args.fixed/row['image'], row['image'], row['png_sha256'])
    for index in range(4):
        duplicate(args.fixed/'gradients'/('shard%d.json' % index), 'baseline_tasks/shard%d.json' % index)
    assert sha(args.checkpoint) == CHECKPOINT_SHA
    inputs[args.checkpoint.resolve().as_posix()] = CHECKPOINT_SHA
    for pair in spec['train']:
        path = args.data/pair['sketch']
        assert sha(path) == pair['hashes']['sketch']
        inputs[path.resolve().as_posix()] = sha(path)
    vector = fixed_perturbation()
    torch.save(dict(vector=vector, purpose='fixed diagnostic perturbation, not trained weights'), args.out/'probe_delta.pt')
    copied['probe_delta.pt'] = sha(args.out/'probe_delta.pt')
    sources = set(read(args.fixed/'solver_overlay.json')['source_sha256']) | set(EXTRA)
    write(args.out/'preflight_protocol.json', dict(original_protocol_sha256=PROTOCOL_SHA,
        checkpoint_sha256=CHECKPOINT_SHA, architecture='frozen f68 + terminal output convolution weight/bias increments',
        trainable_parameters=PARAMETERS, frozen_base_parameters=274372, active_gate=dict(global_index=49, timestep=1, local_B_call=24),
        expression='conv2d(feature, base.output.weight+delta_weight, base.output.bias+delta_bias)*mask; same function for full/cache',
        sampler=spec['sampler'], head_initialization='all zero FP32 increments; base unchanged',
        source_gradient_policy='only old JSON scalar baselines and features; old large gradient PT never loaded as directions',
        sentinels=[list(value) for value in SENTINELS], nonzero_probes=[list(value) for value in NONZERO_PROBES],
        perturbation=dict(formula='float32(sin(float64(arange(1,2309))) normalized in FP64)',
            requested_norm=PERTURBATION_NORM, actual_norm_fp64=float(vector.double().norm()), sha256=tensor_sha(vector),
            purpose='nonzero parity control only; not a new-head trust radius or qualification threshold'),
        loss_scale=1024., replica_gate=dict(max_relative_l2=.005, minimum_cosine=.99999, require_all_finite=True),
        planned_zero_s0=48, planned_nonzero_s0=2, planned_head_reverse_passes=112,
        optimizer_updates=0, projection_calls=0, adopted_checkpoints=[], holdout_results_read=0,
        full_B_or_UNet_reverse_passes=0, training_loop_implemented=False,
        input_sha256=inputs, copied_sha256=copied, source_sha256={name: sha(SOURCE/name) for name in sorted(sources)},
        execution_commit=subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip(),
        interpretation='new time-local parameterization; its local gradient is not a truncated gradient of old full B; parity/repeatability alone do not prove improvement'))
    verify(args)
    print('TERMINAL_PREFLIGHT_PREPARED', flush=True)


def verify(args):
    from tools import e2_b_scaled_projection_retry as retry
    retry.verify(fixed_args(args))
    protocol = read(args.out/'preflight_protocol.json')
    assert protocol['trainable_parameters'] == PARAMETERS and protocol['loss_scale'] == 1024.
    assert protocol['sentinels'] == [list(value) for value in SENTINELS]
    assert protocol['nonzero_probes'] == [list(value) for value in NONZERO_PROBES]
    assert all(sha(Path(path)) == digest for path, digest in protocol['input_sha256'].items())
    assert all(sha(args.out/path) == digest for path, digest in protocol['copied_sha256'].items())
    assert all(sha(SOURCE/path) == digest for path, digest in protocol['source_sha256'].items())
    return protocol


def run_cache(args):
    from tools import e2_state_b_pilot as pilot, e2_state_b_endpoint as endpoint
    from tools.e2_state_b_pair_finetune import Objective
    from tools.e2_oracle_texture import weights_digest
    from tools.e2_fft_experiment import environment
    from tools.e2_b_chain_repro import backend_flags
    started = time.monotonic()
    protocol = verify(args)
    destination = args.out/'cache'/('shard%d.json' % args.shard)
    assert not destination.exists(), '不覆盖已完成shard'
    destination.parent.mkdir(exist_ok=True)
    spec = read(args.out/'protocol.json')
    pair = spec['train'][args.shard]
    items = [r for r in spec['inputs'] if r['case'] == pair['case_id']]
    assert len(items) == 12
    old = read(args.out/'baseline_tasks'/('shard%d.json' % args.shard))
    old_values = {r['sample_condition']: r for r in old['values']}
    history = {key(row): row for row in read(args.out/'historical_f68.json')['rows']}
    runtime = copy.copy(args)
    runtime.action = 'train'
    core, pipe, net = pilot.load_core(runtime)
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    base = StateTextureB(color_channels=2, reference_channels=192).to(args.device)
    base.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=False)['b'])
    head = TerminalOutputHead(base).to(args.device).eval()
    parameters = head.increment_parameters()
    assert sum(p.numel() for p in parameters) == PARAMETERS
    assert sum(p.numel() for p in base.parameters()) == 274372
    assert [id(p) for p in head.parameters() if p.requires_grad] == [id(p) for p in parameters]
    net.eval().requires_grad_(False)
    pipe.vae.eval().requires_grad_(False)
    initial, frozen = weights_digest(base), weights_digest(net)
    assert initial == old['b_hash'] and frozen == old['net_hash'] and torch.count_nonzero(flatten(parameters)) == 0
    assert type(pipe.scheduler).__name__ == 'DDIMScheduler'
    assert pipe.scheduler.config.prediction_type == 'epsilon' and not pipe.scheduler.config.clip_sample
    assert not pipe.scheduler.config.thresholding and not pipe.scheduler.config.set_alpha_to_one
    scheduler_path = Path(inspect.getsourcefile(type(pipe.scheduler)))
    scheduler_sha = sha(scheduler_path)
    cache, captures, checks = {}, {}, []

    def capture(item):
        head.reset_capture()
        trace = []
        def hook(module, positional, keywords, output):
            trace.append(dict(timestep=int(keywords['timestep']), sha256=tensor_sha(output)))
        with torch.no_grad(), head.register_forward_hook(hook, with_kwargs=True):
            image, digest, tap, latent = endpoint.final_sample(runtime, core, pipe, net, pair, item['sample_condition'], head)
        assert head.active_calls == len(trace) == len(tap.states) == 25 and tap.index == 50
        assert [r['timestep'] for r in trace] == pipe.scheduler.timesteps[25:].tolist()
        assert digest == old_values[item['sample_condition']]['noise_digest']
        saved, last = cpu_tree(head.last_record), tap.states[-1]
        assert saved['index'] == last['index'] == 49 and saved['timestep'] == last['t'] == 1
        assert saved['feature'].dtype == saved['mask'].dtype == saved['base_delta'].dtype == torch.float32
        record = dict(item, pair=pair, seed=pair['seed'], zt=last['zt'].clone(), raw_epsilon=last['epsilon'].clone(),
            epsilon_kind='raw_unmodified_CFG', index=49, t=1,
            alpha=float(pipe.scheduler.alphas_cumprod[1].float()), feature=saved['feature'], mask=saved['mask'],
            base_delta=saved['base_delta'], full_delta=saved['full_delta'], final=latent.detach().cpu(), noise_digest=digest,
            prefix_states=[dict(index=s['index'], timestep=s['t'], zt=tensor_sha(s['zt']), raw_epsilon=tensor_sha(s['epsilon'])) for s in tap.states],
            B_trace=trace)
        return record, np.asarray(image).copy()

    for item in items:
        record, picture = capture(item)
        assert tensor_sha(record['full_delta']) == tensor_sha(record['base_delta']), '零增量conv与原同调用conv不等'
        assert np.array_equal(picture, pixels(args.out/history[key(item)]['image'])), '零增量历史PNG改变'
        cache[item['sample_condition']], captures[item['sample_condition']] = record, picture
        print('TERMINAL_ZERO_CAPTURE', args.shard, item['sample_condition'], flush=True)
    objective = Objective(runtime, pipe, None, head, pair, items)
    vae_hash = weights_digest(pipe.vae)
    pipe.scheduler.set_timesteps(50, device=args.device)
    late = {}
    for item in items:
        condition = item['sample_condition']
        record = cache[condition]
        entry = dict(pair=pair, condition=condition)
        with torch.no_grad():
            latent, no_b, delta, corrected = terminal_replay(head, record, pipe.scheduler, args.device)
            assert tensor_sha(latent) == tensor_sha(record['final']) and tensor_sha(delta) == tensor_sha(record['full_delta'])
            tasks, _ = objective.score_latents(entry, latent.float(), no_b, residue=objective.metric(entry).surrogate)
            full_tasks, _ = objective.score_latents(entry, record['final'].to(args.device).float(), no_b,
                residue=objective.metric(entry).surrogate)
            values, full_values = endpoint.numbers(tasks), endpoint.numbers(full_tasks)
            assert values == full_values == old_values[condition]['tasks'], '零增量原七项任务不逐位相同'
            record['target_record'] = cpu_tree(objective.records[pair['case_id'], condition])
            record['tasks'] = values
            record['corrected_epsilon_sha256'] = tensor_sha(corrected)
            record['no_B_base'] = no_b.cpu()
            if item['group'] == 'pastel':
                _, positive, _, _, weight, _ = objective.records[pair['case_id'], condition]
                positive_base = objective.records[pair['case_id'], item['condition']+'_a100'][1]
                features = projection(objective.rgb(latent.float()), positive, positive_base, weight)
                features = {name: float(value) for name, value in features.items()}
                assert features == old['late_features'][item['arm']], '原late feature改变'
                late[item['arm']] = features
        checks.append(dict(item, same_call_base_delta_equal=True, cached_full_latent_equal=True,
            cached_full_tasks_equal=True, tasks_equal_historical=True, historical_png_equal=True,
            historical_full_latent_sha_available=False, cached_png_equal=False,
            feature_sha256=tensor_sha(record['feature']), feature_dtype=str(record['feature'].dtype),
            mask_sha256=tensor_sha(record['mask']), latent_sha256=tensor_sha(record['final']), tasks=values))
    coefficients = pair_coefficients(*(torch.tensor(late[arm][feature]) for arm, feature in
        (('a075', 'y'), ('a125', 'y'), ('a075', 'x'), ('a125', 'x'))))
    assert float(coefficients['pair']) == old['pair_loss']
    absolute = float((torch.tensor(late['a100']['g'])-1).square())
    replicas, gradients = [], {}
    for item in items:
        if key(item) not in SENTINELS:
            continue
        condition = item['sample_condition']
        entry = dict(pair=pair, condition=condition)
        samples = []
        for repeat in range(2):
            # 两次从缓存独立构图；仅2308增量参数进入autograd，绝无UNet展开。
            latent, no_b, delta, corrected = terminal_replay(head, cache[condition], pipe.scheduler, args.device)
            tasks, _ = objective.score_latents(entry, latent.float(), no_b, residue=objective.metric(entry).surrogate)
            assert endpoint.numbers(tasks) == old_values[condition]['tasks']
            current = scaled_tasks(tasks, parameters).detach().cpu()
            assert current.shape == (7, PARAMETERS)
            if not torch.isfinite(current).all():
                raise NumericalFailure(condition+'/repeat%d' % repeat, current)
            samples.append(current)
            del latent, no_b, delta, corrected, tasks
        repeated = replica_report(samples[1], samples[0])
        gradients[condition] = torch.stack(samples)
        replicas.append(dict(item, **repeated,
            gradient_summaries=[{task: vector_summary(sample[index]) for index, task in enumerate(TASKS)} for sample in samples]))
        print('TERMINAL_HEAD_REPLICA', args.shard, condition, repeated['passed'], flush=True)
    assert len(replicas) == 2
    # 所有局部反传已完成，之后才允许pixels的VAE dtype切换。
    for check in checks:
        condition = check['sample_condition']
        cached_png = objective.pixels(cache[condition]['final'].to(args.device))
        assert np.array_equal(cached_png, captures[condition])
        check['cached_png_equal'] = True
    perturbation = torch.load(args.out/'probe_delta.pt', map_location='cpu', weights_only=False)['vector']
    assert tensor_sha(perturbation) == protocol['perturbation']['sha256']
    nonzero = []
    for item in items:
        if key(item) not in NONZERO_PROBES:
            continue
        condition = item['sample_condition']
        record = cache[condition]
        set_parameters(parameters, perturbation)
        try:
            full, full_png = capture(item)
            assert full['prefix_states'] == record['prefix_states'] and full['B_trace'][:24] == record['B_trace'][:24]
            assert tensor_sha(full['feature']) == tensor_sha(record['feature']) and tensor_sha(full['base_delta']) == tensor_sha(record['base_delta'])
            with torch.no_grad():
                cached, no_b, delta, corrected = terminal_replay(head, record, pipe.scheduler, args.device)
                assert tensor_sha(cached) == tensor_sha(full['final']) and tensor_sha(delta) == tensor_sha(full['full_delta'])
                correction = delta.detach().cpu()-record['base_delta']
                assert torch.count_nonzero(correction) > 0 and tensor_sha(cached) != tensor_sha(record['final']), '固定扰动被完全吞掉'
                entry = dict(pair=pair, condition=condition)
                cache_tasks, _ = objective.score_latents(entry, cached.float(), no_b, residue=objective.metric(entry).surrogate)
                full_tasks, _ = objective.score_latents(entry, full['final'].to(args.device).float(), no_b,
                    residue=objective.metric(entry).surrogate)
                values = endpoint.numbers(cache_tasks)
                assert values == endpoint.numbers(full_tasks)
                picture = objective.pixels(cached)
                assert np.array_equal(picture, full_png)
            relative = 'nonzero_probe/%s_%s.png' % (pair['case_id'], condition)
            path = args.out/relative
            path.parent.mkdir(exist_ok=True)
            Image.fromarray(full_png).save(path)
            nonzero.append(dict(item, full_cache_latent_equal=True, full_cache_png_equal=True,
                full_cache_tasks_equal=True, all_preterminal_states_equal=True, first24_B_outputs_equal=True,
                feature_equal=True, perturbation_sha256=tensor_sha(flatten(parameters)),
                correction=vector_summary(correction), latent_change=vector_summary(cached.cpu().float()-record['final'].float()),
                corrected_epsilon_changed=tensor_sha(corrected) != record['corrected_epsilon_sha256'],
                png_changed_channels=int(np.count_nonzero(full_png != captures[condition])), tasks=values,
                image=relative, png_sha256=sha(path), adopted=False))
        finally:
            set_parameters(parameters, torch.zeros_like(perturbation))
    assert weights_digest(base) == initial and weights_digest(net) == frozen and weights_digest(pipe.vae) == vae_hash
    assert not torch.count_nonzero(flatten(parameters))
    assert not any(p.grad is not None for module in (head, net, pipe.vae) for p in module.parameters())
    assert all(not p.requires_grad for module in (base, net, pipe.vae) for p in module.parameters())
    assert sha(scheduler_path) == scheduler_sha
    verify(args)
    tensor_file = args.out/'cache'/('shard%d.pt' % args.shard)
    assert not tensor_file.exists()
    torch.save(dict(records=cache, local_head_gradients=gradients, task_order=TASKS,
        parameter_count=PARAMETERS, zero_head=True, b_hash=initial, net_hash=frozen,
        preflight_protocol_sha256=sha(args.out/'preflight_protocol.json')), tensor_file)
    result = dict(shard=args.shard, case=pair['case_id'], zero_controls=checks, replicas=replicas, nonzero_controls=nonzero,
        late_features=late, pair_loss=float(coefficients['pair']), absolute_loss=absolute,
        all_finite=True, numerical_qualification=all(row['passed'] for row in replicas),
        tensor_file=tensor_file.relative_to(args.out).as_posix(), tensor_sha256=sha(tensor_file), tensor_bytes=tensor_file.stat().st_size,
        preflight_protocol_sha256=sha(args.out/'preflight_protocol.json'), source_sha256=protocol['source_sha256'],
        zero_s0_images=12, nonzero_s0_images=len(nonzero), new_head_reverse_passes=28,
        new_full_B_or_UNet_reverse_passes=0, optimizer_updates=0, projection_calls=0,
        diagnostic_parameter_assignments=2*len(nonzero), head_restored_zero=True, frozen_unchanged=True,
        b_hash=initial, net_hash=frozen, vae_hash_fp32=vae_hash, trainable_parameters=PARAMETERS,
        holdout_results_read=0, adopted_checkpoints=[], training_loop_implemented=False,
        scheduler=dict(class_name=type(pipe.scheduler).__name__, config=dict(pipe.scheduler.config), source_sha256=scheduler_sha),
        environment=environment(), backend_flags=backend_flags(),
        hardware=dict(hostname=socket.gethostname(), gpu=torch.cuda.get_device_name(args.device), job=os.environ.get('SLURM_JOB_ID')),
        seconds=time.monotonic()-started)
    write(destination, result)
    print('TERMINAL_CACHE_COMPLETE', args.shard, result['numerical_qualification'], flush=True)
    if not result['numerical_qualification']:
        raise SystemExit(2)


def review(args):
    protocol = verify(args)
    assert not (args.out/'qualification.json').exists(), '不覆盖已有资格结果'
    reports = [read(args.out/'cache'/('shard%d.json' % index)) for index in range(4)]
    for index, report in enumerate(reports):
        assert report['shard'] == index and report['preflight_protocol_sha256'] == sha(args.out/'preflight_protocol.json')
        assert report['source_sha256'] == protocol['source_sha256'] and report['trainable_parameters'] == PARAMETERS
        assert sha(args.out/report['tensor_file']) == report['tensor_sha256']
        assert report['optimizer_updates'] == report['projection_calls'] == report['new_full_B_or_UNet_reverse_passes'] == report['holdout_results_read'] == 0
        assert report['head_restored_zero'] and report['frozen_unchanged'] and not report['adopted_checkpoints']
    controls = [row for report in reports for row in report['zero_controls']]
    replicas = [row for report in reports for row in report['replicas']]
    nonzero = [row for report in reports for row in report['nonzero_controls']]
    assert len(controls) == 48 and len({key(r) for r in controls}) == 48
    assert len(replicas) == 8 and {key(r) for r in replicas} == set(SENTINELS)
    assert len(nonzero) == 2 and {key(r) for r in nonzero} == set(NONZERO_PROBES)
    assert sum(r['new_head_reverse_passes'] for r in reports) == 112
    for row in controls:
        assert all(row[name] for name in ('same_call_base_delta_equal', 'cached_full_latent_equal',
            'cached_full_tasks_equal', 'tasks_equal_historical', 'historical_png_equal', 'cached_png_equal'))
    for row in nonzero:
        assert row['full_cache_latent_equal'] and row['full_cache_png_equal'] and row['full_cache_tasks_equal']
        assert sha(args.out/row['image']) == row['png_sha256'] and row['latent_change']['norm'] > 0
    passed = all(r['all_finite'] and r['numerical_qualification'] for r in reports)
    result = dict(passed=passed, protocol=protocol, shards=reports, comparisons=dict(zero_conditions=48,
        historical_task_values=336, repeated_head_task_gradients=56, nonzero_full_cache_controls=2),
        zero_s0_images=48, nonzero_s0_images=2, new_head_reverse_passes=112, new_full_B_or_UNet_reverse_passes=0,
        maximum_relative_l2=max(r['max_relative_l2'] for r in replicas),
        minimum_cosine=min(r['minimum_cosine'] for r in replicas),
        optimizer_updates=0, projection_calls=0, holdout_results_read=0, adopted_checkpoints=[],
        training_started=False, candidate_generated=False,
        cache_sha256={r['tensor_file']: r['tensor_sha256'] for r in reports},
        interpretation='qualification only for new 2308-parameter terminal head; not evidence of quality gain, feasible geometry, or accepted update; old 274372-dimensional G/P/norm bounds are not transferred')
    verify(args)
    write(args.out/'qualification.json', result)
    files = [args.out/'preflight_protocol.json', args.out/'qualification.json']
    files += [args.out/'cache'/('shard%d.json' % i) for i in range(4)]
    files += [args.out/row['image'] for row in nonzero]
    write(args.out/'review_manifest.json', {p.relative_to(args.out).as_posix(): sha(p) for p in files})
    with zipfile.ZipFile(args.out/'review.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files+[args.out/'review_manifest.json']:
            archive.write(path, path.relative_to(args.out).as_posix())
    print('TERMINAL_PREFLIGHT_REVIEW', passed, sha(args.out/'review.zip'), flush=True)
    if not passed:
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'cache', 'review', 'verify'))
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--project', type=Path, default=Path('/share/home/u2515283058/Mymodel'))
    parser.add_argument('--shard', type=int, choices=range(4), default=0)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.prior = args.prior.resolve()
    args.fixed = args.prior/'autonomous_repair_20261116/scaled_projection_retry'
    args.out = (args.out or args.prior/'autonomous_repair_20261116/terminal_head_preflight').resolve()
    args.data = args.project.parent/'datasets/BF'
    args.checkpoint = args.project/'output_eval/e2_state_b_pastel_followup_20261115/trust_retry/training/step1.pt'
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    try:
        {'prepare': prepare, 'cache': run_cache, 'review': review, 'verify': verify}[args.action](args)
    except Exception as error:
        if args.action == 'cache' and args.out.exists():
            write(args.out/('shard%d.failure.json' % args.shard), dict(error_type=type(error).__name__, message=str(error),
                details=getattr(error, 'details', None), complete=False, optimizer_updates=0,
                candidate_generated=False, adopted_checkpoints=[]))
        raise


if __name__ == '__main__':
    main()
