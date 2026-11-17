"""参考双基底末步头零训练资格：50完整S0、112真实局部反传，复用已锁定旧缓存。"""
import argparse
import copy
import inspect
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
from types import SimpleNamespace
import zipfile

import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from models.e2_reference_terminal_head import ReferenceTerminalHead, reference_statistics, LOG_RMS_MEAN, LOG_RMS_STD, PARAMETERS
from models.e2_state_texture_b import StateTextureB
from tools import e2_terminal_head_preflight as old
from tools.e2_terminal_head_one_step import objective_from_cache
from tools.e2_b_endpoint_losses import TASKS, flatten, set_parameters
from tools.e2_b_pair_finetune_losses import projection, pair_coefficients
from tools.e2_state_b_scaled_retry import SENTINELS, scaled_tasks, replica_report, NumericalFailure
from tools.e2_state_b_backward_repro import tensor_sha, vector_summary, compare_tensor
from tools.e2_state_b_amplitude_response import read, write, sha, pixels, CHECKPOINT_SHA
from tools.e2_state_b_pair_retry import PROTOCOL_SHA


SOURCE = Path(__file__).resolve().parents[1]
EXTRA = ('models/e2_reference_terminal_head.py', 'tools/e2_reference_terminal_preflight.py',
    'scripts/e2_reference_terminal_preflight.sbatch', 'tools/e2_terminal_head_one_step.py')
LOCAL_CALIBRATION = dict(mean=LOG_RMS_MEAN, std=LOG_RMS_STD, unbiased=False,
    references=48, unique_reference_sha256=39,
    phi_sha256='c7fef0bed23e67dd6b3a2b55d27d43e86800c8f7f8720d77d8f10333a747e19b',
    local_geometry_json_sha256='775379f2c9c5976c8907e8c5972d623bc3e25f1efc461442abb8c7cef8e1b891')


def source_args(args):
    result = copy.copy(args)
    result.out = args.preflight
    return result


def terminal_replay(head, record, scheduler, device):
    reference = record['target_record'][3]['reference'].to(device)
    statistics = head.statistics(reference)
    adapter = SimpleNamespace(from_features=lambda feature, mask: head.from_features(feature, mask, statistics['phi']))
    return old.terminal_replay(adapter, record, scheduler, device)


def reference_audit(reference):
    statistics = reference_statistics(reference)
    return dict(reference_sha256=tensor_sha(reference), first64_sha256=tensor_sha(reference[:, :64]),
        recovered_Y_sha256=tensor_sha((F.pixel_shuffle(reference[:, :64], 8)+1)/2),
        **{name: float(value) for name, value in statistics.items()})


def scheduler_configuration(actual, reference):
    """只忽略默认参数名称的列表顺序；实际调度配置与键集合保持精确比较。"""
    actual, reference = dict(actual), dict(reference)
    differences = {key: dict(actual=actual.get(key), reference=reference.get(key),
        actual_present=key in actual, reference_present=key in reference)
        for key in sorted(set(actual) | set(reference))
        if key not in actual or key not in reference or actual[key] != reference[key]}
    normalized_actual, normalized_reference = copy.deepcopy(actual), copy.deepcopy(reference)
    for config in (normalized_actual, normalized_reference):
        if '_use_default_values' in config and isinstance(config['_use_default_values'], list):
            config['_use_default_values'] = sorted(config['_use_default_values'])
    return dict(actual=actual, reference=reference, raw_diff=differences,
        normalized_equal=normalized_actual == normalized_reference,
        normalization='only sort the _use_default_values list; all other keys and values exact')


def prepare(args):
    previous = old.verify(source_args(args))
    qualification = read(args.preflight/'qualification.json')
    assert qualification['passed'] and qualification['new_head_reverse_passes'] == 112
    assert not args.out.exists(), '不覆盖资格实验'
    spec = read(args.preflight/'protocol.json')
    assert sha(args.preflight/'protocol.json') == PROTOCOL_SHA
    args.out.mkdir(parents=True)
    inputs, copied = {}, {}
    def record(path, expected=None):
        digest = sha(path)
        assert expected is None or digest == expected
        inputs[path.resolve().as_posix()] = digest
        return digest
    def duplicate(source, relative, expected=None):
        digest = record(source, expected)
        target = args.out/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        assert sha(target) == digest
        copied[relative] = digest
    duplicate(args.preflight/'protocol.json', 'protocol.json', PROTOCOL_SHA)
    for name, digest in spec['assets'].items():
        duplicate(args.preflight/name, name, digest)
    duplicate(args.preflight/'historical_f68.json', 'historical_f68.json')
    for row in read(args.out/'historical_f68.json')['rows']:
        duplicate(args.preflight/row['image'], row['image'], row['png_sha256'])
    for name in ('preflight_protocol.json', 'qualification.json'):
        record(args.preflight/name)
    references = []
    for index in range(4):
        name = 'cache/shard%d.pt' % index
        record(args.preflight/name, qualification['cache_sha256'][name])
        record(args.preflight/('cache/shard%d.json' % index))
        cache = torch.load(args.preflight/name, map_location='cpu', weights_only=False)
        for item in [i for i in spec['inputs'] if i['case'] == spec['train'][index]['case_id']]:
            reference = cache['records'][item['sample_condition']]['target_record'][3]['reference']
            name = 'cases/%s/%s_ref.png' % (item['case'], item['sample_condition'])
            assert sha(args.out/name) == item['reference_sha256']
            references.append(dict(item, reference_file=name, **reference_audit(reference)))
    vector = old.fixed_perturbation(PARAMETERS)
    torch.save(dict(vector=vector, purpose='fixed parity perturbation only; no training trust radius'), args.out/'probe_delta.pt')
    copied['probe_delta.pt'] = sha(args.out/'probe_delta.pt')
    record(args.checkpoint, CHECKPOINT_SHA)
    sources = set(previous['source_sha256']) | set(EXTRA)
    write(args.out/'basis_protocol.json', dict(original_protocol_sha256=PROTOCOL_SHA,
        checkpoint_sha256=CHECKPOINT_SHA, source_preflight=str(args.preflight), trainable_parameters=PARAMETERS,
        frozen_base_parameters=274372, parameter_order=['delta_weight0', 'delta_bias0', 'delta_weight1', 'delta_bias1'],
        expression='conv(feature, Wbase+(dW0+phi*dW1), bbase+(db0+phi*db1))*mask',
        gate=dict(global_index=49, timestep=1, local_B_call=24), calibration=LOCAL_CALIBRATION,
        phi='FP32 first64 reference -> pixel_shuffle -> Y=(u+1)/2 -> HP17 RMS -> log(rms+1e-6) -> fixed mu/std -> tanh',
        cpu_reference_statistics=references, calibration_reestimated=False,
        calibration_scope='fixed48 reference-only local CPU calibration; device-tail differences recorded, not refitted',
        sampler=spec['sampler'], source_cache_reused=True, new_full_cache_builds=0,
        planned_zero_s0=48, planned_nonzero_s0=2, planned_head_reverse_passes=112,
        sentinels=[list(v) for v in SENTINELS], nonzero_probes=[list(v) for v in old.NONZERO_PROBES],
        perturbation=dict(formula='FP64 sin(arange(1,4617)), normalized to .007854645 then FP32',
            tensor_sha256=tensor_sha(vector), actual_norm_fp64=float(vector.double().norm()),
            purpose='parity only; not a new parameter metric training budget'),
        loss_scale=1024., replica_gate=dict(max_relative_l2=.005, minimum_cosine=.99999, require_all_finite=True),
        analytic_lift_is_not_measured_G=True, optimizer_updates=0, projection_calls=0, new_candidates=0,
        holdout_results_read=0, adopted_checkpoints=[], training_loop_implemented=False,
        input_sha256=inputs, copied_sha256=copied, source_sha256={name: sha(SOURCE/name) for name in sorted(sources)},
        execution_commit=subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()))
    verify(args)
    print('REFERENCE_TERMINAL_PREPARED', flush=True)


def verify(args):
    old.verify(source_args(args))
    protocol = read(args.out/'basis_protocol.json')
    assert protocol['trainable_parameters'] == PARAMETERS and protocol['calibration'] == LOCAL_CALIBRATION
    assert protocol['planned_zero_s0'] == 48 and protocol['planned_head_reverse_passes'] == 112
    assert all(sha(Path(path)) == digest for path, digest in protocol['input_sha256'].items())
    assert all(sha(args.out/path) == digest for path, digest in protocol['copied_sha256'].items())
    assert all(sha(SOURCE/path) == digest for path, digest in protocol['source_sha256'].items())
    return protocol


def run_cache(args):
    from tools import e2_state_b_pilot as pilot, e2_state_b_endpoint as endpoint
    from tools.e2_oracle_texture import weights_digest
    from tools.e2_fft_experiment import environment
    from tools.e2_b_chain_repro import backend_flags
    started = time.monotonic()
    protocol = verify(args)
    destination = args.out/'cache'/('shard%d.json' % args.shard)
    assert not destination.exists()
    destination.parent.mkdir(exist_ok=True)
    spec = read(args.out/'protocol.json'); pair = spec['train'][args.shard]
    items = [r for r in spec['inputs'] if r['case'] == pair['case_id']]
    cache = torch.load(args.preflight/('cache/shard%d.pt' % args.shard), map_location='cpu', weights_only=False)
    source_report = read(args.preflight/('cache/shard%d.json' % args.shard))
    assert cache['parameter_count'] == 2308 and cache['zero_head'] and len(items) == 12
    history = {old.key(row): row for row in read(args.out/'historical_f68.json')['rows']}
    cpu_statistics = {old.key(row): row for row in protocol['cpu_reference_statistics']}
    runtime = copy.copy(args); runtime.action = 'train'
    core, pipe, net = pilot.load_core(runtime)
    torch.set_num_threads(2); cv2.setNumThreads(2)
    base = StateTextureB(color_channels=2, reference_channels=192).to(args.device)
    base.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=False)['b'])
    head = ReferenceTerminalHead(base).to(args.device).eval(); parameters = head.increment_parameters()
    net.eval().requires_grad_(False); pipe.vae.eval().requires_grad_(False)
    initial, frozen = weights_digest(base), weights_digest(net)
    assert initial == cache['b_hash'] and frozen == cache['net_hash'] and not torch.count_nonzero(flatten(parameters))
    assert sum(p.numel() for p in parameters) == PARAMETERS and sum(p.numel() for p in base.parameters()) == 274372
    assert [id(p) for p in head.parameters() if p.requires_grad] == [id(p) for p in parameters]
    assert type(pipe.scheduler).__name__ == 'DDIMScheduler' and pipe.scheduler.config.prediction_type == 'epsilon'
    assert not pipe.scheduler.config.clip_sample and not pipe.scheduler.config.thresholding and not pipe.scheduler.config.set_alpha_to_one
    scheduler_path = Path(inspect.getsourcefile(type(pipe.scheduler))); scheduler_sha = sha(scheduler_path)
    assert scheduler_sha == source_report['scheduler']['source_sha256']
    scheduler_audit = scheduler_configuration(pipe.scheduler.config, source_report['scheduler']['config'])
    scheduler_audit.update(source_sha256=scheduler_sha, source_reference_sha256=source_report['scheduler']['source_sha256'])
    write(args.out/('cache/shard%d.scheduler.json' % args.shard), scheduler_audit)
    assert scheduler_audit['normalized_equal'], '原DDIM配置存在非元数据顺序差异'

    def capture(item):
        head.reset_capture(); trace = []
        def hook(module, positional, keywords, output):
            trace.append(dict(timestep=int(keywords['timestep']), sha256=tensor_sha(output)))
        with torch.no_grad(), head.register_forward_hook(hook, with_kwargs=True):
            image, digest, tap, latent = endpoint.final_sample(runtime, core, pipe, net, pair, item['sample_condition'], head)
        assert head.active_calls == len(trace) == len(tap.states) == 25 and tap.index == 50
        saved = old.cpu_tree(head.last_record); previous = cache['records'][item['sample_condition']]
        prefix = [dict(index=s['index'], timestep=s['t'], zt=tensor_sha(s['zt']), raw_epsilon=tensor_sha(s['epsilon'])) for s in tap.states]
        assert prefix == previous['prefix_states'] and trace[:24] == previous['B_trace'][:24]
        assert digest == previous['noise_digest'] and tensor_sha(saved['feature']) == tensor_sha(previous['feature'])
        assert tensor_sha(saved['base_delta']) == tensor_sha(previous['base_delta'])
        assert tensor_sha(saved['mask']) == tensor_sha(previous['mask'])
        assert tensor_sha(saved['reference']) == tensor_sha(previous['target_record'][3]['reference'])
        return saved, latent.detach().cpu(), np.asarray(image).copy(), trace

    controls, pictures, full_latents, device_statistics = [], {}, {}, {}
    for item in items:
        c = item['sample_condition']; record = cache['records'][c]
        saved, latent, picture, trace = capture(item)
        assert tensor_sha(saved['full_delta']) == tensor_sha(saved['base_delta']) == tensor_sha(record['full_delta'])
        assert tensor_sha(latent) == tensor_sha(record['final']) and trace == record['B_trace']
        assert np.array_equal(picture, pixels(args.out/history[old.key(item)]['image']))
        actual = {k: float(v) for k, v in saved['statistics'].items()}
        stats = reference_audit(record['target_record'][3]['reference'].to(args.device))
        assert all(actual[k] == stats[k] for k in actual)
        controls.append(dict(item, historical_latent_equal=True, historical_png_equal=True, base_delta_equal=True,
            first24_B_outputs_equal=True, all_preterminal_states_equal=True, reference_equal=True,
            feature_sha256=tensor_sha(saved['feature']), latent_sha256=tensor_sha(latent), device_reference_statistics=stats,
            device_minus_cpu={k: stats[k]-cpu_statistics[old.key(item)][k] for k in actual}))
        pictures[c], full_latents[c], device_statistics[c] = picture, latent, saved['statistics']
        print('REFERENCE_ZERO_CONTROL', args.shard, c, flush=True)
    objective = objective_from_cache(runtime, pipe, head, pair, cache)
    vae_hash = weights_digest(pipe.vae); pipe.scheduler.set_timesteps(50, device=args.device)
    assert vae_hash == source_report['vae_hash_fp32']
    late = {}
    for item, control in zip(items, controls):
        c = item['sample_condition']; record = cache['records'][c]; entry = dict(pair=pair, condition=c)
        with torch.no_grad():
            latent, no_b, delta, corrected = terminal_replay(head, record, pipe.scheduler, args.device)
            assert tensor_sha(latent) == tensor_sha(full_latents[c]) and tensor_sha(corrected) == record['corrected_epsilon_sha256']
            tasks, _ = objective.score_latents(entry, latent.float(), no_b, residue=objective.metric(entry).surrogate)
            full_tasks, _ = objective.score_latents(entry, full_latents[c].to(args.device).float(), no_b, residue=objective.metric(entry).surrogate)
            values = endpoint.numbers(tasks)
            assert values == endpoint.numbers(full_tasks) == record['tasks']
            if item['group'] == 'pastel':
                _, positive, _, _, weight, _ = objective.records[pair['case_id'], c]
                anchor = objective.records[pair['case_id'], item['condition']+'_a100'][1]
                features = {k: float(v) for k, v in projection(objective.rgb(latent.float()), positive, anchor, weight).items()}
                assert features == source_report['late_features'][item['arm']]
                late[item['arm']] = features
        control.update(cached_full_latent_equal=True, historical_tasks_equal=True, cached_full_tasks_equal=True, tasks=values)
    coefficients = pair_coefficients(*(torch.tensor(late[a][k]) for a, k in (('a075', 'y'), ('a125', 'y'), ('a075', 'x'), ('a125', 'x'))))
    absolute = float((torch.tensor(late['a100']['g'])-1).square())
    assert float(coefficients['pair']) == source_report['pair_loss'] and absolute == source_report['absolute_loss']
    gradients, replicas, analytic = {}, [], []
    for item in items:
        if old.key(item) not in SENTINELS:
            continue
        c = item['sample_condition']; samples = []
        for repeat in range(2):
            latent, no_b, delta, corrected = terminal_replay(head, cache['records'][c], pipe.scheduler, args.device)
            entry = dict(pair=pair, condition=c)
            tasks, _ = objective.score_latents(entry, latent.float(), no_b, residue=objective.metric(entry).surrogate)
            assert endpoint.numbers(tasks) == cache['records'][c]['tasks']
            current = scaled_tasks(tasks, parameters).detach().cpu()
            assert current.shape == (7, PARAMETERS)
            if not torch.isfinite(current).all():
                raise NumericalFailure(c+'/repeat%d' % repeat, current)
            samples.append(current)
            del latent, no_b, delta, corrected, tasks
        repeated = replica_report(samples[1], samples[0]); gradients[c] = torch.stack(samples)
        replicas.append(dict(item, **repeated, gradient_summaries=[{task: vector_summary(g[i]) for i, task in enumerate(TASKS)} for g in samples]))
        old_g = cache['local_head_gradients'][c][0]
        predicted = torch.cat((old_g, old_g*device_statistics[c]['phi'].reshape(())), 1)
        analytic.append(dict(item, measured_sha256=tensor_sha(samples[0]), analytic_sha256=tensor_sha(predicted),
            tasks={task: compare_tensor(samples[0][i], predicted[i]) for i, task in enumerate(TASKS)},
            qualification_gate=False, interpretation='analytic comparison only; cannot replace measured repeatability'))
        print('REFERENCE_REAL_GRADIENT', args.shard, c, repeated['passed'], flush=True)
    assert len(replicas) == 2
    # 所有局部反传完成后，才切VAE到half生成PNG。
    with torch.no_grad():
        for item, control in zip(items, controls):
            c = item['sample_condition']
            assert np.array_equal(objective.pixels(full_latents[c].to(args.device)), pictures[c])
            control['cached_png_equal'] = True
    perturbation = torch.load(args.out/'probe_delta.pt', map_location='cpu', weights_only=False)['vector']
    assert tensor_sha(perturbation) == protocol['perturbation']['tensor_sha256']
    nonzero = []
    for item in items:
        if old.key(item) not in old.NONZERO_PROBES:
            continue
        c = item['sample_condition']; record = cache['records'][c]
        set_parameters(parameters, perturbation)
        try:
            saved, full, picture, trace = capture(item)
            with torch.no_grad():
                latent, no_b, delta, corrected = terminal_replay(head, record, pipe.scheduler, args.device)
                assert tensor_sha(latent) == tensor_sha(full) and tensor_sha(delta) == tensor_sha(saved['full_delta'])
                correction = delta.cpu()-record['base_delta']
                assert torch.count_nonzero(correction) > 0 and tensor_sha(latent) != tensor_sha(record['final'])
                entry = dict(pair=pair, condition=c)
                tasks, _ = objective.score_latents(entry, latent.float(), no_b, residue=objective.metric(entry).surrogate)
                full_tasks, _ = objective.score_latents(entry, full.to(args.device).float(), no_b, residue=objective.metric(entry).surrogate)
                values = endpoint.numbers(tasks); assert values == endpoint.numbers(full_tasks)
                assert np.array_equal(objective.pixels(latent), picture)
            name = 'nonzero_probe/%s_%s.png' % (pair['case_id'], c); path = args.out/name
            path.parent.mkdir(exist_ok=True); Image.fromarray(picture).save(path)
            nonzero.append(dict(item, full_cache_latent_equal=True, full_cache_png_equal=True, full_cache_tasks_equal=True,
                all_preterminal_states_equal=True, first24_B_outputs_equal=True, feature_equal=True, reference_equal=True,
                perturbation_sha256=tensor_sha(flatten(parameters)), correction=vector_summary(correction),
                latent_change=vector_summary(latent.cpu().float()-record['final'].float()),
                png_changed_channels=int(np.count_nonzero(picture != pictures[c])), tasks=values,
                image=name, png_sha256=sha(path), adopted=False))
        finally:
            set_parameters(parameters, torch.zeros_like(perturbation))
    assert weights_digest(base) == initial and weights_digest(net) == frozen and weights_digest(pipe.vae) == vae_hash
    assert not torch.count_nonzero(flatten(parameters)) and sha(scheduler_path) == scheduler_sha
    assert not any(p.grad is not None for module in (head, net, pipe.vae) for p in module.parameters())
    assert all(not p.requires_grad for module in (base, net, pipe.vae) for p in module.parameters())
    verify(args)
    tensor_file = destination.with_suffix('.pt'); assert not tensor_file.exists()
    torch.save(dict(local_head_gradients=gradients, device_reference_statistics=device_statistics, task_order=TASKS,
        parameter_count=PARAMETERS, zero_head=True, source_cache_sha256=sha(args.preflight/('cache/shard%d.pt' % args.shard)),
        b_hash=initial, net_hash=frozen, basis_protocol_sha256=sha(args.out/'basis_protocol.json')), tensor_file)
    result = dict(shard=args.shard, case=pair['case_id'], zero_controls=controls, replicas=replicas,
        analytic_lift_audit=analytic, nonzero_controls=nonzero, late_features=late, pair_loss=float(coefficients['pair']),
        absolute_loss=absolute, all_finite=True, numerical_qualification=all(r['passed'] for r in replicas),
        tensor_file=tensor_file.relative_to(args.out).as_posix(), tensor_sha256=sha(tensor_file),
        basis_protocol_sha256=sha(args.out/'basis_protocol.json'), source_sha256=protocol['source_sha256'],
        zero_s0_images=12, nonzero_s0_images=len(nonzero), new_head_reverse_passes=28, new_full_cache_builds=0,
        new_full_B_or_UNet_reverse_passes=0, optimizer_updates=0, projection_calls=0, new_candidates=0,
        diagnostic_parameter_assignments=2*len(nonzero), head_restored_zero=True, frozen_unchanged=True,
        b_hash=initial, net_hash=frozen, vae_hash_fp32=vae_hash, trainable_parameters=PARAMETERS,
        holdout_results_read=0, adopted_checkpoints=[], training_loop_implemented=False,
        scheduler=dict(class_name=type(pipe.scheduler).__name__, config=dict(pipe.scheduler.config), source_sha256=scheduler_sha,
            configuration_audit=scheduler_audit),
        environment=environment(), backend_flags=backend_flags(),
        hardware=dict(hostname=socket.gethostname(), gpu=torch.cuda.get_device_name(args.device), job=os.environ.get('SLURM_JOB_ID')),
        seconds=time.monotonic()-started)
    write(destination, result)
    print('REFERENCE_TERMINAL_SHARD', args.shard, result['numerical_qualification'], flush=True)
    if not result['numerical_qualification']:
        raise SystemExit(2)


def review(args):
    protocol = verify(args)
    assert not (args.out/'qualification.json').exists()
    reports = [read(args.out/('cache/shard%d.json' % i)) for i in range(4)]
    for i, report in enumerate(reports):
        assert report['shard'] == i and report['basis_protocol_sha256'] == sha(args.out/'basis_protocol.json')
        assert report['source_sha256'] == protocol['source_sha256'] and report['trainable_parameters'] == PARAMETERS
        assert sha(args.out/report['tensor_file']) == report['tensor_sha256']
        assert report['head_restored_zero'] and report['frozen_unchanged']
        assert report['scheduler']['configuration_audit']['normalized_equal']
        assert read(args.out/('cache/shard%d.scheduler.json' % i)) == report['scheduler']['configuration_audit']
        assert report['optimizer_updates'] == report['projection_calls'] == report['new_candidates'] == report['holdout_results_read'] == 0
    controls = [r for s in reports for r in s['zero_controls']]
    replicas = [r for s in reports for r in s['replicas']]
    nonzero = [r for s in reports for r in s['nonzero_controls']]
    assert len(controls) == 48 and len({old.key(r) for r in controls}) == 48
    assert len(replicas) == 8 and {old.key(r) for r in replicas} == set(SENTINELS)
    assert len(nonzero) == 2 and {old.key(r) for r in nonzero} == set(old.NONZERO_PROBES)
    assert sum(r['new_head_reverse_passes'] for r in reports) == 112
    for row in controls:
        assert all(row[k] for k in ('historical_latent_equal', 'historical_png_equal', 'base_delta_equal',
            'cached_full_latent_equal', 'historical_tasks_equal', 'cached_full_tasks_equal', 'cached_png_equal'))
    for row in nonzero:
        assert row['full_cache_latent_equal'] and row['full_cache_png_equal'] and row['full_cache_tasks_equal']
        assert row['latent_change']['norm'] > 0 and sha(args.out/row['image']) == row['png_sha256']
    passed = all(r['all_finite'] and r['numerical_qualification'] for r in reports)
    result = dict(passed=passed, protocol=protocol, shards=reports, zero_s0_images=48, nonzero_s0_images=2,
        new_head_reverse_passes=112, new_full_B_or_UNet_reverse_passes=0, new_full_cache_builds=0,
        maximum_relative_l2=max(r['max_relative_l2'] for r in replicas), minimum_cosine=min(r['minimum_cosine'] for r in replicas),
        maximum_device_cpu_phi_abs=max(abs(r['device_minus_cpu']['phi']) for r in controls),
        optimizer_updates=0, projection_calls=0, new_candidates=0, holdout_results_read=0, adopted_checkpoints=[],
        training_started=False, cache_sha256={r['tensor_file']: r['tensor_sha256'] for r in reports},
        interpretation='real4616 head parity/repeatability only; analytic geometry is not quality or equal function-budget evidence')
    verify(args); write(args.out/'qualification.json', result)
    files = [args.out/'basis_protocol.json', args.out/'qualification.json', args.out/'probe_delta.pt']
    files += [args.out/('cache/shard%d.%s' % (i, ext)) for i in range(4) for ext in ('json', 'pt')]
    files += [args.out/('cache/shard%d.scheduler.json' % i) for i in range(4)]
    files += [args.out/r['image'] for r in nonzero]
    write(args.out/'review_manifest.json', {p.relative_to(args.out).as_posix(): sha(p) for p in files})
    with zipfile.ZipFile(args.out/'review.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files+[args.out/'review_manifest.json']:
            archive.write(path, path.relative_to(args.out).as_posix())
    print('REFERENCE_TERMINAL_REVIEW', passed, sha(args.out/'review.zip'), flush=True)
    if not passed:
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'cache', 'review', 'verify'))
    parser.add_argument('--prior', type=Path, required=True); parser.add_argument('--out', type=Path)
    parser.add_argument('--project', type=Path, default=Path('/share/home/u2515283058/Mymodel'))
    parser.add_argument('--shard', type=int, choices=range(4), default=0); parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(); args.prior = args.prior.resolve()
    args.fixed = args.prior/'autonomous_repair_20261116/scaled_projection_retry'
    args.preflight = args.prior/'autonomous_repair_20261116/terminal_head_preflight'
    args.out = (args.out or args.prior/'autonomous_repair_20261116/reference_terminal_preflight').resolve()
    args.data = args.project.parent/'datasets/BF'
    args.checkpoint = args.project/'output_eval/e2_state_b_pastel_followup_20261115/trust_retry/training/step1.pt'
    torch.set_num_threads(2); cv2.setNumThreads(2)
    try:
        {'prepare': prepare, 'cache': run_cache, 'review': review, 'verify': verify}[args.action](args)
    except Exception as error:
        if args.action == 'cache' and args.out.exists():
            write(args.out/('shard%d.failure.json' % args.shard), dict(error_type=type(error).__name__, message=str(error),
                details=getattr(error, 'details', None), complete=False, optimizer_updates=0, adopted_checkpoints=[]))
        raise


if __name__ == '__main__':
    main()
