"""合格4616参考双基底零头的C0/C1独立单步；原损失、QR约束及完整S0保护。"""
import argparse
import copy
import inspect
from pathlib import Path
import shutil
import subprocess
import time
import zipfile

import cv2
import numpy as np
from PIL import Image
import torch

from models.e2_state_texture_b import StateTextureB
from models.e2_reference_terminal_head import ReferenceTerminalHead, PARAMETERS
from tools import e2_reference_terminal_preflight as preflight
from tools import e2_terminal_head_qr_diagnostic as qr
from tools import e2_state_b_pair_summary as summary
from tools.e2_terminal_head_one_step import objective_from_cache, combine_gradients
from tools.e2_terminal_head_projection_retry import original_step
from tools.e2_b_endpoint_losses import TASKS, flatten, set_parameters
from tools.e2_b_balanced_losses import assign
from tools.e2_b_pair_finetune_losses import projection, pair_coefficients
from tools.e2_state_b_scaled_retry import SENTINELS, scaled_tasks, scaled_scalar, replica_report, NumericalFailure
from tools.e2_state_b_backward_repro import tensor_sha, vector_summary
from tools.e2_state_b_amplitude_response import read, write, sha, pixels, CHECKPOINT_SHA
from tools.e2_state_b_pair_retry import PROTOCOL_SHA
from tools.e2_b_scaled_replay_review import Inputs, validate_report, replay_eligibility, same_numbers
from tools.e2_state_b_pair_evaluation import replay_inputs, score_entry, aggregate, strict_replay_protection, training_comparison


SOURCE = Path(__file__).resolve().parents[1]
ARMS = ('C0', 'C1')
INCREMENTS = ('delta_weight0', 'delta_bias0', 'delta_weight1', 'delta_bias1')
EXTRA = ('tools/e2_reference_terminal_one_step.py', 'scripts/e2_reference_terminal_one_step.sbatch',
    'tools/e2_terminal_head_qr_diagnostic.py', 'tools/e2_terminal_head_projection_retry.py',
    'tools/e2_b_scaled_replay_review.py', 'tools/e2_state_b_pair_evaluation.py', 'tools/e2_state_b_pair_summary.py')


def qualified_preflight(args):
    previous = copy.copy(args); previous.out = args.basis
    protocol = preflight.verify(previous)
    result = read(args.basis/'qualification.json')
    assert result['protocol'] == protocol and result['passed'], '新4616头资格未通过，禁止单步实验'
    assert (result['zero_s0_images'], result['nonzero_s0_images'], result['new_head_reverse_passes']) == (48, 2, 112)
    assert result['maximum_relative_l2'] <= .005 and result['minimum_cosine'] >= .99999
    assert result['optimizer_updates'] == result['new_full_B_or_UNet_reverse_passes'] == 0
    assert all(sha(args.basis/name) == digest for name, digest in result['cache_sha256'].items())
    return protocol, result


def prepare(args):
    prior, qualification = qualified_preflight(args)
    assert not args.out.exists(), '不覆盖已有单步实验'
    spec = read(args.basis/'protocol.json'); assert sha(args.basis/'protocol.json') == PROTOCOL_SHA
    cfg = spec['training']
    assert (cfg['lr'], cfg['weight_decay'], cfg['gradient_clip'], cfg['scale'], cfg['displacement_cap']) == (3e-5, .01, 1., .5, .007854645)
    assert (cfg['default_margin'], cfg['protected_amplitude_margin']) == (.01, .18)
    args.out.mkdir(parents=True)
    inputs, copied = {}, {}
    def record(path, expected=None):
        value = sha(path); assert expected is None or value == expected
        inputs[path.resolve().as_posix()] = value
        return value
    def duplicate(path, name, expected=None):
        value = record(path, expected); target = args.out/name
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
        assert sha(target) == value; copied[name] = value
    duplicate(args.basis/'protocol.json', 'protocol.json', PROTOCOL_SHA)
    for name, digest in spec['assets'].items():
        duplicate(args.basis/name, name, digest)
    duplicate(args.basis/'historical_f68.json', 'evaluation/f68/replay/result.json')
    for row in read(args.basis/'historical_f68.json')['rows']:
        duplicate(args.basis/row['image'], row['image'], row['png_sha256'])
    for name in ('basis_protocol.json', 'qualification.json'):
        record(args.basis/name)
    original = read(args.preflight/'qualification.json')
    for index in range(4):
        name = 'cache/shard%d.pt' % index
        record(args.preflight/name, original['cache_sha256'][name])
        record(args.preflight/('cache/shard%d.json' % index))
        record(args.basis/name, qualification['cache_sha256'][name])
        record(args.basis/('cache/shard%d.json' % index))
    record(args.checkpoint, CHECKPOINT_SHA)
    sources = set(prior['source_sha256']) | set(EXTRA)
    write(args.out/'one_step_overlay.json', dict(parameterization='reference-statistic terminal two-basis increments',
        parameter_count=PARAMETERS, parameter_order=list(INCREMENTS), base_checkpoint_sha256=CHECKPOINT_SHA,
        original_protocol_sha256=PROTOCOL_SHA, head_start='four independent zero FP32 tensors',
        basis_run=str(args.basis), basis_qualification_sha256=sha(args.basis/'qualification.json'),
        original_cache_sha256=original['cache_sha256'], basis_reference_G_sha256=qualification['cache_sha256'],
        calibration=prior['calibration'], phi_expression=prior['phi'], sampler=spec['sampler'],
        gradient_loss_scale=1024., new_reverse_passes=dict(tasks=336, y=12, absolute=4),
        solver=dict(method='unpivoted economic QR/SLSQP, same original-space Euclidean QP',
            ftol=1e-12, maxiter=500, limits=qr.LIMITS, retries=0, original_configured_sweeps=cfg['projection_sweeps']),
        original_step=cfg, original_losses_and_gates_unchanged=True, independent_adamw_steps=2,
        planned_complete_s0_images=96, holdout_results_read=0, adopted_checkpoints=[], continuous_training=False,
        functional_budget_matched_to_2308=False, analytic_lift_used_as_fresh_G=False,
        input_sha256=inputs, copied_sha256=copied, source_sha256={name: sha(SOURCE/name) for name in sorted(sources)},
        execution_commit=subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()))
    verify(args)


def verify(args):
    qualified_preflight(args)
    overlay = read(args.out/'one_step_overlay.json')
    assert overlay['parameter_count'] == PARAMETERS and overlay['parameter_order'] == list(INCREMENTS)
    assert overlay['calibration'] == preflight.LOCAL_CALIBRATION and sha(args.out/'protocol.json') == PROTOCOL_SHA
    assert all(sha(Path(path)) == value for path, value in overlay['input_sha256'].items())
    assert all(sha(args.out/path) == value for path, value in overlay['copied_sha256'].items())
    assert all(sha(SOURCE/path) == value for path, value in overlay['source_sha256'].items())
    proposal = args.out/'training/proposals.json'
    if proposal.exists():
        reports = read(proposal)
        assert reports['overlay_sha256'] == sha(args.out/'one_step_overlay.json')
        assert all(sha(args.out/name) == digest for name, digest in reports['numerical_file_sha256'].items())
        _, _, _, qualification = gradient_inputs(args, overlay)
        assert qualification == read(args.out/'gradient_qualification.json') and qualification['passed']
        if not reports['stop_before_replay']:
            raw = torch.load(args.out/'training/raw.pt', map_location='cpu', weights_only=False)
            numeric = torch.load(args.out/'training/numerics.pt', map_location='cpu', weights_only=False)
            diagnostic = read(args.out/'training/solver_diagnostic.json')
            assert diagnostic['all_qualified'] and tensor_sha(raw['constraints']) == reports['constraint_sha256']
            assert tensor_sha(raw['margins']) == reports['margins_sha256']
            assert reports['new_gradient_sha256'] == qualification['new_gradient_sha256']
            for arm in ARMS:
                vector, report = raw['raws'][arm]['raw'], reports['arms'][arm]
                assert tensor_sha(vector) == report['raw_sha256'] and float(vector.norm()) == report['raw_norm']
                assert diagnostic['arms'][arm]['numerical_qualification']
                projected = numeric['arms'][arm]['delta32']
                slack = numeric['N32'].double()@projected.double()/report['raw_norm']-raw['margins'].double()
                assert bool((slack >= -1e-6).all())
                expected, step = original_step(vector, projected, raw['constraints'], read(args.out/'protocol.json')['training'])
                assert all(report[k] == v for k, v in step.items())
                head = load_candidate(args, arm, 'cpu', report)
                assert torch.equal(flatten(head.increment_parameters()), expected)
    return overlay


def load_cache(args, index, overlay):
    name = 'cache/shard%d.pt' % index
    assert sha(args.preflight/name) == overlay['original_cache_sha256'][name]
    assert sha(args.basis/name) == overlay['basis_reference_G_sha256'][name]
    cache = torch.load(args.preflight/name, map_location='cpu', weights_only=False)
    reference = torch.load(args.basis/name, map_location='cpu', weights_only=False)
    assert cache['parameter_count'] == 2308 and reference['parameter_count'] == PARAMETERS
    assert cache['zero_head'] and reference['zero_head'] and cache['b_hash'] == reference['b_hash']
    assert reference['basis_protocol_sha256'] == sha(args.basis/'basis_protocol.json')
    return cache, reference


def load_head(args, device):
    base = StateTextureB(color_channels=2, reference_channels=192).to(device)
    base.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=False)['b'])
    head = ReferenceTerminalHead(base).to(device).eval()
    assert sum(p.numel() for p in head.increment_parameters()) == PARAMETERS
    assert [id(p) for p in head.parameters() if p.requires_grad] == [id(p) for p in head.increment_parameters()]
    return head


def effective_difference(head, phi):
    """先按模型完全相同顺序形成FP32合并权重，再升double减去f68。"""
    phi = phi.detach().reshape(()); layer = head.base.output
    original = torch.cat((layer.weight.detach().flatten(), layer.bias.detach().flatten()))
    change = torch.cat(((head.delta_weight0+phi*head.delta_weight1).flatten(),
                        (head.delta_bias0+phi*head.delta_bias1).flatten()))
    merged = torch.cat(((layer.weight+(head.delta_weight0+phi*head.delta_weight1)).flatten(),
                       (layer.bias+(head.delta_bias0+phi*head.delta_bias1)).flatten()))
    return merged.double()-original.double(), change


def raw_once(parameters, direction, cfg):
    """四个实际形状参数独立AdamW一次；不是旧空间raw重建或连续训练。"""
    start = flatten(parameters); assert not torch.count_nonzero(start)
    optimizer = torch.optim.AdamW(parameters, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    assign(parameters, direction)
    clipped = torch.nn.utils.clip_grad_norm_(parameters, cfg['gradient_clip']); assert torch.isfinite(clipped)
    optimizer.step()
    raw = start-flatten(parameters); assert torch.isfinite(raw).all()
    return raw, dict(raw_norm=float(raw.norm()), raw_sha256=tensor_sha(raw), gradient_norm=float(direction.norm()),
        gradient_clip_return=float(clipped), direction_sha256=tensor_sha(direction), optimizer_steps=1)


def gradient(args):
    from tools import e2_state_b_pilot as pilot, e2_state_b_endpoint as endpoint
    from tools.e2_oracle_texture import weights_digest
    from tools.e2_fft_experiment import environment
    started = time.monotonic(); overlay = verify(args)
    destination = args.out/('gradients/shard%d.pt' % args.shard); assert not destination.exists()
    destination.parent.mkdir(exist_ok=True)
    spec = read(args.out/'protocol.json'); pair = spec['train'][args.shard]
    items = [r for r in spec['inputs'] if r['case'] == pair['case_id']]
    cache, reference = load_cache(args, args.shard, overlay)
    metadata = read(args.basis/('cache/shard%d.json' % args.shard))
    runtime = copy.copy(args); runtime.action = 'train'
    _, pipe, net = pilot.load_core(runtime); torch.set_num_threads(2); cv2.setNumThreads(2)
    net.eval().requires_grad_(False); head = load_head(args, args.device); parameters = head.increment_parameters()
    initial, frozen = weights_digest(head.base), weights_digest(net)
    assert initial == cache['b_hash'] and frozen == cache['net_hash']
    objective = objective_from_cache(runtime, pipe, head, pair, cache)
    vae_hash = weights_digest(pipe.vae); assert vae_hash == metadata['vae_hash_fp32']
    assert sha(Path(inspect.getsourcefile(type(pipe.scheduler)))) == metadata['scheduler']['source_sha256']
    scheduler_audit = preflight.scheduler_configuration(pipe.scheduler.config, metadata['scheduler']['config'])
    write(destination.with_suffix('.scheduler.json'), scheduler_audit)
    assert scheduler_audit['normalized_equal']
    pipe.scheduler.set_timesteps(50, device=args.device)
    gradients, values, late, replicas = [], [], {}, []
    for item in items:
        c = item['sample_condition']; record = cache['records'][c]; entry = dict(pair=pair, condition=c)
        statistics = head.statistics(record['target_record'][3]['reference'].to(args.device))
        phi = statistics['phi'].detach().cpu()
        latent, base, delta, corrected = preflight.terminal_replay(head, record, pipe.scheduler, args.device)
        assert tensor_sha(latent) == tensor_sha(record['final']) and tensor_sha(delta) == tensor_sha(record['base_delta'])
        tasks, _ = objective.score_latents(entry, latent.float(), base, residue=objective.metric(entry).surrogate)
        numeric = endpoint.numbers(tasks); assert numeric == record['tasks']
        current = scaled_tasks(tasks, parameters).detach().cpu()
        if not torch.isfinite(current).all():
            raise NumericalFailure(c, current)
        assert current.shape == (7, PARAMETERS)
        gradients.append(current)
        values.append(dict(item, tasks=numeric, noise_digest=record['noise_digest'], zero_latent_equal=True,
            phi=float(phi), phi_sha256=tensor_sha(phi),
            phi_minus_preflight=float(phi-reference['device_reference_statistics'][c]['phi'])))
        if preflight.old.key(item) in SENTINELS:
            controls = reference['local_head_gradients'][c]
            replicas.append(dict(item, **replica_report(current, controls[0]),
                reference='real4616 preflight graph0 and graph1', reference_tensor_sha256=tensor_sha(controls[0]),
                fresh_tensor_sha256=tensor_sha(current), additional_reference_graph1=replica_report(current, controls[1])))
        del latent, base, delta, corrected, tasks
        if item['group'] == 'pastel':
            latent, base, delta, corrected = preflight.terminal_replay(head, record, pipe.scheduler, args.device)
            _, positive, _, _, weight, _ = objective.records[pair['case_id'], c]
            anchor = objective.records[pair['case_id'], item['condition']+'_a100'][1]
            feature = projection(objective.rgb(latent.float()), positive, anchor, weight)
            features = {k: float(v.detach()) for k, v in feature.items()}
            assert features == metadata['late_features'][item['arm']]
            row = dict(features=features, gradient_y=scaled_scalar(feature['y'], parameters, retain_graph=item['arm'] == 'a100').cpu(),
                phi=float(phi), phi_sha256=tensor_sha(phi))
            if item['arm'] == 'a100':
                row['gradient_absolute'] = scaled_scalar((feature['g']-1).square(), parameters).cpu()
            assert all(torch.isfinite(v).all() for k, v in row.items() if k.startswith('gradient'))
            late[item['arm']] = row
            del latent, base, delta, corrected, feature
        print('REFERENCE_FRESH_G', args.shard, c, flush=True)
    coefficients = pair_coefficients(*(torch.tensor(late[a]['features'][k]) for a, k in (('a075', 'y'), ('a125', 'y'), ('a075', 'x'), ('a125', 'x'))))
    pair_gradient = coefficients['low']*late['a075']['gradient_y']+coefficients['high']*late['a125']['gradient_y']
    assert float(coefficients['pair']) == metadata['pair_loss'] and torch.isfinite(pair_gradient).all()
    assert len(replicas) == 2 and len(gradients) == 12 and set(late) == {'a075', 'a100', 'a125'}
    assert weights_digest(head.base) == initial and weights_digest(net) == frozen and weights_digest(pipe.vae) == vae_hash
    assert not torch.count_nonzero(flatten(parameters)) and not any(p.grad is not None for p in head.parameters())
    verify(args)
    payload = dict(shard=args.shard, case=pair['case_id'], items=items, values=values, gradients=torch.stack(gradients),
        absolute_gradient=late['a100']['gradient_absolute'], pair_gradient=pair_gradient, late_gradients=late,
        late_features={a: r['features'] for a, r in late.items()}, pair_loss=float(coefficients['pair']),
        pair_coefficients={k: float(v) for k, v in coefficients.items()}, replicas=replicas, all_finite=True,
        numerical_qualification=all(r['passed'] and r['additional_reference_graph1']['passed'] for r in replicas),
        base_b_hash=initial, head_hash=tensor_sha(flatten(parameters)), net_hash=frozen, vae_hash_fp32=vae_hash,
        protocol_sha256=PROTOCOL_SHA, overlay_sha256=sha(args.out/'one_step_overlay.json'), source_sha256=overlay['source_sha256'],
        parameter_count=PARAMETERS, loss_scale=1024., gradients_unscaled=True, reverse_passes=dict(tasks=84, y=3, absolute=1),
        fresh_gradients=True, optimizer_updates=0, frozen_unchanged=True, scheduler_audit=scheduler_audit,
        environment=environment(), seconds=time.monotonic()-started)
    torch.save(payload, destination)
    meta = {k: v for k, v in payload.items() if k not in ('gradients', 'absolute_gradient', 'pair_gradient', 'late_gradients')}
    meta.update(tensor_file=destination.relative_to(args.out).as_posix(), tensor_sha256=sha(destination))
    write(destination.with_suffix('.json'), meta)
    if not payload['numerical_qualification']:
        raise SystemExit(2)


def gradient_inputs(args, overlay, tensors=False):
    spec = read(args.out/'protocol.json'); shards, values, hashes, pt_hashes, replicas = [], {}, {}, {}, []
    for i in range(4):
        path = args.out/('gradients/shard%d.json' % i); meta = read(path); pt = path.with_suffix('.pt')
        assert sha(pt) == meta['tensor_sha256'] and meta['shard'] == i and meta['case'] == spec['train'][i]['case_id']
        assert meta['parameter_count'] == PARAMETERS and meta['fresh_gradients'] and meta['gradients_unscaled']
        assert meta['all_finite'] and meta['frozen_unchanged'] and meta['optimizer_updates'] == 0 and meta['loss_scale'] == 1024.
        assert meta['protocol_sha256'] == PROTOCOL_SHA and meta['overlay_sha256'] == sha(args.out/'one_step_overlay.json')
        assert meta['source_sha256'] == overlay['source_sha256'] and meta['reverse_passes'] == dict(tasks=84, y=3, absolute=1)
        assert meta['items'] == [r for r in spec['inputs'] if r['case'] == spec['train'][i]['case_id']]
        for row in meta['values']:
            k = row['case'], row['sample_condition']; assert k not in values
            values[k] = row
        hashes[path.name], pt_hashes[pt.name] = sha(path), sha(pt); replicas.extend(meta['replicas'])
        if tensors:
            payload = torch.load(pt, map_location='cpu', weights_only=False)
            assert payload['items'] == meta['items'] and payload['values'] == meta['values']
            assert payload['gradients'].shape == (12, 7, PARAMETERS)
            assert all(torch.isfinite(payload[k]).all() for k in ('gradients', 'pair_gradient', 'absolute_gradient'))
            late = payload['late_gradients']; coefficients = payload['pair_coefficients']
            assert all(late[a]['gradient_y'].shape == (PARAMETERS,) and torch.isfinite(late[a]['gradient_y']).all() for a in ('a075', 'a100', 'a125'))
            assert torch.equal(payload['pair_gradient'], coefficients['low']*late['a075']['gradient_y']+coefficients['high']*late['a125']['gradient_y'])
            assert torch.equal(payload['absolute_gradient'], late['a100']['gradient_absolute'])
            shards.append(payload)
        else:
            shards.append(meta)
    assert len(values) == 48 and len(replicas) == 8 and {preflight.old.key(r) for r in replicas} == set(SENTINELS)
    qualification = dict(passed=all(r['passed'] and r['additional_reference_graph1']['passed'] for r in replicas), all_finite=True,
        maximum_relative_l2=max(max(r['max_relative_l2'], r['additional_reference_graph1']['max_relative_l2']) for r in replicas),
        minimum_cosine=min(min(r['minimum_cosine'], r['additional_reference_graph1']['minimum_cosine']) for r in replicas),
        replicas=replicas, parameter_count=PARAMETERS, new_gradient_sha256=pt_hashes, gradient_json_sha256=hashes,
        new_task_reverse_passes=336, new_y_reverse_passes=12, new_absolute_reverse_passes=4, saved_y_branches=12,
        loss_scale=1024., optimizer_updates=0, analytic_lift_used_as_fresh_G=False,
        overlay_sha256=sha(args.out/'one_step_overlay.json'))
    return shards, values, hashes, qualification


def load_candidate(args, arm, device, report=None):
    from tools.e2_oracle_texture import weights_digest
    if report is None:
        report = read(args.out/'training/proposals.json')['arms'][arm]
    path = args.out/'training'/(arm+'.pt'); assert sha(path) == report['checkpoint_sha256']
    payload = torch.load(path, map_location='cpu', weights_only=False)
    assert set(payload) == {'increments', 'calibration', 'base_checkpoint_sha256', 'report'}
    assert payload['base_checkpoint_sha256'] == CHECKPOINT_SHA and payload['calibration'] == preflight.LOCAL_CALIBRATION
    assert payload['report'] == {k: v for k, v in report.items() if k != 'checkpoint_sha256'}
    head = load_head(args, device); assert list(payload['increments']) == list(INCREMENTS)
    with torch.no_grad():
        for name in INCREMENTS:
            value = payload['increments'][name]; target = getattr(head, name)
            assert value.dtype == torch.float32 and value.shape == target.shape and torch.isfinite(value).all()
            target.copy_(value)
    vector = flatten(head.increment_parameters())
    assert weights_digest(head.base) == report['base_b_hash'] and tensor_sha(vector) == report['head_sha256']
    assert tensor_sha(torch.zeros_like(vector)-vector) == report['actual_descent_sha256']
    assert report['parameter_updates'] == int(bool(torch.count_nonzero(vector)))
    return head


def propose(args):
    from tools.e2_oracle_texture import weights_digest
    started = time.monotonic(); overlay = verify(args)
    destination = args.out/'training'; assert not destination.exists()
    shards, _, hashes, qualification = gradient_inputs(args, overlay, tensors=True)
    assert not (args.out/'gradient_qualification.json').exists()
    write(args.out/'gradient_qualification.json', qualification)
    if not qualification['passed']:
        raise SystemExit(2)
    spec = read(args.out/'protocol.json'); cfg = spec['training']
    directions, gradients, margins, component_norms = combine_gradients(shards, spec)
    assert gradients.shape == (336, PARAMETERS) and all(torch.isfinite(v).all() for v in directions.values())
    norms, active, rows32 = qr.normalization(gradients); assert bool(active.all())
    rows = rows32.double().numpy(); required = margins.double().numpy()
    q, r, qr_report = qr.factorize(rows)
    destination.mkdir(); raws, numerics, tensors, heads = {}, {}, {}, {}
    for arm in ARMS:
        head = load_head(args, 'cpu'); heads[arm] = head
        raw, raw_report = raw_once(head.increment_parameters(), directions[arm], cfg); raws[arm] = dict(raw=raw, **raw_report)
        try:
            report, result = qr.solve_arm(rows, q, raw, raw_report['raw_norm'], required, qr_report)
            numerics[arm], tensors[arm] = report, result
        except Exception as error:
            numerics[arm] = dict(numerical_qualification=False, error_type=type(error).__name__, message=str(error))
    torch.save(dict(raws=raws, constraints=gradients, margins=margins, directions=directions), destination/'raw.pt')
    torch.save(dict(norms=norms, active=active, N32=rows32, Q=torch.from_numpy(q), R=torch.from_numpy(r), arms=tensors), destination/'numerics.pt')
    diagnostic = dict(qr=qr_report, arms=numerics, all_qualified=qr_report['passed'] and all(r['numerical_qualification'] for r in numerics.values()),
        limits=qr.LIMITS, slsqp_calls=2, raw_adamw_steps=2, input_gradient_space=PARAMETERS,
        constraints_sha256=tensor_sha(gradients), margins_sha256=tensor_sha(margins), new_reverse_passes=0)
    write(destination/'solver_diagnostic.json', diagnostic)
    numerical_files = {str(p.relative_to(args.out).as_posix()): sha(p) for p in (destination/'raw.pt', destination/'numerics.pt', destination/'solver_diagnostic.json')}
    proposals = dict(arms={}, parameter_count=PARAMETERS, conditions=48, full_task_reverse_passes=336,
        late_y_reverse_passes=12, late_abs_reverse_passes=4, new_reverse_passes=352, saved_y_branches=12,
        independent_adamw_proposals=2, sequential_training_updates=0, slsqp_calls=2,
        gradient_seconds=sum(s['seconds'] for s in shards), component_norms=component_norms,
        constraint_sha256=tensor_sha(gradients), margins_sha256=tensor_sha(margins), margins=margins.tolist(),
        new_gradient_sha256=qualification['new_gradient_sha256'], gradient_json_sha256=hashes,
        numerical_file_sha256=numerical_files, overlay_sha256=sha(args.out/'one_step_overlay.json'),
        original_losses_and_gates_unchanged=True, functional_budget_matched_to_2308=False,
        frozen_unchanged=True, heldout_used_for_training=False, adopted_checkpoints=[],
        stop_before_replay=not diagnostic['all_qualified'], saved_candidate_parameter_updates=0)
    if diagnostic['all_qualified']:
        candidates, steps = {}, {}
        for arm in ARMS:
            candidates[arm], steps[arm] = original_step(raws[arm]['raw'], tensors[arm]['delta32'], gradients, cfg)
        proposals['stop_before_replay'] = not all(s['parameter_updates'] == 1 and s['minimum_cosine'] >= -1e-5 for s in steps.values())
        if not proposals['stop_before_replay']:
            for arm in ARMS:
                head = heads[arm]; base_hash = weights_digest(head.base)
                set_parameters(head.increment_parameters(), candidates[arm])
                report = dict(steps[arm], **{k: v for k, v in raws[arm].items() if k != 'raw'}, arm=arm,
                    trainable_parameters=PARAMETERS, base_checkpoint_sha256=CHECKPOINT_SHA, base_b_hash=base_hash,
                    initial_head_sha256=tensor_sha(torch.zeros_like(candidates[arm])),
                    protocol_sha256=PROTOCOL_SHA, overlay_sha256=sha(args.out/'one_step_overlay.json'),
                    gradient_qualification_sha256=sha(args.out/'gradient_qualification.json'),
                    projection=dict(method='QR/SLSQP same original Euclidean QP', feasible=True, tolerance=1e-6,
                        actual_slack_min=numerics[arm]['fp32_slack_min'], actual_violations=numerics[arm]['original_fp32_violations'],
                        delta32_sha256=tensor_sha(tensors[arm]['delta32']), numerical_qualification_is_not_quality=True),
                    frozen_unchanged=True, adopted=False, acceptance='pending original complete48 S0 replay')
                path = destination/(arm+'.pt')
                torch.save(dict(increments={n: getattr(head, n).detach().cpu() for n in INCREMENTS},
                    calibration=preflight.LOCAL_CALIBRATION, base_checkpoint_sha256=CHECKPOINT_SHA, report=report), path)
                proposals['arms'][arm] = dict(report, checkpoint_sha256=sha(path))
                assert weights_digest(head.base) == base_hash
            proposals['saved_candidate_parameter_updates'] = 2
        else:
            proposals['rejected_step_reports'] = steps
    proposals['proposal_seconds'] = time.monotonic()-started
    write(destination/'proposals.json', proposals); verify(args)
    print('REFERENCE_SINGLE_PROPOSALS', not proposals['stop_before_replay'], flush=True)
    if proposals['stop_before_replay']:
        raise SystemExit(2)


@torch.no_grad()
def evaluate(args):
    from tools import e2_state_b_pilot as pilot, e2_state_b_endpoint as endpoint
    from tools.e2_oracle_texture import weights_digest
    from tools.e2_fft_experiment import environment
    started = time.monotonic(); overlay = verify(args)
    spec = read(args.out/'protocol.json'); proposals = read(args.out/'training/proposals.json')
    assert not proposals['stop_before_replay']
    _, task_baselines, gradient_hashes, qualification = gradient_inputs(args, overlay)
    assert qualification == read(args.out/'gradient_qualification.json') and qualification['passed']
    destination = args.out/'evaluation'/args.arm/'replay'; assert not destination.exists()
    destination.mkdir(parents=True)
    entries, _, _ = replay_inputs(args, spec)
    baseline = read(args.out/'evaluation/f68/replay/result.json')
    old_rows = {summary.row_key(r, 'replay'): r for r in baseline['rows']}
    runtime = copy.copy(args); runtime.action = 'train'
    core, pipe, net = pilot.load_core(runtime); torch.set_num_threads(2); cv2.setNumThreads(2)
    net.eval().requires_grad_(False); head = load_candidate(args, args.arm, args.device).eval().requires_grad_(False)
    base_hash, head_hash, net_hash = weights_digest(head.base), tensor_sha(flatten(head.increment_parameters())), weights_digest(net)
    assert net_hash == baseline['net_hash']
    rows, responses, scheduler_audits = [], {}, {}
    current_case, objective, cache = None, None, None
    for entry in entries:
        case, c = entry['case'], entry['sample_condition']
        if case != current_case:
            index = next(i for i, pair in enumerate(spec['train']) if pair['case_id'] == case)
            cache, _ = load_cache(args, index, overlay)
            metadata = read(args.basis/('cache/shard%d.json' % index))
            assert cache['b_hash'] == base_hash and cache['net_hash'] == net_hash
            objective = objective_from_cache(runtime, pipe, head, entry['pair'], cache)
            assert weights_digest(pipe.vae) == metadata['vae_hash_fp32']
            assert sha(Path(inspect.getsourcefile(type(pipe.scheduler)))) == metadata['scheduler']['source_sha256']
            audit = preflight.scheduler_configuration(pipe.scheduler.config, metadata['scheduler']['config'])
            scheduler_audits[case] = audit; write(destination/'scheduler_audits.json', scheduler_audits)
            assert audit['normalized_equal']; current_case = case
        record = cache['records'][c]; head.reset_capture(); trace = []
        def capture(module, positional, keywords, output):
            trace.append(dict(timestep=int(keywords['timestep']), sha256=tensor_sha(output)))
        with head.register_forward_hook(capture, with_kwargs=True):
            image, digest, tap, latent = endpoint.final_sample(runtime, core, pipe, net, entry['pair'], c, head)
        assert tap.index == 50 and len(tap.states) == head.active_calls == len(trace) == 25
        assert digest == entry['expected_digest'] == record['noise_digest']
        prefix = [dict(index=s['index'], timestep=s['t'], zt=tensor_sha(s['zt']), raw_epsilon=tensor_sha(s['epsilon'])) for s in tap.states]
        assert prefix == record['prefix_states'] and trace[:24] == record['B_trace'][:24]
        saved = head.last_record
        assert tensor_sha(saved['feature']) == tensor_sha(record['feature'])
        assert tensor_sha(saved['base_delta']) == tensor_sha(record['base_delta'])
        assert tensor_sha(saved['reference']) == tensor_sha(record['target_record'][3]['reference'])
        phi = saved['statistics']['phi']
        assert tensor_sha(phi) == task_baselines[case, c]['phi_sha256'], '评测实际phi与本轮fresh G点不一致'
        cached, base, delta, corrected = preflight.terminal_replay(head, record, pipe.scheduler, args.device)
        assert tensor_sha(cached) == tensor_sha(latent) and tensor_sha(delta) == tensor_sha(saved['full_delta'])
        task_entry = dict(pair=entry['pair'], condition=c)
        tasks, _ = objective.score_latents(task_entry, latent.float(), base, residue=objective.metric(task_entry).surrogate)
        cached_tasks, _ = objective.score_latents(task_entry, cached.float(), base, residue=objective.metric(task_entry).surrogate)
        numbers = endpoint.numbers(tasks); assert numbers == endpoint.numbers(cached_tasks)
        picture = np.asarray(image).copy(); assert np.array_equal(picture, objective.pixels(cached))
        target = destination/'images'/(case+'_'+c+'.png'); target.parent.mkdir(exist_ok=True); image.save(target)
        effective, ideal = effective_difference(head, phi)
        alpha = pipe.scheduler.alphas_cumprod[1].to(args.device).float()
        original_epsilon = (record['raw_epsilon'].to(args.device).float()-(alpha/(1-alpha)).sqrt()*record['base_delta'].to(args.device)).half()
        assert tensor_sha(original_epsilon) == record['corrected_epsilon_sha256']
        correction = delta.cpu()-record['base_delta']
        epsilon_change = corrected.cpu().float()-original_epsilon.cpu().float()
        latent_change = latent.cpu().float()-record['final'].float()
        response = dict(phi=phi.cpu(), effective_weight_difference=effective.cpu(), ideal_modulated_increment=ideal.cpu(),
            output_delta=delta.cpu(), output_correction=correction, corrected_epsilon=corrected.cpu(),
            corrected_epsilon_change=epsilon_change, latent=latent.cpu(), latent_change=latent_change)
        responses[case+'/'+c] = response
        original = pixels(args.out/old_rows[case, entry['condition'], entry['arm']]['image'])
        row = dict(case=case, condition=entry['condition'], sample_condition=c, arm=entry['arm'], factor=entry['factor'],
            group=entry['group'], model_arm=args.arm, seed=entry['pair']['seed'], noise_digest=digest, b_calls=25,
            png_sha256=sha(target), image=target.relative_to(args.out).as_posix(), training_tasks=numbers,
            training_task_comparison=training_comparison(numbers, task_baselines[case, c]['tasks']),
            latent_sha256=tensor_sha(latent), full_cache_latent_equal=True, full_cache_tasks_equal=True,
            full_cache_png_equal=True, prefix_unchanged=True, actual_phi=float(phi),
            response_sha256={k: tensor_sha(v) for k, v in response.items()},
            effective_weight_difference=vector_summary(effective), ideal_modulated_increment=vector_summary(ideal),
            output_correction=vector_summary(correction), corrected_epsilon_change=vector_summary(epsilon_change),
            latent_change=vector_summary(latent_change), changed_pixel_channels=int(np.count_nonzero(picture != original)),
            **score_entry(entry, picture, 'replay'))
        rows.append(row); write(destination/'partial.json', dict(rows=rows, complete=False))
        print('REFERENCE_SINGLE_REPLAY', args.arm, len(rows), case, c, flush=True)
    assert len(rows) == 48 and len(responses) == 48
    assert weights_digest(head.base) == base_hash and tensor_sha(flatten(head.increment_parameters())) == head_hash
    assert weights_digest(net) == net_hash and not any(p.grad is not None for p in head.parameters())
    assert weights_digest(pipe.vae) == metadata['vae_hash_fp32']
    torch.save(dict(rows=responses, base_checkpoint_sha256=CHECKPOINT_SHA, head_sha256=head_hash,
        source_cache_sha256=overlay['original_cache_sha256']), destination/'responses.pt')
    failures = [dict(case=r['case'], condition=r['condition'], arm=r['arm'], failures=r['training_task_comparison']['failures'])
        for r in rows if r['training_task_comparison']['failures']]
    verify(args)
    write(destination/'result.json', dict(arm=args.arm, split='replay', rows=rows, **aggregate(rows, 'replay'),
        protocol_sha256=PROTOCOL_SHA, overlay_sha256=sha(args.out/'one_step_overlay.json'), source_sha256=overlay['source_sha256'],
        checkpoint_sha256=proposals['arms'][args.arm]['checkpoint_sha256'], base_checkpoint_sha256=CHECKPOINT_SHA,
        base_b_hash=base_hash, head_hash=head_hash, net_hash=net_hash, frozen_unchanged=True,
        evaluation_optimizer_updates=0, heldout_used_for_training=False, heldout_used_for_evaluation=False,
        adopted_checkpoints=[], complete_s0_images=48, new_reverse_passes=0, environment=environment(),
        response_tensor_sha256=sha(destination/'responses.pt'), scheduler_audits_sha256=sha(destination/'scheduler_audits.json'),
        seconds=time.monotonic()-started, scope='48 seen conditions; no holdout, adoption, or equal function-budget claim',
        training_task_protection=dict(passed=not failures, failures=failures, comparisons=336,
            tolerance='1e-5+1e-4*abs(initial)', baseline_gradient_json_sha256=gradient_hashes),
        strict_replay_protection=strict_replay_protection(rows, old_rows)))


def review(args):
    overlay = verify(args); assert not (args.out/'result.json').exists()
    proposals = read(args.out/'training/proposals.json')
    _, values, hashes, qualification = gradient_inputs(args, overlay)
    assert qualification == read(args.out/'gradient_qualification.json')
    result = dict(phase='reference4616 independent C0/C1 one-step replay', overlay=overlay, proposals=proposals,
        qualification=qualification, new_reverse_passes=352, adamw_steps=2, slsqp_calls=2, sequential_training_updates=0,
        complete_s0_images=0, holdout_results_read=0, adopted_checkpoints=[], f68_unchanged=True,
        functional_budget_matched_to_2308=False, preflight_cost_recounted=False,
        status='stopped_numerical_or_zero_step' if proposals['stop_before_replay'] else 'complete_replay')
    names = ['protocol.json', 'one_step_overlay.json', 'gradient_qualification.json', 'training/proposals.json']
    names += list(proposals['numerical_file_sha256'])
    names += ['gradients/shard%d.%s' % (i, ext) for i in range(4) for ext in ('json', 'pt', 'scheduler.json')]
    if not proposals['stop_before_replay']:
        audit, spec = Inputs(), read(args.out/'protocol.json')
        reports = {model: {'replay': audit.read(args.out/'evaluation'/model/'replay/result.json')} for model in ('f68',)+ARMS}
        baseline = reports['f68']['replay']; base_rows = {summary.row_key(r, 'replay'): r for r in baseline['rows']}
        assert baseline['checkpoint_sha256'] == CHECKPOINT_SHA
        assert all(values[r['case'], r['sample_condition']]['tasks'] == r['training_tasks'] for r in baseline['rows'])
        decisions = {}
        for arm in ARMS:
            replay, proposal = reports[arm]['replay'], proposals['arms'][arm]
            assert replay['overlay_sha256'] == sha(args.out/'one_step_overlay.json') and replay['source_sha256'] == overlay['source_sha256']
            assert replay['checkpoint_sha256'] == proposal['checkpoint_sha256'] and replay['head_hash'] == proposal['head_sha256']
            assert replay['base_b_hash'] == proposal['base_b_hash'] == baseline['b_hash'] and replay['net_hash'] == baseline['net_hash']
            assert replay['base_checkpoint_sha256'] == CHECKPOINT_SHA and replay['complete_s0_images'] == 48 and replay['new_reverse_passes'] == 0
            validate_report(replay, arm, spec, PROTOCOL_SHA, values, hashes, audit, args.out)
            assert same_numbers(replay['strict_replay_protection'], strict_replay_protection(replay['rows'], base_rows))
            assert all(all(r[k] for k in ('prefix_unchanged', 'full_cache_latent_equal', 'full_cache_tasks_equal', 'full_cache_png_equal')) for r in replay['rows'])
            folder = args.out/'evaluation'/arm/'replay'
            assert sha(folder/'responses.pt') == replay['response_tensor_sha256']
            assert sha(folder/'scheduler_audits.json') == replay['scheduler_audits_sha256']
            responses = torch.load(folder/'responses.pt', map_location='cpu', weights_only=False)
            assert responses['head_sha256'] == proposal['head_sha256'] and len(responses['rows']) == 48
            for row in replay['rows']:
                tensors = responses['rows'][row['case']+'/'+row['sample_condition']]
                assert {k: tensor_sha(v) for k, v in tensors.items()} == row['response_sha256']
            decisions[arm] = replay_eligibility(proposal, replay, baseline)
            names += ['training/'+arm+'.pt', 'evaluation/'+arm+'/replay/responses.pt', 'evaluation/'+arm+'/replay/scheduler_audits.json']
        innovation = summary.innovation(reports, {'C1': {'eligible': decisions['C1']['holdout_candidate']}})
        innovation['eligibility_scope'] = 'original replay only; no holdout or adoption'
        result.update(decisions=decisions, innovation=innovation, pastel=summary.pastel_rows(reports), complete_s0_images=96,
            comparisons={a+'_minus_'+b: dict(paired=summary.paired_changes(reports[b]['replay'], reports[a]['replay'], 'replay'),
                response=summary.response_changes(reports[b]['replay'], reports[a]['replay'])) for a, b in (('C0', 'f68'), ('C1', 'f68'), ('C1', 'C0'))},
            groups={k: r['replay']['groups'] for k, r in reports.items()}, review_input_sha256=audit.hashes)
        for model, report in reports.items():
            names += ['evaluation/'+model+'/replay/result.json']+[r['image'] for r in report['replay']['rows']]
        assert sum(name.endswith('.png') for name in names) == 144
    verify(args); write(args.out/'result.json', result); names.append('result.json')
    write(args.out/'manifest.json', {name: sha(args.out/name) for name in names})
    with zipfile.ZipFile(args.out/'review.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in names+['manifest.json']:
            archive.write(args.out/name, name)
    print('REFERENCE_SINGLE_REVIEW', result['status'], sha(args.out/'review.zip'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'gradient', 'propose', 'evaluate', 'review', 'verify'))
    parser.add_argument('--prior', type=Path, required=True); parser.add_argument('--out', type=Path)
    parser.add_argument('--project', type=Path, default=Path('/share/home/u2515283058/Mymodel'))
    parser.add_argument('--shard', type=int, choices=range(4), default=0)
    parser.add_argument('--arm', choices=ARMS, default='C0'); parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(); args.prior = args.prior.resolve(); args.project = args.project.resolve()
    args.fixed = args.prior/'autonomous_repair_20261116/scaled_projection_retry'
    args.preflight = args.prior/'autonomous_repair_20261116/terminal_head_preflight'
    args.basis = args.prior/'autonomous_repair_20261116/reference_terminal_preflight_v2'
    args.out = (args.out or args.prior/'autonomous_repair_20261116/reference_terminal_one_step').resolve()
    args.data = args.project.parent/'datasets/BF'
    args.checkpoint = args.project/'output_eval/e2_state_b_pastel_followup_20261115/trust_retry/training/step1.pt'
    torch.set_num_threads(2); cv2.setNumThreads(2)
    try:
        dict(prepare=prepare, gradient=gradient, propose=propose, evaluate=evaluate, review=review, verify=verify)[args.action](args)
    except Exception as error:
        if args.out.exists():
            name = '%s_%s.failure.json' % (args.action, args.shard if args.action == 'gradient' else args.arm)
            if not (args.out/name).exists():
                write(args.out/name, dict(error_type=type(error).__name__, message=str(error),
                    details=getattr(error, 'details', None), complete=False, adopted_checkpoints=[]))
        raise


if __name__ == '__main__':
    main()
