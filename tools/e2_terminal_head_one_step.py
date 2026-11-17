"""合格末步头从零出发的C0/C1各一步；原f68、S0和全部replay门槛保持。"""
import argparse
import copy
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np
from PIL import Image
import torch

from models.e2_state_texture_b import StateTextureB
from models.e2_terminal_output_head import TerminalOutputHead
from tools import e2_terminal_head_preflight as preflight
from tools.e2_b_endpoint_losses import TASKS, flatten, set_parameters
from tools.e2_b_pair_finetune_losses import projection, pair_coefficients, project_displacement
from tools.e2_state_b_scaled_retry import SENTINELS, scaled_tasks, scaled_scalar, replica_report, NumericalFailure
from tools.e2_state_b_pair_finetune import LEGACY_FOCUS
from tools.e2_state_b_amplitude_response import CHECKPOINT_SHA, read, write, sha, pixels
from tools.e2_state_b_backward_repro import tensor_sha, vector_summary
from tools.e2_state_b_pair_retry import PROTOCOL_SHA


SOURCE = Path(__file__).resolve().parents[1]
ARMS = ('C0', 'C1')
PARAMETERS = 2308
SWEEPS = 8192
TOLERANCE = 1e-6
EXTRA = ('tools/e2_terminal_head_one_step.py', 'scripts/e2_terminal_head_one_step.sbatch',
         'tools/e2_terminal_head_step_summary.py')


def previous_args(args):
    value = copy.copy(args)
    value.out = args.preflight
    return value


def device_tree(value, device):
    if torch.is_tensor(value):
        return value.detach().to(device)
    if isinstance(value, dict):
        return {k: device_tree(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(device_tree(v, device) for v in value)
    return value


def load_head(args, device):
    base = StateTextureB(color_channels=2, reference_channels=192).to(device)
    base.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=False)['b'])
    head = TerminalOutputHead(base).to(device).eval()
    assert sum(p.numel() for p in head.increment_parameters()) == PARAMETERS
    assert [id(p) for p in head.parameters() if p.requires_grad] == [id(p) for p in head.increment_parameters()]
    return head


def objective_from_cache(args, pipe, head, pair, cache):
    """恢复资格时的目标字节，不重新encode目标；评分实现仍是原Objective。"""
    from tools.e2_state_b_pair_finetune import Objective
    from tools.e2_b_garment_projection import GarmentProjection
    objective = Objective.__new__(Objective)
    objective.args, objective.pipe, objective.predictor, objective.b = args, pipe, None, head
    objective.records, objective.projections = {}, {}
    folder = args.out/'cases'/pair['case_id']
    mask = np.asarray(Image.open(folder/'mask.png')) > 0
    anchor = pixels(folder/'train_anchor.png')
    pipe.vae.float().eval().requires_grad_(False)
    for condition, record in cache['records'].items():
        target = device_tree(record['target_record'], args.device)
        objective.records[pair['case_id'], condition] = target
        positive = pixels(folder/(condition+'_positive.png'))
        restored = np.rint(target[1][0].permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)
        assert np.array_equal(positive, restored)
        objective.projections[pair['case_id'], condition] = GarmentProjection(positive, anchor, mask, args.device)
    return objective


def combine_gradients(shards, spec):
    cfg = spec['training']
    gradients = torch.cat([s['gradients'] for s in shards])
    items = [item for shard in shards for item in shard['items']]
    weights = torch.tensor(spec['replay_weights'])
    merged = (gradients.mean(0)*weights[:, None]).sum(0)
    old_abs = torch.stack([s['absolute_gradient'] for s in shards if s['case'] in LEGACY_FOCUS]).mean(0)
    new_abs = torch.stack([s['absolute_gradient'] for s in shards if s['case'] not in LEGACY_FOCUS]).mean(0)
    pair = torch.stack([s['pair_gradient'] for s in shards]).mean(0)
    normalized = lambda value: value*(merged.norm()/value.norm().clamp_min(1e-12))
    directions = dict(C0=merged+normalized(old_abs),
        C1=merged+normalized(old_abs)+cfg['pair_gradient_norm_fraction']*normalized(pair)
            +cfg['extra_absolute_gradient_norm_fraction']*normalized(new_abs))
    constraints = gradients.flatten(0, 1)
    margins = torch.full((len(constraints),), cfg['default_margin'])
    for index, item in enumerate(items):
        if [item['case'], item['condition']] == cfg['protected_condition']:
            margins[index*7+TASKS.index('amplitude')] = cfg['protected_amplitude_margin']
    return directions, constraints, margins, dict(original=float(merged.norm()), legacy_absolute=float(old_abs.norm()),
        new_absolute=float(new_abs.norm()), pair=float(pair.norm()))


def one_update(parameters, direction, constraints, margins, cfg):
    """新AdamW只调用一次；保持原clip/normscale/scale/cap，仅固定8192预算。"""
    from tools.e2_b_balanced_losses import assign, alignment
    start = flatten(parameters)
    assert not torch.count_nonzero(start)
    optimizer = torch.optim.AdamW(parameters, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    assign(parameters, direction)
    norm = torch.nn.utils.clip_grad_norm_(parameters, cfg['gradient_clip'])
    assert torch.isfinite(norm)
    optimizer.step()
    raw = start-flatten(parameters)
    delta, projection_report = project_displacement(raw, constraints, margins, max_sweeps=SWEEPS, tolerance=TOLERANCE)
    norm_scale = min(1., float(raw.norm()/delta.norm().clamp_min(1e-18)))
    delta = delta*norm_scale*cfg['scale']
    cap_scale = min(1., cfg['displacement_cap']/float(delta.norm().clamp_min(1e-18)))
    delta = delta*cap_scale
    set_parameters(parameters, start-delta)
    actual = start-flatten(parameters)
    changed = bool(torch.count_nonzero(actual))
    return dict(optimizer_proposals=1, parameter_updates=int(changed), raw_norm=float(raw.norm()),
        actual_norm=float(actual.norm()), raw_sha256=tensor_sha(raw), actual_descent_sha256=tensor_sha(actual),
        projection=projection_report, gradient_norm=float(direction.norm()), gradient_clip_return=float(norm),
        norm_scale=norm_scale, fixed_scale=cfg['scale'], cap_scale=cap_scale,
        minimum_cosine=min(alignment(constraints, actual)) if changed else None,
        configured_projection_sweeps=cfg['projection_sweeps'], effective_projection_sweeps=SWEEPS)


def effective_difference(head):
    layer = head.base.output
    original = torch.cat([layer.weight.detach().flatten(), layer.bias.detach().flatten()]).double()
    merged = torch.cat([(layer.weight.detach()+head.delta_weight.detach()).flatten(),
                        (layer.bias.detach()+head.delta_bias.detach()).flatten()]).double()
    return merged-original


def prepare(args):
    preflight.verify(previous_args(args))
    qualification = read(args.preflight/'qualification.json')
    assert qualification['passed'] and qualification['maximum_relative_l2'] <= .005 and qualification['minimum_cosine'] >= .99999
    assert not (args.out/'one_step_overlay.json').exists(), '不覆盖已有单步实验'
    spec = read(args.preflight/'protocol.json')
    assert sha(args.preflight/'protocol.json') == PROTOCOL_SHA
    cfg = spec['training']
    assert (cfg['lr'], cfg['weight_decay'], cfg['gradient_clip'], cfg['scale'], cfg['displacement_cap']) == (3e-5, .01, 1., .5, .007854645)
    assert (cfg['projection_sweeps'], cfg['default_margin'], cfg['protected_amplitude_margin']) == (4096, .01, .18)
    assert cfg['pair_gradient_norm_fraction'] == cfg['extra_absolute_gradient_norm_fraction'] == .25
    args.out.mkdir(parents=True, exist_ok=True)
    inputs, copied = {}, {}
    def record(path, expected=None):
        path = Path(path).resolve()
        assert 'holdout' not in path.parts
        value = sha(path)
        if expected is not None:
            assert value == expected
        inputs[path.as_posix()] = value
        return value
    def duplicate(source, relative, expected=None):
        digest = record(source, expected)
        dest = args.out/relative
        assert not dest.exists()
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        assert sha(dest) == digest
        copied[relative] = digest
    duplicate(args.preflight/'protocol.json', 'protocol.json', PROTOCOL_SHA)
    for name, digest in spec['assets'].items():
        duplicate(args.preflight/name, name, digest)
    duplicate(args.preflight/'historical_f68.json', 'evaluation/f68/replay/result.json')
    for row in read(args.preflight/'historical_f68.json')['rows']:
        duplicate(args.preflight/row['image'], row['image'], row['png_sha256'])
    for name in ('preflight_protocol.json', 'qualification.json'):
        record(args.preflight/name)
    for index in range(4):
        name = 'cache/shard%d.pt' % index
        record(args.preflight/name, qualification['cache_sha256'][name])
        record(args.preflight/('cache/shard%d.json' % index))
    record(args.checkpoint, CHECKPOINT_SHA)
    sources = set(read(args.preflight/'preflight_protocol.json')['source_sha256']) | set(EXTRA)
    write(args.out/'one_step_overlay.json', dict(parameterization='terminal output weight/bias increments',
        trainable_parameters=PARAMETERS, base_checkpoint_sha256=CHECKPOINT_SHA, head_start='zero',
        original_protocol_sha256=PROTOCOL_SHA, preflight_run=str(args.preflight), preflight_job='125453',
        preflight_cache_sha256=qualification['cache_sha256'], source_qualification_sha256=sha(args.preflight/'qualification.json'),
        configured_projection_sweeps=4096, effective_projection_sweeps=SWEEPS, projection_tolerance=TOLERANCE,
        original_losses_and_gates_unchanged=True, new_task_reverse_passes=336, new_y_reverse_passes=12, new_absolute_reverse_passes=4,
        gradient_loss_scale=1024., input_gradient_space=PARAMETERS, old_full_B_gradient_directions_used=False,
        independent_optimizer_proposals_per_arm=1, original_step_and_cap=cfg,
        plan='fresh gradients -> qualify -> two independent single proposals -> nonzero candidates complete original-S0 replay; no holdout or adoption',
        input_sha256=inputs, copied_sha256=copied, source_sha256={name: sha(SOURCE/name) for name in sorted(sources)},
        execution_commit=subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()))
    verify(args)
    print('TERMINAL_ONE_STEP_PREPARED', flush=True)


def verify(args):
    preflight.verify(previous_args(args))
    overlay = read(args.out/'one_step_overlay.json')
    assert overlay['trainable_parameters'] == PARAMETERS and overlay['effective_projection_sweeps'] == SWEEPS
    assert sha(args.out/'protocol.json') == PROTOCOL_SHA
    assert all(sha(Path(path)) == value for path, value in overlay['input_sha256'].items())
    assert all(sha(args.out/path) == value for path, value in overlay['copied_sha256'].items())
    assert all(sha(SOURCE/path) == value for path, value in overlay['source_sha256'].items())
    return overlay


def load_cache(args, shard, overlay):
    path = args.preflight/('cache/shard%d.pt' % shard)
    assert sha(path) == overlay['preflight_cache_sha256']['cache/shard%d.pt' % shard]
    cache = torch.load(path, map_location='cpu', weights_only=False)
    assert cache['parameter_count'] == PARAMETERS and cache['zero_head']
    assert cache['preflight_protocol_sha256'] == sha(args.preflight/'preflight_protocol.json')
    return cache


def gradient(args):
    from tools import e2_state_b_pilot as pilot, e2_state_b_endpoint as endpoint
    from tools.e2_oracle_texture import weights_digest
    from tools.e2_fft_experiment import environment
    started = time.monotonic()
    overlay = verify(args)
    destination = args.out/'gradients'/('shard%d.pt' % args.shard)
    assert not destination.exists()
    destination.parent.mkdir(exist_ok=True)
    cache = load_cache(args, args.shard, overlay)
    spec = read(args.out/'protocol.json')
    pair = spec['train'][args.shard]
    items = [r for r in spec['inputs'] if r['case'] == pair['case_id']]
    runtime = copy.copy(args)
    runtime.action = 'train'
    _, pipe, net = pilot.load_core(runtime)
    torch.set_num_threads(2)
    net.eval().requires_grad_(False)
    head = load_head(args, args.device)
    parameters = head.increment_parameters()
    initial, frozen = weights_digest(head.base), weights_digest(net)
    assert initial == cache['b_hash'] and frozen == cache['net_hash']
    objective = objective_from_cache(runtime, pipe, head, pair, cache)
    vae_hash = weights_digest(pipe.vae)
    pipe.scheduler.set_timesteps(50, device=args.device)
    gradients, values, late, replicas = [], [], {}, []
    expected_metadata = read(args.preflight/('cache/shard%d.json' % args.shard))
    for item in items:
        condition = item['sample_condition']
        record = cache['records'][condition]
        entry = dict(pair=pair, condition=condition)
        latent, base, delta, corrected = preflight.terminal_replay(head, record, pipe.scheduler, args.device)
        assert tensor_sha(latent) == tensor_sha(record['final']) and tensor_sha(delta) == tensor_sha(record['base_delta'])
        tasks, _ = objective.score_latents(entry, latent.float(), base, residue=objective.metric(entry).surrogate)
        numeric = endpoint.numbers(tasks)
        assert numeric == record['tasks'], 'fresh zero-head任务未与原f68逐位相同'
        current = scaled_tasks(tasks, parameters).detach().cpu()
        if not torch.isfinite(current).all():
            raise NumericalFailure(condition, current)
        assert current.shape == (7, PARAMETERS)
        gradients.append(current)
        values.append(dict(item, tasks=numeric, noise_digest=record['noise_digest'], zero_latent_equal=True))
        if preflight.key(item) in SENTINELS:
            reference = cache['local_head_gradients'][condition]
            comparison = replica_report(current, reference[0])
            replicas.append(dict(item, **comparison, reference='preflight independent graph0',
                reference_tensor_sha256=tensor_sha(reference[0]), fresh_tensor_sha256=tensor_sha(current),
                additional_reference_graph1=replica_report(current, reference[1])))
        del latent, base, delta, corrected, tasks
        if item['group'] == 'pastel':
            latent, base, delta, corrected = preflight.terminal_replay(head, record, pipe.scheduler, args.device)
            _, positive, _, _, weight, _ = objective.records[pair['case_id'], condition]
            positive_base = objective.records[pair['case_id'], item['condition']+'_a100'][1]
            feature = projection(objective.rgb(latent.float()), positive, positive_base, weight)
            raw_features = {name: float(value.detach()) for name, value in feature.items()}
            assert raw_features == expected_metadata['late_features'][item['arm']]
            gy = scaled_scalar(feature['y'], parameters, retain_graph=item['arm'] == 'a100').cpu()
            row = dict(features=raw_features, gradient_y=gy)
            if item['arm'] == 'a100':
                row['gradient_absolute'] = scaled_scalar((feature['g']-1).square(), parameters).cpu()
            assert all(torch.isfinite(v).all() for k, v in row.items() if k.startswith('gradient'))
            late[item['arm']] = row
            del latent, base, delta, corrected, feature
        print('TERMINAL_FRESH_G', args.shard, condition, flush=True)
    coefficients = pair_coefficients(*(torch.tensor(late[arm]['features'][name]) for arm, name in
        (('a075', 'y'), ('a125', 'y'), ('a075', 'x'), ('a125', 'x'))))
    pair_gradient = coefficients['low']*late['a075']['gradient_y']+coefficients['high']*late['a125']['gradient_y']
    assert float(coefficients['pair']) == expected_metadata['pair_loss'] and torch.isfinite(pair_gradient).all()
    assert len(replicas) == 2 and len(gradients) == 12
    assert weights_digest(head.base) == initial and weights_digest(net) == frozen and weights_digest(pipe.vae) == vae_hash
    assert not torch.count_nonzero(flatten(parameters)) and not any(p.grad is not None for p in head.parameters())
    verify(args)
    payload = dict(shard=args.shard, case=pair['case_id'], items=items, values=values, gradients=torch.stack(gradients),
        absolute_gradient=late['a100']['gradient_absolute'], pair_gradient=pair_gradient,
        late_features={arm: row['features'] for arm, row in late.items()}, pair_loss=float(coefficients['pair']),
        replicas=replicas, all_finite=True,
        numerical_qualification=all(r['passed'] and r['additional_reference_graph1']['passed'] for r in replicas),
        base_b_hash=initial, head_hash=tensor_sha(flatten(parameters)), net_hash=frozen, vae_hash_fp32=vae_hash,
        protocol_sha256=PROTOCOL_SHA, overlay_sha256=sha(args.out/'one_step_overlay.json'),
        source_sha256=overlay['source_sha256'], parameter_count=PARAMETERS, loss_scale=1024., gradients_unscaled=True,
        preflight_cache_sha256=overlay['preflight_cache_sha256']['cache/shard%d.pt' % args.shard],
        reverse_passes=dict(tasks=84, y=3, absolute=1), fresh_gradients=True, optimizer_updates=0,
        frozen_unchanged=True, environment=environment(), seconds=time.monotonic()-started)
    torch.save(payload, destination)
    metadata = {k: v for k, v in payload.items() if k not in ('gradients', 'absolute_gradient', 'pair_gradient')}
    metadata.update(tensor_file=destination.relative_to(args.out).as_posix(), tensor_sha256=sha(destination))
    write(destination.with_suffix('.json'), metadata)
    print('TERMINAL_FRESH_G_COMPLETE', args.shard, payload['numerical_qualification'], flush=True)
    if not payload['numerical_qualification']:
        raise SystemExit(2)


def gradient_inputs(args, overlay, tensors=False):
    """只读本轮新2308维梯度；旧全B梯度不进入此接口。"""
    shards, values, hashes, tensor_hashes, replicas = [], {}, {}, {}, []
    spec = read(args.out/'protocol.json')
    for index in range(4):
        path = args.out/'gradients'/('shard%d.json' % index)
        meta = read(path)
        pt = path.with_suffix('.pt')
        assert sha(pt) == meta['tensor_sha256']
        assert meta['shard'] == index and meta['case'] == spec['train'][index]['case_id']
        assert meta['parameter_count'] == PARAMETERS and meta['fresh_gradients'] and meta['gradients_unscaled']
        assert meta['all_finite'] and meta['frozen_unchanged'] and meta['optimizer_updates'] == 0
        assert meta['protocol_sha256'] == PROTOCOL_SHA and meta['overlay_sha256'] == sha(args.out/'one_step_overlay.json')
        assert meta['source_sha256'] == overlay['source_sha256'] and meta['loss_scale'] == 1024.
        assert meta['reverse_passes'] == dict(tasks=84, y=3, absolute=1)
        assert meta['preflight_cache_sha256'] == overlay['preflight_cache_sha256']['cache/shard%d.pt' % index]
        assert len(meta['items']) == len(meta['values']) == 12
        for row in meta['values']:
            key = row['case'], row['sample_condition']
            assert key not in values and set(row['tasks']) == set(TASKS)
            values[key] = row
        hashes[path.name], tensor_hashes[pt.name] = sha(path), sha(pt)
        replicas.extend(meta['replicas'])
        if tensors:
            payload = torch.load(pt, map_location='cpu', weights_only=False)
            for field in ('items', 'values', 'case', 'parameter_count', 'overlay_sha256'):
                assert payload[field] == meta[field]
            assert payload['gradients'].shape == (12, 7, PARAMETERS)
            for name in ('gradients', 'absolute_gradient', 'pair_gradient'):
                assert torch.isfinite(payload[name]).all()
            assert payload['absolute_gradient'].shape == payload['pair_gradient'].shape == (PARAMETERS,)
            shards.append(payload)
        else:
            shards.append(meta)
    assert len(values) == 48 and len(replicas) == 8
    assert {preflight.key(row) for row in replicas} == set(SENTINELS)
    qualification = dict(passed=all(row['passed'] and row['additional_reference_graph1']['passed'] for row in replicas), all_finite=True,
        replicas=replicas, maximum_relative_l2=max(row['max_relative_l2'] for row in replicas),
        minimum_cosine=min(row['minimum_cosine'] for row in replicas), parameter_count=PARAMETERS,
        new_gradient_sha256=tensor_hashes, gradient_json_sha256=hashes,
        new_task_reverse_passes=336, new_y_reverse_passes=12, new_absolute_reverse_passes=4,
        loss_scale=1024., optimizer_updates=0, old_full_B_gradient_directions_used=False,
        reference='fresh versus qualified preflight graph0; graph1 comparison also reported',
        overlay_sha256=sha(args.out/'one_step_overlay.json'))
    return shards, values, hashes, qualification


def load_candidate(args, arm, device, proposal=None):
    from tools.e2_oracle_texture import weights_digest
    path = args.out/'training'/(arm+'.pt')
    if proposal is None:
        proposal = read(args.out/'training/proposals.json')['arms'][arm]
    assert sha(path) == proposal['checkpoint_sha256']
    payload = torch.load(path, map_location='cpu', weights_only=False)
    assert set(payload) == {'delta_weight', 'delta_bias', 'report', 'base_checkpoint_sha256'}
    assert payload['base_checkpoint_sha256'] == CHECKPOINT_SHA
    head = load_head(args, device)
    with torch.no_grad():
        head.delta_weight.copy_(payload['delta_weight'])
        head.delta_bias.copy_(payload['delta_bias'])
    assert weights_digest(head.base) == proposal['base_b_hash']
    assert tensor_sha(flatten(head.increment_parameters())) == proposal['head_sha256']
    assert tensor_sha(effective_difference(head)) == proposal['effective_weight_difference_sha256']
    assert payload['report'] == {k: v for k, v in proposal.items() if k != 'checkpoint_sha256'}
    return head


def propose(args):
    from tools.e2_oracle_texture import weights_digest
    started = time.monotonic()
    overlay = verify(args)
    destination = args.out/'training'
    assert not destination.exists(), '不覆盖已存在的独立候选'
    assert not (args.out/'gradient_qualification.json').exists()
    shards, values, hashes, qualification = gradient_inputs(args, overlay, tensors=True)
    write(args.out/'gradient_qualification.json', qualification)
    if not qualification['passed']:
        raise SystemExit(2)
    spec = read(args.out/'protocol.json')
    directions, constraints, margins, component_norms = combine_gradients(shards, spec)
    assert constraints.shape == (336, PARAMETERS) and all(torch.isfinite(v).all() for v in directions.values())
    destination.mkdir()
    arms = {}
    for arm in ARMS:
        head = load_head(args, 'cpu')
        base_hash = weights_digest(head.base)
        initial_head = tensor_sha(flatten(head.increment_parameters()))
        report = one_update(head.increment_parameters(), directions[arm], constraints, margins, spec['training'])
        actual_head = flatten(head.increment_parameters())
        effective = effective_difference(head)
        assert weights_digest(head.base) == base_hash and torch.isfinite(actual_head).all()
        report.update(arm=arm, protocol_sha256=PROTOCOL_SHA, overlay_sha256=sha(args.out/'one_step_overlay.json'),
            gradient_qualification_sha256=sha(args.out/'gradient_qualification.json'),
            base_checkpoint_sha256=CHECKPOINT_SHA, base_b_hash=base_hash, initial_head_sha256=initial_head,
            head_sha256=tensor_sha(actual_head), trainable_parameters=PARAMETERS,
            effective_weight_difference_sha256=tensor_sha(effective), effective_weight_difference=vector_summary(effective),
            effective_weight_difference_definition='FP64 difference of FP32 (baseW+deltaW,baseb+deltab) and original base',
            frozen_unchanged=True, heldout_used_for_training=False, adopted=False,
            acceptance='unassessed candidate; original complete S0 replay required',
            numerical_qualification_scope='2308-dimensional local head only; no transferred old full-B geometry guarantee')
        path = destination/(arm+'.pt')
        torch.save(dict(delta_weight=head.delta_weight.detach().cpu(), delta_bias=head.delta_bias.detach().cpu(),
            report=report, base_checkpoint_sha256=CHECKPOINT_SHA), path)
        arms[arm] = dict(report, checkpoint_sha256=sha(path))
        load_candidate(args, arm, 'cpu', arms[arm])
        print('TERMINAL_ONE_STEP_PROPOSAL', arm, report['parameter_updates'], report['projection']['feasible'], flush=True)
    verify(args)
    result = dict(arms=arms, conditions=48, parameter_count=PARAMETERS, full_task_reverse_passes=336,
        late_y_reverse_passes=12, late_abs_reverse_passes=4, new_reverse_passes=352,
        independent_adamw_proposals=2, sequential_training_updates=0,
        frozen_unchanged=True, heldout_used_for_training=False, adopted_checkpoints=[],
        component_norms=component_norms, constraint_sha256=tensor_sha(constraints), margins=margins.tolist(),
        new_gradient_sha256=qualification['new_gradient_sha256'], gradient_json_sha256=hashes,
        gradient_seconds=sum(s['seconds'] for s in shards), proposal_seconds=time.monotonic()-started,
        preflight_cost_recounted=False, overlay_sha256=sha(args.out/'one_step_overlay.json'),
        stop_before_replay=not any(r['parameter_updates'] for r in arms.values()))
    write(destination/'proposals.json', result)
    if result['stop_before_replay']:
        print('TERMINAL_ONE_STEP_STOP_BOTH_ZERO', flush=True)
        raise SystemExit(2)


@torch.no_grad()
def evaluate(args):
    from tools import e2_state_b_pilot as pilot, e2_state_b_endpoint as endpoint
    from tools.e2_oracle_texture import weights_digest
    from tools.e2_fft_experiment import environment
    from tools.e2_state_b_pair_evaluation import replay_inputs, score_entry, aggregate, strict_replay_protection, training_comparison
    started = time.monotonic()
    overlay = verify(args)
    spec = read(args.out/'protocol.json')
    _, task_baselines, gradient_hashes, qualification = gradient_inputs(args, overlay)
    assert qualification == read(args.out/'gradient_qualification.json') and qualification['passed']
    proposals = read(args.out/'training/proposals.json')
    assert not proposals['stop_before_replay']
    destination = args.out/'evaluation'/args.arm/'replay'
    assert not destination.exists(), '不覆盖已有完整或部分评测'
    destination.mkdir(parents=True)
    entries, _, _ = replay_inputs(args, spec)
    baseline = read(args.out/'evaluation/f68/replay/result.json')
    old_rows = {(r['case'], r['condition'], r['arm']): r for r in baseline['rows']}
    runtime = copy.copy(args)
    runtime.action = 'train'
    core, pipe, net = pilot.load_core(runtime)
    torch.set_num_threads(2)
    net.eval().requires_grad_(False)
    head = load_candidate(args, args.arm, args.device).eval().requires_grad_(False)
    base_hash, head_hash, net_hash = weights_digest(head.base), tensor_sha(flatten(head.increment_parameters())), weights_digest(net)
    assert net_hash == baseline['net_hash']
    rows, objective_case, objective, cache = [], None, None, None
    for entry in entries:
        case, condition = entry['case'], entry['sample_condition']
        if case != objective_case:
            index = next(i for i, pair in enumerate(spec['train']) if pair['case_id'] == case)
            cache = load_cache(args, index, overlay)
            assert cache['b_hash'] == base_hash and cache['net_hash'] == net_hash
            objective = objective_from_cache(runtime, pipe, head, entry['pair'], cache)
            objective_case = case
        head.reset_capture()
        trace = []
        def capture(module, positional, keywords, output):
            trace.append(dict(timestep=int(keywords['timestep']), sha256=tensor_sha(output)))
        with head.register_forward_hook(capture, with_kwargs=True):
            image, digest, tap, latent = endpoint.final_sample(runtime, core, pipe, net, entry['pair'], condition, head)
        record = cache['records'][condition]
        assert tap.index == 50 and len(tap.states) == head.active_calls == len(trace) == 25
        assert digest == entry['expected_digest'] == record['noise_digest']
        prefix = [dict(index=s['index'], timestep=s['t'], zt=tensor_sha(s['zt']), raw_epsilon=tensor_sha(s['epsilon'])) for s in tap.states]
        assert prefix == record['prefix_states'] and trace[:24] == record['B_trace'][:24]
        assert tensor_sha(head.last_record['feature']) == tensor_sha(record['feature'])
        assert tensor_sha(head.last_record['base_delta']) == tensor_sha(record['base_delta'])
        cached, base, delta, corrected = preflight.terminal_replay(head, record, pipe.scheduler, args.device)
        assert tensor_sha(cached) == tensor_sha(latent) and tensor_sha(delta) == tensor_sha(head.last_record['full_delta'])
        task_entry = dict(pair=entry['pair'], condition=condition)
        tasks, _ = objective.score_latents(task_entry, latent.float(), base, residue=objective.metric(task_entry).surrogate)
        numbers = endpoint.numbers(tasks)
        cached_tasks, _ = objective.score_latents(task_entry, cached.float(), base, residue=objective.metric(task_entry).surrogate)
        assert numbers == endpoint.numbers(cached_tasks)
        picture = np.asarray(image).copy()
        assert np.array_equal(picture, objective.pixels(cached))
        target = destination/'images'/(case+'_'+condition+'.png')
        target.parent.mkdir(exist_ok=True)
        assert not target.exists()
        image.save(target)
        comparison = training_comparison(numbers, task_baselines[case, condition]['tasks'])
        original = pixels(args.out/old_rows[case, entry['condition'], entry['arm']]['image'])
        row = dict(case=case, condition=entry['condition'], sample_condition=condition, arm=entry['arm'], factor=entry['factor'],
            group=entry['group'], model_arm=args.arm, seed=entry['pair']['seed'], noise_digest=digest, b_calls=25,
            png_sha256=sha(target), image=target.relative_to(args.out).as_posix(), training_tasks=numbers,
            training_task_comparison=comparison, latent_sha256=tensor_sha(latent), full_cache_latent_equal=True,
            full_cache_tasks_equal=True, full_cache_png_equal=True, prefix_unchanged=True,
            output_correction=vector_summary(delta.cpu()-record['base_delta']),
            changed_pixel_channels=int(np.count_nonzero(picture != original)),
            **score_entry(entry, picture, 'replay'))
        rows.append(row)
        write(destination/'partial.json', dict(rows=rows, complete=False))
        print('TERMINAL_ONE_STEP_REPLAY', args.arm, len(rows), case, condition, flush=True)
    assert len(rows) == 48
    assert weights_digest(head.base) == base_hash and tensor_sha(flatten(head.increment_parameters())) == head_hash
    assert weights_digest(net) == net_hash and not any(p.grad is not None for p in head.parameters())
    verify(args)
    failures = [dict(case=r['case'], condition=r['condition'], arm=r['arm'], failures=r['training_task_comparison']['failures'])
        for r in rows if r['training_task_comparison']['failures']]
    result = dict(arm=args.arm, split='replay', rows=rows, **aggregate(rows, 'replay'),
        protocol_sha256=PROTOCOL_SHA, overlay_sha256=sha(args.out/'one_step_overlay.json'), source_sha256=overlay['source_sha256'],
        checkpoint_sha256=proposals['arms'][args.arm]['checkpoint_sha256'], base_checkpoint_sha256=CHECKPOINT_SHA,
        base_b_hash=base_hash, head_hash=head_hash, net_hash=net_hash, frozen_unchanged=True,
        evaluation_optimizer_updates=0, heldout_used_for_training=False, heldout_used_for_evaluation=False,
        adopted_checkpoints=[], complete_s0_images=48, new_reverse_passes=0, environment=environment(),
        seconds=time.monotonic()-started, scope='48 seen replay conditions only; no adoption or holdout',
        training_task_protection=dict(passed=not failures, failures=failures, comparisons=336,
            tolerance='1e-5+1e-4*abs(initial)', baseline_gradient_json_sha256=gradient_hashes),
        strict_replay_protection=strict_replay_protection(rows, old_rows))
    write(destination/'result.json', result)


def review(args):
    from tools.e2_terminal_head_step_summary import review as summarize
    verify(args)
    return summarize(args.out, args.prior)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'gradient', 'propose', 'evaluate', 'review', 'verify'))
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--project', type=Path, default=Path('/share/home/u2515283058/Mymodel'))
    parser.add_argument('--shard', type=int, choices=range(4), default=0)
    parser.add_argument('--arm', choices=ARMS, default='C0')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.prior, args.project = args.prior.resolve(), args.project.resolve()
    args.fixed = args.prior/'autonomous_repair_20261116/scaled_projection_retry'
    args.preflight = args.prior/'autonomous_repair_20261116/terminal_head_preflight'
    args.out = (args.out or args.prior/'autonomous_repair_20261116/terminal_head_one_step').resolve()
    args.data = args.project.parent/'datasets/BF'
    args.checkpoint = args.project/'output_eval/e2_state_b_pastel_followup_20261115/trust_retry/training/step1.pt'
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
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
