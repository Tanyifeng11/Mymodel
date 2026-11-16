"""固定五探针的S0方向差分与解码桥接；只求六项终latent头梯度，不训练。"""
import argparse
import copy
import math
import os
from pathlib import Path
import shutil
import socket
import statistics
import subprocess
import time
import zipfile

import cv2
import numpy as np
from PIL import Image
import torch

from tools.e2_b_endpoint_losses import TASKS, flatten, set_parameters
from tools.e2_b_direction_diagnostics import interpolate, FrozenPattern
from tools.e2_state_b_amplitude_response import CHECKPOINT_SHA, read, write, sha
from tools.e2_state_b_backward_repro import tensor_sha, compare_tensor
from tools.e2_state_b_pair_retry import PROTOCOL_SHA


SOURCE = Path(__file__).resolve().parents[1]
PROBES = (
    ('validation_870', 'r1033_native', 'a075', 'amplitude task failure'),
    ('validation_1641', 'r2692_native', 'a075', 'correlation task failure'),
    ('validation_1526', 'r752_native', 'a100', 'light task failure'),
    ('validation_1641', 'r5121_native', 'a125', 'PNG residue and training surrogate split'),
    ('validation_1641', 'r5121_native', 'a100', 'predeclared normal control'),
)
DIRECTIONS = ('C0', 'C1')
VARIANTS = (.25, .125, -.125, .03125, -.03125)
STEPS = (.125, .03125)
HEAD_TASKS = tuple(task for task in TASKS if task != 'pattern')
SCALE = 1024.
STATIONS = ('fp32_continuous_analytic', 'fp16_continuous_analytic',
            'uint8_analytic', 'uint8_opencv_exact')
EXTRA_SOURCES = (
    'tools/e2_state_b_scaled_direction_bridge.py', 'scripts/e2_state_b_scaled_direction_bridge.sbatch',
    'tools/e2_b_direction_diagnostics.py', 'tools/e2_b_finite_step_audit.py',
    'tools/e2_b_chain_repro.py', 'tools/e2_state_b_backward_repro.py',
    'tools/e2_state_b_pair_finetune.py', 'tools/e2_state_b_pair_evaluation.py',
    'tools/e2_state_b_endpoint.py', 'tools/e2_state_b_pilot.py',
    'tools/e2_state_b_material.py', 'tools/e2_state_b_garment.py',
    'tools/e2_state_b_balanced.py', 'tools/e2_state_b_joint.py',
    'tools/e2_state_b_joint_data.py', 'tools/e2_state_b_material_data.py',
    'tools/e2_state_b_repair.py', 'tools/e2_b_garment_projection.py',
    'tools/e2_b_balanced_losses.py', 'tools/e2_b_joint_losses.py', 'tools/e2_b_material_losses.py',
    'tools/e2_oracle_texture.py', 'tools/e2_fft_experiment.py', 'tools/bf_t3_frozen_core.py',
    'tools/m1_evaluation.py', 'models/bf_sd_skip.py', 'models/e2_state_texture_b.py',
    'models/e2_lab_anchor.py', 'adapter/attention_processor.py', 'pipelines/IMAGGarment_pipeline.py')


def key(row):
    return row['case'], row['condition'], row['arm']


def label(direction, scale, role):
    return '%s_%s_%s' % (direction, format(scale, 'g').replace('-', 'm').replace('.', 'p'), role)


def scalar_difference(a, b):
    denominator = max(abs(a), abs(b), 1e-12)
    return dict(difference=a-b, absolute_difference=abs(a-b),
        symmetric_relative_difference=abs(a-b)/denominator, relative_denominator=denominator,
        relative_floor_used=max(abs(a), abs(b)) < 1e-12,
        same_strict_sign=(a > 0 and b > 0) or (a < 0 and b < 0), both_zero=a == b == 0.)


def three_level_fd(plus, minus, h, q, gradient, z_plus, z_minus, z0, theta_plus, theta_minus):
    """实际FP64端点弦；A为头部值FD，B为q·latent弦，C为缓存G·参数弦。"""
    dz = z_plus.detach().double() - z_minus.detach().double()
    dt = theta_plus.detach().double() - theta_minus.detach().double()
    midpoint = (z_plus.double() + z_minus.double()) / 2 - z0.double()
    a = (float(plus)-float(minus)) / (2*h)
    b = float(q.double().reshape(-1) @ dz.reshape(-1)) / (2*h)
    c = float(gradient.double().reshape(-1) @ dt.reshape(-1)) / (2*h)
    assert all(math.isfinite(x) for x in (a, b, c))
    return dict(A_head_value_fd=a, B_head_q_latent_chord=b, C_cached_G_parameter_chord=c,
        head_difference=scalar_difference(a, b), chain_difference=scalar_difference(b, c),
        total_difference=scalar_difference(a, c), task_signal_abs=abs(float(plus)-float(minus)),
        latent_chord_sha256=tensor_sha(dz), latent_chord_norm_fp64=float(dz.norm()),
        latent_chord_nonzero=int(torch.count_nonzero(dz)), latent_midpoint_offset_norm_fp64=float(midpoint.norm()),
        parameter_chord_sha256=tensor_sha(dt), parameter_chord_norm_fp64=float(dt.norm()),
        parameter_chord_nonzero=int(torch.count_nonzero(dt)))


def head_derivatives(objective, entry, latent, base):
    """唯一反传入口：独立FP32终latent叶子；原损失乘1024后梯度除回。"""
    leaf = latent.detach().float().clone().requires_grad_(True)
    tasks, _ = objective.score_latents(entry, leaf, base.detach(), residue=objective.metric(entry).surrogate)
    values = {task: float(tasks[task].detach()) for task in TASKS}
    result = []
    for index, task in enumerate(HEAD_TASKS):
        q = torch.autograd.grad(tasks[task] * SCALE, leaf, retain_graph=index+1 < len(HEAD_TASKS))[0]
        q = (q.detach() / SCALE).cpu()
        assert torch.isfinite(q).all(), 'nonfinite head q: ' + task
        result.append(q)
    return torch.stack(result), values


@torch.no_grad()
def half_rgb(vae, latent):
    """严格保留原pixels的half解码、half归一化/clamp，再转float；无活图时调用。"""
    previous = vae.dtype
    vae.half()
    try:
        decoded = vae.decode(latent.half() / vae.config.scaling_factor, return_dict=False)[0]
        return (decoded / 2 + .5).clamp(0, 1).float()
    finally:
        vae.to(dtype=previous)


@torch.no_grad()
def residue_bridge(metric, rgb32, rgb16, expected_png):
    from models.e2_lab_anchor import rgb_to_lab
    from tools.e2_b_garment_projection import lab, highpass
    assert metric.energy >= .25, '预设探针需有效旧纹样模板'
    rgb8 = np.rint(rgb16[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    assert np.array_equal(rgb8, expected_png), 'half连续RGB量化后与原S0 PNG不同'
    quantized = torch.from_numpy(rgb8.copy()).permute(2, 0, 1)[None].to(rgb32.device).float()/255
    stations = {}
    for name, value in zip(STATIONS[:3], (rgb32, rgb16, quantized)):
        signed = (rgb_to_lab(value)[0] * metric.coefficients).sum() - metric.offset
        original = float(metric.surrogate(value))
        assert abs(float(signed)) == original
        stations[name] = dict(signed=float(signed), absolute=original)
    signed = float(np.mean((highpass(lab(rgb8))[metric.inside]-metric.positive)*metric.template)/metric.energy)
    exact = metric.exact(rgb8)
    assert abs(abs(signed)-exact) <= 1e-12
    stations[STATIONS[-1]] = dict(signed=signed, absolute=exact)
    return dict(stations=stations, adjacent={right+'_minus_'+left: {
        field: stations[right][field]-stations[left][field] for field in ('signed', 'absolute')}
        for left, right in zip(STATIONS[:-1], STATIONS[1:])},
        rgb32_to_rgb16=dict(max_abs=float((rgb16-rgb32).abs().max()), mean_abs=float((rgb16-rgb32).abs().mean())),
        rgb16_to_uint8=dict(max_abs=float((quantized-rgb16).abs().max()), mean_abs=float((quantized-rgb16).abs().mean())),
        png_equal=True, template_energy=metric.energy)


def fixed_args(args):
    result = copy.copy(args)
    result.out = args.fixed
    return result


def load_parameters(args):
    from models.e2_state_texture_b import StateTextureB
    from tools.e2_oracle_texture import weights_digest
    b = StateTextureB(color_channels=2, reference_channels=192)
    b.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=False)['b'])
    start, initial = flatten(list(b.parameters())), weights_digest(b)
    proposals = read(args.fixed / 'training/proposals.json')
    diagnostic = read(args.fixed / 'solver_diagnostic.json')['records'][-1]['arms']
    targets, records = {}, {}
    for arm in DIRECTIONS:
        report = proposals['arms'][arm]
        path = args.fixed / 'training' / (arm+'.pt')
        assert sha(path) == report['checkpoint_sha256']
        b.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['b'])
        target = flatten(list(b.parameters()))
        assert weights_digest(b) == report['final_b_hash'] and initial == report['initial_b_hash']
        # 来源身份用实际FP32位移字节核对；FP32并行范数仅报告，不添加跨平台逐位断言。
        assert tensor_sha(start-target) == diagnostic[arm]['actual_sha256']
        actual = target.double()-start.double()
        targets[arm] = target
        records[arm] = dict(actual_update_sha256_fp64=tensor_sha(actual), actual_norm_fp64=float(actual.norm()),
            legacy_subtraction_norm_fp32=float((start-target).norm()), source_actual_norm_fp32=report['actual_norm'],
            source_descent_sha256_fp32=diagnostic[arm]['actual_sha256'], checkpoint_sha256=report['checkpoint_sha256'])
    return start, targets, initial, records


def prepare(args):
    from tools import e2_b_scaled_projection_retry as retry
    retry.verify(fixed_args(args))
    assert not (args.out/'diagnostic_protocol.json').exists(), '不覆盖已有诊断协议'
    spec = read(args.fixed/'protocol.json')
    assert sha(args.fixed/'protocol.json') == PROTOCOL_SHA
    audit = read(args.fixed/'finite_step_audit.json')
    assert audit['counts']['total_comparisons'] == 672 and audit['new_backward_passes'] == 0
    assert audit['input_gradients_already_unscaled'] and audit['source_gradient_scale'] == SCALE
    selected = [dict(next(row for row in spec['inputs'] if key(row) == probe[:3]),
                     probe_index=index, selection_reason=probe[3]) for index, probe in enumerate(PROBES)]
    inputs, copied = {}, {}
    args.out.mkdir(parents=True, exist_ok=True)

    def record(path, expected=None):
        path = Path(path).resolve()
        assert 'holdout' not in path.parts
        value = sha(path)
        if expected is not None:
            assert value == expected, str(path)
        inputs[path.as_posix()] = value
        return value

    def duplicate(path, relative, expected=None):
        value = record(path, expected)
        destination = args.out/relative
        assert not destination.exists(), str(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        assert sha(destination) == value
        copied[relative] = value

    duplicate(args.fixed/'protocol.json', 'protocol.json', PROTOCOL_SHA)
    asset_names = set()
    for item in selected:
        prefix = 'cases/'+item['case']+'/'
        asset_names.update(prefix+name for name in ('mask.png', 'train_anchor.png', item['condition']+'_positive.png'))
        asset_names.update(prefix+item['sample_condition']+'_'+suffix+'.png' for suffix in ('ref', 'goal', 'positive'))
    for name in sorted(asset_names):
        duplicate(args.fixed/name, name, spec['assets'][name])
    record(args.checkpoint, CHECKPOINT_SHA)
    for name in ('solver_overlay.json', 'solver_diagnostic.json', 'solver_proposal_receipt.json',
                 'scaled_overlay.json', 'numerical_qualification.json', 'finite_step_audit.json',
                 'replay_decision.json', 'training/proposals.json', 'training/C0.pt', 'training/C1.pt'):
        record(args.fixed/name)
    for index, pair in enumerate(spec['train']):
        if not any(item['case'] == pair['case_id'] for item in selected):
            continue
        record(args.data/pair['sketch'], pair['hashes']['sketch'])
        for suffix in ('json', 'pt'):
            record(args.fixed/'gradients'/('shard%d.%s' % (index, suffix)))
    history = {}
    for model in ('f68',)+DIRECTIONS:
        path = args.fixed/'evaluation'/model/'replay/result.json'
        record(path)
        report = read(path)
        rows = []
        for item in selected:
            original = next(row for row in report['rows'] if key(row) == key(item))
            row = copy.deepcopy(original)
            relative = 'history/%s/probe_%d.png' % (model, item['probe_index'])
            duplicate(args.fixed/row['image'], relative, row['png_sha256'])
            row['image'] = relative
            rows.append(row)
        history[model] = rows
    write(args.out/'history.json', history)
    copied['history.json'] = sha(args.out/'history.json')
    _, _, initial, vectors = load_parameters(args)
    sources = set(read(args.fixed/'solver_overlay.json')['source_sha256']) | set(EXTRA_SOURCES)
    write(args.out/'diagnostic_protocol.json', dict(selected=selected, fixed_run=str(args.fixed),
        checkpoint_sha256=CHECKPOINT_SHA, protocol_sha256=PROTOCOL_SHA, initial_b_hash=initial,
        vectors=vectors, variants=list(VARIANTS), difference_steps=list(STEPS), shrink_scales=[.25, .125],
        source_gradient_job='125421', source_gradient_scale=SCALE, cached_gradients_already_unscaled=True,
        head_tasks=list(HEAD_TASKS), head_loss_scale=SCALE, planned_new_s0_images=70, planned_head_reverse_passes=30,
        sampler=spec['sampler'], input_sha256=inputs, copied_sha256=copied,
        source_sha256={name: sha(SOURCE/name) for name in sorted(sources)},
        execution_commit=subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip(),
        optimizer_updates=0, projection_calls=0, new_parameter_reverse_passes=0, adoption_allowed=False,
        holdout_results_read=0, selection_scope='predeclared train-side failure probes and one normal control; not a new validation population',
        parameter_definition='FP64 interpolation between stored FP32 f68 and C endpoints; cast once to original dtype; exact endpoints',
        q_definition='six original tasks at a detached FP32 f68 final-latent leaf; loss*1024 backward then q/1024; no UNet/B graph',
        pattern_policy='raw pattern remains unchanged; frozen f68 spectrum focus and no-grad base reported separately; no pattern q',
        interpretation='A-B includes head finite-step/non-smooth/arithmetic effects; B-C includes chain finite-step/rounding/AD error; neither is a unique root-cause proof'))
    verify(args)
    print('SCALED_DIRECTION_BRIDGE_PREPARED', len(selected), flush=True)


def verify(args):
    from tools import e2_b_scaled_projection_retry as retry
    retry.verify(fixed_args(args))
    spec = read(args.out/'diagnostic_protocol.json')
    assert [(r['case'], r['condition'], r['arm'], r['selection_reason']) for r in spec['selected']] == list(PROBES)
    assert spec['variants'] == list(VARIANTS) and spec['difference_steps'] == list(STEPS)
    assert all(sha(Path(path)) == value for path, value in spec['input_sha256'].items())
    assert all(sha(args.out/path) == value for path, value in spec['copied_sha256'].items())
    assert all(sha(SOURCE/path) == value for path, value in spec['source_sha256'].items())
    return spec


def run_probe(args):
    from models.e2_state_texture_b import StateTextureB
    from tools import e2_state_b_pilot as pilot, e2_state_b_endpoint as endpoint
    from tools.e2_state_b_pair_finetune import Objective
    from tools.e2_state_b_pair_evaluation import score_entry, training_comparison
    from tools.e2_oracle_texture import weights_digest
    from tools.e2_fft_experiment import environment
    from tools.e2_b_chain_repro import backend_flags

    started = time.monotonic()
    diagnostic = verify(args)
    destination = args.out/('probe_%d.json' % args.probe_index)
    assert not destination.exists(), '不覆盖已完成探针'
    selected = diagnostic['selected'][args.probe_index]
    spec = read(args.out/'protocol.json')
    case_index = next(i for i, pair in enumerate(spec['train']) if pair['case_id'] == selected['case'])
    pair = spec['train'][case_index]
    history = {model: next(row for row in rows if key(row) == key(selected))
               for model, rows in read(args.out/'history.json').items()}
    shard = torch.load(args.fixed/'gradients'/('shard%d.pt' % case_index), map_location='cpu', weights_only=False)
    gradient_index = next(i for i, row in enumerate(shard['items']) if key(row) == key(selected))
    cached_g = shard['gradients'][gradient_index].clone()
    replica_g = shard['replica_gradients'].get(selected['sample_condition'])
    replica_metadata = next((row for row in shard['replicas'] if row['sample_condition'] == selected['sample_condition']), None)
    assert (replica_g is not None) == (replica_metadata is not None)
    cached_values = next(row for row in shard['values'] if key(row) == key(selected))
    assert shard['loss_scale'] == SCALE and shard['all_finite'] and shard['numerical_qualification']
    assert cached_values['full_latent_equal'] and cached_values['full_latent_max_error'] == 0.
    start, targets, initial, source_vectors = load_parameters(args)
    variants = {model: {factor: interpolate(start, target, factor) for factor in VARIANTS+(1.,)}
                for model, target in targets.items()}
    runtime = copy.copy(args)
    runtime.action = 'train'
    core, pipe, net = pilot.load_core(runtime)
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    b = StateTextureB(color_channels=2, reference_channels=192).to(args.device)
    parameters = list(b.parameters())
    set_parameters(parameters, start)
    b.eval().requires_grad_(False)
    net.eval().requires_grad_(False)
    pipe.vae.eval().requires_grad_(False)
    initial_net = weights_digest(net)
    assert initial == weights_digest(b) == shard['b_hash'] and initial_net == shard['net_hash']
    entry = dict(selected, pair=pair, folder=args.out/'cases'/pair['case_id'],
        expected_digest=cached_values['noise_digest'], source_image=args.out/history['f68']['image'],
        assets={name: digest for name, digest in diagnostic['copied_sha256'].items()
                if name.startswith('cases/'+pair['case_id']+'/')})
    task_entry = dict(pair=pair, condition=selected['sample_condition'])
    rows, latents, controls = [], {}, []

    def capture(direction, factor, role):
        vector = start if direction == 'f68' else variants[direction][factor]
        set_parameters(parameters, vector)
        trace = []
        def hook(module, positional, keywords, output):
            trace.append(dict(timestep=int(keywords['timestep']), output_sha256=tensor_sha(output)))
        with torch.no_grad(), b.register_forward_hook(hook, with_kwargs=True):
            image, digest, tap, latent = endpoint.final_sample(runtime, core, pipe, net, pair,
                selected['sample_condition'], b)
            assert [r['timestep'] for r in trace] == pipe.scheduler.timesteps[25:].tolist()
            assert tap.index == 50 and len(tap.states) == 25 and digest == entry['expected_digest']
            state = tap.states[-1]
            base = pipe.scheduler.step(state['epsilon'].to(args.device), pipe.scheduler.timesteps[-1],
                state['zt'].to(args.device), return_dict=False)[0].float()
        return image, latent.detach(), base.detach(), trace, digest

    image0, z0, base0, trace0, digest0 = capture('f68', 0., 'baseline')
    objective = Objective(runtime, pipe, None, b, pair, [selected])
    assert pipe.vae.dtype == torch.float32
    initial_vae = weights_digest(pipe.vae)
    # q图先全部完成并释放，再执行任何pixels/half桥接；B和底座始终无梯度。
    q, head_values = head_derivatives(objective, task_entry, z0.to(args.device).float(), base0)
    assert not any(p.grad is not None or p.requires_grad for module in (b, net, pipe.vae) for p in module.parameters())
    _, positive, _, values, weight, _ = objective.records[pair['case_id'], selected['sample_condition']]
    with torch.no_grad():
        frozen_pattern = FrozenPattern(objective.rgb(z0.float()), positive, objective.rgb(base0), weight, pair['patch_coordinates'])

    def score(captured, direction, factor, role):
        image, latent, base, trace, digest = captured
        name = label(direction, factor, role)
        with torch.no_grad():
            tasks, details = objective.score_latents(task_entry, latent.float(), base, residue=objective.metric(task_entry).surrogate)
            numeric = endpoint.numbers(tasks)
            assert all(math.isfinite(value) for value in numeric.values())
            rgb32 = objective.rgb(latent.float())
            frozen = float(frozen_pattern.pattern(rgb32, tasks['pattern'].new_tensor(details['terms']['latent'])))
            rgb16 = half_rgb(pipe.vae, latent)
            picture = np.asarray(image)
            bridge = residue_bridge(objective.metric(task_entry), rgb32, rgb16, picture)
            assert abs(bridge['stations'][STATIONS[0]]['absolute']-numeric['residue']) <= 1e-6
            scored = score_entry(entry, picture, 'replay')
            assert abs(bridge['stations'][STATIONS[-1]]['absolute']-scored['scores']['old_projection']) <= 1e-12
        relative = 'images/probe_%d/%s.png' % (args.probe_index, name)
        path = args.out/relative
        assert not path.exists(), '不覆盖已有采样图'
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        vector = start if direction == 'f68' else variants[direction][factor]
        actual = vector.double()-start.double()
        ideal = torch.zeros_like(actual) if direction == 'f68' else factor*(targets[direction].double()-start.double())
        row = dict(probe_index=args.probe_index, case=selected['case'], condition=selected['condition'],
            arm=selected['arm'], sample_condition=selected['sample_condition'], group=selected['group'],
            direction=direction, scale=factor, role=role, label=name, image=relative, png_sha256=sha(path),
            noise_digest=digest, b_calls=len(trace), forward_trace=trace, b_hash=weights_digest(b),
            latent_sha256=tensor_sha(latent), latent_dtype=str(latent.dtype),
            latent_change_from_f68=compare_tensor(latent.cpu(), z0.cpu()),
            parameters=dict(sha256=tensor_sha(vector), displacement_sha256_fp64=tensor_sha(actual),
                norm_fp64=float(actual.norm()), changed_parameters=int(torch.count_nonzero(actual)),
                interpolation_rounding_error_norm=float((actual-ideal).norm())),
            training_tasks=numeric, training_task_comparison=training_comparison(numeric, cached_values['tasks']),
            frozen_pattern=frozen, residue_bridge=bridge, **scored)
        rows.append(row)
        latents[name] = latent.detach().cpu()
        write(args.out/('probe_%d.partial.json' % args.probe_index), dict(rows=rows, complete=False))
        print('DIRECTION_BRIDGE_SAMPLE', args.probe_index, len(rows), name, flush=True)
        return row

    baseline = score((image0, z0, base0, trace0, digest0), 'f68', 0., 'baseline')
    baseline_error = max(abs(baseline['training_tasks'][task]-cached_values['tasks'][task]) for task in TASKS)
    head_error = max(abs(head_values[task]-baseline['training_tasks'][task]) for task in TASKS)
    equal0 = np.array_equal(np.asarray(image0), np.asarray(Image.open(args.out/history['f68']['image'])))
    controls.append(dict(kind='historical_f68', png_equal=equal0, task_max_abs_to_new_G_values=baseline_error,
        tasks_equal_new_G=baseline['training_tasks'] == cached_values['tasks'], head_task_max_abs=head_error,
        head_tasks_equal=head_values == baseline['training_tasks'], cached_rollout_sample_latent_equal=True,
        cross_job_historical_latent_sha_available=False))
    assert equal0 and baseline_error <= 1e-6 and head_error <= 1e-6
    assert abs(baseline['frozen_pattern']-baseline['training_tasks']['pattern']) <= 1e-6
    repeated = score(capture('f68', 0., 'repeat'), 'f68', 0., 'repeat')
    repeat_noise = {task: abs(repeated['training_tasks'][task]-baseline['training_tasks'][task]) for task in TASKS}
    repeat = dict(kind='repeat_f68', png_equal=repeated['png_sha256'] == baseline['png_sha256'],
        latent_equal=torch.equal(latents[repeated['label']], latents[baseline['label']]),
        forward_trace_equal=repeated['forward_trace'] == baseline['forward_trace'], task_noise=repeat_noise,
        frozen_pattern_noise=abs(repeated['frozen_pattern']-baseline['frozen_pattern']))
    controls.append(repeat)
    assert repeat['png_equal'] and repeat['latent_equal'] and repeat['forward_trace_equal'] and max(repeat_noise.values()) == 0.
    for direction in DIRECTIONS:
        endpoint_row = score(capture(direction, 1., 'endpoint'), direction, 1., 'endpoint')
        old = history[direction]
        error = max(abs(endpoint_row['training_tasks'][task]-old['training_tasks'][task]) for task in TASKS)
        equal = np.array_equal(np.asarray(Image.open(args.out/endpoint_row['image'])), np.asarray(Image.open(args.out/old['image'])))
        controls.append(dict(kind='saved_endpoint', direction=direction, png_equal=equal,
            tasks_equal=endpoint_row['training_tasks'] == old['training_tasks'], task_max_abs_difference=error))
        assert equal and error <= 1e-6
        for factor in VARIANTS:
            score(capture(direction, factor, 'variant'), direction, factor, 'variant')
    for row in rows:
        row['residue_bridge']['delta_from_f68'] = {station: {field:
            row['residue_bridge']['stations'][station][field]-baseline['residue_bridge']['stations'][station][field]
            for field in ('signed', 'absolute')} for station in STATIONS}
    by_variant = {(row['direction'], row['scale']): row for row in rows if row['role'] == 'variant'}
    differences, patterns = [], []
    for direction in DIRECTIONS:
        for h in STEPS:
            plus, minus = by_variant[direction, h], by_variant[direction, -h]
            for task in HEAD_TASKS:
                result = three_level_fd(plus['training_tasks'][task], minus['training_tasks'][task], h,
                    q[HEAD_TASKS.index(task)], cached_g[TASKS.index(task)],
                    latents[plus['label']], latents[minus['label']], z0.cpu(), variants[direction][h], variants[direction][-h])
                replica = None
                if replica_g is not None:
                    chord = variants[direction][h].double()-variants[direction][-h].double()
                    prediction = float(replica_g[TASKS.index(task)].double() @ chord)/(2*h)
                    assert math.isfinite(prediction)
                    replica = dict(P_rep=prediction, P_main=result['C_cached_G_parameter_chord'],
                        replica_minus_main=scalar_difference(prediction, result['C_cached_G_parameter_chord']),
                        head_q_minus_replica=scalar_difference(result['B_head_q_latent_chord'], prediction))
                result.update(cached_replica_available=replica_g is not None, cached_replica_prediction=replica)
                differences.append(dict(direction=direction, h=h, task=task, repeat_task_noise=repeat_noise[task], **result))
            chord = variants[direction][h].double()-variants[direction][-h].double()
            prediction = float(cached_g[0].double() @ chord)/(2*h)
            raw = (plus['training_tasks']['pattern']-minus['training_tasks']['pattern'])/(2*h)
            frozen = (plus['frozen_pattern']-minus['frozen_pattern'])/(2*h)
            patterns.append(dict(direction=direction, h=h, raw_pattern_fd=raw, frozen_pattern_fd=frozen,
                cached_pattern_prediction=prediction, raw_difference=scalar_difference(raw, prediction),
                frozen_difference=scalar_difference(frozen, prediction), repeat_raw_noise=repeat_noise['pattern'],
                repeat_frozen_noise=repeat['frozen_pattern_noise'], head_q_computed=False))
    assert len(rows) == 14 and len(differences) == 24 and len(patterns) == 4
    set_parameters(parameters, start)
    assert weights_digest(b) == initial and weights_digest(net) == initial_net and weights_digest(pipe.vae) == initial_vae
    assert not any(p.grad is not None for module in (b, net, pipe.vae) for p in module.parameters())
    tensor_path = args.out/('probe_%d_tensors.npz' % args.probe_index)
    assert not tensor_path.exists()
    np.savez_compressed(tensor_path, q=q.numpy(), base0=base0.cpu().numpy(),
        **{'latent_'+name: value.numpy() for name, value in latents.items()})
    verify(args)
    report = dict(probe_index=args.probe_index, selected=selected, rows=rows, controls=controls,
        finite_differences=differences, pattern_differences=patterns, source_vectors=source_vectors,
        head_q={task: dict(sha256=tensor_sha(q[index]), norm_fp64=float(q[index].double().norm()),
            nonzero=int(torch.count_nonzero(q[index])), finite=True) for index, task in enumerate(HEAD_TASKS)},
        head_q_tasks=list(HEAD_TASKS), head_values=head_values, head_scale=SCALE,
        cached_replica_available=replica_g is not None, cached_replica_metadata=replica_metadata,
        tensor_file=tensor_path.name, tensor_sha256=sha(tensor_path), tensor_bytes=tensor_path.stat().st_size,
        new_s0_images=14, new_head_reverse_passes=6, new_parameter_reverse_passes=0,
        optimizer_updates=0, projection_calls=0, holdout_results_read=0, adopted_checkpoints=[],
        b_restored=True, frozen_unchanged=True, b_hash=initial, net_hash=initial_net, vae_hash_fp32=initial_vae,
        diagnostic_protocol_sha256=sha(args.out/'diagnostic_protocol.json'), source_sha256=diagnostic['source_sha256'],
        environment=environment(), backend_flags=backend_flags(), threads=torch.get_num_threads(),
        hardware=dict(hostname=socket.gethostname(), gpu=torch.cuda.get_device_name(args.device),
            job=os.environ.get('SLURM_JOB_ID'), node=os.environ.get('SLURM_JOB_NODELIST')),
        seconds=time.monotonic()-started, full_eligibility_evaluated=False,
        interpretation=diagnostic['interpretation'])
    write(destination, report)
    print('DIRECTION_BRIDGE_PROBE_COMPLETE', args.probe_index, sha(destination), flush=True)


def summarize(args):
    diagnostic = verify(args)
    assert not (args.out/'result.json').exists(), '不覆盖已有汇总'
    probes = [read(args.out/('probe_%d.json' % i)) for i in range(5)]
    files = {args.out/'protocol.json', args.out/'diagnostic_protocol.json', args.out/'history.json'}
    for index, report in enumerate(probes):
        assert report['probe_index'] == index and report['selected'] == diagnostic['selected'][index]
        assert report['diagnostic_protocol_sha256'] == sha(args.out/'diagnostic_protocol.json')
        assert report['source_sha256'] == diagnostic['source_sha256']
        assert report['b_restored'] and report['frozen_unchanged'] and not report['adopted_checkpoints']
        assert report['optimizer_updates'] == report['projection_calls'] == report['new_parameter_reverse_passes'] == report['holdout_results_read'] == 0
        assert len(report['rows']) == report['new_s0_images'] == 14 and report['new_head_reverse_passes'] == 6
        assert len(report['finite_differences']) == 24 and len(report['pattern_differences']) == 4
        tensor = args.out/report['tensor_file']
        assert sha(tensor) == report['tensor_sha256']
        files.update((tensor, args.out/('probe_%d.json' % index)))
        for row in report['rows']:
            path = args.out/row['image']
            assert sha(path) == row['png_sha256'] and row['residue_bridge']['png_equal']
            files.add(path)
    rows = [dict(probe_index=p['probe_index'], **row) for p in probes for row in p['finite_differences']]
    stats = {}
    for task in HEAD_TASKS:
        current = [row for row in rows if row['task'] == task]
        stats[task] = {level: dict(count=len(current), same_strict_sign=sum(row[level]['same_strict_sign'] for row in current),
            both_zero=sum(row[level]['both_zero'] for row in current),
            mean_absolute_difference=statistics.mean(row[level]['absolute_difference'] for row in current),
            median_symmetric_relative_difference=statistics.median(row[level]['symmetric_relative_difference'] for row in current))
            for level in ('head_difference', 'chain_difference', 'total_difference')}
    total_images = sum(p['new_s0_images'] for p in probes)
    total_heads = sum(p['new_head_reverse_passes'] for p in probes)
    assert total_images == 70 and total_heads == 30
    result = dict(protocol=diagnostic, probes=probes, finite_difference_summary_by_task=stats,
        new_s0_images=total_images, new_head_reverse_passes=total_heads, new_parameter_reverse_passes=0,
        optimizer_updates=0, projection_calls=0, holdout_results_read=0, adopted_checkpoints=[],
        full_eligibility_evaluated=False, scaled_gradients_are_ground_truth=False,
        formulas=dict(A='(L32(z_plus)-L32(z_minus))/(2h)',
            B='q32_f68 dot (z_plus.double()-z_minus.double())/(2h)',
            C='unscaled_cached_G dot (theta_plus.double()-theta_minus.double())/(2h)',
            symmetric_relative='abs(a-b)/max(abs(a),abs(b),1e-12)',
            head='A-B includes finite-step curvature, nonsmooth losses, and FP32 arithmetic',
            chain='B-C includes finite-step parameter-to-latent curvature/rounding and remaining AD error; not a unique attribution',
            residue='fixed old template; signed and absolute at four stations; decode half before normalization/clamp; uint8 OpenCV is actual acceptance',
            pattern='raw and frozen semantics separate; no new pattern q; agreement is not proof of true derivative'),
        repeat_scope='same-job exact controls do not guarantee cross-job or cross-node reproducibility')
    verify(args)
    write(args.out/'result.json', result)
    files.add(args.out/'result.json')
    files.update(args.out/name for name in diagnostic['copied_sha256'])
    manifest = {path.relative_to(args.out).as_posix(): sha(path) for path in sorted(files)}
    write(args.out/'manifest.json', manifest)
    with zipfile.ZipFile(args.out/'review.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files | {args.out/'manifest.json'}):
            archive.write(path, path.relative_to(args.out).as_posix())
    print('DIRECTION_BRIDGE_COMPLETE', total_images, total_heads, sha(args.out/'review.zip'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'verify', 'probe', 'summarize'))
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--project', type=Path, default=Path('/share/home/u2515283058/Mymodel'))
    parser.add_argument('--probe-index', type=int, choices=range(5), default=0)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.prior = args.prior.resolve()
    args.fixed = args.prior/'autonomous_repair_20261116/scaled_projection_retry'
    args.out = (args.out or args.prior/'autonomous_repair_20261116/scaled_direction_bridge').resolve()
    args.data = args.project.parent/'datasets/BF'
    args.checkpoint = args.project/'output_eval/e2_state_b_pastel_followup_20261115/trust_retry/training/step1.pt'
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    try:
        {'prepare': prepare, 'verify': verify, 'probe': run_probe, 'summarize': summarize}[args.action](args)
    except Exception as error:
        if args.action == 'probe' and args.out.exists():
            write(args.out/('probe_%d.failure.json' % args.probe_index), dict(error_type=type(error).__name__,
                message=str(error), complete=False, optimizer_updates=0, adopted_checkpoints=[]))
        raise


if __name__ == '__main__':
    main()
