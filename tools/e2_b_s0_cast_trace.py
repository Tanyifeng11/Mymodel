"""固定probe0/C1五条S0，只记录cast与scheduler状态；无新损失或反传。"""
import argparse
import copy
import inspect
import os
from pathlib import Path
import socket
import time
import zipfile

import numpy as np
from PIL import Image
import torch

from tools import e2_state_b_scaled_direction_bridge as bridge
from tools.e2_b_direction_diagnostics import interpolate
from tools.e2_b_endpoint_losses import set_parameters
from tools.e2_state_b_amplitude_response import read, write, sha
from tools.e2_state_b_backward_repro import tensor_sha


SOURCE = Path(__file__).resolve().parents[1]
SCALES = (0., .125, -.125, .03125, -.03125)
STEPS = (.125, .03125)
FIELDS = ('precast', 'native', 'raw_epsilon', 'zt', 'previous', 'delta')
EXTRA = ('tools/e2_b_s0_cast_trace.py', 'scripts/e2_b_s0_cast_trace.sbatch')


def half_ulp_distance(left, right):
    """有限FP16的有序格点距离；+0与-0视为同一格点。"""
    assert left.dtype == right.dtype == np.float16 and np.isfinite(left).all() and np.isfinite(right).all()
    def rank(value):
        bits = np.ascontiguousarray(value).view(np.uint16).astype(np.int32)
        return np.where(bits & 0x8000, 0x8000-(bits & 0x7fff), 0x8000+bits)
    return np.abs(rank(left)-rank(right))


def vector_stats(value):
    value = np.asarray(value, np.float64)
    return dict(elements=value.size, nonzero=int(np.count_nonzero(value)),
        norm=float(np.linalg.norm(value.ravel())), max_abs=float(np.max(np.abs(value))),
        mean_abs=float(np.mean(np.abs(value))))


def chord_stats(plus, minus, zero, h):
    chord = plus.astype(np.float64)-minus.astype(np.float64)
    midpoint = (plus.astype(np.float64)+minus.astype(np.float64))/2-zero.astype(np.float64)
    stats = vector_stats(chord)
    result = dict(chord=stats, chord_norm_per_unit=stats['norm']/(2*h), midpoint_offset=vector_stats(midpoint),
        midpoint_to_half_chord_norm=None if stats['norm'] == 0 else float(np.linalg.norm(midpoint.ravel())/(stats['norm']/2)))
    if plus.dtype == minus.dtype == np.float16:
        ulp = half_ulp_distance(plus, minus)
        result['native_ulp_distance'] = dict(definition='ordered FP16 grid distance between plus and minus; signed zeros merged',
            maximum=int(ulp.max()), mean=float(ulp.mean()),
            counts=dict(zero=int((ulp == 0).sum()), one=int((ulp == 1).sum()),
                two=int((ulp == 2).sum()), three=int((ulp == 3).sum()), at_least_four=int((ulp >= 4).sum())))
    return result


def cast_comparison(plus, minus, zero, h):
    """只分解已观察弦；precast已经包含前序步骤的量化反馈。"""
    pre = plus['precast'].astype(np.float64)-minus['precast'].astype(np.float64)
    native = plus['native'].astype(np.float64)-minus['native'].astype(np.float64)
    residual = lambda values: values['native'].astype(np.float64)-values['precast'].astype(np.float64)
    error_chord = residual(plus)-residual(minus)
    error_midpoint = (residual(plus)+residual(minus))/2-residual(zero)
    np_norm, nn_norm = np.linalg.norm(pre.ravel()), np.linalg.norm(native.ravel())
    return dict(fields={name: chord_stats(plus[name], minus[name], zero[name], h) for name in FIELDS},
        rounding_residual_chord=vector_stats(error_chord), rounding_residual_chord_norm_per_unit=float(np.linalg.norm(error_chord.ravel())/(2*h)),
        rounding_residual_midpoint_offset=vector_stats(error_midpoint),
        decomposition_max_abs=float(np.max(np.abs(native-pre-error_chord))),
        precast_nonzero_native_zero=int(((pre != 0) & (native == 0)).sum()),
        native_precast_chord_cosine=None if np_norm*nn_norm == 0 else float(np.dot(pre.ravel(), native.ravel())/(np_norm*nn_norm)),
        native_to_precast_chord_norm=None if np_norm == 0 else float(nn_norm/np_norm),
        rounding_chord_to_precast_norm=None if np_norm == 0 else float(np.linalg.norm(error_chord.ravel())/np_norm))


def bridge_args(args):
    value = copy.copy(args)
    value.out = args.bridge
    return value


def verify(args):
    bridge.verify(bridge_args(args))
    protocol = read(args.out/'trace_protocol.json')
    assert protocol['scales'] == list(SCALES) and protocol['selected'] == list(bridge.PROBES[0][:3])
    assert all(sha(Path(path)) == digest for path, digest in protocol['input_sha256'].items())
    assert all(sha(SOURCE/path) == digest for path, digest in protocol['source_sha256'].items())
    manifest = args.out/'manifest.json'
    if manifest.exists():
        assert all(sha(args.out/path) == digest for path, digest in read(manifest).items())
    return protocol


def prepare(args):
    diagnostic = bridge.verify(bridge_args(args))
    assert not (args.out/'trace_protocol.json').exists(), '不覆盖已有cast诊断'
    args.out.mkdir(parents=True, exist_ok=True)
    original = read(args.bridge/'probe_0.json')
    manifest = read(args.bridge/'manifest.json')
    assert original['selected'] == diagnostic['selected'][0]
    assert tuple(bridge.key(original['selected'])) == bridge.PROBES[0][:3]
    names = ['diagnostic_protocol.json', 'result.json', 'manifest.json', 'probe_0.json', original['tensor_file']]
    selected = []
    for scale in SCALES:
        direction, role = ('f68', 'baseline') if scale == 0 else ('C1', 'variant')
        row = next(r for r in original['rows'] if r['direction'] == direction and r['scale'] == scale and r['role'] == role)
        selected.append(row)
        names.append(row['image'])
    inputs = {}
    for name in names:
        path = args.bridge/name
        digest = sha(path)
        if name != 'manifest.json':
            assert digest == manifest[name]
        inputs[path.as_posix()] = digest
    assert inputs[(args.bridge/original['tensor_file']).as_posix()] == original['tensor_sha256']
    write(args.out/'trace_protocol.json', dict(selected=list(bridge.PROBES[0][:3]), direction='C1', scales=list(SCALES),
        source_bridge=str(args.bridge), source_rows=selected, source_gradient_job='125421', source_bridge_job='125444',
        input_sha256=inputs, source_sha256=dict(diagnostic['source_sha256'], **{name: sha(SOURCE/name) for name in EXTRA}),
        planned_s0_images=5, planned_active_steps_per_sample=25, new_loss_evaluations=0, new_backward_passes=0,
        optimizer_updates=0, optimization_projection_calls=0, holdout_results_read=0, adopted_checkpoints=[],
        policy='observe original Tap; external scheduler wrapper forwards unchanged; reconstruct pre-cast only on GPU and require exact native cast',
        interpretation='precast already contains earlier quantization feedback; grid/chord observations alone cannot establish the whole root cause'))
    return verify(args), original


def run(args):
    from models.e2_state_texture_b import StateTextureB
    from tools import e2_state_b_pilot as pilot, e2_state_b_endpoint as endpoint
    from tools.e2_oracle_texture import weights_digest
    from tools.e2_fft_experiment import environment
    from tools.e2_b_chain_repro import backend_flags
    started = time.monotonic()
    protocol, original = prepare(args)
    diagnostic = read(args.bridge/'diagnostic_protocol.json')
    item = diagnostic['selected'][0]
    spec = read(args.bridge/'protocol.json')
    pair = next(p for p in spec['train'] if p['case_id'] == item['case'])
    with np.load(args.bridge/original['tensor_file']) as tensors:
        q = torch.from_numpy(tensors['q'][original['head_q_tasks'].index('amplitude')].copy())
        old_latents = {row['label']: tensors['latent_'+row['label']].copy() for row in protocol['source_rows']}
    assert tensor_sha(q) == original['head_q']['amplitude']['sha256']
    start, targets, initial, vector_sources = bridge.load_parameters(args)
    runtime = bridge_args(args)
    runtime.action = 'train'
    core, pipe, net = pilot.load_core(runtime)
    torch.set_num_threads(2)
    b = StateTextureB(color_channels=2, reference_channels=192).to(args.device)
    parameters = list(b.parameters())
    b.eval().requires_grad_(False)
    net.eval().requires_grad_(False)
    pipe.vae.eval().requires_grad_(False)
    net_hash, vae_hash = weights_digest(net), weights_digest(pipe.vae)
    assert net_hash == original['net_hash']
    scheduler_source = Path(inspect.getsourcefile(type(pipe.scheduler)))
    scheduler_sha = sha(scheduler_source)
    records, arrays = [], {}
    for source_row in protocol['source_rows']:
        scale = source_row['scale']
        vector = interpolate(start, targets['C1'], scale)
        set_parameters(parameters, vector)
        assert tensor_sha(vector) == source_row['parameters']['sha256'] and weights_digest(b) == source_row['b_hash']
        native_step = pipe.scheduler.step
        observed, deltas, trace = [], [], []
        counter = [0]
        def step(epsilon, timestep, sample, *positional, **keywords):
            index = counter[0]
            counter[0] += 1
            result = native_step(epsilon, timestep, sample, *positional, **keywords)
            if index >= 25:
                previous = result[0] if isinstance(result, tuple) else result.prev_sample
                observed.append(dict(index=index, timestep=int(timestep), native=epsilon.detach().clone(),
                    zt=sample.detach().clone(), previous=previous.detach().clone()))
            return result
        def hook(module, positional, keywords, output):
            deltas.append(output.detach().clone())
            trace.append(dict(timestep=int(keywords['timestep']), output_sha256=tensor_sha(output)))
        pipe.scheduler.step = step
        try:
            with torch.no_grad(), b.register_forward_hook(hook, with_kwargs=True):
                image, digest, tap, latent = endpoint.final_sample(runtime, core, pipe, net, pair, item['sample_condition'], b)
        finally:
            pipe.scheduler.step = native_step
        assert counter[0] == 50 and len(observed) == len(deltas) == len(tap.states) == 25
        assert trace == source_row['forward_trace'] and digest == source_row['noise_digest']
        latent_array = latent.detach().cpu().numpy()
        assert np.array_equal(latent_array, old_latents[source_row['label']]) and tensor_sha(latent) == source_row['latent_sha256']
        assert np.array_equal(np.asarray(image), np.asarray(Image.open(args.bridge/source_row['image'])))
        assert torch.equal(observed[-1]['previous'], latent)
        data = {name: [] for name in FIELDS}
        alphas, controls = [], []
        with torch.no_grad():
            for state, measured, delta in zip(tap.states, observed, deltas):
                assert state['index'] == measured['index'] and state['t'] == measured['timestep']
                assert measured['native'].dtype == torch.float16 and measured['zt'].dtype == torch.float16
                raw = state['epsilon'].to(args.device)
                assert raw.dtype == torch.float16 and torch.equal(state['zt'].to(args.device), measured['zt'])
                alpha = pipe.scheduler.alphas_cumprod[state['t']].to(args.device).float()
                precast = raw.float()-(alpha/(1-alpha)).sqrt()*delta
                cast = precast.to(measured['native'].dtype)
                assert torch.equal(cast, measured['native']), 'GPU原表达式重构cast不等，拒绝分析'
                values = dict(precast=precast, raw_epsilon=raw, delta=delta,
                    **{name: measured[name] for name in ('native', 'zt', 'previous')})
                assert all(torch.isfinite(value).all() for value in values.values())
                for name, value in values.items():
                    data[name].append(value.detach().cpu().numpy())
                alphas.append(float(alpha))
                controls.append(dict(index=state['index'], timestep=state['t'], gpu_cast_equal=True, tap_zt_equal=True))
        data = {name: np.stack(values) for name, values in data.items()}
        data.update(alpha=np.asarray(alphas, np.float32), timesteps=np.asarray([row['timestep'] for row in observed], np.int64))
        arrays[scale] = data
        name = source_row['label']
        tensor_path, image_path = args.out/(name+'.npz'), args.out/(name+'.png')
        assert not tensor_path.exists() and not image_path.exists()
        np.savez_compressed(tensor_path, **data)
        image.save(image_path)
        record = dict(label=name, scale=scale, controls=controls, b_hash=weights_digest(b), noise_digest=digest,
            bridge_latent_equal=True, bridge_png_equal=True, bridge_B_trace_equal=True, forward_trace=trace,
            final_latent_sha256=tensor_sha(latent), tensor_file=tensor_path.name, tensor_sha256=sha(tensor_path),
            tensor_bytes=tensor_path.stat().st_size, image=image_path.name, png_sha256=sha(image_path),
            field_metadata={field: dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=tensor_sha(torch.from_numpy(value))) for field, value in data.items()})
        records.append(record)
        write(args.out/'partial.json', dict(records=records, complete=False))
        print('S0_CAST_CAPTURE', len(records), name, flush=True)
    zero = arrays[0.]
    assert all(np.array_equal(data[name][0], zero[name][0]) for data in arrays.values() for name in ('zt', 'raw_epsilon'))
    comparisons, final_q = [], []
    for h in STEPS:
        for index in range(25):
            select = lambda data: {field: data[field][index] for field in FIELDS}
            comparisons.append(dict(h=h, index=index+25, timestep=int(zero['timesteps'][index]),
                **cast_comparison(select(arrays[h]), select(arrays[-h]), select(zero), h)))
        chord = torch.from_numpy(arrays[h]['previous'][-1]).double()-torch.from_numpy(arrays[-h]['previous'][-1]).double()
        prediction = float(q.double().reshape(-1) @ chord.reshape(-1))/(2*h)
        previous = next(row for row in original['finite_differences'] if row['direction'] == 'C1' and row['h'] == h and row['task'] == 'amplitude')
        final_q.append(dict(h=h, reused_amplitude_q_final_chord=prediction,
            bridge_B_head_q_latent_chord=previous['B_head_q_latent_chord'],
            difference=prediction-previous['B_head_q_latent_chord'],
            exact_equal=prediction == previous['B_head_q_latent_chord']))
    set_parameters(parameters, start)
    assert weights_digest(b) == initial and weights_digest(net) == net_hash and weights_digest(pipe.vae) == vae_hash
    assert all(p.grad is None for module in (b, net, pipe.vae) for p in module.parameters())
    assert sha(scheduler_source) == scheduler_sha
    verify(args)
    result = dict(protocol_sha256=sha(args.out/'trace_protocol.json'), records=records, comparisons=comparisons,
        final_amplitude_q_projection=final_q, q_reused=True, q_sha256=tensor_sha(q),
        first_active_step_shared_zt_and_raw_epsilon=True, vector_sources=vector_sources,
        scheduler=dict(class_name=type(pipe.scheduler).__name__, config=dict(pipe.scheduler.config),
            source_file=str(scheduler_source), source_sha256=scheduler_sha),
        new_s0_images=5, captured_active_steps=125, new_loss_evaluations=0, new_backward_passes=0,
        new_head_reverse_passes=0, new_parameter_reverse_passes=0, optimizer_updates=0, optimization_projection_calls=0,
        holdout_results_read=0, adopted_checkpoints=[], b_restored=True, frozen_unchanged=True,
        b_hash=initial, net_hash=net_hash, vae_hash=vae_hash, environment=environment(), backend_flags=backend_flags(),
        hardware=dict(hostname=socket.gethostname(), gpu=torch.cuda.get_device_name(args.device), job=os.environ.get('SLURM_JOB_ID')),
        seconds=time.monotonic()-started, interpretation=protocol['interpretation'],
        definitions=dict(chord='plus-minus in FP64; norm_per_unit divides by2h',
            rounding_residual='native.double-precast.double; residual chord decomposes observed native/precast chord',
            midpoint='(plus.double+minus.double)/2-f68.double; ratio denominator=half chord norm',
            ulp='ordered finite FP16 value ranks; reports changes in observed values, not zero gradients'))
    write(args.out/'result.json', result)
    files = [args.out/'trace_protocol.json', args.out/'result.json']
    files += [args.out/r[key] for r in records for key in ('tensor_file', 'image')]
    manifest = {path.name: sha(path) for path in files}
    write(args.out/'manifest.json', manifest)
    with zipfile.ZipFile(args.out/'review.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files+[args.out/'manifest.json']:
            archive.write(path, path.name)
    verify(args)
    print('S0_CAST_TRACE_COMPLETE', sha(args.out/'review.zip'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', 'verify'))
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--project', type=Path, default=Path('/share/home/u2515283058/Mymodel'))
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.prior = args.prior.resolve()
    auto = args.prior/'autonomous_repair_20261116'
    args.bridge, args.fixed = auto/'scaled_direction_bridge', auto/'scaled_projection_retry'
    args.out = (args.out or auto/'scaled_cast_trace').resolve()
    args.data = args.project.parent/'datasets/BF'
    args.checkpoint = args.project/'output_eval/e2_state_b_pastel_followup_20261115/trust_retry/training/step1.pt'
    torch.set_num_threads(2)
    try:
        {'run': run, 'verify': verify}[args.action](args)
    except Exception as error:
        if args.action == 'run' and args.out.exists():
            write(args.out/'failure.json', dict(error_type=type(error).__name__, message=str(error), complete=False,
                new_backward_passes=0, optimizer_updates=0, adopted_checkpoints=[]))
        raise


if __name__ == '__main__':
    main()
