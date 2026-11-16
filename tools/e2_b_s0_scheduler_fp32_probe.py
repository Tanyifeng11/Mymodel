"""五点零训练对照：只将active25:50的DDIM内部算术改为FP32，状态仍为FP16。"""
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

from tools import e2_b_s0_cast_trace as cast_trace
from tools import e2_state_b_scaled_direction_bridge as bridge
from tools.e2_b_direction_diagnostics import interpolate
from tools.e2_b_endpoint_losses import set_parameters
from tools.e2_state_b_amplitude_response import read, write, sha
from tools.e2_state_b_backward_repro import tensor_sha


SOURCE = Path(__file__).resolve().parents[1]
EXTRA = ('tools/e2_b_s0_scheduler_fp32_probe.py', 'scripts/e2_b_s0_scheduler_fp32_probe.sbatch')


def cast_output(result, dtype):
    """保留DDIM两项返回值及返回形式，仅将输出恢复输入的native dtype。"""
    if isinstance(result, tuple):
        return tuple(value.to(dtype) if torch.is_tensor(value) else value for value in result)
    return type(result)(prev_sample=result.prev_sample.to(dtype),
        pred_original_sample=result.pred_original_sample.to(dtype))


class PrecisionStep:
    def __init__(self, original):
        self.original, self.count, self.prefix, self.active = original, 0, [], []

    def __call__(self, epsilon, timestep, sample, *positional, **keywords):
        index = self.count
        self.count += 1
        assert keywords.get('eta', 0.) == 0. and not keywords.get('use_clipped_model_output', False)
        if index < 25:
            result = self.original(epsilon, timestep, sample, *positional, **keywords)
            previous = result[0] if isinstance(result, tuple) else result.prev_sample
            self.prefix.append(dict(index=index, timestep=int(timestep), epsilon_sha256=tensor_sha(epsilon),
                zt_sha256=tensor_sha(sample), previous_sha256=tensor_sha(previous)))
            return result
        assert index < 50 and epsilon.dtype == sample.dtype == torch.float16
        result32 = self.original(epsilon.float(), timestep, sample.float(), *positional, **keywords)
        result = cast_output(result32, sample.dtype)
        previous32 = result32[0] if isinstance(result32, tuple) else result32.prev_sample
        previous = result[0] if isinstance(result, tuple) else result.prev_sample
        assert previous32.dtype == torch.float32 and previous.dtype == torch.float16
        self.active.append(dict(index=index, timestep=int(timestep), native=epsilon.detach().clone(),
            zt=sample.detach().clone(), previous32=previous32.detach().clone(), previous=previous.detach().clone()))
        return result


def old_args(args):
    value = copy.copy(args)
    value.out = args.trace
    return value


def verify(args):
    cast_trace.verify(old_args(args))
    protocol = read(args.out/'probe_protocol.json')
    assert protocol['scales'] == list(cast_trace.SCALES) and protocol['active'] == [25, 50]
    assert all(sha(Path(path)) == digest for path, digest in protocol['input_sha256'].items())
    assert all(sha(SOURCE/path) == digest for path, digest in protocol['source_sha256'].items())
    if (args.out/'manifest.json').exists():
        assert all(sha(args.out/path) == digest for path, digest in read(args.out/'manifest.json').items())
    return protocol


def prepare(args):
    previous_protocol = cast_trace.verify(old_args(args))
    assert not (args.out/'probe_protocol.json').exists(), '不覆盖已有数值对照'
    args.out.mkdir(parents=True, exist_ok=True)
    old = read(args.trace/'result.json')
    assert old['new_s0_images'] == 5 and old['first_active_step_shared_zt_and_raw_epsilon']
    paths = [args.trace/name for name in ('trace_protocol.json', 'result.json', 'manifest.json')]
    paths += [args.trace/row[key] for row in old['records'] for key in ('tensor_file', 'image')]
    paths += [args.bridge/name for name in ('diagnostic_protocol.json', 'probe_0.json')]
    write(args.out/'probe_protocol.json', dict(selected=list(bridge.PROBES[0][:3]), direction='C1',
        scales=list(cast_trace.SCALES), active=[25, 50], scheduler=old['scheduler'],
        source_rows=previous_protocol['source_rows'], source_cast_run=str(args.trace),
        input_sha256={path.as_posix(): sha(path) for path in paths},
        source_sha256=dict(previous_protocol['source_sha256'], **{name: sha(SOURCE/name) for name in EXTRA}),
        planned_s0_images=5, new_backward_passes=0, optimizer_updates=0, optimization_projection_calls=0,
        holdout_results_read=0, adopted_checkpoints=[],
        numeric_variant='active25:50 FP32 scheduler arithmetic then FP16 output; original FP16 UNet/CFG/corrected epsilon/state',
        pixel_trajectory_changed_by_design=True, original_forward_equivalence_claimed=False,
        primary_comparison='old f68 under original native scheduler; new f68 baseline is reported separately, never used to hide regression',
        pattern_base='same original bridge head: no-B base computed by original native-half scheduler on the new last-step state',
        no_derivative_claim='old G/P/q are not used; two-scale latent chords are observations of this new numeric variant'))
    return verify(args), old


def run(args):
    from models.e2_state_texture_b import StateTextureB
    from tools import e2_state_b_pilot as pilot, e2_state_b_endpoint as endpoint
    from tools.e2_state_b_pair_finetune import Objective
    from tools.e2_state_b_pair_evaluation import score_entry, training_comparison, strict_replay_protection
    from tools.e2_state_b_pair_summary import METRICS
    from tools.e2_oracle_texture import weights_digest
    from tools.e2_fft_experiment import environment
    from tools.e2_b_chain_repro import backend_flags
    started = time.monotonic()
    protocol, old_trace = prepare(args)
    item = read(args.bridge/'diagnostic_protocol.json')['selected'][0]
    spec = read(args.bridge/'protocol.json')
    pair = next(p for p in spec['train'] if p['case_id'] == item['case'])
    old_f68 = protocol['source_rows'][0]
    old_records = {row['label']: row for row in old_trace['records']}
    start, targets, initial, vector_sources = bridge.load_parameters(args)
    runtime = copy.copy(args)
    runtime.action, runtime.out = 'train', args.bridge
    core, pipe, net = pilot.load_core(runtime)
    torch.set_num_threads(2)
    b = StateTextureB(color_channels=2, reference_channels=192).to(args.device)
    parameters = list(b.parameters())
    b.eval().requires_grad_(False)
    net.eval().requires_grad_(False)
    pipe.vae.eval().requires_grad_(False)
    net_hash = weights_digest(net)
    assert net_hash == old_trace['net_hash']
    scheduler_source = Path(inspect.getsourcefile(type(pipe.scheduler)))
    scheduler_sha = sha(scheduler_source)
    assert scheduler_sha == protocol['scheduler']['source_sha256'] and type(pipe.scheduler).__name__ == 'DDIMScheduler'
    config = dict(pipe.scheduler.config)
    for name, value in protocol['scheduler']['config'].items():
        assert (set(config[name]) == set(value)) if name == '_use_default_values' else config[name] == value
    assert config['prediction_type'] == 'epsilon' and not config['clip_sample'] and not config['thresholding']
    objective, vae_hash, prefix0 = None, None, None
    rows, arrays = [], {}
    entry = dict(item, pair=pair, folder=args.bridge/'cases'/pair['case_id'],
        source_image=args.bridge/old_f68['image'], assets=old_f68['assets'])
    task_entry = dict(pair=pair, condition=item['sample_condition'])
    for old_row in protocol['source_rows']:
        scale, name = old_row['scale'], old_row['label']
        vector = interpolate(start, targets['C1'], scale)
        set_parameters(parameters, vector)
        assert tensor_sha(vector) == old_row['parameters']['sha256'] and weights_digest(b) == old_row['b_hash']
        original_step = pipe.scheduler.step
        wrapper = PrecisionStep(original_step)
        trace, first_delta = [], []
        def hook(module, positional, keywords, output):
            trace.append(dict(timestep=int(keywords['timestep']), output_sha256=tensor_sha(output)))
            if not first_delta:
                first_delta.append(output.detach().cpu().clone())
        pipe.scheduler.step = wrapper
        try:
            with torch.no_grad(), b.register_forward_hook(hook, with_kwargs=True):
                image, digest, tap, latent = endpoint.final_sample(runtime, core, pipe, net, pair, item['sample_condition'], b)
        finally:
            pipe.scheduler.step = original_step
        assert wrapper.count == 50 and len(wrapper.prefix) == len(wrapper.active) == len(trace) == len(tap.states) == 25
        assert digest == old_row['noise_digest'] and trace[0] == old_row['forward_trace'][0]
        if prefix0 is None:
            prefix0 = wrapper.prefix
        assert wrapper.prefix == prefix0, '五点的未改动prefix不一致'
        old_record = old_records[name]
        assert sha(args.trace/old_record['tensor_file']) == old_record['tensor_sha256']
        with np.load(args.trace/old_record['tensor_file']) as previous:
            first = wrapper.active[0]
            checks = dict(zt=tensor_sha(first['zt']) == tensor_sha(torch.from_numpy(previous['zt'][0])),
                native=tensor_sha(first['native']) == tensor_sha(torch.from_numpy(previous['native'][0])),
                raw_epsilon=tensor_sha(tap.states[0]['epsilon']) == tensor_sha(torch.from_numpy(previous['raw_epsilon'][0])),
                delta=tensor_sha(first_delta[0]) == tensor_sha(torch.from_numpy(previous['delta'][0])))
            assert all(checks.values()), '首active输入不同，不能解释为仅scheduler算术变化'
            assert wrapper.prefix[-1]['previous_sha256'] == tensor_sha(torch.from_numpy(previous['zt'][0]))
            old_final = previous['previous'][-1].copy()
        assert torch.equal(wrapper.active[-1]['previous'], latent)
        if objective is None:
            objective = Objective(runtime, pipe, None, b, pair, [item])
            vae_hash = weights_digest(pipe.vae)
        state = tap.states[-1]
        with torch.no_grad():
            # 原bridge评测头不改；pattern的no-B参照仍用原native scheduler。
            base = original_step(state['epsilon'].to(args.device), pipe.scheduler.timesteps[-1],
                state['zt'].to(args.device), return_dict=False)[0].float()
            tasks, _ = objective.score_latents(task_entry, latent.float(), base, residue=objective.metric(task_entry).surrogate)
            numeric = endpoint.numbers(tasks)
            assert all(np.isfinite(value) for value in numeric.values())
            scored = score_entry(entry, np.asarray(image), 'replay')
        data = {field: np.stack([r[field].cpu().numpy() for r in wrapper.active])
                for field in ('native', 'zt', 'previous32', 'previous')}
        data.update(raw_epsilon=np.stack([state['epsilon'].numpy() for state in tap.states]),
            timesteps=np.asarray([r['timestep'] for r in wrapper.active], np.int64))
        arrays[scale] = data
        image_path, tensor_path = args.out/(name+'.png'), args.out/(name+'.npz')
        assert not image_path.exists() and not tensor_path.exists()
        image.save(image_path)
        np.savez_compressed(tensor_path, **data)
        picture, old_picture = np.asarray(image), np.asarray(Image.open(args.bridge/old_row['image']))
        final = latent.detach().cpu().numpy()
        row = dict(case=item['case'], condition=item['condition'], arm=item['arm'], group=item['group'],
            label=name, scale=scale, sample_condition=item['sample_condition'], direction=old_row['direction'],
            numeric_variant=protocol['numeric_variant'], image=image_path.name, png_sha256=sha(image_path),
            tensor_file=tensor_path.name, tensor_sha256=sha(tensor_path), tensor_bytes=tensor_path.stat().st_size,
            latent_sha256=tensor_sha(latent), latent_dtype=str(latent.dtype), b_hash=weights_digest(b), noise_digest=digest,
            b_calls=len(trace), forward_trace=trace, first_active_inputs_equal_old_trace=checks,
            standard_prefix_checksums=wrapper.prefix, old_prefix_full_checksums_available=False,
            old_prefix_end_zt_equal=True, prefix_equal_across_five_points=True,
            actual_pixel_trajectory_changed=not np.array_equal(final, old_final),
            old_same_scale_png_equal=bool(np.array_equal(picture, old_picture)),
            old_same_scale_png_changed_channels=int(np.count_nonzero(picture != old_picture)),
            old_same_scale_latent_change=cast_trace.vector_stats(final.astype(np.float64)-old_final.astype(np.float64)),
            training_tasks=numeric, old_f68_task_comparison=training_comparison(numeric, old_f68['training_tasks']),
            old_same_scale_task_comparison=training_comparison(numeric, old_row['training_tasks']),
            old_f68_score_delta={k: scored['scores'][k]-old_f68['scores'][k] for k in METRICS},
            old_same_scale_score_delta={k: scored['scores'][k]-old_row['scores'][k] for k in METRICS}, **scored)
        row['old_f68_pixel_protection_subset'] = strict_replay_protection([row], {bridge.key(old_f68): old_f68})
        rows.append(row)
        write(args.out/'partial.json', dict(rows=rows, complete=False))
        print('SCHEDULER_FP32_SAMPLE', len(rows), name, row['old_f68_task_comparison']['failures'], flush=True)
    baseline = rows[0]
    for row in rows:
        row['new_f68_task_delta_descriptive_only'] = {task: row['training_tasks'][task]-baseline['training_tasks'][task] for task in bridge.TASKS}
        row['new_f68_score_delta_descriptive_only'] = {k: row['scores'][k]-baseline['scores'][k] for k in METRICS}
    chord_records = []
    for h in cast_trace.STEPS:
        chord_records.append(dict(h=h, final_latent=cast_trace.chord_stats(arrays[h]['previous'][-1], arrays[-h]['previous'][-1], arrays[0.]['previous'][-1], h),
            first_step_previous32=cast_trace.chord_stats(arrays[h]['previous32'][0], arrays[-h]['previous32'][0], arrays[0.]['previous32'][0], h),
            first_step_previous16=cast_trace.chord_stats(arrays[h]['previous'][0], arrays[-h]['previous'][0], arrays[0.]['previous'][0], h)))
    large = arrays[.125]['previous'][-1].astype(np.float64)-arrays[-.125]['previous'][-1].astype(np.float64)
    small = arrays[.03125]['previous'][-1].astype(np.float64)-arrays[-.03125]['previous'][-1].astype(np.float64)
    norms = np.linalg.norm(large.ravel()), np.linalg.norm(small.ravel())
    cross_h = dict(h_small_over_large=.25, chord_norm_small_over_large=None if norms[0] == 0 else float(norms[1]/norms[0]),
        chord_cosine=None if norms[0]*norms[1] == 0 else float(np.dot(large.ravel(), small.ravel())/(norms[0]*norms[1])))
    set_parameters(parameters, start)
    assert weights_digest(b) == initial and weights_digest(net) == net_hash and weights_digest(pipe.vae) == vae_hash
    assert all(p.grad is None for module in (b, net, pipe.vae) for p in module.parameters())
    assert sha(scheduler_source) == scheduler_sha
    verify(args)
    result = dict(protocol_sha256=sha(args.out/'probe_protocol.json'), rows=rows, f68_precision_effect=rows[0],
        latent_chords=chord_records, latent_cross_h=cross_h, vector_sources=vector_sources,
        new_s0_images=5, b_calls=125, new_backward_passes=0, new_q_or_G=False, old_gradient_predictions_used=False,
        optimizer_updates=0, optimization_projection_calls=0, holdout_results_read=0, adopted_checkpoints=[],
        frozen_unchanged=True, b_restored=True, b_hash=initial, net_hash=net_hash, vae_hash_fp32=vae_hash,
        scheduler=dict(source_sha256=scheduler_sha, config=config), environment=environment(), backend_flags=backend_flags(),
        hardware=dict(hostname=socket.gethostname(), gpu=torch.cuda.get_device_name(args.device), job=os.environ.get('SLURM_JOB_ID')),
        seconds=time.monotonic()-started, full_eligibility_evaluated=False,
        interpretation='five-point forward-only numeric variant; old f68 remains protection reference; new baseline cannot hide regression; latent chords are not old-direction derivatives; output FP16 quantization and UNet/CFG remain',
        repeat_noise_measured=False, pattern_base=protocol['pattern_base'])
    write(args.out/'result.json', result)
    files = [args.out/'probe_protocol.json', args.out/'result.json']+[args.out/r[key] for r in rows for key in ('image', 'tensor_file')]
    write(args.out/'manifest.json', {path.name: sha(path) for path in files})
    with zipfile.ZipFile(args.out/'review.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files+[args.out/'manifest.json']:
            archive.write(path, path.name)
    verify(args)
    print('SCHEDULER_FP32_PROBE_COMPLETE', sha(args.out/'review.zip'), flush=True)


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
    args.bridge, args.fixed, args.trace = auto/'scaled_direction_bridge', auto/'scaled_projection_retry', auto/'scaled_cast_trace'
    args.out = (args.out or auto/'scaled_scheduler_fp32_probe').resolve()
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
