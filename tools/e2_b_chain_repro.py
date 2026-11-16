"""固定终latent的residue q，只诊断rollout的J^Tq重复性，不计算loss或更新参数。"""
import argparse
import copy
import gc
import os
import socket
import subprocess
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import torch

from tools.e2_b_endpoint_losses import TASKS, set_parameters
from tools.e2_state_b_backward_repro import compare_tensor, tensor_sha, vector_summary, SELECTED
from tools.e2_state_b_amplitude_response import CHECKPOINT_SHA, read, write, sha


GROUPS = (
    dict(name='original_full', start=0, hops=25, scale=1., math_only=False),
    dict(name='original_last_b', start=24, hops=1, scale=1., math_only=False),
    dict(name='original_full_scale1024', start=0, hops=25, scale=1024., math_only=False),
    dict(name='math_full', start=0, hops=25, scale=1., math_only=True),
)
CAPTURED_SHA = '32d9a3da5d38e6135a7a88b4936139094d0aea2243009557b5290bc2f21b319b'
PREVIOUS_RESULT_SHA = '6ad1cd43cf7ad68a5f3e86bdf31bd509851c1fbe6d403f90c559f80207d12c74'


def backend_flags():
    return dict(deterministic=torch.are_deterministic_algorithms_enabled(),
        warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        flash=torch.backends.cuda.flash_sdp_enabled(),
        efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
        math=torch.backends.cuda.math_sdp_enabled(), cudnn=torch.backends.cuda.cudnn_sdp_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32)


def repeated_vjp(predicted, parameters, q, repeats=2, scale=1., terminal_hook=False):
    """同一外层图重复；原non-reentrant checkpoint在每次反传中仍会重算UNet。"""
    assert q.shape == predicted.shape and not q.requires_grad
    seed = q if scale == 1 else q * scale
    rows, terminals = [], []
    active = []
    def capture(grad_inputs, grad_outputs):
        native, incoming = grad_inputs[0].detach(), grad_outputs[0].detach()
        assert native.dtype == torch.float16 and incoming.dtype == torch.float32
        active.append(dict(native=native.cpu(), incoming=incoming.cpu()))
    handle = None
    if terminal_hook:
        assert predicted.dtype == torch.float32 and type(predicted.grad_fn).__name__ == 'ToCopyBackward0'
        handle = predicted.grad_fn.register_hook(capture)
    try:
        for index in range(repeats):
            active.clear()
            parts = torch.autograd.grad(predicted, parameters, grad_outputs=seed,
                retain_graph=index + 1 < repeats, allow_unused=True)
            gradient = torch.cat([torch.zeros_like(p).flatten() if g is None else g.detach().flatten()
                                  for p, g in zip(parameters, parts)])
            rows.append((gradient / scale if scale != 1 else gradient).cpu())
            if terminal_hook:
                assert len(active) == 1
                native, incoming = active[0]['native'], active[0]['incoming']
                terminals.append(dict(native_summary=vector_summary(native),
                    incoming_summary=vector_summary(incoming), native_sha256=tensor_sha(native),
                    incoming_sha256=tensor_sha(incoming),
                    matches_expected_cast=bool(torch.equal(native, incoming.half())),
                    zero_from_nonzero_seed=int(((native == 0) & (incoming != 0)).sum()),
                    native_unscaled=(native.float() / scale).reshape(-1)))
            assert all(p.grad is None for p in parameters)
    finally:
        if handle is not None:
            handle.remove()
    return rows, terminals


def prepare(args):
    assert not args.out.exists(), '不覆盖已有诊断目录'
    previous = read(args.previous / 'result.json')
    manifest = read(args.previous / 'manifest.json')
    assert manifest['result.json'] == sha(args.previous / 'result.json') == PREVIOUS_RESULT_SHA
    old_protocol = previous['protocol']
    clean = next(row for row in previous['records'] if row['name'] == 'clean0')
    inputs = {str(args.previous / name): value for name, value in manifest.items()}
    inputs[str(args.previous / 'manifest.json')] = sha(args.previous / 'manifest.json')
    inputs.update(old_protocol['input_sha256'])
    inputs[str(args.previous / clean['tensor_file'])] = clean['tensor_sha256']
    inputs[str(args.previous / 'captured_states.pt')] = args.captured_sha
    assert len(args.captured_sha) == 64 and all(c in '0123456789abcdef' for c in args.captured_sha)
    assert sha(args.checkpoint) == CHECKPOINT_SHA
    assert all(sha(Path(name)) == value for name, value in inputs.items())
    source = Path(__file__).resolve().parents[1]
    names = set(old_protocol['source_sha256']) | {
        'tools/e2_b_chain_repro.py', 'scripts/e2_b_chain_repro.sbatch',
        'tools/e2_fft_experiment.py', 'tools/e2_state_b_joint_data.py',
        'tools/e2_state_b_material_data.py', 'tools/bf_t3_frozen_core.py',
        'tools/m1_evaluation.py', 'models/bf_sd_skip.py', 'adapter/attention_processor.py'}
    args.out.mkdir(parents=True)
    protocol = dict(selected=SELECTED, groups=list(GROUPS), q_task='residue', q_source=clean['tensor_file'],
        input_sha256=inputs, source_sha256={n: sha(source / n) for n in sorted(names)},
        captured_sha256_receipt=args.captured_sha,
        execution_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        planned_vjp=12, loss_evaluations=0, optimizer_updates=0, new_weights=False,
        graph_repeats='graph0 twice, independent graph1 once for each group',
        interpretation='fixed q; same graph still recomputes checkpointed UNet; last-step B does not test 25 shared-parameter accumulations; math forward may differ; neither backend is assumed ground truth',
        assets='read-only references to completed backward_repro assets; no PNG or score call',
        tensor_files_server_only=True)
    write(args.out / 'diagnostic_protocol.json', protocol)
    return protocol, clean


def run(args):
    from models.e2_state_texture_b import StateTextureB
    from tools import e2_state_b_endpoint as endpoint, e2_state_b_pilot as pilot
    from tools import e2_state_b_repair as repair, e2_state_b_joint_data as data
    from tools.e2_state_b_direction_diagnostic import load_vectors
    from tools.e2_oracle_texture import weights_digest
    from tools.e2_fft_experiment import environment

    started = time.monotonic()
    protocol, clean = prepare(args)
    derivatives = torch.load(args.previous / clean['tensor_file'], map_location='cpu', weights_only=False)
    assert derivatives['protocol_sha256'] == sha(args.previous / 'diagnostic_protocol.json')
    entry = torch.load(args.previous / 'captured_states.pt', map_location='cpu', weights_only=False)
    assert entry['pair']['case_id'] == SELECTED[0] and len(entry['states']) == 25
    assert entry['condition'] == read(args.previous / 'diagnostic_protocol.json')['item']['sample_condition']
    q = derivatives['q'][TASKS.index('residue')].reshape(entry['final'].shape).to(args.device)
    assert torch.isfinite(q).all() and q.abs().max() > 0
    historical_g = derivatives['gradient'][TASKS.index('residue')]
    start, targets, initial = load_vectors(args)
    directions = {name: target.double() - start.double() for name, target in targets.items()}
    runtime = copy.copy(args)
    runtime.action, runtime.out = 'train', args.previous
    core, pipe, net = pilot.load_core(runtime)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    b = StateTextureB(color_channels=2, reference_channels=192).to(args.device)
    parameters = list(b.parameters())
    set_parameters(parameters, start)
    b.eval().requires_grad_(True)
    net.eval().requires_grad_(False)
    assert weights_digest(b) == initial == clean['hashes_before']['b']
    assert weights_digest(net) == clean['hashes_before']['net']
    # 只构建rollout所需的同一份固定条件，完全跳过Objective的VAE loss head。
    pipe.vae.half()
    predictor = repair.FrozenPredictor(runtime, core, pipe, net, [entry['pair']], conditions=[entry['condition']])
    values, _ = data.inputs(runtime, entry['pair'], entry['condition'], 'raw')
    pipe.vae.float()  # 与原clean0开始反传时的共享模块dtype一致；rollout不调用VAE。
    objective = SimpleNamespace(args=runtime, pipe=pipe, b=b, predictor=predictor,
        records={(entry['pair']['case_id'], entry['condition']): (None, None, None, values)})
    pipe.scheduler.set_timesteps(50, device=args.device)
    def snapshot():
        return dict(b=weights_digest(b), net=weights_digest(net), vae=weights_digest(pipe.vae),
            vae_dtype=str(pipe.vae.dtype), b_training=b.training, net_training=net.training,
            b_requires_grad=all(p.requires_grad for p in parameters),
            net_requires_grad=any(p.requires_grad for p in net.parameters()),
            vae_requires_grad=any(p.requires_grad for p in pipe.vae.parameters()))
    initial_snapshot = snapshot()
    assert initial_snapshot == clean['hashes_before']
    original_flags = backend_flags()
    assert original_flags['deterministic'] and not original_flags['warn_only']
    assert original_flags['efficient'] and original_flags['math'] and not original_flags['flash'] and not original_flags['cudnn']
    records, vectors, native_vectors = [], {}, {}
    for config in GROUPS:
        group_start = time.monotonic()
        torch.backends.cuda.enable_mem_efficient_sdp(not config['math_only'])
        torch.backends.cuda.enable_math_sdp(True)
        torch.cuda.reset_peak_memory_stats(args.device)
        record = dict(config, backend=backend_flags(), before=snapshot(), graphs=[], status='running')
        try:
            for graph_index, repeats in enumerate((2, 1)):
                trace = []
                def trace_b(module, positional, keywords, output):
                    trace.append(dict(timestep=int(keywords['timestep']), z_sha256=tensor_sha(positional[0]),
                        x0_sha256=tensor_sha(positional[1]), delta_sha256=tensor_sha(output),
                        z_dtype=str(positional[0].dtype), delta_dtype=str(output.dtype)))
                with b.register_forward_hook(trace_b, with_kwargs=True):
                    predicted, base = endpoint.Objective.rollout(objective, entry, start=config['start'], hops=config['hops'])
                final = predicted.detach().cpu()
                control = compare_tensor(final, entry['final'])
                assert len(trace) == config['hops']
                if not config['math_only']:
                    assert control['equal'], '原backend必须精确复现历史latent'
                    expected_trace = clean['trace'][config['start']:config['start'] + config['hops']]
                    assert trace == expected_trace, '原backend的逐步前向发生改变'
                gradients, terminals = repeated_vjp(predicted, parameters, q, repeats,
                    config['scale'], terminal_hook=True)
                graph = dict(index=graph_index, latent_sha256=tensor_sha(final), historical_latent=control,
                    trace=trace, vjp=[])
                for repeat_index, (gradient, terminal) in enumerate(zip(gradients, terminals)):
                    name = '%s_g%d_r%d' % (config['name'], graph_index, repeat_index)
                    vectors[name] = gradient
                    native_vectors[name] = terminal.pop('native_unscaled')
                    graph['vjp'].append(dict(name=name, summary=vector_summary(gradient, directions),
                        vs_historical_clean0=compare_tensor(gradient, historical_g), terminal_cast=terminal))
                record['graphs'].append(graph)
                del predicted, base, gradients, terminals
            record['status'] = 'complete'
        except torch.cuda.OutOfMemoryError as error:
            if not config['math_only']:
                raise
            record.update(status='oom', failure=str(error))
            predicted = base = None
        gc.collect()
        torch.cuda.empty_cache()
        record.update(after=snapshot(), seconds=time.monotonic() - group_start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device))
        assert record['after'] == record['before'] == initial_snapshot
        records.append(record)
        write(args.out / 'partial.json', dict(records=records, complete=False, optimizer_updates=0))
        print('CHAIN_REPRO_GROUP', config['name'], record['status'], record['seconds'], flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    torch.backends.cuda.enable_mem_efficient_sdp(original_flags['efficient'])
    tensor_path = args.out / 'derivatives.pt'
    torch.save(dict(gradients=vectors, terminal_native_unscaled=native_vectors,
                    q_sha256=tensor_sha(q), protocol_sha256=sha(args.out / 'diagnostic_protocol.json')), tensor_path)
    comparisons = {left + '__' + right: compare_tensor(vectors[right], vectors[left])
        for i, left in enumerate(vectors) for right in list(vectors)[i+1:]}
    native_comparisons = {left + '__' + right: dict(compare_tensor(native_vectors[right], native_vectors[left]),
        zero_recovered=int(((native_vectors[left] == 0) & (native_vectors[right] != 0)).sum()),
        nonzero_lost=int(((native_vectors[left] != 0) & (native_vectors[right] == 0)).sum()))
        for i, left in enumerate(native_vectors) for right in list(native_vectors)[i+1:]}
    source = Path(__file__).resolve().parents[1]
    assert all(sha(Path(n)) == h for n, h in protocol['input_sha256'].items())
    assert all(sha(source / n) == h for n, h in protocol['source_sha256'].items())
    result = dict(protocol=protocol, records=records, comparisons=comparisons,
        terminal_comparisons=native_comparisons, completed_vjp=len(vectors), loss_evaluations=0,
        optimizer_updates=0, new_weights=False, frozen_unchanged=True, q_sha256=tensor_sha(q),
        q_summary=vector_summary(q), tensor_file=tensor_path.name, tensor_sha256=sha(tensor_path),
        seconds=time.monotonic()-started, environment=environment(), backend_restored=backend_flags(),
        hardware=dict(hostname=socket.gethostname(), job=os.environ.get('SLURM_JOB_ID'),
            node=os.environ.get('SLURM_JOB_NODELIST'), gpu=torch.cuda.get_device_name(args.device),
            total_memory=torch.cuda.get_device_properties(args.device).total_memory),
        reproducibility_scope='same-process only; no cross-job or mathematical ground-truth claim')
    write(args.out / 'result.json', result)
    files = sorted(args.out.glob('*.json'))
    write(args.out / 'manifest.json', {p.name: sha(p) for p in files})
    with zipfile.ZipFile(args.out / 'review.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files + [args.out / 'manifest.json']:
            archive.write(path, path.name)
    print('CHAIN_REPRO_COMPLETE', sha(args.out / 'review.zip'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--captured-sha', default=CAPTURED_SHA)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--project', type=Path, default=Path('/share/home/u2515283058/Mymodel'))
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.prior = args.prior.resolve()
    args.previous = args.prior / 'autonomous_repair_20261116/backward_repro'
    args.fixed = args.prior / 'projection_fp64_retry'
    args.out = (args.out or args.prior / 'autonomous_repair_20261116/chain_repro').resolve()
    args.data = args.project.parent / 'datasets/BF'
    args.checkpoint = args.project / 'output_eval/e2_state_b_pastel_followup_20261115/trust_retry/training/step1.pt'
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    run(args)


if __name__ == '__main__':
    main()
