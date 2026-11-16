"""单条件冻结f68反传复现；只比较执行因素，不创建优化器或新权重。"""
import argparse
import copy
import hashlib
import os
import shutil
import socket
import subprocess
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from tools.e2_b_endpoint_losses import TASKS, task_gradients
from tools.e2_state_b_amplitude_response import CHECKPOINT_SHA, read, write, sha, pixels
from tools.e2_state_b_pair_retry import PROTOCOL_SHA


SELECTED = ('validation_1526', 'r4157_native', 'a075')
VARIANTS = (
    dict(name='clean0', threads=1, pixels_before=False, loss_scale=1.),
    dict(name='clean1', threads=1, pixels_before=False, loss_scale=1.),
    dict(name='pixels_before', threads=1, pixels_before=True, loss_scale=1.),
    dict(name='threads2', threads=2, pixels_before=False, loss_scale=1.),
    dict(name='loss_scale1024', threads=1, pixels_before=False, loss_scale=1024.),
)


def tensor_sha(value):
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def compare_tensor(current, reference):
    a, b = current.detach().double().reshape(-1), reference.detach().double().reshape(-1)
    assert a.shape == b.shape
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not finite:
        return dict(finite=False, equal=False, max_abs=None, relative_l2=None, cosine=None)
    na, nb = a.norm(), b.norm()
    return dict(finite=True, equal=bool(torch.equal(a, b)), max_abs=float((a-b).abs().max()),
                relative_l2=float((a-b).norm() / nb.clamp_min(1e-18)),
                cosine=float(a @ b / (na*nb).clamp_min(1e-18)))


def vector_summary(value, directions=None):
    flat = value.detach().double().reshape(-1)
    finite = bool(torch.isfinite(flat).all())
    nonzero = flat[torch.isfinite(flat) & (flat != 0)].abs()
    result = dict(finite=finite, elements=flat.numel(), nonfinite=int((~torch.isfinite(flat)).sum()),
                  zero=int((flat == 0).sum()), norm=float(flat.norm()) if finite else None,
                  min_abs_nonzero=float(nonzero.min()) if len(nonzero) else None,
                  max_abs=float(flat.abs().max()) if finite else None)
    if directions is not None:
        result['directional'] = {name: float(flat @ direction.double()) if finite else None
                                 for name, direction in directions.items()}
    return result


def scaled_gradients(tasks, parameters, scale):
    # scale=1 原样调用，不在原计算图上额外乘1；不改变旧loss实现。
    values = tasks if scale == 1 else {key: tasks[key] * scale for key in TASKS}
    return task_gradients(values, parameters) / scale if scale != 1 else task_gradients(values, parameters)


def gradients_with_head(objective, entry, parameters, scale=1., before_backward=None):
    """先求原7项G；再用独立终latent叶子图求q，不让q先干扰原G路径。"""
    predicted, base = objective.rollout(entry)
    tasks, _ = objective.score_latents(entry, predicted, base, residue=objective.metric(entry).surrogate)
    values = {key: float(tasks[key].detach()) for key in TASKS}
    final, no_b = predicted.detach(), base.detach()
    if before_backward is not None:
        before_backward(final)
    gradient = scaled_gradients(tasks, parameters, scale).cpu()
    del tasks, predicted, base
    leaf = final.clone().requires_grad_(True)
    head, _ = objective.score_latents(entry, leaf, no_b, residue=objective.metric(entry).surrogate)
    head_values = {key: float(head[key].detach()) for key in TASKS}
    q = scaled_gradients(head, [leaf], scale).cpu()
    return dict(gradient=gradient, q=q, values=values, head_values=head_values,
                final=final.cpu(), base=no_b.cpu())


def prepare(args):
    assert not (args.out / 'diagnostic_protocol.json').exists(), '不覆盖已有诊断'
    spec = read(args.fixed / 'protocol.json')
    provenance = read(args.fixed / 'run_provenance.json')
    assert sha(args.fixed / 'protocol.json') == PROTOCOL_SHA
    assert sha(args.checkpoint) == CHECKPOINT_SHA
    pair = next(p for p in spec['train'] if p['case_id'] == SELECTED[0])
    item = next(p for p in spec['inputs'] if (p['case'], p['condition'], p['arm']) == SELECTED)
    args.out.mkdir(parents=True, exist_ok=True)
    input_hashes = {str(args.checkpoint): CHECKPOINT_SHA,
                    str(args.fixed / 'protocol.json'): PROTOCOL_SHA,
                    str(args.fixed / 'run_provenance.json'): sha(args.fixed / 'run_provenance.json')}
    copied = {}
    for name, expected in spec['assets'].items():
        if name.startswith('cases/' + SELECTED[0] + '/'):
            source, dest = args.fixed / name, args.out / name
            assert sha(source) == expected
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
            assert sha(dest) == expected
            input_hashes[str(source)] = copied[name] = expected
    shutil.copyfile(args.fixed / 'protocol.json', args.out / 'protocol.json')
    copied['protocol.json'] = PROTOCOL_SHA
    for name in ('gradients/shard3.pt', 'gradients/shard3.json'):
        source = args.fixed / name
        assert sha(source) == provenance['copied_sha256'][name]
        input_hashes[str(source)] = sha(source)
    for name in ('training/proposals.json', 'training/C0.pt', 'training/C1.pt'):
        input_hashes[str(args.fixed / name)] = sha(args.fixed / name)
    amplitude = args.project / 'output_eval/e2_state_b_amplitude_response_20261116'
    assert sha(amplitude / 'result.json') == spec['amplitude_source_result_sha256']
    assert sha(amplitude / 'generated.json') == spec['amplitude_source_generated_sha256']
    for name in ('result.json', 'generated.json'):
        input_hashes[str(amplitude / name)] = sha(amplitude / name)
    source = Path(__file__).resolve().parents[1]
    names = ('tools/e2_state_b_backward_repro.py', 'tools/e2_state_b_pair_finetune.py',
             'tools/e2_state_b_endpoint.py', 'tools/e2_b_endpoint_losses.py',
             'tools/e2_state_b_material.py', 'tools/e2_state_b_garment.py',
             'tools/e2_state_b_repair.py', 'tools/e2_state_b_direction_diagnostic.py',
             'tools/e2_b_direction_diagnostics.py', 'tools/e2_state_b_pilot.py',
             'models/e2_state_texture_b.py', 'scripts/e2_state_b_backward_repro.sbatch')
    protocol = dict(case=pair['case_id'], item=item, variants=list(VARIANTS),
        checkpoint_sha256=CHECKPOINT_SHA, original_protocol_sha256=PROTOCOL_SHA,
        input_sha256=input_hashes, copied_sha256=copied,
        source_sha256={name: sha(source / name) for name in names},
        execution_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        q_definition='independent final-latent leaf graph, built only after original seven parameter gradients; same loss head and detached no-B base',
        scale_policy='multiply scalar loss by1024 and divide resulting G/q by1024; diagnostic only, unchanged forward and no optimizer',
        target='one fixed condition and one fresh f68 capture reused by all independent graphs',
        optimizer_updates=0, new_weights=False, holdout_used=False,
        planned_parameter_reverse_passes=35, planned_head_reverse_passes=35,
        archive_excludes='all .pt tensors; diagnostic G/q/state tensors remain on server')
    write(args.out / 'diagnostic_protocol.json', protocol)
    print('BACKWARD_REPRO_PREPARED', SELECTED, flush=True)
    return protocol


def run(args):
    from models.e2_state_texture_b import StateTextureB
    from tools import e2_state_b_pilot as pilot
    from tools import e2_state_b_endpoint as endpoint
    from tools import e2_state_b_repair as repair
    from tools.e2_state_b_pair_finetune import Objective
    from tools.e2_state_b_direction_diagnostic import load_vectors
    from tools.e2_b_endpoint_losses import set_parameters
    from tools.e2_oracle_texture import weights_digest
    from tools.e2_fft_experiment import environment

    started = time.monotonic()
    protocol = prepare(args)
    spec = read(args.out / 'protocol.json')
    pair = next(p for p in spec['train'] if p['case_id'] == SELECTED[0])
    item = protocol['item']
    condition = item['sample_condition']
    shard = torch.load(args.fixed / 'gradients/shard3.pt', map_location='cpu', weights_only=False)
    assert shard['b_hash'] and shard['checkpoint_sha256'] == CHECKPOINT_SHA and shard['case'] == pair['case_id']
    index = next(i for i, row in enumerate(shard['items']) if row['sample_condition'] == condition)
    cached = shard['gradients'][index]
    cached_values = next(row['tasks'] for row in shard['values'] if row['sample_condition'] == condition)
    start, targets, initial = load_vectors(args)
    directions = {name: target.double() - start.double() for name, target in targets.items()}
    runtime = copy.copy(args)
    runtime.action = 'train'
    core, pipe, net = pilot.load_core(runtime)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    b = StateTextureB(color_channels=2, reference_channels=192).to(args.device)
    parameters = list(b.parameters())
    set_parameters(parameters, start)
    b.eval().requires_grad_(True)
    net.eval().requires_grad_(False)
    assert weights_digest(b) == initial == shard['b_hash']
    frozen = weights_digest(net)
    assert frozen == shard['net_hash']
    picture, digest, tap, latent = endpoint.final_sample(runtime, core, pipe, net, pair, condition, b)
    amplitude = args.project / 'output_eval/e2_state_b_amplitude_response_20261116'
    expected = pixels(amplitude / 'images' / SELECTED[2] / (SELECTED[0] + '_' + SELECTED[1] + '.png'))
    assert np.array_equal(np.asarray(picture), expected)
    reference_row = next(row for row in read(amplitude / 'generated.json')['rows']
                         if (row['case'], row['condition'], row['arm']) == SELECTED)
    assert digest == reference_row['noise_digest'] and len(tap.states) == 25
    picture.save(args.out / 'baseline.png')
    entry = dict(pair=pair, condition=condition, states=tap.states, digest=digest, final=latent.detach().cpu())
    torch.save(entry, args.out / 'captured_states.pt')
    pipe.vae.half()
    predictor = repair.FrozenPredictor(runtime, core, pipe, net, [pair], conditions=[condition])
    objective = Objective(runtime, pipe, predictor, b, pair, [item])
    pipe.scheduler.set_timesteps(50, device=args.device)

    def snapshot():
        return dict(b=weights_digest(b), net=weights_digest(net), vae=weights_digest(pipe.vae),
                    vae_dtype=str(pipe.vae.dtype), b_training=b.training, net_training=net.training,
                    b_requires_grad=all(p.requires_grad for p in parameters),
                    net_requires_grad=any(p.requires_grad for p in net.parameters()),
                    vae_requires_grad=any(p.requires_grad for p in pipe.vae.parameters()))

    fixed_snapshot = snapshot()
    records, tensors = [], {}
    for config in VARIANTS:
        variant_start = time.monotonic()
        torch.set_num_threads(config['threads'])
        before = snapshot()
        assert before == fixed_snapshot
        trace, pixel_event = [], {}

        def trace_b(module, positional, keywords, output):
            trace.append(dict(timestep=int(keywords['timestep']),
                              z_sha256=tensor_sha(positional[0]), x0_sha256=tensor_sha(positional[1]),
                              delta_sha256=tensor_sha(output), z_dtype=str(positional[0].dtype),
                              delta_dtype=str(output.dtype)))

        def before_backward(final):
            pixel_event['before'] = dict(vae=weights_digest(pipe.vae), dtype=str(pipe.vae.dtype))
            image = objective.pixels(final)
            pixel_event['after'] = dict(vae=weights_digest(pipe.vae), dtype=str(pipe.vae.dtype))
            pixel_event['png_equal'] = bool(np.array_equal(image, expected))
            assert pixel_event['png_equal']

        callback = before_backward if config['pixels_before'] else None
        with b.register_forward_hook(trace_b, with_kwargs=True):
            value = gradients_with_head(objective, entry, parameters, config['loss_scale'], callback)
        assert len(trace) == 25
        latent_error = float((value['final'].float() - entry['final'].float()).abs().max())
        task_error = max(abs(value['values'][t] - cached_values[t]) for t in TASKS)
        head_error = max(abs(value['head_values'][t] - value['values'][t]) for t in TASKS)
        assert latent_error <= 2e-4 and task_error <= 1e-6 and head_error <= 1e-6
        # 两种反传完成后才做常规PNG控制，避免无意切换活图共享VAE精度。
        final_image = objective.pixels(value['final'].to(args.device))
        assert np.array_equal(final_image, expected)
        Image.fromarray(final_image).save(args.out / (config['name'] + '.png'))
        after = snapshot()
        assert after == before
        assert not any(p.grad is not None for p in parameters)
        path = args.out / (config['name'] + '_derivatives.pt')
        torch.save(dict(gradient=value['gradient'], q=value['q'], config=config,
                        protocol_sha256=sha(args.out / 'diagnostic_protocol.json')), path)
        tensors[config['name']] = dict(gradient=value['gradient'], q=value['q'])
        clean = tensors['clean0']
        record = dict(config, torch_threads=torch.get_num_threads(), values=value['values'],
            head_values=value['head_values'], finite=bool(torch.isfinite(value['gradient']).all() and torch.isfinite(value['q']).all()),
            controls=dict(latent_max_abs=latent_error, task_max_abs=task_error, head_task_max_abs=head_error,
                          png_equal=True, b_calls=25, forward_trace_equal_to_clean0=not records or trace == records[0]['trace']),
            hashes_before=before, hashes_after=after, pixels_event=pixel_event, trace=trace,
            tensor_file=path.name, tensor_sha256=sha(path), seconds=time.monotonic()-variant_start,
            gradients={}, q={})
        for i, task in enumerate(TASKS):
            record['gradients'][task] = dict(summary=vector_summary(value['gradient'][i], directions),
                vs_cached=compare_tensor(value['gradient'][i], cached[i]),
                vs_clean0=compare_tensor(value['gradient'][i], clean['gradient'][i]))
            record['q'][task] = dict(summary=vector_summary(value['q'][i]),
                                     vs_clean0=compare_tensor(value['q'][i], clean['q'][i]))
        records.append(record)
        write(args.out / 'partial.json', dict(records=records, optimizer_updates=0, complete=False))
        print('BACKWARD_REPRO_VARIANT', config['name'], record['finite'], record['seconds'], flush=True)
        del value
    comparisons = {}
    for left in tensors:
        for right in tensors:
            if left >= right:
                continue
            comparisons[left + '__' + right] = {field: {task: compare_tensor(tensors[right][field][i], tensors[left][field][i])
                for i, task in enumerate(TASKS)} for field in ('gradient', 'q')}
    assert all(sha(Path(name)) == expected for name, expected in protocol['input_sha256'].items())
    assert all(sha(args.out / name) == expected for name, expected in protocol['copied_sha256'].items())
    source = Path(__file__).resolve().parents[1]
    assert all(sha(source / name) == expected for name, expected in protocol['source_sha256'].items())
    result = dict(protocol=protocol, records=records, comparisons=comparisons,
        cached_directional={task: vector_summary(cached[i], directions) for i, task in enumerate(TASKS)},
        environment=environment(), optimizer_updates=0, new_weights=False, frozen_unchanged=True,
        baseline_png_equal=True, holdout_used=False, fresh_gradients_are_ground_truth=False,
        reverse_passes=dict(parameter=35, head=35), seconds=time.monotonic()-started,
        interpretation='Execution-factor diagnostic only; equal forward does not establish equal backward; scaled loss is not a training change',
        hardware=dict(hostname=socket.gethostname(), device=args.device,
            slurm_job_id=os.environ.get('SLURM_JOB_ID'), slurm_nodelist=os.environ.get('SLURM_JOB_NODELIST'),
            gpu_name=torch.cuda.get_device_name(args.device),
            compute_capability=list(torch.cuda.get_device_capability(args.device)),
            gpu_memory_bytes=torch.cuda.get_device_properties(args.device).total_memory),
        numeric_configuration=dict(cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_benchmark=torch.backends.cudnn.benchmark, matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_tf32=torch.backends.cudnn.allow_tf32),
        reproducibility_scope='same-process clean repetitions only; bitwise agreement does not establish cross-job reproducibility',
        tensor_files_server_only=True)
    write(args.out / 'result.json', result)
    files = sorted(path for path in args.out.rglob('*') if path.is_file() and path.suffix in ('.json', '.png'))
    manifest = {path.relative_to(args.out).as_posix(): sha(path) for path in files}
    write(args.out / 'manifest.json', manifest)
    with zipfile.ZipFile(args.out / 'review.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files + [args.out / 'manifest.json']:
            archive.write(path, path.relative_to(args.out).as_posix())
    print('BACKWARD_REPRO_COMPLETE', sha(args.out / 'review.zip'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--project', type=Path, default=Path('/share/home/u2515283058/Mymodel'))
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.prior = args.prior.resolve()
    args.fixed = args.prior / 'projection_fp64_retry'
    args.out = (args.out or args.prior / 'autonomous_repair_20261116/backward_repro').resolve()
    args.data = args.project.parent / 'datasets/BF'
    args.checkpoint = args.project / 'output_eval/e2_state_b_pastel_followup_20261115/trust_retry/training/step1.pt'
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    run(args)


if __name__ == '__main__':
    main()
