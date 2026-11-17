"""原生FP16解码诊断：f68/G1各48保存latent观测，f68零4616头32 VJP，零更新。"""
import argparse
import copy
import inspect
import json
from pathlib import Path
import subprocess
import time
import zipfile

import cv2
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from tools import e2_reference_terminal_global_amplitude as global_run
from tools import e2_reference_terminal_one_step as source
from tools.e2_b_endpoint_losses import TASKS, flatten
from tools.e2_state_b_scaled_retry import SENTINELS, scaled_scalar, NumericalFailure
from tools.e2_state_b_backward_repro import tensor_sha, compare_tensor, vector_summary
from tools.e2_state_b_amplitude_response import read, write, sha, pixels, CHECKPOINT_SHA
from tools.e2_oracle_texture import weights_digest


SOURCE = Path(__file__).resolve().parents[1]
LOSSES = ('global_amplitude', 'residue')
EXTRA = ('tools/e2_reference_terminal_r16_diagnostic.py', 'scripts/e2_reference_terminal_r16_diagnostic.sbatch')


def native_rgb_half(vae, latent):
    """不改dtype/梯度模式；调用方必须在整张AD图及重算释放后才恢复VAE。"""
    assert vae.dtype == torch.float16
    decoded = vae.decode(latent.half()/vae.config.scaling_factor, return_dict=False)[0]
    assert decoded.dtype == torch.float16
    return (decoded/2+.5).clamp(0, 1)


def quantize(rgb_half):
    assert rgb_half.dtype == torch.float16 and rgb_half.shape[0] == 1
    value = rgb_half.detach()[0].permute(1, 2, 0).float().cpu().numpy()
    return np.rint(value*255).astype(np.uint8)


def repeat_report(current, reference):
    report = compare_tensor(current, reference)
    report['passed'] = bool(report['finite'] and report['relative_l2'] <= .005 and report['cosine'] >= .99999)
    return report


@torch.no_grad()
def scalar_stations(native, image, positive, weight, metric, fp32, official):
    """两次原标量计算；R32及官方PNG指标只引用旧记录，不重解码或换指标。"""
    rgb8 = torch.from_numpy(image.copy()).permute(2, 0, 1)[None].to(native.device).float()/255
    evaluate = lambda rgb: dict(global_amplitude=float(global_run.global_loss(rgb, positive, weight)),
        residue=float(metric.surrogate(rgb)))
    return dict(R32_metadata=fp32, R16=evaluate(native.float()), R16_uint8_analytic=evaluate(rgb8),
        official_PNG={key: official[key] for key in ('l_amplitude', 'l_correlation', 'l_target_gain', 'old_projection')},
        interpretation='original scalar at three RGB representations; official PNG metrics have different definitions/units')


def prior_args(args):
    value = copy.copy(args); value.out = args.global_source
    return value


def source_inputs(args):
    previous = prior_args(args)
    protocol = global_run.verify(previous)  # 完整祖先验证只在本进程首/末各一次。
    report = read(args.global_source/'result.json')
    assert report['status'] == 'complete_replay' and (report['new_reverse_passes'], report['complete_s0_images']) == (56, 48)
    global_g, values, _, qualification = global_run.gradient_inputs(previous)
    assert qualification == read(args.global_source/'global_qualification.json') and qualification['passed']
    spec = read(args.source_run/'protocol.json')
    identities = [(item['case'], item['sample_condition']) for item in spec['inputs']]
    assert len(identities) == global_g.shape[0] == 48
    return protocol, spec, dict(zip(identities, global_g)), values


def archive(args, result, tensors):
    torch.save(tensors, args.out/'diagnostic.pt')
    result['tensor_file_sha256'] = sha(args.out/'diagnostic.pt')
    write(args.out/'result.json', result)
    names = ['diagnostic_protocol.json', 'diagnostic.pt', 'result.json']
    write(args.out/'manifest.json', {name: sha(args.out/name) for name in names})
    with zipfile.ZipFile(args.out/'review.zip', 'w', zipfile.ZIP_DEFLATED) as output:
        for name in names+['manifest.json']:
            output.write(args.out/name, name)


def run(args):
    from tools import e2_state_b_pilot as pilot
    from tools.e2_fft_experiment import environment
    started = time.monotonic(); previous, spec, fp32_global, global_values = source_inputs(args)
    assert not args.out.exists(), '不覆盖已有R16诊断'
    source_overlay = read(args.source_run/'one_step_overlay.json')
    baseline = read(args.source_run/'evaluation/f68/replay/result.json')
    baseline_rows = {(row['case'], row['sample_condition']): row for row in baseline['rows']}
    g1_report = read(args.global_source/'evaluation/G1/replay/result.json')
    g1_tensor_file = args.global_source/'evaluation/G1/replay/responses.pt'
    assert sha(g1_tensor_file) == g1_report['response_tensor_sha256']
    g1_latents = torch.load(g1_tensor_file, map_location='cpu', weights_only=False)
    g1_rows = {(row['case'], row['sample_condition']): row for row in g1_report['rows']}
    assert len(g1_rows) == 48 and set(g1_rows) == set(baseline_rows)
    names = [args.global_source/name for name in ('global_protocol.json', 'global_qualification.json', 'result.json')]
    names += [args.source_run/name for name in ('one_step_overlay.json', 'gradient_qualification.json', 'evaluation/f68/replay/result.json')]
    names += [folder/('gradients/shard%d.%s' % (i, ext)) for folder in (args.global_source, args.source_run)
        for i in range(4) for ext in ('json', 'pt')]
    names += [args.preflight/('cache/shard%d.pt' % i) for i in range(4)]
    names += [args.basis/('cache/shard%d.json' % i) for i in range(4)]
    vae_config_file = args.project/'models/stable-diffusion-v1-5/vae/config.json'
    names += [args.checkpoint, vae_config_file]
    names += [args.global_source/'evaluation/G1/replay/result.json', g1_tensor_file]
    for row in baseline['rows']:
        path = args.source_run/row['image']; assert sha(path) == row['png_sha256']; names.append(path)
    for row in g1_report['rows']:
        path = args.global_source/row['image']; assert sha(path) == row['png_sha256']; names.append(path)
    args.out.mkdir(parents=True)
    protocol = dict(source_point='f68 + four zero4616 increments, not saved G1/C1 candidate',
        parameter_count=source.PARAMETERS, losses=list(LOSSES), sentinels=[list(v) for v in SENTINELS],
        native_decode='latent.half()/scale -> half VAE -> half /2+.5 -> half clamp -> float for original scalar loss',
        target='original cached FP32 positive, mask, and GarmentProjection template unchanged',
        reference_gradients='G1 zero-point FP32 global; 125548 zero-point FP32 residue; different forwards, not AD error',
        loss_scale=1024., repeat_gate=dict(max_relative_l2=.005, minimum_cosine=.99999, finite=True),
        planned_f68_cache_pixel_controls=48, planned_G1_saved_latent_pixel_controls=48,
        planned_cache_pixel_controls=96, planned_new_reverse_passes=32, checkpoint_recomputation=True,
        G1_observation='saved final latent only; no G1 head loaded into inference/AD model',
        R32_observation='existing zero-point/source metadata only; no extra FP32 VAE decode',
        new_S0=0, new_full_B_or_UNet_reverse_passes=0, optimizer_steps=0, projection_calls=0,
        saved_candidate_updates=0, holdout_results_read=0, adopted_checkpoints=[],
        source_qualification_passed=True, source_quality_not_required_for_diagnostic=True,
        input_sha256={str(p.resolve()): sha(p) for p in names},
        source_sha256=dict(previous['source_sha256'], **{name: sha(SOURCE/name) for name in EXTRA}),
        execution_commit=subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip())
    write(args.out/'diagnostic_protocol.json', protocol)
    result = dict(status='incomplete', passed=False, controls=[], sentinels=[], model_controls=[],
        completed_new_reverse_passes=0, completed_cache_pixel_controls=0,
        completed_f68_cache_pixel_controls=0, completed_G1_saved_latent_pixel_controls=0,
        new_S0=0, optimizer_steps=0, projection_calls=0, saved_candidate_updates=0,
        new_full_B_or_UNet_reverse_passes=0, holdout_results_read=0, adopted_checkpoints=[],
        interpretation='repeatability/PNG parity and representation-stage observations; not AD error, quality adoption, or unique causality')
    tensors = dict(sentinels={}, parameter_count=source.PARAMETERS, parameter_order=list(source.INCREMENTS))
    error = None
    try:
        runtime = copy.copy(args); runtime.action = 'train'
        _, pipe, net = pilot.load_core(runtime); torch.set_num_threads(2); cv2.setNumThreads(2)
        net.eval().requires_grad_(False); pipe.vae.float().eval().requires_grad_(False)
        head = source.load_head(args, args.device); parameters = head.increment_parameters()
        assert sum(p.numel() for p in parameters) == 4616 and not torch.count_nonzero(flatten(parameters))
        base_hash, net_hash, vae_hash = weights_digest(head.base), weights_digest(net), weights_digest(pipe.vae)
        vae_config = json.loads(json.dumps(dict(pipe.vae.config))); config_file = read(vae_config_file)
        assert all(vae_config[k] == value for k, value in config_file.items() if not k.startswith('_'))
        assert vae_config['scaling_factor'] == .18215
        scheduler_file = Path(inspect.getsourcefile(type(pipe.scheduler))); scheduler_sha = sha(scheduler_file)
        result.update(environment=environment(), vae_config=vae_config, vae_config_file=config_file,
            vae_config_file_sha256=sha(vae_config_file), vae_hash_fp32=vae_hash, base_b_hash=base_hash, net_hash=net_hash,
            vae_class_source_sha256=sha(Path(inspect.getsourcefile(type(pipe.vae)))), scheduler_source_sha256=scheduler_sha)
        objective_runtime = copy.copy(runtime); objective_runtime.out = args.source_run
        for index, pair in enumerate(spec['train']):
            cache, _ = source.load_cache(args, index, source_overlay)
            metadata = read(args.basis/('cache/shard%d.json' % index))
            assert base_hash == cache['b_hash'] and net_hash == cache['net_hash'] and vae_hash == metadata['vae_hash_fp32']
            assert scheduler_sha == metadata['scheduler']['source_sha256']
            scheduler_audit = source.preflight.scheduler_configuration(pipe.scheduler.config, metadata['scheduler']['config'])
            assert scheduler_audit['normalized_equal']
            objective = source.objective_from_cache(objective_runtime, pipe, head, pair, cache)
            assert weights_digest(pipe.vae) == vae_hash
            pipe.scheduler.set_timesteps(50, device=args.device)
            old = torch.load(args.source_run/('gradients/shard%d.pt' % index), map_location='cpu', weights_only=False)
            old_rows = {(r['case'], r['sample_condition']): i for i, r in enumerate(old['values'])}
            items = [r for r in spec['inputs'] if r['case'] == pair['case_id']]
            pipe.vae.half(); half_hash = weights_digest(pipe.vae)
            try:
                for item in items:
                    case, c = item['case'], item['sample_condition']; key = case+'/'+c
                    record = cache['records'][c]; original = baseline_rows[case, c]
                    _, positive, _, _, weight, _ = objective.records[case, c]
                    metric = objective.metric(dict(pair=pair, condition=c)); position = old_rows[case, c]
                    with torch.no_grad():
                        latent, _, delta, corrected = source.preflight.terminal_replay(head, record, pipe.scheduler, args.device)
                        assert tensor_sha(latent) == tensor_sha(record['final']) and tensor_sha(delta) == tensor_sha(record['base_delta'])
                        assert tensor_sha(corrected) == record['corrected_epsilon_sha256']
                        native = native_rgb_half(pipe.vae, latent); image = quantize(native)
                        historical = pixels(args.source_run/original['image'])
                        phi = head.statistics(record['target_record'][3]['reference'].to(args.device))['phi']
                        saved_g1_latent = g1_latents['rows'][key]['latent']
                        assert tensor_sha(saved_g1_latent) == g1_rows[case, c]['latent_sha256']
                        native_g1 = native_rgb_half(pipe.vae, saved_g1_latent.to(args.device))
                        image_g1 = quantize(native_g1); historical_g1 = pixels(args.global_source/g1_rows[case, c]['image'])
                    control = dict(item, latent_sha256=tensor_sha(latent), corrected_epsilon_sha256=tensor_sha(corrected),
                        native_rgb_half_sha256=tensor_sha(native), pixel_array_sha256=tensor_sha(torch.from_numpy(image)),
                        historical_pixel_array_sha256=tensor_sha(torch.from_numpy(historical)), historical_png_sha256=original['png_sha256'],
                        pixel_equal=bool(np.array_equal(image, historical)), maximum_pixel_error=int(np.abs(image.astype(int)-historical.astype(int)).max()),
                        phi=float(phi), phi_sha256=tensor_sha(phi), source_noise_digest=record['noise_digest'])
                    control['f68_stations'] = scalar_stations(native, image, positive, weight, metric,
                        dict(global_amplitude=global_values[case, c]['global_loss'], residue=old['values'][position]['tasks']['residue']), original['scores'])
                    control['G1_observation'] = dict(latent_sha256=tensor_sha(saved_g1_latent),
                        native_rgb_half_sha256=tensor_sha(native_g1), pixel_array_sha256=tensor_sha(torch.from_numpy(image_g1)),
                        historical_pixel_array_sha256=tensor_sha(torch.from_numpy(historical_g1)), historical_png_sha256=g1_rows[case, c]['png_sha256'],
                        pixel_equal=bool(np.array_equal(image_g1, historical_g1)),
                        maximum_pixel_error=int(np.abs(image_g1.astype(int)-historical_g1.astype(int)).max()),
                        stations=scalar_stations(native_g1, image_g1, positive, weight, metric,
                            dict(global_amplitude=g1_rows[case, c]['global_loss'], residue=g1_rows[case, c]['training_tasks']['residue']),
                            g1_rows[case, c]['scores']))
                    result['controls'].append(control); result['completed_cache_pixel_controls'] += 2
                    result['completed_f68_cache_pixel_controls'] += 1; result['completed_G1_saved_latent_pixel_controls'] += 1
                    assert control['pixel_equal'] and control['phi_sha256'] == global_values[case, c]['phi_sha256']
                    assert control['G1_observation']['pixel_equal']
                    if source.preflight.old.key(item) in SENTINELS:
                        point = dict(native_rgb_half=native.cpu(), G1_native_rgb_half=native_g1.cpu(),
                            positive=positive.detach().cpu(), weight=weight.detach().cpu(),
                            residue_coefficients=metric.coefficients.detach().cpu(), residue_offset=metric.offset,
                            residue_energy=metric.energy, phi=phi.cpu(), reference={}, gradients={})
                        tensors['sentinels'][key] = point
                        point['reference'] = dict(global_amplitude=fp32_global[case, c],
                            residue=old['gradients'][position, TASKS.index('residue')])
                        task_records = {}
                        for task in LOSSES:
                            samples, numbers = [], []
                            for repeat in range(2):
                                z, _, _, _ = source.preflight.terminal_replay(head, record, pipe.scheduler, args.device)
                                half = checkpoint(lambda value: native_rgb_half(pipe.vae, value), z, use_reentrant=False)
                                assert tensor_sha(half) == control['native_rgb_half_sha256']
                                rgb = half.float()
                                loss = global_run.global_loss(rgb, positive, weight) if task == 'global_amplitude' else metric.surrogate(rgb)
                                if not torch.isfinite(loss):
                                    raise NumericalFailure(key+'/'+task+'/loss', loss)
                                value = float(loss.detach()); gradient = scaled_scalar(loss, parameters).detach().cpu()
                                result['completed_new_reverse_passes'] += 1
                                samples.append(gradient); numbers.append(value); point['gradients'][task] = torch.stack(samples)
                                # 这之后才能释放图或切换dtype；不调用pixels/Objective.rgb干扰活图。
                                del z, half, rgb, loss
                                if not torch.isfinite(gradient).all():
                                    raise NumericalFailure(key+'/'+task+'/'+str(repeat), gradient)
                            repeat_control = repeat_report(samples[1], samples[0]); assert numbers[0] == numbers[1]
                            reference = point['reference'][task]
                            task_records[task] = dict(value=numbers[0], reference_fp32_value=(global_values[case, c]['global_loss']
                                if task == 'global_amplitude' else old['values'][position]['tasks']['residue']),
                                repeats=repeat_control, gradient=vector_summary(samples[0]),
                                different_forward_comparison=compare_tensor(samples[0], reference))
                        result['sentinels'].append(dict(item, tasks=task_records))
                        assert all(v['repeats']['passed'] for v in task_records.values()), 'R16重复资格未通过'
                    del native, native_g1, latent, delta, corrected, phi
                    print('R16_CACHE', len(result['controls']), key, result['completed_new_reverse_passes'], flush=True)
            finally:
                # autograd.grad同步结束、活图已释放后恢复，不在checkpoint图内切dtype。
                pipe.vae.float()
                restored = weights_digest(pipe.vae)
                result['model_controls'].append(dict(case=pair['case_id'], vae_half_hash=half_hash,
                    vae_restored_fp32_hash=restored, scheduler=scheduler_audit,
                    zero_head=not bool(torch.count_nonzero(flatten(parameters))), frozen_unchanged=restored == vae_hash))
                assert restored == vae_hash
            assert weights_digest(pipe.vae) == vae_hash and weights_digest(head.base) == base_hash
            assert not torch.count_nonzero(flatten(parameters)) and not any(p.grad is not None for p in head.parameters())
        assert result['completed_cache_pixel_controls'] == 96 and result['completed_new_reverse_passes'] == 32
        assert result['completed_f68_cache_pixel_controls'] == result['completed_G1_saved_latent_pixel_controls'] == 48
        assert len(result['sentinels']) == 8 and {source.preflight.old.key(r) for r in result['sentinels']} == set(SENTINELS)
        assert weights_digest(net) == net_hash and weights_digest(head.base) == base_hash and weights_digest(pipe.vae) == vae_hash
        assert all(not p.requires_grad and p.grad is None for model in (net, pipe.vae, head.base) for p in model.parameters())
        assert sha(scheduler_file) == scheduler_sha
        result.update(status='complete', passed=True, frozen_unchanged=True)
    except Exception as caught:
        error = caught
        result.update(status='failed', error_type=type(caught).__name__, message=str(caught), details=getattr(caught, 'details', None))
    try:
        source_inputs(args)
        assert all(sha(Path(p)) == digest for p, digest in protocol['input_sha256'].items())
        assert all(sha(SOURCE/p) == digest for p, digest in protocol['source_sha256'].items())
        result['source_final_verification'] = True
    except Exception as caught:
        result.update(status='failed', passed=False, source_final_verification=False,
            source_verification_error=dict(type=type(caught).__name__, message=str(caught)))
        error = error or caught
    result['seconds'] = time.monotonic()-started
    archive(args, result, tensors)
    print('R16_DIAGNOSTIC', result['passed'], sha(args.out/'review.zip'), flush=True)
    if error is not None:
        raise error


def verify(args):
    source_inputs(args); protocol = read(args.out/'diagnostic_protocol.json')
    assert all(sha(Path(p)) == digest for p, digest in protocol['input_sha256'].items())
    assert all(sha(SOURCE/p) == digest for p, digest in protocol['source_sha256'].items())
    manifest = read(args.out/'manifest.json'); assert all(sha(args.out/p) == digest for p, digest in manifest.items())
    result = read(args.out/'result.json')
    assert result['passed'] and result['status'] == 'complete' and result['frozen_unchanged']
    assert (result['completed_cache_pixel_controls'], result['completed_new_reverse_passes']) == (96, 32)
    assert result['completed_f68_cache_pixel_controls'] == result['completed_G1_saved_latent_pixel_controls'] == 48
    assert result['new_S0'] == result['optimizer_steps'] == result['saved_candidate_updates'] == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', 'verify')); parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--out', type=Path); parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--project', type=Path, default=Path('/share/home/u2515283058/Mymodel'))
    args = parser.parse_args(); args.prior = args.prior.resolve(); args.project = args.project.resolve()
    auto = args.prior/'autonomous_repair_20261116'
    args.fixed = auto/'scaled_projection_retry'; args.preflight = auto/'terminal_head_preflight'
    args.basis = auto/'reference_terminal_preflight_v2'; args.source_run = auto/'reference_terminal_one_step'
    args.global_source = auto/'reference_terminal_global_amplitude'
    args.out = (args.out or auto/'reference_terminal_r16_diagnostic').resolve()
    args.data = args.project.parent/'datasets/BF'
    args.checkpoint = args.project/'output_eval/e2_state_b_pastel_followup_20261115/trust_retry/training/step1.pt'
    torch.set_num_threads(2); cv2.setNumThreads(2)
    dict(run=run, verify=verify)[args.action](args)


if __name__ == '__main__':
    main()
