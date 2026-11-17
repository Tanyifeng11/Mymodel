"""固定f68/G1射线的Q8方向诊断；STE只作估计，不生成训练候选。"""
import argparse
import copy
import inspect
from pathlib import Path
import subprocess
import time
import zipfile

import cv2
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from tools import e2_reference_terminal_r16_diagnostic as r16
from tools.e2_b_endpoint_losses import flatten, set_parameters
from tools.e2_b_last_step_metrics import score_pixels
from tools.e2_state_b_amplitude_response import read, write, sha, pixels
from tools.e2_state_b_backward_repro import tensor_sha, vector_summary
from tools.e2_state_b_scaled_retry import scaled_scalar, NumericalFailure
from tools.e2_oracle_texture import weights_digest


SOURCE = Path(__file__).resolve().parents[1]
MODES = ('R16', 'Q8_STE')
LOSSES = ('global_amplitude', 'residue')
FACTORS = (-1., -.5, 0., .5, 1.)
EXTRA = ('tools/e2_reference_terminal_q8_direction.py', 'scripts/e2_reference_terminal_q8_direction.sbatch',
         'tools/e2_b_last_step_metrics.py')


def q8_ste(rgb):
    """前向严格为round(x*255)/255；反向借用identity，不声称是真实量化导数。"""
    assert rgb.dtype == torch.float32
    quantized = (rgb.detach()*255).round().clamp(0, 255)/255
    return quantized+(rgb-rgb.detach())


def select_controls(controls):
    selected = []
    for item in controls:
        b = item['f68_stations']['official_PNG']['old_projection']
        g = item['G1_observation']['stations']['official_PNG']['old_projection']
        reasons = []
        if g-b > 1e-5+1e-4*abs(b):
            reasons.append('already_observed_G1_PNG_residue_failure')
        if item['group'] == 'pastel' and item['arm'] == 'a100':
            reasons.append('already_observed_pastel_alpha1')
        if reasons:
            selected.append(dict(case=item['case'], sample_condition=item['sample_condition'], reasons=reasons))
    assert len(selected) == 11
    assert sum('already_observed_G1_PNG_residue_failure' in r['reasons'] for r in selected) == 7
    assert sum('already_observed_pastel_alpha1' in r['reasons'] for r in selected) == 4
    return selected


def central(values, h):
    return (values[h]-values[-h])/(2*h)


def verify_inputs(args):
    previous, spec, _, _ = r16.source_inputs(args)
    protocol = read(args.r16/'diagnostic_protocol.json')
    manifest = read(args.r16/'manifest.json')
    assert all(sha(args.r16/name) == digest for name, digest in manifest.items())
    assert all(sha(Path(name)) == digest for name, digest in protocol['input_sha256'].items())
    assert all(sha(SOURCE/name) == digest for name, digest in protocol['source_sha256'].items())
    result = read(args.r16/'result.json')
    assert result['passed'] and result['source_final_verification'] and result['frozen_unchanged']
    assert (result['completed_cache_pixel_controls'], result['completed_new_reverse_passes']) == (96, 32)
    return previous, spec, protocol, result


def archive(args, result, tensors):
    torch.save(tensors, args.out/'diagnostic.pt')
    result['tensor_file_sha256'] = sha(args.out/'diagnostic.pt')
    write(args.out/'result.json', result)
    names = ['protocol.json', 'diagnostic.pt', 'result.json']
    write(args.out/'manifest.json', {n: sha(args.out/n) for n in names})
    with zipfile.ZipFile(args.out/'review.zip', 'w', zipfile.ZIP_DEFLATED) as output:
        for name in names+['manifest.json']:
            output.write(args.out/name, name)


def run(args):
    from tools import e2_state_b_pilot as pilot
    from tools.e2_fft_experiment import environment
    started = time.monotonic()
    previous, spec, parent_protocol, parent_result = verify_inputs(args)
    selected = select_controls(parent_result['controls'])
    selected_keys = {(r['case'], r['sample_condition']) for r in selected}
    controls = {(r['case'], r['sample_condition']): r for r in parent_result['controls']}
    overlay = read(args.source_run/'one_step_overlay.json')
    prior = r16.prior_args(args)
    candidate = r16.source.load_candidate(prior, 'G1', 'cpu')
    direction = flatten(candidate.increment_parameters()); del candidate
    assert torch.isfinite(direction).all() and torch.count_nonzero(direction)
    assert not args.out.exists(), '不覆盖已有Q8诊断'
    args.out.mkdir(parents=True)
    files = [args.r16/name for name in ('diagnostic_protocol.json', 'manifest.json', 'result.json')]
    files += [args.global_source/'training/G1.pt']
    for item in selected:
        row = next(r for r in read(args.source_run/'evaluation/f68/replay/result.json')['rows']
                   if (r['case'], r['sample_condition']) == (item['case'], item['sample_condition']))
        files += [args.source_run/'cases'/item['case']/name for name in row['assets']]
    protocol = dict(selection=selected, selection_scope='targeted already-seen replay, not holdout/generalization',
        modes=list(MODES), losses=list(LOSSES), factors=list(FACTORS), derivative_point='f68 zero4616 only',
        direction='fixed actual saved G1 head increments; never recalculated or optimized',
        direction_sha256=tensor_sha(direction), direction_norm=float(direction.double().norm()),
        quantizer='round(native_half.float()*255)/255; identity STE backward; exact PNG forward checked',
        interpretation='STE is biased proxy; finite differences are at specified finite ray scales, not a proof of differentiability',
        planned_reverse_passes=88, planned_repeat_groups=44, planned_cached_probe_observations=55,
        planned_nonzero_probe_points=44, planned_historical_full_S0_pixel_controls=22,
        new_S0=0, optimizer_steps=0, projection_solves=0, saved_candidates=0, holdout_read=0,
        quality_gates='original unchanged; subset probes cannot establish replay eligibility',
        loss_scale=1024., repeat_gate=dict(max_relative_l2=.005, minimum_cosine=.99999, finite=True),
        input_sha256={str(p.resolve()): sha(p) for p in files},
        parent_R16_protocol_sha256=sha(args.r16/'diagnostic_protocol.json'),
        source_sha256=dict(parent_protocol['source_sha256'], **{n: sha(SOURCE/n) for n in EXTRA}),
        execution_commit=subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip())
    write(args.out/'protocol.json', protocol)
    result = dict(status='incomplete', passed=False, rows=[], model_controls=[], completed_reverse_passes=0,
        completed_cached_probe_observations=0, completed_nonzero_probe_points=0, historical_pixel_controls=0,
        new_S0=0, optimizer_steps=0, projection_solves=0, saved_candidates=0, holdout_read=0, adopted_checkpoints=[])
    tensors = dict(direction=direction, points={})
    error = None
    try:
        runtime = copy.copy(args); runtime.action = 'train'
        _, pipe, net = pilot.load_core(runtime)
        net.eval().requires_grad_(False); pipe.vae.float().eval().requires_grad_(False)
        head = r16.source.load_head(args, args.device); parameters = head.increment_parameters()
        zero = torch.zeros_like(direction)
        base_hash, net_hash, vae_hash = weights_digest(head.base), weights_digest(net), weights_digest(pipe.vae)
        assert vae_hash == parent_result['vae_hash_fp32'] and base_hash == parent_result['base_b_hash']
        assert net_hash == parent_result['net_hash']
        assert sha(Path(inspect.getsourcefile(type(pipe.scheduler)))) == parent_result['scheduler_source_sha256']
        result['environment'] = environment()
        result.update(base_b_hash=base_hash, net_hash=net_hash, vae_hash_fp32=vae_hash,
                      scheduler_source_sha256=parent_result['scheduler_source_sha256'])
        objective_runtime = copy.copy(runtime); objective_runtime.out = args.source_run
        for index, pair in enumerate(spec['train']):
            cache, _ = r16.source.load_cache(args, index, overlay)
            metadata = read(args.basis/('cache/shard%d.json' % index))
            audit = r16.source.preflight.scheduler_configuration(pipe.scheduler.config, metadata['scheduler']['config'])
            assert audit['normalized_equal']
            objective = r16.source.objective_from_cache(objective_runtime, pipe, head, pair, cache)
            pipe.scheduler.set_timesteps(50, device=args.device)
            pipe.vae.half()
            try:
                for item in [r for r in spec['inputs'] if (r['case'], r['sample_condition']) in selected_keys and r['case'] == pair['case_id']]:
                    case, c = item['case'], item['sample_condition']; key = case+'/'+c
                    record = cache['records'][c]; historical = controls[case, c]
                    phi = head.statistics(record['target_record'][3]['reference'].to(args.device))['phi']
                    assert tensor_sha(phi) == historical['phi_sha256']
                    _, positive, _, _, weight, _ = objective.records[case, c]
                    metric = objective.metric(dict(pair=pair, condition=c))
                    point = dict(positive=positive.detach().cpu(), weight=weight.detach().cpu(),
                        residue_coefficients=metric.coefficients.detach().cpu(), residue_offset=metric.offset,
                        residue_energy=metric.energy, zt=record['zt'].cpu(), raw_epsilon=record['raw_epsilon'].cpu(),
                        alpha=record['alpha'], alpha_prev=float(pipe.scheduler.final_alpha_cumprod),
                        gradients={}, probes={})
                    tensors['points'][key] = point
                    derivatives = {}
                    for mode in MODES:
                        derivatives[mode] = {}
                        for task in LOSSES:
                            samples, values = [], []
                            for repeat in range(2):
                                set_parameters(parameters, zero.to(args.device))
                                z, _, _, _ = r16.source.preflight.terminal_replay(head, record, pipe.scheduler, args.device)
                                native = checkpoint(lambda v: r16.native_rgb_half(pipe.vae, v), z, use_reentrant=False)
                                assert tensor_sha(native) == historical['native_rgb_half_sha256']
                                rgb = native.float() if mode == 'R16' else q8_ste(native.float())
                                if mode == 'Q8_STE':
                                    image = r16.quantize(native)
                                    restored = torch.from_numpy(image.copy()).permute(2, 0, 1)[None].to(args.device).float()/255
                                    assert torch.equal(rgb.detach(), restored)
                                loss = r16.global_run.global_loss(rgb, positive, weight) if task == 'global_amplitude' else metric.surrogate(rgb)
                                if not torch.isfinite(loss):
                                    raise NumericalFailure(key+'/'+mode+'/'+task, loss)
                                values.append(float(loss.detach()))
                                gradient = scaled_scalar(loss, parameters).detach().cpu()
                                result['completed_reverse_passes'] += 1
                                samples.append(gradient)
                                point['gradients'][mode+'/'+task] = torch.stack(samples)
                                del z, native, rgb, loss
                                if not torch.isfinite(gradient).all():
                                    raise NumericalFailure(key+'/'+mode+'/'+task, gradient)
                            repeat_check = r16.repeat_report(samples[1], samples[0])
                            assert values[0] == values[1] and repeat_check['passed']
                            derivatives[mode][task] = dict(value=values[0], repeats=repeat_check,
                                directional=float(samples[0].double()@direction.double()), gradient=vector_summary(samples[0]))
                    probes = {}
                    folder = args.source_run/'cases'/case
                    png_positive, goal, anchor = [pixels(folder/n) for n in (c+'_positive.png', c+'_goal.png', 'train_anchor.png')]
                    mask = pixels(folder/'mask.png')
                    if mask.ndim == 3:
                        mask = mask[..., 0]
                    for factor in FACTORS:
                        with torch.no_grad():
                            vector = zero if factor == 0 else direction*factor
                            set_parameters(parameters, vector.to(args.device))
                            z, _, delta, corrected = r16.source.preflight.terminal_replay(head, record, pipe.scheduler, args.device)
                            native = r16.native_rgb_half(pipe.vae, z)
                            image = r16.quantize(native); q8 = torch.from_numpy(image.copy()).permute(2, 0, 1)[None].to(args.device).float()/255
                            values = {}
                            for mode, rgb in [('R16', native.float()), ('Q8_STE', q8)]:
                                values[mode] = dict(global_amplitude=float(r16.global_run.global_loss(rgb, positive, weight)),
                                                    residue=float(metric.surrogate(rgb)))
                            pixel_sha = tensor_sha(torch.from_numpy(image))
                            if factor in (0., 1.):
                                old = historical if factor == 0 else historical['G1_observation']
                                assert tensor_sha(z) == old['latent_sha256'] and pixel_sha == old['pixel_array_sha256']
                                assert tensor_sha(native) == old['native_rgb_half_sha256']
                                result['historical_pixel_controls'] += 1
                            if factor == 0:
                                assert tensor_sha(delta) == tensor_sha(record['base_delta'])
                                assert tensor_sha(corrected) == historical['corrected_epsilon_sha256']
                            scores = score_pixels(image, png_positive, goal, anchor, mask > 0, pair['patch_coordinates'])
                            probes[factor] = dict(factor=factor, latent_sha256=tensor_sha(z), native_rgb_half_sha256=tensor_sha(native),
                                delta_B_sha256=tensor_sha(delta), corrected_epsilon_sha256=tensor_sha(corrected),
                                pixel_array_sha256=pixel_sha, values=values, scores=scores,
                                actual_parameter_sha256=tensor_sha(flatten(parameters)))
                            point['probes'][str(factor)] = dict(delta_B=delta.cpu(), corrected_epsilon=corrected.cpu(),
                                latent=z.cpu(), native_rgb_half=native.cpu())
                        result['completed_cached_probe_observations'] += 1
                        result['completed_nonzero_probe_points'] += int(factor != 0)
                    set_parameters(parameters, zero.to(args.device))
                    finite = {}
                    for mode in MODES:
                        finite[mode] = {}
                        for task in LOSSES:
                            observations = {f: probes[f]['values'][mode][task] for f in FACTORS}
                            finite[mode][task] = dict(central={str(h): central(observations, h) for h in (.5, 1.)},
                                positive_endpoint=observations[1.]-observations[0.],
                                estimated_directional=derivatives[mode][task]['directional'])
                    result['rows'].append(dict(item, phi_sha256=tensor_sha(phi), derivatives=derivatives,
                                              probes=list(probes.values()), finite=finite))
                    print('Q8_DIRECTION', len(result['rows']), key, result['completed_reverse_passes'],
                          result['completed_cached_probe_observations'], flush=True)
            finally:
                set_parameters(parameters, zero.to(args.device))
                pipe.vae.float()
                assert weights_digest(pipe.vae) == vae_hash
            result['model_controls'].append(dict(case=pair['case_id'], scheduler=audit,
                zero_head=not bool(torch.count_nonzero(flatten(parameters))), vae_restored_hash=weights_digest(pipe.vae)))
        assert (result['completed_reverse_passes'], result['completed_cached_probe_observations'],
                result['completed_nonzero_probe_points'], result['historical_pixel_controls']) == (88, 55, 44, 22)
        assert len(result['rows']) == 11
        assert not torch.count_nonzero(flatten(parameters))
        assert weights_digest(head.base) == base_hash and weights_digest(net) == net_hash and weights_digest(pipe.vae) == vae_hash
        assert not any(p.grad is not None for model in (head, net, pipe.vae) for p in model.parameters())
        result.update(status='complete', passed=True, frozen_unchanged=True,
            meaning_of_passed='repeatability and historical endpoint parity only; no directional or quality pass is implied')
    except Exception as caught:
        error = caught
        result.update(status='failed', passed=False, error_type=type(caught).__name__, message=str(caught),
                      details=getattr(caught, 'details', None))
    try:
        verify_inputs(args)
        assert all(sha(Path(name)) == digest for name, digest in protocol['input_sha256'].items())
        assert all(sha(SOURCE/name) == digest for name, digest in protocol['source_sha256'].items())
        result['source_final_verification'] = True
    except Exception as caught:
        error = error or caught
        result.update(status='failed', passed=False, source_final_verification=False, source_error=str(caught))
    result['seconds'] = time.monotonic()-started
    archive(args, result, tensors)
    print('Q8_DIRECTION_DIAGNOSTIC', result['passed'], sha(args.out/'review.zip'), flush=True)
    if error is not None:
        raise error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', type=Path, required=True); parser.add_argument('--out', type=Path)
    parser.add_argument('--project', type=Path, default=Path('/share/home/u2515283058/Mymodel'))
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(); args.prior = args.prior.resolve(); args.project = args.project.resolve()
    auto = args.prior/'autonomous_repair_20261116'
    args.fixed = auto/'scaled_projection_retry'; args.preflight = auto/'terminal_head_preflight'
    args.basis = auto/'reference_terminal_preflight_v2'; args.source_run = auto/'reference_terminal_one_step'
    args.global_source = auto/'reference_terminal_global_amplitude'; args.r16 = auto/'reference_terminal_r16_diagnostic'
    args.out = (args.out or auto/'reference_terminal_q8_direction_diagnostic').resolve()
    args.data = args.project.parent/'datasets/BF'
    args.checkpoint = args.project/'output_eval/e2_state_b_pastel_followup_20261115/trust_retry/training/step1.pt'
    torch.set_num_threads(2); cv2.setNumThreads(2)
    run(args)


if __name__ == '__main__':
    main()
