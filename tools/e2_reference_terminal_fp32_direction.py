"""锚定FP32差分支路的fresh局部梯度/有限响应；固定f68和旧G1射线，不训练。"""
import argparse
import copy
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from tools import e2_reference_terminal_fp32_delta_probe as delta
from tools.e2_b_endpoint_losses import flatten, set_parameters
from tools.e2_state_b_amplitude_response import read, write, sha, pixels
from tools.e2_state_b_backward_repro import tensor_sha, vector_summary
from tools.e2_state_b_scaled_retry import scaled_scalar, NumericalFailure
from tools.e2_oracle_texture import weights_digest
from models.e2_lab_anchor import rgb_to_lab


SOURCE = Path(__file__).resolve().parents[1]
MODES = ('continuous_RGB', 'Q8_STE')
TASKS = ('global_amplitude', 'residue')
FACTORS = (-1., -.5, 0., .5, 1.)
EXTRA = ('tools/e2_reference_terminal_fp32_direction.py',
         'scripts/e2_reference_terminal_fp32_direction.sbatch')


def continuous_latent(zbase, kappa, current_b, base_b):
    """同冻结f68零点，末步B输出差用FP32保持到差分解码；不再走half epsilon。"""
    assert all(v.dtype == torch.float32 for v in (kappa, current_b, base_b))
    return zbase.float()+kappa*(current_b-base_b)


def verify(args):
    spec, _, _, q8_result, _ = delta.verify_inputs(args)
    protocol = read(args.fp32/'protocol.json')
    assert all(sha(args.fp32/n) == h for n, h in read(args.fp32/'manifest.json').items())
    assert all(sha(Path(n)) == h for n, h in protocol['input_sha256'].items())
    assert all(sha(SOURCE/n) == h for n, h in protocol['source_sha256'].items())
    result = read(args.fp32/'result.json')
    assert result['passed'] and result['frozen_unchanged'] and result['source_final_verification']
    assert (result['completed_fp32_vae_decodes'], result['zero_RGB_PNG_controls']) == (44, 22)
    return spec, protocol, result, q8_result


def run(args):
    from tools import e2_state_b_pilot as pilot
    from tools.e2_fft_experiment import environment
    started = time.monotonic()
    spec, previous, fp32_result, q8_result = verify(args)
    q8 = delta.post.q8
    r16 = q8.r16
    source = r16.source
    saved = torch.load(args.q8/'diagnostic.pt', map_location='cpu', weights_only=False)
    old32 = torch.load(args.fp32/'diagnostic.pt', map_location='cpu', weights_only=False)
    direction = saved['direction']
    assert tensor_sha(direction) == read(args.q8/'protocol.json')['direction_sha256']
    assert not args.out.exists(), '不覆盖已有FP32支路梯度诊断'
    args.out.mkdir(parents=True)
    files = [args.fp32/n for n in ('protocol.json', 'result.json', 'diagnostic.pt', 'manifest.json')]
    files += [args.q8/'diagnostic.pt']
    protocol = dict(kind='fresh gradients of new anchored FP32 output branch;not original-S0 repair',
        original_S0_code_modified=False, anchor='native f68 half expanded to FP32',
        factors=list(FACTORS), modes=list(MODES), losses=list(TASKS), selected_conditions=11,
        parameter_count=4616, parameter_point='same f68 zero4616;fresh G for this new forward only',
        direction='saved actual G1 parameter increments,not optimized', direction_sha256=tensor_sha(direction),
        forward='clamp(native_f68+(raw32(z_f68.float()+kappa32*(B_head(theta)-B_head(0)))-raw32(z_f68.float())),0,1)',
        Q8_STE='original exact PNG quantizer forward,identity backward is a biased heuristic',
        derivative_interpretation='finite-scale central differences and one-sided endpoints are separate observations',
        planned_reverse_passes=88, planned_repeat_groups=44, planned_cache_observations=55,
        planned_positive_source_RGB_PNG_controls=33, planned_reference_raw_controls=11,
        repeat_gate=dict(max_relative_l2=.005, minimum_cosine=.99999, finite=True),
        finite_difference_eligibility_gate=None, quality_gates='original unchanged;no training or adoption eligibility from subset',
        new_S0=0, UNet_sampling_calls=0, optimizer_steps=0, projection_solves=0,
        saved_candidates=0, holdout_read=0, adopted_checkpoints=[], loss_scale=1024.,
        input_sha256={str(p.resolve()): sha(p) for p in files},
        source_sha256=dict(previous['source_sha256'], **{n: sha(SOURCE/n) for n in EXTRA}),
        execution_commit=subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip())
    write(args.out/'protocol.json', protocol)
    result = dict(status='incomplete', passed=False, rows=[], completed_reverse_passes=0,
        completed_cache_observations=0, positive_source_RGB_PNG_controls=0, reference_raw_controls=0,
        new_S0=0, UNet_sampling_calls=0, optimizer_steps=0, projection_solves=0,
        saved_candidates=0, holdout_read=0, adopted_checkpoints=[])
    tensors = dict(direction=direction, points={})
    error = None
    try:
        runtime = copy.copy(args); runtime.action = 'train'
        _, pipe, net = pilot.load_core(runtime)
        net.eval().requires_grad_(False); pipe.vae.float().eval().requires_grad_(False)
        head = source.load_head(args, args.device)
        parameters = head.increment_parameters(); zero = torch.zeros_like(direction)
        base_hash, net_hash, vae_hash = [weights_digest(v) for v in (head.base, net, pipe.vae)]
        assert net_hash == fp32_result['net_hash'] and vae_hash == fp32_result['vae_hash_fp32']
        assert base_hash == q8_result['base_b_hash']
        result.update(environment=environment(), base_b_hash=base_hash, net_hash=net_hash, vae_hash_fp32=vae_hash)
        overlay = read(args.source_run/'one_step_overlay.json')
        old_rows = {(r['case'], r['sample_condition'], r['probe_factor']): r
                    for r in fp32_result['rows'] if r['anchor'] == 'native_anchor'}
        for index, pair in enumerate(spec['train']):
            cache, _ = source.load_cache(args, index, overlay)
            selected = [r for r in q8_result['rows'] if r['case'] == pair['case_id']]
            for item in selected:
                case, condition = item['case'], item['sample_condition']; key = case+'/'+condition
                record = cache['records'][condition]; point = saved['points'][key]
                feature, mask = [record[n].to(args.device) for n in ('feature', 'mask')]
                phi = head.statistics(record['target_record'][3]['reference'].to(args.device))['phi']
                assert tensor_sha(phi) == item['phi_sha256']
                base = point['probes']['0.0']
                zbase = base['latent'].to(args.device).float()
                native = base['native_rgb_half'].to(args.device).float()
                kappa = delta.post.coefficient(*[torch.tensor(point[n], device=args.device, dtype=torch.float32)
                                                for n in ('alpha', 'alpha_prev')])
                positive, weight, coefficients = [point[n].to(args.device) for n in ('positive', 'weight', 'residue_coefficients')]
                with torch.no_grad():
                    set_parameters(parameters, zero.to(args.device))
                    base_b = head.from_features(feature, mask, phi).detach()
                    assert tensor_sha(base_b) == tensor_sha(base['delta_B'])
                    reference = delta.decode_raw32(pipe.vae, zbase)
                    assert tensor_sha(reference) == tensor_sha(old32['points'][key]['reference_raw32'])
                    result['reference_raw_controls'] += 1
                def loss_values(rgb):
                    return dict(global_amplitude=r16.global_run.global_loss(rgb, positive, weight),
                        residue=((rgb_to_lab(rgb)[0]*coefficients).sum()-point['residue_offset']).abs())
                tensors['points'][key] = dict(gradients={}, probes={})
                derivatives = {}
                for mode in MODES:
                    derivatives[mode] = {}
                    for task in TASKS:
                        gradients, values = [], []
                        for repeat in range(2):
                            set_parameters(parameters, zero.to(args.device))
                            z = continuous_latent(zbase, kappa, head.from_features(feature, mask, phi), base_b)
                            current = checkpoint(lambda v: delta.decode_raw32(pipe.vae, v), z, use_reentrant=False)
                            rgb = delta.anchored_rgb(native, reference, current)
                            if mode == 'Q8_STE':
                                rgb = q8.q8_ste(rgb)
                            loss = loss_values(rgb)[task]
                            if not torch.isfinite(loss):
                                raise NumericalFailure(key+'/'+mode+'/'+task, loss)
                            gradient = scaled_scalar(loss, parameters).detach().cpu()
                            result['completed_reverse_passes'] += 1
                            assert torch.isfinite(gradient).all()
                            values.append(float(loss.detach())); gradients.append(gradient)
                            del z, current, rgb, loss
                        repeat_check = r16.repeat_report(gradients[1], gradients[0])
                        assert values[0] == values[1] and repeat_check['passed']
                        tensors['points'][key]['gradients'][mode+'/'+task] = torch.stack(gradients)
                        derivatives[mode][task] = dict(value=values[0], repeats=repeat_check,
                            directional=float(gradients[0].double()@direction.double()), gradient=vector_summary(gradients[0]))
                folder = args.source_run/'cases'/case
                png_positive, goal, anchor = [pixels(folder/n) for n in (condition+'_positive.png', condition+'_goal.png', 'train_anchor.png')]
                png_mask = pixels(folder/'mask.png'); png_mask = (png_mask[..., 0] if png_mask.ndim == 3 else png_mask) > 0
                probes = {}
                with torch.no_grad():
                    for h in FACTORS:
                        set_parameters(parameters, (direction*h).to(args.device))
                        current_b = head.from_features(feature, mask, phi)
                        assert tensor_sha(current_b) == tensor_sha(point['probes'][str(h)]['delta_B'])
                        z = continuous_latent(zbase, kappa, current_b, base_b)
                        raw = delta.decode_raw32(pipe.vae, z)
                        rgb = delta.anchored_rgb(native, reference, raw)
                        image = delta.quantize(rgb)
                        pixel_sha = tensor_sha(torch.from_numpy(image))
                        if h >= 0:
                            old = old_rows[case, condition, h]
                            assert tensor_sha(rgb) == old['rgb_float32_sha256'] and pixel_sha == old['pixel_array_sha256']
                            result['positive_source_RGB_PNG_controls'] += 1
                        if h == 0:
                            assert torch.equal(rgb, native)
                        rgb8 = torch.from_numpy(image.copy()).permute(2, 0, 1)[None].to(args.device).float()/255
                        numbers = {mode: {n: float(v) for n, v in loss_values(value).items()}
                                   for mode, value in [('continuous_RGB', rgb), ('Q8_STE', rgb8)]}
                        scores = q8.score_pixels(image, png_positive, goal, anchor, png_mask, pair['patch_coordinates'])
                        probes[h] = dict(factor=h, values=numbers, scores=scores, pixel_array_sha256=pixel_sha,
                            rgb_float32_sha256=tensor_sha(rgb), latent_continuous_sha256=tensor_sha(z))
                        tensors['points'][key]['probes'][str(h)] = dict(rgb=rgb.cpu(), latent_continuous=z.cpu())
                        result['completed_cache_observations'] += 1
                finite = {mode: {task: dict(
                    central={str(h): q8.central({f: probes[f]['values'][mode][task] for f in FACTORS}, h) for h in (.5, 1.)},
                    positive_endpoint=probes[1.]['values'][mode][task]-probes[0.]['values'][mode][task],
                    estimated_directional=derivatives[mode][task]['directional']) for task in TASKS} for mode in MODES}
                result['rows'].append(dict(case=case, sample_condition=condition, group=item['group'],
                    derivatives=derivatives, finite=finite, probes=list(probes.values())))
                print('FP32_DIRECTION', len(result['rows']), key, result['completed_reverse_passes'], flush=True)
                set_parameters(parameters, zero.to(args.device))
        assert (result['completed_reverse_passes'], result['completed_cache_observations'],
                result['positive_source_RGB_PNG_controls'], result['reference_raw_controls']) == (88, 55, 33, 11)
        assert not torch.count_nonzero(flatten(parameters))
        assert [weights_digest(v) for v in (head.base, net, pipe.vae)] == [base_hash, net_hash, vae_hash]
        assert not any(p.grad is not None for model in (head, net, pipe.vae) for p in model.parameters())
        result.update(status='complete', passed=True, frozen_unchanged=True,
            meaning_of_passed='repeatability/source controls only;no finite-direction,quality or training qualification implied')
    except Exception as caught:
        error = caught
        result.update(status='failed', passed=False, error_type=type(caught).__name__, message=str(caught))
    try:
        verify(args)
        assert all(sha(Path(n)) == h for n, h in protocol['input_sha256'].items())
        assert all(sha(SOURCE/n) == h for n, h in protocol['source_sha256'].items())
        result['source_final_verification'] = True
    except Exception as caught:
        error = error or caught
        result.update(status='failed', passed=False, source_final_verification=False, source_error=str(caught))
    result['seconds'] = time.monotonic()-started
    q8.archive(args, result, tensors)
    print('FP32_DIRECTION_REVIEW', result['passed'], sha(args.out/'review.zip'), flush=True)
    if error is not None:
        raise error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--project', type=Path, default=Path('/share/home/u2515283058/Mymodel'))
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(); args.prior = args.prior.resolve(); args.project = args.project.resolve()
    auto = args.prior/'autonomous_repair_20261116'
    for name, value in dict(fixed='scaled_projection_retry', preflight='terminal_head_preflight',
        basis='reference_terminal_preflight_v2', source_run='reference_terminal_one_step',
        global_source='reference_terminal_global_amplitude', r16='reference_terminal_r16_diagnostic',
        q8='reference_terminal_q8_direction_diagnostic', post='reference_terminal_post_latent_probe',
        fp32='reference_terminal_fp32_delta_probe', out='reference_terminal_fp32_direction').items():
        setattr(args, name, auto/value)
    args.data = args.project.parent/'datasets/BF'
    args.checkpoint = args.project/'output_eval/e2_state_b_pastel_followup_20261115/trust_retry/training/step1.pt'
    torch.set_num_threads(2); cv2.setNumThreads(2)
    run(args)


if __name__ == '__main__':
    main()
