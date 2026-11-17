"""原f68图像锚定的FP32解码差分反事实；零增量必须复现，不训练或采用。"""
import argparse
import copy
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch

from tools import e2_reference_terminal_post_latent_probe as post
from tools.e2_state_b_amplitude_response import read, write, sha, pixels
from tools.e2_state_b_backward_repro import tensor_sha
from tools.e2_b_last_step_metrics import score_pixels
from tools.e2_oracle_texture import weights_digest
from models.e2_lab_anchor import rgb_to_lab


SOURCE = Path(__file__).resolve().parents[1]
FACTORS = (0., .5, 1.)
EXTRA = ('tools/e2_reference_terminal_fp32_delta_probe.py',
         'scripts/e2_reference_terminal_fp32_delta_probe.sbatch')


def anchored_rgb(native_base, reference_raw32, current_raw32):
    """两个FP32解码只取差，原native f68图像作为锚；最后一次clamp。"""
    assert reference_raw32.dtype == current_raw32.dtype == torch.float32
    return (native_base.float()+(current_raw32-reference_raw32)).clamp(0, 1)


def decode_raw32(vae, z):
    assert vae.dtype == torch.float32 and z.dtype == torch.float32
    decoded = vae.decode(z/vae.config.scaling_factor, return_dict=False)[0]
    assert decoded.dtype == torch.float32
    return decoded/2+.5


def quantize(rgb):
    return np.rint(rgb.detach()[0].permute(1, 2, 0).float().cpu().numpy()*255).astype(np.uint8)


def verify_inputs(args):
    spec, _, q8_result, r16_result = post.verify_inputs(args)
    protocol = read(args.post/'protocol.json')
    manifest = read(args.post/'manifest.json')
    assert all(sha(args.post/n) == h for n, h in manifest.items())
    assert all(sha(Path(n)) == h for n, h in protocol['input_sha256'].items())
    assert all(sha(SOURCE/n) == h for n, h in protocol['source_sha256'].items())
    result = read(args.post/'result.json')
    assert result['passed'] and result['frozen_unchanged'] and result['source_final_verification']
    assert (result['completed_native_vae_decodes'], result['historical_cache_controls']) == (66, 44)
    return spec, protocol, result, q8_result, r16_result


@torch.no_grad()
def run(args):
    from tools import e2_state_b_pilot as pilot
    from tools.e2_fft_experiment import environment
    started = time.monotonic()
    spec, previous, parent, q8_result, r16_result = verify_inputs(args)
    data = torch.load(args.q8/'diagnostic.pt', map_location='cpu', weights_only=False)
    assert sha(args.q8/'diagnostic.pt') == q8_result['tensor_file_sha256']
    assert not args.out.exists(), '不覆盖已有FP32差分反事实'
    args.out.mkdir(parents=True)
    files = [args.post/n for n in ('protocol.json', 'result.json', 'diagnostic.pt', 'manifest.json')]
    files += [args.q8/'diagnostic.pt', args.r16/'result.json']
    protocol = dict(kind='new anchored FP32 decode-difference branch counterfactual;not original-S0 numerical repair',
        original_S0_code_modified=False, factors=list(FACTORS), selected_conditions=11,
        formula='clamp(anchor+(raw32(z_f68.float()+kappa32*(delta_B_h-delta_B_0))-raw32(z_f68.float())),0,1)',
        anchors=['native_f68_half_expanded', 'original_f68_PNG_restored_float32'],
        raw32='FP32 VAE with FP32 latent scaling and normalization, no inner clamp or half cast',
        matched_budget='same actual saved B perturbation;measure actual RGB functional budget',
        planned_fp32_vae_decodes=44, planned_new_observations=66, planned_zero_RGB_PNG_controls=22,
        baseline_FP32_scalar_controls=22, baseline_FP32_scalar_max_abs_error=1e-6,
        baseline_scalar_gate_scope='source compatibility only, not a replacement for original quality protection',
        new_S0=0, B_or_UNet_sampling_calls=0, reverse_passes=0, optimizer_steps=0, projection_solves=0,
        saved_candidates=0, holdout_read=0, adopted_checkpoints=[], quality_gates='original unchanged',
        input_sha256={str(p.resolve()): sha(p) for p in files},
        source_sha256=dict(previous['source_sha256'], **{n: sha(SOURCE/n) for n in EXTRA}),
        execution_commit=subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip())
    write(args.out/'protocol.json', protocol)
    result = dict(status='incomplete', passed=False, rows=[], baseline_controls=[], completed_fp32_vae_decodes=0,
        zero_RGB_PNG_controls=0, new_S0=0, B_or_UNet_sampling_calls=0, reverse_passes=0, optimizer_steps=0,
        projection_solves=0, saved_candidates=0, holdout_read=0, adopted_checkpoints=[],
        historical_context=dict(sha256=sha(args.post/'result.json'), rows=parent['rows'], rerun=False))
    tensors = dict(points={})
    error = None
    try:
        runtime = copy.copy(args); runtime.action = 'train'
        _, pipe, net = pilot.load_core(runtime)
        net.eval().requires_grad_(False); pipe.vae.float().eval().requires_grad_(False)
        net_hash, vae_hash = weights_digest(net), weights_digest(pipe.vae)
        assert net_hash == parent['net_hash'] and vae_hash == parent['vae_hash_fp32']
        result.update(environment=environment(), net_hash=net_hash, vae_hash_fp32=vae_hash)
        pairs = {p['case_id']: p for p in spec['train']}
        controls = {(r['case'], r['sample_condition']): r for r in r16_result['controls']}
        for row in q8_result['rows']:
            case, c = row['case'], row['sample_condition']; key = case+'/'+c
            point = data['points'][key]; baseline = point['probes']['0.0']
            native_base = baseline['native_rgb_half'].to(args.device)
            base_pixels = quantize(native_base)
            png_base = torch.from_numpy(base_pixels.copy()).permute(2, 0, 1)[None].to(args.device).float()/255
            zbase, dbase = [baseline[n].to(args.device) for n in ('latent', 'delta_B')]
            a, ap = [torch.tensor(point[n], device=args.device, dtype=torch.float32) for n in ('alpha', 'alpha_prev')]
            positive, weight, coefficients = [point[n].to(args.device) for n in ('positive', 'weight', 'residue_coefficients')]
            def values(rgb):
                return dict(global_amplitude=float(post.q8.r16.global_run.global_loss(rgb, positive, weight)),
                    residue=float(((rgb_to_lab(rgb)[0]*coefficients).sum()-point['residue_offset']).abs()))
            reference = decode_raw32(pipe.vae, zbase.float()); result['completed_fp32_vae_decodes'] += 1
            observed = values(reference.clamp(0, 1))
            expected = controls[case, c]['f68_stations']['R32_metadata']
            errors = {n: abs(observed[n]-expected[n]) for n in observed}
            result['baseline_controls'].append(dict(case=case, sample_condition=c, values=observed,
                expected_R32=expected, absolute_errors=errors))
            assert max(errors.values()) <= 1e-6, 'FP32参考解码与原R32来源控制不一致'
            tensors['points'][key] = dict(reference_raw32=reference.cpu(), records={})
            folder = args.source_run/'cases'/case
            png_positive, goal, anchor = [pixels(folder/n) for n in (c+'_positive.png', c+'_goal.png', 'train_anchor.png')]
            mask = pixels(folder/'mask.png'); mask = (mask[..., 0] if mask.ndim == 3 else mask) > 0
            for factor in FACTORS:
                d = point['probes'][str(factor)]['delta_B'].to(args.device)
                z = zbase.float()+post.coefficient(a, ap)*(d-dbase)
                current = decode_raw32(pipe.vae, z); result['completed_fp32_vae_decodes'] += 1
                delta_rgb = current-reference
                if factor == 0:
                    assert not torch.count_nonzero(delta_rgb)
                for anchor_name, rgb_anchor in [('native_anchor', native_base.float()), ('PNG_anchor', png_base)]:
                    rgb = anchored_rgb(rgb_anchor, reference, current)
                    image = quantize(rgb); pixel_sha = tensor_sha(torch.from_numpy(image))
                    if factor == 0:
                        assert torch.equal(rgb, rgb_anchor)
                        assert pixel_sha == controls[case, c]['pixel_array_sha256']
                        result['zero_RGB_PNG_controls'] += 1
                    rgb8 = torch.from_numpy(image.copy()).permute(2, 0, 1)[None].to(args.device).float()/255
                    numbers = dict(continuous_RGB=values(rgb), Q8_analytic=values(rgb8))
                    assert all(np.isfinite(v) for fields in numbers.values() for v in fields.values())
                    scores = score_pixels(image, png_positive, goal, anchor, mask, pairs[case]['patch_coordinates'])
                    result['rows'].append(dict(case=case, condition=row['condition'], sample_condition=c, group=row['group'],
                        arm=row['arm'], factor=row['factor'], probe_factor=factor, route='anchored_fp32_decode_difference',
                        anchor=anchor_name, latent_continuous_sha256=tensor_sha(z), rgb_float32_sha256=tensor_sha(rgb),
                        raw_delta_rgb_sha256=tensor_sha(delta_rgb), pixel_array_sha256=pixel_sha,
                        values=numbers, scores=scores, delta_rgb_norm=float((rgb.double()-rgb_anchor.double()).norm())))
                tensors['points'][key]['records'][str(factor)] = dict(latent_continuous=z.cpu(),
                    raw_delta_rgb=delta_rgb.cpu())
            print('FP32_DELTA_PROBE', len(tensors['points']), key, result['completed_fp32_vae_decodes'], flush=True)
        assert len(result['rows']) == 66 and (result['completed_fp32_vae_decodes'], result['zero_RGB_PNG_controls']) == (44, 22)
        assert weights_digest(pipe.vae) == vae_hash and weights_digest(net) == net_hash
        result.update(status='complete', passed=True, frozen_unchanged=True,
            meaning_of_passed='zero/source controls only;no gradient qualification,quality eligibility or adoption implied')
    except Exception as caught:
        error = caught
        result.update(status='failed', passed=False, error_type=type(caught).__name__, message=str(caught))
    try:
        verify_inputs(args)
        assert all(sha(Path(n)) == h for n, h in protocol['input_sha256'].items())
        assert all(sha(SOURCE/n) == h for n, h in protocol['source_sha256'].items())
        result['source_final_verification'] = True
    except Exception as caught:
        error = error or caught
        result.update(status='failed', passed=False, source_final_verification=False, source_error=str(caught))
    result['seconds'] = time.monotonic()-started
    post.q8.archive(args, result, tensors)
    print('FP32_DELTA_REVIEW', result['passed'], sha(args.out/'review.zip'), flush=True)
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
    args.q8 = auto/'reference_terminal_q8_direction_diagnostic'; args.post = auto/'reference_terminal_post_latent_probe'
    args.out = (args.out or auto/'reference_terminal_fp32_delta_probe').resolve()
    args.data = args.project.parent/'datasets/BF'
    args.checkpoint = args.project/'output_eval/e2_state_b_pastel_followup_20261115/trust_retry/training/step1.pt'
    torch.set_num_threads(2); cv2.setNumThreads(2)
    run(args)


if __name__ == '__main__':
    main()
