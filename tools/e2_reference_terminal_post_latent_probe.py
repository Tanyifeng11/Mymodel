"""同一已保存B扰动的注入位置反事实；不修改S0，不训练或采用新结构。"""
import argparse
import copy
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch

from tools import e2_reference_terminal_q8_direction as q8
from tools.e2_state_b_amplitude_response import read, write, sha, pixels
from tools.e2_state_b_backward_repro import tensor_sha
from tools.e2_b_last_step_metrics import score_pixels
from tools.e2_oracle_texture import weights_digest
from models.e2_lab_anchor import rgb_to_lab


SOURCE = Path(__file__).resolve().parents[1]
FACTORS = (0., .5, 1.)
EXTRA = ('tools/e2_reference_terminal_post_latent_probe.py',
         'scripts/e2_reference_terminal_post_latent_probe.sbatch')


def coefficient(alpha, alpha_prev):
    """eta=0、epsilon预测、无clipping时，连续DDIM的B→末态系数；不拟合。"""
    return alpha_prev.sqrt()-(alpha/(1-alpha)).sqrt()*(1-alpha_prev).sqrt()


def post_latent(base, base_delta, delta, alpha, alpha_prev):
    # delta已含原mask；不再乘一次。该公式是新输出结构假设，不替换scheduler.step。
    return (base.float()+coefficient(alpha, alpha_prev)*(delta.float()-base_delta.float())).half()


def verify_inputs(args):
    _, spec, _, r16_result = q8.verify_inputs(args)
    protocol = read(args.q8/'protocol.json')
    manifest = read(args.q8/'manifest.json')
    assert all(sha(args.q8/n) == h for n, h in manifest.items())
    assert all(sha(Path(n)) == h for n, h in protocol['input_sha256'].items())
    assert all(sha(SOURCE/n) == h for n, h in protocol['source_sha256'].items())
    result = read(args.q8/'result.json')
    assert result['passed'] and result['frozen_unchanged'] and result['source_final_verification']
    assert (result['completed_reverse_passes'], result['completed_cached_probe_observations']) == (88, 55)
    return spec, protocol, result, r16_result


@torch.no_grad()
def run(args):
    from tools import e2_state_b_pilot as pilot
    from tools.e2_fft_experiment import environment
    started = time.monotonic()
    spec, previous, parent, r16_result = verify_inputs(args)
    data = torch.load(args.q8/'diagnostic.pt', map_location='cpu', weights_only=False)
    assert sha(args.q8/'diagnostic.pt') == parent['tensor_file_sha256']
    assert not args.out.exists(), '不覆盖已有注入位置反事实'
    args.out.mkdir(parents=True)
    files = [args.q8/n for n in ('protocol.json', 'result.json', 'diagnostic.pt', 'manifest.json')]
    files += [args.r16/'result.json']
    protocol = dict(kind='new B output-location counterfactual, not an original-S0 numerical repair',
        original_S0_code_modified=False, selected_conditions=11, selection=previous['selection'],
        factors=list(FACTORS), routes=['original_epsilon_cached', 'post_latent_hypothesis'],
        formula='half(z_f68.float()+kappa32*(delta_B_h-delta_B_0));kappa=sqrt(alpha_prev)-sqrt(alpha/(1-alpha))*sqrt(1-alpha_prev)',
        matched_budget='same saved actual delta_B; actual latent/RGB norms are not assumed matched',
        mask='already present in saved delta_B; no second mask',
        planned_native_vae_decodes=66, planned_historical_cache_controls=44,
        planned_new_nonzero_structure_observations=22,
        new_S0=0, B_or_UNet_sampling_calls=0, reverse_passes=0, optimizer_steps=0, projection_solves=0,
        saved_candidates=0, holdout_read=0, adopted_checkpoints=[],
        quality_gates='original unchanged;11-condition hypothetical subset cannot qualify adoption',
        input_sha256={str(p.resolve()): sha(p) for p in files},
        source_sha256=dict(previous['source_sha256'], **{n: sha(SOURCE/n) for n in EXTRA}),
        execution_commit=subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip())
    write(args.out/'protocol.json', protocol)
    result = dict(status='incomplete', passed=False, rows=[], completed_native_vae_decodes=0,
        historical_cache_controls=0, new_nonzero_structure_observations=0, new_S0=0,
        B_or_UNet_sampling_calls=0, reverse_passes=0, optimizer_steps=0, projection_solves=0,
        saved_candidates=0, holdout_read=0, adopted_checkpoints=[])
    tensors = dict(post_latent={})
    error = None
    try:
        runtime = copy.copy(args); runtime.action = 'train'
        _, pipe, net = pilot.load_core(runtime)
        net.eval().requires_grad_(False); pipe.vae.float().eval().requires_grad_(False)
        net_hash, vae_hash = weights_digest(net), weights_digest(pipe.vae)
        assert net_hash == parent['net_hash'] and vae_hash == parent['vae_hash_fp32']
        pipe.vae.half()
        assert weights_digest(pipe.vae) == r16_result['model_controls'][0]['vae_half_hash']
        result.update(environment=environment(), net_hash=net_hash, vae_hash_fp32=vae_hash)
        pairs = {p['case_id']: p for p in spec['train']}
        for row in parent['rows']:
            case, c = row['case'], row['sample_condition']; key = case+'/'+c
            point = data['points'][key]; baseline = point['probes']['0.0']
            alpha, alpha_prev = [torch.tensor(point[n], device=args.device, dtype=torch.float32)
                                 for n in ('alpha', 'alpha_prev')]
            base, base_delta = [baseline[n].to(args.device) for n in ('latent', 'delta_B')]
            positive, weight, coefficients = [point[n].to(args.device) for n in
                                              ('positive', 'weight', 'residue_coefficients')]
            old = {r['factor']: r for r in row['probes']}
            folder = args.source_run/'cases'/case
            png_positive, goal, anchor = [pixels(folder/n) for n in
                                         (c+'_positive.png', c+'_goal.png', 'train_anchor.png')]
            mask = pixels(folder/'mask.png')
            mask = (mask[..., 0] if mask.ndim == 3 else mask) > 0
            tensors['post_latent'][key] = {}
            for factor in FACTORS:
                previous_point = point['probes'][str(factor)]
                original = previous_point['latent'].to(args.device)
                delta = previous_point['delta_B'].to(args.device)
                hypothesized = post_latent(base, base_delta, delta, alpha, alpha_prev)
                for route, latent in [('original_epsilon_cached', original),
                                      ('post_latent_hypothesis', hypothesized)]:
                    native = q8.r16.native_rgb_half(pipe.vae, latent)
                    image = q8.r16.quantize(native)
                    rgb8 = torch.from_numpy(image.copy()).permute(2, 0, 1)[None].to(args.device).float()/255
                    native_sha = tensor_sha(native); pixel_sha = tensor_sha(torch.from_numpy(image))
                    if route == 'original_epsilon_cached' or factor == 0:
                        assert tensor_sha(latent) == old[factor]['latent_sha256']
                        assert native_sha == old[factor]['native_rgb_half_sha256']
                        assert pixel_sha == old[factor]['pixel_array_sha256']
                        result['historical_cache_controls'] += 1
                    values = {}
                    for stage, rgb in [('R16', native.float()), ('Q8_analytic', rgb8)]:
                        values[stage] = dict(global_amplitude=float(q8.r16.global_run.global_loss(rgb, positive, weight)),
                            residue=float(((rgb_to_lab(rgb)[0]*coefficients).sum()-point['residue_offset']).abs()))
                    assert all(np.isfinite(v) for fields in values.values() for v in fields.values())
                    scores = score_pixels(image, png_positive, goal, anchor, mask, pairs[case]['patch_coordinates'])
                    change = latent.double()-base.double()
                    reference = coefficient(alpha, alpha_prev).double()*(delta.double()-base_delta.double())
                    result['rows'].append(dict(case=case, condition=row['condition'], sample_condition=c,
                        group=row['group'], arm=row['arm'], factor=row['factor'], probe_factor=factor, route=route,
                        kappa32=float(coefficient(alpha, alpha_prev)), latent_sha256=tensor_sha(latent),
                        native_rgb_half_sha256=native_sha, pixel_array_sha256=pixel_sha,
                        values=values, scores=scores, latent_change_norm=float(change.norm()),
                        continuous_reference_norm=float(reference.norm()),
                        continuous_reference_residual=(float((change-reference).norm())/max(float(reference.norm()), 1e-18)
                                                       if factor else 0.)))
                    if route == 'post_latent_hypothesis':
                        tensors['post_latent'][key][str(factor)] = dict(latent=latent.cpu(), native_rgb_half=native.cpu())
                        result['new_nonzero_structure_observations'] += int(factor != 0)
                    result['completed_native_vae_decodes'] += 1
            print('POST_LATENT_PROBE', len(tensors['post_latent']), key, result['completed_native_vae_decodes'], flush=True)
        pipe.vae.float()
        assert weights_digest(pipe.vae) == vae_hash and weights_digest(net) == net_hash
        assert (result['completed_native_vae_decodes'], result['historical_cache_controls'],
                result['new_nonzero_structure_observations']) == (66, 44, 22)
        result.update(status='complete', passed=True, frozen_unchanged=True,
            meaning_of_passed='historical controls/frozen identity only;new-route quality is observational,not adoption')
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
    q8.archive(args, result, tensors)
    print('POST_LATENT_REVIEW', result['passed'], sha(args.out/'review.zip'), flush=True)
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
    args.q8 = auto/'reference_terminal_q8_direction_diagnostic'
    args.out = (args.out or auto/'reference_terminal_post_latent_probe').resolve()
    args.data = args.project.parent/'datasets/BF'
    args.checkpoint = args.project/'output_eval/e2_state_b_pastel_followup_20261115/trust_retry/training/step1.pt'
    torch.set_num_threads(2); cv2.setNumThreads(2)
    run(args)


if __name__ == '__main__':
    main()
