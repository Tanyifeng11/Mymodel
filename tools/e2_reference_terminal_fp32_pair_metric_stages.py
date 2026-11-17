"""固定已存f68/C0/C1浅彩latent，只分解配对指标的精度与量化路径；不训练。"""
import argparse
import copy
from pathlib import Path
import subprocess
import time
import zipfile

import cv2
import numpy as np
import torch

from models.e2_lab_anchor import rgb_to_lab
from tools.e2_b_material_losses import highpass
from tools.e2_b_pair_finetune_losses import projection
from tools.e2_b_amplitude_response_metrics import projection_features
from tools import e2_reference_terminal_fp32_propose as proposal
from tools.e2_state_b_amplitude_response import read, write, sha, pixels
from tools.e2_state_b_backward_repro import tensor_sha
from tools.e2_oracle_texture import weights_digest


SOURCE = Path(__file__).resolve().parents[1]
EXTRA = ('tools/e2_reference_terminal_fp32_pair_metric_stages.py',
         'scripts/e2_reference_terminal_fp32_pair_metric_stages.sbatch')
STAGES = ('train32_native', 'train64_native', 'train32_current_Q8',
          'official_native_float_targets', 'official_native_PNG_targets', 'official_PNG')


def projection_typed(rgb, positive, base_positive, weight, dtype):
    """同原公式；float64仅作诊断参照，不接入生产loss或更新梯度。"""
    h, p, base = [highpass(rgb_to_lab(value.to(dtype))[:, :1]/100) for value in (rgb, positive, base_positive)]
    inside = (weight >= .999).to(dtype); count = inside.sum()
    mean = lambda value: (value*inside).sum()/count
    base_energy, target_energy = mean(base.square()), mean(p.square())
    assert base_energy > 1e-8 and target_energy > 1e-8
    x = mean(p*base)/base_energy; y = mean(h*base)/base_energy
    g = mean(h*p)/target_energy
    return dict(x=x, y=y, g=g, orth=mean((h-g*p).square())/target_energy)


def official_fields(value):
    return dict(x=value['fixed_target_projection'], y=value['fixed_output_projection'],
                g=value['own_target_gain'], orth=value['own_orthogonal_energy'])


def paired_numbers(arms, fp32):
    low, base, high = [arms[a] for a in ('a075', 'a100', 'a125')]
    span = high['x']-low['x']; output = high['y']-low['y']
    assert span > 1e-6
    if fp32:
        yh, yl, xh, xl, g = [torch.tensor(v, dtype=torch.float32) for v in
                            (high['y'], low['y'], high['x'], low['x'], base['g'])]
        pair_loss = float(((yh-yl)-(xh-xl)).square())
        absolute_loss = float((g-1).square())
    else:
        pair_loss, absolute_loss = (output-span)**2, (base['g']-1)**2
    return dict(target_span=span, output_span=output, center_slope=output/span,
        slope_absolute_error=abs(output/span-1), alpha1_gain=base['g'], gain_absolute_error=abs(base['g']-1),
        pair_loss=pair_loss, absolute_loss=absolute_loss)


def verify(args):
    parent_args = copy.copy(args); parent_args.out = args.proposals
    spec, _, _ = proposal.inputs(parent_args); proposal.verify(parent_args)
    protocol = read(args.metadata/'protocol.json')
    assert all(sha(args.metadata/n) == h for n, h in read(args.metadata/'manifest.json').items())
    assert all(sha(Path(n)) == h for n, h in protocol['input_sha256'].items())
    assert all(sha(SOURCE/n) == h for n, h in protocol['source_sha256'].items())
    summary = read(args.metadata/'result.json')
    assert summary['summary_completed'] and summary['source_integrity_verified']
    assert summary['model_or_metric_values_changed'] is False
    assert summary['original_runtime']['final_frozen_hash_checks_executed'] is False
    assert all(sha(args.replay/n) == h for n, h in read(args.replay/'manifest.json').items())
    return spec, protocol, summary


@torch.no_grad()
def run(args):
    from tools import e2_state_b_pilot as pilot
    from tools.e2_fft_experiment import environment
    started = time.monotonic()
    spec, previous, repaired = verify(args)
    source = proposal.fresh.direction.delta.post.q8.r16.source
    branch = proposal.fresh.direction
    observed = read(args.replay/'result.json')
    saved = torch.load(args.replay/'diagnostic.pt', map_location='cpu', weights_only=False)
    assert sha(args.replay/'diagnostic.pt') == observed['tensor_file_sha256']
    baseline = read(args.source_run/'evaluation/f68/replay/result.json')
    model_rows = dict(f68={(r['case'], r['sample_condition']): r for r in baseline['rows']},
        **{a: {(r['case'], r['sample_condition']): r for r in v['rows']} for a, v in repaired['models'].items()})
    selected = [r for r in spec['inputs'] if r['group'] == 'pastel']; assert len(selected) == 12
    assert not args.out.exists(), '不覆盖配对指标阶段诊断'
    args.out.mkdir(parents=True)
    files = [args.metadata/n for n in ('protocol.json', 'result.json', 'manifest.json')]
    files += [args.replay/n for n in ('protocol.json', 'result.json', 'manifest.json', 'diagnostic.pt')]
    for item in selected:
        folder = args.source_run/'cases'/item['case']
        files += [folder/(item['sample_condition']+'_positive.png'), folder/(item['condition']+'_positive.png'), folder/'mask.png']
    files = list(dict.fromkeys(files))
    protocol = dict(kind='fixed saved latent pair-metric stages only', selection='12 already-seen pastel conditions',
        stages=list(STAGES), original_S0_code_modified=False, original_loss_and_gates_unchanged=True,
        precision_reference='same original auxiliary formula in float64;diagnostic only',
        official_native='precise-L/CV highpass/official mask on continuous double RGB*255 without rounding',
        planned_vae_decodes=dict(native_half=12, fp32=36, total=48), planned_observations=36,
        planned_historical_PNG_controls=36, planned_historical_candidate_RGB_controls=24,
        new_S0=0, B_or_UNet_sampling_calls=0, reverse_passes=0, optimizer_steps=0,
        holdout_read=0, adopted_checkpoints=[], historical_final_hash_checks_recovered=False,
        input_sha256={str(p.resolve()): sha(p) for p in files},
        source_sha256=dict(previous['source_sha256'], **{n: sha(SOURCE/n) for n in EXTRA}),
        execution_commit=subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip())
    write(args.out/'protocol.json', protocol)
    result = dict(status='incomplete', passed=False, rows=[], cases={}, controls=[], completed_vae_decodes=0,
        historical_PNG_controls=0, historical_candidate_RGB_controls=0, historical_actual_RGB_controls=0,
        f68_auxiliary_controls=0, f68_native_RGB_controls=0, typed32_controls=0, new_S0=0, reverse_passes=0, optimizer_steps=0,
        B_or_UNet_sampling_calls=0, holdout_read=0, adopted_checkpoints=[])
    tensors = dict(points={}); error = None
    try:
        runtime = copy.copy(args); runtime.action = 'train'
        _, pipe, net = pilot.load_core(runtime)
        net.eval().requires_grad_(False); pipe.vae.float().eval().requires_grad_(False)
        net_hash, vae_hash = weights_digest(net), weights_digest(pipe.vae)
        assert net_hash == observed['frozen_hashes']['net'] and vae_hash == observed['frozen_hashes']['vae']
        result.update(environment=environment(), net_hash=net_hash, vae_hash=vae_hash)
        overlay = read(args.source_run/'one_step_overlay.json')
        for index, pair in enumerate(spec['train']):
            cache, _ = source.load_cache(args, index, overlay)
            gradients = torch.load(args.gradients/('gradients/shard%d.pt' % index), map_location='cpu', weights_only=False)
            chosen = [r for r in selected if r['case'] == pair['case_id']]; assert len(chosen) == 3
            folder = args.source_run/'cases'/pair['case_id']
            base_name = chosen[0]['condition']
            result['controls'].append(dict(case=pair['case_id'],
                base_PNG_equals_alpha1_PNG=bool(np.array_equal(pixels(folder/(base_name+'_positive.png')),
                    pixels(folder/(base_name+'_a100_positive.png'))))))
            mask_image = pixels(folder/'mask.png'); mask_image = (mask_image[..., 0] if mask_image.ndim == 3 else mask_image) > 0
            for item in chosen:
                case, condition = item['case'], item['sample_condition']; key = case+'/'+condition
                record = cache['records'][condition]; point = saved['points'][key]
                positive = record['target_record'][1].to(args.device)
                weight = record['target_record'][4].to(args.device)
                base_positive = cache['records'][item['condition']+'_a100']['target_record'][1].to(args.device)
                png_positive, png_base = [pixels(folder/n) for n in (condition+'_positive.png', item['condition']+'_positive.png')]
                zbase = point['base_latent'].to(args.device)
                pipe.vae.half()
                try:
                    native_half = branch.delta.post.q8.r16.native_rgb_half(pipe.vae, zbase)
                finally:
                    pipe.vae.float()
                zero = next(v for v in gradients['values'] if v['sample_condition'] == condition)
                assert tensor_sha(native_half.float()) == zero['native_RGB_sha256']
                result['f68_native_RGB_controls'] += 1
                result['completed_vae_decodes'] += 1
                reference = branch.delta.decode_raw32(pipe.vae, zbase.float()); result['completed_vae_decodes'] += 1
                tensors['points'][key] = dict(positive=positive.cpu(), weight=weight.cpu(), native_rgb_half=native_half.cpu(), outputs={})
                to_float_pixels = lambda value: value[0].permute(1, 2, 0).double().cpu().numpy()*255
                target_float, base_float = to_float_pixels(positive), to_float_pixels(base_positive)
                for model in ('f68', 'C0', 'C1'):
                    if model == 'f68':
                        rgb = native_half.float()
                    else:
                        z = point['candidates'][model]['latent'].to(args.device)
                        rgb = branch.delta.anchored_rgb(native_half, reference, branch.delta.decode_raw32(pipe.vae, z))
                        result['completed_vae_decodes'] += 1
                        assert tensor_sha(rgb) == model_rows[model][case, condition]['rgb_float32_sha256']
                        result['historical_candidate_RGB_controls'] += 1
                        if 'rgb' in point['candidates'][model]:
                            assert torch.equal(rgb.cpu(), point['candidates'][model]['rgb'])
                            result['historical_actual_RGB_controls'] += 1
                        tensors['points'][key]['outputs'][model] = rgb.cpu()
                    image = branch.delta.quantize(rgb)
                    old_row = model_rows[model][case, condition]
                    old_image = (args.source_run if model == 'f68' else args.replay)/old_row['image']
                    assert np.array_equal(image, pixels(old_image))
                    result['historical_PNG_controls'] += 1
                    values32 = projection(rgb, positive, base_positive, weight)
                    typed32 = projection_typed(rgb, positive, base_positive, weight, torch.float32)
                    assert all(torch.equal(values32[n], typed32[n]) for n in values32)
                    result['typed32_controls'] += 1
                    values32 = {n: float(v) for n, v in values32.items()}
                    if model == 'f68':
                        assert values32 == gradients['late'][item['arm']]['features']
                        result['f68_auxiliary_controls'] += 1
                    q8 = torch.from_numpy(image.copy()).permute(2, 0, 1)[None].to(args.device).float()/255
                    floating = to_float_pixels(rgb)
                    official_png = projection_features(image, png_positive, png_base, mask_image)
                    assert max(abs(official_png[n]-old_row['projection'][n]) for n in official_png) < 1e-12
                    stages = dict(train32_native=values32,
                        train64_native={n: float(v) for n, v in projection_typed(rgb, positive, base_positive, weight, torch.float64).items()},
                        train32_current_Q8={n: float(v) for n, v in projection(q8, positive, base_positive, weight).items()},
                        official_native_float_targets=official_fields(projection_features(floating, target_float, base_float, mask_image)),
                        official_native_PNG_targets=official_fields(projection_features(floating, png_positive, png_base, mask_image)),
                        official_PNG=official_fields(official_png))
                    result['rows'].append(dict(item, model=model, stages=stages, rgb_float32_sha256=tensor_sha(rgb),
                        pixel_array_sha256=tensor_sha(torch.from_numpy(image)), seed=pair['seed'], noise_digest=record['noise_digest']))
                result['controls'].append(dict(case=case, condition=condition,
                    training_vs_official_inside_pixel_disagreement=int(np.count_nonzero((weight[0, 0].cpu().numpy() >= .999) != (cv2.distanceTransform(mask_image.astype(np.uint8), cv2.DIST_L2, 5) >= 16)))))
            case_values = {}
            for model in ('f68', 'C0', 'C1'):
                case_values[model] = {}
                rr = {v['arm']: v for v in result['rows'] if v['case'] == pair['case_id'] and v['model'] == model}
                for stage in STAGES:
                    case_values[model][stage] = paired_numbers({a: rr[a]['stages'][stage] for a in rr}, stage.startswith('train32'))
            assert case_values['f68']['train32_native']['pair_loss'] == gradients['pair_loss']
            for model in ('C0', 'C1'):
                candidate = torch.load(args.proposals/(model+'.pt'), map_location='cpu', weights_only=False)
                vector = torch.cat([candidate['increments'][n].flatten() for n in source.INCREMENTS]).double()
                prediction = dict(pair_loss=float(gradients['pair_gradient'].double()@vector), absolute_loss=float(gradients['absolute_gradient'].double()@vector))
                case_values[model]['train32_directional_comparison'] = {n: dict(predicted=prediction[n],
                    actual=case_values[model]['train32_native'][n]-case_values['f68']['train32_native'][n]) for n in prediction}
            result['cases'][pair['case_id']] = case_values
            print('PAIR_METRIC_STAGES', len(result['cases']), result['completed_vae_decodes'], flush=True)
        assert (result['completed_vae_decodes'], result['historical_PNG_controls'], result['historical_candidate_RGB_controls'],
                result['f68_auxiliary_controls'], result['f68_native_RGB_controls'], result['typed32_controls']) == (48, 36, 24, 12, 12, 36)
        assert weights_digest(net) == net_hash and weights_digest(pipe.vae) == vae_hash
        result.update(status='complete', passed=True, net_and_vae_frozen_unchanged=True,
            historical_runtime_final_hash_checks_recovered=False, meaning_of_passed='source/decoder controls,not image quality or gradient qualification')
    except Exception as caught:
        error = caught; result.update(status='failed', passed=False, error_type=type(caught).__name__, message=str(caught))
    try:
        verify(args)
        assert all(sha(Path(n)) == h for n, h in protocol['input_sha256'].items())
        assert all(sha(SOURCE/n) == h for n, h in protocol['source_sha256'].items())
        result['source_final_verification'] = True
    except Exception as caught:
        error = error or caught
        result.update(status='failed', passed=False, source_final_verification=False, source_error=str(caught))
    result['seconds'] = time.monotonic()-started
    torch.save(tensors, args.out/'diagnostic.pt'); result['tensor_file_sha256'] = sha(args.out/'diagnostic.pt')
    write(args.out/'result.json', result)
    names = ['protocol.json', 'diagnostic.pt', 'result.json']
    write(args.out/'manifest.json', {n: sha(args.out/n) for n in names})
    with zipfile.ZipFile(args.out/'review.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for n in names+['manifest.json']:
            archive.write(args.out/n, n)
    print('PAIR_METRIC_STAGES_REVIEW', result['passed'], sha(args.out/'review.zip'), flush=True)
    if error is not None:
        raise error


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--project', type=Path, default=Path('/share/home/u2515283058/Mymodel'))
    args = parser.parse_args(); args.prior = args.prior.resolve(); args.project = args.project.resolve()
    auto = args.prior/'autonomous_repair_20261116'
    for name, value in dict(fixed='scaled_projection_retry', preflight='terminal_head_preflight',
        basis='reference_terminal_preflight_v2', source_run='reference_terminal_one_step',
        global_source='reference_terminal_global_amplitude', r16='reference_terminal_r16_diagnostic',
        q8='reference_terminal_q8_direction_diagnostic', post='reference_terminal_post_latent_probe',
        fp32='reference_terminal_fp32_delta_probe', direction='reference_terminal_fp32_direction',
        gradients='reference_terminal_fp32_gradients', proposals='reference_terminal_fp32_proposals',
        replay='reference_terminal_fp32_replay', metadata='reference_terminal_fp32_replay_summary',
        out='reference_terminal_fp32_pair_metric_stages').items():
        setattr(args, name, auto/value)
    args.data = args.project.parent/'datasets/BF'
    args.checkpoint = args.project/'output_eval/e2_state_b_pastel_followup_20261115/trust_retry/training/step1.pt'
    args.device = 'cuda:0'
    torch.set_num_threads(2); cv2.setNumThreads(2)
    run(args)
