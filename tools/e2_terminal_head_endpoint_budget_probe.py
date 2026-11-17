"""固定三条件×四状态的末步缓存诊断；零训练，不产生候选或完整组验收。"""
import argparse
import copy
from pathlib import Path
import time
import zipfile

import numpy as np
from PIL import Image
import torch
import cv2

from models.e2_lab_anchor import rgb_to_lab
from tools import e2_terminal_head_one_step as old, e2_terminal_head_qr_replay as source
from tools.e2_b_material_losses import highpass
from tools.e2_b_endpoint_losses import TASKS, flatten, set_parameters
from tools.e2_state_b_amplitude_response import read, write, sha, pixels
from tools.e2_state_b_backward_repro import tensor_sha, vector_summary
from tools.e2_state_b_pair_evaluation import replay_inputs, score_entry, training_comparison, PROTECTION
from tools.e2_oracle_texture import weights_digest


SOURCE = Path(__file__).resolve().parents[1]
PROBES = (('validation_870', 'r6857_native', 'a100'), ('validation_1641', 'r5121_native', 'a100'),
          ('validation_1641', 'r2692_native', 'a125'))
STATES = ('zero', 'C1', 'twice_C1', 'norm_raw')


def configurations(current, raw):
    assert current.dtype == raw.dtype == torch.float32 and current.norm() > 0 and raw.norm() > 0
    return dict(zero=torch.zeros_like(current), C1=current.clone(), twice_C1=current*2,
        norm_raw=-raw*(current.norm()/raw.norm()))


def amplitude_parts(rgb, positive, weight, coords):
    a, b = [highpass(rgb_to_lab(x)[:, :1]/100) for x in (rgb, positive)]
    inside = (weight >= .999).to(a.dtype)
    fields = [(a, b, inside)]+[(a[..., y:y+64, x:x+64], b[..., y:y+64, x:x+64],
        inside[..., y:y+64, x:x+64]) for x, y in coords]
    losses, records, continuous = [], [], None
    for index, (p, q, w) in enumerate(fields):
        count = w.sum().clamp_min(1)
        p2, q2, dot = (p.square()*w).sum()/count, (q.square()*w).sum()/count, (p*q*w).sum()/count
        ratio = ((p2+1e-10)/(q2+1e-10)).sqrt()
        cosine = dot/((p2+1e-10)*(q2+1e-10)).sqrt()
        valid = (q2.detach() > 1e-8).to(p.dtype)
        loss = valid*ratio.log().square()
        losses.append(loss)
        records.append(dict(index=index, scope='global' if index == 0 else 'patch',
            coordinates=None if index == 0 else coords[index-1], interior_pixels=int(w.sum()), valid=bool(valid),
            amplitude_stabilized=float(ratio), correlation_stabilized=float(cosine), log_amplitude_loss=float(loss)))
        if index == 0:
            assert p2 > 0 and q2 > 1e-8
            amplitude, correlation, gain = (p2/q2).sqrt(), dot/(p2*q2).sqrt(), dot/q2
            orth = ((p-gain*q).square()*w).sum()/count/q2
            continuous = dict(A=float(amplitude), c=float(correlation), g=float(gain), orth=float(orth),
                identity_residual=float(amplitude.square()-gain.square()-orth), absolute_gain_loss=float((gain-1).square()),
                target_energy=float(q2), output_energy=float(p2))
    return dict(parts=records, mean=float(torch.stack(losses).mean()), global_weight=1/len(fields),
        patch_mean=float(torch.stack(losses[1:]).mean()), continuous=continuous)


def pixel_comparison(current, baseline):
    changes = {name: None if current[name] is None or baseline[name] is None else current[name]-baseline[name] for name in PROTECTION}
    residue = baseline['old_projection']
    failed = residue is not None and current['old_projection'] is not None and current['old_projection'] > residue+1e-5+1e-4*abs(residue)
    return dict(delta=changes, per_image_residue_cap_failed=bool(failed),
        per_image_residue_tolerance='1e-5+1e-4*abs(f68)', original_group_gate_assessed=False,
        original_group_tolerances=dict(default=1e-5, old_projection=1e-8, correlation_direction='higher is better'))


@torch.no_grad()
def run(args):
    from tools import e2_state_b_pilot as pilot, e2_state_b_endpoint as endpoint
    from tools.e2_fft_experiment import environment
    assert not args.out.exists(), '不覆盖已有诊断'
    origin = copy.copy(args)
    origin.out = args.prior/'autonomous_repair_20261116/terminal_head_qr_replay'
    source_overlay, proposals = source.verify(origin)
    source_state = source.qualified_inputs(origin)
    spec = read(origin.out/'protocol.json')
    entries, _, _ = replay_inputs(origin, spec)
    entries = [next(row for row in entries if (row['case'], row['condition'], row['arm']) == key) for key in PROBES]
    reports = {arm: read(origin.out/'evaluation'/arm/'replay/result.json') for arm in ('f68', 'C1')}
    historical = {arm: {(r['case'], r['condition'], r['arm']): r for r in report['rows']} for arm, report in reports.items()}
    overlay = read(origin.out/'one_step_overlay.json')
    shards, _, _, qualification = old.gradient_inputs(origin, overlay, tensors=True)
    assert qualification['passed']
    gradients = torch.cat([shard['gradients'] for shard in shards]).flatten(0, 1)
    items = [item for shard in shards for item in shard['items']]
    head_cpu = old.load_candidate(origin, 'C1', 'cpu')
    vectors = configurations(flatten(head_cpu.increment_parameters()), source_state['raw']['raws']['C1']['raw'])
    inputs = {}
    def bind(path):
        path = Path(path).resolve()
        inputs[str(path)] = sha(path)
    for name in ('qr_replay_overlay.json', 'training/proposals.json', 'training/C1.pt', 'result.json',
                 'evaluation/f68/replay/result.json', 'evaluation/C1/replay/result.json'):
        bind(origin.out/name)
    for index in range(4):
        for extension in ('pt', 'json'):
            bind(origin.out/('gradients/shard%d.%s' % (index, extension)))
    selected_indices = sorted({next(i for i, pair in enumerate(spec['train']) if pair['case_id'] == row['case']) for row in entries})
    for index in selected_indices:
        bind(args.preflight/('cache/shard%d.pt' % index))
    assets = {}
    for entry in entries:
        for name in entry['assets']:
            path = entry['folder']/name
            bind(path)
            assets['assets/'+path.relative_to(origin.out).as_posix()] = path
        for arm in ('f68', 'C1'):
            bind(origin.out/historical[arm][entry['case'], entry['condition'], entry['arm']]['image'])
    sources = dict(source_overlay['source_sha256'])
    sources.update({name: sha(SOURCE/name) for name in ('tools/e2_terminal_head_endpoint_budget_probe.py',
        'scripts/e2_terminal_head_endpoint_budget_probe.sbatch', 'models/e2_lab_anchor.py',
        'tools/e2_b_material_losses.py', 'tools/e2_b_last_step_metrics.py', 'tools/e2_b_amplitude_response_metrics.py')})
    args.out.mkdir(parents=True)
    protocol = dict(probes=PROBES, states=STATES, fixed_configurations=12, input_sha256=inputs, source_sha256=sources,
        zero_point_only_gradient_predictions=True, counterfactuals=('twice_C1', 'norm_raw'),
        new_full_s0_samples=0, new_reverse_passes=0, slsqp_calls=0, adamw_steps=0,
        candidate_checkpoints_saved=0, holdout_results_read=0, adopted_checkpoints=[])
    write(args.out/'protocol.json', protocol)
    runtime = copy.copy(origin)
    runtime.action = 'train'
    _, pipe, net = pilot.load_core(runtime)
    torch.set_num_threads(2)
    net.eval().requires_grad_(False)
    head = old.load_candidate(origin, 'C1', args.device).requires_grad_(False)
    base_hash, net_hash = weights_digest(head.base), weights_digest(net)
    rows, tensors = [], dict(vectors=vectors, G=gradients, G_items=items, tasks=TASKS,
        N32=source_state['normalization']['N32'], cases={}, base_output_weight=head.base.output.weight.cpu(), base_output_bias=head.base.output.bias.cpu())
    caches = {index: old.load_cache(origin, index, overlay) for index in selected_indices}
    started = time.monotonic()
    for probe_index, entry in enumerate(entries):
        index = next(i for i, pair in enumerate(spec['train']) if pair['case_id'] == entry['case'])
        cache, key = caches[index], (entry['case'], entry['condition'], entry['arm'])
        assert cache['b_hash'] == base_hash and cache['net_hash'] == net_hash
        record = cache['records'][entry['sample_condition']]
        objective = old.objective_from_cache(runtime, pipe, head, entry['pair'], cache)
        pipe.scheduler.set_timesteps(50, device=args.device)
        task_entry = dict(pair=entry['pair'], condition=entry['sample_condition'])
        target_latent, positive, original_rgb, _, weight, _ = objective.records[entry['case'], entry['sample_condition']]
        row_index = next(i for i, item in enumerate(items) if (item['case'], item['sample_condition']) == (entry['case'], entry['sample_condition']))
        saved = dict(positive=positive.cpu(), original_rgb=original_rgb.cpu(), target_latent=target_latent.cpu(),
            weight=weight.cpu(), coords=entry['pair']['patch_coordinates'], no_B_latent=record['no_B_base'],
            no_B_rgb32=objective.rgb(record['no_B_base'].to(args.device)).cpu(),
            cache={name: record[name] for name in ('feature', 'mask', 'zt', 'raw_epsilon', 'alpha', 't', 'base_delta')}, states={})
        for name in STATES:
            vector = vectors[name]
            set_parameters(head.increment_parameters(), vector.to(args.device))
            latent, base, delta, corrected = old.preflight.terminal_replay(head, record, pipe.scheduler, args.device)
            assert tensor_sha(base) == tensor_sha(record['no_B_base'])
            tasks, _ = objective.score_latents(task_entry, latent.float(), base, residue=objective.metric(task_entry).surrogate)
            values = endpoint.numbers(tasks)
            rgb = objective.rgb(latent.float())
            parts = amplitude_parts(rgb, positive, weight, entry['pair']['patch_coordinates'])
            assert parts['mean'] == values['amplitude'] and len(parts['parts']) == 9
            picture = objective.pixels(latent)
            control = None
            if name in ('zero', 'C1'):
                prior = historical['f68' if name == 'zero' else 'C1'][key]
                assert values == prior['training_tasks'] and np.array_equal(picture, pixels(origin.out/prior['image']))
                expected = tensor_sha(record['final']) if name == 'zero' else prior['latent_sha256']
                assert tensor_sha(latent) == expected
                control = dict(tasks_equal=True, png_pixels_equal=True, latent_equal=True)
            target = args.out/('%d_%s.png' % (probe_index, name))
            Image.fromarray(picture).save(target)
            scored = score_entry(entry, picture, 'replay')
            baseline = historical['f68'][key]
            actual = {task: values[task]-baseline['training_tasks'][task] for task in TASKS}
            prediction = gradients[row_index*7:(row_index+1)*7].double()@vector.double()
            row = dict(probe_index=probe_index, case=entry['case'], condition=entry['sample_condition'], state=name,
                head_vector_sha256=tensor_sha(vector), head_norm=float(vector.norm()), control=control,
                image=target.name, png_sha256=sha(target), amplitude=parts, training_tasks=values,
                training_task_comparison=training_comparison(values, baseline['training_tasks']),
                png_comparison=pixel_comparison(scored['scores'], baseline['scores']),
                png=dict(A=scored['scores']['l_amplitude'], c=scored['scores']['l_correlation'],
                    g=scored['scores']['l_target_gain'], orth=scored['projection']['own_orthogonal_energy']),
                zero_point_prediction={task: float(prediction[j]) for j, task in enumerate(TASKS)},
                observed_task_delta=actual, output_correction=vector_summary(delta.cpu()-record['base_delta']), **scored)
            rows.append(row)
            saved['states'][name] = dict(latent=latent.cpu(), rgb32=rgb.cpu(), delta=delta.cpu(), corrected_epsilon=corrected.cpu())
            print('ENDPOINT_BUDGET', probe_index, name, parts['continuous']['A'], row['png']['A'], flush=True)
        tensors['cases'][probe_index] = saved
    assert len(rows) == 12 and weights_digest(head.base) == base_hash and weights_digest(net) == net_hash
    source.verify(origin)
    assert all(sha(Path(path)) == digest for path, digest in inputs.items())
    assert all(sha(SOURCE/path) == digest for path, digest in sources.items())
    torch.save(tensors, args.out/'diagnostic.pt')
    linear = {name: dict(predicted_task_increases=int(((gradients.double()@vector.double()) > 0).sum()),
        norm=float(vector.norm()), valid_at='zero head only; not a gradient at this diagnostic endpoint') for name, vector in vectors.items()}
    result = dict(protocol=protocol, rows=rows, zero_point_linear_bank=linear, configurations=12,
        controls_passed=all(row['control'] is None or all(row['control'].values()) for row in rows),
        original_full_group_protection_assessed=False, eligibility_assessed=False, pair_response_assessed=False,
        candidate_checkpoints_saved=0, adopted_checkpoints=[], frozen_unchanged=True, environment=environment(),
        seconds=time.monotonic()-started, tensor_sha256=sha(args.out/'diagnostic.pt'),
        interpretation='three-condition counterfactuals only; raw lacks protection, twice_C1 is not the fixed training step; no adoption or full-bank gate pass')
    write(args.out/'result.json', result)
    files = {name: args.out/name for name in ('protocol.json', 'diagnostic.pt', 'result.json')}
    files.update({row['image']: args.out/row['image'] for row in rows})
    files.update(assets)
    write(args.out/'manifest.json', {name: sha(path) for name, path in files.items()})
    with zipfile.ZipFile(args.out/'review.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            archive.write(path, name)
        archive.write(args.out/'manifest.json', 'manifest.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--project', type=Path, default=Path('/share/home/u2515283058/Mymodel'))
    parser.add_argument('--device', default='cuda:0')
    options = parser.parse_args()
    args = source.previous.make_args(options.out or options.prior/'autonomous_repair_20261116/terminal_head_endpoint_budget_probe', options.prior, options.project)
    args.device = options.device
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    run(args)


if __name__ == '__main__':
    main()
