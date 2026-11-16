"""固定f68和已保存的C0/C1参数方向；只读差分与缩步回放，不创建优化器。"""
import argparse
import copy
import hashlib
import os
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from tools.e2_fft_experiment import read, write, sha
from tools.e2_state_b_amplitude_response import CHECKPOINT_SHA
from tools.e2_state_b_pair_retry import PROTOCOL_SHA
from tools.e2_b_endpoint_losses import TASKS, flatten, set_parameters
from tools.e2_b_direction_diagnostics import interpolate, FrozenPattern


SOURCE = Path(__file__).resolve().parents[1]
EXTRA = (('validation_621', 'r2779_native', 'a125'),
         ('validation_1641', 'r2692_native', 'a075'),
         ('validation_870', 'r1033_native', 'a075'),
         ('validation_1526', 'r752_native', 'a125'))
PASTEL_PROBES = (('validation_621', 'r4157_native', 'a125'),
                 ('validation_1641', 'r5121_native', 'a100'),
                 ('validation_870', 'r6857_native', 'a075'),
                 ('validation_1526', 'r4157_native', 'a075'))
SCALES = (.25, .125)
STEPS = (.125, .03125)
VARIANTS = (.25, .125, -.125, .03125, -.03125)
SOURCES = ('tools/e2_state_b_direction_diagnostic.py', 'tools/e2_b_direction_diagnostics.py',
           'tools/e2_state_b_direction_summary.py', 'tools/e2_state_b_pair_finetune.py',
           'tools/e2_state_b_pair_evaluation.py', 'tools/e2_state_b_pair_summary.py',
           'tools/e2_state_b_endpoint.py', 'tools/e2_b_endpoint_losses.py',
           'tools/e2_b_balanced_losses.py', 'tools/e2_b_joint_losses.py',
           'tools/e2_b_garment_projection.py', 'tools/e2_state_b_repair.py',
           'models/e2_state_texture_b.py', 'scripts/e2_state_b_direction_diagnostic.sbatch')


def key(row):
    return row['case'], row['condition'], row['arm']


def tensor_sha(vector):
    return hashlib.sha256(vector.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def load_vectors(args):
    from models.e2_state_texture_b import StateTextureB
    from tools.e2_oracle_texture import weights_digest
    proposals = read(args.fixed / 'training/proposals.json')['arms']
    module = StateTextureB(color_channels=2, reference_channels=192)
    module.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=False)['b'])
    start = flatten(list(module.parameters()))
    initial_hash = weights_digest(module)
    targets = {}
    for name in ('C0', 'C1'):
        report = proposals[name]
        path = args.fixed / 'training' / (name + '.pt')
        assert sha(path) == report['checkpoint_sha256']
        assert report['checkpoint_start'] == CHECKPOINT_SHA and report['initial_b_hash'] == initial_hash
        assert report['parameter_updates'] == 1 and report['projection']['feasible']
        module.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['b'])
        assert weights_digest(module) == report['final_b_hash']
        targets[name] = flatten(list(module.parameters()))
        assert float((start - targets[name]).norm()) == report['actual_norm']
    return start, targets, initial_hash


def prepare(args):
    assert not (args.out / 'diagnostic_protocol.json').exists()
    assert sha(args.checkpoint) == CHECKPOINT_SHA
    assert sha(args.fixed / 'protocol.json') == PROTOCOL_SHA
    previous = read(args.fixed / 'run_provenance.json')
    manifest = read(args.fixed / 'manifest.json')
    spec = read(args.fixed / 'protocol.json')
    assert previous['cached_gradients_reused'] and previous['new_backward_passes'] == 0
    selected = [r for r in spec['inputs'] if r['group'] == 'pastel' or key(r) in EXTRA]
    sentinels = [r for r in selected if key(r) in EXTRA + PASTEL_PROBES]
    assert len(selected) == 16 and len(sentinels) == 8
    inputs, copied = {}, {}

    def record(source, expected=None):
        value = sha(source)
        if expected is not None:
            assert value == expected, str(source)
        inputs[str(source.resolve())] = value
        return value

    def duplicate(source, relative, expected=None):
        value = record(source, expected)
        target = args.out / relative
        assert not target.exists(), str(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        assert sha(target) == value
        copied[relative] = value

    args.out.mkdir(parents=True, exist_ok=True)
    duplicate(args.fixed / 'protocol.json', 'protocol.json', PROTOCOL_SHA)
    duplicate(args.fixed / 'training/proposals.json', 'fixed_proposals.json', manifest['training/proposals.json'])
    for name, digest in spec['assets'].items():
        duplicate(args.fixed / name, name, digest)
    record(args.checkpoint, CHECKPOINT_SHA)
    for name in ('C0', 'C1'):
        record(args.fixed / 'training' / (name + '.pt'), read(args.fixed / 'training/proposals.json')['arms'][name]['checkpoint_sha256'])
    for index in range(4):
        for suffix in ('pt', 'json'):
            name = 'gradients/shard%d.%s' % (index, suffix)
            record(args.fixed / name, previous['copied_sha256'][name])
    record(args.fixed / 'result.json', manifest['result.json'])
    record(args.fixed / 'run_provenance.json', manifest['run_provenance.json'])
    chosen = {key(r) for r in selected}
    for model in ('f68', 'C0', 'C1'):
        source = args.fixed / 'evaluation' / model / 'replay/result.json'
        report = read(source)
        record(source, manifest[source.relative_to(args.fixed).as_posix()])
        rows = []
        for original in report['rows']:
            if key(original) not in chosen:
                continue
            row = copy.deepcopy(original)
            relative = 'history/images/%s/%s' % (model, Path(row['image']).name)
            duplicate(args.fixed / row['image'], relative, row['png_sha256'])
            row.update(source_image=row['image'], image=relative, direction=model,
                       scale=0. if model == 'f68' else 1., role='history')
            rows.append(row)
        assert len(rows) == 16
        write(args.out / 'history' / (model + '.json'), dict(rows=rows))
    start, targets, initial = load_vectors(args)
    vectors = {}
    for model, target in targets.items():
        rows = {}
        for factor in (0., 1.) + VARIANTS:
            value = interpolate(start, target, factor)
            actual = value.double() - start.double()
            ideal = factor * (target.double() - start.double())
            rows[str(factor)] = dict(parameter_sha256=tensor_sha(value), norm=float(actual.norm()),
                changed_parameters=int((value != start).sum()),
                rounding_error_norm=float((actual - ideal).norm()))
        vectors[model] = rows
    write(args.out / 'diagnostic_protocol.json', dict(
        fixed_run=str(args.fixed), checkpoint=str(args.checkpoint), checkpoint_sha256=CHECKPOINT_SHA,
        original_protocol_sha256=PROTOCOL_SHA, initial_b_hash=initial,
        selected=selected, sentinels=sentinels, repeat_and_endpoint_controls=[list(x) for x in PASTEL_PROBES],
        shrink_scales=list(SCALES), difference_steps=list(STEPS), vectors=vectors,
        direction_definition='v = theta_saved_C - theta_f68; theta(lambda) = float32(theta_f68 + lambda*v) using FP64 interpolation',
        input_sha256=inputs, copied_sha256=copied, source_sha256={name: sha(SOURCE / name) for name in SOURCES},
        execution_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        planned_new_s0_images=140, planned_diagnostic_reverse_passes=56,
        optimizer_updates=0, continuous_finetuning=False, holdout_evaluated=False,
        full_eligibility_evaluated=False, adoption_allowed=False,
        task_tolerance=spec['acceptance']['task_tolerance'], sampler=spec['sampler'],
        policy='fixed two shrink scales, fixed two central-difference radii; no searching, no new optimizer or projection; restore f68 after each case',
        pattern_policy='true seven losses unchanged; frozen f68 focus and no-B base diagnostic reported separately for stop-gradient semantics'))
    print('DIRECTION_PREPARED', len(selected), len(sentinels), flush=True)


def verify(args):
    diagnostic = read(args.out / 'diagnostic_protocol.json')
    assert sha(args.out / 'protocol.json') == PROTOCOL_SHA
    assert all(sha(Path(name)) == digest for name, digest in diagnostic['input_sha256'].items())
    assert all(sha(args.out / name) == digest for name, digest in diagnostic['copied_sha256'].items())
    assert all(sha(SOURCE / name) == digest for name, digest in diagnostic['source_sha256'].items())
    print('DIRECTION_INPUTS_UNCHANGED', flush=True)
    return diagnostic


def run_case(args):
    from models.e2_state_texture_b import StateTextureB
    from tools import e2_state_b_pilot as pilot
    from tools import e2_state_b_endpoint as endpoint
    from tools import e2_state_b_repair as repair
    from tools.e2_state_b_pair_finetune import Objective
    from tools.e2_state_b_pair_evaluation import replay_inputs, score_entry, training_comparison
    from tools.e2_b_endpoint_losses import task_gradients
    from tools.e2_oracle_texture import weights_digest
    from tools.e2_fft_experiment import environment

    diagnostic = verify(args)
    spec = read(args.out / 'protocol.json')
    destination = args.out / ('case_%d.json' % args.case_index)
    assert not destination.exists()
    pair = spec['train'][args.case_index]
    case = pair['case_id']
    selected = {key(r) for r in diagnostic['selected'] if r['case'] == case}
    sentinel = {key(r) for r in diagnostic['sentinels'] if r['case'] == case}
    entries, _, _ = replay_inputs(args, spec)
    entries = [r for r in entries if key(r) in selected]
    assert len(entries) == 4 and len(sentinel) == 2
    history = {model: {key(r): r for r in read(args.out / 'history' / (model + '.json'))['rows']}
               for model in ('f68', 'C0', 'C1')}
    shard = torch.load(args.fixed / 'gradients' / ('shard%d.pt' % args.case_index), map_location='cpu', weights_only=False)
    assert shard['case'] == case and shard['checkpoint_sha256'] == CHECKPOINT_SHA
    cached = {key(item): shard['gradients'][i] for i, item in enumerate(shard['items'])}
    cached_values = {key(r): r['tasks'] for r in shard['values']}
    start, targets, initial = load_vectors(args)
    variants = {model: {scale: interpolate(start, target, scale) for scale in VARIANTS + (1.,)}
                for model, target in targets.items()}
    runtime = copy.copy(args)
    runtime.action = 'train'
    started = time.monotonic()
    core, pipe, net = pilot.load_core(runtime)
    b = StateTextureB(color_channels=2, reference_channels=192).to(args.device)
    parameters = list(b.parameters())
    set_parameters(parameters, start)
    b.eval().requires_grad_(False)
    net.eval().requires_grad_(False)
    frozen = weights_digest(net)
    assert initial == weights_digest(b) == shard['b_hash'] and frozen == shard['net_hash']
    pipe.vae.half()
    predictor = repair.FrozenPredictor(runtime, core, pipe, net, [pair],
        conditions=[r['sample_condition'] for r in entries if key(r) in sentinel])
    items = [r for r in spec['inputs'] if r['case'] == case]
    objective = Objective(runtime, pipe, predictor, b, pair, items)
    rows, controls, gradient_checks = [], [], []
    baseline, snapshots, patterns, fresh = {}, {}, {}, {}

    def sample(entry, model, scale, role):
        vector = start if model == 'f68' else variants[model][scale]
        set_parameters(parameters, vector)
        b.requires_grad_(False)
        before_hash = weights_digest(b)
        recorded = []
        def hook(module, positional, keywords, output):
            recorded.append(int(keywords['timestep']))
        with torch.no_grad(), b.register_forward_hook(hook, with_kwargs=True):
            image, digest, tap, latent = endpoint.final_sample(runtime, core, pipe, net, pair, entry['sample_condition'], b)
            assert recorded == pipe.scheduler.timesteps[25:].tolist() and tap.index == 50
            assert digest == entry['expected_digest']
            state = tap.states[-1]
            base = pipe.scheduler.step(state['epsilon'].to(args.device), pipe.scheduler.timesteps[-1],
                                       state['zt'].to(args.device), return_dict=False)[0].float()
            task_entry = dict(pair=pair, condition=entry['sample_condition'])
            tasks, details = objective.score_latents(task_entry, latent.float(), base,
                                                     residue=objective.metric(task_entry).surrogate)
            picture = np.asarray(image)
            label = ('%.5f' % scale).replace('-', 'm').replace('.', 'p')
            path = args.out / 'images' / case / ('%s_%s_%s_%s.png' % (entry['sample_condition'], model, label, role))
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)
            row = dict(case=case, condition=entry['condition'], arm=entry['arm'],
                sample_condition=entry['sample_condition'], group=entry['group'], factor=entry['factor'],
                direction=model, scale=scale, role=role, seed=pair['seed'], noise_digest=digest,
                b_calls=len(recorded), b_hash=before_hash, image=path.relative_to(args.out).as_posix(),
                png_sha256=sha(path), training_tasks=endpoint.numbers(tasks),
                frozen_pattern=None, **score_entry(entry, picture, 'replay'))
            row['training_task_comparison'] = training_comparison(row['training_tasks'], cached_values[key(entry)])
            if model != 'f68':
                row['parameters'] = diagnostic['vectors'][model][str(scale)]
            if key(entry) in patterns:
                row['frozen_pattern'] = float(patterns[key(entry)].pattern(objective.rgb(latent.float()),
                    tasks['pattern'].new_tensor(details['terms']['latent'])))
            if key(entry) in snapshots:
                reference = snapshots[key(entry)]['final'].to(args.device)
                row['latent_change_from_f68'] = dict(max_abs=float((latent.float() - reference.float()).abs().max()),
                    changed_elements=int((latent != reference).sum()))
        assert weights_digest(b) == before_hash
        rows.append(row)
        write(args.out / ('case_%d.partial.json' % args.case_index), dict(case=case, rows=rows, complete=False))
        print('DIRECTION_SAMPLE', args.case_index, len(rows), entry['sample_condition'], model, scale, role, flush=True)
        return row, dict(pair=pair, condition=entry['sample_condition'], states=tap.states,
                         digest=digest, final=latent.detach().cpu()), base.detach()

    for entry in entries:
        row, cached_entry, base = sample(entry, 'f68', 0., 'baseline')
        original = history['f68'][key(entry)]
        assert np.array_equal(np.asarray(Image.open(args.out / row['image'])), np.asarray(Image.open(args.out / original['image'])))
        error = max(abs(row['training_tasks'][t] - cached_values[key(entry)][t]) for t in TASKS)
        assert error <= 1e-6, ('f68_task_mismatch', key(entry), error)
        baseline[key(entry)], snapshots[key(entry)] = row, cached_entry
        controls.append(dict(kind='fresh_f68', case=case, condition=entry['condition'], arm=entry['arm'],
                             png_equal=True, task_max_abs_difference=error))
        if key(entry) not in sentinel:
            continue
        with torch.no_grad():
            _, positive, _, values, weight, _ = objective.records[case, entry['sample_condition']]
            patterns[key(entry)] = FrozenPattern(objective.rgb(cached_entry['final'].to(args.device).float()),
                positive, objective.rgb(base), weight, pair['patch_coordinates'])
            target = objective.records[case, entry['sample_condition']][0]
            latent_term = pilot.weighted_mean((cached_entry['final'].to(args.device).float() - target).square(), values['mask'])
            row['frozen_pattern'] = float(patterns[key(entry)].pattern(
                objective.rgb(cached_entry['final'].to(args.device).float()), latent_term))
            assert abs(row['frozen_pattern'] - row['training_tasks']['pattern']) <= 1e-6
        set_parameters(parameters, start)
        b.requires_grad_(True)
        pipe.scheduler.set_timesteps(50, device=args.device)
        predicted, base_rollout = objective.rollout(cached_entry)
        latent_error = float((predicted.detach().cpu() - cached_entry['final'].float()).abs().max())
        assert latent_error <= 2e-4
        tasks, _ = objective.score_latents(cached_entry, predicted, base_rollout,
                                           residue=objective.metric(cached_entry).surrogate)
        task_error = max(abs(float(tasks[t].detach()) - row['training_tasks'][t]) for t in TASKS)
        png_equal = np.array_equal(objective.pixels(predicted.detach()), np.asarray(Image.open(args.out / row['image'])))
        assert png_equal and task_error <= 1e-6
        current = task_gradients(tasks, parameters).cpu()
        fresh[key(entry)] = current
        old = cached[key(entry)]
        comparisons = {}
        for index, task in enumerate(TASKS):
            g0, g1 = old[index].double(), current[index].double()
            comparisons[task] = dict(max_abs_difference=float((g1 - g0).abs().max()),
                relative_l2=float((g1 - g0).norm() / g0.norm().clamp_min(1e-18)),
                cosine=float(g0 @ g1 / (g0.norm() * g1.norm()).clamp_min(1e-18)),
                cached_directional={m: float(g0 @ (v.double() - start.double())) for m, v in targets.items()},
                fresh_directional={m: float(g1 @ (v.double() - start.double())) for m, v in targets.items()})
        gradient_checks.append(dict(case=case, condition=entry['condition'], arm=entry['arm'],
            latent_max_abs_difference=latent_error, task_max_abs_difference=task_error,
            png_equal=png_equal, tasks=comparisons))
        b.requires_grad_(False)
        del predicted, base_rollout, tasks
        print('DIRECTION_GRADIENT_CHECK', args.case_index, entry['sample_condition'], flush=True)

    for entry in entries:
        if key(entry) in PASTEL_PROBES:
            repeated, _, _ = sample(entry, 'f68', 0., 'repeat')
            original = baseline[key(entry)]
            noise = {t: abs(repeated['training_tasks'][t] - original['training_tasks'][t]) for t in TASKS}
            controls.append(dict(kind='repeat_f68', case=case, condition=entry['condition'], arm=entry['arm'],
                png_equal=repeated['png_sha256'] == original['png_sha256'], task_noise=noise))
            assert repeated['png_sha256'] == original['png_sha256'] and max(noise.values()) == 0.
        for model in ('C0', 'C1'):
            if key(entry) in PASTEL_PROBES:
                endpoint_row, _, _ = sample(entry, model, 1., 'endpoint_control')
                original = history[model][key(entry)]
                error = max(abs(endpoint_row['training_tasks'][t] - original['training_tasks'][t]) for t in TASKS)
                equal = np.array_equal(np.asarray(Image.open(args.out / endpoint_row['image'])), np.asarray(Image.open(args.out / original['image'])))
                controls.append(dict(kind='saved_endpoint', case=case, condition=entry['condition'], arm=entry['arm'],
                    direction=model, png_equal=equal, task_max_abs_difference=error))
                assert equal and error <= 1e-6
            for factor in VARIANTS:
                if factor not in SCALES and key(entry) not in sentinel:
                    continue
                sample(entry, model, factor, 'shrink' if factor in SCALES else 'difference')

    by_variant = {(key(r), r['direction'], r['scale']): r for r in rows if r['role'] in ('shrink', 'difference')}
    differences = []
    for item in diagnostic['sentinels']:
        if item['case'] != case:
            continue
        k = key(item)
        repeat = next((c for c in controls if c['kind'] == 'repeat_f68' and
                       (c['case'], c['condition'], c['arm']) == k), None)
        for model in ('C0', 'C1'):
            for h in STEPS:
                plus, minus = by_variant[k, model, h], by_variant[k, model, -h]
                chord = (variants[model][h].double() - variants[model][-h].double()) / (2 * h)
                nominal = targets[model].double() - start.double()
                for index, task in enumerate(TASKS):
                    fd = (plus['training_tasks'][task] - minus['training_tasks'][task]) / (2 * h)
                    prediction = float(cached[k][index].double() @ chord)
                    fresh_prediction = float(fresh[k][index].double() @ chord)
                    frozen_fd = ((plus['frozen_pattern'] - minus['frozen_pattern']) / (2 * h)
                                 if task == 'pattern' else None)
                    differences.append(dict(case=case, condition=k[1], arm=k[2], task=task, direction=model, h=h,
                        fd=fd, prediction_cached=prediction, prediction_fresh=fresh_prediction,
                        nominal_prediction_cached=float(cached[k][index].double() @ nominal),
                        error=fd - prediction,
                        symmetric_relative_error=abs(fd - prediction) / max(abs(fd), abs(prediction), 1e-12),
                        sign_agrees=fd * prediction > 0., frozen_pattern_fd=frozen_fd,
                        frozen_pattern_error=None if frozen_fd is None else frozen_fd - prediction,
                        signal_abs=abs(plus['training_tasks'][task] - minus['training_tasks'][task]),
                        noise_abs=None if repeat is None else repeat['task_noise'][task],
                        plus_latent_change=plus['latent_change_from_f68'], minus_latent_change=minus['latent_change_from_f68']))
    set_parameters(parameters, start)
    assert weights_digest(b) == initial and weights_digest(net) == frozen
    assert len(rows) == 35 and len(differences) == 56
    write(destination, dict(case=case, rows=rows, controls=controls, fresh_gradient_checks=gradient_checks,
        finite_differences=differences, frozen_unchanged=True, b_restored=True, b_hash=initial, net_hash=frozen,
        optimizer_updates=0, new_reverse_passes=14, new_s0_images=len(rows),
        protocol_sha256=PROTOCOL_SHA, diagnostic_protocol_sha256=sha(args.out / 'diagnostic_protocol.json'),
        environment=environment(), seconds=time.monotonic() - started,
        pattern_semantics='frozen_pattern is a separate derivative diagnostic; actual losses and protection use training_tasks',
        heldout_used=False, adopted=False))
    print('DIRECTION_CASE_COMPLETE', args.case_index, len(rows), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'verify', 'case'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--fixed', type=Path, required=True)
    parser.add_argument('--project', type=Path, default=Path('/share/home/u2515283058/Mymodel'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--case-index', type=int, default=0)
    args = parser.parse_args()
    args.fixed, args.out = args.fixed.resolve(), args.out.resolve()
    args.data = args.project.parent / 'datasets/BF'
    args.amplitude_source = args.project / 'output_eval/e2_state_b_amplitude_response_20261116'
    args.checkpoint = args.project / 'output_eval/e2_state_b_pastel_followup_20261115/trust_retry/training/step1.pt'
    torch.set_num_threads(min(2, int(os.environ.get('SLURM_CPUS_PER_TASK', '2'))))
    cv2.setNumThreads(2)
    {'prepare': prepare, 'verify': verify, 'case': run_case}[args.action](args)


if __name__ == '__main__':
    main()
