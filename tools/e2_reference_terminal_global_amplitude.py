"""同f68零4616点只替换全衣幅度优化方向；原336约束与全部验收保持。"""
import argparse
import copy
import inspect
from pathlib import Path
import shutil
import subprocess
import time
import zipfile

import cv2
import numpy as np
import torch

from tools import e2_reference_terminal_one_step as source
from tools.e2_b_endpoint_losses import TASKS, luminance, flatten, set_parameters
from tools.e2_state_b_backward_repro import tensor_sha, vector_summary, compare_tensor
from tools.e2_state_b_scaled_retry import SENTINELS, scaled_scalar, NumericalFailure
from tools.e2_state_b_amplitude_response import read, write, sha, pixels, CHECKPOINT_SHA
from tools.e2_state_b_pair_retry import PROTOCOL_SHA
from tools.e2_oracle_texture import weights_digest
from tools.e2_b_scaled_replay_review import Inputs, validate_report, replay_eligibility, same_numbers
from tools.e2_state_b_pair_evaluation import replay_inputs, score_entry, aggregate, strict_replay_protection, training_comparison


SOURCE = Path(__file__).resolve().parents[1]
ARM = 'G1'
EXTRA = ('tools/e2_reference_terminal_global_amplitude.py', 'scripts/e2_reference_terminal_global_amplitude.sbatch')


def source_args(args):
    value = copy.copy(args); value.out = args.source_run
    return value


def source_context(args):
    previous = source_args(args); overlay = source.verify(previous)
    _, values, hashes, qualification = source.gradient_inputs(previous, overlay)
    assert qualification == read(args.source_run/'gradient_qualification.json') and qualification['passed']
    result = read(args.source_run/'result.json')
    assert result['status'] == 'complete_replay' and result['complete_s0_images'] == 96
    proposals = read(args.source_run/'training/proposals.json')
    assert not proposals['stop_before_replay'] and proposals['new_reverse_passes'] == 352
    return previous, overlay, values, hashes, qualification, proposals


def global_loss(rgb, positive, weight):
    # 空coords只保留原首项；原epsilon、valid、Lab/highpass、mask及可微公式完全复用。
    return luminance(rgb, positive, weight, [])[0]['amplitude']


def replace_direction(saved_c1, source_gradients, global_gradients, weights):
    assert source_gradients.shape == (48, 7, source.PARAMETERS)
    assert global_gradients.shape == (48, source.PARAMETERS) and saved_c1.shape == (source.PARAMETERS,)
    amplitude = TASKS.index('amplitude')
    old_mean = source_gradients.mean(0)[amplitude]
    replacement = torch.tensor(weights, dtype=saved_c1.dtype)[amplitude]*(global_gradients.mean(0)-old_mean)
    return saved_c1+replacement, replacement


def prepare(args):
    _, previous, _, _, qualification, _ = source_context(args)
    assert not args.out.exists(), '不覆盖已有全衣幅度对照'
    spec = read(args.source_run/'protocol.json'); args.out.mkdir(parents=True)
    inputs, copied = {}, {}
    def record(path, expected=None):
        digest = sha(path); assert expected is None or digest == expected
        inputs[path.resolve().as_posix()] = digest
        return digest
    def duplicate(path, relative, expected=None):
        digest = record(path, expected); target = args.out/relative
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
        assert sha(target) == digest; copied[relative] = digest
    duplicate(args.source_run/'protocol.json', 'protocol.json', PROTOCOL_SHA)
    for name, digest in spec['assets'].items():
        duplicate(args.source_run/name, name, digest)
    for model in ('f68', 'C0', 'C1'):
        duplicate(args.source_run/('evaluation/%s/replay/result.json' % model), 'context/'+model+'.json')
    names = ['one_step_overlay.json', 'gradient_qualification.json', 'result.json', 'training/proposals.json',
        'training/raw.pt', 'training/numerics.pt', 'training/solver_diagnostic.json', 'evaluation/C1/replay/responses.pt']
    names += ['gradients/shard%d.%s' % (i, ext) for i in range(4) for ext in ('json', 'pt')]
    for name in names:
        record(args.source_run/name)
    record(args.checkpoint, CHECKPOINT_SHA)
    sources = set(previous['source_sha256']) | set(EXTRA)
    write(args.out/'global_protocol.json', dict(arm=ARM, source_run=str(args.source_run),
        parameter_count=source.PARAMETERS, source_point='f68 + four zero4616 increments; never source C1 endpoint',
        direction='saved_source_C1 + w_amplitude * (mean(G_global) - mean(G_original_amp9))',
        auxiliary_norm_match_coefficients='frozen implicitly in saved source C1 direction; no renormalization',
        constraint_matrix='source336 real zero-head G; amplitude remains original nine-term mean',
        original_loss='global plus eight patches, equal average', replacement_loss='luminance(rgb,positive,weight,[])[0][amplitude]',
        source_reverse_passes_reused=352, source_task_constraints_reused=336,
        new_global_main_vjp=48, new_global_replica_vjp=8, loss_scale=1024., sentinels=[list(v) for v in SENTINELS],
        replica_gate=dict(max_relative_l2=.005, minimum_cosine=.99999, finite=True),
        calibration=previous['calibration'], original_step=spec['training'], original_solver=previous['solver'],
        source_qualification_sha256=sha(args.source_run/'gradient_qualification.json'),
        source_gradient_sha256=qualification['new_gradient_sha256'],
        historical_image_root=str(args.source_run), historical_PNGs_copied=0,
        planned_new_S0=48, source_C1_saved_latent_global_head_decodes=48,
        extra_G1_global_head_decodes=48, source_C1_S0_reruns=0,
        optimizer_steps=1, slsqp_calls=1, source_QR_factorization_reused=True,
        heldout_used=False, adopted_checkpoints=[], continuous_training=False, functional_budget_matched_to_source=False,
        input_sha256=inputs, copied_sha256=copied, source_sha256={n: sha(SOURCE/n) for n in sorted(sources)},
        execution_commit=subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()))
    verify(args)


def verify(args):
    source_context(args)
    protocol = read(args.out/'global_protocol.json')
    assert protocol['arm'] == ARM and protocol['parameter_count'] == source.PARAMETERS
    assert sha(args.out/'protocol.json') == PROTOCOL_SHA
    assert all(sha(Path(p)) == h for p, h in protocol['input_sha256'].items())
    assert all(sha(args.out/p) == h for p, h in protocol['copied_sha256'].items())
    assert all(sha(SOURCE/p) == h for p, h in protocol['source_sha256'].items())
    proposal = args.out/'training/proposals.json'
    if proposal.exists():
        record = read(proposal)
        assert record['protocol_sha256'] == sha(args.out/'global_protocol.json')
        assert all(sha(args.out/p) == h for p, h in record['numerical_file_sha256'].items())
        _, _, hashes, qualification = gradient_inputs(args)
        assert qualification == read(args.out/'global_qualification.json') and qualification['passed']
        assert hashes == record['gradient_file_sha256']
        if not record['stop_before_replay']:
            assert read(args.out/'training/solver_diagnostic.json')['numerical_qualification']
            head = source.load_candidate(args, ARM, 'cpu', record['arms'][ARM])
            raw = torch.load(args.out/'training/raw.pt', map_location='cpu', weights_only=False)
            numeric = torch.load(args.out/'training/numerics.pt', map_location='cpu', weights_only=False)
            source_raw, source_numeric, _ = source_math(args)
            assert float(raw['raw'].norm()) == raw['report']['raw_norm']
            slack = source_numeric['N32'].double()@numeric['delta32'].double()/raw['report']['raw_norm']-source_raw['margins'].double()
            assert bool((slack >= -1e-6).all())
            expected, step = source.original_step(raw['raw'], numeric['delta32'], source_raw['constraints'], read(args.out/'protocol.json')['training'])
            assert all(record['arms'][ARM][k] == v for k, v in step.items())
            assert torch.equal(flatten(head.increment_parameters()), expected)
    return protocol


def source_math(args):
    raw = torch.load(args.source_run/'training/raw.pt', map_location='cpu', weights_only=False)
    numeric = torch.load(args.source_run/'training/numerics.pt', map_location='cpu', weights_only=False)
    diagnostic = read(args.source_run/'training/solver_diagnostic.json')
    assert diagnostic['all_qualified'] and diagnostic['qr']['passed']
    assert raw['constraints'].shape == (336, source.PARAMETERS) and bool(numeric['active'].all())
    assert tensor_sha(raw['constraints']) == diagnostic['constraints_sha256']
    assert tensor_sha(raw['margins']) == diagnostic['margins_sha256']
    assert tensor_sha(raw['directions']['C1']) == raw['raws']['C1']['direction_sha256']
    return raw, numeric, diagnostic


def gradient(args):
    from tools import e2_state_b_pilot as pilot
    from tools.e2_fft_experiment import environment
    started = time.monotonic(); protocol = verify(args)
    prior, source_overlay, old_values, _, _, _ = source_context(args)
    path = args.out/('gradients/shard%d.pt' % args.shard); assert not path.exists()
    path.parent.mkdir(exist_ok=True)
    spec = read(args.out/'protocol.json'); pair = spec['train'][args.shard]
    items = [r for r in spec['inputs'] if r['case'] == pair['case_id']]
    cache, _ = source.load_cache(prior, args.shard, source_overlay)
    metadata = read(args.basis/('cache/shard%d.json' % args.shard))
    runtime = copy.copy(args); runtime.action = 'train'
    _, pipe, net = pilot.load_core(runtime); torch.set_num_threads(2); cv2.setNumThreads(2)
    net.eval().requires_grad_(False); head = source.load_head(args, args.device); parameters = head.increment_parameters()
    objective = source.objective_from_cache(runtime, pipe, head, pair, cache)
    base_hash, net_hash, vae_hash = weights_digest(head.base), weights_digest(net), weights_digest(pipe.vae)
    assert base_hash == cache['b_hash'] and net_hash == cache['net_hash'] and vae_hash == metadata['vae_hash_fp32']
    assert sha(Path(inspect.getsourcefile(type(pipe.scheduler)))) == metadata['scheduler']['source_sha256']
    audit = source.preflight.scheduler_configuration(pipe.scheduler.config, metadata['scheduler']['config'])
    write(path.with_suffix('.scheduler.json'), audit); assert audit['normalized_equal']
    pipe.scheduler.set_timesteps(50, device=args.device)
    values, gradients, replicas, repeated_tensors = [], [], [], {}
    args.completed_global_vjp = 0
    for item in items:
        c = item['sample_condition']; record = cache['records'][c]
        samples = []
        for repeat in range(2 if source.preflight.old.key(item) in SENTINELS else 1):
            latent, _, delta, _ = source.preflight.terminal_replay(head, record, pipe.scheduler, args.device)
            assert tensor_sha(latent) == tensor_sha(record['final']) and tensor_sha(delta) == tensor_sha(record['base_delta'])
            _, positive, _, _, weight, _ = objective.records[pair['case_id'], c]
            phi = head.statistics(record['target_record'][3]['reference'].to(args.device))['phi']
            assert tensor_sha(phi) == old_values[item['case'], c]['phi_sha256']
            rgb = objective.rgb(latent.float())
            loss = global_loss(rgb, positive, weight)
            original = luminance(rgb, positive, weight, pair['patch_coordinates'])[0]['amplitude']
            assert float(original.detach()) == old_values[item['case'], c]['tasks']['amplitude']
            current = scaled_scalar(loss, parameters).detach().cpu(); args.completed_global_vjp += 1
            if not torch.isfinite(current).all():
                raise NumericalFailure(c+'/global%d' % repeat, current)
            assert current.shape == (source.PARAMETERS,)
            if repeat == 0:
                values.append(dict(item, global_loss=float(loss.detach()), original_amp9=float(original.detach()),
                    phi=float(phi), phi_sha256=tensor_sha(phi), zero_latent_equal=True, original_amplitude_equal=True,
                    noise_digest=record['noise_digest'], gradient_summary=vector_summary(current)))
                gradients.append(current)
            else:
                assert float(loss.detach()) == values[-1]['global_loss']
            samples.append(current)
            del latent, delta, rgb, loss, original
        if len(samples) == 2:
            comparison = compare_tensor(samples[1], samples[0])
            passed = comparison['finite'] and comparison['relative_l2'] <= .005 and comparison['cosine'] >= .99999
            replicas.append(dict(item, **comparison, passed=passed))
            repeated_tensors[c] = torch.stack(samples)
        print('GLOBAL_AMPLITUDE_G', args.shard, c, len(samples), flush=True)
    assert len(values) == 12 and len(replicas) == 2 and args.completed_global_vjp == 14
    assert not torch.count_nonzero(flatten(parameters)) and not any(p.grad is not None for p in head.parameters())
    assert weights_digest(head.base) == base_hash and weights_digest(net) == net_hash and weights_digest(pipe.vae) == vae_hash
    verify(args)
    payload = dict(shard=args.shard, case=pair['case_id'], values=values, global_gradients=torch.stack(gradients),
        repeated_gradients=repeated_tensors, replicas=replicas, all_finite=True, passed=all(r['passed'] for r in replicas),
        parameter_count=source.PARAMETERS, zero_head=True, base_b_hash=base_hash, net_hash=net_hash, vae_hash_fp32=vae_hash,
        loss_scale=1024., gradients_unscaled=True, global_main_vjp=12, global_replica_vjp=2, optimizer_updates=0,
        source_G_reused_not_regenerated=True, protocol_sha256=sha(args.out/'global_protocol.json'),
        source_sha256=protocol['source_sha256'], frozen_unchanged=True, environment=environment(), seconds=time.monotonic()-started)
    torch.save(payload, path)
    meta = {k: v for k, v in payload.items() if k not in ('global_gradients', 'repeated_gradients')}
    meta.update(tensor_sha256=sha(path), tensor_file=path.relative_to(args.out).as_posix())
    write(path.with_suffix('.json'), meta)
    if not payload['passed']:
        raise SystemExit(2)


def gradient_inputs(args):
    spec = read(args.out/'protocol.json'); arrays, values, hashes, replicas = [], {}, {}, []
    protocol = read(args.out/'global_protocol.json')
    for i in range(4):
        path = args.out/('gradients/shard%d.json' % i); meta = read(path); pt = path.with_suffix('.pt')
        assert meta['shard'] == i and meta['case'] == spec['train'][i]['case_id']
        assert sha(pt) == meta['tensor_sha256'] and meta['protocol_sha256'] == sha(args.out/'global_protocol.json')
        assert meta['parameter_count'] == source.PARAMETERS and meta['loss_scale'] == 1024. and meta['gradients_unscaled']
        assert meta['zero_head'] and meta['all_finite'] and meta['frozen_unchanged'] and meta['optimizer_updates'] == 0
        assert meta['source_sha256'] == protocol['source_sha256'] and (meta['global_main_vjp'], meta['global_replica_vjp']) == (12, 2)
        payload = torch.load(pt, map_location='cpu', weights_only=False)
        assert payload['values'] == meta['values'] and payload['global_gradients'].shape == (12, source.PARAMETERS)
        assert torch.isfinite(payload['global_gradients']).all()
        expected = [r for r in spec['inputs'] if r['case'] == spec['train'][i]['case_id']]
        assert [source.preflight.old.key(r) for r in meta['values']] == [source.preflight.old.key(r) for r in expected]
        arrays.append(payload['global_gradients']); replicas.extend(meta['replicas'])
        for row in meta['values']:
            values[row['case'], row['sample_condition']] = row
        hashes[path.name] = sha(path); hashes[pt.name] = sha(pt)
    assert len(values) == 48 and len(replicas) == 8 and {source.preflight.old.key(r) for r in replicas} == set(SENTINELS)
    qualification = dict(passed=all(r['passed'] for r in replicas), all_finite=True, replicas=replicas,
        maximum_relative_l2=max(r['relative_l2'] for r in replicas), minimum_cosine=min(r['cosine'] for r in replicas),
        parameter_count=source.PARAMETERS, new_global_main_vjp=48, new_global_replica_vjp=8,
        source_reverse_passes_reused=352, source_gradient_sha256=protocol['source_gradient_sha256'],
        gradient_file_sha256=hashes, protocol_sha256=sha(args.out/'global_protocol.json'))
    return torch.cat(arrays), values, hashes, qualification


def propose(args):
    protocol = verify(args); directory = args.out/'training'; assert not directory.exists()
    global_gradients, _, hashes, qualification = gradient_inputs(args)
    assert not (args.out/'global_qualification.json').exists()
    write(args.out/'global_qualification.json', qualification)
    if not qualification['passed']:
        raise SystemExit(2)
    raw_source, numeric_source, diagnostic_source = source_math(args)
    spec = read(args.out/'protocol.json'); cfg = spec['training']
    old_g = raw_source['constraints'].reshape(48, 7, source.PARAMETERS)
    direction, replacement = replace_direction(raw_source['directions']['C1'], old_g, global_gradients, spec['replay_weights'])
    head = source.load_head(args, 'cpu'); base_hash = weights_digest(head.base)
    raw, raw_report = source.raw_once(head.increment_parameters(), direction, cfg)
    directory.mkdir()
    torch.save(dict(raw=raw, report=raw_report, direction=direction, replacement=replacement,
        source_C1_direction_sha256=tensor_sha(raw_source['directions']['C1']),
        source_constraints_sha256=tensor_sha(raw_source['constraints'])), directory/'raw.pt')
    tensors = {}
    try:
        numeric, tensors = source.qr.solve_arm(numeric_source['N32'].double().numpy(), numeric_source['Q'].numpy(),
            raw, raw_report['raw_norm'], raw_source['margins'].double().numpy(), diagnostic_source['qr'])
    except Exception as error:
        numeric = dict(numerical_qualification=False, error_type=type(error).__name__, message=str(error))
    torch.save(tensors, directory/'numerics.pt'); write(directory/'solver_diagnostic.json', numeric)
    files = {p.relative_to(args.out).as_posix(): sha(p) for p in (directory/'raw.pt', directory/'numerics.pt', directory/'solver_diagnostic.json')}
    proposal = dict(arms={}, protocol_sha256=sha(args.out/'global_protocol.json'), parameter_count=source.PARAMETERS,
        stop_before_replay=not numeric['numerical_qualification'], source_reverse_passes_reused=352,
        new_reverse_passes=56, adamw_steps=1, slsqp_calls=1, qr_factorizations=0,
        source_QR_factorization_reused=True, saved_candidate_parameter_updates=0, sequential_training_updates=0,
        numerical_file_sha256=files, gradient_file_sha256=hashes, source_constraints_sha256=tensor_sha(raw_source['constraints']),
        source_margins_sha256=tensor_sha(raw_source['margins']), replacement=vector_summary(replacement),
        source_C1_direction_sha256=tensor_sha(raw_source['directions']['C1']), new_direction_sha256=tensor_sha(direction),
        auxiliary_norm_match_recomputed=False, adopted_checkpoints=[], holdout_results_read=0)
    if numeric['numerical_qualification']:
        candidate, step = source.original_step(raw, tensors['delta32'], raw_source['constraints'], cfg)
        proposal['stop_before_replay'] = not (step['parameter_updates'] == 1 and step['minimum_cosine'] >= -1e-5)
        if not proposal['stop_before_replay']:
            set_parameters(head.increment_parameters(), candidate)
            report = dict(step, **raw_report, arm=ARM, trainable_parameters=source.PARAMETERS,
                base_checkpoint_sha256=CHECKPOINT_SHA, base_b_hash=base_hash,
                initial_head_sha256=tensor_sha(torch.zeros_like(candidate)),
                protocol_sha256=PROTOCOL_SHA, global_protocol_sha256=sha(args.out/'global_protocol.json'),
                global_qualification_sha256=sha(args.out/'global_qualification.json'),
                projection=dict(method='same original QP, source QR basis, fixed SLSQP', feasible=True, tolerance=1e-6,
                    actual_slack_min=numeric['fp32_slack_min'], actual_violations=numeric['original_fp32_violations']),
                frozen_unchanged=True, adopted=False, acceptance='pending original complete48 S0 gates')
            path = directory/(ARM+'.pt')
            torch.save(dict(increments={name: getattr(head, name).detach().cpu() for name in source.INCREMENTS},
                calibration=source.preflight.LOCAL_CALIBRATION, base_checkpoint_sha256=CHECKPOINT_SHA, report=report), path)
            proposal['arms'][ARM] = dict(report, checkpoint_sha256=sha(path)); proposal['saved_candidate_parameter_updates'] = 1
        else:
            proposal['rejected_step_report'] = step
    assert weights_digest(head.base) == base_hash
    write(directory/'proposals.json', proposal); verify(args)
    print('GLOBAL_AMPLITUDE_PROPOSAL', not proposal['stop_before_replay'], flush=True)
    if proposal['stop_before_replay']:
        raise SystemExit(2)


def evaluate(args):
    from tools import e2_state_b_pilot as pilot, e2_state_b_endpoint as endpoint
    from tools.e2_fft_experiment import environment
    started = time.monotonic(); protocol = verify(args)
    prior, source_overlay, task_baselines, gradient_hashes, _, _ = source_context(args)
    _, global_baselines, _, qualification = gradient_inputs(args)
    assert qualification['passed'] and qualification == read(args.out/'global_qualification.json')
    proposals = read(args.out/'training/proposals.json'); assert not proposals['stop_before_replay']
    spec = read(args.out/'protocol.json'); destination = args.out/'evaluation'/ARM/'replay'
    assert not destination.exists(); destination.mkdir(parents=True)
    entries, _, _ = replay_inputs(args, spec)
    baseline = read(args.out/'context/f68.json'); old_c1 = read(args.out/'context/C1.json')
    old_rows = {source.summary.row_key(r, 'replay'): r for r in baseline['rows']}
    old_c1_rows = {(r['case'], r['sample_condition']): r for r in old_c1['rows']}
    old_response = torch.load(args.source_run/'evaluation/C1/replay/responses.pt', map_location='cpu', weights_only=False)
    assert sha(args.source_run/'evaluation/C1/replay/responses.pt') == old_c1['response_tensor_sha256']
    runtime = copy.copy(args); runtime.action = 'train'
    core, pipe, net = pilot.load_core(runtime); torch.set_num_threads(2); cv2.setNumThreads(2)
    net.eval().requires_grad_(False); head = source.load_candidate(args, ARM, args.device).eval().requires_grad_(False)
    base_hash, head_hash, net_hash = weights_digest(head.base), tensor_sha(flatten(head.increment_parameters())), weights_digest(net)
    assert net_hash == baseline['net_hash']
    rows, responses, scheduler_audits = [], {}, {}
    current_case, objective, cache = None, None, None
    for entry in entries:
        case, c = entry['case'], entry['sample_condition']
        if case != current_case:
            index = next(i for i, pair in enumerate(spec['train']) if pair['case_id'] == case)
            cache, _ = source.load_cache(prior, index, source_overlay)
            metadata = read(args.basis/('cache/shard%d.json' % index))
            assert cache['b_hash'] == base_hash and cache['net_hash'] == net_hash
            objective = source.objective_from_cache(runtime, pipe, head, entry['pair'], cache)
            assert weights_digest(pipe.vae) == metadata['vae_hash_fp32']
            assert sha(Path(inspect.getsourcefile(type(pipe.scheduler)))) == metadata['scheduler']['source_sha256']
            audit = source.preflight.scheduler_configuration(pipe.scheduler.config, metadata['scheduler']['config'])
            scheduler_audits[case] = audit; write(destination/'scheduler_audits.json', scheduler_audits)
            assert audit['normalized_equal']; current_case = case
        record = cache['records'][c]; head.reset_capture(); trace = []
        def capture(module, positional, keywords, output):
            trace.append(dict(timestep=int(keywords['timestep']), sha256=tensor_sha(output)))
        with head.register_forward_hook(capture, with_kwargs=True):
            image, digest, tap, latent = endpoint.final_sample(runtime, core, pipe, net, entry['pair'], c, head)
        assert tap.index == 50 and len(tap.states) == head.active_calls == len(trace) == 25
        assert digest == entry['expected_digest'] == record['noise_digest']
        prefix = [dict(index=s['index'], timestep=s['t'], zt=tensor_sha(s['zt']), raw_epsilon=tensor_sha(s['epsilon'])) for s in tap.states]
        assert prefix == record['prefix_states'] and trace[:24] == record['B_trace'][:24]
        saved = head.last_record; phi = saved['statistics']['phi']
        assert tensor_sha(saved['feature']) == tensor_sha(record['feature'])
        assert tensor_sha(saved['base_delta']) == tensor_sha(record['base_delta'])
        assert tensor_sha(saved['reference']) == tensor_sha(record['target_record'][3]['reference'])
        assert tensor_sha(phi) == task_baselines[case, c]['phi_sha256'] == global_baselines[case, c]['phi_sha256']
        cached, base, delta, corrected = source.preflight.terminal_replay(head, record, pipe.scheduler, args.device)
        assert tensor_sha(cached) == tensor_sha(latent) and tensor_sha(delta) == tensor_sha(saved['full_delta'])
        task_entry = dict(pair=entry['pair'], condition=c)
        tasks, _ = objective.score_latents(task_entry, latent.float(), base, residue=objective.metric(task_entry).surrogate)
        cached_tasks, _ = objective.score_latents(task_entry, cached.float(), base, residue=objective.metric(task_entry).surrogate)
        numbers = endpoint.numbers(tasks); assert numbers == endpoint.numbers(cached_tasks)
        _, positive, _, _, weight, _ = objective.records[case, c]
        # 只给历史C1保存latent补FP32头部观测，不加载C1增量、不新生成C1样本。
        historical_latent = old_response['rows'][case+'/'+c]['latent']
        assert tensor_sha(historical_latent) == old_c1_rows[case, c]['latent_sha256']
        with torch.no_grad():
            global_g1 = float(global_loss(objective.rgb(latent.float()), positive, weight))
            global_c1 = float(global_loss(objective.rgb(historical_latent.to(args.device).float()), positive, weight))
        picture = np.asarray(image).copy(); assert np.array_equal(picture, objective.pixels(cached))
        target = destination/'images'/(case+'_'+c+'.png'); target.parent.mkdir(exist_ok=True); image.save(target)
        effective, ideal = source.effective_difference(head, phi)
        alpha = pipe.scheduler.alphas_cumprod[1].to(args.device).float()
        original_epsilon = (record['raw_epsilon'].to(args.device).float()-(alpha/(1-alpha)).sqrt()*record['base_delta'].to(args.device)).half()
        assert tensor_sha(original_epsilon) == record['corrected_epsilon_sha256']
        correction = delta.cpu()-record['base_delta']; epsilon_change = corrected.cpu().float()-original_epsilon.cpu().float()
        latent_change = latent.cpu().float()-record['final'].float()
        response = dict(phi=phi.cpu(), effective_weight_difference=effective.cpu(), ideal_modulated_increment=ideal.cpu(),
            output_delta=delta.cpu(), output_correction=correction, corrected_epsilon=corrected.cpu(),
            corrected_epsilon_change=epsilon_change, latent=latent.cpu(), latent_change=latent_change)
        responses[case+'/'+c] = response
        original = pixels(args.source_run/old_rows[case, entry['condition'], entry['arm']]['image'])
        row = dict(case=case, condition=entry['condition'], sample_condition=c, arm=entry['arm'], factor=entry['factor'],
            group=entry['group'], model_arm=ARM, seed=entry['pair']['seed'], noise_digest=digest, b_calls=25,
            png_sha256=sha(target), image=target.relative_to(args.out).as_posix(), training_tasks=numbers,
            training_task_comparison=training_comparison(numbers, task_baselines[case, c]['tasks']),
            latent_sha256=tensor_sha(latent), full_cache_latent_equal=True, full_cache_tasks_equal=True,
            full_cache_png_equal=True, prefix_unchanged=True, actual_phi=float(phi),
            response_sha256={k: tensor_sha(v) for k, v in response.items()},
            effective_weight_difference=vector_summary(effective), ideal_modulated_increment=vector_summary(ideal),
            output_correction=vector_summary(correction), corrected_epsilon_change=vector_summary(epsilon_change),
            latent_change=vector_summary(latent_change), changed_pixel_channels=int(np.count_nonzero(picture != original)),
            global_loss=global_g1, source_C1_global_loss=global_c1,
            f68_global_loss=global_baselines[case, c]['global_loss'], source_C1_latent_sha256=tensor_sha(historical_latent),
            **score_entry(entry, picture, 'replay'))
        rows.append(row); write(destination/'partial.json', dict(rows=rows, complete=False))
        print('GLOBAL_AMPLITUDE_REPLAY', len(rows), case, c, flush=True)
    assert len(rows) == len(responses) == 48
    assert weights_digest(head.base) == base_hash and tensor_sha(flatten(head.increment_parameters())) == head_hash
    assert weights_digest(net) == net_hash and not any(p.grad is not None for p in head.parameters())
    assert weights_digest(pipe.vae) == metadata['vae_hash_fp32']
    torch.save(dict(rows=responses, base_checkpoint_sha256=CHECKPOINT_SHA, head_sha256=head_hash,
        source_cache_sha256=source_overlay['original_cache_sha256']), destination/'responses.pt')
    failures = [dict(case=r['case'], condition=r['condition'], arm=r['arm'], failures=r['training_task_comparison']['failures'])
        for r in rows if r['training_task_comparison']['failures']]
    verify(args)
    write(destination/'result.json', dict(arm=ARM, split='replay', rows=rows, **aggregate(rows, 'replay'),
        protocol_sha256=PROTOCOL_SHA, global_protocol_sha256=sha(args.out/'global_protocol.json'), source_sha256=protocol['source_sha256'],
        checkpoint_sha256=proposals['arms'][ARM]['checkpoint_sha256'], base_checkpoint_sha256=CHECKPOINT_SHA,
        base_b_hash=base_hash, head_hash=head_hash, net_hash=net_hash, frozen_unchanged=True,
        evaluation_optimizer_updates=0, heldout_used_for_training=False, heldout_used_for_evaluation=False,
        adopted_checkpoints=[], complete_s0_images=48, source_C1_S0_reruns=0, new_reverse_passes=0,
        source_C1_saved_latent_global_head_decodes=48, extra_G1_global_head_decodes=48, environment=environment(),
        response_tensor_sha256=sha(destination/'responses.pt'), scheduler_audits_sha256=sha(destination/'scheduler_audits.json'),
        seconds=time.monotonic()-started, scope='same48 seen conditions; original336 protection; no holdout/adoption',
        training_task_protection=dict(passed=not failures, failures=failures, comparisons=336,
            tolerance='1e-5+1e-4*abs(initial)', baseline_gradient_json_sha256=gradient_hashes),
        strict_replay_protection=strict_replay_protection(rows, old_rows)))


def review(args):
    protocol = verify(args); assert not (args.out/'result.json').exists()
    proposals_path = args.out/'training/proposals.json'
    proposals = read(proposals_path) if proposals_path.exists() else None
    qualified = args.out/'global_qualification.json'
    # 受控失败仍归档已完成证据；不把部分梯度/回放计成完整56/48。
    completed_vjp = 0
    for i in range(4):
        path = args.out/('gradients/shard%d.json' % i)
        if path.exists():
            row = read(path); completed_vjp += row['global_main_vjp']+row['global_replica_vjp']
        elif (args.out/('gradient_%d.failure.json' % i)).exists():
            completed_vjp += read(args.out/('gradient_%d.failure.json' % i)).get('completed_global_vjp', 0)
    result = dict(phase='global-only amplitude direction G1 versus saved source C1', protocol=protocol,
        proposals=proposals, qualification=read(qualified) if qualified.exists() else None,
        source_reverse_passes_reused=352, new_reverse_passes=completed_vjp,
        adamw_steps=int((args.out/'training/raw.pt').exists()), slsqp_calls=int((args.out/'training/numerics.pt').exists()),
        qr_factorizations=0, sequential_training_updates=0, complete_s0_images=0,
        source_C1_S0_reruns=0, historical_PNGs_copied=0, holdout_results_read=0, adopted_checkpoints=[], f68_unchanged=True,
        status='stopped_or_incomplete', baseline_image_root=str(args.source_run))
    path = args.out/'evaluation/G1/replay/result.json'
    if proposals and not proposals['stop_before_replay'] and path.exists():
        audit, spec = Inputs(), read(args.out/'protocol.json')
        _, _, values, hashes, _, _ = source_context(args)
        _, _, _, qualification = gradient_inputs(args)
        assert qualification == read(qualified) and qualification['passed'] and completed_vjp == 56
        baseline = audit.read(args.out/'context/f68.json'); old_c0 = audit.read(args.out/'context/C0.json')
        old_c1 = audit.read(args.out/'context/C1.json'); replay = audit.read(path); proposal = proposals['arms'][ARM]
        assert replay['global_protocol_sha256'] == sha(args.out/'global_protocol.json') and replay['source_sha256'] == protocol['source_sha256']
        assert replay['checkpoint_sha256'] == proposal['checkpoint_sha256'] and replay['head_hash'] == proposal['head_sha256']
        assert replay['base_b_hash'] == proposal['base_b_hash'] == baseline['b_hash'] and replay['net_hash'] == baseline['net_hash']
        assert replay['base_checkpoint_sha256'] == CHECKPOINT_SHA and replay['complete_s0_images'] == 48 and replay['new_reverse_passes'] == 0
        validate_report(replay, ARM, spec, PROTOCOL_SHA, values, hashes, audit, args.out)
        base_rows = {source.summary.row_key(r, 'replay'): r for r in baseline['rows']}
        assert same_numbers(replay['strict_replay_protection'], strict_replay_protection(replay['rows'], base_rows))
        assert all(all(r[k] for k in ('prefix_unchanged', 'full_cache_latent_equal', 'full_cache_tasks_equal', 'full_cache_png_equal')) for r in replay['rows'])
        folder = path.parent
        assert sha(folder/'responses.pt') == replay['response_tensor_sha256']
        assert sha(folder/'scheduler_audits.json') == replay['scheduler_audits_sha256']
        responses = torch.load(folder/'responses.pt', map_location='cpu', weights_only=False)
        assert responses['head_sha256'] == proposal['head_sha256'] and len(responses['rows']) == 48
        for row in replay['rows']:
            tensors = responses['rows'][row['case']+'/'+row['sample_condition']]
            assert {k: tensor_sha(v) for k, v in tensors.items()} == row['response_sha256']
        decision = replay_eligibility(proposal, replay, baseline)
        # 只适配原纯汇总函数的角色名，不修改ARMS/报告/模型身份或验收公式。
        roles = {'f68': {'replay': baseline}, 'C0': {'replay': old_c0}, 'C1': {'replay': replay}}
        innovation = source.summary.innovation(roles, {'C1': {'eligible': decision['holdout_candidate']}})
        reports = {'f68': baseline, 'C0': old_c0, 'C1': old_c1, ARM: replay}
        result.update(status='complete_replay', complete_s0_images=48, decision=decision, innovation=innovation,
            innovation_role_mapping=dict(f68='source f68', C0='source C0', C1='new G1'),
            eligibility_scope='original replay only; never automatic holdout/adoption',
            source_C1_saved_latent_global_head_decodes=48, extra_G1_global_head_decodes=48,
            groups={k: r['groups'] for k, r in reports.items()},
            comparisons={ARM+'_minus_'+name: dict(paired=source.summary.paired_changes(report, replay, 'replay'),
                response=source.summary.response_changes(report, replay)) for name, report in reports.items() if name != ARM},
            global_loss=dict(f68_mean=source.summary.mean([r['f68_global_loss'] for r in replay['rows']]),
                source_C1_mean=source.summary.mean([r['source_C1_global_loss'] for r in replay['rows']]),
                G1_mean=source.summary.mean([r['global_loss'] for r in replay['rows']]),
                interpretation='same original global luminance formula; observational only, not a new quality gate'),
            review_input_sha256=audit.hashes)
    elif (args.out/'evaluation/G1/replay/partial.json').exists():
        result['partial_s0_rows'] = len(read(args.out/'evaluation/G1/replay/partial.json')['rows'])
    verify(args); write(args.out/'result.json', result)
    # 新目录仅含48张候选图；历史图保留在已SHA绑定的source目录，不重复归档。
    excluded = set(protocol['copied_sha256'])-{'protocol.json', 'context/f68.json', 'context/C0.json', 'context/C1.json'}
    names = sorted(p.relative_to(args.out).as_posix() for p in args.out.rglob('*')
        if p.is_file() and p.suffix in ('.json', '.pt', '.png') and p.name != 'manifest.json'
        and p.relative_to(args.out).as_posix() not in excluded)
    if result['status'] == 'complete_replay':
        assert sum(n.endswith('.png') for n in names) == 48
    write(args.out/'manifest.json', {name: sha(args.out/name) for name in names})
    with zipfile.ZipFile(args.out/'review.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in names+['manifest.json']:
            archive.write(args.out/name, name)
    print('GLOBAL_AMPLITUDE_REVIEW', result['status'], sha(args.out/'review.zip'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'gradient', 'propose', 'evaluate', 'review', 'verify'))
    parser.add_argument('--prior', type=Path, required=True); parser.add_argument('--out', type=Path)
    parser.add_argument('--source-run', type=Path); parser.add_argument('--project', type=Path, default=Path('/share/home/u2515283058/Mymodel'))
    parser.add_argument('--shard', type=int, choices=range(4), default=0); parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(); args.prior = args.prior.resolve(); args.project = args.project.resolve()
    args.fixed = args.prior/'autonomous_repair_20261116/scaled_projection_retry'
    args.preflight = args.prior/'autonomous_repair_20261116/terminal_head_preflight'
    args.basis = args.prior/'autonomous_repair_20261116/reference_terminal_preflight_v2'
    args.source_run = (args.source_run or args.prior/'autonomous_repair_20261116/reference_terminal_one_step').resolve()
    args.out = (args.out or args.prior/'autonomous_repair_20261116/reference_terminal_global_amplitude').resolve()
    args.data = args.project.parent/'datasets/BF'
    args.checkpoint = args.project/'output_eval/e2_state_b_pastel_followup_20261115/trust_retry/training/step1.pt'
    torch.set_num_threads(2); cv2.setNumThreads(2)
    try:
        dict(prepare=prepare, gradient=gradient, propose=propose, evaluate=evaluate, review=review, verify=verify)[args.action](args)
    except Exception as error:
        if args.out.exists():
            name = '%s_%s.failure.json' % (args.action, args.shard if args.action == 'gradient' else ARM)
            if not (args.out/name).exists():
                write(args.out/name, dict(error_type=type(error).__name__, message=str(error), details=getattr(error, 'details', None),
                    completed_global_vjp=getattr(args, 'completed_global_vjp', 0), complete=False, adopted_checkpoints=[]))
        raise


if __name__ == '__main__':
    main()
