"""零772参考直连的完整fresh G；原f68/S0前缀冻结，尚不训练。"""
import argparse
import copy
import inspect
from pathlib import Path
import subprocess
import time

import cv2
import torch
from torch.utils.checkpoint import checkpoint

from models.e2_reference_terminal_skip import ReferenceTerminalSkip, PARAMETERS
from tools import e2_reference_terminal_fp32_direction as direction
from tools.e2_reference_terminal_fp32_gradients import task_values
from tools import e2_b_endpoint_losses as legacy
from tools.e2_b_pair_finetune_losses import projection, pair_coefficients
from tools.e2_state_b_scaled_retry import SENTINELS, scaled_tasks, scaled_scalar, replica_report
from tools.e2_state_b_amplitude_response import read, write, sha
from tools.e2_state_b_backward_repro import tensor_sha
from tools.e2_oracle_texture import weights_digest


SOURCE = Path(__file__).resolve().parents[1]
EXTRA = ('models/e2_reference_terminal_skip.py', 'tools/e2_reference_terminal_skip_gradients.py',
         'scripts/e2_reference_terminal_skip_gradients.sbatch', 'tools/e2_reference_terminal_fp32_gradients.py')


def source_inputs(args):
    spec, previous, _, _ = direction.verify(args)
    old = read(args.direction/'protocol.json')
    assert all(sha(args.direction/n) == h for n, h in read(args.direction/'manifest.json').items())
    assert all(sha(Path(n)) == h for n, h in old['input_sha256'].items())
    assert all(sha(SOURCE/n) == h for n, h in old['source_sha256'].items())
    result = read(args.direction/'result.json')
    assert result['passed'] and result['frozen_unchanged'] and result['source_final_verification']
    assert (result['completed_reverse_passes'], result['completed_cache_observations']) == (88, 55)
    return spec, old


def prepare(args):
    spec, previous = source_inputs(args)
    assert not args.out.exists(), '不覆盖参考直连新梯度'
    args.out.mkdir(parents=True)
    files = [args.direction/n for n in ('protocol.json', 'result.json', 'diagnostic.pt', 'manifest.json')]
    files += [args.prior/'autonomous_repair_20261116/reference_terminal_fp32_gradients/gradients'/('shard%d.json'%i) for i in range(4)]
    write(args.out/'protocol.json', dict(kind='fresh complete48/7task G of new reference192-to-latent4 terminal skip',
        original_S0_code_modified=False, parameters=PARAMETERS, point='f68 plus zero772 reference skip only',
        forward='z=z_f68.float()+kappa32*conv1x1(reference192)*mask;native anchored FP32 decode',
        intervention='old f68 B and all old4616 increments frozen;new direct reference path only',
        reference='unchanged pixel-unshuffled luma64+spatial_ab128;no target in network input',
        functional_budget_matched_to_4616=False,
        baseline='new branch continuous native f68 RGB;original f68 PNG must remain exact for all48',
        loss='same seven task formulas,positive,mask,projection and weights;fresh baseline values for this forward',
        gradient_reuse=0, planned_main_VJP=336, planned_repeat_VJP=56, planned_y_VJP=12,
        planned_absolute_VJP=4, planned_total_VJP=408, repeat_groups=56,
        repeat_gate=dict(max_relative_l2=.005, minimum_cosine=.99999),
        original_losses=spec['replay_weights'], original_training=spec['training'],
        original_quality_gates=spec['acceptance'], original_step_changed=False,
        optimizer_steps=0, projection_solves=0, saved_candidates=0, new_S0=0, UNet_sampling_calls=0,
        holdout_read=0, adopted_checkpoints=[],
        input_sha256={str(p.resolve()): sha(p) for p in files},
        source_sha256=dict(previous['source_sha256'], **{n: sha(SOURCE/n) for n in EXTRA}),
        execution_commit=subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()))


def verify(args):
    spec, _ = source_inputs(args)
    protocol = read(args.out/'protocol.json')
    assert all(sha(Path(n)) == h for n, h in protocol['input_sha256'].items())
    assert all(sha(SOURCE/n) == h for n, h in protocol['source_sha256'].items())
    assert protocol['original_training'] == spec['training'] and protocol['original_quality_gates'] == spec['acceptance']
    return spec, protocol


def gradient(args):
    from tools import e2_state_b_pilot as pilot
    from tools.e2_fft_experiment import environment
    started = time.monotonic()
    spec, protocol = verify(args)
    dest = args.out/('gradients/shard%d.pt' % args.shard)
    assert not dest.exists()
    dest.parent.mkdir(exist_ok=True)
    r16 = direction.delta.post.q8.r16; source = r16.source
    overlay = read(args.source_run/'one_step_overlay.json')
    cache, _ = source.load_cache(args, args.shard, overlay)
    pair = spec['train'][args.shard]; items = [r for r in spec['inputs'] if r['case'] == pair['case_id']]
    runtime = copy.copy(args); runtime.action = 'train'; runtime.out = args.source_run
    _, pipe, net = pilot.load_core(runtime)
    net.eval().requires_grad_(False); head = source.load_head(args, args.device).requires_grad_(False)
    skip = ReferenceTerminalSkip().to(args.device).eval()
    parameters = list(skip.parameters())
    assert sum(v.numel() for v in parameters) == PARAMETERS
    objective = source.objective_from_cache(runtime, pipe, head, pair, cache)
    hashes = [weights_digest(v) for v in (head.base, net, pipe.vae)]
    metadata = read(args.basis/('cache/shard%d.json' % args.shard))
    assert hashes == [cache['b_hash'], cache['net_hash'], metadata['vae_hash_fp32']]
    assert sha(Path(inspect.getsourcefile(type(pipe.scheduler)))) == metadata['scheduler']['source_sha256']
    audit = source.preflight.scheduler_configuration(pipe.scheduler.config, metadata['scheduler']['config'])
    assert audit['normalized_equal']
    pipe.scheduler.set_timesteps(50, device=args.device)
    controls = {(v['case'], v['sample_condition']): v for v in read(args.r16/'result.json')['controls']}
    gradients, values, replicas, late = [], [], [], {}
    old_values = read(args.prior/'autonomous_repair_20261116/reference_terminal_fp32_gradients/gradients'/('shard%d.json'%args.shard))['values']
    old_values = {v['sample_condition']:v for v in old_values}
    for item in items:
        condition = item['sample_condition']; record = cache['records'][condition]
        entry = dict(pair=pair, condition=condition)
        with torch.no_grad():
            z0, no_b, base_b, _ = source.preflight.terminal_replay(head, record, pipe.scheduler, args.device)
            assert tensor_sha(z0) == tensor_sha(record['final'])
            assert tensor_sha(base_b) == tensor_sha(record['base_delta'])
            pipe.vae.half()
            try:
                native = r16.native_rgb_half(pipe.vae, z0).float()
            finally:
                pipe.vae.float()
            image = direction.delta.quantize(native)
            assert tensor_sha(torch.from_numpy(image)) == controls[pair['case_id'], condition]['pixel_array_sha256']
            reference = direction.delta.decode_raw32(pipe.vae, z0.float())
            no_b_rgb = objective.rgb(no_b).detach()
            mask = record['mask'].to(args.device)
            conditioning = record['target_record'][3]['reference'].to(args.device)
            phi = head.statistics(record['target_record'][3]['reference'].to(args.device))['phi']
            kappa = direction.delta.post.coefficient(torch.tensor(record['alpha'], device=args.device),
                pipe.scheduler.final_alpha_cumprod.to(args.device).float())
        def forward():
            difference = skip(conditioning, mask)
            z = z0.float()+kappa*difference
            current = checkpoint(lambda value: direction.delta.decode_raw32(pipe.vae, value), z, use_reentrant=False)
            rgb = direction.delta.anchored_rgb(native, reference, current)
            assert torch.equal(rgb.detach(), native)
            return z, rgb
        z, rgb = forward()
        tasks = task_values(objective, entry, z, rgb, no_b_rgb)
        numbers = {n: float(v.detach()) for n, v in tasks.items()}
        assert numbers == old_values[condition]['tasks'], '同f68零点七任务须逐值一致；旧G不进入新空间'
        current = scaled_tasks(tasks, parameters).detach().cpu()
        assert current.shape == (7, PARAMETERS) and torch.isfinite(current).all()
        gradients.append(current)
        values.append(dict(item, tasks=numbers, zero_PNG_equal=True,
            native_RGB_sha256=tensor_sha(native), pixel_array_sha256=tensor_sha(torch.from_numpy(image)),
            continuous_zero_latent_equal=tensor_sha(z.detach()) == tensor_sha(z0.float()), phi=float(phi)))
        del z, rgb, tasks
        if source.preflight.old.key(item) in SENTINELS:
            z, rgb = forward(); repeated_tasks = task_values(objective, entry, z, rgb, no_b_rgb)
            assert {n: float(v.detach()) for n, v in repeated_tasks.items()} == numbers
            repeated = scaled_tasks(repeated_tasks, parameters).detach().cpu()
            report = replica_report(repeated, current); assert report['passed']
            replicas.append(dict(item, **report, repeated_gradients=repeated))
            del z, rgb, repeated_tasks
        if item['group'] == 'pastel':
            z, rgb = forward()
            _, positive, _, _, weight, _ = objective.records[pair['case_id'], condition]
            anchor = objective.records[pair['case_id'], item['condition']+'_a100'][1]
            features = projection(rgb, positive, anchor, weight)
            numbers_late = {n: float(v.detach()) for n, v in features.items()}
            y = scaled_scalar(features['y'], parameters, retain_graph=item['arm'] == 'a100').cpu()
            value = dict(features=numbers_late, gradient_y=y)
            if item['arm'] == 'a100':
                value['gradient_absolute'] = scaled_scalar((features['g']-1).square(), parameters).cpu()
            late[item['arm']] = value
            del z, rgb, features
        print('REFERENCE_SKIP_FRESH_G', args.shard, condition, flush=True)
    assert len(gradients) == 12 and len(replicas) == 2 and set(late) == {'a075', 'a100', 'a125'}
    coefficients = pair_coefficients(*(torch.tensor(late[a]['features'][n]) for a, n in
        (('a075', 'y'), ('a125', 'y'), ('a075', 'x'), ('a125', 'x'))))
    pair_gradient = coefficients['low']*late['a075']['gradient_y']+coefficients['high']*late['a125']['gradient_y']
    assert torch.isfinite(pair_gradient).all() and torch.isfinite(late['a100']['gradient_absolute']).all()
    assert [weights_digest(v) for v in (head.base, net, pipe.vae)] == hashes
    assert not torch.count_nonzero(legacy.flatten(parameters))
    assert not any(p.grad is not None for model in (head, skip, net, pipe.vae) for p in model.parameters())
    verify(args)
    payload = dict(shard=args.shard, case=pair['case_id'], items=items, values=values, gradients=torch.stack(gradients),
        replicas=replicas, late=late, pair_gradient=pair_gradient, absolute_gradient=late['a100']['gradient_absolute'],
        pair_loss=float(coefficients['pair']), protocol_sha256=sha(args.out/'protocol.json'),
        parameter_count=PARAMETERS, all_finite=True, numerical_qualification=True, frozen_unchanged=True,
        optimizer_steps=0, new_S0=0, holdout_read=0, adopted=False,
        reverse_passes=dict(tasks=84, repeats=14, y=3, absolute=1),
        base_b_hash=hashes[0], net_hash=hashes[1], vae_hash_fp32=hashes[2],
        environment=environment(), scheduler_audit=audit, seconds=time.monotonic()-started)
    torch.save(payload, dest)
    meta = {n: v for n, v in payload.items() if n not in ('gradients', 'replicas', 'late', 'pair_gradient', 'absolute_gradient')}
    meta['replicas'] = [{n: v for n, v in r.items() if n != 'repeated_gradients'} for r in replicas]
    meta.update(tensor_sha256=sha(dest))
    write(dest.with_suffix('.json'), meta)


def qualify(args):
    spec, protocol = verify(args)
    shards = []; manifest = {'protocol.json': sha(args.out/'protocol.json')}
    for index in range(4):
        path = args.out/('gradients/shard%d.pt' % index); meta = read(path.with_suffix('.json'))
        assert sha(path) == meta['tensor_sha256']
        data = torch.load(path, map_location='cpu', weights_only=False)
        assert data['shard'] == index and data['protocol_sha256'] == sha(args.out/'protocol.json')
        assert data['case'] == spec['train'][index]['case_id']
        assert data['items'] == [r for r in spec['inputs'] if r['case'] == data['case']]
        assert data['all_finite'] and data['numerical_qualification'] and data['frozen_unchanged']
        assert data['gradients'].shape == (12, 7, PARAMETERS)
        assert all(torch.isfinite(data[n]).all() for n in ('gradients', 'pair_gradient', 'absolute_gradient'))
        assert len(data['values']) == 12 and all(v['zero_PNG_equal'] and v['continuous_zero_latent_equal'] for v in data['values'])
        assert data['reverse_passes'] == dict(tasks=84, repeats=14, y=3, absolute=1)
        assert len(meta['replicas']) == 2
        assert all(r['passed'] for r in meta['replicas'])
        for replica in data['replicas']:
            position = next(i for i,v in enumerate(data['items']) if v['sample_condition']==replica['sample_condition'])
            assert replica_report(replica['repeated_gradients'],data['gradients'][position])['passed']
        shards.append(meta)
        for p in (path, path.with_suffix('.json')):
            manifest[p.relative_to(args.out).as_posix()] = sha(p)
    result = dict(passed=True, status='complete', shards=shards, conditions=48,
        fresh_VJP=408, repeat_groups=56, zero_PNG_controls=48, new_S0=0, optimizer_steps=0,
        projection_solves=0, saved_candidates=0, holdout_read=0, adopted_checkpoints=[],
        meaning='complete new-G repeat/source qualification only;not candidate,PNG protection or training efficacy',
        protocol_sha256=sha(args.out/'protocol.json'), execution_commit=protocol['execution_commit'])
    write(args.out/'result.json', result); manifest['result.json'] = sha(args.out/'result.json')
    write(args.out/'manifest.json', manifest)
    import zipfile
    with zipfile.ZipFile(args.out/'review.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for n in list(manifest)+['manifest.json']:
            z.write(args.out/n, n)
    print('REFERENCE_SKIP_G_REVIEW', sha(args.out/'review.zip'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--project', type=Path, default=Path('/share/home/u2515283058/Mymodel'))
    parser.add_argument('--action', choices=('prepare', 'gradient', 'qualify'), required=True)
    parser.add_argument('--shard', type=int); parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(); args.prior = args.prior.resolve(); args.project = args.project.resolve()
    auto = args.prior/'autonomous_repair_20261116'
    for n, value in dict(fixed='scaled_projection_retry', preflight='terminal_head_preflight',
        basis='reference_terminal_preflight_v2', source_run='reference_terminal_one_step',
        global_source='reference_terminal_global_amplitude', r16='reference_terminal_r16_diagnostic',
        q8='reference_terminal_q8_direction_diagnostic', post='reference_terminal_post_latent_probe',
        fp32='reference_terminal_fp32_delta_probe', direction='reference_terminal_fp32_direction',
        out='reference_terminal_skip_gradients').items():
        setattr(args, n, auto/value)
    args.data = args.project.parent/'datasets/BF'
    args.checkpoint = args.project/'output_eval/e2_state_b_pastel_followup_20261115/trust_retry/training/step1.pt'
    torch.set_num_threads(2); cv2.setNumThreads(2)
    dict(prepare=prepare, gradient=gradient, qualify=qualify)[args.action](args)


if __name__ == '__main__':
    main()
