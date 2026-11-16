"""f68固定loss scale1024的新鲜梯度与重复性资格检查；不自动生成候选。"""
import argparse
import copy
import os
import socket
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from tools.e2_b_endpoint_losses import TASKS, task_gradients
from tools.e2_state_b_amplitude_response import CHECKPOINT_SHA, read, write, sha, pixels
from tools.e2_state_b_backward_repro import compare_tensor, vector_summary
from tools.e2_state_b_direction_diagnostic import EXTRA, PASTEL_PROBES
from tools.e2_state_b_pair_retry import PROTOCOL_SHA, copy_verified


SCALE = 1024.
MAX_RELATIVE_L2 = .005
MIN_COSINE = .99999
SENTINELS = EXTRA + PASTEL_PROBES
SOURCE = Path(__file__).resolve().parents[1]
SOURCES = (
    'tools/e2_state_b_scaled_retry.py', 'scripts/e2_state_b_scaled_retry.sbatch',
    'tools/e2_state_b_pair_finetune.py', 'tools/e2_b_pair_finetune_losses.py',
    'tools/e2_state_b_pair_evaluation.py', 'tools/e2_state_b_pair_summary.py',
    'tools/e2_state_b_pair_retry.py', 'tools/e2_state_b_direction_diagnostic.py',
    'tools/e2_state_b_backward_repro.py', 'tools/e2_state_b_endpoint.py',
    'tools/e2_b_endpoint_losses.py', 'tools/e2_b_balanced_losses.py',
    'tools/e2_b_joint_losses.py', 'tools/e2_b_garment_projection.py',
    'tools/e2_state_b_balanced.py', 'tools/e2_state_b_garment.py',
    'tools/e2_state_b_material.py', 'tools/e2_state_b_repair.py',
    'tools/e2_state_b_pilot.py', 'tools/e2_state_b_joint_data.py',
    'tools/e2_state_b_material_data.py', 'tools/e2_fft_experiment.py',
    'tools/bf_t3_frozen_core.py', 'tools/m1_evaluation.py',
    'models/e2_state_texture_b.py', 'models/bf_sd_skip.py', 'adapter/attention_processor.py')


class NumericalFailure(RuntimeError):
    def __init__(self, label, tensor):
        self.details = dict(label=label, summary=vector_summary(tensor))
        super().__init__('非有限梯度：' + label)


def finite(value, label):
    if not torch.isfinite(value).all():
        raise NumericalFailure(label, value)
    return value


def scaled_tasks(values, parameters):
    # 数值记录仍用原values；只在反传入口缩放一次，在输出G处还原一次。
    return task_gradients({key: values[key] * SCALE for key in TASKS}, parameters) / SCALE


def scaled_scalar(value, parameters, retain_graph=False):
    parts = torch.autograd.grad(value * SCALE, parameters, retain_graph=retain_graph, allow_unused=True)
    return torch.cat([torch.zeros_like(p).flatten() if g is None else g.detach().flatten()
                      for p, g in zip(parameters, parts)]) / SCALE


def replica_report(current, baseline):
    rows = {task: compare_tensor(current[i], baseline[i]) for i, task in enumerate(TASKS)}
    passed = all(r['finite'] and r['relative_l2'] <= MAX_RELATIVE_L2 and r['cosine'] >= MIN_COSINE
                 for r in rows.values())
    return dict(passed=passed, tasks=rows, max_relative_l2=max(
        (r['relative_l2'] for r in rows.values() if r['finite']), default=None),
        minimum_cosine=min((r['cosine'] for r in rows.values() if r['finite']), default=None))


def row_key(row):
    return row['case'], row['condition'], row['arm']


def prepare(args):
    assert not args.out.exists(), '不覆盖已有实验'
    spec = read(args.fixed / 'protocol.json')
    manifest = read(args.fixed / 'manifest.json')
    prior_provenance = read(args.fixed / 'run_provenance.json')
    assert sha(args.fixed / 'protocol.json') == PROTOCOL_SHA
    assert sha(args.checkpoint) == spec['checkpoint_sha256'] == CHECKPOINT_SHA
    assert len(spec['inputs']) == 48 and len(spec['train']) == 4
    assert len([r for r in spec['inputs'] if row_key(r) in SENTINELS]) == 8
    selected = dict(spec['assets'], **{'protocol.json': PROTOCOL_SHA})
    inputs = {str(args.checkpoint): CHECKPOINT_SHA,
              str(args.fixed / 'manifest.json'): sha(args.fixed / 'manifest.json'),
              str(args.fixed / 'run_provenance.json'): sha(args.fixed / 'run_provenance.json')}
    for pair in spec['train']:
        source = args.data / pair['sketch']
        expected = spec['assets']['cases/' + pair['case_id'] + '/sketch.jpg']
        assert sha(source) == expected
        inputs[str(source)] = expected
    # 原PT只读用于身份/损失/特征对照，禁止复制成此次的新梯度。
    for index in range(4):
        for suffix in ('pt', 'json'):
            name = 'gradients/shard%d.%s' % (index, suffix)
            expected = prior_provenance['copied_sha256'][name]
            assert sha(args.fixed / name) == expected
            inputs[str(args.fixed / name)] = expected
    for split, count in (('replay', 48), ('holdout', 44)):
        name = 'evaluation/f68/%s/result.json' % split
        report = read(args.fixed / name)
        assert report['checkpoint_sha256'] == CHECKPOINT_SHA and report['protocol_sha256'] == PROTOCOL_SHA
        assert report['frozen_unchanged'] and len(report['rows']) == count
        selected[name] = manifest[name]
        for row in report['rows']:
            assert row['png_sha256'] == manifest[row['image']]
            selected[row['image']] = row['png_sha256']
    for name, expected in sorted(selected.items()):
        copy_verified(args.fixed / name, args.out / name, expected)
        inputs[str(args.fixed / name)] = expected
    for name, expected in (('result.json', spec['amplitude_source_result_sha256']),
                           ('generated.json', spec['amplitude_source_generated_sha256'])):
        assert sha(args.amplitude_source / name) == expected
        inputs[str(args.amplitude_source / name)] = expected
    overlay = dict(original_protocol_sha256=PROTOCOL_SHA, original_protocol_unchanged=True,
        starting_checkpoint_sha256=CHECKPOINT_SHA, loss_scale=SCALE,
        input_sha256=inputs, copied_sha256=selected,
        source_sha256={name: sha(SOURCE / name) for name in SOURCES},
        execution_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        fixed_source=str(args.fixed), sentinels=[list(row) for row in SENTINELS],
        replica_gate=dict(max_relative_l2=MAX_RELATIVE_L2, minimum_cosine=MIN_COSINE, require_all_finite=True),
        new_full_gradient_conditions=48, repeated_full_gradient_conditions=8,
        planned_reverse_passes=dict(full_tasks=336, replica_tasks=56, late_y=12, late_absolute=4),
        cached_gradients_reused=False, reused_f68_images=92, optimizer_updates=0,
        automatic_candidates=False, new_weights=False, holdout_used_for_training=False,
        change='all scalar backward seeds multiplied by1024; G divided by1024; original losses, projection, margins and proposal rules unchanged',
        policy='only prepare and fresh gradients now; qualification is numerical only and does not replace original pixel acceptance',
        png_order='initial no-grad S0 capture retains original sampler; all PNG checks and objective.pixels calls occur after every full/late/replica backward')
    write(args.out / 'scaled_overlay.json', overlay)
    verify(args)
    print('SCALED_RETRY_PREPARED', len(selected), flush=True)


def verify(args):
    overlay = read(args.out / 'scaled_overlay.json')
    assert sha(args.out / 'protocol.json') == PROTOCOL_SHA
    assert sha(args.checkpoint) == CHECKPOINT_SHA
    assert all(sha(Path(name)) == value for name, value in overlay['input_sha256'].items())
    assert all(sha(args.out / name) == value for name, value in overlay['copied_sha256'].items())
    assert all(sha(SOURCE / name) == value for name, value in overlay['source_sha256'].items())
    assert overlay['loss_scale'] == SCALE and not overlay['cached_gradients_reused']
    return overlay


def gradient_shard(args):
    from models.e2_state_texture_b import StateTextureB
    from tools import e2_state_b_pair_finetune as old
    from tools.e2_oracle_texture import weights_digest
    from tools.e2_fft_experiment import environment

    started = time.monotonic()
    overlay = verify(args)
    spec = read(args.out / 'protocol.json')
    pair = spec['train'][args.shard]
    items = [row for row in spec['inputs'] if row['case'] == pair['case_id']]
    sentinels = {row_key(row) for row in items if row_key(row) in SENTINELS}
    assert len(items) == 12 and len(sentinels) == 2
    dest = args.out / 'gradients'
    dest.mkdir(exist_ok=True)
    final = dest / ('shard%d.pt' % args.shard)
    failed = dest / ('shard%d.failed.json' % args.shard)
    assert not final.exists() and not failed.exists(), '不覆盖已完成或失败的shard'
    baseline = torch.load(args.fixed / 'gradients' / final.name, map_location='cpu', weights_only=False)
    assert baseline['case'] == pair['case_id'] and baseline['protocol_sha256'] == PROTOCOL_SHA
    old_values = {row['sample_condition']: row for row in baseline['values']}
    old_indices = {row['sample_condition']: index for index, row in enumerate(baseline['items'])}
    image_rows = {row_key(row): row for row in read(args.out / 'evaluation/f68/replay/result.json')['rows']}
    reports, replica_reports, gradients, replicas, late, cache, final_latents = [], [], {}, {}, {}, {}, {}
    replica_latents = {}
    try:
        runtime = copy.copy(args)
        runtime.action = 'train'
        core, pipe, net = old.pilot.load_core(runtime)
        torch.set_num_threads(2)
        cv2.setNumThreads(2)
        b = StateTextureB(color_channels=2, reference_channels=192).to(args.device)
        b.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=False)['b'])
        b.eval().requires_grad_(True)
        net.eval().requires_grad_(False)
        initial, frozen = weights_digest(b), weights_digest(net)
        assert initial == baseline['b_hash'] and frozen == baseline['net_hash']
        parameters = list(b.parameters())
        # 完整S0前缀与旧代码相同；此时没有活反传图，检查图片留到所有反传之后。
        for item in items:
            condition = item['sample_condition']
            picture, digest, tap, latent = old.endpoint.final_sample(args, core, pipe, net, pair, condition, b)
            assert len(tap.states) == 25 and digest == old_values[condition]['noise_digest']
            cache[condition] = dict(pair=pair, condition=condition, states=tap.states, digest=digest,
                final=latent.detach().cpu(), capture_png=np.asarray(picture).copy())
        pipe.vae.half()
        predictor = old.repair.FrozenPredictor(args, core, pipe, net, [pair],
            conditions=[row['sample_condition'] for row in items])
        objective = old.Objective(args, pipe, predictor, b, pair, items)
        pipe.scheduler.set_timesteps(50, device=args.device)
        for item in items:
            condition = item['sample_condition']
            entry = cache[condition]
            predicted, base = objective.rollout(entry)
            latent_control = compare_tensor(predicted.detach().cpu(), entry['final'])
            assert latent_control['finite'] and latent_control['max_abs'] <= 2e-4, '完整rollout超出原S0 latent容差'
            tasks, _ = objective.score_latents(entry, predicted, base, residue=objective.metric(entry).surrogate)
            finite(torch.stack([tasks[key].detach() for key in TASKS]), condition + '/unscaled_losses')
            values = old.endpoint.numbers(tasks)
            task_error = max(abs(values[key] - old_values[condition]['tasks'][key]) for key in TASKS)
            assert task_error <= 1e-6, '原七项loss超出历史容差'
            gradient = scaled_tasks(tasks, parameters).detach().cpu()
            for index, task in enumerate(TASKS):
                finite(gradient[index], condition + '/' + task)
            gradients[condition] = gradient
            final_latents[condition] = predicted.detach().cpu()
            reports.append(dict(item, tasks=values, noise_digest=entry['digest'],
                full_latent_max_error=latent_control['max_abs'], full_latent_equal=latent_control['equal'],
                full_png_equal=False, tasks_equal_old=values == old_values[condition]['tasks'],
                tasks_max_abs_old=task_error,
                gradient_vs_old={task: compare_tensor(gradient[i], baseline['gradients'][old_indices[condition], i])
                                 for i, task in enumerate(TASKS)}))
            if item['group'] == 'pastel':
                last = dict(entry, states=[entry['states'][20]])
                endpoint_z, _ = objective.rollout(last, hops=5)
                assert torch.equal(endpoint_z.detach(), predicted.detach()), '末5步与完整展开不一致'
                _, positive, _, _, weight, _ = objective.records[pair['case_id'], condition]
                base_positive = objective.records[pair['case_id'], item['condition'] + '_a100'][1]
                feature = old.projection(objective.rgb(endpoint_z), positive, base_positive, weight)
                finite(torch.stack([value.detach() for value in feature.values()]), condition + '/unscaled_late_features')
                feature_values = {key: float(value.detach()) for key, value in feature.items()}
                assert feature_values == baseline['late_features'][item['arm']], '未缩放的late特征改变'
                gy = finite(scaled_scalar(feature['y'], parameters, retain_graph=item['arm'] == 'a100'),
                            condition + '/late_y').cpu()
                row = dict(features=feature_values, gradient_y=gy)
                if item['arm'] == 'a100':
                    row['gradient_absolute'] = finite(scaled_scalar((feature['g']-1).square(), parameters),
                                                     condition + '/late_absolute').cpu()
                late[item['arm']] = row
                del endpoint_z, feature
            del predicted, base, tasks
            print('SCALED_GRADIENT', args.shard, condition, flush=True)
        # 八个预先指定的哨兵独立重建图，绝不把缓存G或第一个图当作重建结果。
        for item in items:
            if row_key(item) not in sentinels:
                continue
            condition = item['sample_condition']
            entry = cache[condition]
            predicted, base = objective.rollout(entry)
            latent_control = compare_tensor(predicted.detach().cpu(), entry['final'])
            same_forward = compare_tensor(predicted.detach().cpu(), final_latents[condition])
            assert latent_control['finite'] and latent_control['max_abs'] <= 2e-4
            tasks, _ = objective.score_latents(entry, predicted, base, residue=objective.metric(entry).surrogate)
            finite(torch.stack([tasks[key].detach() for key in TASKS]), condition + '/replica/unscaled_losses')
            values = old.endpoint.numbers(tasks)
            task_error = max(abs(values[key] - old_values[condition]['tasks'][key]) for key in TASKS)
            assert task_error <= 1e-6
            repeated = scaled_tasks(tasks, parameters).detach().cpu()
            for index, task in enumerate(TASKS):
                finite(repeated[index], condition + '/replica/' + task)
            replicas[condition] = repeated
            replica_latents[condition] = predicted.detach().cpu()
            replica_reports.append(dict(item, **replica_report(repeated, gradients[condition]),
                forward_equal=same_forward['equal'], forward_vs_main=same_forward,
                forward_vs_historical=latent_control, task_equal=values == old_values[condition]['tasks'],
                task_max_abs_old=task_error, png_equal=False))
            del predicted, base, tasks
            print('SCALED_REPLICA', args.shard, condition, replica_reports[-1]['passed'], flush=True)
        coefficient = old.pair_coefficients(*(torch.tensor(late[arm]['features'][key]) for arm, key in
            (('a075', 'y'), ('a125', 'y'), ('a075', 'x'), ('a125', 'x'))))
        pair_gradient = finite(coefficient['low'] * late['a075']['gradient_y'] +
            coefficient['high'] * late['a125']['gradient_y'], 'pair_gradient')
        assert pair_gradient.norm() > 0 and float(coefficient['pair']) == baseline['pair_loss']
        # 到这里所有反传均已结束，此后才允许pixels内部切换共享VAE精度。
        for item, report in zip(items, reports):
            condition = item['sample_condition']
            expected = pixels(args.out / image_rows[row_key(item)]['image'])
            assert np.array_equal(cache[condition]['capture_png'], expected), 'S0前缀图像改变'
            assert np.array_equal(objective.pixels(final_latents[condition].to(args.device)), expected), 'rollout PNG改变'
            report['full_png_equal'] = True
            if condition in replica_latents:
                assert np.array_equal(objective.pixels(replica_latents[condition].to(args.device)), expected), 'replica PNG改变'
                next(row for row in replica_reports if row['sample_condition'] == condition)['png_equal'] = True
        assert weights_digest(b) == initial and weights_digest(net) == frozen
        assert not any(p.grad is not None for p in parameters)
        verify(args)
        payload = dict(shard=args.shard, case=pair['case_id'], items=items, values=reports,
            gradients=torch.stack([gradients[row['sample_condition']] for row in items]),
            absolute_gradient=late['a100']['gradient_absolute'], pair_gradient=pair_gradient,
            replica_gradients=replicas, replicas=replica_reports,
            late_features={arm: row['features'] for arm, row in late.items()},
            pair_loss=float(coefficient['pair']), b_hash=initial, net_hash=frozen,
            checkpoint_sha256=CHECKPOINT_SHA, protocol_sha256=PROTOCOL_SHA,
            overlay_sha256=sha(args.out / 'scaled_overlay.json'), source_sha256=overlay['source_sha256'],
            frozen_unchanged=True, training_updates=0, baseline_parity=12, loss_scale=SCALE,
            all_finite=True, numerical_qualification=all(row['passed'] for row in replica_reports),
            new_gradient_generation=True, cached_gradients_reused=False, seconds=time.monotonic()-started,
            environment=environment(), hardware=dict(hostname=socket.gethostname(),
                job=os.environ.get('SLURM_JOB_ID'), node=os.environ.get('SLURM_JOB_NODELIST')),
            reverse_passes=dict(full_tasks=84, replica_tasks=14, late_y=3, late_absolute=1))
        old.balanced.atomic_torch(payload, final)
        metadata = {key: value for key, value in payload.items()
                    if key not in ('gradients', 'absolute_gradient', 'pair_gradient', 'replica_gradients')}
        metadata['tensor_sha256'] = sha(final)
        write(dest / ('shard%d.json' % args.shard), metadata)
        if not payload['numerical_qualification']:
            write(failed, dict(shard=args.shard, case=pair['case_id'], error_type='ReplicaQualificationFailure',
                error='预设重复性数值门槛未通过，拒绝生成proposal', replicas=replica_reports,
                loss_scale=SCALE, all_finite=True, candidate_generated=False, proposal_eligible=False,
                optimizer_updates=0, tensor_sha256=metadata['tensor_sha256']))
        print('SCALED_SHARD_COMPLETE', args.shard, payload['numerical_qualification'], metadata['tensor_sha256'], flush=True)
    except Exception as error:
        try:
            verify(args)
            unchanged, verification_error = True, None
        except Exception as verification:
            unchanged, verification_error = False, str(verification)
        write(failed, dict(shard=args.shard, case=pair['case_id'], error_type=type(error).__name__,
            error=str(error), details=getattr(error, 'details', None), values=reports, replicas=replica_reports,
            loss_scale=SCALE, complete=False, candidate_generated=False, optimizer_updates=0,
            proposal_eligible=False, inputs_and_sources_unchanged=unchanged, verification_error=verification_error,
            seconds=time.monotonic()-started))
        raise


def aggregate(args):
    verify(args)
    reports = [read(args.out / 'gradients' / ('shard%d.json' % i)) for i in range(4)]
    for index, report in enumerate(reports):
        assert sha(args.out / 'gradients' / ('shard%d.pt' % index)) == report['tensor_sha256']
        assert report['overlay_sha256'] == sha(args.out / 'scaled_overlay.json')
    replicas = [row for report in reports for row in report['replicas']]
    assert len(replicas) == 8 and {row_key(row) for row in replicas} == set(SENTINELS)
    result = dict(passed=all(r['all_finite'] and r['numerical_qualification'] for r in reports),
        all_finite=all(r['all_finite'] for r in reports), replicas=replicas,
        maximum_relative_l2=max(r['max_relative_l2'] for r in replicas),
        minimum_cosine=min(r['minimum_cosine'] for r in replicas),
        new_gradient_sha256={('shard%d.pt' % i): r['tensor_sha256'] for i, r in enumerate(reports)},
        loss_scale=SCALE, optimizer_updates=0, candidates_generated=False,
        policy='数值资格检查，不改变像素验收；候选生成需主流程单独决定，先replay再决定holdout')
    write(args.out / 'numerical_qualification.json', result)
    print('SCALED_NUMERICAL_QUALIFICATION', result['passed'], result['maximum_relative_l2'], result['minimum_cosine'], flush=True)
    if not result['passed']:
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'gradient', 'verify'))
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--project', type=Path, default=Path('/share/home/u2515283058/Mymodel'))
    parser.add_argument('--shard', type=int, choices=range(4), default=0)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.prior = args.prior.resolve()
    args.fixed = args.prior / 'projection_fp64_retry'
    args.out = (args.out or args.prior / 'autonomous_repair_20261116/scaled_retry').resolve()
    args.data = args.project.parent / 'datasets/BF'
    args.previous = args.project / 'output_eval/e2_state_b_material_20261113'
    args.amplitude_source = args.project / 'output_eval/e2_state_b_amplitude_response_20261116'
    args.checkpoint = args.project / 'output_eval/e2_state_b_pastel_followup_20261115/trust_retry/training/step1.pt'
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    {'prepare': prepare, 'gradient': gradient_shard, 'verify': aggregate}[args.action](args)


if __name__ == '__main__':
    main()
