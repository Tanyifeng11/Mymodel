"""从零残差训练 E2 单头 B，全程采用 127619 A 的损失。"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from tools import e2_b_single_pants_cov_ab as cov
from tools.e2_fft_experiment import SOURCE

single, previous, init, tail = cov.single, cov.previous, cov.init, cov.tail
RUN = 'direct_h2_alexnet_960'
ARM = 'direct'
CHECKPOINTS = (256, 512, 768, 960)
CATEGORIES = tuple(tail.distill.CATEGORIES)
HISTORY = (
    (127194, 768, 'H2', '4de6fe244e5c5c319a8074a177d5f8946c24dcdfe4b61165dfab01af83c3ec0a'),
    (127459, 64, 'A', 'a908a1a339d27db7323c8f349b41d982f6c324d515075b70c087a1f53df64ce2'),
    (127556, 64, 'B', '2b0fddcac11cf66d9cd436b21850ebf04b9e9ac72513a40b0017c89763867f05'),
    (127619, 64, 'A', '897246cc75c29e11c124d952b3e4cbe49c8aeec90866e385c086809a99d0b0f7'),
)


def summarize(result, arm):
    all_metrics = init.metrics_from_group(result['groups']['all'], arm)
    texture = {}
    for category in CATEGORIES:
        group = result['groups'][category]
        texture[category] = dict(group['texture'][arm], eligible_pairs=int(group['texture_count']))
    return dict(metrics=all_metrics, texture=texture)


def setup(prior, device):
    args, _ = previous.setup(prior, device)
    original_run, band_run = args.parent_run, args.run
    cov_run = band_run / cov.RUN
    runs = (original_run, band_run, cov_run, cov_run / 'b64_pants_inception_band50_ab')
    training_pairs = {pair['case_id']: pair for pair in args.pairs['train']}
    read_checked = single.fusion.read_checked
    cohort_sha = tail.sha(args.out / 'protocol.json')
    order, reset_after, history = [], [], []
    for run, (job, steps, arm, digest) in zip(runs, HISTORY):
        assert tail.sha(run / 'review.zip') == digest
        manifest = tail.read(run / 'review_manifest.json')
        spec = read_checked(run / 'protocol.json', manifest['protocol.json'])
        trace = read_checked(run / 'training.json', manifest['training.json'])['trace']
        assert spec['cohort_sha256'] == cohort_sha and spec['sampler'] == args.protocol['sampler']
        assert spec.get('base_sha256', spec.get('model_sha256')) == init.BASE_SHA
        rows = trace[:steps]
        assert [row['step'] for row in rows] == list(range(1, steps + 1))
        cases = []
        for row in rows:
            records = row['rows'] if job == 127194 else row[arm]['rows']
            assert len(records) == single.BATCH
            cases.extend(entry['case'] for entry in records)
        if job != 127194:
            assert cases == spec['training_order'][:len(cases)]
        order.extend(training_pairs[case] for case in cases)
        if job != HISTORY[-1][0]:
            reset_after.append(len(order) // single.BATCH)
        history.append(dict(job=job, steps=steps, arm=arm, review_sha256=digest))
    assert len(order) == CHECKPOINTS[-1] * single.BATCH
    assert spec['common_loss']['band_H2'] == previous.BASE_WEIGHT
    assert spec['A_extra']['weight'] == cov.COV_WEIGHT and spec['A_extra']['category'] == 'pants'
    rel = 'validation/step64'
    a64 = read_checked(runs[-1] / rel / 'result.json', manifest[f'{rel}/result.json'])
    a64_rows = tail.read(runs[-1] / rel / 'per_pair.json')
    zero_manifest = tail.read(original_run / 'review_manifest.json')
    zero, zero_rows, _, gt = previous.diagnosis.load_validation(original_run, zero_manifest, 0)
    expected_ids = [(pair['case_id'], pair['category']) for pair in args.pairs['val']]
    for result, rows in ((a64, a64_rows), (zero, zero_rows)):
        assert result['pairs'] == len(rows) == 256
        assert [(row['case_id'], row['category']) for row in rows] == expected_ids
    assert tail.sha(runs[-1] / rel / 'pool3.npz') == a64['pool3_sha256']
    with np.load(runs[-1] / rel / 'pool3.npz', allow_pickle=False) as pool:
        np.testing.assert_array_equal(pool['gt'], gt)
    baseline = summarize(a64, 'A')
    off = summarize(zero, 'H2')
    assert baseline['metrics']['fid'] > off['metrics']['fid']
    tolerance = single.rules.TOLERANCE
    limits = dict(fid=off['metrics']['fid'], lpips=baseline['metrics']['lpips'],
                  ssim=baseline['metrics']['ssim'] * (1 - tolerance),
                  contour_f1=baseline['metrics']['contour_f1'] * (1 - tolerance),
                  band_mae={key: value['band_mae'] for key, value in baseline['texture'].items()})
    args.train_order = order
    original_attempt = original_run / RUN
    args.run = original_attempt / 'scaled_qualification_retry'
    args.run.mkdir(parents=True, exist_ok=True)
    args.spec = args.run / 'protocol.json'
    contract = dict(
        cohort_sha256=cohort_sha, sampler=spec['sampler'], model_sha256=init.BASE_SHA,
        initial='f68 features and residual block; zero output layer',
        f68_sha256=tail.CHECKPOINT_SHA, optimizer='AdamW', optimizer_reset_after=reset_after,
        learning_rate=single.LR, effective_batch=single.BATCH, gpus=1,
        training_history=history, training_order=[pair['case_id'] for pair in order],
        common_loss=spec['common_loss'], alexnet_covariance=spec['A_extra'],
        validation_steps=list(CHECKPOINTS), thresholds=limits, tolerance=tolerance,
        baseline_a64=baseline, baseline_b_off=off, formal_reference=spec['formal_reference'],
        code={filename: tail.sha(SOURCE / filename) for filename in (
            'tools/e2_b_single_direct_960.py', 'scripts/e2_b_single_direct_960.sbatch',
            'tools/e2_b_single_pants_cov_ab.py', 'tools/e2_b_epoch20_single.py')},
    )
    original_contract = tail.read(original_attempt / 'protocol.json')
    assert {k: v for k, v in contract.items() if k != 'code'} == {
        k: v for k, v in original_contract.items() if k != 'code'
    }
    return args, contract


def validation_gate(result, contract):
    entry = summarize(result, ARM)
    metrics, texture = entry['metrics'], entry['texture']
    limits = contract['thresholds']
    checks = dict(fid=metrics['fid'] < limits['fid'], lpips=metrics['lpips'] <= limits['lpips'],
                  ssim=metrics['ssim'] >= limits['ssim'], contour_f1=metrics['contour_f1'] >= limits['contour_f1'])
    for name, summary in texture.items():
        assert summary['eligible_pairs'] == contract['baseline_a64']['texture'][name]['eligible_pairs']
        checks[f'band_mae_{name}'] = summary['band_mae'] <= limits['band_mae'][name]
    return dict(entry, checks=checks, passed=all(checks.values()))


def qualify(args, models, objective):
    path = args.run / 'qualification.json'
    if path.exists():
        assert tail.read(path)['protocol_sha256'] == args.spec_sha
        return
    for index, pair in enumerate(args.pairs['qualification']):
        def probe_objective(pred, gt, mask, _weight):
            return objective(pred, gt, mask, 'B', pair['category'])
        single.qualification(args, models, probe_objective, pair, index)
    pair = next(p for p in args.train_order if p['category'] == 'pants'
                and cov.roi.regions(tail.case_mask(args, p), cov.CORE_RADIUS)[0].sum() >= cov.MIN_PIXELS)
    head, params = single.initialize(models)
    pred, gt, mask = previous.forward(args, models, head, pair)
    common, _ = objective(pred, gt, mask, 'A', pair['category'])
    full, terms = objective(pred, gt, mask, 'B', pair['category'])
    extra = terms['pants_covariance']
    torch.testing.assert_close(full - common, cov.COV_WEIGHT * extra, atol=5e-7, rtol=1e-4)
    # 独立诊断既复现旧探针，也按训练的单样本放缩尺度检查同一张计算图。
    weights = tuple(params.values())
    output_index = tuple(params).index('output.weight')
    term = cov.COV_WEIGHT * extra
    rgb_grad, *legacy = torch.autograd.grad(term, (pred, *weights), retain_graph=True)
    scale = single.SCALE / single.BATCH
    scaled = torch.autograd.grad(scale * term, weights, retain_graph=True)
    total = torch.autograd.grad(single.SCALE * full, weights)

    def norm(gradients, divisor=1):
        norms = [torch.linalg.vector_norm(grad.float() / divisor) for grad in gradients]
        return float(torch.linalg.vector_norm(torch.stack(norms)))

    gradients = dict(
        total_all_trainable=norm(total, single.SCALE),
        total_output_weight=norm((total[output_index],), single.SCALE),
        covariance_rgb=norm((rgb_grad,)),
        covariance_legacy_all_trainable=norm(legacy),
        covariance_legacy_output_weight=norm((legacy[output_index],)),
        covariance_scaled_all_trainable=norm(scaled, scale),
        covariance_scaled_output_weight=norm((scaled[output_index],), scale),
    )
    record = dict(
        protocol_sha256=args.spec_sha, case=pair['case_id'], covariance=float(extra.detach()),
        total_loss=float(full.detach()), h2_loss=float(common.detach()),
        covariance_weight=cov.COV_WEIGHT, covariance_scale=scale, total_scale=single.SCALE,
        vae_dtype=str(next(models[1].vae.parameters()).dtype),
        gradients={key: val if np.isfinite(val) else None for key, val in gradients.items()},
        optimizer_updates=0,
    )
    tail.write(args.run / 'qualification_gradients.json', record)
    for name, value in gradients.items():
        print('DIRECT_GRADIENT', name, value, flush=True)
    required = ('total_all_trainable', 'total_output_weight', 'covariance_rgb',
                'covariance_scaled_all_trainable', 'covariance_scaled_output_weight')
    assert all(np.isfinite(gradients[key]) and gradients[key] > 0 for key in required), gradients
    models[2].adapter.clear()
    single.ab.audit_frozen(args, models, head, params)
    tail.write(path, record)
    print('DIRECT_QUALIFIED', pair['case_id'], flush=True)


def train(args, models, objective, metric_models, contract):
    head, params = single.initialize(models)
    optimizer = torch.optim.AdamW(params.values(), lr=single.LR)
    resume = args.run / 'resume.pt'
    step, trace, checks = 0, [], {}
    if resume.exists():
        state = torch.load(resume, map_location='cpu', weights_only=True)
        assert state['protocol_sha256'] == args.spec_sha
        step, trace, checks = state['step'], state['trace'], state['checks']
        assert step == len(trace)
        single.rules.restore(params, state['head'])
        optimizer.load_state_dict(state['optimizer'])

    def snapshot():
        return {name: value.detach().cpu() for name, value in params.items()}

    def save():
        tail.save_torch(resume, dict(protocol_sha256=args.spec_sha, step=step, trace=trace,
                                    checks=checks, head=snapshot(), optimizer=optimizer.state_dict()))

    def validate():
        dest = args.run / 'validation' / f'step{step}'
        tail.save_torch(dest / f'{ARM}.pt', snapshot())
        result = previous.score(args, models, {ARM: head}, metric_models, 'val', dest)
        checks[str(step)] = validation_gate(result, contract)
        value = checks[str(step)]
        save()
        print('DIRECT_VAL', step, value['metrics']['fid'], value['checks'], flush=True)

    if step in CHECKPOINTS and str(step) not in checks:
        validate()
    for step in range(step + 1, CHECKPOINTS[-1] + 1):
        if step - 1 in contract['optimizer_reset_after']:
            optimizer = torch.optim.AdamW(params.values(), lr=single.LR)
        offset = (step - 1) * single.BATCH
        pairs = args.train_order[offset:offset + single.BATCH]
        trace.append(dict(step=step, **cov.update(args, models, objective, 'B', head, params, optimizer, pairs)))
        save()
        if step % 16 == 0:
            print('DIRECT_TRAIN', step, CHECKPOINTS[-1], flush=True)
        if step in CHECKPOINTS:
            validate()

    eligible = [int(n) for n, result in checks.items() if result['passed']]
    selected = min(eligible, key=lambda n: (checks[str(n)]['metrics']['fid'], n)) if eligible else None
    if selected is not None:
        single.ab.update_head(head, params, args.run / 'validation' / f'step{selected}' / f'{ARM}.pt')
    single.ab.audit_frozen(args, models, head, params)
    tail.write(args.run / 'training.json', dict(updates=step, selected=selected, validation=checks,
                                             trace=trace, frozen_unchanged=True))
    return head.eval().requires_grad_(False), selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prior', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--qualify-only', action='store_true')
    options = parser.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(4 if options.check else 6)
    args, contract = setup(options.prior, 'cpu' if options.check else 'cuda:0')
    if options.check:
        single.archived64(args, args.pairs['test64'])
        print('DIRECT_PRECHECK', len(args.train_order), len({p['case_id'] for p in args.train_order}),
              len(args.pairs['val']), contract['optimizer_reset_after'], contract['thresholds'], flush=True)
        return

    assert torch.cuda.device_count() == 1
    if args.spec.exists():
        assert tail.read(args.spec) == contract
    else:
        tail.write(args.spec, contract)
    args.spec_sha = tail.sha(args.spec)
    if (args.run / 'completion.json').exists() and (args.run / 'review.zip').exists():
        return
    models = init.load(args)
    assert args.original['frozen_model_hashes'] == tail.read(args.parent_run / 'model.json')['frozen_model_hashes']
    objective = cov.Objective(args)
    if options.qualify_only:
        qualify(args, models, objective)
        return
    qualification = tail.read(args.run / 'qualification.json')
    assert qualification['protocol_sha256'] == args.spec_sha
    assert qualification['optimizer_updates'] == 0
    metric_models = tail.metric_models('cpu', args.protocol['metric_assets'])
    head, selected = train(args, models, objective, metric_models, contract)
    files = ['protocol.json', 'model.json', 'execution_environment.json', 'qualification.json',
             'qualification_probe0.json', 'qualification_probe1.json',
             'qualification_gradients.json', 'training.json']
    files.extend(f'validation/step{step}/result.json' for step in CHECKPOINTS)
    stages = {}
    if selected is not None:
        for split, label in (('test64', 'eval64'), ('test500', 'eval500')):
            single.test_controls(args, models, args.pairs[split], split)
            dest = args.run / label
            result = previous.score(args, models, {ARM: head}, metric_models, split, dest)
            decision = single.acceptance(args, dest, result, {ARM: selected}, split)
            tail.write(dest / 'acceptance.json', decision)
            accepted = decision['decisions'][ARM]['accepted']
            stages[label] = dict(pairs=result['pairs'], accepted=accepted)
            files.extend(f'{label}/{name}' for name in ('result.json', 'acceptance.json', 'per_pair.json', 'pool3.npz'))
            print('DIRECT_TEST', label, stages[label], flush=True)
            if not accepted:
                break

    assert tail.tensor_set(models[4]) == args.original['frozen_model_hashes']
    tail.write(args.run / 'completion.json', dict(updates=CHECKPOINTS[-1], selected=selected, evaluation=stages,
                                               frozen_unchanged=True, trajectories=1, vae_decodes=1))
    init.fid.pack(args.run, [*files, 'completion.json'])
    print('DIRECT_DONE', selected, stages, tail.sha(args.run / 'review.zip'), flush=True)


if __name__ == '__main__':
    main()
