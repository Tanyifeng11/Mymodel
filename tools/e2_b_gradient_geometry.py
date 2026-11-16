"""缓存梯度凸包的只读几何诊断；不反传、不生成或采用新权重。"""
import argparse
import hashlib
import json
import os
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


CHECKPOINT_SHA = 'f68f22a106c3f0a08044c754b44c738ba3d6f67274edf5d281d5b54f39ddcd96'
PROTOCOL_SHA = 'acdbf13ae5d7f0c893c627957fc8e50d4431004266a7bc845514d039f6146fc9'
TASKS = ('pattern', 'amplitude', 'correlation', 'residue', 'color', 'light', 'edge')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')


def geometry(rows, gap_tolerance=1e-5, maxiter=2000):
    """任意可行w都给上界；直接向量重建给可达下界，不依赖求解器成功标志。"""
    rows = np.asarray(rows, dtype=np.float64)
    assert rows.ndim == 2 and np.isfinite(rows).all()
    assert np.max(np.abs(np.linalg.norm(rows, axis=1) - 1)) < 1e-10
    gram = rows @ rows.T
    count = len(rows)
    objective = lambda w: .5 * float(w @ gram @ w)
    solution = minimize(objective, np.full(count, 1. / count), jac=lambda w: gram @ w,
        method='SLSQP', bounds=[(0., 1.)] * count,
        constraints=[dict(type='eq', fun=lambda w: w.sum() - 1., jac=lambda w: np.ones(count))],
        options=dict(ftol=1e-14, maxiter=maxiter, disp=False))
    raw = solution.x
    assert np.isfinite(raw).all() and raw.min() >= -1e-10 and abs(raw.sum() - 1) <= 1e-8
    # 只修复浮点simplex尾差，然后重新核验全部界限；不修改训练梯度。
    weights = np.maximum(raw, 0.)
    weights /= weights.sum()
    mix = weights @ rows
    norm = float(np.linalg.norm(mix))
    dots = rows @ mix
    gram_dots = gram @ weights
    squared = float(mix @ mix)
    lower = float(dots.min() / norm) if norm > 1e-12 else None
    upper = norm
    gap = upper - lower if lower is not None else None
    support = weights > 1e-8
    return dict(solver=dict(method='SLSQP', success=bool(solution.success), status=int(solution.status),
        message=str(solution.message), iterations=int(solution.nit), maximum_iterations=maxiter, ftol=1e-14),
        weights=weights.tolist(), upper_bound=upper, lower_bound=lower, interval_width=gap,
        gap_tolerance=gap_tolerance, interval_resolved=bool(gap is not None and gap <= gap_tolerance),
        positive_common_direction_found=bool(lower is not None and lower > 0),
        near_zero_mix=norm <= 1e-12,
        simplex=dict(raw_minimum=float(raw.min()), raw_sum_error=float(abs(raw.sum()-1)),
            repair_l1=float(np.abs(raw-weights).sum()), minimum=float(weights.min()), sum_error=float(abs(weights.sum()-1))),
        kkt=dict(frank_wolfe_gap=squared-float(dots.min()),
            active_stationarity_max_abs=float(np.max(np.abs(dots[support]-squared))),
            minimum_stationarity=float((dots-squared).min()),
            complementarity_max_abs=float(np.max(np.abs(weights*(dots-squared))))),
        reconstruction=dict(gram_dot_max_abs_error=float(np.max(np.abs(dots-gram_dots))),
            gram_squared_norm_error=float(abs(squared-float(weights @ gram_dots))),
            row_norm_max_error=float(np.max(np.abs(np.diag(gram)-1)))),
        mixed_direction_cosines=(dots / norm).tolist() if norm > 1e-12 else None,
        interpretation='Bounds describe cached unit-gradient geometry only. Near-zero upper bound does not certify exact infeasibility; sphere max-min need not be zero when the origin is in the convex hull.')


def module_vector(module):
    import torch
    digest = hashlib.sha256()
    for name, value in module.named_parameters():
        digest.update(name.encode())
        digest.update(str((list(value.shape), str(value.dtype))).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return torch.cat([p.detach().flatten() for p in module.parameters()]), digest.hexdigest()


def run(args):
    import torch
    from models.e2_state_texture_b import StateTextureB
    started = time.monotonic()
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '2')))
    prior = args.prior.resolve()
    fixed = prior / 'projection_fp64_retry'
    out = (args.out or prior / 'autonomous_repair_20261116/gradient_geometry').resolve()
    assert not out.exists(), '不覆盖已有诊断目录'
    inputs = {}

    def verified(path, expected=None):
        digest = sha(path)
        assert expected is None or digest == expected, str(path)
        inputs[str(path)] = digest
        return digest

    verified(prior / 'protocol.json', PROTOCOL_SHA)
    verified(fixed / 'protocol.json', PROTOCOL_SHA)
    spec = read(prior / 'protocol.json')
    verified(args.checkpoint, CHECKPOINT_SHA)
    manifest = read(fixed / 'manifest.json')
    verified(fixed / 'manifest.json')
    verified(fixed / 'run_provenance.json', manifest['run_provenance.json'])
    verified(fixed / 'training/proposals.json', manifest['training/proposals.json'])
    provenance = read(fixed / 'run_provenance.json')
    proposals = read(fixed / 'training/proposals.json')['arms']
    model = StateTextureB(color_channels=2, reference_channels=192)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=False)['b'])
    start, initial_hash = module_vector(model)
    matrices, descriptions, identities = [], [], []
    for i in range(4):
        for suffix in ('pt', 'json'):
            name = 'gradients/shard%d.%s' % (i, suffix)
            verified(prior / name, provenance['copied_sha256'][name])
        shard = torch.load(prior / ('gradients/shard%d.pt' % i), map_location='cpu', weights_only=False)
        assert shard['protocol_sha256'] == PROTOCOL_SHA and shard['checkpoint_sha256'] == CHECKPOINT_SHA
        assert shard['b_hash'] == initial_hash and shard['frozen_unchanged'] and shard['training_updates'] == 0
        assert shard['case'] == spec['train'][i]['case_id'] and len(shard['items']) == 12
        assert shard['items'] == [r for r in spec['inputs'] if r['case'] == shard['case']]
        assert shard['gradients'].shape == (12, 7, start.numel())
        identities.append(shard['net_hash'])
        matrices.append(shard['gradients'].reshape(84, -1))
        descriptions.extend(dict(case=item['case'], condition=item['condition'], arm=item['arm'], task=task)
                            for item in shard['items'] for task in TASKS)
        del shard
    assert len(set(identities)) == 1
    gradients = torch.cat(matrices)
    del matrices
    assert gradients.dtype == torch.float32 and bool(torch.isfinite(gradients).all())
    norms32 = gradients.norm(dim=1)
    assert bool((norms32 > 1e-12).all()), '发现零/近零约束，不能无声丢弃'
    rows = gradients.double()
    norms64 = rows.norm(dim=1)
    rows.div_(norms64[:, None])
    differences = []
    old_norm_errors = []
    for i in range(0, len(rows), 16):
        old = (gradients[i:i+16] / norms32[i:i+16, None]).double()
        differences.append(float((old-rows[i:i+16]).abs().max()))
        old_norm_errors.append(float((old.norm(dim=1)-1).abs().max()))
    del gradients
    normalization = dict(source_dtype='torch.float32', geometry_dtype='torch.float64',
        method='cast raw gradients to float64 before norm and division',
        old_fp32_normalized_rows_max_abs_difference=max(differences),
        old_fp32_row_norm_max_error=max(old_norm_errors),
        raw_gradient_norm_min=float(norms64.min()), raw_gradient_norm_max=float(norms64.max()))
    result = geometry(rows.numpy())
    candidates = {}
    for arm in ('C0', 'C1'):
        report = proposals[arm]
        verified(fixed / ('training/%s.pt' % arm), report['checkpoint_sha256'])
        assert report['checkpoint_start'] == CHECKPOINT_SHA and report['initial_b_hash'] == initial_hash
        assert report['parameter_updates'] == 1 and report['projection']['feasible']
        model.load_state_dict(torch.load(fixed / ('training/%s.pt' % arm), map_location='cpu', weights_only=False)['b'])
        target, digest = module_vector(model)
        assert digest == report['final_b_hash'] and float((start-target).norm()) == report['actual_norm']
        # 差值也在double中求，描述已写回参数的真实位移。
        delta = start.double() - target.double()
        assert bool(torch.isfinite(delta).all()) and float(delta.norm()) > 0
        cosines = (rows @ (delta / delta.norm())).numpy()
        index = int(np.argmin(cosines))
        candidates[arm] = dict(minimum_cosine=float(cosines[index]), minimum_row=descriptions[index],
            cosines=cosines.tolist(), actual_norm_fp64=float(delta.norm()),
            original_report_minimum_cosine=report['minimum_cosine'], final_b_hash=digest,
            upper_bound_consistent=bool(float(cosines.min()) <= result['upper_bound']+1e-10))
        assert candidates[arm]['upper_bound_consistent']
    weights = result.pop('weights')
    result['constraint_rows'] = [dict(item, weight=weights[i],
        mixed_cosine=None if result['mixed_direction_cosines'] is None else result['mixed_direction_cosines'][i])
        for i, item in enumerate(descriptions)]
    result['top_weight_rows'] = sorted(result['constraint_rows'], key=lambda row: -row['weight'])[:20]
    source = Path(__file__).resolve().parents[1]
    names = ('tools/e2_b_gradient_geometry.py', 'tests/test_e2_b_gradient_geometry.py',
             'scripts/e2_b_gradient_geometry.sbatch', 'models/e2_state_texture_b.py')
    result.update(input_sha256=inputs, source_sha256={name: sha(source/name) for name in names},
        execution_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(source), text=True).strip(),
        initial_b_hash=initial_hash, net_hash=identities[0], constraints=336, normalization=normalization,
        actual_candidates=candidates, new_backward_passes=0, new_training_updates=0, new_candidates=0,
        cached_gradients_are_ground_truth=False, holdout_used=False, seconds=time.monotonic()-started,
        projection_comparison='Existing projection minimizes distance to raw with unequal margins; wider max-min geometry does not establish a solver defect. Positive scalar step scaling preserves cosine before parameter rounding.',
        best_available_lower_bound=max([v['minimum_cosine'] for v in candidates.values()] +
                                      ([] if result['lower_bound'] is None else [result['lower_bound']])))
    assert all(sha(Path(path)) == digest for path, digest in inputs.items())
    out.mkdir(parents=True)
    write(out/'result.json', result)
    write(out/'manifest.json', {'result.json': sha(out/'result.json')})
    with zipfile.ZipFile(out/'review.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in ('result.json', 'manifest.json'):
            archive.write(out/name, name)
    print('GRADIENT_GEOMETRY_COMPLETE', result['lower_bound'], result['upper_bound'], sha(out/'review.zip'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=Path('/share/home/u2515283058/Mymodel/output_eval/e2_state_b_pastel_followup_20261115/trust_retry/training/step1.pt'))
    run(parser.parse_args())


if __name__ == '__main__':
    main()
