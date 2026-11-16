"""新scaled G的非均匀margin几何界限；只读CPU诊断，不生成模型候选。"""
import argparse
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
import zipfile

import numpy as np
from scipy.optimize import minimize

from tools.e2_b_gradient_geometry import read, write, sha, module_vector


SOURCE = Path(__file__).resolve().parents[1]
TASKS = ('pattern', 'amplitude', 'correlation', 'residue', 'color', 'light', 'edge')
DEFAULT = '/share/home/u2515283058/Mymodel/output_eval/e2_state_b_pair_finetune_20261116/autonomous_repair_20261116/scaled_projection_retry'
EXTRA = ('tools/e2_b_weighted_geometry.py', 'scripts/e2_b_weighted_geometry.sbatch',
         'tools/e2_b_gradient_geometry.py', 'tools/e2_b_scaled_projection_retry.py',
         'tools/e2_b_pair_finetune_losses.py', 'models/e2_state_texture_b.py')


def weighted_geometry(rows, margins, maxiter=2000, gap_tolerance=1e-5):
    """min simplex ||sum w_i N_i/m_i||；真向量重建给U/L，不依赖success标志。"""
    rows, margins = np.asarray(rows, np.float64), np.asarray(margins, np.float64)
    assert rows.ndim == 2 and margins.shape == (len(rows),)
    assert np.isfinite(rows).all() and np.isfinite(margins).all() and (margins > 0).all()
    # 统一缩放仅改善SLSQP条件数；最终界限/KKT全部换回N_i/m_i单位。
    scale = float(margins.min())
    scaled = rows * (scale / margins)[:, None]
    gram = scaled @ scaled.T
    count = len(rows)
    solution = minimize(lambda w: .5 * float(w @ gram @ w), np.full(count, 1. / count),
        jac=lambda w: gram @ w, method='SLSQP', bounds=[(0., 1.)] * count,
        constraints=[dict(type='eq', fun=lambda w: w.sum()-1., jac=lambda w: np.ones(count))],
        options=dict(ftol=1e-14, maxiter=maxiter, disp=False))
    raw = solution.x
    assert np.isfinite(raw).all() and raw.min() >= -1e-10 and abs(raw.sum()-1.) <= 1e-8
    weights = np.maximum(raw, 0.)
    weights /= weights.sum()
    mix = (weights @ scaled) / scale
    upper = float(np.linalg.norm(mix))
    squared = float(mix @ mix)
    dots = (rows @ mix) / margins
    gram_dots = (gram @ weights) / (scale * scale)
    lower = float(dots.min()/upper) if upper > 1e-12 else None
    gap = None if lower is None else upper-lower
    support = weights > 1e-8
    return dict(solver=dict(method='SLSQP', success=bool(solution.success), status=int(solution.status),
        message=str(solution.message), iterations=int(solution.nit), maximum_iterations=maxiter,
        objective_scale=scale, ftol=1e-14),
        weights=weights.tolist(), upper_bound=upper, lower_bound=lower, interval_width=gap,
        interval_resolved=bool(gap is not None and gap <= gap_tolerance), gap_tolerance=gap_tolerance,
        positive_common_direction_found=bool(lower is not None and lower > 0), near_zero_mix=upper <= 1e-12,
        simplex=dict(raw_minimum=float(raw.min()), raw_sum_error=float(abs(raw.sum()-1)),
            repair_l1=float(np.abs(raw-weights).sum()), minimum=float(weights.min()), sum_error=float(abs(weights.sum()-1))),
        kkt=dict(frank_wolfe_gap=squared-float(dots.min()),
            active_stationarity_max_abs=float(np.max(np.abs(dots[support]-squared))),
            minimum_stationarity=float((dots-squared).min()),
            complementarity_max_abs=float(np.max(np.abs(weights*(dots-squared))))),
        reconstruction=dict(gram_dot_max_abs_error=float(np.max(np.abs(dots-gram_dots))),
            gram_squared_norm_error=float(abs(squared-float(weights @ gram_dots))),
            input_row_norm_min=float(np.linalg.norm(rows, axis=1).min()),
            input_row_norm_max=float(np.linalg.norm(rows, axis=1).max())),
        mixture_unit_direction_weighted_dots=(dots/upper).tolist() if upper > 1e-12 else None,
        interpretation='Weighted projections are not cosines. Bounds use cached G only; near-zero U does not prove mathematical incompatibility or finite-step behavior.')


def candidate_margin(rows, margins, start, target):
    """真实写回端点先转FP64再相减；下降方向与参数更新方向符号相反。"""
    rows, margins = np.asarray(rows, np.float64), np.asarray(margins, np.float64)
    descent = np.asarray(start, np.float64)-np.asarray(target, np.float64)
    length = float(np.linalg.norm(descent))
    assert length > 0 and np.isfinite(descent).all()
    dots = rows @ (descent/length)
    weighted = dots/margins
    cosines = dots/np.linalg.norm(rows, axis=1)
    return dict(gamma=float(weighted.min()), minimum_weighted_row=int(np.argmin(weighted)),
        minimum_row_dot=float(dots.min()), ordinary_minimum_cosine=float(cosines.min()),
        actual_descent_norm_fp64=length, row_dots=dots.tolist(), weighted_row_projections=weighted.tolist(),
        definition='d=(theta_f68.double()-theta_candidate.double())/norm; gamma=min_i(N_i dot d/m_i)')


def run(args):
    import torch
    from models.e2_state_texture_b import StateTextureB
    from tools import e2_b_scaled_projection_retry as retry
    from tools.e2_state_b_backward_repro import tensor_sha
    from tools.e2_state_b_amplitude_response import CHECKPOINT_SHA
    from tools.e2_state_b_pair_retry import PROTOCOL_SHA

    started = time.monotonic()
    fixed = args.prior
    assert not args.out.exists(), '不覆盖已有weighted geometry目录'
    checked = SimpleNamespace(out=fixed, checkpoint=args.checkpoint)
    retry.verify(checked)
    inputs = {}
    def record(path, expected=None):
        path = Path(path).resolve()
        assert 'holdout' not in path.parts
        value = sha(path)
        assert expected is None or value == expected, str(path)
        inputs[path.as_posix()] = value
        return value

    overlay = read(fixed/'solver_overlay.json')
    sources = {name: sha(SOURCE/name) for name in sorted(set(overlay['source_sha256']) | set(EXTRA))}
    qualification = read(fixed/'numerical_qualification.json')
    proposals = read(fixed/'training/proposals.json')
    spec = read(fixed/'protocol.json')
    diagnostic = read(fixed/'solver_diagnostic.json')
    for name in ('protocol.json', 'solver_overlay.json', 'solver_diagnostic.json', 'solver_proposal_receipt.json',
                 'scaled_overlay.json', 'numerical_qualification.json', 'training/proposals.json'):
        record(fixed/name)
    assert record(args.checkpoint, CHECKPOINT_SHA) == spec['checkpoint_sha256']
    assert inputs[(fixed/'protocol.json').as_posix()] == PROTOCOL_SHA
    model = StateTextureB(color_channels=2, reference_channels=192)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=False)['b'])
    start, initial = module_vector(model)
    matrices, descriptions, identities = [], [], []
    for index, pair in enumerate(spec['train']):
        path = fixed/'gradients'/('shard%d.pt' % index)
        record(path, qualification['new_gradient_sha256'][path.name])
        record(path.with_suffix('.json'))
        shard = torch.load(path, map_location='cpu', weights_only=False)
        assert shard['loss_scale'] == 1024 and shard['all_finite'] and shard['numerical_qualification']
        assert shard['new_gradient_generation'] and not shard['cached_gradients_reused']
        assert shard['protocol_sha256'] == PROTOCOL_SHA and shard['checkpoint_sha256'] == CHECKPOINT_SHA
        assert shard['b_hash'] == initial and shard['frozen_unchanged'] and shard['training_updates'] == 0
        assert shard['overlay_sha256'] == record(fixed/'scaled_overlay.json')
        assert shard['case'] == pair['case_id']
        assert shard['items'] == [r for r in spec['inputs'] if r['case'] == pair['case_id']]
        assert shard['gradients'].shape == (12, 7, start.numel())
        identities.append(shard['net_hash'])
        matrices.append(shard['gradients'].reshape(84, -1))
        descriptions.extend(dict(case=item['case'], condition=item['condition'], arm=item['arm'], task=task)
                            for item in shard['items'] for task in TASKS)
        del shard
    assert len(matrices) == 4 and len(set(identities)) == 1
    gradients = torch.cat(matrices)
    del matrices
    assert gradients.dtype == torch.float32 and torch.isfinite(gradients).all()
    assert tensor_sha(gradients) == diagnostic['constraints_sha256']
    cfg = spec['training']
    assert cfg['default_margin'] == .01 and cfg['protected_amplitude_margin'] == .18
    margins = torch.full((336,), cfg['default_margin'], dtype=gradients.dtype)
    for index, row in enumerate(descriptions):
        if [row['case'], row['condition']] == cfg['protected_condition'] and row['task'] == 'amplitude':
            margins[index] = cfg['protected_amplitude_margin']
    assert int((margins > .01).sum()) == 3 and tensor_sha(margins) == diagnostic['margins_sha256']
    norms32 = gradients.norm(dim=1)
    assert (norms32 > 1e-12).all()
    # 与project_displacement逐式相同：norm和除法都在原FP32，随后才转double。
    rows32 = (gradients / norms32[:, None]).double()
    rows64 = gradients.double()
    rows64.div_(rows64.norm(dim=1)[:, None])
    normalization = dict(primary='FP32 gradient norm and division, then FP64; same project_displacement definition',
        sensitivity='FP64 gradient norm and division', primary_rows_sha256=tensor_sha(rows32),
        sensitivity_rows_sha256=tensor_sha(rows64),
        row_max_abs_difference=float((rows32-rows64).abs().max()),
        primary_row_norm_max_error=float((rows32.norm(dim=1)-1).abs().max()))
    del gradients
    candidates = {}
    for arm in ('C0', 'C1'):
        report = proposals['arms'][arm]
        record(fixed/'training'/(arm+'.pt'), report['checkpoint_sha256'])
        model.load_state_dict(torch.load(fixed/'training'/(arm+'.pt'), map_location='cpu', weights_only=False)['b'])
        target, digest = module_vector(model)
        assert report['initial_b_hash'] == initial and digest == report['final_b_hash']
        candidates[arm] = dict(final_b_hash=digest, checkpoint_sha256=report['checkpoint_sha256'],
            actual_descent_sha256_fp64=tensor_sha(start.double()-target.double()),
            source_actual_norm_fp32=report['actual_norm'], legacy_subtraction_norm_fp32=float((start-target).norm()),
            primary=candidate_margin(rows32.numpy(), margins.double().numpy(), start.numpy(), target.numpy()),
            fp64_normalization=candidate_margin(rows64.numpy(), margins.double().numpy(), start.numpy(), target.numpy()))
    geometries = {}
    for name, rows in (('primary', rows32), ('fp64_normalization', rows64)):
        result = weighted_geometry(rows.numpy(), margins.double().numpy())
        for arm, candidate in candidates.items():
            current = candidate[name]
            current['minimum_weighted_identity'] = descriptions[current['minimum_weighted_row']]
            assert current['gamma'] <= result['upper_bound']+1e-9
            current['reachable_lower_over_gamma'] = None if result['lower_bound'] is None else result['lower_bound']/current['gamma']
            current['upper_over_gamma'] = result['upper_bound']/current['gamma']
        result['rows'] = [dict(row, margin=float(margins[i]), simplex_weight=result['weights'][i],
            weighted_projection=None if result['mixture_unit_direction_weighted_dots'] is None else result['mixture_unit_direction_weighted_dots'][i])
            for i, row in enumerate(descriptions)]
        geometries[name] = result
        print('WEIGHTED_GEOMETRY', name, result['lower_bound'], result['upper_bound'], flush=True)
    retry.verify(checked)
    assert all(sha(Path(path)) == value for path, value in inputs.items())
    assert all(sha(SOURCE/path) == value for path, value in sources.items())
    result = dict(analysis='new scaled-G unequal-margin common-direction bounds; no candidate generation',
        fixed_run=str(fixed), input_sha256=inputs, source_sha256=sources, protocol_sha256=PROTOCOL_SHA,
        execution_commit=subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip(),
        source_gradient_job='125421', source_gradient_scale=1024, gradients_already_unscaled=True,
        constraints_sha256=diagnostic['constraints_sha256'], margins_sha256=tensor_sha(margins),
        initial_b_hash=initial, frozen_net_hash=identities[0], constraints=336, protected_margin_rows=3,
        normalization=normalization, geometry=geometries, actual_candidates=candidates,
        definitions=dict(weighted_rows='A_i=N_i/m_i', target='rho=max_{||d||=1} min_i A_i dot d',
            upper='U=||sum_i w_i A_i|| for simplex-feasible w',
            lower='L=min_i A_i dot normalized(sum_i w_i A_i)',
            original_QP='min ||delta-raw||^2 subject to N delta >= m*||raw||',
            zero_center_relation='min ||delta||^2 with same RHS corresponds to weighted max-min when common strict descent exists; does not preserve raw loss preference'),
        new_forward_passes=0, new_backward_passes=0, optimizer_updates=0, new_candidates=0,
        projection_calls=0, adopted_checkpoints=[], holdout_results_read=0,
        limitations=['Cached gradient geometry is not finite-step S0 acceptance or capacity diagnosis.',
            'FP32-normalized row dots are not exact cosines; all margin-weighted projections are not cosines.',
            'No mixed direction is applied to parameters. C0/C1 raw preference is not tested by the optimum bound.'],
        seconds=time.monotonic()-started)
    args.out.mkdir(parents=True)
    write(args.out/'result.json', result)
    write(args.out/'manifest.json', {'result.json': sha(args.out/'result.json')})
    with zipfile.ZipFile(args.out/'review.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in ('result.json', 'manifest.json'):
            archive.write(args.out/name, name)
    print('WEIGHTED_GEOMETRY_COMPLETE', sha(args.out/'review.zip'), flush=True)


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', type=Path, default=Path(DEFAULT), help='本轮scaled_projection_retry目录')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--project', type=Path, default=Path('/share/home/u2515283058/Mymodel'))
    args = parser.parse_args()
    args.prior = args.prior.resolve()
    args.out = (args.out or args.prior.parent/'scaled_weighted_geometry').resolve()
    args.checkpoint = args.project/'output_eval/e2_state_b_pastel_followup_20261115/trust_retry/training/step1.pt'
    torch.set_num_threads(2)
    run(args)


if __name__ == '__main__':
    main()
