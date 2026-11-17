"""同一原始空间欧氏投影的QR/SLSQP诊断；不生成head候选或运行采样。"""
import argparse
import os
from pathlib import Path
import platform
import socket
import subprocess
import time
import zipfile

import numpy as np
import scipy
from scipy.linalg import qr
from scipy.optimize import LinearConstraint, minimize, nnls
import torch

from tools import e2_terminal_head_projection_retry as prior_tool
from tools.e2_state_b_amplitude_response import read, write, sha
from tools.e2_state_b_backward_repro import tensor_sha


SOURCE = Path(__file__).resolve().parents[1]
ARMS = ('C0', 'C1')
EXTRA = ('tools/e2_terminal_head_qr_diagnostic.py', 'scripts/e2_terminal_head_qr_diagnostic.sbatch')
LIMITS = dict(qr_residual=1e-10, continuous_slack=1e-8, original_fp32_slack=1e-6,
    stationarity_relative=1e-6, relative_gap=1e-6, relative_complementarity=1e-6, active_slack=1e-7)
COUNTS = dict(new_s0_images=0, new_reverse_passes=0, adamw_steps=0, head_updates=0,
    holdout_results_read=0, adopted_checkpoints=[])


def normalization(gradients):
    assert gradients.dtype == torch.float32 and torch.isfinite(gradients).all()
    norms = gradients.norm(dim=1)  # 保留原solver的FP32归约/归一化，之后才转double。
    active = norms > 1e-12
    rows32 = gradients[active]/norms[active, None]
    assert active.any() and torch.isfinite(rows32).all()
    return norms, active, rows32


def factorize(rows):
    q, r = qr(rows.T, mode='economic', pivoting=False, check_finite=True)
    metrics = dict(orthogonality_max_abs=float(np.max(np.abs(q.T@q-np.eye(q.shape[1])))),
        reconstruction_max_abs=float(np.max(np.abs(rows.T-q@r))),
        row_space_max_abs=float(np.max(np.abs(rows-(rows@q)@q.T))),
        reduced_matrix_max_abs=float(np.max(np.abs(rows@q-r.T))),
        shape_Q=list(q.shape), shape_R=list(r.shape), pivoting=False, rank_truncation=False)
    metrics['passed'] = max(metrics['orthogonality_max_abs'], metrics['reconstruction_max_abs'],
        metrics['row_space_max_abs']) <= LIMITS['qr_residual']
    return q, r, metrics


def original_step_preview(raw, raw_norm, delta32, gradients, cfg, feasible):
    """仅张量运算预览原缩步；不构造模型，不写参数。投影失败仍按原规则返回零。"""
    delta = delta32 if feasible else torch.zeros_like(raw)
    norm_scale = min(1., float(torch.tensor(raw_norm, dtype=raw.dtype)/delta.norm().clamp_min(1e-18)))
    delta = delta*norm_scale*cfg['scale']
    cap_scale = min(1., cfg['displacement_cap']/float(delta.norm().clamp_min(1e-18)))
    delta = delta*cap_scale
    zero = torch.zeros_like(raw)
    actual = zero-(zero-delta)
    denominator = gradients.norm(dim=1)*actual.norm()
    minimum = float(((gradients@actual)/denominator.clamp_min(1e-18)).min()) if torch.count_nonzero(actual) else None
    return dict(norm_scale=norm_scale, fixed_scale=cfg['scale'], cap_scale=cap_scale,
        actual_descent_sha256=tensor_sha(actual), actual_norm=float(actual.norm()), minimum_cosine=minimum,
        projection_would_return_zero=not feasible, model_parameters_written=False)


def solve_arm(rows, q, raw, length, margins, qr_report):
    """目标只降维不换形；KKT独立在原参数空间核验，不依赖SLSQP乘子。"""
    raw64 = raw.double().numpy()
    assert raw.dtype == torch.float32 and length > 0 and np.isfinite(length)
    a = rows@q
    b = margins-rows@(raw64/length)
    started = time.monotonic()
    solution = minimize(lambda v: .5*float(v@v), np.zeros(q.shape[1]), jac=lambda v: v,
        method='SLSQP', constraints=[LinearConstraint(a, b, np.inf)],
        options=dict(ftol=1e-12, maxiter=500, disp=False))
    v = solution.x
    assert np.isfinite(v).all()
    delta64 = raw64+length*(q@v)
    delta32 = torch.from_numpy(delta64).to(raw.dtype)
    continuous = rows@delta64/length-margins
    written = rows@delta32.double().numpy()/length-margins
    correction = (delta64-raw64)/length
    slack = rows@correction-b
    active = continuous <= LIMITS['active_slack']
    multiplier = np.zeros(rows.shape[0])
    nnls_error = None
    try:
        if active.any():
            # scipy1.10.1不传atol；活跃集合只用于乘子诊断，不修改原约束。
            multiplier[active], _ = nnls(rows[active].T, correction)
    except Exception as error:
        nnls_error = dict(type=type(error).__name__, message=str(error))
        multiplier = None
    p = .5*float(correction@correction)
    kkt = dict(primal=p, solver_objective=float(solution.fun), primal_original_units=p*length**2,
        objective_equivalence_abs=abs(p-float(solution.fun)),
        active_count=int(active.sum()), active_rule='continuous_slack <= 1e-7', nnls_error=nnls_error)
    if multiplier is not None:
        normal = rows.T@multiplier
        d = float(b@multiplier-.5*(normal@normal))
        stationarity = float(np.linalg.norm(correction-normal)/max(np.linalg.norm(correction), 1e-12))
        gap = p-d
        comp = float(np.max(np.abs(multiplier*slack)))
        kkt.update(dual=d, dual_original_units=d*length**2, gap=gap, gap_original_units=gap*length**2,
            stationarity_relative=stationarity, stationarity_denominator='max(norm(correction),1e-12)',
            relative_gap=abs(gap)/max(1., abs(p), abs(d)),
            relative_complementarity=comp/max(1., p), complementarity_max_abs=comp,
            multiplier_min=float(multiplier.min()), dual_feasible=bool((multiplier >= 0).all()))
    checks = dict(qr=qr_report['passed'], slsqp_success=bool(solution.success),
        finite=bool(np.isfinite(delta64).all() and torch.isfinite(delta32).all()),
        continuous_feasible=float(continuous.min()) >= -LIMITS['continuous_slack'],
        original_fp32_feasible=float(written.min()) >= -LIMITS['original_fp32_slack'],
        nnls_success=nnls_error is None,
        dual_feasible=kkt.get('dual_feasible', False),
        stationarity=kkt.get('stationarity_relative', np.inf) <= LIMITS['stationarity_relative'],
        gap=kkt.get('relative_gap', np.inf) <= LIMITS['relative_gap'],
        complementarity=kkt.get('relative_complementarity', np.inf) <= LIMITS['relative_complementarity'])
    report = dict(numerical_qualification=all(checks.values()), checks=checks,
        failed_checks=[key for key, value in checks.items() if not value],
        slsqp=dict(success=bool(solution.success), status=int(solution.status), message=str(solution.message),
            iterations=int(solution.nit), evaluations=int(solution.nfev), ftol=1e-12, maxiter=500, analytic_jacobian=True),
        length=length, raw_sha256=tensor_sha(raw), continuous_slack_min=float(continuous.min()),
        fp32_slack_min=float(written.min()), original_fp32_violations=int((written < -1e-6).sum()),
        reduced_slack_min=float((a@v-b).min()), reconstruction_slack_difference=float(np.max(np.abs((a@v-b)-continuous))),
        cast_max_abs=float(np.max(np.abs(delta32.double().numpy()-delta64))), kkt=kkt,
        seconds=time.monotonic()-started,
        interpretation='numerical KKT evidence only; tolerance-feasible does not imply an exact mathematical certificate or pixel quality')
    tensors = dict(v=torch.from_numpy(v), lambda_=None if multiplier is None else torch.from_numpy(multiplier),
        delta64=torch.from_numpy(delta64), delta32=delta32, active_nnls=torch.from_numpy(active),
        b=torch.from_numpy(b), continuous_slack=torch.from_numpy(continuous), fp32_slack=torch.from_numpy(written))
    return report, tensors


def run(args):
    assert not args.out.exists(), '拒绝覆盖已有诊断'
    old_args = prior_tool.make_args(args.source_run, args.prior, args.project)
    previous = prior_tool.verify_inputs(old_args)
    raw_path = args.source_run/'raw.pt'
    source_diagnostic = read(args.source_run/'solver_diagnostic.json')
    assert sha(raw_path) == source_diagnostic['raw_pt_sha256'] and all(source_diagnostic['raw_sha_match'].values())
    raw_data = torch.load(raw_path, map_location='cpu', weights_only=False)
    historical = read(args.source_run/'source_failure/proposals.json')
    assert raw_data['parameter_count'] == 2308 and raw_data['source_gradient_sha256'] == previous['source_gradient_sha256']
    gradients = torch.cat([torch.load(args.source_run/('gradients/shard%d.pt' % i), map_location='cpu',
        weights_only=False)['gradients'] for i in range(4)]).flatten(0, 1)
    assert torch.equal(gradients, raw_data['constraints']) and gradients.shape == (336, 2308)
    assert tensor_sha(gradients) == source_diagnostic['constraints_sha256']
    margins32 = raw_data['margins']
    assert margins32.dtype == torch.float32 and tensor_sha(margins32) == source_diagnostic['margins_sha256']
    for arm in ARMS:
        raw = raw_data['raws'][arm]
        assert tensor_sha(raw['raw']) == raw['raw_sha256'] == historical['arms'][arm]['raw_sha256']
        assert raw['raw_norm'] == historical['arms'][arm]['raw_norm']
    args.out.mkdir(parents=True)
    sources = dict(previous['source_sha256'])
    sources.update({name: sha(SOURCE/name) for name in EXTRA})
    input_names = ('raw.pt', 'solver_overlay.json', 'solver_diagnostic.json', 'source_failure/proposals.json',
        'protocol.json', 'one_step_overlay.json', 'gradient_qualification.json')
    inputs = {str((args.source_run/name).resolve()): sha(args.source_run/name) for name in input_names}
    for i in range(4):
        for ext in ('pt', 'json'):
            path = args.source_run/('gradients/shard%d.%s' % (i, ext))
            inputs[str(path.resolve())] = sha(path)
    protocol = dict(source_run=str(args.source_run), method='unpivoted economic QR; original-space Euclidean projection',
        loss_constraints_step_and_original_gates_unchanged=True, input_sha256=inputs, source_sha256=sources,
        limits=LIMITS, slsqp=dict(ftol=1e-12, maxiter=500, jacobian='analytic', initial_v='zero'),
        nnls=dict(active_slack=1e-7, atol_not_supplied=True), **COUNTS,
        execution_commit=subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip(),
        environment=dict(torch=str(torch.__version__), scipy=scipy.__version__, numpy=np.__version__,
            python=platform.python_version(), hostname=socket.gethostname(), threads=torch.get_num_threads(),
            job=os.environ.get('SLURM_JOB_ID'), node=os.environ.get('SLURM_JOB_NODELIST')))
    write(args.out/'diagnostic_protocol.json', protocol)
    norms, active, rows32 = normalization(gradients)
    rows = rows32.double().numpy()
    margins = margins32[active].double().numpy()
    torch.save(dict(norms=norms, active=active, N32=rows32, margins32=margins32,
        source_constraints_sha256=tensor_sha(gradients)), args.out/'normalization.pt')
    result = dict(protocol=protocol, normalization=dict(norms_sha256=tensor_sha(norms), active_sha256=tensor_sha(active),
        N32_sha256=tensor_sha(rows32), N64_sha256=tensor_sha(rows32.double()), active_rows=int(active.sum())), arms={}, **COUNTS)
    tensors = {}
    try:
        q, r, qr_report = factorize(rows)
        result['qr'] = qr_report
        tensors.update(Q=torch.from_numpy(q), R=torch.from_numpy(r), N64=torch.from_numpy(rows))
        cfg = read(args.source_run/'protocol.json')['training']
        for arm in ARMS:
            try:
                raw = raw_data['raws'][arm]
                report, arm_tensors = solve_arm(rows, q, raw['raw'], raw['raw_norm'], margins, qr_report)
                report['original_step_preview'] = original_step_preview(raw['raw'], raw['raw_norm'], arm_tensors['delta32'], gradients,
                    cfg, report['checks']['original_fp32_feasible'])
                result['arms'][arm], tensors[arm] = report, arm_tensors
            except Exception as error:
                result['arms'][arm] = dict(numerical_qualification=False, error_type=type(error).__name__, message=str(error))
    except Exception as error:
        result['factorization_failure'] = dict(error_type=type(error).__name__, message=str(error))
    result['numerical_qualification'] = len(result['arms']) == 2 and all(r['numerical_qualification'] for r in result['arms'].values())
    result['quality_assessed'] = False
    result['candidate_generated'] = False
    torch.save(tensors, args.out/'diagnostic.pt')
    result['tensor_sha256'] = {name: sha(args.out/name) for name in ('normalization.pt', 'diagnostic.pt')}
    prior_tool.verify_inputs(old_args)
    assert all(sha(Path(path)) == digest for path, digest in inputs.items())
    assert all(sha(SOURCE/path) == digest for path, digest in sources.items())
    write(args.out/'result.json', result)
    names = ('diagnostic_protocol.json', 'normalization.pt', 'diagnostic.pt', 'result.json')
    write(args.out/'manifest.json', {name: sha(args.out/name) for name in names})
    with zipfile.ZipFile(args.out/'review.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in names+('manifest.json',):
            archive.write(args.out/name, name)
    print('TERMINAL_QR_DIAGNOSTIC', result['numerical_qualification'], sha(args.out/'review.zip'), flush=True)
    if not result['numerical_qualification']:
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--project', type=Path, default=Path('/share/home/u2515283058/Mymodel'))
    args = parser.parse_args()
    args.prior = args.prior.resolve()
    args.source_run = args.prior/'autonomous_repair_20261116/terminal_head_projection_retry'
    args.out = (args.out or args.prior/'autonomous_repair_20261116/terminal_head_qr_diagnostic').resolve()
    torch.set_num_threads(2)
    run(args)


if __name__ == '__main__':
    main()
