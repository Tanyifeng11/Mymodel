"""零5388特征加参考直连完整fresh G资格后，原损失/步长的独立C0/C1一次提议。"""
import argparse
from pathlib import Path
import subprocess
import time
import zipfile

import torch

from models.e2_reference_terminal_hybrid import ReferenceTerminalHybrid, PARAMETERS
from tools import e2_reference_terminal_hybrid_gradients as fresh
from tools.e2_state_b_amplitude_response import read, write, sha, CHECKPOINT_SHA
from tools.e2_state_b_backward_repro import tensor_sha
from tools.e2_oracle_texture import weights_digest


SOURCE = Path(__file__).resolve().parents[1]
EXTRA = ('tools/e2_reference_terminal_hybrid_propose.py',
         'scripts/e2_reference_terminal_hybrid_propose.sbatch')


def inputs(args):
    parent_args = argparse.Namespace(**vars(args)); parent_args.out = args.gradients
    spec, parent = fresh.verify(parent_args)
    manifest = read(args.gradients/'manifest.json')
    assert all(sha(args.gradients/n) == h for n, h in manifest.items())
    result = read(args.gradients/'result.json')
    assert result['passed'] and (result['conditions'], result['fresh_VJP'], result['zero_PNG_controls']) == (48, 408, 48)
    shards = []
    for index in range(4):
        data = torch.load(args.gradients/('gradients/shard%d.pt' % index), map_location='cpu', weights_only=False)
        assert data['shard'] == index and data['protocol_sha256'] == sha(args.gradients/'protocol.json')
        assert data['numerical_qualification'] and data['frozen_unchanged']
        assert data['gradients'].shape == (12, 7, PARAMETERS)
        for replica in data['replicas']:
            item = {n: replica[n] for n in ('case', 'sample_condition')}
            position = next(i for i, v in enumerate(data['items']) if all(v[n] == value for n, value in item.items()))
            assert fresh.replica_report(replica['repeated_gradients'], data['gradients'][position])['passed']
        shards.append(data)
    assert sha(args.checkpoint) == CHECKPOINT_SHA
    return spec, parent, shards


def verify(args):
    protocol = read(args.out/'protocol.json')
    assert all(sha(Path(n)) == h for n, h in protocol['input_sha256'].items())
    assert all(sha(SOURCE/n) == h for n, h in protocol['source_sha256'].items())
    return protocol


def propose(args):
    started = time.monotonic()
    spec, previous, shards = inputs(args)
    assert not args.out.exists(), '不覆盖FP32支路C0/C1提议'
    args.out.mkdir(parents=True)
    files = [args.gradients/n for n in read(args.gradients/'manifest.json')]+[args.gradients/'manifest.json', args.checkpoint]
    protocol = dict(kind='feature4616+reference772 hybrid,new5388 C0/C1 proposals only',
        functional_budget_matched_to_4616=False, point='same f68 plus zero5388 hybrid',
        original_S0_code_modified=False, original_loss_formulas_and_step_gates_unchanged=True,
        parent_fresh_VJP=408, new_reverse_passes=0, reused_G_scope='qualified new branch at unchanged f68 zero only',
        original_training=spec['training'], original_acceptance=spec['acceptance'],
        source_sha256=dict(previous['source_sha256'], **{n: sha(SOURCE/n) for n in EXTRA}),
        input_sha256={str(p.resolve()): sha(p) for p in files},
        execution_commit=subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip(),
        new_S0=0, planned_independent_AdamW_proposals=2, planned_SLSQP_calls=2,
        continuous_training=False, holdout_read=0, adopted_checkpoints=[])
    write(args.out/'protocol.json', protocol)
    source = fresh.direction.delta.post.q8.r16.source
    directions, gradients, margins, component_norms = source.combine_gradients(shards, spec)
    assert gradients.shape == (336, PARAMETERS) and torch.isfinite(gradients).all()
    norms, active, normalized = source.qr.normalization(gradients)
    assert bool(active.all()), '新支路出现零约束，先诊断，不移用旧336求解'
    rows = normalized.double().numpy()
    q, r, qr_report = source.qr.factorize(rows)
    raw_reports, numeric_reports, numeric_tensors, heads = {}, {}, {}, {}
    for arm in ('C0', 'C1'):
        head = ReferenceTerminalHybrid(source.load_head(args,'cpu')); heads[arm] = head
        assert weights_digest(head.head.base)==shards[0]['base_b_hash']
        raw, report = source.raw_once(head.increment_parameters(), directions[arm], spec['training'])
        assert weights_digest(head.head.base)==shards[0]['base_b_hash']
        raw_reports[arm] = dict(raw=raw, **report)
        try:
            diagnostic, tensors = source.qr.solve_arm(rows, q, raw, report['raw_norm'], margins.double().numpy(), qr_report)
            numeric_reports[arm], numeric_tensors[arm] = diagnostic, tensors
        except Exception as caught:
            numeric_reports[arm] = dict(numerical_qualification=False, error_type=type(caught).__name__, message=str(caught))
    torch.save(dict(raws=raw_reports, directions=directions, constraints=gradients, margins=margins), args.out/'raw.pt')
    torch.save(dict(norms=norms, active=active, N32=normalized, Q=torch.from_numpy(q), R=torch.from_numpy(r), arms=numeric_tensors), args.out/'numerics.pt')
    qualified = qr_report['passed'] and all(v['numerical_qualification'] for v in numeric_reports.values())
    write(args.out/'solver_diagnostic.json', dict(qr=qr_report, arms=numeric_reports, all_qualified=qualified,
        constraints_sha256=tensor_sha(gradients), margins_sha256=tensor_sha(margins), limits=source.qr.LIMITS))
    proposals = dict(arms={}, steps={}, component_norms=component_norms, stop_before_replay=not qualified,
        numerical_qualification=qualified, independent_AdamW_steps=2, SLSQP_calls=2, new_reverse_passes=0,
        parent_fresh_VJP=408, parameter_count=PARAMETERS, conditions=48, new_S0=0,
        continuous_training=False, holdout_read=0, adopted_checkpoints=[], source_final_verification=False)
    if qualified:
        candidates = {}
        for arm in ('C0', 'C1'):
            candidates[arm], proposals['steps'][arm] = source.original_step(
                raw_reports[arm]['raw'], numeric_tensors[arm]['delta32'], gradients, spec['training'])
        proposals['stop_before_replay'] = not all(v['parameter_updates'] == 1 and v['minimum_cosine'] >= -1e-5 for v in proposals['steps'].values())
        if not proposals['stop_before_replay']:
            for arm in ('C0', 'C1'):
                head = heads[arm]; base_hash = shards[0]['base_b_hash']
                fresh.legacy.set_parameters(head.increment_parameters(), candidates[arm])
                report = dict(proposals['steps'][arm], arm=arm, base_b_hash=base_hash,
                    base_checkpoint_sha256=CHECKPOINT_SHA, parameter_count=PARAMETERS,
                    projection=dict(method='original QR/SLSQP QP', feasible=True, tolerance=1e-6,
                        actual_slack_min=numeric_reports[arm]['fp32_slack_min']),
                    adopted=False, acceptance='pending full48/336 and original PNG protection',
                    protocol_sha256=sha(args.out/'protocol.json'))
                path = args.out/(arm+'.pt')
                torch.save(dict(state=head.state_dict(),report=report), path)
                proposals['arms'][arm] = dict(report, checkpoint_sha256=sha(path))
                assert tensor_sha(fresh.legacy.flatten(head.increment_parameters())) == report['head_sha256']
    inputs(args); verify(args)
    proposals.update(source_final_verification=True, seconds=time.monotonic()-started)
    write(args.out/'proposals.json', proposals)
    names = ['protocol.json', 'raw.pt', 'numerics.pt', 'solver_diagnostic.json', 'proposals.json']
    names += [arm+'.pt' for arm in proposals['arms']]
    write(args.out/'manifest.json', {n: sha(args.out/n) for n in names})
    with zipfile.ZipFile(args.out/'review.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for n in names+['manifest.json']:
            archive.write(args.out/n, n)
    print('REFERENCE_HYBRID_PROPOSALS_REVIEW', not proposals['stop_before_replay'], sha(args.out/'review.zip'), flush=True)
    if proposals['stop_before_replay']:
        raise SystemExit(2)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--project', type=Path, default=Path('/share/home/u2515283058/Mymodel'))
    args = parser.parse_args(); args.prior = args.prior.resolve(); args.project = args.project.resolve()
    auto = args.prior/'autonomous_repair_20261116'
    for n, value in dict(fixed='scaled_projection_retry', preflight='terminal_head_preflight',
        basis='reference_terminal_preflight_v2', source_run='reference_terminal_one_step',
        global_source='reference_terminal_global_amplitude', r16='reference_terminal_r16_diagnostic',
        q8='reference_terminal_q8_direction_diagnostic', post='reference_terminal_post_latent_probe',
        fp32='reference_terminal_fp32_delta_probe', direction='reference_terminal_fp32_direction',
        gradients='reference_terminal_hybrid_gradients', out='reference_terminal_hybrid_proposals').items():
        setattr(args, n, auto/value)
    args.checkpoint = args.project/'output_eval/e2_state_b_pastel_followup_20261115/trust_retry/training/step1.pt'
    args.data = args.project.parent/'datasets/BF'
    args.device = 'cpu'
    torch.set_num_threads(2)
    return args


if __name__ == '__main__':
    propose(arguments())
