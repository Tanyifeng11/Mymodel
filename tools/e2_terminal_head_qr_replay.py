"""仅应用已通过数值资格的QR投影，再做原S0完整replay；不重新求解或优化。"""
import argparse
import copy
from pathlib import Path
import shutil
import zipfile

import cv2
import torch

from tools import e2_terminal_head_one_step as old
from tools import e2_terminal_head_projection_retry as previous
from tools import e2_state_b_pair_summary as summary
from tools.e2_state_b_amplitude_response import read, write, sha, CHECKPOINT_SHA
from tools.e2_state_b_backward_repro import tensor_sha, vector_summary
from tools.e2_b_endpoint_losses import flatten, set_parameters
from tools.e2_oracle_texture import weights_digest
from tools.e2_b_scaled_replay_review import Inputs, validate_report, replay_eligibility, same_numbers
from tools.e2_state_b_pair_evaluation import strict_replay_protection
from tools.e2_terminal_head_step_summary import validate_candidate


SOURCE = Path(__file__).resolve().parents[1]
ARMS = ('C0', 'C1')
COUNTS = dict(new_reverse_passes=0, source_reverse_passes=352, slsqp_calls=0, adamw_steps=0,
    saved_candidate_parameter_updates=2, holdout_results_read=0, adopted_checkpoints=[])


def qualified_inputs(args):
    qr_run = args.prior/'autonomous_repair_20261116/terminal_head_qr_diagnostic'
    raw_run = args.prior/'autonomous_repair_20261116/terminal_head_projection_retry'
    manifest = read(qr_run/'manifest.json')
    assert all(sha(qr_run/name) == digest for name, digest in manifest.items())
    result, protocol = read(qr_run/'result.json'), read(qr_run/'diagnostic_protocol.json')
    assert result['protocol'] == protocol and result['numerical_qualification'] and result['qr']['passed']
    assert not result['candidate_generated'] and not result['quality_assessed']
    assert all(result[key] == 0 for key in ('new_reverse_passes', 'adamw_steps', 'head_updates', 'new_s0_images'))
    assert all(sha(Path(path)) == digest for path, digest in protocol['input_sha256'].items())
    assert all(sha(SOURCE/path) == digest for path, digest in protocol['source_sha256'].items())
    payload = torch.load(qr_run/'diagnostic.pt', map_location='cpu', weights_only=False)
    norm = torch.load(qr_run/'normalization.pt', map_location='cpu', weights_only=False)
    raw = torch.load(raw_run/'raw.pt', map_location='cpu', weights_only=False)
    assert tensor_sha(raw['constraints']) == norm['source_constraints_sha256']
    assert torch.equal(payload['N64'], norm['N32'].double()) and torch.equal(norm['margins32'], raw['margins'])
    assert tensor_sha(norm['N32']) == result['normalization']['N32_sha256']
    assert torch.equal(norm['active'], norm['norms'] > 1e-12)
    assert torch.equal(norm['N32'], raw['constraints'][norm['active']]/norm['norms'][norm['active'], None])
    historical = read(raw_run/'source_failure/proposals.json')
    for arm in ARMS:
        assert result['arms'][arm]['numerical_qualification']
        assert tensor_sha(raw['raws'][arm]['raw']) == historical['arms'][arm]['raw_sha256']
        assert raw['raws'][arm]['raw_norm'] == result['arms'][arm]['length'] == historical['arms'][arm]['raw_norm']
        assert float(raw['raws'][arm]['raw'].norm()) == raw['raws'][arm]['raw_norm'], '原FP32 raw norm必须逐值相同'
    return dict(qr_run=qr_run, raw_run=raw_run, result=result, protocol=protocol, tensors=payload, normalization=norm, raw=raw)


def qualified_step(raw, delta32, rows, margins, length, gradients, cfg):
    assert raw.dtype == delta32.dtype == torch.float32 and torch.isfinite(delta32).all()
    slack = rows@delta32.double()/length-margins
    assert bool((slack >= -1e-6).all()), '保留原投影FP32写回门槛'
    vector, report = previous.original_step(raw, delta32, gradients, cfg)
    assert report['parameter_updates'] == 1 and report['minimum_cosine'] >= -1e-5
    report['projection'] = dict(method='qualified QR/SLSQP original Euclidean projection', feasible=True,
        tolerance=1e-6, actual_slack_min=float(slack.min()), actual_violations=int((slack < -1e-6).sum()),
        delta32_sha256=tensor_sha(delta32), numerical_qualification_is_not_quality=True)
    return vector, report


def prepare(args):
    assert not args.out.exists(), '不覆盖已有QR replay'
    source_args = copy.copy(args)
    source_args.out = args.prior/'autonomous_repair_20261116/terminal_head_one_step'
    original = old.verify(source_args)
    state = qualified_inputs(args)
    assert sha(source_args.out/'gradient_qualification.json') == sha(state['raw_run']/'gradient_qualification.json')
    copied, inputs = {}, {}
    names = dict(original['copied_sha256'])
    for name in ('one_step_overlay.json', 'gradient_qualification.json'):
        names[name] = sha(source_args.out/name)
    for index in range(4):
        for extension in ('pt', 'json'):
            name = 'gradients/shard%d.%s' % (index, extension)
            names[name] = sha(source_args.out/name)
            assert names[name] == sha(state['raw_run']/name)
            if extension == 'pt':
                assert names[name] == state['raw']['source_gradient_sha256']['shard%d.pt' % index]
    for name, digest in names.items():
        target = args.out/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_args.out/name, target)
        assert sha(target) == digest
        copied[name] = inputs[str((source_args.out/name).resolve())] = digest
    for name in ('diagnostic_protocol.json', 'manifest.json', 'result.json', 'normalization.pt', 'diagnostic.pt'):
        path = state['qr_run']/name
        inputs[str(path.resolve())] = sha(path)
    inputs[str((state['raw_run']/'raw.pt').resolve())] = sha(state['raw_run']/'raw.pt')
    sources = dict(state['protocol']['source_sha256'])
    sources.update({name: sha(SOURCE/name) for name in ('tools/e2_terminal_head_qr_replay.py', 'scripts/e2_terminal_head_qr_replay.sbatch')})
    overlay = dict(method='apply saved qualified QR delta32; original normscale -> .5 -> cap; original S0 replay',
        source_qr_run=str(state['qr_run']), source_qr_result_sha256=sha(state['qr_run']/'result.json'),
        source_qr_training_updates=0, source_gradient_job='125463', parameter_count=2308,
        old_overlay_is_source_only=True, input_sha256=inputs, copied_sha256=copied, source_sha256=sources, **COUNTS)
    write(args.out/'qr_replay_overlay.json', overlay)
    spec = read(args.out/'protocol.json')
    arms = {}
    (args.out/'training').mkdir()
    for arm in ARMS:
        raw = state['raw']['raws'][arm]
        norm = state['normalization']
        vector, report = qualified_step(raw['raw'], state['tensors'][arm]['delta32'], norm['N32'].double(),
            norm['margins32'][norm['active']].double(), raw['raw_norm'], state['raw']['constraints'], spec['training'])
        head = old.load_head(args, 'cpu')
        initial, base_hash = tensor_sha(flatten(head.increment_parameters())), weights_digest(head.base)
        set_parameters(head.increment_parameters(), vector)
        effective = old.effective_difference(head)
        report.update(arm=arm, trainable_parameters=2308, base_checkpoint_sha256=CHECKPOINT_SHA,
            base_b_hash=base_hash, initial_head_sha256=initial, head_sha256=tensor_sha(flatten(head.increment_parameters())),
            effective_weight_difference_sha256=tensor_sha(effective), effective_weight_difference=vector_summary(effective),
            protocol_sha256=sha(args.out/'protocol.json'), overlay_sha256=sha(args.out/'one_step_overlay.json'),
            gradient_qualification_sha256=sha(args.out/'gradient_qualification.json'),
            qr_replay_overlay_sha256=sha(args.out/'qr_replay_overlay.json'),
            source_qr_result_sha256=overlay['source_qr_result_sha256'], source_raw_sha256=tensor_sha(raw['raw']),
            optimizer_proposals=0, frozen_unchanged=weights_digest(head.base) == base_hash,
            heldout_used_for_training=False, adopted=False, acceptance='pending original complete S0 replay')
        path = args.out/'training'/(arm+'.pt')
        torch.save(dict(delta_weight=head.delta_weight.detach(), delta_bias=head.delta_bias.detach(),
            base_checkpoint_sha256=CHECKPOINT_SHA, report=report), path)
        arms[arm] = dict(report, checkpoint_sha256=sha(path))
    write(args.out/'training/proposals.json', dict(arms=arms, conditions=48, parameter_count=2308,
        stop_before_replay=False, frozen_unchanged=True, heldout_used_for_training=False,
        qr_replay_overlay_sha256=sha(args.out/'qr_replay_overlay.json'), **COUNTS))
    verify(args)


def verify(args):
    overlay = read(args.out/'qr_replay_overlay.json')
    old.verify(args)
    for path, digest in overlay['input_sha256'].items():
        assert sha(Path(path)) == digest
    assert all(sha(args.out/path) == digest for path, digest in overlay['copied_sha256'].items())
    assert all(sha(SOURCE/path) == digest for path, digest in overlay['source_sha256'].items())
    state = qualified_inputs(args)
    proposals, spec = read(args.out/'training/proposals.json'), read(args.out/'protocol.json')
    assert proposals['qr_replay_overlay_sha256'] == sha(args.out/'qr_replay_overlay.json')
    assert all(proposals[key] == value and overlay[key] == value for key, value in COUNTS.items())
    for arm in ARMS:
        report, raw, norm = proposals['arms'][arm], state['raw']['raws'][arm], state['normalization']
        expected, step = qualified_step(raw['raw'], state['tensors'][arm]['delta32'], norm['N32'].double(),
            norm['margins32'][norm['active']].double(), raw['raw_norm'], state['raw']['constraints'], spec['training'])
        assert all(report[key] == value for key, value in step.items())
        head = old.load_candidate(args, arm, 'cpu', report)
        assert torch.equal(flatten(head.increment_parameters()), expected)
        assert report['qr_replay_overlay_sha256'] == sha(args.out/'qr_replay_overlay.json')
        assert report['source_qr_result_sha256'] == overlay['source_qr_result_sha256']
    return overlay, proposals


def evaluate(args):
    verify(args)
    old.evaluate(args)
    verify(args)


def review(args):
    assert not (args.out/'result.json').exists()
    overlay, proposals = verify(args)
    _, values, hashes, qualification = old.gradient_inputs(args, read(args.out/'one_step_overlay.json'))
    assert qualification == read(args.out/'gradient_qualification.json') and qualification['passed']
    audit, spec = Inputs(), read(args.out/'protocol.json')
    reports = {model: {'replay': audit.read(args.out/'evaluation'/model/'replay/result.json')} for model in ('f68',)+ARMS}
    baseline = reports['f68']['replay']
    prior_rows = {summary.row_key(row, 'replay'): row for row in baseline['rows']}
    decisions = {}
    for arm in ARMS:
        replay, proposal = reports[arm]['replay'], proposals['arms'][arm]
        assert replay['overlay_sha256'] == sha(args.out/'one_step_overlay.json')
        assert replay['source_sha256'] == read(args.out/'one_step_overlay.json')['source_sha256']
        assert replay['base_checkpoint_sha256'] == proposal['base_checkpoint_sha256'] == CHECKPOINT_SHA
        assert replay['checkpoint_sha256'] == proposal['checkpoint_sha256'] and replay['head_hash'] == proposal['head_sha256']
        assert replay['base_b_hash'] == proposal['base_b_hash'] == baseline['b_hash'] and replay['net_hash'] == baseline['net_hash']
        assert replay['complete_s0_images'] == 48 and replay['new_reverse_passes'] == 0
        validate_report(replay, arm, spec, sha(args.out/'protocol.json'), values, hashes, audit, args.out)
        assert same_numbers(replay['strict_replay_protection'], strict_replay_protection(replay['rows'], prior_rows))
        assert all(all(row[key] for key in ('prefix_unchanged', 'full_cache_latent_equal', 'full_cache_tasks_equal', 'full_cache_png_equal')) for row in replay['rows'])
        payload = torch.load(args.out/'training'/(arm+'.pt'), map_location='cpu', weights_only=False)
        validate_candidate(payload, proposal, old.load_head(args, 'cpu').base)
        decisions[arm] = replay_eligibility(proposal, replay, baseline)
    innovation = summary.innovation(reports, {'C1': {'eligible': decisions['C1']['holdout_candidate']}})
    innovation['eligibility_scope'] = 'replay only; no holdout or checkpoint adoption'
    result = dict(phase='apply qualified QR projection and original S0 replay', overlay=overlay, proposals=proposals,
        qualification=qualification, decisions=decisions, innovation=innovation, pastel=summary.pastel_rows(reports),
        comparisons={a+'_minus_'+b: dict(paired=summary.paired_changes(reports[b]['replay'], reports[a]['replay'], 'replay'),
            response=summary.response_changes(reports[b]['replay'], reports[a]['replay'])) for a, b in (('C0', 'f68'), ('C1', 'f68'), ('C1', 'C0'))},
        groups={model: item['replay']['groups'] for model, item in reports.items()}, complete_s0_images=96,
        f68_unchanged=True, source_numeric_qualification_is_not_quality=True, review_input_sha256=audit.hashes, **COUNTS)
    verify(args)
    write(args.out/'result.json', result)
    names = ['qr_replay_overlay.json', 'protocol.json', 'one_step_overlay.json', 'gradient_qualification.json', 'training/proposals.json', 'result.json']
    names += ['training/'+arm+'.pt' for arm in ARMS]
    for model, item in reports.items():
        names += ['evaluation/'+model+'/replay/result.json']+[row['image'] for row in item['replay']['rows']]
    assert sum(name.endswith('.png') for name in names) == 144
    write(args.out/'manifest.json', {name: sha(args.out/name) for name in names})
    with zipfile.ZipFile(args.out/'review.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in names+['manifest.json']:
            archive.write(args.out/name, name)
    print('TERMINAL_QR_REPLAY', {arm: value['holdout_candidate'] for arm, value in decisions.items()}, sha(args.out/'review.zip'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'verify', 'evaluate', 'review'))
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--project', type=Path, default=Path('/share/home/u2515283058/Mymodel'))
    parser.add_argument('--arm', choices=ARMS, default='C0')
    parser.add_argument('--device', default='cuda:0')
    options = parser.parse_args()
    args = previous.make_args(options.out or options.prior/'autonomous_repair_20261116/terminal_head_qr_replay', options.prior, options.project)
    args.action, args.arm, args.device = options.action, options.arm, options.device
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    dict(prepare=prepare, verify=verify, evaluate=evaluate, review=review)[args.action](args)


if __name__ == '__main__':
    main()
