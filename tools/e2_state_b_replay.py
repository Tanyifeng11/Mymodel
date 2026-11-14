"""先检查跨身份/参考共同下降，再做4次全量回放；11个身份及供体始终隔离。"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from tools import e2_b_endpoint_losses as loss
from tools import e2_b_replay as replay
from tools import e2_state_b_endpoint as endpoint
from tools import e2_state_b_garment as garment
from tools import e2_state_b_pilot as pilot
from tools.e2_fft_experiment import SOURCE, read, sha, write
from tools.e2_oracle_texture import weights_digest


def source_hashes():
    hashes = garment.source_hashes()
    hashes.update({name: sha(SOURCE / name) for name in (
        'tools/e2_b_replay.py', 'tools/e2_state_b_replay.py',
        'tests/test_e2_b_replay.py', 'scripts/e2_state_b_replay.sbatch')})
    return hashes


def prepare(args):
    previous = read(args.comparison / 'diagnostics.json')
    assert previous['passed'] and previous['updates_committed'] == 0
    bank = replay.anchors(read(args.parent / 'protocol.json'))
    endpoint.prepare(args, max_attempts=4, checkpoint_steps=(0, 2, 4), source=source_hashes,
                     protocol_fields=dict(
                         replay_bank=bank, replay_weights=previous['weights'],
                         comparison_protocol_sha256=sha(args.comparison / 'protocol.json'),
                         comparison_diagnostics_sha256=sha(args.comparison / 'diagnostics.json'),
                         controls='same oldB400, network, train4, refs9, 40 input conditions and 7 losses; full-bank replay replaces pair-only updates',
                         schedule=[dict(step=i, bank_size=16, full_fresh_gradients=True) for i in range(1, 5)],
                         gradient_budget='16 anchor gradients for diagnosis, reused at update1; 16 fresh anchor gradients at updates2-4; no stale training gradients',
                         joint_constraints=112, projection_sweeps=4096,
                         update_gate='all112 task losses nonincreasing within original numeric tolerance; fixed initial per-image residue caps; exact group palette/light/L-detail nonregression and joint residue descent',
                         diagnostic_gate='one joint actual AdamW update passes full-bank finite pixel guard; four full-pipeline consistency controls; no committed update'))


def protocol(args):
    spec = endpoint.protocol(args, source_hashes)
    assert sha(args.comparison / 'protocol.json') == spec['comparison_protocol_sha256']
    assert sha(args.comparison / 'diagnostics.json') == spec['comparison_diagnostics_sha256']
    assert spec['replay_bank'] == replay.anchors(spec)
    assert len(spec['holdout']) == 11 and len(spec['protected_product_roots']) == 22
    assert not set(spec['protected_product_roots']) & {p['identity_group'] for p in spec['train'] + spec['references']}
    return spec


class Objective(garment.Objective):
    def __init__(self, args, spec, pipe, predictor, b):
        super().__init__(args, spec, pipe, predictor, b)
        self.capture_dir = None
        self.positive_pixels = {}
        for row in spec['replay_bank']:
            path = args.out / 'cases' / row['case'] / (row['condition'] + '_positive.png')
            self.positive_pixels[row['case'], row['condition']] = pilot.rgb_tensor(np.asarray(Image.open(path).convert('RGB')), args.device)

    @torch.no_grad()
    def measured(self, entry, latent, tasks):
        pixels = self.pixels(latent)
        pair, condition = entry['pair'], entry['condition']
        if self.capture_dir is not None:
            self.capture_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pixels).save(self.capture_dir / (pair['case_id'] + '_' + condition + '.png'))
        metrics = endpoint.old.score(self.args, pair, condition, Image.fromarray(pixels))
        positive = self.positive_pixels[pair['case_id'], condition]
        weight = self.records[pair['case_id'], condition][4]
        _, luminance = loss.luminance(pilot.rgb_tensor(pixels, self.args.device), positive, weight, pair['patch_coordinates'])
        metrics.update(luminance)
        exact = self.metric(entry).exact(pixels)
        assert exact is not None and abs(exact - metrics['old_projection']) < 1e-8
        return dict(endpoint.numbers(tasks), residue=exact, metrics=metrics)

    def gradients_and_values(self, entry, parameters):
        predicted, base = self.rollout(entry)
        tasks, _ = self.score_latents(entry, predicted, base, residue=self.metric(entry).surrogate)
        gradients = loss.task_gradients(tasks, parameters)
        return gradients, self.measured(entry, predicted, tasks)

    @torch.no_grad()
    def evaluate(self, entry):
        predicted, base = self.rollout(entry)
        tasks, _ = self.score_latents(entry, predicted, base, residue=self.metric(entry).surrogate)
        return self.measured(entry, predicted, tasks)


def load(args, spec):
    return endpoint.load(args, spec, Objective)


def entries(spec, cache):
    return [cache[row['case'], row['condition']] for row in spec['replay_bank']]


def full_gradients(args, spec, objective, bank_entries, b, label):
    gradients, before = [], []
    objective.capture_dir = args.out / 'diagnostic_images/baseline' if label == 'diagnostic' else None
    for row, entry in zip(spec['replay_bank'], bank_entries):
        gradient, values = objective.gradients_and_values(entry, list(b.parameters()))
        assert torch.isfinite(gradient).all() and (gradient[:6].norm(dim=1) > 0).all()
        gradients.append(gradient)
        before.append(values)
        write(args.out / (label + '_partial.json'), dict(rows=spec['replay_bank'][:len(before)], values=before, heldout_used=False))
        print('REPLAY_GRADIENT', label, len(before), row['case'], row['condition'], flush=True)
    objective.capture_dir = None
    return torch.stack(gradients), before


def guarded(args, spec, b, optimizer, gradients, objective, bank_entries, before, initial, label='diagnostic'):
    weights = torch.tensor(spec['replay_weights'], device=args.device)
    destination = args.out / 'candidate_images' / label
    def evaluate():
        # 只保留最后一次实际评估的像素，失败诊断也能独立复算和目视检查。
        objective.capture_dir = destination
        try:
            return [objective.evaluate(entry) for entry in bank_entries]
        finally:
            objective.capture_dir = None
    check = lambda after: replay.joint_gate(spec['replay_bank'], before, after, initial)
    report = endpoint.guarded_step(b, optimizer, gradients, weights, evaluate, before,
                                   check=check, projection_sweeps=spec['projection_sweeps'])
    if any('values' in row for row in report['backtracks']):
        report['last_evaluated_images'] = str(destination.relative_to(args.out))
    return report


def archive(args):
    endpoint.archive(args, exclude=('base_cache.pt', 'initial_gradients.pt'))


def diagnose(args):
    spec = protocol(args)
    core, pipe, net, b, cache, objective, initial = load(args, spec)
    frozen = weights_digest(net)
    bank_entries = entries(spec, cache)
    # 四个身份各一个控制，同时覆盖四个参考组；完整采样像素必须与展开完全一致。
    controls = []
    for j, pair in enumerate(spec['train']):
        row = next(r for r in spec['replay_bank'] if r['case'] == pair['case_id'] and r['group'] == replay.GROUPS[j])
        entry = cache[row['case'], row['condition']]
        picture, digest, tap, actual = endpoint.final_sample(args, core, pipe, net, pair, row['condition'], b)
        assert digest == entry['digest']
        assert all(torch.equal(entry['states'][0][k], tap.states[0][k]) for k in ('zt', 'epsilon'))
        with torch.no_grad():
            predicted, _ = objective.rollout(entry)
            difference = float((predicted - actual.float()).abs().max())
            equal = np.array_equal(objective.pixels(predicted), np.asarray(picture))
        assert difference <= 2e-4 and equal
        controls.append(dict(row, latent_max_error=difference, pixels_equal=equal))
        print('REPLAY_CONTROL', row['case'], row['group'], flush=True)

    gradients, before = full_gradients(args, spec, objective, bank_entries, b, 'diagnostic')
    endpoint.old.atomic_torch(dict(gradients=gradients.cpu(), before=before, initial_b_hash=initial,
                                  protocol_sha256=sha(args.out / 'protocol.json')), args.out / 'initial_gradients.pt')
    flat = gradients.flatten(0, 1)
    normalized = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cosine = (normalized @ normalized.T).cpu().tolist()
    parameters = list(b.parameters())
    original = loss.flatten(parameters)
    preview = guarded(args, spec, b, endpoint.optimizer(b), gradients, objective, bank_entries, before, before)
    loss.set_parameters(parameters, original)
    unchanged = weights_digest(b) == initial and weights_digest(net) == frozen
    passed = bool(preview['accepted'] and unchanged)
    write(args.out / 'diagnostics.json', dict(
        passed=passed, controls=controls, bank=spec['replay_bank'], before=before, preview=preview,
        cosine=cosine, task_order=loss.TASKS, weights=spec['replay_weights'], constraints=flat.shape[0],
        initial_b_hash=initial, m0_hash=frozen, frozen_unchanged=unchanged, updates_committed=0,
        heldout_used=False, protocol_sha256=sha(args.out / 'protocol.json'),
        initial_gradients_sha256=sha(args.out / 'initial_gradients.pt')))
    archive(args)
    assert passed, '未找到通过整组最终像素把关的共同更新；禁止启动小试'
    print('REPLAY_DIAGNOSTICS_PASS', flat.shape[0], preview['scale'], flush=True)


def train(args):
    spec, diagnostic = protocol(args), read(args.out / 'diagnostics.json')
    assert diagnostic['passed'] and diagnostic['protocol_sha256'] == sha(args.out / 'protocol.json')
    folder = args.out / 'training'
    folder.mkdir(exist_ok=True)
    assert not (folder / 'last.pt').exists(), '禁止覆盖已有更新或绕过止损续跑'
    core, pipe, net, b, cache, objective, initial = load(args, spec)
    assert initial == diagnostic['initial_b_hash'] and weights_digest(net) == diagnostic['m0_hash']
    bank_entries = entries(spec, cache)
    optimizer = endpoint.optimizer(b)
    endpoint.old.atomic_torch(dict(step=0, b=b.state_dict()), folder / 'step0.pt')
    baseline = endpoint.probe(args, spec, core, pipe, net, b, 0)
    for row in baseline['rows']:
        name = row['case'] + '_' + row['condition'] + '.png'
        assert np.array_equal(np.asarray(Image.open(folder / 'probe0' / name)),
                              np.asarray(Image.open(args.comparison / 'training/probe0' / name)))
    history = []
    for row in spec['schedule']:
        step, start = row['step'], time.monotonic()
        if step == 1:
            path = args.out / 'initial_gradients.pt'
            assert sha(path) == diagnostic['initial_gradients_sha256']
            saved = torch.load(path, map_location=args.device, weights_only=False)
            assert saved['initial_b_hash'] == initial and saved['protocol_sha256'] == sha(args.out / 'protocol.json')
            gradients, before = saved['gradients'], saved['before']
        else:
            gradients, before = full_gradients(args, spec, objective, bank_entries, b, 'step' + str(step))
        report = guarded(args, spec, b, optimizer, gradients, objective, bank_entries, before, diagnostic['before'], label='step' + str(step))
        record = dict(row, **report, before=before, seconds=time.monotonic() - start)
        history.append(record)
        payload = dict(step=step, b=b.state_dict(), optimizer=optimizer.state_dict(), history=history,
                       accepted_updates=sum(r['accepted'] for r in history), protocol_sha256=sha(args.out / 'protocol.json'))
        endpoint.old.atomic_torch(payload, folder / 'last.pt')
        endpoint.old.atomic_torch(payload, folder / ('step' + str(step) + '.pt'))
        with (folder / 'loss.jsonl').open('a', encoding='utf8') as stream:
            stream.write(json.dumps(record) + '\n')
        print('REPLAY_UPDATE', step, report['accepted'], report.get('scale'), flush=True)
        reasons = [] if report['accepted'] else ['joint_update_rejected']
        if step in spec['checkpoint_steps'] or reasons:
            reasons += endpoint.probe(args, spec, core, pipe, net, b, step, baseline)['stop_reasons']
        if reasons:
            write(folder / 'stopped.json', dict(attempts=step, accepted_updates=payload['accepted_updates'],
                                               reasons=reasons, heldout_used=False))
            archive(args)
            raise RuntimeError('联合回放止损，禁止继续或使用留出身份调参')
    assert weights_digest(net) == diagnostic['m0_hash']
    write(folder / 'complete.json', dict(attempts=4, accepted_updates=payload['accepted_updates'],
        initial_b_hash=initial, frozen_unchanged=True, heldout_used=False, no_expansion=True,
        checkpoint_sha256=sha(folder / 'step4.pt')))
    archive(args)
    print('REPLAY_PILOT_COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'diagnose', 'train', 'archive'))
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--project', type=Path, default=pilot.PROJECT)
    parser.add_argument('--data', type=Path, default=pilot.PROJECT.parent / 'datasets/BF')
    parser.add_argument('--previous', type=Path, default=pilot.PROJECT / 'output_eval/e2_state_b_material_20261113')
    parser.add_argument('--parent', type=Path, default=pilot.PROJECT / 'output_eval/e2_state_b_balanced_20261114')
    parser.add_argument('--comparison', type=Path, default=pilot.PROJECT / 'output_eval/e2_state_b_garment_20261114')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    globals()[args.action](args)


if __name__ == '__main__':
    main()
