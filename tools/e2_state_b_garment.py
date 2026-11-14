"""复用完整去噪小试，只对齐残留训练目标和验收指标。"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from tools import e2_b_endpoint_losses as loss
from tools import e2_state_b_endpoint as endpoint
from tools import e2_state_b_pilot as pilot
from tools.e2_b_garment_projection import GarmentProjection
from tools.e2_fft_experiment import SOURCE, read, sha, write
from tools.e2_oracle_texture import weights_digest


def source_hashes():
    hashes = endpoint.source_hashes()
    hashes.update({name: sha(SOURCE / name) for name in (
        'tools/e2_b_garment_projection.py', 'tools/e2_state_b_garment.py',
        'tests/test_e2_b_garment_projection.py', 'scripts/e2_state_b_garment.sbatch')})
    return hashes


def prepare(args):
    previous = args.parent.with_name('e2_state_b_endpoint_strict_20261114') / 'diagnostics.json'
    old = read(previous)
    assert old['passed'] and list(old['task_order']) == list(loss.TASKS)
    endpoint.prepare(args, max_attempts=16, checkpoint_steps=(0, 10, 16), source=source_hashes,
                     protocol_fields=dict(
                         controls='old B400, same train4, refs9 and first16 pairs; same guarded AdamW; change only residue objective',
                         residue='same train_anchor, VAE positive, 16-px interior, Lab high-pass projection as acceptance; exact pixel guard',
                         residue_guard='pipeline fp16 VAE plus uint8 OpenCV score after every candidate update',
                         old_task_weights=old['weights'], weight_source_sha256=sha(previous),
                         diagnostic_gate='3 of 5 previews accepted; both low-chroma stripe probes strictly improve residue'))


def protocol(args):
    return endpoint.protocol(args, source_hashes)


class Objective(endpoint.Objective):
    def __init__(self, args, spec, pipe, predictor, b):
        super().__init__(args, spec, pipe, predictor, b)
        self.projections = {}
        for pair in spec['train']:
            case = pair['case_id']
            folder = args.out / 'cases' / case
            mask = np.asarray(Image.open(folder / 'mask.png')) > 0
            anchor = np.asarray(Image.open(folder / 'train_anchor.png').convert('RGB'))
            for condition in spec['conditions']:
                positive = np.asarray(Image.open(folder / f'{condition}_positive.png').convert('RGB'))
                self.projections[case, condition] = GarmentProjection(positive, anchor, mask, args.device)

    def metric(self, entry):
        return self.projections[entry['pair']['case_id'], entry['condition']]

    @torch.no_grad()
    def pixels(self, latent):
        vae = self.pipe.vae
        previous = vae.dtype
        vae.half()
        try:
            decoded = vae.decode(latent.half() / vae.config.scaling_factor, return_dict=False)[0]
            rgb = (decoded / 2 + .5).clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy()
            return np.rint(rgb * 255).astype(np.uint8)
        finally:
            vae.to(dtype=previous)

    def gradients_and_values(self, entry, parameters):
        predicted, base = self.rollout(entry)
        metric = self.metric(entry)
        tasks, _ = self.score_latents(entry, predicted, base, residue=metric.surrogate)
        gradients = loss.task_gradients(tasks, parameters)
        values = endpoint.numbers(tasks)
        values['residue'] = metric.exact(self.pixels(predicted)) or 0.
        return gradients, values

    @torch.no_grad()
    def evaluate(self, entry):
        predicted, base = self.rollout(entry)
        tasks, _ = self.score_latents(entry, predicted, base, residue=self.metric(entry).surrogate)
        values = endpoint.numbers(tasks)
        values['residue'] = self.metric(entry).exact(self.pixels(predicted)) or 0.
        return values


def load(args, spec):
    return endpoint.load(args, spec, Objective)


def diagnose(args):
    spec = protocol(args)
    core, pipe, net, b, cache, objective, initial = load(args, spec)
    frozen = weights_digest(net)
    parameters = list(b.parameters())
    # 包含上轮推高残留的621、1526条纹，以及浅彩、多色和合成参考对。
    rows = [spec['schedule'][i - 1] for i in (1, 4, 5, 9, 13)]
    cases = {p['case_id']: p for p in spec['train']}
    checks = []
    for row in rows:
        pair = cases[row['case']]
        entries = [cache[row['case'], c] for c in row['conditions']]
        gradients, before, checks_for_pair = [], [], []
        for entry in entries:
            condition = entry['condition']
            picture, digest, tap, actual = endpoint.final_sample(args, core, pipe, net, pair, condition, b)
            assert digest == entry['digest']
            assert all(torch.equal(entry['states'][0][k], tap.states[0][k]) for k in ('zt', 'epsilon'))
            with torch.no_grad():
                predicted, _ = objective.rollout(entry)
                pixels = objective.pixels(predicted)
            error = float((predicted - actual.float()).abs().max())
            assert error <= 2e-4 and np.array_equal(pixels, np.asarray(picture))
            g, values = objective.gradients_and_values(entry, parameters)
            expected = endpoint.old.score(args, pair, condition, picture)['old_projection']
            assert abs(values['residue'] - (expected or 0.)) <= 1e-8
            assert torch.isfinite(g).all() and (expected is None or float(g[3].norm()) > 0)
            gradients.append(g)
            before.append(values)
            checks_for_pair.append(dict(condition=condition, latent_max_error=error,
                                        pixels_equal=True, residue=expected, residue_gradient_norm=float(g[3].norm())))
        checks.append((row, entries, torch.stack(gradients), before, checks_for_pair))
        print('GARMENT_METRIC_MATCH', row['step'], row['case'], row['group'], flush=True)

    norms = torch.cat([g.norm(dim=2) for _, _, g, _, _ in checks]).cpu().median(0).values.to(args.device)
    weights = torch.tensor(spec['old_task_weights'], device=args.device)
    weights[3] = (norms[0] / norms[3].clamp_min(1e-12)).clamp(.05, 20.)
    reports = []
    for row, entries, gradients, before, checks_for_pair in checks:
        start = loss.flatten(parameters)
        preview = endpoint.guarded_step(b, endpoint.optimizer(b), gradients, weights,
                                        lambda: [objective.evaluate(e) for e in entries], before)
        loss.set_parameters(parameters, start)
        reports.append(dict(row=row, controls=checks_for_pair, before=before, preview=preview))
        print('GARMENT_PREVIEW', row['step'], row['group'], preview['accepted'], preview.get('scale'), flush=True)

    accepted = sum(r['preview']['accepted'] for r in reports)
    critical = [r for r in reports if r['row']['step'] in (1, 4)]
    critical_improvement = all(r['preview']['accepted'] and
        r['preview']['backtracks'][-1]['values'][0]['residue'] < r['before'][0]['residue'] for r in critical)
    unchanged = weights_digest(b) == initial and weights_digest(net) == frozen
    passed = accepted >= 3 and critical_improvement and unchanged
    write(args.out / 'diagnostics.json', dict(
        passed=passed, rows=reports, accepted_previews=accepted, critical_residue_improves=critical_improvement,
        weights=weights.tolist(), task_order=loss.TASKS, initial_b_hash=initial, m0_hash=frozen,
        protocol_sha256=sha(args.out / 'protocol.json'), updates_committed=0, heldout_used=False))
    endpoint.archive(args)
    assert passed, '全衣身残留零更新检查未通过'
    print('GARMENT_DIAGNOSTICS_PASS', accepted, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'diagnose', 'train', 'archive'))
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--project', type=Path, default=pilot.PROJECT)
    parser.add_argument('--data', type=Path, default=pilot.PROJECT.parent / 'datasets/BF')
    parser.add_argument('--previous', type=Path, default=pilot.PROJECT / 'output_eval/e2_state_b_material_20261113')
    parser.add_argument('--parent', type=Path, default=pilot.PROJECT / 'output_eval/e2_state_b_balanced_20261114')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    if args.action == 'train':
        endpoint.train(args, read_protocol=protocol, load_objective=load)
    elif args.action == 'archive':
        endpoint.archive(args)
    else:
        globals()[args.action](args)


if __name__ == '__main__':
    main()
