"""空间色度 B 的真实裁片诊断、400步小试与固定留出验证。"""
import argparse
import gc
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.checkpoint import checkpoint

from tools import e2_state_b_material_data as data
from tools import e2_state_b_pilot as pilot
from tools import e2_state_b_repair as repair
from tools.e2_b_material_losses import loss_terms, material_target, old_template
from tools.e2_fft_experiment import SOURCE, environment, read, sha, write
from tools.e2_oracle_texture import decode, encode, image, projected, weights_digest
from tools.e2_state_b_lab_anchor import CHECKPOINT

UPDATES = 400
END = 50
WEIGHTS = dict(latent=1., rgb=2., pattern=3., chroma=2., palette=2., light=1., old=.06, protection=2.)


def source_hashes():
    files = ('tools/e2_state_b_material.py', 'tools/e2_state_b_material_data.py',
             'tools/e2_state_b_material_report.py', 'tools/e2_b_material_losses.py',
             'tools/e2_state_b_repair.py', 'models/e2_state_texture_b.py',
             'models/e2_lab_anchor.py', 'pipelines/IMAGGarment_pipeline.py')
    return {name: sha(SOURCE / name) for name in files}


def protocol(args):
    spec = read(args.out / 'protocol.json')
    assert spec['code'] == source_hashes() and spec['source']['commit'] == environment()['commit']
    return spec


def folder(args, pair):
    return args.out / 'cases' / pair['case_id']


def sample(args, core, pipe, net, pair, condition, b=None, capture=False):
    reference = folder(args, pair) / (condition + '_ref.png')
    assert sha(reference) == pair['reference_hashes'][condition]
    values, _ = data.inputs(args, pair, condition, args.device)
    features = projected(net, core.tensor(args.data / pair['sketch']).to(args.device).half(),
                         core.tensor(reference).to(args.device).half())
    with repair.Tap(pipe, b, values, end=END, capture=capture) as tap:
        picture, digest = image(args, core, pipe, net, pair, features)
    assert tap.index == END and (not capture or len(tap.states) == 25)
    return picture, digest, tap.states


def cache(args, core, pipe, net, pairs, b=None, originals=None):
    entries = {}
    with torch.no_grad():
        for pair in pairs:
            d = folder(args, pair)
            for condition in data.CONDITIONS:
                picture, digest, states = sample(args, core, pipe, net, pair, condition, b, capture=True)
                key = pair['case_id'], condition
                entries[key] = dict(pair=pair, condition=condition, states=states, digest=digest)
                if originals is None and condition == f"r{pair['index']}_native":
                    picture.save(d / 'train_anchor.png')
                if originals is not None:
                    assert digest == originals[key]['digest']
            print('MATERIAL_CACHE', pair['case_id'], 'M0' if b is None else 'B200', flush=True)
    return entries


class Objective:
    def __init__(self, args, pipe, predictor, b, pairs):
        self.args, self.pipe, self.predictor, self.b = args, pipe, predictor, b
        self.records = {}
        self.targets = []
        for pair in pairs:
            d = folder(args, pair)
            mask = np.asarray(Image.open(d / 'mask.png')) > 0
            anchor = np.asarray(Image.open(d / 'train_anchor.png').convert('RGB'))
            anchor_tensor = pilot.rgb_tensor(anchor, args.device)
            self.targets.append(dict(case=pair['case_id'], conditions={}))
            board = Image.new('RGB', (7 * 160, 3 * 155), 'white')
            draw = ImageDraw.Draw(board)
            for j, condition in enumerate(data.CONDITIONS):
                reference = np.asarray(Image.open(d / (condition + '_ref.png')).convert('RGB'))
                target, audit = material_target(anchor, reference, mask)
                assert audit['outside_unchanged'] and audit['chroma_quantization_error'] < .9
                goal = Image.fromarray(target)
                goal.save(d / f'{condition}_goal.png')
                with torch.no_grad():
                    z = encode(pipe.vae, goal, args.device)
                    positive = decode(pipe.vae, z)
                positive.save(d / f'{condition}_positive.png')
                values, weight = data.inputs(args, pair, condition, args.device)
                weight = torch.from_numpy(weight)[None, None].to(args.device)
                goal_rgb = pilot.rgb_tensor(target, args.device)
                positive_rgb = pilot.rgb_tensor(positive, args.device)
                coords = pair['patch_coordinates']
                old = old_template(anchor_tensor, positive_rgb, coords)
                terms = loss_terms(goal_rgb, goal_rgb, goal_rgb, weight, old, coords)
                assert max(float(v) for v in terms.values()) < 1e-6
                from tools.e2_state_b_material_report import aligned
                upper = aligned(np.asarray(positive), target, np.asarray(positive), anchor, mask)
                audit.update(vae=upper, target_sha256=sha(d / f'{condition}_goal.png'))
                self.targets[-1]['conditions'][condition] = audit
                self.records[pair['case_id'], condition] = z, goal_rgb, values, weight, old
                col, row = j // 3, j % 3
                draw.text((col * 160 + 2, row * 155 + 2), condition + ' target / VAE', fill='black')
                for side, tile in enumerate((goal, positive)):
                    board.paste(tile.resize((78, 125)), (col * 160 + side * 79, row * 155 + 20))
            board.save(d / 'training_targets.jpg', quality=95)

    def rgb(self, z):
        def decode_latent(value):
            return (self.pipe.vae.decode(value / self.pipe.vae.config.scaling_factor).sample.float() / 2 + .5).clamp(0, 1)
        return checkpoint(decode_latent, z, use_reentrant=False) if z.requires_grad else decode_latent(z)

    def view(self, entry, state_index):
        pair = entry['pair']
        target_z, goal, values, weight, old = self.records[pair['case_id'], entry['condition']]
        state = entry['states'][state_index]
        z = state['zt'].to(self.args.device)
        epsilon = state['epsilon'].to(self.args.device)
        losses = []
        for hop in range(2):
            t = torch.tensor(state['t'] - 20 * hop, device=self.args.device)
            a = self.pipe.scheduler.alphas_cumprod[int(t)].to(z.device).float()
            x0 = (z.float() - (1 - a).sqrt() * epsilon.float()) / a.sqrt()
            delta = self.b(z.float(), x0, timestep=t.reshape(1), **values)
            corrected = (epsilon.float() - (a / (1 - a)).sqrt() * delta).to(epsilon.dtype)
            final = state['index'] + hop == 49
            predicted = self.pipe.scheduler.step(corrected, t, z, return_dict=False)[0].float() if final else x0 + delta
            with torch.no_grad():
                base_z = self.pipe.scheduler.step(epsilon, t, z.detach(), return_dict=False)[0].float() if final else x0.detach()
                base = self.rgb(base_z)
            current = loss_terms(self.rgb(predicted), goal, base, weight, old, pair['patch_coordinates'])
            current['latent'] = pilot.weighted_mean((predicted - target_z).square(), values['mask'])
            losses.append(current)
            if hop == 0:
                z = self.pipe.scheduler.step(corrected, t, z, return_dict=False)[0]
                epsilon = self.predictor.epsilon(z, t - 20, entry)
        return {key: (losses[0][key] + losses[1][key]) / 2 for key in WEIGHTS}


def train(args):
    spec = protocol(args)
    source = args.repair / 'training/step400.pt'
    assert sha(source) == CHECKPOINT
    core, pipe, net = pilot.load_core(args)
    pairs = spec['train']
    before = weights_digest(net)
    assert before == read(args.validation / 'shard0.json')['m0_hash']
    b = data.initialize(args, args.device)
    d = args.out / 'training'
    d.mkdir(exist_ok=True)
    def save(step):
        torch.save(dict(step=step, b=b.state_dict()), d / f'step{step}.pt')
    save(0)
    m0 = cache(args, core, pipe, net, pairs)
    predictor = repair.FrozenPredictor(args, core, pipe, net, pairs, conditions=data.CONDITIONS)
    with torch.no_grad():
        for pair in pairs:
            entry = m0[pair['case_id'], f"r{pair['index']}_native"]
            state = entry['states'][0]
            t = torch.tensor(state['t'], device=args.device)
            replay = predictor.epsilon(state['zt'].to(args.device), t, entry)
            assert torch.equal(replay.cpu(), state['epsilon'])
    objective = Objective(args, pipe, predictor, b, pairs)
    pipe.vae.float()
    probes = []
    for pair in pairs:
        entry = m0[pair['case_id'], f"r{pair['index']}_native"]
        terms = objective.view(entry, 9)
        gradients = {k: torch.autograd.grad(v, b.output.weight, retain_graph=True)[0].detach().flatten()
                     for k, v in terms.items()}
        total = sum(WEIGHTS[k] * v for k, v in gradients.items())
        assert torch.isfinite(total).all() and total.norm() > 0
        probes.append(dict(case=pair['case_id'], loss={k: float(v.detach()) for k, v in terms.items()},
            cosine_to_total={k: float(torch.nn.functional.cosine_similarity(v, total, dim=0)) for k, v in gradients.items()},
            gradient_norm={k: float(v.norm()) for k, v in gradients.items()}))
        del terms, gradients, total
    vae = [r['vae'] for p in objective.targets for r in p['conditions'].values()]
    material_ok = all(v['correlation'] >= .5 and v['palette_ratio'] >= .5 for v in vae)
    gradient_ok = all(np.mean([p['cosine_to_total'][k] for p in probes]) >= 0 for k in ('pattern', 'chroma', 'palette', 'light'))
    write(d / 'diagnostics.json', dict(updates=0, targets=objective.targets, loss_weights=WEIGHTS,
        gradients=probes, short_rollout_replay=True, positive_controls_pass=material_ok,
        gradient_alignment_pass=gradient_ok, passed=material_ok and gradient_ok))
    print('MATERIAL_DIAGNOSTIC', material_ok, gradient_ok, flush=True)
    if args.action == 'diagnose':
        return
    assert material_ok and gradient_ok, '训练前诊断未通过，请检查固定训练目标和梯度记录'
    optimizer = torch.optim.AdamW(b.parameters(), lr=1e-4, weight_decay=.01)
    rng = np.random.default_rng(20261113)
    refreshed = None
    families = ('stripes', 'checks', 'dots')
    conditions = {family: [c for c in data.CONDITIONS if data.REFERENCES[int(c.split('_')[0][1:])] == family]
                  for family in families}
    records = []
    with (d / 'loss.jsonl').open('w', encoding='utf-8') as log:
        for step in range(1, UPDATES + 1):
            if step == 201:
                b.eval()
                pipe.vae.half()
                refreshed = cache(args, core, pipe, net, pairs, b, originals=m0)
                pipe.vae.float()
            b.train()
            optimizer.zero_grad(set_to_none=True)
            start = time.monotonic()
            pair = pairs[(step - 1) % len(pairs)]
            first = ((step - 1) // len(pairs)) % 3
            second = (first + 1) % 3
            stream = refreshed if refreshed is not None and ((step - 1) // 12) % 2 else m0
            selected = [str(rng.choice(conditions[families[i]])) for i in (first, second)]
            state_index = int(rng.integers(24))
            assert stream[pair['case_id'], selected[0]]['digest'] == stream[pair['case_id'], selected[1]]['digest']
            means = dict.fromkeys(WEIGHTS, 0.)
            for condition in selected:
                terms = objective.view(stream[pair['case_id'], condition], state_index)
                total = sum(WEIGHTS[k] * v for k, v in terms.items())
                assert torch.isfinite(total)
                (total / 2).backward()
                for k, v in terms.items():
                    means[k] += float(v.detach()) / 2
            grad = torch.nn.utils.clip_grad_norm_(b.parameters(), 1.)
            assert torch.isfinite(grad)
            optimizer.step()
            row = dict(step=step, case=pair['case_id'], conditions=selected, state_index=state_index,
                       trajectory='m0' if stream is m0 else 'b200', loss=means, grad=float(grad), seconds=time.monotonic() - start)
            log.write(json.dumps(row) + '\n')
            records.append(row)
            if step % 20 == 0:
                log.flush()
                print('MATERIAL_UPDATE', step, means, flush=True)
            if step in (200, 400):
                save(step)
    assert weights_digest(net) == before and sha(source) == CHECKPOINT
    write(d / 'complete.json', dict(updates=len(records), frozen_unchanged=True, m0_hash=before,
        checkpoint_sha256={str(i): sha(d / f'step{i}.pt') for i in (0, 200, 400)},
        protocol_sha256=sha(args.out / 'protocol.json'), loss_weights=WEIGHTS,
        mean_seconds=float(np.mean([r['seconds'] for r in records]))))
    print('MATERIAL_TRAIN_COMPLETE', len(records), flush=True)


def generate(args):
    from tools.e2_state_b_material_report import evaluate
    spec = protocol(args)
    complete = read(args.out / 'training/complete.json')
    assert complete['updates'] == UPDATES and complete['frozen_unchanged']
    core, pipe, net = pilot.load_core(args)
    before = weights_digest(net)
    assert before == complete['m0_hash']
    b = data.initialize(args, args.device).eval().requires_grad_(False)
    checkpoints = {}
    for step in (0, 400):
        path = args.out / 'training' / f'step{step}.pt'
        assert sha(path) == complete['checkpoint_sha256'][str(step)]
        checkpoints[step] = torch.load(path, map_location='cpu', weights_only=False)['b']
    with torch.inference_mode():
        for p in spec['holdout'][args.shard::2]:
            d = folder(args, p)
            assert sha(args.data / p['sketch']) == p['source_hashes']['target']['sketch']
            assert sha(d / 'mask.png') == p['mask_sha256']
            previous = read(d / 'previous_generation.json')
            mask = np.asarray(Image.open(d / 'mask.png')) > 0
            rows = []
            for seed in range(2):
                pair = dict(p, seed=p['seed'] + 100000 * seed)
                for c in spec['validation_conditions']:
                    key = f'{c}_s{seed}'
                    meta = next(r for r in previous['rows'] if r['seed'] == pair['seed'] and r['condition'] == c)
                    for arm, old in (('m0', 'm0'), ('old_raw', 'raw'), ('old_lab', 'ab_l')):
                        assert sha(d / f'{key}_{arm}.png') == meta['output_sha256'][old]
                    hashes = {}
                    for step, state in checkpoints.items():
                        b.load_state_dict(state)
                        picture, digest, _ = sample(args, core, pipe, net, pair, c, b)
                        assert digest == meta['digest']
                        path = d / f'{key}_b{step}.png'
                        picture.save(path)
                        hashes[f'b{step}'] = sha(path)
                    anchor_c = 'real_a' if c.startswith('real') else 'v16'
                    anchor = Image.open(d / f'{anchor_c}_s{seed}_m0.png').convert('RGB')
                    ref = Image.open(d / f'{c}_ref.png').convert('RGB')
                    target, audit = material_target(anchor, ref, mask)
                    assert audit['outside_unchanged']
                    goal = Image.fromarray(target)
                    goal.save(d / f'{key}_goal.png')
                    decode(pipe.vae, encode(pipe.vae, goal, args.device)).save(d / f'{key}_positive.png')
                    rows.append(dict(key=key, seed=pair['seed'], digest=meta['digest'], output_sha256=hashes, target=audit))
                print('MATERIAL_GENERATED', p['case_id'], seed, flush=True)
            write(d / 'generation.json', dict(rows=rows))
    assert weights_digest(net) == before
    del b, net, pipe, core
    gc.collect()
    torch.cuda.empty_cache()
    evaluate(args, spec)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'diagnose', 'train', 'generate', 'evaluate', 'summarize'))
    parser.add_argument('--project', type=Path, default=pilot.PROJECT)
    parser.add_argument('--data', type=Path, default=pilot.PROJECT.parent / 'datasets/BF')
    parser.add_argument('--baseline', type=Path, default=pilot.COLOR_BASELINE)
    parser.add_argument('--repair', type=Path, default=pilot.PROJECT / 'output_eval/e2_state_b_repair_20261112')
    parser.add_argument('--validation', type=Path, default=pilot.PROJECT / 'output_eval/e2_state_b_newid_real_20261113')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--shard', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_ID', '0')))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    if args.action == 'prepare':
        data.prepare(args, source_hashes())
    elif args.action in ('diagnose', 'train'):
        train(args)
    elif args.action == 'generate':
        generate(args)
    else:
        from tools import e2_state_b_material_report as report
        getattr(report, args.action)(args, protocol(args))


if __name__ == '__main__':
    main()
