"""先检查冻结轨迹，再统一Lab目标、配对采样与两步监督，固定训练400步。"""
import argparse
import collections
import copy
import json
import shutil
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from models.e2_state_texture_b import StateTextureB
from tools import e2_state_b_pilot as pilot
from tools.e2_fft_experiment import environment, read, write, sha
from tools.e2_oracle_color import lab_image, color_metrics
from tools.e2_oracle_texture import encode, decode, projected, image, weights_digest
from tools.e2_b_repair_losses import make_targets, image_losses, template

BASELINE = pilot.COLOR_BASELINE
CONDITIONS = pilot.CONDITIONS
SIZE = pilot.SIZE


def case_dir(args, pair):
    return args.out / 'cases' / pair['case_id']


def prepare(args):
    original = read(args.baseline / 'protocol.json')
    complete = read(args.baseline / 'training/complete.json')
    assert complete['updates'] == 400 and complete['frozen_unchanged']
    assert sha(args.baseline / 'training/step400.pt') == complete['checkpoint_sha256']
    for p in original['pairs']:
        assert sha(args.data / p['sketch']) == p['hashes']['sketch']
        source = args.baseline / 'cases' / p['case_id']
        dest = case_dir(args, p)
        dest.mkdir(parents=True, exist_ok=True)
        for name in ['mask.png', 'sketch.png'] + [c + '_ref.png' for c in CONDITIONS] + [f'anchor_s{s}.png' for s in range(2)]:
            shutil.copyfile(source / name, dest / name)
            assert sha(source / name) == sha(dest / name)
        assert all(sha(dest / (c + '_ref.png')) == p['reference_hashes'][c] for c in CONDITIONS)
    spec = dict(pairs=original['pairs'], source=environment(), baseline=str(args.baseline),
        baseline_checkpoint=complete['checkpoint_sha256'], baseline_protocol=sha(args.baseline / 'protocol.json'),
        architecture='original width64/two residual blocks, plus2 common ab maps; output4; 200644 parameters',
        sampler=original['sampler'], evaluation=original['evaluation'], training_updates=400,
        probe='train4 only, seed0, v16/h16/v32; first B400 window25:45 and oracle windows25:45/25:50',
        window_rule='extend to index49 if oracle final mean chroma AND old projection improve versus stopping at44',
        target='shared garment ab, mask-normalized Lab L Gaussian sigma24 plus reference L stripes; joint gamut contrast',
        training='same original initialization, AdamW1e-4/batch4 (two pairs), two-step input gradients through frozen UNet; refresh100/200/300',
        sampling='pair type and trajectory source independently balanced; initial100 M0 then half M0/half latest B',
        comparisons='primary unchanged original-target/positive-control endpoints for both B models; repaired targets secondary',
        scope=original['scope'])
    write(args.out / 'protocol.json', spec)
    print('REPAIR_PREPARED', len(original['pairs']), flush=True)


class Tap:
    def __init__(self, pipe, module, values, end=45, oracle=None, capture=False, snapshots=False):
        self.pipe, self.module, self.values, self.end = pipe, module, values, end
        self.oracle, self.capture, self.snapshots = oracle, capture, snapshots
        self.index, self.states, self.pictures = 0, [], {}

    def __enter__(self):
        self.original = self.pipe.scheduler.step
        self.pipe.scheduler.step = self.step
        return self

    def __exit__(self, *unused):
        self.pipe.scheduler.step = self.original

    def step(self, epsilon, timestep, sample, *args, **kwargs):
        a = self.pipe.scheduler.alphas_cumprod[int(timestep)].to(sample.device).float()
        x0 = (sample.float() - (1 - a).sqrt() * epsilon.float()) / a.sqrt()
        if self.capture and 25 <= self.index < 50:
            self.states.append(dict(zt=sample.detach().cpu().half(), x0=x0.detach().cpu().half(),
                                    epsilon=epsilon.detach().cpu().half(), t=int(timestep), index=self.index))
        if self.snapshots and self.index in (25, 44, 45, 49):
            self.pictures[str(self.index) + '_before'] = decode(self.pipe.vae, x0)
        if 25 <= self.index < self.end:
            if self.oracle is not None:
                delta = (self.oracle - x0) * self.values['mask']
            elif self.module is not None:
                delta = self.module(sample.float(), x0, timestep=timestep.reshape(1), **self.values)
            else:
                delta = torch.zeros_like(x0)
            epsilon = (epsilon.float() - (a / (1 - a)).sqrt() * delta).to(epsilon.dtype)
            if self.snapshots and self.index in (25, 44, 45, 49):
                self.pictures[str(self.index) + '_after'] = decode(self.pipe.vae, x0 + delta)
        self.index += 1
        return self.original(epsilon, timestep, sample, *args, **kwargs)


def sample(args, core, pipe, net, pair, condition, module=None, end=45, oracle=None, capture=False, snapshots=False, color=False):
    d = case_dir(args, pair)
    values, weight = pilot.inputs(args, pair, condition, args.device)
    if color:
        seed = 0 if pair['seed'] == next(p['seed'] for p in read(args.out / 'protocol.json')['pairs'] if p['case_id'] == pair['case_id']) else 1
        ab = read(d / f'goal_s{seed}.json')['ab']
        values['color'] = torch.tensor(ab, device=args.device).float().view(1, 2, 1, 1).expand(1, 2, 64, 48) / 100
    features = projected(net, core.tensor(args.data / pair['sketch']).to(args.device).half(),
                         core.tensor(d / (condition + '_ref.png')).to(args.device).half())
    with Tap(pipe, module, values, end, oracle, capture, snapshots) as tap:
        picture, digest = image(args, core, pipe, net, pair, features)
    assert tap.index == 50 and (not capture or len(tap.states) == 25)
    return picture, digest, tap


def measure(picture, anchor, target, positive, mask, pair, condition):
    texture = pilot.texture_metrics(picture, pair['patch_coordinates'], condition.replace('_shift', ''))
    return dict(chroma=color_metrics(picture, target, mask)['chroma_shift'],
                lab=color_metrics(picture, target, mask)['lab_shift'], texture=texture,
                **pilot.old_pattern(picture, anchor, positive, pair['patch_coordinates'], condition.replace('_shift', '')))


def probe(args):
    spec = read(args.out / 'protocol.json')
    core, pipe, net = pilot.load_core(args)
    before = weights_digest(net)
    module = StateTextureB().to(args.device)
    module.load_state_dict(torch.load(args.baseline / 'training/step400.pt', map_location=args.device, weights_only=False)['b'])
    module.eval()
    results = []
    with torch.no_grad():
        for p in (p for p in spec['pairs'] if p['split'] == 'train'):
            source = args.baseline / 'cases' / p['case_id']
            mask = np.asarray(Image.open(source / 'mask.png')) > 0
            anchor = Image.open(source / 'anchor_s0.png').convert('RGB')
            for c in ('v16', 'h16', 'v32'):
                target = Image.open(source / (c + '_s0_target.png')).convert('RGB')
                positive = Image.open(source / (c + '_s0_vae.png')).convert('RGB')
                goal = encode(pipe.vae, target, args.device)
                for arm, end in (('first_b', 45), ('oracle45', 45), ('oracle50', 50)):
                    pic, digest, tap = sample(args, core, pipe, net, p, c,
                        module=module if arm == 'first_b' else None, end=end,
                        oracle=None if arm == 'first_b' else goal, snapshots=True)
                    d = args.out / 'probe' / p['case_id'] / (c + '_' + arm)
                    d.mkdir(parents=True, exist_ok=True)
                    tap.pictures['final'] = pic
                    scores = {}
                    for name, picture in tap.pictures.items():
                        picture.save(d / (name + '.png'))
                        scores[name] = measure(picture, anchor, target, positive, mask, p, c)
                    if arm == 'first_b':
                        assert np.array_equal(np.asarray(pic), np.asarray(Image.open(source / (c + '_s0_b400.png'))))
                    results.append(dict(case_id=p['case_id'], condition=c, arm=arm, digest=digest, scores=scores))
            print('ZERO_TRAIN_PROBE', p['case_id'], flush=True)
    means = {arm: {k: float(np.mean([r['scores']['final'][k] for r in results if r['arm'] == arm]))
                   for k in ('chroma', 'lab', 'old_projection')} for arm in ('first_b', 'oracle45', 'oracle50')}
    end = 50 if all(means['oracle50'][k] < means['oracle45'][k] for k in ('chroma', 'old_projection')) else 45
    assert before == weights_digest(net)
    write(args.out / 'probe.json', dict(training_updates=0, first_b_bitwise=True, frozen_unchanged=True,
        selected_end=end, selection_uses='train4 only, predeclared oracle rule', means=means, rows=results))
    print('ZERO_TRAIN_COMPLETE', end, means, flush=True)


def targets(args):
    spec = read(args.out / 'protocol.json')
    probe_result = read(args.out / 'probe.json')
    audit = []
    for p in spec['pairs']:
        d = case_dir(args, p)
        mask = np.asarray(Image.open(d / 'mask.png')) > 0
        _, weight = pilot.inputs(args, p, 'v16', 'cpu')
        references = {c: Image.open(d / (c + '_ref.png')).convert('RGB') for c in CONDITIONS}
        for seed in range(2):
            anchor = Image.open(d / f'anchor_s{seed}.png').convert('RGB')
            pictures, info = make_targets(anchor, references, mask, weight)
            checks = {}
            for c, pixels in pictures.items():
                picture = Image.fromarray(pixels)
                picture.save(d / f'{c}_s{seed}_goal.png')
                checks[c] = dict(sha256=sha(d / f'{c}_s{seed}_goal.png'),
                    texture=pilot.texture_metrics(picture, p['patch_coordinates'], c.replace('_shift', '')),
                    color=color_metrics(picture, anchor, mask))
            info['targets'] = checks
            write(d / f'goal_s{seed}.json', info)
            audit.append(dict(case_id=p['case_id'], seed=seed, contrast=info['joint_contrast'], targets=checks))
    spec.update(active_end=probe_result['selected_end'], repaired_targets=audit, training_source=environment(),
        paired_color=dict(baseline=str(args.baseline)))
    write(args.out / 'protocol.json', spec)
    print('UNIFIED_LAB_TARGETS', len(audit) * len(CONDITIONS), flush=True)


def initialize(args):
    torch.manual_seed(20261112)
    b = StateTextureB(color_channels=2).to(args.device)
    old = torch.load(args.baseline / 'training/step0.pt', map_location='cpu', weights_only=False)['b']
    state = b.state_dict()
    for key, value in old.items():
        if key == 'input.weight':
            expanded = torch.zeros_like(state[key])
            expanded[:, :value.shape[1]] = value.to(expanded)
            state[key] = expanded
        else:
            state[key] = value
    b.load_state_dict(state)
    assert sum(p.numel() for p in b.parameters()) == 200644
    return b


def cache(args, core, pipe, net, pairs, module, end, name):
    entries = []
    with torch.no_grad():
        for p in pairs:
            d = case_dir(args, p)
            for c in CONDITIONS:
                _, digest, tap = sample(args, core, pipe, net, p, c, module, end=end, capture=True, color=True)
                entries.append(dict(pair=p, condition=c, states=tap.states, digest=digest))
            print('REPAIR_CACHE', name, p['case_id'], flush=True)
    torch.save(entries, args.out / ('cache_' + name + '.pt'))
    return entries


class FrozenPredictor:
    """复用原pipeline条件，短展开保留UNet对latent输入的梯度。"""
    def __init__(self, args, core, pipe, net, pairs):
        self.pipe, self.net, self.contexts = pipe, net, {}
        pipe.set_scale(.6)
        assert pipe.unet.config.time_cond_proj_dim is None
        with torch.no_grad():
            for p in pairs:
                text, negative = pipe.encode_prompt(p['caption'], args.device, 1, True, ' worst quality, low quality')
                sketch = core.tensor(args.data / p['sketch']).to(args.device).half()
                z = pipe.vae.encode(sketch).latent_dist.mean * .18215
                sa = core.sketch_states(pipe, z, torch.tensor(981, device=args.device))
                for c in CONDITIONS:
                    features = projected(net, sketch, core.tensor(case_dir(args, p) / (c + '_ref.png')).to(args.device).half())
                    self.contexts[p['case_id'], c] = text, negative, sa, features

    def epsilon(self, z, t, entry):
        text, negative, sa, features = self.contexts[entry['pair']['case_id'], entry['condition']]
        def conditional(value):
            self.net.adapter.features = features
            return self.pipe.unet(value.half(), t, encoder_hidden_states=text,
                cross_attention_kwargs={'sa_hidden_states': sa}, return_dict=False)[0]
        def unconditional(value):
            self.net.adapter.features = features
            return self.pipe.unet(value.half(), t, encoder_hidden_states=negative, return_dict=False)[0]
        if z.requires_grad and torch.is_grad_enabled():
            cond = checkpoint(conditional, z, use_reentrant=False)
            unc = checkpoint(unconditional, z, use_reentrant=False)
        else:
            cond, unc = conditional(z), unconditional(z)
        return unc + 7. * (cond - unc)


class Objective:
    def __init__(self, args, pipe, b, predictor, pairs, end):
        self.args, self.pipe, self.b, self.predictor, self.end = args, pipe, b, predictor, end
        self.goals, self.values, self.weights, self.templates = {}, {}, {}, {}
        encoded = {}
        with torch.no_grad():
            for p in pairs:
                for c in CONDITIONS:
                    path = case_dir(args, p) / (c + '_s0_goal.png')
                    encoded[p['case_id'], c] = encode(pipe.vae, Image.open(path).convert('RGB'), args.device)
        pipe.vae.float()
        with torch.no_grad():
            for p in pairs:
                d = case_dir(args, p)
                anchor = Image.open(d / 'anchor_s0.png').convert('RGB')
                for c in CONDITIONS:
                    goal_file = d / (c + '_s0_goal.png')
                    assert sha(goal_file) == read(d / 'goal_s0.json')['targets'][c]['sha256']
                    z = encoded[p['case_id'], c]
                    positive = decode(pipe.vae, z)
                    positive.save(d / (c + '_s0_goal_vae.png'))
                    rgb = self.decode_rgb(z)
                    key = p['case_id'], c
                    self.goals[key] = z, rgb
                    values, weight = pilot.inputs(args, p, c, args.device)
                    ab = read(d / 'goal_s0.json')['ab']
                    values['color'] = torch.tensor(ab, device=args.device).float().view(1, 2, 1, 1).expand(1, 2, 64, 48) / 100
                    self.values[key] = values
                    self.weights[key] = torch.from_numpy(weight)[None, None].to(args.device)
                    self.templates[key] = template(anchor, positive, p['patch_coordinates'], c, args.device)
                    losses, error = image_losses(rgb, rgb, self.weights[key], self.templates[key], p['patch_coordinates'])
                    assert max(float(v) for v in losses.values()) < 1e-7 and error.abs().max() == 0

    def decode_rgb(self, z):
        def fn(value):
            return (self.pipe.vae.decode(value / self.pipe.vae.config.scaling_factor).sample.float() / 2 + .5).clamp(0, 1)
        if z.requires_grad and torch.is_grad_enabled():
            return checkpoint(fn, z, use_reentrant=False)
        return fn(z)

    def view(self, entry, state_index):
        state = entry['states'][state_index]
        key = entry['pair']['case_id'], entry['condition']
        values, weight = self.values[key], self.weights[key]
        zgoal, goal = self.goals[key]
        sample_z = state['zt'].to(self.args.device)
        epsilon = state['epsilon'].to(self.args.device)
        index = state['index']
        accumulated, residuals = [], []
        for hop in range(2):
            t = torch.tensor(state['t'] - 20 * hop, device=self.args.device)
            a = self.pipe.scheduler.alphas_cumprod[int(t)].to(self.args.device).float()
            x0 = (sample_z.float() - (1 - a).sqrt() * epsilon.float()) / a.sqrt()
            if index + hop < self.end:
                delta = self.b(sample_z.float(), x0, timestep=t.reshape(1), **values)
            else:
                delta = torch.zeros_like(x0)
            predicted = x0 + delta
            rgb = self.decode_rgb(predicted)
            losses, residual = image_losses(rgb, goal, weight, self.templates[key], entry['pair']['patch_coordinates'])
            losses['latent'] = pilot.weighted_mean((predicted - zgoal).square(), values['mask'])
            with torch.no_grad():
                base = self.decode_rgb(x0.detach())
            losses['protection'] = pilot.weighted_mean((rgb - base).abs(), 1 - weight)
            accumulated.append(losses)
            residuals.append(residual)
            if hop == 0:
                corrected = (epsilon.float() - (a / (1 - a)).sqrt() * delta).to(epsilon.dtype)
                sample_z = self.pipe.scheduler.step(corrected, t, sample_z, return_dict=False)[0]
                epsilon = self.predictor.epsilon(sample_z, t - 20, entry)
        return {k: .5 * (accumulated[0][k] + accumulated[1][k]) for k in accumulated[0]}, residuals

    def pair(self, first, second, state_index):
        a, ar = self.view(first, state_index)
        b, br = self.view(second, state_index)
        losses = {k: .5 * (a[k] + b[k]) for k in a}
        weight = F.avg_pool2d(self.weights[first['pair']['case_id'], first['condition']], 32)
        losses['color'] += .5 * sum(pilot.weighted_mean((x - y).abs(), weight) for x, y in zip(ar, br))
        return losses


def training(args):
    spec = read(args.out / 'protocol.json')
    assert spec['training_source']['commit'] == environment()['commit']
    end = spec['active_end']
    pairs = [p for p in spec['pairs'] if p['split'] == 'train']
    core, pipe, net = pilot.load_core(args)
    before = weights_digest(net)
    b = initialize(args)
    with torch.no_grad():
        plain, digest, _ = sample(args, core, pipe, net, pairs[0], 'h32', end=end, color=True)
        zero, digest2, _ = sample(args, core, pipe, net, pairs[0], 'h32', b, end=end, color=True)
        assert digest == digest2 and np.array_equal(np.asarray(plain), np.asarray(zero))
    m0 = cache(args, core, pipe, net, pairs, None, end, 'm0')
    predictor = FrozenPredictor(args, core, pipe, net, pairs)
    # 校验短展开使用的条件与原50步采样逐位一致。
    with torch.no_grad():
        for entry in m0[::8]:
            state = entry['states'][0]
            replay = predictor.epsilon(state['zt'].to(args.device), torch.tensor(state['t'], device=args.device), entry)
            assert torch.equal(replay.cpu(), state['epsilon']), '短展开与原pipeline不一致'
    objective = Objective(args, pipe, b, predictor, pairs, end)
    pipe.vae.float()
    directory = args.out / 'training'
    directory.mkdir(exist_ok=True)
    optimizer = torch.optim.AdamW(b.parameters(), lr=1e-4, weight_decay=.01)
    def save(step):
        torch.save(dict(step=step, b=b.state_dict(), optimizer=optimizer.state_dict()), directory / f'step{step}.pt')
    save(0)
    # 只用固定训练探针一次标定输出层的梯度比例，随后锁定权重。
    ratios = collections.defaultdict(list)
    for identity in range(4):
        first, second = m0[identity * 8], m0[identity * 8 + 1]
        losses = objective.pair(first, second, 9)
        base = losses['latent'] + 2 * losses['gray'] + 2 * losses['protection']
        base_grad = torch.autograd.grad(base, b.output.weight, retain_graph=True)[0]
        for name, fraction in (('color', .5), ('old', .25)):
            gradient = torch.autograd.grad(losses[name], b.output.weight, retain_graph=True)[0]
            ratios[name].append(float(fraction * base_grad.norm() / gradient.norm().clamp_min(1e-9)))
        del losses, base, base_grad
    weights = dict(latent=1., gray=2., protection=2., **{k: float(np.median(v)) for k, v in ratios.items()})
    assert all(np.isfinite(v) for v in weights.values())
    write(args.out / 'calibration.json', dict(weights=weights, ratios=dict(ratios), optimizer_updates=0,
        method='median output-layer norm ratio on4 fixed training pairs, color=.5 base and old=.25 base'))
    print('LOCKED_LOSS_WEIGHTS', weights, flush=True)
    rng = np.random.default_rng(20261112)
    stream, records, counts = m0, [], collections.Counter()
    for step in range(1, 401):
        if step in (101, 201, 301):
            b.eval(); pipe.vae.half()
            current = cache(args, core, pipe, net, pairs, b, end, f'b{step-1}')
            pipe.vae.float(); stream = m0 + current
        b.train(); optimizer.zero_grad(set_to_none=True)
        start = time.monotonic()
        totals = dict.fromkeys(weights, 0.)
        tokens = []
        # 每更新两对共四个视图；轨迹来源和配对类型分别取不同计数位。
        for slot in range(2):
            identity = 2 * ((step - 1) % 2) + slot
            kind = ((step - 1) // 2 + slot) % 2
            stage = int(step > 100 and ((step - 1) // 4 + slot) % 2 == 0)
            choices = stream[stage * 32 + identity * 8:stage * 32 + (identity + 1) * 8]
            first = choices[int(rng.integers(8))]
            companion = pilot.paired_condition(first['condition'], bool(kind))
            second = next(e for e in choices if e['condition'] == companion)
            state_index = int(rng.integers(24))
            assert first['digest'] == second['digest']
            losses = objective.pair(first, second, state_index)
            total = sum(weights[k] * losses[k] for k in weights)
            assert torch.isfinite(total)
            (total / 2).backward()
            for k in totals:
                totals[k] += float(losses[k].detach()) / 2
            tokens.append([pairs[identity]['case_id'], first['condition'], companion, state_index, stage])
            counts[('early' if step <= 100 else 'late', stage, kind)] += 1
        norm = torch.nn.utils.clip_grad_norm_(b.parameters(), 1.)
        assert torch.isfinite(norm)
        optimizer.step()
        row = dict(step=step, losses=totals, tokens=tokens, grad_norm=float(norm),
                   seconds=time.monotonic()-start, peak_gib=torch.cuda.max_memory_allocated()/2**30)
        records.append(row)
        with (directory / 'loss.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        if step % 20 == 0:
            print('REPAIR_UPDATE', step, totals, flush=True)
        if step in (100, 200, 300, 400):
            save(step)
    assert all(counts['late', s, k] == 150 for s in (0, 1) for k in (0, 1))
    assert weights_digest(net) == before
    net.adapter.clear()
    write(directory / 'complete.json', dict(updates=400, finite_updates=400, zero_init_bitwise=True,
        frozen_unchanged=True, frozen_hash=before, parameters=200644, short_rollout_pipeline_bitwise=True,
        positive_control_zero_loss=True, pairing_balanced={str(k): v for k,v in counts.items()},
        checkpoint_sha256=sha(directory/'step400.pt'), loss_weights=weights,
        first20={k:float(np.mean([r['losses'][k] for r in records[:20]])) for k in weights},
        last20={k:float(np.mean([r['losses'][k] for r in records[-20:]])) for k in weights}))
    print('REPAIR_TRAINING_COMPLETE_400', flush=True)


def generate(args):
    spec = read(args.out/'protocol.json')
    complete = read(args.out/'training/complete.json')
    assert sha(args.out/'training/step400.pt') == complete['checkpoint_sha256']
    core, pipe, net = pilot.load_core(args)
    before = weights_digest(net)
    b = StateTextureB(color_channels=2).to(args.device)
    b.load_state_dict(torch.load(args.out/'training/step400.pt', map_location=args.device, weights_only=False)['b'])
    b.eval()
    with torch.no_grad():
        for p in spec['pairs'][args.shard::2]:
            d = case_dir(args, p)
            source = args.baseline/'cases'/p['case_id']
            expected = read(source/'generation.json')
            rows = []
            for seed in range(2):
                pair = dict(p, seed=p['seed']+100000*seed)
                for c in CONDITIONS:
                    picture, digest, _ = sample(args, core, pipe, net, pair, c, b, end=spec['active_end'], color=True)
                    old = next(r for r in expected['rows'] if r['seed']==pair['seed'] and r['condition']==c)
                    assert digest == old['digest']
                    picture.save(d/f'{c}_s{seed}_b400.png')
                    rows.append(dict(condition=c, seed=pair['seed'], digest=digest))
                    for arm in ('m0','target','vae'):
                        shutil.copyfile(source/f'{c}_s{seed}_{arm}.png', d/f'{c}_s{seed}_{arm}.png')
                print('REPAIR_GENERATED',p['case_id'],seed,flush=True)
            write(d/'generation.json',dict(paired_noise=True,rows=rows))
    assert weights_digest(net)==before
    write(args.out/f'shard{args.shard}.json',dict(frozen_unchanged=True,checkpoint_sha256=complete['checkpoint_sha256']))


def summarize(args):
    pilot.summarize(args)
    result = read(args.out/'summary.json')
    result['repair'] = dict(probe=read(args.out/'probe.json'),calibration=read(args.out/'calibration.json'),
        primary='all original B endpoints use identical original targets and VAE controls for both models',
        target_changes='new training targets stored as *_goal.png; no final image postprocessing')
    write(args.out/'summary.json',result)
    # 原比较图保留第一轮目标；补充新目标图供核对训练目标变化。
    for p in read(args.out/'protocol.json')['pairs']:
        d=case_dir(args,p)
        canvas=Image.new('RGB',(4*154,4*226),'white'); draw=ImageDraw.Draw(canvas)
        for row,c in enumerate(pilot.VARIANTS):
            paths=[('first B',args.baseline/'cases'/p['case_id']/f'{c}_s0_b400.png'),
                   ('repaired target',d/f'{c}_s0_goal.png'),('repair B',d/f'{c}_s0_b400.png'),
                   ('original anchor',d/'anchor_s0.png')]
            for col,(label,path) in enumerate(paths):
                x,y=col*154,row*226
                draw.text((x+3,y+3),c+' '+label,fill='black')
                canvas.paste(Image.open(path).convert('RGB').resize((150,200)),(x,y+22))
        canvas.save(d/'repaired_panel.jpg',quality=94)
    original=args.out/'b400_paired_color_review.zip'
    archive=args.out/'b400_repair_review.zip'
    with zipfile.ZipFile(original) as old, zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
        for name in old.namelist():
            if name!='summary.json':
                bundle.writestr(name,old.read(name))
        extras=[args.out/'summary.json',args.out/'probe.json',args.out/'calibration.json']
        extras+=list((args.out/'cases').glob('*/*_goal.png'))+list((args.out/'cases').glob('*/goal_s*.json'))
        extras+=list((args.out/'cases').glob('*/repaired_panel.jpg'))+list((args.out/'probe').glob('*/*/*.png'))
        for path in extras:
            bundle.write(path,path.relative_to(args.out))
    print('REPAIR_ARCHIVE',archive.stat().st_size,sha(archive),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'probe', 'targets', 'training', 'generate', 'summarize'))
    parser.add_argument('--project', type=Path, default=pilot.PROJECT)
    parser.add_argument('--data', type=Path, default=pilot.PROJECT.parent / 'datasets/BF')
    parser.add_argument('--baseline', type=Path, default=BASELINE)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--shard', type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    globals()[args.action](args)


if __name__ == '__main__':
    main()
