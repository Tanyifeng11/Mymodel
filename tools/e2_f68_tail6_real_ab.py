"""从 f68 出发，用真实成衣监督原 B 的末六步衣内修正。"""
import argparse
import copy
import inspect
import os
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from models.e2_lab_anchor import rgb_to_lab
from models.e2_state_texture_b import StateTextureB
from tools import e2_f68_region_diagnostic as roi, e2_f68_rgb_distill as distill
from tools import e2_state_b_joint as joint, e2_state_b_repair as repair
from tools.e2_f68_edge_residual import ACCEPTANCE_SHA, mix_pixels, summarize, verified_files
from tools.e2_f68_real_paired import (
    ARMS, CHECKPOINT_SHA, CONDITION, METRICS, SIZE, image_scores, metric_assets,
    metric_models, rgb, verify, write,
)
from tools.e2_f68_single_s0_gate import ALPHA, latent_gate, subset
from tools.e2_fft_experiment import read, sha
from tools.e2_oracle_texture import weights_digest
from tools.e2_state_b_backward_repro import tensor_sha

RUN = 'tail6_real_ab_20261119'
FIRST, LAST, STEPS = 44, 50, 128
LEARNING_RATE, LPIPS_WEIGHT, BAND_WEIGHT = .0001, .1, .25
PARAMETERS = 76420
ARMS_AB = ('A', 'B')


def save_torch(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    torch.save(value, temp)
    temp.replace(path)


class TailResidual:
    def __init__(self, original, candidate, gate, index=25):
        self.original, self.candidate, self.gate, self.index = original, candidate, gate, index
        self.interior = ((1 - gate) / (1 - ALPHA)).clamp(0, 1)

    def __call__(self, *args, **kwargs):
        residual = self.original(*args, **kwargs)
        result = residual * self.gate
        if self.index >= FIRST and self.candidate is not None:
            result = result + self.gate * self.interior * (self.candidate(*args, **kwargs) - residual)
        self.index += 1
        return result


def tensor_set(models):
    return {name: weights_digest(model) for name, model in models.items()}


def load_models(args):
    assert sha(args.checkpoint) == CHECKPOINT_SHA
    context = copy.copy(args)
    context.action, context.out = 'train', args.run
    core, pipe, net = joint.pilot.load_core(context)
    pipe.vae.half().eval().requires_grad_(False)
    net.eval().requires_grad_(False)
    original = StateTextureB(color_channels=2, reference_channels=192).to(args.device)
    original.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=False)['b'])
    original.eval().requires_grad_(False)
    models = dict(B=original, net=net, vae=pipe.vae)
    assert tensor_set(models) == args.original['frozen_model_hashes']
    assert type(pipe.scheduler).__name__ == 'DDIMScheduler'
    assert sha(Path(inspect.getsourcefile(type(pipe.scheduler)))) == args.original['scheduler_source_sha256']
    return core, pipe, net, original, models


def candidates(original):
    head = copy.deepcopy(original)
    head.blocks[-1].requires_grad_(True)
    head.output.requires_grad_(True)
    parameters = {name: value for name, value in head.named_parameters() if value.requires_grad}
    assert sum(value.numel() for value in parameters.values()) == PARAMETERS
    return head, parameters


def case_mask(args, pair):
    folder = args.out / 'cases' / pair['case_id']
    path = folder / 'mask.png'
    assert sha(path) == pair['mask_sha256']
    assert sha(args.data / pair['sketch']) == pair['input_sha256']['sketch']
    assert sha(folder / (CONDITION + '_ref.png')) == pair['reference_hashes'][CONDITION]
    return np.asarray(rgb(path))[..., 0] > 0


class PrefixComplete(Exception):
    pass


class PrefixTap(repair.Tap):
    def __init__(self, *args, qualify=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.qualify = qualify

    def step(self, epsilon, timestep, z, *args, **kwargs):
        index = self.index
        if index == FIRST:
            self.start = dict(z=z.detach().cpu(), epsilon=epsilon.detach().cpu(), t=int(timestep))
            if not self.qualify:
                raise PrefixComplete
        result = super().step(epsilon, timestep, z, *args, **kwargs)
        if index == LAST - 1:
            self.final = result[0].detach().cpu()
        return result


@torch.no_grad()
def cache_prefix(args, core, pipe, net, original, pair, qualify=False):
    mask = case_mask(args, pair)
    module = TailResidual(original, None, latent_gate(mask, args.device))
    values, _ = joint.data.inputs(args, pair, CONDITION, 'raw')
    ref = args.out / 'cases' / pair['case_id'] / (CONDITION + '_ref.png')
    features = joint.projected(net, core.tensor(args.data / pair['sketch']).to(args.device).half(),
                               core.tensor(ref).to(args.device).half())
    with PrefixTap(pipe, module, values, end=LAST, qualify=qualify) as tap:
        try:
            picture, _ = joint.image(args, core, pipe, net, pair, features)
        except PrefixComplete:
            assert tap.index == FIRST and module.index == FIRST
            return tap.start
        finally:
            net.adapter.clear()
    assert qualify and tap.index == module.index == LAST
    return tap.start, tap.final, picture


def decode(pipe, latent):
    def forward(value):
        image = pipe.vae.decode(value / pipe.vae.config.scaling_factor, return_dict=False)[0]
        return (image / 2 + .5).clamp(0, 1).float()
    if latent.requires_grad and torch.is_grad_enabled():
        return checkpoint(forward, latent, use_reentrant=False)
    return forward(latent)


def trajectory(args, models, candidate, pair, state, mask):
    core, pipe, net, original, _ = models
    predictor = repair.FrozenPredictor(args, core, pipe, net, [pair], (CONDITION,))
    descriptor = dict(pair=pair, condition=CONDITION)
    values, _ = joint.data.inputs(args, pair, CONDITION, 'raw')
    tail = TailResidual(original, candidate, latent_gate(mask, args.device), index=FIRST)
    pipe.scheduler.set_timesteps(LAST, device=args.device)
    z, epsilon = (state[name].to(args.device) for name in ('z', 'epsilon'))
    times = pipe.scheduler.timesteps
    assert int(times[FIRST]) == state['t']
    for index in range(FIRST, LAST):
        t = times[index]
        if index != FIRST:
            epsilon = predictor.epsilon(z, t, descriptor)
        a = pipe.scheduler.alphas_cumprod[int(t)].to(z.device).float()
        x0 = (z.float() - (1 - a).sqrt() * epsilon.float()) / a.sqrt()
        delta = tail(z.float(), x0, timestep=t.reshape(1), **values)
        corrected = (epsilon.float() - (a / (1 - a)).sqrt() * delta).to(epsilon.dtype)
        z = pipe.scheduler.step(corrected, t, z, eta=0, return_dict=False)[0]
    assert tail.index == LAST
    return z


def qualify(args):
    models = load_models(args)
    core, pipe, net, original, frozen = models
    head, parameters = candidates(original)
    pair = next(p for p in args.trainval['pairs'] if p['split'] == 'val')
    mask = case_mask(args, pair)
    state, expected, picture = cache_prefix(args, core, pipe, net, original, pair, qualify=True)
    with torch.no_grad():
        output = trajectory(args, models, head, pair, state, mask)
        pixels = decode(pipe, output).cpu().permute(0, 2, 3, 1).numpy()
        replay = pipe.numpy_to_pil(pixels)[0]
    assert torch.equal(output.cpu(), expected), float((output.cpu() - expected).abs().max())
    assert np.array_equal(np.asarray(picture), np.asarray(replay))
    path = args.run / 'prefix' / (pair['case_id'] + '.pt')
    save_torch(path, dict(**state, case=pair['case_id'], contract=args.contract))
    prediction, target, mask = render_pair(args, models, head, pair, {pair['case_id']: sha(path)})
    assert torch.equal(prediction.detach().cpu().permute(0, 2, 3, 1), torch.from_numpy(pixels))
    objective, _ = RealPairLoss(args, True)(prediction, target, mask)
    norm = backward(objective, parameters)
    assert norm > 0
    assert tensor_set(frozen) == args.original['frozen_model_hashes']
    write(args.run / 'qualification.json', dict(case=pair['case_id'], cached_sha256=sha(path),
          expected_latent=tensor_sha(expected), replay_latent=tensor_sha(output.cpu()),
          exact_latent=True, exact_pixels=True, grad_norm=norm, updates=0, contract=args.contract))
    print('TAIL6_QUALIFIED', pair['case_id'], norm, flush=True)


def cache(args):
    core, pipe, net, original, frozen = load_models(args)
    pairs = args.trainval['pairs'][args.shard::args.shards]
    covered = {}
    for pair in pairs:
        case = pair['case_id']
        path = args.run / 'prefix' / (case + '.pt')
        if not path.exists():
            state = cache_prefix(args, core, pipe, net, original, pair)
            save_torch(path, dict(**state, case=case, contract=args.contract))
        entry = torch.load(path, map_location='cpu', weights_only=True)
        assert entry['case'] == case and entry['contract'] == args.contract
        covered[case] = sha(path)
        print('TAIL6_PREFIX', args.shard, len(covered), len(pairs), case, flush=True)
    assert tensor_set(frozen) == args.original['frozen_model_hashes']
    write(args.run / f'prefix_{args.shard}.json', dict(cases=covered, contract=args.contract))


class RealPairLoss:
    def __init__(self, args, with_band):
        import lpips
        assets = args.protocol['metric_assets']
        model = lpips.LPIPS(net='alex', pnet_rand=True, model_path=assets['lpips_calibration']['path'])
        alex = torch.load(assets['alexnet']['path'], map_location='cpu', weights_only=False)
        model.net.load_state_dict({name: alex['features.' + name.split('.', 1)[1]]
                                   for name in model.net.state_dict()})
        self.lpips = model.to(args.device).eval().requires_grad_(False)
        self.with_band = with_band
        self.kernels = [torch.as_tensor(cv2.getGaussianKernel(8*sigma+1, sigma),
                                       device=args.device, dtype=torch.float32) for sigma in (1, 4)]

    def band(self, image):
        light = rgb_to_lab(image)[:, :1] / 100
        smooth = []
        for kernel in self.kernels:
            radius = kernel.numel() // 2
            value = F.conv2d(light, kernel.reshape(1, 1, 1, -1), padding=(0, radius))
            smooth.append(F.conv2d(value, kernel.reshape(1, 1, -1, 1), padding=(radius, 0)))
        return smooth[0] - smooth[1]

    def __call__(self, prediction, target, mask, measure_band=False):
        error = (prediction - target).abs().mean(1, keepdim=True)
        region = torch.from_numpy(roi.regions(mask, 8)).to(prediction.device)
        counts = region.sum((-1, -2))
        l1 = ((region * error).sum((-1, -2)) / counts.clamp_min(1)).sum() / (counts > 0).sum()
        lpips = self.lpips(prediction * 2 - 1, target * 2 - 1).mean()
        band = prediction.new_zeros(())
        if self.with_band or measure_band:
            interior = roi.regions(mask, 20)[0]
            if interior.sum() >= 256:
                keep = torch.from_numpy(interior).to(prediction.device)
                band = (self.band(prediction) - self.band(target)).abs()[0, 0][keep].mean()
        return l1 + LPIPS_WEIGHT * lpips + (BAND_WEIGHT * band if self.with_band else 0), (l1, lpips, band)


def render_pair(args, models, head, pair, cache_hashes):
    case = pair['case_id']
    path = args.run / 'prefix' / (case + '.pt')
    assert sha(path) == cache_hashes[case]
    state = torch.load(path, map_location='cpu', weights_only=True)
    assert state['case'] == case and state['contract'] == args.contract
    mask = case_mask(args, pair)
    target_path = args.data / pair['gt']
    assert sha(target_path) == args.gt_hashes[case]
    target = roi.tensors([np.asarray(rgb(target_path).resize(SIZE, Image.Resampling.BILINEAR))]).to(args.device)
    return decode(models[1], trajectory(args, models, head, pair, state, mask)), target, mask


def backward(objective, parameters):
    # FP16 冻结路径保留原前向，FP32 残差块梯度在裁剪前还原。
    scale = 128
    (objective * scale).backward()
    assert all(p.grad is not None for p in parameters.values())
    for param in parameters.values():
        param.grad.div_(scale)
    return float(torch.nn.utils.clip_grad_norm_(parameters.values(), 1, error_if_nonfinite=True))


def train(args):
    run = args.run / args.arm
    run.mkdir(exist_ok=True)
    done = run / 'training.json'
    if done.exists():
        prior = read(done)
        assert prior['contract'] == args.contract and prior['updates'] == STEPS
        assert all(sha(run / f'step{step}.pt') == value['sha256']
                   for step, value in prior['checkpoints'].items())
        return
    qualified = read(args.run / 'qualification.json')
    assert qualified['exact_latent'] and qualified['exact_pixels'] and qualified['contract'] == args.contract
    cache_hashes = {}
    for shard in range(args.shards):
        manifest = read(args.run / f'prefix_{shard}.json')
        assert manifest['contract'] == args.contract and not cache_hashes.keys() & manifest['cases'].keys()
        cache_hashes.update(manifest['cases'])
    trained = [p for p in args.trainval['pairs'] if p['split'] == 'train']
    valid = [p for p in args.trainval['pairs'] if p['split'] == 'val']
    assert len(trained) == STEPS and len(valid) == 32
    assert set(cache_hashes) == {p['case_id'] for p in (*trained, *valid)}
    assert metric_assets() == args.protocol['metric_assets']
    models = load_models(args)
    original, frozen = models[3:]
    head, parameters = candidates(original)
    optimizer = torch.optim.AdamW(parameters.values(), lr=LEARNING_RATE)
    loss_function = RealPairLoss(args, args.arm == 'B')
    trace, snapshots = [], {}
    order = np.random.default_rng(42).permutation(STEPS)
    for update, index in enumerate(order, start=1):
        pair = trained[index]
        optimizer.zero_grad(set_to_none=True)
        prediction, target, mask = render_pair(args, models, head, pair, cache_hashes)
        objective, values = loss_function(prediction, target, mask)
        norm = backward(objective, parameters)
        optimizer.step()
        models[2].adapter.clear()
        trace.append(dict(update=update, case=pair['case_id'], grad_norm=norm, loss=float(objective.detach()),
                          rgb_l1=float(values[0].detach()), lpips=float(values[1].detach()),
                          band=float(values[2].detach())))
        del prediction, target, objective, values
        print('TAIL6_TRAIN', args.arm, update, pair['case_id'], trace[-1]['loss'], norm, flush=True)
        if update not in (64, STEPS):
            continue
        with torch.no_grad():
            validation = []
            for p in valid:
                prediction, target, mask = render_pair(args, models, head, p, cache_hashes)
                _, terms = loss_function(prediction, target, mask, measure_band=True)
                validation.append(dict(case=p['case_id'], rgb_l1=float(terms[0]), lpips=float(terms[1]),
                                       band=float(terms[2])))
                models[2].adapter.clear()
        assert all(torch.equal(value, original.state_dict()[name]) for name, value in head.state_dict().items()
                   if name not in parameters)
        path = run / f'step{update}.pt'
        save_torch(path, {name: value.detach().cpu() for name, value in parameters.items()})
        snapshots[str(update)] = dict(sha256=sha(path), rows=validation,
            **{metric: float(np.mean([row[metric] for row in validation])) for metric in ('rgb_l1', 'lpips', 'band')})
        print('TAIL6_VALIDATION', args.arm, update, snapshots[str(update)]['lpips'], flush=True)
    assert tensor_set(frozen) == args.original['frozen_model_hashes']
    write(done, dict(arm=args.arm, updates=len(trace), steps=trace, checkpoints=snapshots,
                     trainable=list(parameters), count=PARAMETERS, frozen_unchanged=True, contract=args.contract))


def select(args):
    first, second = (read(args.run / arm / 'training.json') for arm in ARMS_AB)
    assert first['contract'] == second['contract'] == args.contract
    assert first['updates'] == second['updates'] == STEPS
    assert [x['case'] for x in first['steps']] == [x['case'] for x in second['steps']]
    validation = {step: {arm: row['checkpoints'][step]['lpips']
                         for arm, row in zip(ARMS_AB, (first, second))}
                  for step in ('64', '128')}
    best = min(validation, key=lambda step: (sum(validation[step].values()), -int(step)))
    write(args.run / 'selection.json', dict(update=int(best), validation_lpips=validation,
        checkpoints={arm: row['checkpoints'][best]['sha256'] for arm, row in zip(ARMS_AB, (first, second))},
        contract=args.contract))
    print('TAIL6_SELECTED', best, validation, flush=True)


@torch.no_grad()
def generate(args):
    args.out = args.baseline
    models = load_models(args)
    core, pipe, net, original, frozen = models
    head, parameters = candidates(original)
    selection = read(args.run / 'selection.json')
    assert selection['contract'] == args.contract
    path = args.run / args.arm / f"step{selection['update']}.pt"
    checkpoint_hash = selection['checkpoints'][args.arm]
    assert sha(path) == checkpoint_hash
    state = torch.load(path, map_location=args.device, weights_only=True)
    assert set(state) == set(parameters)
    for k, v in parameters.items():
        v.copy_(state[k])
    head.eval().requires_grad_(False)
    previous = read(args.baseline / 'per_image.json')
    ids = subset(args.protocol)
    if args.action == 'gen500':
        assert read(args.run / 'eval64' / 'result.json')['acceptance'][args.arm]['passed']
        ids = sorted(set(range(len(previous))) - set(ids))
    ids = ids[args.shard::args.shards]
    for index in ids:
        pair = args.protocol['pairs'][index]
        case = pair['case_id']
        assert case == previous[index]['case_id']
        directory = args.run / args.arm / 'test'
        directory.mkdir(exist_ok=True)
        png, metadata = (directory / (case + suffix) for suffix in ('.png', '.json'))
        if metadata.exists():
            saved = read(metadata)
            assert saved['checkpoint'] == checkpoint_hash and saved['png_sha256'] == sha(png)
            continue
        mask = case_mask(args, pair)
        tail = TailResidual(original, head, latent_gate(mask, args.device))
        picture, noise, tap = joint.sample(args, core, pipe, net, pair, CONDITION, b=tail, mode='raw', capture=True)
        assert tap.index == tail.index == LAST and len(tap.states) == 25 and picture.size == SIZE
        first = {name: tensor_sha(tap.states[0][name]) for name in ('zt', 'epsilon')}
        assert noise == previous[index]['arms']['off']['initial_noise_digest']
        assert first == previous[index]['arms']['off']['first_b_step']
        temp = png.with_suffix('.tmp')
        picture.save(temp, format='PNG')
        temp.replace(png)
        write(metadata, dict(case=case, index=index, steps=50, corrected_steps=list(range(FIRST, LAST)),
             checkpoint=checkpoint_hash, png_sha256=sha(png), initial_noise_digest=noise, first_b_step=first))
        print('TAIL6_GENERATED', args.action, args.arm, index, case, flush=True)
    assert tensor_set(frozen) == args.original['frozen_model_hashes']
    write(args.run / f'{args.action}_{args.arm}_{args.shard}.json',
          dict(ids=ids, checkpoint=checkpoint_hash, contract=args.contract, frozen_unchanged=True))


def appearance(image, target, interior):
    if interior.sum() < 256:
        return dict.fromkeys(('band_mae', 'band_rms', 'band_rms_error', 'l_bias', 'low_ab_error'))
    value = roi.lab(image)
    texture, truth = (roi.band(x[..., 0])[interior] for x in (value, target))
    return dict(band_mae=float(np.abs(texture-truth).mean()), band_rms=roi.rms(texture),
                band_rms_error=abs(roi.rms(texture)-roi.rms(truth)),
                l_bias=float((value[..., 0]-target[..., 0])[interior].mean()),
                low_ab_error=float(np.linalg.norm(roi.gaussian(value-target, 8)[interior, 1:], axis=1).mean()))


@torch.no_grad()
def score(args):
    few = args.action == 'score64'
    label = 'eval64' if few else 'eval500'
    out = args.run / label
    out.mkdir(exist_ok=True)
    if (out / 'tail6_review.zip').exists():
        assert sha(out / 'result.json') == read(out / 'review_manifest.json')['result.json']
        return
    qualified = read(args.run / 'qualification.json')
    selection = read(args.run / 'selection.json')
    assert qualified['contract'] == selection['contract'] == args.contract
    arms = ARMS_AB if few else tuple(
        k for k, v in read(args.run / 'eval64' / 'result.json')['acceptance'].items() if v['passed'])
    assert arms
    base = args.baseline
    blends, gates = base / 'edge_residual_acceptance_20261118', base / 'single_s0_gate_20261119'
    verified_files(base, roi.ARCHIVE_SHA, ('protocol.json', 'per_image.json', 'fid_pool3.npz'))
    verified_files(blends, ACCEPTANCE_SHA, ('per_pair.json', 'inner_mix_pool3.npz'))
    ids = subset(args.protocol) if few else list(range(len(args.protocol['pairs'])))
    controls = (*ARMS, 'blend', 'gate') if few else (*ARMS, 'blend')
    previous, mixed = read(base / 'per_image.json'), read(blends / 'per_pair.json')
    with np.load(base / 'fid_pool3.npz') as saved:
        features = {name: saved[name][ids] for name in (*ARMS, 'gt')}
    with np.load(blends / 'inner_mix_pool3.npz') as saved:
        features['blend'] = saved['edge_a025'][ids]
    if few:
        verified_files(gates, distill.GATE_ARCHIVE_SHA, ('per_pair.json', 'pool3.npz'))
        gate_rows = {row['index']: row for row in read(gates / 'per_pair.json')}
        assert list(gate_rows) == ids
        with np.load(gates / 'pool3.npz') as saved:
            features['gate'] = saved['gate']
    assert metric_assets() == args.protocol['metric_assets']
    lpips, inception = metric_models(args.device, args.protocol['metric_assets'])
    pending = {arm: [] for arm in arms}
    rows = []
    for offset in range(0, len(ids), 4):
        images, targets = {name: [] for name in arms}, []
        for index in ids[offset:offset + 4]:
            pair = args.protocol['pairs'][index]
            case = pair['case_id']
            assert case == previous[index]['case_id'] == mixed[index]['case_id']
            raw, mask = roi.checked_case(base, args.data, pair, previous[index])
            original = dict(**previous[index]['scores'], blend=mixed[index]['scores']['edge_a025'])
            raw['blend'] = mix_pixels(raw['off'], raw['on'], mask, {'blend': .25})['blend']
            interior = roi.regions(mask, 20)[0]
            target = roi.lab(raw['gt'])
            if few:
                assert gate_rows[index]['case_id'] == case
                original['gate'] = gate_rows[index]['scores']['gate']
                gate_path = gates / 'images' / (case + '.png')
                assert sha(gate_path) == gate_rows[index]['image_sha256']
                raw['gate'] = np.asarray(rgb(gate_path))
            pngs = {}
            for arm in arms:
                path = args.run / arm / 'test' / (case + '.png')
                record = read(path.with_suffix('.json'))
                assert record['index'] == index and record['case'] == case and record['steps'] == LAST
                assert record['checkpoint'] == selection['checkpoints'][arm]
                pngs[arm] = sha(path)
                assert record['png_sha256'] == pngs[arm]
                assert record['first_b_step'] == previous[index]['arms']['off']['first_b_step']
                assert record['initial_noise_digest'] == previous[index]['arms']['off']['initial_noise_digest']
                picture = rgb(path)
                raw[arm] = np.asarray(picture)
                images[arm].append(raw[arm])
                original[arm] = image_scores(picture, Image.fromarray(raw['gt']), path, args.data / pair['gt'], mask)
            targets.append(raw['gt'])
            rows.append(dict(index=index, case_id=case, category=pair['category'], scores=original,
                             pngs=pngs, interior_px=int(interior.sum()),
                             texture={name: appearance(raw[name], target, interior) for name in ('gt', *controls, *arms)}))
        target = roi.tensors(targets).to(args.device)
        for arm in arms:
            x = roi.tensors(images[arm]).to(args.device)
            for row, value in zip(rows[-len(targets):], lpips(x * 2 - 1, target * 2 - 1).flatten().tolist()):
                row['scores'][arm]['lpips'] = value
            pending[arm].append(inception(x)[0].flatten(1).cpu().numpy())
        if offset % 40 == 0:
            print('TAIL6_SCORED', label, len(rows), flush=True)
    features.update({arm: np.concatenate(pending[arm]) for arm in arms})
    assert all(x.shape == (len(ids), 2048) and np.isfinite(x).all() for x in features.values())
    result = summarize(rows, features, (*controls, *arms), controls, ('blend', *arms))
    for group, report in result['groups'].items():
        selected = [row for row in rows if row['interior_px'] >= 256 and
                    (group == 'all' or row['category'] == 'outwear')]
        assert selected
        report['texture_count'] = len(selected)
        report['texture'] = {name: {key: float(np.mean([row['texture'][name][key] for row in selected]))
            for key in selected[0]['texture'][name]} for name in ('gt', *controls, *arms)}
    result['acceptance'] = {}
    all_arms = result['groups']['all']['arms']
    texture = result['groups']['all']['texture']
    for arm in arms:
        seven = distill.acceptance(dict(groups=dict(all=dict(arms={**all_arms, 'student': all_arms[arm]}))))
        texture_pass = texture[arm]['band_mae'] < min(texture[name]['band_mae'] for name in ARMS)
        result['acceptance'][arm] = dict(seven=seven, texture_pass=texture_pass,
                                        passed=seven['passed'] and texture_pass)
    result.update(samples=len(ids), selection=selection, qualification=qualified,
                  reused_candidate_pairs=0 if few else len(subset(args.protocol)), contract=args.contract)
    write(out / 'result.json', result)
    write(out / 'per_pair.json', rows)
    np.savez_compressed(out / 'candidate_pool3.npz', **{k: features[k] for k in arms})
    files = {p.name: p for p in (out/'result.json', out/'per_pair.json', out/'candidate_pool3.npz',
                                  args.run/'selection.json', args.run/'qualification.json', args.run/'protocol.json')}
    files.update({f'training_{arm}.json': args.run/arm/'training.json' for arm in ARMS_AB})
    write(out / 'review_manifest.json', {name: sha(path) for name, path in files.items()})
    temp = out / 'tail6_review.tmp'
    with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            archive.write(path, name)
        archive.write(out / 'review_manifest.json', 'review_manifest.json')
    temp.replace(out / 'tail6_review.zip')
    print('TAIL6_RESULT', label, {name: {k: row[k]['fid' if k == 'fid' else 'mean']
          for k in ('fid', *METRICS)} for name, row in all_arms.items()}, flush=True)
    print('TAIL6_ACCEPTED', label, result['acceptance'], sha(out / 'tail6_review.zip'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = dict(prepare=None, qualify=qualify, cache=cache, train=train, select=select, gen64=generate, gen500=generate,
                   score64=score, score500=score)
    parser.add_argument('action', choices=actions)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS_AB)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=2)
    args = parser.parse_args()
    args.project = args.prior.parents[1]
    args.data = args.project.parent / 'datasets/BF'
    args.checkpoint = args.project / 'output_eval/e2_state_b_pastel_followup_20261115/trust_retry/training/step1.pt'
    args.baseline = args.prior / distill.BASE
    args.out = args.baseline / distill.RUN
    args.run = args.baseline / RUN
    args.run.mkdir(exist_ok=True)
    args.protocol = verify(argparse.Namespace(out=args.baseline))
    args.original = read(args.baseline / 'result.json')
    args.trainval = distill.experiment_protocol(args)
    roots = [{Path(p['gt']).stem for p in group}
             for group in ([p for p in args.trainval['pairs'] if p['split'] == 'train'],
                           [p for p in args.trainval['pairs'] if p['split'] == 'val'],
                           args.protocol['pairs'])]
    assert [len(group) for group in roots] == [128, 32, 500]
    assert len(set.union(*roots)) == sum(map(len, roots))
    protocol_path = args.run / 'protocol.json'
    gt_hashes = (read(protocol_path)['gt_sha256'] if protocol_path.exists() else
                 {p['case_id']: sha(args.data / p['gt']) for p in args.trainval['pairs']})
    args.gt_hashes = gt_hashes
    args.contract = dict(code=sha(Path(__file__)), source_sha256=sha(args.out/'protocol.json'),
        original_protocol_sha256=sha(args.baseline/'protocol.json'), old_B_sha256=CHECKPOINT_SHA,
        seed=42, tail_indices=list(range(FIRST, LAST)), updates=STEPS, batch=1, lr=LEARNING_RATE,
        lpips_weight=LPIPS_WEIGHT, band_weight_B=BAND_WEIGHT, gt_sha256=gt_hashes)
    if protocol_path.exists():
        assert read(protocol_path) == args.contract
    else:
        write(protocol_path, args.contract)
    cv2.setNumThreads(1)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '2')))
    if args.action != 'prepare':
        actions[args.action](args)


if __name__ == '__main__':
    main()
