"""固定400更新的状态相关B小试；只训练B，分开评价4个训练/8个留出身份。"""
import argparse
import copy
import json
import shutil
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.signal import find_peaks
import torch
from torch.nn import functional as F

from eval.eval_utils import prepare_evaluation_masks, estimate_foreground_mask
from models.e2_compat import load_trainable_strict
from models.e2_state_texture_b import StateTextureB
from models.e2_texture_diagnostics import reference_control
from tools.bfm_skip_probe import PROJECT, RUN, PREVIOUS, PAIRS_SHA, CHECKPOINT_SHA, projected
from tools.e2_fft_experiment import read, write, sha, setup, environment
from tools.e2_oracle_texture import SIZE, ACTIVE, RGB_LUMA, oracle_image, encode, decode, image, weights_digest
from tools.e2_oracle_color import calibrate_color, color_metrics
from tools.m1_evaluation import prepare_inference
from tools.m1_experiment import load_m
from tools.e2_a_evaluation import boundary_f1

VARIANTS = ('v16', 'h16', 'v32', 'h32')
CONDITIONS = tuple(v + suffix for suffix in ('', '_shift') for v in VARIANTS)
EXCLUDED = (7814, 5735, 7871, 6550, 4022, 974, 8365, 6722)
TRAIN = (621, 1641, 870, 1526)
HOLDOUT = (2063, 6372, 3775, 5038, 1011, 2779, 1917, 8504)
COLOR_BASELINE = PROJECT / 'output_eval/e2_state_b_pilot_20261112'


def directory(args, pair):
    return args.out / 'cases' / pair['case_id']


def prepare(args):
    if args.paired_color:
        prepare_paired_color(args)
        return
    assert sha(RUN / 'strict_pairs.json') == PAIRS_SHA
    pairs = read(RUN / 'strict_pairs.json')['pairs']
    chosen = [copy.deepcopy(next(p for p in pairs if p['index'] == i)) for i in TRAIN + HOLDOUT]
    assert len({p['identity_group'] for p in chosen}) == 12
    assert not set(EXCLUDED) & set(TRAIN + HOLDOUT)
    for p in chosen:
        p['split'] = 'train' if p['index'] in TRAIN else 'holdout'
        p['seed'] += 300000
        d = directory(args, p)
        d.mkdir(parents=True, exist_ok=True)
        for key in ('sketch', 'gt', 'original_reference'):
            assert sha(args.data / p[key]) == p['hashes'][key]
        masks = prepare_evaluation_masks(SIZE, sketch_path=str(args.data / p['sketch']), mask_policy='sketch_only')
        assert masks['garment'].any() and len(p['patch_coordinates']) == 8
        Image.fromarray(masks['garment'].astype(np.uint8) * 255).save(d / 'mask.png')
        Image.open(args.data / p['sketch']).save(d / 'sketch.png')
        rgb = np.asarray(Image.open(args.data / p['original_reference']).convert('RGB').resize(SIZE, Image.Resampling.BICUBIC))
        assert (rgb @ RGB_LUMA).std() / 255 >= .02
        hist = np.sort(np.ascontiguousarray(rgb).view('V3'), axis=None)
        for c in CONDITIONS:
            ref = reference_control(rgb, c, p['seed'])
            assert np.array_equal(hist, np.sort(np.ascontiguousarray(ref).view('V3'), axis=None))
            Image.fromarray(ref).save(d / (c + '_ref.png'))
        p['reference_hashes'] = {c: sha(d / (c + '_ref.png')) for c in CONDITIONS}
    write(args.out / 'protocol.json', dict(pairs=chosen, excluded_old_cases=list(EXCLUDED),
        source=environment(), checkpoint_sha256=CHECKPOINT_SHA, manifest_sha256=PAIRS_SHA,
        selection='input-only fixed IDs; train first2 remaining stripes/first dot/first check; holdout first6 remaining stripes/next2 dots',
        scope='new to B and previous two oracle rounds; drawn from previously inspected validation32; not a new independent test population',
        architecture='zt4+x0_CFG4+PixelUnshuffle8(reference_luma)64+mask1+structure1; width64, 2 residual blocks, time32, zero output',
        intervention='after CFG before scheduler.step; delta_x0=mask*B; epsilon -= sqrt(alpha/(1-alpha))*delta_x0',
        active_steps=list(ACTIVE), sampler=dict(steps=50, cfg=7., sketch_scale=.6),
        reference='BFM and B read same condition; no GT as generation or training condition',
        training=dict(updates=400, seed=20261112, optimizer='AdamW', lr=1e-4, weight_decay=.01,
            effective_batch=4, micro_batch=1, gradient_clip=1., first200='M0 cached states',
            last200='equal M0 and one fixed step200 B rollout cache; same optimizer retained',
            loss_weights=dict(latent=1., gray=2., low_lab=1., protection=2.)),
        supervision='M0 v16 phase0 anchor, garment-interior lowpass+reference stripes, fixed20 Lab calibration; no original GT interior loss',
        evaluation=dict(seeds_per_identity=2, conditions=list(CONDITIONS), final_checkpoint=400,
            metrics='paired FFT + tensor orientation + ACF period; old-template projection excluding target harmonics; mean Lab to calibrated target and VAE floor; outside MAE',
            decision='exploratory joint improvement on heldout identities, train-fit diagnostic, no checkpoint selection or assumed absolute Lab cutoff')))
    print('PREPARED_SPLIT', [(p['case_id'], p['split']) for p in chosen], flush=True)


def prepare_paired_color(args):
    baseline = args.color_baseline.resolve()
    assert args.out.resolve() != baseline
    protocol = read(baseline / 'protocol.json')
    complete = read(baseline / 'training/complete.json')
    assert complete['updates'] == 400 and complete['frozen_unchanged']
    assert sha(baseline / 'training/step400.pt') == complete['checkpoint_sha256']
    assert tuple(p['index'] for p in protocol['pairs']) == TRAIN + HOLDOUT
    assert protocol['checkpoint_sha256'] == CHECKPOINT_SHA
    assert tuple(protocol['evaluation']['conditions']) == CONDITIONS
    for p in protocol['pairs']:
        assert sha(args.data / p['sketch']) == p['hashes']['sketch']
        source = baseline / 'cases' / p['case_id']
        dest = directory(args, p)
        dest.mkdir(parents=True, exist_ok=True)
        files = ['mask.png', 'sketch.png', 'anchor_s0.png', 'anchor_s1.png']
        files += [c + '_ref.png' for c in CONDITIONS]
        files += [f'{c}_s{seed}_{arm}.png' for c in CONDITIONS for seed in range(2)
                  for arm in ('m0', 'target', 'vae')]
        if p['split'] == 'train':
            files += ['train_anchor.png'] + [c + '_train_target.png' for c in CONDITIONS]
        for name in files:
            shutil.copyfile(source / name, dest / name)
            assert sha(source / name) == sha(dest / name)
        assert all(sha(dest / (c + '_ref.png')) == p['reference_hashes'][c] for c in CONDITIONS)
    protocol['source'] = environment()
    protocol['paired_color'] = dict(
        baseline=str(baseline), baseline_checkpoint_sha256=complete['checkpoint_sha256'],
        baseline_protocol_sha256=sha(baseline / 'protocol.json'),
        m0_cache_sha256=sha(baseline / 'cache_m0.pt'),
        loss='old L* loss / 3 + 2 * common-anchor ab MAE + same-colour pair ab MAE',
        anchor='original per-identity train_anchor.png (M0 v16 phase0); same anchor for both predictions',
        chroma='Lab ab after 32x32 garment-interior weighted RGB pooling; mean absolute error in Lab/100',
        pair='same identity, seed, phase, and timestep; alternate orientation and period swaps',
        budget='400 optimizer updates, original primary stream and batch of four; extra decoded predictions only for colour loss',
        controls='reuse exact M0 cache, train targets, and evaluation M0/target/VAE images; regenerate B step200 cache',
        evaluation='compare step400 with original B step400; global and local ab error, pair ab consistency, original texture and protection endpoints')
    write(args.out / 'protocol.json', protocol)
    print('PAIRED_COLOR_PREPARED', len(protocol['pairs']), flush=True)


def load_core(args):
    core, _ = setup(args)
    checkpoint = PREVIOUS / 'training/M0/latest.pt'
    assert sha(checkpoint) == CHECKPOINT_SHA
    pipe, net, _ = load_m(args, core, 'M0')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    load_trainable_strict(net, payload['network'])
    del payload
    prepare_inference(args, pipe, net)
    # Diffusers通过组件推断执行设备，未使用的辅助组件也需与主干同设备。
    return core, pipe, net


def inputs(args, pair, condition, device):
    d = directory(args, pair)
    ref = np.asarray(Image.open(d / (condition + '_ref.png')), dtype=np.float32) @ RGB_LUMA / 255
    mask = np.asarray(Image.open(d / 'mask.png')) > 0
    weight = np.clip((cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5) - 8) / 8, 0, 1)
    sketch = np.asarray(Image.open(args.data / pair['sketch']).convert('RGB').resize(SIZE), dtype=np.float32) @ RGB_LUMA / 255
    tensor = lambda a: torch.from_numpy(a.copy()).float()[None, None].to(device)
    return dict(reference=F.pixel_unshuffle(tensor(ref) * 2 - 1, 8),
                mask=F.avg_pool2d(tensor(weight), 8), structure=F.avg_pool2d(tensor(1 - sketch), 8)), weight


class SamplerB:
    def __init__(self, scheduler, module, condition, capture=False):
        self.scheduler, self.module, self.condition = scheduler, module, condition
        self.capture, self.states, self.index = capture, [], 0

    def __enter__(self):
        self.original = self.scheduler.step
        self.scheduler.step = self.step
        return self

    def __exit__(self, *unused):
        self.scheduler.step = self.original

    def step(self, model_output, timestep, sample, *args, **kwargs):
        if self.index in ACTIVE:
            a = self.scheduler.alphas_cumprod[int(timestep)].to(device=sample.device, dtype=torch.float32)
            x0 = (sample.float() - (1 - a).sqrt() * model_output.float()) / a.sqrt()
            if self.capture:
                self.states.append(dict(zt=sample.detach().cpu().half(), x0=x0.detach().cpu().half(), t=int(timestep)))
            if self.module is not None:
                delta = self.module(sample.float(), x0, timestep=timestep.reshape(1), **self.condition)
                model_output = (model_output.float() - (a / (1 - a)).sqrt() * delta).to(model_output.dtype)
        self.index += 1
        return self.original(model_output, timestep, sample, *args, **kwargs)


def sample(args, core, pipe, net, pair, condition, module=None, capture=False):
    d = directory(args, pair)
    assert sha(d / (condition + '_ref.png')) == pair['reference_hashes'][condition]
    features = projected(net, core.tensor(args.data / pair['sketch']).to(args.device).half(),
                         core.tensor(d / (condition + '_ref.png')).to(args.device).half())
    values, _ = inputs(args, pair, condition, args.device)
    with SamplerB(pipe.scheduler, module, values, capture) as hook:
        picture, digest = image(args, core, pipe, net, pair, features)
    assert hook.index == 50 and (not capture or len(hook.states) == 20)
    return picture, digest, hook.states


def target_images(args, pair, condition, anchor, suffix):
    d = directory(args, pair)
    mask = np.asarray(Image.open(d / 'mask.png')) > 0
    target, weight = oracle_image(anchor, Image.open(d / (condition + '_ref.png')).convert('RGB'), mask)
    target = calibrate_color(target, anchor, mask, weight)
    assert color_metrics(target, anchor, mask)['lab_shift'] <= .25
    target.save(d / (condition + suffix + '_target.png'))
    return target


def cache_states(args, core, pipe, net, pairs, module, stage, frozen_targets=None):
    entries = []
    with torch.no_grad():
        for p in pairs:
            d = directory(args, p)
            if stage == 'm0':
                anchor, _, _ = sample(args, core, pipe, net, p, 'v16')
                anchor.save(d / 'train_anchor.png')
            if frozen_targets is None:
                anchor = Image.open(d / 'train_anchor.png').convert('RGB')
            for c in CONDITIONS:
                picture, digest, states = sample(args, core, pipe, net, p, c, module, True)
                picture.save(d / (c + '_cache_' + stage + '.png'))
                if frozen_targets is None:
                    target = target_images(args, p, c, anchor, '_train')
                    z = encode(pipe.vae, target, args.device).cpu()
                else:
                    z = frozen_targets[p['case_id'], c]
                entries.append(dict(pair=p, condition=c, target=z, states=states, initial_latent=digest))
            print('CACHE', stage, p['case_id'], flush=True)
    torch.save(entries, args.out / ('cache_' + stage + '.pt'))
    return entries


def rgb_tensor(picture, device):
    return torch.from_numpy(np.asarray(picture, dtype=np.float32).transpose(2, 0, 1).copy())[None].to(device) / 255


def lab_tensor(rgb):
    rgb = rgb.clamp(0, 1)
    linear = torch.where(rgb > .04045, ((rgb + .055) / 1.055).pow(2.4), rgb / 12.92)
    matrix = rgb.new_tensor([[.4124564, .3575761, .1804375], [.2126729, .7151522, .0721750], [.0193339, .1191920, .9503041]])
    xyz = torch.einsum('ij,bjhw->bihw', matrix, linear) / rgb.new_tensor([.95047, 1., 1.08883])[None, :, None, None]
    f = torch.where(xyz > .008856, xyz.clamp_min(.008856).pow(1 / 3), 7.787 * xyz + 16 / 116)
    return torch.cat((116 * f[:, 1:2] - 16, 500 * (f[:, :1] - f[:, 1:2]), 200 * (f[:, 1:2] - f[:, 2:])), 1) / 100


def weighted_mean(value, mask):
    return (value * mask).sum() / (mask.sum() * value.shape[1]).clamp_min(1)


def paired_condition(condition, swap_direction):
    if swap_direction:
        return ('h' if condition[0] == 'v' else 'v') + condition[1:]
    period = '32' if condition[1:3] == '16' else '16'
    return condition[0] + period + condition[3:]


def pooled_chroma(rgb, weight, pooled_weight):
    mean_rgb = F.avg_pool2d(rgb * weight, 32) / pooled_weight.clamp_min(1e-6)
    return lab_tensor(mean_rgb)[:, 1:]


def color_supervision(rgb, companion_rgb, goal, weight, anchor_ab, pooled_weight):
    # L* 沿用旧监督的贡献，两个参考共用底色；一致性项在两个方向上传递梯度。
    delta_l = lab_tensor(F.avg_pool2d(rgb, 16))[:, :1] - lab_tensor(F.avg_pool2d(goal, 16))[:, :1]
    luminance = weighted_mean(delta_l.abs(), F.avg_pool2d(weight, 16))
    current = pooled_chroma(rgb, weight, pooled_weight)
    companion = pooled_chroma(companion_rgb, weight, pooled_weight)
    anchor = .5 * (weighted_mean((current - anchor_ab).abs(), pooled_weight)
                   + weighted_mean((companion - anchor_ab).abs(), pooled_weight))
    consistency = weighted_mean((current - companion).abs(), pooled_weight)
    return luminance / 3 + 2 * anchor + consistency, anchor, consistency


def training(args):
    protocol = read(args.out / 'protocol.json')
    assert protocol['source']['commit'] == environment()['commit']
    core, pipe, net = load_core(args)
    frozen_before = weights_digest(net)
    torch.manual_seed(20261112)
    b = StateTextureB().to(args.device)
    optimizer = torch.optim.AdamW(b.parameters(), lr=1e-4, weight_decay=.01)
    color = protocol.get('paired_color')
    train_pairs = [p for p in protocol['pairs'] if p['split'] == 'train']
    with torch.no_grad():
        plain, digest, _ = sample(args, core, pipe, net, train_pairs[0], 'h32')
        zero, digest2, _ = sample(args, core, pipe, net, train_pairs[0], 'h32', b)
    assert digest == digest2 and np.array_equal(np.asarray(plain), np.asarray(zero))
    targets = None
    if color:
        prior = Path(color['baseline'])
        assert sha(prior / 'cache_m0.pt') == color['m0_cache_sha256']
        cache = torch.load(prior / 'cache_m0.pt', map_location='cpu', weights_only=False)
        assert [e['pair'] for e in cache] == [p for p in train_pairs for _ in CONDITIONS]
        assert [e['condition'] for e in cache] == list(CONDITIONS) * len(train_pairs)
        assert frozen_before == read(prior / 'training/complete.json')['frozen_hash']
        initial = torch.load(prior / 'training/step0.pt', map_location='cpu', weights_only=False)['b']
        assert all(torch.equal(p.cpu(), initial[k]) for k, p in b.state_dict().items())
        targets = {(e['pair']['case_id'], e['condition']): e['target'] for e in cache}
        print('REUSED_EXACT_M0_CACHE_AND_B_INITIALIZATION', flush=True)
    else:
        cache = cache_states(args, core, pipe, net, train_pairs, None, 'm0')
    checkpoint_dir = args.out / 'training'
    checkpoint_dir.mkdir(exist_ok=True)
    def save(step):
        torch.save(dict(step=step, b=b.state_dict(), optimizer=optimizer.state_dict()), checkpoint_dir / ('step%d.pt' % step))
    save(0)
    rng = np.random.default_rng(20261112)
    records = []
    prepared = {}
    anchors = {}
    def tensors(entry):
        key = (entry['pair']['case_id'], entry['condition'])
        if key not in prepared:
            values, weight = inputs(args, entry['pair'], entry['condition'], args.device)
            d = directory(args, entry['pair'])
            goal = rgb_tensor(Image.open(d / (entry['condition'] + '_train_target.png')), args.device)
            prepared[key] = values, torch.from_numpy(weight)[None, None].to(args.device), goal
        return prepared[key]
    def predict(state, values):
        zt, x0 = state['zt'].to(args.device).float(), state['x0'].to(args.device).float()
        t = torch.tensor([state['t']], device=args.device)
        return x0 + b(zt, x0, timestep=t, **values), x0

    def decode_rgb(z):
        return (pipe.vae.decode(z / pipe.vae.config.scaling_factor).sample.float() / 2 + .5).clamp(0, 1)

    audit = dict()
    if color:
        baseline_log = [json.loads(line) for line in (prior / 'training/loss.jsonl').read_text().splitlines()]
        assert [row['step'] for row in baseline_log] == list(range(1, 401))
        audit = dict(exact_initial_b=True, exact_m0_cache=True, exact_primary_stream=True)
    pipe.vae.float()
    for step in range(1, 401):
        if step == 201:
            pipe.vae.half()
            b.eval()
            cache += cache_states(args, core, pipe, net, train_pairs, b, 'b200', targets)
            pipe.vae.float()
        b.train()
        optimizer.zero_grad(set_to_none=True)
        loss_values = dict.fromkeys(('latent', 'gray', 'low_lab', 'protection'), 0.)
        color_values = dict.fromkeys(('anchor_ab', 'pair_ab'), 0.)
        tokens = []
        color_tokens = []
        start = time.monotonic()
        # 每次更新覆盖四个身份；后200步两个M0、两个B200状态，轮换身份。
        for micro, p in enumerate(train_pairs):
            stage = int(step > 200 and (micro + step) % 2 == 0)
            choices = [e for e in cache[stage * 32:(stage + 1) * 32] if e['pair']['case_id'] == p['case_id']]
            entry = choices[int(rng.integers(len(choices)))]
            state_index = int(rng.integers(20))
            state = entry['states'][state_index]
            values, weight, goal = tensors(entry)
            prediction, x0 = predict(state, values)
            zgoal = entry['target'].to(args.device)
            latent = weighted_mean((prediction - zgoal).square(), values['mask'])
            rgb = decode_rgb(prediction)
            with torch.no_grad():
                base = decode_rgb(x0)
                representable = decode_rgb(zgoal)
            luma = rgb.new_tensor(RGB_LUMA)[None, :, None, None]
            gray, target_gray = (rgb * luma).sum(1, keepdim=True), (representable * luma).sum(1, keepdim=True)
            highpass = lambda x: x - F.avg_pool2d(x, 17, 1, 8)
            gray_loss = weighted_mean((highpass(gray) - highpass(target_gray)).abs(), weight)
            if color:
                companion = paired_condition(entry['condition'], (step + micro) % 2 == 0)
                other = next(e for e in choices if e['condition'] == companion)
                other_state = other['states'][state_index]
                assert other['initial_latent'] == entry['initial_latent'] and other_state['t'] == state['t']
                other_values, _, _ = tensors(other)
                other_rgb = decode_rgb(predict(other_state, other_values)[0])
                case = p['case_id']
                if case not in anchors:
                    anchor = rgb_tensor(Image.open(directory(args, p) / 'train_anchor.png'), args.device)
                    pooled_weight = F.avg_pool2d(weight, 32)
                    anchors[case] = pooled_chroma(anchor, weight, pooled_weight), pooled_weight
                low_lab, anchor_loss, pair_loss = color_supervision(rgb, other_rgb, goal, weight, *anchors[case])
                color_values['anchor_ab'] += float(anchor_loss.detach()) / 4
                color_values['pair_ab'] += float(pair_loss.detach()) / 4
                color_tokens.append(companion)
            else:
                low_lab = weighted_mean((lab_tensor(F.avg_pool2d(rgb, 16)) - lab_tensor(F.avg_pool2d(goal, 16))).abs(), F.avg_pool2d(weight, 16))
            protection = weighted_mean((rgb - base).abs(), 1 - weight)
            losses = dict(latent=latent, gray=gray_loss, low_lab=low_lab, protection=protection)
            total = sum(protocol['training']['loss_weights'][k] * v for k, v in losses.items())
            assert torch.isfinite(total)
            (total / 4).backward()
            for k, v in losses.items():
                loss_values[k] += float(v.detach()) / 4
            tokens.append([p['case_id'], entry['condition'], state['t'], stage])
        norm = torch.nn.utils.clip_grad_norm_(b.parameters(), 1.)
        assert torch.isfinite(norm)
        optimizer.step()
        row = dict(step=step, losses=loss_values, grad_norm=float(norm), tokens=tokens,
                   seconds=time.monotonic() - start, peak_gib=torch.cuda.max_memory_allocated() / 2**30)
        if color:
            assert tokens == baseline_log[step - 1]['tokens']
            row.update(color_losses=color_values, color_pairs=color_tokens)
        records.append(row)
        with (checkpoint_dir / 'loss.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        if step % 20 == 0:
            print('UPDATE', step, loss_values, flush=True)
        if step in (200, 400):
            save(step)
    assert frozen_before == weights_digest(net)
    write(checkpoint_dir / 'complete.json', dict(updates=400, zero_init_bitwise=True, frozen_unchanged=True,
        frozen_hash=frozen_before, parameters=sum(p.numel() for p in b.parameters()),
        **audit,
        checkpoint_sha256=sha(checkpoint_dir / 'step400.pt'), finite_updates=len(records),
        first20={k: float(np.mean([r['losses'][k] for r in records[:20]])) for k in loss_values},
        last20={k: float(np.mean([r['losses'][k] for r in records[-20:]])) for k in loss_values}))
    print('TRAINING_COMPLETE_400', flush=True)


def generate(args):
    protocol = read(args.out / 'protocol.json')
    complete = read(args.out / 'training/complete.json')
    assert complete['updates'] == 400 and sha(args.out / 'training/step400.pt') == complete['checkpoint_sha256']
    core, pipe, net = load_core(args)
    before = weights_digest(net)
    b = StateTextureB().to(args.device)
    b.load_state_dict(torch.load(args.out / 'training/step400.pt', map_location=args.device, weights_only=False)['b'])
    b.eval()
    color = protocol.get('paired_color')
    with torch.no_grad():
        for p in protocol['pairs'][args.shard::2]:
            d = directory(args, p)
            audit = []
            if color:
                source = Path(color['baseline']) / 'cases' / p['case_id']
                original = read(source / 'generation.json')
                assert original['paired_noise']
                expected = {(r['seed'], r['condition']): r['digest'] for r in original['rows']}
            for seed_index in range(2):
                pair = dict(p, seed=p['seed'] + seed_index * 100000)
                suffix = '_s%d' % seed_index
                if not color:
                    anchor, anchor_digest, _ = sample(args, core, pipe, net, pair, 'v16')
                    anchor.save(d / ('anchor' + suffix + '.png'))
                for c in CONDITIONS:
                    if color:
                        digest = expected[pair['seed'], c]
                        anchor_digest = expected[pair['seed'], 'v16']
                    else:
                        target = target_images(args, pair, c, anchor, suffix)
                        z = encode(pipe.vae, target, args.device)
                        decode(pipe.vae, z).save(d / (c + suffix + '_vae.png'))
                        m0, digest, _ = sample(args, core, pipe, net, pair, c)
                        m0.save(d / (c + suffix + '_m0.png'))
                    current, bdigest, _ = sample(args, core, pipe, net, pair, c, b)
                    assert digest == bdigest == anchor_digest
                    current.save(d / (c + suffix + '_b400.png'))
                    audit.append(dict(condition=c, seed=pair['seed'], digest=digest))
                print('GENERATED', p['case_id'], seed_index, flush=True)
            write(d / 'generation.json', dict(paired_noise=True, rows=audit))
    assert before == weights_digest(net)
    write(args.out / ('shard%d.json' % args.shard), dict(frozen_unchanged=True, checkpoint_sha256=complete['checkpoint_sha256']))


def texture_metrics(picture, coords, wanted):
    values = np.asarray(picture, dtype=np.float64) @ RGB_LUMA / 255
    bands = {v: [] for v in VARIANTS}
    angles, periods = [], []
    for x, y in coords:
        gray = values[y:y+64, x:x+64]
        window = np.outer(np.hanning(64), np.hanning(64))
        power = abs(np.fft.fft2((gray - gray.mean()) * window)) ** 2
        power[0, 0] = 0
        for v in VARIANTS:
            k = 64 // int(v[1:])
            iy, ix = (0, k) if v[0] == 'v' else (k, 0)
            bands[v].append(float((power[iy, ix] + power[-iy, -ix]) / max(power.sum(), 1e-12)))
        dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)[1:-1, 1:-1]
        dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)[1:-1, 1:-1]
        xx, yy, xy = (dx * dx).mean(), (dy * dy).mean(), (dx * dy).mean()
        coherence = np.hypot(xx - yy, 2 * xy) / max(xx + yy, 1e-12)
        angle = abs(np.degrees(.5 * np.arctan2(2 * xy, xx - yy)))
        direction = (angle <= 20 if wanted[0] == 'v' else angle >= 70) and coherence >= .35 and gray.std() >= 2 / 255
        angles.append(bool(direction))
        if not direction:
            continue
        profile = gray.mean(axis=0 if wanted[0] == 'v' else 1)
        axis = np.arange(64)
        # 64像素窗只有两个宽条周期，线性去趋势会删掉32像素锯齿的真实峰。
        profile -= profile.mean()
        corr = [1.]
        for lag in range(1, 42):
            a, b = profile[:-lag], profile[lag:]
            corr.append(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
        peaks, info = find_peaks(corr, height=.25, prominence=.20)
        picks = [(int(i), float(h)) for i, h in zip(peaks, info['peak_heights']) if 8 <= i <= 40]
        if picks:
            periods.append(next(i for i, h in picks if h >= .8 * max(v for _, v in picks)))
    bands = {k: float(np.mean(v)) for k, v in bands.items()}
    minimum = (len(coords) + 1) // 2
    period = float(np.median(periods)) if len(periods) >= minimum else None
    fft = bands[wanted] >= .075 and all(bands[wanted] > bands[v] for v in VARIANTS if v != wanted)
    direction = sum(angles) >= minimum
    period_hit = period is not None and abs(period - int(wanted[1:])) <= .25 * int(wanted[1:])
    return dict(bands=bands, fft=bool(fft), direction=direction, period=period,
                period_hit=period_hit, hit=bool(fft and direction and period_hit))


def old_pattern(picture, anchor, positive, coords, wanted):
    def gray(im):
        v = np.asarray(im, dtype=np.float32) @ RGB_LUMA / 255
        return v - cv2.GaussianBlur(v, (0, 0), 8)
    current, original, goal = map(gray, (picture, anchor, positive))
    ratios, errors = [], []
    period = int(wanted[1:])
    def remove_target(patch):
        # 删除目标周期及其全部谐波；32像素目标中的16像素谐波不计为旧纹样。
        profile = patch.mean(axis=0 if wanted[0] == 'v' else 1)
        phase = np.arange(64) % period
        average = np.array([profile[phase == i].mean() for i in range(period)])[phase]
        return patch - (average[None, :] if wanted[0] == 'v' else average[:, None])
    for x, y in coords:
        sl = np.s_[y:y+64, x:x+64]
        template = remove_target(original[sl] - goal[sl])
        error = remove_target(current[sl] - goal[sl])
        energy = np.mean(template ** 2)
        if energy >= (.5 / 255) ** 2:
            ratios.append(float(max(0, np.mean(error * template) / energy)))
        errors.append(float(np.abs(current[sl] - goal[sl]).mean()))
    return dict(old_projection=float(np.mean(ratios)) if ratios else None,
                valid_templates=len(ratios), gray_highpass_mae=float(np.mean(errors)))


def panel(args, pair, prior=None):
    d = directory(args, pair)
    if prior is not None:
        canvas = Image.new('RGB', (4 * 154, 4 * 226), 'white')
        draw = ImageDraw.Draw(canvas)
        for row, c in enumerate(VARIANTS):
            old = prior / 'cases' / pair['case_id']
            entries = [('M0', d / (c + '_s0_m0.png')), ('target', d / (c + '_s0_target.png')),
                       ('original B', old / (c + '_s0_b400.png')), ('paired-color B', d / (c + '_s0_b400.png'))]
            for col, (label, path) in enumerate(entries):
                x, y = col * 154, row * 226
                draw.text((x + 3, y + 3), c + ' ' + label, fill='black')
                canvas.paste(Image.open(path).convert('RGB').resize((150, 200)), (x, y + 22))
        canvas.save(d / 'panel.jpg', quality=94)
        return
    tiles = [('sketch', d / 'sketch.png'), ('anchor', d / 'anchor_s0.png')]
    for c in VARIANTS:
        tiles += [(c + ' target', d / (c + '_s0_target.png')), (c + ' VAE', d / (c + '_s0_vae.png')),
                  (c + ' M0', d / (c + '_s0_m0.png')), (c + ' B400', d / (c + '_s0_b400.png'))]
    canvas = Image.new('RGB', (6 * 154, 3 * 230), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (label, path) in enumerate(tiles):
        x, y = i % 6 * 154, i // 6 * 230
        draw.text((x + 3, y + 3), label, fill='black')
        canvas.paste(Image.open(path).convert('RGB').resize((150, 200)), (x, y + 24))
    canvas.save(d / 'panel.jpg', quality=94)


def summarize(args):
    from tools.bf_t3_three_arm_report import paired
    protocol = read(args.out / 'protocol.json')
    complete = read(args.out / 'training/complete.json')
    color = protocol.get('paired_color')
    prior = Path(color['baseline']) if color else None
    assert complete['updates'] == 400 and complete['frozen_unchanged']
    for shard in range(2):
        assert read(args.out / ('shard%d.json' % shard))['frozen_unchanged']
    identities = []
    for p in protocol['pairs']:
        d = directory(args, p)
        rows, switches = {}, {}
        mask = np.asarray(Image.open(d / 'mask.png')) > 0
        original = read(prior / 'cases' / p['case_id'] / 'result.json') if color else None
        for seed in range(2):
            anchor = Image.open(d / ('anchor_s%d.png' % seed)).convert('RGB')
            for c in CONDITIONS:
                suffix = c + '_s%d' % seed
                goal = Image.open(d / (suffix + '_target.png')).convert('RGB')
                positive = Image.open(d / (suffix + '_vae.png')).convert('RGB')
                wanted = c.replace('_shift', '')
                record = {}
                m0_pixels = np.asarray(Image.open(d / (suffix + '_m0.png')).convert('RGB'), dtype=np.float32) / 255
                goal_rgb = np.asarray(goal, dtype=np.float32) / 255
                goal_lab = cv2.cvtColor(cv2.GaussianBlur(goal_rgb, (0, 0), 12), cv2.COLOR_RGB2LAB)
                boundary = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) != cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8))
                for arm in ('m0', 'b400', 'vae', 'target'):
                    if original and arm != 'b400':
                        record[arm] = original['rows'][suffix][arm]
                        continue
                    pic = Image.open(d / (suffix + '_' + arm + '.png')).convert('RGB')
                    rgb, base = np.asarray(pic, dtype=np.float32) / 255, np.asarray(anchor, dtype=np.float32) / 255
                    pred_mask = estimate_foreground_mask(pic, SIZE)
                    local_lab = cv2.cvtColor(cv2.GaussianBlur(rgb, (0, 0), 12), cv2.COLOR_RGB2LAB)
                    record[arm] = dict(texture=texture_metrics(pic, p['patch_coordinates'], wanted),
                        color_to_target=color_metrics(pic, goal, mask)['lab_shift'],
                        color_to_anchor=color_metrics(pic, anchor, mask)['lab_shift'],
                        lowfreq_lab_error=float(np.linalg.norm(local_lab[mask] - goal_lab[mask], axis=1).mean()),
                        outside_mae=float(np.abs(rgb[~mask] - base[~mask]).mean()),
                        outside_change_from_m0=float(np.abs(rgb[~mask] - m0_pixels[~mask]).mean()),
                        boundary_change_from_m0=float(np.abs(rgb[boundary] - m0_pixels[boundary]).mean()),
                        silhouette_iou=float((pred_mask & mask).sum() / (pred_mask | mask).sum()),
                        boundary_f1=boundary_f1(pred_mask, mask, 2),
                        **old_pattern(pic, anchor, positive, p['patch_coordinates'], wanted))
                rows[suffix] = record
            for phase in ('', '_shift'):
                for kind, pairs in [('direction', [('v16', 'h16'), ('v32', 'h32')]),
                                    ('period', [('v16', 'v32'), ('h16', 'h32')])]:
                    for first, second in pairs:
                        a, b = (rows[v + phase + '_s%d' % seed] for v in (first, second))
                        switches['%s_%s_%s_s%d' % (kind, first, phase, seed)] = {
                            arm: dict(strict=bool(a[arm]['texture']['hit'] and b[arm]['texture']['hit']),
                                      fft=bool(a[arm]['texture']['fft'] and b[arm]['texture']['fft']))
                            for arm in ('m0', 'b400', 'vae', 'target')}
        agg = {}
        for arm in ('m0', 'b400', 'vae', 'target'):
            values = [r[arm] for r in rows.values()]
            agg[arm] = {k: float(np.mean([r[k] for r in values if r[k] is not None])) for k in (
                'color_to_target', 'color_to_anchor', 'lowfreq_lab_error', 'outside_mae',
                'outside_change_from_m0', 'boundary_change_from_m0', 'silhouette_iou', 'boundary_f1',
                'old_projection', 'gray_highpass_mae')}
            for kind in ('direction', 'period'):
                for endpoint in ('strict', 'fft'):
                    agg[arm][kind + '_' + endpoint] = float(np.mean([r[arm][endpoint] for k, r in switches.items() if k.startswith(kind)]))
            agg[arm]['valid_templates'] = sum(r['valid_templates'] for r in values)
        result = dict(case_id=p['case_id'], split=p['split'], rows=rows, switches=switches, aggregate=agg)
        write(d / 'result.json', result)
        identities.append(result)
        panel(args, p, prior)
    summary = {}
    for split in ('train', 'holdout'):
        selected = [p for p in identities if p['split'] == split]
        keys = [k for k in selected[0]['aggregate']['m0'] if k != 'valid_templates']
        summary[split] = dict(n=len(selected), arms={arm: {k: float(np.mean([p['aggregate'][arm][k] for p in selected])) for k in keys}
            for arm in ('m0', 'b400', 'vae', 'target')}, paired_b_minus_m0={
                k: paired([p['aggregate']['b400'][k] - p['aggregate']['m0'][k] for p in selected]) for k in keys})
    comparison = {}
    if color:
        from tools.e2_state_b_color_evaluation import compare
        comparison['paired_color_vs_original_b'] = compare(args.out, prior, identities)
    write(args.out / 'summary.json', dict(training=complete, results=summary, evaluation_source=environment(),
        **comparison,
        period_preflight='before viewing generated results: ideal32 sawtooth failed linear detrending; fixed ACF centering to mean-only, all4 ideal controls pass',
        interpretation='identity-clustered paired descriptive bootstrap; exploratory, multiple endpoints not corrected; all fixed400 outputs included'))
    files = [args.out / 'protocol.json', args.out / 'summary.json', *args.out.glob('shard*.json'),
             *list((args.out / 'training').glob('*.json*')), args.out / 'training/step400.pt']
    cases = args.out / 'cases'
    if color:
        for pattern in ('*_s*_b400.png', 'panel.jpg', 'result.json', 'generation.json', 'paired_color_comparison.json'):
            files += list(cases.glob('*/' + pattern))
    else:
        files += list(cases.glob('*/*'))
    archive = args.out / ('b400_paired_color_review.zip' if color else 'b400_review.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(files):
            if path.is_file():
                z.write(path, path.relative_to(args.out))
    print('SUMMARY', summary, flush=True)
    print('ARCHIVE', archive.stat().st_size, sha(archive), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'training', 'generate', 'summarize'))
    parser.add_argument('--project', type=Path, default=PROJECT)
    parser.add_argument('--data', type=Path, default=PROJECT.parent / 'datasets/BF')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--paired-color', action='store_true', help='准备仅更换颜色监督的复现实验')
    parser.add_argument('--color-baseline', type=Path, default=COLOR_BASELINE)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    globals()[args.action](args)


if __name__ == '__main__':
    main()
