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
