"""冻结 f68：S0 正常末步，S1 仅关闭 index49 的 B；不创建优化器。"""
import argparse
import copy
import hashlib
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

from models.e2_state_texture_b import StateTextureB
from tools import e2_state_b_balanced as balanced
from tools import e2_state_b_pilot as pilot
from tools.e2_fft_experiment import read, write, sha, environment
from tools.e2_oracle_texture import weights_digest
from tools.e2_b_last_step_metrics import score_pixels, summarize


CHECKPOINT_SHA = 'f68f22a106c3f0a08044c754b44c738ba3d6f67274edf5d281d5b54f39ddcd96'


def prepare(args):
    spec = read(args.source_run / 'protocol.json')
    baseline = read(args.baseline_run / 'trust_result.json')
    assert sha(args.checkpoint) == baseline['checkpoint_sha256'] == CHECKPOINT_SHA
    bank = spec['replay_bank']
    assert len(bank) == 16 and len(spec['train']) == 4
    assert len({(r['case'], r['condition']) for r in bank}) == 16
    assert not set(spec['protected_product_roots']) & {p['identity_group'] for p in spec['train']}
    assert spec['sampler'] == dict(steps=50, cfg=7., sketch_scale=.6, active=[25, 50])
    args.out.mkdir(parents=True, exist_ok=True)
    assets = {}
    for p in spec['train']:
        case = p['case_id']
        src, dst = args.source_run / 'cases' / case, args.out / 'cases' / case
        dst.mkdir(parents=True, exist_ok=True)
        names = ['mask.png', 'train_anchor.png']
        for r in bank:
            if r['case'] == case:
                names += [r['condition'] + suffix for suffix in ('_ref.png', '_goal.png', '_positive.png')]
        for name in sorted(set(names)):
            origin, target = src / name, dst / name
            if not target.exists():
                shutil.copyfile(origin, target)
            assert sha(origin) == sha(target)
            assets[target.relative_to(args.out).as_posix()] = sha(target)
        sketch = args.data / p['sketch']
        assert sha(sketch) == p['hashes']['sketch']
        shutil.copyfile(sketch, dst / 'sketch.jpg')
        assets[(dst / 'sketch.jpg').relative_to(args.out).as_posix()] = sha(sketch)
        for r in bank:
            if r['case'] == case:
                assert assets[f"cases/{case}/{r['condition']}_ref.png"] == p['reference_hashes'][r['condition']]
    provenance = dict(checkpoint_sha256=CHECKPOINT_SHA,
        source_protocol_sha256=sha(args.source_run / 'protocol.json'),
        baseline_report_sha256=sha(args.baseline_run / 'trust_result.json'),
        train=spec['train'], replay_bank=bank, sampler=spec['sampler'], assets=assets,
        heldout_used=False, training_updates=0, postprocessing=False,
        intervention='S1 shares S0 states through the input of index49; only final B delta_x0 is zero',
        full_S1_parity='one fixed neutral condition per identity; compare all25 captured states and final PNG/latent',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    write(args.out / 'protocol.json', provenance)
    print('LAST_STEP_PREPARED', len(bank), CHECKPOINT_SHA, flush=True)


def final_sample(args, core, pipe, net, pair, condition, b):
    original, record = pipe.scheduler.step, {}

    def capture(epsilon, timestep, z, *a, **kw):
        result = original(epsilon, timestep, z, *a, **kw)
        record.update(latent=(result[0] if isinstance(result, tuple) else result.prev_sample).detach().clone(),
                      timestep=timestep.detach().clone(), kwargs=kw)
        return result

    pipe.scheduler.step = capture
    try:
        picture, digest, tap = balanced.sample(args, core, pipe, net, pair, condition, b, capture=True)
    finally:
        pipe.scheduler.step = original
    return picture, digest, tap, record


def decode_pipeline(pipe, z):
    # 与真实 pipeline 的最终解码和取整完全一致，不额外加入颜色/亮度后处理。
    rgb = pipe.vae.decode(z / pipe.vae.config.scaling_factor, return_dict=False)[0]
    rgb = (rgb / 2 + .5).clamp(0, 1)
    return pipe.numpy_to_pil(rgb.cpu().permute(0, 2, 3, 1).float().numpy())[0]


def state_digest(states):
    digest = hashlib.sha256()
    for state in states:
        for key in ('zt', 'epsilon'):
            digest.update(state[key].contiguous().numpy().tobytes())
    return digest.hexdigest()


def pixels(path):
    return np.asarray(Image.open(path).convert('RGB'))


@torch.no_grad()
def generate(args):
    protocol = read(args.out / 'protocol.json')
    assert sha(args.checkpoint) == protocol['checkpoint_sha256'] == CHECKPOINT_SHA
    runtime = copy.copy(args)
    runtime.action = 'train'  # 只选择历史确定性推理后端；本文件没有训练调用。
    core, pipe, net = pilot.load_core(runtime)
    assert pipe.scheduler.__class__.__name__ == 'DDIMScheduler'
    b = StateTextureB(color_channels=2, reference_channels=192).to(args.device)
    b.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=False)['b'])
    b.eval().requires_grad_(False)
    net.eval().requires_grad_(False)
    pipe.vae.half()
    b_hash, net_hash = weights_digest(b), weights_digest(net)
    assert net_hash == read(args.source_run / 'training/result.json')['frozen_hash']
    pairs = {p['case_id']: p for p in protocol['train']}
    selected = {p['case_id'] for i, p in enumerate(protocol['train']) if i % args.shards == args.shard}
    rows, checked = [], set()
    started = time.monotonic()
    for item in protocol['replay_bank']:
        case, condition = item['case'], item['condition']
        if case not in selected:
            continue
        pair = pairs[case]
        picture0, digest, tap, record = final_sample(args, core, pipe, net, pair, condition, b)
        assert tap.index == 50 and len(tap.states) == 25
        last = tap.states[-1]
        assert last['index'] == 49 and last['t'] == int(pipe.scheduler.timesteps[-1])
        assert record['kwargs'].get('eta', 0.) == 0.
        z1 = pipe.scheduler.step(last['epsilon'].to(args.device), record['timestep'],
                                 last['zt'].to(args.device), **record['kwargs'])
        z1 = z1[0] if isinstance(z1, tuple) else z1.prev_sample
        assert z1.dtype == record['latent'].dtype == torch.float16
        picture1 = decode_pipeline(pipe, z1)
        assert np.array_equal(np.asarray(picture0), np.asarray(decode_pipeline(pipe, record['latent'])))
        name = case + '_' + condition
        audit = dict(shared_prefix_sha256=state_digest(tap.states), shared_last_input=True,
                     final_index=49, final_timestep=last['t'], noise_digest=digest)
        if case not in checked:
            def last_only(module, positional, keywords, delta):
                return torch.zeros_like(delta) if int(keywords['timestep']) == last['t'] else delta

            with b.register_forward_hook(last_only, with_kwargs=True):
                explicit, explicit_digest, explicit_tap, explicit_record = final_sample(
                    args, core, pipe, net, pair, condition, b)
            assert explicit_digest == digest
            assert all(torch.equal(a[k], c[k]) for a, c in zip(tap.states, explicit_tap.states)
                       for k in ('zt', 'epsilon'))
            error = float((z1.float() - explicit_record['latent'].float()).abs().max())
            assert error <= 2e-4 and np.array_equal(np.asarray(explicit), np.asarray(picture1))
            audit.update(explicit_full_S1_verified=True, latent_max_error=error, prefix_bitwise=True,
                         full_S1_png_bitwise=True)
            checked.add(case)
        folder = args.out / 'cases' / case
        positive, goal, anchor = [pixels(folder / f) for f in
                                  (condition + '_positive.png', condition + '_goal.png', 'train_anchor.png')]
        mask = np.asarray(Image.open(folder / 'mask.png')) > 0
        row = dict(item, seed=pair['seed'], audit=audit, png_sha256={})
        for arm, picture in (('S0', picture0), ('S1', picture1)):
            dest = args.out / 'images' / arm / (name + '.png')
            dest.parent.mkdir(parents=True, exist_ok=True)
            picture.save(dest)
            row[arm] = score_pixels(np.asarray(picture), positive, goal, anchor, mask, pair['patch_coordinates'])
            row['png_sha256'][arm] = sha(dest)
        delta = np.abs(np.asarray(picture1, np.float32) - np.asarray(picture0, np.float32)) / 255
        distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        row['S1_minus_S0_pixels'] = dict(outside_mae=float(delta[~mask].mean()),
            boundary_band_mae=float(delta[mask & (distance < 16)].mean()),
            interior_mae=float(delta[distance >= 16].mean()))
        prior = args.baseline_run / 'training/probe1' / (name + '.png')
        if prior.exists():
            err = np.abs(pixels(prior).astype(np.int16) - np.asarray(picture0, np.int16))
            row['historical_S0'] = dict(sha256=sha(prior), max_pixel_error=int(err.max()),
                                       mean_pixel_error=float(err.mean()))
        dest = args.out / 'latents' / (name + '.pt')
        dest.parent.mkdir(exist_ok=True)
        torch.save(dict(S0=record['latent'].cpu(), S1=z1.cpu(), final_input=last), dest)
        rows.append(row)
        write(args.out / f'shard{args.shard}_partial.json', dict(rows=rows))
        print('LAST_STEP_PAIR', name, flush=True)
    assert len(checked) == len(selected)
    assert weights_digest(b) == b_hash and weights_digest(net) == net_hash
    assert sha(args.checkpoint) == CHECKPOINT_SHA
    write(args.out / f'shard{args.shard}.json', dict(rows=rows, shard=args.shard, shards=args.shards,
        frozen_unchanged=True, b_hash=b_hash, net_hash=net_hash, checkpoint_sha256=CHECKPOINT_SHA,
        training_updates=0, heldout_used=False, seconds=time.monotonic()-started, environment=environment()))
    print('LAST_STEP_SHARD_COMPLETE', args.shard, len(rows), flush=True)


def finish(args):
    protocol = read(args.out / 'protocol.json')
    shards = [read(args.out / f'shard{i}.json') for i in range(args.shards)]
    rows = [r for s in shards for r in s['rows']]
    assert {(r['case'], r['condition']) for r in rows} == {
        (r['case'], r['condition']) for r in protocol['replay_bank']}
    assert len(rows) == 16 and all(s['frozen_unchanged'] and s['training_updates'] == 0 for s in shards)
    assert len({s['b_hash'] for s in shards}) == len({s['net_hash'] for s in shards}) == 1
    assert sum(bool(r['audit'].get('explicit_full_S1_verified')) for r in rows) == 4
    write(args.out / 'result.json', dict(rows=rows, summary=summarize(rows), training_updates=0,
        heldout_used=False, frozen_unchanged=True, checkpoint_sha256=CHECKPOINT_SHA,
        full_S1_parity_cases=4, interpretation='train4 fixed16 diagnostics; no tuning or checkpoint selection'))
    for case in sorted({r['case'] for r in rows}):
        selected = [r for r in rows if r['case'] == case]
        board = Image.new('RGB', (4*192, len(selected)*280), 'white')
        draw = ImageDraw.Draw(board)
        for j, row in enumerate(selected):
            c, name = row['condition'], case+'_'+row['condition']
            files = [args.out/'cases'/case/(c+'_ref.png'), args.out/'cases'/case/(c+'_positive.png')]
            files += [args.out/'images'/a/(name+'.png') for a in ('S0','S1')]
            for i, (label, path) in enumerate(zip(('reference','VAE target','S0','S1'), files)):
                board.paste(Image.open(path).convert('RGB').resize((192,256)), (192*i,280*j+24))
                draw.text((192*i+3,280*j+3), c+' '+label, fill='black')
        board.save(args.out/(case+'_comparison.png'))
    files = sorted(p for p in args.out.rglob('*') if p.is_file()
                   and 'source' not in p.relative_to(args.out).parts and p.suffix not in ('.zip','.tmp','.log','.err')
                   and p.name != 'manifest.json')
    manifest = {p.relative_to(args.out).as_posix():sha(p) for p in files}
    write(args.out/'manifest.json', manifest)
    archive = args.out/'s0_s1_review.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in files+[args.out/'manifest.json']:
            z.write(p, p.relative_to(args.out).as_posix())
    print('LAST_STEP_COMPLETE', len(rows), sha(archive), archive.stat().st_size, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare','generate','finish'))
    parser.add_argument('--project', type=Path, default=Path('/share/home/u2515283058/Mymodel'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--baseline-run', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    args = parser.parse_args()
    args.data = args.project.parent/'datasets/BF'
    args.previous = args.project/'output_eval/e2_state_b_material_20261113'
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    {'prepare':prepare, 'generate':generate, 'finish':finish}[args.action](args)


if __name__ == '__main__':
    main()
