"""在 E1-BFM-T3 第十轮底座上复现原 B 的末六步真实配对 A/B。"""
import argparse
import copy
import inspect
import os
import zipfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
from PIL import Image

from tools import e2_f68_tail6_real_ab as tail
from tools.e2_fft_experiment import configure, setup
from tools.m1_evaluation import prepare_inference


RUN = 'tail6_real_ab_e1_epoch10_20261119'
BASE_CHECKPOINT = 'output_eval/e1_bfm_continue_20261117/epochs/epoch_10/training/E1-BFM-T3-epoch10/final.pt'
CONTROL64 = ('off', 'on', 'gate')
ARM_SET = ('off', 'on', 'blend', 'gate', *tail.ARMS_AB)


def prepare(args):
    complete = tail.read(args.base_checkpoint.with_name('complete.json'))
    assert (complete['epoch'], complete['total_updates']) == (10, 56730)
    assert tail.sha(args.base_checkpoint) == complete['checkpoint_sha256']
    assert tail.sha(args.checkpoint) == tail.CHECKPOINT_SHA
    tail.verified_files(args.baseline, tail.roi.ARCHIVE_SHA, ('protocol.json','per_image.json','fid_pool3.npz'))
    training = args.base_checkpoint.parents[4] / 'training_manifest.json'
    base_protocol = tail.read(training.with_name('protocol.json'))
    train_sha = tail.sha(training)
    assert train_sha == base_protocol['train_sha256']
    core = configure(args)
    assert tail.sha(core.E1) == base_protocol['e1_sha256'] == core.E1_SHA
    training_roots = {Path(row['cloth']).stem.split('_')[0] for row in tail.read(training)}
    test_roots = {Path(row['gt']).stem.split('_')[0] for row in args.protocol['pairs']}
    tv = args.trainval['pairs']
    train_roots, val_roots = [
        {Path(row['gt']).stem.split('_')[0] for row in tv if row['split'] == split}
        for split in ('train', 'val')
    ]
    assert len(test_roots) == 500 and not (test_roots & (training_roots | val_roots | train_roots))
    assert len(val_roots) == 32 and len(train_roots) == 128 and not (train_roots & val_roots)
    file = args.base_checkpoint.stat()
    source = dict(code=tail.sha(Path(__file__)), reused_training_code=tail.sha(Path(tail.__file__)),
        training_protocol=tail.sha(args.out/'protocol.json'), evaluation_protocol=tail.sha(args.baseline/'protocol.json'),
        base_path=str(args.base_checkpoint), base_sha256=complete['checkpoint_sha256'],
        base_bytes=file.st_size, base_mtime_ns=file.st_mtime_ns, base_train_manifest=train_sha,
        e1_reference_source_sha256=core.E1_SHA,
        base_train_roots=len(training_roots), base_val_overlap=len(training_roots & val_roots),
        base_test_overlap=len(training_roots & test_roots), old_B_sha256=tail.CHECKPOINT_SHA,
        shape=list(tail.SIZE), seed=42, tail_indices=list(range(tail.FIRST, tail.LAST)),
        updates=tail.STEPS, batch=1, lr=tail.LEARNING_RATE,
        lpips_weight=tail.LPIPS_WEIGHT, band_weight_B=tail.BAND_WEIGHT,
        gt_sha256={row['case_id']:tail.sha(args.data/row['gt']) for row in tv})
    tail.write(args.run/'protocol.json', source)
    print('E1_TAIL_PREPARED', source['base_sha256'], '500 test roots disjoint', flush=True)


def load_backbone(args):
    context = copy.copy(args)
    context.action, context.out = 'train', args.run
    core, _ = setup(context)
    file = args.base_checkpoint.stat()
    assert (file.st_size, file.st_mtime_ns) == (args.contract['base_bytes'], args.contract['base_mtime_ns'])
    assert tail.sha(args.checkpoint) == tail.CHECKPOINT_SHA
    pipe, net = core.load_network(args.device)
    state = torch.load(args.base_checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert state['total_updates'] == 56730
    # 这份联合训练权重同时包含生成 U-Net 和条件分支。
    net.load_state_dict(state['network'], strict=True)
    del state
    prepare_inference(args, pipe, net)
    pipe.vae.eval().requires_grad_(False)
    assert type(pipe.scheduler).__name__ == 'DDIMScheduler'
    original = tail.StateTextureB(color_channels=2, reference_channels=192).to(args.device)
    original.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=False)['b'])
    original.eval().requires_grad_(False)
    models = dict(B=original, net=net, vae=pipe.vae)
    frozen = tail.tensor_set(models)
    old = args.m0_result
    assert frozen['B'] == old['frozen_model_hashes']['B'] and frozen['vae'] == old['frozen_model_hashes']['vae']
    assert frozen['net'] != old['frozen_model_hashes']['net']
    scheduler = tail.sha(Path(inspect.getsourcefile(type(pipe.scheduler))))
    assert scheduler == old['scheduler_source_sha256']
    audit = dict(frozen_model_hashes=frozen, scheduler_source_sha256=scheduler,
                 base_sha256=args.contract['base_sha256'])
    marker = args.run/'model.json'
    if marker.exists():
        assert tail.read(marker) == audit
    else:
        assert args.action == 'qualify'
        tail.write(marker, audit)
    args.original = audit
    return core, pipe, net, original, models


def eligible(args):
    return tuple(name for name, v in tail.read(args.run/'eval64/result.json')['acceptance'].items() if v['passed'])


def pair_indices(args, few):
    small = tail.subset(args.protocol)
    return small if few else [i for i in range(len(args.protocol['pairs'])) if i not in small]


def checkpoint_head(args, original, arm, selection):
    candidate, parameters = tail.candidates(original)
    path = args.run/arm/f"step{selection['update']}.pt"
    assert tail.sha(path) == selection['checkpoints'][arm]
    state = torch.load(path, map_location=args.device, weights_only=True)
    assert set(state) == set(parameters)
    for name, param in parameters.items():
        param.copy_(state[name])
    return candidate.eval().requires_grad_(False)


@torch.no_grad()
def generate(args):
    few = args.action == 'generate64'
    arms = (*CONTROL64, *tail.ARMS_AB) if few else ('off', 'on', *eligible(args))
    assert few or len(arms) > 2
    selection = tail.read(args.run/'selection.json')
    assert selection['contract'] == args.contract
    core, pipe, net, original, frozen = load_backbone(args)
    heads = {arm: checkpoint_head(args, original, arm, selection) for arm in arms if arm in tail.ARMS_AB}
    args.out = args.baseline
    root = args.run/'test'
    old_images = tail.read(args.baseline/'per_image.json')
    for name in arms:
        (root/name).mkdir(parents=True, exist_ok=True)
    ids = pair_indices(args, few)[args.shard::args.shards]
    count = 0
    for index in ids:
        pair = args.protocol['pairs'][index]
        case = pair['case_id']
        assert old_images[index]['case_id'] == case
        mask = tail.case_mask(args, pair)
        gate = tail.latent_gate(mask, args.device)
        reference = None
        for arm in arms:
            png = root/arm/(case+'.png')
            meta = png.with_suffix('.json')
            checkpoint = selection['checkpoints'].get(arm, tail.CHECKPOINT_SHA if arm != 'off' else None)
            if meta.exists():
                record = tail.read(meta)
                assert record['png_sha256'] == tail.sha(png)
                assert (record['index'], record['case'], record['arm']) == (index, case, arm)
                assert record['checkpoint'] == checkpoint and record['base_sha256'] == args.contract['base_sha256']
            else:
                module = None if arm == 'off' else original if arm == 'on' else tail.TailResidual(original, heads.get(arm), gate)
                picture, noise, tap = tail.joint.sample(args, core, pipe, net, pair, tail.CONDITION,
                                                       b=module, mode='raw', capture=True)
                assert picture.size == tail.SIZE and tap.index == tail.LAST and len(tap.states) == 25
                before = tap.states[0]
                first = {name:tail.tensor_sha(before[name]) for name in ('zt','epsilon')}
                temp = png.with_suffix('.tmp')
                picture.save(temp, format='PNG')
                temp.replace(png)
                record = dict(index=index, case=case, arm=arm, png_sha256=tail.sha(png),
                              base_sha256=args.contract['base_sha256'], checkpoint=checkpoint,
                              initial_noise_digest=noise, first_b_step=first, steps=tail.LAST)
                tail.write(meta, record)
            current = record['initial_noise_digest'], record['first_b_step']
            assert record['steps'] == tail.LAST and (reference is None or current == reference)
            assert current[0] == old_images[index]['arms']['off']['initial_noise_digest']
            reference = current
        count += 1
        print('E1_TAIL_GENERATED', args.action, args.shard, count, len(ids), case, flush=True)
    assert tail.tensor_set(frozen) == args.original['frozen_model_hashes']
    tail.write(args.run/f'{args.action}_{args.shard}.json', dict(ids=ids, arms=arms, frozen_unchanged=True))


def gate_check(result, arm):
    scores = result['groups']['all']['arms']
    gate = tail.distill.acceptance(dict(groups=dict(all=dict(arms={**scores, 'student':scores[arm]}))))
    texture = result['groups']['all']['texture']
    texture_pass = texture[arm]['band_mae'] < min(texture[name]['band_mae'] for name in tail.ARMS)
    better_lpips = scores[arm]['lpips']['mean'] < scores['off']['lpips']['mean']
    return dict(seven=gate, texture_pass=texture_pass, lpips_better_than_off=better_lpips,
                passed=gate['passed'] and texture_pass and better_lpips)


@torch.no_grad()
def score(args):
    few = args.action == 'score64'
    mode = 'eval64' if few else 'eval500'
    out = args.run/mode
    out.mkdir(exist_ok=True)
    candidate_arms = tail.ARMS_AB if few else eligible(args)
    assert candidate_arms
    arms = tuple(arm for arm in ARM_SET if (few or arm != 'gate') and
                 (arm not in tail.ARMS_AB or arm in candidate_arms))
    ids = pair_indices(args, few)
    markers = [tail.read(args.run/f'generate{64 if few else 500}_{shard}.json') for shard in range(args.shards)]
    assert all(marker['ids']==ids[shard::args.shards] and marker['frozen_unchanged'] and
               set(marker['arms'])==set(arms)-{'blend'} for shard,marker in enumerate(markers))
    if (out/'review.zip').exists():
        assert tail.sha(out/'result.json') == tail.read(out/'review_manifest.json')['result.json']
        return
    assert tail.metric_assets() == args.protocol['metric_assets']
    lpips, inception = tail.metric_models(args.device, args.protocol['metric_assets'])
    rows, features = {}, {arm:{} for arm in arms}
    if not few:
        previous = args.run/'eval64'
        for row in tail.read(previous/'per_pair.json'):
            rows[row['index']] = row
        with np.load(previous/'pool3.npz', allow_pickle=False) as saved:
            for arm in arms:
                features[arm].update(zip(tail.subset(args.protocol), saved[arm]))
    root = args.run/'test'
    (root/'blend').mkdir(exist_ok=True)
    args.out = args.baseline
    for offset in range(0, len(ids), 4):
        batch = ids[offset:offset+4]
        pictures, targets = {arm:[] for arm in arms}, []
        for index in batch:
            pair = args.protocol['pairs'][index]
            case = pair['case_id']
            mask = tail.case_mask(args, pair)
            paths = {arm:root/arm/(case+'.png') for arm in arms}
            target = args.data/pair['gt']
            assert tail.sha(target) == pair['input_sha256']['gt']
            gt = tail.rgb(target).resize(tail.SIZE, Image.Resampling.BILINEAR)
            targets.append(np.asarray(gt))
            raw, prior = {}, None
            for arm in (a for a in arms if a != 'blend'):
                path = paths[arm]
                observation = tail.read(path.with_suffix('.json'))
                assert (observation['index'], observation['case'], observation['arm']) == (index,case,arm)
                assert observation['base_sha256'] == args.contract['base_sha256']
                assert observation['png_sha256'] == tail.sha(path)
                pair_state = observation['initial_noise_digest'], observation['first_b_step']
                assert prior is None or pair_state == prior
                prior = pair_state
                raw[arm] = np.asarray(tail.rgb(path))
            raw['blend'] = tail.mix_pixels(raw['off'], raw['on'], mask, {'blend':.25})['blend']
            Image.fromarray(raw['blend']).save(paths['blend'])
            interior = tail.roi.regions(mask, 20)[0]
            gt_lab = tail.roi.lab(np.asarray(gt))
            rows[index] = dict(index=index, case_id=case, category=pair['category'],
                pngs={arm:tail.sha(paths[arm]) for arm in arms}, interior_px=int(interior.sum()),
                scores={arm:tail.image_scores(Image.fromarray(raw[arm]), gt, paths[arm],target,mask) for arm in arms},
                texture={arm:tail.appearance(im,gt_lab,interior) for arm,im in (*raw.items(),('gt',np.asarray(gt)))})
            for arm in arms:
                pictures[arm].append(raw[arm])
        gt = tail.roi.tensors(targets).to(args.device)
        for arm in arms:
            x = tail.roi.tensors(pictures[arm]).to(args.device)
            for index, value in zip(batch, lpips(x*2-1,gt*2-1).flatten().tolist()):
                rows[index]['scores'][arm]['lpips'] = value
            features[arm].update(zip(batch, inception(x)[0].flatten(1).cpu().numpy()))
        if offset % 40 == 0:
            print('E1_TAIL_SCORED', mode, offset+len(batch), len(ids), flush=True)
    full = tail.subset(args.protocol) if few else list(range(len(args.protocol['pairs'])))
    ordered = [rows[i] for i in full]
    arrays = {arm:np.stack([features[arm][i] for i in full]) for arm in arms}
    with np.load(args.baseline/'fid_pool3.npz', allow_pickle=False) as old:
        arrays['gt'] = old['gt'][full]
    assert all(arr.shape==(len(full),2048) and np.isfinite(arr).all() for arr in arrays.values())
    result = tail.summarize(ordered, arrays, arms, ('off','on','blend'), ('blend',*candidate_arms))
    for label, group in result['groups'].items():
        selected = [row for row in ordered if row['interior_px']>=256 and
                    (label=='all' or row['category']=='outwear')]
        assert selected
        group['texture_count'] = len(selected)
        group['texture'] = {arm:{key:float(np.mean([r['texture'][arm][key] for r in selected]))
                            for key in selected[0]['texture'][arm]} for arm in (*arms,'gt')}
    result.update(samples=len(full), reused_pairs=0 if few else len(full)-len(ids),
                  acceptance={arm:gate_check(result,arm) for arm in candidate_arms})
    tail.write(out/'result.json', result)
    tail.write(out/'per_pair.json', ordered)
    np.savez_compressed(out/'pool3.npz', **arrays)
    files = {path.name:path for path in (out/'result.json',out/'per_pair.json',out/'pool3.npz',
             *[args.run/f'{name}.json' for name in ('protocol','model','qualification','selection')])}
    files.update({f'training_{arm}.json':args.run/arm/'training.json' for arm in tail.ARMS_AB})
    manifest = out/'review_manifest.json'
    tail.write(manifest, {name:tail.sha(path) for name,path in files.items()})
    temporary = out/'review.tmp'
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            archive.write(path, name)
        archive.write(manifest, manifest.name)
    temporary.replace(out/'review.zip')
    print('E1_TAIL_ACCEPTED', mode, result['acceptance'], tail.sha(out/'review.zip'), flush=True)


def main():
    actions = dict(prepare=prepare, qualify=tail.qualify, cache=tail.cache, train=tail.train,
                   select=tail.select, generate64=generate, generate500=generate, score64=score, score500=score)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=actions)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--arm', choices=tail.ARMS_AB)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=2)
    args = parser.parse_args()
    args.project = args.prior.parents[1]
    args.data = args.project.parent/'datasets/BF'
    args.checkpoint = args.project/'output_eval/e2_state_b_pastel_followup_20261115/trust_retry/training/step1.pt'
    args.base_checkpoint = args.project/BASE_CHECKPOINT
    args.baseline = args.prior/tail.distill.BASE
    args.out = args.baseline/tail.distill.RUN
    args.run = args.baseline/RUN
    args.run.mkdir(exist_ok=True)
    args.protocol = tail.verify(argparse.Namespace(out=args.baseline))
    args.trainval = tail.distill.experiment_protocol(args)
    args.m0_result = tail.read(args.baseline/'result.json')
    cv2.setNumThreads(1)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','2')))
    if args.action == 'prepare':
        prepare(args)
        return
    args.contract = tail.read(args.run/'protocol.json')
    assert args.contract['code'] == tail.sha(Path(__file__))
    assert args.contract['reused_training_code'] == tail.sha(Path(tail.__file__))
    args.gt_hashes = args.contract['gt_sha256']
    # 仅此入口注入新底座，原训练/缓存/资格函数与历史实现保持一致。
    with patch.object(tail, 'load_models', load_backbone):
        actions[args.action](args)


if __name__ == '__main__':
    main()
