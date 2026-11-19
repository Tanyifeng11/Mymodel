"""从第 11 轮 RGB 残差头做固定预算的衣内感知续训对照。"""
import argparse
import os
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from tools import e2_f68_rgb_distill as distill
from tools.e2_f68_edge_residual import verified_files
from tools.e2_f68_real_paired import METRICS, SEED, SIZE, metric_assets, metric_models, rgb, write
from tools.e2_fft_experiment import environment, read, sha

RUN = 'ab_inside_lpips_20261119'
ARMS = {'A_l1': 0., 'B_inside_lpips': .02}
EPOCHS, LR = 4, .0002
START_SHA = '3b26a5b6f0dd1f68856dcfd6367fbaadd2093e19dd7f8aee6c6435cb73867c7f'
PRIOR_ARCHIVE_SHA = '83a3548d6f8287c9246d94b1ff16126ea6d4d0f4d38634ea35ac513b7c36509e'


def rgb_tensor(image):
    return torch.from_numpy(np.asarray(image).transpose(2, 0, 1).copy())


def cached_pairs(args, protocol):
    folders = {'train': [], 'val': []}
    protocol_sha = sha(args.previous/'protocol.json')
    source_hashes = {}
    for pair in protocol['pairs']:
        case = pair['case_id']
        row = read(args.previous/'pairs'/(case+'.json'))
        assert row['protocol_sha256'] == protocol_sha and row['case_id'] == case
        mask, sketch = distill.read_mask_and_sketch(args.previous/'cases', args.data, pair)
        images = {}
        for arm in distill.ARMS:
            path = args.previous/'images'/arm/(case+'.png')
            assert sha(path) == row['arms'][arm]['png_sha256'], path
            images[arm] = np.asarray(rgb(path))
        teacher_file = args.previous/'teachers'/(case+'.png')
        teacher = rgb(teacher_file)
        expected = distill.mix_pixels(images['off'], images['on'], mask, distill.TEACHER)['blend']
        assert np.array_equal(teacher, expected), case
        target_file = args.data/pair['gt']
        target = rgb(target_file).resize(SIZE, Image.Resampling.BILINEAR)
        pixels = distill.pixel_inputs(images['off'], mask, sketch)
        folders[pair['split']].append((pixels, rgb_tensor(teacher), rgb_tensor(target)))
        source_hashes[case] = {'teacher': sha(teacher_file), 'gt': sha(target_file)}
    assert {key: len(rows) for key, rows in folders.items()} == {'train': 128, 'val': 32}
    return folders, source_hashes


def inside_lpips(model, pred, gt, interior):
    # 在同一真实成衣上下文中只替换衣内，再对衣内的 LPIPS 空间图取均值。
    model.spatial = True
    composed = torch.lerp(gt, pred, interior)
    distance = model(composed * 2 - 1, gt * 2 - 1)
    return ((distance * interior).sum(dim=(-1, -2)) /
            interior.sum(dim=(-1, -2)).clamp_min(1)).mean()


@torch.no_grad()
def validation(head, metric, batches, device):
    head.eval()
    metric.spatial = False
    total = np.zeros(2)
    for pixels, teacher, target in batches:
        pixels = pixels.to(device)
        pred = head.render(pixels)
        l1 = distill.region_l1(pred, teacher.to(device).float()/255, pixels)
        lpips = metric((pred*255).round()/127.5 - 1, target.to(device).float()/127.5 - 1).mean()
        total += np.array([float(l1), float(lpips)])*len(pixels)
    return dict(zip(('teacher_l1', 'full_gt_lpips'), (total/len(batches.dataset)).tolist()))


def train(args):
    completed = args.out/'complete_training.json'
    if completed.exists():
        recorded = read(completed)
        assert all(sha(args.out/arm/'head.pt') == digest for arm, digest in recorded['heads'].items())
        return
    verification = argparse.Namespace(out=args.previous, baseline=args.baseline)
    protocol = distill.experiment_protocol(verification)
    verified_files(args.previous/'eval64', PRIOR_ARCHIVE_SHA, ('result.json', 'per_pair.json', 'student_pool3.npz'))
    assert sha(args.previous/'head.pt') == START_SHA
    initial = read(args.previous/'training.json')
    assert initial['best_epoch'] == 11 and initial['checkpoint_sha256'] == START_SHA
    frozen = distill.verify_teacher_shards(verification, read(args.baseline/'result.json'))
    assert metric_assets() == protocol['metric_assets']
    data, target_hashes = cached_pairs(args, protocol)
    write(args.out/'source_manifest.json', dict(
        previous_protocol_sha256=sha(args.previous/'protocol.json'), frozen_hashes=frozen,
        initial_epoch=11, initial_checkpoint_sha256=START_SHA, seed=SEED, sampler=protocol['sampler'],
        targets=target_hashes, implementation_sha256=sha(Path(__file__)),
        lr=LR, epochs=EPOCHS, batch=distill.BATCH, inside_lpips_weights=ARMS))
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    training = DataLoader(data['train'], batch_size=distill.BATCH, shuffle=True,
                          generator=torch.Generator().manual_seed(SEED))
    valid = DataLoader(data['val'], batch_size=distill.BATCH)
    metric, inception = metric_models(args.device, protocol['metric_assets'])
    del inception
    checkpoint = torch.load(args.previous/'head.pt', map_location='cpu', weights_only=True)
    heads = {arm: distill.RGBResidualHead().to(args.device) for arm in ARMS}
    for arm, head in heads.items():
        head.load_state_dict(checkpoint)
        (args.out/arm).mkdir(exist_ok=True)
        shutil.copyfile(args.previous/'protocol.json', args.out/arm/'protocol.json')
    optimizers = {arm: torch.optim.Adam(head.parameters(), lr=LR) for arm, head in heads.items()}
    start = validation(heads['A_l1'], metric, valid, args.device)
    history, best, best_epoch = [], float('inf'), 0
    for epoch in range(1, EPOCHS+1):
        for head in heads.values():
            head.train()
        totals = {arm: np.zeros(3) for arm in ARMS}
        for pixels, teacher, target in training:
            pixels, teacher, target = (value.to(args.device) for value in (pixels, teacher, target))
            teacher, target = teacher.float()/255, target.float()/255
            for arm, weight in ARMS.items():
                optimizer = optimizers[arm]
                optimizer.zero_grad(set_to_none=True)
                pred = heads[arm].render(pixels)
                l1 = distill.region_l1(pred, teacher, pixels)
                perception = inside_lpips(metric, pred, target, pixels[:, 5:6]) if weight else l1.new_zeros(())
                loss = l1 + weight*perception
                loss.backward()
                optimizer.step()
                totals[arm] += np.array([float(v.detach()) for v in (l1, perception, loss)])*len(pixels)
        measures = {arm: validation(head, metric, valid, args.device) for arm, head in heads.items()}
        assert all(np.isfinite(x).all() for x in totals.values())
        assert all(np.isfinite(list(row.values())).all() for row in measures.values())
        history.append(dict(epoch=epoch, updates=epoch*len(training), validation=measures,
                            training={arm: dict(zip(('teacher_l1', 'inside_gt_lpips', 'total'),
                                                    (value/len(training.dataset)).tolist()))
                                      for arm, value in totals.items()}))
        write(args.out/'progress.json', {'initial': start, 'history': history})
        print('RGB_AB_EPOCH', epoch, measures, flush=True)
        # 共用验证选轮规则，确保两个待评估权重具有完全相同的更新步数。
        score = np.mean([row['full_gt_lpips'] for row in measures.values()])
        if score < best:
            best, best_epoch = score, epoch
            for arm, head in heads.items():
                temporary = args.out/arm/'head.pt.tmp'
                torch.save(head.state_dict(), temporary)
                temporary.replace(args.out/arm/'head.pt')
    hashes = {}
    for arm, weight in ARMS.items():
        hashes[arm] = sha(args.out/arm/'head.pt')
        write(args.out/arm/'training.json', dict(arm=arm, checkpoint_sha256=hashes[arm],
              initial_checkpoint_sha256=START_SHA, initial_epoch=11, epochs=EPOCHS, best_epoch=best_epoch,
              checkpoint_updates=best_epoch*len(training), head_optimizer_updates=EPOCHS*len(training),
              learning_rate=LR, inside_lpips_weight=weight, training_pairs=len(data['train']),
              validation_pairs=len(data['val']), initial_validation=start, history=history,
              frozen_hashes=frozen, frozen_optimizer_updates=0, ground_truth_used_for_loss=bool(weight),
              new_s0_trajectories=0, environment=environment()))
    write(completed, dict(heads=hashes, shared_epoch=best_epoch, updates=best_epoch*len(training)))
    print('RGB_AB_TRAINED', best_epoch, hashes, flush=True)


def report(args):
    completed = read(args.out/'complete_training.json')
    summary = dict(arms={}, initial_checkpoint_sha256=START_SHA, new_s0_trajectories=0, **completed)
    archives = ['source_manifest.json', 'complete_training.json', 'progress.json']
    for arm in ARMS:
        path = args.out/arm
        assert sha(path/'head.pt') == completed['heads'][arm]
        opts = argparse.Namespace(baseline=args.baseline, data=args.data, out=path, device=args.device, action='evaluate64')
        distill.evaluate(opts)
        sixty_four = read(path/'eval64/result.json')
        accepted = sixty_four['acceptance']['passed']
        summary['arms'][arm] = {'eval64': sixty_four['acceptance'], 'eval500': None}
        archives.append(f'{arm}/eval64/review.zip')
        if accepted:
            opts.action = 'evaluate500'
            distill.evaluate(opts)
            summary['arms'][arm]['eval500'] = read(path/'eval500/result.json')['acceptance']
            archives.append(f'{arm}/eval500/review.zip')
    write(args.out/'summary.json', summary)
    archives.append('summary.json')
    manifest = {name: sha(args.out/name) for name in archives}
    write(args.out/'manifest.json', manifest)
    temporary = args.out/'ab_review.zip.tmp'
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in (*archives, 'manifest.json'):
            z.write(args.out/name, name)
    temporary.replace(args.out/'ab_review.zip')
    for arm, row in summary['arms'].items():
        print('RGB_AB_RESULT', arm, row['eval64']['passed'],
              {key: round(row['eval64']['checks'][key]['value'], 8) for key in (*METRICS, 'fid')}, flush=True)
    print('RGB_AB_ARCHIVE', sha(args.out/'ab_review.zip'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('train', 'evaluate'))
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--project', type=Path, default=Path('/share/home/u2515283058/Mymodel'))
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.baseline = args.prior/distill.BASE
    args.previous = args.baseline/distill.RUN
    args.out = args.previous/RUN
    args.out.mkdir(exist_ok=True)
    args.data = args.project.parent/'datasets/BF'
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))
    cv2.setNumThreads(1)
    {'train': train, 'evaluate': report}[args.action](args)
