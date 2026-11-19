"""双图 0.25 教师的 RGB 残差蒸馏和冻结原配对验收。"""
import argparse
import hashlib
import os
import re
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from eval.eval_utils import prepare_evaluation_masks
from tools.e2_f68_edge_residual import ACCEPTANCE_SHA, mix_pixels, summarize, verified_files
from tools.e2_f68_real_paired import (ARMS, CATEGORIES, CHECKPOINT_SHA, CONDITION, SEED, SIZE,
                                     generate, image_scores, metric_assets, metric_models, rgb, verify, write)
from tools.e2_f68_region_diagnostic import ARCHIVE_SHA, gaussian, regions, tensors
from tools.e2_f68_single_s0_gate import subset
from tools.e2_fft_experiment import environment, read, sha


RUN = 'rgb_distill_a025_20261119'
BASE = 'real_paired_B_on_off_20261118'
GATE_ARCHIVE_SHA = 'b6cc1124894e86d998d33980563bd1cc5baedcdfd37454bdb6ea401c93f2e754'
TEACHER = {'blend': .25}
TRAIN_PER_CLASS, VAL_PER_CLASS = 32, 8
N = len(CATEGORIES) * (TRAIN_PER_CLASS + VAL_PER_CLASS)
EPOCHS, BATCH, LR = 12, 4, .001
GARMENT_TERMS = {
    'dress': r'\b(?:dress(?:es)?(?![- ]shirts?)|skirts?)\b',
    'outwear': r'\b(?:coats?|jackets?|blazers?|trench(?:es)?)\b',
    'pants': r'\b(?:pants|trousers|jeans|slacks|leggings)\b',
    'top': r'\b(?:tops?|blouses?|shirts?|t-?shirts?|sweaters?|pullovers?)\b',
}


def caption_category(caption):
    matches = [category for category, pattern in GARMENT_TERMS.items() if re.search(pattern, caption, re.I)]
    return matches[0] if len(matches) == 1 else None


def choose_pairs(training, frozen_pairs):
    used = {Path(row['gt']).stem for row in frozen_pairs}
    selected = []
    groups = {category: [] for category in CATEGORIES}
    for row in training:
        category = caption_category(row['caption'])
        if category is not None:
            groups[category].append(row)
    for category, group in groups.items():
        group.sort(key=lambda row: hashlib.sha256(('f68-rgb-distill-v1:' + row['sample_id']).encode()).hexdigest())
        current = []
        for row in group:
            product = Path(row['cloth']).stem
            if product in used:
                continue
            assert all(Path(row[key]).stem == product for key in ('cloth', 'sketch', 'texture'))
            assert row['caption'].strip()
            used.add(product)
            current.append(dict(case_id=category+'_'+product, sample_id=row['sample_id'], category=category,
                                sketch=row['sketch'], texture=row['texture'], gt=row['cloth'],
                                caption=row['caption'], seed=SEED,
                                split='train' if len(current) < TRAIN_PER_CLASS else 'val'))
            if len(current) == TRAIN_PER_CLASS + VAL_PER_CLASS:
                break
        assert len(current) == TRAIN_PER_CLASS + VAL_PER_CLASS, category
        selected.extend(current)
    return selected


def experiment_protocol(args):
    protocol = verify(args)
    assert protocol['baseline_sha256'] == sha(args.baseline/'protocol.json')
    assert protocol['implementation_sha256'] == sha(Path(__file__))
    return protocol


def prepare(args):
    baseline = verify(argparse.Namespace(out=args.baseline))
    if (args.out/'protocol.json').exists():
        experiment_protocol(args)
        return
    path = args.project/'data/processed/bf_full_audit_v1/training_clean.json'
    pairs = choose_pairs(read(path), baseline['pairs'])
    assert sha(args.checkpoint) == CHECKPOINT_SHA
    assert metric_assets() == baseline['metric_assets']
    for pair in pairs:
        folder = args.out/'cases'/pair['case_id']
        folder.mkdir(parents=True, exist_ok=True)
        pair['input_sha256'] = {name: sha(args.data/pair[name]) for name in ('sketch', 'texture')}
        reference = folder/(CONDITION+'_ref.png')
        rgb(args.data/pair['texture']).resize(SIZE, Image.Resampling.BICUBIC).save(reference)
        pair['reference_hashes'] = {CONDITION: sha(reference)}
        bundle = prepare_evaluation_masks(SIZE, sketch_path=str(args.data/pair['sketch']), mask_policy='sketch_only')
        mask = bundle['garment']
        assert mask is not None and mask.any(), pair['case_id']
        Image.fromarray(mask.astype(np.uint8)*255).save(folder/'mask.png')
        pair['mask_sha256'] = sha(folder/'mask.png')
    protocol = {name: baseline[name] for name in ('source_sha256', 'checkpoint_sha256', 'sampler', 'metric_assets')}
    protocol.update(pairs=pairs, input_manifest_sha256={str(path): sha(path)},
                    baseline_sha256=sha(args.baseline/'protocol.json'), implementation_sha256=sha(Path(__file__)),
                    alpha=.25, radius=8, feather_sigma=2, epochs=EPOCHS, batch=BATCH, lr=LR,
                    train_per_class=TRAIN_PER_CLASS, val_per_class=VAL_PER_CLASS, seed=SEED,
                    category_source='Explicit unambiguous garment terms from official training captions',
                    category_patterns=GARMENT_TERMS)
    write(args.out/'protocol.json', protocol)
    print('RGB_DISTILL_PREPARED', len(pairs), flush=True)


def pixel_inputs(off, mask, sketch_path):
    drawing = np.asarray(rgb(sketch_path).convert('L').resize(SIZE, Image.Resampling.BILINEAR))
    inside, edge, _ = regions(mask, 8)
    wi, we = (gaussian(region.astype('float32'), 2) for region in (inside, edge))
    pixels = np.concatenate((off.astype('float32').transpose(2, 0, 1)/255, drawing[None]/np.float32(255),
                             we[None], wi[None]), axis=0)
    return torch.from_numpy(pixels)


class RGBResidualHead(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Conv2d(6, 32, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(32, 32, 3, padding=2, dilation=2), nn.ReLU(),
                         nn.Conv2d(32, 32, 3, padding=4, dilation=4), nn.ReLU(),
                         nn.Conv2d(32, 3, 3, padding=1))
        nn.init.zeros_(self[-1].weight)
        nn.init.zeros_(self[-1].bias)

    def render(self, pixels):
        gate = pixels[:, 4:5] + .25*pixels[:, 5:6]
        return (pixels[:, :3] + gate*self(pixels)).clamp(0, 1)


def region_l1(prediction, target, pixels):
    error = (prediction - target).abs().mean(dim=1, keepdim=True)
    weights = pixels[:, 4:6]
    return ((error*weights).sum(dim=(-1, -2)) / weights.sum(dim=(-1, -2)).clamp_min(1)).mean()


def read_mask_and_sketch(folder, data, pair):
    mask_file = folder/pair['case_id']/'mask.png'
    sketch = data/pair['sketch']
    assert sha(mask_file) == pair['mask_sha256'] and sha(sketch) == pair['input_sha256']['sketch']
    return np.asarray(rgb(mask_file))[..., 0] > 0, sketch


class TrainingPairs(Dataset):
    def __init__(self, args, pairs, split):
        self.examples = []
        protocol_sha = sha(args.out/'protocol.json')
        target_dir = args.out/'teachers'
        target_dir.mkdir(exist_ok=True)
        for pair in pairs:
            if pair['split'] != split:
                continue
            case = pair['case_id']
            row = read(args.out/'pairs'/(case+'.json'))
            assert row['protocol_sha256'] == protocol_sha
            mask, sketch = read_mask_and_sketch(args.out/'cases', args.data, pair)
            images = []
            for arm in ARMS:
                path = args.out/'images'/arm/(case+'.png')
                assert sha(path) == row['arms'][arm]['png_sha256']
                images.append(np.asarray(rgb(path)))
            off, on = images
            target = mix_pixels(off, on, mask, TEACHER)['blend']
            Image.fromarray(target).save(target_dir/(case+'.png'))
            self.examples.append((pixel_inputs(off, mask, sketch), torch.from_numpy(target.transpose(2, 0, 1))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def verify_teacher_shards(args, original):
    covers = []
    for path in args.out.glob('generated_*.json'):
        row = read(path)
        assert row['protocol_sha256'] == sha(args.out/'protocol.json')
        assert row['hashes'] == original['frozen_model_hashes']
        assert row['scheduler_sha256'] == original['scheduler_source_sha256']
        assert row['frozen_unchanged'] and row['optimizer_updates'] == 0
        covers.extend(range(row['start'], row['end']))
    assert sorted(covers) == list(range(N))
    return original['frozen_model_hashes']


@torch.no_grad()
def validate(head, loader, device):
    head.eval()
    total = 0.
    for x, y in loader:
        x, y = x.to(device), y.to(device).float()/255
        total += float(region_l1(head.render(x), y, x))*len(x)
    return total/len(loader.dataset)


def train(args):
    protocol = experiment_protocol(args)
    original = read(args.baseline/'result.json')
    frozen = verify_teacher_shards(args, original)
    if (args.out/'training.json').exists():
        assert sha(args.out/'head.pt') == read(args.out/'training.json')['checkpoint_sha256']
        return
    torch.manual_seed(SEED)
    training = TrainingPairs(args, protocol['pairs'], 'train')
    validation = TrainingPairs(args, protocol['pairs'], 'val')
    assert len(training) == 128 and len(validation) == 32
    train_data = DataLoader(training, batch_size=BATCH, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    val_data = DataLoader(validation, batch_size=BATCH)
    head = RGBResidualHead().to(args.device)
    optimizer = torch.optim.Adam(head.parameters(), lr=LR)
    initial = validate(head, val_data, args.device)
    best, best_epoch = float('inf'), 0
    history = []
    for epoch in range(1, EPOCHS+1):
        head.train()
        total = 0.
        for x, y in train_data:
            x, y = x.to(args.device), y.to(args.device).float()/255
            optimizer.zero_grad(set_to_none=True)
            loss = region_l1(head.render(x), y, x)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())*len(x)
        value = validate(head, val_data, args.device)
        assert np.isfinite(value) and np.isfinite(total)
        history.append(dict(epoch=epoch, training_l1=total/len(training), validation_l1=value))
        print('RGB_DISTILL_EPOCH', history[-1], flush=True)
        if value < best:
            best, best_epoch = value, epoch
            temporary = args.out/'head.pt.tmp'
            torch.save(head.state_dict(), temporary)
            temporary.replace(args.out/'head.pt')
    write(args.out/'training.json', dict(training_pairs=len(training), validation_pairs=len(validation),
          initialization_validation_l1=initial, best_epoch=best_epoch, best_validation_l1=best, history=history,
          head_parameters=sum(p.numel() for p in head.parameters()), checkpoint_sha256=sha(args.out/'head.pt'),
          head_optimizer_updates=EPOCHS*len(train_data), frozen_hashes=frozen, frozen_optimizer_updates=0,
          ground_truth_used=False, teacher_s0_trajectories=2*N, environment=environment()))


def acceptance(result):
    arms = result['groups']['all']['arms']
    checks = {}
    for field in ('lpips', 'fid', 'ssim', 'psnr', 'contour_f1', 'iou', 'leak'):
        key = 'fid' if field == 'fid' else 'mean'
        off, blend, student = (arms[arm][field][key] for arm in ('off', 'blend', 'student'))
        lower = field in ('lpips', 'fid', 'leak')
        goal = off + .8*(blend-off) if field in ('lpips', 'ssim', 'contour_f1') else off
        passed = (student < goal if field == 'fid' else student <= goal) if lower else student >= goal
        checks[field] = dict(value=student, threshold=goal, passed=passed, off=off, teacher=blend)
    return dict(passed=all(item['passed'] for item in checks.values()), checks=checks)


@torch.no_grad()
def evaluate(args):
    few = args.action == 'evaluate64'
    label = 'eval64' if few else 'eval500'
    out = args.out/label
    out.mkdir(exist_ok=True)
    if (out/'review.zip').exists():
        assert sha(out/'result.json') == read(out/'review_manifest.json')['result.json']
        return
    if not few:
        assert read(args.out/'eval64/result.json')['acceptance']['passed']
    experiment_protocol(args)
    protocol = verify(argparse.Namespace(out=args.baseline))
    blend_dir = args.baseline/'edge_residual_acceptance_20261118'
    gate_dir = args.baseline/'single_s0_gate_20261119'
    verified_files(args.baseline, ARCHIVE_SHA, ('protocol.json', 'per_image.json', 'fid_pool3.npz'))
    verified_files(blend_dir, ACCEPTANCE_SHA, ('per_pair.json', 'inner_mix_pool3.npz'))
    assert metric_assets() == protocol['metric_assets']
    ids = subset(protocol) if few else list(range(len(protocol['pairs'])))
    original, blends = read(args.baseline/'per_image.json'), read(blend_dir/'per_pair.json')
    controls = (*ARMS, 'blend', 'gate') if few else (*ARMS, 'blend')
    with np.load(args.baseline/'fid_pool3.npz', allow_pickle=False) as data:
        features = {name: data[name][ids] for name in (*ARMS, 'gt')}
    with np.load(blend_dir/'inner_mix_pool3.npz', allow_pickle=False) as data:
        features['blend'] = data['edge_a025'][ids]
    if few:
        verified_files(gate_dir, GATE_ARCHIVE_SHA, ('per_pair.json', 'pool3.npz'))
        gates = {r['index']: r for r in read(gate_dir/'per_pair.json')}
        assert list(gates) == ids
        with np.load(gate_dir/'pool3.npz', allow_pickle=False) as data:
            features['gate'] = data['gate']
    model, inception = metric_models(args.device, protocol['metric_assets'])
    head = RGBResidualHead().to(args.device)
    training = read(args.out/'training.json')
    assert sha(args.out/'head.pt') == training['checkpoint_sha256']
    head.load_state_dict(torch.load(args.out/'head.pt', map_location=args.device, weights_only=True))
    head.eval()
    images = out/'images'
    images.mkdir(exist_ok=True)
    rows, student_features = [], []
    cached = args.out/'eval64/images'
    cached_hashes = {} if few else {r['case_id']: r['png_sha256'] for r in read(args.out/'eval64/per_pair.json')}
    for offset in range(0, len(ids), BATCH):
        predicted, targets = [], []
        for index in ids[offset:offset+BATCH]:
            pair = protocol['pairs'][index]
            case = pair['case_id']
            prev, blend = original[index], blends[index]
            assert case == prev['case_id'] == blend['case_id'] and prev['index'] == index
            mask, sketch = read_mask_and_sketch(args.baseline/'cases', args.data, pair)
            off_path = args.baseline/'images/off'/(case+'.png')
            target_path = args.data/pair['gt']
            assert sha(off_path) == prev['arms']['off']['png_sha256']
            assert sha(target_path) == pair['input_sha256']['gt']
            pixels = pixel_inputs(np.asarray(rgb(off_path)), mask, sketch)
            png = images/(case+'.png')
            if case in cached_hashes:
                path = cached/(case+'.png')
                assert sha(path) == cached_hashes[case]
                picture = rgb(path)
                png = path
            else:
                raw = head.render(pixels[None].to(args.device))[0].permute(1, 2, 0).cpu().numpy()
                picture = Image.fromarray(np.rint(raw*255).astype('uint8'))
                picture.save(png)
            target = rgb(target_path).resize(SIZE, Image.Resampling.BILINEAR)
            score = {**prev['scores'], 'blend': blend['scores']['edge_a025'],
                     'student': image_scores(picture, target, png, target_path, mask)}
            if few:
                assert gates[index]['case_id'] == case
                score['gate'] = gates[index]['scores']['gate']
            predicted.append(np.asarray(picture))
            targets.append(np.asarray(target))
            rows.append(dict(index=index, case_id=case, category=pair['category'], png_sha256=sha(png), scores=score))
        inputs, expected = [tensors(batch).to(args.device) for batch in (predicted, targets)]
        lpips = model(inputs*2-1, expected*2-1).flatten().tolist()
        for row, value in zip(rows[-len(predicted):], lpips):
            row['scores']['student']['lpips'] = value
        student_features.append(inception(inputs)[0].flatten(1).cpu().numpy())
        if offset % 40 == 0:
            print('RGB_DISTILL_SCORED', label, len(rows), flush=True)
    features['student'] = np.concatenate(student_features)
    assert all(v.shape == (len(ids), 2048) and np.isfinite(v).all() for v in features.values())
    result = summarize(rows, features, (*controls, 'student'), controls, ('blend', 'student'))
    result['acceptance'] = acceptance(result)
    result.update(samples=len(ids), new_s0_trajectories=0, reused_student_images=len(cached_hashes),
                  head_sha256=training['checkpoint_sha256'], environment=environment())
    write(out/'result.json', result)
    write(out/'per_pair.json', rows)
    # 教师和基线特征已有完整归档，此处只保存新增特征。
    np.savez_compressed(out/'student_pool3.npz', student=features['student'])
    files = [out/name for name in ('result.json', 'per_pair.json', 'student_pool3.npz')]
    files += [args.out/name for name in ('training.json', 'protocol.json', 'head.pt')]
    files += sorted(args.out.glob('generated_*.json'))
    write(out/'review_manifest.json', {p.name: sha(p) for p in files})
    temporary = out/'review.zip.tmp'
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in (*files, out/'review_manifest.json'):
            archive.write(path, path.name)
    temporary.replace(out/'review.zip')
    print('RGB_DISTILL_COMPLETE', label, result['acceptance'], sha(out/'review.zip'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'generate', 'train', 'evaluate64', 'evaluate500'))
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--project', type=Path, default=Path('/share/home/u2515283058/Mymodel'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=N)
    args = parser.parse_args()
    args.baseline = args.prior/BASE
    args.out = args.baseline/RUN
    args.out.mkdir(parents=True, exist_ok=True)
    args.data = args.project.parent/'datasets/BF'
    args.checkpoint = args.project/'output_eval/e2_state_b_pastel_followup_20261115/trust_retry/training/step1.pt'
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '2')))
    cv2.setNumThreads(1)
    actions = dict(prepare=prepare, generate=generate, train=train, evaluate64=evaluate, evaluate500=evaluate)
    actions[args.action](args)
