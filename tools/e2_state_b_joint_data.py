"""9个独立真实面料来源、24个固定条件；22个验证产品根始终隔离。"""
import copy
import hashlib
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

from models.e2_state_texture_b import StateTextureB
from tools import e2_state_b_material_data as old
from tools.e2_b_joint_losses import geometry
from tools.e2_b_repair_losses import lab_to_rgb
from tools.e2_fft_experiment import SOURCE, environment, read, sha, write
from tools.e2_state_b_generalization_data import perceptual_hash


def configuration():
    return read(SOURCE / 'configs/e2_b_joint_refs.json')


def initialize(args):
    source = args.previous / 'training/step400.pt'
    assert sha(source) == configuration()['base_checkpoint_sha256']
    b = StateTextureB(color_channels=2, reference_channels=192).to(args.device)
    b.load_state_dict(torch.load(source, map_location='cpu', weights_only=False)['b'])
    return b


def inputs(args, p, c, mode):
    values, weight = old.inputs(args, p, c, args.device)
    if mode != 'raw':
        mask = np.asarray(Image.open(args.out / 'cases' / p['case_id'] / 'mask.png')) > 0
        field = geometry(mask) if mode == 'outline' else np.zeros(mask.shape, np.float32)
        tensor = torch.from_numpy(field)[None, None].to(args.device)
        values['structure'] = torch.nn.functional.avg_pool2d(tensor, 8)
    return values, weight


def prepare(args, hashes):
    cfg = configuration()
    previous = read(args.previous / 'protocol.json')
    assert sha(args.previous / 'protocol.json') == cfg['protected_protocol_sha256']
    assert sha(args.previous / 'training/step400.pt') == cfg['base_checkpoint_sha256']
    records = read(args.project / 'output_eval/e2_b_screen_20261108/inputs/registry.json')['records']
    by_id = {v['index']: v for v in records}
    protected = set(previous['protected_product_roots'])
    train_roots = {v['identity_group'] for v in previous['train']}
    refs = cfg['references']
    roots = {v['identity_group'] for v in refs}
    assert len(refs) == len(roots) == 9 and not roots & (protected | train_roots)
    pixel = lambda path: hashlib.sha256(np.asarray(Image.open(path).convert('RGB')).tobytes()).hexdigest()
    held = [v for v in records if v['identity_group'] in protected]
    assert len(held) == 22
    held_pixels = {pixel(args.data / v[k]) for v in held for k in ('gt', 'texture')}
    held_phash = [perceptual_hash(args.data / v['gt']) for v in held]
    fingerprints = []
    independence = []
    folder = args.out / 'references'
    folder.mkdir(parents=True, exist_ok=True)
    conditions, descriptions = [], {}
    board = Image.new('RGB', (9 * 160, 240), 'white')
    draw = ImageDraw.Draw(board)
    for j, entry in enumerate(refs):
        i = entry['index']; row = by_id[i]
        assert row['identity_group'] == entry['identity_group']
        for k in ('gt', 'texture', 'sketch'):
            assert sha(args.data / row[k]) == entry['hashes'][k]
        assert all(pixel(args.data / row[k]) not in held_pixels for k in ('gt', 'texture'))
        fingerprint = perceptual_hash(args.data / row['gt'])
        distance = min(bin(fingerprint ^ v).count('1') for v in held_phash)
        assert distance > 4
        assert all(bin(fingerprint ^ v).count('1') > 4 for v in fingerprints)
        fingerprints.append(fingerprint)
        source = Image.open(args.data / row['gt']).convert('RGB').crop(entry['crop_box'])
        source.save(folder / f'r{i}_source.png')
        assert pixel(folder / f'r{i}_source.png') == entry['crop_pixel_sha256']
        native = source.resize((384, 512), Image.Resampling.BICUBIC)
        lab = cv2.cvtColor(np.asarray(native, np.float32) / 255, cv2.COLOR_RGB2LAB).astype(np.float64)
        ab_mean = lab[..., 1:].mean((0, 1))
        desired = lab.copy()
        centre = np.array([7., 9.]) if j % 2 == 0 else np.array([5., -8.])
        desired[..., 1:] = .6 * (lab[..., 1:] - ab_mean) + centre
        # 配对增强只用于训练；输入和目标共同使用实际生成的RGB，不假定裁剪后仍是理想Lab。
        recolored = Image.fromarray(np.rint(np.clip(lab_to_rgb(desired), 0, 1) * 255).astype(np.uint8))
        variants = [('native', native), ('color', recolored)]
        if entry['palette_group'] == 'neutral':
            rotate = native.transpose(Image.Transpose.ROTATE_90).resize((384, 512), Image.Resampling.BICUBIC)
            half = native.resize((192, 256), Image.Resampling.BICUBIC)
            tile = Image.new('RGB', (384, 512))
            for xy in ((0, 0), (192, 0), (0, 256), (192, 256)):
                tile.paste(half, xy)
            variants.extend([('r90', rotate), ('s50', tile)])
        for variant, picture in variants:
            condition = f'r{i}_{variant}'
            picture.save(folder / (condition + '.png'))
            conditions.append(condition)
            descriptions[condition] = dict(index=i, family=entry['family'], group=entry['palette_group'], variant=variant)
        draw.text((j * 160 + 2, 2), f"{i} {entry['family']} {entry['palette_group']}", fill='black')
        board.paste(native.resize((156, 208)), (j * 160, 24))
        independence.append(dict(index=i, root=row['identity_group'], nearest_heldout_gt_phash=distance,
                                 gt_sha256=sha(args.data / row['gt']), crop_sha256=sha(folder / f'r{i}_source.png')))
    assert len(conditions) == 24
    board.save(args.out / 'training_reference_panel.jpg', quality=96)
    train = copy.deepcopy(previous['train'])
    for p in train:
        old_d = args.previous / 'cases' / p['case_id']; d = args.out / 'cases' / p['case_id']
        d.mkdir(parents=True, exist_ok=True)
        for name in ('mask.png', 'sketch.jpg'):
            shutil.copyfile(old_d / name, d / name)
        for c in conditions:
            shutil.copyfile(folder / (c + '.png'), d / (c + '_ref.png'))
        p['reference_hashes'] = {c: sha(d / (c + '_ref.png')) for c in conditions}
    for p in previous['holdout']:
        old_d = args.previous / 'cases' / p['case_id']; d = args.out / 'cases' / p['case_id']
        d.mkdir(parents=True, exist_ok=True)
        names = ['mask.png', 'sketch.jpg'] + [c + '_ref.png' for c in previous['validation_conditions']]
        for c in previous['validation_conditions']:
            for seed in range(2):
                names += [f'{c}_s{seed}_{arm}.png' for arm in ('m0', 'goal', 'positive', 'b400')]
        for name in names:
            dest = name.replace('_b400.png', '_base.png')
            shutil.copyfile(old_d / name, d / dest)
            assert sha(old_d / name) == sha(d / dest)
        shutil.copyfile(old_d / 'generation.json', d / 'generation_base.json')
    spec = dict(source=environment(), code=hashes, references=refs, independence=independence,
        protected_product_roots=sorted(protected), train=train, holdout=copy.deepcopy(previous['holdout']),
        conditions=conditions, descriptions=descriptions, validation_conditions=previous['validation_conditions'],
        base_checkpoint_sha256=cfg['base_checkpoint_sha256'], previous_protocol_sha256=sha(args.previous / 'protocol.json'),
        updates_per_arm=400, input_only_review=True, reference_source=cfg['reference_source'],
        architecture='unchanged274372 parameters; A raw sketch channel; B mask-derived outline channel',
        sampler=previous['sampler'], previous_training_goals=84, training_goals=96,
        controls='same current B400 weights, reference pool, seeds and paired order; existing losses versus joint recipe; full recipe comparison, not single-component ablation',
        diagnostic='train4 only; raw/zero/outline trajectories; 12 palette/family probes, 3 times, full-B pairwise gradient matrix',
        gates=dict(color='all12 multi images retain75% to125% VAE-positive spread; mean-ab error <=positive error+2Lab; palette SWD improves',
            pattern='direction44/44 and period44/44; real strict counts increase, multi6/6; low-chroma count does not regress',
            residue='all3 multi identities do not regress versus base; all-real mean decreases',
            protection='all-real low-L and outside mean do not regress; visual garment shape check'),
        interpretation='same11 reused validation identities; real photo crops and controlled planar targets, not captured counterfactual garment truth')
    write(args.out / 'input_protocol.json', spec)
    print('JOINT_PREPARED', 9, len(conditions), len(train), len(spec['holdout']), flush=True)
