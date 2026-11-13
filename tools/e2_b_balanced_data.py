"""原4训练身份、9独立裁片；几何增强覆盖三色组，验证22产品根隔离。"""
import copy
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

SYNTHETIC = ('v16', 'h16', 'v32', 'h32')


def geometric_variants(picture):
    # 保留完整RGB三元组；不做全衣身统一色度，也不读取验证参考。
    rotate = picture.transpose(Image.Transpose.ROTATE_90).resize(picture.size, Image.Resampling.BICUBIC)
    half = picture.resize((192, 256), Image.Resampling.BICUBIC)
    tiled = Image.new('RGB', picture.size)
    for xy in ((0, 0), (192, 0), (0, 256), (192, 256)):
        tiled.paste(half, xy)
    return rotate, tiled


def schedule(spec):
    rng = np.random.default_rng(20261114)
    rows = []
    # 每16步四种数据各4次，400步恰好100次/类、100次/身份。
    counts = dict.fromkeys(('neutral', 'pastel', 'multi', 'synthetic'), 0)
    for block in range(25):
        for slot in range(4):
            group = ('neutral', 'pastel', 'multi', 'synthetic')[(block + slot) % 4]
            for j in range(4):
                case = spec['train'][(j + block) % 4]['case_id']
                n = counts[group]; counts[group] += 1
                if group == 'synthetic':
                    # 交替方向/粗细，共50步/类，四个身份均覆盖。
                    kind = 'direction' if n % 2 == 0 else 'period'
                    pair = ('v16', 'h16') if kind == 'direction' else ('v16', 'v32')
                    if (n // 2) % 2:
                        pair = ('v32', 'h32') if kind == 'direction' else ('h16', 'h32')
                else:
                    refs = [r for r in spec['references'] if r['palette_group'] == group]
                    ref = refs[(n // 4 + j) % 3]
                    kind = 'geometry' if (n // 4) % 2 == 0 else 'colour'
                    variant = ('r90', 's50')[(n // 8) % 2] if kind == 'geometry' else 'color'
                    pair = (f"r{ref['index']}_native", f"r{ref['index']}_{variant}")
                step = len(rows) + 1
                four = (j + block + slot) % 4 == 0
                rows.append(dict(step=step, case=case, group=group, kind=kind, conditions=list(pair),
                                 state_index=21 if four else int(rng.integers(24)), hops=4 if four else 2,
                                 online=(j + block + slot) % 2 == 0))
    assert len(rows) == 400 and all(n == 100 for n in counts.values())
    assert sum(r['online'] for r in rows) == 200 and sum(r['hops']==4 for r in rows) == 100
    assert sum(r['kind']=='direction' for r in rows) == sum(r['kind']=='period' for r in rows) == 50
    return rows


def prepare(args, hashes):
    from models.e2_texture_diagnostics import reference_control
    from tools.e2_fft_experiment import read, write, sha, environment

    prior = read(args.joint/'input_protocol.json')
    cfg = read(Path(__file__).resolve().parents[1]/'configs/e2_b_joint_refs.json')
    assert sha(args.joint/'input_protocol.json') == '3a0b9fa9dad76b810bf40ec2fd9b6a2b8bb9b69bf677f3b3747b224a4c0cc4b9'
    assert prior['references']==cfg['references']
    assert sha(args.previous/'protocol.json')==cfg['protected_protocol_sha256']
    assert sha(args.previous/'training/step400.pt') == prior['base_checkpoint_sha256'] == cfg['base_checkpoint_sha256']
    assert sha(args.joint/'protocol.json') == '28569f84b49cdacd74ed9eee28f3d65b39c30e4ac2079598be1f3857119732ac'
    assert len(prior['protected_product_roots']) == 22 and len(prior['holdout']) == 11
    assert not set(prior['protected_product_roots']) & {r['identity_group'] for r in prior['references']+prior['train']}
    spec = copy.deepcopy(prior)
    folder = args.out/'references'; folder.mkdir(parents=True, exist_ok=True)
    conditions, descriptions, audit = [], {}, []
    lab = lambda im: cv2.cvtColor(np.asarray(im,np.float32)/255,cv2.COLOR_RGB2LAB)
    def dense_palette(value):
        angles=np.arange(16)*np.pi/16
        directions=np.stack((np.cos(angles),np.sin(angles)))
        return np.quantile(value[...,1:].reshape(-1,2)@directions,(np.arange(1024)+.5)/1024,axis=0)
    board = Image.new('RGB',(9*160,240),'white');draw=ImageDraw.Draw(board)
    for j, r in enumerate(spec['references']):
        prefix=f"r{r['index']}"
        source=args.joint/'references'/f'{prefix}_source.png'
        assert sha(source)==r['crop_sha256']
        native=Image.open(args.joint/'references'/f'{prefix}_native.png').convert('RGB')
        colour=Image.open(args.joint/'references'/f'{prefix}_color.png').convert('RGB')
        rotate,half=geometric_variants(native)
        original=lab(native);dist=dense_palette(original)
        for variant,im in (('native',native),('color',colour),('r90',rotate),('s50',half)):
            c=f'{prefix}_{variant}';im.save(folder/f'{c}.png');conditions.append(c)
            descriptions[c]=dict(index=r['index'],family=r['family'],group=r['palette_group'],variant=variant)
            if variant in ('r90','s50'):
                value=lab(im);current=dense_palette(value)
                row=dict(condition=c,mean_ab=float(np.linalg.norm(value[...,1:].mean((0,1))-original[...,1:].mean((0,1)))),
                         palette_swd=float(np.abs(current-dist).mean()))
                # 仅在训练裁片上筛查几何增强的改色；不根据验证表现调整阈值。
                assert row['mean_ab']<=.5 and row['palette_swd']<=1.,row
                audit.append(row)
        shutil.copyfile(source,folder/f'{prefix}_source.png')
        draw.text((j*160+2,2),f"{r['index']} {r['family']} {r['palette_group']}",fill='black')
        board.paste(native.resize((156,208)),(j*160,24))
    stripe=Image.open(folder/'r2779_native.png').convert('RGB')
    signature=lambda a:np.sort(np.ascontiguousarray(a).view('V3'),axis=None)
    for c in SYNTHETIC:
        value=reference_control(np.asarray(stripe),c,20261114)
        assert np.array_equal(signature(value),signature(np.asarray(stripe)))
        Image.fromarray(value).save(folder/f'{c}.png');conditions.append(c)
        descriptions[c]=dict(index=2779,family='stripes',group='synthetic',variant=c)
    assert len(conditions)==40
    for p in spec['train']:
        src=args.joint/'cases'/p['case_id'];dst=args.out/'cases'/p['case_id'];dst.mkdir(parents=True,exist_ok=True)
        for name in ('mask.png','sketch.jpg','train_anchor.png'):shutil.copyfile(src/name,dst/name)
        for c in conditions:shutil.copyfile(folder/f'{c}.png',dst/f'{c}_ref.png')
        p['reference_hashes']={c:sha(dst/f'{c}_ref.png') for c in conditions}
    for p in spec['holdout']:
        src=args.joint/'cases'/p['case_id'];dst=args.out/'cases'/p['case_id'];dst.mkdir(parents=True,exist_ok=True)
        names=['mask.png','sketch.jpg','generation_base.json']+[c+'_ref.png' for c in spec['validation_conditions']]
        names += [f'{c}_s{s}_{a}.png' for c in spec['validation_conditions'] for s in range(2) for a in ('m0','goal','positive','base')]
        for name in names:
            shutil.copyfile(src/name,dst/name);assert sha(src/name)==sha(dst/name)
    board.save(args.out/'training_reference_panel.jpg',quality=95)
    spec.update(source=environment(),code=hashes,conditions=conditions,descriptions=descriptions,
                architecture='unchanged274372; both arms raw sketch and full spatial reference',arm_modes={'A':'raw','B':'raw'},
                labels={'A':'A2 weighted sum','B':'B2 PCGrad'},updates_per_arm=400,
                previous_joint_protocol_sha256=sha(args.joint/'protocol.json'),geometry_palette_audit=audit,
                training=dict(seed=20261114,optimizer='AdamW',lr=3e-5,weight_decay=.01,gradient_clip=1.,
                              updates_per_arm=400,online_fraction=.5,terminal_four_fraction=.25,
                              checkpoint_steps=[0,50,100,200,400]),
                controls='identical initial B400, data, schedule, loss groups, calibrated weights, raw input; only PCGrad differs',
                target='same encoded latent and float32 VAE decode for spatial supervision; original palette with VAE tolerance',
                validation_policy='frozen step400 only; no validation-based selection or tuning; all22 roots and variants excluded')
    spec['schedule']=schedule(spec)
    write(args.out/'input_protocol.json',spec)
    print('BALANCED_PREPARED',len(spec['train']),len(conditions),22,flush=True)
