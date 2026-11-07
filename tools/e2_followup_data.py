"""已批准的 E2 后续诊断：先冻结输入资格，再查看模型输出。"""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage.color import rgb2lab

SIZE = (384, 512)
PROTOCOL = dict(version='e2_followup_20261108_v2_frozen_before_outputs', seed=110800, updates=0,
    pool='target=previous_dev64; references=fixed_training_black_white_regular_candidates', pretraining_exposure=True,
    reference_pool='16 per stripes/checks/dots, black+white caption, stable SHA order, exclude dev64; visual qualification before inference',
    same_color=dict(mean_lab_delta_max=5., lab_quantile_delta_max=10.),
    regular_pattern=dict(period_ratio_min=1.5, orientation_difference_min=45.),
    scales=[.75, 1., 1.33], maximum_targets=32,
    derived_controls=dict(direction='clean stripes rotated 90 degrees at native size',
        period='clean regular reference downsampled by 2 and tiled 2x2; image-scale diagnostic only',
        maximum_source_identities=8, reported_separately_from_real_reference_pairs=True),
    primary='appearance_neutral_real_reference_swap',
    reference_resize='legacy_BICUBIC_384x512', physical_scale_known=False,
    selection='input-only AI visual review plus fixed color/period eligibility; never model outputs',
    patch='fixed 64px fully inside sketch body; maximum 8; same coordinates across arms',
    evaluation='v2_separate_from_legacy_v1; paired identity response change, DISTS and protection metrics')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


def freeze(path, value):
    if Path(path).exists():
        assert read(path) == value, '已冻结清单发生变化：%s' % path
    else:
        write(path, value)


def rgb(path, size=SIZE):
    image = Image.open(path).convert('RGB')
    return image.resize(size, Image.Resampling.BICUBIC) if size else image


def asset(args, row, key):
    if key == 'cloth' and args.cloth_dir:
        return Path(args.cloth_dir)/Path(row[key]).name
    return Path(args.data)/row[key]


def spectral_proxy(pixels):
    """返回离散图像周期/条纹走向代理；不声称物理尺度或语义花型真值。"""
    a = np.asarray(pixels, dtype=np.float64)
    if a.ndim == 3:
        a = a @ np.array([.299, .587, .114])
    if a.max(initial=0) > 2:
        a = a / 255.
    std = float(a.std())
    if std < .02:
        return dict(valid=False, std=std, period=None, line_angle=None, peak_fraction=None)
    h, w = a.shape
    fy, fx = np.meshgrid(np.fft.fftfreq(h), np.fft.fftfreq(w), indexing='ij')
    radius = np.hypot(fx, fy)
    band = (radius >= max(1/h, 1/w)) & (radius <= .5)
    z = (a - a.mean()) / max(std, .02) * np.outer(np.hanning(h), np.hanning(w))
    power = np.abs(np.fft.fft2(z, norm='ortho')) ** 2
    selected = np.where(band, power, 0.)
    y, x = np.unravel_index(selected.argmax(), selected.shape)
    return dict(valid=True, std=std, period=float(1/radius[y, x]),
        line_angle=float((np.degrees(np.arctan2(fy[y, x], fx[y, x])) + 90) % 180),
        peak_fraction=float(selected[y, x]/max(selected.sum(), 1e-12)))


def color_summary(image):
    lab = rgb2lab(np.asarray(image, dtype=np.float64) / 255.)
    return dict(lab_mean=lab.mean((0, 1)).tolist(),
                lab_quantiles=np.quantile(lab.reshape(-1, 3), [.1, .5, .9], axis=0).tolist())


def color_distance(a, b):
    return (float(np.linalg.norm(np.asarray(a['lab_mean'])-b['lab_mean'])),
            float(np.linalg.norm(np.asarray(a['lab_quantiles'])-b['lab_quantiles'], axis=1).mean()))


def neutral_prompt(caption):
    # 只保留输入文本中的服装名词；颜色、花型、材料不进入此新评测协议。
    words = str(caption).lower().replace('-', ' ').split()
    noun = next((n for n in ['dress', 'coat', 'jacket', 'pants', 'trousers', 'skirt',
                 'blouse', 'shirt', 'sweater', 'top'] if n in words), 'garment')
    return 'a %s following the provided garment sketch.' % noun


def prepare(args):
    out = Path(args.out)
    rows, split = read(args.manifest), read(Path(args.parent)/'split.json')
    indices = split['validation_indices']
    assert len(indices) == 64 and len(rows) == 45384
    def record_for(index):
        row = rows[index]
        ref = rgb(asset(args, row, 'texture'))
        record = dict(index=index, **row,
            hashes={k: sha(asset(args, row, k)) for k in ['cloth', 'sketch', 'texture']},
            original_sizes={k: list(Image.open(asset(args, row, k)).size) for k in ['cloth', 'sketch', 'texture']},
            color=color_summary(ref), spectral_proxy=spectral_proxy(np.asarray(ref.resize((64, 64), Image.Resampling.BICUBIC))),
            neutral_prompt=neutral_prompt(row['caption']))
        return record
    records = [record_for(i) for i in indices]
    freeze(out/'inputs/dev64.json', dict(protocol=PROTOCOL, manifest_canonical_sha256=canonical_sha(rows),
        split_sha256=sha(Path(args.parent)/'split.json'), records=records))
    reference_indices=[]
    for words in [('stripe',),('check','plaid','gingham'),('polka','dotted')]:
        candidates=[i for i,row in enumerate(rows) if i not in set(indices)
            and 'black' in str(row['caption']).lower() and 'white' in str(row['caption']).lower()
            and any(w in str(row['caption']).lower() for w in words)]
        chosen=sorted(candidates,key=lambda i:hashlib.sha256(('followup_ref_pool/'+str(i)).encode()).hexdigest())[:16]
        reference_indices+=chosen
    refs=[record_for(i) for i in sorted(set(reference_indices))]
    freeze(out/'inputs/reference_pool.json',dict(protocol=PROTOCOL,records=refs,selection_uses_model_outputs=False))
    for page in range((len(records)+len(refs)+15)//16):
        group=records if page<4 else refs
        offset=page if page<4 else page-4
        panel = Image.new('RGB', (1536, 880), 'white')
        draw = ImageDraw.Draw(panel)
        for slot, row in enumerate(group[offset*16:(offset+1)*16]):
            x, y = slot % 4 * 384, slot // 4 * 220
            draw.text((x+3, y+3), '%s | sketch / GT / reference' % row['index'], fill='black')
            for col, key in enumerate(['sketch', 'cloth', 'texture']):
                im = rgb(asset(args, row, key), (128, 184))
                panel.paste(im, (x+col*128, y+22))
        name='dev64' if page<4 else 'reference_pool'
        panel.save(out/('inputs/%s_%d.png' % (name,offset+1)))
    print('PREPARED_DEV64', canonical_sha(rows), flush=True)


def qualifies(a, b, review):
    qa, qb = review[str(a['index'])], review[str(b['index'])]
    if not (qa['clean'] and qb['clean']):
        return None
    if qa['pattern'] not in ['stripes', 'checks', 'dots'] or qb['pattern'] not in ['stripes', 'checks', 'dots']:
        return None
    color, palette = color_distance(a['color'], b['color'])
    if color > 5. or palette > 10.:
        return None
    # 人眼核验的规律纹样才使用主峰代理，避免用褶皱/裁剪边缘作花型真值。
    pa, pb = a['spectral_proxy'], b['spectral_proxy']
    if not (pa['valid'] and pb['valid']):
        return None
    ratio = max(pa['period'], pb['period'])/min(pa['period'], pb['period'])
    angle = abs(pa['line_angle']-pb['line_angle'])
    angle = min(angle, 180-angle)
    family = qa['pattern'] != qb['pattern']
    if not family and ratio < 1.5 and not (qa['pattern'] == 'stripes' and angle >= 45):
        return None
    return dict(color_delta=color, palette_delta=palette, period_ratio=ratio,
                orientation_difference=angle, pattern_family_change=family)


def freeze_pairs(args):
    out = Path(args.out)
    source = read(out/'inputs/dev64.json')
    # 候选资格来自模型输出生成前的输入图板审阅；不可在出结果后更换。
    review_file = Path(args.review)
    review = read(review_file)['records']
    records = source['records'];refs=read(out/'inputs/reference_pool.json')['records']
    assert set(review) == {str(x['index']) for x in records+refs}
    reference_pairs=[]
    for ai,a in enumerate(refs):
        for b in refs[ai+1:]:
            score=qualifies(a,b,review)
            if score is not None:reference_pairs.append((score,a,b))
    reference_pairs=sorted(reference_pairs,key=lambda v:hashlib.sha256(('fixed_ref_pair/%s/%s'%(v[1]['index'],v[2]['index'])).encode()).hexdigest())[:32]
    targets=sorted(records,key=lambda x:hashlib.sha256(('followup_target/'+str(x['index'])).encode()).hexdigest())
    choices=[]
    for target,(score,a,b) in zip(targets,reference_pairs):
            choices.append(dict(case_id='real_%s'%target['index'],kind='real_reference_pair',index=target['index'],
                reference_indices=[a['index'],b['index']],
                caption=target['neutral_prompt'], sketch=target['sketch'], gt=target['cloth'],
                original_reference=a['texture'], swapped_reference=b['texture'],
                seed=110800+target['index'], qualification=score,
                hashes=dict(sketch=target['hashes']['sketch'], gt=target['hashes']['cloth'],
                    original_reference=a['hashes']['texture'], swapped_reference=b['hashes']['texture'])))
    choices = sorted(choices, key=lambda x: hashlib.sha256(('followup/'+str(x['index'])).encode()).hexdigest())[:32]
    controls=[]
    clean=[a for a in records if review[str(a['index'])]['clean'] and review[str(a['index'])]['pattern'] in ['stripes','checks','dots']][:8]
    for a in clean:
        original=rgb(Path(args.data)/a['texture'],None)
        transformations=['period2x']+(['rotate90'] if review[str(a['index'])]['pattern']=='stripes' else [])
        for transform in transformations:
            if transform=='rotate90':
                altered=original.transpose(Image.Transpose.ROTATE_90)
            else:
                tile=original.resize((original.width//2,original.height//2),Image.Resampling.BICUBIC)
                altered=Image.new('RGB',original.size)
                for y in [0,tile.height]:
                    for x in [0,tile.width]:altered.paste(tile,(x,y))
            mean_delta,palette_delta=color_distance(a['color'],color_summary(altered.resize(SIZE,Image.Resampling.BICUBIC)))
            if mean_delta>5 or palette_delta>10:continue
            name='controlled_refs/%s_%s.png'%(a['index'],transform)
            path=out/name;path.parent.mkdir(exist_ok=True);altered.save(path)
            controls.append(dict(case_id='%s_%s'%(transform,a['index']),kind='derived_'+transform,index=a['index'],
                caption=a['neutral_prompt'],sketch=a['sketch'],gt=a['cloth'],seed=110800+a['index'],
                original_reference=a['texture'],swapped_reference=name,swapped_reference_root='out',
                qualification=dict(color_delta=mean_delta,palette_delta=palette_delta,transform=transform),
                hashes=dict(sketch=a['hashes']['sketch'],gt=a['hashes']['cloth'],
                    original_reference=a['hashes']['texture'],swapped_reference=sha(path))))
    value = dict(protocol=PROTOCOL, input_registry_sha256=sha(out/'inputs/dev64.json'),
                 reference_registry_sha256=sha(out/'inputs/reference_pool.json'),
                 review_sha256=sha(review_file), review=read(review_file), pairs=choices+controls,
                 planned_maximum=32, actual_real_targets=len(choices),actual_derived_controls=len(controls),
                 sufficient_for_confirmatory_claim=False,
                 qualification_note='固定dev64的开发诊断；不作为新独立泛化测试，不补选低资格样本凑数')
    freeze(out/'strict_pairs.json', value)
    print('FROZEN_STRICT_TARGETS', len(choices),'DERIVED_CONTROLS',len(controls),flush=True)


def sketch_maps(sketch, mask):
    import cv2
    from skimage.morphology import skeletonize
    gray = np.asarray(sketch.convert('L'))
    line = 1-gray.astype(np.float32)/255.
    binary = line > 10/255.
    gx = cv2.Sobel(gray.astype(np.float32)/255., cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray.astype(np.float32)/255., cv2.CV_32F, 0, 1, ksize=3)
    sobel = np.clip(np.hypot(gx, gy)/4., 0, 1)
    canny = cv2.Canny(gray, 40, 120).astype(np.float32)/255.
    outer = mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3,3), np.uint8)).astype(bool)
    near_outer = cv2.dilate(outer.astype(np.uint8), np.ones((9,9), np.uint8)).astype(bool)
    skeleton = skeletonize(binary)
    internal = skeleton & mask & ~near_outer
    layer = np.ones((*gray.shape, 3), np.uint8)*255
    layer[outer] = [220, 30, 30]
    layer[internal] = [30, 60, 220]
    return dict(original_line=line, sobel=sobel, canny=canny,
                skeleton=skeleton.astype(np.float32), hierarchy=layer), internal


def offline(args):
    import cv2
    from eval.eval_utils import prepare_evaluation_masks
    from tools.e2_followup_pidinet import detector
    out = Path(args.out)
    records = read(out/'inputs/dev64.json')['records']
    pidinet, provenance = detector()
    stats, tiles = [], []
    for row in records:
        sketch = rgb(Path(args.data)/row['sketch'])
        mask = prepare_evaluation_masks(SIZE, sketch_path=str(Path(args.data)/row['sketch']), mask_policy='sketch_only')['garment']
        maps, internal = sketch_maps(sketch, mask)
        if pidinet is not None:
            maps['pidinet'] = pidinet(np.asarray(sketch))
        numbers = {}
        for name, value in maps.items():
            if value.ndim == 2:
                binary = value > .1 if name in ['sobel', 'pidinet'] else value > 10/255.
                n, _ = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
                numbers[name] = dict(pixel_n=int(binary.sum()), density=float(binary.mean()), components=n-1)
        stats.append(dict(index=row['index'], mask_area=float(mask.mean()), internal_skeleton_pixels=int(internal.sum()), maps=numbers))
        rendered = []
        for name, value in maps.items():
            arr = value if value.ndim == 3 else np.repeat(((1-value)*255).clip(0,255).astype(np.uint8)[...,None],3,axis=2)
            im = Image.fromarray(arr).resize((144,192), Image.Resampling.NEAREST)
            tile = Image.new('RGB', (144,214), 'white');tile.paste(im,(0,22))
            ImageDraw.Draw(tile).text((3,3), name, fill='black');rendered.append(tile)
        tiles.append((row['index'], rendered))
    for page in range(8):
        part = tiles[page*8:(page+1)*8]
        panel = Image.new('RGB', (len(part[0][1])*144+68,8*214), 'white')
        for y,(index, values) in enumerate(part):
            ImageDraw.Draw(panel).text((2,y*214+5),str(index),fill='black')
            for x,im in enumerate(values):panel.paste(im,(68+x*144,y*214))
        path=out/('a_offline/page_%d.png'%(page+1));path.parent.mkdir(exist_ok=True);panel.save(path)
    write(out/'a_offline/result.json',dict(records=stats, detector=provenance,
        hed=dict(status='not_run',reason='作者Caffe实现；本批不迁移训练框架',source='https://github.com/s9xie/hed'),
        thresholds=dict(line=10/255.,sobel=.1,canny=[40,120],pidinet=.1),
        quantitative_scope='输入一致性代理与连通性统计；无人工边缘真值，不是检测准确率',
        downstream_A_training_executed=False, gt_used_as_condition=False))
    print('OFFLINE_COMPLETE', len(stats), provenance['status'], flush=True)
