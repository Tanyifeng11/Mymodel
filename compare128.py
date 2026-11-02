"""官方生成器推理复现：固定草图+参考局部块；GT仅供独立评价。"""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from skimage.color import rgb2lab, lab2rgb

HERE = Path(__file__).resolve().parent
HOME = Path('/share/home/u2515283058')
WEIGHT_SHA = '37a10d847feb1f1516a9ff9606162eed0e41bfce974b05121edcd27ff96f2d64'
UPSTREAM = '8b20e678c482c216a816569087aaea247b2c2ebb'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(path, obj):
    if Path(path).exists():
        assert read(path) == obj, '冻结文件不一致: '+str(path)
    else:
        write(path, obj)


def load_model(weights, device):
    assert sha(weights) == WEIGHT_SHA, '权重SHA不匹配'
    state = torch.load(weights, map_location='cpu', weights_only=True)
    assert state['image_size'] == 256 and state['color_space'] == 'lab'
    spec = importlib.util.spec_from_file_location('upstream_texturegan', HERE/'models/texturegan.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.TextureGAN(5, 3, 32)
    model.load_state_dict(state['state_dict'], strict=True)
    return model.to(device).eval(), {k: v for k, v in state.items() if k != 'state_dict'}


def normalized_lab(image):
    arr = rgb2lab(np.asarray(image.convert('RGB'), dtype=np.float64)/255)
    return ((arr-np.array([50, 0, 0]))/np.array([50, 128, 128])).astype(np.float32)


def build_input(row, source):
    # 仅读取两项推理输入；不打开GT，也不以GT生成mask或选择纹理。
    from eval.eval_utils import sketch_to_garment_mask
    sketch_path, reference_path = Path(row['sketch']), Path(row['reference'])
    assert sha(sketch_path) == row['sketch_sha256']
    assert sha(reference_path) == row['reference_sha256']
    sketch = Image.open(sketch_path).convert('RGB').resize((256, 256), Image.Resampling.BILINEAR)
    ref = Image.open(reference_path).convert('RGB')
    mask, info = sketch_to_garment_mask(sketch, (256, 256), return_info=True)
    assert mask is not None and mask.any(), row['id']
    import cv2
    distance = cv2.distanceTransform(np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    if info['mask_low_confidence'] or distance.max() < 16:
        # 开口轮廓可能只被旧flood-fill识别为线条；仅放置提示时用草图墨迹凸包。
        # 评价仍完全使用原冻结mask协议，不随这个推理接口变化。
        points = np.argwhere(np.asarray(sketch.convert('L')) < 220)[:, ::-1].astype(np.int32)
        hull_mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.fillConvexPoly(hull_mask, cv2.convexHull(points), 1)
        distance = cv2.distanceTransform(np.pad(hull_mask, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
        info = dict(info, hint_only_fallback='sketch_ink_convex_hull')
    rng = np.random.default_rng(row['seed'])
    patches = []
    data = np.ones((256, 256, 5), dtype=np.float32)
    data[:, :, 0] = normalized_lab(sketch)[:, :, 0]
    data[:, :, 4] = -1
    # 预设两块、边长20—40；原代码的未提示Lab三个通道均为+1，mask=-1。
    # 从参考图独立采块后放入草图内部，这是自动接口适配，不是原论文人工拖放。
    for _ in range(2):
        size = int(rng.integers(20, 41))
        candidates = np.argwhere(distance > size*.72+1)
        if not len(candidates):
            size = min(size, int((float(distance.max())-1)/.72)-1)
            if size < 4:
                raise ValueError('草图内部过窄，不能放置纹理块: '+row['id'])
            candidates = np.argwhere(distance > size*.72+1)
        assert len(candidates), row['id']
        y, x = map(int, candidates[int(rng.integers(len(candidates)))])
        left, top = x-size//2, y-size//2
        # 保持参考原像素尺度；不足时仅放大至能采样，禁止整幅拼贴成结果。
        scale = max(1., size/ref.width, size/ref.height)
        sample = ref.resize((int(np.ceil(ref.width*scale)), int(np.ceil(ref.height*scale))), Image.Resampling.BILINEAR)
        sx = int(rng.integers(0, sample.width-size+1))
        sy = int(rng.integers(0, sample.height-size+1))
        patch = sample.crop((sx, sy, sx+size, sy+size))
        data[top:top+size, left:left+size, 1:4] = normalized_lab(patch)
        data[top:top+size, left:left+size, 4] = 1
        patches.append(dict(x=left, y=top, size=size, reference_x=sx, reference_y=sy, reference_scale=scale))
    tensor = torch.from_numpy(data.transpose(2, 0, 1).copy())[None]
    assert tuple(tensor.shape) == (1, 5, 256, 256) and torch.isfinite(tensor).all()
    return tensor, patches, info


def prepare(args):
    assert (args.source/'tools/garment_test128.py').is_file()
    rows = read(args.previous/'protocol/input_manifest.json')
    assert len(rows) == 128 and len({r['id'] for r in rows}) == 128
    # 本地和服务器都重用服务器冻结的输入字节；不误用本地同名但不同版本的草图。
    for row in rows:
        for key in ['reference', 'sketch', 'gt']:
            path = args.previous/'inputs'/Path(row[key]).name
            assert sha(path) == row[key+'_sha256']
            row[key] = str(path.resolve())
    freeze(args.root/'protocol/input_manifest.json', rows)
    protocol = dict(model='TextureGAN', n=128, training=False, input_resolution=[256, 256],
        output_resolution=[256, 256], evaluation_resolution=[384, 512],
        modalities=['sketch', 'reference_texture'], text_used=False, GT_used_in_inference=False,
        hint_policy='2 reference-native crops, side uniform integer 20..40; positions from fixed-seed sketch-only interior; Lab defaults all +1 and loc mask -1 as upstream',
        seed_policy='same identity seed as frozen128; selects hints, not diffusion noise',
        mask_source='frozen Mymodel sketch_to_garment_mask; hint-only sketch ink convex hull if low confidence or narrow; no GT mask; evaluation masks unchanged',
        alignment='PyTorch0.2 bilinear corner aligned => align_corners=True',
        weights_sha256=WEIGHT_SHA, upstream_commit=UPSTREAM,
        provenance=read(HERE/'weight_provenance.json'),
        inference_scope='full official pretrained generator + automated reference hint interface; not training reproduction',
        comparison_limits=['no text input', 'unknown historical pretraining identity overlap',
            'mirror artifact; identity against deleted official S3 cannot be verified',
            'automated patch placement differs from original manual dragging'],
        metrics='same original128 Edge F1/Leak/IoU/Edge L1/LPIPS/TCF LAB implementation; no swapped reference',
        gpu_limit_seconds=3600)
    freeze(args.root/'protocol/controlled_protocol.json', protocol)
    print('PREPARED', len(rows), sha(args.root/'protocol/input_manifest.json'), flush=True)


@torch.inference_mode()
def generate(args):
    sys.path.insert(0, str(args.source))
    import platform
    import skimage
    import cv2
    write(args.root/'audit/environment.json', dict(python=sys.version, platform=platform.platform(),
        torch=torch.__version__, numpy=np.__version__, skimage=skimage.__version__, opencv=cv2.__version__,
        cuda=torch.version.cuda, device=args.device, git_commit=subprocess.check_output(
            ['git', '-C', str(HERE), 'rev-parse', 'HEAD'], text=True).strip()))
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model, state_meta = load_model(args.weights, args.device)
    rows = read(args.root/'protocol/input_manifest.json')
    if args.smoke:
        rows = [next(r for r in rows if r['category'] == c) for c in ['top', 'outwear', 'pants', 'dress']]
    for row in rows:
        output = args.root/'outputs/TextureGAN/original'/(row['id']+'.png')
        native = args.root/'native/TextureGAN/original'/output.name
        meta_path = output.with_suffix('.json')
        if meta_path.exists():
            old = read(meta_path)
            assert old['checkpoint_sha256'] == WEIGHT_SHA and old['output_sha256'] == sha(output)
            assert old['input_sketch_sha256'] == row['sketch_sha256']
            continue
        tensor, patches, mask_info = build_input(row, args.source)
        start = time.monotonic()
        result = model(tensor.to(args.device))
        if args.device == 'cuda':
            torch.cuda.synchronize()
        seconds = time.monotonic()-start
        assert torch.isfinite(result).all()
        lab = result[0].cpu().numpy().transpose(1, 2, 0)*np.array([50, 128, 128])+np.array([50, 0, 0])
        rgb = np.clip(lab2rgb(lab), 0, 1)
        image = Image.fromarray(np.rint(rgb*255).astype(np.uint8))
        native.parent.mkdir(parents=True, exist_ok=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(native)
        image.resize((384, 512), Image.Resampling.BILINEAR).save(output)
        Image.fromarray(np.rint(np.clip((tensor[0, 0].numpy()+1)*.5, 0, 1)*255).astype(np.uint8)).save(native.with_name(row['id']+'_sketch.png'))
        hint_lab = tensor[0, 1:4].numpy().transpose(1, 2, 0)*np.array([50, 128, 128])+np.array([50, 0, 0])
        hint_rgb = np.clip(lab2rgb(hint_lab), 0, 1)
        hint_view = np.repeat(np.clip((tensor[0, 0].numpy()+1)*.5, 0, 1)[:, :, None], 3, axis=2)
        hint_mask = tensor[0, 4].numpy() > 0
        hint_view[hint_mask] = hint_rgb[hint_mask]
        Image.fromarray(np.rint(hint_view*255).astype(np.uint8)).save(native.with_name(row['id']+'_hints.png'))
        write(meta_path, dict(id=row['id'], arm='TextureGAN', seed=row['seed'],
            checkpoint_sha256=WEIGHT_SHA, checkpoint_metadata=state_meta, parameters=sum(p.numel() for p in model.parameters()),
            input_sketch_sha256=row['sketch_sha256'], input_reference_sha256=row['reference_sha256'],
            input_tensor_sha256=hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
            GT_used_in_inference=False, sketch_used=True, reference_texture_used=True, text_used=False,
            native_size=[256, 256], evaluation_size=[384, 512], inference_seconds=seconds,
            hint_patches=patches, mask_info=mask_info, output_sha256=sha(output), native_sha256=sha(native)))
        print('GENERATED', row['id'], round(seconds, 3), flush=True)
    write(args.root/'audit'/('smoke_complete.json' if args.smoke else 'generation_complete.json'),
        dict(n=len(rows), weights_sha256=WEIGHT_SHA, device=args.device, torch=torch.__version__,
            code_commit=subprocess.check_output(['git', '-C', str(HERE), 'rev-parse', 'HEAD'], text=True).strip()))


def evaluate(args):
    sys.path.insert(0, str(args.source))
    from tools import t3_p0p1_evaluate as legacy
    legacy.ROOT = args.root
    legacy.output_path = lambda arm, row: args.root/'outputs'/arm/row['condition']/(row['id']+'.png')
    legacy.evaluate('TextureGAN', include_swap=True)


def resources(args):
    ledger = read(args.root/'audit/jobs.json') if (args.root/'audit/jobs.json').exists() else []
    total = 0
    records = []
    for job in ledger:
        raw = subprocess.check_output(['sacct', '-j', str(job['id']), '-X', '-n', '-P',
            '-o', 'JobIDRaw,State,ElapsedRaw,AllocTRES'], text=True).strip()
        line = next((x.split('|') for x in raw.splitlines() if x.split('|')[0] == str(job['id'])), None)
        assert line is not None, raw
        state, elapsed, tres = line[1], int(line[2]), line[3]
        gpu = next((int(x.split('=')[1]) for x in tres.split(',') if x.startswith('gres/gpu=')), 0)
        if elapsed and state != 'PENDING':
            assert gpu == job['gpus'], '无法核实GPU分配量: '+raw
        used = elapsed*gpu
        total += used if state in ['COMPLETED', 'FAILED', 'TIMEOUT', 'OUT_OF_MEMORY'] or state.startswith('CANCELLED') else job['limit_seconds']
        records.append(dict(job=job['id'], state=state, elapsed_seconds=elapsed, gpu_count=gpu, actual_gpu_seconds=used))
    result = dict(jobs=records, maximum_accounted_gpu_seconds=total, limit_gpu_seconds=3600)
    write(args.root/'audit/resources.json', result)
    assert total <= 3600
    return result


def submit(args):
    resources(args)
    ledger = read(args.root/'audit/jobs.json') if (args.root/'audit/jobs.json').exists() else []
    assert not ledger, '已有作业记录，先核查而非重复提交'
    assert sha(args.weights) == WEIGHT_SHA
    (args.root/'logs').mkdir(parents=True, exist_ok=True)
    command = ['sbatch', '--parsable', '-N1', '--ntasks=1', '--partition=gpu', '--gpus=1',
        '--cpus-per-task=4', '--mem=24G', '--time=00:30:00', '--job-name=texturegan128',
        '--output='+str(args.root/'logs/run_%j.out'), str(HERE/'run128.sbatch')]
    ident = int(subprocess.check_output(command, text=True).strip().split(';')[0])
    write(args.root/'audit/jobs.json', [dict(id=ident, gpus=1, limit_seconds=1800, command=command)])
    print('SUBMITTED', ident, flush=True)


def package(args):
    resource = resources(args)
    assert all(j['state'] == 'COMPLETED' for j in resource['jobs'])
    complete = read(args.root/'metrics/TextureGAN/complete.json')
    assert complete['n'] == 128 and complete['missing_generation'] == 0
    for row in read(args.root/'protocol/input_manifest.json'):
        p = args.root/'outputs/TextureGAN/original'/(row['id']+'.png')
        assert read(p.with_suffix('.json'))['output_sha256'] == sha(p)
    # 原始输入沿用上一批，附包使下载后可单独复核和展示。
    for row in read(args.root/'protocol/input_manifest.json'):
        for key in ['reference', 'sketch', 'gt']:
            target = args.root/'inputs'/Path(row[key]).name
            target.parent.mkdir(exist_ok=True)
            if not target.exists():
                shutil.copy2(row[key], target)
    for name in ['compare128.py', 'run128.sbatch', 'weight_provenance.json', 'models/texturegan.py']:
        p = args.root/'code'/name
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(HERE/name, p)
    files = sorted(p for folder in ['audit', 'protocol', 'outputs', 'native', 'metrics', 'logs', 'inputs', 'code']
        for p in (args.root/folder).rglob('*') if p.is_file() and p.name != 'bundle_sha256sums.txt')
    sums = args.root/'audit/bundle_sha256sums.txt'
    sums.write_text(''.join(sha(p)+'  '+p.relative_to(args.root).as_posix()+'\n' for p in files))
    target = args.root/'review_results.tar.gz'
    with tarfile.open(target, 'w:gz') as tar:
        for p in files+[sums]:
            tar.add(p, arcname=p.relative_to(args.root).as_posix(), recursive=False)
    print('BUNDLE', target.stat().st_size, sha(target), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare', 'generate', 'evaluate', 'submit', 'resources', 'package'])
    p.add_argument('--source', type=Path, default=HOME/'Mymodel/output_eval/t3_p0_p1_20261028/source')
    p.add_argument('--previous', type=Path, default=HOME/'Mymodel/output_eval/garment_test128_20261031')
    p.add_argument('--root', type=Path, default=HOME/'Mymodel/output_eval/texturegan_comparison_20261102')
    p.add_argument('--weights', type=Path, default=HERE/'weights/final_cloth_finetune.pth')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    result = globals()[args.action](args)
    if result is not None:
        print(json.dumps(result), flush=True)
