"""BF 真实原配对的完整 S0 开关 B 对照。"""
import argparse
import copy
import hashlib
import importlib.metadata as metadata
import inspect
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from eval.eval_utils import derive_region_masks, estimate_foreground_mask, prepare_evaluation_masks
from tools.e2_a_evaluation import boundary_f1
from tools.e2_fft_experiment import SOURCE, environment, read, sha
from tools.e2_oracle_texture import weights_digest


SIZE = (384, 512)
ARMS = ('off', 'on')
CATEGORIES = ('dress', 'outwear', 'pants', 'top')
METRICS = ('lpips', 'ssim', 'psnr', 'contour_f1', 'iou', 'leak')
CONDITION = 'original'
SEED = 42
CHECKPOINT_SHA = 'f68f22a106c3f0a08044c754b44c738ba3d6f67274edf5d281d5b54f39ddcd96'
B_HASH = '43346a544a1b82de7792e3b9a49c122a061333fd8b31e21f5e1b36ad9c3141f0'
NET_HASH = 'b846a8c58b4a25a62250e062a4bdf3f1302e19341c3d1903118e6d1c291fa0ca'
FID_WHEEL = 'vendor/pytorch_fid-0.3.0-py3-none-any.whl'
FID_WHEEL_SHA = '0e1d4fe66441a8dff2a2aca31394e9389fb197dfeff046f2f971f271d4137767'
CODE = ('tools/e2_f68_real_paired.py', 'scripts/e2_f68_real_paired.sbatch', 'tools/e2_state_b_pilot.py',
        'tools/e2_state_b_joint.py', 'tools/e2_state_b_joint_data.py', 'tools/e2_state_b_material_data.py',
        'tools/e2_state_b_repair.py', 'tools/e2_texture_bottleneck.py', 'models/e2_state_texture_b.py',
        'pipelines/IMAGGarment_pipeline.py', 'tools/e2_a_evaluation.py', 'eval/eval_utils.py',
        'eval/metrics.py', 'garment_mask_utils.py', 'tools/e2a_attr_eval.py', FID_WHEEL)


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def rgb(path):
    with Image.open(path) as image:
        return image.convert('RGB')


def choose_pairs(test, training):
    training_ids = {Path(row['cloth']).stem for row in training}
    selected = []
    ids = set(training_ids)
    for category in CATEGORIES:
        group = [r for r in test if r['sample_id'].split('/')[:2] == ['test', category]]
        group.sort(key=lambda r: hashlib.sha256(('f68-real-paired-v1:' + r['sample_id']).encode()).hexdigest())
        count = 0
        for row in group:
            product = Path(row['cloth']).stem
            if product in ids:
                continue
            assert all(Path(row[k]).stem == product for k in ('cloth', 'sketch', 'texture'))
            assert isinstance(row['caption'], str) and row['caption'].strip()
            ids.add(product)
            selected.append(dict(case_id=category+'_'+product, sample_id=row['sample_id'], category=category,
                                 sketch=row['sketch'], texture=row['texture'], gt=row['cloth'],
                                 caption=row['caption'], seed=SEED))
            count += 1
            if count == 125:
                break
        assert count == 125, category
    return selected


def metric_assets():
    import lpips
    assert sha(SOURCE/FID_WHEEL) == FID_WHEEL_SHA
    directory = Path(torch.hub.get_dir()) / 'checkpoints'
    paths = dict(alexnet=directory/'alexnet-owt-4df8aa71.pth',
                 lpips_calibration=Path(lpips.__file__).parent/'weights/v0.1/alex.pth',
                 inception=directory/'pt_inception-2015-12-05-6726825d.pth')
    hashes = {name: dict(path=str(path), sha256=sha(path)) for name, path in paths.items()}
    assert hashes['alexnet']['sha256'].startswith('4df8aa71')
    assert hashes['inception']['sha256'].startswith('6726825d')
    return hashes


def prepare(args):
    assert not (args.out/'protocol.json').exists()
    manifests = args.project/'data/processed/bf_full_audit_v1'
    test, train = (manifests/name for name in ('test_clean.json', 'training_clean.json'))
    manifest_sha = {str(p): sha(p) for p in (test, train)}
    pairs = choose_pairs(read(test), read(train))
    assets = metric_assets()
    args.out.mkdir(parents=True, exist_ok=True)
    for pair in pairs:
        folder = args.out/'cases'/pair['case_id']
        folder.mkdir(parents=True, exist_ok=True)
        pair['input_sha256'] = {n: sha(args.data/pair[n]) for n in ('sketch', 'texture', 'gt')}
        reference = folder/(CONDITION+'_ref.png')
        rgb(args.data/pair['texture']).resize(SIZE, Image.Resampling.BICUBIC).save(reference)
        pair['reference_hashes'] = {CONDITION: sha(reference)}
        bundle = prepare_evaluation_masks(SIZE, sketch_path=str(args.data/pair['sketch']), mask_policy='sketch_only')
        mask = bundle['garment']
        if mask is None:
            mask = np.zeros((SIZE[1], SIZE[0]), dtype=bool)
        Image.fromarray(mask.astype(np.uint8)*255).save(folder/'mask.png')
        pair['mask_sha256'] = sha(folder/'mask.png')
        pair['mask_valid'] = bool(mask.any())
        pair['mask_stats'] = bundle['stats']
    assert sha(args.checkpoint) == CHECKPOINT_SHA
    protocol = dict(pairs=pairs, input_manifest_sha256=manifest_sha,
        split='BF official test clean, four clothing categories;125 hash-ranked original pairs per category',
        identity_check='unique product stem and absent from current clean training manifest', selection_has_no_output_filter=True,
        checkpoint_sha256=CHECKPOINT_SHA, sampler=dict(steps=50, cfg=7., sketch_scale=.6, b_active=[25,50], size=SIZE, seed=SEED),
        comparison='same native raster generation, frozen B is the only arm change; true GT used only in scoring',
        gt_resize='384x512 bilinear RGB for LPIPS, FID, SSIM and PSNR; repository resizing for leak',
        metrics=dict(lpips='primary, AlexNet v0.1 full RGB [-1,1]', fid='same500 GT and same Inception pool3 backend for both arms',
                     ssim='RGB, data range255, standard scikit-image defaults', psnr='per-image RGB, data range255',
                     contour_f1='inner edge of estimated generated foreground versus fixed sketch mask, radius3px',
                     iou='estimated generated foreground versus fixed sketch mask', leak='repository leak_colored_frac, sketch mask kernel9, original GT'),
        metric_assets=assets, packages={n: metadata.version(n) for n in ('torch','pytorch-fid','scikit-image','lpips')},
        source_sha256={n: sha(SOURCE/n) for n in CODE}, environment=environment())
    write(args.out/'protocol.json', protocol)
    print('F68_REAL_PREPARED', len(pairs), 'valid_masks', sum(p['mask_valid'] for p in pairs), flush=True)


def verify(args):
    protocol = read(args.out/'protocol.json')
    assert all(sha(SOURCE/n) == value for n, value in protocol['source_sha256'].items())
    assert all(sha(Path(n)) == value for n, value in protocol['input_manifest_sha256'].items())
    return protocol


def row_files(args, pair):
    return {arm: args.out/'images'/arm/(pair['case_id']+'.png') for arm in ARMS}


@torch.no_grad()
def generate(args):
    from models.e2_state_texture_b import StateTextureB
    from tools import e2_state_b_joint as joint, e2_state_b_pilot as pilot
    from tools.e2_state_b_backward_repro import tensor_sha
    protocol = verify(args)
    selected = protocol['pairs'][args.start:args.end]
    assert selected and 0 <= args.start < args.end <= len(protocol['pairs'])
    assert sha(args.checkpoint) == CHECKPOINT_SHA
    runtime = copy.copy(args)
    runtime.action = 'train'  # 原 S0 诊断用这一入口配置确定性后端。
    core, pipe, net = pilot.load_core(runtime)
    pipe.vae.half().eval().requires_grad_(False)
    net.eval().requires_grad_(False)
    b = StateTextureB(color_channels=2, reference_channels=192).to(args.device)
    b.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=False)['b'])
    b.eval().requires_grad_(False)
    initial = {n: weights_digest(v) for n,v in [('B',b),('net',net),('vae',pipe.vae)]}
    assert initial['B'] == B_HASH and initial['net'] == NET_HASH
    assert type(pipe.scheduler).__name__ == 'DDIMScheduler'
    calls = []
    handle = b.register_forward_hook(lambda module, inputs, output: calls.append(1))
    protocol_sha = sha(args.out/'protocol.json')
    (args.out/'pairs').mkdir(exist_ok=True)
    for arm in ARMS:
        (args.out/'images'/arm).mkdir(parents=True, exist_ok=True)
    for index, pair in enumerate(selected, args.start):
        paths = row_files(args, pair)
        row_path = args.out/'pairs'/(pair['case_id']+'.json')
        if row_path.exists():
            previous = read(row_path)
            assert previous['protocol_sha256'] == protocol_sha
            assert all(sha(path) == previous['arms'][arm]['png_sha256'] for arm,path in paths.items())
            continue
        folder = args.out/'cases'/pair['case_id']
        assert sha(args.data/pair['sketch']) == pair['input_sha256']['sketch']
        assert sha(folder/'mask.png') == pair['mask_sha256']
        row = dict(case_id=pair['case_id'], index=index, seed=pair['seed'], protocol_sha256=protocol_sha, arms={})
        for arm in ARMS:
            calls.clear()
            picture, noise, tap = joint.sample(args, core, pipe, net, pair, CONDITION,
                                                b if arm == 'on' else None, mode='raw', capture=True)
            assert picture.size == SIZE and tap.index == 50 and len(tap.states) == 25
            assert len(calls) == (25 if arm == 'on' else 0)
            first = tap.states[0]
            assert first['index'] == 25
            prefix = {n: tensor_sha(first[n]) for n in ('zt','epsilon')}
            picture.save(paths[arm], format='PNG')
            row['arms'][arm] = dict(png_sha256=sha(paths[arm]), initial_noise_digest=noise, first_b_step=prefix,
                                    steps=tap.index, b_calls=len(calls))
        assert row['arms']['off']['initial_noise_digest'] == row['arms']['on']['initial_noise_digest']
        assert row['arms']['off']['first_b_step'] == row['arms']['on']['first_b_step']
        write(row_path, row)
        print('F68_REAL_PAIR', index+1, pair['case_id'], flush=True)
    handle.remove()
    assert initial == {n: weights_digest(v) for n,v in [('B',b),('net',net),('vae',pipe.vae)]}
    assert sha(args.checkpoint) == CHECKPOINT_SHA
    verify(args)
    write(args.out/('generated_%d_%d.json'%(args.start,args.end)), dict(start=args.start,end=args.end,
        protocol_sha256=protocol_sha, hashes=initial, frozen_unchanged=True, environment=environment(),
        scheduler_sha256=sha(Path(inspect.getsourcefile(type(pipe.scheduler)))), optimizer_updates=0))


def paired_interval(delta, categories):
    rng = np.random.default_rng(20261118)
    # 保持四类在重抽样中的固定比例，按身份抽取开关配对。
    blocks = [delta[categories == name] for name in CATEGORIES if np.any(categories == name)]
    totals = sum(v[rng.integers(0, len(v), (10000,len(v)))].sum(1) for v in blocks)
    return [float(v) for v in np.quantile(totals/len(delta), [.025,.975])]


def fid_from_pool3(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    x = (a-a.mean(0))/np.sqrt(len(a)-1)
    y = (b-b.mean(0))/np.sqrt(len(b)-1)
    # 非零协方差乘积特征值的平方根是样本交叉矩阵的奇异值。
    cross = np.linalg.svd(x@y.T, compute_uv=False).sum()
    value = np.square(a.mean(0)-b.mean(0)).sum()+np.square(x).sum()+np.square(y).sum()-2*cross
    assert value >= -1e-7
    return float(max(value,0))


def metric_models(device, assets):
    import lpips
    from pytorch_fid.inception import InceptionV3
    model = lpips.LPIPS(net='alex', pnet_rand=True, model_path=assets['lpips_calibration']['path'])
    pretrained = torch.load(assets['alexnet']['path'], map_location='cpu', weights_only=False)
    model.net.load_state_dict({k: pretrained['features.'+k.split('.',1)[1]] for k in model.net.state_dict()}, strict=True)
    inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]])
    return model.eval().requires_grad_(False).to(device), inception.eval().requires_grad_(False).to(device)


def image_scores(picture, target, path, target_path, mask):
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    from eval.metrics import compute_texture_leakage
    a,b = np.asarray(picture),np.asarray(target)
    exact = bool(np.array_equal(a,b))
    values = dict(ssim=float(structural_similarity(a,b,data_range=255,channel_axis=2)),
                  psnr=None if exact else float(peak_signal_noise_ratio(b,a,data_range=255)),
                  psnr_infinite_identical=exact, contour_f1=None,iou=None,leak=None)
    if mask.any():
        predicted = estimate_foreground_mask(picture, SIZE)
        if predicted is None:
            predicted = np.zeros_like(mask)
        values.update(contour_f1=boundary_f1(predicted,mask,3), iou=float((predicted & mask).sum()/(predicted | mask).sum()))
        outside,boundary = derive_region_masks(mask,kernel_size=9)
        if outside.sum() >= 50:
            scores = compute_texture_leakage(str(path), target_path=str(target_path),
                                             mask_bundle=dict(outside=outside,boundary=boundary))
            values['leak'] = scores['leak_colored_frac']
    return values


@torch.no_grad()
def evaluate(args):
    protocol = verify(args)
    assert metric_assets() == protocol['metric_assets']
    protocol_sha = sha(args.out/'protocol.json')
    assert not (args.out/'review.zip').exists()
    covers, frozen_hashes, scheduler_hashes = [], [], set()
    for file in sorted(args.out.glob('generated_*.json')):
        job = read(file)
        assert job['frozen_unchanged'] and job['protocol_sha256'] == protocol_sha and job['optimizer_updates'] == 0
        covers.extend(range(job['start'],job['end']))
        frozen_hashes.append(job['hashes'])
        scheduler_hashes.add(job['scheduler_sha256'])
    pairs = protocol['pairs']
    assert sorted(covers) == list(range(len(pairs)))
    assert all(v == frozen_hashes[0] for v in frozen_hashes) and len(scheduler_hashes) == 1
    model, inception = metric_models(args.device,protocol['metric_assets'])
    feature_batches = {k: [] for k in (*ARMS,'gt')}
    tensor = lambda images: torch.from_numpy(np.stack([np.asarray(im) for im in images])).permute(0,3,1,2).contiguous().to(args.device)
    rows = []
    for offset in range(0,len(pairs),4):
        batch = {k: [] for k in (*ARMS,'gt')}
        for pair in pairs[offset:offset+4]:
            row = read(args.out/'pairs'/(pair['case_id']+'.json'))
            assert row['protocol_sha256'] == protocol_sha and row['index'] == len(rows)
            assert row['arms']['off']['initial_noise_digest'] == row['arms']['on']['initial_noise_digest']
            assert row['arms']['off']['first_b_step'] == row['arms']['on']['first_b_step']
            assert all(row['arms'][a]['steps']==50 for a in ARMS) and row['arms']['on']['b_calls']==25 and row['arms']['off']['b_calls']==0
            target_path = args.data/pair['gt']
            assert sha(target_path) == pair['input_sha256']['gt']
            target = rgb(target_path).resize(SIZE,Image.Resampling.BILINEAR)
            mask_file = args.out/'cases'/pair['case_id']/'mask.png'
            assert sha(mask_file) == pair['mask_sha256']
            mask = np.asarray(rgb(mask_file))[...,0]>0
            row.update(category=pair['category'], mask_valid=pair['mask_valid'], scores={})
            batch['gt'].append(target)
            for arm,path in row_files(args,pair).items():
                assert sha(path) == row['arms'][arm]['png_sha256']
                picture = rgb(path)
                batch[arm].append(picture)
                row['scores'][arm] = image_scores(picture,target,path,target_path,mask)
            rows.append(row)
        count = len(batch['gt'])
        target_tensor = tensor(batch['gt']).float()/127.5-1
        for arm in ARMS:
            value = model(tensor(batch[arm]).float()/127.5-1,target_tensor).flatten().cpu().tolist()
            assert len(value) == count and np.isfinite(value).all()
            for row,score in zip(rows[-count:],value):row['scores'][arm]['lpips'] = score
        for key,images in batch.items():
            feature = inception(tensor(images).float()/255)[0].flatten(1).cpu().numpy()
            assert feature.shape == (count,2048) and np.isfinite(feature).all()
            feature_batches[key].append(feature)
        if offset%40==0:print('F68_REAL_METRICS',len(rows),flush=True)
    features = {k: np.concatenate(v) for k,v in feature_batches.items()}
    np.savez_compressed(args.out/'fid_pool3.npz',**features)
    fid = {a: fid_from_pool3(features[a],features['gt']) for a in ARMS}
    summary = {}
    for name in METRICS:
        valid = [r for r in rows if all(r['scores'][a][name] is not None for a in ARMS)]
        x,y = [np.array([r['scores'][a][name] for r in valid]) for a in ARMS]
        delta = y-x
        assert len(delta)>0 and np.isfinite(delta).all(),name
        summary[name] = dict(n=len(valid), off=float(x.mean()), on=float(y.mean()), delta=float(delta.mean()),
                             ci95=paired_interval(delta,np.array([r['category'] for r in valid])),
                             on_less=int((delta<0).sum()), on_greater=int((delta>0).sum()), equal=int((delta==0).sum()))
    write(args.out/'per_image.json', rows)
    write(args.out/'result.json',dict(complete=True,conditions=len(rows),generated_images=2*len(rows),summary=summary,
        fid=dict(**fid,delta=fid['on']-fid['off'],n=len(rows),backend='pytorch-fid 0.3.0 Inception pool3 2048 and float64 sample-space SVD'),
        mask_valid=sum(r['mask_valid'] for r in rows), frozen_model_hashes=frozen_hashes[0],
        scheduler_source_sha256=next(iter(scheduler_hashes)), original_ground_truth_used_for_generation=False,
        protocol_sha256=protocol_sha, optimizer_updates=0, environment=environment()))
    files = [args.out/n for n in ['protocol.json','result.json','per_image.json','fid_pool3.npz']]
    files += sorted(args.out.glob('generated_*.json'))
    rank = sorted(rows,key=lambda r:r['scores']['on']['lpips']-r['scores']['off']['lpips'])
    for row in rank[:3]+rank[-3:]:
        files.extend(row_files(args,row).values())
    manifest = {p.relative_to(args.out).as_posix():sha(p) for p in files}
    write(args.out/'review_manifest.json',manifest)
    temporary = args.out/'review.zip.tmp'
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in files+[args.out/'review_manifest.json']:
            archive.write(path,path.relative_to(args.out).as_posix())
    temporary.replace(args.out/'review.zip')
    print('F68_REAL_COMPLETE',sha(args.out/'review.zip'),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare','generate','evaluate'))
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--project',type=Path,default=Path('/share/home/u2515283058/Mymodel'))
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--end',type=int,default=500)
    args = parser.parse_args()
    args.out = args.prior/'real_paired_B_on_off_20261118'
    args.data = args.project.parent/'datasets/BF'
    args.checkpoint = args.project/'output_eval/e2_state_b_pastel_followup_20261115/trust_retry/training/step1.pt'
    torch.set_num_threads(4 if args.action=='evaluate' else 2)
    cv2.setNumThreads(1)
    dict(prepare=prepare,generate=generate,evaluate=evaluate)[args.action](args)
