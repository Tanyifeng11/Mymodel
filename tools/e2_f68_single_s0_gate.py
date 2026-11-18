"""冻结 f68 的单次 S0 残差门控，与保存的三条基线作 64 对配对比较。"""
import argparse
import inspect
import os
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

from tools.e2_f68_edge_residual import LOWER_BETTER, mix_pixels, verified_files
from tools.e2_f68_real_paired import (ARMS, CATEGORIES, CHECKPOINT_SHA, CONDITION, METRICS, SIZE,
                                     image_scores, metric_assets, metric_models, rgb, verify, write)
from tools.e2_f68_region_diagnostic import ARCHIVE_SHA, checked_case, fid_terms, gaussian, paired, tensors
from tools.e2_fft_experiment import environment, read, sha
from tools.e2_oracle_texture import weights_digest


SUBSET_SIZE = 16
RADIUS = 24
ALPHA = .25
BLEND_ARCHIVE_SHA = '2a39cc9e7194f7a9ad7731e93b8d455f155e2b29b99283d30e2411ecdefc00ca'
CONTROLS = ARMS + ('blend',)
ALL_ARMS = CONTROLS + ('gate',)


def subset(protocol):
    pairs = protocol['pairs']
    picks = []
    for category in CATEGORIES:
        indices = [i for i, pair in enumerate(pairs) if pair['category'] == category]
        picks.extend(indices[:SUBSET_SIZE])
    assert len(picks) == len(set(picks)) == SUBSET_SIZE * len(CATEGORIES)
    assert all(pairs[i]['seed'] == 42 for i in picks)
    return picks


def latent_gate(mask, device):
    distance = cv2.distanceTransform(mask.astype('uint8'), cv2.DIST_L2, 5)
    local = gaussian((distance <= RADIUS).astype('float32'), 2)
    factor = ALPHA + (1 - ALPHA) * np.clip(local, 0, 1)
    return F.avg_pool2d(torch.from_numpy(factor)[None, None].to(device), 8).clamp(ALPHA, 1)


def contract(baseline, protocol):
    return dict(indices=subset(protocol), per_category=SUBSET_SIZE, seed=42,
                original_protocol_sha256=sha(baseline/'protocol.json'), source_sha256=sha(Path(__file__)),
                sampler=protocol['sampler'], checkpoint_sha256=CHECKPOINT_SHA,
                gate=dict(hook='multiply existing masked B residual before S0 epsilon conversion',
                          lower_scale=ALPHA, near_edge_scale=1, original_feather_px=[8,16],
                          gate_change_at_inward_distance_px=RADIUS, feather_sigma_px=2, latent_pool=8))


@torch.no_grad()
def generate(args, baseline, out, protocol, run_contract, previous, original):
    from models.e2_state_texture_b import StateTextureB
    from tools import e2_state_b_joint as joint, e2_state_b_pilot as pilot
    from tools.e2_state_b_backward_repro import tensor_sha
    assert sha(args.checkpoint) == CHECKPOINT_SHA
    # 沿用原开关生成的确定性后端配置。此处仅调用推理。
    args.action = 'train'
    core, pipe, net = pilot.load_core(args)
    pipe.vae.half().eval().requires_grad_(False)
    net.eval().requires_grad_(False)
    b = StateTextureB(color_channels=2,reference_channels=192).to(args.device)
    b.load_state_dict(torch.load(args.checkpoint,map_location='cpu',weights_only=False)['b'])
    b.eval().requires_grad_(False)
    models = {'B':b,'net':net,'vae':pipe.vae}
    frozen = {k: weights_digest(v) for k,v in models.items()}
    assert frozen == original['frozen_model_hashes']
    assert type(pipe.scheduler).__name__ == 'DDIMScheduler'
    scheduler = sha(Path(inspect.getsourcefile(type(pipe.scheduler))))
    assert scheduler == original['scheduler_source_sha256']
    pngs = out/'images'
    rows = out/'pairs'
    pngs.mkdir(exist_ok=True)
    rows.mkdir(exist_ok=True)
    ids = run_contract['indices'][args.shard::2]
    for index in ids:
        p = protocol['pairs'][index]
        controls = previous[index]
        assert p['case_id'] == controls['case_id'] and controls['index'] == index
        case = p['case_id']
        png, metadata = pngs/(case+'.png'),rows/(case+'.json')
        if metadata.exists():
            prior = read(metadata)
            assert prior['contract'] == run_contract and prior['png_sha256'] == sha(png)
            print('S0_GATE_REUSED', index, case, flush=True)
            continue
        folder = baseline/'cases'/case
        assert sha(args.data/p['sketch']) == p['input_sha256']['sketch']
        assert sha(folder/'mask.png') == p['mask_sha256']
        gate = latent_gate(np.asarray(rgb(folder/'mask.png'))[...,0]>0,args.device)
        calls = []

        def apply_gate(module, inputs, delta):
            scaled = delta * gate
            calls.append(torch.stack((delta.square().sum(), scaled.square().sum())))
            return scaled

        handle = b.register_forward_hook(apply_gate)
        try:
            picture, noise, tap = joint.sample(args, core, pipe, net, p, CONDITION, b, mode='raw', capture=True)
        finally:
            handle.remove()
        assert picture.size == SIZE and tap.index == 50 and len(tap.states) == len(calls) == 25
        first = tap.states[0]
        assert first['index'] == 25
        prefix = {name: tensor_sha(first[name]) for name in ('zt','epsilon')}
        for arm in ARMS:
            assert noise == controls['arms'][arm]['initial_noise_digest']
            assert prefix == controls['arms'][arm]['first_b_step']
        norms = torch.stack(calls).sqrt().cpu().numpy()
        assert np.isfinite(norms).all()
        assert np.all(norms[:,1] <= norms[:,0] + 1e-6)
        picture.save(png,format='PNG')
        write(metadata,dict(index=index,case_id=case,contract=run_contract,seed=p['seed'],
                            initial_noise_digest=noise,first_b_step=prefix,steps=50,b_calls=25,
                            png_sha256=sha(png),raw_and_gated_residual_l2=norms.tolist()))
        print('S0_GATE_PAIR', index, case, flush=True)
    assert frozen == {k: weights_digest(v) for k,v in models.items()}
    assert sha(args.checkpoint) == CHECKPOINT_SHA
    write(out/f'shard{args.shard}.json',dict(indices=ids,contract=run_contract,hashes=frozen,
        scheduler_sha256=scheduler,frozen_unchanged=True,optimizer_updates=0,environment=environment()))
    print('S0_GATE_SHARD_COMPLETE',args.shard,len(ids),flush=True)


def compare(rows, features):
    labels = np.array([r['category'] for r in rows])
    groups = {}
    for group, keep in (('all', np.ones(len(rows),bool)),('outwear',labels=='outwear')):
        selected = [r for r, ok in zip(rows,keep) if ok]
        result = {name: dict(fid=fid_terms(features[name][keep],features['gt'][keep]),scores={})
                  for name in ALL_ARMS}
        for name in ALL_ARMS:
            for metric in METRICS:
                v = np.array([r['scores'][name][metric] for r in selected],dtype='float64')
                current = dict(mean=float(np.nanmean(v)), n=int(np.isfinite(v).sum()),comparisons={})
                for control in CONTROLS:
                    ref = np.array([r['scores'][control][metric] for r in selected],dtype='float64')
                    change = v-ref
                    direction = -change if metric in LOWER_BETTER else change
                    diff = paired(ref,v,labels[keep])
                    diff.update(improved=int((direction>0).sum()),worse=int((direction<0).sum()),
                                equal=int((direction==0).sum()))
                    current['comparisons'][control] = diff
                result[name]['scores'][metric] = current
            result[name]['fid']['delta'] = {a:result[name]['fid']['fid']-result[a]['fid']['fid'] for a in CONTROLS}
        groups[group] = dict(n=len(selected),arms=result)
    sensitivity = {}
    for category in CATEGORIES:
        keep = labels != category
        fids = {a:fid_terms(features[a][keep],features['gt'][keep])['fid'] for a in ALL_ARMS}
        sensitivity[category] = {a:fids[a]-fids['off'] for a in ALL_ARMS}
    return dict(groups=groups,fid_exclude_category_delta_vs_off=sensitivity)


def example_panel(out, baseline, args, protocol, prior, rows):
    order = sorted(rows,key=lambda r:r['scores']['gate']['lpips']-r['scores']['off']['lpips'])
    panel = Image.new('RGB',(5*168,3*242),'white')
    draw = ImageDraw.Draw(panel)
    for j,row in enumerate((order[0],order[len(order)//2],order[-1])):
        index, case = row['index'], row['case_id']
        raw, mask = checked_case(baseline,args.data,protocol['pairs'][index],prior[index])
        blend = mix_pixels(raw['off'],raw['on'],mask)['edge_a025']
        gate = np.asarray(rgb(out/'images'/(case+'.png')))
        draw.text((3,j*242+2),case,fill='black')
        for k,(im,label) in enumerate(zip((raw['gt'],raw['off'],raw['on'],blend,gate),
                                         ('GT','B off','B on','2-output blend','1-pass gate'))):
            panel.paste(Image.fromarray(im).resize((156,208),Image.Resampling.BILINEAR),(k*168,j*242+16))
            draw.text((k*168+3,j*242+226),label,fill='black')
    panel.save(out/'examples.png')


@torch.no_grad()
def evaluate(args,baseline,out,protocol,run_contract,previous,original):
    p = baseline/'edge_residual_acceptance_20261118'
    verified_files(p,BLEND_ARCHIVE_SHA,('per_pair.json','inner_mix_pool3.npz','result.json'))
    old_blend = read(p/'per_pair.json')
    assert metric_assets()==protocol['metric_assets']
    ids = run_contract['indices']
    for shard in (0,1):
        marker = read(out/f'shard{shard}.json')
        assert marker['indices']==ids[shard::2] and marker['contract']==run_contract
        assert marker['frozen_unchanged'] and marker['optimizer_updates']==0
        assert marker['hashes']==original['frozen_model_hashes']
        assert marker['scheduler_sha256']==original['scheduler_source_sha256']
    model, inception = metric_models('cpu',protocol['metric_assets'])
    feature_groups, rows = [], []
    for offset in range(0, len(ids), 4):
        batch, ground_truth = [], []
        for index in ids[offset:offset+4]:
            item = protocol['pairs'][index]
            case = item['case_id']
            assert case == previous[index]['case_id'] == old_blend[index]['case_id']
            raw, mask = checked_case(baseline, args.data, item, previous[index])
            path = out/'images'/(case+'.png')
            observation = read(out/'pairs'/(case+'.json'))
            assert observation['index'] == index and observation['contract'] == run_contract
            assert observation['png_sha256'] == sha(path) and observation['steps'] == 50 and observation['b_calls'] == 25
            before = previous[index]['arms']['off']
            assert observation['initial_noise_digest'] == before['initial_noise_digest']
            assert observation['first_b_step'] == before['first_b_step']
            image = rgb(path)
            target = Image.fromarray(raw['gt'])
            val = image_scores(image, target, path, args.data/item['gt'], mask)
            rows.append(dict(index=index, case_id=case, category=item['category'], image_sha256=observation['png_sha256'],
                             scores={**previous[index]['scores'], 'blend':old_blend[index]['scores']['edge_a025'],
                                     'gate':val}))
            batch.append(np.asarray(image))
            ground_truth.append(raw['gt'])
        inputs, targets = tensors(batch), tensors(ground_truth)
        for row, value in zip(rows[-len(batch):], model(inputs*2-1, targets*2-1).flatten().tolist()):
            row['scores']['gate']['lpips'] = float(value)
        feature_groups.append(inception(inputs)[0].flatten(1).numpy())
    with np.load(baseline/'fid_pool3.npz',allow_pickle=False) as base_features:
        features = {n:base_features[n][ids] for n in (*ARMS,'gt')}
    with np.load(p/'inner_mix_pool3.npz',allow_pickle=False) as blend_features:
        features['blend'] = blend_features['edge_a025'][ids]
    features['gate'] = np.concatenate(feature_groups)
    assert len(rows)==len(ids) and all(f.shape==(len(ids),2048) and np.isfinite(f).all() for f in features.values())
    result = compare(rows,features)
    meta = dict(contract=run_contract,original_archive_sha256=ARCHIVE_SHA,
                blend_archive_sha256=BLEND_ARCHIVE_SHA,environment=environment(),
                optimizer_updates=0,new_s0_trajectories=len(ids),frozen_hashes=original['frozen_model_hashes'])
    write(out/'meta.json',meta)
    write(out/'per_pair.json',rows)
    write(out/'result.json',result)
    np.savez_compressed(out/'pool3.npz',**features)
    example_panel(out,baseline,args,protocol,previous,rows)
    names = ['meta.json','per_pair.json','result.json','pool3.npz','examples.png','shard0.json','shard1.json']
    names.extend(f'{folder}/{row["case_id"]}.{extension}' for row in rows
                 for folder,extension in (('pairs','json'),('images','png')))
    write(out/'review_manifest.json',{n:sha(out/n) for n in names})
    tmp = out/'review.zip.tmp'
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in (*names,'review_manifest.json'):
            archive.write(out/name,name)
    tmp.replace(out/'review.zip')
    print('S0_GATE_COMPLETE',len(rows),sha(out/'review.zip'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prior',type=Path)
    parser.add_argument('action',choices=('generate','evaluate'))
    parser.add_argument('--shard',type=int,choices=(0,1))
    args=parser.parse_args()
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','2')))
    cv2.setNumThreads(1)
    args.project=args.prior.parents[1]
    args.data=args.project.parent/'datasets/BF'
    args.device='cuda:0'
    args.checkpoint=args.project/'output_eval/e2_state_b_pastel_followup_20261115/trust_retry/training/step1.pt'
    baseline=args.prior/'real_paired_B_on_off_20261118'
    args.out=baseline
    out=baseline/'single_s0_gate_20261119'
    out.mkdir(exist_ok=True)
    verified_files(baseline,ARCHIVE_SHA,('protocol.json','per_image.json','fid_pool3.npz','result.json'))
    protocol=verify(args)
    run_contract=contract(baseline,protocol)
    context=(args,baseline,out,protocol,run_contract,read(baseline/'per_image.json'),read(baseline/'result.json'))
    if args.action=='generate':
        assert args.shard is not None
        generate(*context)
    else:
        assert not (out/'review.zip').exists()
        evaluate(*context)
