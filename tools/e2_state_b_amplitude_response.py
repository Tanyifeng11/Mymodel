"""冻结 f68 的三档参考幅度响应；完整 S0 推理，无优化器或权重更新。"""
import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

from tools.e2_b_last_step_metrics import highpass, precise_l, lab, score_pixels
from tools.e2_b_amplitude_response_metrics import transform_reference, projection_features, response_summary


CHECKPOINT_SHA = 'f68f22a106c3f0a08044c754b44c738ba3d6f67274edf5d281d5b54f39ddcd96'
ARMS = {'a075': .75, 'a100': 1., 'a125': 1.25}


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pixels(path):
    return np.asarray(Image.open(path).convert('RGB'))


def change_audit(current, baseline, mask):
    """实际 PNG 的串扰审计；不把有限窗口高通称为严格频率投影。"""
    inside = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5) >= 16
    a, b = lab(current), lab(baseline)
    x, p = highpass(precise_l(current))[inside], highpass(precise_l(baseline))[inside]
    den = max(float(np.sum(p*p)), 1e-12)
    delta = a-b
    weight = mask.astype(np.float32)
    low = cv2.GaussianBlur(delta[..., 0]*weight, (0, 0), 24)
    low /= cv2.GaussianBlur(weight, (0, 0), 24).clip(.001)
    return dict(fixed_projection=float(np.sum(x*p)/den),
        amplitude_ratio=float(np.sqrt(np.sum(x*x)/den)),
        correlation=float(np.sum(x*p)/max(float(np.linalg.norm(x)*np.linalg.norm(p)), 1e-12)),
        mean_l_delta=float(delta[inside, 0].mean()), low_l_change=float(np.abs(low[inside]).mean()),
        ab_mae=float(np.abs(delta[inside, 1:]).mean()),
        saturated_pixel_fraction=float(((current==0)|(current==255)).any(-1)[inside].mean()))


def prepare(args):
    old = read(args.source_run/'protocol.json')
    result = read(args.source_run/'result.json')
    assert old['checkpoint_sha256'] == result['checkpoint_sha256'] == CHECKPOINT_SHA
    assert result['frozen_unchanged'] and result['training_updates'] == 0
    assert len(old['train']) == 4 and len(old['replay_bank']) == 16
    assert old['sampler'] == dict(steps=50, cfg=7., sketch_scale=.6, active=[25, 50])
    args.out.mkdir(parents=True, exist_ok=True)
    assert not (args.out/'generated.json').exists(), '已有输出，不能覆盖诊断'
    pairs, inputs, assets = copy.deepcopy(old['train']), [], {}
    for pair in pairs:
        case = pair['case_id']
        source, dest = args.source_run/'cases'/case, args.out/'cases'/case
        dest.mkdir(parents=True, exist_ok=True)
        for name in ('mask.png', 'train_anchor.png', 'sketch.jpg'):
            shutil.copyfile(source/name, dest/name)
            assert sha(dest/name) == old['assets'][f'cases/{case}/{name}']
        for item in old['replay_bank']:
            if item['case'] != case:
                continue
            c = item['condition']
            for suffix in ('ref', 'goal', 'positive'):
                name = f'{c}_{suffix}.png'
                shutil.copyfile(source/name, dest/name)
                assert sha(dest/name) == old['assets'][f'cases/{case}/{name}']
            original = pixels(dest/f'{c}_ref.png')
            assert original.shape == (512, 384, 3)
            for arm, factor in ARMS.items():
                sample_condition = c+'_'+arm
                transformed, audit = transform_reference(original, factor)
                target = dest/f'{sample_condition}_ref.png'
                if factor == 1:
                    shutil.copyfile(dest/f'{c}_ref.png', target)
                    assert np.array_equal(transformed, original)
                else:
                    Image.fromarray(transformed).save(target)
                pair['reference_hashes'][sample_condition] = sha(target)
                inputs.append(dict(item, arm=arm, factor=factor, sample_condition=sample_condition,
                                   reference_audit=audit, reference_sha256=sha(target)))
        for path in dest.iterdir():
            if path.is_file():
                assets[path.relative_to(args.out).as_posix()] = sha(path)
    protocol = dict(train=pairs, replay_bank=old['replay_bank'], inputs=inputs, assets=assets,
        sampler=old['sampler'], arms=ARMS, checkpoint_sha256=CHECKPOINT_SHA,
        source_protocol_sha256=sha(args.source_run/'protocol.json'),
        source_result_sha256=sha(args.source_run/'result.json'),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        training_updates=0, heldout_used=False,
        intervention='final reference PNG L highpass amplitude; same PNG to BFM and B; full independent S0 rollout',
        positive_precision='fp16 VAE mean encoder, float32 latent and decoder, rint uint8; generation remains fp16',
        response='fixed p1 basis x=<p_alpha,p1>/<p1,p1>, y=<h_alpha,p1>/<p1,p1>; actual endpoint slope dy/dx',
        interpretation='16 repeated conditions in 4 seen identities; no tuning or checkpoint selection')
    assert len(inputs) == 48
    write(args.out/'protocol.json', protocol)
    print('AMPLITUDE_PREPARED', len(inputs), flush=True)


@torch.no_grad()
def generate(args):
    from models.e2_state_texture_b import StateTextureB
    from tools import e2_state_b_balanced as balanced
    from tools import e2_state_b_pilot as pilot
    from tools.e2_b_material_losses import material_target
    from tools.e2_oracle_texture import encode, weights_digest
    from tools.e2_fft_experiment import environment

    spec = read(args.out/'protocol.json')
    assert sha(args.checkpoint) == spec['checkpoint_sha256'] == CHECKPOINT_SHA
    assert all(sha(args.out/name)==digest for name,digest in spec['assets'].items())
    assert sha(args.source_run/'result.json') == spec['source_result_sha256']
    old_rows = {(r['case'],r['condition']):r for r in read(args.source_run/'result.json')['rows']}
    runtime = copy.copy(args)
    runtime.action = 'train'  # 仅选择旧确定性后端；没有调用训练函数。
    core, pipe, net = pilot.load_core(runtime)
    assert pipe.scheduler.__class__.__name__ == 'DDIMScheduler'
    b = StateTextureB(color_channels=2, reference_channels=192).to(args.device)
    b.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=False)['b'])
    b.eval().requires_grad_(False)
    net.eval().requires_grad_(False)
    b_hash, net_hash = weights_digest(b), weights_digest(net)
    assert net_hash == read(args.source_run/'shard0.json')['net_hash']
    pairs = {p['case_id']:p for p in spec['train']}
    rows = []
    started = time.monotonic()
    for item in spec['inputs']:
        case, c, arm = item['case'], item['condition'], item['arm']
        pair, sample_c = pairs[case], item['sample_condition']
        folder = args.out/'cases'/case
        anchor, ref = pixels(folder/'train_anchor.png'), pixels(folder/f'{sample_c}_ref.png')
        mask = np.asarray(Image.open(folder/'mask.png')) > 0
        goal, goal_audit = material_target(anchor, ref, mask)
        pipe.vae.half()
        z = encode(pipe.vae, Image.fromarray(goal), args.device)
        pipe.vae.float()
        rgb = (pipe.vae.decode(z/pipe.vae.config.scaling_factor).sample.float()/2+.5).clamp(0,1)
        positive = np.rint(rgb[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        base_goal, base_positive = pixels(folder/f'{c}_goal.png'), pixels(folder/f'{c}_positive.png')
        if arm == 'a100':
            assert np.array_equal(goal, base_goal), (case,c,'goal identity')
            assert np.array_equal(positive, base_positive), (case,c,'positive identity')
        Image.fromarray(goal).save(folder/f'{sample_c}_goal.png')
        Image.fromarray(positive).save(folder/f'{sample_c}_positive.png')
        pipe.vae.half()
        calls = []
        def record(module, positional, keywords, output):
            calls.append(int(keywords['timestep']))
        with b.register_forward_hook(record, with_kwargs=True):
            picture, digest, tap = balanced.sample(args, core, pipe, net, pair, sample_c, b)
        assert tap.index == 50 and len(calls) == 25 and calls == pipe.scheduler.timesteps[25:].tolist()
        old = old_rows[case,c]
        assert digest == old['audit']['noise_digest']
        array = np.asarray(picture)
        dest = args.out/'images'/arm/(case+'_'+c+'.png')
        dest.parent.mkdir(parents=True, exist_ok=True)
        picture.save(dest)
        row = dict(item, seed=pair['seed'], noise_digest=digest, b_calls=25,
            png_sha256=sha(dest), goal_audit=goal_audit,
            scores=score_pixels(array, positive, goal, anchor, mask, pair['patch_coordinates']),
            projection=projection_features(array, positive, base_positive, mask),
            goal_change=change_audit(goal,base_goal,mask),
            positive_change=change_audit(positive,base_positive,mask),
            vae_fidelity=change_audit(positive,goal,mask))
        prior = pixels(args.source_run/'images'/'S0'/(case+'_'+c+'.png'))
        if arm == 'a100':
            assert np.array_equal(array, prior), (case,c,'S0 replay')
            row['identity_parity'] = dict(goal=True,positive=True,output=True,max_pixel_error=0)
        delta = np.abs(array.astype(np.float32)-prior.astype(np.float32))/255
        distance = cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,5)
        row['baseline_pixel_change'] = dict(outside_mae=float(delta[~mask].mean()),
            boundary_band_mae=float(delta[mask&(distance<16)].mean()),interior_mae=float(delta[distance>=16].mean()))
        rows.append(row)
        write(args.out/'partial.json', dict(rows=rows))
        print('AMPLITUDE_IMAGE', case,c,arm,flush=True)
    assert len(rows)==48 and weights_digest(b)==b_hash and weights_digest(net)==net_hash
    assert sha(args.checkpoint)==CHECKPOINT_SHA
    write(args.out/'generated.json',dict(rows=rows,training_updates=0,heldout_used=False,
        frozen_unchanged=True,b_hash=b_hash,net_hash=net_hash,checkpoint_sha256=CHECKPOINT_SHA,
        seconds=time.monotonic()-started,environment=environment()))
    print('AMPLITUDE_GENERATED',len(rows),flush=True)


def finish(args):
    spec, generated = read(args.out/'protocol.json'), read(args.out/'generated.json')
    rows = generated['rows']
    assert len(rows)==48 and generated['frozen_unchanged'] and generated['training_updates']==0
    assert sum('identity_parity' in r for r in rows)==16
    pairs = []
    for item in spec['replay_bank']:
        rr = {r['arm']:r for r in rows if (r['case'],r['condition'])==(item['case'],item['condition'])}
        assert set(rr)==set(ARMS)
        pairs.append(dict(item,response=response_summary({a:r['projection'] for a,r in rr.items()}),
                          arms=rr))
    groups = {}
    keys = ('l_amplitude','l_amplitude_error','l_correlation','l_target_gain','l_target_gain_error',
            'palette_swd','spatial_ab_mae_goal','low_l','old_projection','old_projection_channel_abs_sum',
            'old_l_projection','edge','l_edge')
    for group in ('all','neutral','pastel','multi','synthetic'):
        rr = rows if group=='all' else [r for r in rows if r['group']==group]
        groups[group]={}
        for arm in ARMS:
            aa = [r for r in rr if r['arm']==arm]
            values = {k:[r['scores'][k] for r in aa if r['scores'][k] is not None] for k in keys}
            groups[group][arm] = dict(count=len(aa),
                metrics={k:float(np.mean(v)) if v else None for k,v in values.items()},
                amplitude_pass=sum(r['scores']['amplitude_pass'] for r in aa),
                l_quality_pass=sum(r['scores']['l_quality_pass'] for r in aa))
    write(args.out/'result.json',dict(pairs=pairs,groups=groups,training_updates=0,heldout_used=False,
        frozen_unchanged=True,identity_replays=16,checkpoint_sha256=CHECKPOINT_SHA,
        interpretation='Slopes use actual VAE target projection span; response is distinct from accuracy'))
    for case in sorted({r['case'] for r in rows}):
        cc = [x for x in spec['replay_bank'] if x['case']==case]
        board = Image.new('RGB',(6*160,len(cc)*252),'white')
        draw = ImageDraw.Draw(board)
        for j,item in enumerate(cc):
            c=item['condition']
            for i,arm in enumerate(ARMS):
                files=[args.out/'cases'/case/f'{c}_{arm}_positive.png',args.out/'images'/arm/(case+'_'+c+'.png')]
                for k,path in enumerate(files):
                    x=(2*i+k)*160
                    board.paste(Image.open(path).convert('RGB').resize((160,213)),(x,j*252+36))
                    draw.text((x+2,j*252+2),c,fill='black')
                    draw.text((x+2,j*252+17),arm+' '+('target' if k==0 else 'S0'),fill='black')
        board.save(args.out/(case+'_comparison.png'))
    files=sorted(p for p in args.out.rglob('*') if p.is_file()
        and 'source' not in p.relative_to(args.out).parts and p.suffix not in ('.zip','.tmp','.log','.err')
        and p.name!='manifest.json')
    write(args.out/'manifest.json',{p.relative_to(args.out).as_posix():sha(p) for p in files})
    archive=args.out/'amplitude_response_review.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[args.out/'manifest.json']:
            z.write(p,p.relative_to(args.out).as_posix())
    print('AMPLITUDE_COMPLETE',len(rows),sha(archive),archive.stat().st_size,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','generate','finish'))
    parser.add_argument('--project',type=Path,default=Path('/share/home/u2515283058/Mymodel'))
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    args.data=args.project.parent/'datasets/BF'
    args.previous=args.project/'output_eval/e2_state_b_material_20261113'
    if args.checkpoint is None:
        args.checkpoint=args.project/'output_eval/e2_state_b_pastel_followup_20261115/trust_retry/training/step1.pt'
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    {'prepare':prepare,'generate':generate,'finish':finish}[args.action](args)


if __name__=='__main__':
    main()
