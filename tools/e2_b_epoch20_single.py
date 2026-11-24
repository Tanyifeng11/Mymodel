"""E2第20轮单次S0真实配对监督，H2仅比H1多衣内带通项。"""
import argparse
from collections import Counter
import copy
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from models.e2_lab_anchor import rgb_to_lab
from tools import e2_b_epoch20_expanded_long as fusion
from tools.e2_fft_experiment import SOURCE

init, tail, roi, ft = fusion.init, fusion.tail, fusion.roi, fusion.ft
rules, ab = fusion.rules, fusion.ab
RUN = 'b_single_s0_e2_epoch20_real_ab'
ARMS = ('H1', 'H2')
CONTROLS = ('off', 'on', 'blend', 'four_image')
PHASES, PERIOD, EARLY = fusion.PHASES, fusion.PERIOD, fusion.EARLY
LR, BATCH, SCALE = fusion.LR, fusion.BATCH, fusion.SCALE


def setup(prior, device):
    args = fusion.paths(prior, device)
    args.fusion_run = args.run
    source = args.fusion_run/'protocol.json'
    spec = tail.read(source)
    assert spec['base_sha256'] == init.BASE_SHA
    assert spec['f68_sha256'] == tail.CHECKPOINT_SHA
    assert spec['start_sha256'] == fusion.START_SHA and spec['retention'] == .125
    assert spec['sampler'] == args.protocol['sampler']
    assert spec['metric_assets'] == tail.metric_assets()
    args.pairs = {split:[p for p in spec['pairs'] if p['split'] == split]
                  for split in ('train','val','test64','test500','qualification')}
    expected = dict(train=512,val=64,test64=16,test500=125)
    products = set()
    for split, per_class in expected.items():
        pairs = args.pairs[split]
        roots = {fusion.root(p) for p in pairs}
        assert Counter(p['category'] for p in pairs) == dict.fromkeys(tail.distill.CATEGORIES,per_class)
        assert len(roots) == len(pairs) and None not in roots and not products & roots
        products.update(roots)
    base_manifest = args.base_checkpoint.parents[4]/'training_manifest.json'
    base_roots = {ft.independent.product(r['cloth'])
                  for r in fusion.read_checked(base_manifest,spec['base_train_manifest_sha256'])}
    assert not base_roots & {fusion.root(p) for s in ('val','test64','test500') for p in args.pairs[s]}
    for path,digest in ((args.base_checkpoint,init.BASE_SHA),(args.checkpoint,tail.CHECKPOINT_SHA),
                        (args.start,fusion.START_SHA),(args.a_checkpoint,init.A_SHA)):
        assert tail.sha(path) == digest, path
    args.run = args.baseline/RUN
    args.out = args.fusion_run  # 只读既有样本掩码和参考图。
    args.run.mkdir(parents=True,exist_ok=True)
    args.gt_hashes = spec['gt_sha256']
    args.spec = args.run/'protocol.json'
    contract = dict(cohort_sha256=tail.sha(source),pairs={k:len(p) for k,p in args.pairs.items()},
        source_protocol=spec['source_protocol_sha256'],base_sha256=init.BASE_SHA,f68_sha256=tail.CHECKPOINT_SHA,
        sampler=spec['sampler'],metrics=spec['metric_assets'],four_image_start_sha256=fusion.START_SHA,
        formal_reference=spec['formal_reference'],seed=42,stages=list(PHASES),validation_period=PERIOD,
        early_validation=list(EARLY),lr=LR,batch=BATCH,gpus=1,gradient_accumulation=BATCH,
        parameters=tail.PARAMETERS,
        initial='f68 features; zero output; no residual before step index 44',
        mask='clip(inward sketch distance / 8, 0, 1), average pool 8, applied once by StateTextureB',
        loss=dict(rgb=1,global_lpips=.05,inside_lpips=.05,low_lab=.10,frame=.20,band_H1=0,band_H2=.10),
        low_lab_channels=[100,100,100],lab_blur_sigma=8,
        select='existing relative-tradeoff vs own step0 on validation; both arms extend together',
        replacement='old absolute gates; four-image LPIPS/FID/PSNR/IoU/leak/texture no worse; '
                    'SSIM/F1 within 0.5% of both references; gains at least twice any losses',
        tolerance=rules.TOLERANCE, gain_ratio=rules.GAIN_RATIO,
        code={name:tail.sha(SOURCE/name) for name in (
            'tools/e2_b_epoch20_single.py','scripts/e2_b_epoch20_single.sbatch',
            'tools/e2_b_epoch20_expanded_long.py','tools/e2_b_init_epoch20_ab.py',
            'tools/e2_f68_tail6_real_ab.py','tools/e2_f68_e1_a_roi_ablation.py',
            'tools/e2_f68_e1_a_fused25_lstar_ab.py','models/e2_state_texture_b.py')})
    return args,contract


def initialize(models):
    head,params = tail.candidates(models[3])
    with torch.no_grad():
        head.output.weight.zero_()
        head.output.bias.zero_()
    return head,params


def conditions(args,pair,mask):
    values,_ = tail.joint.data.inputs(args,pair,tail.CONDITION,'raw')
    distance = cv2.distanceTransform(mask.astype('uint8'),cv2.DIST_L2,5)
    weight = torch.from_numpy(np.clip(distance/8,0,1)).to(args.device)
    values['mask'] = F.avg_pool2d(weight[None,None],8)
    return values


class LastSix:
    def __init__(self,head):
        self.head = head
        self.index = 25

    def __call__(self,z,x0,**values):
        index = self.index
        self.index += 1
        return self.head(z,x0,**values) if index >= tail.FIRST else torch.zeros_like(x0)


@torch.no_grad()
def full(args,models,pair,head=None,stop_at_prefix=False):
    core,pipe,net = models[:3]
    mask = tail.case_mask(args,pair)
    values = conditions(args,pair,mask)
    ref = args.out/'cases'/pair['case_id']/'original_ref.png'
    features = tail.joint.projected(net,core.tensor(args.data/pair['sketch']).to(args.device).half(),
                                   core.tensor(ref).to(args.device).half())
    module = LastSix(head) if head is not None else None
    try:
        with tail.PrefixTap(pipe,module,values,end=tail.LAST,qualify=not stop_at_prefix) as tap:
            try:
                picture,noise = tail.joint.image(args,core,pipe,net,pair,features)
            except tail.PrefixComplete:
                assert stop_at_prefix and tap.index == tail.FIRST
                return tap.start
        assert not stop_at_prefix and tap.index == tail.LAST
        assert module is None or module.index == tail.LAST
        return tap.start,tap.final,np.asarray(picture),noise
    finally:
        net.adapter.clear()


def prefix(args,models,pair):
    case = pair['case_id']
    path = args.run/'prefix_b_off'/(case+'.pt')
    record = dict(case=case,protocol_sha256=args.spec_sha,trajectory='B off, steps 0..43')
    if not path.exists():
        tail.save_torch(path,dict(full(args,models,pair,stop_at_prefix=True),**record))
    saved = torch.load(path,map_location='cpu',weights_only=True)
    assert all(saved[k] == v for k,v in record.items()), path
    return saved


def trajectory(args,models,head,pair,state,mask):
    core,pipe,net = models[:3]
    predictor = tail.repair.FrozenPredictor(args,core,pipe,net,[pair],(tail.CONDITION,))
    entry = dict(pair=pair,condition=tail.CONDITION)
    values = conditions(args,pair,mask)
    pipe.scheduler.set_timesteps(tail.LAST,device=args.device)
    z,epsilon = (state[k].to(args.device) for k in ('z','epsilon'))
    assert int(pipe.scheduler.timesteps[tail.FIRST]) == state['t']
    for index,t in enumerate(pipe.scheduler.timesteps[tail.FIRST:],tail.FIRST):
        if index > tail.FIRST:
            epsilon = predictor.epsilon(z,t,entry)
        a = pipe.scheduler.alphas_cumprod[int(t)].to(z.device).float()
        x0 = (z.float()-(1-a).sqrt()*epsilon.float())/a.sqrt()
        delta = head(z.float(),x0,timestep=t.reshape(1),**values)
        corrected = (epsilon.float()-(a/(1-a)).sqrt()*delta).to(epsilon.dtype)
        z = pipe.scheduler.step(corrected,t,z,eta=0,return_dict=False)[0]
    return z


def forward(args,models,head,pair):
    mask = tail.case_mask(args,pair)
    state = prefix(args,models,pair)
    z = trajectory(args,models,head,pair,state,mask)
    path = args.data/pair['gt']
    assert tail.sha(path) == args.gt_hashes[pair['case_id']]
    gt = np.asarray(tail.rgb(path).resize(tail.SIZE,Image.Resampling.BILINEAR))
    return tail.decode(models[1],z),roi.tensors([gt]).to(args.device),mask


class Objective(ft.light.LightingObjective):
    def __call__(self,pred,gt,mask,band_weight):
        # 复用归一化三域L1和LPIPS图，低频项改为真实配对的L、a、b。
        _,terms = ab.Objective.__call__(self,pred,gt,mask,'both')
        inside = torch.from_numpy(roi.regions(mask,8)[0]).to(pred)[None,None]
        lab_delta = (rgb_to_lab(pred)-rgb_to_lab(gt))/100
        weighted = (lab_delta*inside).reshape(-1,1,*pred.shape[-2:])
        smoothed = self.blur(weighted).reshape_as(lab_delta)/self.blur(inside).clamp_min(1e-6)
        low_lab = (smoothed.abs()*inside).sum()/(3*inside.sum().clamp_min(1))
        band = pred.new_zeros(())
        if band_weight:
            core = roi.regions(mask,20)[0]
            if core.sum() >= 256:
                keep = torch.from_numpy(core).to(pred.device)
                band = (self.base.band(pred)-self.base.band(gt)).abs()[0,0][keep].mean()
        loss = (terms['rgb_l1']+.05*terms['lpips']+.05*terms['inside_lpips']
                +.10*low_lab+.20*terms['frame']+band_weight*band)
        return loss,dict(rgb_l1=terms['rgb_l1'],lpips=terms['lpips'],inside_lpips=terms['inside_lpips'],
                         frame=terms['frame'],low_lab=low_lab,band=band)


def qualification(args,models,objective,pair,index):
    path = args.run/f'qualification_probe{index}.json'
    if path.exists():
        assert tail.read(path)['protocol_sha256'] == args.spec_sha
        return
    head,params = initialize(models)
    state = prefix(args,models,pair)
    mask = tail.case_mask(args,pair)
    start,off_latent,off_rgb,off_noise = full(args,models,pair)
    other,latent,rgb,noise = full(args,models,pair,head)
    assert noise == off_noise
    for key in ('z','epsilon'):
        assert torch.equal(start[key],state[key]) and torch.equal(other[key],state[key])
    assert torch.equal(latent,off_latent)
    np.testing.assert_array_equal(rgb,off_rgb)
    with torch.no_grad():
        replay = trajectory(args,models,head,pair,state,mask)
        assert torch.equal(replay.cpu(),latent)
        np.testing.assert_array_equal(ft.rgb8(tail.decode(models[1],replay)),rgb)
    pred,gt,mask = forward(args,models,head,pair)
    loss,_ = objective(pred,gt,mask,.10)
    norm = tail.backward(loss,params)
    output_norm = float(torch.linalg.vector_norm(head.output.weight.grad))
    assert np.isfinite(norm) and output_norm > 0
    # 仅在临时头上扰动，验证非零残差的回放和残差块梯度。
    with torch.no_grad():
        head.output.weight.add_(head.output.weight.grad,alpha=-.001/output_norm)
    head.zero_grad(set_to_none=True)
    del loss,pred,gt
    models[2].adapter.clear()
    _,probe_latent,probe_rgb,_ = full(args,models,pair,head)
    with torch.no_grad():
        replay = trajectory(args,models,head,pair,state,mask)
        assert torch.equal(replay.cpu(),probe_latent)
        np.testing.assert_array_equal(ft.rgb8(tail.decode(models[1],replay)),probe_rgb)
    pred,gt,mask = forward(args,models,head,pair)
    loss,_ = objective(pred,gt,mask,.10)
    tail.backward(loss,params)
    block_norm = sum(float(p.grad.square().sum()) for n,p in params.items() if n.startswith('blocks.1.'))**.5
    assert block_norm > 0 and np.isfinite(block_norm)
    ab.audit_frozen(args,models,head,params)
    models[2].adapter.clear()
    tail.write(path,dict(protocol_sha256=args.spec_sha,case=pair['case_id'],probe=index,
        zero_equals_b_off=True,zero_and_nonzero_full_equal_replay=True,output_grad_norm=output_norm,
        block_grad_norm_after_temporary_output_perturbation=block_norm,optimizer_updates=0))
    print('SINGLE_QUALIFIED',index,pair['case_id'],output_norm,block_norm,flush=True)


@torch.no_grad()
def render(args,models,head,pair,digest):
    case = pair['case_id']
    path = args.run/'renders'/digest/(case+'.png')
    meta = path.with_suffix('.json')
    record = dict(protocol_sha256=args.spec_sha,case=case,head_sha256=digest)
    if init.checked_image(path,meta,record) is None:
        pred,_,_ = forward(args,models,head,pair)
        init.save_image(path,meta,ft.rgb8(pred),record)
        models[2].adapter.clear()
    return path


def control_context(args):
    context = copy.copy(args)
    context.run = args.run/'controls'
    context.old_pairs = {}
    return context


def test_controls(args,models,pairs,split):
    if split == 'test64':
        return
    context = control_context(args)
    a,_ = init.head_for(context,models,'A')
    start,_ = init.head_for(context,models,'f68')
    for i,pair in enumerate(pairs,1):
        existing = args.fusion_run/'fixed'/pair['case_id']/'record.json'
        if existing.exists():
            record = tail.read(existing)
            assert record['protocol_sha256'] == tail.sha(args.fusion_run/'protocol.json')
            for key in ('off','on','blend','start'):
                assert tail.sha(existing.parent/(key+'.png')) == record['pngs'][key]
        else:
            cache_control(context,models,pair,a,start)
        if i%16 == 0:
            print('SINGLE_CONTROL500',i,len(pairs),flush=True)


@torch.no_grad()
def cache_control(context,models,pair,a,start):
    folder = context.run/'fixed'/pair['case_id']
    marker = folder/'record.json'
    if marker.exists():
        record = tail.read(marker)
        assert record['protocol_sha256'] == context.spec_sha
        for name,digest in record['pngs'].items():
            assert tail.sha(folder/(name+'.png')) == digest
        return
    images = init.controls(context,models,pair)
    mask = tail.case_mask(context,pair)
    donor = tail.mix_pixels(images['off'],images['on'],mask,{'blend':.25})['blend']
    alpha = ft.fused.alpha(mask)
    a_rgb = ft.fused.pixels_mixed(init.raw_cached(context,models,a,pair),donor,alpha)
    b_rgb = ft.fused.pixels_mixed(init.raw_cached(context,models,start,pair),donor,alpha)
    images.update(blend=donor,start=ft.uniform_mix(b_rgb,a_rgb,mask))
    for name,image in images.items():
        ft.fused.save_png(folder/(name+'.png'),image)
    tail.write(marker,dict(protocol_sha256=context.spec_sha,
                           pngs={n:tail.sha(folder/(n+'.png')) for n in images}))


def archived64(args,pairs):
    directory = args.source/'eval64'
    manifest = tail.read(args.source/'review_manifest.json')
    rows = fusion.read_checked(directory/'per_pair.json',manifest['eval64/per_pair.json'])
    path = directory/'pool3.npz'
    assert tail.sha(path) == manifest['eval64/pool3.npz']
    assert [r['case_id'] for r in rows] == [p['case_id'] for p in pairs]
    with np.load(path,allow_pickle=False) as pool:
        values = {key:pool['f68_zero' if key=='four_image' else key] for key in (*CONTROLS,'gt')}
    digests = {}
    for row,pair in zip(rows,pairs):
        case = pair['case_id']
        paths = test_paths(args,pair,'test64')
        assert all(tail.sha(p) == row['pngs']['f68_zero' if n=='four_image' else n] for n,p in paths.items())
        digests[case] = row
    return digests,values


def test_paths(args,pair,split):
    case = pair['case_id']
    if split == 'test64':
        base = args.source/'fixed'/case
        four = args.source/'eval64/images/f68_selected'/(case+'.png')
    else:
        base = args.fusion_run/'fixed'/case
        if not (base/'record.json').exists():
            base = args.run/'controls/fixed'/case
        four = base/'start.png'
    return dict({n:base/(n+'.png') for n in ('off','on','blend')},four_image=four)


@torch.no_grad()
def score(args,models,pairs,heads,dest,metrics,split):
    dest.mkdir(parents=True,exist_ok=True)
    digest = {name:init.head_hash(head) for name,head in heads.items()}
    result_path = dest/'result.json'
    if result_path.exists():
        result = tail.read(result_path)
        assert result['protocol_sha256'] == args.spec_sha and result['head_sha256'] == digest
        assert tail.sha(dest/'pool3.npz') == result['pool3_sha256']
        return result
    controls = CONTROLS if split.startswith('test') else ()
    names = (*controls,*heads)
    old,old_features = archived64(args,pairs) if split=='test64' else ({},{})
    lpips,inception = metrics
    rows,features = [],{n:[] for n in (*names,'gt')}
    for i in range(0,len(pairs),4):
        batch = pairs[i:i+4]
        masks = [tail.case_mask(args,p) for p in batch]
        cores = [roi.regions(m,20)[0] for m in masks]
        targets = [np.asarray(tail.rgb(args.data/p['gt']).resize(tail.SIZE,Image.Resampling.BILINEAR)) for p in batch]
        assert all(tail.sha(args.data/p['gt']) == args.gt_hashes[p['case_id']] for p in batch)
        target = roi.tensors(targets)
        target_features = inception(target)[0].flatten(1).numpy()
        if old:
            np.testing.assert_allclose(target_features,old_features['gt'][i:i+len(batch)],atol=7e-5,rtol=1e-4)
            target_features = old_features['gt'][i:i+len(batch)]
        features['gt'].append(target_features)
        labels = [roi.lab(x) for x in targets]
        records = [dict(case_id=p['case_id'],category=p['category'],interior_px=int(core.sum()),
            pngs={},scores={},texture={'gt':tail.appearance(gt,lab,core)},regional_lpips={})
            for p,core,gt,lab in zip(batch,cores,targets,labels)]
        if controls:
            target_maps = roi.lpips_features(lpips,target)
            regions = torch.from_numpy(np.stack([roi.regions(m,8) for m in masks])).float()
        for name in names:
            paths = [render(args,models,heads[name],p,digest[name]) if name in heads
                     else test_paths(args,p,split)[name] for p in batch]
            pixels = [np.asarray(tail.rgb(p)) for p in paths]
            tensor = roi.tensors(pixels)
            reused = bool(old) and name in controls
            if reused:
                key = 'f68_zero' if name=='four_image' else name
                features[name].append(old_features[name][i:i+len(batch)])
            else:
                features[name].append(inception(tensor)[0].flatten(1).numpy())
                distances = lpips(tensor*2-1,target*2-1).flatten().tolist()
            if controls:
                local = roi.apportion(roi.maps(lpips,target_maps,tensor),regions)
            for j,(r,png,im) in enumerate(zip(records,paths,pixels)):
                r['pngs'][name] = tail.sha(png)
                if reused:
                    prior = old[r['case_id']]
                    r['scores'][name],r['texture'][name] = prior['scores'][key],prior['texture'][key]
                else:
                    scores = tail.image_scores(Image.fromarray(im),Image.fromarray(targets[j]),png,
                                               args.data/batch[j]['gt'],masks[j])
                    r['scores'][name] = dict(scores,lpips=distances[j])
                    r['texture'][name] = tail.appearance(im,labels[j],cores[j])
                if controls:
                    r['regional_lpips'][name] = dict(zip(('inside','edge','background'),local[j].tolist()))
        rows.extend(records)
        if len(rows)%64 == 0:
            print('SINGLE_SCORED',dest.name,len(rows),len(pairs),flush=True)
    features = {n:np.concatenate(v) for n,v in features.items()}
    assert all(v.shape == (len(pairs),2048) and np.isfinite(v).all() for v in features.values())
    result = fusion.summarize(rows,features,names)
    if controls:
        for group,summary in result['groups'].items():
            subset = [r for r in rows if group=='all' or r['category']==group]
            summary['regional_lpips'] = {n:{k:float(np.mean([r['regional_lpips'][n][k] for r in subset]))
                for k in ('inside','edge','background')} for n in names}
    np.savez_compressed(dest/'pool3.npz',**features)
    result.update(protocol_sha256=args.spec_sha,head_sha256=digest,pairs=len(pairs),
                  pool3_sha256=tail.sha(dest/'pool3.npz'))
    tail.write(dest/'per_pair.json',rows)
    tail.write(result_path,result)
    return result


def update_batch(args,models,objective,head,parameters,optimizer,pairs,band_weight):
    observations = []
    for pair in pairs:
        pred,gt,mask = forward(args,models,head,pair)
        loss,terms = objective(pred,gt,mask,band_weight)
        assert torch.isfinite(loss)
        # 同一组权重累计两对样本的梯度，裁剪并更新一次。
        (loss*SCALE/BATCH).backward()
        observations.append(dict(loss=float(loss.detach()),**{k:float(v.detach()) for k,v in terms.items()}))
        models[2].adapter.clear()
        del loss,pred,gt,terms
    for p in parameters.values():
        p.grad.div_(SCALE)
    norm = float(torch.nn.utils.clip_grad_norm_(parameters.values(),1,error_if_nonfinite=True))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return [dict(row,grad_norm=norm) for row in observations]


def train(args,models,objective,metrics):
    training,validation = args.pairs['train'],args.pairs['val']
    heads,params,optimizers = {},{},{}
    for name in ARMS:
        heads[name],params[name] = initialize(models)
        optimizers[name] = torch.optim.AdamW(params[name].values(),lr=LR)
    resume = args.run/'resume.pt'
    completed,trace,checks,phases = 0,[],{},[]
    if resume.exists():
        state = torch.load(resume,map_location='cpu',weights_only=False)
        assert state['protocol_sha256'] == args.spec_sha
        completed,trace,checks,phases = (state[k] for k in ('step','trace','checks','phases'))
        for name in ARMS:
            rules.restore(params[name],state['heads'][name])
            optimizers[name].load_state_dict(state['optimizers'][name])

    def snapshot():
        return {name:{k:p.detach().cpu() for k,p in params[name].items()} for name in ARMS}

    def save():
        tail.save_torch(resume,dict(protocol_sha256=args.spec_sha,step=completed,trace=trace,
            checks=checks,phases=phases,heads=snapshot(),
            optimizers={n:o.state_dict() for n,o in optimizers.items()}))

    def validate(step):
        dest = args.run/'validation'/f'step{step}'
        for n,weights in snapshot().items():
            path = dest/(n+'.pt')
            if not path.exists():
                tail.save_torch(path,weights)
            stored = torch.load(path,map_location='cpu',weights_only=True)
            assert all(torch.equal(stored[k],v) for k,v in weights.items())
        result = score(args,models,validation,heads,dest,metrics,'val')['groups']['all']
        checks[str(step)] = {n:init.metrics_from_group(result,n) for n in ARMS}
        print('SINGLE_VALIDATION',step,checks[str(step)],flush=True)
        save()

    if str(completed) not in checks and (completed%PERIOD==0 or completed in EARLY):
        validate(completed)
    previous = 0
    for end in PHASES:
        recorded = next((p for p in phases if p['end']==end),None)
        if recorded:
            previous = end
            if not recorded['continue_training']:
                break
            continue
        order = None
        for step in range(completed+1,end+1):
            epoch,position = divmod(step-1,len(training)//BATCH)
            if order is None or position==0:
                order = np.random.default_rng(42+epoch).permutation(len(training))
            pairs = [training[j] for j in order[position*BATCH:(position+1)*BATCH]]
            records = [dict(case=p['case_id']) for p in pairs]
            for n in ARMS:
                values = update_batch(args,models,objective,heads[n],params[n],optimizers[n],pairs,
                                      .10 if n=='H2' else 0.)
                for row,value in zip(records,values):
                    row[n] = value
            trace.append(dict(step=step,rows=records))
            if step%16==0:
                print('SINGLE_TRAIN',step,end,records,flush=True)
            completed = step
            save()
            if step%PERIOD==0 or step in EARLY:
                validate(step)
        selected = {n:rules.choose({s:v[n] for s,v in checks.items()})[0] for n in ARMS}
        continued = end<PHASES[-1] and any(s>previous for s in selected.values())
        phases.append(dict(end=end,selected=selected,continue_training=continued))
        for n in ARMS:
            ab.audit_frozen(args,models,heads[n],params[n])
        save()
        print('SINGLE_PHASE',phases[-1],flush=True)
        previous = end
        if not continued:
            break
    selections = {n:rules.choose({s:v[n] for s,v in checks.items()}) for n in ARMS}
    steps = {n:s[0] for n,s in selections.items()}
    for n,step in steps.items():
        ab.update_head(heads[n],params[n],args.run/'validation'/f'step{step}'/(n+'.pt'))
        ab.audit_frozen(args,models,heads[n],params[n])
        heads[n].eval().requires_grad_(False)
    tail.write(args.run/'training.json',dict(updates_per_arm=completed,selected=steps,validation=checks,
        assessments={n:s[1] for n,s in selections.items()},phases=phases,trace=trace,frozen_unchanged=True))
    return heads,steps,completed


def acceptance(args,dest,result,steps,split):
    n = '64' if split == 'test64' else '500'
    source = args.formal64 if n=='64' else args.formal500
    manifest = tail.read(args.spec)['formal_reference'][n]
    old = fusion.read_checked(source/'result.json',manifest['result.json'])
    old_rows = fusion.read_checked(source/'per_pair.json',manifest['per_pair.json'])
    assert [r['case_id'] for r in old_rows] == [p['case_id'] for p in args.pairs[split]]
    assert tail.sha(source/'pool3.npz') == manifest['pool3.npz']
    with np.load(source/'pool3.npz',allow_pickle=False) as a,np.load(dest/'pool3.npz',allow_pickle=False) as b:
        np.testing.assert_allclose(a['gt'],b['gt'],atol=7e-5,rtol=1e-4)
    group = result['groups']['all']
    four = init.metrics_from_group(group,'four_image')
    historical = init.metrics_from_group(old['groups']['all'],'retain_125')
    decisions = {}
    for name in result['head_sha256']:
        value = init.metrics_from_group(group,name)
        absolute = init.gates(group,name)
        tradeoff = rules.tradeoff(value,four)
        checks = {}
        for k in rules.KEYS:
            if k in ('ssim','contour_f1'):
                checks[k] = value[k] >= max(four[k],historical[k])*(1-rules.TOLERANCE)
            else:
                checks[k] = value[k] <= four[k] if k in rules.LOWER else value[k] >= four[k]
        checks['tradeoff_if_degraded'] = tradeoff['gain'] >= rules.GAIN_RATIO*tradeoff['harm']
        decisions[name] = dict(metrics=value,absolute=absolute,checks=checks,tradeoff=tradeoff,
            accepted=steps[name]>0 and absolute['passed'] and all(checks.values()))
    return dict(four_image=four,historical=historical,decisions=decisions)


@torch.no_grad()
def benchmark(args,models,heads,pairs):
    context = control_context(args)
    a,_ = init.head_for(context,models,'A')
    start,_ = init.head_for(context,models,'f68')
    core,pipe,net,original = models[:4]

    def four(p):
        off,_,_ = tail.joint.sample(args,core,pipe,net,p,tail.CONDITION)
        on,_,_ = tail.joint.sample(args,core,pipe,net,p,tail.CONDITION,b=original)
        mask = tail.case_mask(args,p)
        # A与固定B对照共享一次冻结的门控前缀。
        state = tail.cache_prefix(args,core,pipe,net,original,p)
        rgb = [ft.rgb8(tail.decode(pipe,tail.trajectory(args,models,h,p,state,mask))) for h in (a,start)]
        donor = tail.mix_pixels(np.asarray(off),np.asarray(on),mask,{'blend':.25})['blend']
        alpha = ft.fused.alpha(mask)
        return ft.uniform_mix(ft.fused.pixels_mixed(rgb[1],donor,alpha),
                              ft.fused.pixels_mixed(rgb[0],donor,alpha),mask)

    arms = dict(four_image=four)
    arms.update({n:(lambda p,h=h:full(args,models,p,h)[2]) for n,h in heads.items()})
    samples = []
    for category in tail.distill.CATEGORIES:
        samples.extend([p for p in pairs if p['category']==category][:2])
    results = {n:[] for n in arms}
    for n,fn in arms.items():
        produced = fn(samples[0])
        expected = test_paths(args,samples[0],'test500')[n] if n=='four_image' else \
            args.run/'renders'/init.head_hash(heads[n])/(samples[0]['case_id']+'.png')
        np.testing.assert_array_equal(produced,np.asarray(tail.rgb(expected)))
    for pair in samples:
        for n,fn in arms.items():
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            fn(pair)
            torch.cuda.synchronize()
            results[n].append(dict(case=pair['case_id'],seconds=time.perf_counter()-t0,
                peak_extra_bytes=torch.cuda.max_memory_allocated()-before,
                peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    tail.write(args.run/'latency.json',dict(batch=1,device=torch.cuda.get_device_name(),warmup_per_arm=1,
        four_image_shared_prefix=True,results=results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prior',type=Path)
    parser.add_argument('--check',action='store_true')
    options = parser.parse_args()
    args,contract = setup(options.prior,'cpu' if options.check else 'cuda:0')
    if options.check:
        archived64(args,args.pairs['test64'])
        assert len(args.pairs['qualification']) == BATCH
        print('SINGLE_CPU_CHECK',contract['pairs'],contract['cohort_sha256'],flush=True)
        return
    assert torch.cuda.device_count() == 1
    if args.spec.exists():
        assert tail.read(args.spec) == contract
    else:
        tail.write(args.spec,contract)
    args.spec_sha = tail.sha(args.spec)
    if (args.run/'completion.json').exists() and (args.run/'review.zip').exists():
        return
    cv2.setNumThreads(1)
    models = init.load(args)
    torch.set_num_threads(6)
    objective = Objective(args)
    for index,pair in enumerate(args.pairs['qualification']):
        qualification(args,models,objective,pair,index)
    pairs = [*args.pairs['train'],*args.pairs['val']]
    for i,pair in enumerate(pairs,1):
        prefix(args,models,pair)
        if i%32==0:
            print('SINGLE_PREFIX',i,len(pairs),flush=True)
    metrics = tail.metric_models('cpu',args.protocol['metric_assets'])
    heads,steps,updates = train(args,models,objective,metrics)
    active = heads
    evaluation = {}
    for split in ('test64','test500'):
        if not active:
            break
        pairs = args.pairs[split]
        test_controls(args,models,pairs,split)
        dest = args.run/('eval64' if split=='test64' else 'eval500')
        result = score(args,models,pairs,active,dest,metrics,split)
        decision = acceptance(args,dest,result,steps,split)
        tail.write(dest/'acceptance.json',decision)
        accepted = [n for n,d in decision['decisions'].items() if d['accepted']]
        evaluation[dest.name] = dict(scored=list(active),accepted=accepted)
        print('SINGLE_ACCEPTANCE',dest.name,decision,flush=True)
        if split=='test500':
            benchmark(args,models,active,pairs)
        active = {n:heads[n] for n in accepted}
    assert tail.tensor_set(models[4]) == args.original['frozen_model_hashes']
    tail.write(args.run/'completion.json',dict(updates_per_arm=updates,selected=steps,evaluation=evaluation,
        trained_heads=2,frozen_unchanged=True,inference_trajectories=1,vae_decodes=1))
    files = ['protocol.json','model.json','execution_environment.json','qualification_probe0.json',
             'qualification_probe1.json','training.json','completion.json']
    files.extend(str(p.relative_to(args.run)) for p in sorted((args.run/'validation').glob('step*/result.json')))
    for label in evaluation:
        files.extend(f'{label}/{n}' for n in ('result.json','per_pair.json','pool3.npz','acceptance.json'))
    if 'eval500' in evaluation:
        files.append('latency.json')
    init.fid.pack(args.run,files)
    print('SINGLE_FINISHED',tail.sha(args.run/'review.zip'),flush=True)


if __name__=='__main__':
    main()
