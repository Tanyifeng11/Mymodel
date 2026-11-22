"""固定统一 12.5% 的最终融合，用真实成衣短程续训 B。"""
import argparse
import copy
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from tools import e2_f68_e1_a_fused25_continue as fused
from tools import e2_f68_e1_a_fused25_lstar_ab as light
from tools import e2_f68_e1_a_roi_ablation as ab
from tools import e2_f68_e1_band_class_ablation as class_ab
from tools import e2_f68_e1_fid_regions as fid
from tools import e2_f68_e1_outwear_independent as independent
from tools import e2_f68_e1_retention500 as retention
from tools import e2_f68_region_diagnostic as roi
from tools import e2_f68_tail6_real_ab as tail


RUN = 'retain125_finetune_20261122'
BASE = retention.CANDIDATE
ARM = 'retain125_ft'
ARMS = (*retention.SCORED, ARM)
LR = 1e-5
ACCUMULATE = 2
STEPS = 64
SNAPSHOTS = (16, 32, STEPS)
SCALE = 128


def rgb8(image):
    return np.rint(image.detach()[0].permute(1,2,0).cpu().numpy()*255).astype('uint8')


def uniform_mix(current, offline, mask):
    weight = (1-retention.RETENTION)*fid.regions(mask)[0]
    return fused.pixels_mixed(current,offline,weight)


def fuse_final(prediction, donor, offline, mask):
    current = fused.integer_st(fused.fuse_training(prediction,donor,mask)*255)
    source = torch.from_numpy(np.ascontiguousarray(offline.transpose(2,0,1)))[None].to(current)
    weight = torch.as_tensor((1-retention.RETENTION)*fid.regions(mask)[0],
                             device=current.device,dtype=current.dtype)[None,None]
    return fused.integer_st(current+weight*(source-current))/255


def setup(prior):
    args = retention.setup(prior)
    args.run = args.light_run/RUN
    args.run.mkdir(exist_ok=True)
    args.action,args.device = 'train','cuda:0'
    args.out = args.baseline/tail.distill.RUN
    trainval_path = args.out/'protocol.json'
    args.trainval = tail.distill.experiment_protocol(args)
    args.start = args.light_run/'B/step64.pt'
    assert tail.sha(args.start)==args.heads['B']
    assert tail.sha(args.base_checkpoint)==args.contract['parent_contract']['base_sha256']
    tail.verified_files(args.light_run/'eval64',fid.SOURCE_SHA,('per_pair.json','pool3.npz'))
    tail.verified_files(args.archived,retention.CONTROL_SHA,('per_pair.json','pool3.npz','result.json'))
    manifest = tail.read(args.light_run/'eval64/review_manifest.json')
    assert tail.sha(args.light_run/'selection.json')==manifest['selection.json']
    assert tail.sha(args.light_run/'protocol.json')==manifest['protocol.json']
    assert tail.metric_assets()==args.protocol['metric_assets']
    spec = dict(start_step=64,start_sha256=args.heads['B'],start_a_sha256=args.heads['A'],
                base_sha256=args.contract['parent_contract']['base_sha256'],f68_sha256=tail.CHECKPOINT_SHA,
                seed=42,sampler=args.protocol['sampler'],trainval_sha256=tail.sha(trainval_path),
                prior_protocol_sha256=tail.sha(args.light_run/'protocol.json'),
                retention=retention.RETENTION,donor_interior=fused.COEFFICIENT,lr=LR,optimizer_updates=STEPS,
                accumulation=ACCUMULATE,validation_steps=[0,*SNAPSHOTS],train_pairs=128,validation_pairs=32,
                texture_gate='strict decrease vs retain_125',rms_gate=False,code_sha256=tail.sha(Path(__file__)))
    path = args.run/'protocol.json'
    if path.is_file():
        assert tail.read(path)==spec
    else:
        tail.write(path,spec)
    args.spec = spec
    return args


@torch.no_grad()
def training_anchors(args, models):
    original = ab.head_from_a(args,models[3],False)[0]
    path = args.run/'anchors.json'
    recorded = tail.read(path) if path.is_file() else {}
    hashes, anchors = {}, {}
    for pair in args.trainval['pairs']:
        case = pair['case_id']
        donor = fid.checked_png(args.parent/fused.RUN/'training_donors'/case/'blend.png',
                                args.contract['donor_hashes'][case])
        dest = args.run/'training_anchors'/(case+'.png')
        if case in recorded:
            offline = fid.checked_png(dest,recorded[case])
        else:
            a,_,mask = ab.render(args,models,original,pair)
            offline = fused.pixels_mixed(rgb8(a),donor,fused.alpha(mask))
            fused.save_png(dest,offline)
            models[2].adapter.clear()
        anchors[case] = donor,offline
        hashes[case] = tail.sha(dest)
        if len(anchors)%32==0:
            print('RETAIN125_ANCHORS',len(anchors),160,flush=True)
    tail.write(path,hashes)
    return anchors


def forward(args, models, head, pair, anchors):
    prediction,gt,mask = ab.render(args,models,head,pair)
    donor,offline = anchors[pair['case_id']]
    return fuse_final(prediction,donor,offline,mask),gt,mask


@torch.no_grad()
def validate(args, models, head, criterion, pairs, anchors):
    rows = []
    for pair in pairs:
        image,gt,mask = forward(args,models,head,pair,anchors)
        loss,terms = ab.collect(criterion,image,gt,mask,'inside')
        core = roi.regions(mask,20)[0]
        texture = tail.appearance(rgb8(image),roi.lab(rgb8(gt)),core)
        rows.append(dict(case=pair['case_id'],loss=float(loss),band_mae=texture['band_mae'],**terms))
        models[2].adapter.clear()
    values = {k:float(np.mean([r[k] for r in rows if r[k] is not None]))
              for k in rows[0] if k!='case' and any(r[k] is not None for r in rows)}
    return dict(rows=rows,**values)


def train(args, models):
    train_pairs = [p for p in args.trainval['pairs'] if p['split']=='train']
    valid_pairs = [p for p in args.trainval['pairs'] if p['split']=='val']
    assert (len(train_pairs),len(valid_pairs))==(128,32)
    assert set(args.contract['prefix'])=={p['case_id'] for p in (*train_pairs,*valid_pairs)}
    training = args.run/'training.json'
    head,params = tail.candidates(models[3])
    if training.is_file():
        saved = tail.read(training)
        assert saved['protocol']==args.spec and saved['updates']==STEPS
        path = args.run/f"step{saved['selected']}.pt"
        assert tail.sha(path)==saved['validation'][str(saved['selected'])]['sha256']
        return ab.update_head(head,params,path).eval().requires_grad_(False),saved

    ab.update_head(head,params,args.start)
    assert set(params)=={n for n,p in head.named_parameters() if p.requires_grad}
    assert not any(p.requires_grad for model in models[4].values() for p in model.parameters())
    anchors = training_anchors(args,models)
    criterion = light.LightingObjective(args)
    probe = valid_pairs[0]
    pred,target,mask = ab.render(args,models,head,probe)
    donor,offline = anchors[probe['case_id']]
    prediction = fuse_final(pred,donor,offline,mask)
    expected = uniform_mix(fused.pixels_mixed(rgb8(pred),donor,fused.alpha(mask)),offline,mask)
    np.testing.assert_array_equal(rgb8(prediction),expected)
    value,_ = criterion(prediction,target,mask,'inside')
    grad = tail.backward(value,params)
    assert grad>0
    models[2].adapter.clear()
    del pred,prediction,value,target
    tail.write(args.run/'qualification.json',dict(case=probe['case_id'],old_B_sha256=args.heads['B'],
               expected_pixel_match=True,gradient_norm=grad,optimizer_updates=0,trainable_parameters=tail.PARAMETERS))
    optimizer = torch.optim.AdamW(params.values(),lr=LR)
    optimizer.zero_grad(set_to_none=True)
    checkpoints = {'0':validate(args,models,head,criterion,valid_pairs,anchors)}
    checkpoints['0']['sha256'] = args.heads['B']
    print('RETAIN125_VAL',0,checkpoints['0']['loss'],checkpoints['0']['lpips'],flush=True)
    order = np.random.default_rng(42).permutation(len(train_pairs))
    trace = []
    for i,index in enumerate(order,start=1):
        pair = train_pairs[index]
        rgb,gt,mask = forward(args,models,head,pair,anchors)
        loss,terms = ab.collect(criterion,rgb,gt,mask,'inside')
        (loss*(SCALE/ACCUMULATE)).backward()
        trace.append(dict(pair=pair['case_id'],loss=float(loss.detach()),**terms))
        models[2].adapter.clear()
        del loss,rgb,gt
        if i%ACCUMULATE:
            continue
        assert all(p.grad is not None for p in params.values())
        for p in params.values():
            p.grad.div_(SCALE)
        step = i//ACCUMULATE
        norm = float(torch.nn.utils.clip_grad_norm_(params.values(),1,error_if_nonfinite=True))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        trace[-1].update(step=step,grad_norm=norm)
        if step%8==0:
            print('RETAIN125_TRAIN',step,STEPS,trace[-1]['loss'],flush=True)
        if step in SNAPSHOTS:
            report = validate(args,models,head,criterion,valid_pairs,anchors)
            path = args.run/f'step{step}.pt'
            tail.save_torch(path,{k:p.detach().cpu() for k,p in params.items()})
            checkpoints[str(step)] = dict(report,sha256=tail.sha(path))
            print('RETAIN125_VAL',step,report['loss'],report['lpips'],flush=True)
    ab.audit_frozen(args,models,head,params)
    picked = min(SNAPSHOTS,key=lambda step:checkpoints[str(step)]['loss'])
    assert tail.sha(args.run/f'step{picked}.pt')==checkpoints[str(picked)]['sha256']
    ab.update_head(head,params,args.run/f'step{picked}.pt')
    saved = dict(protocol=args.spec,selected=picked,validation=checkpoints,updates=STEPS,trace=trace,
                 trainable=list(params),parameter_count=tail.PARAMETERS,frozen_unchanged=True,
                 selection_rule='minimum validation fused objective among 16, 32, 64 updates')
    tail.write(training,saved)
    print('RETAIN125_SELECTED',picked,checkpoints[str(picked)]['loss'],flush=True)
    return head.eval().requires_grad_(False),saved


def cached_scores(args, small, context):
    if small:
        prior = args.light_run/'eval64'
        rows = tail.read(prior/'per_pair.json')
        uniform = tail.read(args.archived/'per_pair.json')
        assert [r['index'] for r in rows]==tail.subset(context.protocol)
        assert len(uniform)==len(rows)==64
        for row,baseline in zip(rows,uniform):
            assert all(row[k]==baseline[k] for k in ('index','case_id','category','interior_px'))
            for key in ('pngs','scores','texture'):
                row[key][BASE] = baseline[key][BASE]
        with np.load(prior/'pool3.npz',allow_pickle=False) as control, \
                np.load(args.archived/'pool3.npz',allow_pickle=False) as cached:
            features = {n:cached[n] if n==BASE else control[n] for n in (*retention.SCORED,'gt')}
        archived = tail.read(args.archived/'result.json')
    else:
        prior = context.out/'eval500'
        tail.verified_files(prior,class_ab.SOURCE_SHA,('protocol.json','per_pair.json','pool3.npz','result.json'))
        rows = tail.read(prior/'per_pair.json')
        assert len(rows)==500 and [r['index'] for r in rows]==list(range(500))
        with np.load(prior/'pool3.npz',allow_pickle=False) as cached:
            features = {n:cached[n] for n in (*retention.SCORED,'gt')}
        archived = tail.read(prior/'result.json')
    assert all(x.shape==(len(rows),2048) and np.isfinite(x).all() for x in features.values())
    return rows,features,archived


def archived_pixels(args, context, row, mask, small):
    case = row['case_id']+'.png'
    if small:
        donor = fid.checked_png(args.parent/'test/blend'/case,row['pngs']['blend'])
        a = fid.checked_png(args.parent/'test/A'/case,row['pngs']['A'])
        offline = fused.pixels_mixed(a,donor,fused.alpha(mask))
        old = fid.checked_png(args.light_run/'images/eval64'/case,row['pngs']['B'])
        saved = fid.checked_png(args.archived/'images'/BASE/case,row['pngs'][BASE])
        np.testing.assert_array_equal(uniform_mix(old,offline,mask),saved)
    else:
        donor = fid.checked_png(context.out/'images/blend'/case,row['pngs']['blend'])
        offline = fid.checked_png(context.out/'images/offline25'/case,row['pngs']['offline25'])
    return donor,offline


def candidate_raw(context, models, head, pair, mask, row, path, reference, checkpoint, frozen_sha):
    assert reference['base_sha256']==frozen_sha and reference['png_sha256']==row['pngs']['off']
    assert reference['case']==pair['case_id'] and reference['index']==row['index']
    meta = path.with_suffix('.json')
    audit = dict(case=pair['case_id'],index=row['index'],seed=pair['seed'],checkpoint=checkpoint,
                 base_sha256=frozen_sha,steps=tail.LAST,
                 initial_noise_digest=reference['initial_noise_digest'],first_b_step=reference['first_b_step'])
    if meta.is_file():
        saved = tail.read(meta)
        assert all(saved[k]==v for k,v in audit.items())
        return fid.checked_png(path,saved['png_sha256']),True
    core,pipe,net,original,_ = models
    gate = tail.latent_gate(mask,context.device)
    module = tail.TailResidual(original,head,gate)
    picture,noise,tap = tail.joint.sample(context,core,pipe,net,pair,tail.CONDITION,b=module,mode='raw',capture=True)
    assert picture.size==tail.SIZE and tap.index==tail.LAST and len(tap.states)==25
    assert noise==audit['initial_noise_digest']
    assert {k:tail.tensor_sha(tap.states[0][k]) for k in ('zt','epsilon')}==audit['first_b_step']
    pixels = np.asarray(picture)
    fused.save_png(path,pixels)
    tail.write(meta,dict(audit,png_sha256=tail.sha(path)))
    net.adapter.clear()
    return pixels,False


@torch.no_grad()
def evaluate(args, models, head, trained, small):
    context = copy.copy(args)
    if small:
        context.out = args.baseline
        reference_dir = args.parent/'test/off'
    else:
        context.out = args.light_run/independent.BAND_RUN
        context.protocol = tail.read(context.out/'eval500/protocol.json')
        assert context.protocol['base_sha256']==args.spec['base_sha256'] and context.protocol['heads']==args.heads
        reference_dir = context.out/'test/off'
    cohort = 'eval64' if small else 'eval500'
    dest = args.run/cohort
    dest.mkdir(exist_ok=True)
    if (dest/'review.zip').is_file():
        return tail.read(dest/'result.json')
    rows,features,archived = cached_scores(args,small,context)
    lpips,inception = tail.metric_models('cpu',context.protocol['metric_assets'])
    features[ARM] = np.empty_like(features[BASE])
    checkpoint = trained['validation'][str(trained['selected'])]['sha256']
    reused = 0
    for start in range(0,len(rows),4):
        pics,truth = [],[]
        for row in rows[start:start+4]:
            pair = context.protocol['pairs'][row['index']]
            case = row['case_id']
            assert (case,row['category'],pair['seed'])==(pair['case_id'],pair['category'],42)
            mask = tail.case_mask(context,pair)
            target_path = args.data/pair['gt']
            assert tail.sha(target_path)==pair['input_sha256']['gt']
            gt = tail.rgb(target_path).resize(tail.SIZE,Image.Resampling.BILINEAR)
            inside = roi.regions(mask,20)[0]
            assert int(inside.sum())==row['interior_px']
            donor,offline = archived_pixels(args,context,row,mask,small)
            ref = tail.read(reference_dir/(case+'.json'))
            raw,hit = candidate_raw(context,models,head,pair,mask,row,dest/'raw'/(case+'.png'),
                                    ref,checkpoint,args.spec['base_sha256'])
            reused += hit
            current = fused.pixels_mixed(raw,donor,fused.alpha(mask))
            photo = uniform_mix(current,offline,mask)
            path = dest/'images'/(case+'.png')
            fused.save_png(path,photo)
            row['pngs'][ARM] = tail.sha(path)
            row['scores'][ARM] = tail.image_scores(Image.fromarray(photo),gt,path,target_path,mask)
            row['texture'][ARM] = tail.appearance(photo,roi.lab(np.asarray(gt)),inside)
            pics.append(photo)
            truth.append(np.asarray(gt))
        x,y = roi.tensors(pics),roi.tensors(truth)
        features[ARM][start:start+len(pics)] = inception(x)[0].flatten(1).numpy()
        for row,value in zip(rows[start:start+len(pics)],lpips(x*2-1,y*2-1).flatten().tolist()):
            row['scores'][ARM]['lpips'] = value
        if start%20==0 or start+len(pics)==len(rows):
            print('RETAIN125_EVAL',cohort,start+len(pics),len(rows),flush=True)
    assert np.isfinite(features[ARM]).all()
    assert tail.tensor_set(models[4])==args.original['frozen_model_hashes']
    result = tail.summarize(rows,features,ARMS,(BASE,'offline25'),(ARM,),fid_reference=BASE)
    class_ab.summarize_texture(result,rows,ARMS)
    group = result['groups']['all']
    ref = archived['groups']['all']
    for k,v in group['arms'][BASE].items():
        field = 'fid' if k=='fid' else 'mean'
        np.testing.assert_allclose(v[field],ref['arms'][BASE][k][field],atol=1e-9,rtol=1e-12)
    np.testing.assert_allclose(group['texture'][BASE]['band_mae'],ref['texture'][BASE]['band_mae'],atol=1e-12,rtol=0)
    gate = class_ab.uniform_gate(result,ARM)
    gate['accepted'] = gate['passed'] and all(gate['vs_uniform'].values())
    result['acceptance'] = gate
    result['texture_threshold'] = min(group['texture'][n]['band_mae'] for n in ('off','on'))
    valid = [r for r in rows if r['interior_px']>=256]
    result['band_mae_paired'] = roi.paired([r['texture'][BASE]['band_mae'] for r in valid],
                                         [r['texture'][ARM]['band_mae'] for r in valid],
                                         np.array([r['category'] for r in valid]))
    result['fid_delta_ci95'] = class_ab.fid_intervals(
        features,np.array([r['category'] for r in rows]),arms=(BASE,ARM),
        contrast=lambda values:{ARM:values[ARM]-values[BASE]})[ARM]
    tail.write(dest/'result.json',result)
    tail.write(dest/'per_pair.json',rows)
    np.savez_compressed(dest/'candidate_pool3.npz',indices=[r['index'] for r in rows],**{ARM:features[ARM]})
    tail.write(dest/'audit.json',dict(pairs=len(rows),baseline_pairs_reused=len(rows),baseline_rollouts=0,
        candidate_rollouts=len(rows)-reused,cached_candidate_rollouts=reused,selected=trained['selected'],
        checkpoint_sha256=checkpoint,model_frozen=True,protocol=args.spec))
    fid.pack(dest,('result.json','per_pair.json','candidate_pool3.npz','audit.json'))
    print('RETAIN125_GATE',cohort,gate['accepted'],gate['vs_uniform'],flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prior',type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
    args = setup(args.prior)
    models = ab.load(args)
    head,trained = train(args,models)
    small = evaluate(args,models,head,trained,True)
    passed = small['acceptance']['accepted']
    if passed:
        evaluate(args,models,head,trained,False)
    files = ['protocol.json','qualification.json','anchors.json','training.json','eval64/review.zip']
    if passed:
        files.append('eval500/review.zip')
    tail.write(args.run/'completion.json',dict(eval64_passed=passed,eval500_executed=passed,
               eval500_passed=tail.read(args.run/'eval500/result.json')['acceptance']['accepted'] if passed else None))
    fid.pack(args.run,(*files,'completion.json'))
    print('RETAIN125_FINISHED',passed,tail.sha(args.run/'review.zip'),flush=True)


if __name__=='__main__':
    main()
