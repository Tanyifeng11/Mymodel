"""在最终统一 12.5% 图上，以感知保护和边缘起点约束短程续训 B。"""
import argparse
import copy
import os
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from tools import e2_f68_e1_retain125_finetune as ft


RUN = 'retain125_guarded_finetune_20261122'
ARM = 'retain125_guard'
SOURCE_SHA = 'ed5b0d1cff1b8cde198732058402db4db0e2f899dd45cffe1f00c4797102b311'
SNAPSHOTS = (8, 16, 32, 64)
EDGE_WEIGHT = .20
GUARD_WEIGHT = 1.
tail, ab, roi, fid = ft.tail, ft.ab, ft.roi, ft.fid


def setup(prior):
    args = ft.retention.setup(prior)
    args.previous = args.light_run/ft.RUN
    tail.verified_files(args.previous,SOURCE_SHA,('protocol.json','anchors.json','training.json'))
    previous = tail.read(args.previous/'protocol.json')
    args.run = args.light_run/RUN
    args.run.mkdir(exist_ok=True)
    args.action,args.device = 'train','cuda:0'
    args.out = args.baseline/tail.distill.RUN
    args.trainval = tail.distill.experiment_protocol(args)
    args.start = args.light_run/'B/step64.pt'
    assert tail.sha(args.start)==args.heads['B']==previous['start_sha256']
    assert tail.sha(args.base_checkpoint)==args.contract['parent_contract']['base_sha256']==previous['base_sha256']
    assert tail.sha(args.out/'protocol.json')==previous['trainval_sha256']
    assert tail.metric_assets()==args.protocol['metric_assets']
    spec = dict(previous,code_sha256=tail.sha(Path(__file__)),validation_steps=[0,*SNAPSHOTS],
        protected_start_archive=SOURCE_SHA,guard_weight=GUARD_WEIGHT,edge_weight=EDGE_WEIGHT,
        edge_region='sketch eight pixel inner and outer band',training_module_sha256=tail.sha(Path(ft.__file__)),
        selection='five relative gates, then LPIPS, band MAE, earlier step, fallback zero')
    path = args.run/'protocol.json'
    if path.is_file():
        assert tail.read(path)==spec
    else:
        tail.write(path,spec)
    args.spec = spec
    return args


def anchors(args):
    hashes = tail.read(args.previous/'anchors.json')
    assert set(hashes)==set(args.contract['prefix'])
    values = {}
    for case in hashes:
        donor = fid.checked_png(args.parent/ft.fused.RUN/'training_donors'/case/'blend.png',
                                args.contract['donor_hashes'][case])
        offline = fid.checked_png(args.previous/'training_anchors'/(case+'.png'),hashes[case])
        values[case] = donor,offline
    return values


@torch.no_grad()
def references(args,models,head,base,pairs,fixed):
    manifest = args.run/'references.json'
    recorded = tail.read(manifest) if manifest.is_file() else {}
    images = {}
    previous = {r['case']:r for r in tail.read(args.previous/'training.json')['validation']['0']['rows']}
    for pair in pairs:
        case = pair['case_id']
        path = args.run/'references'/(case+'.png')
        if case in recorded:
            image = fid.checked_png(path,recorded[case]['png_sha256'])
        else:
            pred,target,mask = ft.forward(args,models,head,pair,fixed)
            loss,terms = ab.collect(base,pred,target,mask,'inside')
            image = ft.rgb8(pred)
            ft.fused.save_png(path,image)
            recorded[case] = dict(png_sha256=tail.sha(path),loss=float(loss),**terms)
            tail.write(manifest,recorded)
            models[2].adapter.clear()
        if case in previous:
            for key in ('lpips','inside_lpips','rgb_l1','low_lstar'):
                np.testing.assert_allclose(recorded[case][key],previous[case][key],atol=5e-6,rtol=0)
        images[case] = image
        if len(images)%32==0:
            print('GUARDED_REFERENCES',len(images),160,flush=True)
    assert len(images)==len(recorded)==160
    return recorded,images


class Objective:
    def __init__(self,base,reference,images):
        self.base,self.reference,self.images = base,reference,images

    def __call__(self,prediction,target,mask,pair):
        loss,terms = self.base(prediction,target,mask,'inside')
        ref = self.reference[pair['case_id']]
        guard = F.relu(terms['lpips']-prediction.new_tensor(ref['lpips']))
        guard = guard+F.relu(terms['inside_lpips']-prediction.new_tensor(ref['inside_lpips']))
        image = roi.tensors([self.images[pair['case_id']]]).to(prediction)
        edge = torch.as_tensor(roi.regions(mask,8)[1],device=prediction.device)
        error = (prediction-image).abs().mean(1)[0]
        edge_loss = ab.weighted_mean(error,edge)
        return loss+GUARD_WEIGHT*guard+EDGE_WEIGHT*edge_loss,dict(
            terms,l0=loss,perceptual_guard=guard,edge_anchor=edge_loss)


def collect(criterion,rgb,target,mask,pair):
    loss,terms = criterion(rgb,target,mask,pair)
    return loss,{k:float(v.detach()) for k,v in terms.items()}


@torch.no_grad()
def validate(args,models,head,criterion,pairs,fixed,reference,images,inception,step):
    directory = args.run/'validation'
    directory.mkdir(exist_ok=True)
    record = directory/f'step{step}.json'
    features = directory/f'step{step}_pool3.npz'
    if record.is_file():
        result = tail.read(record)
        assert result['features_sha256']==tail.sha(features) and result['step']==step
        assert result['protocol']==args.spec
        return result
    rows,pixels,targets = [],[],[]
    for pair in pairs:
        case = pair['case_id']
        if step==0:
            image = images[case]
            mask = tail.case_mask(args,pair)
            gt_path = args.data/pair['gt']
            target = tail.rgb(gt_path).resize(tail.SIZE,Image.Resampling.BILINEAR)
            values = dict(reference[case])
            values.pop('png_sha256')
            values.update(l0=values['loss'],perceptual_guard=0.,edge_anchor=0.)
        else:
            pred,truth,mask = ft.forward(args,models,head,pair,fixed)
            loss,values = collect(criterion,pred,truth,mask,pair)
            values['loss'] = float(loss)
            image = ft.rgb8(pred)
            target = Image.fromarray(ft.rgb8(truth))
            gt_path = args.data/pair['gt']
        path = directory/f'step{step}'/(case+'.png')
        if step==0:
            path = args.run/'references'/(case+'.png')
        else:
            ft.fused.save_png(path,image)
        metrics = tail.image_scores(Image.fromarray(image),target,path,gt_path,mask)
        core = roi.regions(mask,20)[0]
        appearance = tail.appearance(image,roi.lab(np.asarray(target)),core)
        rows.append(dict(case=case,category=pair['category'],png_sha256=tail.sha(path),
                         band_mae=appearance['band_mae'],**values,**metrics))
        pixels.append(image)
        targets.append(np.asarray(target))
        models[2].adapter.clear()
    predicted,gt = [],[]
    for i in range(0,len(pixels),4):
        predicted.append(inception(roi.tensors(pixels[i:i+4]))[0].flatten(1).numpy())
        gt.append(inception(roi.tensors(targets[i:i+4]))[0].flatten(1).numpy())
    predicted,gt = np.concatenate(predicted),np.concatenate(gt)
    terms = roi.fid_terms(predicted,gt)
    np.savez_compressed(features,prediction=predicted,gt=gt)
    means = {k:float(np.mean([r[k] for r in rows if r[k] is not None]))
             for k in rows[0] if k not in ('case','category','png_sha256') and any(r[k] is not None for r in rows)}
    result = dict(step=step,rows=rows,**means,fid=terms['fid'],fid_terms=terms,
                  features_sha256=tail.sha(features),protocol=args.spec)
    tail.write(record,result)
    print('GUARDED_VALIDATION',step,result['lpips'],result['fid'],result['contour_f1'],flush=True)
    return result


def choose(checkpoints):
    zero = checkpoints['0']
    requirements = {}
    for s,c in checkpoints.items():
        if s=='0':
            continue
        checks = dict(fid=c['fid']<=zero['fid'],lpips=c['lpips']<zero['lpips'],
            band_mae=c['band_mae']<zero['band_mae'],ssim=c['ssim']>=zero['ssim'],
            contour_f1=c['contour_f1']>=zero['contour_f1'])
        requirements[s] = checks
    eligible = [int(s) for s,v in requirements.items() if all(v.values())]
    selected = min(eligible,key=lambda s:(checkpoints[str(s)]['lpips'],
                   checkpoints[str(s)]['band_mae'],s)) if eligible else 0
    return selected,requirements


def train(args,models):
    train_pairs = [p for p in args.trainval['pairs'] if p['split']=='train']
    valid_pairs = [p for p in args.trainval['pairs'] if p['split']=='val']
    assert (len(train_pairs),len(valid_pairs))==(128,32)
    head,params = tail.candidates(models[3])
    if (args.run/'training.json').is_file():
        done = tail.read(args.run/'training.json')
        assert done['protocol']==args.spec and done['updates']==ft.STEPS and done['frozen_unchanged']
        path = args.start if done['selected']==0 else args.run/f"step{done['selected']}.pt"
        assert tail.sha(path)==done['validation'][str(done['selected'])]['sha256']
        return ab.update_head(head,params,path).eval().requires_grad_(False),done
    ab.update_head(head,params,args.start)
    assert sum(v.numel() for v in params.values())==tail.PARAMETERS
    assert not any(p.requires_grad for module in models[4].values() for p in module.parameters())
    fixed = anchors(args)
    base = ft.light.LightingObjective(args)
    reference,images = references(args,models,head,base,args.trainval['pairs'],fixed)
    criterion = Objective(base,reference,images)
    probe = valid_pairs[0]
    rgb,target,mask = ft.forward(args,models,head,probe,fixed)
    np.testing.assert_array_equal(ft.rgb8(rgb),images[probe['case_id']])
    loss,terms = criterion(rgb,target,mask,probe)
    assert float(terms['perceptual_guard'])==0. and float(terms['edge_anchor'])==0.
    norm = tail.backward(loss,params)
    assert norm>0
    models[2].adapter.clear()
    del loss,rgb,target
    ab.audit_frozen(args,models,head,params)
    tail.write(args.run/'qualification.json',dict(start_sha256=args.heads['B'],reference_pairs=160,
        start_pixels_equal=True,guard_at_start=0.,edge_at_start=0.,gradient_norm=norm,
        optimizer_updates=0,trainable_parameters=tail.PARAMETERS))
    from pytorch_fid.inception import InceptionV3
    inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(params.values(),lr=ft.LR)
    optimizer.zero_grad(set_to_none=True)
    checkpoints = {'0':validate(args,models,head,criterion,valid_pairs,fixed,reference,images,inception,0)}
    checkpoints['0']['sha256'] = args.heads['B']
    trace = []
    completed = 0
    resume = args.run/'resume.pt'
    if resume.is_file():
        state = torch.load(resume,map_location='cpu',weights_only=False)
        assert state['protocol']==args.spec
        completed = state['step']
        ab.update_head(head,params,args.run/f'step{completed}.pt')
        optimizer.load_state_dict(state['optimizer'])
        trace,checkpoints = state['trace'],state['validation']
    for i,index in enumerate(np.random.default_rng(42).permutation(128),start=1):
        if i<=completed*ft.ACCUMULATE:
            continue
        pair = train_pairs[index]
        rgb,target,mask = ft.forward(args,models,head,pair,fixed)
        loss,terms = collect(criterion,rgb,target,mask,pair)
        (loss*(ft.SCALE/ft.ACCUMULATE)).backward()
        trace.append(dict(pair=pair['case_id'],loss=float(loss.detach()),**terms))
        models[2].adapter.clear()
        del loss,rgb,target
        if i%ft.ACCUMULATE:
            continue
        assert all(p.grad is not None for p in params.values())
        for p in params.values():
            p.grad.div_(ft.SCALE)
        step = i//ft.ACCUMULATE
        norm = float(torch.nn.utils.clip_grad_norm_(params.values(),1,error_if_nonfinite=True))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        trace[-1].update(step=step,grad_norm=norm)
        if step%8==0:
            print('GUARDED_TRAIN',step,ft.STEPS,trace[-1]['loss'],flush=True)
        if step in SNAPSHOTS:
            path = args.run/f'step{step}.pt'
            tail.save_torch(path,{k:p.detach().cpu() for k,p in params.items()})
            report = validate(args,models,head,criterion,valid_pairs,fixed,reference,images,inception,step)
            checkpoints[str(step)] = dict(report,sha256=tail.sha(path))
            tail.save_torch(resume,dict(step=step,protocol=args.spec,optimizer=optimizer.state_dict(),
                                       trace=trace,validation=checkpoints))
    assert len(trace)==128
    ab.audit_frozen(args,models,head,params)
    selected,checks = choose(checkpoints)
    path = args.start if selected==0 else args.run/f'step{selected}.pt'
    assert tail.sha(path)==checkpoints[str(selected)]['sha256']
    ab.update_head(head,params,path)
    done = dict(protocol=args.spec,updates=ft.STEPS,trace=trace,validation=checkpoints,
                validation_checks=checks,selected=selected,parameter_count=tail.PARAMETERS,
                trainable=list(params),frozen_unchanged=True)
    tail.write(args.run/'training.json',done)
    print('GUARDED_SELECTED',selected,checks,flush=True)
    return head.eval().requires_grad_(False),done


def reused_start64(args,trained):
    context = copy.copy(args)
    context.out = args.baseline
    rows,features,_ = ft.cached_scores(args,True,context)
    for row in rows:
        for key in ('scores','texture','pngs'):
            row[key][ARM] = row[key][ft.BASE]
    features[ARM] = features[ft.BASE]
    variants = (*ft.retention.SCORED,ARM)
    result = tail.summarize(rows,features,variants,(ft.BASE,'offline25'),(ARM,),fid_reference=ft.BASE)
    ft.class_ab.summarize_texture(result,rows,variants)
    result['acceptance'] = ft.class_ab.uniform_gate(result,ARM)
    result['acceptance']['accepted'] = False
    result['fid_delta_ci95'] = [0.,0.]
    result['texture_threshold'] = min(result['groups']['all']['texture'][k]['band_mae'] for k in ('off','on'))
    good = [r for r in rows if r['interior_px']>=256]
    values = [r['texture'][ft.BASE]['band_mae'] for r in good]
    result['band_mae_paired'] = roi.paired(values,values,np.array([r['category'] for r in good]))
    dest = args.run/'eval64'
    dest.mkdir(exist_ok=True)
    (dest/'images').mkdir(exist_ok=True)
    for row in rows:
        path = dest/'images'/(row['case_id']+'.png')
        if not path.exists():
            path.symlink_to(args.archived/'images'/ft.BASE/path.name)
        assert tail.sha(path)==row['pngs'][ARM]
    tail.write(dest/'result.json',result)
    tail.write(dest/'per_pair.json',rows)
    np.savez_compressed(dest/'candidate_pool3.npz',indices=[r['index'] for r in rows],**{ARM:features[ARM]})
    tail.write(dest/'audit.json',dict(pairs=64,baseline_pairs_reused=64,baseline_rollouts=0,
        candidate_rollouts=0,reused_start_images=64,selected=0,
        checkpoint_sha256=args.heads['B'],model_frozen=True,protocol=args.spec))
    fid.pack(dest,('result.json','per_pair.json','candidate_pool3.npz','audit.json'))
    print('GUARDED_REUSED_START64',64,flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prior',type=Path)
    cv2.setNumThreads(1)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
    args = setup(parser.parse_args().prior)
    models = ab.load(args)
    head,trained = train(args,models)
    with patch.object(ft,'ARM',ARM),patch.object(ft,'ARMS',(*ft.retention.SCORED,ARM)):
        small = reused_start64(args,trained) if trained['selected']==0 else ft.evaluate(args,models,head,trained,True)
        passed = small['acceptance']['accepted']
        if passed:
            ft.evaluate(args,models,head,trained,False)
    finish = dict(selected=trained['selected'],eval64_passed=passed,eval500_executed=passed,
        eval500_passed=tail.read(args.run/'eval500/result.json')['acceptance']['accepted'] if passed else None)
    tail.write(args.run/'completion.json',finish)
    files = ['protocol.json','qualification.json','references.json','training.json','completion.json','eval64/review.zip']
    for step in (0,*SNAPSHOTS):
        files.extend([f'validation/step{step}.json',f'validation/step{step}_pool3.npz'])
    if passed:
        files.append('eval500/review.zip')
    fid.pack(args.run,files)
    print('GUARDED_FINISHED',trained['selected'],passed,tail.sha(args.run/'review.zip'),flush=True)


if __name__=='__main__':
    main()
