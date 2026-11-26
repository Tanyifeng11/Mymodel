"""从 127720 第960步等步比较原损失与固定半量带通目标。"""
import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

from tools import e2_b_single_direct_960 as direct
from tools.e2_b_direct_127720_region_diagnostic import A64_SHA, DIRECT_SHA
from tools.e2_fft_experiment import SOURCE

single, previous, init, tail, cov = direct.single, direct.previous, direct.init, direct.tail, direct.cov
roi, ab, CATEGORIES = cov.roi, cov.ab, direct.CATEGORIES
RUN, STEPS, ARMS = 'fixed_band50_ab_64', 64, ('A', 'B')


def setup(prior, device):
    args, inherited = direct.setup(prior, device)
    origin = args.run
    assert tail.sha(origin/'review.zip') == DIRECT_SHA
    manifest = tail.read(origin/'review_manifest.json')
    checked = single.fusion.read_checked
    source = checked(origin/'protocol.json', manifest['protocol.json'])
    assert source == inherited
    done = checked(origin/'completion.json', manifest['completion.json'])
    assert done['updates']==960 and done['frozen_unchanged'] and done['trajectories']==1
    stage = origin/'validation/step960'
    result = checked(stage/'result.json', manifest['validation/step960/result.json'])
    assert result['pairs']==256 and result['protocol_sha256']==manifest['protocol.json']
    assert tail.sha(stage/'pool3.npz')==result['pool3_sha256']
    ids = [p['case_id'] for p in args.pairs['val']]
    rows = tail.read(stage/'per_pair.json')
    assert [r['case_id'] for r in rows]==ids

    comparison = args.parent_run/previous.RUN/cov.RUN/'b64_pants_inception_band50_ab'
    assert tail.sha(comparison/'review.zip')==A64_SHA
    expected = tail.read(comparison/'review_manifest.json')
    ref = checked(comparison/'protocol.json',expected['protocol.json'])
    order = ref['training_order']
    assert len(order)==len(set(order))==STEPS*single.BATCH and order==source['training_order'][-len(order):]
    prior_trace = checked(comparison/'training.json',expected['training.json'])['trace']
    assert order==[r['case'] for step in prior_trace for r in step['A']['rows']]
    train = {p['case_id']:p for p in args.pairs['train']}
    args.train_order = [train[k] for k in order]
    assert Counter(p['category'] for p in args.train_order)==dict.fromkeys(CATEGORIES,32)
    train_ids = set(order)
    assert all(train_ids.isdisjoint(p['case_id'] for p in args.pairs[s]) for s in ('val','test64','test500'))
    with np.load(stage/'pool3.npz',allow_pickle=False) as u, np.load(
            comparison/'validation/step64/pool3.npz',allow_pickle=False) as v:
        np.testing.assert_array_equal(u['gt'],v['gt'])

    args.start_head = stage/'direct.pt'
    args.start_digest = result['head_sha256']['direct']
    args.anchor_origin = origin/'renders'/args.start_digest
    args.anchor_spec = manifest['protocol.json']
    args.target_run = origin/RUN
    args.run = args.target_run/'gradient_qualification_retry'
    args.run.mkdir(exist_ok=True,parents=True)
    args.spec = args.run/'protocol.json'
    contract = dict(parent_job=127720, parent_step=960, parent_review_sha256=DIRECT_SHA,
        baseline_a64_review_sha256=A64_SHA, start_sha256=tail.sha(args.start_head), start_head=args.start_digest,
        sampler=source['sampler'], model_sha256=source['model_sha256'], cohort_sha256=source['cohort_sha256'],
        training_order=order, steps_per_arm=STEPS, optimizer='fresh AdamW per arm', learning_rate=single.LR,
        effective_batch=single.BATCH, gpus=1, A_loss='H2 + original AlexNet covariance',
        B_change=dict(old='0.10 H2 band', new='0.10 fixed target band', fraction=.5, core_radius=20,
                      feather_px=4, band_sigma=[1,4], min_core_pixels=256, categories=list(CATEGORIES),
                      anchor='fixed 960 RGB8', target='training GT only', gradient='only r20 in RGB'),
        parent_metrics=direct.summarize(result,'direct'),
        parent_pants_fid=result['groups']['pants']['arms']['direct']['fid']['fid'],
        common_loss=source['common_loss'], alexnet_covariance=source['alexnet_covariance'],
        thresholds=source['thresholds'], baseline_a64=source['baseline_a64'], baseline_b_off=source['baseline_b_off'],
        tolerance=source['tolerance'], formal_reference=source['formal_reference'],
        code={n:tail.sha(SOURCE/n) for n in (
            'tools/e2_b_fixed_band50_ab.py','scripts/e2_b_fixed_band50_ab.sbatch',
            'tools/e2_b_single_pants_ab.py','tools/e2_b_single_pants_cov_ab.py',
            'tools/e2_b_single_pants_inception_ab.py','tools/e2_b_single_pants_lstar_equal50.py',
            'tools/e2_b_epoch20_single.py')})
    return args,contract


@torch.no_grad()
def prepare_targets(args, contract):
    source = args.target_run
    prior_spec = tail.read(source/'protocol.json')
    assert {k:v for k,v in prior_spec.items() if k!='code'}=={
        k:v for k,v in contract.items() if k!='code'
    }
    source_sha = tail.sha(source/'protocol.json')
    manifest = tail.read(source/'target_manifest.json')
    assert manifest['protocol_sha256']==source_sha
    assert set(manifest['rows'])=={p['case_id'] for p in args.train_order}
    targets,counts = {},Counter()
    for pair in args.train_order:
        case,category = pair['case_id'],pair['category']
        row = manifest['rows'][case]
        core = roi.regions(tail.case_mask(args,pair),cov.CORE_RADIUS)[0]
        assert (category,int(core.sum()),bool(core.sum()>=cov.MIN_PIXELS))==(
            row['category'],row['core_px'],row['eligible'])
        if not row['eligible']:
            continue
        assert row['protocol_sha256']==source_sha and row['mask_sha256']==pair['mask_sha256']
        assert row['gt_sha256']==args.gt_hashes[case]==tail.sha(args.data/pair['gt'])
        image = source/'fixed_train_targets'/(case+'.png')
        assert tail.read(image.with_suffix('.json'))=={k:v for k,v in row.items() if k!='changed_pixels'}
        assert tail.sha(image)==row['png_sha256']
        anchor = args.anchor_origin/(case+'.png')
        if not anchor.is_file():
            anchor = source/'renders'/args.start_digest/(case+'.png')
        assert tail.sha(anchor)==row['anchor_sha256']
        target,reference = np.asarray(tail.rgb(image)),np.asarray(tail.rgb(anchor))
        np.testing.assert_array_equal(target[~core],reference[~core])
        assert np.count_nonzero(np.any(target!=reference,axis=2))==row['changed_pixels']
        targets[case] = roi.tensors([target])
        counts[category] += 1
    assert len(targets)==115 and all(counts[c]==manifest['counts'][f'eligible_{c}'] for c in CATEGORIES)
    tail.write(args.run/'target_manifest.json',dict(manifest,protocol_sha256=args.spec_sha,
        reused_from_job=127788,source_manifest_sha256=tail.sha(source/'target_manifest.json')))
    print('FIXED50_TARGETS_REUSED',len(targets),dict(counts),flush=True)
    return targets


class Objective(cov.Objective):
    def fixed_band(self,pred,target,core):
        keep = torch.from_numpy(core).to(pred.device)
        scoped = torch.where(keep[None,None],pred,pred.detach())
        with torch.no_grad():
            reference = self.base.band(target.to(pred))
        return (self.base.band(scoped)-reference).abs()[0,0][keep].mean()

    def __call__(self,pred,gt,mask,arm,category,target):
        if arm=='A':
            return super().__call__(pred,gt,mask,'B',category)
        total,terms = single.Objective.__call__(self,pred,gt,mask,0.)
        core = roi.regions(mask,cov.CORE_RADIUS)[0]
        assert (target is not None)==(core.sum()>=cov.MIN_PIXELS)
        band = pred.new_zeros(())
        covariance = pred.new_zeros(())
        if target is not None:
            band = self.fixed_band(pred,target,core)
            if category=='pants':
                covariance = cov.covariance_loss(self.base.lpips,pred,gt,core)
        loss = total+previous.BASE_WEIGHT*band+cov.COV_WEIGHT*covariance
        return loss,dict(terms,band=band,pants_covariance=covariance)


def qualify(args,models,head,params,objective,targets):
    path = args.run/'qualification.json'
    if path.is_file():
        checked=tail.read(path)
        assert checked['protocol_sha256']==args.spec_sha and checked['optimizer_updates']==0
        return
    records = {}
    sources = tail.read(args.run/'target_manifest.json')['rows']
    scale = single.SCALE/single.BATCH
    for category in CATEGORIES:
        pair = next(p for p in args.train_order if p['category']==category and sources[p['case_id']].get('changed_pixels',0)>0)
        case = pair['case_id']
        pred,gt,mask = previous.forward(args,models,head,pair)
        core = roi.regions(mask,cov.CORE_RADIUS)[0]
        a,terms_a = objective(pred,gt,mask,'A',category,targets[case])
        b,terms_b = objective(pred,gt,mask,'B',category,targets[case])
        for k in terms_a.keys()-{'band'}:
            torch.testing.assert_close(terms_a[k],terms_b[k],atol=0,rtol=0)
        torch.testing.assert_close(b-a,previous.BASE_WEIGHT*(terms_b['band']-terms_a['band']),atol=5e-7,rtol=1e-4)
        band = previous.BASE_WEIGHT*terms_b['band']
        weight = head.output.weight
        rg, old = torch.autograd.grad(scale*band,(pred,weight),retain_graph=True)
        # 独立小项用更高尺度观测，正式训练仍沿用原尺度和总损失。
        large = 4096.
        isolated, = torch.autograd.grad(large*band,weight,retain_graph=True)
        ga, = torch.autograd.grad(scale*a,weight,retain_graph=True)
        gb, = torch.autograd.grad(scale*b,weight,retain_graph=True)
        gb_common, = torch.autograd.grad(scale*(b-band),weight)
        keep = torch.from_numpy(core).to(pred.device)
        def gradient_norm(value, divisor):
            return float(torch.linalg.vector_norm(value.float()/divisor))
        gradient = dict(
            rgb=gradient_norm(rg,scale), isolated_scale64=gradient_norm(old,scale),
            isolated_scale4096=gradient_norm(isolated,large),
            total_A=gradient_norm(ga,scale),total_B=gradient_norm(gb,scale),
            band_increment_in_B=gradient_norm(gb.float()-gb_common.float(),scale),
        )
        inside = int(torch.count_nonzero(rg[...,keep]))
        outside = int(torch.count_nonzero(rg[...,~keep]))
        records[category] = dict(case=case, gradients=gradient,rgb_inside_nonzero=inside,
            rgb_outside_nonzero=outside, band_A=float(terms_a['band'].detach()),
            band_B=float(terms_b['band'].detach()), production_scale=scale, isolated_probe_scale=large)
        tail.write(args.run/'qualification_probe.json',dict(protocol_sha256=args.spec_sha,probes=records))
        print('FIXED50_PROBE',category,records[category],flush=True)
        assert outside==0 and inside>0 and all(np.isfinite(v) for v in gradient.values())
        assert all(v>0 for k,v in gradient.items() if k!='isolated_scale64'), gradient
        models[2].adapter.clear()
    ab.audit_frozen(args,models,head,params)
    tail.write(path,dict(protocol_sha256=args.spec_sha,common_terms_equal=True,optimizer_updates=0,probes=records))
    print('FIXED50_QUALIFIED',records,flush=True)


def update(args,models,head,params,opt,objective,arm,pairs,targets):
    records = []
    for pair in pairs:
        pred,gt,mask = previous.forward(args,models,head,pair)
        case = pair['case_id']
        loss,terms = objective(pred,gt,mask,arm,pair['category'],targets.get(case))
        assert torch.isfinite(loss)
        (single.SCALE*loss/single.BATCH).backward()
        records.append(dict(case=case,category=pair['category'],loss=float(loss.detach()),
                            **{k:float(v.detach()) for k,v in terms.items()}))
        models[2].adapter.clear()
    for w in params.values():
        w.grad.div_(single.SCALE)
    norm = float(torch.nn.utils.clip_grad_norm_(params.values(),1.,error_if_nonfinite=True))
    opt.step()
    opt.zero_grad(set_to_none=True)
    return dict(grad_norm=norm,rows=records)


def validation(result,arm,contract):
    entry = direct.summarize(result,arm)
    m = entry['metrics']
    z = contract['thresholds']
    checks = dict(fid=m['fid']<z['fid'],lpips=m['lpips']<=z['lpips'],
                  ssim=m['ssim']>=z['ssim'],contour_f1=m['contour_f1']>=z['contour_f1'])
    for c,v in entry['texture'].items():
        assert v['eligible_pairs']==contract['baseline_a64']['texture'][c]['eligible_pairs']
        checks[f'band_mae_{c}'] = v['band_mae']<=z['band_mae'][c]
    return dict(entry, pants_fid=result['groups']['pants']['arms'][arm]['fid']['fid'],
                checks=checks,passed=all(checks.values()))


def train(args,models,head_and_params,objective,targets,metrics,contract):
    heads = {k:v[0] for k,v in head_and_params.items()}
    params = {k:v[1] for k,v in head_and_params.items()}
    optimizers = {k:torch.optim.AdamW(params[k].values(),lr=single.LR) for k in ARMS}
    resume = args.run/'resume.pt'
    step,trace = 0,[]
    if resume.is_file():
        state = torch.load(resume,map_location='cpu',weights_only=True)
        assert state['protocol_sha256']==args.spec_sha
        step,trace = state['step'],state['trace']
        assert step==len(trace)
        for k in ARMS:
            single.rules.restore(params[k],state['heads'][k])
            optimizers[k].load_state_dict(state['optimizers'][k])

    def snapshot():
        return {k:{n:w.detach().cpu() for n,w in params[k].items()} for k in ARMS}

    for step in range(step+1,STEPS+1):
        offset = (step-1)*single.BATCH
        pairs = args.train_order[offset:offset+single.BATCH]
        trace.append(dict(step=step,**{k:update(args,models,heads[k],params[k],optimizers[k],objective,k,pairs,targets)
                                        for k in ARMS}))
        tail.save_torch(resume,dict(protocol_sha256=args.spec_sha,step=step,trace=trace,heads=snapshot(),
                                   optimizers={k:o.state_dict() for k,o in optimizers.items()}))
        if step%16==0:
            print('FIXED50_TRAIN',step,STEPS,flush=True)

    for k in ARMS:
        ab.audit_frozen(args,models,heads[k],params[k])
        heads[k].eval().requires_grad_(False)
    dest = args.run/f'validation/step{STEPS}'
    for k,state in snapshot().items():
        tail.save_torch(dest/(k+'.pt'),state)
    result = previous.score(args,models,heads,metrics,'val',dest)
    assert result['pairs']==256
    verdicts = {k:validation(result,k,contract) for k in ARMS}
    tail.write(dest/'acceptance.json',verdicts)
    tail.write(args.run/'training.json',dict(updates_per_arm=STEPS,trace=trace,validation=verdicts,frozen_unchanged=True))
    for k,v in verdicts.items():
        print('FIXED50_VAL',k,v['metrics']['fid'],v['pants_fid'],v['checks'],flush=True)
    return {k:heads[k] for k in ARMS if verdicts[k]['passed']}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('prior',type=Path)
    cli.add_argument('--check',action='store_true')
    opt = cli.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(4 if opt.check else 6)
    args,contract = setup(opt.prior,'cpu' if opt.check else 'cuda:0')
    if opt.check:
        single.archived64(args,args.pairs['test64'])
        for pair in args.train_order:
            assert tail.sha(args.data/pair['gt'])==args.gt_hashes[pair['case_id']]
        print('FIXED50_PRECHECK',len(args.train_order),len(args.pairs['val']),contract['start_sha256'],flush=True)
        return
    assert torch.cuda.device_count()==1
    if args.spec.is_file():
        assert tail.read(args.spec)==contract
    else:
        tail.write(args.spec,contract)
    args.spec_sha = tail.sha(args.spec)
    if (args.run/'completion.json').is_file() and (args.run/'review.zip').is_file():
        return
    models = init.load(args)
    assert args.original['frozen_model_hashes']==tail.read(args.parent_run/'model.json')['frozen_model_hashes']
    prepared = {k:previous.start_head(args,models) for k in ARMS}
    targets = prepare_targets(args,contract)
    objective = Objective(args)
    qualify(args,models,*prepared['B'],objective,targets)
    metrics = tail.metric_models('cpu',args.protocol['metric_assets'])
    active = train(args,models,prepared,objective,targets,metrics,contract)
    selected = {k:STEPS for k in active}
    stages = {}
    files = ['protocol.json','model.json','execution_environment.json','target_manifest.json',
             'qualification.json','qualification_probe.json','training.json',f'validation/step{STEPS}/result.json',
             f'validation/step{STEPS}/acceptance.json']
    for split,label in (('test64','eval64'),('test500','eval500')):
        if not active:
            break
        single.test_controls(args,models,args.pairs[split],split)
        dest = args.run/label
        result = previous.score(args,models,active,metrics,split,dest)
        decision = single.acceptance(args,dest,result,selected,split)
        tail.write(dest/'acceptance.json',decision)
        accepted = [k for k,d in decision['decisions'].items() if d['accepted']]
        stages[label] = dict(pairs=result['pairs'],scored=list(active),accepted=accepted)
        active = {k:active[k] for k in accepted}
        print('FIXED50_TEST',label,stages[label],flush=True)
        files.extend(f'{label}/{n}' for n in ('result.json','acceptance.json','per_pair.json','pool3.npz'))

    assert tail.tensor_set(models[4])==args.original['frozen_model_hashes']
    tail.write(args.run/'completion.json',dict(updates_per_arm=STEPS,selected=selected,evaluation=stages,
        frozen_unchanged=True,trajectories=1,vae_decodes=1))
    init.fid.pack(args.run,[*files,'completion.json'])
    print('FIXED50_DONE',selected,stages,tail.sha(args.run/'review.zip'),flush=True)


if __name__=='__main__':
    main()
