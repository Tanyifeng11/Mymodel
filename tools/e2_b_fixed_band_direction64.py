"""固定半量带通方向监督，使用 127791 B 的同预算归档对照。"""
import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from tools import e2_b_127791_region_diagnostic as diagnostic
from tools import e2_b_fixed_band50_ab as fixed
from tools.e2_fft_experiment import SOURCE

single, previous, init, tail, cov = fixed.single, fixed.previous, fixed.init, fixed.tail, fixed.cov
roi, ab, CATEGORIES = fixed.roi, fixed.ab, fixed.CATEGORIES
ARM = 'direction'
RUN = 'fixed_band50_direction_64'
STEPS = fixed.STEPS
POSITION = DIRECTION = .05


def setup(prior, device):
    args, inherited = fixed.setup(prior,device)
    control = args.run
    assert tail.sha(control/'review.zip') == diagnostic.RUN_SHA
    manifest = tail.read(control/'review_manifest.json')
    checked = single.fusion.read_checked
    spec = checked(control/'protocol.json',manifest['protocol.json'])
    assert {k:v for k,v in spec.items() if k!='code'} == {
        k:v for k,v in inherited.items() if k!='code'
    }
    original_targets = args.target_run/'target_manifest.json'
    control_targets = checked(control/'target_manifest.json',manifest['target_manifest.json'])
    assert control_targets['source_manifest_sha256'] == tail.sha(original_targets)
    assert control_targets['rows'] == tail.read(original_targets)['rows']

    stage = 'validation/step64'
    score = checked(control/stage/'result.json',manifest[stage+'/result.json'])
    assert score['pairs']==len(args.pairs['val'])==256
    assert score['protocol_sha256']==manifest['protocol.json']
    assert tail.sha(control/stage/'pool3.npz')==score['pool3_sha256']
    ids = [(p['case_id'],p['category']) for p in args.pairs['val']]
    assert [(r['case_id'],r['category']) for r in tail.read(control/stage/'per_pair.json')]==ids
    verdict = checked(control/stage/'acceptance.json',manifest[stage+'/acceptance.json'])
    assert fixed.validation(score,'B',inherited)==verdict['B']
    training = checked(control/'training.json',manifest['training.json'])
    assert training['frozen_unchanged'] and training['updates_per_arm']==len(training['trace'])==STEPS
    assert [r['case'] for x in training['trace'] for r in x['B']['rows']]==inherited['training_order']
    with np.load(control/stage/'pool3.npz',allow_pickle=False) as a, np.load(
            args.start_head.parent/'pool3.npz',allow_pickle=False) as b:
        np.testing.assert_array_equal(a['gt'],b['gt'])

    assert POSITION+DIRECTION==previous.BASE_WEIGHT
    args.run = control/RUN
    args.spec = args.run/'protocol.json'
    contract = dict(inherited, optimizer='fresh AdamW', baseline_127791_B=verdict['B'],
        baseline_127791_review_sha256=diagnostic.RUN_SHA,
        fixed_target_manifest_sha256=tail.sha(original_targets), train_arms=[ARM],
        change=dict(position_weight=POSITION,direction_weight=DIRECTION,
            goal='fixed target band minus fixed 960 anchor band',
            response='differentiable predicted band minus fixed 960 anchor band',
            direction='mean(abs(G))*abs(1-dot(V,G)/dot(G,G)), zero when G is zero',
            core_radius=cov.CORE_RADIUS,minimum_core_pixels=cov.MIN_PIXELS,band_sigma=[1,4],
            categories=list(CATEGORIES),gradient='only r20 in RGB'),
        code={**inherited['code'],**{name:tail.sha(SOURCE/name) for name in (
            'tools/e2_b_fixed_band_direction64.py','scripts/e2_b_fixed_band_direction64.sbatch',
            'tools/e2_b_127791_region_diagnostic.py')}})
    return args, inherited, contract, control


@dataclass
class BandTarget:
    anchor: torch.Tensor
    target: torch.Tensor
    goal_mae: torch.Tensor
    basis: torch.Tensor


def directional_loss(response, target):
    # basis 已包含 mean(abs(G))/dot(G,G)，避免在反传中除以小量。
    return (target.goal_mae-(response*target.basis).sum()).abs()


class Objective(fixed.Objective):
    @torch.no_grad()
    def cache(self,args,targets):
        rows = tail.read(args.run/'target_manifest.json')['rows']
        frozen, counts = {}, Counter()
        for pair in args.train_order:
            case = pair['case_id']
            if case not in targets:
                continue
            row = rows[case]
            path = args.anchor_origin/(case+'.png')
            if not path.is_file():
                path = args.target_run/'renders'/args.start_digest/(case+'.png')
            assert tail.sha(path)==row['anchor_sha256']
            keep = torch.from_numpy(roi.regions(tail.case_mask(args,pair),cov.CORE_RADIUS)[0]).to(args.device)
            anchor = roi.tensors([np.asarray(tail.rgb(path))]).to(args.device)
            ref_band = self.base.band(targets[case].to(args.device))[0,0][keep]
            anchor_band = self.base.band(anchor)[0,0][keep]
            goal = ref_band-anchor_band
            energy = goal.square().sum()
            mean = goal.abs().mean()
            basis = goal*(mean/energy) if float(energy)>0 else torch.zeros_like(goal)
            target = BandTarget(anchor_band,ref_band,mean,basis)
            assert torch.isfinite(basis).all()
            torch.testing.assert_close(directional_loss(goal,target),mean.new_zeros(()),atol=5e-8,rtol=0)
            frozen[case] = target
            counts[pair['category']] += int(float(energy)>0)
        assert len(frozen)==115 and all(counts[k] for k in CATEGORIES)
        tail.write(args.run/'band_cache.json',dict(protocol_sha256=args.spec_sha,
            reused_anchors=len(frozen),reused_targets=len(frozen),nonzero_goals=dict(counts)))
        print('DIRECTION_TARGETS_REUSED',len(frozen),dict(counts),flush=True)
        return frozen

    def components(self,pred,gt,mask,category,target):
        base,terms = single.Objective.__call__(self,pred,gt,mask,0.)
        core = roi.regions(mask,cov.CORE_RADIUS)[0]
        assert (target is not None)==(core.sum()>=cov.MIN_PIXELS)
        pos = direction = covariance = pred.new_zeros(())
        if target is not None:
            keep = torch.from_numpy(core).to(pred.device)
            scoped = torch.where(keep[None,None],pred,pred.detach())
            band = self.base.band(scoped)[0,0][keep]
            pos = (band-target.target).abs().mean()
            direction = directional_loss(band-target.anchor,target)
            if category=='pants':
                covariance = cov.covariance_loss(self.base.lpips,pred,gt,core)
        return base,dict(terms,band=pos,direction=direction,pants_covariance=covariance)

    def __call__(self,pred,gt,mask,arm,category,target):
        assert arm==ARM
        base,t = self.components(pred,gt,mask,category,target)
        loss = base+POSITION*t['band']+DIRECTION*t['direction']+cov.COV_WEIGHT*t['pants_covariance']
        return loss,t


def qualify(args,models,head,params,objective,targets,images):
    path = args.run/'qualification.json'
    if path.is_file():
        q = tail.read(path)
        assert q['protocol_sha256']==args.spec_sha and q['optimizer_updates']==0
        return

    records = {}
    scale,large = single.SCALE/single.BATCH,4096.
    for category in CATEGORIES:
        pair = next(p for p in args.train_order if p['category']==category and
                    p['case_id'] in targets and float(targets[p['case_id']].goal_mae)>0)
        case = pair['case_id']
        pred,gt,mask = previous.forward(args,models,head,pair)
        core = roi.regions(mask,cov.CORE_RADIUS)[0]
        keep = torch.from_numpy(core).to(pred.device)
        base,t = objective.components(pred,gt,mask,category,targets[case])
        pos, direction = POSITION*t['band'],DIRECTION*t['direction']
        covariance = cov.COV_WEIGHT*t['pants_covariance']
        old_pos = fixed.Objective.fixed_band(objective,pred,images[case],core)
        torch.testing.assert_close(t['band'],old_pos,atol=1e-8,rtol=1e-5)
        control = base+previous.BASE_WEIGHT*old_pos+covariance
        total = base+pos+direction+covariance
        torch.testing.assert_close(total-control,direction-pos,atol=5e-7,rtol=1e-4)

        weight = head.output.weight
        rg,g64 = torch.autograd.grad(scale*direction,(pred,weight),retain_graph=True)
        pg, = torch.autograd.grad(scale*pos,pred,retain_graph=True)
        g4096, = torch.autograd.grad(large*direction,weight,retain_graph=True)
        full, = torch.autograd.grad(scale*total,weight,retain_graph=True)
        without, = torch.autograd.grad(scale*(base+pos+covariance),weight)
        norm = lambda x,s:float(torch.linalg.vector_norm(x.float()/s))
        record = dict(case=case,position_l1=float(t['band'].detach()),
            direction=float(t['direction'].detach()),goal_mae=float(targets[case].goal_mae),
            position_rgb_inside=int(torch.count_nonzero(pg[...,keep])),
            position_rgb_outside=int(torch.count_nonzero(pg[...,~keep])),
            direction_rgb_inside=int(torch.count_nonzero(rg[...,keep])),
            direction_rgb_outside=int(torch.count_nonzero(rg[...,~keep])),
            isolated_scale64=norm(g64,scale),isolated_scale4096=norm(g4096,large),
            production_total=norm(full,scale),production_increment=norm(full.float()-without.float(),scale))
        records[category] = record
        tail.write(args.run/'qualification_probe.json',dict(protocol_sha256=args.spec_sha,probes=records))
        print('DIRECTION_PROBE',category,record,flush=True)
        assert record['direction_rgb_outside']==record['position_rgb_outside']==0
        assert record['direction_rgb_inside']>0 and record['position_rgb_inside']>0
        assert torch.isfinite(rg).all() and torch.isfinite(pg).all()
        assert all(np.isfinite(record[k]) and record[k]>0 for k in (
            'isolated_scale4096','production_total','production_increment'))
        assert np.isfinite(record['isolated_scale64'])
        models[2].adapter.clear()
    ab.audit_frozen(args,models,head,params)
    tail.write(path,dict(protocol_sha256=args.spec_sha,optimizer_updates=0,probes=records,
        production_scale=scale,isolated_probe_scale=large,position_baseline_matched=True))
    print('DIRECTION_QUALIFIED',len(records),flush=True)


def train(args,models,head,params,objective,targets):
    optimizer = torch.optim.AdamW(params.values(),lr=single.LR)
    resume = args.run/'resume.pt'
    start,trace = 0,[]
    if resume.is_file():
        saved = torch.load(resume,map_location='cpu',weights_only=True)
        assert saved['protocol_sha256']==args.spec_sha
        start,trace = saved['step'],saved['trace']
        assert start==len(trace)
        single.rules.restore(params,saved['head'])
        optimizer.load_state_dict(saved['optimizer'])
    for step in range(start+1,STEPS+1):
        batch = args.train_order[(step-1)*single.BATCH:step*single.BATCH]
        record = fixed.update(args,models,head,params,optimizer,objective,ARM,batch,targets)
        trace.append(dict(step=step,**record))
        tail.save_torch(resume,dict(protocol_sha256=args.spec_sha,step=step,trace=trace,
            head={k:w.detach().cpu() for k,w in params.items()},optimizer=optimizer.state_dict()))
        if step%16==0:
            print('DIRECTION_TRAIN',step,STEPS,flush=True)
    assert [row['case'] for step in trace for row in step['rows']]==[p['case_id'] for p in args.train_order]
    ab.audit_frozen(args,models,head,params)
    head.eval().requires_grad_(False)
    return trace


@torch.inference_mode()
def response_audit(prior,args,stage,head,score,inception):
    out = args.run/'response_audit.json'
    if out.is_file():
        saved = tail.read(out)
        assert saved['protocol_sha256']==args.spec_sha
        assert tail.sha(args.run/'response_pool3.npz')==saved['region_pool3_sha256']
        return
    old_args, old_dest, pairs, hashes, albums, features, _, _, _, _, _, _ = diagnostic.load(prior)
    graph = diagnostic.regions
    with np.load(stage/'pool3.npz',allow_pickle=False) as archive:
        np.testing.assert_allclose(archive['gt'],features['gt'],atol=7e-5,rtol=1e-4)
        features[ARM] = archive[ARM]
    labels = np.array([p['category'] for p in pairs])
    old = tail.read(old_dest/'result.json')
    digest = init.head_hash(head)
    assert score['head_sha256'][ARM]==digest
    arms = {k:albums[k] for k in ('off','step960','B')}
    arms[ARM] = (args.run/'renders'/digest,ARM,tail.read(stage/'per_pair.json'))
    pools,response = {},{}
    for category in CATEGORIES:
        ids = np.flatnonzero(labels==category)
        end = (1<<len(graph.region_names(category)))-1
        pool = {0:features['off'][ids],end:features[ARM][ids]}
        pool.update({b:np.empty_like(pool[end]) for b in range(1,end)})
        cases = []
        for offset in range(0,len(ids),4):
            batch = []
            for i in ids[offset:offset+4]:
                picture,gt,hard,coarse,row = diagnostic.prior.previous.inputs(old_args,pairs[i],hashes,arms,i)
                mask = tail.case_mask(old_args,pairs[i])
                _,soft = graph.split_regions(mask,hard,coarse,category)
                core = roi.regions(mask,20)[0]
                if row['interior_px']>=256:
                    goal = diagnostic.target_band50(picture['step960'],gt,core)[0]
                    measured = diagnostic.response(dict(step960=picture['step960'],A=picture['B'],B=picture[ARM]),goal,core)
                    if measured:
                        cases.append(measured)
                batch.append((picture,soft,coarse))
            for b in range(1,end):
                blend = [graph.mix((x['off'],x[ARM]),soft,coarse,b,category) for x,soft,coarse in batch]
                pool[b][offset:offset+len(batch)] = inception(roi.tensors(blend))[0].flatten(1).numpy()
        pools[category] = pool
        n = len(cases)
        total = sum(x['A']['goal_sq'] for x in cases)
        old_projection = sum(x['A']['dot'] for x in cases)/total
        archived = old['texture'][category]['directions']['B']
        assert n==archived['n'] and abs(old_projection-archived['pooled_projection'])<1e-9
        response[category] = dict(n=n,control=old_projection,
            candidate=sum(x['B']['dot'] for x in cases)/total,
            delta=sum(x['B_A']['dot'] for x in cases)/total,
            median_increment_rms_ratio=float(np.median([x['B_A']['energy_ratio'] for x in cases])))
        print('DIRECTION_AUDIT_CATEGORY',category,n,flush=True)
    attributed = graph.contribution(pools,dict(step0=features['off'],candidate=features[ARM]),labels,features['gt'])
    # 区域归因沿用 127792 的关 B 端点与全套排序。
    a = attributed['category_region']['pants']['deep']
    b = old['attribution']['off_B']['category_region']['pants']['deep']
    path = args.run/'response_pool3.npz'
    np.savez_compressed(path,**{f'{c}_{bits}':v for c,ps in pools.items()
                               for bits,v in ps.items() if bits not in (0,len(ps)-1)})
    tail.write(out,dict(protocol_sha256=args.spec_sha,categories=response,attribution=attributed,
        pants_deep=dict(control=b,candidate=a,delta={k:a[k]-b[k] for k in a}),
        new_region_features=2048,region_pool3_sha256=tail.sha(path)))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('prior',type=Path)
    cli.add_argument('--check',action='store_true')
    opts = cli.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(4 if opts.check else 6)
    args,inherited,contract,control = setup(opts.prior,'cpu' if opts.check else 'cuda:0')
    if opts.check:
        single.archived64(args,args.pairs['test64'])
        assert tail.sha(control/'fid_texture_regions_127791/review.zip')==(
            '3532c0ad47630a8de91483877b6de9d2866c79c1b7ff66e943e06b68ab18213d')
        assert all((args.parent_run/'prefix_b_off'/(p['case_id']+'.pt')).is_file()
                   for p in (*args.train_order,*args.pairs['val']))
        print('DIRECTION_PRECHECK',len(args.train_order),len(args.pairs['val']),contract['start_sha256'],flush=True)
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
    head,params = previous.start_head(args,models)
    images = fixed.prepare_targets(args,inherited)
    objective = Objective(args)
    targets = objective.cache(args,images)
    qualify(args,models,head,params,objective,targets,images)
    del images
    trace = train(args,models,head,params,objective,targets)
    del objective,targets
    stage = args.run/'validation/step64'
    tail.save_torch(stage/(ARM+'.pt'),{n:w.detach().cpu() for n,w in params.items()})
    metrics = tail.metric_models('cpu',args.protocol['metric_assets'])
    score = previous.score(args,models,{ARM:head},metrics,'val',stage)
    assert score['pairs']==256
    decision = fixed.validation(score,ARM,contract)
    tail.write(stage/'acceptance.json',{ARM:decision})
    tail.write(args.run/'training.json',dict(updates=STEPS,trace=trace,validation=decision,frozen_unchanged=True))
    print('DIRECTION_VAL',decision['metrics']['fid'],decision['pants_fid'],decision['checks'],flush=True)
    selected = {ARM:STEPS} if decision['passed'] else {}
    stages, active = {},bool(selected)
    files = ['protocol.json','model.json','execution_environment.json','target_manifest.json','band_cache.json',
             'qualification.json','qualification_probe.json','training.json','validation/step64/result.json',
             'validation/step64/acceptance.json']
    for split,label in (('test64','eval64'),('test500','eval500')):
        if not active:
            break
        single.test_controls(args,models,args.pairs[split],split)
        dest = args.run/label
        result = previous.score(args,models,{ARM:head},metrics,split,dest)
        formal = single.acceptance(args,dest,result,selected,split)
        tail.write(dest/'acceptance.json',formal)
        active = formal['decisions'][ARM]['accepted']
        stages[label] = dict(pairs=result['pairs'],scored=[ARM],accepted=[ARM] if active else [])
        print('DIRECTION_TEST',label,stages[label],flush=True)
        files.extend(f'{label}/{n}' for n in ('result.json','acceptance.json','per_pair.json','pool3.npz'))
    response_audit(opts.prior,args,stage,head,score,metrics[1])
    assert tail.tensor_set(models[4])==args.original['frozen_model_hashes']
    tail.write(args.run/'completion.json',dict(updates=STEPS,selected=selected,evaluation=stages,
        frozen_unchanged=True,trajectories=1,vae_decodes=1,control_retrained=False))
    init.fid.pack(args.run,[*files,'response_audit.json','completion.json'])
    print('DIRECTION_DONE',selected,stages,tail.sha(args.run/'review.zip'),flush=True)


if __name__=='__main__':
    main()
