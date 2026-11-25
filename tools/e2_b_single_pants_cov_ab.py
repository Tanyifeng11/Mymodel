"""从 A+64 做同预算短训，单独检验裤装深衣内的特征协方差监督。"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from tools import e2_b_single_pants_ab as previous
from tools.e2_fft_experiment import SOURCE

single, init, tail, roi, ab = previous.single, previous.init, previous.tail, previous.roi, previous.ab
RUN = 'a64_pants_feature_cov_ab'
ARMS = ('A', 'B')
STEPS = 64
CHECKPOINTS = (32, STEPS)
COV_WEIGHT = .01
CORE_RADIUS, MIN_PIXELS = 20, 256


def spatial_covariance(features, mask):
    weights = F.interpolate(mask.to(features), size=features.shape[-2:], mode='area').flatten(2)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    values = features.flatten(2)
    centered = values - (values * weights).sum(dim=-1, keepdim=True)
    return (centered * weights) @ centered.transpose(1, 2)


def covariance_loss(metric, pred, gt, core):
    mask = torch.from_numpy(core).to(pred.device)[None, None]
    # 固定新项在深衣内的输出梯度，保留原 H2 损失覆盖的区域。
    scoped_pred = torch.where(mask, pred, pred.detach())
    x = spatial_covariance(roi.lpips_features(metric, scoped_pred)[1], mask)
    with torch.no_grad():
        y = spatial_covariance(roi.lpips_features(metric, gt)[1], mask)
    return (x-y).abs().mean() / y.abs().mean().clamp_min(1e-6)


class Objective(single.Objective):
    def __call__(self,pred,gt,mask,arm,category):
        loss,terms = super().__call__(pred,gt,mask,previous.BASE_WEIGHT)
        extra = pred.new_zeros(())
        if arm == 'B' and category == 'pants':
            core = roi.regions(mask,CORE_RADIUS)[0]
            if core.sum() >= MIN_PIXELS:
                extra = covariance_loss(self.base.lpips,pred,gt,core)
        return loss + COV_WEIGHT*extra, dict(terms,pants_covariance=extra)


def setup(prior,device):
    args,_ = previous.setup(prior,device)
    archived = args.run
    review_digest = 'a908a1a339d27db7323c8f349b41d982f6c324d515075b70c087a1f53df64ce2'
    assert tail.sha(archived/'review.zip') == review_digest
    manifest = tail.read(archived/'review_manifest.json')
    parent_spec = single.fusion.read_checked(archived/'protocol.json',manifest['protocol.json'])
    assert parent_spec['cohort_sha256'] == tail.sha(args.out/'protocol.json')
    assert parent_spec['training_order'] == [p['case_id'] for p in args.train_order]
    assert parent_spec['sampler'] == args.protocol['sampler']
    assert parent_spec['common_loss']['band_H2'] == previous.BASE_WEIGHT
    args.train_order = args.train_order[:STEPS * single.BATCH]
    location = archived/'validation/step64'
    result = single.fusion.read_checked(location/'result.json',manifest['validation/step64/result.json'])
    assert result['protocol_sha256'] == manifest['protocol.json']
    rows = tail.read(location/'per_pair.json')
    assert result['pairs'] == len(rows) == len(args.pairs['val']) == 256
    assert [r['case_id'] for r in rows] == [p['case_id'] for p in args.pairs['val']]
    assert tail.sha(location/'pool3.npz') == result['pool3_sha256']
    with np.load(location/'pool3.npz',allow_pickle=False) as pool:
        args.start_features = {n:pool[n] for n in ('A','gt')}
    assert all(a.shape == (256,2048) and np.isfinite(a).all() for a in args.start_features.values())
    args.start_head = location/'A.pt'
    args.start_digest = result['head_sha256']['A']
    args.start_renders = archived/'renders'/args.start_digest
    args.start_rows = {r['case_id']:r for r in rows}
    args.start_metrics = init.metrics_from_group(result['groups']['all'],'A')
    args.start_pants = result['groups']['pants']['texture']['A']
    args.run = archived/RUN
    args.spec = args.run/'protocol.json'
    args.run.mkdir(exist_ok=True,parents=True)
    contract = dict(parent_job=127459,parent_review_sha256=review_digest,
        parent_protocol_sha256=manifest['protocol.json'],cohort_sha256=parent_spec['cohort_sha256'],
        initial_weights_sha256=tail.sha(args.start_head),initial_head_digest=args.start_digest,
        initial='127459 A +64',model_sha256=parent_spec['model_sha256'],sampler=parent_spec['sampler'],
        gpus=1,optimizer='fresh AdamW',updates_per_arm=STEPS,effective_batch=single.BATCH,
        learning_rate=single.LR,training_order=[p['case_id'] for p in args.train_order],
        validation_pairs=256,validation_steps=list(CHECKPOINTS),
        common_loss=parent_spec['common_loss'],covariance=dict(weight=COV_WEIGHT,category='pants',
            core_radius=CORE_RADIUS,minimum_core_pixels=MIN_PIXELS,metric='LPIPS AlexNet relu2, normalized',
            statistic='centered channel covariance with area weights over spatial sites',
            error='mean L1 / max(mean absolute target covariance, 1e-6)',output_gradient_scope='deep core'),
        validation_start=args.start_metrics,validation_pants=args.start_pants,validation_step0=args.zero_metrics,
        tolerance=single.rules.TOLERANCE,formal_reference=parent_spec['formal_reference'],
        code={n:tail.sha(SOURCE/n) for n in ('tools/e2_b_single_pants_cov_ab.py',
            'scripts/e2_b_single_pants_cov_ab.sbatch','tools/e2_b_single_pants_ab.py','tools/e2_b_epoch20_single.py')})
    return args,contract


def qualify(args,models,objective):
    path = args.run/'qualification.json'
    if path.exists():
        assert tail.read(path)['protocol_sha256'] == args.spec_sha
        return
    head,params = previous.start_head(args,models)
    pair = next(p for p in args.train_order if p['category']=='pants'
                and roi.regions(tail.case_mask(args,p),CORE_RADIUS)[0].sum()>=MIN_PIXELS)
    pred,gt,mask = previous.forward(args,models,head,pair)
    a,terms_a = objective(pred,gt,mask,'A','pants')
    b,terms_b = objective(pred,gt,mask,'B','pants')
    for key in terms_a.keys()-{'pants_covariance'}:
        torch.testing.assert_close(terms_a[key], terms_b[key],atol=0,rtol=0)
    extra = terms_b['pants_covariance']
    torch.testing.assert_close(b-a,COV_WEIGHT*extra,atol=5e-7,rtol=1e-4)
    grads = torch.autograd.grad(extra,(pred,*params.values()))
    core = roi.regions(mask,CORE_RADIUS)[0]
    assert grads[0][...,torch.from_numpy(~core).to(pred.device)].count_nonzero().item()==0
    norm = float(torch.linalg.vector_norm(torch.stack([g.float().norm() for g in grads[1:]])))
    assert bool(torch.isfinite(extra)) and float(extra.detach())>0 and np.isfinite(norm) and norm>0
    models[2].adapter.clear()
    ab.audit_frozen(args,models,head,params)
    tail.write(path,dict(protocol_sha256=args.spec_sha,case=pair['case_id'],a64_head_match=True,
        cached_prefix=True,identical_h2_terms=True,b_only_extra=float((b-a).detach()),
        covariance_head_grad_norm=norm,extra_gradient_outside_core=0,optimizer_updates=0))
    print('PANTS_COV_QUALIFIED',pair['case_id'],norm,flush=True)


def update(args,models,objective,arm,head,params,optimizer,pairs):
    records = []
    for pair in pairs:
        pred,gt,mask = previous.forward(args,models,head,pair)
        loss,terms = objective(pred,gt,mask,arm,pair['category'])
        assert torch.isfinite(loss)
        (single.SCALE*loss/single.BATCH).backward()
        records.append(dict(case=pair['case_id'],category=pair['category'],loss=float(loss.detach()),
                           **{k:float(v.detach()) for k,v in terms.items()}))
        models[2].adapter.clear()
        del pred,gt,loss,terms
    for param in params.values():
        param.grad.div_(single.SCALE)
    norm = float(torch.nn.utils.clip_grad_norm_(params.values(),1.,error_if_nonfinite=True))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return dict(grad_norm=norm,rows=records)


def validation_gate(now,start,zero,pants,start_pants):
    outcome = previous.validation_gate(now,start,zero)
    outcome['checks']['pants_band_mae_kept'] = pants['band_mae']<=start_pants['band_mae']
    outcome['passed'] = all(outcome['checks'].values())
    return outcome


@torch.inference_mode()
def pants_probe(args,scored,dest,inception):
    """固定 A+64 的其他区域，只换入候选裤装衣内。"""
    path = dest/'pants_inside_probe.json'
    if path.exists():
        result = tail.read(path)
        assert result['protocol_sha256'] == args.spec_sha
        return result['arms']
    pairs = [(i,p) for i,p in enumerate(args.pairs['val']) if p['category']=='pants']
    reference = roi.fid_terms(args.start_features['A'],args.start_features['gt'])
    result = {}
    for name,digest in scored['head_sha256'].items():
        features = args.start_features['A'].copy()
        for offset in range(0,len(pairs),4):
            batch = pairs[offset:offset+4]
            images = []
            for _,pair in batch:
                case = pair['case_id']
                base_path = args.start_renders/(case+'.png')
                assert tail.sha(base_path)==args.start_rows[case]['pngs']['A']
                base = np.asarray(tail.rgb(base_path))
                now = np.asarray(tail.rgb(args.run/'renders'/digest/(case+'.png')))
                inside = roi.regions(tail.case_mask(args,pair),8)[0].astype('float32')
                w = roi.gaussian(inside,2)[...,None]
                images.append(np.rint(base+w*(now.astype('float32')-base)).astype('uint8'))
            features[[i for i,_ in batch]] = inception(roi.tensors(images))[0].flatten(1).numpy()
        observed = roi.fid_terms(features,args.start_features['gt'])
        result[name] = dict(**observed,delta_fid=observed['fid']-reference['fid'],
            delta_covariance=observed['covariance_term']-reference['covariance_term'])
    tail.write(path,dict(protocol_sha256=args.spec_sha,reference=reference,
        pairs_changed=len(pairs),region='inside erode 8px, Gaussian feather sigma=2',arms=result))
    return result


def train(args,models,objective,metrics):
    heads,params,optimizers = {},{},{}
    for n in ARMS:
        heads[n],params[n] = previous.start_head(args,models)
        optimizers[n] = torch.optim.AdamW(params[n].values(),lr=single.LR)
    path = args.run/'resume.pt'
    step,trace,checks = 0,[],{}
    if path.exists():
        state = torch.load(path,map_location='cpu',weights_only=False)
        assert state['protocol_sha256']==args.spec_sha
        step,trace,checks = (state[k] for k in ('step','trace','checks'))
        for n in ARMS:
            single.rules.restore(params[n],state['heads'][n])
            optimizers[n].load_state_dict(state['optimizers'][n])

    def weights(n):
        return {k:p.detach().cpu() for k,p in params[n].items()}

    def save():
        tail.save_torch(path,dict(protocol_sha256=args.spec_sha,step=step,trace=trace,
            checks=checks,heads={n:weights(n) for n in ARMS},
            optimizers={n:o.state_dict() for n,o in optimizers.items()}))

    def validate():
        dest = args.run/'validation'/f'step{step}'
        for n in ARMS:
            tail.save_torch(dest/(n+'.pt'),weights(n))
        scored = previous.score(args,models,heads,metrics,'val',dest)
        groups = scored['groups']
        probe = pants_probe(args,scored,dest,metrics[1])
        summary = {}
        for n in ARMS:
            value = init.metrics_from_group(groups['all'],n)
            pants = groups['pants']['texture'][n]
            summary[n] = dict(metrics=value,pants=pants,pants_inside_probe=probe[n],
                **validation_gate(value,args.start_metrics,args.zero_metrics,pants,args.start_pants))
        checks[str(step)] = summary
        save()
        print('PANTS_COV_VAL',step,{n:(v['metrics']['fid'],v['checks']) for n,v in summary.items()},flush=True)

    if step in CHECKPOINTS and str(step) not in checks:
        validate()
    for step in range(step+1,STEPS+1):
        pairs = args.train_order[(step-1)*single.BATCH:step*single.BATCH]
        trace.append(dict(step=step,**{n:update(args,models,objective,n,heads[n],params[n],optimizers[n],pairs)
                                      for n in ARMS}))
        save()
        if step%16==0:
            print('PANTS_COV_TRAIN',step,STEPS,flush=True)
        if step in CHECKPOINTS:
            validate()

    selected = {}
    for n in ARMS:
        passing = [(int(s),r[n]) for s,r in checks.items() if r[n]['passed']]
        if passing:
            selected[n],_ = min(passing,key=lambda x:(x[1]['metrics']['fid'],x[1]['metrics']['lpips'],
                                                       x[1]['pants']['band_mae'],x[0]))
            ab.update_head(heads[n],params[n],args.run/'validation'/f'step{selected[n]}'/(n+'.pt'))
        ab.audit_frozen(args,models,heads[n],params[n])
    tail.write(args.run/'training.json',dict(protocol_sha256=args.spec_sha,updates_per_arm=STEPS,
        validation=checks,selected=selected,trace=trace,frozen_unchanged=True))
    return {n:heads[n].eval().requires_grad_(False) for n in selected},selected


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('prior',type=Path)
    cli.add_argument('--check',action='store_true')
    options = cli.parse_args()
    args,contract = setup(options.prior,'cpu' if options.check else 'cuda:0')
    if options.check:
        single.archived64(args,args.pairs['test64'])
        for case,row in args.start_rows.items():
            assert tail.sha(args.start_renders/(case+'.png'))==row['pngs']['A']
        available = sum((args.fusion_run/'fixed'/p['case_id']/'record.json').is_file()
                        for p in args.pairs['test500'])
        print('PANTS_COV_PRECHECK',len(args.train_order),len(args.pairs['val']),
              args.zero_metrics['fid'],contract['initial_weights_sha256'],
              'existing_500_controls',available,flush=True)
        return
    assert torch.cuda.device_count()==1
    cv2.setNumThreads(1)
    torch.set_num_threads(6)
    if args.spec.exists():
        assert tail.read(args.spec)==contract
    else:
        tail.write(args.spec,contract)
    args.spec_sha = tail.sha(args.spec)
    if (args.run/'completion.json').is_file() and (args.run/'review.zip').is_file():
        return
    models = init.load(args)
    assert args.original['frozen_model_hashes']==tail.read(args.parent_run/'model.json')['frozen_model_hashes']
    objective = Objective(args)
    qualify(args,models,objective)
    metrics = tail.metric_models('cpu',args.protocol['metric_assets'])
    active,selected = train(args,models,objective,metrics)
    stages = {}
    files = ['protocol.json','model.json','execution_environment.json','qualification.json','training.json']
    files.extend(f'validation/step{s}/{n}' for s in CHECKPOINTS for n in ('result.json','pants_inside_probe.json'))
    for split,label in (('test64','eval64'),('test500','eval500')):
        if not active:
            break
        dest = args.run/label
        single.test_controls(args,models,args.pairs[split],split)
        result = previous.score(args,models,active,metrics,split,dest)
        decision = single.acceptance(args,dest,result,selected,split)
        tail.write(dest/'acceptance.json',decision)
        accepted = [n for n,v in decision['decisions'].items() if v['accepted']]
        stages[label] = dict(scored=list(active),accepted=accepted)
        active = {n:active[n] for n in accepted}
        files.extend(f'{label}/{n}' for n in ('result.json','acceptance.json','per_pair.json','pool3.npz'))
        print('PANTS_COV_TEST',label,stages[label],flush=True)
    assert tail.tensor_set(models[4])==args.original['frozen_model_hashes']
    tail.write(args.run/'completion.json',dict(updates_per_arm=STEPS,selected=selected,evaluation=stages))
    init.fid.pack(args.run,[*files,'completion.json'])
    print('PANTS_COV_DONE',selected,stages,tail.sha(args.run/'review.zip'),flush=True)


if __name__ == '__main__':
    main()
