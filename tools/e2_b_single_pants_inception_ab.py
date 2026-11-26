"""从 127556 B64 续训，等预算比较原损失与冻结 Inception 配对监督。"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from tools import e2_b_single_pants_cov_ab as cov
from tools import e2_b_single_pants_lstar_equal50 as offline
from tools.e2_fft_experiment import SOURCE

single, previous, init, tail, roi, ab = cov.single, cov.previous, cov.init, cov.tail, cov.roi, cov.ab
RUN = 'b64_pants_inception_band50_ab'
ARMS, STEPS, CHECKPOINTS = cov.ARMS, cov.STEPS, cov.CHECKPOINTS
WEIGHT = cov.COV_WEIGHT
PARENT_SHA = '2b0fddcac11cf66d9cd436b21850ebf04b9e9ac72513a40b0017c89763867f05'
TARGET_SHA = 'd3ff3f84315e359a7efd841f8ede2a1cabd8197c90459c4f5d90d3a6a543d3cb'


def target_band50(anchor,gt,core):
    weight = np.minimum(cv2.distanceTransform(core.astype('uint8'),cv2.DIST_L2,5)/offline.FEATHER_WIDTH,1)
    source = offline.lab_pixels(anchor)
    difference = offline.lab_pixels(gt)[...,0]-source[...,0]
    return offline.apply_lstar(anchor,source,offline.FRACTION*weight*roi.band(difference),core)


def paired_inputs(pred,gt,core):
    anchor = single.ft.rgb8(pred.detach())
    reference,_ = target_band50(anchor,single.ft.rgb8(gt.detach()),core)
    anchor_tensor = roi.tensors([anchor]).to(pred)
    ref_tensor = roi.tensors([reference]).to(pred)
    mask = torch.from_numpy(core).to(pred.device)[None,None]
    # Inception 前向使用与离线诊断一致的 8 位 RGB，反传只开放深衣内。
    x = anchor_tensor + torch.where(mask,pred-pred.detach(),0.)
    return x,ref_tensor


def pool3(extractor,image):
    # 与 CPU 评分使用同一插值，避开 CUDA 插值反传的确定性限制。
    resized = F.interpolate(image.cpu(),size=(299,299),mode='bilinear',align_corners=False)
    return extractor(resized.to(image.device))[0].flatten(1)


def inception_loss(extractor,pred,gt,core):
    x,y = paired_inputs(pred,gt,core)
    with torch.no_grad():
        target = pool3(extractor,y)
    observed = pool3(extractor,x)
    return (observed-target).abs().mean()/target.abs().mean().clamp_min(1e-6)


def load_inception(device):
    from pytorch_fid.inception import InceptionV3
    block = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    return InceptionV3([block],resize_input=False).to(device).eval().requires_grad_(False)


def check_inception():
    model = load_inception('cpu')
    rgb = torch.linspace(.1,.9,3*64*64).reshape(1,3,64,64).requires_grad_(True)
    explicit = pool3(model,rgb)
    model.resize_input = True
    with torch.no_grad():
        native = model(rgb)[0].flatten(1)
    torch.testing.assert_close(explicit,native,atol=0,rtol=0)
    grad, = torch.autograd.grad(explicit.square().mean(),rgb)
    assert explicit.shape==(1,2048) and torch.isfinite(grad).all() and grad.count_nonzero()>0


class Objective(cov.Objective):
    def __init__(self,args):
        super().__init__(args)
        self.inception = load_inception(args.device)

    def __call__(self,pred,gt,mask,arm,category):
        # 本轮 A 继承 127556 的 B；本轮 B 继承其共同 H2 部分。
        total,terms = super().__call__(pred,gt,mask,'B' if arm=='A' else 'A',category)
        feature = pred.new_zeros(())
        if arm=='B' and category=='pants':
            core = roi.regions(mask,cov.CORE_RADIUS)[0]
            if core.sum()>=cov.MIN_PIXELS:
                feature = inception_loss(self.inception,pred,gt,core)
        return total+WEIGHT*feature,dict(terms,pants_inception=feature)


def setup(prior,device):
    args,_ = cov.setup(prior,device)
    archived = args.run
    assert tail.sha(archived/'review.zip')==PARENT_SHA
    manifest = tail.read(archived/'review_manifest.json')
    parent = single.fusion.read_checked(archived/'protocol.json',manifest['protocol.json'])
    assert parent['training_order']==[p['case_id'] for p in args.train_order]
    assert parent['covariance']['weight']==WEIGHT and parent['covariance']['category']=='pants'
    assert parent['validation_start']==args.start_metrics and parent['validation_pants']==args.start_pants
    assert parent['validation_step0']==args.zero_metrics and parent['tolerance']==single.rules.TOLERANCE
    target_run = archived/offline.RUN
    assert tail.sha(target_run/'review.zip')==TARGET_SHA
    target_manifest = tail.read(target_run/'review_manifest.json')
    target_spec = single.fusion.read_checked(target_run/'protocol.json',target_manifest['protocol.json'])
    target_result = single.fusion.read_checked(target_run/'result.json',target_manifest['result.json'])
    assert target_spec['source_job']==127556 and target_spec['fraction']==offline.FRACTION
    assert target_spec['core_radius']==cov.CORE_RADIUS and target_spec['feather_px']==offline.FEATHER_WIDTH
    assert target_result['pairs']==256 and target_result['acceptance']['band50']['passed']
    assert tail.metric_assets()==args.protocol['metric_assets']
    loc = archived/'validation/step64'
    score = single.fusion.read_checked(loc/'result.json',manifest['validation/step64/result.json'])
    assert score['protocol_sha256']==manifest['protocol.json'] and score['pairs']==256
    rows = tail.read(loc/'per_pair.json')
    assert [p['case_id'] for p in args.pairs['val']]==[r['case_id'] for r in rows]
    assert tail.sha(loc/'pool3.npz')==score['pool3_sha256']
    args.start_head = loc/'B.pt'
    args.start_digest = score['head_sha256']['B']
    args.initial_renders = archived/'renders'/args.start_digest
    args.initial_rows = rows
    args.run = archived/RUN
    args.run.mkdir(exist_ok=True,parents=True)
    args.spec = args.run/'protocol.json'
    initial_metrics = init.metrics_from_group(score['groups']['all'],'B')
    thresholds = dict(fid=args.zero_metrics['fid'],lpips=args.start_metrics['lpips'],
        ssim=args.start_metrics['ssim']*(1-single.rules.TOLERANCE),
        contour_f1=args.start_metrics['contour_f1']*(1-single.rules.TOLERANCE),
        pants_band_mae=args.start_pants['band_mae'])
    assert target_result['thresholds']==thresholds
    contract = dict(parent_job=127556,parent_review_sha256=PARENT_SHA,target_job=127596,
        target_review_sha256=TARGET_SHA,initial='127556 B step64',initial_head_digest=args.start_digest,
        initial_weights_sha256=tail.sha(args.start_head),initial_metrics=initial_metrics,
        initial_pants=score['groups']['pants']['texture']['B'],
        cohort_sha256=parent['cohort_sha256'],sampler=parent['sampler'],model_sha256=parent['model_sha256'],
        training_order=parent['training_order'],validation_pairs=256,gpus=1,updates_per_arm=STEPS,
        effective_batch=single.BATCH,learning_rate=single.LR,optimizer='fresh AdamW',
        validation_steps=list(CHECKPOINTS),thresholds=thresholds,formal_reference=parent['formal_reference'],
        common_loss=parent['common_loss'],A_extra=parent['covariance'],
        B_extra=dict(weight=WEIGHT,category='pants',feature='frozen FID Inception pool3 (2048)',
            inception_sha256=args.protocol['metric_assets']['inception']['sha256'],
            target='detached current training RGB + half paired L* DoG residual in deep core',
            bands_sigma=[1,4],fraction=offline.FRACTION,core_radius=cov.CORE_RADIUS,
            min_pixels=cov.MIN_PIXELS,feather_px=offline.FEATHER_WIDTH,
            normalization='feature L1 / max(mean absolute target feature, 1e-6)',
            resize='CPU bilinear 299 align_corners=False, same as score',
            direct_gradient='straight-through RGB8 inside core only'),
        code={n:tail.sha(SOURCE/n) for n in ('tools/e2_b_single_pants_inception_ab.py',
            'scripts/e2_b_single_pants_inception_ab.sbatch','tools/e2_b_single_pants_cov_ab.py',
            'tools/e2_b_single_pants_lstar_equal50.py','tools/e2_b_epoch20_single.py')})
    return args,contract


def qualify(args,models,objective):
    path = args.run/'qualification.json'
    if path.exists():
        assert tail.read(path)['protocol_sha256']==args.spec_sha
        return
    head,params = previous.start_head(args,models)
    probe = next(p for p in args.train_order if p['category']=='pants'
        and roi.regions(tail.case_mask(args,p),cov.CORE_RADIUS)[0].sum()>=cov.MIN_PIXELS)
    pred,gt,mask = previous.forward(args,models,head,probe)
    a,ta = objective(pred,gt,mask,'A','pants')
    b,tb = objective(pred,gt,mask,'B','pants')
    for k in ta.keys()-{'pants_covariance','pants_inception'}:
        torch.testing.assert_close(ta[k],tb[k],atol=0,rtol=0)
    assert ta['pants_covariance']>0 and ta['pants_inception']==0
    assert tb['pants_inception']>0 and tb['pants_covariance']==0
    torch.testing.assert_close(b-a,WEIGHT*(tb['pants_inception']-ta['pants_covariance']),atol=5e-7,rtol=1e-4)
    core = roi.regions(mask,cov.CORE_RADIUS)[0]
    outside = torch.from_numpy(~core).to(pred.device)
    x,y = paired_inputs(pred,gt,core)
    assert torch.equal(x[...,outside],y[...,outside])
    grads = torch.autograd.grad(tb['pants_inception'],(pred,*params.values()))
    assert grads[0][...,outside].count_nonzero().item()==0
    norm = float(torch.linalg.vector_norm(torch.stack([g.float().norm() for g in grads[1:]])))
    assert np.isfinite(norm) and norm>0
    assert all(not p.requires_grad and p.grad is None for p in objective.inception.parameters())
    models[2].adapter.clear()
    ab.audit_frozen(args,models,head,params)
    tail.write(path,dict(protocol_sha256=args.spec_sha,case=probe['case_id'],b64_head_match=True,
        cached_prefix=True,common_h2_equal=True,loss_replacement_checked=True,same_context_outside_core=True,
        b_old_covariance=0,a_covariance=float(ta['pants_covariance'].detach()),
        b_inception=float(tb['pants_inception'].detach()),feature_gradient_outside_core=0,
        inception_head_grad_norm=norm,optimizer_updates=0))
    print('PANTS_INCEPTION_QUALIFIED',probe['case_id'],norm,flush=True)


def train(args,models,objective,metrics):
    heads,params,opts = {},{},{}
    for n in ARMS:
        heads[n],params[n] = previous.start_head(args,models)
        opts[n] = torch.optim.AdamW(params[n].values(),lr=single.LR)
    resume = args.run/'resume.pt'
    step,trace,checks = 0,[],{}
    if resume.exists():
        state = torch.load(resume,map_location='cpu',weights_only=False)
        assert state['protocol_sha256']==args.spec_sha
        step,trace,checks = (state[k] for k in ('step','trace','checks'))
        for n in ARMS:
            single.rules.restore(params[n],state['heads'][n])
            opts[n].load_state_dict(state['optimizers'][n])

    def weights(n):
        return {k:p.detach().cpu() for k,p in params[n].items()}

    def save():
        tail.save_torch(resume,dict(protocol_sha256=args.spec_sha,step=step,trace=trace,checks=checks,
            heads={n:weights(n) for n in ARMS},optimizers={n:o.state_dict() for n,o in opts.items()}))

    def validate():
        dest = args.run/'validation'/f'step{step}'
        for n in ARMS:
            tail.save_torch(dest/(n+'.pt'),weights(n))
        groups = previous.score(args,models,heads,metrics,'val',dest)['groups']
        decision = {}
        for n in ARMS:
            value = init.metrics_from_group(groups['all'],n)
            pants = groups['pants']['texture'][n]
            decision[n] = dict(metrics=value,pants=pants,
                **cov.validation_gate(value,args.start_metrics,args.zero_metrics,pants,args.start_pants))
        checks[str(step)] = decision
        save()
        print('PANTS_INCEPTION_VAL',step,{n:(v['metrics']['fid'],v['checks']) for n,v in decision.items()},flush=True)

    if step in CHECKPOINTS and str(step) not in checks:
        validate()
    for step in range(step+1,STEPS+1):
        pairs = args.train_order[(step-1)*single.BATCH:step*single.BATCH]
        trace.append(dict(step=step,**{n:cov.update(args,models,objective,n,heads[n],params[n],opts[n],pairs)
                                      for n in ARMS}))
        save()
        if step%16==0:
            print('PANTS_INCEPTION_TRAIN',step,STEPS,flush=True)
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
    cv2.setNumThreads(1)
    torch.set_num_threads(6 if not options.check else 4)
    args,contract = setup(options.prior,'cpu' if options.check else 'cuda:0')
    if options.check:
        check_inception()
        print('PANTS_INCEPTION_CPU_PARITY',2048,flush=True)
        single.archived64(args,args.pairs['test64'])
        for row in args.initial_rows:
            assert tail.sha(args.initial_renders/(row['case_id']+'.png'))==row['pngs']['B']
        print('PANTS_INCEPTION_PRECHECK',len(args.train_order),len(args.pairs['val']),
              contract['initial_weights_sha256'],flush=True)
        return
    assert torch.cuda.device_count()==1
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
    feature_state = tail.tensor_set({'inception':objective.inception})
    qualify(args,models,objective)
    metrics = tail.metric_models('cpu',args.protocol['metric_assets'])
    active,selected = train(args,models,objective,metrics)
    stages = {}
    files = ['protocol.json','model.json','execution_environment.json','qualification.json','training.json']
    files.extend(f'validation/step{s}/result.json' for s in CHECKPOINTS)
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
        print('PANTS_INCEPTION_TEST',label,stages[label],flush=True)
    assert tail.tensor_set(models[4])==args.original['frozen_model_hashes']
    assert tail.tensor_set({'inception':objective.inception})==feature_state
    tail.write(args.run/'completion.json',dict(updates_per_arm=STEPS,selected=selected,evaluation=stages,
        frozen_inception_sha256=feature_state['inception']))
    init.fid.pack(args.run,[*files,'completion.json'])
    print('PANTS_INCEPTION_DONE',selected,stages,tail.sha(args.run/'review.zip'),flush=True)


if __name__ == '__main__':
    main()
