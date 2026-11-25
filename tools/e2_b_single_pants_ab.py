"""从 H2 第768步短训，B组仅增强裤装深衣内的真实配对带通监督。"""
import argparse
from collections import Counter
import copy
from pathlib import Path

import cv2
import numpy as np
import torch

from tools import e2_b_epoch20_single as single
from tools import e2_b_single_pants_roi as diagnosis
from tools.e2_fft_experiment import SOURCE

fusion, init, tail, roi, ab = single.fusion, single.init, single.tail, single.roi, single.ab
RUN = 'h2_step768_pants_band_ab'
ARMS = ('A', 'B')
STEPS = 128
CHECKPOINTS = (32, 64, STEPS)
BASE_WEIGHT, PANTS_WEIGHT = .10, .25


def batch_order(pairs):
    rng = np.random.default_rng(42)
    classes = [rng.permutation([i for i,p in enumerate(pairs) if p['category']==c])
               for c in tail.distill.CATEGORIES]
    order = np.column_stack(classes).ravel()[:STEPS*single.BATCH]
    assert len(set(order.tolist())) == STEPS*single.BATCH
    selected = [pairs[i] for i in order]
    assert Counter(p['category'] for p in selected) == dict.fromkeys(tail.distill.CATEGORIES,64)
    return selected


def band_weight(arm,pair):
    return PANTS_WEIGHT if arm=='B' and pair['category']=='pants' else BASE_WEIGHT


def setup(prior,device):
    args,_ = single.setup(prior,device)
    args.parent_run = args.run
    args.prefix_source = copy.copy(args)
    archive = tail.read(args.parent_run/'review_manifest.json')
    protocol = fusion.read_checked(args.spec,archive['protocol.json'])
    assert protocol['cohort_sha256'] == tail.sha(args.out/'protocol.json')
    assert protocol['gpus']==1 and protocol['sampler']==args.protocol['sampler']
    assert protocol['loss']['band_H2']==BASE_WEIGHT and protocol['parameters']==tail.PARAMETERS
    args.prefix_source.spec_sha = archive['protocol.json']
    zero,zrows,_,gt = diagnosis.load_validation(args.parent_run,archive,0)
    start,rows,_,start_gt = diagnosis.load_validation(args.parent_run,archive,768)
    np.testing.assert_array_equal(gt,start_gt)
    assert [r['case_id'] for r in zrows] == [r['case_id'] for r in rows] == [p['case_id'] for p in args.pairs['val']]
    args.start_head = args.parent_run/'validation/step768/H2.pt'
    args.start_digest = start['head_sha256']['H2']
    args.start_rows = {r['case_id']:r for r in rows}
    args.train_order = batch_order(args.pairs['train'])
    for pair in [*args.pairs['train'],*args.pairs['val']]:
        assert (args.parent_run/'prefix_b_off'/(pair['case_id']+'.pt')).is_file(), pair['case_id']
    args.start_metrics = init.metrics_from_group(start['groups']['all'],'H2')
    args.zero_metrics = init.metrics_from_group(zero['groups']['all'],'H2')
    args.run = args.parent_run/RUN
    args.spec = args.run/'protocol.json'
    contract = dict(parent_protocol_sha256=archive['protocol.json'],cohort_sha256=protocol['cohort_sha256'],
        model_sha256=protocol['base_sha256'],sampler=protocol['sampler'],initial_weights_sha256=tail.sha(args.start_head),
        initial_h2_head_digest=args.start_digest,initial_step_in_parent=768,gpus=1,optimizer='fresh AdamW',
        updates_per_arm=STEPS,learning_rate=single.LR,effective_batch=single.BATCH,training_order=[
            p['case_id'] for p in args.train_order],training_pool=2048,training_sampled=256,validation_pairs=256,
        samples_per_training_category=64,validation_updates=list(CHECKPOINTS),
        band_weight=dict(A=BASE_WEIGHT,B_nonpants=BASE_WEIGHT,B_pants=PANTS_WEIGHT),
        bands_sigma=[1,4],pants_core_radius=20,minimum_core_pixels=256,
        common_loss=protocol['loss'],validation_start=args.start_metrics,validation_step0=args.zero_metrics,
        tolerance=single.rules.TOLERANCE,formal_reference=protocol['formal_reference'],
        code={f:tail.sha(SOURCE/f) for f in ('tools/e2_b_single_pants_ab.py','scripts/e2_b_single_pants_ab.sbatch',
                                           'tools/e2_b_epoch20_single.py','tools/e2_b_single_pants_roi.py')})
    return args,contract


def start_head(args,models):
    head,parameters = tail.candidates(models[3])
    ab.update_head(head,parameters,args.start_head)
    assert init.head_hash(head) == args.start_digest
    return head,parameters


def forward(args,models,head,pair):
    cache = args.parent_run/'prefix_b_off'/(pair['case_id']+'.pt')
    source = args.prefix_source if cache.is_file() else args
    return single.forward(source,models,head,pair)


@torch.no_grad()
def render(args,models,head,pair,digest):
    path = args.run/'renders'/digest/(pair['case_id']+'.png')
    record = dict(protocol_sha256=args.spec_sha,case=pair['case_id'],head_sha256=digest)
    if init.checked_image(path,path.with_suffix('.json'),record) is None:
        pred,_,_ = forward(args,models,head,pair)
        init.save_image(path,path.with_suffix('.json'),single.ft.rgb8(pred),record)
        models[2].adapter.clear()


def score(args,models,heads,metrics,split,dest):
    pairs = args.pairs[split]
    if not (dest/'result.json').exists():
        for name,head in heads.items():
            digest = init.head_hash(head)
            for i,pair in enumerate(pairs,1):
                render(args,models,head,pair,digest)
                if i%64==0:
                    print('PANTS_AB_RENDER',dest.name,name,i,len(pairs),flush=True)
    # 每张候选图已落盘，标准评估器仅读取像素并计算指标。
    return single.score(args,models,pairs,heads,dest,metrics,split)


def qualification(args,models,objective):
    path = args.run/'qualification.json'
    if path.exists():
        assert tail.read(path)['protocol_sha256']==args.spec_sha
        return
    pair = next(p for p in args.pairs['val'] if p['category']=='pants'
                and args.start_rows[p['case_id']]['interior_px']>=256)
    head,params = start_head(args,models)
    _,_,full,_ = single.full(args,models,pair,head)
    pred,gt,mask = forward(args,models,head,pair)
    rgb = single.ft.rgb8(pred)
    np.testing.assert_array_equal(full,rgb)
    case = pair['case_id']
    archived = args.parent_run/'renders'/args.start_digest/(case+'.png')
    assert tail.sha(archived)==args.start_rows[case]['pngs']['H2']
    np.testing.assert_array_equal(rgb,np.asarray(tail.rgb(archived)))
    a,terms = objective(pred,gt,mask,band_weight('A',pair))
    b,_ = objective(pred,gt,mask,band_weight('B',pair))
    torch.testing.assert_close(b-a,(PANTS_WEIGHT-BASE_WEIGHT)*terms['band'],atol=5e-7,rtol=1e-4)
    assert terms['band'].item()>0
    norm = tail.backward(b,params)
    assert np.isfinite(norm) and norm>0
    models[2].adapter.clear()
    ab.audit_frozen(args,models,head,params)
    tail.write(path,dict(protocol_sha256=args.spec_sha,case=case,full_replay_archived_rgb_equal=True,
        grad_norm=norm,b_only_extra_band=float((b-a).detach()),optimizer_updates=0))
    print('PANTS_AB_QUALIFIED',case,norm,flush=True)


def update(args,models,objective,arm,head,params,optimizer,pairs):
    rows = []
    for pair in pairs:
        pred,gt,mask = forward(args,models,head,pair)
        weight = band_weight(arm,pair)
        loss,terms = objective(pred,gt,mask,weight)
        assert torch.isfinite(loss)
        (loss*single.SCALE/single.BATCH).backward()
        rows.append(dict(case=pair['case_id'],category=pair['category'],band_weight=weight,
                         loss=float(loss.detach()),**{k:float(v.detach()) for k,v in terms.items()}))
        models[2].adapter.clear()
        del pred,gt,loss,terms
    for p in params.values():
        p.grad.div_(single.SCALE)
    norm = float(torch.nn.utils.clip_grad_norm_(params.values(),1.,error_if_nonfinite=True))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return dict(grad_norm=norm,rows=rows)


def validation_gate(now,start,zero):
    tolerance = single.rules.TOLERANCE
    checks = dict(fid_below_original_step0=now['fid']<zero['fid'],lpips_kept=now['lpips']<=start['lpips'],
                  ssim_kept=now['ssim']>=start['ssim']*(1-tolerance),
                  contour_f1_kept=now['contour_f1']>=start['contour_f1']*(1-tolerance))
    return dict(checks=checks,passed=all(checks.values()))


def train(args,models,objective,metrics):
    heads,params,opts = {},{},{}
    for n in ARMS:
        heads[n],params[n] = start_head(args,models)
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
        tail.save_torch(resume,dict(protocol_sha256=args.spec_sha,step=step,trace=trace,
            checks=checks,heads={n:weights(n) for n in ARMS},optimizers={n:o.state_dict() for n,o in opts.items()}))

    def validate():
        dest = args.run/'validation'/f'step{step}'
        for n in ARMS:
            tail.save_torch(dest/(n+'.pt'),weights(n))
        result = score(args,models,heads,metrics,'val',dest)['groups']['all']
        entries = {n:init.metrics_from_group(result,n) for n in ARMS}
        checks[str(step)] = {n:dict(metrics=m,**validation_gate(m,args.start_metrics,args.zero_metrics))
                             for n,m in entries.items()}
        save()
        print('PANTS_AB_VAL',step,checks[str(step)],flush=True)

    if step in CHECKPOINTS and str(step) not in checks:
        validate()
    for step in range(step+1,STEPS+1):
        pairs = args.train_order[(step-1)*single.BATCH:step*single.BATCH]
        trace.append(dict(step=step,**{n:update(args,models,objective,n,heads[n],params[n],opts[n],pairs)
                                       for n in ARMS}))
        save()
        if step%16==0:
            print('PANTS_AB_TRAIN',step,STEPS,flush=True)
        if step in CHECKPOINTS:
            validate()
    selected = {}
    for n in ARMS:
        passing = [(s,m[n]['metrics']) for s,m in checks.items() if m[n]['passed']]
        if passing:
            s,_ = min(passing,key=lambda x:(x[1]['fid'],x[1]['lpips'],x[1]['band_mae'],int(x[0])))
            selected[n] = int(s)
            ab.update_head(heads[n],params[n],args.run/'validation'/f'step{s}'/(n+'.pt'))
        ab.audit_frozen(args,models,heads[n],params[n])
    tail.write(args.run/'training.json',dict(protocol_sha256=args.spec_sha,updates_per_arm=STEPS,
        selected=selected,validation=checks,trace=trace,frozen_unchanged=True))
    return {n:heads[n].eval().requires_grad_(False) for n in selected},selected


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('prior',type=Path)
    cli.add_argument('--check',action='store_true')
    options = cli.parse_args()
    args,contract = setup(options.prior,'cpu' if options.check else 'cuda:0')
    if options.check:
        single.archived64(args,args.pairs['test64'])
        print('PANTS_AB_CPU_CHECK',len(args.train_order),len(args.pairs['val']),
              args.zero_metrics['fid'],contract['initial_weights_sha256'],flush=True)
        return
    assert torch.cuda.device_count()==1
    if args.spec.exists():
        assert tail.read(args.spec)==contract
    else:
        tail.write(args.spec,contract)
    args.spec_sha = tail.sha(args.spec)
    if (args.run/'completion.json').is_file() and (args.run/'review.zip').is_file():
        return
    cv2.setNumThreads(1)
    models = init.load(args)
    torch.set_num_threads(6)
    assert args.original['frozen_model_hashes']==tail.read(args.parent_run/'model.json')['frozen_model_hashes']
    objective = single.Objective(args)
    qualification(args,models,objective)
    metrics = tail.metric_models('cpu',args.protocol['metric_assets'])
    active,selected = train(args,models,objective,metrics)
    stages = {}
    files = ['protocol.json','model.json','execution_environment.json','qualification.json','training.json']
    files.extend(f'validation/step{s}/result.json' for s in CHECKPOINTS)
    for split in ('test64','test500'):
        if not active:
            break
        label = 'eval64' if split=='test64' else 'eval500'
        dest = args.run/label
        single.test_controls(args,models,args.pairs[split],split)
        result = score(args,models,active,metrics,split,dest)
        decision = single.acceptance(args,dest,result,selected,split)
        tail.write(dest/'acceptance.json',decision)
        print('PANTS_AB_TEST',label,decision,flush=True)
        accepted = [n for n,x in decision['decisions'].items() if x['accepted']]
        stages[label] = dict(scored=list(active),accepted=accepted)
        active = {n:active[n] for n in accepted}
        files.extend(f'{label}/{n}' for n in ('result.json','acceptance.json','per_pair.json','pool3.npz'))
    assert tail.tensor_set(models[4])==args.original['frozen_model_hashes']
    tail.write(args.run/'completion.json',dict(updates_per_arm=STEPS,selected=selected,evaluation=stages))
    init.fid.pack(args.run,[*files,'completion.json'])
    print('PANTS_AB_DONE',selected,stages,tail.sha(args.run/'review.zip'),flush=True)


if __name__=='__main__':
    main()
