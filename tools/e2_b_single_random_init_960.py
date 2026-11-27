"""复用 127720 的同批对照，训练随机初始化并且输出置零的单头 B。"""
import argparse
import copy
from pathlib import Path

import cv2
import numpy as np
import torch

from tools import e2_b_single_direct_960 as direct
from tools.e2_fft_experiment import SOURCE

single, previous, init, tail, cov = direct.single, direct.previous, direct.init, direct.tail, direct.cov
CHECKPOINTS = direct.CHECKPOINTS
CONTROL_SHA = 'c1725c2ba8ceda4097b8cbbb9638a4b085083a3449a802d0bc6060e0a7032c82'
CONTROL_960_SHA = '1ec4b413cc39ceaab78347f56a6d9cf491b3d77c4d339f061196b288d5747b42'
SEED = 42
ARM = 'random'
CONTROL = 'f68_960'
RUN = 'random_full_head_zero_output_960'


def setup(prior, device):
    args, source = direct.setup(prior,device)
    archived = args.run
    assert tail.sha(archived/'review.zip')==CONTROL_SHA
    manifest = tail.read(archived/'review_manifest.json')
    checked = single.fusion.read_checked
    assert source==checked(archived/'protocol.json',manifest['protocol.json'])
    done = checked(archived/'completion.json',manifest['completion.json'])
    trace = checked(archived/'training.json',manifest['training.json'])
    assert done['updates']==trace['updates']==CHECKPOINTS[-1]
    assert done['frozen_unchanged'] and trace['frozen_unchanged'] and done['evaluation']=={}
    assert [r['case'] for step in trace['trace'] for r in step['rows']]==source['training_order']
    assert source['optimizer_reset_after']==[768,832,896]

    identities = [(p['case_id'],p['category']) for p in args.pairs['val']]
    controls = {}
    truth = None
    for step in CHECKPOINTS:
        name = f'validation/step{step}'
        value = checked(archived/name/'result.json',manifest[name+'/result.json'])
        rows = tail.read(archived/name/'per_pair.json')
        assert value['pairs']==len(rows)==256 and value['protocol_sha256']==manifest['protocol.json']
        assert [(r['case_id'],r['category']) for r in rows]==identities
        pool = archived/name/'pool3.npz'
        assert tail.sha(pool)==value['pool3_sha256']
        with np.load(pool,allow_pickle=False) as arrays:
            if truth is None:
                truth = arrays['gt']
            else:
                np.testing.assert_array_equal(arrays['gt'],truth)
        controls[step] = value
    tail_path = archived/f'validation/step{CHECKPOINTS[-1]}/direct.pt'
    assert tail.sha(tail_path)==CONTROL_960_SHA
    assert all((args.parent_run/'prefix_b_off'/(p['case_id']+'.pt')).is_file()
               for p in (*args.train_order,*args.pairs['val']))

    args.run = archived/RUN
    args.spec = args.run/'protocol.json'
    contract = {k:v for k,v in source.items() if k not in ('initial','thresholds','code')}
    contract.update(initial='random full StateTextureB; zero output Conv2d; freeze input, time and first block',
        init_seed=SEED,trainable_parameters=tail.PARAMETERS,control_job=127720,control_review_sha256=CONTROL_SHA,
        control_tail960_sha256=CONTROL_960_SHA,control_full960_digest=controls[960]['head_sha256']['direct'],
        evaluation='same validation 256 at each checkpoint; both step 960 heads on original 64',
        fid_policy='report only; no gate or selection',code={**source['code'],**{
            name:tail.sha(SOURCE/name)
            for name in ('tools/e2_b_single_random_init_960.py','scripts/e2_b_single_random_init_960.sbatch',
                         'models/e2_state_texture_b.py')}})
    return args,contract,tail_path,controls,truth


def random_models(args,models):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(SEED)
        original = tail.StateTextureB(color_channels=2,reference_channels=192)
    assert not torch.count_nonzero(original.output.weight) and not torch.count_nonzero(original.output.bias)
    assert all(not torch.equal(original.state_dict()[key],models[3].state_dict()[key].cpu())
               for key in ('input.weight','time.0.weight','blocks.0.2.weight','blocks.1.2.weight'))
    digest = init.head_hash(original)
    path,meta = args.run/'random_initial_head.pt',args.run/'random_initial_head.json'
    if path.is_file():
        saved = torch.load(path,map_location='cpu',weights_only=True)
        assert saved.keys()==original.state_dict().keys()
        assert all(torch.equal(w,saved[k]) for k,w in original.state_dict().items())
    else:
        tail.save_torch(path,original.state_dict())
    record = dict(seed=SEED,head_digest=digest,sha256=tail.sha(path),protocol_sha256=args.spec_sha)
    if meta.is_file():
        assert tail.read(meta)==record
    else:
        tail.write(meta,record)
    original.to(args.device).eval().requires_grad_(False)
    return (*models[:3],original,models[4])


def qualify(args,models,objective):
    path = args.run/'qualification.json'
    if path.is_file():
        assert tail.read(path)['protocol_sha256']==args.spec_sha
        return
    initial_digest = init.head_hash(models[3])
    for index,pair in enumerate(args.pairs['qualification']):
        def probe(pred,gt,mask,_):
            return objective(pred,gt,mask,'B',pair['category'])
        single.qualification(args,models,probe,pair,index)
    assert init.head_hash(models[3])==initial_digest

    pair = next(p for p in args.train_order if p['category']=='pants' and
                cov.roi.regions(tail.case_mask(args,p),cov.CORE_RADIUS)[0].sum()>=cov.MIN_PIXELS)
    head,params = single.initialize(models)
    pred,gt,mask = previous.forward(args,models,head,pair)
    common,_ = objective(pred,gt,mask,'A','pants')
    full,terms = objective(pred,gt,mask,'B','pants')
    term = cov.COV_WEIGHT*terms['pants_covariance']
    torch.testing.assert_close(full-common,term,atol=5e-7,rtol=1e-4)
    output = params['output.weight']
    scale,probe_scale = single.SCALE/single.BATCH,4096.
    rgb,g64 = torch.autograd.grad(scale*term,(pred,output),retain_graph=True)
    g4096, = torch.autograd.grad(probe_scale*term,output,retain_graph=True)
    gfull, = torch.autograd.grad(scale*full,output,retain_graph=True)
    gcommon, = torch.autograd.grad(scale*common,output)
    norm = lambda x,s:float(torch.linalg.vector_norm(x.float()/s))
    gradients = dict(rgb=norm(rgb,scale),isolated64=norm(g64,scale),
        isolated4096=norm(g4096,probe_scale),production_total=norm(gfull,scale),
        production_increment=norm(gfull.float()-gcommon.float(),scale))
    record = dict(protocol_sha256=args.spec_sha,case=pair['case_id'],optimizer_updates=0,
        production_scale=scale,probe_scale=probe_scale,gradients={
            name:v if np.isfinite(v) else None for name,v in gradients.items()})
    tail.write(args.run/'qualification_gradients.json',record)
    assert all(np.isfinite(v) and v>0 for k,v in gradients.items() if k!='isolated64'),gradients
    assert np.isfinite(gradients['isolated64'])
    single.ab.audit_frozen(args,models,head,params)
    models[2].adapter.clear()
    tail.write(path,record)
    print('RANDOM960_QUALIFIED',gradients,flush=True)


def measure(result, arm):
    return dict(direct.summarize(result,arm),pants_fid=result['groups']['pants']['arms'][arm]['fid']['fid'])


def compare(result, arm, control, control_arm):
    new,old = measure(result,arm),measure(control,control_arm)
    assert result['pairs']==control['pairs']
    assert all(new['texture'][k]['eligible_pairs']==old['texture'][k]['eligible_pairs']
               for k in direct.CATEGORIES)
    return dict(pairs=result['pairs'],random=new,f68=old,
        metric_delta={k:v-old['metrics'][k] for k,v in new['metrics'].items()},
        pants_fid_delta=new['pants_fid']-old['pants_fid'],
        texture_delta={k:v['band_mae']-old['texture'][k]['band_mae'] for k,v in new['texture'].items()})


def train(args,models,objective,metrics,contract,controls,truth):
    head,params = single.initialize(models)
    optimizer = torch.optim.AdamW(params.values(),lr=single.LR)
    resume = args.run/'resume.pt'
    step,trace,checks = 0,[],{}
    if resume.is_file():
        state = torch.load(resume,map_location='cpu',weights_only=True)
        assert state['protocol_sha256']==args.spec_sha
        step,trace,checks = state['step'],state['trace'],state['comparison']
        assert step==len(trace) and set(checks)<={str(s) for s in CHECKPOINTS if s<=step}
        single.rules.restore(params,state['head'])
        optimizer.load_state_dict(state['optimizer'])

    def snapshot():
        return {n:w.detach().cpu() for n,w in params.items()}

    def save():
        tail.save_torch(resume,dict(protocol_sha256=args.spec_sha,step=step,trace=trace,
            comparison=checks,head=snapshot(),optimizer=optimizer.state_dict()))

    def validate():
        dest = args.run/'validation'/f'step{step}'
        tail.save_torch(dest/(ARM+'.pt'),snapshot())
        result = previous.score(args,models,{ARM:head},metrics,'val',dest)
        rows = tail.read(dest/'per_pair.json')
        assert [r['case_id'] for r in rows]==[p['case_id'] for p in args.pairs['val']]
        with np.load(dest/'pool3.npz',allow_pickle=False) as arrays:
            np.testing.assert_allclose(arrays['gt'],truth,atol=7e-5,rtol=1e-4)
        checks[str(step)] = compare(result,ARM,controls[step],'direct')
        tail.write(dest/'comparison.json',checks[str(step)])
        save()
        m = checks[str(step)]['random']['metrics']
        print('RANDOM960_VAL',step,m['band_mae'],m['contour_f1'],m['lpips'],m['ssim'],m['fid'],flush=True)

    if step in CHECKPOINTS and str(step) not in checks:
        validate()
    for step in range(step+1,CHECKPOINTS[-1]+1):
        if step-1 in contract['optimizer_reset_after']:
            optimizer = torch.optim.AdamW(params.values(),lr=single.LR)
        offset = (step-1)*single.BATCH
        pairs = args.train_order[offset:offset+single.BATCH]
        trace.append(dict(step=step,**cov.update(args,models,objective,'B',head,params,optimizer,pairs)))
        save()
        if step%16==0:
            print('RANDOM960_TRAIN',step,CHECKPOINTS[-1],flush=True)
        if step in CHECKPOINTS:
            validate()
    assert step==CHECKPOINTS[-1] and set(checks)=={str(s) for s in CHECKPOINTS}
    assert [r['case'] for s in trace for r in s['rows']]==contract['training_order']
    single.ab.audit_frozen(args,models,head,params)
    tail.save_torch(args.run/'random_full_step960.pt',{n:w.detach().cpu() for n,w in head.state_dict().items()})
    tail.write(args.run/'training.json',dict(updates=step,trace=trace,comparison=checks,frozen_unchanged=True))
    return head.eval().requires_grad_(False)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('prior',type=Path)
    cli.add_argument('--check',action='store_true')
    options = cli.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(4 if options.check else 6)
    args,contract,control_path,controls,truth = setup(options.prior,'cpu' if options.check else 'cuda:0')
    if options.check:
        single.archived64(args,args.pairs['test64'])
        print('RANDOM960_PRECHECK',len(args.train_order),len(args.pairs['val']),len(args.pairs['test64']),
              contract['optimizer_reset_after'],flush=True)
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
    rnd_models = random_models(args,models)
    objective = cov.Objective(args)
    qualify(args,rnd_models,objective)
    metrics = tail.metric_models('cpu',args.protocol['metric_assets'])
    candidate = train(args,rnd_models,objective,metrics,contract,controls,truth)
    del objective

    control_args = copy.copy(args)
    control_args.start_head = control_path
    control_args.start_digest = contract['control_full960_digest']
    control,_ = previous.start_head(control_args,models)
    control.eval().requires_grad_(False)
    dest = args.run/'eval64'
    result = previous.score(args,models,{CONTROL:control,ARM:candidate},metrics,'test64',dest)
    assert result['pairs']==len(args.pairs['test64'])==64
    decision = compare(result,ARM,result,CONTROL)
    tail.write(dest/'comparison.json',decision)
    print('RANDOM960_TEST64',decision['metric_delta'],decision['texture_delta'],flush=True)

    assert tail.tensor_set(models[4])==args.original['frozen_model_hashes']
    assert init.head_hash(rnd_models[3])==tail.read(args.run/'random_initial_head.json')['head_digest']
    tail.write(args.run/'completion.json',dict(updates=CHECKPOINTS[-1],frozen_unchanged=True,
        trajectories=1,vae_decodes=1,control_retrained=False,fid_policy='report only',
        evaluation=dict(validation256=dict(pairs=256,checkpoints=list(CHECKPOINTS)),
                        eval64=dict(pairs=64,checkpoint=960,arms=[CONTROL,ARM]))))
    files = ['protocol.json','model.json','execution_environment.json','random_initial_head.json',
        'qualification.json','qualification_gradients.json','qualification_probe0.json','qualification_probe1.json',
        'training.json','completion.json']
    files.extend(f'validation/step{s}/{n}' for s in CHECKPOINTS for n in ('result.json','comparison.json'))
    files.extend(f'eval64/{n}' for n in ('result.json','comparison.json','per_pair.json','pool3.npz'))
    init.fid.pack(args.run,files)
    print('RANDOM960_DONE',CHECKPOINTS[-1],tail.sha(args.run/'review.zip'),flush=True)


if __name__=='__main__':
    main()
