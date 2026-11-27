"""先验收冻结的 127791 B，再有条件续训。FID 仅记录。"""
import argparse
import copy
from pathlib import Path

import cv2
import numpy as np
import torch

from tools import e2_b_fixed_band_direction64 as archived
from tools.e2_fft_experiment import SOURCE

fixed = archived.fixed
single, previous, init, tail, ab = fixed.single, fixed.previous, fixed.init, fixed.tail, fixed.ab
rules = single.rules
RUN = 'fixed_band50_continue_64_without_fid_gate'
STEPS = fixed.STEPS
CHECKPOINTS = (32,64)
CONTROL = 'B_frozen'
ACTIVE_KEYS = tuple(k for k in rules.KEYS if k!='fid')


def setup(prior, device):
    args,inherited,base,folder = archived.setup(prior,device)
    manifest = tail.read(folder/'review_manifest.json')
    checked = single.fusion.read_checked
    q = checked(folder/'qualification.json',manifest['qualification.json'])
    assert q['optimizer_updates']==0 and set(q['probes'])==set(fixed.CATEGORIES)
    assert q['protocol_sha256']==manifest['protocol.json']
    stage = folder/'validation/step64'
    score = checked(stage/'result.json',manifest['validation/step64/result.json'])

    weights = stage/'B.pt'
    snapshot = torch.load(weights,map_location='cpu',weights_only=True)
    resume = folder/'resume.pt'
    state = torch.load(resume,map_location='cpu',weights_only=True)
    assert state['protocol_sha256']==manifest['protocol.json'] and state['step']==STEPS
    assert state['trace']==checked(folder/'training.json',manifest['training.json'])['trace']
    assert snapshot.keys()==state['heads']['B'].keys()
    assert all(torch.equal(w,state['heads']['B'][k]) for k,w in snapshot.items())
    adam = state['optimizers']['B']
    flat = [pid for g in adam['param_groups'] for pid in g['params']]
    assert len(flat)==len(snapshot) and set(flat)==set(adam['state'])
    assert all(g['lr']==single.LR for g in adam['param_groups'])
    for pid,w in zip(flat,snapshot.values()):
        moment = adam['state'][pid]
        assert int(moment['step'])==STEPS and moment['exp_avg'].shape==moment['exp_avg_sq'].shape==w.shape

    args.run = folder/RUN
    args.spec = args.run/'protocol.json'
    args.control_weights = weights
    args.control_digest = score['head_sha256']['B']
    args.control_pool = stage/'pool3.npz'
    args.control_rows = stage/'per_pair.json'
    contract = dict(inherited,control_job=127791,control_step=STEPS,
        control_weights_sha256=tail.sha(weights),control_resume_sha256=tail.sha(resume),
        control_head_digest=args.control_digest,control_validation=base['baseline_127791_B'],
        control_review_sha256=base['baseline_127791_review_sha256'],
        optimizer='restore 127791 B AdamW moments and step 64',additional_updates=STEPS,
        validation_updates=list(CHECKPOINTS),loss='exact 127791 B including 0.10 fixed target position L1',
        fid_policy='report only at every stage',validation_selection='strictly lower pooled band MAE than step 0; original non-FID guards; ties earlier',
        guard_keys=list(ACTIVE_KEYS),
        code={**inherited['code'],**{name:tail.sha(SOURCE/name) for name in (
            'tools/e2_b_fixed_band_continue64.py','scripts/e2_b_fixed_band_continue64.sbatch',
            'tools/e2_b_fixed_band_direction64.py','tools/e2_f68_e1_retain125_long.py',
            'tools/e2_f68_rgb_distill.py','tools/e2_b_init_epoch20_ab.py')}})
    return args,inherited,contract,adam


def restore_control(args,models):
    source = copy.copy(args)
    source.start_head = args.control_weights
    source.start_digest = args.control_digest
    head,params = previous.start_head(source,models)
    assert list(params)==list(torch.load(args.control_weights,map_location='cpu',weights_only=True))
    return head,params


def holdout_decision(args,contract,dest,result,name,split):
    n = '64' if split=='test64' else '500'
    source = args.formal64 if n=='64' else args.formal500
    manifest = contract['formal_reference'][n]
    old = single.fusion.read_checked(source/'result.json',manifest['result.json'])
    rows = single.fusion.read_checked(source/'per_pair.json',manifest['per_pair.json'])
    assert result['pairs']==len(rows)==int(n)
    assert [r['case_id'] for r in rows]==[p['case_id'] for p in args.pairs[split]]
    assert tail.sha(source/'pool3.npz')==manifest['pool3.npz']
    with np.load(source/'pool3.npz',allow_pickle=False) as a, np.load(dest/'pool3.npz',allow_pickle=False) as b:
        np.testing.assert_allclose(a['gt'],b['gt'],atol=7e-5,rtol=1e-4)

    group = result['groups']['all']
    value = init.metrics_from_group(group,name)
    four = init.metrics_from_group(group,'four_image')
    historical = init.metrics_from_group(old['groups']['all'],'retain_125')
    absolute = init.gates(group,name)
    six = {k:v for k,v in absolute['seven']['checks'].items() if k!='fid'}
    absolute_pass = all(v['passed'] for v in six.values()) and absolute['texture_pass']
    relative = {}
    thresholds = {}
    signed = {}
    for k in ACTIVE_KEYS:
        assert np.isfinite(value[k]) and four[k]>0
        thresholds[k] = max(four[k],historical[k])*(1-rules.TOLERANCE) if k in ('ssim','contour_f1') else four[k]
        relative[k] = value[k]<=thresholds[k] if k in rules.LOWER else value[k]>=thresholds[k]
        signed[k] = ((four[k]-value[k]) if k in rules.LOWER else (value[k]-four[k]))/four[k]
    gain = sum(max(v,0.) for v in signed.values())
    harm = sum(max(-v,0.) for v in signed.values())
    tradeoff = gain>=rules.GAIN_RATIO*harm
    return dict(metrics=value,six=six,texture_pass=absolute['texture_pass'],
        texture_threshold=absolute['texture_threshold'],absolute_pass=absolute_pass,
        four_image=four,historical=historical,relative=relative,relative_thresholds=thresholds,
        tradeoff=dict(relative=signed,gain=gain,harm=harm,passed=tradeoff),
        accepted=absolute_pass and all(relative.values()) and tradeoff,fid_policy='report only')


def verify_optimizer(optimizer,steps):
    assert optimizer.state and all(int(x['step'])==steps for x in optimizer.state.values())
    assert all(g['lr']==single.LR for g in optimizer.param_groups)


def train(args,models,head,params,targets,adam):
    criterion = fixed.Objective(args)
    optimizer = torch.optim.AdamW(params.values(),lr=single.LR)
    resume = args.run/'resume.pt'
    step,trace,checkpoints = 0,[],{}
    snapshot = lambda: {k:w.detach().cpu() for k,w in params.items()}
    if resume.is_file():
        state = torch.load(resume,map_location='cpu',weights_only=True)
        assert state['protocol_sha256']==args.spec_sha
        step,trace,checkpoints = state['step'],state['trace'],state['checkpoints']
        assert step==len(trace) and set(checkpoints)=={str(s) for s in CHECKPOINTS if s<=step}
        single.rules.restore(params,state['head'])
        adam = state['optimizer']
    optimizer.load_state_dict(adam)
    verify_optimizer(optimizer,STEPS+step)
    for step in range(step+1,STEPS+1):
        batch = args.train_order[(step-1)*single.BATCH:step*single.BATCH]
        update = fixed.update(args,models,head,params,optimizer,criterion,'B',batch,targets)
        trace.append(dict(step=step,**update))
        payload = snapshot()
        if step in CHECKPOINTS:
            ab.audit_frozen(args,models,head,params)
            checkpoints[str(step)] = payload
        tail.save_torch(resume,dict(protocol_sha256=args.spec_sha,step=step,trace=trace,
            checkpoints=checkpoints,head=payload,optimizer=optimizer.state_dict()))
        if step%16==0:
            print('CONTINUE64_TRAIN',step,STEPS,flush=True)
    verify_optimizer(optimizer,2*STEPS)
    assert [r['case'] for s in trace for r in s['rows']]==[p['case_id'] for p in args.train_order]
    return checkpoints,trace


def validation_decisions(args,contract,dest,result):
    control = contract['control_validation']
    threshold = control['metrics']['band_mae']
    common = {k:v for k,v in control['checks'].items() if k!='fid'}
    choices = {'0':dict(control,checks=common,passed=all(common.values()),texture_gain=0.,eligible=False)}
    archived_rows = tail.read(args.control_rows)
    assert [r['case_id'] for r in archived_rows]==[r['case_id'] for r in tail.read(dest/'per_pair.json')]
    with np.load(args.control_pool,allow_pickle=False) as a, np.load(dest/'pool3.npz',allow_pickle=False) as b:
        np.testing.assert_allclose(a['gt'],b['gt'],atol=7e-5,rtol=1e-4)
    for step in CHECKPOINTS:
        label = str(step)
        verdict = fixed.validation(result,'plus'+label,contract)
        del verdict['checks']['fid']
        verdict['passed'] = all(verdict['checks'].values())
        gain = threshold-verdict['metrics']['band_mae']
        choices[label] = dict(verdict,texture_gain=gain,eligible=verdict['passed'] and gain>0.)
    eligible = [s for s in CHECKPOINTS if choices[str(s)]['eligible']]
    winner = min(eligible,key=lambda s:(choices[str(s)]['metrics']['band_mae'],s)) if eligible else None
    return dict(choices=choices,selected=winner,fid_policy='report only')


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('prior',type=Path)
    cli.add_argument('--check',action='store_true')
    opts = cli.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(4 if opts.check else 6)
    args,inherited,contract,adam = setup(opts.prior,'cpu' if opts.check else 'cuda:0')
    if opts.check:
        single.archived64(args,args.pairs['test64'])
        assert all((args.parent_run/'prefix_b_off'/(p['case_id']+'.pt')).is_file()
                   for p in (*args.train_order,*args.pairs['val']))
        print('CONTINUE64_PRECHECK',len(args.pairs['test64']),len(args.train_order),len(args.pairs['val']),
              contract['control_weights_sha256'],flush=True)
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
    head,params = restore_control(args,models)
    ab.audit_frozen(args,models,head,params)
    head.eval().requires_grad_(False)
    metrics = tail.metric_models('cpu',args.protocol['metric_assets'])
    frozen = args.run/'frozen64'
    result = previous.score(args,models,{CONTROL:head},metrics,'test64',frozen)
    gate = holdout_decision(args,contract,frozen,result,CONTROL,'test64')
    tail.write(frozen/'acceptance.json',gate)
    print('CONTINUE64_FROZEN_GATE',gate['accepted'],gate['metrics']['fid'],gate['relative'],flush=True)
    files = ['protocol.json','model.json','execution_environment.json',
             *[f'frozen64/{n}' for n in ('result.json','acceptance.json','per_pair.json','pool3.npz')]]
    stages = dict(frozen64=dict(pairs=result['pairs'],accepted=gate['accepted']))
    selected,updates = None,0
    if gate['accepted']:
        del head,params
        targets = fixed.prepare_targets(args,inherited)
        head,params = restore_control(args,models)
        checkpoints,trace = train(args,models,head,params,targets,adam)
        updates = len(trace)
        stage = args.run/'validation256'
        heads = {}
        for step in CHECKPOINTS:
            label = 'plus'+str(step)
            chosen,chosen_params = restore_control(args,models)
            single.rules.restore(chosen_params,checkpoints[str(step)])
            ab.audit_frozen(args,models,chosen,chosen_params)
            heads[label] = chosen.eval().requires_grad_(False)
            tail.save_torch(stage/(label+'.pt'),checkpoints[str(step)])
        result = previous.score(args,models,heads,metrics,'val',stage)
        assert result['pairs']==256
        verdict = validation_decisions(args,contract,stage,result)
        selected = verdict['selected']
        tail.write(stage/'acceptance.json',verdict)
        tail.write(args.run/'training.json',dict(additional_updates=updates,optimizer_steps=[STEPS,2*STEPS],
            trace=trace,validation=verdict,frozen_unchanged=True))
        print('CONTINUE64_VAL',selected,{k:v['texture_gain'] for k,v in verdict['choices'].items()},flush=True)
        files.extend(('target_manifest.json','training.json',*[f'validation256/{n}' for n in (
            'result.json','acceptance.json','per_pair.json','pool3.npz')]))
        stages['validation256'] = dict(pairs=result['pairs'],selected=selected)
        if selected is not None:
            single.test_controls(args,models,args.pairs['test500'],'test500')
            dest = args.run/'eval500'
            label = 'plus'+str(selected)
            result = previous.score(args,models,{label:heads[label]},metrics,'test500',dest)
            decision = holdout_decision(args,contract,dest,result,label,'test500')
            tail.write(dest/'acceptance.json',decision)
            print('CONTINUE64_TEST500',decision['accepted'],decision['metrics']['fid'],flush=True)
            files.extend(f'eval500/{n}' for n in ('result.json','acceptance.json','per_pair.json','pool3.npz'))
            stages['eval500'] = dict(pairs=result['pairs'],accepted=decision['accepted'])
    assert tail.tensor_set(models[4])==args.original['frozen_model_hashes']
    tail.write(args.run/'completion.json',dict(additional_updates=updates,selected=selected,
        frozen_unchanged=True,evaluation=stages,fid_policy='report only',control_retrained=False))
    init.fid.pack(args.run,[*files,'completion.json'])
    print('CONTINUE64_DONE',updates,selected,stages,tail.sha(args.run/'review.zip'),flush=True)


if __name__=='__main__':
    main()
