"""复用第 64 次优化器状态，按预先固定的指标取舍规则分阶段续训。"""
import argparse
import os
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

from tools import e2_f68_e1_retain125_guarded as guard


RUN = 'retain125_long_tradeoff_20261123'
ARM = 'retain125_long'
SOURCE_SHA = 'ca0c7747434ba1f4a32f2e4e06f2711b864a5539dde33deba4280f3ce9f56e91'
PHASES = (256, 512, 1024)
PERIOD = 64
TOLERANCE = .005
GAIN_RATIO = 2.
KEYS = ('lpips','fid','ssim','psnr','contour_f1','iou','leak','band_mae')
LOWER = {'lpips','fid','leak','band_mae'}
ft,tail,ab,roi,fid = guard.ft,guard.tail,guard.ab,guard.roi,guard.fid


def tradeoff(candidate, baseline):
    # 相对变化仅作续训/选点启发式；正式七项底线在测试时逐项另验。
    assert all(np.isfinite(candidate[k]) and baseline[k]>0 for k in KEYS)
    relative = {k:((baseline[k]-candidate[k]) if k in LOWER else
                   (candidate[k]-baseline[k]))/baseline[k] for k in KEYS}
    gain = sum(max(v,0.) for v in relative.values())
    harm = sum(max(-v,0.) for v in relative.values())
    checks = dict(lpips=candidate['lpips']<=baseline['lpips'],fid=candidate['fid']<=baseline['fid'],
        primary_improved=candidate['lpips']<baseline['lpips'] or candidate['fid']<baseline['fid'],
        band_mae=candidate['band_mae']<baseline['band_mae'],
        ssim=candidate['ssim']>=baseline['ssim']*(1-TOLERANCE),
        contour_f1=candidate['contour_f1']>=baseline['contour_f1']*(1-TOLERANCE),
        benefit_dominates=gain>0 and gain>=GAIN_RATIO*harm)
    return dict(relative=relative,gain=gain,harm=harm,net=gain-harm,
                gain_harm_ratio=gain/harm if harm else None,checks=checks,accepted=all(checks.values()))


def choose(checkpoints):
    scores = {s:tradeoff(c,checkpoints['0']) for s,c in checkpoints.items()}
    eligible = [int(s) for s,c in scores.items() if c['accepted']]
    selected = max(eligible,key=lambda s:(scores[str(s)]['net'],-checkpoints[str(s)]['lpips'],
                                         -checkpoints[str(s)]['fid'],-s)) if eligible else 0
    return selected,scores


def link(path, source):
    path.parent.mkdir(exist_ok=True,parents=True)
    if not path.exists():
        path.symlink_to(source,target_is_directory=source.is_dir())
    assert path.resolve()==source.resolve()


def setup(prior):
    args = guard.setup(prior)
    args.source,args.source_spec = args.run,args.spec
    names = ['protocol.json','training.json','references.json','qualification.json','completion.json']
    for s in (0,*guard.SNAPSHOTS):
        names.extend([f'validation/step{s}.json',f'validation/step{s}_pool3.npz'])
    tail.verified_files(args.source,SOURCE_SHA,names)
    source = tail.read(args.source/'training.json')
    assert source['protocol']==args.source_spec and source['updates']==PERIOD
    assert source['frozen_unchanged'] and source['parameter_count']==tail.PARAMETERS
    assert tail.sha(args.source/'step64.pt')==source['validation']['64']['sha256']
    args.run = args.light_run/RUN
    args.run.mkdir(exist_ok=True)
    spec = dict(args.source_spec,code_sha256=tail.sha(Path(__file__)),source_archive_sha256=SOURCE_SHA,
        source_protocol_sha256=tail.sha(args.source/'protocol.json'),
        resume_sha256=tail.sha(args.source/'resume.pt'),source_training_updates=PERIOD,
        resume_checkpoint_sha256=source['validation']['64']['sha256'],
        phases=list(PHASES),validation_period=PERIOD,relative_tolerance=TOLERANCE,
        gain_ratio=GAIN_RATIO,tradeoff_metrics=list(KEYS),checkpoint_every_update=True,
        selection='accepted maximum relative net benefit; source checkpoints and zero retained',
        continuation='new phase best must qualify and exceed preceding best; maximum 1024 updates')
    spec.pop('optimizer_updates')
    spec.pop('validation_steps')
    path = args.run/'protocol.json'
    if path.is_file():
        assert tail.read(path)==spec
    else:
        tail.write(path,spec)
    args.spec = spec
    for name in ('protocol','training','qualification'):
        path = args.run/f'source_{name}.json'
        original = tail.read(args.source/f'{name}.json')
        if path.is_file():
            assert tail.read(path)==original
        else:
            tail.write(path,original)
    link(args.run/'references',args.source/'references')
    link(args.run/'references.json',args.source/'references.json')
    return args,source


def seed_validation(args, source):
    checkpoints = {}
    for s in (0,*guard.SNAPSHOTS):
        name = f'validation/step{s}'
        original = tail.read(args.source/(name+'.json'))
        assert original['protocol']==args.source_spec and original['step']==s
        features = args.source/(name+'_pool3.npz')
        assert tail.sha(features)==original['features_sha256']
        link(args.run/(name+'_pool3.npz'),features)
        report = dict(original,protocol=args.spec,reused_source_step=s)
        path = args.run/(name+'.json')
        if path.is_file():
            assert tail.read(path)==report
        else:
            tail.write(path,report)
        checkpoint = source['validation'][str(s)]['sha256']
        if s:
            original_weight = args.source/f'step{s}.pt'
            assert tail.sha(original_weight)==checkpoint
            link(args.run/f'step{s}.pt',original_weight)
        else:
            assert tail.sha(args.start)==checkpoint
        checkpoints[str(s)] = dict(report,sha256=checkpoint)
    return checkpoints


@torch.no_grad()
def restore(parameters, state):
    assert set(parameters)==set(state)
    for name,value in parameters.items():
        value.copy_(state[name])


def train(args, models, source):
    head,parameters = tail.candidates(models[3])
    assert list(parameters)==source['trainable']
    completed_path = args.run/'training.json'
    if completed_path.is_file():
        done = tail.read(completed_path)
        assert done['protocol']==args.spec and done['frozen_unchanged']
        path = args.start if done['selected']==0 else args.run/f"step{done['selected']}.pt"
        assert tail.sha(path)==done['validation'][str(done['selected'])]['sha256']
        return ab.update_head(head,parameters,path).eval().requires_grad_(False),done
    training = [p for p in args.trainval['pairs'] if p['split']=='train']
    validation = [p for p in args.trainval['pairs'] if p['split']=='val']
    assert (len(training),len(validation))==(128,32)
    fixed = guard.anchors(args)
    reference = tail.read(args.run/'references.json')
    assert set(reference)==set(args.contract['prefix'])
    images = {case:fid.checked_png(args.run/'references'/(case+'.png'),record['png_sha256'])
              for case,record in reference.items()}
    criterion = guard.Objective(ft.light.LightingObjective(args),reference,images)
    optimizer = torch.optim.AdamW(parameters.values(),lr=ft.LR)
    checkpoints = seed_validation(args,source)
    trace,phases = list(source['trace']),[]
    state = torch.load(args.source/'resume.pt',map_location='cpu',weights_only=False)
    assert state['step']==PERIOD and state['protocol']==args.source_spec
    assert state['trace']==trace and state['validation']==source['validation']
    completed = PERIOD
    ab.update_head(head,parameters,args.source/'step64.pt')
    optimizer.load_state_dict(state['optimizer'])
    resume = args.run/'resume.pt'
    if resume.is_file():
        state = torch.load(resume,map_location='cpu',weights_only=False)
        assert state['protocol']==args.spec and state['source_resume_sha256']==args.spec['resume_sha256']
        restore(parameters,state['head'])
        optimizer.load_state_dict(state['optimizer'])
        completed,trace,checkpoints,phases = (state[k] for k in ('step','trace','validation','phases'))
    assert len(trace)==completed*ft.ACCUMULATE and trace[:128]==source['trace']
    assert all(group['lr']==ft.LR for group in optimizer.param_groups)
    optimizer.zero_grad(set_to_none=True)
    ab.audit_frozen(args,models,head,parameters)
    from pytorch_fid.inception import InceptionV3
    inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).eval().requires_grad_(False)

    def save_resume():
        tail.save_torch(resume,dict(step=completed,protocol=args.spec,
            source_resume_sha256=args.spec['resume_sha256'],head={k:p.detach().cpu() for k,p in parameters.items()},
            optimizer=optimizer.state_dict(),trace=trace,validation=checkpoints,phases=phases))

    def complete_validation(step):
        path = args.run/f'step{step}.pt'
        weights = {k:p.detach().cpu() for k,p in parameters.items()}
        if path.is_file():
            saved = torch.load(path,map_location='cpu',weights_only=True)
            assert all(torch.equal(saved[k],v) for k,v in weights.items())
        else:
            tail.save_torch(path,weights)
        result = guard.validate(args,models,head,criterion,validation,fixed,reference,images,inception,step)
        checkpoints[str(step)] = dict(result,sha256=tail.sha(path))
        assessment = tradeoff(result,checkpoints['0'])
        print('LONG_VALIDATION',step,assessment['accepted'],assessment['net'],flush=True)
        save_resume()

    if completed>PERIOD and completed%PERIOD==0 and str(completed) not in checkpoints:
        complete_validation(completed)
    boundary = PERIOD
    for goal in PHASES:
        earlier = next((p for p in phases if p['end']==goal),None)
        if earlier:
            boundary = goal
            if not earlier['continue_training']:
                break
            continue
        before = {s:c for s,c in checkpoints.items() if int(s)<=boundary}
        previous_best,previous_scores = choose(before)
        for step in range(completed+1,goal+1):
            epoch,position = divmod(step-1,PERIOD)
            order = np.random.default_rng(42+epoch).permutation(128)
            for micro in range(ft.ACCUMULATE):
                pair = training[order[position*ft.ACCUMULATE+micro]]
                rgb,target,mask = ft.forward(args,models,head,pair,fixed)
                loss,terms = guard.collect(criterion,rgb,target,mask,pair)
                assert torch.isfinite(loss)
                (loss*(ft.SCALE/ft.ACCUMULATE)).backward()
                trace.append(dict(pair=pair['case_id'],epoch=epoch+1,loss=float(loss.detach()),**terms))
                models[2].adapter.clear()
                del loss,rgb,target
            assert all(p.grad is not None for p in parameters.values())
            for p in parameters.values():
                p.grad.div_(ft.SCALE)
            norm = float(torch.nn.utils.clip_grad_norm_(parameters.values(),1,error_if_nonfinite=True))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            completed = step
            trace[-1].update(step=step,grad_norm=norm)
            save_resume()
            if step%8==0:
                print('LONG_TRAIN',step,goal,trace[-1]['loss'],flush=True)
            if step%PERIOD==0:
                complete_validation(step)
        selected,assessments = choose(checkpoints)
        new_best = selected>boundary and assessments[str(selected)]['net']>previous_scores[str(previous_best)]['net']
        continued = new_best and goal<PHASES[-1]
        phases.append(dict(start=boundary,end=goal,previous_best=previous_best,selected=selected,
                           new_best=bool(new_best),continue_training=bool(continued),
                           net=assessments[str(selected)]['net']))
        ab.audit_frozen(args,models,head,parameters)
        save_resume()
        print('LONG_PHASE',goal,selected,continued,flush=True)
        boundary = goal
        if not continued:
            break
    selected,assessments = choose(checkpoints)
    path = args.start if selected==0 else args.run/f'step{selected}.pt'
    assert tail.sha(path)==checkpoints[str(selected)]['sha256']
    ab.update_head(head,parameters,path)
    ab.audit_frozen(args,models,head,parameters)
    done = dict(protocol=args.spec,updates=completed,new_updates=completed-PERIOD,source_updates=PERIOD,
        selected=selected,selected_reused=selected<=PERIOD,validation=checkpoints,
        tradeoff=assessments,phases=phases,trace=trace,trainable=list(parameters),
        parameter_count=tail.PARAMETERS,frozen_unchanged=True)
    tail.write(completed_path,done)
    print('LONG_SELECTED',selected,completed,flush=True)
    return head.eval().requires_grad_(False),done


def group_metrics(group, name):
    result = {k:group['arms'][name][k]['fid' if k=='fid' else 'mean'] for k in KEYS if k!='band_mae'}
    return dict(result,band_mae=group['texture'][name]['band_mae'])


def evaluate(args, models, head, trained, small):
    if small and trained['selected']==0:
        result = guard.reused_start64(args,trained)
    else:
        result = ft.evaluate(args,models,head,trained,small)
    dest = args.run/('eval64' if small else 'eval500')
    strict = result.get('strict_acceptance',result['acceptance'])
    group = result['groups']['all']
    assessment = tradeoff(group_metrics(group,ARM),group_metrics(group,ft.BASE))
    accepted = strict['seven']['passed'] and strict['texture_pass'] and assessment['accepted']
    result['strict_acceptance'] = strict
    result['acceptance'] = dict(seven=strict['seven'],texture_pass=strict['texture_pass'],
                                tradeoff=assessment,accepted=bool(accepted))
    tail.write(dest/'result.json',result)
    fid.pack(dest,('result.json','per_pair.json','candidate_pool3.npz','audit.json'))
    print('LONG_TEST',dest.name,accepted,assessment['net'],flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prior',type=Path)
    cv2.setNumThreads(1)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
    args,source = setup(parser.parse_args().prior)
    if (args.run/'review.zip').is_file() and (args.run/'completion.json').is_file():
        print('LONG_ALREADY_COMPLETE',tail.sha(args.run/'review.zip'),flush=True)
        return
    models = ab.load(args)
    head,trained = train(args,models,source)
    with patch.object(ft,'ARM',ARM),patch.object(ft,'ARMS',(*ft.retention.SCORED,ARM)),patch.object(guard,'ARM',ARM):
        small = evaluate(args,models,head,trained,True)
        passed = small['acceptance']['accepted']
        if passed:
            evaluate(args,models,head,trained,False)
    finish = dict(updates=trained['updates'],new_updates=trained['new_updates'],selected=trained['selected'],
        selected_reused=trained['selected_reused'],eval64_passed=passed,eval500_executed=passed,
        eval500_passed=tail.read(args.run/'eval500/result.json')['acceptance']['accepted'] if passed else None)
    tail.write(args.run/'completion.json',finish)
    files = ['protocol.json','source_protocol.json','source_training.json','source_qualification.json',
             'references.json','training.json','completion.json','eval64/review.zip']
    for step in trained['validation']:
        files.extend([f'validation/step{step}.json',f'validation/step{step}_pool3.npz'])
    if passed:
        files.append('eval500/review.zip')
    fid.pack(args.run,files)
    print('LONG_FINISHED',trained['updates'],trained['selected'],passed,tail.sha(args.run/'review.zip'),flush=True)


if __name__=='__main__':
    main()
