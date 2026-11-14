"""先诊断实际AdamW位移及短/完整轨迹，再做不超过50次尝试的小试。"""
import argparse
import copy
import json
import os
import shutil
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from tools import e2_b_balanced_losses as old_loss
from tools import e2_b_endpoint_losses as loss
from tools import e2_state_b_balanced as old
from tools import e2_state_b_joint_data as data
from tools import e2_state_b_pilot as pilot
from tools import e2_state_b_repair as repair
from tools.e2_fft_experiment import SOURCE, read, write, sha
from tools.e2_oracle_texture import weights_digest


def source_hashes():
    names=set(old.source_hashes())|{'tools/e2_b_endpoint_losses.py','tools/e2_state_b_endpoint.py',
                                  'tests/test_e2_b_endpoint.py','scripts/e2_state_b_endpoint.sbatch'}
    return {name:sha(SOURCE/name) for name in sorted(names)}


def linked(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if not destination.exists():
        try:os.link(source,destination)
        except OSError:shutil.copyfile(source,destination)
    assert sha(source)==sha(destination)


def prepare(args):
    assert not (args.out/'protocol.json').exists()
    spec=read(args.parent/'protocol.json')
    protected=set(spec['protected_product_roots'])
    assert len(protected)==22 and len(spec['holdout'])==11
    assert not protected & {p['identity_group'] for p in spec['train']+spec['references']}
    assert len(spec['train'])==4 and len(spec['references'])==9
    # 不复制、读取或生成任何留出身份/供体的像素。
    for p in spec['train']:
        source=args.parent/'cases'/p['case_id'];dest=args.out/'cases'/p['case_id']
        for name in ('mask.png','sketch.jpg','train_anchor.png'):
            linked(source/name,dest/name)
        for c in spec['conditions']:
            for suffix in ('ref','goal','positive','baseline'):
                linked(source/f'{c}_{suffix}.png',dest/f'{c}_{suffix}.png')
    linked(args.parent/'base_cache.pt',args.out/'base_cache.pt')
    assert sha(args.out/'base_cache.pt')==read(args.parent/'diagnostics.json')['base_cache_sha256']
    spec.update(code=source_hashes(),parent_protocol_sha256=sha(args.parent/'protocol.json'),
                heldout_used=False,max_attempts=50,checkpoint_steps=[0,10,25,50],
                training_rule='full25 endpoint, independent L amplitude/correlation/light, project actual AdamW displacement; finite-step backtracking',
                update_gate='each reference x each task nonincreasing within 1e-5+1e-4*abs(loss); no pair averaging for protection',
                no_expansion=True)
    write(args.out/'protocol.json',spec)
    print('ENDPOINT_PREPARED train4 refs9 protected22 max50',flush=True)


def protocol(args):
    spec=read(args.out/'protocol.json');assert spec['code']==source_hashes()
    assert sha(args.parent/'protocol.json')==spec['parent_protocol_sha256']
    return spec


class Objective(old.Objective):
    def rollout(self,entry,start=0,hops=25):
        p,c=entry['pair'],entry['condition'];values=self.records[p['case_id'],c][3]
        state=entry['states'][start]
        z=state['zt'].to(self.args.device);epsilon=state['epsilon'].to(self.args.device)
        for hop in range(hops):
            index=state['index']+hop;t=self.pipe.scheduler.timesteps[index]
            assert int(t)==state['t']-20*hop
            alpha=self.pipe.scheduler.alphas_cumprod[int(t)].to(z.device).float()
            x0=(z.float()-(1-alpha).sqrt()*epsilon.float())/alpha.sqrt()
            delta=self.b(z.float(),x0,timestep=t.reshape(1),**values)
            corrected=(epsilon.float()-(alpha/(1-alpha)).sqrt()*delta).to(epsilon.dtype)
            previous=self.pipe.scheduler.step(corrected,t,z,return_dict=False)[0]
            if hop+1==hops:
                predicted=previous.float() if index==49 else x0+delta
                with torch.no_grad():
                    base=self.pipe.scheduler.step(epsilon,t,z.detach(),return_dict=False)[0].float() if index==49 else x0.detach()
                return predicted,base
            z=previous
            epsilon=self.predictor.epsilon(z,self.pipe.scheduler.timesteps[index+1],entry)

    def tasks(self,entry,start=0,hops=25):
        p,c=entry['pair'],entry['condition']
        target,positive,original,values,weight,template=self.records[p['case_id'],c]
        predicted,base=self.rollout(entry,start,hops)
        rgb=self.rgb(predicted)
        with torch.no_grad():base=self.rgb(base)
        terms=old_loss.terms(rgb,positive,original,base,weight,template,p['patch_coordinates'])
        terms['latent']=pilot.weighted_mean((predicted-target).square(),values['mask'])
        extra,metrics=loss.luminance(rgb,positive,weight,p['patch_coordinates'])
        return loss.groups(terms,extra),dict(metrics,terms={k:float(v.detach()) for k,v in terms.items()})


def load(args,spec):
    # 诊断和小试使用相同确定性反传后端，避免action名称改变SDPA/cuDNN配置。
    runtime=copy.copy(args);runtime.action='train'
    core,pipe,net=pilot.load_core(runtime)
    b=data.initialize(args);initial=weights_digest(b)
    cache=torch.load(args.out/'base_cache.pt',map_location='cpu',weights_only=False)
    pipe.vae.half()
    predictor=repair.FrozenPredictor(args,core,pipe,net,spec['train'],conditions=spec['conditions'])
    objective=Objective(args,spec,pipe,predictor,b)
    pipe.scheduler.set_timesteps(spec['sampler']['steps'],device=args.device)
    return core,pipe,net,b,cache,objective,initial


def final_sample(args,core,pipe,net,p,c,b):
    original=pipe.scheduler.step;record={}
    def step(*a,**kw):
        result=original(*a,**kw)
        record['z']=(result[0] if isinstance(result,tuple) else result.prev_sample).detach()
        return result
    pipe.scheduler.step=step
    try:picture,digest,tap=old.sample(args,core,pipe,net,p,c,b,capture=True)
    finally:pipe.scheduler.step=original
    return picture,digest,tap,record['z']


def numbers(tasks):return {k:float(tasks[k].detach()) for k in loss.TASKS}


def optimizer(b):return torch.optim.AdamW(b.parameters(),lr=3e-5,weight_decay=.01)


def guarded_step(b,opt,gradients,weights,evaluate,before_values):
    params=list(b.parameters());before=loss.flatten(params);state=copy.deepcopy(opt.state_dict())
    merged=(gradients.mean(0)*weights[:,None]).sum(0)
    old_loss.assign(params,merged);norm=torch.nn.utils.clip_grad_norm_(params,1.)
    assert torch.isfinite(norm)
    opt.step();raw=before-loss.flatten(params)
    corrected,projection=loss.project_update(raw,gradients.flatten(0,1))
    report=dict(projection=projection,raw_alignment=old_loss.alignment(gradients.flatten(0,1),raw),
                accepted=False,backtracks=[],gradient_norm=float(norm))
    if projection['feasible']:
        for scale in (1.,.5,.25,.125):
            loss.set_parameters(params,before-scale*corrected)
            # 参数float32舍入后再次检查，不能只检查理想位移。
            actual=before-loss.flatten(params)
            cos=old_loss.alignment(gradients.flatten(0,1),actual)
            if min(cos)<-1e-5:
                report['backtracks'].append(dict(scale=scale,reason='rounded_update_conflict'));continue
            with torch.no_grad():after_values=evaluate()
            passing=all(after[k]<=prior[k]+1e-5+1e-4*abs(prior[k])
                        for prior,after in zip(before_values,after_values) for k in loss.TASKS)
            report['backtracks'].append(dict(scale=scale,passing=passing,values=after_values))
            if passing and float(actual.norm())>0:
                report.update(accepted=True,scale=scale,actual_alignment=cos,displacement=float(actual.norm()))
                return report
    loss.set_parameters(params,before);opt.load_state_dict(state)
    report.update(actual_alignment=[0.]*gradients.flatten(0,1).shape[0],displacement=0.)
    return report


def archive(args):
    files=[p for p in args.out.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.png','.pt')
           and 'source' not in p.parts and 'cases' not in p.parts and p.name!='base_cache.pt']
    dest=args.out/'endpoint_review.zip'
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.relative_to(args.out))
    print('ENDPOINT_ARCHIVE',dest.name,sha(dest),flush=True)


def diagnose(args):
    spec=protocol(args);core,pipe,net,b,cache,obj,initial=load(args,spec)
    params=list(b.parameters());frozen=weights_digest(net);base=copy.deepcopy(b.state_dict())
    saved=torch.load(args.parent/'training/B/step50.pt',map_location='cpu',weights_only=False)
    probes=[]
    for i,group in enumerate(('neutral','pastel','multi','synthetic')):
        p=spec['train'][i]
        row=next(r for r in spec['schedule'] if r['group']==group and r['case']==p['case_id'])
        probes.append((p,row['conditions'][0],group))
    rows=[];norms=[];good=[]
    for label,state in (('oldB400',base),('B2step50',saved['b'])):
        b.load_state_dict(state)
        for p,c,group in probes:
            picture,digest,tap,actual=final_sample(args,core,pipe,net,p,c,b)
            cached=cache[p['case_id'],c]['states'][0]
            assert all(torch.equal(cached[k],tap.states[0][k]) for k in ('zt','epsilon'))
            entry=dict(cache[p['case_id'],c],states=tap.states)
            with torch.no_grad():replay,_=obj.rollout(entry)
            difference=float((replay-actual.float()).abs().max())
            assert difference<=2e-4,('full_replay_mismatch',label,p['case_id'],c,difference)
            folder=args.out/'diagnostic_images';folder.mkdir(exist_ok=True)
            picture.save(folder/f'{label}_{p["case_id"]}_{c}.png')
            tasks,values=obj.tasks(entry);full=loss.task_gradients(tasks,params)
            assert torch.isfinite(full).all() and (full[[0,1,2,4,5]].norm(dim=1)>0).all()
            record=dict(model=label,case=p['case_id'],condition=c,group=group,
                        full=numbers(tasks),full_metrics=values,latent_max_error=difference,short=[])
            if label=='oldB400':norms.append(full.norm(dim=1).cpu())
            for start,hops in ((9,2),(21,4)):
                short_tasks,short_values=obj.tasks(entry,start,hops)
                g=loss.task_gradients(short_tasks,params)
                assert torch.isfinite(g).all()
                legacy=torch.stack((g[0],g[3]+g[6],g[4]+5*g[5]))
                weights=torch.tensor([spec['task_weights'][k] for k in old_loss.TASKS],device=args.device)
                direction=old_loss.merge(legacy,weights,True,20261114)
                opt=optimizer(b)
                if label=='B2step50':opt.load_state_dict(saved['optimizer'])
                old_loss.assign(params,direction);torch.nn.utils.clip_grad_norm_(params,1.)
                before=loss.flatten(params);opt.step();actual_direction=before-loss.flatten(params)
                record['short'].append(dict(start=start,hops=hops,values=numbers(short_tasks),metrics=short_values,
                    short_full_cosine=((g*full).sum(1)/(g.norm(dim=1)*full.norm(dim=1)).clamp_min(1e-18)).cpu().tolist(),
                    original_3task_raw_alignment=old_loss.alignment(legacy,direction),
                    actual_update_full_alignment=old_loss.alignment(full,actual_direction)))
                loss.set_parameters(params,before)
            # 使用完整终点梯度，预览修复后真实位移；随后还原权重，不提交训练更新。
            opt=optimizer(b);before=loss.flatten(params)
            local_weights=(full.norm(dim=1).median()/full.norm(dim=1).clamp_min(1e-12)).clamp(.05,20.)
            evaluate=lambda:[numbers(obj.tasks(entry)[0])]
            repaired=guarded_step(b,opt,full[None],local_weights,evaluate,[record['full']])
            loss.set_parameters(params,before)
            record['repair_preview']=repaired;good.append(repaired['accepted'])
            rows.append(record);write(args.out/'diagnostics_partial.json',dict(updates=0,rows=rows,heldout_used=False))
            print('ENDPOINT_DIAG',label,p['case_id'],group,'replay',difference,'accepted',repaired['accepted'],flush=True)
    b.load_state_dict(base)
    median=torch.stack(norms).median(0).values
    weights=(median[0]/median.clamp_min(1e-12)).clamp(.05,20.)
    passed=all(good) and weights_digest(b)==initial and weights_digest(net)==frozen
    result=dict(passed=passed,updates_committed=0,rows=rows,task_order=loss.TASKS,weights=weights.tolist(),
                full_replay_pass=True,repair_previews_pass=all(good),initial_b_hash=initial,m0_hash=frozen,
                frozen_unchanged=weights_digest(net)==frozen,heldout_used=False,protected_roots=22,
                scope='four train-side strata probes at oldB400/B2step50; no claim of heldout generalization',
                protocol_sha256=sha(args.out/'protocol.json'))
    write(args.out/'diagnostics.json',result);archive(args)
    assert passed,'零训练诊断未通过，禁止启动小试'
    print('ENDPOINT_DIAGNOSTIC_PASS',flush=True)


def probe(args,spec,core,pipe,net,b,step,baseline=None):
    rows=[];dest=args.out/'training'/f'probe{step}';dest.mkdir(parents=True,exist_ok=True)
    for p,c,group in old.probe_definitions(spec,full_synthetic=True):
        picture,_,_=old.sample(args,core,pipe,net,p,c,b)
        picture.save(dest/f'{p["case_id"]}_{c}.png')
        row=old.score(args,p,c,picture)
        positive=pilot.rgb_tensor(np.asarray(Image.open(args.out/'cases'/p['case_id']/f'{c}_positive.png')),args.device)
        rgb=pilot.rgb_tensor(np.asarray(picture),args.device)
        _,weight=data.inputs(args,p,c,'raw');weight=torch.from_numpy(weight)[None,None].to(args.device)
        _,lum=loss.luminance(rgb,positive,weight,p['patch_coordinates'])
        rows.append(dict(case=p['case_id'],condition=c,group=group,scores=dict(row,**lum)))
    fields=('l_amplitude','l_correlation','correlation','old_projection','palette_swd','low_l')
    groups={g:{k:float(np.mean([r['scores'][k] for r in rows if r['group']==g and r['scores'][k] is not None]))
               for k in fields} for g in ('neutral','pastel','multi','synthetic')}
    bad=[]
    if baseline:
        for g,cur in groups.items():
            prior=baseline['groups'][g]
            if cur['l_correlation']<prior['l_correlation']-.02:bad.append(g+':L_correlation')
            if abs(cur['l_amplitude']-1)>abs(prior['l_amplitude']-1)+.05:bad.append(g+':L_amplitude')
            if cur['old_projection']>1.1*prior['old_projection']+.001:bad.append(g+':old_projection')
            if cur['palette_swd']>1.1*prior['palette_swd']+.1:bad.append(g+':palette')
            if cur['low_l']>prior['low_l']+.25:bad.append(g+':low_l')
        for r in rows:
            if r['group']!='synthetic':continue
            prior=next(x for x in baseline['rows'] if (x['case'],x['condition'])==(r['case'],r['condition']))
            if prior['scores']['texture']['hit'] and not r['scores']['texture']['hit']:bad.append(r['case']+':'+r['condition'])
    report=dict(step=step,rows=rows,groups=groups,stop_reasons=bad,heldout_used=False)
    write(dest/'result.json',report);print('ENDPOINT_PROBE',step,groups,bad,flush=True)
    return report


def train(args):
    spec=protocol(args);diag=read(args.out/'diagnostics.json')
    assert diag['passed'] and diag['protocol_sha256']==sha(args.out/'protocol.json')
    folder=args.out/'training';folder.mkdir(exist_ok=True)
    assert not (folder/'last.pt').exists(),'50步首段只运行一次；禁止自动覆盖或越过止损续跑'
    core,pipe,net,b,cache,obj,initial=load(args,spec)
    assert initial==diag['initial_b_hash'] and weights_digest(net)==diag['m0_hash']
    opt=optimizer(b);params=list(b.parameters());weights=torch.tensor(diag['weights'],device=args.device)
    old.atomic_torch(dict(step=0,b=b.state_dict()),folder/'step0.pt')
    baseline=probe(args,spec,core,pipe,net,b,0);history=[]
    for row in spec['schedule'][:spec['max_attempts']]:
        start=time.monotonic();p=next(p for p in spec['train'] if p['case_id']==row['case'])
        entries=[cache[p['case_id'],c] for c in row['conditions']]
        assert entries[0]['digest']==entries[1]['digest']
        # 第25步之前B未参与，缓存起点独立于B；其后25次调用全部保留梯度。
        gradients=[];prior=[]
        for entry in entries:
            tasks,_=obj.tasks(entry);prior.append(numbers(tasks));gradients.append(loss.task_gradients(tasks,params))
        gradients=torch.stack(gradients)
        evaluate=lambda:[numbers(obj.tasks(e)[0]) for e in entries]
        report=guarded_step(b,opt,gradients,weights,evaluate,prior)
        record=dict(row,**report,before=prior,seconds=time.monotonic()-start)
        history.append(record)
        payload=dict(step=row['step'],b=b.state_dict(),optimizer=opt.state_dict(),history=history,
                     protocol_sha256=sha(args.out/'protocol.json'),accepted_updates=sum(r['accepted'] for r in history))
        old.atomic_torch(payload,folder/'last.pt')
        with (folder/'loss.jsonl').open('a',encoding='utf8') as stream:stream.write(json.dumps(record)+'\n')
        print('ENDPOINT_UPDATE',row['step'],'accepted',report['accepted'],'seconds',record['seconds'],flush=True)
        reasons=[]
        if row['step'] in spec['checkpoint_steps']:
            old.atomic_torch(payload,folder/f'step{row["step"]}.pt')
            reasons+=probe(args,spec,core,pipe,net,b,row['step'],baseline)['stop_reasons']
        if len(history)>=10 and sum(not r['accepted'] for r in history[-10:])>=5:
            reasons.append('at_least_half_recent_attempts_rejected')
        if reasons:
            write(folder/'stopped.json',dict(attempts=len(history),accepted_updates=payload['accepted_updates'],
                                           reasons=reasons,heldout_used=False));archive(args)
            raise RuntimeError('训练侧止损；禁止继续或使用11身份调参')
    assert weights_digest(net)==diag['m0_hash']
    write(folder/'complete.json',dict(attempts=50,accepted_updates=sum(r['accepted'] for r in history),
        initial_b_hash=initial,frozen_unchanged=True,heldout_used=False,no_expansion=True,
        checkpoint_sha256=sha(folder/'step50.pt')))
    archive(args);print('ENDPOINT_PILOT_COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','diagnose','train','archive'))
    p.add_argument('--project',type=Path,default=pilot.PROJECT)
    p.add_argument('--data',type=Path,default=pilot.PROJECT.parent/'datasets/BF')
    p.add_argument('--previous',type=Path,default=pilot.PROJECT/'output_eval/e2_state_b_material_20261113')
    p.add_argument('--parent',type=Path,default=pilot.PROJECT/'output_eval/e2_state_b_balanced_20261114')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(2);torch.set_num_threads(2)
    globals()[args.action](args)


if __name__=='__main__':main()
