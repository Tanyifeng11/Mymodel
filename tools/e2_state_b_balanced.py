"""训练侧闭环诊断后，旧B400起点的A2/B2单因素400步对照。"""
import argparse
import copy
import gc
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

from tools import e2_b_balanced_data as data
from tools import e2_b_balanced_losses as loss
from tools import e2_state_b_joint as joint
from tools import e2_state_b_joint_data as old_data
from tools import e2_state_b_material as material
from tools import e2_state_b_material_report as metrics
from tools import e2_state_b_pilot as pilot
from tools import e2_state_b_repair as repair
from tools.e2_b_material_losses import material_target
from tools.e2_fft_experiment import SOURCE, read, write, sha
from tools.e2_oracle_texture import encode, decode, weights_digest


def source_hashes():
    names=set(joint.source_hashes()) | {'tools/e2_b_balanced_data.py','tools/e2_b_balanced_losses.py',
                                      'tools/e2_state_b_balanced.py','scripts/e2_state_b_balanced.sbatch','tests/test_e2_b_balanced.py'}
    return {name:sha(SOURCE/name) for name in sorted(names)}


def protocol(args, final=False):
    spec=read(args.out/('protocol.json' if final else 'input_protocol.json'))
    assert spec['code']==source_hashes()
    return spec


def atomic_torch(payload,path):
    temporary=path.with_suffix(path.suffix+'.tmp');torch.save(payload,temporary);temporary.replace(path)


def sample(args,core,pipe,net,p,c,b,mode='raw',capture=False,snapshots=False):
    was_training=b.training;vae_dtype=pipe.vae.dtype;b.eval();pipe.vae.half()
    try:
        with torch.no_grad():return joint.sample(args,core,pipe,net,p,c,b,mode,capture,snapshots)
    finally:
        pipe.vae.to(dtype=vae_dtype);b.train(was_training)


def baseline_cache(args,spec,core,pipe,net,b):
    path=args.out/'base_cache.pt'
    cache=torch.load(path,map_location='cpu',weights_only=False) if path.exists() else {}
    for p in spec['train']:
        d=args.out/'cases'/p['case_id']
        for c in spec['conditions']:
            key=p['case_id'],c
            if key in cache:
                assert sha(d/f'{c}_baseline.png')==cache[key]['output_sha256'];continue
            picture,digest,tap=sample(args,core,pipe,net,p,c,b,capture=True)
            picture.save(d/f'{c}_baseline.png')
            cache[key]=dict(pair=p,condition=c,digest=digest,states=tap.states,output_sha256=sha(d/f'{c}_baseline.png'))
            assert len(tap.states)==25
        atomic_torch(cache,path)
        print('BALANCED_BASE_CACHE',p['case_id'],len(cache),flush=True)
    assert len(cache)==len(spec['train'])*len(spec['conditions'])
    return cache


class Objective(material.Objective):
    def __init__(self,args,spec,pipe,predictor,b,save_targets=False):
        self.args,self.pipe,self.predictor,self.b=args,pipe,predictor,b
        self.records,self.targets={},[]
        for p in spec['train']:
            d=args.out/'cases'/p['case_id'];mask=np.asarray(Image.open(d/'mask.png'))>0
            anchor=metrics.pixels(d/'train_anchor.png');audit={}
            for c in spec['conditions']:
                goal,info=material_target(anchor,metrics.pixels(d/f'{c}_ref.png'),mask)
                if save_targets:Image.fromarray(goal).save(d/f'{c}_goal.png')
                else:assert np.array_equal(goal,metrics.pixels(d/f'{c}_goal.png'))
                with torch.no_grad():
                    pipe.vae.half();z=encode(pipe.vae,Image.fromarray(goal),args.device);pipe.vae.float()
                    positive=self.rgb(z).detach()
                positive_image=Image.fromarray(np.rint(positive[0].permute(1,2,0).cpu().numpy()*255).astype('uint8'))
                if save_targets:positive_image.save(d/f'{c}_positive.png')
                else:assert np.array_equal(np.asarray(positive_image),metrics.pixels(d/f'{c}_positive.png'))
                values,w=old_data.inputs(args,p,c,'raw')
                weight=torch.from_numpy(w)[None,None].to(args.device)
                baseline=pilot.rgb_tensor(metrics.pixels(d/f'{c}_baseline.png'),args.device)
                template=loss.templates(baseline,positive,p['patch_coordinates'])
                original=pilot.rgb_tensor(goal,args.device)
                self.records[p['case_id'],c]=z,positive,original,values,weight,template
                zero=loss.terms(positive,positive,original,positive,weight,template,p['patch_coordinates'])
                assert max(float(v) for v in zero.values())<1e-6,(p['case_id'],c)
                info.update(vae=metrics.aligned(np.asarray(positive_image),goal,np.asarray(positive_image),anchor,mask),
                            target_sha256=sha(d/f'{c}_goal.png'),positive_sha256=sha(d/f'{c}_positive.png'))
                audit[c]=info
            self.targets.append(dict(case=p['case_id'],conditions=audit))
            print('BALANCED_TARGETS',p['case_id'],len(audit),flush=True)

    def view(self,entry,state_index,hops):
        p,c=entry['pair'],entry['condition']
        target,positive,original,values,weight,template=self.records[p['case_id'],c]
        state=entry['states'][state_index]
        z,epsilon=state['zt'].to(self.args.device),state['epsilon'].to(self.args.device)
        for hop in range(hops):
            index=state['index']+hop;t=torch.tensor(state['t']-20*hop,device=self.args.device)
            alpha=self.pipe.scheduler.alphas_cumprod[int(t)].to(z.device).float()
            x0=(z.float()-(1-alpha).sqrt()*epsilon.float())/alpha.sqrt()
            delta=self.b(z.float(),x0,timestep=t.reshape(1),**values)
            corrected=(epsilon.float()-(alpha/(1-alpha)).sqrt()*delta).to(epsilon.dtype)
            if hop+1<hops:
                z=self.pipe.scheduler.step(corrected,t,z,return_dict=False)[0]
                epsilon=self.predictor.epsilon(z,t-20,entry)
                continue
            # 只在短展开末端监督；末次调用与真实DDIM最终输出一致。
            predicted=self.pipe.scheduler.step(corrected,t,z,return_dict=False)[0].float() if index==49 else x0+delta
            with torch.no_grad():
                base=self.pipe.scheduler.step(epsilon,t,z.detach(),return_dict=False)[0].float() if index==49 else x0.detach()
                base=self.rgb(base)
            values_=loss.terms(self.rgb(predicted),positive,original,base,weight,template,p['patch_coordinates'])
            values_['latent']=pilot.weighted_mean((predicted-target).square(),values['mask'])
            return loss.groups(values_),values_


def probe_definitions(spec,full_synthetic=False):
    probes=list(joint.probes(spec))
    for j,p in enumerate(spec['train']):
        for c in data.SYNTHETIC if full_synthetic else (('v16','h32')[j%2],):
            probes.append((p,c,'synthetic'))
    return probes


def score(args,p,c,picture):
    d=args.out/'cases'/p['case_id'];mask=np.asarray(Image.open(d/'mask.png'))>0
    row=metrics.aligned(np.asarray(picture),metrics.pixels(d/f'{c}_goal.png'),metrics.pixels(d/f'{c}_positive.png'),
                        metrics.pixels(d/'train_anchor.png'),mask)
    current=metrics.lab(np.asarray(picture))/100
    target=metrics.lab(metrics.pixels(d/f'{c}_positive.png'))/100
    inside=cv2.distanceTransform(mask.astype('uint8'),cv2.DIST_L2,5)>=16
    edge=lambda x:np.concatenate((np.pad(np.diff(x,axis=1),((0,0),(0,1),(0,0))),
                                 np.pad(np.diff(x,axis=0),((0,1),(0,0),(0,0)))),axis=2)
    row['edge']=float(np.abs(edge(current-target))[inside].mean())
    if c in data.SYNTHETIC:row['texture']=pilot.texture_metrics(np.asarray(picture),p['patch_coordinates'],c)
    return row


def checkpoint_probes(args,spec,core,pipe,net,b,step,baseline=None):
    rows=[];dest=args.out/'training'/args.arm/f'probe{step}';dest.mkdir(parents=True,exist_ok=True)
    with torch.no_grad():
        for p,c,group in probe_definitions(spec,full_synthetic=True):
            picture,digest,_=sample(args,core,pipe,net,p,c,b)
            name=p['case_id']+'_'+c;picture.save(dest/f'{name}.png')
            row=dict(case=p['case_id'],condition=c,group=group,digest=digest,scores=score(args,p,c,picture))
            if baseline is not None:
                assert row['digest']==next(v['digest'] for v in baseline if v['case']==p['case_id'] and v['condition']==c)
            rows.append(row)
    groups={}
    for group in ('neutral','pastel','multi','synthetic'):
        rr=[v['scores'] for v in rows if v['group']==group]
        groups[group]={k:float(np.mean([r[k] for r in rr if r[k] is not None])) for k in ('correlation','palette_swd','old_projection','edge','low_l')}
    bad=[]
    if baseline is not None:
        for group,current in groups.items():
            rr=[r['scores'] for r in baseline if r['group']==group]
            base={k:float(np.mean([r[k] for r in rr if r[k] is not None])) for k in current}
            # 预先固定的训练内止损，不读取11身份结果。
            if current['correlation']<base['correlation']-.02:
                bad.append(group+':correlation_drop_gt_.02')
            if current['old_projection']>1.1*base['old_projection']+.001 and current['edge']>1.1*base['edge']:
                bad.append(group+':both_residue_errors_up_gt_10pct')
        for row in rows:
            if row['group']!='synthetic':continue
            old=next(r for r in baseline if r['case']==row['case'] and r['condition']==row['condition'])
            if old['scores']['texture']['hit'] and not row['scores']['texture']['hit']:
                bad.append(row['case']+':lost_'+row['condition'])
    report=dict(step=step,rows=rows,groups=groups,stop_reasons=bad,heldout_used=False)
    write(dest/'result.json',report)
    print('BALANCED_PROBE',args.arm,step,groups,bad,flush=True)
    return report


def diagnostics_archive(args):
    from tools.e2_state_b_joint_report import archive
    files=[args.out/n for n in ('input_protocol.json','protocol.json','diagnostics.json','target_controls.json','training_reference_panel.jpg')]
    files+=list((args.out/'diagnostic_panels').glob('*'))
    archive(args.out,'balanced_diagnostics_review.zip',files)


def diagnose(args):
    spec=protocol(args);args.action='train'
    core,pipe,net=pilot.load_core(args);frozen=weights_digest(net)
    b=old_data.initialize(args).eval();initial=weights_digest(b)
    cache=baseline_cache(args,spec,core,pipe,net,b)
    pipe.vae.half()  # FrozenPredictor编码的原草图与冻结M0条件均为half。
    predictor=repair.FrozenPredictor(args,core,pipe,net,spec['train'],conditions=spec['conditions'])
    objective=Objective(args,spec,pipe,predictor,b,save_targets=True)
    pipe.scheduler.set_timesteps(spec['sampler']['steps'],device=args.device)
    with torch.no_grad():
        for p in spec['train']:
            e=cache[p['case_id'],spec['conditions'][0]];state=e['states'][0]
            replay=predictor.epsilon(state['zt'].to(args.device),torch.tensor(state['t'],device=args.device),e)
            assert torch.equal(replay.cpu(),state['epsilon'])
    write(args.out/'target_controls.json',dict(updates=0,targets=objective.targets))
    trajectory=[];folder=args.out/'diagnostic_panels';folder.mkdir(exist_ok=True)
    states={}
    for a in ('A','B'):
        path=args.joint/'training'/a/'step400.pt'
        assert sha(path)==read(path.parent/'complete.json')['checkpoint_sha256']['400']
        states[a]=torch.load(path,map_location='cpu',weights_only=False)['b']
    base_state=copy.deepcopy(b.state_dict())
    for p,c,group in list(joint.probes(spec)):
        images={};scores={}
        for label,state,mode in (('base',base_state,'raw'),('A',states['A'],'raw'),('B',states['B'],'outline')):
            b.load_state_dict(state)
            picture,digest,tap=sample(args,core,pipe,net,p,c,b,mode,snapshots=True)
            tap.pictures['final']=picture;scores[label]={}
            for name in ('25_after','34_after','44_after','45_before','49_after','final'):
                im=tap.pictures[name];scores[label][name]=score(args,p,c,im);images[label,name]=im
        board=Image.new('RGB',(6*124,3*188),'white');draw=ImageDraw.Draw(board)
        for y,label in enumerate(('base','A','B')):
            for x,name in enumerate(('25_after','34_after','44_after','45_before','49_after','final')):
                draw.text((x*124+2,y*188+2),label+' '+name,fill='black')
                board.paste(images[label,name].resize((120,160)),(x*124,y*188+23))
        board.save(folder/(p['case_id']+'_'+c+'.jpg'),quality=95)
        trajectory.append(dict(case=p['case_id'],condition=c,group=group,scores=scores))
        print('BALANCED_TRAJECTORY',p['case_id'],c,flush=True)
    b.load_state_dict(base_state);parameters=list(b.parameters());gradients=[];raw=[]
    for p,c,group in probe_definitions(spec):
        for index,hops in ((0,2),(9,2),(21,4)):
            losses,values=objective.view(cache[p['case_id'],c],index,hops)
            g=loss.task_gradients(losses,parameters);assert torch.isfinite(g).all() and (g.norm(dim=1)>0).all()
            gradients.append(g.cpu());raw.append(dict(case=p['case_id'],condition=c,group=group,state_index=index,hops=hops,
                                                      losses={k:float(v.detach()) for k,v in losses.items()},gram=(g@g.T).cpu().tolist()))
        print('BALANCED_GRADIENT',p['case_id'],c,flush=True)
    norms=torch.stack([g.norm(dim=1) for g in gradients]).median(0).values
    weights=(norms[0]/norms.clamp_min(1e-12)).clamp(.05,20.)
    alignments={}
    for arm in ('A','B'):
        aligned=[]
        for i,g in enumerate(gradients):
            v=loss.merge(g,weights,arm=='B',20261114+i)
            a=loss.alignment(g,v);raw[i][arm+'_alignment']=a;aligned.append(a)
        aligned=np.asarray(aligned)
        alignments[arm]={group:dict(zip(loss.TASKS,np.median(aligned[[i for i,r in enumerate(raw) if r['group']==group]],axis=0).tolist()))
                         for group in ('neutral','pastel','multi','synthetic')}
    controls=[v['vae']['correlation'] for p in objective.targets for v in p['conditions'].values()]
    gradient_ok=all(value>=-1e-6 for group in alignments['B'].values() for value in group.values())
    passed=min(controls)>=.5 and gradient_ok and weights_digest(net)==frozen and weights_digest(b)==initial
    result=dict(updates=0,passed=passed,positive_controls_pass=min(controls)>=.5,minimum_vae_correlation=min(controls),
                gradient_alignment_pass=gradient_ok,alignment=alignments,gradients=raw,weights=dict(zip(loss.TASKS,weights.tolist())),
                trajectories=trajectory,frozen_unchanged=weights_digest(net)==frozen,m0_hash=frozen,initial_b_hash=initial,
                input_protocol_sha256=sha(args.out/'input_protocol.json'),base_cache_sha256=sha(args.out/'base_cache.pt'),
                gate='B2 all4 strata x all3 task median cosine >=0; A2 is the unprojected control',
                protected_roots=22,heldout_used=False)
    write(args.out/'diagnostics.json',result)
    if passed:
        final=copy.deepcopy(spec);final.update(diagnostics_sha256=sha(args.out/'diagnostics.json'),task_weights=result['weights'])
        write(args.out/'protocol.json',final)
    diagnostics_archive(args)
    assert passed,'训练侧零更新诊断未通过，禁止启动A2/B2训练'
    print('BALANCED_DIAGNOSTIC_PASS',alignments,flush=True)


def train(args):
    spec=protocol(args,True);diag=read(args.out/'diagnostics.json')
    assert diag['passed'] and sha(args.out/'diagnostics.json')==spec['diagnostics_sha256']
    assert sha(args.out/'base_cache.pt')==diag['base_cache_sha256']
    folder=args.out/'training'/args.arm;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():
        assert read(folder/'complete.json')['updates']==400;print('BALANCED_ALREADY_COMPLETE',args.arm);return
    assert not (folder/'stopped.json').exists(),'训练内止损已触发，需先分析，禁止自动续跑'
    core,pipe,net=pilot.load_core(args);frozen=weights_digest(net)
    assert frozen==diag['m0_hash']
    b=old_data.initialize(args);initial=weights_digest(b);assert initial==diag['initial_b_hash']
    cache=torch.load(args.out/'base_cache.pt',map_location='cpu',weights_only=False)
    pipe.vae.half()
    predictor=repair.FrozenPredictor(args,core,pipe,net,spec['train'],conditions=spec['conditions'])
    objective=Objective(args,spec,pipe,predictor,b)
    pipe.scheduler.set_timesteps(spec['sampler']['steps'],device=args.device)
    parameters=list(b.parameters());weights=torch.tensor([spec['task_weights'][k] for k in loss.TASKS],device=args.device)
    optimizer=torch.optim.AdamW(parameters,lr=3e-5,weight_decay=.01);history=[]
    if (folder/'last.pt').exists():
        saved=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        assert saved['protocol_sha256']==sha(args.out/'protocol.json') and saved['arm']==args.arm
        b.load_state_dict(saved['b']);optimizer.load_state_dict(saved['optimizer']);history=saved['history']
        assert len(history)==saved['step']
    else:
        atomic_torch(dict(step=0,b=b.state_dict()),folder/'step0.pt')
    if (folder/'probe0/result.json').exists():baseline=read(folder/'probe0/result.json')['rows']
    else:baseline=checkpoint_probes(args,spec,core,pipe,net,b,0)['rows']
    if history and len(history) in (50,100,200,400):
        # 中断若发生在检查点探针阶段，先补完该检查，不能跳过门槛继续更新。
        step=len(history);path=folder/f'probe{step}/result.json'
        review=read(path) if path.exists() else checkpoint_probes(args,spec,core,pipe,net,b,step,baseline)
        if review['stop_reasons']:
            write(folder/'stopped.json',dict(updates=step,reasons=review['stop_reasons'],heldout_used=False))
            raise RuntimeError('恢复后训练内止损触发，禁止继续')
    (folder/'loss.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history),encoding='utf8')
    for row in spec['schedule'][len(history):]:
        start=time.monotonic();p=next(p for p in spec['train'] if p['case_id']==row['case'])
        entries=[]
        for c in row['conditions']:
            e=cache[p['case_id'],c]
            if row['online']:
                picture,digest,tap=sample(args,core,pipe,net,p,c,b,capture=True)
                assert digest==e['digest'];e=dict(e,states=tap.states)
            entries.append(e)
        assert entries[0]['digest']==entries[1]['digest']
        b.train();optimizer.zero_grad(set_to_none=True);g=None;values=dict.fromkeys(loss.TASKS,0.)
        for e in entries:
            tasks,_=objective.view(e,row['state_index'],row['hops'])
            current=loss.task_gradients(tasks,parameters)
            g=current/2 if g is None else g+current/2
            for k in loss.TASKS:values[k]+=float(tasks[k].detach())/2
        direction=loss.merge(g,weights,args.arm=='B',20261114+row['step'])
        assert torch.isfinite(direction).all()
        loss.assign(parameters,direction);grad=torch.nn.utils.clip_grad_norm_(parameters,1.);assert torch.isfinite(grad)
        before=torch.cat([p.detach().flatten() for p in parameters]).clone();optimizer.step()
        after=torch.cat([p.detach().flatten() for p in parameters])
        record=dict(row,loss=values,grad=float(grad),alignment=loss.alignment(g,direction),
                    adam_alignment=loss.alignment(g,before-after),seconds=time.monotonic()-start)
        history.append(record)
        payload=dict(step=row['step'],arm=args.arm,b=b.state_dict(),optimizer=optimizer.state_dict(),history=history,
                     protocol_sha256=sha(args.out/'protocol.json'))
        # 每次更新保存参数和优化器，作业中断可精确继续，不重跑已保存的预算。
        atomic_torch(payload,folder/'last.pt')
        with (folder/'loss.jsonl').open('a',encoding='utf8') as stream:stream.write(json.dumps(record)+'\n')
        if row['step']%20==0:print('BALANCED_UPDATE',args.arm,row['step'],values,record['adam_alignment'],flush=True)
        if row['step'] in (50,100,200,400):
            atomic_torch(payload,folder/f"step{row['step']}.pt")
            review=checkpoint_probes(args,spec,core,pipe,net,b,row['step'],baseline)
            if review['stop_reasons']:
                write(folder/'stopped.json',dict(updates=row['step'],reasons=review['stop_reasons'],heldout_used=False))
                raise RuntimeError('训练内止损触发；禁止自动进入11身份验证')
    assert len(history)==400 and weights_digest(net)==frozen
    assert (folder/'probe400/result.json').exists() and not read(folder/'probe400/result.json')['stop_reasons']
    write(folder/'complete.json',dict(updates=400,arm=args.arm,frozen_unchanged=True,m0_hash=frozen,initial_b_hash=initial,
          protocol_sha256=sha(args.out/'protocol.json'),schedule=spec['schedule'],
          checkpoint_sha256={str(i):sha(folder/f'step{i}.pt') for i in (0,50,100,200,400)},
          weights=spec['task_weights'],gradient_rule='PCGrad' if args.arm=='B' else 'weighted sum'))
    print('BALANCED_TRAIN_COMPLETE',args.arm,400,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','diagnose','train','generate','summarize','evaluate'))
    parser.add_argument('--project',type=Path,default=pilot.PROJECT)
    parser.add_argument('--data',type=Path,default=pilot.PROJECT.parent/'datasets/BF')
    parser.add_argument('--previous',type=Path,default=pilot.PROJECT/'output_eval/e2_state_b_material_20261113')
    parser.add_argument('--joint',type=Path,default=pilot.PROJECT/'output_eval/e2_state_b_joint_recrop_20261113')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--arm',choices=('A','B'),default='A')
    parser.add_argument('--shard',type=int,default=0)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(2);torch.set_num_threads(2)
    if args.action=='prepare':data.prepare(args,source_hashes())
    elif args.action in ('diagnose','train'):globals()[args.action](args)
    else:
        from tools import e2_state_b_joint_report as report
        getattr(report,args.action)(args,protocol(args,True))


if __name__=='__main__':main()
