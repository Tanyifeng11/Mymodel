"""f68固定一次C0/C1微调：共享新鲜梯度，完整25步保护，末5步配对修正。"""
import argparse
import copy
import os
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from models.e2_state_texture_b import StateTextureB
from tools import e2_b_balanced_losses as balanced_loss
from tools import e2_b_endpoint_losses as legacy_loss
from tools import e2_state_b_balanced as balanced
from tools import e2_state_b_endpoint as endpoint
from tools import e2_state_b_garment as garment
from tools import e2_state_b_joint_data as data
from tools import e2_state_b_pilot as pilot
from tools import e2_state_b_repair as repair
from tools.e2_b_garment_projection import GarmentProjection
from tools.e2_b_pair_finetune_losses import projection, pair_coefficients, project_displacement
from tools.e2_state_b_amplitude_response import read, write, sha, pixels, CHECKPOINT_SHA
from tools.e2_oracle_texture import encode, weights_digest


LEGACY_FOCUS = ('validation_621', 'validation_1526')
ARMS = ('a075', 'a100', 'a125')


def prepare(args):
    assert not (args.out/'protocol.json').exists(), '不覆盖实验'
    source = read(args.amplitude_source/'protocol.json')
    prior = read(args.amplitude_source/'generated.json')
    old_path = args.project/'output_eval/e2_state_b_residue_search_20261115/protocol.json'
    old = read(old_path)
    material_path = args.project/'output_eval/e2_state_b_material_20261113/protocol.json'
    assert sha(args.checkpoint) == source['checkpoint_sha256'] == CHECKPOINT_SHA
    assert prior['frozen_unchanged'] and prior['training_updates'] == 0
    args.out.mkdir(parents=True, exist_ok=True)
    assets = {}
    for pair in source['train']:
        case = pair['case_id']
        destination = args.out/'cases'/case
        destination.mkdir(parents=True)
        for path in sorted((args.amplitude_source/'cases'/case).glob('*')):
            if path.is_file():
                shutil.copyfile(path, destination/path.name)
                assets[f'cases/{case}/{path.name}'] = sha(destination/path.name)
    assert len(source['inputs']) == 48
    protected = set(old['protected_product_roots'])
    assert not protected & {p['identity_group'] for p in source['train']+old['references']}
    spec = dict(source)
    spec.update(assets=assets, holdout=old['holdout'], protected_product_roots=sorted(protected),
        references=old['references'], replay_weights=old['replay_weights'],
        amplitude_source_result_sha256=sha(args.amplitude_source/'result.json'),
        amplitude_source_generated_sha256=sha(args.amplitude_source/'generated.json'),
        material_source_protocol_sha256=sha(material_path),
        legacy_protocol_sha256=sha(old_path), source_commit=subprocess.check_output(
            ['git','rev-parse','HEAD'],text=True).strip(),
        training=dict(attempts=1, optimizer='AdamW', lr=3e-5, weight_decay=.01, gradient_clip=1.,
            scale=.5, displacement_cap=.007854645, projection_sweeps=4096,
            legacy_focus=list(LEGACY_FOCUS), pair_gradient_norm_fraction=.25,
            extra_absolute_gradient_norm_fraction=.25,
            full_loss_window=[25,50], correction_window=[45,50],
            default_margin=.01, protected_amplitude_margin=.18,
            protected_condition=['validation_1641','r2692_native'],
            C0='original seven losses over all48 + existing absolute gain correction on621/1526 at alpha1',
            C1='C0 + four-pastel paired difference + absolute gain on remaining1641/870, no duplicate absolute term',
            budget='same shared full48 x seven gradients and all four late-pair diagnostics; each arm one reset AdamW proposal; no adaptive retries'),
        acceptance=dict(task_tolerance='1e-5+1e-4*abs(initial)', group_tolerance=1e-5,
            residue_group_tolerance=1e-8, require_mean_residue_descent=True,
            require_projection_feasible=True, require_nonzero_update=True, minimum_actual_cosine=-1e-5,
            holdout_group_tolerance=1e-5,
            innovation='pastel mean abs slope error and alpha1 gain error each decrease >1e-6 against both f68 and C0; correlation does not fall >1e-5',
            protect=['palette_swd','spatial_ab_mae_goal','low_l','old_l_projection','edge','l_edge'],
            policy='retain all fixed candidates for comparison; rejected candidates are not adopted; no holdout-driven tuning'),
        evaluation=dict(holdout_seeds=[0,100000],holdout_conditions=['real_a','real_b'],
            scope='11 training-isolated but previously viewed identities; static regression, not unseen slope generalization'),
        planned_optimizer_proposals_per_arm=1, heldout_used_for_training=False,
        intervention='same three reference amplitudes, C0/C1 fixed one projected B parameter proposal; frozen backbone and S0',
        interpretation='controlled one-step candidate comparison, acceptance determined after complete pixel evaluation')
    spec.pop('training_updates',None)
    write(args.out/'protocol.json',spec)
    print('PAIR_PREPARED',len(spec['inputs']),len(spec['holdout']),flush=True)


class Objective(garment.Objective):
    def __init__(self,args,pipe,predictor,b,pair,items):
        self.args,self.pipe,self.predictor,self.b = args,pipe,predictor,b
        self.records,self.projections = {},{}
        folder=args.out/'cases'/pair['case_id']
        mask=np.asarray(Image.open(folder/'mask.png'))>0
        anchor=pixels(folder/'train_anchor.png')
        for item in items:
            c=item['sample_condition']
            goal=pixels(folder/f'{c}_goal.png')
            with torch.no_grad():
                pipe.vae.half()
                z=encode(pipe.vae,Image.fromarray(goal),args.device)
                pipe.vae.float()
                positive=self.rgb(z).detach()
            png=np.rint(positive[0].permute(1,2,0).cpu().numpy()*255).astype('uint8')
            assert np.array_equal(png,pixels(folder/f'{c}_positive.png'))
            values,weight=data.inputs(args,pair,c,'raw')
            weight=torch.from_numpy(weight)[None,None].to(args.device)
            self.records[pair['case_id'],c]=(z,positive,pilot.rgb_tensor(goal,args.device),values,weight,None)
            self.projections[pair['case_id'],c]=GarmentProjection(png,anchor,mask,args.device)


def gradient_shard(args):
    started=time.monotonic()
    spec=read(args.out/'protocol.json')
    assert sha(args.checkpoint)==CHECKPOINT_SHA
    assert all(sha(args.out/p)==h for p,h in spec['assets'].items())
    pair=spec['train'][args.shard]
    items=[r for r in spec['inputs'] if r['case']==pair['case_id']]
    assert len(items)==12
    dest=args.out/'gradients'
    dest.mkdir(exist_ok=True)
    final=dest/f'shard{args.shard}.pt'
    assert not final.exists()
    runtime=copy.copy(args);runtime.action='train'
    core,pipe,net=pilot.load_core(runtime)
    torch.set_num_threads(min(2,int(os.environ.get('SLURM_CPUS_PER_TASK','2'))))
    b=StateTextureB(color_channels=2,reference_channels=192).to(args.device)
    b.load_state_dict(torch.load(args.checkpoint,map_location='cpu',weights_only=False)['b'])
    b.eval().requires_grad_(True)
    net.eval().requires_grad_(False)
    initial,frozen=weights_digest(b),weights_digest(net)
    assert frozen==read(args.amplitude_source/'generated.json')['net_hash']
    params=list(b.parameters())
    # 每个alpha完整生成独立前缀，B之前的状态可由两组共享。
    cache={}
    for item in items:
        c=item['sample_condition']
        picture,digest,tap,latent=endpoint.final_sample(args,core,pipe,net,pair,c,b)
        previous=args.amplitude_source/'images'/item['arm']/(item['case']+'_'+item['condition']+'.png')
        assert np.array_equal(np.asarray(picture),pixels(previous))
        assert len(tap.states)==25
        cache[c]=dict(pair=pair,condition=c,states=tap.states,digest=digest,final=latent.detach().cpu())
    pipe.vae.half()
    predictor=repair.FrozenPredictor(args,core,pipe,net,[pair],conditions=[r['sample_condition'] for r in items])
    obj=Objective(args,pipe,predictor,b,pair,items)
    pipe.scheduler.set_timesteps(50,device=args.device)
    gradients,values,late={},[],{}
    for item in items:
        c=item['sample_condition'];entry=cache[c]
        predicted,base=obj.rollout(entry)
        latent_error=float((predicted.detach()-entry['final'].to(args.device).float()).abs().max())
        assert latent_error<=2e-4, '完整反传轨迹不一致'
        tasks,_=obj.score_latents(entry,predicted,base,residue=obj.metric(entry).surrogate)
        gradients[c]=legacy_loss.task_gradients(tasks,params).cpu()
        expected_png=pixels(args.amplitude_source/'images'/item['arm']/(item['case']+'_'+item['condition']+'.png'))
        assert np.array_equal(obj.pixels(predicted.detach()),expected_png), '完整反传输出像素不一致'
        values.append(dict(item,tasks=endpoint.numbers(tasks),noise_digest=entry['digest'],
                           full_latent_max_error=latent_error,full_png_equal=True))
        assert torch.isfinite(gradients[c]).all()
        if item['group']=='pastel':
            # 旧绝对gain修正的相同末5步路径；不截短用于七项保护的完整路径。
            last=dict(entry,states=[entry['states'][20]])
            endpoint_z,_=obj.rollout(last,hops=5)
            assert torch.equal(endpoint_z.detach(),predicted.detach()), '末5步与完整展开不一致'
            _,positive,_,_,weight,_=obj.records[pair['case_id'],c]
            base_positive=obj.records[pair['case_id'],item['condition']+'_a100'][1]
            feature=projection(obj.rgb(endpoint_z),positive,base_positive,weight)
            gy=torch.autograd.grad(feature['y'],params,retain_graph=item['arm']=='a100',allow_unused=True)
            pack=lambda gs:torch.cat([torch.zeros_like(p).flatten() if g is None else g.detach().flatten()
                                    for p,g in zip(params,gs)]).cpu()
            row=dict(features={k:float(v.detach()) for k,v in feature.items()},gradient_y=pack(gy))
            if item['arm']=='a100':
                ga=torch.autograd.grad((feature['g']-1).square(),params,allow_unused=True)
                row['gradient_absolute']=pack(ga)
            late[item['arm']]=row
        del predicted,base,tasks
        print('PAIR_GRADIENT',args.shard,c,flush=True)
    coefficient=pair_coefficients(*(torch.tensor(late[a]['features'][k]) for a,k in
                                  (('a075','y'),('a125','y'),('a075','x'),('a125','x'))))
    assert late['a125']['features']['x']-late['a075']['features']['x']>1e-6
    pair_gradient=coefficient['low']*late['a075']['gradient_y']+coefficient['high']*late['a125']['gradient_y']
    assert torch.isfinite(pair_gradient).all() and pair_gradient.norm()>0
    assert weights_digest(b)==initial and weights_digest(net)==frozen
    payload=dict(shard=args.shard,case=pair['case_id'],items=items,values=values,
        gradients=torch.stack([gradients[r['sample_condition']] for r in items]),
        absolute_gradient=late['a100']['gradient_absolute'],pair_gradient=pair_gradient,
        late_features={a:r['features'] for a,r in late.items()},
        pair_loss=float(coefficient['pair']),b_hash=initial,net_hash=frozen,
        checkpoint_sha256=CHECKPOINT_SHA,protocol_sha256=sha(args.out/'protocol.json'),
        frozen_unchanged=True,training_updates=0,baseline_parity=12,seconds=time.monotonic()-started)
    balanced.atomic_torch(payload,final)
    write(dest/f'shard{args.shard}.json',{k:v for k,v in payload.items()
        if k not in ('gradients','absolute_gradient','pair_gradient')})
    print('PAIR_GRADIENT_COMPLETE',args.shard,payload['seconds'],flush=True)


def propose(args):
    spec=read(args.out/'protocol.json');cfg=spec['training']
    shards=[torch.load(args.out/'gradients'/f'shard{i}.pt',map_location='cpu',weights_only=False) for i in range(4)]
    assert all(s['protocol_sha256']==sha(args.out/'protocol.json') and s['frozen_unchanged'] for s in shards)
    assert len({s['b_hash'] for s in shards})==len({s['net_hash'] for s in shards})==1
    gradients=torch.cat([s['gradients'] for s in shards])
    rows=[r for s in shards for r in s['items']]
    assert gradients.shape[:2]==(48,7)
    weights=torch.tensor(spec['replay_weights'])
    merged=(gradients.mean(0)*weights[:,None]).sum(0)
    old_abs=torch.stack([s['absolute_gradient'] for s in shards if s['case'] in LEGACY_FOCUS]).mean(0)
    new_abs=torch.stack([s['absolute_gradient'] for s in shards if s['case'] not in LEGACY_FOCUS]).mean(0)
    pair=torch.stack([s['pair_gradient'] for s in shards]).mean(0)
    normalized=lambda x:x*(merged.norm()/x.norm().clamp_min(1e-12))
    directions=dict(C0=merged+normalized(old_abs),
        C1=merged+normalized(old_abs)+cfg['pair_gradient_norm_fraction']*normalized(pair)
            +cfg['extra_absolute_gradient_norm_fraction']*normalized(new_abs))
    constraints=gradients.flatten(0,1)
    margins=torch.full((len(constraints),),cfg['default_margin'])
    for i,r in enumerate(rows):
        if [r['case'],r['condition']]==cfg['protected_condition']:
            margins[i*7+legacy_loss.TASKS.index('amplitude')]=cfg['protected_amplitude_margin']
    folder=args.out/'training';folder.mkdir(exist_ok=True)
    summaries={}
    for arm,direction in directions.items():
        assert not (folder/f'{arm}.pt').exists()
        b=StateTextureB(color_channels=2,reference_channels=192)
        b.load_state_dict(torch.load(args.checkpoint,map_location='cpu',weights_only=False)['b'])
        parameters=list(b.parameters());start=legacy_loss.flatten(parameters)
        optimizer=endpoint.optimizer(b)
        balanced_loss.assign(parameters,direction)
        torch.nn.utils.clip_grad_norm_(parameters,1.)
        optimizer.step()
        raw=start-legacy_loss.flatten(parameters)
        delta,projection_report=project_displacement(raw,constraints,margins,max_sweeps=cfg['projection_sweeps'])
        # 同一固定规则：不增加投影后的总步长，不通过评测挑选scale。
        norm_scale=min(1.,float(raw.norm()/delta.norm().clamp_min(1e-18)))
        delta=delta*norm_scale
        delta=delta*cfg['scale']
        cap_scale=min(1.,cfg['displacement_cap']/float(delta.norm().clamp_min(1e-18)))
        delta=delta*cap_scale
        legacy_loss.set_parameters(parameters,start-delta)
        actual=start-legacy_loss.flatten(parameters)
        changed=weights_digest(b)!=shards[0]['b_hash']
        report=dict(arm=arm,optimizer_proposals=1,parameter_updates=int(changed),
            adopted=False,acceptance='pending_complete_evaluation',projection=projection_report,
            raw_norm=float(raw.norm()),actual_norm=float(actual.norm()),
            norm_scale=norm_scale,fixed_scale=cfg['scale'],cap_scale=cap_scale,
            minimum_cosine=min(balanced_loss.alignment(constraints,actual)) if changed else None,
            checkpoint_start=CHECKPOINT_SHA,initial_b_hash=shards[0]['b_hash'],final_b_hash=weights_digest(b),
            gradient_norm=float(direction.norm()),protocol_sha256=sha(args.out/'protocol.json'))
        balanced.atomic_torch(dict(b=b.state_dict(),report=report),folder/f'{arm}.pt')
        report['checkpoint_sha256']=sha(folder/f'{arm}.pt')
        summaries[arm]=report
    write(folder/'proposals.json',dict(arms=summaries,shared_gradients=True,conditions=48,
        full_task_reverse_passes=336,late_y_reverse_passes=12,late_abs_reverse_passes=4,
        gradient_seconds=sum(s['seconds'] for s in shards),frozen_unchanged=True,
        heldout_used_for_training=False,gradient_norms=dict(original=float(merged.norm()),
            legacy_absolute=float(old_abs.norm()),new_absolute=float(new_abs.norm()),pair=float(pair.norm()))))
    print('PAIR_PROPOSED',summaries,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','gradient','propose'))
    parser.add_argument('--project',type=Path,default=Path('/share/home/u2515283058/Mymodel'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    args.data=args.project.parent/'datasets/BF'
    args.previous=args.project/'output_eval/e2_state_b_material_20261113'
    args.amplitude_source=args.project/'output_eval/e2_state_b_amplitude_response_20261116'
    args.checkpoint=args.project/'output_eval/e2_state_b_pastel_followup_20261115/trust_retry/training/step1.pt'
    threads=min(2,int(os.environ.get('SLURM_CPUS_PER_TASK','2')))
    torch.set_num_threads(threads);cv2.setNumThreads(threads)
    {'prepare':prepare,'gradient':gradient_shard,'propose':propose}[args.action](args)


if __name__=='__main__':
    main()
