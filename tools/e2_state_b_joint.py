"""先做训练集零更新诊断，再从同一空间色度B400开展两臂各400步。"""
import argparse
import copy
import gc
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from tools import e2_state_b_joint_data as data
from tools import e2_b_joint_losses as improved
from tools import e2_state_b_material as material
from tools import e2_state_b_material_report as measurements
from tools import e2_state_b_repair as repair
from tools import e2_state_b_pilot as pilot
from tools.e2_b_material_losses import material_target, loss_terms as old_terms, old_template
from tools.e2_fft_experiment import SOURCE, read, write, sha
from tools.e2_oracle_texture import encode, decode, image, projected, weights_digest

KEYS = tuple(material.WEIGHTS)


def source_hashes():
    files = list(material.source_hashes()) + ['tools/e2_state_b_joint.py', 'tools/e2_state_b_joint_data.py',
        'tools/e2_b_joint_losses.py', 'tools/e2_state_b_joint_report.py', 'configs/e2_b_joint_refs.json',
        'scripts/e2_state_b_joint.sbatch']
    return {name: sha(SOURCE / name) for name in files}


def protocol(args, final=False):
    spec = read(args.out / ('protocol.json' if final else 'input_protocol.json'))
    assert spec['code'] == source_hashes()
    return spec


def sample(args, core, pipe, net, pair, condition, b=None, mode='raw', capture=False, snapshots=False):
    values, _ = data.inputs(args, pair, condition, mode)
    d = args.out / 'cases' / pair['case_id']
    assert sha(d / (condition + '_ref.png')) == pair['reference_hashes'][condition]
    features = projected(net, core.tensor(args.data / pair['sketch']).to(args.device).half(),
                         core.tensor(d / (condition + '_ref.png')).to(args.device).half())
    class Tap(repair.Tap):
        def step(self, epsilon, timestep, z, *a, **kw):
            index = self.index
            keep = snapshots and index in (25, 34, 44, 45, 49)
            if keep:
                alpha = self.pipe.scheduler.alphas_cumprod[int(timestep)].to(z.device).float()
                x0 = (z.float() - (1-alpha).sqrt()*epsilon.float()) / alpha.sqrt()
                self.pictures[f'{index}_before'] = decode(self.pipe.vae, x0)
                if b is not None:
                    delta = b(z.float(), x0, timestep=timestep.reshape(1), **values)
                    self.pictures[f'{index}_after'] = decode(self.pipe.vae, x0+delta)
            return super().step(epsilon, timestep, z, *a, **kw)
    with Tap(pipe, b, values, end=50, capture=capture) as tap:
        picture, digest = image(args, core, pipe, net, pair, features)
    assert tap.index == 50
    return picture, digest, tap


def cache(args, spec, core, pipe, net, b=None, mode='raw', original=None, reusable=None):
    entries = {}
    reusable = {} if reusable is None else reusable
    with torch.no_grad():
        for p in spec['train']:
            for c in spec['conditions']:
                key = p['case_id'], c
                if key in reusable:
                    entries[key] = dict(reusable[key], pair=p)
                    continue
                picture, digest, tap = sample(args, core, pipe, net, p, c, b, mode, capture=True)
                entries[key] = dict(pair=p, condition=c, digest=digest, states=tap.states)
                assert len(tap.states) == 25
                if original is None and c == spec['conditions'][0]:
                    picture.save(args.out / 'cases' / p['case_id'] / 'train_anchor.png')
                if original is not None:
                    assert digest == original[p['case_id'], c]['digest']
            print('JOINT_CACHE', mode, p['case_id'], flush=True)
    return entries


def reusable_trajectories(args, spec):
    if args.reuse_cache_from is None:
        return {}, None
    previous = args.reuse_cache_from
    old = read(previous / 'input_protocol.json')
    cache_path = previous / 'm0_cache.pt'
    assert old['sampler'] == spec['sampler'] and old['holdout'] == spec['holdout']
    assert old['conditions'] == spec['conditions']
    cached = torch.load(cache_path, map_location='cpu', weights_only=False)
    keep = {}
    anchors = {}
    for p, prior in zip(spec['train'], old['train']):
        assert {k:v for k,v in p.items() if k!='reference_hashes'} == {
            k:v for k,v in prior.items() if k!='reference_hashes'}
        for c in spec['conditions']:
            if p['reference_hashes'][c] == prior['reference_hashes'][c]:
                key = p['case_id'], c
                assert len(cached[key]['states']) == 25
                keep[key] = cached[key]
        # 训练锚点取相同的第一项参考，本轮只换了另外两个来源的裁片。
        assert (p['case_id'], spec['conditions'][0]) in keep
        name = p['case_id']
        source = previous / 'cases' / name / 'train_anchor.png'
        target = args.out / 'cases' / name / 'train_anchor.png'
        shutil.copyfile(source, target)
        anchors[name] = sha(target)
    return keep, dict(source_protocol_sha256=sha(previous/'input_protocol.json'),
                      source_cache_sha256=sha(cache_path), reused=len(keep), anchor_sha256=anchors)


class Objective(material.Objective):
    def __init__(self, args, spec, pipe, predictor, b, arm):
        self.args, self.pipe, self.predictor, self.b, self.arm = args, pipe, predictor, b, arm
        self.records, self.targets = {}, []
        for p in spec['train']:
            d = args.out / 'cases' / p['case_id']
            mask = np.asarray(Image.open(d / 'mask.png')) > 0
            anchor = np.asarray(Image.open(d / 'train_anchor.png').convert('RGB'))
            anchor_rgb = pilot.rgb_tensor(anchor, args.device)
            audits = {}
            for c in spec['conditions']:
                ref = np.asarray(Image.open(d / (c + '_ref.png')).convert('RGB'))
                goal, audit = material_target(anchor, ref, mask)
                Image.fromarray(goal).save(d / (c + '_goal.png'))
                with torch.no_grad():
                    z = encode(pipe.vae, Image.fromarray(goal), args.device)
                    positive = decode(pipe.vae, z)
                positive.save(d / (c + '_positive.png'))
                goal_rgb = pilot.rgb_tensor(goal, args.device)
                pos_rgb = pilot.rgb_tensor(positive, args.device)
                values, weight = data.inputs(args, p, c, 'raw' if arm == 'A' else 'outline')
                weight = torch.from_numpy(weight)[None, None].to(args.device)
                old = old_template(anchor_rgb, pos_rgb, p['patch_coordinates'])
                support = improved.suppression(anchor_rgb, pos_rgb, weight)
                self.records[p['case_id'], c] = z, goal_rgb, values, weight, old, support
                test = self.terms(goal_rgb, goal_rgb, goal_rgb, weight, old, p['patch_coordinates'], support)
                assert max(float(v) for v in test.values()) < 1e-6
                audit['vae'] = measurements.aligned(np.asarray(positive), goal, np.asarray(positive), anchor, mask)
                audit['family'] = spec['descriptions'][c]['family']
                audit['group'] = spec['descriptions'][c]['group']
                audit['target_sha256'] = sha(d / (c + '_goal.png'))
                audits[c] = audit
            self.targets.append(dict(case=p['case_id'], conditions=audits))

    def terms(self, rgb, goal, base, weight, template, coords, support):
        if self.arm == 'A':
            return old_terms(rgb, goal, base, weight, template, coords)
        return improved.loss_terms(rgb, goal, base, weight, template, coords, support)

    def view(self, entry, state_index, hops=2):
        p, c = entry['pair'], entry['condition']
        target_z, goal, values, weight, template, support = self.records[p['case_id'], c]
        state = entry['states'][state_index]
        z, epsilon = state['zt'].to(self.args.device), state['epsilon'].to(self.args.device)
        losses = []
        for hop in range(hops):
            t = torch.tensor(state['t'] - 20*hop, device=self.args.device)
            alpha = self.pipe.scheduler.alphas_cumprod[int(t)].to(z.device).float()
            x0 = (z.float() - (1-alpha).sqrt()*epsilon.float()) / alpha.sqrt()
            delta = self.b(z.float(), x0, timestep=t.reshape(1), **values)
            corrected = (epsilon.float() - (alpha/(1-alpha)).sqrt()*delta).to(epsilon.dtype)
            final = state['index'] + hop == 49
            predicted = self.pipe.scheduler.step(corrected, t, z, return_dict=False)[0].float() if final else x0+delta
            with torch.no_grad():
                base_z = self.pipe.scheduler.step(epsilon, t, z.detach(), return_dict=False)[0].float() if final else x0.detach()
                base = self.rgb(base_z)
            terms = self.terms(self.rgb(predicted), goal, base, weight, template, p['patch_coordinates'], support)
            terms['latent'] = pilot.weighted_mean((predicted-target_z).square(), values['mask'])
            losses.append(terms)
            if hop+1 < hops:
                z = self.pipe.scheduler.step(corrected, t, z, return_dict=False)[0]
                epsilon = self.predictor.epsilon(z, t-20, entry)
        return {k: sum(v[k] for v in losses)/hops for k in KEYS}


def probes(spec):
    families = ('stripes', 'checks', 'dots')
    for j, p in enumerate(spec['train']):
        for n, group in enumerate(('neutral', 'pastel', 'multi')):
            c = next(c for c, v in spec['descriptions'].items() if v['variant']=='native'
                     and v['group']==group and v['family']==families[(j+n)%3])
            yield p, c, group


def diagnose(args):
    spec = protocol(args)
    args.action = 'train'  # 诊断和后续训练使用同一确定性反传后端。
    core, pipe, net = pilot.load_core(args)
    before = weights_digest(net)
    assert before == read(args.previous / 'training/complete.json')['m0_hash']
    b = data.initialize(args).eval()
    reusable, cache_audit = reusable_trajectories(args, spec)
    m0 = cache(args, spec, core, pipe, net, reusable=reusable)
    del reusable
    torch.save(m0, args.out / 'm0_cache.pt')
    predictor = repair.FrozenPredictor(args, core, pipe, net, spec['train'], conditions=spec['conditions'])
    with torch.no_grad():
        for p in spec['train']:
            e = m0[p['case_id'], spec['conditions'][0]]; state = e['states'][0]
            replay = predictor.epsilon(state['zt'].to(args.device), torch.tensor(state['t'], device=args.device), e)
            assert torch.equal(replay.cpu(), state['epsilon'])
    objective = Objective(args, spec, pipe, predictor, b, 'B')
    write(args.out / 'target_controls.json', dict(targets=objective.targets, updates=0))
    trajectories = []
    with torch.no_grad():
        for p, c, group in probes(spec):
            d = args.out / 'cases' / p['case_id']
            mask = np.asarray(Image.open(d / 'mask.png')) > 0
            anchor = measurements.pixels(d / 'train_anchor.png')
            goal, positive = (measurements.pixels(d / (c+'_'+arm+'.png')) for arm in ('goal', 'positive'))
            for mode in ('raw', 'zero', 'outline'):
                picture, digest, tap = sample(args, core, pipe, net, p, c, b, mode, snapshots=True)
                dest = args.out / 'probe' / p['case_id'] / (c+'_'+mode)
                dest.mkdir(parents=True, exist_ok=True)
                tap.pictures['final'] = picture
                scores = {}
                for name, im in tap.pictures.items():
                    im.save(dest / (name+'.png'))
                    scores[name] = measurements.aligned(np.asarray(im), goal, positive, anchor, mask)
                trajectories.append(dict(case=p['case_id'], condition=c, group=group, mode=mode, digest=digest, scores=scores))
            print('JOINT_TRAJECTORY', p['case_id'], c, flush=True)
    pipe.vae.float()
    gradients = []
    parameters = list(b.parameters())
    for p, c, group in probes(spec):
        for state_index in (0, 9, 23):
            terms = objective.view(m0[p['case_id'], c], state_index)
            gs = []
            for k in KEYS:
                parts = torch.autograd.grad(terms[k], parameters, retain_graph=True, allow_unused=True)
                gs.append(torch.cat([torch.zeros_like(v).flatten() if g is None else g.detach().flatten()
                                     for v, g in zip(parameters, parts)]))
            g = torch.stack(gs)
            gram = g @ g.T
            norms = gram.diag().clamp_min(0).sqrt()
            cosine = gram / (norms[:,None]*norms[None,:]).clamp_min(1e-12)
            assert torch.isfinite(g).all() and float(norms[KEYS.index('rgb')]) > 0
            gradients.append(dict(case=p['case_id'], condition=c, group=group, state_index=state_index,
                losses={k:float(terms[k].detach()) for k in KEYS}, norms=norms.tolist(), cosine=cosine.tolist(), gram=gram.tolist()))
            del terms, gs, g, gram, norms, cosine
        print('JOINT_GRADIENT', p['case_id'], c, flush=True)
    med = np.median([p['norms'] for p in gradients], axis=0)
    desired = np.array([.5, 1., 1., 1., .5, .5, 1., 1.])
    weights = np.clip(desired*med[KEYS.index('rgb')]/np.maximum(med, 1e-12), .05, 20.)
    alignment = {}
    for group in ('neutral', 'pastel', 'multi'):
        values = []
        for p in (p for p in gradients if p['group']==group):
            gram = np.asarray(p['gram']); norm = np.sqrt(np.maximum(np.diag(gram),0)); total_norm = np.sqrt(max(float(weights@gram@weights),1e-18))
            values.append((gram@weights)/(norm*total_norm).clip(1e-12))
        alignment[group] = dict(zip(KEYS, np.median(values,axis=0).tolist()))
    controls = [v['vae'] for p in objective.targets for v in p['conditions'].values()]
    positive_ok = all(v['correlation'] >= .5 for v in controls)
    # 只阻止持续显著反向的组；完整两两矩阵保留，不用总梯度均值掩盖冲突。
    gradient_ok = all(v[k] >= -.25 for v in alignment.values() for k in ('pattern','chroma','palette','old'))
    rebound = [p for p in trajectories if p['mode']=='outline']
    old_delta = [p['scores']['45_before']['old_projection']-p['scores']['44_after']['old_projection'] for p in rebound
                 if p['scores']['45_before']['old_projection'] is not None and p['scores']['44_after']['old_projection'] is not None]
    palette_delta = [p['scores']['45_before']['palette_swd']-p['scores']['44_after']['palette_swd'] for p in rebound]
    four = bool((old_delta and np.median(old_delta)>.001) or np.median(palette_delta)>.25)
    passed = positive_ok and gradient_ok and weights_digest(net)==before
    result = dict(updates=0, passed=passed, positive_controls_pass=positive_ok, gradient_alignment_pass=gradient_ok,
        m0_hash=before, initial_b_hash=weights_digest(b), m0_cache_sha256=sha(args.out/'m0_cache.pt'), cache_reuse=cache_audit,
        input_protocol_sha256=sha(args.out/'input_protocol.json'), keys=KEYS, gradients=gradients,
        median_norms=med.tolist(), B_weights=dict(zip(KEYS,weights.tolist())), group_alignment=alignment,
        trajectories=trajectories, old_rebound_median=float(np.median(old_delta)) if old_delta else None,
        palette_rebound_median=float(np.median(palette_delta)), use_terminal_four_hops=four, frozen_unchanged=weights_digest(net)==before)
    write(args.out/'diagnostics.json',result)
    if passed:
        final = copy.deepcopy(spec)
        final.update(diagnostics_sha256=sha(args.out/'diagnostics.json'), A_weights=material.WEIGHTS,
            B_weights=result['B_weights'], use_terminal_four_hops=four,
            training=dict(seed=20261113, optimizer='AdamW', lr=1e-4, weight_decay=.01, gradient_clip=1.,
                          updates_per_arm=400, terminal_four_hop_fraction=.25 if four else 0., refresh=200))
        write(args.out/'protocol.json',final)
    from tools.e2_state_b_joint_report import diagnostic_archive
    diagnostic_archive(args, spec)
    assert passed, '训练数据零更新诊断未通过，禁止进入400步训练'
    print('JOINT_DIAGNOSTIC_PASS', four, result['B_weights'], flush=True)


def schedule(spec):
    rng = np.random.default_rng(20261113)
    refs = spec['references']
    neutral = [v for v in refs if v['palette_group']=='neutral']
    rows = []
    for step in range(1,401):
        p = spec['train'][(step-1)%4]
        if ((step-1)//4)%2 == 0:
            ref = refs[((step-1)//8)%len(refs)]; i=ref['index']; conditions=[f'r{i}_native',f'r{i}_color']; kind='colour'
        else:
            ref = neutral[((step-1)//8)%len(neutral)]; i=ref['index']; variant=('r90','s50')[((step-1)//24)%2];conditions=[f'r{i}_native',f'r{i}_{variant}'];kind='geometry'
        # 四步展开轮换分配给4个身份，避免只落在每个batch末尾的同一身份。
        four = spec['use_terminal_four_hops'] and ((step-1)//4)%4 == (step-1)%4
        rows.append(dict(step=step,case=p['case_id'],conditions=conditions,kind=kind,
            state_index=21 if four else int(rng.integers(24)),hops=4 if four else 2,
            refreshed=step>200 and ((step-1)//12)%2==1))
    return rows


def train(args):
    spec = protocol(args, final=True)
    assert read(args.out/'diagnostics.json')['passed'] and sha(args.out/'diagnostics.json')==spec['diagnostics_sha256']
    core, pipe, net = pilot.load_core(args); before=weights_digest(net)
    assert before==read(args.previous/'training/complete.json')['m0_hash']
    # 训练进程从磁盘读取轨迹，没有先调用管线来初始化调度器。
    pipe.scheduler.set_timesteps(spec['sampler']['steps'], device=args.device)
    b = data.initialize(args); folder=args.out/'training'/args.arm;folder.mkdir(parents=True,exist_ok=True)
    save=lambda n:torch.save(dict(step=n,b=b.state_dict()),folder/f'step{n}.pt')
    initial_b_hash=weights_digest(b);save(0)
    diag=read(args.out/'diagnostics.json')
    assert initial_b_hash==diag['initial_b_hash'] and sha(args.out/'m0_cache.pt')==diag['m0_cache_sha256']
    m0=torch.load(args.out/'m0_cache.pt',map_location='cpu',weights_only=False)
    timesteps = pipe.scheduler.timesteps.tolist()
    for entry in m0.values():
        assert all(timesteps[state['index']] == state['t'] for state in entry['states'])
    predictor=repair.FrozenPredictor(args,core,pipe,net,spec['train'],conditions=spec['conditions'])
    objective=Objective(args,spec,pipe,predictor,b,args.arm);pipe.vae.float()
    weights=spec[args.arm+'_weights'];optimizer=torch.optim.AdamW(b.parameters(),lr=1e-4,weight_decay=.01)
    rows=schedule(spec);refreshed=None
    with (folder/'loss.jsonl').open('w',encoding='utf8') as log:
        for row in rows:
            step=row['step']
            if step==201:
                b.eval();pipe.vae.half();refreshed=cache(args,spec,core,pipe,net,b,'raw' if args.arm=='A' else 'outline',m0);pipe.vae.float()
            b.train();optimizer.zero_grad(set_to_none=True);start=time.monotonic()
            stream=refreshed if row['refreshed'] else m0
            assert stream[row['case'],row['conditions'][0]]['digest']==stream[row['case'],row['conditions'][1]]['digest']
            losses=dict.fromkeys(KEYS,0.)
            for c in row['conditions']:
                terms=objective.view(stream[row['case'],c],row['state_index'],row['hops'])
                total=sum(weights[k]*terms[k] for k in KEYS);assert torch.isfinite(total)
                (total/2).backward()
                for k in KEYS:losses[k]+=float(terms[k].detach())/2
            grad=torch.nn.utils.clip_grad_norm_(b.parameters(),1.);assert torch.isfinite(grad);optimizer.step()
            log.write(json.dumps(dict(row,loss=losses,grad=float(grad),seconds=time.monotonic()-start))+'\n')
            if step%20==0:log.flush();print('JOINT_UPDATE',args.arm,step,losses,flush=True)
            if step in (200,400):save(step)
    assert weights_digest(net)==before and sha(args.previous/'training/step400.pt')==spec['base_checkpoint_sha256']
    write(folder/'complete.json',dict(updates=400,arm=args.arm,frozen_unchanged=True,m0_hash=before,initial_b_hash=initial_b_hash,
        checkpoint_sha256={str(i):sha(folder/f'step{i}.pt') for i in (0,200,400)},
        protocol_sha256=sha(args.out/'protocol.json'),schedule=rows,weights=weights))
    print('JOINT_TRAIN_COMPLETE',args.arm,400,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','diagnose','train','generate','evaluate','summarize'))
    parser.add_argument('--project',type=Path,default=pilot.PROJECT)
    parser.add_argument('--data',type=Path,default=pilot.PROJECT.parent/'datasets/BF')
    parser.add_argument('--previous',type=Path,default=pilot.PROJECT/'output_eval/e2_state_b_material_20261113')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--arm',choices=('A','B'),default='A')
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--reuse-cache-from',type=Path)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(2);torch.set_num_threads(2)
    if args.action=='prepare':data.prepare(args,source_hashes())
    elif args.action in ('diagnose','train'):globals()[args.action](args)
    else:
        from tools import e2_state_b_joint_report as report
        getattr(report,args.action)(args,protocol(args,final=True))


if __name__=='__main__':main()
