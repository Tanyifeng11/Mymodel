"""从 E1 第十轮底座的 A 权重继续优化，检验外框背景与衣内区域 LPIPS。"""
import argparse
import copy
import os
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from tools import e2_f68_region_diagnostic as roi
from tools import e2_f68_tail6_e1_epoch10 as epoch
from tools import e2_f68_tail6_real_ab as tail
from tools.e2_f68_rgb_student_roi import pixels_mixed
from tools.e2_f68_tail6_e1_roi import FOLLOWUP_RUN


RUN = 'a_roi_frame_ablation_20261120'
PARENT_SHA = '3472521b98bfba47ccb06fe000512bdb7017176a0d42fd7dc1981aa877155373'
OFFLINE_SHA = '071d465f6a7205c64f0f65384332ff70503800833f5ee1a5f6c2d4bb1de47639'
ARMS = ('frame', 'inside', 'both')
OPTIONS = {'frame': (True, False), 'inside': (False, True), 'both': (True, True)}
CONTROLS = ('off', 'on', 'blend', 'A', 'offline25')
LEARNING_RATE = 5e-5
FRAME_WEIGHT = .20
FRAME_WIDTH = 4
STEPS = 128


def prepare(args):
    parent = args.parent / 'eval64'
    tail.verified_files(parent, PARENT_SHA, ('protocol.json','model.json','selection.json','per_pair.json',
                                            'result.json','pool3.npz','training_A.json'))
    tail.verified_files(args.parent/FOLLOWUP_RUN, OFFLINE_SHA, ('result.json','per_pair.json','edit_pool3.npz'))
    parent_contract = tail.read(args.parent/'protocol.json')
    for artifact in ('protocol', 'model', 'selection'):
        assert tail.sha(parent/f'{artifact}.json') == tail.sha(args.parent/f'{artifact}.json')
    selection = tail.read(args.parent/'selection.json')
    assert selection['update'] == 128 and selection['contract'] == parent_contract
    start = args.parent/'A/step128.pt'
    assert tail.sha(start) == selection['checkpoints']['A']
    assert tail.sha(args.parent/'A/training.json') == tail.sha(parent/'training_A.json')
    assert tail.sha(args.out/'protocol.json') == parent_contract['training_protocol']
    assert tail.sha(args.baseline/'protocol.json') == parent_contract['evaluation_protocol']
    hashes = {}
    for shard in range(2):
        part = tail.read(args.parent/f'prefix_{shard}.json')
        assert part['contract'] == parent_contract and not (hashes.keys() & part['cases'].keys())
        hashes.update(part['cases'])
    assert set(hashes) == {p['case_id'] for p in args.trainval['pairs']}
    assert all(tail.sha(args.parent/'prefix'/(case+'.pt')) == checksum for case,checksum in hashes.items())
    file = args.base_checkpoint.stat()
    assert (file.st_size, file.st_mtime_ns) == (parent_contract['base_bytes'], parent_contract['base_mtime_ns'])
    contract = dict(code=tail.sha(Path(__file__)), parent_contract=parent_contract, parent_review_sha256=PARENT_SHA,
                    offline_review_sha256=OFFLINE_SHA, start_A_sha256=tail.sha(start), prefix=hashes,
                    updates=STEPS, seed=42, lr=LEARNING_RATE, batch=1, frame_weight=FRAME_WEIGHT, frame_width=FRAME_WIDTH,
                    rgb='mean of three nonempty regions', lpips_global=.1, lpips_inside_transfer=.05,
                    lpips_inside_radius=8, trainable=['blocks.1','output'], options=OPTIONS)
    tail.write(args.run/'protocol.json',contract)
    print('E1_ROI_PREPARED', len(hashes), contract['start_A_sha256'], flush=True)


def load(args):
    frozen = copy.copy(args)
    frozen.run = args.parent
    frozen.contract = args.contract['parent_contract']
    models = epoch.load_backbone(frozen)
    args.original = frozen.original
    return models


def head_from_a(args, original, trainable):
    head, params = tail.candidates(original)
    path = args.parent/'A/step128.pt'
    assert tail.sha(path) == args.contract['start_A_sha256']
    update_head(head, params, path)
    if not trainable:
        head.requires_grad_(False)
    return head, params


@torch.no_grad()
def update_head(head, params, path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    assert set(state) == set(params)
    for name,tensor in params.items():
        tensor.copy_(state[name])
    return head


def render(args, models, head, pair):
    # 末六步以外的状态来自已核验的上一轮缓存。
    cached = copy.copy(args)
    cached.run = args.parent
    cached.contract = args.contract['parent_contract']
    cached.gt_hashes = cached.contract['gt_sha256']
    return tail.render_pair(cached,models,head,pair,args.contract['prefix'])


def border_median(rgb):
    values = torch.cat((rgb[...,0,:],rgb[...,-1,:],rgb[...,:,0],rgb[...,:,-1]),dim=-1)
    values = values.sort(dim=-1,stable=True).values
    center = values.shape[-1]//2
    return values[...,center-1:center+1].mean(-1)


def weighted_mean(value, mask):
    return (value*mask).sum()/mask.sum().clamp_min(1.)


class Objective:
    def __init__(self,args):
        self.base = tail.RealPairLoss(args,False)

    def __call__(self, prediction, target, mask, arm):
        partition = torch.from_numpy(roi.regions(mask,8)).to(prediction.device)
        inside,_,outside = partition
        error = (prediction-target).abs().mean(1)[0]
        counts = partition.sum((-2,-1))
        l1 = ((error*partition).sum((-2,-1))/counts.clamp_min(1)).sum()/(counts>0).sum()
        metric = self.base.lpips
        maps = roi.maps(metric,roi.lpips_features(metric,target),prediction)
        lpips = sum(layer.mean() for layer in maps)
        area = inside.float()[None,None]
        inside_lpips = sum(weighted_mean(layer,F.interpolate(area,size=layer.shape[-2:],mode='area')[0,0])
                           for layer in maps)
        if not bool(inside.any()):
            inside_lpips = lpips
        frame = torch.ones_like(outside)
        frame[FRAME_WIDTH:-FRAME_WIDTH,FRAME_WIDTH:-FRAME_WIDTH] = False
        ring = frame & outside
        median = (border_median(prediction)-border_median(target)).abs().mean()
        frame_loss = (weighted_mean(error,ring)+weighted_mean(error,outside)+median)/3
        has_frame,has_inside = OPTIONS[arm]
        score = l1+.1*lpips
        if has_frame:
            score = score + FRAME_WEIGHT*frame_loss
        if has_inside:
            score = score + .05*(inside_lpips-lpips)
        return score,dict(lpips=lpips,rgb_l1=l1,inside_lpips=inside_lpips,frame=frame_loss,median=median)


def audit_frozen(args,models,head,params):
    original,frozen=models[3:]
    assert tail.tensor_set(frozen)==args.original['frozen_model_hashes']
    assert all(torch.equal(t,original.state_dict()[n]) for n,t in head.state_dict().items() if n not in params)


def qualify(args):
    models=load(args)
    head,params=head_from_a(args,models[3],True)
    val=next(p for p in args.trainval['pairs'] if p['split']=='val')
    expected=tail.read(args.parent/'A/training.json')['checkpoints']['128']['rows'][0]
    assert expected['case']==val['case_id']
    pred,gt,mask=render(args,models,head,val)
    criterion=Objective(args)
    loss,terms=criterion(pred,gt,mask,'both')
    for key in ('lpips','rgb_l1'):
        assert abs(float(terms[key].detach())-expected[key]) < 5e-6, (key,expected[key],float(terms[key].detach()))
    for key in ('inside_lpips','frame'):
        grad=torch.autograd.grad(terms[key],pred,retain_graph=True)[0]
        assert torch.isfinite(grad).all() and grad.abs().sum()>0
    grad=tail.backward(loss,params)
    assert grad>0
    audit_frozen(args,models,head,params)
    tail.write(args.run/'qualification.json',dict(case=val['case_id'], start_A_match=True, grad_norm=grad,
                                                 new_gradient_terms=['frame','inside_lpips'], contract=args.contract))
    print('A_ROI_QUALIFIED',val['case_id'],grad,flush=True)


def collect(criterion,rgb,gt,mask,arm):
    loss,terms=criterion(rgb,gt,mask,arm)
    return loss,{k:float(v.detach()) for k,v in terms.items()}


def train(args):
    assert tail.read(args.run/'qualification.json')['start_A_match']
    folder=args.run/args.arm
    folder.mkdir(exist_ok=True)
    if (folder/'training.json').is_file():
        done=tail.read(folder/'training.json')
        assert done['contract']==args.contract and done['updates']==STEPS
        return
    train_pairs=[p for p in args.trainval['pairs'] if p['split']=='train']
    validation=[p for p in args.trainval['pairs'] if p['split']=='val']
    assert (len(train_pairs),len(validation))==(128,32)
    models=load(args)
    head,params=head_from_a(args,models[3],True)
    criterion=Objective(args)
    optimizer=torch.optim.AdamW(params.values(),lr=LEARNING_RATE)
    snapshots,trace={},[]
    resume=folder/'resume.pt'
    completed=0
    if resume.is_file():
        state=torch.load(resume,map_location='cpu',weights_only=False)
        assert state['contract']==args.contract and state['arm']==args.arm
        completed=state['step']
        update_head(head,params,folder/f'step{completed}.pt')
        optimizer.load_state_dict(state['optimizer'])
        trace,snapshots=state['trace'],state['snapshots']
    for step,index in enumerate(np.random.default_rng(42).permutation(STEPS),start=1):
        if step<=completed:
            continue
        pair=train_pairs[index]
        optimizer.zero_grad(set_to_none=True)
        rgb,gt,mask=render(args,models,head,pair)
        loss,terms=collect(criterion,rgb,gt,mask,args.arm)
        grad=tail.backward(loss,params)
        optimizer.step()
        models[2].adapter.clear()
        trace.append(dict(step=step,case=pair['case_id'],grad_norm=grad,loss=float(loss.detach()),**terms))
        del loss,rgb,gt
        print('A_ROI_TRAIN',args.arm,step,pair['case_id'],flush=True)
        if step not in (64,STEPS):
            continue
        with torch.no_grad():
            rows=[]
            for val in validation:
                rgb,gt,mask=render(args,models,head,val)
                _,terms=collect(criterion,rgb,gt,mask,args.arm)
                rows.append(dict(case=val['case_id'],**terms))
                models[2].adapter.clear()
        audit_frozen(args,models,head,params)
        path=folder/f'step{step}.pt'
        tail.save_torch(path,{k:t.detach().cpu() for k,t in params.items()})
        metrics={k:float(np.mean([row[k] for row in rows])) for k in terms}
        snapshots[str(step)]=dict(sha256=tail.sha(path),rows=rows,**metrics)
        tail.save_torch(resume,dict(step=step,optimizer=optimizer.state_dict(),trace=trace,snapshots=snapshots,
                                    arm=args.arm,contract=args.contract))
        print('A_ROI_VALIDATION',args.arm,step,metrics['lpips'],flush=True)
    assert len(trace)==STEPS
    tail.write(folder/'training.json',dict(arm=args.arm,updates=len(trace),trace=trace,checkpoints=snapshots,
        trainable=list(params),parameter_count=tail.PARAMETERS,frozen_unchanged=True,contract=args.contract))


def select(args):
    logs={a:tail.read(args.run/a/'training.json') for a in ARMS}
    assert all(log['contract']==args.contract and log['updates']==STEPS and log['frozen_unchanged'] for log in logs.values())
    assert len({tuple(v['case'] for v in log['trace']) for log in logs.values()})==1
    means={step:{a:log['checkpoints'][step]['lpips'] for a,log in logs.items()} for step in ('64','128')}
    step=min(means,key=lambda k:(sum(means[k].values()),-int(k)))
    tail.write(args.run/'selection.json',dict(step=int(step), validation_lpips=means,
        checkpoints={a:log['checkpoints'][step]['sha256'] for a,log in logs.items()},contract=args.contract))
    print('A_ROI_SELECTION',step,means,flush=True)


def accepted(args):
    return tuple(k for k,v in tail.read(args.run/'eval64/result.json')['acceptance'].items() if v['passed'])


def png_path(args,name,case):
    old=args.parent/'test'/name/(case+'.png')
    return old if old.is_file() else args.run/'test'/name/(case+'.png')


@torch.no_grad()
def generate(args):
    few=args.action=='generate64'
    arms=ARMS if few else (*accepted(args),'off','on','A')
    assert few or len(arms)>3
    ids=epoch.pair_indices(args,few)[args.shard::args.shards]
    core,pipe,net,original,frozen=load(args)
    selection=tail.read(args.run/'selection.json')
    assert selection['contract']==args.contract
    heads={}
    for arm in arms:
        if arm in ('off','on'):
            continue
        head,params=head_from_a(args,original,False)
        if arm in ARMS:
            path=args.run/arm/f"step{selection['step']}.pt"
            assert tail.sha(path)==selection['checkpoints'][arm]
            update_head(head,params,path)
        heads[arm]=head
    args.out=args.baseline
    for count,index in enumerate(ids,start=1):
        pair=args.protocol['pairs'][index]
        case=pair['case_id']
        mask=tail.case_mask(args,pair)
        gate=tail.latent_gate(mask,args.device)
        ref=tail.read((args.parent/'test/off'/(case+'.json'))) if few else None
        for arm in arms:
            png=png_path(args,arm,case)
            meta=png.with_suffix('.json')
            expected=selection['checkpoints'].get(arm, {'off':None,'on':tail.CHECKPOINT_SHA,'A':args.contract['start_A_sha256']}.get(arm))
            if png.exists() and meta.exists():
                record=tail.read(meta)
                assert (record['case'],record['index'],record['checkpoint'])==(case,index,expected)
                assert record['png_sha256']==tail.sha(png)
            else:
                assert args.run in png.parents
                png.parent.mkdir(parents=True,exist_ok=True)
                module=None if arm=='off' else original if arm=='on' else tail.TailResidual(original,heads[arm],gate)
                pic,noise,tap=tail.joint.sample(args,core,pipe,net,pair,tail.CONDITION,b=module,mode='raw',capture=True)
                assert pic.size==tail.SIZE and tap.index==tail.LAST and len(tap.states)==25
                first={name:tail.tensor_sha(tap.states[0][name]) for name in ('zt','epsilon')}
                temp=png.with_suffix('.tmp')
                pic.save(temp,format='PNG')
                temp.replace(png)
                record=dict(case=case,index=index,arm=arm,png_sha256=tail.sha(png),steps=tail.LAST,
                            initial_noise_digest=noise,first_b_step=first,checkpoint=expected,
                            base_sha256=args.contract['parent_contract']['base_sha256'])
                tail.write(meta,record)
            assert record['base_sha256']==args.contract['parent_contract']['base_sha256']
            if ref is None:
                ref=record
            assert all(record[k]==ref[k] for k in ('initial_noise_digest','first_b_step'))
        print('A_ROI_GENERATED',args.action,args.shard,count,len(ids),case,flush=True)
    assert tail.tensor_set(frozen)==args.original['frozen_model_hashes']
    tail.write(args.run/f'{args.action}_{args.shard}.json',dict(ids=ids,arms=arms,frozen=True))


def control64(args,rows,features):
    parent=args.parent/'eval64'
    offline=args.parent/FOLLOWUP_RUN
    previous=tail.read(parent/'per_pair.json')
    derived=tail.read(offline/'per_pair.json')
    assert [r['index'] for r in previous]==[r['index'] for r in derived]==tail.subset(args.protocol)
    with np.load(parent/'pool3.npz',allow_pickle=False) as f, np.load(offline/'edit_pool3.npz',allow_pickle=False) as d:
        pooled={k:f[k] for k in CONTROLS if k!='offline25'}
        pooled['offline25']=d['A_bg_inner025']
        for pos,(old,probe) in enumerate(zip(previous,derived)):
            i=old['index']
            assert old['case_id']==probe['case_id']
            row=dict(index=i,case_id=old['case_id'],category=old['category'],interior_px=old['interior_px'],
                     scores={k:old['scores'][k] for k in CONTROLS if k!='offline25'},
                     texture={k:old['texture'][k] for k in (*CONTROLS,'gt') if k!='offline25'},
                     pngs={k:old['pngs'][k] for k in CONTROLS if k!='offline25'})
            row['scores']['offline25']=probe['scores']['A_bg_inner025']
            row['texture']['offline25']=probe['texture']['A_bg_inner025']
            rows[i]=row
            for k in CONTROLS:
                features[k][i]=pooled[k][pos]


def scoring_image(args,arm,case,rows,i):
    path=png_path(args,arm,case)
    digest=tail.sha(path)
    observation=tail.read(path.with_suffix('.json'))
    assert (observation['case'],observation['index'],observation['png_sha256'])==(case,i,digest)
    rows[i]['pngs'][arm]=digest
    return path,np.asarray(tail.rgb(path))


@torch.no_grad()
def score(args):
    few=args.action=='score64'
    mode='eval64' if few else 'eval500'
    out=args.run/mode
    out.mkdir(exist_ok=True)
    if (out/'review.zip').is_file():
        assert tail.sha(out/'result.json')==tail.read(out/'review_manifest.json')['result.json']
        return
    arms=ARMS if few else accepted(args)
    assert arms
    ids=epoch.pair_indices(args,few)
    action='generate64' if few else 'generate500'
    for shard in range(args.shards):
        marker=tail.read(args.run/f'{action}_{shard}.json')
        assert marker['ids']==ids[shard::args.shards] and marker['frozen']
        assert set(marker['arms'])==(set(arms) if few else set(arms)|{'off','on','A'})
    output_arms=(*CONTROLS,*arms)
    rows,features={}, {k:{} for k in output_arms}
    if few:
        control64(args,rows,features)
    else:
        previous=args.run/'eval64'
        rows={r['index']:r for r in tail.read(previous/'per_pair.json')}
        with np.load(previous/'pool3.npz',allow_pickle=False) as f:
            for k in output_arms:
                features[k].update(zip(tail.subset(args.protocol),f[k]))
    assert tail.metric_assets()==args.protocol['metric_assets']
    lpips,inception=tail.metric_models('cpu',args.protocol['metric_assets'])
    args.out=args.baseline
    new_arms=arms if few else output_arms
    with tempfile.TemporaryDirectory(prefix='a_roi_score_',dir=os.environ.get('SLURM_TMPDIR')) as temp:
        for offset in range(0,len(ids),4):
            batch=ids[offset:offset+4]
            pictures,targets={name:[] for name in new_arms},[]
            for i in batch:
                pair=args.protocol['pairs'][i]
                case=pair['case_id']
                mask=tail.case_mask(args,pair)
                core=roi.regions(mask,20)[0]
                target=args.data/pair['gt']
                assert tail.sha(target)==pair['input_sha256']['gt']
                gt=tail.rgb(target).resize(tail.SIZE,Image.Resampling.BILINEAR)
                gt_np=np.asarray(gt)
                gt_lab=roi.lab(gt_np)
                targets.append(gt_np)
                if not few:
                    rows[i]=dict(index=i,case_id=case,category=pair['category'],interior_px=int(core.sum()),
                                 scores={},texture={'gt':tail.appearance(gt_np,gt_lab,core)},pngs={})
                paths,raw={},{}
                for arm in (new_arms if few else (*arms,'off','on','A')):
                    paths[arm],raw[arm]=scoring_image(args,arm,case,rows,i)
                if not few:
                    raw['blend']=tail.mix_pixels(raw['off'],raw['on'],mask,{'blend':.25})['blend']
                    inside,_,outside=(roi.gaussian(a.astype('float32'),2) for a in roi.regions(mask,8))
                    raw['offline25']=pixels_mixed(raw['A'],raw['blend'],outside+.25*inside)
                    for k in ('blend','offline25'):
                        paths[k]=Path(temp)/(k+'.png')
                        Image.fromarray(raw[k]).save(paths[k])
                        rows[i]['pngs'][k]=tail.sha(paths[k])
                for arm in new_arms:
                    image=Image.fromarray(raw[arm])
                    rows[i]['scores'][arm]=tail.image_scores(image,gt,paths[arm],target,mask)
                    rows[i]['texture'][arm]=tail.appearance(raw[arm],gt_lab,core)
                    pictures[arm].append(raw[arm])
            gt=roi.tensors(targets)
            for arm in new_arms:
                x=roi.tensors(pictures[arm])
                for i,v in zip(batch,lpips(x*2-1,gt*2-1).flatten().tolist()):
                    rows[i]['scores'][arm]['lpips']=v
                features[arm].update(zip(batch,inception(x)[0].flatten(1).numpy()))
            if offset%40==0:
                print('A_ROI_SCORED',mode,offset+len(batch),len(ids),flush=True)
    full=tail.subset(args.protocol) if few else list(range(len(args.protocol['pairs'])))
    arrays={k:np.stack([features[k][i] for i in full]) for k in output_arms}
    with np.load(args.baseline/'fid_pool3.npz',allow_pickle=False) as old:
        arrays['gt']=old['gt'][full]
    assert all(f.shape==(len(full),2048) and np.isfinite(f).all() for f in arrays.values())
    ordered=[rows[i] for i in full]
    result=tail.summarize(ordered,arrays,output_arms,('off','A','offline25'),('A','offline25',*arms))
    for label,group in result['groups'].items():
        valid=[r for r in ordered if r['interior_px']>=256 and (label=='all' or r['category']==label)]
        group['texture_count']=len(valid)
        group['texture']={arm:{key:float(np.mean([r['texture'][arm][key] for r in valid]))
            for key in valid[0]['texture'][arm]} for arm in (*output_arms,'gt')}
    result.update(samples=len(full), reused_baseline_pairs=64, reused_candidate_pairs=0 if few else 64,
                  acceptance={arm:epoch.gate_check(result,arm) for arm in arms})
    tail.write(out/'result.json',result)
    tail.write(out/'per_pair.json',ordered)
    np.savez_compressed(out/'pool3.npz',**arrays)
    files={path.name:path for path in (out/'result.json',out/'per_pair.json',out/'pool3.npz',
             *[args.run/f'{n}.json' for n in ('protocol','qualification','selection')])}
    files.update({f'training_{a}.json':args.run/a/'training.json' for a in ARMS})
    manifest=out/'review_manifest.json'
    tail.write(manifest,{n:tail.sha(p) for n,p in files.items()})
    temporary=out/'review.tmp'
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as zip_out:
        for name,path in files.items():
            zip_out.write(path,name)
        zip_out.write(manifest,manifest.name)
    temporary.replace(out/'review.zip')
    print('A_ROI_ACCEPTED',mode,{a:v['passed'] for a,v in result['acceptance'].items()},tail.sha(out/'review.zip'),flush=True)


def main():
    actions=dict(prepare=prepare,qualify=qualify,train=train,select=select,
                 generate64=generate,generate500=generate,score64=score,score500=score)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=actions)
    parser.add_argument('--prior',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=4)
    args=parser.parse_args()
    args.project=args.prior.parents[1]
    args.data=args.project.parent/'datasets/BF'
    args.checkpoint=args.project/'output_eval/e2_state_b_pastel_followup_20261115/trust_retry/training/step1.pt'
    args.base_checkpoint=args.project/epoch.BASE_CHECKPOINT
    args.baseline=args.prior/tail.distill.BASE
    args.out=args.baseline/tail.distill.RUN
    args.parent=args.baseline/epoch.RUN
    args.run=args.parent/RUN
    args.run.mkdir(exist_ok=True)
    args.protocol=tail.verify(argparse.Namespace(out=args.baseline))
    args.trainval=tail.distill.experiment_protocol(args)
    args.m0_result=tail.read(args.baseline/'result.json')
    cv2.setNumThreads(1)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','2')))
    if args.action!='prepare':
        args.contract=tail.read(args.run/'protocol.json')
        assert args.contract['code']==tail.sha(Path(__file__))
    actions[args.action](args)


if __name__=='__main__':
    main()
