"""第20轮底座上的 B 初始化对照；共享融合参照，仅改变可训练尾块的起点。"""
import argparse
import copy
import inspect
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from tools import e2_f68_e1_retain125_long as rules
from tools.e2_fft_experiment import environment, setup as setup_core
from tools.m1_evaluation import prepare_inference

ft, tail, ab, roi, fid = rules.ft, rules.tail, rules.ab, rules.roi, rules.fid
guard = rules.guard
RUN = 'b_init_e2_epoch20_ab_20261123'
BASE = 'output_eval/e1_bfm_continue20_20261121/epochs/epoch_20/training/E1-BFM-T3-epoch20/final.pt'
BASE_SHA = '292ade058ed8db056ce4d1f01b36e05b07c60935f73783586ca84398d729692a'
A_SHA = '0062c3b15d2bc31c2199c8e1565b6f4a58af3cf63c209a9f25abcf89838b549b'
B64_SHA = '9926326896c15bb2be18f2f6869b564216d21cb383e58b4188668355c9667af9'
ARMS = ('f68', 'B64')
STEPS, ACCUMULATE, LR = 128, 2, 1e-5
SNAPSHOTS = (0, 32, 64, 128)


def identity(pair):
    return Path(pair['gt']).stem.split('_')[0]


def setup(prior):
    args = argparse.Namespace(prior=prior, project=prior.parents[1], device='cuda:0', action='train')
    args.data = args.project.parent/'datasets/BF'
    args.baseline = prior/tail.distill.BASE
    args.run = args.baseline/RUN
    args.run.mkdir(parents=True, exist_ok=True)
    args.out = args.run
    args.checkpoint = args.project/'output_eval/e2_state_b_pastel_followup_20261115/trust_retry/training/step1.pt'
    args.base_checkpoint = args.project/BASE
    old = args.baseline/'tail6_real_ab_e1_epoch10_20261119'
    args.a_checkpoint = old/'A/step128.pt'
    args.b64_checkpoint = old/'a_fused25_lstar_ab_20261121/B/step64.pt'
    args.protocol = tail.verify(argparse.Namespace(out=args.baseline))
    training = args.baseline/tail.distill.RUN
    args.trainval = tail.distill.experiment_protocol(argparse.Namespace(out=training, baseline=args.baseline))
    args.test = [dict(args.protocol['pairs'][i], index=i, split='test') for i in tail.subset(args.protocol)]
    args.pairs = [*args.trainval['pairs'], *args.test]
    splits = {s:{identity(p) for p in args.pairs if p['split']==s} for s in ('train','val','test')}
    assert [len(splits[s]) for s in splits]==[128,32,64]
    assert not (splits['train'] & splits['val'] or splits['train'] & splits['test'] or splits['val'] & splits['test'])
    assert len({p['case_id'] for p in args.pairs})==224
    for path, digest in ((args.base_checkpoint,BASE_SHA),(args.checkpoint,tail.CHECKPOINT_SHA),
                         (args.a_checkpoint,A_SHA),(args.b64_checkpoint,B64_SHA)):
        assert tail.sha(path)==digest, str(path)
    complete = tail.read(args.base_checkpoint.with_name('complete.json'))
    assert (complete['epoch'],complete['total_updates'],complete['checkpoint_sha256'])==(20,113460,BASE_SHA)
    assert tail.metric_assets()==args.protocol['metric_assets']
    args.gt_hashes = {}
    for pair in args.pairs:
        folder = args.run/'cases'/pair['case_id']
        folder.mkdir(parents=True, exist_ok=True)
        source = args.baseline if pair['split']=='test' else training
        for name, digest in (('mask.png',pair['mask_sha256']),
                             ('original_ref.png',pair['reference_hashes'][tail.CONDITION])):
            src, dst = source/'cases'/pair['case_id']/name, folder/name
            assert tail.sha(src)==digest
            if not dst.exists():
                dst.symlink_to(src)
            assert tail.sha(dst)==digest
        for name in ('sketch','texture'):
            assert tail.sha(args.data/pair[name])==pair['input_sha256'][name]
        args.gt_hashes[pair['case_id']] = tail.sha(args.data/pair['gt'])
        if pair['split']=='test':
            assert args.gt_hashes[pair['case_id']]==pair['input_sha256']['gt']
    manifest = args.base_checkpoint.parents[4]/'training_manifest.json'
    base_protocol = tail.read(manifest.with_name('protocol.json'))
    assert tail.sha(manifest)==base_protocol['train_sha256']
    roots = {Path(r['cloth']).stem.split('_')[0] for r in tail.read(manifest)}
    assert not roots & splits['test']
    modules = (tail,ab,roi,fid,ft,guard,rules,ft.fused,ft.light,tail.joint,tail.repair,
               tail.joint.data,tail.joint.data.old)
    root = Path(__file__).resolve().parents[1]
    files = {Path(m.__file__) for m in modules}|{Path(__file__)}
    files.update(root/name for name in ('models/e2_state_texture_b.py','tools/e2_fft_experiment.py',
        'tools/m1_evaluation.py','tools/e2_texture_bottleneck.py','pipelines/IMAGGarment_pipeline.py',
        'eval/eval_utils.py','eval/metrics.py','scripts/e2_b_init_epoch20_ab.sbatch'))
    source_hashes = {str(p.relative_to(Path(__file__).resolve().parents[1])):tail.sha(p) for p in files}
    args.spec = dict(base_path=str(args.base_checkpoint),base_sha256=BASE_SHA,
        f68_sha256=tail.CHECKPOINT_SHA,B64_sha256=B64_SHA,frozen_A_sha256=A_SHA,
        base_train_manifest_sha256=base_protocol['train_sha256'],base_train_val_overlap=len(roots & splits['val']),
        base_train_test_overlap=0,trainval_protocol_sha256=tail.sha(training/'protocol.json'),
        test_protocol_sha256=tail.sha(args.baseline/'protocol.json'),pairs=args.pairs,gt_sha256=args.gt_hashes,
        seed=42,sampler=args.protocol['sampler'],metric_assets=args.protocol['metric_assets'],
        optimizer_updates_per_arm=STEPS,accumulation=ACCUMULATE,lr=LR,optimizer='AdamW, fresh state in both arms',
        validation_steps=list(SNAPSHOTS),trainable_parameters=tail.PARAMETERS,
        trainable=['blocks.1','output'],tail_indices=list(range(44,50)),retention=.125,
        frozen_A_role='same historical A weights, regenerated on epoch20; never trained in this comparison',
        common_anchor='epoch20 B64 zero-update final fused image, identical G/E reference in both arms',
        loss='L0 + G + 0.20 E; L0=regionL1+.05globalLPIPS+.05insideLPIPS+.10lowLstar',
        fusion='unchanged donor0.25, background recovery, edge feathering, final retain12.5%',
        selection='validation only: original seven+texture, common B64-zero relative tradeoff; retain zero fallback',
        tolerance=.005,gain_ratio=2.,test_role='reused original64, initialization selection study, not independent confirmation',
        expanded_evaluation=False,code=source_hashes)
    path = args.run/'protocol.json'
    if path.exists():
        assert tail.read(path)==args.spec
    else:
        tail.write(path,args.spec)
    args.spec_sha = tail.sha(path)
    return args


def load(args):
    context = copy.copy(args)
    core,_ = setup_core(context)
    pipe,net = core.load_network(args.device)
    state = torch.load(args.base_checkpoint,map_location='cpu',weights_only=False,mmap=True)
    assert state['total_updates']==113460
    net.load_state_dict(state['network'],strict=True)
    del state
    prepare_inference(args,pipe,net)
    pipe.vae.eval().requires_grad_(False)
    original = tail.StateTextureB(color_channels=2,reference_channels=192).to(args.device)
    original.load_state_dict(torch.load(args.checkpoint,map_location='cpu',weights_only=False)['b'])
    original.eval().requires_grad_(False)
    assert type(pipe.scheduler).__name__=='DDIMScheduler'
    frozen = dict(B=original,net=net,vae=pipe.vae)
    args.original = dict(frozen_model_hashes=tail.tensor_set(frozen),
        scheduler_sha256=tail.sha(Path(inspect.getsourcefile(type(pipe.scheduler)))),base_sha256=BASE_SHA)
    path = args.run/'model.json'
    if path.exists():
        assert tail.read(path)==args.original
    else:
        tail.write(path,args.original)
        tail.write(args.run/'environment.json',environment())
    return core,pipe,net,original,frozen


def head_for(args,models,name):
    head,params = tail.candidates(models[3])
    path = dict(A=args.a_checkpoint,B64=args.b64_checkpoint).get(name)
    if path:
        ab.update_head(head,params,path)
    return head,params


def head_hash(head):
    return tail.tensor_set({'head':head})['head']


def prefix(args,models,pair):
    path = args.run/'prefix'/(pair['case_id']+'.pt')
    if not path.exists():
        state = tail.cache_prefix(args,*models[:4],pair)
        tail.save_torch(path,dict(state,case=pair['case_id'],protocol_sha256=args.spec_sha))
    state = torch.load(path,map_location='cpu',weights_only=True)
    assert state['case']==pair['case_id'] and state['protocol_sha256']==args.spec_sha
    return state


def raw_forward(args,models,head,pair):
    mask = tail.case_mask(args,pair)
    state = prefix(args,models,pair)
    prediction = tail.decode(models[1],tail.trajectory(args,models,head,pair,state,mask))
    gt_path = args.data/pair['gt']
    assert tail.sha(gt_path)==args.gt_hashes[pair['case_id']]
    gt = roi.tensors([np.asarray(tail.rgb(gt_path).resize(tail.SIZE,Image.Resampling.BILINEAR))]).to(args.device)
    return prediction,gt,mask


def checked_image(path,meta,expected):
    if not meta.exists():
        return None
    record = tail.read(meta)
    assert all(record[k]==v for k,v in expected.items())
    assert tail.sha(path)==record['png_sha256']
    return np.asarray(tail.rgb(path))


def save_image(path,meta,image,record):
    ft.fused.save_png(path,image)
    tail.write(meta,dict(record,png_sha256=tail.sha(path)))
    return image


@torch.no_grad()
def raw_cached(args,models,head,pair):
    digest = head_hash(head)
    path = args.run/'raw'/digest/(pair['case_id']+'.png')
    meta = path.with_suffix('.json')
    record = dict(protocol_sha256=args.spec_sha,case=pair['case_id'],head_sha256=digest)
    image = checked_image(path,meta,record)
    if image is None:
        prediction,_,_ = raw_forward(args,models,head,pair)
        image = save_image(path,meta,ft.rgb8(prediction),record)
        models[2].adapter.clear()
    return image


@torch.no_grad()
def controls(args,models,pair):
    core,pipe,net,original,_ = models
    values,records = {},{}
    for arm in ('off','on'):
        path = args.run/'raw'/arm/(pair['case_id']+'.png')
        meta = path.with_suffix('.json')
        record = dict(protocol_sha256=args.spec_sha,case=pair['case_id'],arm=arm)
        image = checked_image(path,meta,record)
        if image is None:
            picture,noise,tap = tail.joint.sample(args,core,pipe,net,pair,tail.CONDITION,
                b=original if arm=='on' else None,mode='raw',capture=True)
            record.update(initial_noise_digest=noise,first_b_step={k:tail.tensor_sha(tap.states[0][k]) for k in ('zt','epsilon')})
            image = save_image(path,meta,np.asarray(picture),record)
        records[arm] = tail.read(meta)
        values[arm] = image
    assert all(records['off'][k]==records['on'][k] for k in ('initial_noise_digest','first_b_step'))
    return values


@torch.no_grad()
def fixed(args,models):
    a,_ = head_for(args,models,'A')
    b64,_ = head_for(args,models,'B64')
    a.requires_grad_(False)
    b64.requires_grad_(False)
    anchors,images = {},{}
    for i,pair in enumerate(args.pairs,1):
        case = pair['case_id']
        mask = tail.case_mask(args,pair)
        folder = args.run/'fixed'/case
        marker = folder/'record.json'
        if marker.exists():
            record = tail.read(marker)
            assert record['protocol_sha256']==args.spec_sha
            source = {n:fid.checked_png(folder/(n+'.png'),h) for n,h in record['pngs'].items()}
        else:
            prefix(args,models,pair)
            source = controls(args,models,pair)
            source['blend'] = tail.mix_pixels(source['off'],source['on'],mask,{'blend':.25})['blend']
            source['offline25'] = ft.fused.pixels_mixed(raw_cached(args,models,a,pair),source['blend'],ft.fused.alpha(mask))
            current = ft.fused.pixels_mixed(raw_cached(args,models,b64,pair),source['blend'],ft.fused.alpha(mask))
            source['baseline125'] = ft.uniform_mix(current,source['offline25'],mask)
            for name,image in source.items():
                ft.fused.save_png(folder/(name+'.png'),image)
            tail.write(marker,dict(protocol_sha256=args.spec_sha,
                prefix_sha256=tail.sha(args.run/'prefix'/(case+'.pt')),
                pngs={n:tail.sha(folder/(n+'.png')) for n in source}))
        assert tail.sha(args.run/'prefix'/(case+'.pt'))==tail.read(marker)['prefix_sha256']
        anchors[case] = source['blend'],source['offline25']
        images[case] = source['baseline125']
        if i%8==0:
            print('INIT20_FIXED',i,len(args.pairs),case,flush=True)
        models[2].adapter.clear()
    del a,b64
    return anchors,images


def forward(args,models,head,pair,anchors):
    prediction,target,mask = raw_forward(args,models,head,pair)
    donor,offline = anchors[pair['case_id']]
    return ft.fuse_final(prediction,donor,offline,mask),target,mask


@torch.no_grad()
def final_cached(args,models,head,pair,anchors):
    digest = head_hash(head)
    path = args.run/'renders'/digest/(pair['case_id']+'.png')
    meta = path.with_suffix('.json')
    record = dict(protocol_sha256=args.spec_sha,case=pair['case_id'],head_sha256=digest)
    image = checked_image(path,meta,record)
    if image is None:
        donor,offline = anchors[pair['case_id']]
        mask = tail.case_mask(args,pair)
        current = ft.fused.pixels_mixed(raw_cached(args,models,head,pair),donor,ft.fused.alpha(mask))
        image = save_image(path,meta,ft.uniform_mix(current,offline,mask),record)
    return image,path


@torch.no_grad()
def reference_losses(args,models,base,images):
    path = args.run/'references.json'
    if path.exists():
        references = tail.read(path)
        assert set(references)=={p['case_id'] for p in args.trainval['pairs']}
        return references
    references = {}
    for pair in args.trainval['pairs']:
        rgb = guard.anchor_tensor(images[pair['case_id']],torch.empty((),device=args.device))
        gt = guard.anchor_tensor(np.asarray(tail.rgb(args.data/pair['gt']).resize(tail.SIZE,Image.Resampling.BILINEAR)),rgb)
        mask = tail.case_mask(args,pair)
        loss,terms = ab.collect(base,rgb,gt,mask,'inside')
        references[pair['case_id']] = dict(loss=float(loss),**terms)
    tail.write(path,references)
    return references


def metrics_from_group(group,arm):
    scores = group['arms'][arm]
    return dict({k:scores[k]['fid' if k=='fid' else 'mean'] for k in rules.KEYS if k!='band_mae'},
                band_mae=group['texture'][arm]['band_mae'])


def gates(group,arm):
    scores = group['arms']
    seven = tail.distill.acceptance(dict(groups=dict(all=dict(arms={**scores,'student':scores[arm]}))))
    texture = group['texture']
    threshold = min(texture[n]['band_mae'] for n in ('off','on'))
    passed = texture[arm]['band_mae']<threshold
    return dict(seven=seven,texture_pass=passed,texture_threshold=threshold,passed=seven['passed'] and passed)


@torch.no_grad()
def score(args,models,pairs,heads,label,anchors,metrics):
    dest = args.run/label
    dest.mkdir(parents=True,exist_ok=True)
    result_path = dest/'result.json'
    if result_path.exists():
        result = tail.read(result_path)
        assert result['protocol_sha256']==args.spec_sha
        assert result['pool3_sha256']==tail.sha(dest/'pool3.npz')
        return result
    names = ('off','on','blend','offline25','baseline125',*heads)
    lpips,inception = metrics
    features = {n:[] for n in (*names,'gt')}
    rows = []
    for start in range(0,len(pairs),4):
        images = {n:[] for n in names}
        truth = []
        for pair in pairs[start:start+4]:
            case = pair['case_id']
            mask = tail.case_mask(args,pair)
            gt_path = args.data/pair['gt']
            assert tail.sha(gt_path)==args.gt_hashes[case]
            gt = tail.rgb(gt_path).resize(tail.SIZE,Image.Resampling.BILINEAR)
            inside = roi.regions(mask,20)[0]
            gt_lab = roi.lab(np.asarray(gt))
            row = dict(index=pair.get('index',start+len(truth)),case_id=case,category=pair['category'],
                       interior_px=int(inside.sum()),pngs={},scores={},texture={})
            for name in names:
                if name in heads:
                    image,path = final_cached(args,models,heads[name],pair,anchors)
                else:
                    path = args.run/'fixed'/case/(name+'.png')
                    image = np.asarray(tail.rgb(path))
                row['pngs'][name] = tail.sha(path)
                row['scores'][name] = tail.image_scores(Image.fromarray(image),gt,path,gt_path,mask)
                row['texture'][name] = tail.appearance(image,gt_lab,inside)
                images[name].append(image)
            row['texture']['gt'] = tail.appearance(np.asarray(gt),gt_lab,inside)
            truth.append(np.asarray(gt))
            rows.append(row)
        target = roi.tensors(truth)
        features['gt'].append(inception(target)[0].flatten(1).numpy())
        for name in names:
            tensor = roi.tensors(images[name])
            features[name].append(inception(tensor)[0].flatten(1).numpy())
            distances = lpips(tensor*2-1,target*2-1).flatten().tolist()
            for row,value in zip(rows[start:],distances):
                row['scores'][name]['lpips'] = value
        print('INIT20_SCORED',label,len(rows),len(pairs),flush=True)
        models[2].adapter.clear()
    features = {n:np.concatenate(v) for n,v in features.items()}
    assert all(v.shape==(len(pairs),2048) and np.isfinite(v).all() for v in features.values())
    result = tail.summarize(rows,features,names,('baseline125',),tuple(heads),fid_reference='baseline125')
    # 全类别使用同一批图和特征，避免把服装占比差异混入初始化比较。
    for category in tail.distill.CATEGORIES:
        if category in result['groups']:
            continue
        chosen = [r for r in rows if r['category']==category]
        keep = np.array([r['category']==category for r in rows])
        result['groups'][category] = dict(n=len(chosen),arms={n:dict(
            fid=roi.fid_terms(features[n][keep],features['gt'][keep]),
            **{k:dict(mean=float(np.mean([r['scores'][n][k] for r in chosen]))) for k in tail.METRICS}) for n in names})
    ft.class_ab.summarize_texture(result,rows,(*names,'gt'))
    result.update(protocol_sha256=args.spec_sha,pairs=len(pairs),acceptance={n:gates(result['groups']['all'],n) for n in names})
    np.savez_compressed(dest/'pool3.npz',**features)
    result['pool3_sha256'] = tail.sha(dest/'pool3.npz')
    tail.write(dest/'per_pair.json',rows)
    tail.write(result_path,result)
    return result


def select(checkpoints):
    # 所有臂共用 B64 的零更新结果；不能分别使用各自起点定义 G/E 或相对门槛。
    zero = checkpoints['0']['groups']['all']
    baseline = metrics_from_group(zero,'baseline125')
    selected,assessments = {},{}
    for arm in ARMS:
        candidates = []
        assessments[arm] = {}
        for step,result in checkpoints.items():
            group = result['groups']['all']
            value = metrics_from_group(group,arm)
            trade = rules.tradeoff(value,baseline)
            same_anchor = arm=='B64' and step=='0'
            eligible = result['acceptance'][arm]['passed'] and (trade['accepted'] or same_anchor)
            assessments[arm][step] = dict(absolute=result['acceptance'][arm],tradeoff=trade,eligible=eligible)
            if eligible:
                candidates.append((trade['net'], -value['lpips'], -value['fid'], -int(step), int(step)))
        selected[arm] = max(candidates)[-1] if candidates else 0
    return selected,assessments


def train(args,models,anchors,images,metrics):
    training = [p for p in args.trainval['pairs'] if p['split']=='train']
    validation = [p for p in args.trainval['pairs'] if p['split']=='val']
    heads,parameters,optimizers = {},{},{}
    for arm in ARMS:
        heads[arm],parameters[arm] = head_for(args,models,arm)
        optimizers[arm] = torch.optim.AdamW(parameters[arm].values(),lr=LR)
    base = ft.light.LightingObjective(args)
    references = reference_losses(args,models,base,images)
    criterion = guard.Objective(base,references,images)
    trace,checkpoints = [],{}
    rng = np.random.default_rng(42)
    order = np.concatenate([rng.permutation(128) for _ in range(2)])
    assert len(order)==STEPS*ACCUMULATE
    completed = 0
    resume = args.run/'resume.pt'
    if resume.exists():
        state = torch.load(resume,map_location='cpu',weights_only=False)
        assert state['protocol_sha256']==args.spec_sha
        completed,trace = state['step'],state['trace']
        for arm in ARMS:
            rules.restore(parameters[arm],state['heads'][arm])
            optimizers[arm].load_state_dict(state['optimizers'][arm])
    else:
        for arm in ARMS:
            tail.save_torch(args.run/arm/'step0.pt',{n:p.detach().cpu() for n,p in parameters[arm].items()})
    for step in SNAPSHOTS:
        if step<=completed and (args.run/f'validation/step{step}/result.json').exists():
            checkpoints[str(step)] = tail.read(args.run/f'validation/step{step}/result.json')
    # 中断可能发生在两臂更新落盘之后、当次验证归档之前。
    if completed in SNAPSHOTS and str(completed) not in checkpoints:
        for arm in ARMS:
            tail.save_torch(args.run/arm/f'step{completed}.pt',{n:p.detach().cpu() for n,p in parameters[arm].items()})
        checkpoints[str(completed)] = score(args,models,validation,heads,f'validation/step{completed}',anchors,metrics)
    if completed==0:
        checkpoints['0'] = score(args,models,validation,heads,'validation/step0',anchors,metrics)
    for step in range(completed+1,STEPS+1):
        batch = [training[j] for j in order[(step-1)*ACCUMULATE:step*ACCUMULATE]]
        for arm in ARMS:
            optimizer,params = optimizers[arm],parameters[arm]
            optimizer.zero_grad(set_to_none=True)
            records = []
            for pair in batch:
                pred,target,mask = forward(args,models,heads[arm],pair,anchors)
                loss,terms = guard.collect(criterion,pred,target,mask,pair)
                (loss*(128/ACCUMULATE)).backward()
                records.append(dict(case=pair['case_id'],loss=float(loss.detach()),**terms))
                models[2].adapter.clear()
                del loss,pred,target
            assert all(p.grad is not None for p in params.values())
            for p in params.values():
                p.grad.div_(128)
            norm = float(torch.nn.utils.clip_grad_norm_(params.values(),1,error_if_nonfinite=True))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            trace.append(dict(step=step,arm=arm,cases=[p['case_id'] for p in batch],grad_norm=norm,rows=records))
        # 两臂一起断点，恢复时不会使其中一臂多做一次更新。
        tail.save_torch(resume,dict(protocol_sha256=args.spec_sha,step=step,trace=trace,
            heads={a:{n:p.detach().cpu() for n,p in parameters[a].items()} for a in ARMS},
            optimizers={a:optimizers[a].state_dict() for a in ARMS}))
        if step%8==0:
            print('INIT20_TRAIN',step,STEPS,flush=True)
        if step in SNAPSHOTS:
            for arm in ARMS:
                tail.save_torch(args.run/arm/f'step{step}.pt',{n:p.detach().cpu() for n,p in parameters[arm].items()})
            checkpoints[str(step)] = score(args,models,validation,heads,f'validation/step{step}',anchors,metrics)
    assert set(checkpoints)==set(map(str,SNAPSHOTS))
    for arm in ARMS:
        ab.audit_frozen(args,models,heads[arm],parameters[arm])
    selected,assessments = select(checkpoints)
    saved = dict(protocol_sha256=args.spec_sha,updates_per_arm=STEPS,trace=trace,selected=selected,
        selection=assessments,trainable=list(parameters['f68']),parameter_count=tail.PARAMETERS,
        frozen_unchanged=True,checkpoints={a:{str(s):tail.sha(args.run/a/f'step{s}.pt') for s in SNAPSHOTS} for a in ARMS})
    tail.write(args.run/'training.json',saved)
    tail.write(args.run/'selection.json',dict(selected=selected,validation=assessments,protocol_sha256=args.spec_sha))
    chosen = {}
    for arm in ARMS:
        for label,step in (('zero',0),('step128',128),('selected',selected[arm])):
            head,params = head_for(args,models,'f68')
            ab.update_head(head,params,args.run/arm/f'step{step}.pt')
            chosen[arm+'_'+label] = head.eval().requires_grad_(False)
    return saved,chosen


def qualify(args,models,anchors,images):
    path = args.run/'qualification.json'
    if path.exists():
        assert tail.read(path)['protocol_sha256']==args.spec_sha
        return
    pair = next(p for p in args.trainval['pairs'] if p['split']=='val')
    mask = tail.case_mask(args,pair)
    records = {}
    core,pipe,net,original,_ = models
    for arm in ARMS:
        head,params = head_for(args,models,arm)
        values,_ = tail.joint.data.inputs(args,pair,tail.CONDITION,'raw')
        features = tail.joint.projected(net,core.tensor(args.data/pair['sketch']).to(args.device).half(),
            core.tensor(args.out/'cases'/pair['case_id']/'original_ref.png').to(args.device).half())
        module = tail.TailResidual(original,head,tail.latent_gate(mask,args.device))
        with torch.no_grad(),tail.PrefixTap(pipe,module,values,qualify=True) as tap:
            picture,_ = tail.joint.image(args,core,pipe,net,pair,features)
        state = prefix(args,models,pair)
        assert all(torch.equal(state[n],tap.start[n]) for n in ('z','epsilon'))
        with torch.no_grad():
            latent = tail.trajectory(args,models,head,pair,state,mask)
            assert torch.equal(latent.cpu(),tap.final)
            decoded = tail.decode(pipe,latent)
            np.testing.assert_array_equal(ft.rgb8(decoded),np.asarray(picture))
        pred,target,mask = forward(args,models,head,pair,anchors)
        expected,_ = final_cached(args,models,head,pair,anchors)
        np.testing.assert_array_equal(ft.rgb8(pred),expected)
        base = ft.light.LightingObjective(args)
        loss,_ = base(pred,target,mask,'inside')
        grad = tail.backward(loss,params)
        assert grad>0
        if arm=='B64':
            np.testing.assert_array_equal(ft.rgb8(pred),images[pair['case_id']])
            assert (pred.detach()-guard.anchor_tensor(images[pair['case_id']],pred)).abs().max()==0
        records[arm] = dict(exact_prefix=True,exact_latent=True,exact_raw_pixels=True,
            exact_fused_pixels=True,gradient_norm=grad,head_sha256=head_hash(head))
        models[2].adapter.clear()
        del pred,target,loss,head,base
    tail.write(path,dict(case=pair['case_id'],arms=records,
        common_reference=True,optimizer_updates=0,trainable_parameters=tail.PARAMETERS,protocol_sha256=args.spec_sha))
    print('INIT20_QUALIFIED',records,flush=True)


def examples(args):
    rows = tail.read(args.run/'eval64/per_pair.json')
    sample = [next(r for r in rows if r['category']==c) for c in tail.distill.CATEGORIES]
    names = ('gt','off','baseline125','f68_step128','B64_step128','f68_selected','B64_selected')
    sheet = Image.new('RGB',(len(names)*144,len(sample)*216),'white')
    for i,row in enumerate(sample):
        pair = next(p for p in args.test if p['case_id']==row['case_id'])
        for j,name in enumerate(names):
            if name=='gt':
                path = args.data/pair['gt']
            elif name in ('off','baseline125'):
                path = args.run/'fixed'/pair['case_id']/(name+'.png')
            else:
                path = args.run/'eval64/images'/name/(pair['case_id']+'.png')
            sheet.paste(tail.rgb(path).resize((138,184)),(j*144,i*216+16))
            ImageDraw.Draw(sheet).text((j*144+2,i*216+200),name,fill='black')
    sheet.save(args.run/'examples.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prior',type=Path)
    args = setup(parser.parse_args().prior)
    if (args.run/'completion.json').exists() and (args.run/'review.zip').exists():
        print('INIT20_ALREADY_COMPLETE',tail.sha(args.run/'review.zip'),flush=True)
        return
    cv2.setNumThreads(1)
    models = load(args)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','8')))
    with torch.no_grad():
        anchors,images = fixed(args,models)
    qualify(args,models,anchors,images)
    metrics = tail.metric_models('cpu',args.protocol['metric_assets'])
    trained,heads = train(args,models,anchors,images,metrics)
    result = score(args,models,args.test,heads,'eval64',anchors,metrics)
    group = result['groups']['all']
    reference = metrics_from_group(group,'baseline125')
    assessments = {}
    for arm in ARMS:
        name = arm+'_selected'
        trade = rules.tradeoff(metrics_from_group(group,name),reference)
        unchanged = arm=='B64' and trained['selected'][arm]==0
        eligible = result['acceptance'][name]['passed'] and (trade['accepted'] or unchanged)
        assessments[arm] = dict(step=trained['selected'][arm],gate=result['acceptance'][name],tradeoff=trade,eligible=eligible)
        for pair in args.test:
            image,_ = final_cached(args,models,heads[name],pair,anchors)
            ft.fused.save_png(args.run/'eval64/images'/name/(pair['case_id']+'.png'),image)
            final,_ = final_cached(args,models,heads[arm+'_step128'],pair,anchors)
            ft.fused.save_png(args.run/'eval64/images'/(arm+'_step128')/(pair['case_id']+'.png'),final)
    passing = [a for a in ARMS if assessments[a]['eligible']]
    picked = max(passing,key=lambda a:assessments[a]['tradeoff']['net']) if passing else None
    assert tail.tensor_set(models[4])==args.original['frozen_model_hashes']
    tail.write(args.run/'completion.json',dict(optimizer_updates_per_arm=STEPS,selected_steps=trained['selected'],
        test_pairs=64,assessments=assessments,preferred_initialization=picked,base_sha256=BASE_SHA,
        frozen_unchanged=True,eval500_executed=False,long_training_started=False,
        interpretation='original64 used to choose initialization; independent confirmation still needed'))
    examples(args)
    names = ['protocol.json','model.json','environment.json','qualification.json','references.json',
             'training.json','selection.json','completion.json','examples.png','eval64/result.json',
             'eval64/per_pair.json','eval64/pool3.npz']
    for step in SNAPSHOTS:
        names.extend([f'validation/step{step}/{n}' for n in ('result.json','per_pair.json','pool3.npz')])
    fid.pack(args.run,names)
    print('INIT20_FINISHED',picked,trained['selected'],tail.sha(args.run/'review.zip'),flush=True)


if __name__=='__main__':
    main()
