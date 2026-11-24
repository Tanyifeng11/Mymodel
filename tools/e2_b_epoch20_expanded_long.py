"""E2第20轮/f68零更新起点：单作业内扩样、双卡等效batch=2、分阶段长训及条件验收。"""
import argparse
import copy
import datetime
import functools
import hashlib
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from tools import e2_b_init_epoch20_ab as init

tail, ft, ab, roi, fid, rules, guard = init.tail, init.ft, init.ab, init.roi, init.fid, init.rules, init.guard
RUN = 'b_f68_e2_epoch20_expanded_long_20261124'
START_SHA = '5fd0f126b7ce6d9b70b8fc28cfeec2e8303dafbc4d5ba4e1505dc661cc1c5320'
SOURCE_PROTOCOL = 'ae2fbbfd1a8fd26ca7a1a879a8416377648262447d2e39cf29e088f6bcc11ea3'
TRAIN_N, VAL_N, BATCH, LR, SCALE = 2048, 256, 2, 5e-6, 128
PHASES, PERIOD = (1024, 2048, 4096, 8192), 256
EARLY = (32,64,128)
FIXED = ('off', 'on', 'blend', 'offline25', 'start')
SALT = 'e2-epoch20-f68-expanded-v1:'
MANIFEST_SHA = {
    'training':'cffbd62c2bb3dbdc6635c0ec7dff63843d95a8e3f828423f406774dde9584c4e',
    'validation':'a63790ee10e2b6c511fd0ec1f788281933f3d1774d33172106cb27533be171f7',
}


def paths(prior, device='cuda:0'):
    args = argparse.Namespace(prior=prior, project=prior.parents[1], device=device, action='train')
    args.data = args.project.parent/'datasets/BF'
    args.baseline = prior/tail.distill.BASE
    args.source = args.baseline/init.RUN
    args.run = args.out = args.baseline/RUN
    args.checkpoint = args.project/'output_eval/e2_state_b_pastel_followup_20261115/trust_retry/training/step1.pt'
    args.base_checkpoint = args.project/init.BASE
    args.parent = args.baseline/'tail6_real_ab_e1_epoch10_20261119'
    args.a_checkpoint = args.parent/'A/step128.pt'
    args.light_run = args.parent/'a_fused25_lstar_ab_20261121'
    # 共用head_for会先构造全部候选路径；补齐接口字段，不把B64用作本轮起点。
    args.b64_checkpoint = args.light_run/'B/step64.pt'
    args.start = args.source/'f68/step0.pt'
    args.formal64 = args.light_run/'fid_regions_64_20261121/retention_125_250'
    args.formal500 = args.light_run/'retention_band_rms475_independent_20261121/eval500'
    args.protocol = tail.read(args.baseline/'protocol.json')
    return args


def read_checked(path, digest):
    assert tail.sha(path)==digest, str(path)
    return tail.read(path)


def root(pair):
    return ft.independent.product(pair['gt'])


def draw(rows, category, used, number, split):
    candidates = [r for r in rows if tail.distill.caption_category(r['caption'])==category]
    candidates.sort(key=lambda r:hashlib.sha256((SALT+split+':'+r['sample_id']).encode()).hexdigest())
    chosen = []
    for row in candidates:
        product = ft.independent.product(row['cloth'])
        assert product and all(ft.independent.product(row[k])==product for k in ('sketch','texture'))
        if product in used:
            continue
        used.add(product)
        chosen.append(dict(case_id=category+'_'+Path(row['cloth']).stem, sample_id=row['sample_id'],
            category=category, sketch=row['sketch'], texture=row['texture'], gt=row['cloth'],
            caption=row['caption'], seed=42, split=split, product=product))
        if len(chosen)==number:
            break
    assert len(chosen)==number, (category,split,len(chosen),number)
    return chosen


def materialize(args, pair, source=None):
    pair = dict(pair)
    case = args.run/'cases'/pair['case_id']
    case.mkdir(parents=True,exist_ok=True)
    digest = {k:tail.sha(args.data/pair[k]) for k in ('sketch','texture','gt')}
    assert all(digest[k]==v for k,v in pair.get('input_sha256',{}).items())
    pair['input_sha256'] = digest
    for name in ('mask.png','original_ref.png'):
        path = case/name
        if source:
            original = source/'cases'/pair['case_id']/name
            expected = pair['mask_sha256'] if name=='mask.png' else pair['reference_hashes'][tail.CONDITION]
            assert tail.sha(original)==expected
            rules.link(path,original)
        elif not path.exists():
            if name=='original_ref.png':
                tail.rgb(args.data/pair['texture']).resize(tail.SIZE,Image.Resampling.BICUBIC).save(path)
            else:
                mask = tail.distill.prepare_evaluation_masks(tail.SIZE,
                    sketch_path=str(args.data/pair['sketch']),mask_policy='sketch_only')['garment']
                assert mask is not None and mask.any(), pair['case_id']
                Image.fromarray(mask.astype('uint8')*255).save(path)
    pair['mask_sha256'] = tail.sha(case/'mask.png')
    pair['reference_hashes'] = {tail.CONDITION:tail.sha(case/'original_ref.png')}
    return pair


def prepare(args):
    args.run.mkdir(parents=True,exist_ok=True)
    if (args.run/'protocol.json').exists():
        return
    old = init.setup(args.prior)
    assert old.spec_sha==SOURCE_PROTOCOL
    assert tail.read(args.source/'completion.json')['frozen_unchanged']
    assert tail.sha(args.start)==START_SHA
    assert tail.sha(args.base_checkpoint)==init.BASE_SHA
    assert tail.sha(args.a_checkpoint)==init.A_SHA
    assert tail.metric_assets()==args.protocol['metric_assets']
    manifest = args.base_checkpoint.parents[4]/'training_manifest.json'
    base_protocol = tail.read(manifest.with_name('protocol.json'))
    base_train = {ft.independent.product(r['cloth']) for r in read_checked(manifest,base_protocol['train_sha256'])}
    official = args.project/'data/processed/bf_full_audit_v1'
    files = {n:official/(n+'_clean.json') for n in ('training','validation')}
    source_rows = {n:read_checked(p,MANIFEST_SHA[n]) for n,p in files.items()}
    # 训练保留原128对以复用；新验证来自官方validation，排除底座训练及历史B试验身份。
    history = {}
    blocked = set()
    for location in (args.prior,args.checkpoint.parents[2]):
        for path in location.rglob('protocol.json'):
            if 'source' in path.parts or RUN in path.parts:
                continue
            members = set(ft.independent.seen(tail.read(path)))
            if members:
                history[str(path)] = dict(sha256=tail.sha(path),products=len(members))
                blocked.update(members)
    test = [dict(p,split='test64') for p in old.test]
    expanded = [dict(p,split='test500') for p in tail.read(args.formal500/'protocol.json')['pairs']]
    training = [dict(p,split='train') for p in old.trainval['pairs'] if p['split']=='train']
    old_val = [dict(p,split='qualification') for p in old.trainval['pairs'] if p['split']=='val']
    qualification = [next(p for p in old_val if p['category']==c) for c in ('dress','outwear')]
    reserved = blocked | {root(p) for p in (*training,*old_val,*test,*expanded)}
    used = reserved | base_train
    validation = []
    for category in tail.distill.CATEGORIES:
        validation.extend(draw(source_rows['validation'],category,used,VAL_N//4,'val'))
    used = reserved | {root(p) for p in validation}
    for category in tail.distill.CATEGORIES:
        count = sum(p['category']==category for p in training)
        training.extend(draw(source_rows['training'],category,used,TRAIN_N//4-count,'train'))
    assert len(training)==TRAIN_N and len(validation)==VAL_N
    groups = [set(root(p) for p in group) for group in (training,validation,test,expanded)]
    assert all(len(group)==n for group,n in zip(groups,(TRAIN_N,VAL_N,64,500)))
    assert all(not a&b for i,a in enumerate(groups) for b in groups[i+1:])
    assert not groups[1]&base_train and not groups[2]&base_train and not groups[3]&base_train
    source_cases = {p['case_id'] for p in old.pairs}
    pairs = []
    for pair in (*training,*validation,*test,*expanded,*qualification):
        source = args.source if pair['case_id'] in source_cases else (args.formal500.parent if pair['split']=='test500' else None)
        pairs.append(materialize(args,pair,source))
    code = dict(tail.read(args.source/'implementation.json')['code'])
    repository = Path(__file__).resolve().parents[1]
    for name in ('tools/e2_b_epoch20_expanded_long.py','scripts/e2_b_epoch20_expanded_long.sbatch'):
        code[name] = tail.sha(repository/name)
    formal = {}
    for n,path in (('64',args.formal64),('500',args.formal500)):
        hashes = tail.read(path/'review_manifest.json')
        for name in ('result.json','per_pair.json','pool3.npz'):
            assert tail.sha(path/name)==hashes[name]
        formal[n] = {name:tail.sha(path/name) for name in ('result.json','per_pair.json','pool3.npz')}
    spec = dict(base_path=str(args.base_checkpoint),base_sha256=init.BASE_SHA,f68_sha256=tail.CHECKPOINT_SHA,
        frozen_A_sha256=init.A_SHA,start_sha256=START_SHA,initialization='f68 zero updates from audited127009',
        source_protocol_sha256=SOURCE_PROTOCOL,pairs=pairs,gt_sha256={p['case_id']:p['input_sha256']['gt'] for p in pairs},
        train_pairs=TRAIN_N,validation_pairs=VAL_N,retained_training_pairs=128,
        base_train_val_overlap=0,product_disjoint=True,base_train_manifest_sha256=base_protocol['train_sha256'],
        input_manifests={n:dict(path=str(p),sha256=tail.sha(p)) for n,p in files.items()},
        selection_salt=SALT,excluded_products=len(blocked),history=history,
        seed=42,sampler=args.protocol['sampler'],metric_assets=args.protocol['metric_assets'],
        lr=LR,optimizer='fresh AdamW, default betas and weight_decay',effective_batch=BATCH,world_size=2,
        phases=list(PHASES),validation_period=PERIOD,early_validation_steps=list(EARLY),
        maximum_updates=PHASES[-1],trainable=['blocks.1','output'],
        trainable_parameters=tail.PARAMETERS,tail_indices=list(range(44,50)),retention=.125,
        loss='unchanged L0 + G + 0.20 E on final uint8-STE fused image',
        common_anchor='own f68 zero-update final fused image on E2 epoch20',
        fusion='unchanged background recovery,8px edge,Gaussian sigma2,donor .25,uniform .125',
        validation_selection='relative tradeoff versus own start only; include zero fallback; no test absolute gate on validation',
        continuation='phase best must be a new accepted checkpoint and exceed preceding best net benefit',
        final_acceptance='original seven and texture; tradeoff vs own start AND previous E1 formal version',
        tolerance=rules.TOLERANCE,gain_ratio=rules.GAIN_RATIO,formal_reference=formal,
        test64='reused selection/regression set; not independent',
        test500='previous independent500, now historical regression; no new independent claim',code=code)
    tail.write(args.run/'protocol.json',spec)
    print('EXPANDED_PREPARED',TRAIN_N,VAL_N,'retained_train128','GPU_test500_deferred',flush=True)


def configure(args):
    args.spec = tail.read(args.run/'protocol.json')
    args.spec_sha = tail.sha(args.run/'protocol.json')
    for name,digest in args.spec['code'].items():
        assert tail.sha(Path(__file__).resolve().parents[1]/name)==digest, name
    args.pairs = args.spec['pairs']
    args.gt_hashes = args.spec['gt_sha256']
    args.old_spec = tail.read(args.source/'protocol.json')
    assert tail.sha(args.source/'protocol.json')==SOURCE_PROTOCOL
    args.old_pairs = {p['case_id']:p for p in args.old_spec['pairs']}
    return args


def compatible(args,pair):
    old = args.old_pairs.get(pair['case_id'])
    if old is None:
        return False
    assert all(pair[k]==old[k] for k in ('sketch','texture','gt','caption','seed','category','mask_sha256','reference_hashes'))
    assert all(pair['input_sha256'][k]==old['input_sha256'][k] for k in ('sketch','texture'))
    assert pair['input_sha256']['gt']==args.old_spec['gt_sha256'][pair['case_id']]
    return True


def reuse_image(args,pair,relative,expected):
    path,source = args.run/relative,args.source/relative
    if path.with_suffix('.json').exists() or not compatible(args,pair) or not source.exists():
        return
    meta = source.with_suffix('.json')
    record = tail.read(meta)
    assert record['protocol_sha256']==SOURCE_PROTOCOL and tail.sha(source)==record['png_sha256']
    assert all(record[k]==v for k,v in expected.items())
    rules.link(path,source)
    tail.write(path.with_suffix('.json'),dict(record,protocol_sha256=args.spec_sha,reused_source=str(source)))


def prefix(args,models,pair):
    path = args.run/'prefix'/(pair['case_id']+'.pt')
    if not path.exists() and compatible(args,pair):
        source = args.source/'prefix'/path.name
        fixed = tail.read(args.source/'fixed'/pair['case_id']/'record.json')
        assert tail.sha(source)==fixed['prefix_sha256']
        state = torch.load(source,map_location='cpu',weights_only=True)
        assert state['protocol_sha256']==SOURCE_PROTOCOL and state['case']==pair['case_id']
        tail.save_torch(path,dict(state,protocol_sha256=args.spec_sha,reused_source_sha256=tail.sha(source)))
    return init.prefix(args,models,pair)


class FixedImages:
    def __init__(self,args):
        self.args = args

    @functools.lru_cache(maxsize=12)
    def record(self,case):
        record = tail.read(self.args.run/'fixed'/case/'record.json')
        assert record['protocol_sha256']==self.args.spec_sha
        return record

    def image(self,case,name):
        path = self.args.run/'fixed'/case/(name+'.png')
        return fid.checked_png(path,self.record(case)['pngs'][name])

    def __getitem__(self,case):
        return self.image(case,'blend'),self.image(case,'offline25')


class StartImages:
    def __init__(self,fixed):
        self.fixed = fixed

    def __getitem__(self,case):
        return self.fixed.image(case,'start')


class References:
    def __init__(self,args):
        self.args = args

    @functools.lru_cache(maxsize=16)
    def __getitem__(self,case):
        result = tail.read(self.args.run/'fixed'/case/'reference.json')
        assert result['protocol_sha256']==self.args.spec_sha
        return result


@torch.no_grad()
def cache_one(args,models,pair,a,start,base):
    folder = args.run/'fixed'/pair['case_id']
    marker = folder/'record.json'
    prefix(args,models,pair)
    if marker.exists():
        record = tail.read(marker)
        assert record['protocol_sha256']==args.spec_sha
        assert record['prefix_sha256']==tail.sha(args.run/'prefix'/(pair['case_id']+'.pt'))
        for name,digest in record['pngs'].items():
            assert tail.sha(folder/(name+'.png'))==digest
    else:
        for arm in ('off','on'):
            reuse_image(args,pair,Path('raw')/arm/(pair['case_id']+'.png'),dict(case=pair['case_id'],arm=arm))
        for head in (a,start):
            digest = init.head_hash(head)
            reuse_image(args,pair,Path('raw')/digest/(pair['case_id']+'.png'),dict(case=pair['case_id'],head_sha256=digest))
        images = init.controls(args,models,pair)
        mask = tail.case_mask(args,pair)
        images['blend'] = tail.mix_pixels(images['off'],images['on'],mask,{'blend':.25})['blend']
        images['offline25'] = ft.fused.pixels_mixed(init.raw_cached(args,models,a,pair),images['blend'],ft.fused.alpha(mask))
        current = ft.fused.pixels_mixed(init.raw_cached(args,models,start,pair),images['blend'],ft.fused.alpha(mask))
        images['start'] = ft.uniform_mix(current,images['offline25'],mask)
        for name,image in images.items():
            ft.fused.save_png(folder/(name+'.png'),image)
        tail.write(marker,dict(protocol_sha256=args.spec_sha,
            prefix_sha256=tail.sha(args.run/'prefix'/(pair['case_id']+'.pt')),
            pngs={n:tail.sha(folder/(n+'.png')) for n in images}))
    path = folder/'reference.json'
    if not path.exists():
        record = tail.read(marker)
        image = fid.checked_png(folder/'start.png',record['pngs']['start'])
        prediction = guard.anchor_tensor(image,torch.empty((),device=args.device))
        target = roi.tensors([np.asarray(tail.rgb(args.data/pair['gt']).resize(tail.SIZE,Image.Resampling.BILINEAR))]).to(args.device)
        loss,terms = ab.collect(base,prediction,target,tail.case_mask(args,pair),'inside')
        tail.write(path,dict(protocol_sha256=args.spec_sha,png_sha256=record['pngs']['start'],loss=float(loss),**terms))
    models[2].adapter.clear()


def cache_pairs(args,models,pairs,base,rank,world):
    a,_ = init.head_for(args,models,'A')
    start,_ = init.head_for(args,models,'f68')
    a.requires_grad_(False)
    start.requires_grad_(False)
    for i,pair in enumerate(pairs[rank::world],1):
        cache_one(args,models,pair,a,start,base)
        if i%8==0:
            print('EXPANDED_CACHE',rank,i,len(pairs[rank::world]),pair['case_id'],flush=True)
    del a,start


def qualify(args,models,base,fixed,criterion,rank,world):
    path = args.run/'qualification.json'
    if path.exists():
        assert tail.read(path)['protocol_sha256']==args.spec_sha
        return
    probes = [p for p in args.pairs if p['split']=='qualification']
    pair = probes[rank]
    cache_pairs(args,models,[pair],base,0,1)
    head,parameters = init.head_for(args,models,'f68')
    core,pipe,net,original,_ = models
    mask = tail.case_mask(args,pair)
    values,_ = tail.joint.data.inputs(args,pair,tail.CONDITION,'raw')
    features = tail.joint.projected(net,core.tensor(args.data/pair['sketch']).to(args.device).half(),
        core.tensor(args.out/'cases'/pair['case_id']/'original_ref.png').to(args.device).half())
    module = tail.TailResidual(original,head,tail.latent_gate(mask,args.device))
    with torch.no_grad(),tail.PrefixTap(pipe,module,values,end=tail.LAST,qualify=True) as tap:
        picture,_ = tail.joint.image(args,core,pipe,net,pair,features)
    state = prefix(args,models,pair)
    assert all(torch.equal(state[n],tap.start[n]) for n in ('z','epsilon'))
    with torch.no_grad():
        latent = tail.trajectory(args,models,head,pair,state,mask)
        assert torch.equal(latent.cpu(),tap.final)
        np.testing.assert_array_equal(ft.rgb8(tail.decode(pipe,latent)),np.asarray(picture))
    prediction,target,mask = init.forward(args,models,head,pair,fixed)
    np.testing.assert_array_equal(ft.rgb8(prediction),fixed.image(pair['case_id'],'start'))
    loss,terms = criterion(prediction,target,mask,pair)
    assert float(terms['perceptual_guard'].detach())==float(terms['edge_anchor'].detach())==0.
    (loss*SCALE/BATCH).backward()
    for parameter in parameters.values():
        dist.all_reduce(parameter.grad)
        parameter.grad.div_(SCALE)
    averaged = {n:p.grad.detach().cpu().clone() for n,p in parameters.items()}
    norm = float(torch.linalg.vector_norm(torch.cat([g.flatten() for g in averaged.values()])))
    assert np.isfinite(norm) and norm>0
    for p in parameters.values():
        p.grad = None
    models[2].adapter.clear()
    del prediction,target,loss
    dist.barrier()
    # 在更新前核对双卡平均梯度与单卡累积，避免扩卡悄悄改变有效batch或梯度缩放。
    if rank==0:
        for probe in probes:
            prediction,target,mask = init.forward(args,models,head,probe,fixed)
            loss,_ = criterion(prediction,target,mask,probe)
            (loss*SCALE/world).backward()
            models[2].adapter.clear()
            del prediction,target,loss
        maximum = 0.
        for name,p in parameters.items():
            value = p.grad.detach().cpu()/SCALE
            torch.testing.assert_close(value,averaged[name],atol=1e-6,rtol=1e-4)
            maximum = max(maximum,float((value-averaged[name]).abs().max()))
            p.grad = None
        ab.audit_frozen(args,models,head,parameters)
        tail.write(path,dict(protocol_sha256=args.spec_sha,optimizer_updates=0,exact_prefix=True,
            exact_latent=True,exact_raw_pixels=True,exact_fused_pixels=True,guard_at_start=0.,edge_at_start=0.,
            gradient_norm=norm,distributed_vs_accumulated_max_error=maximum,effective_batch=2,
            trainable_parameters=tail.PARAMETERS,active_end=tail.LAST,qualification_cases=[p['case_id'] for p in probes]))
        print('EXPANDED_QUALIFIED',norm,maximum,flush=True)
    dist.barrier()
    del head


def summarize(rows,features,names):
    labels = np.array([r['category'] for r in rows])
    groups = {}
    for name in ('all',*tail.distill.CATEGORIES):
        keep = np.ones(len(rows),dtype=bool) if name=='all' else labels==name
        chosen = [r for r,k in zip(rows,keep) if k]
        arms = {}
        for arm in names:
            metrics = {}
            for key in tail.METRICS:
                values = np.asarray([r['scores'][arm][key] for r in chosen],dtype='float64')
                metrics[key] = dict(n=int(np.isfinite(values).sum()),mean=float(np.nanmean(values)))
            arms[arm] = dict(metrics,fid=roi.fid_terms(features[arm][keep],features['gt'][keep]))
        groups[name] = dict(n=len(chosen),arms=arms)
    result = dict(groups=groups)
    ft.class_ab.summarize_texture(result,rows,(*names,'gt'))
    return result


@torch.no_grad()
def render(args,models,pairs,head,fixed,rank,world):
    for pair in pairs[rank::world]:
        init.final_cached(args,models,head,pair,fixed)
        models[2].adapter.clear()


def reused64(args,pair,name,image):
    if pair['split']!='test64':
        return None
    if not hasattr(args,'source64'):
        rows = tail.read(args.source/'eval64/per_pair.json')
        with np.load(args.source/'eval64/pool3.npz',allow_pickle=False) as pool:
            args.source64 = {r['case_id']:(r,{n:pool[n][i] for n in pool.files}) for i,r in enumerate(rows)}
    row,features = args.source64[pair['case_id']]
    source_name = 'f68_zero' if name=='start' else name
    if name=='gt':
        return dict(feature=features['gt'],texture=row['texture']['gt'])
    if name=='candidate' or source_name not in row['scores']:
        return None
    assert row['pngs'][source_name]==image['png_sha256']
    return dict(scores=row['scores'][source_name],texture=row['texture'][source_name],feature=features[source_name])


@torch.no_grad()
def score(args,models,pairs,head,fixed,label,metrics):
    out = args.run/label
    result_path = out/'result.json'
    digest = init.head_hash(head)
    if result_path.exists():
        result = tail.read(result_path)
        assert result['protocol_sha256']==args.spec_sha and result['head_sha256']==digest
        assert tail.sha(out/'pool3.npz')==result['pool3_sha256']
        return result
    out.mkdir(parents=True,exist_ok=True)
    names = (*FIXED,'candidate')
    rows,features = [],{n:[] for n in (*names,'gt')}
    lpips,inception = metrics
    # 逐PNG摘要缓存观测；验证起点及相同图的评分可跨选点复用。
    for start in range(0,len(pairs),4):
        batch = pairs[start:start+4]
        targets = [np.asarray(tail.rgb(args.data/p['gt']).resize(tail.SIZE,Image.Resampling.BILINEAR)) for p in batch]
        target_tensor = roi.tensors(targets)
        gt_feature = inception(target_tensor)[0].flatten(1).numpy()
        records = []
        for i,pair in enumerate(batch):
            mask = tail.case_mask(args,pair)
            inside = roi.regions(mask,20)[0]
            old = reused64(args,pair,'gt',{})
            feature = old['feature'] if old else gt_feature[i]
            if old:
                np.testing.assert_allclose(gt_feature[i],feature,atol=7e-5,rtol=1e-4)
            features['gt'].append(feature)
            records.append(dict(case_id=pair['case_id'],category=pair['category'],interior_px=int(inside.sum()),
                pngs={},scores={},texture={'gt':tail.appearance(targets[i],roi.lab(targets[i]),inside)}))
        for name in names:
            observations,unknown,pixels = {},[],[]
            for i,pair in enumerate(batch):
                if name=='candidate':
                    image,path = init.final_cached(args,models,head,pair,fixed)
                else:
                    path = args.run/'fixed'/pair['case_id']/(name+'.png')
                    image = fixed.image(pair['case_id'],name)
                png_sha = tail.sha(path)
                pixels.append(image)
                key = args.run/'observations'/pair['case_id']/png_sha
                record_path,feature_path = key.with_suffix('.json'),key.with_suffix('.npz')
                source_name = 'start' if name=='candidate' and png_sha==fixed.record(pair['case_id'])['pngs']['start'] else name
                prior = reused64(args,pair,source_name,dict(png_sha256=png_sha))
                if record_path.exists():
                    observation = tail.read(record_path)
                    assert observation['protocol_sha256']==args.spec_sha and observation['png_sha256']==png_sha
                    assert observation['features_sha256']==tail.sha(feature_path)
                    with np.load(feature_path,allow_pickle=False) as f:
                        observation['feature'] = f['prediction']
                    observations[i] = observation
                elif prior:
                    observations[i] = prior
                else:
                    unknown.append((i,pair,image,path,record_path,feature_path,png_sha))
                records[i]['pngs'][name] = png_sha
            if unknown:
                # 缺一张评分也沿用完整四张批次，固定CPU卷积的数值路径。
                tensor = roi.tensors(pixels)
                distances = lpips(tensor*2-1,target_tensor*2-1).flatten().tolist()
                values = inception(tensor)[0].flatten(1).numpy()
                for j,(i,pair,image,path,record_path,feature_path,png_sha) in enumerate(unknown):
                    mask = tail.case_mask(args,pair)
                    score_value = tail.image_scores(Image.fromarray(image),Image.fromarray(targets[i]),path,args.data/pair['gt'],mask)
                    score_value['lpips'] = distances[i]
                    observation = dict(protocol_sha256=args.spec_sha,png_sha256=png_sha,scores=score_value,
                        texture=tail.appearance(image,roi.lab(targets[i]),roi.regions(mask,20)[0]))
                    feature_path.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(feature_path,prediction=values[i])
                    tail.write(record_path,dict(observation,features_sha256=tail.sha(feature_path)))
                    observations[i] = dict(observation,feature=values[i])
            for i in range(len(batch)):
                observation = observations[i]
                records[i]['scores'][name] = observation['scores']
                records[i]['texture'][name] = observation['texture']
                features[name].append(observation['feature'])
        rows.extend(records)
        if len(rows)%32==0:
            print('EXPANDED_SCORED',label,len(rows),len(pairs),flush=True)
    features = {n:np.stack(v) for n,v in features.items()}
    assert all(v.shape==(len(pairs),2048) for v in features.values())
    result = summarize(rows,features,names)
    result.update(protocol_sha256=args.spec_sha,head_sha256=digest,pairs=len(pairs))
    np.savez_compressed(out/'pool3.npz',**features)
    result['pool3_sha256'] = tail.sha(out/'pool3.npz')
    tail.write(out/'per_pair.json',rows)
    tail.write(result_path,result)
    return result


def formal_acceptance(args,result,label,selected):
    source = args.formal64 if label=='eval64' else args.formal500
    manifest = args.spec['formal_reference']['64' if label=='eval64' else '500']
    previous = read_checked(source/'result.json',manifest['result.json'])
    rows = read_checked(source/'per_pair.json',manifest['per_pair.json'])
    current = tail.read(args.run/label/'per_pair.json')
    assert [r['case_id'] for r in rows]==[r['case_id'] for r in current]
    assert tail.sha(source/'pool3.npz')==manifest['pool3.npz']
    with np.load(source/'pool3.npz',allow_pickle=False) as old, np.load(args.run/label/'pool3.npz',allow_pickle=False) as new:
        np.testing.assert_allclose(old['gt'],new['gt'],atol=7e-5,rtol=1e-4)
    group = result['groups']['all']
    value = init.metrics_from_group(group,'candidate')
    own = rules.tradeoff(value,init.metrics_from_group(group,'start'))
    inherited = rules.tradeoff(value,init.metrics_from_group(previous['groups']['all'],'retain_125'))
    absolute = init.gates(group,'candidate')
    passed = selected>0 and absolute['passed'] and own['accepted'] and inherited['accepted']
    return dict(absolute=absolute,own_start=own,inherited_formal=inherited,
        selected=selected,accepted=bool(passed),reference='previous E1 formal; 0.5% budget never reset')


def distributed_validation(args,models,pairs,head,fixed,label,metrics,rank,world):
    render(args,models,pairs,head,fixed,rank,world)
    dist.barrier()
    result = score(args,models,pairs,head,fixed,label,metrics) if rank==0 else None
    dist.barrier()
    return result


def train(args,models,base,fixed,criterion,metrics,rank,world):
    head,params = init.head_for(args,models,'f68')
    assert tail.sha(args.start)==START_SHA
    ab.update_head(head,params,args.start)
    training = [p for p in args.pairs if p['split']=='train']
    validation = [p for p in args.pairs if p['split']=='val']
    resume = args.run/'resume.pt'
    optimizer = torch.optim.AdamW(params.values(),lr=LR)
    completed,trace,checkpoints,phases = 0,[],{},[]
    if resume.exists():
        state = torch.load(resume,map_location='cpu',weights_only=False)
        assert state['protocol_sha256']==args.spec_sha
        rules.restore(params,state['head'])
        optimizer.load_state_dict(state['optimizer'])
        completed,trace,checkpoints,phases = (state[n] for n in ('step','trace','validation','phases'))
    optimizer.zero_grad(set_to_none=True)

    def save():
        if rank==0:
            tail.save_torch(resume,dict(protocol_sha256=args.spec_sha,step=completed,
                head={n:p.detach().cpu() for n,p in params.items()},optimizer=optimizer.state_dict(),
                trace=trace,validation=checkpoints,phases=phases))

    def validate(step):
        if rank==0:
            tail.save_torch(args.run/f'step{step}.pt',{n:p.detach().cpu() for n,p in params.items()})
        dist.barrier()
        result = distributed_validation(args,models,validation,head,fixed,f'validation/step{step}',metrics,rank,world)
        if rank==0:
            checkpoints[str(step)] = init.metrics_from_group(result['groups']['all'],'candidate')
            checkpoints[str(step)]['sha256'] = tail.sha(args.run/f'step{step}.pt')
            print('EXPANDED_VALIDATION',step,checkpoints[str(step)],flush=True)
        boxes = [checkpoints if rank==0 else None]
        dist.broadcast_object_list(boxes,src=0)
        snapshot = dict(boxes[0])
        checkpoints.clear()
        checkpoints.update(snapshot)
        save()

    if '0' not in checkpoints:
        assert completed==0
        validate(0)
    if completed and (completed%PERIOD==0 or completed in EARLY) and str(completed) not in checkpoints:
        validate(completed)
    boundary = 0
    for goal in PHASES:
        earlier = next((p for p in phases if p['end']==goal),None)
        if earlier:
            boundary = goal
            if not earlier['continue_training']:
                break
            continue
        previous_best,previous_scores = rules.choose({s:c for s,c in checkpoints.items() if int(s)<=boundary})
        for step in range(completed+1,goal+1):
            epoch,position = divmod(step-1,TRAIN_N//BATCH)
            order = np.random.default_rng(42+epoch).permutation(TRAIN_N)
            pair = training[order[position*BATCH+rank]]
            prediction,target,mask = init.forward(args,models,head,pair,fixed)
            loss,terms = guard.collect(criterion,prediction,target,mask,pair)
            assert torch.isfinite(loss)
            total = float(loss.detach())
            # 与原单卡累积2次使用相同的FP16反传缩放64；双卡求和后再除128。
            (loss*SCALE/BATCH).backward()
            models[2].adapter.clear()
            del loss,prediction,target
            assert all(p.grad is not None for p in params.values())
            for p in params.values():
                dist.all_reduce(p.grad)
                p.grad.div_(SCALE)
            norm = float(torch.nn.utils.clip_grad_norm_(params.values(),1,error_if_nonfinite=True))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            row = dict(case=pair['case_id'],epoch=epoch+1,loss=total,**terms)
            gathered = [None]*world
            dist.all_gather_object(gathered,row)
            if rank==0:
                trace.append(dict(step=step,grad_norm=norm,rows=gathered))
            completed = step
            save()
            if rank==0 and step%16==0:
                print('EXPANDED_TRAIN',step,goal,gathered,flush=True)
            if step%PERIOD==0 or step in EARLY:
                validate(step)
        selected,scores = rules.choose(checkpoints)
        new_best = selected>boundary and scores[str(selected)]['net']>previous_scores[str(previous_best)]['net']
        continued = bool(new_best and goal<PHASES[-1])
        phases.append(dict(start=boundary,end=goal,selected=selected,previous_best=previous_best,
            net=scores[str(selected)]['net'],new_best=bool(new_best),continue_training=continued))
        ab.audit_frozen(args,models,head,params)
        save()
        if rank==0:
            print('EXPANDED_PHASE',phases[-1],flush=True)
        boundary = goal
        if not continued:
            break
    selected,scores = rules.choose(checkpoints)
    ab.update_head(head,params,args.run/f'step{selected}.pt')
    digest = init.head_hash(head)
    hashes = [None]*world
    dist.all_gather_object(hashes,digest)
    assert len(set(hashes))==1
    ab.audit_frozen(args,models,head,params)
    if rank==0:
        tail.write(args.run/'training.json',dict(protocol_sha256=args.spec_sha,updates=completed,selected=selected,
            validation=checkpoints,tradeoff=scores,phases=phases,trace=trace,parameter_count=tail.PARAMETERS,
            trainable=list(params),frozen_unchanged=True,final_head_sha256=digest))
    return head.eval().requires_grad_(False),selected,completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prior',type=Path)
    parser.add_argument('--prepare-only',action='store_true')
    ns = parser.parse_args()
    if ns.prepare_only:
        cv2.setNumThreads(1)
        prepare(paths(ns.prior))
        return
    rank,world,local = (int(os.environ[n]) for n in ('RANK','WORLD_SIZE','LOCAL_RANK'))
    assert world==BATCH==2
    torch.cuda.set_device(local)
    dist.init_process_group('nccl',timeout=datetime.timedelta(hours=4))
    cv2.setNumThreads(1)
    torch.set_num_threads(6)
    args = paths(ns.prior,f'cuda:{local}')
    if rank==0:
        prepare(args)
    dist.barrier()
    configure(args)
    if (args.run/'completion.json').exists() and (args.run/'review.zip').exists():
        if rank==0:
            print('EXPANDED_ALREADY_COMPLETE',tail.sha(args.run/'review.zip'),flush=True)
        dist.destroy_process_group()
        return
    # 先由rank0写冻结模型记录，其余rank核对；资格检查在大批缓存之前进行。
    models = init.load(args) if rank==0 else None
    dist.barrier()
    if rank!=0:
        models = init.load(args)
    dist.barrier()
    torch.set_num_threads(6)
    base = ft.light.LightingObjective(args)
    fixed = FixedImages(args)
    criterion = guard.Objective(base,References(args),StartImages(fixed))
    qualify(args,models,base,fixed,criterion,rank,world)
    trainval = [p for p in args.pairs if p['split'] in ('train','val')]
    cache_pairs(args,models,trainval,base,rank,world)
    dist.barrier()
    metrics = tail.metric_models('cpu',args.protocol['metric_assets']) if rank==0 else None
    head,selected,updates = train(args,models,base,fixed,criterion,metrics,rank,world)
    test = [p for p in args.pairs if p['split']=='test64']
    cache_pairs(args,models,test,base,rank,world)
    dist.barrier()
    result = distributed_validation(args,models,test,head,fixed,'eval64',metrics,rank,world)
    if rank==0:
        acceptance = formal_acceptance(args,result,'eval64',selected)
        tail.write(args.run/'eval64/acceptance.json',acceptance)
        print('EXPANDED_TEST64',acceptance,flush=True)
    boxes = [acceptance['accepted'] if rank==0 else None]
    dist.broadcast_object_list(boxes,src=0)
    expanded = bool(boxes[0])
    if expanded:
        pairs = [p for p in args.pairs if p['split']=='test500']
        cache_pairs(args,models,pairs,base,rank,world)
        dist.barrier()
        result = distributed_validation(args,models,pairs,head,fixed,'eval500',metrics,rank,world)
        if rank==0:
            acceptance500 = formal_acceptance(args,result,'eval500',selected)
            tail.write(args.run/'eval500/acceptance.json',acceptance500)
    assert tail.tensor_set(models[4])==args.original['frozen_model_hashes']
    dist.barrier()
    if rank==0:
        tail.write(args.run/'completion.json',dict(updates=updates,selected=selected,
            eval64_passed=expanded,eval500_executed=expanded,
            eval500_passed=acceptance500['accepted'] if expanded else None,frozen_unchanged=True))
        files = ['protocol.json','model.json','environment.json','execution_environment.json','qualification.json',
                 'training.json','completion.json','eval64/result.json','eval64/per_pair.json',
                 'eval64/pool3.npz','eval64/acceptance.json']
        for path in sorted((args.run/'validation').glob('step*/result.json')):
            files.extend(str((path.parent/n).relative_to(args.run)) for n in ('result.json','per_pair.json','pool3.npz'))
        if expanded:
            files.extend('eval500/'+n for n in ('result.json','per_pair.json','pool3.npz','acceptance.json'))
        fid.pack(args.run,files)
        print('EXPANDED_FINISHED',updates,selected,expanded,tail.sha(args.run/'review.zip'),flush=True)
    dist.destroy_process_group()


if __name__=='__main__':
    main()
