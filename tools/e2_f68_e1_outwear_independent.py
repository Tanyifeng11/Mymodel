"""在互斥的 BF 测试身份上验证固定的衣内修正规则。"""
import argparse
from collections import Counter
import copy
import hashlib
import os
from pathlib import Path
import re
from unittest.mock import patch

import cv2
import numpy as np
import torch
from PIL import Image

from tools import e2_f68_e1_retention500 as expand
from tools import e2_f68_e1_fid_regions as fid_regions
from tools import e2_f68_e1_a_fused25_continue as fused
from tools import e2_f68_real_paired as paired
from tools import e2_f68_region_diagnostic as roi
from tools import e2_f68_tail6_e1_epoch10 as epoch
from tools import e2_f68_tail6_real_ab as tail


RUN = 'retention_outwear_independent_20261121'
SALT = 'e1-epoch10-outwear-interior-independent-v1:'
N_PER_CLASS = 125
FIXED = 'outwear_restore'
PREVIOUS_SHA = 'b5e3867e51dcf860cc1c563a2f7a21bce16ed1c28e6f61446cb6caac0fc6c42b'
BAND_RUN = 'retention_band_rms475_independent_20261121'
BAND_SALT = 'e1-epoch10-band-rms475-independent-v1:'
BAND_CANDIDATE = 'band_rms475'
BAND_RULE = dict(threshold=4.75, coefficient={'dress': .10, 'outwear': .05},
                 core_radius=20, min_core_pixels=256, feather_sigma=2, band_sigmas=[1, 4])
HOLDOUT_SHA = '8f83d9e9fa15205cd9a1e06398ddcab752c96633001e2c78e8e4c736c89953eb'
DIAGNOSTIC_SHA = '78de0b72562491ab8852972c1a713e40e6a0fe633aa2bbb2fd51b6f565ee6704'
IDENTITY = re.compile(r'^(?:(?:dress|outwear|pants|top|bag)_)?(\d{5,})(?:[_ -][\w -]+)?$')


def product(value):
    m = IDENTITY.fullmatch(Path(value).stem)
    return m.group(1) if m else None


def seen(value):
    if isinstance(value, str):
        identity = product(value)
        if identity:
            yield identity
    elif isinstance(value, list):
        for element in value:
            yield from seen(element)
    elif isinstance(value, dict):
        for key, element in value.items():
            yield from seen(key)
            yield from seen(element)


def historical(args, prior):
    entries = {}
    blocked = set()
    scope = (prior, args.checkpoint.parents[2])
    for path in (p for root in scope for p in root.rglob('protocol.json')):
        if any(d in path.parts for d in ('source', args.run.name)):
            continue
        document = tail.read(path)
        members = set(seen(document))
        if members:
            entries[str(path)] = dict(sha256=tail.sha(path), products=len(members))
            blocked.update(members)
    return blocked, entries


def select_rows(rows, excluded, n=N_PER_CLASS, salt=SALT):
    drawn = []
    used = set(excluded)
    for category in paired.CATEGORIES:
        candidates = [r for r in rows if r['sample_id'].split('/')[:2]==['test',category]]
        ordered = sorted(candidates,key=lambda r:hashlib.sha256((salt+r['sample_id']).encode()).hexdigest())
        group = []
        for row in ordered:
            key = product(row['cloth'])
            assert key and all(product(row[k])==key for k in ('sketch','texture'))
            if key in used:
                continue
            used.add(key)
            assert isinstance(row['caption'],str) and row['caption'].strip()
            group.append(dict(case_id=category+'_'+Path(row['cloth']).stem, sample_id=row['sample_id'],
                              category=category,sketch=row['sketch'],texture=row['texture'],gt=row['cloth'],
                              caption=row['caption'],seed=42,product=key))
            if len(group)==n:
                break
        assert len(group)==n,category
        drawn.extend(group)
    return drawn


def paths(prior, band_gate=False):
    args = expand.setup(prior)
    args.previous = args.run
    args.old_base = args.baseline
    args.old_protocol = args.protocol
    args.band_gate = band_gate
    args.run = args.light_run/(BAND_RUN if band_gate else RUN)
    args.baseline = args.out = args.run
    return args


def prepare(args, prior):
    args.run.mkdir(exist_ok=True)
    cohort = args.run/'protocol.json'
    if cohort.is_file():
        return
    old = args.previous/'eval500'
    tail.verified_files(old,PREVIOUS_SHA,('audit.json','per_pair.json','result.json'))
    complete = args.contract['parent_contract']
    training = args.base_checkpoint.parents[4]/'training_manifest.json'
    assert tail.sha(training)==complete['base_train_manifest']
    assert tail.sha(args.base_checkpoint)==complete['base_sha256']
    assert tail.sha(args.parent/'A/step128.pt')==args.heads['A']
    assert tail.sha(args.light_run/'B/step64.pt')==args.heads['B']
    assert tail.sha(args.light_run/'selection.json')==tail.read(args.light_run/'eval64/review_manifest.json')['selection.json']
    manifests = args.project/'data/processed/bf_full_audit_v1'
    files = {name:manifests/(name+'_clean.json') for name in ('test','training','validation')}
    assert all(tail.sha(Path(p))==v for p,v in args.old_protocol['input_manifest_sha256'].items())
    assert all(x['seed']==42 for x in args.old_protocol['pairs'])
    assert len({x['arms']['off']['initial_noise_digest'] for x in args.original_rows})==1
    train = {product(r['cloth']) for r in tail.read(files['training'])}
    train.update(product(r['cloth']) for r in tail.read(training))
    val = {product(r['cloth']) for r in tail.read(files['validation'])}
    old500 = {product(r['gt']) for r in args.old_protocol['pairs']}
    trainval_b = {product(k) for k in args.contract['prefix']}
    assert len(old500)==500 and len(trainval_b)==160 and None not in train|val|old500|trainval_b
    seen_before, history = historical(args,prior)
    diagnosed = set()
    salt = SALT
    frozen_rule = dict(outwear_restore=True)
    if args.band_gate:
        holdout = args.light_run/RUN
        tail.verified_files(holdout/'eval500',HOLDOUT_SHA,('protocol.json','result.json'))
        tail.verified_files(holdout/'band_attribution_20261121',DIAGNOSTIC_SHA,('audit.json',))
        old_cohort = tail.read(holdout/'eval500/protocol.json')
        diagnosed = {p['product'] for p in old_cohort['pairs']}
        assert len(diagnosed)==500 and not diagnosed & old500 and diagnosed <= seen_before
        salt = BAND_SALT
        frozen_rule = dict(band_gate=BAND_RULE, previous_independent_sha256=HOLDOUT_SHA,
                           diagnostic_sha256=DIAGNOSTIC_SHA, excluded_diagnosis_products=len(diagnosed))
    excluded = train|val|old500|trainval_b|seen_before|diagnosed
    pairs = select_rows(tail.read(files['test']),excluded,salt=salt)
    assert len({p['product'] for p in pairs})==len(pairs)==500
    assert not {p['product'] for p in pairs} & excluded
    assets = tail.metric_assets()
    assert assets==args.old_protocol['metric_assets']
    valid_masks = Counter()
    for pair in pairs:
        case = args.run/'cases'/pair['case_id']
        case.mkdir(parents=True,exist_ok=True)
        pair['input_sha256'] = {k:tail.sha(args.data/pair[k]) for k in ('sketch','texture','gt')}
        path = case/(tail.CONDITION+'_ref.png')
        tail.rgb(args.data/pair['texture']).resize(tail.SIZE,Image.Resampling.BICUBIC).save(path)
        pair['reference_hashes'] = {tail.CONDITION:tail.sha(path)}
        mask = paired.prepare_evaluation_masks(tail.SIZE,sketch_path=str(args.data/pair['sketch']),
                                               mask_policy='sketch_only')['garment']
        if mask is None:
            mask = np.zeros((tail.SIZE[1],tail.SIZE[0]),dtype=bool)
        path = case/'mask.png'
        Image.fromarray(mask.astype('uint8')*255).save(path)
        pair['mask_sha256'] = tail.sha(path)
        valid_masks[pair['category']] += int(mask.any())

    # 原 500 对使用的原图目录属于已排除的试验。保留任何已存在的同协议续跑图。
    roots = [x for x in args.parent.rglob('test') if x!=args.run/'test']
    cases = {p['case_id'] for p in pairs}
    collisions = []
    for path in roots:
        for arm in expand.RAW:
            directory = path/arm
            if directory.is_dir():
                collisions.extend(str(x) for x in directory.glob('*.png') if x.stem in cases)
    assert not collisions, collisions[:10]

    checkpoint = args.run/'B/step64.pt'
    checkpoint.parent.mkdir(exist_ok=True)
    if not checkpoint.exists():
        checkpoint.symlink_to(args.light_run/'B/step64.pt')
    assert tail.sha(checkpoint)==args.heads['B']
    tail.write(args.run/'selection.json',args.selection)
    launch_protocol = dict(pairs=pairs, sampler=args.old_protocol['sampler'], metric_assets=assets,
        heads=args.heads, base_sha256=complete['base_sha256'], cases_per_class=N_PER_CLASS,
        selection_hash_salt=salt, manifest_sha256={k:tail.sha(v) for k,v in files.items()},
        e1_training_sha256=tail.sha(training), previous_expansion_sha256=PREVIOUS_SHA,
        excluded=dict(previous500=len(old500),b_trainval=len(trainval_b),e1_and_official_train=len(train),
                      official_validation=len(val),other_known_protocol_products=len(seen_before),union=len(excluded)),
        history=history, retained_coefficient=.125, mask_radius=8, feather_sigma=2, **frozen_rule,
        checked_same_protocol_images_before_run=0, valid_masks=dict(valid_masks), optimizer_updates=0)
    tail.write(cohort,launch_protocol)
    print('OUTWEAR_INDEPENDENT_PREPARED',len(pairs),dict(valid_masks),len(excluded),flush=True)


def generate(args):
    # 沿用已验证的四路生成、配对噪声审计和已完成 PNG 的幂等续跑。
    with patch.object(epoch,'pair_indices',return_value=list(range(500))):
        expand.generate(args)


def generation_records(args):
    records = []
    covered = set()
    for shard in range(args.shards):
        path = args.run/f'generate500_{shard}.json'
        if path.is_file():
            record = tail.read(path)
            assert record['ids']==list(range(shard,500,args.shards))
            records.append(record)
            covered.update(record['ids'])
    remaining = sorted(set(range(500))-covered)
    consolidated = args.run/'remaining_generation.json'
    if consolidated.is_file():
        record = tail.read(consolidated)
        assert record['ids']==remaining
        records.append(record)
        remaining = []
    assert all(set(record['arms'])==set(expand.RAW) and record['frozen'] for record in records)
    return records,remaining


def complete(args):
    _,remaining = generation_records(args)
    print('OUTWEAR_INDEPENDENT_REMAINING',len(remaining),flush=True)
    if remaining:
        worker = copy.copy(args)
        worker.device,worker.shard,worker.shards = 'cuda:0',0,1
        with patch.object(epoch,'pair_indices',return_value=remaining):
            expand.generate(worker,marker_name='remaining_generation.json')
    score(args)


def fit_gate(result, candidate=FIXED):
    gate = epoch.gate_check(result,candidate)
    arms = result['groups']['all']['arms']
    fields = dict(lpips='mean',fid='fid',ssim='mean',contour_f1='mean')
    delta = {arm:{k:arms[candidate][k][field]-arms[arm][k][field] for k,field in fields.items()}
             for arm in ('offline25',expand.CANDIDATE)}
    d = delta['offline25']
    gate['delta'] = delta
    gate['passed_all'] = gate['passed'] and d['lpips']<0 and d['fid']<0 and d['ssim']>=0 and d['contour_f1']>=0
    return gate


def band_edit(image, core, category):
    from tools.e2_f68_e1_independent_band import attenuate

    rule = BAND_RULE
    if int(core.sum()) < rule['min_core_pixels']:
        return image, dict(rms=None, coefficient=0.)
    lab = roi.lab(image)
    band = roi.band(lab[..., 0])
    rms = roi.rms(band[core])
    coefficient = rule['coefficient'].get(category, 0.) if rms >= rule['threshold'] else 0.
    if coefficient:
        weight = roi.gaussian(core.astype('float32'),rule['feather_sigma'])
        image = attenuate(image,lab,band,roi.as_rgb(lab).astype('int16'),weight,coefficient)
    return image, dict(rms=rms, coefficient=coefficient)


@torch.inference_mode()
def score(args):
    dest = args.run/'eval500'
    dest.mkdir(exist_ok=True)
    if (dest/'review.zip').is_file():
        return
    markers,remaining = generation_records(args)
    assert not remaining
    lpips,inception = tail.metric_models('cpu',args.protocol['metric_assets'])
    assert tail.metric_assets()==args.protocol['metric_assets']
    candidate = BAND_CANDIDATE if args.band_gate else FIXED
    arms = (*expand.SCORED,candidate)
    features = {name:np.zeros((500,2048),dtype='float32') for name in (*arms,'gt')}
    rows = []
    for start in range(0,500,4):
        batch = list(range(start,min(start+4,500)))
        pix = {k:[] for k in arms}
        targets = []
        edited = set()
        for index in batch:
            pair,mask,images,raw = expand.checked_inputs(args,index)
            case = pair['case_id']
            target = args.data/pair['gt']
            assert tail.sha(target)==pair['input_sha256']['gt']
            gt = tail.rgb(target).resize(tail.SIZE,Image.Resampling.BILINEAR)
            gt_rgb = np.asarray(gt)
            targets.append(gt_rgb)
            gt_lab = roi.lab(gt_rgb)
            core = roi.regions(mask,20)[0]
            row = dict(index=index,case_id=case,category=pair['category'],interior_px=int(core.sum()),
                       pngs={},scores={},texture={},raw={k:v['png_sha256'] for k,v in raw.items()})
            before = images[expand.CANDIDATE]
            if args.band_gate:
                after,gate = band_edit(before,core,pair['category'])
                gate['edited_pixels'] = int(np.count_nonzero(np.any(after!=before,axis=-1)))
                row['band_gate'] = gate
                if gate['edited_pixels']:
                    images[candidate] = after
            elif pair['category']=='outwear':
                w = fid_regions.regions(mask)[0]
                images[candidate] = fused.pixels_mixed(before,images['offline25'],w)
                assert np.array_equal(images[candidate][w==0],before[w==0])
            if candidate in images:
                edited.add(index)
            for name,img in images.items():
                if name in ('off','on'):
                    path = expand.raw_path(args,name,case)
                else:
                    path = args.run/'images'/name/(case+'.png')
                    fused.save_png(path,img)
                row['pngs'][name] = tail.sha(path)
                row['scores'][name] = tail.image_scores(Image.fromarray(img),gt,path,target,mask)
                row['texture'][name] = tail.appearance(img,gt_lab,core)
                pix[name].append(img)
            if index not in edited:
                for key in ('pngs','scores','texture'):
                    row[key][candidate] = row[key][expand.CANDIDATE]
                pix[candidate].append(before)
            rows.append(row)
        truth = roi.tensors(targets)
        features['gt'][batch] = inception(truth)[0].flatten(1).numpy()
        for name in arms:
            if name==candidate and not edited:
                features[name][batch] = features[expand.CANDIDATE][batch]
                continue
            prediction = roi.tensors(pix[name])
            values = lpips(prediction*2-1,truth*2-1).flatten().tolist()
            features[name][batch] = inception(prediction)[0].flatten(1).numpy()
            for i,value in zip(batch,values):
                if name==candidate and i not in edited:
                    assert np.allclose(features[name][i],features[expand.CANDIDATE][i],rtol=1e-5,atol=2e-5)
                    features[name][i] = features[expand.CANDIDATE][i]
                else:
                    rows[i]['scores'][name]['lpips'] = value
        if start%20==0 or len(rows)==500:
            print('OUTWEAR_INDEPENDENT_SCORED',len(rows),500,flush=True)
    assert len(rows)==500 and all(np.isfinite(v).all() for v in features.values())
    result = tail.summarize(rows,features,arms,('offline25',expand.CANDIDATE),
                            (expand.CANDIDATE,candidate),fid_reference='offline25')
    for label,group in result['groups'].items():
        good = [x for x in rows if x['interior_px']>=256 and (label=='all' or x['category']==label)]
        group['texture_count'] = len(good)
        group['texture'] = {k:{m:float(np.mean([r['texture'][k][m] for r in good]))
                               for m in good[0]['texture'][k]} for k in arms}
    result['acceptance'] = fit_gate(result,candidate)
    if args.band_gate:
        texture = result['groups']['all']['texture']
        result['band_gate'] = dict(rule=BAND_RULE,
            texture_threshold=min(texture[name]['band_mae'] for name in ('off','on')),
            triggered=dict(Counter(r['category'] for r in rows if r['band_gate']['coefficient'])),
            edited=dict(Counter(r['category'] for r in rows if r['band_gate']['edited_pixels'])))
    tail.write(dest/'result.json',result)
    tail.write(dest/'per_pair.json',rows)
    np.savez_compressed(dest/'pool3.npz',**features)
    tail.write(dest/'audit.json',dict(protocol_sha256=tail.sha(args.run/'protocol.json'),shards=markers,
        optimizer_updates=0,model_frozen=True,started_with_existing_raw=args.started_with_existing_raw,
        scores=500,raw_outputs=2000))
    (dest/'protocol.json').write_bytes((args.run/'protocol.json').read_bytes())
    fid_regions.pack(dest,('result.json','per_pair.json','pool3.npz','audit.json','protocol.json'))
    print('OUTWEAR_INDEPENDENT_DONE',tail.sha(dest/'review.zip'),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('prior',type=Path)
    p.add_argument('action',choices=('prepare','generate','score','complete'))
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--shards',type=int,default=24)
    p.add_argument('--band-gate',action='store_true')
    a = p.parse_args()
    if a.band_gate and a.action!='complete':
        p.error('--band-gate requires complete')
    cv2.setNumThreads(1)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
    args = paths(a.prior,a.band_gate)
    if a.action=='prepare' or a.band_gate:
        prepare(args,a.prior)
    if a.action=='prepare':
        return
    args.protocol = tail.read(args.run/'protocol.json')
    assert args.protocol['heads']==args.heads and args.protocol['base_sha256']==args.contract['parent_contract']['base_sha256']
    if a.band_gate:
        assert args.protocol['band_gate']==BAND_RULE and args.protocol['selection_hash_salt']==BAND_SALT
    args.started_with_existing_raw = sum(
        (args.run/'test'/arm/(pair['case_id']+'.png')).with_suffix('.json').is_file()
        for pair in args.protocol['pairs'] for arm in expand.RAW)
    args.expansion = {'cached':{}}
    args.shard,args.shards = a.shard,a.shards
    if a.action=='generate':
        args.device = 'cuda:0'
        generate(args)
    elif a.action=='score':
        score(args)
    else:
        complete(args)


if __name__=='__main__':
    main()
