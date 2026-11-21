"""只补缺的 E1 S0 轨迹并复用原 64 对，评估固定 12.5% 的 500 对效果。"""
import argparse
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch
from PIL import Image

from tools import e2_f68_e1_a_roi_ablation as ab
from tools import e2_f68_e1_a_fused25_continue as fused
from tools import e2_f68_e1_a_fused25_lstar_ab as light
from tools import e2_f68_e1_fid_regions as regions
from tools import e2_f68_region_diagnostic as roi
from tools import e2_f68_tail6_e1_epoch10 as epoch
from tools import e2_f68_tail6_real_ab as tail


RUN = 'retention_125_eval500_20261121'
CONTROL_SHA = '537ed11967af70fdb0cf829d1fa063eeb7a2029f798f32d9a80ecfd49e9a03d1'
RAW = ('off', 'on', 'A', 'B')
CANDIDATE = 'retain_125'
SCORED = ('off', 'on', 'blend', 'offline25', CANDIDATE)
RETENTION = .125


def checked_meta(args, path, pair, index, arm):
    m = tail.read(path.with_suffix('.json'))
    ckpt = args.heads[arm]
    assert (m['case'], m['index'], m['arm'], m['checkpoint']) == (pair['case_id'], index, arm, ckpt), path
    assert m['steps'] == 50 and m['base_sha256'] == args.contract['parent_contract']['base_sha256'], path
    assert m['initial_noise_digest'] == args.original_rows[index]['arms']['off']['initial_noise_digest'], path
    assert tail.sha(path) == m['png_sha256'], path
    return m


def setup(prior):
    project = prior.parents[1]
    base = prior / tail.distill.BASE
    parent = base / regions.COHORT
    light_run = parent / light.RUN
    run = light_run / RUN
    run.mkdir(exist_ok=True)
    selection = tail.read(light_run / 'selection.json')
    contract = tail.read(light_run / 'protocol.json')
    assert selection['step'] == 64 and selection['contract'] == contract
    return SimpleNamespace(
        project=project, data=project.parent / 'datasets/BF', baseline=base, out=base,
        parent=parent, light_run=light_run, run=run,
        archived=light_run / regions.RUN / regions.RETENTION_RUN,
        protocol=tail.verify(SimpleNamespace(out=base)), contract=contract, selection=selection,
        original_rows=tail.read(base/'per_image.json'), m0_result=tail.read(base/'result.json'),
        heads={'off':None,'on':tail.CHECKPOINT_SHA,'A':contract['start_A_sha256'],
               'B':selection['checkpoints']['B']},
        checkpoint=project/'output_eval/e2_state_b_pastel_followup_20261115/trust_retry/training/step1.pt',
        base_checkpoint=project/epoch.BASE_CHECKPOINT)


def prepare(args):
    original = args.light_run / 'eval64'
    tail.verified_files(original, regions.SOURCE_SHA, ('per_pair.json','pool3.npz','result.json'))
    tail.verified_files(args.archived, CONTROL_SHA, ('per_pair.json','pool3.npz','result.json','audit.json'))
    tail.verified_files(args.baseline, roi.ARCHIVE_SHA, ('protocol.json','per_image.json','fid_pool3.npz'))
    archived = tail.read(args.archived / 'result.json')
    assert archived['acceptance'][CANDIDATE]['passed']
    manifest = tail.read(original/'review_manifest.json')
    assert tail.sha(args.light_run/'selection.json') == manifest['selection.json']
    assert tail.sha(args.base_checkpoint) == args.contract['parent_contract']['base_sha256']
    assert tail.sha(args.parent/'A/step128.pt') == args.heads['A']
    head = args.light_run/'B/step64.pt'
    assert tail.sha(head) == args.heads['B']
    local_head = args.run/'B/step64.pt'
    local_head.parent.mkdir(exist_ok=True)
    if not local_head.exists():
        local_head.symlink_to(head)
    assert tail.sha(local_head) == args.heads['B']
    tail.write(args.run/'selection.json',args.selection)

    rows = tail.read(args.archived/'per_pair.json')
    small = tail.subset(args.protocol)
    assert len(rows) == len(small) == 64 and [r['index'] for r in rows] == small
    for row in rows:
        case = row['case_id']
        assert case == args.protocol['pairs'][row['index']]['case_id']
        assert tail.sha(args.archived/'images'/CANDIDATE/(case+'.png')) == row['pngs'][CANDIDATE]

    # 验证同底座同权重的现存图，保留其他实验的原文件。
    roots = sorted(args.parent.rglob('test'))
    cached = {}
    more = epoch.pair_indices(args,False)
    assert len(more) == 436 and len(args.protocol['pairs']) == 500
    for index in more:
        pair = args.protocol['pairs'][index]
        case = pair['case_id']
        record = {}
        for arm in RAW:
            for root in roots:
                path = root / arm / (case+'.png')
                meta = path.with_suffix('.json')
                if not (path.is_file() and meta.is_file()):
                    continue
                m = tail.read(meta)
                if m.get('checkpoint') != args.heads[arm] or m.get('base_sha256') != args.contract['parent_contract']['base_sha256']:
                    continue
                checked_meta(args,path,pair,index,arm)
                record[arm] = str(path.relative_to(args.parent))
                break
        if record:
            cached[case] = record
    reuse = {k:sum(k in v for v in cached.values()) for k in RAW}
    tail.write(args.run/'protocol.json',dict(retention=RETENTION, seed=42, steps=50, generator='S0',
        test64=small, additional436=more, cached=cached, reused_raw=reuse, initial_archives=[CONTROL_SHA,regions.SOURCE_SHA],
        heads=args.heads, base_sha256=args.contract['parent_contract']['base_sha256'],
        code_sha256=tail.sha(Path(__file__)), optimizer_updates=0))
    print('RETENTION500_PREPARED',reuse,'new',sum(436-n for n in reuse.values()),flush=True)


def raw_path(args, arm, case):
    cached = args.expansion['cached'].get(case, {}).get(arm)
    if cached is not None:
        return args.parent/cached
    return args.run/'test'/arm/(case+'.png')


class SharedNoise:
    """复用同一身份、相同条件下输入逐位相同的 U-Net 前向。"""

    def __init__(self):
        self.sample = tail.joint.sample
        self.case = None
        self.prefix = {}
        self.tail = {}
        self.reused = 0

    def __call__(self, args, core, pipe, net, pair, condition, b=None, **kwargs):
        case = pair['case_id']
        if case != self.case:
            self.case = case
            self.prefix.clear()
            self.tail.clear()
        calls = 0
        original = pipe.unet.forward

        def forward(sample, timestep, **inputs):
            nonlocal calls
            index = calls
            calls += 1
            # S0 每步依次执行有条件、无条件两次 U-Net。
            step = index // 2
            cache = self.prefix if step <= 25 else self.tail if step <= 44 and isinstance(b, tail.TailResidual) else None
            if cache is None:
                return original(sample, timestep, **inputs)
            condition_input = inputs['encoder_hidden_states']
            if index in cache:
                x, t, cond, value = cache[index]
                if torch.equal(sample,x) and torch.equal(timestep,t) and torch.equal(condition_input,cond):
                    self.reused += 1
                    return (value.clone(),)
                return original(sample, timestep, **inputs)
            output = original(sample, timestep, **inputs)
            assert isinstance(output,tuple) and len(output)==1
            cache[index] = (sample.clone(), timestep.clone(), condition_input.clone(), output[0].clone())
            return output

        with patch.object(pipe.unet,'forward',forward):
            output = self.sample(args,core,pipe,net,pair,condition,b=b,**kwargs)
        assert calls == 100
        return output


def generate(args, marker_name=None):
    args.action = 'generate500'
    memo = SharedNoise()
    default = args.run/f'generate500_{args.shard}.json'
    marker = args.run/marker_name if marker_name is not None else default
    save = tail.write

    def write(path, value):
        if path == default:
            return save(marker,dict(value,reused_unet_forwards=memo.reused))
        return save(path,value)

    with patch.object(ab,'ARMS',('B',)), patch.object(ab,'accepted',return_value=('B',)):
        with patch.object(ab,'png_path',raw_path), patch.object(tail.joint,'sample',memo):
            with patch.object(tail,'write',write):
                ab.generate(args)
    result = tail.read(marker)
    print('RETENTION500_GENERATED',args.shard,len(result['ids']),memo.reused,flush=True)


def checked_inputs(args, index):
    pair = args.protocol['pairs'][index]
    case = pair['case_id']
    mask = tail.case_mask(args,pair)
    records = {}
    raw = {}
    for name in RAW:
        path = raw_path(args,name,case)
        records[name] = checked_meta(args,path,pair,index,name)
        raw[name] = np.asarray(tail.rgb(path))
    assert all(m['first_b_step'] == records['off']['first_b_step'] for m in records.values())
    donor = tail.mix_pixels(raw['off'],raw['on'],mask,{'blend':.25})['blend']
    offline = fused.pixels_mixed(raw['A'],donor,fused.alpha(mask))
    current = fused.pixels_mixed(raw['B'],donor,fused.alpha(mask))
    inside = regions.regions(mask)[0]
    candidate = fused.pixels_mixed(current,offline,(1-RETENTION)*inside)
    assert np.array_equal(candidate[inside==0],current[inside==0])
    return pair,mask,dict(off=raw['off'],on=raw['on'],blend=donor,offline25=offline,retain_125=candidate),records


def reuse64(args, features):
    tail.verified_files(args.archived, CONTROL_SHA, ('per_pair.json','pool3.npz','result.json','audit.json'))
    source = args.light_run/'eval64'
    tail.verified_files(source, regions.SOURCE_SHA, ('per_pair.json','pool3.npz','result.json'))
    cached = [tail.read(p/'per_pair.json') for p in (source,args.archived)]
    ids = args.expansion['test64']
    rows = {}
    for a,b in zip(*cached):
        index = a['index']
        assert index==b['index'] and a['case_id']==b['case_id']
        row = {k:b[k] for k in ('index','case_id','category','interior_px')}
        row['pngs'] = {CANDIDATE:b['pngs'][CANDIDATE]}
        for key in ('scores','texture'):
            row[key] = {k:(b if k in ('offline25',CANDIDATE) else a)[key][k] for k in SCORED}
        rows[index] = row
    assert list(rows) == ids
    with np.load(source/'pool3.npz',allow_pickle=False) as former, np.load(args.archived/'pool3.npz',allow_pickle=False) as fixed:
        for name in SCORED:
            values = (fixed if name in ('offline25',CANDIDATE) else former)[name]
            assert values.shape == (64,2048)
            features[name][ids] = values
    return rows


def summarize(rows, features):
    result = tail.summarize(rows,features,SCORED,('offline25',),(CANDIDATE,),fid_reference='offline25')
    for label,group in result['groups'].items():
        selected = [r for r in rows if r['interior_px']>=256 and (label=='all' or r['category']==label)]
        group['texture_count'] = len(selected)
        group['texture'] = {arm:{key:float(np.mean([r['texture'][arm][key] for r in selected]))
                                 for key in selected[0]['texture'][arm]} for arm in SCORED}
    result['acceptance'] = epoch.gate_check(result,CANDIDATE)
    arms = result['groups']['all']['arms']
    delta = {k:arms[CANDIDATE][k][field]-arms['offline25'][k][field]
             for k,field in (('lpips','mean'),('fid','fid'),('ssim','mean'),('contour_f1','mean'))}
    gate = result['acceptance']
    gate['delta_vs_offline25'] = delta
    gate['passed_all'] = gate['passed'] and delta['lpips']<0 and delta['fid']<0 and delta['ssim']>=0 and delta['contour_f1']>=0
    return result


@torch.inference_mode()
def score(args):
    product = args.run/'eval500'
    product.mkdir(exist_ok=True)
    if (product/'review.zip').is_file():
        return
    ids = args.expansion['additional436']
    markers = [tail.read(args.run/f'generate500_{shard}.json') for shard in range(args.shards)]
    assert all(m['ids']==ids[s::args.shards] and set(m['arms'])==set(RAW) and m['frozen']
               for s,m in enumerate(markers))
    features = {name:np.zeros((500,2048),dtype='float32') for name in SCORED}
    with np.load(args.baseline/'fid_pool3.npz',allow_pickle=False) as prev:
        features['gt'] = prev['gt']
    rows = reuse64(args,features)
    assert tail.metric_assets()==args.protocol['metric_assets']
    lpips,inception = tail.metric_models('cpu',args.protocol['metric_assets'])
    for start in range(0,len(ids),4):
        batch = ids[start:start+4]
        pictures, targets = {k:[] for k in SCORED}, []
        for index in batch:
            pair,mask,images,records = checked_inputs(args,index)
            case = pair['case_id']
            target = args.data/pair['gt']
            assert tail.sha(target)==pair['input_sha256']['gt']
            gt = tail.rgb(target).resize(tail.SIZE,Image.Resampling.BILINEAR)
            gt_rgb = np.asarray(gt)
            gt_lab = roi.lab(gt_rgb)
            core = roi.regions(mask,20)[0]
            targets.append(gt_rgb)
            row = dict(index=index,case_id=case,category=pair['category'],interior_px=int(core.sum()),
                       pngs={},scores={},texture={},raw={k:v['png_sha256'] for k,v in records.items()})
            for name,img in images.items():
                if name in ('off','on'):
                    path = raw_path(args,name,case)
                else:
                    path = args.run/'images'/name/(case+'.png')
                    fused.save_png(path,img)
                picture = Image.fromarray(img)
                pictures[name].append(img)
                row['pngs'][name] = tail.sha(path)
                row['scores'][name] = tail.image_scores(picture,gt,path,target,mask)
                row['texture'][name] = tail.appearance(img,gt_lab,core)
            rows[index] = row
        truth = roi.tensors(targets)
        for name in SCORED:
            pred = roi.tensors(pictures[name])
            for index,value in zip(batch,lpips(pred*2-1,truth*2-1).flatten().tolist()):
                rows[index]['scores'][name]['lpips'] = value
            features[name][batch] = inception(pred)[0].flatten(1).numpy()
        if start%20==0 or start+4>=len(ids):
            print('RETENTION500_SCORED',start+len(batch),len(ids),flush=True)
    assert sorted(rows)==list(range(500)) and all(np.isfinite(a).all() for a in features.values())
    all_rows = [rows[i] for i in range(500)]
    more_rows = [rows[i] for i in ids]
    result = dict(all500=summarize(all_rows,features),
                  additional436=summarize(more_rows,{k:v[ids] for k,v in features.items()}))
    tail.write(product/'result.json',result)
    tail.write(product/'per_pair.json',all_rows)
    np.savez_compressed(product/'pool3.npz',**features)
    tail.write(product/'audit.json',dict(contract=args.expansion, shards=markers, reused_scores=64,
        new_scored_pairs=436,new_raw={k:436-args.expansion['reused_raw'][k] for k in RAW},
        frozen_weights=True,optimizer_updates=0))
    regions.pack(product,('result.json','per_pair.json','pool3.npz','audit.json'))
    print('RETENTION500_DONE',tail.sha(product/'review.zip'),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('prior',type=Path)
    p.add_argument('action',choices=('prepare','generate','score'))
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--shards',type=int,default=24)
    parsed = p.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
    args = setup(parsed.prior)
    if parsed.action=='prepare':
        prepare(args)
        return
    args.expansion = tail.read(args.run/'protocol.json')
    args.shard,args.shards = parsed.shard,parsed.shards
    assert args.expansion['heads']==args.heads and args.expansion['retention']==RETENTION
    if parsed.action=='generate':
        args.device = 'cuda:0'
        generate(args)
    else:
        score(args)


if __name__=='__main__':
    main()
