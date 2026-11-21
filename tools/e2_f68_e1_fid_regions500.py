"""复用 500 对终图与归档 pool3，做带羽化的区域互换诊断。"""
import argparse
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from tools import e2_f68_e1_fid_regions as roi64
from tools import e2_f68_e1_retention500 as expand
from tools import e2_f68_e1_a_fused25_continue as fused
from tools import e2_f68_region_diagnostic as roi
from tools import e2_f68_tail6_real_ab as tail


RUN = 'fid_regions_500_20261121'
SOURCE_SHA = 'b5e3867e51dcf860cc1c563a2f7a21bce16ed1c28e6f61446cb6caac0fc6c42b'
VARIANTS = tuple(f'swap_{i}' for i in range(1, 7))
SCORED = ('swap_1', 'swap_2', 'swap_4')
CANDIDATE = expand.CANDIDATE
REGIONS = ('inside','edge','background')


def images(args, row, historical, old_ids):
    index, case = row['index'], row['case_id']
    pair = args.protocol['pairs'][index]
    assert (case, row['category']) == (pair['case_id'], pair['category'])
    mask = tail.case_mask(args, pair)
    if index in old_ids:
        raw = historical[index]['pngs']
        a = roi64.checked_png(args.parent/'test/A'/(case+'.png'),raw['A'])
        donor = roi64.checked_png(args.parent/'test/blend'/(case+'.png'),raw['blend'])
        offline = fused.pixels_mixed(a,donor,fused.alpha(mask))
        current_dir = args.archived/'images'/CANDIDATE
    else:
        offline = roi64.checked_png(args.run/'images/offline25'/(case+'.png'),row['pngs']['offline25'])
        current_dir = args.run/'images'/CANDIDATE
    current = roi64.checked_png(current_dir/(case+'.png'),row['pngs'][CANDIDATE])
    target_path = args.data/pair['gt']
    assert tail.sha(target_path)==pair['input_sha256']['gt']
    gt = tail.rgb(target_path).resize(tail.SIZE,Image.Resampling.BILINEAR)
    return current,offline,mask,gt,target_path


def edits(current,offline,mask):
    weights = roi64.regions(mask)
    np.testing.assert_allclose(sum(weights),1,atol=1e-6,rtol=0)
    return {k:fused.pixels_mixed(current,offline,sum(weights[j] for j in range(3) if i & (1<<j)))
            for i,k in enumerate(VARIANTS,1)}


def score_image(image, gt, path, target_path, mask, deep, gt_lab):
    picture = Image.fromarray(image)
    picture.save(path)
    return tail.image_scores(picture,gt,path,target_path,mask), tail.appearance(image,gt_lab,deep)


@torch.inference_mode()
def measure(args, out, source, shard, shards):
    marker = out/f'part_{shard}.json'
    if marker.is_file():
        doc = tail.read(marker)
        assert doc['source_sha256']==SOURCE_SHA and doc['features_sha256']==tail.sha(out/f'pool3_{shard}.npz')
        return
    tail.verified_files(source,SOURCE_SHA,('audit.json','per_pair.json','pool3.npz','result.json'))
    original = tail.read(source/'per_pair.json')
    historical_path = args.light_run/'eval64'
    tail.verified_files(historical_path,roi64.SOURCE_SHA,('per_pair.json',))
    historical = {r['index']:r for r in tail.read(historical_path/'per_pair.json')}
    old_ids = set(tail.read(source/'audit.json')['contract']['test64'])
    assert old_ids==set(historical) and len(old_ids)==64
    assert len(original)==500 and [r['index'] for r in original]==list(range(500))
    selected = original[shard::shards]
    features = {k:[] for k in VARIANTS}
    measured = []
    assert tail.metric_assets()==args.protocol['metric_assets']
    lpips,inception = tail.metric_models('cpu',args.protocol['metric_assets'])
    with np.load(source/'pool3.npz',allow_pickle=False) as f:
        endpoints = {k:f[k] for k in (CANDIDATE,'offline25')}
    with tempfile.TemporaryDirectory(prefix='fid_region500_',dir=os.environ.get('SLURM_TMPDIR')) as tmp:
        png = Path(tmp)/'current.png'
        for start in range(0,len(selected),4):
            chunk = selected[start:start+4]
            variants = {k:[] for k in VARIANTS}
            equals = {k:[] for k in VARIANTS}
            targets = []
            items = []
            for old in chunk:
                current,offline,mask,gt,target = images(args,old,historical,old_ids)
                targets.append(np.asarray(gt))
                deep = roi.regions(mask,20)[0]
                assert int(deep.sum())==old['interior_px']
                lab = roi.lab(np.asarray(gt))
                item = {k:old[k] for k in ('index','case_id','category','interior_px')}
                diff = current.astype('float32')-offline
                stats = {}
                for name,region in zip(REGIONS,roi.regions(mask,8)):
                    part = diff[region]
                    stats[name] = dict(mae=float(np.abs(part).mean()) if len(part) else None,
                                       changed=int(np.count_nonzero(np.any(part!=0,axis=1))),pixels=len(part))
                item.update(pixel_difference=stats,scores={},texture={})
                for name,im in edits(current,offline,mask).items():
                    eq = (CANDIDATE if np.array_equal(im,current) else
                          'offline25' if np.array_equal(im,offline) else None)
                    variants[name].append(im)
                    equals[name].append(eq)
                    if name in SCORED:
                        if eq:
                            item['scores'][name] = dict(old['scores'][eq])
                            item['texture'][name] = dict(old['texture'][eq])
                        else:
                            item['scores'][name],item['texture'][name] = score_image(im,gt,png,target,mask,deep,lab)
                items.append(item)
            truth = roi.tensors(targets)
            for name,arrays in variants.items():
                eq = equals[name]
                active = [i for i,key in enumerate(eq) if key is None]
                values = {i:endpoints[key][chunk[i]['index']] for i,key in enumerate(eq) if key is not None}
                if active:
                    # 整批前向保持与既有 pool3 的批大小一致。
                    pred = roi.tensors(arrays)
                    result = inception(pred)[0].flatten(1).numpy()
                    values.update({i:result[i] for i in active})
                    if name in SCORED:
                        distance = lpips(pred*2-1,truth*2-1).flatten().tolist()
                        for i in active:
                            items[i]['scores'][name]['lpips'] = distance[i]
                features[name].append(np.stack([values[i] for i in range(len(chunk))]))
            measured.extend(items)
            if start%24==0 or len(measured)==len(selected):
                print('FID500_REGION',shard,len(measured),len(selected),flush=True)
    pool = out/f'pool3_{shard}.npz'
    np.savez_compressed(pool,**{k:np.concatenate(v) for k,v in features.items()})
    tail.write(marker,dict(rows=measured,shard=shard,shards=shards,features_sha256=tail.sha(pool),
                           source_sha256=SOURCE_SHA,code_sha256=tail.sha(Path(__file__)),
                           diffusion_rollouts=0,optimizer_updates=0))
    print('FID500_REGION_DONE',shard,len(measured),flush=True)


def merge(out,shards):
    if (out/'review.zip').is_file():
        return
    features = {k:np.zeros((500,2048),dtype='float32') for k in VARIANTS}
    all_rows = {}
    hashes = {}
    for shard in range(shards):
        marker = out/f'part_{shard}.json'
        pool = out/f'pool3_{shard}.npz'
        part = tail.read(marker)
        assert part['source_sha256']==SOURCE_SHA and part['features_sha256']==tail.sha(pool)
        assert part['code_sha256']==tail.sha(Path(__file__))
        assert part['diffusion_rollouts']==part['optimizer_updates']==0
        ids = [row['index'] for row in part['rows']]
        assert ids==list(range(shard,500,shards)) and not set(ids) & all_rows.keys()
        all_rows.update(zip(ids,part['rows']))
        with np.load(pool,allow_pickle=False) as f:
            for k in VARIANTS:
                assert f[k].shape==(len(ids),2048) and np.isfinite(f[k]).all()
                features[k][ids] = f[k]
        hashes[marker.name] = tail.sha(marker)
    assert len(all_rows)==500
    np.savez_compressed(out/'pool3.npz',**features)
    tail.write(out/'per_pair.json',[all_rows[i] for i in range(500)])
    tail.write(out/'audit.json',dict(samples=500,shards=shards,parts=hashes,source_sha256=SOURCE_SHA,
        code_sha256=tail.sha(Path(__file__)),radius=8,feather_sigma=2,scored=SCORED,
        variants=VARIANTS,diffusion_rollouts=0,optimizer_updates=0))
    roi64.pack(out,('pool3.npz','per_pair.json','audit.json'))
    path = out/'review.zip'
    print('FID500_REGION_PACKED',path.stat().st_size,tail.sha(path),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('prior',type=Path)
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--shards',type=int,default=3)
    p.add_argument('--pack',action='store_true')
    a = p.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','8')))
    args = expand.setup(a.prior)
    out = args.run/RUN
    out.mkdir(exist_ok=True)
    if a.pack:
        merge(out,a.shards)
    else:
        assert 0<=a.shard<a.shards
        measure(args,out,args.run/'eval500',a.shard,a.shards)


if __name__=='__main__':
    main()
