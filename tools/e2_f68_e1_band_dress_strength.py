"""在同一批 500 对上补测固定门槛下的裙装 L* 带通衰减。"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from tools import e2_f68_e1_band_class_ablation as category
from tools import e2_f68_e1_fid_regions as fid_regions
from tools import e2_f68_e1_independent_band as band
from tools import e2_f68_e1_outwear_independent as independent
from tools import e2_f68_region_diagnostic as roi
from tools import e2_f68_tail6_real_ab as tail


RUN = 'band_dress_strength_ablation_20261122'
CATEGORY_SHA = '89434ab1db49c9f670864126de82167570951f6922c223bbc6aaf3175a10139d'
LEVELS = {'dress_025':.025,'dress_050':.05,'dress_075':.075}
ARMS = (category.BASE,*LEVELS,category.DRESS)


def eligible(row):
    rule = independent.BAND_RULE
    return (row['category']=='dress' and row['interior_px']>=rule['min_core_pixels']
            and row['band_gate']['rms']>=rule['threshold'])


def fid_deltas(values):
    return {ref:{a:values[a]-values[ref] for a in ARMS if a!=ref}
            for ref in (category.BASE,category.DRESS)}


def fid_contrasts(values):
    return {ref+'/'+arm:v for ref,terms in fid_deltas(values).items() for arm,v in terms.items()}


@torch.inference_mode()
def run(prior):
    args = independent.paths(prior,band_gate=True)
    source = args.run/'eval500'
    archived = args.run/category.RUN
    tail.verified_files(source,category.SOURCE_SHA,('protocol.json','per_pair.json','pool3.npz','result.json'))
    tail.verified_files(archived,CATEGORY_SHA,('result.json','audit.json'))
    old = tail.read(archived/'result.json')
    args.protocol = tail.read(source/'protocol.json')
    assert args.protocol['band_gate']==independent.BAND_RULE
    assert tail.metric_assets()==args.protocol['metric_assets']
    rows = tail.read(source/'per_pair.json')
    assert len(rows)==len(args.protocol['pairs'])==500
    for i,(r,pair) in enumerate(zip(rows,args.protocol['pairs'])):
        assert (r['index'],r['case_id'],r['category'])==(i,pair['case_id'],pair['category'])
        if r['category']=='dress':
            assert eligible(r)==bool(r['band_gate']['coefficient'])
    selected = np.array([r['index'] for r in rows if eligible(r)])
    assert len(selected)==old['changed_pairs'][category.DRESS]==41
    out = args.run/RUN
    out.mkdir(exist_ok=True)
    labels = np.array([r['category'] for r in rows])
    with np.load(source/'pool3.npz',allow_pickle=False) as pool:
        features = {name:pool[name] for name in (*category.CONTROLS,category.BASE,category.BOTH,'gt')}
    features[category.DRESS] = features[category.BASE].copy()
    features[category.DRESS][selected] = features[category.BOTH][selected]
    for r in rows:
        choice = category.BOTH if eligible(r) else category.BASE
        for key in ('pngs','scores','texture'):
            r[key][category.DRESS] = r[key][choice]
    choices = {name:dict(gain=c) for name,c in LEVELS.items()}
    features.update(band.score_edits(args,rows,choices,features,eligible=lambda r,_:eligible(r),
                                     image_out=out/'images'))
    unchanged = np.ones(len(rows),bool)
    unchanged[selected] = False
    changed = {}
    for name in LEVELS:
        changed[name] = sum(r['edited_pixels'][name]>0 for r in rows)
        np.testing.assert_array_equal(features[name][unchanged],features[category.BASE][unchanged])
        assert all(r[key][name]==r[key][category.BASE] for r in rows if not eligible(r)
                   for key in ('pngs','scores','texture'))
    del features[category.BOTH]
    variants = (*category.CONTROLS,*ARMS)
    refs = (category.BASE,category.DRESS,'offline25')
    result = tail.summarize(rows,features,variants,refs,tuple(LEVELS),fid_reference=category.BASE)
    category.summarize_texture(result,rows,variants)
    group = result['groups']['all']
    for a in (category.BASE,category.DRESS):
        for key,stat in group['arms'][a].items():
            field = 'fid' if key=='fid' else 'mean'
            np.testing.assert_allclose(stat[field],old['groups']['all']['arms'][a][key][field],atol=1e-9,rtol=1e-12)
        np.testing.assert_allclose(group['texture'][a]['band_mae'],
                                   old['groups']['all']['texture'][a]['band_mae'],atol=1e-12,rtol=0)
    valid = [r for r in rows if r['interior_px']>=independent.BAND_RULE['min_core_pixels']]
    result['texture_paired'] = {
        ref:{a:roi.paired([r['texture'][ref]['band_mae'] for r in valid],
                          [r['texture'][a]['band_mae'] for r in valid],
                          np.array([r['category'] for r in valid])) for a in ARMS if a!=ref}
        for ref in refs[:2]}
    result['acceptance'] = {a:category.uniform_gate(result,a) for a in ARMS}
    result['changed_pairs'] = {category.BASE:0,category.DRESS:len(selected),**changed}
    result['strength'] = {category.BASE:0.,category.DRESS:.10,**LEVELS}
    result['texture_threshold'] = min(group['texture'][a]['band_mae'] for a in ('off','on'))
    result['dress_fid'] = {a:roi.fid_terms(features[a][labels=='dress'],features['gt'][labels=='dress']) for a in ARMS}
    values = {a:group['arms'][a]['fid']['fid'] for a in ARMS}
    print('BAND_DRESS_FID',values,flush=True)
    result['fid_deltas'] = fid_deltas(values)
    result['fid_ci95'] = category.fid_intervals(features,labels,arms=ARMS,contrast=fid_contrasts)
    tail.write(out/'result.json',result)
    tail.write(out/'edited_pairs.json',[rows[i] for i in selected])
    np.savez_compressed(out/'edited_pool3.npz',indices=selected,**{a:features[a][selected] for a in LEVELS})
    tail.write(out/'audit.json',dict(pairs=len(rows),eligible_pairs=len(selected),eligible_indices=selected.tolist(),
        changed_pairs=changed,rms_threshold=independent.BAND_RULE['threshold'],dress_strength=LEVELS,
        outwear_enabled=False,source_job=126569,category_job=126641,
        source_sha256=category.SOURCE_SHA,category_sha256=CATEGORY_SHA,
        code_sha256=tail.sha(Path(__file__)),scored=sum(changed.values()),diffusion_rollouts=0,
        optimizer_updates=0,bootstrap=category.N_BOOT,bootstrap_seed=category.BOOT_SEED))
    fid_regions.pack(out,('result.json','edited_pairs.json','edited_pool3.npz','audit.json'))
    print('BAND_DRESS_DONE',tail.sha(out/'review.zip'),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prior',type=Path)
    cv2.setNumThreads(1)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
    run(parser.parse_args().prior)
