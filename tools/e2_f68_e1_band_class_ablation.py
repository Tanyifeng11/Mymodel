"""复用独立 500 对归档，对照固定 RMS 门控的类别开关。"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from tools import e2_f68_e1_fid_regions as fid_regions
from tools import e2_f68_e1_outwear_independent as independent
from tools import e2_f68_region_diagnostic as roi
from tools import e2_f68_tail6_real_ab as tail


SOURCE_SHA = 'fa659c87cfea3f4990d02039d3689ed82d2af3747b60f169139affb7f709661b'
RUN = 'band_category_ablation_20261122'
BASE = 'retain_125'
BOTH = independent.BAND_CANDIDATE
DRESS = 'dress_only'
OUTWEAR = 'outwear_only'
ARMS = (BASE, DRESS, OUTWEAR, BOTH)
CONTROLS = ('off', 'on', 'blend', 'offline25')
N_BOOT = 1200
BOOT_SEED = 20261121


def factorial(values):
    base, dress, outwear, both = (values[name] for name in ARMS)
    return dict(dress_only=dress-base, outwear_only=outwear-base, both=both-base,
                dress_given_outwear=both-outwear, outwear_given_dress=both-dress,
                interaction=both-dress-outwear+base,
                dress_shapley=.5*(dress-base+both-outwear),
                outwear_shapley=.5*(outwear-base+both-dress))


def fid_intervals(features, labels, arms=ARMS, contrast=factorial):
    # 配对重复抽样中的相同产品用计数权重表示，保持普通 FID 的样本协方差。
    gt = features['gt'].astype('float64')
    pred = {a:features[a].astype('float64') for a in arms}
    grams = {a:x @ gt.T for a,x in pred.items()}
    norm = {a:np.square(x).sum(axis=1) for a,x in pred.items()}
    gt_norm = np.square(gt).sum(axis=1)
    rng = np.random.default_rng(BOOT_SEED)
    groups = [np.flatnonzero(labels==c) for c in np.unique(labels)]
    n = len(gt)
    draws = np.concatenate([rng.choice(g,size=(N_BOOT,len(g))) for g in groups],axis=1)
    effects = []
    for draw in draws:
        ids, counts = np.unique(draw,return_counts=True)
        w = counts/n
        scale = np.sqrt(counts)
        sub = np.ix_(ids,ids)
        mu_gt = w @ gt[ids]
        var_gt = gt_norm[ids] @ counts - n*(mu_gt @ mu_gt)
        fids = {}
        for arm in arms:
            mu = w @ pred[arm][ids]
            var = norm[arm][ids] @ counts - n*(mu @ mu)
            gram = grams[arm][sub]
            center = gram - (gram @ w)[:,None] - (w @ gram)[None,:] + w @ gram @ w
            cross = np.linalg.svd(center*scale[:,None]*scale[None,:],compute_uv=False).sum()
            fids[arm] = np.square(mu-mu_gt).sum() + (var+var_gt-2*cross)/(n-1)
        effects.append(contrast(fids))
    return {key:np.quantile([x[key] for x in effects],[.025,.975]).tolist() for key in effects[0]}


def summarize_texture(result, rows, variants):
    for name,group in result['groups'].items():
        good = [r for r in rows if r['interior_px']>=256 and (name=='all' or name==r['category'])]
        group['texture_count'] = len(good)
        group['texture'] = {a:{key:float(np.mean([r['texture'][a][key] for r in good]))
                               for key in good[0]['texture'][a]} for a in variants}


def uniform_gate(result, arm):
    gate = independent.fit_gate(result,arm)
    texture = result['groups']['all']['texture']
    delta = gate['delta'][BASE]
    comparison = dict(fid=delta['fid']<=0,lpips=delta['lpips']<0,ssim=delta['ssim']>=0,
                      contour_f1=delta['contour_f1']>=0,
                      texture=texture[arm]['band_mae']<texture[BASE]['band_mae'])
    gate['vs_uniform'] = comparison
    gate['passed_with_uniform_target'] = gate['passed_all'] and all(comparison.values())
    return gate


def run(source):
    names = ('audit.json','protocol.json','per_pair.json','pool3.npz','result.json')
    tail.verified_files(source,SOURCE_SHA,names)
    out = source.parent/RUN
    out.mkdir(exist_ok=True)
    protocol = tail.read(source/'protocol.json')
    archived = tail.read(source/'result.json')
    rows = tail.read(source/'per_pair.json')
    audit = tail.read(source/'audit.json')
    assert protocol['band_gate']==independent.BAND_RULE and audit['scores']==len(rows)==500
    assert [r['case_id'] for r in rows]==[p['case_id'] for p in protocol['pairs']]
    labels = np.array([r['category'] for r in rows])
    assert Counter(labels)==dict.fromkeys(('dress','outwear','pants','top'),125)
    with np.load(source/'pool3.npz',allow_pickle=False) as pool:
        features = {name:pool[name] for name in (*CONTROLS,BASE,BOTH,'gt')}
    changed = {DRESS:0,OUTWEAR:0,BOTH:0}
    selected = {DRESS:'dress',OUTWEAR:'outwear'}
    for arm,category in selected.items():
        features[arm] = np.where((labels==category)[:,None],features[BOTH],features[BASE])
    for i,row in enumerate(rows):
        edit = row['band_gate']
        rms = edit['rms']
        expected = independent.BAND_RULE['coefficient'].get(row['category'],0.)
        if rms is None or rms < independent.BAND_RULE['threshold']:
            expected = 0.
        assert edit['coefficient']==expected
        edited = bool(edit['edited_pixels'])
        assert edited==bool(expected) and edited==(row['pngs'][BASE]!=row['pngs'][BOTH])
        if not edited:
            np.testing.assert_array_equal(features[BASE][i],features[BOTH][i])
            assert all(row[key][BASE]==row[key][BOTH] for key in ('scores','texture'))
        changed[BOTH] += edited
        for arm,category in selected.items():
            choice = BOTH if row['category']==category else BASE
            changed[arm] += edited and choice==BOTH
            for key in ('scores','texture','pngs'):
                row[key][arm] = row[key][choice]
    assert changed[BOTH]==changed[DRESS]+changed[OUTWEAR]

    variants = (*CONTROLS,*ARMS)
    result = tail.summarize(rows,features,variants,(BASE,'offline25'),(DRESS,OUTWEAR,BOTH),fid_reference=BASE)
    summarize_texture(result,rows,variants)
    all_group = result['groups']['all']
    arms, texture = all_group['arms'], all_group['texture']
    for a in (BASE,BOTH):
        for key,stat in arms[a].items():
            field = 'fid' if key=='fid' else 'mean'
            np.testing.assert_allclose(stat[field],archived['groups']['all']['arms'][a][key][field],
                                       rtol=1e-12,atol=1e-9)
        np.testing.assert_allclose(texture[a]['band_mae'],archived['groups']['all']['texture'][a]['band_mae'],
                                   rtol=0,atol=1e-12)
    good = [r for r in rows if r['interior_px']>=256]
    result['texture_threshold'] = min(texture[a]['band_mae'] for a in ('off','on'))
    result['texture_paired_vs_uniform'] = {
        arm:roi.paired([r['texture'][BASE]['band_mae'] for r in good],
                       [r['texture'][arm]['band_mae'] for r in good],
                       np.array([r['category'] for r in good])) for arm in ARMS}
    result['category_fid'] = {
        str(c):{arm:roi.fid_terms(features[arm][labels==c],features['gt'][labels==c]) for arm in ARMS}
        for c in np.unique(labels)}
    result['changed_pairs'] = changed
    result['acceptance'] = {arm:uniform_gate(result,arm) for arm in ARMS}

    print('BAND_CLASS_FID', {a:arms[a]['fid']['fid'] for a in ARMS},flush=True)
    result['fid_factorial'] = dict(point=factorial({a:arms[a]['fid']['fid'] for a in ARMS}),
                                   ci95=fid_intervals(features,labels))
    tail.write(out/'result.json',result)
    tail.write(out/'audit.json',dict(source_sha256=SOURCE_SHA,source_archive=str(source/'review.zip'),
        source_job=126569,code_sha256=tail.sha(Path(__file__)),pairs=len(rows),changed_pairs=changed,
        band_gate=protocol['band_gate'],bootstrap=N_BOOT,bootstrap_seed=BOOT_SEED,
        edited_image_source={DRESS:'band_rms475/dress',OUTWEAR:'band_rms475/outwear'},
        other_image_source=BASE,new_images=0,model_forwards=0,optimizer_updates=0))
    fid_regions.pack(out,('result.json','audit.json'))
    print('BAND_CLASS_DONE',tail.sha(out/'review.zip'),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    run(parser.parse_args().source)
