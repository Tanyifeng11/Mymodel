"""从便携原件重建 P0；每个身份配对，保留旧确认和探索消融的边界。"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from tools.t3_p0p1 import read, write, sha, csv_write

METRICS = dict(edge_f1=1, leak=-1, iou=1, edge_l1=-1, lpips=-1, tcf_lab=-1)
LABELS = dict(edge_f1='Edge F1', leak='Leak', iou='Sketch IoU', edge_l1='Edge L1',
              lpips='LPIPS', tcf_lab='TCF-LAB')


def paired_stats(values, key):
    values = np.asarray(values, dtype=float)
    seed = 281042 + int(hashlib.sha256(key.encode()).hexdigest()[:7], 16)
    rng = np.random.default_rng(seed)
    boots = values[rng.integers(0, len(values), (10000, len(values)))].mean(1)
    signs = rng.choice([-1., 1.], (10000, len(values)))
    p = (1 + np.count_nonzero(np.abs((signs * values).mean(1)) >= abs(values.mean()) - 1e-15)) / 10001
    low, high = np.quantile(boots, [.025, .975])
    return dict(n=len(values), delta=float(values.mean()), ci_low=float(low), ci_high=float(high),
                p_raw=float(p), sd_difference=float(values.std(ddof=1)),
                paired_dz=float(values.mean()/values.std(ddof=1)) if values.std(ddof=1) else None,
                bootstrap_draws=10000, signflip_draws=10000, statistics_seed=seed)


def holm(rows):
    order = sorted(range(len(rows)), key=lambda index: rows[index]['p_raw'])
    previous = 0.
    for rank, index in enumerate(order):
        previous = max(previous, min(1., (len(rows)-rank)*rows[index]['p_raw']))
        rows[index]['p_holm'] = previous
        rows[index]['family_size'] = len(rows)


def sources(project):
    sup = project/'output_eval/e2a_sup_comp_01_20261021'
    if not (sup/'evaluation/P2b').is_dir():
        sup = sup/'review_results_20261026'
    conf = project/'output_eval/t3_reference_response_20261027'
    if not (conf/'protocol/config.json').is_file():
        conf = conf/'review_results'
    extension = project/'output_eval/t3_reference_response_20261027/exploratory_extension_20261027'
    if not (extension/'ablation_closeout').is_dir():
        extension = extension/'ablation_results'
    return sup, conf, extension


def review(project):
    root = project/'output_eval/t3_p0_p1_20261028'
    sup, conf, extension = sources(project)
    rows = read(sup/'splits/P2b.json')
    ids = [row['id'] for row in rows]
    assert len(ids) == len(set(ids)) == 96
    all_metrics, manifest, table, registry = {}, [], [], []
    definitions = dict(B='基础训练，无额外区域监督', T1='感知 DISTS 与高频监督适配',
                       T2='DINO 特征外观与结构监督适配', T3='完整双区域非对称监督',
                       C='作者历史自研 body/boundary/outside 三域监督')
    invpath = root/'audit/server_inventory.json'
    inventory = read(invpath) if invpath.is_file() else {}
    checks = []
    for arm in ['B', 'T1', 'T2', 'T3', 'C']:
        folder = sup/'evaluation/P2b'/arm
        path = folder/'per_identity_metrics.json'
        records = read(path)
        by_id = {record['id']: record for record in records}
        assert set(by_id) == set(ids) and len(records) == 96
        all_metrics[arm] = by_id
        for metric, direction in METRICS.items():
            values = [by_id[i][metric] for i in ids if by_id[i].get(metric) is not None]
            assert np.isfinite(values).all()
            table.append(dict(model=arm, metric=metric, direction='up' if direction>0 else 'down',
                              mean=float(np.mean(values)), valid_n=len(values), total_n=96,
                              source=str(path), source_sha256=sha(path), protocol='P2b legacy six-metric'))
        checkpoint = None
        for row in rows:
            image = folder/'images'/(row['id']+'.png')
            meta = read(image.with_suffix('.json'))
            assert sha(image) == meta['output_sha256']
            assert meta['gt_used_in_inference'] is False and meta['explicit_latent_verified']
            checkpoint = meta['checkpoint']
            if arm != 'B':
                other = read(sup/'evaluation/P2b/B/images'/(row['id']+'.json'))
                assert all(meta[k] == other[k] for k in ['seed', 'initial_latent_sha256', 'inputs'])
            manifest.append(dict(id=row['id'], model=arm, condition='original', output=str(image),
                output_sha256=meta['output_sha256'], checkpoint=meta['checkpoint'], seed=meta['seed'],
                initial_latent_sha256=meta['initial_latent_sha256'], prompt=row['caption'],
                **{kind+'_sha256': value for kind, value in meta['inputs'].items()}))
        registry.append(dict(model_id=arm, role='historical_self_developed' if arm=='C' else 'primary_internal',
            model_definition=definitions[arm], input_modalities='text+sketch+reference', training_seed=42,
            training_steps=28365, initialization_checkpoint='E1: e51d71dbf17c424fb639293ab99cfa83b9658451e127e6a61e8263e15bd5e39a',
            checkpoint=checkpoint, checkpoint_sha256=inventory.get('models',{}).get(arm,{}).get('sha256','pending server inventory'),
            dataset_split='P2b96 development identities', inference_config='DDIM50 CFG7 384x512 sketch0.6 texture1',
            metric_protocol_version=sha(sup/'evaluation/metric_protocol.json'), result_source=str(path),
            status='local outputs and explicit paired latents verified', comparability_notes='one training seed; retrospective statistics'))
        checks.append(dict(model=arm, n=96, paired_inputs_and_latents=True, output_sha256=True))
    for arm, definition in [('E0','共同预训练纹理适配器初始化后独立训练；不启用层级路由'),
                            ('E1','与 E0 独立训练，启用固定层级路由；为 T3 初始化模型')]:
        registry.append(dict(model_id=arm, role='training_stage_background', model_definition=definition,
            input_modalities='text+sketch+reference', checkpoint_sha256=inventory.get('models',{}).get(arm,{}).get('sha256','pending server inventory'),
            training_seed=42, training_steps=28365, initialization_checkpoint='texture_adapter_bf_e20, independently trained',
            dataset_split='P2b96 evaluation pending', inference_config='same frozen P2b settings, E0 routing disabled',
            metric_protocol_version='pending same-protocol inference', result_source='docs/E0_E1_E2_E5_训练方法_文献依据与性能详解_20261018.md',
            status='definition established; P2b outputs pending', comparability_notes='historical 500 set cannot join P2b ranking'))
    contrasts = []
    for baseline in ['B','T1','T2','C']:
        group = []
        for metric, direction in METRICS.items():
            valid = [i for i in ids if all(all_metrics[a][i].get(metric) is not None for a in ['T3',baseline])]
            delta = [all_metrics['T3'][i][metric]-all_metrics[baseline][i][metric] for i in valid]
            stat = paired_stats(delta, baseline+'/'+metric)
            wins = np.asarray(delta)*direction
            group.append(dict(comparison='T3 minus '+baseline, baseline=baseline, metric=metric,
                favorable_direction=direction, oriented_delta=stat['delta']*direction,
                win=int((wins>1e-12).sum()), tie=int((abs(wins)<=1e-12).sum()), lose=int((wins< -1e-12).sum()),
                evidence='retrospective single-training-seed development analysis', **stat))
        contrasts.extend(group)
    main = [row for row in contrasts if row['baseline']!='C']
    historical = [row for row in contrasts if row['baseline']=='C']
    holm(main)
    holm(historical)
    for row in contrasts:
        row['multiplicity_family'] = '18 internal contrasts (3 baselines x 6 metrics)' if row['baseline']!='C' else '6 historical C contrasts'
    csv_write(root/'tables/internal_metrics.csv', table)
    csv_write(root/'tables/internal_paired_statistics.csv', contrasts)
    csv_write(root/'model_registry.csv', registry)
    csv_write(root/'provenance/internal_generation_manifest.csv', manifest)
    write(root/'protocol/P0_statistics.json', dict(evidence='retrospective exploratory; not new preregistered confirmation',
        primary_internal_models=['B','T1','T2','T3'], historical_self_developed='C',
        independent_unit='identity', training_seeds=1, n=96, bootstrap=10000, signflip=10000,
        confidence_interval='paired identity percentile bootstrap 95%, unadjusted descriptive CI',
        multiple_testing='Holm across all18 main contrasts; historical C six tests separately',
        missing='pairwise complete; no imputation, retain all failed identities',
        original_confirmation_and_ablation_decisions_unchanged=True))
    write(root/'audit/P0_local_verification.json', dict(models=checks, images=len(manifest),
        source_metric_protocol_sha256=sha(sup/'evaluation/metric_protocol.json'),
        source_splits_sha256=sha(sup/'splits/P2b.json'), paired_all_passed=True))
    plot(root, contrasts)
    panels(root, sup, rows, all_metrics)
    print(json.dumps(dict(root=str(root), verified_images=len(manifest), contrasts=len(contrasts),
                         main_holm_significant=[(r['baseline'],r['metric'],r['p_holm']) for r in main if r['p_holm']<.05]),ensure_ascii=False))


def plot(root, contrasts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,3,figsize=(12,6.5))
    for axis, (metric,direction) in zip(axes.flat,METRICS.items()):
        for y, baseline in enumerate(['B','T1','T2']):
            row = next(r for r in contrasts if r['baseline']==baseline and r['metric']==metric)
            low, high = sorted([row['ci_low']*direction,row['ci_high']*direction])
            center = row['oriented_delta']
            axis.errorbar(center,y,xerr=[[center-low],[high-center]],fmt='o',capsize=3,
                color='#167278' if row['p_holm']<.05 else '#7b8793')
        axis.axvline(0,color='#c8c8c8',ls='--',lw=1)
        axis.set_yticks(range(3),['T3 vs '+a for a in ['B','T1','T2']])
        axis.set_title(LABELS[metric]); axis.set_xlabel('Difference oriented toward T3 benefit')
        axis.grid(axis='x',alpha=.2)
    fig.suptitle('P2b96 paired development analysis | one training seed\n95% unadjusted identity CI; Holm across18 tests',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.93])
    target=root/'figures/internal_paired_effects.png';target.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(target,dpi=180);plt.close(fig)


def input_image(sup, meta, kind):
    path = sup/'inputs'/(meta['inputs'][kind]+'.jpg')
    if not path.exists():
        path=sup.parent/'t3_p0_p1_20261028/inputs'/(meta['inputs'][kind]+'.jpg')
    assert sha(path) == meta['inputs'][kind]
    return path


def panels(root, sup, rows, metrics):
    # 固定输入哈希抽样代表普通案例；结果导向好/差案例独立标识，均完整保留。
    ordinary=sorted(rows,key=lambda r:hashlib.sha256(('T3PP/'+r['id']).encode()).hexdigest())[:6]
    ordered=sorted(rows,key=lambda r:metrics['T3'][r['id']]['iou']-metrics['B'][r['id']]['iou'])
    selections=[('ordinary_input_hash',ordinary),('T3_lower_IoU_than_B',ordered[:3]),('T3_higher_IoU_than_B',ordered[-3:])]
    index=[]
    for category, selected in selections:
        for row in selected:
            identity=row['id'];meta=read(sup/'evaluation/P2b/T3/images'/(identity+'.json'))
            for name, arms in [('internal',['B','T1','T2','T3']),('historical_C',['C','T3'])]:
                entries=[('Sketch',input_image(sup,meta,'sketch')),('Reference',input_image(sup,meta,'reference'))]
                entries += [(arm,sup/'evaluation/P2b'/arm/'images'/(identity+'.png')) for arm in arms]
                w,h=192,256
                panel=Image.new('RGB',(len(entries)*w,h+52),'white');draw=ImageDraw.Draw(panel)
                draw.text((8,4),f'{identity} | {category} | qualitative, previously viewed development set',fill='black')
                for j,(label,path) in enumerate(entries):
                    draw.text((j*w+6,28),label,fill='black')
                    panel.paste(Image.open(path).convert('RGB').resize((w,h),Image.Resampling.BILINEAR),(j*w,52))
                target=root/'figures'/name/category/(identity+'.png');target.parent.mkdir(parents=True,exist_ok=True);panel.save(target)
                index.append(dict(id=identity,category=category,panel=str(target),panel_sha256=sha(target),
                    models=','.join(arms),selection_rule='input hash' if category.startswith('ordinary') else 'post-result IoU difference T3-B'))
    csv_write(root/'provenance/panel_manifest.csv',index)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True)
    args=parser.parse_args();review(args.project)
