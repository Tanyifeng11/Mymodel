"""完整对照的颜色配对、样本级汇总与离线mask重算。"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from tools.e14_pattern_probe import write_json

BASE_STEPS=[1,181,481,781,981]
FULL_STEPS=[1,101,141,181,221,261,481,781,981]


def color_pairs(rows,indices,hashes,root):
    hist={}
    for i in indices:
        with Image.open(Path(root)/rows[i].get('texture',rows[i].get('color'))) as im:
            a=np.asarray(im.convert('RGB')).reshape(-1,3)
        h=np.histogramdd(a,bins=(8,8,8),range=((0,256),)*3)[0].reshape(-1)
        hist[i]=h/h.sum()
    result={}
    for i in indices:
        eligible=[j for j in indices if j!=i and hashes[j]!=hashes[i] and rows[j]['cloth']!=rows[i]['cloth']]
        distances={j:float(np.linalg.norm(np.sqrt(hist[i])-np.sqrt(hist[j]))/np.sqrt(2)) for j in eligible}
        j=min(eligible,key=lambda k:(distances[k],k))
        result[i]=dict(index=j,hellinger_distance=distances[j],candidate_count=len(eligible),
                       interpretation='当前32张内颜色直方图最近，不保证同图案或充分颜色匹配')
    return result


def aggregate(records):
    if not records:return {}
    conditions=[k for k in records[0]['losses'] if k!='matched']+['wrong_average']
    result={}
    for condition in conditions:
        samples={}
        for r in records:
            loss=(r['losses']['wrong_1']+r['losses']['wrong_2'])/2 if condition=='wrong_average' else r['losses'][condition]
            samples.setdefault(r['sample_index'],[]).append(loss-r['losses']['matched'])
        means={str(i):float(np.mean(v)) for i,v in samples.items()}
        v=list(means.values())
        result[condition]=dict(mean_delta=float(np.mean(v)),median_delta=float(np.median(v)),
            matched_better_samples=sum(x>0 for x in v),sample_count=len(v),per_sample=means,
            worst_sample_indices=[int(i) for i in sorted(means,key=means.get)[:8]])
    return result


def analyze(root):
    from tools.e14_denoising_regions import partition
    root=Path(root);data=json.loads((root/'denoising_report.json').read_text(encoding='utf-8'))
    if not data['complete']:raise ValueError('推理尚未完成')
    masks={};mask_hashes={}
    import hashlib
    for sample in data['samples']:
        i=sample['sample_index'];p=root/'regions'/('%05d_mask.png'%i)
        mask_hashes[str(i)]=hashlib.sha256(p.read_bytes()).hexdigest()
        with Image.open(p) as im:masks[i]=partition(im,(64,48))
    reconstruction=[]
    for r in data['records']:
        with np.load(root/r['error_maps']) as arrays:
            r['region_losses']={name:{c:float(arrays[c][mask].mean()) if mask.any() else None for c in r['losses']}
                                for name,mask in masks[r['sample_index']].items()}
            for c in r['losses']:
                reconstruction.append(abs(float(arrays[c].mean())-r['losses'][c]))
    result=dict(mask_sha256=mask_hashes,error_map_vs_report_max_abs=max(reconstruction),regions={},
        limits=['正delta表示正确参考更好；以样本为汇总单位。',
                'base_steps用于与旧实验比较；dense_steps不能混入旧五步均值。',
                '自动mask与清单配对均待人工确认；颜色近邻不保证图案不同。'])
    result['mask_review_required']=[s['sample_index'] for s in data['samples']
        if s.get('region_diagnostics',{}).get('mask_low_confidence',True)]
    result['empty_regions']={str(i):[name for name,m in parts.items() if not m.any()]
                            for i,parts in masks.items() if any(not m.any() for m in parts.values())}
    result['validation']=dict(previous_replay_close=data.get('previous_replay_close'),
        previous_replay_max_abs_loss_difference=data.get('previous_replay_max_abs_loss_difference'),
        repeat_max_abs=max((r.get('matched_repeat_max_abs') or 0) for r in data['records']),
        record_count=len(data['records']),expected_record_count=len(data['samples'])*2*len(FULL_STEPS))
    for region in ['whole','interior','boundary','background']:
        records=data['records'] if region=='whole' else [dict(r,losses=r['region_losses'][region]) for r in data['records']
            if all(x is not None for x in r['region_losses'][region].values())]
        result['regions'][region]=dict(base_steps=aggregate([r for r in records if r['timestep'] in BASE_STEPS]),
            by_timestep={str(t):aggregate([r for r in records if r['timestep']==t]) for t in FULL_STEPS})
    write_json(root/'complete_summary.json',result)
    anomalies=[]
    for region in result['regions']:
        for c,v in result['regions'][region]['base_steps'].items():
            for i in v['worst_sample_indices']:
                anomalies.append(dict(region=region,condition=c,sample_index=i,
                    delta=v['per_sample'][str(i)],pair_image='pairs/%05d.png'%i,mask='regions/%05d_mask.png'%i))
    write_json(root/'anomaly_index.json',anomalies)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    analyze(p.parse_args().root)
