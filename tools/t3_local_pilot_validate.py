"""下载后的独立数值复核；不重跑训练或神经网络评价。"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def validate(root):
    raw = root/'server_results'
    rows = read(raw/'inputs/evaluation.json')
    values = read(raw/'metrics/per_image.json')
    assert len(rows) == 40 and len(values) == 120
    assert len({r['identity'] for r in rows if r['condition']=='original'}) == 16
    lookup = {(r['arm'],r['condition'],r['id']):r for r in values}
    assert len(lookup)==120
    # 输出字节和三臂初始噪声逐条件比较；不以随机seed文字相同代替。
    for row in rows:
        latent = []
        for arm in ['T3','residual','reference']:
            path = raw/'outputs'/arm/row['condition']/(row['id']+'.png')
            meta = read(path.with_suffix('.json'))
            assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['sha256']
            assert meta['reference_sha256']==row['inputs_sha256']['reference']
            assert meta['GT_used_in_generation'] is False
            latent.append(meta['initial_latent_sha256'])
        assert len(set(latent))==1
    # 从逐图数字重算全部均值与24个配对区间；不调用报告中的汇总函数。
    errors = []
    for summary in read(raw/'metrics/summary.json'):
        part=[r for r in values if r['arm']==summary['arm'] and r['condition']==summary['condition']]
        for metric in ['gray_dists','gray_margin','spectrum','iou','edge_f1','background_mae']:
            finite=[r[metric] for r in part if r[metric] is not None and np.isfinite(r[metric])]
            assert len(finite)==summary[metric+'_n']
            if finite:
                errors.append(abs(float(np.mean(finite))-summary[metric]))
    for comparison in read(raw/'metrics/paired.json'):
        other=comparison['comparison'].split('-',1)[1]
        differences=[]
        for row in rows:
            if row['condition']!=comparison['condition']:
                continue
            a=lookup['reference',row['condition'],row['id']][comparison['metric']]
            b=lookup[other,row['condition'],row['id']][comparison['metric']]
            if a is not None and b is not None:
                differences.append(a-b)
        assert len(differences)==comparison['n']
        if not differences:
            assert comparison['metric']=='gray_margin' and comparison['condition']=='original'
            assert comparison['mean'] is None and comparison['ci95'] is None
            continue
        assert len(differences)==16
        x=np.asarray(differences)
        rng=np.random.default_rng(110142)
        interval=np.quantile(x[rng.integers(len(x),size=(10000,len(x)))].mean(1),[.025,.975])
        errors.append(abs(float(x.mean())-comparison['mean']))
        errors.extend(abs(interval-np.asarray(comparison['ci95'])).tolist())
    assert max(errors)<1e-12
    train={arm:read(raw/'training'/arm/'complete.json') for arm in ['residual','reference']}
    assert all(r['updates']==400 and r['frozen_before']==r['frozen_after'] for r in train.values())
    assert len({r['stream_sha256'] for r in train.values()})==1
    # 打包作业终态来自之后的网页sacct截图；保留原包RUNNING快照原样。
    jobs=[]
    for line in (raw/'audit/resource_sacct.psv').read_text(encoding='utf-8').splitlines():
        fields=line.split('|')
        job=dict(id=fields[0],name=fields[1],state=fields[2],seconds=int(fields[4]),cpus=int(fields[5]),
                 gpus=1 if 'gres/gpu=1' in fields[6] else 0)
        if job['id']=='121929':
            job.update(state='COMPLETED',seconds=4,final_source='网页sacct终态截图')
        jobs.append(job)
    result=dict(output_files=120,paired_latent_checks=40,summary_and_CI_max_error=max(errors),
        frozen=True,training_stream_equal=True,jobs=jobs,
        gpu_seconds=sum(r['seconds']*r['gpus'] for r in jobs),
        cpu_seconds=sum(r['seconds']*r['cpus'] for r in jobs),
        scope='数值重算与文件审计；未在本地重跑DISTS或训练',
        final_screenshot_sha256=hashlib.sha256((root/'audit/server_final_status.jpg').read_bytes()).hexdigest())
    (root/'audit/local_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    validate(parser.parse_args().root)
