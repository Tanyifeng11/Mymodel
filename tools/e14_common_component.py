"""已有冻结特征的共同成分与留一来源中心化余弦检查，无需 GPU。"""
import argparse
import json
from pathlib import Path
import numpy as np
from tools.e14_pattern_probe import write_json


def cosine(x, y):
    return (x @ y.T) / np.maximum(np.linalg.norm(x, axis=-1)[..., None] * np.linalg.norm(y, axis=-1), 1e-15)


def inspect(x, rows):
    # 全样本均值只用于描述性几何统计，不用于测试样本检索。
    mean = x.mean(0)
    residual = x - mean
    total = np.square(x).sum(1).mean()
    singular = np.linalg.eigvalsh(residual @ residual.T).clip(0)
    weights = singular / max(singular.sum(), 1e-15)
    groups = np.array([r['source_group'] for r in rows])
    task = 'orientation' if all('orientation' in r for r in rows) else 'pattern'
    labels = np.array([r[task] for r in rows])
    result = dict(common_energy_fraction=float(np.square(mean).sum()/total),
        residual_energy_fraction=float(np.square(residual).sum(1).mean()/total),
        centered_effective_rank=float(np.exp(-(weights*np.log(np.maximum(weights,1e-15))).sum())),
        centered_top1_variance=float(weights[-1]), centered_top2_variance=float(weights[-2:].sum()), task=task)
    cross = groups[:,None] != groups[None,:]
    result['cross_source_cosine_mean'] = float(cosine(x,x)[cross].mean())
    result['cross_source_relative_distance_mean'] = float(np.sqrt(np.maximum(
        np.square(x).sum(1)[:,None]+np.square(x).sum(1)[None,:]-2*x@x.T,0))[cross].mean()/np.sqrt(total))
    if task == 'orientation':
        pairs = []
        within = []
        leave_out = []
        for group in sorted(set(groups)):
            ids = np.flatnonzero(groups == group)
            a,b = x[ids]
            pairs.append(dict(source_group=group,relative_l2=float(np.linalg.norm(a-b)/np.linalg.norm(a))))
            within.extend(x[ids]-x[ids].mean(0))
            remaining=x[groups!=group]
            remaining=remaining-remaining.mean(0)
            ev=np.linalg.eigvalsh(remaining@remaining.T).clip(0)
            leave_out.append(float(ev[-1]/max(ev.sum(),1e-15)))
        result['rotation_pairs'] = pairs
        result['within_source_rotation_variance_fraction']=float(np.square(within).sum()/np.square(residual).sum())
        result['leave_one_source_out_top1_variance_range']=[min(leave_out),max(leave_out)]
    for centered in (False,True):
        details = []
        for i,row in enumerate(rows):
            train = np.flatnonzero(groups != groups[i])
            mu = x[train].mean(0) if centered else np.zeros(x.shape[1])
            candidates = train
            if task == 'pattern':
                candidates = np.array([j for j in train if rows[j]['color_group']==row['color_group']])
            scores = cosine((x[i]-mu)[None,:], x[candidates]-mu)[0]
            # 同来源候选等权汇总，避免多个近重复图获得额外权重。
            units = sorted(set((groups[j],labels[j]) for j in candidates))
            scores, truths = zip(*[(float(scores[(groups[candidates]==g)&(labels[candidates]==label)].mean()),label)
                                   for g,label in units])
            scores,truths = np.array(scores),np.array(truths)
            positive = truths == labels[i]
            if not positive.any() or positive.all():
                continue
            top = np.isclose(scores,scores.max(),rtol=0,atol=1e-10)
            delta = scores[positive,None]-scores[None,~positive]
            details.append(dict(sample_id=row['sample_id'],source_group=row['source_group'],
                top1=float(positive[top].mean()),chance=float(positive.mean()),
                triplet=float(((delta>1e-10)+.5*(np.abs(delta)<=1e-10)).mean())))
        result['centered_cosine' if centered else 'raw_cosine'] = dict(
            **{k:float(np.mean([np.mean([r[k] for r in details if r['source_group']==g])
                              for g in sorted(set(r['source_group'] for r in details))]))
               for k in ['top1','triplet','chance']}, details=details)
    # StandardScaler 已做训练折去均值；额外中心化在线性标准化后代数等价。
    train = np.flatnonzero(groups != groups[0])
    mu,sd = x[train].mean(0),x[train].std(0)
    sd = np.where(sd<1e-12,1,sd)
    a = (x-mu)/sd
    z = x-mu
    b = (z-z[train].mean(0))/sd
    result['linear_centering_equivalence_max_abs'] = float(np.max(np.abs(a-b)))
    return result


def run(args):
    root = Path(args.features)
    index = json.loads((root/'index.json').read_text(encoding='utf-8'))
    if not index.get('complete'):
        raise ValueError('特征未完成')
    rows = index['rows']
    with np.load(root/rows[0]['feature_file']) as f:
        keys = [k for k in f.files if k.endswith(('__flatten','__meanstd'))]
    result = dict(features=str(root),sample_count=len(rows),source_count=len(set(r['source_group'] for r in rows)),
        config=index.get('config'), results={},
        limits=['共同均值由当前小样本估计，不代表全数据集公共方向。',
                '均值能量占比不是信息损失率；去均值余弦改变度量，不增加信息。',
                '每次检索仅用其他来源估计均值；pilot 按颜色组选择候选，但不等于精确匹配颜色。',
                '原线性探针已由 StandardScaler 在训练折中心化，无需重复宣称新的线性改善。'])
    for key in keys:
        vectors=[]
        for row in rows:
            with np.load(root/row['feature_file']) as f:
                vectors.append(f[key].reshape(-1).astype(np.float64))
        x=np.stack(vectors)
        if not np.isfinite(x).all():
            raise ValueError(key)
        result['results'][key]=inspect(x,rows)
        print(key,flush=True)
    write_json(Path(args.output),result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',required=True)
    p.add_argument('--output',required=True)
    run(p.parse_args())
