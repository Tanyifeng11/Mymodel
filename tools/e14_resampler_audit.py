"""从缓存检查 resampler 输出冗余、旋转响应及颜色关联，不推断 attention 因果。"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from tools.e14_pattern_probe import write_json, pixel_hash


def main(args):
    root, inputs = Path(args.features), Path(args.inputs)
    index = json.loads((root/'index.json').read_text(encoding='utf-8'))
    if not index.get('complete'):
        raise ValueError('特征不完整')
    rows=index['rows']
    groups=sorted(set(r['source_group'] for r in rows))
    ids=[np.array([i for i,r in enumerate(rows) if r['source_group']==g]) for g in groups]
    for pair in ids:
        if len(pair)!=2 or {rows[i]['variant'] for i in pair}!={'original','rot90'}:
            raise ValueError('需要每来源原图和旋转配对')
    attributes=[]
    for pair in ids:
        row=next(rows[i] for i in pair if rows[i]['variant']=='original')
        with Image.open(inputs/row['texture']) as im:
            rgb=im.convert('RGB')
            if pixel_hash(rgb)!=row['pixel_sha256']:
                raise ValueError('参考 hash 不匹配')
            a=np.asarray(rgb,dtype=float)/255
        gray=a@np.array([.2126,.7152,.0722])
        attributes.append([*a.mean((0,1)),gray.mean(),gray.std(),(a.max(2)-a.min(2)).mean()])
    attributes=np.array(attributes)
    names=['mean_R','mean_G','mean_B','luminance_mean','luminance_std','chroma_range_mean']
    report=dict(features=str(root),sample_count=len(rows),source_count=len(groups),layers={},
        limits=['颜色关联仅10个来源的描述性相关，不是因果或泛化证明。',
                'PCA仅用于几何审计，不用于删特征或训练分类器；主成分符号任意。',
                'token相似不证明attention均匀；没有权重无法定位Q/K/V或各head。'])
    for key in ['clip_projected','cnn_projected_combined','fused','resampler_raw','tokens_pre_ln','tokens_post_ln']:
        x=[]
        for row in rows:
            with np.load(root/row['feature_file']) as f:
                x.append(f[key+'__flatten'].astype(np.float64).reshape(-1,768))
        x=np.stack(x)
        flat=x.reshape(len(rows),-1)
        token_mean=x.mean(1,keepdims=True)
        energy=np.square(x).sum((1,2))
        redundancy=np.square(x-token_mean).sum((1,2))/energy
        source=np.stack([flat[pair].mean(0) for pair in ids])
        centered=source-source.mean(0)
        ev,u=np.linalg.eigh(centered@centered.T)
        order=np.argsort(ev)[::-1];ev=ev[order].clip(0);u=u[:,order]
        scores=u[:,:2]*np.sqrt(ev[:2])
        correlations={n:[float(np.corrcoef(attributes[:,j],scores[:,pc])[0,1]) for pc in range(2)] for j,n in enumerate(names)}
        pairs=[]
        for group,pair in zip(groups,ids):
            a,b=flat[pair]
            ma,mb=token_mean[pair].reshape(2,-1)
            pairs.append(dict(source_group=group,relative_flat=float(np.linalg.norm(a-b)/np.linalg.norm(a)),
                              relative_token_mean=float(np.linalg.norm(ma-mb)/np.linalg.norm(ma))))
        within=sum(np.square(flat[pair]-flat[pair].mean(0)).sum() for pair in ids)
        total=np.square(flat-flat.mean(0)).sum()
        report['layers'][key]=dict(token_count=x.shape[1],
            token_nonmean_energy_fraction_mean=float(redundancy.mean()),
            token_nonmean_energy_fraction_range=[float(redundancy.min()),float(redundancy.max())],
            rotation_fraction_of_centered_variance=float(within/total),
            source_PC_variance_fractions=(ev[:2]/ev.sum()).tolist(),
            source_PC_attribute_correlations=correlations,rotation_pairs=pairs,
            source_scores=[dict(source_group=g,pc1=float(s[0]),pc2=float(s[1]),
                                attributes=dict(zip(names,map(float,a)))) for g,s,a in zip(groups,scores,attributes)])
        print(key,flush=True)
    write_json(Path(args.output),report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',required=True)
    p.add_argument('--inputs',default='eval_outputs/e14_real_orientation_inputs')
    p.add_argument('--output',required=True)
    main(p.parse_args())
