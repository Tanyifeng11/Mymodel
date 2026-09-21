"""四张真实参考的旋转敏感性；仅两种纹样，不训练方向分类器。"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
from tools.e14_pattern_probe import FIELDS, pixel_hash, similarity, write_json


def prepare(bundle):
    root=Path(bundle);data=json.loads((root/'inputs.json').read_text(encoding='utf-8'));rows=[]
    for pair in data['pairs']:
        for variant,item in pair['variants'].items():
            with Image.open(root/item['file']) as im:
                if pixel_hash(im.convert('RGB'))!=item['pixel_sha256']:raise ValueError('参考hash不符')
            row=dict.fromkeys(FIELDS,'')
            row.update(sample_id=pair['pair_id']+'_'+variant,texture=item['file'],caption='a garment',
                       pattern='stripe',color_group='neutral',source_group=pair['pair_id'],confirmed='1',
                       pattern_visible='1',pixel_sha256=item['pixel_sha256'],
                       notes='仅四图配对诊断；旋转变体同来源；不能用于五类检索')
            row.update(pair_id=pair['pair_id'],variant=variant,
                       orientation='vertical' if item['reference_metrics']['vertical_score']>0 else 'horizontal')
            rows.append(row)
    with (root/'feature_labels.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS+['pair_id','variant','orientation']);w.writeheader();w.writerows(rows)
    print(root/'feature_labels.csv')


def analyze(scores, rows):
    rotation=[];cross=[];orientation=[]
    for i in range(len(rows)):
        for j in range(i+1,len(rows)):
            d=float(1-scores[i,j])
            item=dict(left=rows[i]['sample_id'],right=rows[j]['sample_id'],cosine_distance=d)
            if rows[i]['source_group']==rows[j]['source_group']:rotation.append(item)
            else:
                item['same_orientation']=rows[i]['orientation']==rows[j]['orientation'];cross.append(item)
    for i,r in enumerate(rows):
        same=[j for j,s in enumerate(rows) if s['source_group']!=r['source_group'] and s['orientation']==r['orientation']]
        opposite=[j for j,s in enumerate(rows) if s['source_group']!=r['source_group'] and s['orientation']!=r['orientation']]
        margin=float(scores[i,same].mean()-scores[i,opposite].mean())
        orientation.append(dict(sample_id=r['sample_id'],same_minus_opposite_cosine=margin))
    denominator=float(np.mean([p['cosine_distance'] for p in cross]))
    return dict(rotation_pairs=rotation,cross_pattern_pairs=cross,
                mean_rotation_distance=float(np.mean([p['cosine_distance'] for p in rotation])),
                mean_cross_pattern_distance=denominator,
                rotation_to_cross_ratio=float(np.mean([p['cosine_distance'] for p in rotation])/denominator) if denominator>1e-8 else None,
                cross_pattern_orientation_preferences=orientation,
                note='变化大小不是方向可解码性的证明；跨纹样同方向偏好仅为四图探索性检查。')


def report(root):
    root=Path(root);index=json.loads((root/'index.json').read_text(encoding='utf-8'));rows=index['rows']
    if not index.get('complete') or len(rows)!=4:raise ValueError('需要完整四图特征')
    groups={r['source_group'] for r in rows}
    if len(groups)!=2 or any({r['orientation'] for r in rows if r['source_group']==g}!={'horizontal','vertical'} for g in groups):
        raise ValueError('每种参考必须有横竖配对')
    with np.load(root/rows[0]['feature_file']) as f:keys=sorted(f.files)
    results={}
    for key in keys:
        values=[]
        for row in rows:
            with np.load(root/row['feature_file']) as f:values.append(f[key])
        x=np.stack(values)
        if not np.isfinite(x).all():raise ValueError('特征非有限')
        results[key]=analyze(similarity(x),rows)
        if x.ndim==2:
            details=[]
            for g in sorted(groups):
                ids=[i for i,r in enumerate(rows) if r['source_group']==g];a,b=x[ids]
                details.append(dict(source_group=g,exact_equal=bool(np.array_equal(a,b)),
                    relative_l2=float(np.linalg.norm(a-b)/max((np.linalg.norm(a)+np.linalg.norm(b))/2,1e-12))))
            results[key]['vector_changes']=details
    write_json(root/'rotation_feature_report.json',dict(results=results,config=index['config'],
        limits=['四张图、两种纹样；不计算显著性，不拟合分类器。',
                '比较相同读出内的旋转/跨纹样距离比例，不直接跨层比较绝对距离。',
                '较大旋转变化不等于方向语义保留，也可能来自缩放和局部位置变化。',
                '两参考相似度不能完全分离纹样尺度与方向；需结合生成配对结果。']))
    print(root/'rotation_feature_report.json')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','report']);p.add_argument('--root',required=True)
    a=p.parse_args();(prepare if a.action=='prepare' else report)(a.root)
