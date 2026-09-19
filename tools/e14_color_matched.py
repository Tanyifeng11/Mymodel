"""复用 E14 特征，以固定颜色匹配三元组比较各层，无须 GPU。

python -m tools.e14_color_matched --features eval_outputs/e14_pilot15/representations
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from tools.e14_pattern_probe import similarity, write_json


def source_units(rows):
    grouped = defaultdict(list)
    for i, row in enumerate(rows):
        grouped[row['source_group']].append(i)
    units = []
    for group, indices in sorted(grouped.items()):
        labels = {(rows[i]['pattern'], rows[i]['color_group']) for i in indices}
        if len(labels) != 1:
            raise ValueError(f'来源组 {group} 的图案/颜色标签不一致，先核对标注')
        pattern, color = labels.pop()
        units.append(dict(source_group=group, pattern=pattern, color_group=color,
                          indices=indices, sample_ids=[rows[i]['sample_id'] for i in indices]))
    return units


def source_scores(scores, units):
    # 对两个来源组的全部裁剪交叉相似度取平均；同源多图不增加统计票数。
    return np.array([[scores[np.ix_(a['indices'], b['indices'])].mean()
                      for b in units] for a in units], dtype=np.float64)


def match_triplets(color, units, tolerance, min_similarity):
    """仅依赖颜色和标签，不读取待评估层的相似度。保留所有满足条件的组合。"""
    triples = []
    for a, unit in enumerate(units):
        pool = [i for i, other in enumerate(units) if i != a
                and other['color_group'] == unit['color_group']
                and color[a, i] >= min_similarity]
        pos = [i for i in pool if units[i]['pattern'] == unit['pattern']]
        neg = [i for i in pool if units[i]['pattern'] != unit['pattern']]
        for p in pos:
            for n in neg:
                gap = float(color[a, p] - color[a, n])
                if tolerance is None or abs(gap) <= tolerance + 1e-12:
                    triples.append(dict(anchor=a, positive=p, negative=n,
                                        color_positive=float(color[a, p]), color_negative=float(color[a, n]),
                                        color_gap=gap))
    return triples


def evaluate_scores(scores, triples, units):
    anchors = defaultdict(list)
    for t in triples:
        delta = float(scores[t['anchor'], t['positive']] - scores[t['anchor'], t['negative']])
        anchors[t['anchor']].append(delta)
    details = []
    for a, margins in sorted(anchors.items()):
        x = np.asarray(margins)
        details.append(dict(source_group=units[a]['source_group'], pattern=units[a]['pattern'],
                            color_group=units[a]['color_group'], triplets=len(x),
                            accuracy=float(np.mean((x > 1e-7) + .5 * (np.abs(x) <= 1e-7))),
                            margin=float(x.mean())))
    accuracy = float(np.mean([r['accuracy'] for r in details])) if details else None
    return dict(accuracy=accuracy, anchor_sources=len(details), triplets=len(triples), per_anchor=details,
                by_pattern={p: float(np.mean([r['accuracy'] for r in details if r['pattern'] == p]))
                            for p in sorted({r['pattern'] for r in details})})


def compare_layers(metrics):
    pairs = [(f'cnn{i}_native', f'cnn{i}_pool') for i in range(1, 5)]
    pairs += [('clip_pool', 'clip_projected'), ('fused', 'resampler_raw'),
              ('resampler_raw', 'tokens_pre_ln'), ('tokens_pre_ln', 'tokens_post_ln'),
              ('tokens_post_ln', 'tcpm_neutral'), ('tokens_post_ln', 'tcpm_caption')]
    result = []
    for a, b in pairs:
        for readout in ('meanstd', 'flatten', 'set16_s0', 'set16_s1', 'set16_s2'):
            before, after = a + '__' + readout, b + '__' + readout
            if before not in metrics or after not in metrics:
                continue
            x = {r['source_group']: r['accuracy'] for r in metrics[before]['per_anchor']}
            delta = [r['accuracy'] - x[r['source_group']] for r in metrics[after]['per_anchor']]
            result.append(dict(before=before, after=after, anchor_sources=len(delta),
                               accuracy_change=float(np.mean(delta)) if delta else None))
    return result


def run(args):
    root = Path(args.features)
    index_path = root / 'index.json'
    index = json.loads(index_path.read_text(encoding='utf-8'))
    if not index.get('complete'):
        raise ValueError('特征提取未标记完成')
    rows = index['rows']
    units = source_units(rows)
    with np.load(root / rows[0]['feature_file']) as f:
        keys = sorted(f.files)
    values = defaultdict(list)
    # 只需一次读取每个压缩包；本批15张约90MB压缩特征。
    for row in rows:
        with np.load(root / row['feature_file']) as f:
            if sorted(f.files) != keys:
                raise ValueError('特征字段不一致')
            for key in keys:
                x = f[key]
                if not np.isfinite(x).all():
                    raise ValueError('特征包含非有限值：' + key)
                values[key].append(x)
    color = source_scores(similarity(values['image__color_hist']), units)
    settings = [('unmatched', None)] + [(f'tolerance_{t:g}', t) for t in args.tolerances]
    reports = {}
    for name, tolerance in settings:
        triples = match_triplets(color, units, tolerance, args.min_color_similarity)
        gaps = np.array([t['color_gap'] for t in triples])
        used = {t[k] for t in triples for k in ('anchor', 'positive', 'negative')}
        reports[name] = dict(tolerance=tolerance, triplets=triples, metrics={},
            coverage=dict(triplet_count=len(triples), participating_sources=len(used),
                          anchor_sources=len({t['anchor'] for t in triples}),
                          anchor_by_pattern=dict(Counter(
                              units[a]['pattern'] for a in {t['anchor'] for t in triples})),
                          color_abs_gap_max=float(np.abs(gaps).max()) if len(gaps) else None,
                          color_abs_gap_mean=float(np.abs(gaps).mean()) if len(gaps) else None,
                          status='exploratory' if triples else 'no_eligible_triplets'))
    for key in keys:
        scores = source_scores(similarity(values.pop(key)), units)
        for name, report in reports.items():
            report['metrics'][key] = evaluate_scores(scores, report['triplets'], units)
            # 基线也只在该容差保留下的anchor上计算，避免把anchor变化当匹配效果。
            anchor_ids = {t['anchor'] for t in report['triplets']}
            same_anchors = [t for t in reports['unmatched']['triplets'] if t['anchor'] in anchor_ids]
            report.setdefault('unmatched_same_anchors', {})[key] = evaluate_scores(scores, same_anchors, units)
    for name, report in reports.items():
        report['paired_layer_changes'] = compare_layers(report['metrics'])
        print(name, json.dumps(report['coverage']), flush=True)
    output = Path(args.output) if args.output else root / 'color_matched_report.json'
    write_json(output, dict(
        config=dict(features=str(root.resolve()), tolerances=args.tolerances, min_color_similarity=args.min_color_similarity),
        index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(), source_units=units, results=reports,
        interpretation=[
            '三元组：同颜色组、正样本同图案、负样本异图案；三个暂定来源互异。',
            '颜色条件：两个候选与anchor的直方图余弦相似度均达下限，二者相似度差不超过容差。',
            '固定颜色三元组由所有层共用；每个anchor先平均全部三元组，再按来源等权平均。',
            'accuracy为正样本更相似的比例，并列算0.5；这是三元组正确率，不是R@1。',
            'unmatched也采用相同颜色组和相似度下限；unmatched_same_anchors进一步固定anchor。',
            '默认容差0.01/0.025/0.05/0.1与相似度下限0.5是探索性设置，不按模型结果择优。',
            '仅匹配相对anchor的直方图相似度，不保证两候选颜色分布相同，不消除全部颜色混杂。',
            '颜色baseline仍可能偏离50%；须同时检查匹配后残余偏好，不把小gap视为完全无颜色偏差。',
            '来源未获商品身份验证；三元组共享图像而不独立，不按三元组数计算显著性。',
            '小样本结果仅用于收窄机制；不以局部改善/下降证明信息保留或永久丢失。']))
    print(output, flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', required=True)
    parser.add_argument('--output')
    parser.add_argument('--tolerances', type=float, nargs='+', default=[.01, .025, .05, .1])
    parser.add_argument('--min-color-similarity', type=float, default=.5)
    args = parser.parse_args()
    if any(t < 0 or t > 1 for t in args.tolerances) or not 0 <= args.min_color_similarity <= 1:
        parser.error('容差和颜色相似度下限必须在[0,1]内')
    run(args)


if __name__ == '__main__':
    main()
