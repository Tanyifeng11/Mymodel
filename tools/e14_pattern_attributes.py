"""在同类图案内部读出频率/方向；仅使用已提取特征，不训练生成模型。"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score
from tools.e14_grouped_linear import transform


def attributes(rows):
    result = []
    for row in rows:
        m = re.fullmatch(r'(stripe|plaid|dots)_f(\d+)_a(\d+)_p(\d+)', row['sample_id'])
        if not m or m[1] != row['pattern']:
            raise ValueError('要求受控图案数据及生成参数样本ID')
        result.append((int(m[2]), int(m[3]), int(m[4])))
    return np.array(result)


def folds_for(params, task):
    # 频率回归和方向分类均留一频率，整个原始模板组不跨折。
    # 频率分类留一角度：相位变体随角度一起留出；同频率跨角度是任务所需，不能声称独立模板泛化。
    column = 1 if task == 'frequency_cross_angle' else 0
    group = params[:, column]
    return [(int(v), np.flatnonzero(group != v), np.flatnonzero(group == v)) for v in sorted(set(group))]


def evaluate(x, params, task):
    is_regression = task == 'frequency_unseen_scale'
    y = np.log2(params[:, 0]) if is_regression else params[:, 1 if task == 'orientation_unseen_scale' else 0]
    predictions = np.zeros(len(y), dtype=float)
    baseline = np.zeros(len(y), dtype=float)
    folds = []
    for held, train, test in folds_for(params, task):
        dim = min(8, len(train)-1, x.shape[1])
        a, b = transform(x, train, test, dim)
        if is_regression:
            model = Ridge(alpha=1.)
            baseline[test] = y[train].mean()
        else:
            if set(y[train]) != set(y):
                raise ValueError('训练折缺少目标类别')
            model = LogisticRegression(C=1., max_iter=4000, class_weight='balanced', random_state=42)
        model.fit(a, y[train])
        predictions[test] = model.predict(b)
        fold = dict(held_group=held, train=train.tolist(), test=test.tolist(), pca_dim=dim)
        if is_regression:
            fold.update(mae_log2=float(np.abs(predictions[test]-y[test]).mean()),
                        baseline_mae_log2=float(np.abs(baseline[test]-y[test]).mean()))
        else:
            fold['balanced_accuracy'] = float(balanced_accuracy_score(y[test], predictions[test]))
        folds.append(fold)
    result = dict(folds=folds, predictions=predictions.tolist(), targets=y.tolist())
    if is_regression:
        result.update(mae_log2=float(np.abs(predictions-y).mean()),
                      baseline_mae_log2=float(np.abs(baseline-y).mean()),
                      baseline_predictions=baseline.tolist())
    else:
        result.update(balanced_accuracy=float(balanced_accuracy_score(y, predictions)), chance=1/len(set(y)))
    return result


def run(root):
    root = Path(root)
    index = json.loads((root/'index.json').read_text(encoding='utf-8'))
    if not index.get('complete'):
        raise ValueError('特征未完整提取')
    rows = index['rows']
    params = attributes(rows)
    with np.load(root/rows[0]['feature_file']) as f:
        keys = [k for k in sorted(f.files) if k.endswith('__meanstd') or 'color_hist' in k or 'gray_fft' in k
                or k in [s+'__flatten' for s in ('resampler_raw', 'tokens_pre_ln', 'tokens_post_ln', 'tcpm_neutral')]]
    arrays = {k: [] for k in keys}
    for row in rows:
        with np.load(root/row['feature_file']) as f:
            for k in keys:
                arrays[k].append(f[k])
    results = {}
    tasks = ['frequency_unseen_scale', 'orientation_unseen_scale', 'frequency_cross_angle']
    for key in keys:
        x = np.stack(arrays.pop(key)).astype(np.float64)
        if not np.isfinite(x).all():
            raise ValueError('特征包含非有限值')
        results[key] = {}
        for task in tasks:
            per_class = {}
            for pattern in ('stripe', 'plaid', 'dots'):
                indices = [i for i,r in enumerate(rows) if r['pattern'] == pattern]
                item = evaluate(x[indices], params[indices], task)
                item['sample_ids'] = [rows[i]['sample_id'] for i in indices]
                per_class[pattern] = item
            metric = 'mae_log2' if task == 'frequency_unseen_scale' else 'balanced_accuracy'
            result = dict(per_pattern=per_class, macro_score=float(np.mean([v[metric] for v in per_class.values()])), metric=metric)
            if metric == 'mae_log2':
                result['baseline_macro_mae_log2'] = float(np.mean([v['baseline_mae_log2'] for v in per_class.values()]))
            results[key][task] = result
        print(key, {t: round(results[key][t]['macro_score'],4) for t in tasks}, flush=True)
    output = root/'pattern_attribute_report.json'
    output.write_text(json.dumps(dict(results=results, config=dict(features=str(root.resolve()),
        classifier_C=1., ridge_alpha=1., max_pca_dim=8, seed=42),
        protocol=[
            '每个图案类别单独拟合；Scaler/PCA仅训练折拟合；固定正则和PCA上限，不按结果调参。',
            'frequency_unseen_scale：留一频率的log2频率线性回归；MAE越低越好，1表示平均一倍频程误差。',
            'orientation_unseen_scale：留一频率的0/30度分类，相位变体保持在同一折，随机水平50%。',
            'frequency_cross_angle：留一角度的四频率分类，相位变体一起留出，随机水平25%。',
            '频率跨角度分类刻意允许同频率进入训练，不代表新频率/独立模板泛化。',
            'dots方向是圆点阵列方向，不是单个圆点朝向；plaid方向是格线方向。',
            '仅48张规则生成图，结果可能利用周期、边缘或栅格规律；不等于自然参考身份保真。',
            '失败只表明该线性读出和划分未成功，不能证明信息不存在。']),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(output)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features',required=True)
    run(parser.parse_args().features)
