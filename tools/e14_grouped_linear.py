"""受控 E14：外层留一频率，内层留一频率选 C；冻结已有特征。

python -m tools.e14_grouped_linear --features eval_outputs/e14_controlled/112906/representations
"""
import argparse
import hashlib
import json
import re
import warnings
from pathlib import Path

import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


def make_splits(rows):
    frequencies = []
    for row in rows:
        match = re.fullmatch(r'(stripe|plaid|dots)_f(\d+)_a\d+_p\d+', row['sample_id'])
        if match is None or match[1] != row['pattern']:
            raise ValueError('此协议仅适用于已生成的 E14 受控图案集')
        frequencies.append(int(match[2]))
    freq = np.array(frequencies)
    folds = []
    labels = {r['pattern'] for r in rows}
    for held in sorted(set(frequencies)):
        train, test = np.flatnonzero(freq != held), np.flatnonzero(freq == held)
        if {rows[i]['pattern'] for i in test} != labels:
            raise ValueError('留出频率缺少类别')
        inner = [(np.flatnonzero((freq != held) & (freq != val)), np.flatnonzero(freq == val))
                 for val in sorted(set(frequencies) - {held})]
        for a, b in [(train, test)] + inner:
            if {rows[i]['source_group'] for i in a} & {rows[i]['source_group'] for i in b}:
                raise ValueError('训练/测试来源泄漏')
            if {rows[i]['pattern'] for i in a} != labels or {rows[i]['pattern'] for i in b} != labels:
                raise ValueError('折内类别不完整')
        folds.append((held, train, test, inner))
    if len(folds) < 3:
        raise ValueError('至少需要3档频率')
    return folds


def transform(x, train, test, components):
    scaler = StandardScaler()
    a = scaler.fit_transform(x[train])
    b = scaler.transform(x[test])
    # 常数颜色基线不需要PCA，避免零总方差导致解释方差警告。
    if np.max(np.abs(a)) < 1e-12:
        return np.zeros((len(train), 1)), np.zeros((len(test), 1))
    pca = PCA(n_components=components, svd_solver='randomized', random_state=42)
    return pca.fit_transform(a), pca.transform(b)


def probe(x, rows, folds):
    y = np.array([r['pattern'] for r in rows])
    classes = sorted(set(y))
    predictions = np.empty(len(y), dtype=y.dtype)
    results = []
    for held, train, test, inner in folds:
        dim = min(32, x.shape[1], min(len(a) - 1 for a, b in inner))
        encoded = [(a, b, *transform(x, a, b, dim)) for a, b in inner]
        grid = []
        for c in (.01, .1, 1., 10.):
            scores = []
            for a, b, xa, xb in encoded:
                model = LogisticRegression(C=c, max_iter=4000, class_weight='balanced', random_state=42)
                model.fit(xa, y[a])
                scores.append(float(balanced_accuracy_score(y[b], model.predict(xb))))
            grid.append(dict(C=c, inner_balanced_accuracy=float(np.mean(scores))))
        # 并列选择更小 C（更强正则），不读取外层结果。
        best = max(grid, key=lambda r: r['inner_balanced_accuracy'])['C']
        xa, xb = transform(x, train, test, dim)
        model = LogisticRegression(C=best, max_iter=4000, class_weight='balanced', random_state=42)
        model.fit(xa, y[train])
        pred = model.predict(xb)
        predictions[test] = pred
        results.append(dict(held_frequency=held, C=best, pca_dim=dim, inner_search=grid,
                            balanced_accuracy=float(balanced_accuracy_score(y[test], pred))))
    return dict(balanced_accuracy=float(balanced_accuracy_score(y, predictions)), folds=results,
                classes=classes, confusion_matrix=confusion_matrix(y, predictions, labels=classes).tolist(),
                per_class_accuracy={c: float(np.mean(predictions[y == c] == c)) for c in classes},
                predictions=[dict(sample_id=r['sample_id'], source_group=r['source_group'],
                                  target=r['pattern'], predicted=str(p)) for r, p in zip(rows, predictions)])


def run(root):
    root = Path(root)
    index_file = root / 'index.json'
    index = json.loads(index_file.read_text(encoding='utf-8'))
    if not index.get('complete'):
        raise ValueError('特征不完整')
    rows = index['rows']
    folds = make_splits(rows)
    with np.load(root / rows[0]['feature_file']) as f:
        keys = [k for k in sorted(f.files) if k.endswith('__meanstd') or 'color_hist' in k or 'gray_fft' in k
                or k in [s+'__flatten' for s in ('resampler_raw', 'tokens_pre_ln', 'tokens_post_ln', 'tcpm_neutral')]]
    features = {k: [] for k in keys}
    for row in rows:
        with np.load(root / row['feature_file']) as f:
            for k in keys:
                features[k].append(f[k])
    results = {}
    warnings.filterwarnings('error', category=ConvergenceWarning)
    for key in keys:
        x = np.stack(features.pop(key)).astype(np.float64)
        if not np.isfinite(x).all():
            raise ValueError('非有限特征')
        results[key] = probe(x, rows, folds)
        print(key, round(results[key]['balanced_accuracy'], 4), flush=True)
    report = dict(protocol='leave_one_frequency_out_nested_linear', sklearn_version=sklearn.__version__,
                  index_sha256=hashlib.sha256(index_file.read_bytes()).hexdigest(), sample_count=len(rows),
                  chance=1/len({r['pattern'] for r in rows}), results=results,
                  splits=[dict(held_frequency=h, train=a.tolist(), test=b.tolist(),
                               inner=[dict(train=c.tolist(), validation=d.tolist()) for c, d in inner])
                          for h, a, b, inner in folds],
                  limits=['所有读出共用外层和内层划分；Scaler/PCA均仅训练折拟合；内层选C。',
                          'meanstd与flatten分别评估，不以测试分数选择读出或PCA维度。',
                          '每类4个频率模板；变体相关；不能把48张视为48个独立自然来源。',
                          '预处理颜色基线若可分类则存在残余混杂；不能仅据分类分数宣称纹样信息保真。',
                          '低分不证明信息永久丢失；高分不证明生成模型利用了信息。'])
    output = root / 'grouped_linear_report.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', required=True)
    run(parser.parse_args().features)
