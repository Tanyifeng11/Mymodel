"""真实条纹方向探针：冻结特征，按参考来源留出；不训练生成模型。"""
import argparse
import csv
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tools.e14_pattern_probe import FIELDS, pixel_hash, write_json

# 人工核对主方向；排除斜纹、多方向拼接和点链状条纹。
SELECTION = {2: 'vertical', 3: 'vertical', 13: 'horizontal', 17: 'vertical',
             19: 'horizontal', 23: 'vertical', 25: 'horizontal', 30: 'vertical',
             36: 'vertical', 85: 'horizontal'}


def prepare(candidates, output):
    candidates, output = Path(candidates), Path(output)
    with (candidates / 'labels.csv').open(encoding='utf-8-sig') as f:
        source = {int(r['review_index']): r for r in csv.DictReader(f)}
    output.mkdir(parents=True, exist_ok=False)
    (output / 'images').mkdir()
    rows = []
    preview = Image.new('RGB', (512, 280 * len(SELECTION)), 'white')
    draw = ImageDraw.Draw(preview)
    for n, (idx, direction) in enumerate(SELECTION.items()):
        original = source[idx]
        if original['pattern'] != 'stripe' or original['confirmed'] != '1':
            raise ValueError('候选标注不符')
        im = Image.open(candidates / 'thumbnails' / ('%04d.png' % idx)).convert('RGB')
        # 缩略图只有与已审计原参考像素完全一致才可作为实验输入。
        if pixel_hash(im) != original['pixel_sha256']:
            raise ValueError('参考像素不符：%04d' % idx)
        for col, (variant, image) in enumerate([('original', im), ('rot90', im.transpose(Image.Transpose.ROTATE_90))]):
            sample = 'ref_%04d_%s' % (idx, variant)
            path = 'images/' + sample + '.png'
            image.save(output / path)
            row = dict.fromkeys(FIELDS, '')
            row.update(sample_id=sample, texture=path, caption='a garment', pattern='stripe',
                       color_group=original['color_group'], source_group=original['source_group'],
                       confirmed='1', pattern_visible='1', pixel_sha256=pixel_hash(image),
                       notes=original['notes'])
            row.update(variant=variant, original_sample_id=original['sample_id'],
                       orientation=direction if col == 0 else ('horizontal' if direction == 'vertical' else 'vertical'))
            rows.append(row)
            preview.paste(image.resize((256, 256)), (col * 256, n * 280))
            draw.text((col * 256 + 3, n * 280 + 258), sample + ' ' + row['orientation'], fill='black')
    with (output / 'feature_labels.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS + ['variant', 'original_sample_id', 'orientation'])
        writer.writeheader()
        writer.writerows(rows)
    preview.save(output / 'preview.png')
    write_json(output / 'selection.json', dict(rows=rows, limits=[
        '来源组为暂定样本来源，未验证布料身份；含轻微弯曲、接缝。',
        '仅测试横竖方向跨参考的线性可读出性，不等于完整图案保留或生成可利用性。']))
    with zipfile.ZipFile(output.with_suffix('.zip'), 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(output.rglob('*')):
            if p.is_file():
                archive.write(p, p.relative_to(output.parent))
    print(output.with_suffix('.zip'))


def make_splits(rows):
    groups = np.array([r['source_group'] for r in rows])
    unique = sorted(set(groups))
    if len(unique) < 3:
        raise ValueError('至少三个参考来源')
    folds = []
    for group in unique:
        test = np.flatnonzero(groups == group)
        if len(test) != 2 or {rows[i]['orientation'] for i in test} != {'horizontal', 'vertical'}:
            raise ValueError('每个来源必须恰好有横竖两张')
        folds.append((group, np.flatnonzero(groups != group), test))
    return folds


def probe(x, rows, no_pca=False):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    from tools.e14_grouped_linear import transform
    from sklearn.preprocessing import StandardScaler
    y = np.array([r['orientation'] for r in rows])
    pred = np.empty_like(y)
    folds = []
    fixed_pred = np.empty_like(y)
    for group, train, test in make_splits(rows):
        grid = []
        if no_pca:
            # 内层同样按来源留出；每次标准化只拟合当前内层训练数据。
            scores = {c: [] for c in (.0001, .001, .01, .1, 1., 10.)}
            for _, inner_train, inner_test in make_splits([rows[i] for i in train]):
                ia, ib = train[inner_train], train[inner_test]
                scaler = StandardScaler()
                xa = scaler.fit_transform(x[ia])
                xb = scaler.transform(x[ib])
                xa, xb = exact_linear_coordinates(xa, xb)
                for c in scores:
                    model = linear_model(c)
                    model.fit(xa, y[ia])
                    scores[c].append(float(np.mean(model.predict(xb) == y[ib])))
            grid = [dict(C=c, inner_accuracy=float(np.mean(v))) for c, v in scores.items()]
            best = max(grid, key=lambda item: item['inner_accuracy'])['C']
            scaler = StandardScaler()
            a, b = scaler.fit_transform(x[train]), scaler.transform(x[test])
            a, b = exact_linear_coordinates(a, b)
            model = linear_model(best)
            control = linear_model(1.)
            control.fit(a, y[train])
            fixed_pred[test] = control.predict(b)
        else:
            # 固定正则强度，不以外层测试结果挑选参数；标准化/PCA仅拟合训练折。
            dim = min(16, len(train) - 1, x.shape[1])
            a, b = transform(x, train, test, dim)
            model = LogisticRegression(C=1., max_iter=4000, random_state=42)
        model.fit(a, y[train])
        pred[test] = model.predict(b)
        folds.append(dict(source_group=group, accuracy=float(np.mean(pred[test] == y[test]))))
        if no_pca:
            folds[-1].update(selected_C=best, inner_grid=grid)
    result = dict(balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                pairs_both_correct=sum(f['accuracy'] == 1 for f in folds),
                num_pairs=len(folds), folds=folds,
                classes=['horizontal', 'vertical'],
                confusion_matrix=confusion_matrix(y, pred, labels=['horizontal', 'vertical']).tolist(),
                predictions=[dict(sample_id=r['sample_id'], truth=str(a), prediction=str(b))
                             for r, a, b in zip(rows, y, pred)])
    if no_pca:
        result['fixed_C1_control'] = dict(balanced_accuracy=float(balanced_accuracy_score(y, fixed_pred)),
            predictions=fixed_pred.tolist())
    return result


def linear_model(c):
    from sklearn.linear_model import LogisticRegression
    # 与原PCA探针保持相同分类器；不删除任何特征维度。
    return LogisticRegression(C=c, solver='lbfgs', max_iter=4000, random_state=42)


def exact_linear_coordinates(train, test):
    # 高维L2线性模型的最优权重位于训练向量张成的空间。
    # 完整经济型QR只是等距换坐标：不按方差排序、不截断分量，
    # 保留全部训练内积及测试-训练内积，目标函数与原维度一致。
    # 测试数据不参与构造基；低维输入直接使用原坐标。
    if train.shape[1] <= train.shape[0]:
        return train, test
    from scipy.linalg import qr
    basis, r = qr(np.asarray(train, dtype=np.float64).T, mode='economic')
    return r.T, np.asarray(test, dtype=np.float64) @ basis


def evaluate(root, no_pca=False):
    import sklearn
    root = Path(root)
    index = json.loads((root / 'index.json').read_text(encoding='utf-8'))
    if not index.get('complete'):
        raise ValueError('特征提取未完成')
    rows = index['rows']
    make_splits(rows)
    with np.load(root / rows[0]['feature_file']) as f:
        keys = sorted(k for k in f.files if k.endswith(('__meanstd', '__flatten')) or
                      k.endswith(('__color_hist', '__gray_fft64', '__gray_fft128')))
    results = {}
    for key in keys:
        values = []
        for row in rows:
            with np.load(root / row['feature_file']) as f:
                values.append(f[key].reshape(-1))
        x = np.stack(values)
        if not np.isfinite(x).all():
            raise ValueError('特征非有限：' + key)
        results[key] = probe(x, rows, no_pca=no_pca)
        print(key, results[key]['balanced_accuracy'], flush=True)
    protocol = dict(split='leave_one_source_out', C=1., pca_max_components=16,
                      preprocessing='train_fold_only', sklearn_version=sklearn.__version__)
    if no_pca:
        protocol.update(C='inner_leave_one_source_out', C_grid=[.0001, .001, .01, .1, 1., 10.],
                        pca_max_components=None, solver='lbfgs', tie_break='smallest_C',
                        fixed_C1_control=True, coordinates='full_training_span_QR_no_truncation')
    write_json(root / ('real_orientation_no_pca_report.json' if no_pca else 'real_orientation_report.json'),
        dict(results=results, config=index['config'], protocol=protocol,
        limits=['仅少量暂定来源组；无显著性结论。',
                '方向可读出不代表生成器利用；低分也不能证明信息完全丢失。',
                'meanstd与flatten分别比较；不根据测试分数挑选最佳层或超参数。',
                '原图颜色直方图旋转不变，预期50%；预处理后直方图可能因缩放变化。']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--candidates', default='eval_outputs/e14_candidates/candidates')
    p.add_argument('--output', default='eval_outputs/e14_real_orientation_inputs')
    p = sub.add_parser('evaluate')
    p.add_argument('--features', required=True)
    p.add_argument('--no-pca', action='store_true', help='不降维，内层来源交叉验证选择L2正则强度')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.candidates, args.output)
    else:
        evaluate(args.features, args.no_pca)
