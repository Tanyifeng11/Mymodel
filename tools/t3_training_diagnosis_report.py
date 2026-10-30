"""从服务器诊断包在本地汇总；报告不上传服务器。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

STAGES = ['E1_init', 'T3_u28048', 'T3_final']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def paired(a, b, key):
    ids = sorted(a.keys() & b.keys())
    values = np.array([a[i][key]-b[i][key] for i in ids if a[i].get(key) is not None and b[i].get(key) is not None], float)
    if not len(values):
        return dict(n=0, delta=None, ci_low=None, ci_high=None)
    seed = 301042+int(hashlib.sha256((key+'/'+'/'.join(ids)).encode()).hexdigest()[:7], 16)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), (10000, len(values)))].mean(1)
    lo, hi = np.quantile(samples, [.025, .975])
    return dict(n=len(values), delta=float(values.mean()), ci_low=float(lo), ci_high=float(hi), seed=seed)


def run(root):
    server = root/'server_results'
    metrics, loss = [], []
    for stage in STAGES:
        records = [read(p) for p in sorted((server/'metrics'/stage).glob('*/*.json'))]
        assert len(records) == 128
        metrics.extend(records)
        parts = [read(p) for p in sorted((server/'loss_probe'/stage).glob('*.json'))]
        assert len(parts) == 64 and all(len(v) == 4 for v in parts)
        for part in parts:
            row = dict(stage=stage, id=part[0]['id'], split=part[0]['split'])
            assert sorted(r['timestep'] for r in part) == [64, 256, 512, 768]
            for key in ['base', 'body_rgb', 'rest', 'total', 'denoise', 'gram', 'edge', 'color']:
                row[key] = float(np.mean([r[key] for r in part]))
            loss.append(row)
    noise = {}
    for stage in STAGES:
        for p in (server/'loss_probe'/stage).glob('*.json'):
            for row in read(p):
                key = (row['split'], row['id'], row['timestep'])
                fingerprints = tuple(row[k] for k in ['noise_sha256', 'latent_sha256', 'reference_latent_sha256'])
                if key in noise:
                    assert noise[key] == fingerprints, ('FIXED_PROBE_INPUT_MISMATCH', key, stage)
                noise[key] = fingerprints
    from PIL import Image
    manifest = read(server/'protocol/input_manifest.json')
    same = []
    for condition in ['original', 'swapped']:
        rows = [r for r in manifest if r['condition'] == condition]
        identical, changes = 0, []
        for row in rows:
            metas = [read(server/'outputs'/s/condition/(row['id']+'.json')) for s in STAGES]
            assert len({m['initial_latent_sha256'] for m in metas}) == 1
            assert metas[0]['initial_latent_sha256'] == row['historical_initial_latent_sha256']
            identical += metas[1]['output_sha256'] == metas[2]['output_sha256']
            im = [np.asarray(Image.open(server/'outputs'/s/condition/(row['id']+'.png')), np.float64) for s in STAGES[1:]]
            changes.append(float(np.abs(im[1]-im[0]).mean()))
        same.append(dict(condition=condition, n=len(rows), late_final_identical_png=identical,
                         late_final_mean_absolute_RGB_0to255=float(np.mean(changes))))
    csv_write(root/'audit/late_output_pixel_changes.csv', same)
    summaries, contrasts = [], []
    metric_keys = ['dists_rgb', 'dists_gray', 'color_js', 'dists_rgb_margin', 'dists_gray_margin', 'color_margin', 'iou', 'edge_f1', 'leak']
    for stage in STAGES:
        for condition in ['original', 'swapped']:
            group = [r for r in metrics if r['stage'] == stage and r['condition'] == condition]
            for key in metric_keys:
                values = [r[key] for r in group if r.get(key) is not None]
                summaries.append(dict(stage=stage, condition=condition, metric=key, n=len(values), mean=float(np.mean(values)) if values else None))
    for before in STAGES[:2]:
        for condition in ['original', 'swapped']:
            a = {r['id']: r for r in metrics if r['stage'] == 'T3_final' and r['condition'] == condition}
            b = {r['id']: r for r in metrics if r['stage'] == before and r['condition'] == condition}
            for key in metric_keys:
                contrasts.append(dict(comparison='T3_final-'+before, condition=condition, metric=key, **paired(a, b, key)))
    loss_summary, loss_contrasts = [], []
    for stage in STAGES:
        for split in ['train_probe32', 'dev_probe32']:
            group = [r for r in loss if r['stage'] == stage and r['split'] == split]
            assert len(group) == 32
            for key in ['base', 'body_rgb', 'rest', 'total']:
                loss_summary.append(dict(stage=stage, split=split, metric=key, n=32, mean=float(np.mean([r[key] for r in group]))))
    for before in STAGES[:2]:
        for split in ['train_probe32', 'dev_probe32']:
            a = {r['id']: r for r in loss if r['stage'] == 'T3_final' and r['split'] == split}
            b = {r['id']: r for r in loss if r['stage'] == before and r['split'] == split}
            for key in ['base', 'body_rgb', 'rest', 'total']:
                loss_contrasts.append(dict(comparison='T3_final-'+before, split=split, metric=key, **paired(a, b, key)))
    csv_write(root/'metrics/per_checkpoint_summary.csv', summaries)
    csv_write(root/'metrics/fixed_loss_probe_per_identity.csv', loss)
    csv_write(root/'metrics/fixed_loss_probe_summary.csv', loss_summary)
    csv_write(root/'statistics/generation_paired_effects.csv', contrasts)
    csv_write(root/'statistics/fixed_loss_paired_effects.csv', loss_contrasts)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei'], 'axes.unicode_minus': False})
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, key in zip(axes, ['base', 'body_rgb', 'rest']):
        for split, label, offset in [('train_probe32', '固定训练32', -.06), ('dev_probe32', '固定开发32', .06)]:
            values = [next(r['mean'] for r in loss_summary if r['stage'] == s and r['split'] == split and r['metric'] == key) for s in STAGES]
            ax.scatter(np.arange(3)+offset, values, label=label)
        ax.set_xticks(range(3), ['E1初始化', 'T3 28048', 'T3 28365'], fontsize=8)
        ax.set_title(key)
        ax.legend(fontsize=8)
    fig.suptitle('固定样本、固定噪声与时间步：三个实际留存点；早、中期缺失')
    fig.tight_layout()
    fig.savefig(root/'figures/fixed_loss_probe_checkpoints.png', dpi=150)
    plt.close(fig)
    # 展示案例只按输入身份哈希选取，与新模型分数无关。
    originals = {r['id']: r for r in manifest if r['condition'] == 'original'}
    swaps = {r['id']: r for r in manifest if r['condition'] == 'swapped'}
    ids = sorted(swaps, key=lambda i: hashlib.sha256(('T3_DIAG_301042/figure/'+i).encode()).hexdigest())[:4]
    def input_path(path):
        return server/'inputs'/Path(path).name
    from PIL import ImageDraw
    for condition in ['original', 'swapped']:
        page = Image.new('RGB', (6*160, 4*240), 'white')
        draw = ImageDraw.Draw(page)
        for j, identity in enumerate(ids):
            inputs = [('Sketch', input_path(originals[identity]['sketch'])), ('Old ref', input_path(originals[identity]['reference'])),
                      ('New ref', input_path(swaps[identity]['reference']))]
            paths = inputs+[(s, server/'outputs'/s/condition/(identity+'.png')) for s in STAGES]
            for i, (name, path) in enumerate(paths):
                page.paste(Image.open(path).convert('RGB').resize((160, 213)), (160*i, 240*j+26))
                draw.text((160*i+3, 240*j+2), identity+'\n'+name, fill='black')
        page.save(root/'figures'/('fixed_hash4_'+condition+'.png'))
    summary = dict(complete=True, generation_records=len(metrics), fixed_loss_identity_records=len(loss),
                   fixed_noise_latent_checks=len(noise), identical_inputs_across_stages=True,
                   early_middle_formal_checkpoints_available=False, stages=STAGES, pixel_changes=same,
                   caveat='initialization plus two late snapshots cannot establish early/middle generalization trajectory',
                   generation_contrasts=contrasts, fixed_loss_contrasts=loss_contrasts)
    write(root/'analysis_summary.json', summary)
    print(json.dumps(dict(generation_records=len(metrics), fixed_loss_records=len(loss), pixel_changes=same), ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('F:/fuxian/Mymodel/output_eval/t3_training_diagnosis_20261030'))
    run(p.parse_args().root)
