"""下载并解包后，本地校验、复算统计、画曲线与生成256方图对比页。"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from html import escape
from pathlib import Path

import numpy as np
from PIL import Image
from tools.bf_t3_ablation import ARMS, COEFFICIENTS
from tools.bf_t3_ablation_report import PRIMARY, RESPONSE, METRICS, holm
from tools.bf_t3_three_arm_report import paired


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def local(root):
    server = root / 'server_results'
    assert read(server / 'evaluation_complete.json')['complete']
    manifest = read(server / 'bundle_manifest.json')
    for name, info in manifest.items():
        path = server / name
        assert path.is_file() and path.stat().st_size == info['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == info['sha256'], name
    rows = read(server / 'evaluation.json')
    records = read(server / 'metrics/per_image.json')
    assert len(rows) == 192 and len(records) == 768
    assert Counter((r['arm'], r['condition']) for r in records) == {
        (a, c): n for a in ARMS for c, n in [('original', 128), ('neutral_original', 32), ('neutral_swapped', 32)]}
    lookup = {(r['arm'], r['id'], r['condition']): r for r in records}
    summary = read(server / 'metrics/summary.json')
    for entry in summary:
        values = [r[entry['metric']] for r in records if r['arm'] == entry['arm'] and r['condition'] == entry['condition'] and r[entry['metric']] is not None]
        assert len(values) == entry['n']
        assert entry['mean'] is None if not values else abs(np.mean(values) - entry['mean']) < 1e-12
    comparisons = read(server / 'metrics/paired_statistics.json')
    families = {}
    for row in comparisons:
        other = row['comparison'].split(' minus ')[1]
        values = []
        for case in rows:
            if case['condition'] != row['condition']: continue
            a, b = lookup[ARMS[-1], case['id'], row['condition']][row['metric']], lookup[other, case['id'], row['condition']][row['metric']]
            if a is not None and b is not None: values.append(a - b)
        recomputed = paired(values)
        assert row['n'] == recomputed['n']
        for k in ['mean', 'p']:
            assert row[k] is None if recomputed[k] is None else abs(row[k] - recomputed[k]) < 1e-12
        assert row['ci95'] is None if recomputed['ci95'] is None else np.allclose(row['ci95'], recomputed['ci95'], atol=1e-12)
        families.setdefault(row['family'], []).append(dict(recomputed, expected_holm=row['p_holm']))
    for family in families.values():
        holm(family)
        assert all(abs(r['p_holm'] - r['expected_holm']) < 1e-12 for r in family)
    # 每个条件四个模型的图像及输入/latent逐一配对。
    inputs = {Path(r[k]).name for r in rows for k in ['reference', 'sketch', 'gt']}
    for case in rows:
        latent = []
        for arm in ARMS:
            image = server / 'outputs' / arm / case['condition'] / (case['id'] + '.png')
            meta = read(image.with_suffix('.json'))
            assert meta['output_sha256'] == hashlib.sha256(image.read_bytes()).hexdigest()
            assert meta['seed'] == case['seed'] and not meta['GT_used_in_inference']
            assert meta['input_sketch_sha256'] == case['sketch_sha256']
            assert meta['input_reference_sha256'] == case['reference_sha256']
            latent.append(meta['initial_latent_sha256'])
        assert len(set(latent)) == 1
    for case in rows:
        if case['condition'] != 'neutral_swapped': continue
        for arm in ARMS:
            old = read(server / 'outputs' / arm / 'neutral_original' / (case['id'] + '.json'))
            new = read(server / 'outputs' / arm / 'neutral_swapped' / (case['id'] + '.json'))
            assert old['initial_latent_sha256'] == new['initial_latent_sha256']
            assert old['caption'] == new['caption'] == case['caption']
    figures = root / 'figures'; figures.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for arm in ARMS:
        folder = 'training' if arm in ARMS[1:3] else 'reused_training'
        with (server / folder / arm / 'training_log.csv').open(encoding='utf-8') as f:
            logs = list(csv.DictReader(f))
        assert len(logs) == 28365 and [int(r['step']) for r in logs] == list(range(1, 28366))
        b, rest = COEFFICIENTS[arm]
        total = [float(r['base']) + b * float(r['body']) + rest * float(r['rest']) for r in logs]
        for ax, values in zip(axes, [total, [b * float(r['body']) for r in logs], [rest * float(r['rest']) for r in logs]]):
            values = np.asarray(values)
            bins = [np.mean(values[i:i + 250]) for i in range(0, len(values), 250)]
            ax.plot([min(i + 250, len(values)) for i in range(0, len(values), 250)], bins, label=arm, lw=1.2)
    for ax, title in zip(axes, ['Total objective', 'Weighted body loss', 'Weighted rest loss']):
        ax.set_title(title); ax.set_xlabel('Successful updates'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle('Rank0 last microbatch, 250-update bins; different objectives have different totals', fontsize=10)
    fig.savefig(figures / 'training_losses.png', dpi=180); plt.close(fig)
    for condition, metrics, filename in [('original', PRIMARY, 'primary_effects.png'), ('neutral_swapped', RESPONSE, 'response_effects.png')]:
        fig, axes = plt.subplots(1, len(metrics), figsize=(max(10, 3 * len(metrics)), 4), constrained_layout=True)
        for ax, metric in zip(np.atleast_1d(axes), metrics):
            part = [r for r in comparisons if r['condition'] == condition and r['metric'] == metric]
            for i, value in enumerate(part):
                if value['n']:
                    mean = value['mean']; lo, hi = value['ci95']
                    ax.errorbar(mean, i, xerr=[[mean - lo], [hi - mean]], fmt='o', capsize=3)
            ax.axvline(0, color='gray', lw=.8); ax.set_title(metric)
            ax.set_yticks(range(len(part)), [r['comparison'].split(' minus ')[1] for r in part], fontsize=7)
            ax.grid(axis='x', alpha=.2)
        fig.suptitle('Full minus comparator; identity-paired descriptive95%CI')
        fig.savefig(figures / filename, dpi=180); plt.close(fig)
    page_rows = []
    labels = {'original': '主测试原文本', 'neutral_original': '中性文本·原参考', 'neutral_swapped': '中性文本·换参考'}
    for number, case in enumerate(rows, 1):
        cells = []
        for key in ['reference', 'sketch', 'gt']:
            p = 'server_results/inputs/' + Path(case[key]).name
            cells.append('<td><a href="' + p + '"><img class="input" loading="lazy" src="' + p + '"></a></td>')
        for arm in ARMS:
            raw = server / 'outputs' / arm / case['condition'] / (case['id'] + '.png')
            target = root / 'display_256x256' / arm / case['condition'] / raw.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(raw) as image: image.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR).save(target)
            src = target.relative_to(root).as_posix()
            cells.append('<td><a href="' + raw.relative_to(root).as_posix() + '"><img width="256" height="256" class="output" loading="lazy" src="' + src + '"></a></td>')
        info = '<th>%03d · %s<p>%s</p><p>%s</p><small>%s</small></th>' % (number, case['id'], labels[case['condition']], case['category'], escape(case['caption']))
        page_rows.append('<tr data-condition="' + case['condition'] + '" data-id="' + case['id'] + '">' + info + ''.join(cells) + '</tr>')
    main_table = []
    for arm in ARMS:
        values = {r['metric']: r for r in summary if r['arm'] == arm and r['condition'] == 'original'}
        main_table.append('<tr><th>' + arm + '</th>' + ''.join('<td>%.6f<small>n=%d</small></td>' % (values[k]['mean'], values[k]['n']) for k in PRIMARY) + '</tr>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>新T3区域监督消融</title><style>
body{font:15px/1.6 Microsoft YaHei,sans-serif;margin:24px;color:#23394b}h1{font-size:26px}a{color:#176498}table{border-collapse:separate;border-spacing:0}th,td{padding:9px;border-bottom:1px solid #d6e0e7;text-align:center}thead th{position:sticky;top:0;background:#e8f0f5;z-index:2}small{display:block;color:#708190}.scroll{overflow:auto;max-height:80vh;border:1px solid #d6e0e7}tbody th{min-width:235px;max-width:280px;text-align:left;font-weight:normal}td{min-width:175px}.input{width:160px;height:220px;object-fit:contain}.output{width:256px;height:256px;object-fit:fill}select,input{font:inherit;margin:10px;padding:5px}[hidden]{display:none!important}</style>
<h1>新T3区域监督消融：同一BF-SD架构</h1><p>四组为无附加区域监督、仅rest、仅body、完整双区域监督。两项消融重新训练，基础及完整模型复用核验的历史权重。主测试128件，另32件中性文本的原/换参考，共192组。所有输出保留，未按效果筛选。</p>
<p>模型图直接拉伸为256×256，点击查看原生384×512；输入保持比例。换参考行的旧GT仅作结构对照。单训练seed、已查看测试集和64平方patch限制推广与完整纹样保真结论。</p>
<p><a href="server_results/metrics/model_parameters.csv">模型参数</a> · <a href="server_results/metrics/summary.csv">完整指标</a> · <a href="server_results/metrics/paired_statistics.csv">配对统计</a> · <a href="figures/training_losses.png">损失曲线</a> · <a href="figures/primary_effects.png">主指标配对图</a> · <a href="figures/response_effects.png">参考响应配对图</a></p>
<table><thead><tr><th>模型</th>''' + ''.join('<th>' + k + '</th>' for k in PRIMARY) + '</tr></thead><tbody>' + ''.join(main_table) + '''</tbody></table>
<label>分组<select id="condition"><option value="original">主测试128件</option><option value="neutral_original">中性文本原参考32件</option><option value="neutral_swapped">中性文本换参考32件</option><option value="">全部192组</option></select></label><label>编号<input id="sample" type="search"></label><span id="count"></span>
<div class="scroll"><table><thead><tr><th>编号/原始文本</th><th>实际参考面料</th><th>草图</th><th>GT</th>''' + ''.join('<th>' + a + '</th>' for a in ARMS) + '</tr></thead><tbody>' + ''.join(page_rows) + '''</tbody></table></div><script>
const condition=document.getElementById('condition'),sample=document.getElementById('sample'),rows=[...document.querySelectorAll('tr[data-id]')];function filter(){let n=0;for(const r of rows){r.hidden=!!((condition.value&&r.dataset.condition!==condition.value)||!r.dataset.id.includes(sample.value.trim()));if(!r.hidden)n++;}document.getElementById('count').textContent=`显示${n} / ${rows.length}组`;}condition.addEventListener('change',filter);sample.addEventListener('input',filter);filter();</script></html>'''
    (root / 'index.html').write_text(page, encoding='utf-8')
    result = dict(complete=True, bundle_files_verified=len(manifest), summary_rows_recomputed=len(summary),
        paired_tests_recomputed=len(comparisons), output_images_verified=768,
        initial_noise_and_inputs_paired=True, neutral_caption_paired=True,
        plots_created=True, visual_review_pending=True, local_report_pending=True)
    (root / 'audit/local_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    local(parser.parse_args().root)
