"""本地合并12方法指标与指定10组展示；保留之前的原始数据。"""
import argparse
import csv
import hashlib
import json
import os
from html import escape
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

MODELS = ['E0', 'E1', 'B', 'T1', 'T2', 'C', 'T3', 'BF-Fashion', 'ControlNet+IP-Adapter', 'LOTS', 'SGDiff', 'TextureGAN']
KEYS = ['edge_f1', 'leak', 'iou', 'edge_l1', 'lpips', 'tcf_lab']
LABELS = ['Edge F1 ↑', 'Leak ↓', 'IoU ↑', 'Edge L1 ↓', 'LPIPS ↓', 'TCF-LAB ↓']
IDS = ['129524367', '144216852', '88977898', '209081216', '203067804', '64959928', '170363086', '148830337', '77823388', '195714159']


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def csv_read(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def csv_write(p, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def contain(path):
    im = ImageOps.contain(Image.open(path).convert('RGB'), (256, 256))
    tile = Image.new('RGB', (256, 256), 'white')
    tile.paste(im, ((256-im.width)//2, (256-im.height)//2))
    return tile


def main(root, previous):
    server = root/'server_results'
    rows = read(server/'protocol/input_manifest.json')
    old_rows = read(previous/'server_results/protocol/input_manifest.json')
    assert len(rows) == 128 and [r['id'] for r in rows] == [r['id'] for r in old_rows]
    for a, b in zip(rows, old_rows):
        assert all(a[k] == b[k] for k in ['seed', 'caption', 'reference_sha256', 'sketch_sha256', 'gt_sha256'])
    meta = read(server/'metrics/TextureGAN/complete.json')
    assert meta['n'] == 128 and meta['complete'] and meta['missing_generation'] == 0
    hint_fallbacks = 0
    for row in rows:
        generation = read(server/'outputs/TextureGAN/original'/(row['id']+'.json'))
        assert generation['seed'] == row['seed'] and not generation['GT_used_in_inference']
        assert generation['input_sketch_sha256'] == row['sketch_sha256']
        assert generation['input_reference_sha256'] == row['reference_sha256']
        assert not generation['text_used'] and len(generation['hint_patches']) == 2
        hint_fallbacks += int('hint_only_fallback' in generation['mask_info'])
    raw = []
    for model in MODELS:
        base = server if model == 'TextureGAN' else previous/'server_results'
        part = [dict(read(p), source_file=str(p), category=next(r['category'] for r in rows if r['id'] == p.stem))
            for p in sorted((base/'metrics'/model/'original').glob('*.json'))]
        assert len(part) == 128 and {r['id'] for r in part} == {r['id'] for r in rows}
        assert all(r['valid_generation'] for r in part)
        for r in part:
            output = base/'outputs'/model/'original'/(r['id']+'.png')
            assert hashlib.sha256(output.read_bytes()).hexdigest() == r['output_sha256']
        raw += part
    summary, long_summary, category, paired = [], [], [], []
    old = read(previous/'workbook_data.json')
    for model in MODELS:
        entry = dict(model=model, input_modalities='草图+面料（无文本）' if model == 'TextureGAN' else next(r['input_modalities'] for r in old['original'] if r['model'] == model))
        for key in KEYS:
            vals = [r[key] for r in raw if r['arm'] == model and r.get(key) is not None]
            assert all(np.isfinite(x) for x in vals)
            entry.update({key: float(np.mean(vals)) if vals else None, key+'_n': len(vals)})
            long_summary.append(dict(model=model, metric=key, mean=entry[key], n=len(vals), direction='up' if key in ['edge_f1', 'iou'] else 'down'))
            for cat in ['top', 'outwear', 'pants', 'dress']:
                v = [r[key] for r in raw if r['arm'] == model and r['category'] == cat and r.get(key) is not None]
                category.append(dict(model=model, category=cat, metric=key, mean=float(np.mean(v)) if v else None, n=len(v)))
        summary.append(entry)
    records = {(r['arm'], r['id']): r for r in raw}
    for model in [m for m in MODELS if m != 'T3']:
        for key in KEYS:
            dif = np.array([records['T3', r['id']][key]-records[model, r['id']][key] for r in rows
                if records['T3', r['id']].get(key) is not None and records[model, r['id']].get(key) is not None])
            paired.append(dict(comparator=model, metric=key, n=len(dif),
                mean_T3_minus_comparator=float(dif.mean()) if len(dif) else None,
                T3_better_count=int(np.sum(dif > 0 if key in ['edge_f1', 'iou'] else dif < 0)), ties=int(np.sum(dif == 0))))
    csv_write(root/'per_image_metrics_12models.csv', raw)
    csv_write(root/'all_model_metrics.csv', long_summary)
    csv_write(root/'original128_wide.csv', summary)
    csv_write(root/'category_original_metrics.csv', category)
    csv_write(root/'T3_pairwise_metrics.csv', paired)
    params = csv_read(previous/'model_parameters.csv')
    prov = read(server/'protocol/controlled_protocol.json')['provenance']
    params.append(dict(model_id='TextureGAN', role='external_pretrained',
        model_definition='官方完整生成器，现代PyTorch兼容；自动草图内部提示接口', input_modalities='sketch+reference; no text',
        training_seed='unknown', training_steps='checkpoint epoch86 iteration1300; not total training steps',
        initialization_checkpoint='historical pretrained cloth model; no new training', checkpoint='/share/home/u2515283058/TextureGAN/weights/final_cloth_finetune.pth',
        checkpoint_sha256=prov['sha256'], dataset_split='same BF/test 128 garment identities, original only',
        inference_config='CPU 256x256; 2 reference-native patches,20..40px; fixed identity seed; float32 eval(); align_corners=True; 8,001,641 parameters',
        metric_protocol_version='same frozen original128 six metrics at384x512; CPU with16-case GPU replay', result_source=str(server/'metrics/TextureGAN/per_image.csv'),
        status='128 generated/evaluated; pretrained inference reproduction',
        comparability_notes='mirror: https://github.com/spuliz/masterproject ; source: https://github.com/janesjanes/Pytorch-TextureGAN ; official S3 deleted, original byte identity unverified; pretraining overlap unknown; no text; automatic hint placement'))
    csv_write(root/'model_parameters.csv', params)
    resources = read(server/'audit/resources.json')
    notes = old['notes'] + [
        '本次只新增TextureGAN，复用已有11方法冻结结果；12方法各128张，总1536条指标。',
        'TextureGAN使用可追溯应用项目镜像的服装权重；与已失效官方文件无法逐字节核对。未从头训练，不称完整训练复现。',
        f'TextureGAN仅接收草图和参考面料局部块，不接收文本。两块20—40像素，固定seed放置；{hint_fallbacks}件使用草图墨迹凸包作为提示位置后备，评价mask保持原协议。']
    if (server/'audit/CPU_GPU_metric_replay.json').exists():
        replay = read(server/'audit/CPU_GPU_metric_replay.json')
        assert replay['passed'] and replay['n'] == 16
        notes.append('本次TextureGAN使用CPU推理和评价；同环境16例六指标GPU历史回放最大绝对差：'+str(max(replay['maximum_absolute_difference'].values()))+'。')
    notes[0] = '同批128件纯服装：上装、外套、裤装、连衣裙各32；仅原参考；12模型各128张，共1536条记录。'
    notes[7] = '本次10组展示沿用用户指定编号和原顺序，不另外挑选胜出样本；其中原有条纹/波点案例也保留。完整128件始终计入指标，展示不代表总体胜率。'
    write(root/'workbook_data.json', dict(original=summary, raw=raw, parameters=params, notes=notes, pairwise=paired, resources=resources))
    def asset(row, key):
        if key in ['reference', 'sketch', 'gt']:
            return server/'inputs'/Path(row[key]).name
        if key == 'TextureGAN':
            return server/'native/TextureGAN/original'/(row['id']+'.png')
        return previous/'display_256x256'/key/(row['id']+'.png')
    def url(path):
        return escape(Path(os.path.relpath(path, root)).as_posix(), quote=True)
    cols = ['reference', 'sketch', 'gt']+MODELS
    labels = ['参考面料原图', '草图', 'GT']+MODELS
    def table(part):
        out = '<div class="scroll"><table><thead><tr><th>编号 / 实际文本</th>'+''.join('<th>'+escape(x)+'</th>' for x in labels)+'</tr></thead><tbody>'
        for r in part:
            out += '<tr id="sample-'+r['id']+'"><th>'+r['category']+' · '+r['id']+'<p>'+escape(r['caption'])+'</p><small>seed '+str(r['seed'])+'</small></th>'
            for key in cols:
                link = url(asset(r, key))
                out += '<td><a href="'+link+'"><img loading="lazy" src="'+link+'" alt="'+escape(key+' '+r['id'])+'"></a></td>'
            out += '</tr>'
        return out+'</tbody></table></div>'
    selected = [next(r for r in rows if r['id'] == x) for x in IDS]
    prefix = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>TextureGAN补充：12模型对比</title><style>body{font:15px/1.6 Microsoft YaHei,sans-serif;color:#18364c;margin:24px}a{color:#176498}.scroll{overflow:auto;max-height:80vh;border:1px solid #d5e0e8}table{border-collapse:separate;border-spacing:0}th,td{padding:9px;border-bottom:1px solid #d5e0e8;text-align:center}thead th{position:sticky;top:0;background:#e8f0f5;z-index:2;min-width:192px}tbody th{min-width:260px;max-width:300px;text-align:left;font-weight:normal}img{display:block;width:176px;height:176px;object-fit:contain}small{display:block;color:#647e8a}</style><h1>TextureGAN补充：12模型、同批128件</h1><p>模型展示均256×256；参考图、草图和GT保持比例。TextureGAN无文本输入，原生256×256；评价统一384×512。保留全部样本，指定10组只用于展示。</p><p>服装权重来自2020年公开应用项目镜像，官方S3已失效；与原官方文件字节一致性无法核验。未训练或按测试结果调参。</p>'
    metrics = '<table><thead><tr><th>模型</th>'+''.join('<th>'+x+'</th>' for x in LABELS)+'</tr></thead><tbody>'
    for row in summary:
        metrics += '<tr><th>'+row['model']+'</th>'+''.join('<td>'+('N/A' if row[k] is None else f'{row[k]:.6f}')+'<small>n='+str(row[k+'_n'])+'</small></td>' for k in KEYS)+'</tr>'
    metrics += '</tbody></table>'
    nav = '<p><a href="outputs/texturegan_comparison_20261102/12模型指标与参数.xlsx">12模型指标与参数Excel</a> · <a href="original128_wide.csv">12模型指标CSV</a> · <a href="model_parameters.csv">模型与推理参数CSV</a> · <a href="selected10.html">指定10组对比</a> · <a href="全部12模型对比_指定10组.png">高清总图</a></p>'
    (root/'index.html').write_text(prefix+nav+metrics+'<h2>完整128件</h2>'+table(rows)+'</html>', encoding='utf-8')
    (root/'selected10.html').write_text(prefix+nav+'<h2>用户指定10组，原顺序</h2>'+table(selected)+'</html>', encoding='utf-8')
    # 原11方法页面和原图保留，追加新版入口。
    for page, relative in [(previous/'index.html', '../texturegan_comparison_20261102/index.html'),
                           (previous/'selected10_all_models/index.html', '../../texturegan_comparison_20261102/selected10.html')]:
        original_html = page.read_text(encoding='utf-8')
        if relative not in original_html:
            link = '<p><a href="'+relative+'">新增TextureGAN：12模型完整指标和对比图</a></p>'
            page.write_text(original_html.replace('</h1>', '</h1>'+link, 1), encoding='utf-8')
    cell, top, rh, margin = 272, 130, 340, 24
    canvas = Image.new('RGB', (margin*2+len(cols)*cell, top+len(selected)*rh+margin), 'white')
    d = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 20)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 18)
    title = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 30)
    d.text((margin, 12), '用户指定10组：全部12模型对比（新增TextureGAN）', font=title, fill='#18364c')
    d.text((margin, 57), '生成图统一256×256展示，参考/草图/GT保持比例；TextureGAN为服装预训练生成器+自动局部提示，无文本输入。', font=small, fill='#526775')
    d.rectangle((margin, 92, canvas.width-margin, 129), fill='#e8f0f5')
    for j, label in enumerate(labels):
        text = 'ControlNet+IP-Adapter' if j == 11 else label
        d.text((margin+j*cell+(cell-d.textlength(text, font=font))/2, 98), text, font=font, fill='#18364c')
    for i, r in enumerate(selected):
        y = top+i*rh
        d.text((margin, y+5), f'{i+1:02d} · {r["id"]} · {r["category"]}', font=font, fill='#18364c')
        d.text((margin, y+35), 'Text: '+r['caption'], font=small, fill='#526775')
        for j, key in enumerate(cols):
            tile = contain(asset(r, key))
            canvas.paste(tile, (margin+j*cell+8, y+69))
    canvas.save(root/'全部12模型对比_指定10组.png')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, key in zip(axes.flat, KEYS):
        ax.bar(range(12), [r[key] if r[key] is not None else np.nan for r in summary], color=['#bc722a' if m == 'TextureGAN' else '#2679a6' if m == 'T3' else '#8195a0' for m in MODELS])
        ax.set_xticks(range(12), [m.replace('ControlNet+IP-Adapter', 'CN+IPA') for m in MODELS], rotation=60, ha='right', fontsize=8)
        ax.set_title(key+(' (higher better)' if key in ['edge_f1', 'iou'] else ' (lower better)'))
        ax.grid(axis='y', alpha=.15)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Same 128 garments | 12 models | all outputs included | modalities differ')
    fig.savefig(root/'metrics_12models.png', dpi=170)
    plt.close(fig)
    write(root/'audit/local_merge.json', dict(models=12, cases=128, metric_records=len(raw),
        selected_ids=IDS, old_outputs_unchanged=True, input_hashes_match=True))
    print(json.dumps(dict(metrics=summary[-1], texturegan_pairwise=[r for r in paired if r['comparator'] == 'TextureGAN']), ensure_ascii=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('F:/fuxian/Mymodel/output_eval/texturegan_comparison_20261102'))
    p.add_argument('--previous', type=Path, default=Path('F:/fuxian/Mymodel/output_eval/garment_test128_20261031'))
    a = p.parse_args()
    main(a.root, a.previous)
