"""本地归档校验、统计复算、256平方图库与中文报告；原包及原始成员不变。"""
import argparse
import csv
from html import escape
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import shutil
import tarfile

import numpy as np
from PIL import Image,ImageDraw
from tools.bf_t1_t2 import RAW,ARMS,NEW,read,write,sha
from tools.bf_t1_t2_report import summarize,PRIMARY,RESPONSE,CONDITIONS


def main(root):
    archive=root/'review_results.tar.gz';info=read(root/'audit/server_bundle_info.json')
    assert archive.stat().st_size==info['bytes'] and sha(archive)==info['sha256']
    server=root/'server_results'
    if not server.exists():
        with tarfile.open(archive) as tar:
            members=tar.getmembers();assert len(members)==info['members']
            for m in members:assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
            tar.extractall(server,filter='data')
    manifest=read(server/'bundle_manifest.json')
    for name,value in manifest.items():
        p=server/name;assert p.stat().st_size==value['bytes'] and sha(p)==value['sha256'],name
    assert len(manifest)+1==info['members']
    analysis=root/'analysis';(analysis/'metrics').mkdir(parents=True,exist_ok=True)
    shutil.copy2(server/'metrics/per_image.json',analysis/'metrics/per_image.json')
    summarize(analysis)
    for name in ['summary.json','paired_statistics.json']:assert read(analysis/'metrics'/name)==read(server/'metrics'/name),name
    rows=read(server/'evaluation.json');records=read(server/'metrics/per_image.json');assert len(records)==1536
    for row in rows:
        for key in ['reference','sketch','gt']:assert sha(server/'inputs'/row[key+'_sha256'])==row[key+'_sha256']
        latents=[];soft=np.load(server/'masks'/(row['id']+'.npy'))
        for arm in ARMS:
            path=server/'outputs'/arm/row['condition']/(row['id']+'.png');meta=read(path.with_suffix('.json'))
            assert sha(path)==meta['output_sha256'] and Image.open(path).size==(384,512)
            assert meta['caption']==row['caption'] and meta['seed']==row['seed'] and meta['cfg']==7 and not meta['GT_used_in_inference']
            for key in ['reference','sketch']:assert meta['input_'+key+'_sha256']==row[key+'_sha256']
            latents.append(meta['initial_latent_sha256'])
            if arm.endswith('_LAB'):
                a=np.asarray(Image.open(path));b=np.asarray(Image.open(server/'outputs'/arm.removesuffix('_LAB')/row['condition']/(row['id']+'.png')))
                assert np.array_equal(a[soft==0],b[soft==0])
                if not meta['lab_valid']:assert np.array_equal(a,b)
            if row['condition']=='neutral_swapped':
                other=read(server/'outputs'/arm/'neutral_original'/(row['id']+'.json'));assert other['initial_latent_sha256']==meta['initial_latent_sha256']
        assert len(set(latents))==1
    comparable=read(server/'audit/final_comparability.json');assert comparable['passed']
    accounts=list(csv.DictReader((root/'audit/final_sacct.psv').open(),delimiter='|'));gpu=0
    for account in accounts:
        g=re.search(r'(?:^|,)gres/gpu=([0-9]+)',account['AllocTRES'])
        gpu+=int(account['ElapsedRaw'])*(int(g[1]) if g else 0)
    assert gpu<=360000
    write(root/'audit/final_resources.json',dict(gpu_seconds=gpu,gpu_hours=gpu/3600,authorized_limit=360000,jobs=accounts))
    figures=root/'figures';figures.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for col,arm in enumerate(NEW):
        logs=list(csv.DictReader((server/'training'/arm/'training_log.csv').open()))
        assert len(logs)==28365 and [int(r['step']) for r in logs]==list(range(1,28366))
        for name in ['base','weighted_aux','total']:
            v=np.array([float(r[name]) for r in logs]);axes[0,col].plot(np.arange(100,len(v)+1),np.convolve(v,np.ones(100)/100,mode='valid'),label=name)
        axes[0,col].set_title(arm+' rank0 final microbatch / moving mean100');axes[0,col].legend();axes[0,col].set_yscale('symlog',linthresh=.01)
        axes[1,col].plot([int(r['step']) for r in logs],[float(r['lr']) for r in logs]);axes[1,col].set_title(arm+' LR')
        for ax in axes[:,col]:ax.grid(alpha=.2);ax.set_xlabel('successful updates')
    fig.tight_layout();fig.savefig(figures/'training_curves.png',dpi=150);plt.close(fig)
    stats=read(analysis/'metrics/paired_statistics.json');summary=read(analysis/'metrics/summary.json')
    fig,axes=plt.subplots(2,3,figsize=(15,7))
    for ax,key in zip(axes.flatten(),PRIMARY):
        for y,other in enumerate(NEW):
            v=next(r for r in stats if r['family']=='raw_primary12' and r['metric']==key and r['comparison']=='BF-SD-T3 minus '+other)
            mean=v['mean'];lo,hi=v['ci95'];ax.errorbar(mean,y,xerr=[[mean-lo],[hi-mean]],fmt='o',capsize=4)
        ax.set_title(key);ax.set_yticks([0,1],['T3 - T1','T3 - T2']);ax.axvline(0,color='gray',linestyle='--');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(figures/'raw_primary_paired_ci.png',dpi=160);plt.close(fig)
    assets=root/'display_256x256';assets.mkdir(exist_ok=True)
    def url(p):return Path(os.path.relpath(p,root)).as_posix()
    table=[]
    for row in rows:
        paths=[server/'inputs'/row[k+'_sha256'] for k in ['reference','sketch','gt']]+[server/'outputs'/a/row['condition']/(row['id']+'.png') for a in ARMS]
        cells=[]
        for col,path in enumerate(paths):
            dest=assets/(row['condition']+'_'+row['id']+'_'+str(col)+'.png');Image.open(path).convert('RGB').resize((256,256),Image.Resampling.LANCZOS).save(dest)
            cells.append('<td><a href="'+url(path)+'"><img width="256" height="256" loading="lazy" src="'+url(dest)+'"></a></td>')
        table.append('<tbody data-condition="'+row['condition']+'" data-id="'+row['id']+'"><tr><th colspan="11">'+escape(row['id']+' · '+row['condition']+' · '+row['category']+' · '+row['caption'])+'</th></tr><tr>'+''.join(cells)+'</tr></tbody>')
    page='<!doctype html><html lang="zh"><meta charset="utf-8"><title>新BF-SD：T1/T2/T3同架构比较</title><style>body{font:15px Arial;margin:20px}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:4px}thead{position:sticky;top:0;background:#e8f0f5}img{width:256px;height:256px;object-fit:fill}tbody th{text-align:left}select,input{font:inherit;padding:7px;margin:8px}</style><h1>新BF-SD：T1/T2/T3同架构比较</h1><p>四个原始CFG7模型为主比较，四个相同Lab后处理配置为补充。完整192条件，展示256×256；原始输出384×512。GT不参与推理，换参考GT只是原图。</p><label>条件 <select id="condition"><option value="">全部</option>'+''.join('<option>'+c+'</option>' for c in CONDITIONS)+'</select></label><input id="identity" placeholder="按编号查找"><span id="count"></span><table><thead><tr>'+''.join('<th>'+escape(x)+'</th>' for x in ['参考面料','草图','GT原图']+ARMS)+'</tr></thead>'+''.join(table)+'</table><script>const c=document.getElementById("condition"),i=document.getElementById("identity"),rs=[...document.querySelectorAll("tbody[data-id]")];function f(){let n=0;for(const r of rs){r.hidden=!!((c.value&&r.dataset.condition!==c.value)||!r.dataset.id.includes(i.value.trim()));if(!r.hidden)n++;}document.getElementById("count").textContent=n+" / 192";}c.addEventListener("change",f);i.addEventListener("input",f);f();</script></html>'
    (root/'index.html').write_text(page,encoding='utf-8')
    fixed=[]
    for condition in ['original','neutral_swapped']:
        for cat in ['top','outwear','pants','dress']:
            fixed.append(sorted([r for r in rows if r['condition']==condition and r['category']==cat],key=lambda r:r['sketch_sha256'])[0])
    for start in [0,4]:
        canvas=Image.new('RGB',(11*256,4*290),'white');draw=ImageDraw.Draw(canvas)
        for j,row in enumerate(fixed[start:start+4]):
            for col,label in enumerate(['Reference','Sketch','GT original only']+ARMS):
                draw.text((col*256+3,j*290+2),label,fill='black')
                p=assets/(row['condition']+'_'+row['id']+'_'+str(col)+'.png');canvas.paste(Image.open(p),(col*256,j*290+18))
            draw.text((3,j*290+274),row['condition']+' '+row['id'],fill='black')
        canvas.save(figures/('fixed_original.png' if start==0 else 'fixed_swapped.png'))
    def mean(a,c,k):return next(r['mean'] for r in summary if r['arm']==a and r['condition']==c and r['metric']==k)
    def fmt(v):return 'NA' if v is None else f'{v:.6f}'
    text=['# 新BF-SD架构T1/T2训练与测试总结','',
        '本轮只新增T1/T2训练；BF-SD与完整T3复用已核验28365步权重。四组主比较均为CFG7且未加Lab，Lab补充组对四个模型使用相同规则。具体优劣以以下全量指标与配对统计为准，不根据训练loss或精选图预判胜负。','',
        '## 训练与标定','',
        '共同初始化：E1＋历史BF融合＋零投影；45,384训练样本，seed42、单节点双GPU、有效batch8，各28365次成功更新。AdamW、LR5e-5、warmup500、cosine、bf16与既有T3一致；可训练参数101,752,192。T1为已有感知/高频监督适配，T2为已有DINO外观/结构监督适配，均非完整原论文复现；监督网络冻结，eta249。TRAIN-CAL512无优化器更新，各项辅助/base的epsilon梯度RMS正值中位数匹配0.1，标定不访问测试结果。该规则不意味着四模型总梯度预算完全一致。','']
    for arm in NEW:text.append('- '+arm+'标定系数：`'+json.dumps(read(server/'calibration'/arm/'complete.json')['coefficients'])+'`。')
    text+=['','共同初始化、两rank最终训练流哈希、逐步学习率、成功更新数与最终权重均通过审计。日志为rank0每次更新的末微批，不当作全局batch均值；各模型总目标不同，不能按loss绝对值给模型排名。','',
        '## 原文本128件主指标','',
        '|模型|Edge F1↑|IoU↑|Edge L1↓|Leak↓|LPIPS↓|TCF-LAB↓|','|---|---:|---:|---:|---:|---:|---:|']
    for a in ARMS:text.append('|'+a+'|'+'|'.join(fmt(mean(a,'original',k)) for k in PRIMARY)+'|')
    text+=['','## 配对差值与参考响应','',
        '身份配对bootstrap与双侧sign-flip各10,000次，seed110142。原始T3−T1/T2的六项主指标为Holm12族；三项换参考DISTS为独立Holm6族；Lab补充组另立族。CI为单项描述性95%区间，显著性使用对应Holm p。','',
        '|比较|条件|指标|n|平均差|95%CI|Holm p|','|---|---|---|---:|---:|---|---:|']
    for r in stats:
        if r['family'] in ['raw_primary12','raw_response6']:
            text.append('|'+r['comparison']+'|'+r['condition']+'|'+r['metric']+f"|{r['n']}|{fmt(r['mean'])}|{r['ci95']}|{fmt(r['p_holm'])}|")
    text+=['','换参考指标按固定内部64×64 patch评价，少于四块保持NA；margin=到旧参考距离−到新参考距离，越大越偏向新参考。RGB DISTS包含颜色，灰度版弱化颜色但仍不能代表完整纹元。旧GT不用于换参考LPIPS质量结论。T1训练直接使用DISTS，因此该指标不是独立于其训练目标的验证，需结合结构、LPIPS与实际审图。','',
        '## 图表、配置与资源','',
        '[完整192条件八配置对比](../output_eval/'+root.name+'/index.html)；[训练曲线](../output_eval/'+root.name+'/figures/training_curves.png)；[主指标配对CI](../output_eval/'+root.name+'/figures/raw_primary_paired_ci.png)。',
        '[完整指标](../output_eval/'+root.name+'/analysis/metrics/summary.csv)；[参数与推理配置](../output_eval/'+root.name+'/server_results/metrics/model_parameters.csv)；[配对统计](../output_eval/'+root.name+'/analysis/metrics/paired_statistics.csv)。','',
        f'本轮全部GPU作业实际用量（含失败、取消、重试）{gpu}秒，即{gpu/3600:.4f} GPU小时；授权硬上限100 GPU小时。真实终态账单见结果目录audit/final_sacct.psv。原始包及manifest已逐成员校验，统计本地复算一致；1536输出的输入/latent和Lab mask外一致性已核验。','',
        '单训练seed、此前已查看测试身份、供体复用及预训练暴露未知均限制外推；本轮不包含独立测试集确认或真人盲评。Lab是输出后处理，颜色分布改善不等于学习到纹样。固定按输入SHA选择的原/换参考面板以及退化案例需在最终交付前实际审看，并据真实指标填写结论。','']
    report=root.parents[1]/'docs/新BF-SD_T1_T2训练测试总结_20261103.md';report.write_text('\n'.join(text),encoding='utf-8')
    write(root/'audit/local_verification.json',dict(passed=True,members_verified=len(manifest),outputs_verified=1536,
        statistics_recomputed=True,input_latent_exact=True,outside_exact=True,gpu_seconds=gpu,report=str(report)))
    print('VERIFIED',len(manifest),'OUTPUTS',1536,'GPU_HOURS',gpu/3600,'REPORT',report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
