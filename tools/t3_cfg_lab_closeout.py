"""颜色诊断的离线分层、退化案例与中文交付；不修改归档成员。"""
import csv
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw
from tools.t3_cfg_lab import ARMS, CONDITIONS, read, write, sha
from tools.bf_t3_three_arm_report import paired

ROOT=Path('F:/fuxian/Mymodel/output_eval/t3_cfg5_lab_20261103')
SERVER=ROOT/'server_results'


def main():
    rows=read(SERVER/'evaluation.json');records=read(SERVER/'metrics/per_image.json')
    summary=read(ROOT/'analysis/metrics/summary.json');stats=read(ROOT/'analysis/metrics/paired_statistics.json')
    index={(r['arm'],r['condition'],r['id']):r for r in records}
    def ids(c):return sorted(r['id'] for r in rows if r['condition']==c)
    def mean(a,c,k):return next(r for r in summary if r['arm']==a and r['condition']==c and r['metric']==k)['mean']
    def result(a,b,c,k):return next(r for r in stats if r['comparison']==a+' minus '+b and r['condition']==c and r['metric']==k)
    def f(v):return 'NA' if v is None else f'{v:.6f}'
    def ci(v):return 'NA' if v['ci95'] is None else f"{f(v['mean'])} [{f(v['ci95'][0])}, {f(v['ci95'][1])}]"
    accounts=list(csv.DictReader((ROOT/'audit/final_sacct.psv').open(),delimiter='|'))
    gpu=0;cpu_seconds=0
    for row in accounts:
        assert row['State']=='COMPLETED' and row['ExitCode']=='0:0'
        count=re.search(r'(?:^|,)gres/gpu=([0-9]+)',row['AllocTRES'])
        row['gpu_seconds']=int(row['ElapsedRaw'])*(int(count[1]) if count else 0)
        gpu+=row['gpu_seconds'];cpu_seconds+=int(row['AllocCPUS'])*int(row['ElapsedRaw'])
    assert gpu<=3600
    write(ROOT/'audit/final_resources.json',dict(gpu_seconds=gpu,gpu_hours=gpu/3600,
        allocated_cpu_seconds=cpu_seconds,authorized_gpu_seconds=3600,jobs=accounts))
    # 校验所有冻结输入、旧基线血缘、原/换参考latent、无效校准exact OFF。
    parent=ROOT.parent/'bf_t3_ablation_20261102/server_results'
    for row in rows:
        for key in ['sketch','reference','gt']:assert sha(SERVER/'inputs'/row[key+'_sha256'])==row[key+'_sha256']
        base=SERVER/'outputs/CFG7'/row['condition']/(row['id']+'.png')
        assert sha(base)==sha(parent/'outputs/BF-SD-T3'/row['condition']/(row['id']+'.png'))
        for arm in ARMS:
            meta=read((SERVER/'outputs'/arm/row['condition']/(row['id']+'.json')))
            assert all(meta['input_'+k+'_sha256']==row[k+'_sha256'] for k in ['sketch','reference'])
            if row['condition']=='neutral_swapped':
                old=read(SERVER/'outputs'/arm/'neutral_original'/(row['id']+'.json'))
                assert old['initial_latent_sha256']==meta['initial_latent_sha256'] and old['caption']==meta['caption']
            if arm.endswith('_LAB') and not meta['lab_valid']:
                assert np.array_equal(np.asarray(Image.open(SERVER/'outputs'/arm/row['condition']/(row['id']+'.png'))),
                    np.asarray(Image.open(SERVER/'outputs'/arm.replace('_LAB','')/row['condition']/(row['id']+'.png'))))
    stratified=[];direct=[]
    for condition in CONDITIONS:
        for valid in [True,False]:
            selected=[i for i in ids(condition) if index['CFG7',condition,i]['mask_valid']==valid]
            for a,b in [('CFG7_LAB','CFG7'),('CFG5_LAB','CFG5')]:
                for metric in ['ref_rgbuv_h','ref_ab_w1','edge_f1','iou','leak','lpips','gray_dists']:
                    values=[index[a,condition,i][metric]-index[b,condition,i][metric] for i in selected
                        if index[a,condition,i][metric] is not None and index[b,condition,i][metric] is not None]
                    stratified.append(dict(condition=condition,mask_valid=valid,planned_n=len(selected),
                        comparison=a+' minus '+b,metric=metric,**paired(values)))
        for metric in ['ref_rgbuv_h','ref_ab_w1','edge_f1','iou','leak','lpips']:
            values=[index['CFG5_LAB',condition,i][metric]-index['CFG7_LAB',condition,i][metric] for i in ids(condition)
                if index['CFG5_LAB',condition,i][metric] is not None and index['CFG7_LAB',condition,i][metric] is not None]
            direct.append(dict(condition=condition,comparison='CFG5_LAB minus CFG7_LAB',metric=metric,
                prespecified=False,**paired(values)))
    write(ROOT/'analysis/mask_stratified_statistics.json',stratified)
    write(ROOT/'analysis/exploratory_direct_lab_comparison.json',direct)
    failures=[]
    for condition in ['original','neutral_swapped']:
        chosen=min(ids(condition),key=lambda i:index['CFG5',condition,i]['edge_f1']-index['CFG7',condition,i]['edge_f1'])
        failures.append(dict(id=chosen,condition=condition,reason='worst CFG5 minus CFG7 edge F1; post-hoc failure audit'))
    chosen=max([i for i in ids('original') if index['CFG7','original',i]['ref_rgbuv_h'] is not None],
        key=lambda i:index['CFG7_LAB','original',i]['ref_rgbuv_h']-index['CFG7','original',i]['ref_rgbuv_h'])
    failures.append(dict(id=chosen,condition='original',reason='worst CFG7_LAB minus CFG7 RGBuv; post-hoc failure audit'))
    chosen=min(ids('neutral_original'),key=lambda i:index['CFG5_LAB','neutral_original',i]['edge_f1']-index['CFG5','neutral_original',i]['edge_f1'])
    failures.append(dict(id=chosen,condition='neutral_original',reason='worst CFG5_LAB minus CFG5 edge F1; post-hoc failure audit'))
    write(ROOT/'analysis/failure_cases.json',failures)
    panel=Image.new('RGB',(1792,4*300),'white');draw=ImageDraw.Draw(panel)
    for j,selected in enumerate(failures):
        row=next(r for r in rows if r['condition']==selected['condition'] and r['id']==selected['id'])
        paths=[SERVER/'inputs'/row[k+'_sha256'] for k in ['reference','sketch','gt']]
        paths += [SERVER/'outputs'/arm/row['condition']/(row['id']+'.png') for arm in ARMS]
        for col,(label,path) in enumerate(zip(['Reference','Sketch','GT original only']+ARMS,paths)):
            draw.text((col*256+2,j*300+2),label,fill='black')
            panel.paste(Image.open(path).convert('RGB').resize((256,256),Image.Resampling.LANCZOS),(col*256,j*300+20))
        draw.text((3,j*300+278),row['condition']+' '+row['id']+' | '+selected['reason'],fill='black')
    panel.save(ROOT/'figures/failure_audit.png')
    changes={}
    for c in CONDITIONS:
        part=[index['CFG7',c,i] for i in ids(c)]
        changes[c]=dict(identities=len(part),valid=sum(r['mask_valid'] for r in part),off=sum(not r['mask_valid'] for r in part),
            color_n=sum(r['ref_ab_w1'] is not None for r in part),gray_n=sum(r['gray_dists'] is not None for r in part))
    write(ROOT/'audit/supplemental_verification.json',dict(passed=True,baseline_192_equal_parent=True,
        all_input_hashes_verified=True,neutral_latent_caption_exact=True,invalid_lab_exact_off=True,
        counts=changes,resource_seconds=gpu,post_hoc_failure_selection=failures))
    text=['# 新T3：CFG与Lab颜色诊断（2026-11-03）','',
        '本轮建议保留CFG7，按草图有效性门槛增加可选Lab校准。CFG5＋Lab改善了参考颜色距离，但相对原CFG7的结构与衣外泄漏明显退化，不建议直接替换默认推理配置。CFG7＋Lab在三类条件中均通过本轮预设颜色规则和结构护栏；这是当前已查看样本上的探索性诊断，不是独立确认。','',
        '## 冻结配置与核验','',
        '- 完整新BF-SD-T3最终28365步权重不变；没有训练。checkpoint SHA256：`'+read(SERVER/'protocol.json')['checkpoint_sha256']+'`。',
        '- 128件纯服装原文本，另32件中性文本各做原参考/换参考，共192条件×4组=768图。复用192张CFG7，新生成192张CFG5，CPU校准384张；3个预固定CFG7重放样例逐像素一致。',
        '- 四组同输入、同逐身份seed与初始latent；DDIM50，384×512，sketch_scale=0.6，ipa_scale=0，texture tokens关闭。GT不参与生成、mask构建或Lab干预。',
        '- Lab沿用M12代码：只匹配a/b均值和标准差，参考为面料原图全像素，gain裁剪[0.5,2]，强度1；草图mask侵蚀9/羽化8，低置信时完全关闭。L在转RGB前保持，色域裁剪后不保证最终亮度完全不变。',
        '- 192张基线与父消融归档SHA相同；输入SHA、四组及原/换参考latent一致、所有Lab mask外像素不变、invalid整图exact OFF均通过。',
        '- 下载包141,111,010字节，SHA256 `9124804feb5a26ad55ed51c8ec28ffec872ef4565929cb20ac3d2b76fed061d0`；2205个manifest成员逐一校验，通过。服务器汇总与所有配对统计本地复算一致。','',
        '## 原文本128件结果','',
        '颜色评价内部区域不足50像素的3件为NA：88977898、169391821、124955772；两主颜色距离有效n=125，其余下表指标n=128。所有NA保留，没有补样。','',
        '|组别|RGBuv H↓|Lab ab W1↓|Edge F1↑|IoU↑|Leak↓|LPIPS↓|TCF-LAB↓|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for a in ARMS:text.append('|'+a+'|'+'|'.join(f(mean(a,'original',k)) for k in ['ref_rgbuv_h','ref_ab_w1','edge_f1','iou','leak','lpips','tcf_lab'])+'|')
    rgb_gain=100*(1-mean('CFG7_LAB','original','ref_rgbuv_h')/mean('CFG7','original','ref_rgbuv_h'))
    ab_gain=100*(1-mean('CFG7_LAB','original','ref_ab_w1')/mean('CFG7','original','ref_ab_w1'))
    text += ['',f'CFG7＋Lab两项参考色距分别相对下降{rgb_gain:.1f}%和{ab_gain:.1f}%。CFG5本身没有呈现颜色优势，IoU从0.762542降为0.713353，Leak从0.129270升为0.226426。校准只处理衣内颜色，不能消除CFG5生成阶段已产生的背景污染。','',
        '主要配对差值（后者为基线；括号内是身份配对描述性95% bootstrap CI）：','',
        '|对比|ΔRGBuv H|Δab W1|ΔIoU|ΔLeak|ΔLPIPS|','|---|---:|---:|---:|---:|---:|']
    for a,b in read(SERVER/'protocol.json')['statistics']['comparisons']:
        text.append('|'+a+' − '+b+'|'+'|'.join(ci(result(a,b,'original',k)) for k in ['ref_rgbuv_h','ref_ab_w1','iou','leak','lpips'])+'|')
    text += ['',
        'bootstrap与sign-flip各10,000次，固定seed110142。每个condition分别将四对比×两主色距作为Holm8族；两种Lab相对各自未校准组的原文本主色距p_Holm均为0.000800。结构、感知与纹理的其他p值为探索性描述，不与Holm8主色族混用。','',
        '预设实用规则：两主色距的ΔCI上界均<0；ΔF1 CI下界≥−0.02、ΔIoU CI下界≥−0.01、ΔLeak CI上界≤0.005，且无单身份F1下降超过0.15。完整判定见[decision.json](../output_eval/t3_cfg5_lab_20261103/analysis/decision.json)。CFG7＋Lab相对CFG7在三条件均通过。CFG5＋Lab相对CFG5在中性原参考出现单样本F1下降0.193339，未通过该条件护栏；相对原CFG7则三条件均未通过。','',
        '## 中性文本与换参考','',
        '|条件|组别|RGBuv H↓|ab W1↓|IoU↑|Leak↓|新参考灰度DISTS↓|灰度margin↑|','|---|---|---:|---:|---:|---:|---:|---:|']
    for c in CONDITIONS[1:]:
        for a in ARMS:text.append('|'+c+'|'+a+'|'+'|'.join(f(mean(a,c,k)) for k in ['ref_rgbuv_h','ref_ab_w1','iou','leak','gray_dists','gray_margin'])+'|')
    text += ['',
        '中性颜色与结构每组n=32；固定64×64内部patch不足4块的5件保持共同NA，因此DISTS每组n=27。换参考使用同类固定供体，原/新参考具有相同中性文本与latent；没有用旧GT的LPIPS判断换参考质量。',
        'CFG7＋Lab在换参考上的灰度margin均值从−0.002525变为0.001667，ΔCI为'+ci(result('CFG7_LAB','CFG7','neutral_swapped','gray_margin'))+'，sign-flip p=0.076392；灰度DISTS变化为'+ci(result('CFG7_LAB','CFG7','neutral_swapped','gray_dists'))+'，p=0.072393。bootstrap CI与置换p给出的证据不完全一致，这些又是次要探索指标，不能据此声称已经解决纹样迁移。','',
        '## 校准有效与关闭分层','',
        '|条件|有效身份|关闭身份|色距有效n|CFG7＋Lab有效层ΔRGBuv H|有效层Δab W1|','|---|---:|---:|---:|---:|---:|']
    for c in CONDITIONS:
        st=changes[c]
        def layer(k):return next(s['mean'] for s in stratified if s['condition']==c and s['mask_valid'] and s['comparison']=='CFG7_LAB minus CFG7' and s['metric']==k)
        text.append('|'+c+f"|{st['valid']}|{st['off']}|{st['color_n']}|{f(layer('ref_rgbuv_h'))}|{f(layer('ref_ab_w1'))}|")
    text += ['',
        '关闭层两种Lab的所有输出像素和指标均与各自来源相同；原文本17个OFF中14个有颜色指标，另外3个为NA。Lab有效性门槛与颜色评价mask的有效性并非同一概念。分层配对统计保存在[mask_stratified_statistics.json](../output_eval/t3_cfg5_lab_20261103/analysis/mask_stratified_statistics.json)。','',
        '亮度、色相、色度分别报告，色相在近灰色参考或输出上会NA，不能填0或把有符号色度差越小当作越好。原文本CFG7→CFG7＋Lab：L W1为11.404349→11.323081（各n=125），色度绝对对数比为0.371722→0.135303（各n=89）；色相H均值为0.561831（n=83）→0.288194（n=85），有效集合不同，不能把两均值差当作配对效应。完整配对结果保存在paired_statistics.json。完整数值见[summary.csv](../output_eval/t3_cfg5_lab_20261103/analysis/metrics/summary.csv)和[逐图指标](../output_eval/t3_cfg5_lab_20261103/analysis/metrics/per_image.json)。','',
        '## 图表与实际审图','',
        '- [192行完整对比表](../output_eval/t3_cfg5_lab_20261103/index.html)：面料原图、草图、GT与四组输出；展示均拉伸为256×256，原始PNG保持384×512。',
        '- [颜色配对CI](../output_eval/t3_cfg5_lab_20261103/figures/color_paired_ci.png)、[结构/感知配对CI](../output_eval/t3_cfg5_lab_20261103/figures/guardrails_paired_ci.png)。',
        '- [固定原参考面板](../output_eval/t3_cfg5_lab_20261103/figures/fixed_original.png)、[固定换参考面板](../output_eval/t3_cfg5_lab_20261103/figures/fixed_swapped.png)：每类按输入SHA选首件，不按效果挑样。',
        '- [退化案例审计](../output_eval/t3_cfg5_lab_20261103/figures/failure_audit.png)：事后按最差指标确定，专门展示失败，不能作为总体胜率。','',
        '已实际查看两幅配对CI图、固定原/换参考面板及退化审计面板。固定面板中202622944的换参考由偏青色向米白色移动，背景仍保持来源状态；粉色针织外套147017156的颜色更接近参考，但针织细节并非原纹样的严格复制。复杂花纹连衣裙204811634仍保留了生成器自行组织的图案，Lab只改色度，不能补出参考纹元。','',
        '失败面板保留四件：139506765的CFG5白裤引入密集竖纹，Lab因低置信草图关闭；199070462的CFG5换参考上衣增加点状高频细节，F1下降，Lab未消除这些细节；69236132的近黑色背心经Lab后偏棕，ab W1从2.474598降至0.328173，但RGBuv H从0.673211升至0.843415，说明两个色距会发生单例冲突；52898126的橙色外套校准后更接近参考色，但CFG5分支F1从0.447637降至0.254298。最后一例外轮廓并未显著移动，边缘检测会受校准后的灰度/纹理对比影响，不能把F1下降直接解释为服装几何形变。','',
        '事后补充的CFG5＋Lab与CFG7＋Lab直接比较：原文本ΔRGBuv H=0.000974，CI[−0.009794,0.011506]；Δab W1=−0.113205，CI[−0.322412,0.061528]，未见明确额外颜色收益；ΔIoU=−0.049132，CI[−0.077232,−0.024399]。该对比不在冻结四对比中，单独标注探索性，不纳入主Holm8族或代替预设规则。','',
        '## 资源与使用建议','',
        '|作业|阶段|终态|分配秒|GPU秒|','|---|---|---|---:|---:|']
    for r in accounts:text.append(f"|{r['JobID']}|{r['JobName']}|{r['State']}|{r['ElapsedRaw']}|{r['gpu_seconds']}|")
    text += ['',f'实际新增{gpu} GPU秒 = {gpu/60:.1f} GPU分钟 = {gpu/3600:.3f} GPU小时，授权上限3600秒。没有失败、取消或重试GPU作业；CPU后处理229秒。旧消融训练用量另计。','',
        '当前可将“CFG7＋有效草图门槛下的Lab”作为可选颜色后处理，原始T3与后处理版本同时保留并分开标注。不要把CFG5＋Lab设为默认，也不要在对比模型的主表中无标记地替换原始T3；若用于模型公平比较，应单列推理后处理，并给其他方法相同后处理机会。','',
        '结论局限：只有一个训练checkpoint、每身份固定一次随机生成，测试身份此前已查看；中性供体复用，64×64 patch不能代表完整纹元，未做独立样本确认和真人盲评。颜色指标本身与矩校准目标相关，其下降代表颜色分布更近，不等于学会纹样或提升整体审美。T1/T2训练测试请求仍保留，本轮未启动。','']
    report=ROOT.parents[1]/'docs/新T3_CFG与Lab颜色诊断总结_20261103.md'
    report.write_text('\n'.join(text),encoding='utf-8')
    print('REPORT',report,'GPU_SECONDS',gpu,'COUNTS',changes,'FAILURES',failures)


if __name__=='__main__':main()
