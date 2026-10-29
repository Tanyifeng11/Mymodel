"""从已验证原包生成本地docs报告，报告不上传服务器。"""
import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path


def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def report(project):
    root=project/'output_eval/pattern_fidelity_eval_20261029';server=root/'server_results'
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    link=lambda label,p:'['+label+']('+p.as_posix()+')'
    def table(header,data):
        return '| '+' | '.join(header)+' |\n| '+' | '.join(['---']*len(header))+' |\n'+''.join('| '+' | '.join(str(x) for x in row)+' |\n' for row in data)
    def number(value):return 'N/A' if value in ['',None] else f'{float(value):.6f}'
    summary=read(server/'report/analysis_summary.json');release=summary['release'];integrity=read(root/'audit/final_download_integrity.json')
    assert integrity['passed']
    means={(r['split'],r['model'],r['metric']):r for r in rows(server/'statistics/model_means.csv')}
    stats=rows(server/'statistics/paired_statistics.csv');safety=rows(server/'statistics/structure_safety.csv')
    def mean_cell(split,arm,metric):
        r=means[split,arm,metric];return number(r['mean'])+' ('+r['n']+')'
    def means_table(split,arms):
        return table(['模型','灰度频谱距离↓','灰度DISTS新参考距离↓','灰度DISTS Margin↑','Lab直方图Margin↑','方向直方图距离↓','周期误差↓'],
            [[arm]+[mean_cell(split,arm,k) for k in ['gray_fft_new','dists_gray_new','dists_gray_margin','color_margin','orientation_hist_new','period_value_new']] for arm in arms])
    main=means_table('P2b_swap32',['B','T1','T2','T3','C','BF-Fashion','ControlNet+IP-Adapter'])
    supplement=means_table('viewed128',['C','T3','T3-no-body','T3-no-rest'])
    selected=[r for r in stats if r['split']=='P2b_swap32' and r['baseline'] in ['B','T1','T2'] and r['subset']=='all_valid' and r['metric'] in ['gray_fft_new','dists_gray_margin','color_margin']]
    paired=table(['T3−对照','指标','配对n','均值差','95% CI','Holm p'],[[r['baseline'],r['metric'],r['n'],number(r['delta']),('['+number(r['ci_low'])+', '+number(r['ci_high'])+']') if r['ci_low'] else 'N/A',number(r['p_holm'])] for r in selected])
    coverage=table(['参考新纹样类别','32身份','128身份'],[[k,release['reference_only_counts']['P2b_swap32']['new_reference_categories'][k],release['reference_only_counts']['viewed128']['new_reference_categories'][k]] for k in release['reference_only_counts']['P2b_swap32']['new_reference_categories']])
    primary=table(['数据组','身份','严格内部patch有效','冻结主终点有效'],[[split,r['identities'],r['body_valid'],r['primary_eligible']] for split,r in release['reference_only_counts'].items()])
    grouped={}
    for row in safety:grouped.setdefault((row['split'],row['baseline']),[]).append(row)
    structures=table(['数据组 / T3−对照','通过端点 / 6','全部通过'],[[split+' / '+arm,str(sum(r['passed']=='True' for r in group))+'/6','是' if all(r['passed']=='True' for r in group) else '否'] for (split,arm),group in grouped.items()])
    resource=read(root/'audit/resource_summary_final.json')
    visual=read(root/'audit/visual_review.json')
    commit=subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[1]),'rev-parse','HEAD'],text=True).strip()
    content=f'''# T3 纹样保真度客观验证结果（2026-10-29）

本轮完成现有32身份的7模型、已查看128身份的4模型评价，共736个模型—身份配对、1472条旧/新条件记录。新增生成0、训练0。全部分析属于已查看数据上的探索性证据，不是新的独立确认。

**预设规则纹样主终点在两组中的有效身份均为0，不能检验主假设。** 这是当前64像素patch与冻结稳定周期估计规则下的可测性结果，不说明参考面料本身没有周期，也不是T3失败或成功的检验结果。次要频谱距离与灰度纹理距离按其各自适用范围报告，不能据此声称精确纹样复制。

服务器目录：`/share/home/u2515283058/Mymodel/output_eval/pattern_fidelity_eval_20261029`。本地{link('全部结果',root)}与{link('未改动的服务器原包解压',server)}保留原始结果；本文只写本地docs。

## 1. 输入、类别与有效性

主比较为32身份的B/T1/T2/T3，C为历史补充，BF与CN+IPA为外部补充。128身份上的C/T3与T3-no-body/no-rest另表，不与32身份合并。LOTS无独立参考面料输入，沿用已有生成补充结果，不进入本轮参考纹样排名。

173张去重参考图已逐张以参考输入面板作AI标注，未按生成质量或新模型分数归类。褶皱、接缝、纽扣不当作印花条纹；低清、边界、混合图案保留ambiguous。类别是输入整理，不是人类偏好评分或经过人类复核的真值。

{coverage}
参考原图均为256²；其实际纹样清晰度和重复单元覆盖并不等于像素文件尺寸。{link('完整参考类别与说明',server/'pattern_labels/reference_pattern_labels.csv')}、{link('仅参考图面板',server/'figures/reference_only')}、{link('适用性清单',server/'manifests/metric_validity.csv')}保留全部类别、有效与无效身份。

{primary}
所有方法共用与生成结果无关的sketch-only mask。腐蚀核11，固定网格64²、stride32，要求patch100%位于腐蚀后内部，最多等距选8个，至少4个才计算内部纹样指标。相同身份各模型和旧新条件坐标完全一致，先聚合身份内patch，再做统计。这比历史90%占据率更严格，因此新DISTS不是旧107/128或27/32结果的数值回放，不能混用旧均值或直接沿用0.005门槛。

{link('固定mask与patch清单',server/'manifests/mask_manifest.csv')}、{link('mask/patch实际QA',server/'figures/mask_patch_QA.png')}和{link('已有输出及哈希清单',server/'manifests/existing_outputs.csv')}可追溯输入、checkpoint、训练seed、推理、旧新输出和mask哈希。结构/背景代理仍按历史代码评价完整图像；GT只用于其评价，未用于本轮生成或纹样mask选择。

## 2. 指标定义和验证

颜色采用固定Lab8×8×8直方图JS距离；Margin为同一新条件输出到旧参考距离减新参考距离。DISTS使用冻结官方VGG16及权重，对固定patch集做对称最近邻聚合，同时报告RGB与统一亮度归一化灰度视图。灰度处理为Rec.709亮度减均值、除以max(std,2/255)，乘0.18加0.5并clip至[0,1]，复制三通道；没有按模型调色或锐化。

方向为固定结构张量的24-bin无向线方向直方图JS；单patch角度来自未分箱圆周矩，置信度≥0.35才参与主方向误差。周期采用补零线性自相关，在3—31像素滞后寻找稳定峰，至少半数且不少于4个参考patch可靠，周期IQR/中位数≤0.35。主终点另要求规则参考类别及稳定周期，参考规则未通过者不进入主终点。生成端周期不可靠者仍保留输出并记N/A。

频谱采用固定64²、去均值、2D Hann窗、幅度平方、去除DC半径1、sigma1频谱平滑、归一化后JS距离。规则参考类别可报告频谱形状距离；周期误差仍需稳定峰，二者不互相替代。非规则/模糊类别的频谱指标表中为N/A，原始描述子计算值在逐图JSON保留供审计。灰度频谱与RGB频谱分开，任何指标都未合成总分。全部周期均为统一图像尺度下的表观周期，未作物理面料尺度标定。

合成条纹8/16像素及0°/45°/90°方向正控通过；相同图频谱距离为0，旋转90°和周期翻倍能被区分，常量图不产生虚假零距离。{link('合成正控',server/'audit/synthetic_controls.json')}、{link('参考裁剪/亮度/平移/模糊敏感性',server/'audit/reference_sensitivity.csv')}、{link('mask腐蚀与48/64/96/128尺寸可用性',server/'audit/mask_patch_sensitivity.csv')}、{link('FFT与自相关QA',server/'figures/fft_autocorrelation_QA.png')}均保留。敏感性分析不用于事后改主规则；正控通过也不等于指标已被人类感知验证。

STSIM未取得已核验且许可明确的2013等价实现，按可选指标记N/A，未拿SSIM替代；无可靠纹样分类器/OCR，未制造花卉、文字或logo身份准确率。{link('STSIM状态表',server/'metrics/stsim_metrics.csv')}与{link('局部语义指标状态表',server/'metrics/motif_metrics.csv')}明确原因。

## 3. 描述性结果：均值（有效身份n）

32身份：

{main}
128身份补充：

{supplement}
灰度/感知距离不等于纹样语义准确率。颜色Margin可反映颜色分布响应，DISTS可反映感知纹理距离，FFT是频率形状代理。BF缺文本条件、原生256²且DDIM200；其余内部/CN为384×512、DDIM50，BF直接双线性转换评价。当前外部模型训练历史、模态、架构和预算不同，只作披露条件差异后的补充比较。LOTS未被赋予0分或列入上述参考表。

## 4. 身份配对统计与结构保护

以身份为单位，配对bootstrap与sign-flip各10000次。CI为未校正描述区间；冻结主终点3项Holm但有效n=0，未生成主检验p值。主要内部其余3对照×22端点成66项Holm族，包含不适用端点的预定族规模；C、两个外部模型和两个消融按各自预定端点族分别处理。所有结果是单训练seed与已查看数据上的探索，未将patch当独立样本。

{paired}
全量{link('配对统计',server/'statistics/paired_statistics.csv')}还包括中位数差、有效n和随机种子；{link('类别分层统计',server/'statistics/category_stratified_statistics.csv')}按8类×3对照×22端点统一校正，n<2不作检验。类别小样本不能支持强类别结论。{link('配对效果与CI图',server/'figures/paired_effects.png')}保持各指标自己的量纲。

结构保护沿用历史诊断界限IoU0.010、EdgeF1 0.020、Leak0.005，对旧新条件共6端点给Bonferroni单侧界；这是已查看样本上的补充诊断，不是新的确认门槛成功。

{structures}
完整数值见{link('结构保护',server/'statistics/structure_safety.csv')}。Leak是衣外背景代理，不能解释成完整结构质量或纹理泄漏率。无效记录见{link('逐身份N/A原因',server/'manifests/metric_invalid_records.csv')}，不是生成失败。

## 5. 图像与指标互证

{link('普通输入哈希4例',server/'figures/ordinary_hash4.png')}、{link('外部普通面板',server/'figures/external_/ordinary_hash4.png')}显示固定草图、旧新参考及各方法旧新输出。另按预先写入程序的规则选取频谱改善、颜色响应但灰度纹理失败、结构退化候选，属于结果后选择的诊断面板，不能当作无偏总体展示。若没有满足条件的案例，不制造该类成功或失败。

本轮实际AI图审结论：{visual['summary']}

逐图观察与局限见{link('实际图审记录',root/'audit/visual_review.json')}，面板与文件SHA见{link('图源清单',server/'figures_manifest.csv')}；每个面板都可对应逐身份指标。AI观察不是人类盲评，不能替代应用价值验证。

## 6. 是否建立新独立确认集

当前预设主终点有效n=0，不能从这批数据估计其可靠的配对方差或直接规划一次确认。应先解决参考重复覆盖、patch尺度与测量可靠性；在参考输入和合成控制上开发、冻结可用规则后，才考虑与训练及这些已查看身份隔离的新清单。不能将现有32/96/128重新称为未触碰确认，也不机械再取128。

{link('次要端点精度场景',server/'statistics/confirmation_precision_scenarios.csv')}只按观测配对SD给出CI半宽0.01/0.02/0.05的正态近似场景，并按有效比例调整；这些半宽不是最小有意义差异，也不是已授权启动的新样本量。新指标实际意义阈值仍未设定，不能直接借用旧DISTS0.005。原128身份确认中“未达原最小幅度且Leak保护未通过”的结论保持不变。

因此本轮不启动新训练或自动扩充直到显著；没有可靠方向/周期或纹样身份证据时，论文主张应限于实际得到支持的参考响应、颜色响应或感知纹理相似度，并披露结构权衡。

## 7. 复现、资源和来源

本轮{resource['total_jobs']}个实际提交作业已结束，{resource['gpu_jobs']}个GPU评价作业成功，A30分配时间合计{resource['actual_gpu_hours']:.6f} GPU小时。两次最初超出账户提交上限的请求未获作业ID，属于提交拒绝，不计为已提交失败作业；后按账户限制分批补交，未重复计算完成记录。{link('资源账本',root/'resource_ledger.csv')}与最终sacct截图保留凭据。

正式包{integrity['bytes']}字节，{integrity['members']}成员，SHA256=`{integrity['sha256']}`；{integrity['verified_files']}项文件大小与SHA均通过。准备包SHA256=`0c6630e8efab8fb93e03b95681fbf7b68903b606fe3ada41656264eda22dc76c`，1002项文件也全部通过。原有P0/P1、128确认和消融包未覆盖。

评价冻结提交`{release['commit'][:7]}`、评价文件SHA=`{release['evaluator_sha256']}`；完整版本和环境见{link('run manifest',server/'run_manifest.json')}，本地报告代码提交`{commit[:7]}`。代码部署沿用本地提交/推送、服务器git pull；不采用其他代码上传路径。{link('冻结协议',server/'protocol')}、{link('评价代码快照',server/'code')}和{link('原包下载校验',root/'audit/final_download_integrity.json')}提供复跑入口。

文献用于设计背景，本文固定JS频谱诊断并非复现频域论文的训练损失：[Liang等，CGF2023](https://doi.org/10.1111/cgf.14938)、[STSIM原文](https://users.eecs.northwestern.edu/~pappas/papers/zujovic_tip13.pdf)、[DISTS官方代码](https://github.com/dingkeyan93/DISTS)、[纹理特征评价局限](https://doi.org/10.1109/TPAMI.2020.2964533)。
'''
    target=project/'docs/T3_纹样保真度客观验证结果_20261029.md';target.parent.mkdir(exist_ok=True);target.write_text(content,encoding='utf-8')
    dest=root/'code/tools'/Path(__file__).name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(__file__,dest)
    write_path=root/'audit/local_report_source.json';write_path.write_text(json.dumps(dict(commit=commit,report=str(target),server_report_uploaded=False),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('LOCAL_REPORT',target)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);report(parser.parse_args().project)
