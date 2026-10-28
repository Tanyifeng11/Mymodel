"""仅将本轮真实结果写入用户指定的本地docs；不向服务器发送报告。"""
import csv
import json
import shutil
import subprocess
from pathlib import Path


def report(project):
    root=project/'output_eval/t3_p0_p1_20261028';server=root/'server_results'
    read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
    rows=lambda p:list(csv.DictReader(Path(p).open(encoding='utf-8-sig',newline='')))
    link=lambda name,p:'['+name+']('+Path(p).as_posix()+')'
    table=lambda head,data:'| '+' | '.join(head)+' |\n| '+' | '.join(['---']*len(head))+' |\n'+''.join('| '+' | '.join(str(x) for x in row)+' |\n' for row in data)
    values={(r['model'],r['metric']):float(r['mean']) for r in rows(root/'external_comparison_table.csv')}
    arms=['E0','E1','B','T1','T2','C','T3','BF-Fashion','ControlNet+IP-Adapter']
    metrics=['edge_f1','leak','iou','edge_l1','lpips','tcf_lab']
    means=table(['模型','Edge F1↑','Leak↓','Sketch IoU↑','Edge L1↓','LPIPS↓','TCF-LAB↓'],
                [[a]+[f'{values[a,m]:.6f}' for m in metrics] for a in arms])
    stats=rows(root/'tables/internal_uniform_paired_statistics.csv')
    significant=table(['比较','指标','T3−对照','95% CI','Holm p'],[
        ['T3−'+r['baseline'],r['metric'],f"{float(r['delta']):+.6f}",
         f"[{float(r['ci_low']):+.6f}, {float(r['ci_high']):+.6f}]",f"{float(r['p_holm']):.6f}"]
        for r in stats if float(r['p_holm'])<.05])
    contrasts=rows(root/'tables/reference_response_paired.csv')
    response=table(['T3−对照','DISTS Margin差','95% CI','Holm p','配对n'],[
        [r['baseline'],f"{float(r['delta']):+.6f}",f"[{float(r['ci_low']):+.6f}, {float(r['ci_high']):+.6f}]",f"{float(r['p_holm']):.6f}",r['n']] for r in contrasts])
    summary=rows(root/'tables/reference_response_summary.csv')
    lookup={(r['model'],r['metric']):float(r['mean']) for r in summary}
    responses=table(['模型','DISTS Margin↑ (27/32)','颜色Margin↑ (32/32)','IoU↑ (32/32)','Leak↓ (32/32)'],[
        [a]+[f'{lookup[a,m]:.6f}' for m in ['dists_margin','color_margin','iou','leak']]
        for a in ['B','T1','T2','C','T3','BF-Fashion','ControlNet+IP-Adapter']])
    resource=read(root/'audit/resource_summary_local.json')
    commit=subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[1]),'rev-parse','HEAD'],text=True).strip()
    content=f'''# T3 P0/P1 实验结果与交付状态（2026-10-28）

P0复核、E0/E1同协议补推理、BF-Fashion和ControlNet+IP-Adapter完整外部评价已完成。本轮新增训练0，新增图像448张。真人评价只完成材料和分析程序，真实答卷0；LOTS因官方权重访问失败，尚未复现，不能称为冒烟通过或任务不兼容。

服务器本次结果根：`/share/home/u2515283058/Mymodel/output_eval/t3_p0_p1_20261028`。本地保留{link('完整校验后的服务器原包',server)}和{link('可读导出及协议',root)}；本文只存本地docs。

## 1. 模型定义与可比范围

E0与E1均从共同Texture Adapter初始化，分别独立训练5个epoch；E0不启用层级路由，E1启用，E1是B/T1/T2/C/T3的初始化。E1不是从E0续训，两者不是T3单项消融。原始W&B配置已回查：历史训练seed未固定/未记录，历史报告为None；42不能作为其训练seed。28365是其历史global-step终点，本轮未重新审计为28365次成功优化更新。

本次E0/E1各重推理P2b同一96身份，不拼入旧500样本均值。E0/E1历史训练4GPU×batch1×accum2，主要内部后续训练2GPU×1×4；本轮比较不能把阶段差异解释成严格控制训练seed与全部训练条件的路由因果消融。当前两个训练清单均45384行、字节SHA相同，和P2b的身份交集为0；历史不可变清单指纹和预训练曝光仍未知。定义、checkpoint、哈希及原始配置见{link('总模型台账',root/'model_registry.csv')}、{link('E0/E1审计',root/'audit/E0_E1_history.json')}。

B为基础训练，T1为感知/高频监督，T2为DINO特征监督，T3为完整双区域非对称监督；C是作者历史自研三域方案。真正消融为T3-no-body和T3-no-rest，均只读复用已完成的seed42结果。

## 2. 同批96身份的统一评价

下表各项有效分母均96，统一CUDA float32评价、关闭TF32。E0/E1是阶段背景，B/T1/T2/T3是主要内部比较，C是历史补充。BF只输入草图+纹理，其余实测方法输入文本+草图+纹理。BF原生256²、DDIM200；其图像直接双线性转换到384×512评价；其余为384×512、DDIM50。采样预算、原生尺寸、文本条件和架构不同，不能把表解释为同条件因果排名。

{means}
T3的六项均值在B/T1/T2/T3中均更有利，但配对统计不支持“全面显著领先”。身份bootstrap与sign-flip各10000次，95%区间为未校正描述区间；主要3对照×6指标共18项Holm校正，只有下列4项通过0.05。对B/T1的校正后最小p分别为0.099390、0.127387；C六项另成族，均未通过。

{significant}
统计为已查看开发集上的回顾性分析，主要内部只有一个训练seed。{link('统一逐身份配对统计',root/'tables/internal_uniform_paired_statistics.csv')}保留所有24项效果量、CI、原始/校正p和有效分母，另有{link('主要内部效果图',root/'figures/internal_paired_effects.png')}。旧P0统计与均值另存，未覆盖。此次五个内部模型LPIPS逐图重算与旧值最大差7.1734×10⁻⁵，未通过2×10⁻⁶严格回放；其余五项逐图一致。因此本文表与配对统计均用此次统一值，不把旧LPIPS混入。差异见{link('旧新评价审计',root/'audit/legacy_six_metric_replay.csv')}，DISTS三病例回放则在各评价作业中均通过2×10⁻⁶。

BF的Edge F1、Leak、Edge L1、LPIPS均值更有利；T3的IoU和TCF-LAB更有利。这些外部六指标是描述性均值，未据此宣称显著。CN+IPA的背景/分割代理明显较差，定性图也出现有色背景与结构变化。

## 3. 参考替换：已查看的32身份

各适用模型原参考96张、换参考32张。旧新参考在各自架构内部复用显式latent与身份seed；BF的32² float32 latent和SD的64×48 float16 latent不同，相同seed编号不表示跨架构相同噪声。不用旧GT计算换参考LPIPS或FID。

{responses}
以下是逐身份配对差；B/T1/T2三项、C一项、两个外部方法分别作Holm族。不能通过比较两个单模型区间是否重叠代替配对推断。

{response}
T3相对BF的差接近0、CI跨0，既不能证明更优，也不能证明等效；相对当前冻结CN+IPA配置的DISTS响应代理更高。内部B/T1/T2三项校正后均未通过。C结果属于本次开发32身份，不替代原独立确认结论。

DISTS Margin是固定草图内部patch集的响应代理，不是纹样身份、方向或周期准确率；Leak是历史衣外背景代理，不等于完整结构质量。5个身份因不足4个有效patch，DISTS分母为27/32；所有输出保留。审计中70条无效记录=7方法×5身份×旧新2条件，**不是70次生成失败**；9模型全部生成缺失数为0。身份与原因见{link('无效指标记录',root/'audit/metric_failures.json')}。

## 4. 原独立确认与探索消融

原独立确认128身份、DISTS有效107：T3−C=+0.00324244，95%CI[+0.00079938,+0.00593172]，单主比较p=0.01069893；未达到事前0.005最小有意义幅度，旧/新参考Leak非劣保护均未通过。原`continue_new_training=false`保持不变，后续探索消融不能改写这一结论。

探索消融同为单seed42、有效107/128：full−no-body=+0.00662814，CI[+0.00473093,+0.00862243]，原计划五比较Holm p=0.00049995；支持本次body_rgb的响应增量。full−no-rest=−0.00250447，CI[−0.00527906,+0.00005283]，Holm p=0.21087891；不支持rest提高DISTS响应，也不能证明两者等效。full相对no-rest新参考IoU高0.017333、Leak低0.038302，只能作为探索性的结构代理证据。原结果与来源哈希见{link('历史证据归档',server/'historical_evidence')}；{link('消融效果图',server/'figures/ablation_effects.png')}与{link('本次参考响应描述图',server/'figures/reference_response_models.png')}分别保留不同证据范围。

## 5. 图像、外部审计与LOTS

已保存阶段图、B/T1/T2/T3图、C补充图、消融图、参考替换图、普通及失败案例。普通案例按输入哈希选取；失败尾部及T3−BF响应两端是结果后选择，已标明，不能当作无偏总体展示。未只挑T3获胜案例。数据包含包袋等配饰，不能宣称全部是纯服装类别。

{link('阶段面板',root/'figures/QA_stage_comparison.png')}、{link('外部普通面板',root/'figures/QA_external_qualitative_grid.png')}、{link('背景失败面板',root/'figures/QA_external_failure_cases.png')}、{link('换参考面板',root/'figures/QA_reference_swap.png')}均已实际AI查看。T3也有参考响应弱、保留旧纹样和结构变化；BF有弱响应与纹样差异；CN有背景块状伪影。单个案例不能证明方法总体优劣。AI观察记录见{link('图审记录',root/'audit/final_visual_review.json')}，输入/输出SHA见{link('面板来源',root/'provenance/local_panel_manifest.csv')}。

BF已有权重epoch26、global-step49166；CN+IPA复用历史冻结C2（CN/IPA均选checkpoint3000，尺度1.25/0.9），未用本轮P2b再调参。二者当前训练身份与P2b交集为0；历史完整数据指纹、预训练曝光及BF许可仍有未确证项，不宣称完全无污染或可任意分发。完整生成参数、代码与checkpoint版本见{link('外部协议',root/'external_protocol.md')}及{link('外部台账',root/'external_model_registry.csv')}。

LOTS按用户要求独立放在{link('本地LOTS项目',project.parent/'LOTS')}，服务器为`/share/home/u2515283058/LOTS`，和Mymodel同级。[官方源码固定提交](https://github.com/intelligolabs/lots/tree/49e3955c8eece69ee484f1b52bd27335bd11b019)，[官方权重固定版本](https://huggingface.co/federicogirella/lots/tree/1172dbc1aceefd121e84ea547ef39aa62ad994e1)。隔离依赖安装完成，但直接连接、集群已有代理、固定SDK和固定文件地址重试均未取得权重；最终作业121095在5次文件访问尝试后失败。实际推理图像0，未运行四例冒烟，更未扩展正式批次；这属于资产访问阻塞，不说明LOTS质量或任务不兼容。原始代码、下载程序和日志均已保留；{link('LOTS状态',project.parent/'LOTS/output/t3_p0_p1_20261028/candidate_status.json')}。LOTS无独立参考纹理输入，不列入参考Margin排名。

## 6. 真人盲评：材料就绪，真实参与者0

{link('参与者匿名材料ZIP',root/'human_eval/blind_material.zip')}含40题草案及每人8题试评入口、30套随机顺序；每套40题左右各20次，题序独立随机，三个问题顺序固定。案例为2个跨模型困难结构身份+8个输入哈希身份，各与B/T1/T2/C比较，不按T3胜负选题。隐藏模型名，不含GT，不在参与者ZIP中放解盲文件。界面已实际检查，未填写任何评分。

用户需组织5–8名真实参与者试评并提供导出CSV，再依据真实变异冻结正式规模（暂拟20–30人×40题）、刺激和分析规则。正式草案入口不能当作已经批准的正式研究。分析程序保留原始答卷、缺失、相当和无法判断，考虑评价者/案例交叉重复；无答卷时不生成胜率或p值。当前`responses_raw.csv`、`responses_clean.csv`和`human_eval_statistics.csv`仅有表头。协议见{link('真人试评协议',root/'human_eval/human_eval_protocol.md')}；{link('当前人评结果状态',root/'human_eval/human_eval_results.md')}。**尚无真人应用价值验证，P1-B未完成。**

## 7. 资源、复核和复跑入口

本轮{resource['total_jobs']}个已提交作业已结束：20完成、4失败、3取消；12个GPU作业全部成功，共{resource['actual_gpu_hours']:.6f} GPU小时（A30分配时间，不含排队/历史训练）。BF完整批次分为两张单卡并行；按空闲卡与队列限制调度。初次审计路径错误已修复；LOTS下载失败和迁移/重复下载取消均保留。{link('最终资源账本',root/'resource_ledger.csv')}保留原始快照并用{link('最终sacct截图',root/'audit/final_jobs_and_bundle.png')}补全最后两个CPU作业的结束状态，不编造未捕获的时间和节点字段。

结果包166491245字节、2746成员，SHA256=`73211dfeb7fde08cd805ce5322aaec2a38e1b3fde9819bb251ea6b6279ea734a`；2745项文件大小与SHA逐项通过。旧P0的480图也已核验；新增448图无GT进入生成。{link('下载校验',root/'audit/download_integrity.json')}与{link('执行状态',root/'execution_status.json')}可追溯全部范围。

服务器数值汇总代码提交`94ba019`；本地图表/台账/报告程序提交`{commit[:7]}`，分支`codex/t3-p0-p1-20261028`；LOTS独立分支`codex/lots-repro-20261028`提交`201af7d`。代码上传始终为本地Git提交/推送、服务器git pull，未采用其他代码上传路线。服务器原包保持原样；本地仅重排图标签、完善台账，并用同一次统一指标计算配对统计。

本地可从{link('代码快照',root/'code/tools')}复跑统计与报告；服务器主入口是`scripts/t3_p0p1.sbatch`的`generate/evaluate/review/aggregate/package`动作，输入和旧新输出哈希冻结。原始项目中用户未提交代码未被修改。后续剩余工作为真实试评/正式评分，以及LOTS资产可访问后的实际冒烟；本轮不新增内部训练。
'''
    target=project/'docs/T3_P0_P1实验结果_20261028.md';target.write_text(content,encoding='utf-8')
    shutil.copy2(__file__,root/'code/tools'/Path(__file__).name)
    (root/'code/local_source_version.json').write_text(json.dumps(dict(commit=commit,branch='codex/t3-p0-p1-20261028',server_numerical_commit='94ba019',report_local_only=str(target)),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('LOCAL_REPORT',target)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True)
    report(parser.parse_args().project)
