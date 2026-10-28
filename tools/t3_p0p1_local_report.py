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
    lots=read(root/'lots/followup_summary.json')
    lots_means=table(['补充方法 / 输入','Edge F1↑','Leak↓','Sketch IoU↑','Edge L1↓','LPIPS↓','TCF-LAB'],
        [['LOTS / 文本+草图']+[f"{lots['metric_means'][m]:.6f}" for m in metrics[:-1]]+['N/A']])
    commit=subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[1]),'rev-parse','HEAD'],text=True).strip()
    content=f'''# T3 P0/P1 实验结果与交付状态（2026-10-28，LOTS补充至10-29）

P0复核、E0/E1同协议补推理、BF-Fashion和ControlNet+IP-Adapter完整外部评价已完成。LOTS官方预训练推理已完成同批96身份及五项适用指标，作为文本+草图补充基线。本轮新增训练0，新增独立图像544张；LOTS四例冒烟复用在96中，不重复计数。真人评价只完成材料和分析程序，真实答卷0。

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

文本+草图补充组单列如下。LOTS使用官方预训练权重，未在当前数据集微调；与已微调的BF/CN及含参考纹理的内部方法输入条件不同。全部五项有效n=96。原生512²、Euler30、CFG5、scale0.8、fp16，直接双线性转384×512评价；没有为改善本批结果再调参。

{lots_means}
LOTS的Leak接近1，与图审所见的有色摄影背景一致。它是固定旧评价的衣外背景代理，不能解释成“99.1%纹理泄漏”；原生背景、人物入镜、输入线条对比度及任务差异也会影响结构代理。LPIPS使用原GT作图像距离，但LOTS没有独立参考纹理条件，距离差不能单独证明方法更差。TCF-LAB、DISTS Margin和颜色Margin均不适用，未设为0、未进入参考响应排名。该组不与主表合成同条件优劣结论。{link('LOTS逐身份指标',root/'lots/metrics/per_image.csv')}保留全部96条原始记录。

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

LOTS按用户要求独立放在{link('本地LOTS项目',project.parent/'LOTS')}，服务器为`/share/home/u2515283058/LOTS`，和Mymodel同级。[官方源码固定提交](https://github.com/intelligolabs/lots/tree/49e3955c8eece69ee484f1b52bd27335bd11b019)，[官方权重固定版本](https://huggingface.co/federicogirella/lots/tree/1172dbc1aceefd121e84ea547ef39aa62ad994e1)。此前服务器直连/代理下载失败日志保留；用户授权后，本地从官方固定revision下载24项资产，共9134607130字节，经9个分包上传，服务器校验分包、完整归档和全部24文件SHA后离线推理。分包不含源代码，源代码始终通过git pull部署。{link('资产与版本审计',root/'lots/assets_provenance.json')}、{link('上传校验',root/'lots/asset_transfer_verification.json')}及{link('隔离环境',root/'lots/runtime_environment.json')}可追溯依赖和权重。

固定四例先实际检查，能生成独立挂衣/挂裙，但存在包袋结构失败、颜色和背景偏差。AI按最低可运行任务门槛放行后扩展原96身份，网络未修改；沿用官方global prompt、negative prompt、Euler30、CFG5、scale0.8和身份seed，保存实际latent形状/精度/SHA。四例原图原样复用，其余92张新增，未作质量筛除。正式生成351秒，冒烟64秒，统一评价47秒；这些为各自单A30作业分配时间，不是纯采样时间。完整批次不保证每图都是独立白底服装：实际出现包袋变成上衣、人物入镜、拍摄背景和颜色偏离，所以作为有任务限制的补充基线，不称为严格白底任务完全对齐或论文指标复现成功。{link('冒烟放行规则',root/'lots/code/protocols/smoke_release_20261029.json')}、{link('96原生图与逐图元数据',root/'lots/formal')}、{link('最终状态',root/'lots/candidate_status.json')}均保留。

{link('普通案例（输入哈希4例）',root/'lots/figures/ordinary_hash4.png')}和{link('失败尾部（高Leak/低IoU后选择）',root/'lots/figures/failure_tail.png')}同时展示草图、仅评价用GT、原生方图和转换后评价图；{link('96原生图总览',root/'lots/figures/all96_native_contact.png')}与{link('图源清单',root/'lots/panel_provenance.csv')}保留所有输出。AI实际图审记录见{link('LOTS图审',root/'lots/visual_review.json')}，不是人类评分。普通案例中203183240的包袋生成成上衣、144929078生成成外套；失败尾部171531554出现人物局部、131993724的裙子变成短裤，均未删除。官方预训练数据曝光与其训练seed未知，不宣称完全无污染。

## 6. 真人盲评：材料就绪，真实参与者0

{link('参与者匿名材料ZIP',root/'human_eval/blind_material.zip')}含40题草案及每人8题试评入口、30套随机顺序；每套40题左右各20次，题序独立随机，三个问题顺序固定。案例为2个跨模型困难结构身份+8个输入哈希身份，各与B/T1/T2/C比较，不按T3胜负选题。隐藏模型名，不含GT，不在参与者ZIP中放解盲文件。界面已实际检查，未填写任何评分。

用户需组织5–8名真实参与者试评并提供导出CSV，再依据真实变异冻结正式规模（暂拟20–30人×40题）、刺激和分析规则。正式草案入口不能当作已经批准的正式研究。分析程序保留原始答卷、缺失、相当和无法判断，考虑评价者/案例交叉重复；无答卷时不生成胜率或p值。当前`responses_raw.csv`、`responses_clean.csv`和`human_eval_statistics.csv`仅有表头。协议见{link('真人试评协议',root/'human_eval/human_eval_protocol.md')}；{link('当前人评结果状态',root/'human_eval/human_eval_results.md')}。**尚无真人应用价值验证，P1-B未完成。**

## 7. 资源、复核和复跑入口

本轮{resource['total_jobs']}个已提交作业已结束：{resource['job_states']['COMPLETED']}完成、{resource['job_states']['FAILED']}失败、{resource['job_states']['CANCELLED']}取消；{resource['gpu_jobs']}个GPU作业全部成功，共{resource['actual_gpu_hours']:.6f} GPU小时（A30分配时间，不含排队/历史训练）。BF完整批次分为两张单卡并行；LOTS单卡原生512推理约8.73GiB峰值，92张新增在351秒内完成，未为增加卡数引入新的并行实现。按空闲卡与队列限制调度。初次审计路径错误已修复；LOTS早期下载失败和迁移/重复下载取消均保留。{link('最终资源账本',root/'resource_ledger.csv')}按job_id去重，保留原始快照；{link('原轮次最终sacct截图',root/'audit/final_jobs_and_bundle.png')}及{link('LOTS最终sacct字段',root/'lots/server_formal_job_fields.png')}补全各包最后作业的结束状态，不编造未捕获的字段。

原结果包166491245字节、2746成员，SHA256=`73211dfeb7fde08cd805ce5322aaec2a38e1b3fde9819bb251ea6b6279ea734a`；2745项文件大小与SHA逐项通过。LOTS正式包{lots['bundle']['archive_bytes']}字节、{lots['bundle']['members']}成员，SHA256=`{lots['bundle']['archive_sha256']}`；{lots['bundle']['verified_files']}项文件大小与SHA逐项通过，原包均保持不变。旧P0的480图也已核验；本轮新增544图无GT进入生成，LOTS转换后的评价图是派生文件，不重复计为新的独立生成。{link('原包下载校验',root/'audit/download_integrity.json')}、{link('LOTS下载校验',root/'lots/formal_download_integrity.json')}与{link('执行状态',root/'execution_status.json')}可追溯全部范围。

服务器原数值汇总代码提交`94ba019`，LOTS统一评价提交`{lots['metric_commit'][:7]}`；本地图表/台账/报告程序提交`{commit[:7]}`，分支`codex/t3-p0-p1-20261028`；LOTS独立分支`codex/lots-repro-20261028`，生成提交分别为`a394b32`（复用4图）、`f6408a1`（其余92图），审计/打包提交`{lots['audit_commit'][:7]}`。代码上传始终为本地Git提交/推送、服务器git pull，未采用其他代码上传路线。服务器原包保持原样；本地重排图标签、完善台账、并入LOTS补充结果，并用同一次统一指标计算内部配对统计。

本地可从{link('代码快照',root/'code/tools')}复跑统计与报告；服务器主入口是`scripts/t3_p0p1.sbatch`的`generate/evaluate/review/aggregate/package`动作，LOTS入口是其独立项目的`run_experiment.sbatch`，输入和旧新输出哈希冻结。原始项目中用户未提交代码未被修改。后续剩余工作为真实试评、据此冻结正式人评及收集评分；本轮不新增内部训练。
'''
    target=project/'docs/T3_P0_P1实验结果_20261028.md';target.write_text(content,encoding='utf-8')
    shutil.copy2(__file__,root/'code/tools'/Path(__file__).name)
    (root/'code/local_source_version.json').write_text(json.dumps(dict(commit=commit,branch='codex/t3-p0-p1-20261028',server_numerical_commit='94ba019',report_local_only=str(target)),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('LOCAL_REPORT',target)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True)
    report(parser.parse_args().project)
