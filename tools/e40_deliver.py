"""仅在本地整理用户要求的三项交付；待生成时明确保留 pending。"""
import argparse
import json
from pathlib import Path
from datetime import datetime


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def number(value):return 'NA' if value is None else '%.4f'%value


def interval(record):
    text=number(record['mean'])
    if record.get('ci95') is not None:text+=' ['+', '.join(number(v) for v in record['ci95'])+']'
    return text+'；组数='+str(record.get('source_groups',0))


def deliver(root):
    root=Path(root);assert root.is_absolute()
    local=root/'output_eval/e40_pattern_identity_20261011/local_review';docs=root/'docs'
    docs.mkdir(exist_ok=True)
    full=(local/'E40_pattern_representation_comparison.json').exists()
    data=read(local/('E40_pattern_representation_comparison.json' if full else 'cpu_reference_analysis.json'))
    protocol=read(local/'protocol.json');classification=data['classification'];reference=data['reference_tests']
    state=read(root/'output_eval/e40_pattern_identity_20261011/local_state.json')
    status='五项分析已运行；完整身份仍需按证据边界判断' if full else '实验尚未完成：已完成 FFT/颜色参考检验，GPU 生成及全部表示对比仍在排队'
    data['local_delivery']=dict(updated=datetime.now().isoformat(timespec='seconds'),status=status,
        complete=full,job_state=state,protocol=protocol)
    (docs/'E40_pattern_representation_comparison.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# E40：纹样身份定义与可测性验证', '', '**状态：'+status+'。**', '',
        '冻结 E5、seed42、CFG7、DDIM50，生成模型训练更新数为 0。新结果服务器目录：`/share/home/u2515283058/Mymodel/output_eval/e40_pattern_identity_20261011`。', '',
        '纹样身份暂定义为（类别、重复单元、周期结构、空间排列、方向关系），颜色、材质与光照属于应被排除的干扰因素。裁剪前后的物理尺度需要统一；裁剪再缩放会改变像素周期。本轮分别检验粗类别、频域结构与增强后的来源检索，尚未建立完整重复单元身份的真值。', '',
        '## 数据与固定协议', '',
        '- 受控数据：四类各 32 个原型；4 档频率×2 个方向×4 个相位。三类有纹图使用相同的双色像素直方图。素色的亮度方差不同，单独报告三类有纹子集。',
        '- 真实参考：条纹 22、格纹 25、花纹 10、素色 32；按 SHA256 去重。标签来自历史单一 AI 视觉复核，来源组仅为暂定图像来源，未核验布料身份或真实材质。全部 89 张联系表已检查；未据模型结果筛选样本。',
        '- 每个来源包含原图、两次换色、两次裁剪及一次噪声/光照代理，共 1302 个参考视图。来源及其增强始终同折；受控集外折留一频率，真实集按来源分四折。',
        '- 分类器固定 balanced logistic regression，C=1；StandardScaler 和最多 32 维 PCA 只在训练折拟合。不搜索测试集超参数。',
        '- 分类 F1 降幅使用分层、配对来源 bootstrap 1000 次。距离均值使用 2000 次组 bootstrap；真实颜色匹配配对按共享端点的连通分量分组，受控配对按频率/方向分组。少于两个组时置信区间记 NA。',
        '- 三种表示：全图 CLIP image encoder；64×64 灰度 patch 的 CLIP 均值/标准差；相同 patch 上的 FFT magnitude、radial/angular spectrum。最多固定采样 5 个 patch。颜色基线为 LAB 均值/标准差与 4³ RGB 直方图，使用 float64 累加防止像素顺序泄漏。',
        '- 新增生成固定 16 个既有载体×4 类×2 配色=128 张，同载体共享类别中性文本和初始噪声。参考使用留出的频率 7、方向 15°；不通过 caption 提示纹样。E39 原有 133 张只读复用。',
        '- 服装评估仅使用 sketch-only 高置信度掩码并内缩 9 像素；patch 至少 95% 位于服装内。无有效区域的 LAB、局部和 FFT 指标记 NA。全图 CLIP 仍有背景、形状和参考到服装的域偏移。', '',
        '## 实验 1–2：类别分类与换色稳定性', '',
        '| 数据 | 表示 | 原图 Accuracy | 原图 Macro F1 | 换色1 F1 | 换色2 F1 | 裁剪1 F1 | 裁剪2 F1 | 有纹三类原图 F1 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for domain,results in classification.items():
        for rep,row in results.items():
            m=row['metrics'];patterned=m['base'].get('patterned_only',{}).get('macro_f1')
            lines.append('| '+' | '.join([domain,rep,number(m['base']['accuracy']),number(m['base']['macro_f1']),
                number(m['color1']['macro_f1']),number(m['color2']['macro_f1']),number(m['crop1']['macro_f1']),
                number(m['crop2']['macro_f1']),number(patterned)])+' |')
    lines+=['', '固定初筛门槛为原图 Macro F1≥0.70、两次换色各自 F1 降幅≤0.05。该门槛用于表示筛选；真实标签误差、来源独立性及分类器域偏移仍需单独处理。素色可由方差识别，因此四类颜色基线高于随机机会水平本身不能证明它识别了有纹类别。', '',
        '| 数据 | 表示 | 换色1 F1 降幅 [95% CI] | 换色2 F1 降幅 [95% CI] |', '|---|---|---|---|']
    for domain,results in classification.items():
        for rep,row in results.items():
            fields=[]
            for variant in ['color1','color2']:
                p=row['paired_f1_drop'][variant];fields.append(number(p['drop'])+(' ['+', '.join(number(x) for x in p['ci95'])+']' if p['ci95'] else ' [NA]'))
            lines.append('| '+' | '.join([domain,rep]+fields)+' |')
    lines+=['', '## 实验 3：同色异纹敏感性与来源检索', '',
        'D_pattern、D_color 均在同一表示内使用余弦距离，不跨表示比较绝对值。受控主比较排除素色，三种有纹图的直方图相同；真实配对仅满足平均 LAB 距离≤5，不能等同于完整配色或材质匹配。真实候选共 34 个有向配对，共享端点不作为独立样本。', '',
        '| 数据 | 表示 | D_pattern | D_color | 差值及组 bootstrap 95% CI |', '|---|---|---|---|---|']
    for domain,results in reference.items():
        for rep,row in results.items():
            s=row['sensitivity'];lines.append('| '+' | '.join([domain,rep,interval(s['D_pattern']),interval(s['D_color']),interval(s['gap'])])+' |')
    lines+=['', '固定门槛为差值置信区间下界>0。来源检索结果及逐对数值见 JSON。检索不包含素色；受控候选为 24 个频率/方向类别族，真实候选为 57 个有纹来源。查询是同一来源的程序增强，不能据此宣称未见布料或跨材质身份泛化。', '',
        '换色实际改变程度（灰度 MAE 以 0–255 为单位；LAB 为均值的欧氏距离）：', '',
        '| 数据 | 灰度 MAE | LAB 均值变化 |','|---|---|---|']
    for domain,row in data['input_color_drift']['summary'].items():
        lines.append('| '+' | '.join([domain,interval(row['gray_mae']),interval(row['lab_delta'])])+' |')
    if full:
        g=data['controlled_generation'];old=data['e39_reuse']
        lines+=['', '## 实验 4–5：E5 参考利用与生成保持', '',
            f"已评估新增 {g['images']} 张，区域掩码有效 {g['valid_masks']}/128，有有效内部 patch {g['valid_patch_images']}/128；复用 E39 {old['reused_images']} 张。均值及区间按 16 个载体分组。", '',
            '| 量 | 参考到生成距离 [95% CI] |', '|---|---|']
        for k,v in g['reference_preservation'].items():lines.append('| '+k+' | '+interval(v)+' |')
        lines+=['', '同载体参考干预引起的输出变化：', '',
            '| 干预 | 输出 LAB 变化 | 输出 CLIP 余弦距离 | 输出 FFT 余弦距离 | 输出 local 余弦距离 |', '|---|---|---|---|---|']
        for intervention,row in g['intervention_responses'].items():
            s=row['summary'];lines.append('| '+' | '.join([intervention]+[interval(s[k]) for k in ['lab','clip','fft','local']])+' |')
            if 'matched_histogram_three_pattern_subset' in row:
                s=row['matched_histogram_three_pattern_subset'];lines.append('| '+' | '.join(['pattern：三类直方图匹配']+[interval(s[k]) for k in ['lab','clip','fft','local']])+' |')
        lines+=['', '冻结参考分类器在生成图上的诊断（域外代理分数，未验证为生成图真值）：', '',
            '| 分类器训练域 | 表示 | 配色 | 生成正确率（按载体） | 留出模板参考正确率 | 参考正确的子集内生成正确率 |', '|---|---|---|---|---|---|']
        for domain,rs in g['classifier'].items():
            for rep,colors in rs.items():
                for color,row in colors.items():
                    lines.append('| '+' | '.join([domain,rep,color,interval(row['correct']),interval(row['reference_correct']),interval(row['reference_valid_subset'])])+' |')
        lines+=['', 'E39 使用原有文本，不能与受控中性文本合并作同分布证据。R− donor 未知标签、四类之外的 denim/graphic 等不强映射；已知四类的 R+/R90 使用该来源的留出折分类器。逐臂 CLIP、LAB、FFT、local 和分类概率见 JSON。', '',
            '**解释限制：**参考裁块与生成服装的尺度、褶皱和域偏移会增大频域/视觉距离；低颜色距离与高纹样距离仅形成候选诊断。必须结合图板与代理有效性，不能单独判定完整 semantic grounding 失败。', '',
            '## 审计与交付', '',
            f"输入文件核验 {data['analysis']['audits']['input_files_unchanged']} 项、参考视图 {data['analysis']['audits']['reference_views_unchanged']} 项保持不变；两 GPU 进程内部 E5 模块哈希运行前后相同，E39 原 PNG 哈希保持不变。E5 检查点 SHA256：`{protocol['E5_sha256']}`。", '',
            '完整数值、原型、生成图、逐项审计及复核图板留在服务器新目录；本地已下载复核包并核对归档 SHA256。三个总结文件仅存本地 docs。']
    else:
        lines+=['', '## 实验 4–5：等待 GPU 资源', '',
            '128 张新增生成尚未完成；CLIP/local 参考提取、E39 重评和生成保持分析仍待运行。依赖统计作业仅在两 GPU 作业成功后开始。这里不提供 E5 新生成结论或完整 Go/Stop 判定。', '',
            'GPU 作业：`116776`、`116777`；服务器连接及队列信息见本地 `local_state.json`。预计开始时间属于调度器预测，资源释放后可提前。']
    lines+=['', '## 证据边界', '',
        '- H1：本轮仅检验程序换色下的类别/描述符稳定性，不能扩展成任意真实染色过程下完整身份不变。',
        '- H2：**真实同纹跨材质未验证**。carrier 仅改变噪声与照明；D 的真实同材质异纹也没有核验材料标签。',
        '- 粗类别分类、周期统计和增强来源检索覆盖身份定义的一部分；空间排列、重复单元语义、物理尺度与真实跨材质同一性仍缺真值。',
        '- 全图 image encoder、灰度局部 CLIP 和 FFT 各有不同混杂。生成分类器未经过独立生成图标签校准，置信概率不能直接视为身份保持概率。', '']
    (docs/'E40_pattern_identity_report.md').write_text('\n'.join(lines),encoding='utf-8')
    if full:
        decision=data['decision'];passing=', '.join(decision['operational_candidates']) or '无'
        go=['# E40 Go / No-go', '', '**完整纹样身份：证据不足，暂不进入完整 Pattern Identity Grounding 方法验证。**', '',
            '通过受控与真实粗标签的分类、换色和距离初筛的表示：'+passing+'。这只支持相应表示的局部操作性结论。', '',
            '真实同纹跨材质（H2）、同材质异纹（D）、完整重复单元身份与生成图分类器有效性尚未核验。当前状态为 `'+decision['status']+'`。', '',
            '下一步先补核验来源/材质及同纹异材质配对，并给生成图建立独立的纹样标签和尺度校准；确认可测性之后，再决定是否进入方法训练。', '',
            '当前判断属于证据不足；已通过与未通过的门槛、置信区间、有效组数均见数值 JSON 与报告。']
    else:
        go=['# E40 Go / No-go', '', '**待定：GPU 生成及全表示分析未完成。**', '',
            '当前仅有 FFT/颜色参考检验。完整身份的真实跨材质验证仍缺数据；生成模型保持性尚未评估。保留作业排队，完成后更新本文件。']
    (docs/'E40_go_no_go.md').write_text('\n'.join(go)+'\n',encoding='utf-8')
    print(status)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    deliver(parser.parse_args().root)
