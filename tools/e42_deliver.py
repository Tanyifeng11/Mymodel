"""只在本地运行，将下载并验SHA的E42结果写到本地docs。"""
import argparse
from pathlib import Path
from tools.e42_repeat import OUT,read,sha


def display(value):
    if value is None:return '未执行/不可用'
    if isinstance(value,float):return '%.4f'%value
    return str(value)


def deliver(root):
    root=Path(root);out=root/OUT;local=out/'local_review';docs=root/'docs'
    a=read(local/'audit.json');c=read(local/'calibration.json');s=read(local/'paired_statistics.json')
    d=read(out/'download_verification.json');assert d['sha256_verified']
    verification=read(out/'local_audit_verification.json')
    visual=read(local/'visual_review.json')
    docs.mkdir(exist_ok=True)
    meaning='程序测量校准未通过，停止真实RGB评估。' if s['stop_reason']=='measurement_invalid' else (
        '真实资格来源不足，停止生成端主张。' if s['stop_reason']=='insufficient_support' else '按预注册门槛给出真实RGB重复可观测性判定。')
    lines=['# E42：真实面料重复结构可观测性与生成缺陷确认','',
        '**状态：'+s['status']+'；原因：'+(s['stop_reason'] or '见门槛结果')+'。** '+meaning,'',
        '冻结E5，训练0步、新增模型生成0张、GPU0卡时。历史500条配对来自原评估清单，按原BF ID、dataset_index和历史benchmark序号核对；不修改原数据，不新建人工标签。','',
        '## 数据与溯源','',
        '- 历史集合：`eval_outputs/phase1_full_500/e5_tcpm_lite`。',
        '- 原始五路配对、caption和split逐行核验；原始输入及PNG SHA256见归档 `input_hashes.json`。',
        '- E5当前命名权重SHA256：`'+a['E5_sha256']+'`。',
        '- 历史输出清单记录checkpoint路径及代码版本；CFG7/DDIM50由对应历史代码默认值且无benchmark覆盖查证，seed42由清单记录。历史PNG没有逐图运行参数日志和当时权重摘要，不能把当前哈希说成历史逐图绑定证明。',
        '- 审计的历史代码版本：`'+a['historical_code_commit']+'`；本轮代码版本：`'+a['code_commit']+'`。',
        '- 历史G头信息分辨率384×512；后续GT评估将按此分辨率对齐原草图坐标。本轮若在G0停止，则尚未执行该RGB处理。最终原始文件哈希复核：'+str(a['input_hashes_unchanged'])+'。','',
        '## 先锁定、后评估的测量协议','',
        'sketch_only掩码侵蚀9px（分辨率同比缩放为奇数核），96/128/160/192窗口、半窗步长、前景≥95%。大窗优先、同行优先的几何NMS，完全不依据GT/G纹样分数选窗，至少2个互不重叠ROI。已保存各样本的草图掩码和窗口哈希；GT/G阶段将共用该几何并核验窗口哈希，G0停止时尚未执行RGB窗口一致性检查。','',
        'Lab三通道分别去sigma3低频并按RMS归一；常数通道无效。A使用Hann FFT归一自相关及离中心8..W/3峰、峰背景median/MAD标准化和方向谷值排除平移不变脊；B使用固定位置最多12组互不重叠高通NCC块对，以25%分位数判一致，至少3组。B另有无Hann候选提议，不要求候选通过A阈值；A/B共享Lab预处理，不能称统计独立。','',
        '覆盖率是通过窗口数/可测窗口数，不是物理密度。低对比或不足2个可测ROI记NA；若没有检出，结论仅限该窗口的8..W/3周期范围，无法确认更大周期、遮挡或完整纹样身份。GT重复须两处分离ROI方向差≤20°、周期log比≤0.35。','',
        '局部结构相容性为颜色幅度归一的极坐标频谱，扫描旋转和有限径向位移；它不是完整身份准确率。相容性阈值另用已知同模板变换与异模板控制校准。','',
        '## 程序控制与一次性留出校验','',
        '直接复用E40条纹/格纹renderer及E41椭圆renderer，程序贴入仅草图选取的8个有效轮廓。开发period12/20/28、angle0/30；留出period14/18/24、angle15/45，相位另留出；对比16/64/160并包含同亮度色差图。开发和留出各54阳性+54阴性，阴性为纯色、单图徽和随机8px块打乱。它们不构成BF训练数据或真实人工真值。','',
        '阈值只由开发控制选择，锁定后留出仅检验一次；真实GT/G不得反向调阈值。候选阈值27组，兼容阈值65个预定值；留出未用于选优。联合检测TPR≥0.90/FPR≤0.05且两单代理TPR>FPR；相容性检验同样要求TPR≥0.90/FPR≤0.05。','',
        '| 判据 | 留出阳性 | 留出阴性 | 阳性NA | 阴性NA | TPR | FPR |','|---|---:|---:|---:|---:|---:|---:|']
    for m,r in c.get('heldout',{}).items():lines.append('| '+' | '.join([m,str(r['positive_n']),str(r['negative_n']),str(r['positive_NA']),str(r['negative_NA']),display(r['tpr']),display(r['fpr'])])+' |')
    mr=c.get('compatibility_heldout')
    if mr:lines.append('| 局部相容性 | '+' | '.join([display(mr['positive_n']),display(mr['negative_n']),'—','—',display(mr['tpr']),display(mr['fpr'])])+' |')
    lines+=['','检测器阈值：`'+str(c.get('threshold'))+'`；相容性阈值：`'+str(c.get('compatibility_threshold'))+'`。','',
        '检测校准通过：'+str(c.get('detector_pass'))+'；相容性校准通过：'+str(c.get('compatibility_pass'))+'。','']
    for m,r in c.get('heldout',{}).items():
        lines.append(m+'阴性分项：'+str(r['negative_by_type'])+'。')
    lines+=['','FPR以全部已知阴性程序控制为分母；阴性NA单列，不将纯色NA宣称为真实图像可测。阳性NA计入召回损失。']
    lines+=['','### 校准失败的具体含义','',
        '开发集联合检出45/54（83.33%），留出集48/54（88.89%）；54张留出阳性至少需检出49张才能达到90%门槛。所有6张漏检均为条纹：条纹12/18（66.67%）、格纹18/18、椭圆18/18。漏检覆盖对比度16、64、160，不仅发生在低对比条件。','',
        'A单独使用时留出误检8/54（14.81%），B单独使用时误检5/54（9.26%）；联合规则压低误检但未达到预设召回。局部相容性在开发集为17/54（31.48%），留出集21/54（38.89%），不能据此筛出可信的真实A+子集。','',
        '以上说明本轮检测器及相容性代理尚不足以支撑预注册主张，不说明E5成功或失败。未用这些留出结果改阈值，也未追加实验或重跑校准。完整漏检控制ID及按模板统计见本地 `local_audit_verification.json`。']
    lines+=['','## 全量漏斗及NA','', '| 阶段 | 数量 |','|---|---:|']
    for k,v in s['funnel'].items():lines.append('| '+k+' | '+display(v)+' |')
    lines+=['','掩码/ROI原因：`'+str(a['roi_counts'])+'`。','',
        '未执行的参考/GT或E5阶段不填写0；资格CSV或E5 CSV仅有表头时，表示门槛前停止，而不是500条都无重复。','',
        '阶段状态：`'+str(s['stages'])+'`。','',
        '## 配对统计与决策','',
        '预设统计协议为按reference SHA256来源先组内均值、再来源等权，2000次来源簇配对bootstrap；只有进入相应阶段才执行。G1要求A+≥30独立来源；G2要求覆盖差95%CI下界>0.15、可测E5无重复率点估≥0.40、两独立代理方向不反。可测成对来源不足30时不允许GO。','']
    if not s['modes']:
        lines+=['本轮在校准/溯源门槛停止；A/A+、真实GT自一致性、GT/E5覆盖差、无重复率和相容性差均未验证。不计算虚假的配对CI，不生成真实RGB评价图板。','']
    for m,r in s['modes'].items():
        lines+=['### '+m,'', '资格层：`'+str(r['layers'])+'`；A+独立来源='+str(r['Aplus_groups'])+'，占全量='+display(r['Aplus_fraction'])+'。',
            'GT不稳定='+str(r['GT_repeat_unstable'])+'，reference NA='+str(r['na_reference'])+'，GT NA='+str(r['na_GT'])+'，成对E5 NA='+str(r['e5_unmeasurable'])+'。',
            'A统计：`'+str(r['A'])+'`。','A+统计：`'+str(r['Aplus'])+'`。',
            'A+中有重复但局部参考不相容计数='+str(r['has_repeat_but_mismatch'])+'。','']
    lines+=['G0='+str(s['G0'])+'；G1='+str(s['G1'])+'；G2='+str(s['G2'])+'。','',
        '## 复核、资源与交付','',
        '程序控制图板为固定前12个控制的正负对（按程序排序均为条纹，不代表全部模板的视觉覆盖；格纹与椭圆的完整数值结果保留在控制JSON）。真实图板仅在进入E5阶段后按原ID排序固定取24条。视觉复核只作说明，不改阈值、不新增标签。复核记录见本地 `visual_review.json`。','',
        '服务器结果目录：`/share/home/u2515283058/Mymodel/'+str(OUT).replace('\\','/')+'`。作业116813使用CPU4核，申请16GB内存，耗时71秒，COMPLETED、退出码0:0；未记录MaxRSS，不推测峰值内存。Slurm原始记录随归档保存。','',
        '本地核对：500个原始ID、500个reference哈希、2000个输入路径；阈值锁文件和开发控制哈希一致，锁定事件先于留出校验完成。全部保存的NCC块对互不重叠；按锁定阈值复算的留出混淆计数与服务器一致。无真实RGB测量事件，资格及E5占位CSV均无数据行。','',
        '两张控制图板均已复核，共48个缩略图。观察到条纹方向及对比变化、单次X图徽、纯色和打乱块；部分图徽接近服装边缘，不能把代理误检单独归因于某个纹样因素。图板不构成独立人工真值。','',
        '归档SHA256已在本地核验：`'+d['actual_sha256']+'`。数据、逐窗口原分数和控制结果保存在本地 `output_eval/e42_real_fabric_repeat_gate_20261011/local_review`；本报告及Go/No-Go仅写本地docs。','',
        '## 结论边界','',
        '只有G0/G1/G2全过才能声称“协议限定的真实R/GT重复子集上，E5最终RGB局部重复覆盖不足”。任何其他状态均不验证这一论文靶点；即使GO也不能推导E5所有纹样失败、密度是主因或内部唯一因果模块。此次不追加诊断轮次、不开始新方法训练。','']
    (docs/'E42_real_repeat_gate_report.md').write_text('\n'.join(lines),encoding='utf-8')
    short=['# E42 Go/No-Go','', '**'+s['status']+'**','',
        '停止原因：'+(s['stop_reason'] or '无')+'。'+meaning,'',
        '联合检测留出召回48/54=88.89%（门槛90%），误检0/54；其中条纹召回12/18=66.67%。局部相容性留出召回21/54=38.89%。','',
        'G0='+str(s['G0'])+'；G1='+str(s['G1'])+'；G2='+str(s['G2'])+'。','',
        '训练0步、新模型生成0张。未通过门槛时不以重复形成不足作为已验证方法靶点，也不将E41密度探索评分解释为因果结论。','',
        '详情见 [E42_real_repeat_gate_report.md](E42_real_repeat_gate_report.md)。','']
    (docs/'E42_go_no_go.md').write_text('\n'.join(short),encoding='utf-8')
    print(s['status'],s['stop_reason'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);deliver(p.parse_args().root)
