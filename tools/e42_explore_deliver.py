"""本地交付E42跳过G0的探索性结果，保留原G0停止报告。"""
import argparse
from pathlib import Path
import numpy as np
from tools.e42_repeat import read,write
from tools.e42_explore import OUT


def fmt(v):
    return 'NA' if v is None else ('%.4f'%v if isinstance(v,float) else str(v))


def interval(m,key):
    ci=m[key+'_ci95'];return fmt(m[key+'_mean' if key=='delta' else key])+(' ['+', '.join(map(fmt,ci))+']' if ci else '')


def deliver(root):
    root=Path(root);out=root/OUT;local=out/'local_review';docs=root/'docs'
    verify=read(out/'download_verification.json');assert verify['sha256_verified']
    audit=read(local/'audit.json');stats=read(local/'paired_statistics.json');protocol=read(local/'override_protocol.json')
    visual=read(local/'visual_review.json');assert visual['complete']
    records=read(local/'e5_records.json');eligibility=read(local/'eligibility_records.json')
    quality=[];summary={}
    for mode in ['joint','autocorr_only','self_similarity_only']:
        summary[mode]={}
        for layer in ['A','A+']:
            selected=[dict(group_id=r['group_id'],**r['modes'][mode]) for r in records if mode in r['modes'] and
                (layer=='A' or r['modes'][mode]['layer']=='A+') and r['modes'][mode]['delta'] is not None]
            groups={}
            for r in selected:groups.setdefault(r['group_id'],[]).append(r)
            means={k:float(np.mean([np.mean([r[k] for r in v]) for v in groups.values()])) if groups else None for k in ['C_GT','C_E5']}
            summary[mode][layer]=dict(**means,has_repeat_but_mismatch=sum(r['has_repeat_but_mismatch'] for r in selected),
                rows=len(selected),groups=len(groups),ref_compatible_e5=sum(r['ref_compatible_E5']['compatible'] for r in selected))
    for r in records:
        q=r['quality_diagnostic'];quality.append(dict(sample_id=r['sample_id'],sketch_IoU=q.get('sketch_iou'),EdgeF1=q.get('edge_f1')))
    qmeans={k:float(np.mean([r[k] for r in quality if r[k] is not None])) if any(r[k] is not None for r in quality) else None for k in ['sketch_IoU','EdgeF1']}
    delivery=dict(**stats,coverage_points=summary,quality_diagnostic=qmeans,quality_rows=len(quality),
        frozen_calibration_sha256=protocol['calibration_sha256'],download_sha256=verify['actual_sha256'],
        visual_review=visual,analysis_jobs=audit['jobs_seen'],analysis_code_commit=audit['code_commit'],
        evidence='Exploratory unvalidated image proxies; no preregistered GO; independent-source bootstrap cannot remove detector bias.')
    write(docs/'E42_exploratory_statistics.json',delivery)
    lines=['# E42：跳过G0后的探索性真实RGB实验','',
        '**状态：EXPLORATORY_UNVALIDATED_MEASUREMENT。** 按用户“跳过G0测量，继续进行后续实验”执行。原G0失败结论保留；本轮完整运行参考/GT资格分层、已有E5 RGB、成对统计和固定图板，但所有纹样指标仍为未验证代理。','',
        '本轮主要结果：联合规则A/A+均为0，无法估计主分析的GT−E5覆盖差；自相关单代理A为23条、A+为6条。只有单代理小样本的探索性数值，尚不足以判断真实重复结构形成缺陷，不能将主分析空集解释为E5没有缺陷。','',
        '冻结E5，训练0步、新增模型生成0张、GPU0卡时。原500条配对、原始数据及上一轮输出只读；结果放在原E42目录内的独立子目录 `exploratory_override_20261011`。','',
        '## 已锁定规则与本轮变化','',
        '沿用原检测阈值 z≥3、相关峰≥0.15、NCC≥0.60，相容性阈值0.64，不重新校准，不把calibration_pass改成True。原留出联合召回48/54=88.89%，条纹召回12/18=66.67%，相容性召回21/54=38.89%。','',
        'GT/E5继续使用此前冻结的sketch_only掩码和几何窗口。参考patch全部256×256，去除四边5%后，大窗贪心只能保留一窗；本轮预先增加纯几何回退：不足两窗时，从固定尺度中选择能放下两个互不重叠窗口的最大单一尺度。本数据回退到96px，可放4个互不重叠窗口；不看纹样分数决定窗口，不缩放参考，不改变检测阈值。该变化在读取真实RGB前记录在override_protocol中。','',
        '先完成参考/GT，再锁定资格文件，之后读取E5；图板从全体原始ID固定排序取前24条。A为参考和GT均被代理检出重复、GT两ROI方向/周期自一致；A+还需通过未校准的局部相容性代理。G1不足30组时仍完成描述性统计，但明确记录支持不足，不补样本。','',
        '## 全量漏斗','', '| 阶段 | 数量 | 占500比例 |','|---|---:|---:|']
    for k,v in stats['funnel'].items():lines.append('| '+k+' | '+fmt(v)+' | '+(fmt(v/500) if v is not None else 'NA')+' |')
    lines+=['','以上漏斗采用联合规则。草图原因：`'+str(audit['roi_counts'])+'`。不可测不按无重复0处理。GT/E5仅使用双方可测的同位置窗口；掉样会影响可解释范围。','',
        '## 三种代理的资格与成对结果','',
        '以reference SHA256来源先组内均值、再来源等权，2000次配对来源bootstrap，固定seed42042。置信区间描述这些代理的抽样波动，不能覆盖测量偏差；小于30组的结果仅作描述。A包括A+。','',
        '| 代理 | A | A+ | A+独立来源 | reference NA | GT NA | GT不稳定 | E5成对NA |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for mode,m in stats['modes'].items():lines.append('| '+' | '.join(map(str,[mode,m['A_n'],m['Aplus_n'],m['Aplus_groups'],m['na_reference'],m['na_GT'],m['GT_repeat_unstable'],m['e5_unmeasurable']]))+' |')
    lines+=['','reference/GT的NA计数在380条有效草图ROI内统计，可能互相重叠，不能直接与漏斗相加。分层表中的C是未进入参考重复层的剩余标签，也包含reference不可测；不代表这些真实图像均无重复。E5只测量三代理A/A+的并集23条，不将其余477条算作模型失败。','', '| 代理/子集 | 可测样本/来源 | C_GT | C_E5 | GT−E5 [95%CI] | E5无重复率 [95%CI] |','|---|---:|---:|---:|---|---|']
    for mode,m in stats['modes'].items():
        for layer,key in [('A','A'),('A+','Aplus')]:
            b=m[key];q=summary[mode][layer]
            lines.append('| '+' | '.join([mode+'/'+layer,str(b['n'])+'/'+str(b['groups']),fmt(q['C_GT']),fmt(q['C_E5']),interval(b,'delta'),interval(b,'no_repeat')])+' |')
    lines+=['','C为可观测重复窗口比例，不是物理密度；E5无重复要求可测且C_E5=0。图中有少量候选窗口不等于满足两处分离ROI的高置信重复。','',
        '## 形成与相容性分开报告','',
        '| 代理/A+ | 成对有效数 | E5局部参考相容数 | 有重复但参考不相容数 |','|---|---:|---:|---:|']
    for mode in summary:
        q=summary[mode]['A+'];lines.append('| '+' | '.join(map(str,[mode,q['rows'],q['ref_compatible_e5'],q['has_repeat_but_mismatch']]))+' |')
    lines+=['','相容性不是完整motif身份真值；其校准召回较低，A+选择及上述分项可能受较强测量偏差影响。不能把A+外的样本都算成模型失败。','',
        '## 原门槛的数值对照','',
        'G0实际仍为False。G1数值条件='+str(stats['G1'])+'；G2数值条件='+str(stats['G2'])+'。G2数值条件包括A+成对来源≥30、覆盖差CI下界>0.15、无重复点估≥0.40、三代理方向不反。','',
        '**这些数值条件不构成预注册GO。** 无论是否达到，都不能写成已验证的真实重复结构形成缺陷，也不能定位唯一内部因果瓶颈。','',
        '## 质量、审计与复核','',
        '结构质量仅作诊断：E5评估并集中'+str(len(quality))+'张，sketch_IoU均值='+fmt(qmeans['sketch_IoU'])+'、EdgeF1均值='+fmt(qmeans['EdgeF1'])+'；不声称方法提升。','',
        '原始输入哈希保持一致：'+str(audit['input_hashes_unchanged'])+'；原G0产物哈希保持一致：'+str(audit['base_file_hashes_unchanged'])+'。冻结阈值及资格锁、几何窗口哈希见归档JSON。','',
        '图板按原sample_id排序固定选择全体前24条；规则在本轮测量前已写入代码，不依赖生成结果或资格类别。图板不新增人工真值，不用于调阈值，也不代表A+子集的全部视觉表现。视觉观察见本地visual_review.json。','',
        '作业116819在完成参考/GT并保存资格锁后，因日志time字段冲突失败；修复后116820从资格锁断点继续，未重算参考/GT，最终COMPLETED 0:0。两次均4CPU、申请16GB，分别运行102秒与29秒，合计131秒；MaxRSS字段未由Slurm报告。训练0、新模型生成0。详细资源见slurm_resources.txt。','',
        '已查看4张固定联系表，共24条R/sketch/GT/E5。以下仅为缩略图定性观察，不新增人工真值：','']
    lines += ['- '+v for v in visual['observations']]
    lines+=['','归档下载SHA256：`'+verify['actual_sha256']+'`。','',
        '本轮分析代码版本：`'+audit['code_commit']+'`。原G0报告仍为 `E42_real_repeat_gate_report.md` 与 `E42_go_no_go.md`；本轮报告和统计仅写本地docs。','']
    (docs/'E42_exploratory_repeat_report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(stats['status'],stats['funnel'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);deliver(parser.parse_args().root)
