"""只在本地生成E43交付；如实区分实际运行与门槛停止后的未运行阶段。"""
import argparse
from pathlib import Path
from tools.e43_protocol import OUT,read,write,sha


def number(v):
    return 'NA' if v is None else ('%.5f'%v if isinstance(v,float) else str(v))


def deliver(root):
    root=Path(root);out=root/OUT;review=out/'local_review';docs=root/'docs'
    download=read(out/'download_verification.json');assert download['sha256_verified']
    config=read(review/'locked_protocol.json');state=read(review/'state.json');preflight=read(review/'preflight.json');assert preflight['passed']
    verification=read(out/'local_verification.json');assert verification['passed']
    visual=read(review/'visual_review.json');assert visual['complete']
    decisions={phase:read(review/phase/'decision.json') for phase in ['pilot','validation','holdout'] if (review/phase/'decision.json').exists()}
    last=list(decisions)[-1];decision=decisions[last]
    summary=dict(experiment='E43_RRUC',last_stage=last,status=decision['status'],stop_reasons=decision['reasons'],
        stages=state,decisions=decisions,preflight=preflight,download_sha256=download['actual_sha256'],visual_review=visual,
        controller_fit=(review/'policy.pkl').exists(),E5_training_steps=0,local_verification=verification,
        generated_pilot_images=verification['generated_png_n'],resources=verification['resources'],
        execution_stages={k:('未运行：Pilot门槛停止' if state.get(k)=='pending' and not decision['pass_stage'] else state.get(k,'pending'))
            for k in ['precheck','pilot','train','validation','holdout']},
        claims=dict(pattern_identity=False,unique_causal_bottleneck=False,novel_method=False))
    write(docs/'E43_rgb_utility_statistics.json',summary)
    lines=['# E43：参考增量的最终RGB收益校准','',
        '**状态：'+decision['status']+'；最后完成阶段：'+last+'。** 停止原因：`'+str(decision['reasons'])+'`。','',
        '减弱参考存在真实RGB收益，但增强参考没有出现同时满足LPIPS改善、灰度高通改善与结构安全的Pilot样本，因此未通过实验前锁定的双方向门槛。按协议停止控制器训练、验证与正式评估；不等于证明所有参考调节都无效，也没有证明自适应控制器优于固定减弱。','',
        '## 实际执行与冻结协议','',
        'E5、参考编码器与VAE冻结，384×512、CFG7、DDIM50。前10步为共同E5前缀，后40步在各自当前latent上重算完整R+与R0 CFG，系数0.65/1/1.35。alpha=1直接返回原eps。R0沿用E39的CLIP与正BF输入零化路径，保留学习偏置，不换成黑图。','',
        'E5权重SHA256：`'+config['E5_sha256']+'`。预检查两例的完整50步轨迹及最终PNG与原E5一致；另外验证手动完整alpha1、零残差低/高系数和无效掩码回退一致。生成模型训练0步，Ridge是否实际拟合：'+str(summary['controller_fit'])+'。','',
        '按参考SHA256来源互斥锁定Pilot24、控制器训练128、验证48、正式评估128。Pilot与训练seed42；验证与正式评估计划seed42/123。验证/正式集合来自历史500条，已在E39–E42使用，属于回顾性控制器留出，不能称全新盲测。原配对、caption和原始数据不改写。','',
        '掩码只来自sketch：距离侵蚀9px、羽化8px后area下采样到latent。置信掩码失败则精确回退alpha1并保留样本，区域测量为NA。LPIPS采用官方AlexNet v0.1校准层，各特征层按固定sketch服装区域加权；不是仓库lpips_like像素代理。灰度高通误差sigma3，颜色LAB单列，结构与泄漏沿用现有指标，单位均为原始分数/比例。','',
        '| 阶段 | 实际状态 |','|---|---|']
    for stage in ['precheck','pilot','train','validation','holdout']:lines.append('| '+stage+' | '+str(summary['execution_stages'][stage])+' |')
    if 'pilot' in decisions:
        p=decisions['pilot'];lines+=['','## Pilot真实最终RGB结果','',
            '固定来源'+str(p['fixed_n'])+'；可测'+str(p['valid_n'])+'；有任一正LPIPS收益'+str(p['any_positive_gain_n'])+'；有结构安全正收益'+str(p['safe_positive_gain_n'])+'。安全且非颜色改善的低/高系数计数：`'+str(p['safe_noncolor_by_alpha'])+'`。','',
            '每例真实GT选优的LPIPS oracle平均上界='+number(p['oracle_upper_bound']['mean'])+'。这是不可部署上界，不能记作方法成绩；24条只用于筛选，不作论文显著性结论。','',
            '实际完成24×3=72张主分支RGB与前4来源的16张对照，共88张。3个低置信掩码来源：`'+str(verification['invalid_mask_ids'])+'`；均保留并精确回退E5。LPIPS/HP/LAB/IoU/Leak的有效分母为21，EdgeF1为24。安全非颜色改善：low=10/24，high=0/24。','',
            '| 分支相对E5 | ΔLPIPS_cloth [来源bootstrap 95%区间] | ΔHP_gray | ΔLAB | ΔIoU | ΔEdgeF1 | ΔLeak | IoU差第5百分位 | 崩坏标记数 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for arm,c in p['comparisons'].items():
            lp=c['metrics']['LPIPS_cloth'];interval=' ['+', '.join(number(x) for x in lp['ci95'])+']' if lp['ci95'] else ''
            lines.append('| '+' | '.join([arm,number(lp['mean'])+interval]+[number(c['metrics'][k]['mean']) for k in ['HP_gray','LAB','IoU','EdgeF1','Leak']]+[number(c['delta_iou_p05']),str(len(c['catastrophic_flags']))])+' |')
        lines+=['','所有Δ均为分支−E5：LPIPS/HP/LAB/Leak越低越好，IoU/EdgeF1越高越好。区域指标按有效来源统计并报告固定总分母，未将NA填0。详细来源级数值、置信区间和失败ID见统计JSON及rgb_results.csv。','',
            '21来源按来源等权进行2000次bootstrap；仅为Pilot描述性区间，不作正式显著性结论。共同前缀的作用、跨seed稳定性和控制器可预测性未获得正式检验。','',
            '前4个锁定Pilot来源另测从第1步生效与不使用内区掩码的两种系数，各8张；其中3个来源区域可测，不用于主方法训练。控制结果为：','']
        for arm,c in p['controls'].items():lines.append('- '+arm+'：ΔLPIPS='+number(c['metrics']['LPIPS_cloth']['mean'])+'，ΔIoU='+number(c['metrics']['IoU']['mean'])+'，ΔLeak='+number(c['metrics']['Leak']['mean'])+'。')
        lines+=['','固定low的IoU差第5百分位为'+number(p['comparisons']['low']['delta_iou_p05'])+'，仍有逐例退化。数值阈值未标记灾难性崩坏，不代表每例安全；low最差LPIPS来源210013766，low最差IoU来源162607053，high最差LPIPS/IoU来源195833180。详见本地核验JSON的逐例差值清单。','']
    resources=verification['resources']
    lines+=['','## 资源与审计','',
        'Pilot GPU作业116846/116847耗时5分01秒/4分03秒，均1张A30、2CPU、申请40GB内存；预检查116843为2分26秒。合计GPU卡时约'+number(resources['total_GPU_card_hours'])+'小时。CPU准备116841、重复幂等准备116842、统计116848与辅助头审计116849的完整退出状态见slurm_accounting.txt。','',
        '88张采集共'+str(resources['pilot_unet_calls'])+'次UNet调用；分支记录生成耗时总和'+number(resources['pilot_sum_recorded_generation_seconds'])+'秒，峰值已分配显存'+number(resources['pilot_peak_allocated_GiB'])+'GiB。作业时间另含加载、指标与审计开销，不能直接除以图数作为可部署策略速度。'+resources['deployment_note'],'',
        'BF模块跨作业整体哈希不同：旧E5缺失pattern_head的4个辅助头张量，加载器保留随机初始化；生成时pattern_loss_enabled=False，该头不参与输出。CPU作业116849重建两种初始化，核验其余张量一致、正参考/零参考输出token逐位一致，且辅助头未被调用。各生成作业内全部参数前后哈希不变；没有静默忽略整体哈希差异。','',
        '本地重新核验88张PNG哈希、NPZ共同前缀、预检查50步同一性、来源隔离、有效分母、来源bootstrap和停止门槛。原始输入内容与权重文件在服务器核验；本地未重新加载原始数据或运行LPIPS网络。','',
        'CLIP-I Real、CLIP Texture、E36-B1、两seed泛化和Ridge对照属于未进入的后续比较阶段，本次未测，不补填成绩。FID未计算。','']
    lines+=['','## 视觉复核与解释边界','']+['- '+v for v in visual['observations']]
    lines+=['','达到停止条件后未运行的阶段不补填成绩，不训练更大控制器，也不靠FFT或修复模块补救。本实验的RGB/结构误差不能证明纹样身份grounding或唯一内部因果瓶颈；R0仍是可能分布外的操作性消融。','',
        '完整联系表、前4来源对照与失败样本板位于本地output_eval/e43_rgb_utility_20261011/local_review/pilot/contact_sheets；四张固定联系表覆盖全部24例。','',
        '服务器结果目录：`/share/home/u2515283058/Mymodel/'+str(OUT).replace('\\','/')+'`。本报告与统计仅保存本地docs。下载归档SHA256：`'+download['actual_sha256']+'`。','']
    (docs/'E43_rgb_utility_report.md').write_text('\n'.join(lines),encoding='utf-8')
    (docs/'E43_go_no_go.md').write_text('# E43决策\n\n状态：'+decision['status']+'。最后完成阶段：'+last+'。\n\n原因：'+str(decision['reasons'])+'。\n\n参见E43_rgb_utility_report.md和E43_rgb_utility_statistics.json。未运行阶段不代表失败成绩；不支持纹样身份或论文新颖性结论。\n',encoding='utf-8')
    print(summary['status'],summary['stop_reasons'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);deliver(p.parse_args().root)
