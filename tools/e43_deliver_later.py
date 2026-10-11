"""仅本地交付用户要求的探索性延续，原Pilot判断始终保留。"""
import csv
import re
from tools.e43_protocol import OUT,Path,read,write,sha


def num(v):return 'NA' if v is None else ('%.6f'%v if isinstance(v,float) else str(v))


def interval(v):
    return num(v['mean'])+(' ['+', '.join(num(x) for x in v['ci95'])+']' if v['ci95'] else '')


def accounting(path):
    rows=list(csv.DictReader(path.open(encoding='utf-8-sig'),delimiter='|'));jobs=[];card_seconds=0
    for row in rows:
        if not row['JobID'].isdigit():continue
        jobs.append(row);gpus=re.findall(r'gres/gpu(?::[^=,]+)?=(\d+)',row.get('AllocTRES',''))
        if gpus:
            elapsed=row['Elapsed'];days=0
            if '-' in elapsed:day,elapsed=elapsed.split('-',1);days=int(day)
            values=[int(x) for x in elapsed.split(':')];seconds=days*86400
            for value,factor in zip(reversed(values),[1,60,3600]):seconds+=value*factor
            card_seconds+=seconds*max(map(int,gpus))
    return dict(jobs=jobs,total_GPU_card_hours=card_seconds/3600)


def deliver(root):
    root=Path(root);out=root/OUT;review=out/'local_review';docs=root/'docs'
    download=read(out/'download_verification.json');verification=read(out/'local_verification.json')
    later=read(out/'later_verification.json');visual=read(review/'visual_review.json')
    assert download['sha256_verified'] and verification['passed'] and later['passed'] and visual['complete']
    state=read(review/'state.json');assert state['complete'],'仍有实验阶段待运行，不能宣称最终交付'
    assert set(later['stages']).issubset(set(visual.get('stages_reviewed',[]))),'后阶段固定联系表尚未人工视觉复核'
    decisions={p:read(review/p/'decision.json') for p in ['pilot','validation','holdout'] if (review/p/'decision.json').exists()}
    last=list(decisions)[-1];decision=decisions[last];continuation=read(review/'exploratory_continuation.json')
    resources=accounting(review/'slurm_accounting.txt');oof=read(review/'train/oof.json')
    local_state=read(out/'local_state.json');recovery={k:local_state[k] for k in ['train_failure','train_retry_jobs'] if k in local_state}
    summary=dict(experiment='E43_RRUC',status=decision['status'],last_stage=last,stop_reasons=decision['reasons'],
        original_pilot_status=decisions['pilot']['status'],preregistered_pipeline_pass=False,exploratory_continuation=continuation,
        decisions=decisions,stages=state,train_oof=oof,local_verification=verification,later_verification=later,
        visual_review=visual,resources=resources,engineering_recovery=recovery,download_sha256=download['actual_sha256'],E5_training_steps=0,
        generation_model_training_steps=0,claims=dict(pattern_identity=False,unique_causal_bottleneck=False,novel_method=False))
    write(docs/'E43_rgb_utility_statistics.json',summary)
    p=decisions['pilot'];lines=['# E43：参考增量RGB收益校准与探索性延续','',
        '**探索性延续状态：'+decision['status']+'；最后完成阶段：'+last+'。** 原预注册Pilot仍为NO_GO。','',
        '用户在查看Pilot结果后明确要求“继续完成后续的实验”，因此仅豁免Pilot停止门槛，继续原128来源训练收益采集、Ridge拟合及48来源双种子验证。来源名单、预算、生成参数、Ridge设置、验证及正式门槛未放宽。事后延续不能反推原Pilot通过。','',
        '停止原因：`'+str(decision['reasons'])+'`。正式评估只在验证通过后执行；未运行阶段保留未运行，不补填成绩。','',
        '## 冻结设置与可解释范围','',
        'E5、参考编码器和VAE冻结，生成模型训练0步；384×512、CFG7、DDIM50、alpha=0.65/1/1.35。前10步共同E5前缀，step1/5/10采集14个早期特征，后40步在各分支自身latent上重算R+/R0完整CFG。alpha1直接返回原eps。R0沿E39输入零化并保留偏置与原负分支，不换成黑图。','',
        'sketch内区侵蚀9px、羽化8px，失效时保留样本并精确回退E5，服装区域指标NA。真实服装区域官方AlexNet v0.1 LPIPS；灰度高通误差sigma3与LAB颜色误差分列；IoU/EdgeF1/Leak为原分数/比例。CLIP-I Real与Texture仅作副指标，不替代主门槛。','',
        '参考SHA来源隔离：Pilot24、训练128、验证48、正式128。Pilot/训练seed42，验证/正式seed42与123。历史500已在E39–E42使用，后两阶段只是回顾性控制器留出，不能称新盲测。训练监督来自最终RGB与GT；推理动作只读取早期特征和mask，禁止使用GT或后40步值。','',
        '两条StandardScaler+Ridge(alpha=1)分别预测low/high的结构惩罚收益，最大预测收益严格大于0.005才干预。来源4折OOF的标准化/PCA只在折内训练。time/sketch-only、token-only、shuffled采用相同回归框架。实际训练/验证特征第14维为协议指定的sketch置信度；Pilot旧采集该维为有效性标志，Pilot不参加Ridge拟合，不改写其原记录。','',
        '## 实际执行','', '| 阶段 | 状态 | 固定来源 | RGB记录数 | 区域可测来源 |','|---|---|---:|---:|---:|',
        '| Pilot | NO_GO（原结论保留） | 24 | 88 | 21 |']
    for phase,n in [('train',128),('validation',48),('holdout',128)]:
        v=later['stages'].get(phase)
        lines.append('| '+' | '.join([phase,str(state.get(phase,'pending')) if v else '未运行：前序停止',str(n),str(v['generated_pngs']) if v else '—',str(v['valid_sources']) if v else '—'])+' |')
    lines+=['','## 原Pilot结果','',
        '24固定来源中21可测；安全且LPIPS与灰度高通同时改善：low=10/24，high=0/24。因此原双方向门槛未通过。oracle LPIPS平均收益上界='+num(p['oracle_upper_bound']['mean'])+'，不是部署成绩。','']
    for arm in ['low','high']:
        m=p['comparisons'][arm]['metrics'];lines.append('- '+arm+'：ΔLPIPS='+interval(m['LPIPS_cloth'])+'；ΔHP='+num(m['HP_gray']['mean'])+'。')
    lines+=['','## 训练来源OOF泛化','',
        '固定128来源，可测'+str(oof['n_train_measurable'])+'。OOF实际动作收益与逐例GT oracle分开；oracle只作为不可部署上界。','',
        '| 控制器 | OOF ΔLPIPS vs E5 [95%CI] | 实际utility | oracle utility | oracle−实际 | 选择0/low/high |',
        '|---|---:|---:|---:|---:|---|']
    for kind,v in oof['results'].items():
        lines.append('| '+' | '.join([kind,interval(v['actual_oof_vs_e5']['metrics']['LPIPS_cloth']),num(v['actual_utility_mean']),num(v['oracle_utility_mean']),num(v['oracle_minus_actual_utility']),str(v['choices'])])+' |')
    for phase in ['validation','holdout']:
        if phase not in decisions:continue
        d=decisions[phase];lines+=['','## '+phase+'：最终RGB配对结果','',
            '状态：'+d['status']+'；原因：`'+str(d['reasons'])+'`；Best-fixed-val='+d['best_fixed_val']+'。两seed先在同来源内平均，再对来源等权做2000次bootstrap。','',
            '| 模型/策略 | ΔLPIPS vs E5 [95%CI] | ΔLPIPS vs Best-fixed [95%CI] | ΔIoU [95%CI] | ΔEdgeF1 | ΔLeak | ΔHP | ΔLAB | ΔCLIP Real / Texture |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---|']
        candidates=[(a,v,None) for a,v in d['fixed'].items()]+[(k,v['vs_e5'],v['vs_best_fixed']) for k,v in d['policies'].items()]
        if 'E36_B1' in d:candidates.append(('E36_B1',d['E36_B1']['vs_e5'],None))
        for name,v,b in candidates:
            m=v['metrics'];lines.append('| '+' | '.join([name,interval(m['LPIPS_cloth']),interval(b['metrics']['LPIPS_cloth']) if b else '—',interval(m['IoU'])]+[num(m[k]['mean']) for k in ['EdgeF1','Leak','HP_gray','LAB']]+[num(m['CLIP_real']['mean'])+' / '+num(m['CLIP_texture']['mean'])])+' |')
        ours=d['policies']['utility']['vs_e5'];lines+=['','Δ为策略−基线，LPIPS/HP/LAB/Leak越低越好，IoU/EdgeF1/CLIP越高越好。',
            '策略动作计数：`'+str(ours['choices'])+'`；无效mask行数='+str(len(later['stages'][phase]['invalid_mask_rows']))+'；IoU差第5百分位='+num(ours['delta_iou_p05'])+'；灾难性阈值标记='+str(len(ours['catastrophic_flags']))+'。所有失败ID和有效分母见统计JSON及rgb_results.csv。','',
            '简单控制器比较（utility策略−对照）：']
        for name,v in d['simple_controls'].items():lines.append('- '+name+'：ΔLPIPS='+interval(v['metrics']['LPIPS_cloth'])+'。')
        if 'E36_B1' in d:lines.append('- 复用已有E36-B1 step800权重，本实验不重训练；utility−B1 ΔLPIPS='+interval(d['E36_B1']['policy_vs_B1']['metrics']['LPIPS_cloth'])+'。')
        else:lines.append('- E36-B1缺少完整可核验结果，未填成绩。')
        lines+=['','双种子方向：']
        for seed,m in ours['per_seed'].items():lines.append('- seed'+seed+'：ΔLPIPS='+interval(m['LPIPS_cloth'])+'。')
        lines+=['','caption服装词和参考路径目录仅作描述性分层，未经人工语义核验，不是纹样身份真值。主要来源方向应结合统计JSON中的source_directions及有效来源数判断，不能用合并均值掩盖分组退化。','',
            '策略估计UNet调用/图='+num(d['efficiency']['policy_unet_calls_mean'])+'，原E5为100；估计时间/图='+num(d['efficiency']['policy_seconds_composed_mean'])+'秒。'+d['efficiency']['note']+'；未实测独立部署速度，不把三分支采集总时间当部署成本。','']
    lines+=['','## 审计、资源与视觉复核','',
        '完整50步预检查及PNG同一性通过；本地核验来源隔离、PNG/NPZ哈希、失效回退、来源簇bootstrap和动作记录。原始输入和生成权重在服务器核验，本地未重新加载LPIPS网络。BF旧checkpoint缺失4个未启用pattern_head张量导致跨作业整体哈希差异；已审计其他参数一致、正/零tokens逐位一致且辅助头未调用。各作业生成参数前后不变。','',
        'Slurm已分配累计GPU卡时='+num(resources['total_GPU_card_hours'])+'小时，包含准备后的预检查、Pilot及后续实际分配作业，逐作业状态/退出码/申请资源见统计JSON和slurm_accounting.txt。采集计时和真实部署估计分列，失败重跑资源也保留。','']
    lines+=['- '+v for v in visual['observations']]
    if recovery:
        lines+=['','训练分片116858在空sketch内区的RMS特征处失败：fp16将分母下限1e-12舍入为0，产生0/0；原E5前缀有限。提交9583514只将分母下限运算转fp32，非空掩码特征逐位不变。保留67个已完成来源，只续跑剩余61个，不删除失效mask样本、不改变阈值；失败及续跑资源均计入总量。','']
    lines+=['','本实验不支持纹样身份grounding、唯一因果瓶颈或方法新颖性结论。R0可能分布外；未通过门槛不追加FFT或修复器。即使探索性正式评估通过，也只支持后续原型研究，不能消除原Pilot失败和事后继续的限制。FID在128来源规模不报告。','',
        '归档SHA256：`'+download['actual_sha256']+'`。服务器全部产物位于`/share/home/u2515283058/Mymodel/'+str(OUT).replace('\\','/')+'`；三项最终交付只保存本地docs。','']
    (docs/'E43_rgb_utility_report.md').write_text('\n'.join(lines),encoding='utf-8')
    (docs/'E43_go_no_go.md').write_text('# E43探索性延续决策\n\n状态：'+decision['status']+'。最后完成阶段：'+last+'。\n\n原Pilot：NO_GO，未放宽原阈值；用户事后授权继续，不能记作预注册链条通过。\n\n原因：'+str(decision['reasons'])+'。\n\n详见本地E43_rgb_utility_report.md与E43_rgb_utility_statistics.json。\n',encoding='utf-8')
    print('E43 EXPLORATORY DELIVERED',last,decision['status'])
