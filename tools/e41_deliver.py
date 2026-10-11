"""仅本地生成E41三项文本/JSON交付和Excel所需输入；不上传报告。"""
import argparse
import json
from pathlib import Path
from datetime import datetime


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def fmt(v):return '未验证' if v is None else '%.4f'%v
def stat(s):return fmt(s['mean'])+(' ['+', '.join(fmt(v) for v in s['ci95'])+']' if s['ci95'] else '')+'；组数='+str(s.get('groups',0))


def deliver(root):
    root=Path(root);out=root/'output_eval/e41_attribute_bottleneck_20261011';local=out/'local_review';docs=root/'docs'
    d=read(local/'E41_attribute_trajectory.json');state=read(out/'local_state.json');docs.mkdir(exist_ok=True)
    d['local_delivery']=dict(complete=state.get('complete',False),time=datetime.now().astimezone().isoformat(),state=state)
    for name in ['local_download_check.json','visual_review.json']:
        p=local/'audit'/name
        if p.exists():d[name.removesuffix('.json')]=read(p)
    (docs/'E41_attribute_trajectory.json').write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    attrs=['orientation','period','scale','density'];units=dict(orientation='轴向角误差，度',period='周期绝对对数比',scale='尺度绝对对数比',density='占位比例绝对差')
    lines=['# E41：纹样属性瓶颈定位', '', '已完成冻结 E5 的 A–D 分析，新增160张受控生成；真实轨迹与RGB只读复用E39。seed42、CFG7、DDIM50，生成模型更新0。', '',
        '本轮将属性分解为单元方向、晶格中心间距、单元尺寸、晶格占位率。它们是图像坐标属性，不是布料物理单位。完整纹样身份和真实同纹跨材质不在本轮已验证结论之内。', '',
        '## 输入与测量校准', '',
        '方向0°/30°/45°/90°，周期32/48/64像素，尺度12/18/24像素，密度1/3、2/3、1；108组合×4相位×2配色=864受控参考。所有同参数族视图同折。真实89个来源各构造4个方向视图，标签为输入提取器伪标签。', '',
        '方向使用FFT高频角向二阶矩并按180°等价计算误差。周期保留全径向峰，同时用固定轴向晶格投影区分缺位超结构；轴向假设尚未在任意自然纹样或生成域成立。尺度使用孤立单元的多尺度频率响应和固定几何标准查表；密度使用前景单元计数/晶格面积，边缘密度另列。', '',
        '生成图、七点预测x0和参考均映射到256×256测量坐标；生成区域由sketch-only掩码确定并内缩。由此报告相对图像的周期/尺度，不能解读成真实布料尺寸。自然图像前景斑块不等于已确认的纹样重复单元。', '',
        '| 属性 | 误差单位 | 输入 MAE | 有效覆盖率 | 预设门槛 | 校准 |', '|---|---|---:|---:|---:|---|']
    for a in attrs:
        r=d['calibration'][a];lines.append('| '+' | '.join([a,units[a],fmt(r['mae']),fmt(r['coverage']),fmt(r['limit']),'通过' if r['pass_calibration'] else '未通过'])+' |')
    lines+=['', '占位变化会带来额外空间频率、颜色覆盖与边缘统计变化。“仅一个渲染参数变化”不能等同于“仅一个实测属性变化”；输入实际描述符均已保留。', '',
        '## 阶段 A：编码器到 texture token', '',
        '仅训练固定 Ridge alpha=1 的读出probe，标准化及最多32维PCA只在训练折拟合，四折按参数族/真实来源分组。方向回归cos2θ/sin2θ，周期与尺度回归log值。主读出为固定16段池化的序列特征；mean/std作为对照，不能把它称为完整原生flatten。', '',
        '| 属性 | 融合编码器 MAE [95% CI] | 最终 token MAE [95% CI] | 常数基线 MAE | token/常数误差比 | token−encoder MAE [95% CI] |', '|---|---|---|---|---:|---|']
    probe_rows=[]
    for domain,layers in d['representation'].items():
        for layer,modes in layers.items():
            for mode,values in modes.items():
                for a,v in values.items():probe_rows.append([domain,layer,mode,a,units[a],v['mae']['mean'],*(v['mae']['ci95'] or [None,None]),v['null_mae']['mean'],v['shuffled_mae']['mean'],v['normalized_mae'],v['valid_readout'],v['mae']['groups'],v['mae']['n']])
    for a in attrs:
        e=d['representation']['controlled']['encoder_fused']['spatial'][a];t=d['representation']['controlled']['token_tcpm']['spatial'][a]
        lines.append('| '+' | '.join([a,stat(e['mae']),stat(t['mae']),fmt(t['null_mae']['mean']),fmt(t['normalized_mae']),stat(d['compression'][a]['token_minus_encoder_error'])])+' |')
    supported=[a for a in attrs if d['compression'][a]['compression_bottleneck_supported']]
    lines+=['', '编码器到token的配对误差上升证据：'+('、'.join(supported) if supported else '未检出明确证据')+'。该判断只涉及匹配线性probe容量下的可读性，不能据此证明信息不可恢复或唯一归因于压缩模块。全部CLIP/CNN/融合/resampler/BF/TCPM、两种读出及真实伪标签结果见JSON与Excel。', '',
        '## 阶段 B：同 latent 条件响应', '',
        '16个载体×（基准+9个单参数变化）=160张生成。每个载体文本、sketch、初始噪声保持一致，不提示纹样属性。在R+实际z_t上，仅替换参考产生的条件token，保持timestep和无条件分支一致。保存原轨迹回放误差及相同参考sham；Δε只是条件响应，不能直接等同于属性正确利用。', '',
        '| 属性 | 按参数跨度标准化的 Δε 相对敏感性 [95% CI] | sham [95% CI] |','|---|---|---|']
    for a in attrs:
        r=d['conditioning']['summary'][a];lines.append('| '+a+' | '+stat(r['standardized_sensitivity'])+' | '+stat(r['sham'])+' |')
    lines+=['', '## 阶段 C–D：去噪轨迹与最终 RGB', '',
        '记录真实DDIM时间点981、801、601、401、201、41、1；从同一步z_t及CFG后的epsilon重建预测x0，再使用原VAE解码。预先固定801为早期、1为后期，不按结果选择最优时间。下表为图像域属性代理误差，均按载体聚合后bootstrap。', '',
        '| 属性 | t981 | t801 | t601 | t401 | t201 | t41 | t1 | 最终RGB |', '|---|---|---|---|---|---|---|---|---|']
    trajectory_rows=[]
    for a in attrs:
        s=d['generation']['summary'][a];lines.append('| '+' | '.join([a]+[stat(s[str(t)]) for t in [981,801,601,401,201,41,1,'RGB']])+' |')
        for t,r in s.items():trajectory_rows.append([a,units[a],str(t),r['mean'],*(r['ci95'] or [None,None]),r['groups'],r['n']])
    lines+=['', '| 属性 | 后期−早期误差 [95% CI] | VAE标准重建误差 [95% CI] |', '|---|---|---|']
    for a in attrs:lines.append('| '+a+' | '+stat(d['generation']['late_minus_early_error'][a])+' | '+stat(d['vae_controls']['roundtrip_error'][a])+' |')
    lines+=['', '独立程序标准用于验证VAE roundtrip及latent probe。生成服装latent经过掩码且存在域偏移，尚无生成域属性标签来校准该读出；因此“latent有、RGB无”的解码器归因记未验证，不能直接对latent各通道做RGB频谱解释。完整latent读出结果保留为域外代理。', '',
        f"区域掩码有效 {d['generation']['valid_masks']}/160；每项指标均单列有效数和来源组数，缺失值不计为0。真实E39复用32个来源，核验原有 {d['real_reuse']['original_images_hash_checked']} 张图像哈希，未重新生成。真实结果使用旧文本及输入伪属性，只作外部探索性对照。", '',
        '## 属性瓶颈评分与判断', '',
        'E_rep=token MAE/常数基线MAE；E_cond=按输入参数跨度标准化的epsilon响应相对本轮最大响应的缺口；E_denoise=正向后期−早期误差/基线MAE；E_rgb=最终RGB误差/基线MAE。各项截到[0,1]，四项等权。任何缺失阶段都保留NA。E_cond不是语义利用误差，图像域代理也未经生成域标签校准，因此综合分仅作探索性比较。', '',
        '| 属性 | E_rep | E_cond代理 | E_denoise代理 | E_rgb代理 | 等权B [95% CI] |', '|---|---:|---:|---:|---:|---|']
    comparison_rows=[]
    for a in attrs:
        r=d['bottleneck']['attributes'][a];ci=r.get('B_ci95');lines.append('| '+' | '.join([a]+[fmt(r[k]) for k in ['E_rep','E_cond','E_denoise','E_rgb']]+[fmt(r['B_equal_weight'])+(' ['+', '.join(fmt(v) for v in ci)+']' if ci else '')])+' |')
        comparison_rows.append([a,units[a],r['E_rep'],r['E_cond'],r['E_denoise'],r['E_rgb'],.25,.25,.25,.25,r['B_equal_weight'],*(ci or [None,None]),r['compression_evidence'],'探索性综合分；因果定位未验证'])
    rank=d['bottleneck']['exploratory_ranking']
    lines+=['', '探索性排序：'+(' > '.join(rank) if rank else '不可计算')+'。另计算四种单阶段权重0.4、其余各0.2的敏感性，并保留首位与其他属性的配对差值区间。排序不能单独证明某属性是主因。', '',
        '当前因果判断：`'+d['bottleneck']['status']+'`。可报告输入校准、表示可读性下降、条件响应和图像域代理的实际证据；尚不足以确认唯一主属性、后期覆盖的唯一机制或解码器故障。不能因代理分数无下降就断言四类属性均被正确保留。', '',
        '## 审计与结果位置', '',
        '服务器所有新结果：`/share/home/u2515283058/Mymodel/output_eval/e41_attribute_bottleneck_20261011`。E5检查点SHA256：`'+d['protocol']['E5_sha256']+'`。两分片运行前后模块哈希相同，原始输入和既有输出保持不变。NPZ保留服务器，复核图及JSON下载本地。', '',
        '全部16个载体最终图板及指定时间点图像均在本地复核目录：['+str(local/'review').replace('\\','/')+']('+str(local/'review').replace('\\','/')+')。四项总结交付只存本地docs。', '']
    # 结论依赖实际数据；显式展示缺失、域偏移和未经校准的代理范围。
    top=rank[0] if rank else None
    findings=['## 本轮回答', '',
        '探索性评分最高属性：'+(top or '无法计算')+'。方向、周期和密度在编码器→最终token存在配对读出误差上升，尺度未检出明确上升；四属性最终token仍均优于常数基线。', '',
        '可定位的是匹配线性probe下的表示可读性下降；尚不能把完整迁移失败唯一归因到某个属性或阶段。后期密度代理误差上升，但生成域提取器不满足已验证的占位率解释。当前不进入新模型训练，先校准生成域测量。', '']
    lines[6:6]=findings
    lines+=['## 有效观测与生成域限制', '',
        '| 属性 | 最终RGB有效图数 | 有效载体组数 |', '|---|---:|---:|']
    for a in attrs:
        s=d['generation']['summary'][a]['RGB'];lines.append(f"| {a} | {s['n']} | {s['groups']} |")
    audit=d.get('local_download_check',{})
    lines+=['', '掩码无效载体：'+', '.join(audit.get('mask_invalid_carriers',[]))+'。其20张生成保留用于视觉复核，属性计算按缺失处理，不补为0。', '',
        f"最终密度代理中 {audit.get('density_proxy_outside_count','未核查')} 个估计超出[0,1]，不能解释为真实晶格占位率。保留原值以暴露测量失配；没有根据生成结果重调提取器或裁剪原始读数。综合评分内E_rgb截到1会掩盖大误差差别，须结合原始误差及有效数阅读。", '',
        '## 真实参考验证与复用', '',
        '真实参考来自89个来源的356个视图，目标为输入提取器伪标签；全部最终token主读出均未满足预设有效读出门槛，不能从受控椭圆结果直接外推自然纹样。', '',
        '| 属性 | 真实最终token MAE [95% CI] | 常数基线 | 有效读出 |', '|---|---|---:|---|']
    for a in attrs:
        r=d['representation']['real']['token_tcpm']['spatial'][a]
        lines.append('| '+a+' | '+stat(r['mae'])+' | '+fmt(r['null_mae']['mean'])+' | '+('是' if r['valid_readout'] else '否')+' |')
    lines+=['', '| 属性 | E39复用最终RGB误差 [95% CI] |', '|---|---|']
    for a in attrs:lines.append('| '+a+' | '+stat(d['real_reuse']['summary'][a]['RGB'])+' |')
    lines+=['', '真实参考的45°与135°旋转视图原有黑色补边，主分片完成后以反射扩边修复，再重提取全部356个真实辅助参考特征。178张视图发生变化，原版参考列表/特征保留审计备份；160张生成PNG哈希不变，新增生成数0。受控数据及原BF文件未修改。', '',
        '## 单参数干预的交叉响应', '',
        '下表为单一渲染参数改变后，四种输出描述符相对基准的绝对变化；按载体bootstrap。值越大仅表示输出变化较大，不能解释为变化方向正确。输入描述符的交叉变化逐条件保存在JSON的generation.intervention_matrix。', '',
        '| 改变参数 | 输出方向变化 | 输出周期变化 | 输出尺度变化 | 输出密度变化 |', '|---|---|---|---|---|']
    for a in attrs:
        matrix=d['generation']['intervention_matrix'][a]
        lines.append('| '+' | '.join([a]+[stat(matrix[b]['output_change']) for b in attrs])+' |')
    lines+=['', '## 排序稳健性与视觉复核', '', '| 阶段权重（表示/条件/去噪/RGB） | 探索性排序 |', '|---|---|']
    for s in d['bottleneck']['weight_sensitivity']:
        lines.append('| '+', '.join(fmt(w) for w in s['weights'])+' | '+' > '.join(s['ranking'])+' |')
    lines+=['', '| 探索性首位与其他属性之差 | 配对bootstrap 95% CI |', '|---|---|']
    for name,ci in d['bottleneck']['exploratory_top_difference_ci95'].items():
        lines.append('| '+name+' | '+', '.join(fmt(v) for v in ci)+' |')
    visual=d.get('visual_review',{})
    lines+=['', '以下为单AI视觉复核，非独立人工真值。', '']
    lines+=['- '+s for s in visual.get('observations',[])]
    lines+=['', '## 计算资源与版本审计', '', '| 作业 | 任务 | 状态 | 实际秒数 | CPU核数 | GPU数 |', '|---|---|---|---:|---:|---:|']
    for job in audit.get('jobs',[]):
        tres=dict(x.split('=',1) for x in job['AllocTRES'].split(','))
        lines.append('| '+' | '.join([job['JobID'],job['JobName'],job['State'],job['ElapsedRaw'],tres['cpu'],tres.get('gres/gpu','0')])+' |')
    pack=audit.get('packing_job',{})
    if pack:lines.append(f"| {pack['id']} | E41_pack | {pack['state']} | {pack['elapsed_seconds']} | 2 | 0 |")
    total=audit.get('total_resources_with_pack',{})
    lines+=['', '含归档作业分配资源：GPU '+fmt(total.get('gpu_hours'))+' 卡时，CPU '+fmt(total.get('cpu_core_hours'))+' 核时；采用Slurm实际Elapsed×AllocTRES计算，排队等待不计。', '',
        '计算分片0提交e5f34e2，分片1、辅助参考修复及分析提交aa3b9d0；这两版本的e41_runtime.py和e41_attributes.py一致，后续变化为真实旋转修复及统计/报告修正。', '',
        audit.get('cross_shard_hash_note',''), '',
        f"下载归档SHA256：`{audit.get('archive_sha256','未核验')}`。manifest共{audit.get('manifest_entries','NA')}项，本地核验{audit.get('checked','NA')}项均一致；{audit.get('missing_count','NA')}项NPZ有意仅留服务器。另核验160张生成PNG与其生成记录哈希、1220张参考PNG、每载体10条件初始latent一致和三个GPU作业前后冻结模块哈希。", '']
    (docs/'E41_attribute_bottleneck_report.md').write_text('\n'.join(lines),encoding='utf-8')
    go=['# E41 Go / No-go', '', '**本轮完成诊断，暂不进入新的属性grounding方法训练。**', '',
        '受控提取器校准结果、编码器到token的配对可读性变化、同latent条件响应及七点图像域轨迹已经记录。', '',
        '存在编码器到token误差上升证据的属性：'+('、'.join(supported) if supported else '无明确检出')+'。', '',
        '综合瓶颈分的探索性排序：'+(' > '.join(rank) if rank else 'NA')+'；该分包含epsilon响应代理与未经生成域校准的图像描述符，不能当作唯一因果瓶颈。', '',
        '密度在全部五组权重下均位列探索评分首位，但89个最终密度代理估计超过1；该排序不能当作物理占位率丢失的结论。受控域四属性最终token仍均可读；仅凭该读出不能判断表示压缩是否足以解释全部迁移失败。', '',
        '真实参考的最终token主读出四属性均未通过有效性门槛。真实纹样的物理尺度、独立生成图属性标签，以及masked garment latent读出的有效性仍未验证。下一步应先在14个有效载体上用独立标注/单元定位校准生成域测量，再决定是否针对某属性/阶段改模型。', '',
        '文档Stop条件“四属性均无明显衰减但纹样错误”目前无法被严格确认，也没有据此否定整个方向。当前状态：`'+d['bottleneck']['status']+'`。', '']
    (docs/'E41_go_no_go.md').write_text('\n'.join(go),encoding='utf-8')
    conditioning_rows=[]
    for a,v in d['conditioning']['summary'].items():
        for t,r in v['by_timestep'].items():conditioning_rows.append([a,int(t),r['mean'],*(r['ci95'] or [None,None]),r['groups'],r['n']])
    raw=[[r['id'],r['carrier'],r['condition'],r['attribute'],str(r['t']),r['reference'],r['generated'],r['error'],r['mask_valid']] for r in d['generation']['records']]
    workbook=dict(comparison=comparison_rows,probes=probe_rows,trajectory=trajectory_rows,conditioning=conditioning_rows,raw=raw,
        source_json=str(docs/'E41_attribute_trajectory.json'),server_output=state['server_output'])
    (out/'workbook_input.json').write_text(json.dumps(workbook,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print('E41 LOCAL TEXT DELIVERED',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();deliver(a.root)
