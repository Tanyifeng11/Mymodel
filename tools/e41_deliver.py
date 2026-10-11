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
    d['local_delivery']=dict(complete=True,time=datetime.now().astimezone().isoformat(),state=state)
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
    (docs/'E41_attribute_bottleneck_report.md').write_text('\n'.join(lines),encoding='utf-8')
    go=['# E41 Go / No-go', '', '**本轮完成诊断，暂不进入新的属性grounding方法训练。**', '',
        '受控提取器校准结果、编码器到token的配对可读性变化、同latent条件响应及七点图像域轨迹已经记录。', '',
        '存在编码器到token误差上升证据的属性：'+('、'.join(supported) if supported else '无明确检出')+'。', '',
        '综合瓶颈分的探索性排序：'+(' > '.join(rank) if rank else 'NA')+'；该分包含epsilon响应代理与未经生成域校准的图像描述符，不能当作唯一因果瓶颈。', '',
        '真实纹样的物理尺度、独立生成图属性标签，以及masked garment latent读出的有效性仍未验证。下一步应先校准这些测量，再决定是否针对某属性/阶段改模型。', '',
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
