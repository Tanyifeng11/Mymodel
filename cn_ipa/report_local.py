"""实验结束后仅在本地生成结果总结；不在服务器执行。"""
import argparse
import shutil
from pathlib import Path
import numpy as np
from cn_ipa.common import read


def interval(value,signed=False):
    if value['n']==0:return 'NA（n=0）'
    fmt=(lambda x:f'{x:+.6f}') if signed else (lambda x:f'{x:.6f}')
    return f"{fmt(value['mean'])} [{fmt(value['ci95'][0])}, {fmt(value['ci95'][1])}]（n={value['n']}）"


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results',required=True);parser.add_argument('--docs',default='docs')
    args=parser.parse_args();root=Path(args.results).resolve();docs=Path(args.docs).resolve()
    assert read(root/'audit/finalized.json')['complete'], '全部实验完成后才生成总结'
    means=read(root/'metrics/means.json');paired=read(root/'metrics/paired_ci.json');failures=read(root/'metrics/failure_ids.json')
    frozen=read(root/'selection/frozen_final_choice.json');audit=read(root/'manifests/split_audit.json')
    training=read(root/'audit/training_runs.json');versions=read(root/'audit/versions.json')
    budget=read(root/'audit/image_budget.json')
    assert read(root/'audit/sd15_official_validation.json')['passed']
    assert read(root/'audit/historical_reuse.json')['passed']
    assets=docs/'cn_ipa_repro_20261022';assets.mkdir(parents=True,exist_ok=True)
    for folder in ['metrics','selection','audit','qualitative','resources']:
        destination=assets/folder;destination.mkdir(exist_ok=True)
        for source in (root/folder).rglob('*'):
            if source.is_file() and source.suffix in ['.json','.csv','.png']:
                path=destination/source.relative_to(root/folder);path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,path)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    arms=['C0','C1','C2','E5','B1','M10','M10_GAIN','M12','M12_GAIN']
    colors={a:('#1d4ed8' if a=='C2' else '#60a5fa' if a.startswith('C') else '#ea580c' if a.startswith('M12') else '#9ca3af') for a in arms}
    keys=['lpips_gt','contour_f1','iou','leak'];titles=['LPIPS (lower)','Contour F1 (higher)','IoU (higher)','Leak (lower)']
    fig,axes=plt.subplots(1,4,figsize=(13,4.5),sharey=True)
    for axis,key,title in zip(axes,keys,titles):
        for i,arm in enumerate(arms):
            value=means[arm][key]
            if not value['n']:continue
            mean=value['mean'];lo,hi=value['ci95']
            axis.errorbar(mean,i,xerr=[[max(mean-lo,0)],[max(hi-mean,0)]],fmt='o',color=colors[arm],capsize=3)
        axis.set_title(title);axis.grid(axis='x',alpha=.2)
    axes[0].set_yticks(range(len(arms)),arms);axes[0].invert_yaxis()
    fig.suptitle('TEST-A: 96 identities, seeds 42/43/44; identity bootstrap 95% CI')
    fig.tight_layout();fig.savefig(assets/'metrics_ci.png',dpi=180);fig.savefig(assets/'metrics_ci.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.3))
    for axis,name,title in zip(axes,['cn_fashion','ip_fashion'],['ControlNet fine-tuning','IP-Adapter fine-tuning']):
        rows=training[name]['history'];x=[r['step'] for r in rows];y=np.array([r['loss'] for r in rows])
        axis.plot(x,y,alpha=.3,lw=1,label='logged batch loss')
        if len(y)>=5:axis.plot(x[4:],np.convolve(y,np.ones(5)/5,mode='valid'),lw=1.5,label='5-record moving mean')
        axis.set(title=title,xlabel='optimizer updates',ylabel='epsilon MSE');axis.legend(fontsize=8);axis.grid(alpha=.2)
    fig.tight_layout();fig.savefig(assets/'training_loss.png',dpi=180);plt.close(fig)
    lookup={(r['candidate'],r['baseline'],r['metric']):r for r in paired}
    delta=lambda a,b,k:interval(lookup[a,b,k],True)
    lines=[
        '# ControlNet + IP-Adapter 对比 M12：实验结果（2026-10-22）','',
        f"已完成 C0/C1/C2 三个外部实验臂，新增正式测试图 **864 张**；与历史六臂共 **2592 张**图像统一复算四项指标。每身份先平均三个 Seed，再对身份进行 Bootstrap。",'',
        f"C2−M12：LPIPS {delta('C2','M12','lpips_gt')}；F1 {delta('C2','M12','contour_f1')}；IoU {delta('C2','M12','iou')}；Leak {delta('C2','M12','leak')}。这些配对区间按预设口径作为探索性证据，不作未经多重校正的确认性优越结论。",'',
        '## 固定协议与输入','',
        f"训练 {audit['train']}、开发 {audit['dev']}、TEST-A {audit['test_A']} 个身份。训练来源为历史 M54 train1024；开发来源为历史 dev32；测试严格复用 2026-10-20 已有确认集。该测试集已经被研究者审阅，本轮是既有确认集上的新增外部对比，不能称新盲测。",'',
        '输出宽384×高512，DDIM50、CFG7，Seed42/43/44。负提示词与历史 M12 实际调用相同：` worst quality, low quality`。同步 BILINEAR 缩放沿用历史几何协议，保留原始黑线白底，不作方形中心裁剪。', '',
        'identity、文件/RGB哈希及 GT/reference 跨模态 pHash 距离≤6 审计通过；线稿做精确/像素排重。Reference 始终是数据集提供的独立 texture 文件，训练 CLIP 不读取 GT；其是否源自与 GT 同一次采集无法由文件分离证明。公开预训练、Writer 完整预训练暴露未知，不能声称对所有预训练权重绝对未见。','',
        'C0：公开 Lineart CN + 标准 IPA，无新增训练。C1：只用 epsilon MSE 微调 CN，IPA 固定。C2：固定选中 CN，只训练 IPA image projection 与 image K/V；输入为 Text + Sketch + 原始 Reference，目标为 GT。没有引入 M12 专属损失。','',
        'C2 是基于公开方法的本任务适配实现，不是原论文发布的服装三条件一键模型。训练前 forward、终图回放、权重导出重载、参考/草图端点、梯度和冻结哈希检查记录均在审计文件中。', '',
        '## 平均指标与身份级95%区间','',
        '| 方法 | LPIPS↓ | F1↑ | IoU↑ | Leak↓ | 任一Seed极端身份 |',
        '| --- | --- | --- | --- | --- | --- |'
    ]
    for arm in arms:
        extreme=sum(r['variant']==arm for r in failures)
        lines.append('| '+arm+' | '+' | '.join(interval(means[arm][k]) for k in keys)+f' | {extreme}/96 |')
    lines += ['', '![均值和区间](cn_ipa_repro_20261022/metrics_ci.png)','',
        '四项评分沿用历史 LPIPS-Alex、Sketch-only 前景/轮廓与 Leak 实现。LPIPS使用历史FP32/CuDNN TF32口径，先在旧DEV输出回放验证；它与生成的FP16设置分开。历史1728张图逐张核验 PNG 哈希并复算，数值差异限制为1e-6。缺任一Seed的指标则该身份该指标为NA；未把288张图当作288个独立身份。', '',
        '本轮主表使用统一服务器GPU评分，没有声称完成新增输出的完整本地CPU LPIPS重算。历史确认实验原有CPU/GPU 1e-6严格容差失败（最大差约0.00012144）仍保留，不能把本轮同GPU回放通过解释为已经修复历史跨设备精度差异。','',
        '## 配对比较','',
        '| 候选−基线 | ΔLPIPS↓ | ΔF1↑ | ΔIoU↑ | ΔLeak↓ |', '| --- | --- | --- | --- | --- |']
    for a,b in [('C0','M12'),('C1','M12'),('C2','M12'),('C0','M12_GAIN'),('C1','M12_GAIN'),('C2','M12_GAIN'),('C1','C0'),('C2','C1')]:
        lines.append('| '+a+'−'+b+' | '+' | '.join(delta(a,b,k) for k in keys)+' |')
    lines += ['', 'Bootstrap10000次、seed32042；按身份平均三个Seed后配对。逐Seed、删一身份敏感性、NA和极端身份原始表全部保留。极端事件沿用任一Seed的 `F1_方法−F1_E5 < -0.15`。', '',
        '历史 M12_GAIN 的六门禁状态仍为 CONDITIONAL：不能用本轮外部模型排序改写其既有 Leak/极端事件失败；“B1收益保留比例”也没有作为外部方法的唯一资格标准。','',
        '## 检查点、控制强度与开发预算','',
        '| 方法 | CN检查点 | IPA检查点 | CN scale | IPA scale |','| --- | --- | --- | --- | --- |']
    for arm in ['C0','C1','C2']:
        choice=frozen['models'][arm]
        cn='公开预训练' if arm=='C0' else str(read(root/'selection/cn_checkpoint.json')['step'])
        ip=str(read(root/'selection/ip_checkpoint.json')['step']) if arm=='C2' else '公开预训练'
        lines.append(f"| {arm} | {cn} | {ip} | {choice['cn_scale']} | {choice['ip_scale']} |")
    lines += ['', 'CN候选1000/2000/3000/4000/5000更新；IPA候选500/1000/1500/2000/2500/3000更新。每臂开发集控制强度为3×3网格，另有CN-only基线及检查点开发评测；这些图均不计入864张正式测试。结构/外溢边界和无合格组合的回退排序均在生成前固定，完整候选表与最终冻结哈希保留。', '',
        f"实际保存：DEV图 {budget['dev']} 张，烟测/预检图 {budget['smoke']} 张，耗时测试图 {budget['profiles']} 张，正式新测试图 {budget['outputs']} 张。另有定性对照拼图 {budget['qualitative']} 张；拼图不算新增独立生成结果。完整开发分项见 image_budget.json。",'',
        '## 训练与推理成本','',
        '| 阶段 | 最终更新/实际执行 | 实际取入样本 | 可训练参数 | 优化循环秒 | 峰值allocated GiB |','| --- | --- | --- | --- | --- | --- |']
    for name,label in [('cn_fashion','ControlNet'),('ip_fashion','IP-Adapter')]:
        done=training[name]['complete'];parameters=done.get('trainable_parameters')
        if parameters is None:parameters=read(root/'audit/smoke_C1.json')['parameters']['controlnet']
        usage=training[name]['usage']
        lines.append(f"| {label} | {done['updates']}/{usage['actual_optimizer_updates']} | {usage['actual_loop_samples']} | {parameters:,} | {done.get('optimizer_loop_seconds',done['seconds']):.1f} | {done['peak_allocated_gib']:.3f} |")
    lines += ['', '正式最终训练路径为CN5000更新和IPA3000更新，C1/C2共享同一份CN训练。实际执行另计中断后的重放和被丢弃计算，不把重放当作最终模型额外有效更新。全局有效batch=4。CN lr1e-5，IPA lr5e-5，AdamW；IPA image/text/both dropout各0.05。表内正式训练不含CN断点烟测2次和IPA梯度烟测1次；溢出跳过及真实样本消费单独记录。IPA官方权重先按FP16推理图加载，再升为FP32主参数，冻结骨干为FP16。', '',
        f"环境：Python {versions['python']}，PyTorch {versions['torch']}，Diffusers {versions['diffusers']}，Transformers {versions['transformers']}，Accelerate {versions['accelerate']}。训练阶段加载、缓存准备、最终哈希/写盘和失败/重试分配与优化循环计时范围有区别；完整 Slurm 账本和阶段记录可查。",'',
        '| 方法 | 实际总加载参数 | 单图延迟中位数秒 | 峰值allocated GiB | GPU |','| --- | --- | --- | --- | --- |']
    for arm in ['C0','C1','C2','M12','M12_GAIN']:
        profile=read(root/'profiles'/f'{arm}.json');records=profile['records']
        latency=float(np.median([r['seconds'] for r in records]))
        peak=max(r.get('peak_allocated_gib',r.get('peak_memory_gib')) for r in records)
        count=profile.get('total_loaded_parameters',profile.get('parameters',{}).get('total'))
        lines.append(f"| {arm} | {count:,} | {latency:.3f} | {peak:.3f} | {profile['gpu']} |")
    lines += ['', '五臂分别使用全新进程，先预热1次，再在相同四个历史开发身份上测4次；CUDA同步、FP16、DDIM50。计时为原生生产管线调用，排除输入加载、参考输入准备及PNG写盘；模型各自原生编码器/Writer路径的计算范围见记录。历史 E5/B1/Writer 的训练成本未知或不完整恢复，不能把M12_GAIN的1280个gain参数当作整个模型成本。','',
        '![训练损失](cn_ipa_repro_20261022/training_loss.png)','',
        '## 固定定性展示与失败案例','',
        '以下12个身份按生成前固定的输入哈希顺序选择，均展示Seed42。各臂结构最差10身份、最差Seed的同输入对照保存在 `qualitative/worst_*`，没有因结果较差而删除或重选。真人盲评和全新TEST-B均未运行。','']
    for identity in read(root/'manifests/qualitative_ids.json'):
        lines += [f'![{identity}](cn_ipa_repro_20261022/qualitative/fixed/{identity}_seed42.png)','']
    lines += ['## 溯源与可复算文件','',
        f"冻结代码提交：`{frozen['code_commit']}`；GitHub分支：`codex/cn-ipa-repro-20261022`。",
        '基础SD1.5、Lineart、IPA、ViT-H文件均保存SHA256，并与适用的官方LFS文件指纹核对。所有checkpoint、输入、输出与最终选择的哈希可查。','',
        '- [逐图评分](cn_ipa_repro_20261022/metrics/per_image.csv)',
        '- [逐身份评分](cn_ipa_repro_20261022/metrics/per_identity.csv)',
        '- [配对区间](cn_ipa_repro_20261022/metrics/paired_ci.csv)',
        '- [极端身份](cn_ipa_repro_20261022/metrics/failure_ids.csv)',
        '- [最终冻结配置](cn_ipa_repro_20261022/selection/frozen_final_choice.json)',
        '- [训练与冻结审计](cn_ipa_repro_20261022/audit/training_runs.json)',
        '- [历史输出复算核验](cn_ipa_repro_20261022/audit/historical_reuse.json)','',
        '方法来源：[ControlNet论文](https://arxiv.org/abs/2302.05543)、[IP-Adapter论文](https://arxiv.org/abs/2308.06721)、[Diffusers联合条件文档](https://huggingface.co/docs/diffusers/en/using-diffusers/ip_adapter)。本任务改造和真实训练预算以代码及审计为准。','']
    path=docs/'ControlNet_IPAdapter_对比M12_实验结果_20261022.md'
    path.write_text('\n'.join(lines),encoding='utf-8')
    print(path)


if __name__=='__main__':main()
