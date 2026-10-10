"""下载并核验完成后在用户本地运行；不在服务器生成 docs 报告。"""
import argparse
import numpy as np
from tools.e38_protocol import read, write, sha
from tools.e38_run import association, boot_mean
from scipy.stats import rankdata


def paired_rho_gain(x, baseline, y):
    a,b,c=map(lambda v:np.asarray(v,dtype=float),(x,baseline,y))
    if len(a)<3 or any(np.ptp(v)==0 for v in [a,b,c]):return dict(n=len(a),difference=None,ci95=None,valid_bootstrap=0)
    def rho(u,v):
        u=rankdata(u,axis=1);v=rankdata(v,axis=1)
        u-=u.mean(1,keepdims=True);v-=v.mean(1,keepdims=True)
        den=np.sqrt((u*u).sum(1)*(v*v).sum(1))
        return np.divide((u*v).sum(1),den,out=np.full(len(u),np.nan),where=den>0)
    rng=np.random.default_rng(38042);ii=rng.integers(0,len(a),(10000,len(a)))
    values=rho(a[ii],c[ii])-rho(b[ii],c[ii]);values=values[np.isfinite(values)]
    return dict(n=len(a),difference=float(rho(a[None],c[None])[0]-rho(b[None],c[None])[0]),
                ci95=np.quantile(values,[.025,.975]).tolist() if len(values) else None,valid_bootstrap=len(values))


def visual_analysis(review):
    # 调用方必须先保存 ratings_locked.json，并在核对 key 前计算其 SHA。
    ratings_path=review/'blind/ratings_locked.json'
    if not ratings_path.exists():return dict(status='not_run',reason='ratings not available')
    ratings=read(ratings_path);keys=read(review/'blind/key_do_not_read_before_ratings.json')
    assert len(ratings)==len(keys)==32
    lookup={v['review_id']:v for v in ratings};deltas={}
    for k in keys:
        row=lookup[k['review_id']];scores={}
        for arm in ['X','Y']:
            assert len(row[arm])==3 and all(v in [0,1,2,3] for v in row[arm])
            scores[k[arm]]=float(np.mean(row[arm]))
        deltas[k['id']]=scores['B1_CONV']-scores['A0_E5_OFF']
    records=read(review/'prediction/records.json');out={}
    for name in ['A_SDF','B_FOURIER','C_GRAPH']:
        rr=[v for v in records if v['candidate']==name and v['delta_distance'] is not None and v['source_mask_valid'] and v['minus_delta_IoU'] is not None]
        x=[v['delta_distance'] for v in rr];b=[v['minus_delta_IoU'] for v in rr];y=[deltas[v['id']] for v in rr]
        gain=paired_rho_gain(x,b,y);r=association(x,y)
        out[name]=dict(fixed_n=32,n=len(rr),candidate_association=r,legacy_sketch_iou_association=association(b,y),paired_rho_gain=gain,
            pass_gate=bool(len(rr)/32>=.8 and r['rho'] is not None and r['rho']>.4 and gain['ci95'] is not None and gain['ci95'][0]>0))
    return dict(status='complete',assessor='single AI; not human study; prior E36/E37 familiarity limits blinding',
                ratings_sha256=sha(ratings_path),mean_delta_defect=boot_mean(list(deltas.values())),candidates=out)


def fmt(value):return 'NA' if value is None else ('%.4f'%value if isinstance(value,(int,float)) else str(value))


def run(local):
    from pathlib import Path
    local=Path(local);review=local/'local_review';docs=Path('F:/fuxian/Mymodel/docs')
    assert (local/'local_verification.json').exists(),'必须先核验下载产物'
    verification=read(local/'local_verification.json')
    assert verification['all_artifacts_match']
    result=read(review/'E38_structure_representation_eval.json');visual=visual_analysis(review)
    result['blinded_visual_review']=visual;result['local_verification']=verification
    for name,rec in result['candidates'].items():
        if visual['status']=='complete':rec['checks']['blinded_real_superiority']=visual['candidates'][name]['pass_gate']
        # C 的语义正确性需显式审核文件；缺失审核不当作已通过。
        if name=='C_GRAPH' and (review/'audit/semantic_review.json').exists():
            rec['checks']['semantic_graph_validated']=read(review/'audit/semantic_review.json')['coverage']>=.8
        rec['go']=all(v is True for v in rec['checks'].values())
        rec['status']='go' if rec['go'] else 'no_go'
    result['go']=any(v['go'] for v in result['candidates'].values())
    result['structure_supervision_training']='not_run';result['novelty_claim']=False
    write(docs/'E38_structure_representation_eval.json',result)
    lines=['# E38 候选结构表示比较','',
        '日期：2026-10-11。固定 E36/E37 dev32、64张完整 DDIM50 原图；训练更新0，新增生成0，GPU消耗0。',
        '数据集、Caption、E5/B1检查点不修改。dev32 已用于前轮实验，本轮属于探索性验证，不能当作独立泛化证据。','',
        '## 外观稳定性与结构敏感性','',
        'oracle 表示固定 GT 前景提取结果与已知几何变换下的 mask；它不是人工真值分割，不能掩盖原始 GT 提取误差。RGB 表示每次从变化后的 RGB 独立重提取轮廓。',
        '覆盖率按完整32身份统计；每身份六种外观扰动，衣摆／轮廓各两级，袖型扰动只对原 Caption 声明的适用上装运行。NA 不补零。', '',
        '|流程|候选|有效/32|严格分离率|身份平均AUROC [95%CI]|相对IoU的AUROC增量 [95%CI]|门槛|',
        '|---|---|---:|---:|---|---|---|']
    for pipeline in ['oracle','rgb']:
        for name in result['protocol']['candidates']+['one_minus_IoU']:
            v=result['perturbations'][pipeline][name];a=v['auc'];g=v.get('auc_gain_vs_iou',{})
            lines.append('|%s|%s|%d/32|%s|%s %s|%s %s|%s|'%(pipeline,name,v['valid_n'],fmt(v['separation_rate']),fmt(a['mean']),a['ci95'],fmt(g.get('mean')),g.get('ci95'),v['pass_gate']))
    lines+=['','## 与最终 RGB 变化的对应关系','',
        '距离定义为各自生成图到 GT 轮廓的表示距离，Δ为 B1−E5。结构目标使用原 Sketch 指标的 −ΔIoU、−ΔEdgeF1；方向事前固定，不能用绝对值把反向关系算成通过。LPIPS 单列诊断。',
        '每项按身份 bootstrap 10000次，逐项分母及未定义重采样次数见 JSON。门槛为两项结构相关均ρ>0.4且有效覆盖≥80%。','',
        '|候选|目标|n/32|ρ|95%CI|', '|---|---|---:|---:|---|']
    for name,v in result['prediction'].items():
        for target,a in v['comparisons'].items():lines.append('|%s|%s|%d/32|%s|%s|'%(name,target,a['n'],fmt(a['rho']),a['ci95']))
    lines+=['','## 独立结构判断与候选 C','',
        '已知扰动和逐图评分是区别于现有 IoU 的参照。只与 IoU／EdgeF1 相关，不足以证明更强预测能力。',
        '盲评状态：`%s`。'%visual['status'],
        '逐图评分维度为外轮廓、部件／长度比例、内部构造线；每维0无缺陷、1轻微、2明确、3严重。评分者为单个AI，且曾接触前轮案例；不能称为独立人工用户研究。',
        'C 为上装轮廓地标与显式连接的最小实现。裤装、未知类别、领口或袖身无法分离时保留缺失；几何检查通过也不自动等于语义正确。未达到总体覆盖与语义审核门槛，不支持“Garment Graph 已验证”。', '',
        '## 可复核产物','',
        '[冻结协议](%s)；[扰动逐条结果](%s)；[真实RGB逐身份结果](%s)；[输入与权重哈希](%s)；[资源账本](%s)。'%tuple(str(review/p).replace('\\','/') for p in ['protocol_locked.json','perturbations/records.json','prediction/records.json','audit/input_hashes.json','resource_ledger.json']),
        '[本地下载核验](%s)。'%str(local/'local_verification.json').replace('\\','/'), '',
        '局限：SDF/Fourier/C 依赖可提取外轮廓，无法覆盖所有内部领口、门襟、缝线；合成形变不是实际设计编辑；GT 自动 mask 仍可能错误；原边缘指标也受纹理影响。硬 mask→轮廓／EDT→距离不是端到端可微训练损失。']
    (docs/'E38_candidate_comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    decision=['# E38 Go / No-Go','',
        '**%s**。'%('存在通过候选，仍需独立数据确认' if result['go'] else 'No-Go：当前候选没有同时通过冻结门槛'),'',
        '|候选|检查项|结果|','|---|---|---|']
    for name,rec in result['candidates'].items():
        for check,ok in rec['checks'].items():decision.append('|%s|%s|%s|'%(name,check,'not_run / 未确认' if ok is None else ('pass' if ok else 'fail')))
    decision+=['',
        '本轮没有训练结构监督网络、增加生成种子、使用 confirm96／validation500 或调门槛补救。未运行步骤保持 not_run/null。',
        'No-Go 限定为本次提取器、候选定义、扰动及已复用 dev32 上没有充分支持；不代表 SDF、轮廓或语义图在所有任务上无效。',
        '研究边界：prior-art 已覆盖距离场边界监督、服装语义对齐、纸样结构扩散和结构引导纹理生成。当前不支持原创方法成立，也不支持把 SDF loss／contour loss／graph loss 本身作为创新。', '',
        '[候选比较](F:/fuxian/Mymodel/docs/E38_candidate_comparison.md)；[完整JSON](F:/fuxian/Mymodel/docs/E38_structure_representation_eval.json)；[文献矩阵](F:/fuxian/Mymodel/docs/E38_prior_art_matrix.md)。', '',
        '报告仅本地 docs 保存，未上传服务器。']
    (docs/'E38_go_no_go.md').write_text('\n'.join(decision)+'\n',encoding='utf-8')
    write(local/'final_decision.json',dict(go=result['go'],candidates=result['candidates'],training='not_run',novelty_claim=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('local_dir');run(parser.parse_args().local_dir)
