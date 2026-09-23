"""冻结 BF 的来源配比小对照；仅重放 fused，不训练、不生成。"""
import argparse
import json
from pathlib import Path
import numpy as np
from tools.e14_pattern_probe import write_json


VARIANTS = {'baseline': (1., 1.), 'cnn1_075': (1., .75),
            'clip_2': (2., 1.), 'clip_2_cnn1_075': (2., .75)}


def source_bias(length, queries, clip_factor, cnn_factor, device):
    import torch
    if length <= 256 or min(clip_factor, cnn_factor) <= 0:
        raise ValueError('需要 CLIP + 四组64 tokens，倍率必须为正')
    bias = torch.zeros(queries, length, device=device)
    clip_count = length-256
    bias[:, :clip_count] = np.log(clip_factor)
    bias[:, clip_count:clip_count+64] = np.log(cnn_factor)
    return bias


def run(args):
    import torch
    from models.bf_texture_module import BFTextureConditioner
    from tools.e14_resampler_attention import relative
    root, out = Path(args.features), Path(args.output)
    index = json.loads((root/'index.json').read_text(encoding='utf-8'))
    if not index.get('complete'):
        raise ValueError('特征未完成')
    state = torch.load(args.checkpoint, map_location='cpu')['bf_texture_conditioner']
    bf = BFTextureConditioner(clip_embeddings_dim=state['token_source_proj.0.1.weight'].shape[1])
    bf.load_state_dict(state, strict=True)
    bf.to(args.device).requires_grad_(False)
    # 缓存来自 FP16 权重，先量化一次；FP32机制实验使用相同量化后的权重。
    bf.half()
    out.mkdir(parents=True, exist_ok=False)
    for variant in VARIANTS:
        (out/variant/'features').mkdir(parents=True)
        write_json(out/variant/'index.json',dict(rows=index['rows'],complete=False,
            config=dict(source_index=str(root/'index.json'),variant=variant,dtype='fp32')))
    report = dict(complete=False,config=vars(args),variants=VARIANTS,records=[],limits=[
        'logit加log倍率后重新softmax；倍率不是最终attention占比。',
        '旋转响应增大不是正确方向或生成改善，须结合分组方向读出。',
        '相对幅度和单位范数差异都报告，避免把输出缩放当成收益。',
        '仅条纹原图/旋转，不能验证所有图案类别；基线对照均为FP32。'])
    for i,row in enumerate(index['rows']):
        with np.load(root/row['feature_file']) as f:
            fused=torch.from_numpy(f['fused__flatten'].copy()).reshape(1,-1,768).to(args.device)
            cached=torch.from_numpy(f['resampler_raw__flatten'].copy()).reshape(1,16,768).to(args.device)
        with torch.inference_mode():
            bf.half()
            half,_=bf.resampler(bf.resampler_queries,fused.half(),fused.half(),need_weights=False)
            error=relative(half,cached)
            if error>1e-3:
                raise ValueError('重放未通过：%s error=%g'%(row['sample_id'],error))
            bf.float();x=fused.float();q=bf.resampler_queries
            original,_=bf.resampler(q,x,x,need_weights=False)
            original_final=bf.token_norm(original+bf.token_mlp(original))
            for variant,(clip_factor,cnn_factor) in VARIANTS.items():
                bias=source_bias(x.shape[1],q.shape[1],clip_factor,cnn_factor,args.device)
                raw,attention=bf.resampler(q,x,x,attn_mask=bias,need_weights=True,average_attn_weights=False)
                pre=raw+bf.token_mlp(raw);final=bf.token_norm(pre)
                if not torch.isfinite(final).all():
                    raise ValueError('非有限输出')
                if variant=='baseline' and relative(raw,original)>1e-4:
                    raise ValueError('零bias前向不一致')
                values={}
                for name,t in [('resampler_raw',raw),('tokens_pre_ln',pre),('tokens_post_ln',final)]:
                    values[name+'__flatten']=t[0].cpu().numpy().reshape(-1)
                    values[name+'__meanstd']=torch.cat([t[0].mean(0),t[0].std(0,unbiased=False)]).cpu().numpy()
                target=out/variant/row['feature_file'];target.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(target,**values)
                count=x.shape[1]-256
                bounds=[0,count,count+64,count+128,count+192,count+256]
                masses={name:attention[:,:,:,a:b].sum(-1).mean((0,2)).cpu().tolist()
                        for name,a,b in zip(['clip','cnn1','cnn2','cnn3','cnn4'],bounds[:-1],bounds[1:])}
                report['records'].append(dict(sample_id=row['sample_id'],variant=variant,
                    replay_error=error,fp32_vs_fp16=relative(original,half),
                    raw_relative_change=relative(raw,original),final_relative_change=relative(final,original_final),
                    raw_norm_ratio=float(raw.norm()/original.norm()),final_norm_ratio=float(final.norm()/original_final.norm()),
                    attention_mass_per_head=masses))
        write_json(out/'balance_report.json',report)
        print(row['sample_id'],flush=True)
    for variant in VARIANTS:
        p=out/variant/'index.json';data=json.loads(p.read_text(encoding='utf-8'));data['complete']=True;write_json(p,data)
    report['complete']=True
    write_json(out/'balance_report.json',report)
    summarize(out)


def summarize(root):
    root=Path(root);summary={}
    for variant in VARIANTS:
        folder=root/variant
        index=json.loads((folder/'index.json').read_text(encoding='utf-8'))
        if not index['complete']:raise ValueError('特征未完成')
        grouped={}
        for row in index['rows']:grouped.setdefault(row['source_group'],{})[row['variant']]=row
        pairs=[]
        for group,rows in grouped.items():
            with np.load(folder/rows['original']['feature_file']) as a,np.load(folder/rows['rot90']['feature_file']) as b:
                entry={'source_group':group}
                for key in ['resampler_raw__flatten','tokens_post_ln__flatten']:
                    x,y=a[key].astype(float),b[key].astype(float)
                    entry[key]=dict(relative_l2=float(np.linalg.norm(x-y)/np.linalg.norm(x)),
                        unit_norm_distance=float(np.linalg.norm(x/np.linalg.norm(x)-y/np.linalg.norm(y))))
                pairs.append(entry)
        summary[variant]=pairs
    write_json(root/'rotation_summary.json',summary)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['features','checkpoint','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cuda:0')
    run(p.parse_args())
