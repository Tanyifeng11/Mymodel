"""重放缓存 fused，检查 attention 来源分配、可加贡献及来源屏蔽。"""
import argparse
import json
from pathlib import Path
import numpy as np
from tools.e14_pattern_probe import write_json


def relative(a,b):
    return float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12))


def contributions(module, x, weights):
    import torch
    import torch.nn.functional as F
    dim=module.embed_dim
    value=F.linear(x,module.in_proj_weight[2*dim:],module.in_proj_bias[2*dim:])
    value=value.reshape(1,-1,module.num_heads,dim//module.num_heads).transpose(1,2)
    result={}
    sizes=[x.shape[1]-256,64,64,64,64]
    start=0
    for name,size in zip(['clip','cnn1','cnn2','cnn3','cnn4'],sizes):
        part=(weights[:,:,:,start:start+size]@value[:,:,start:start+size])
        part=part.transpose(1,2).reshape(1,-1,dim)
        result[name]=F.linear(part,module.out_proj.weight,None)
        start+=size
    return result


def run(args):
    import torch
    from models.bf_texture_module import BFTextureConditioner
    root=Path(args.features)
    index=json.loads((root/'index.json').read_text(encoding='utf-8'))
    if not index.get('complete'):
        raise ValueError('缓存未完成')
    state=torch.load(args.checkpoint,map_location='cpu')['bf_texture_conditioner']
    bf=BFTextureConditioner(clip_embeddings_dim=state['token_source_proj.0.1.weight'].shape[1])
    bf.load_state_dict(state,strict=True)
    bf.to(args.device).requires_grad_(False)
    # 保留训练模式实现路径，但 MHA dropout 默认为零。
    report=dict(complete=False,config=vars(args),records=[],limits=[
        '来源attention质量按token数校正；权重不是因果贡献。',
        '固定attention的贡献可加，但范数不能相加作为贡献百分比。',
        '屏蔽来源会重新归一化attention，是分布外机制干预，不是生成质量评价。',
        'FP32用于数值审计；先检查FP16缓存重放误差及FP32与FP16差异。'])
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    for row in index['rows']:
        with np.load(root/row['feature_file']) as f:
            x=torch.from_numpy(f['fused__flatten'].copy()).reshape(1,-1,768).to(args.device)
            cached=torch.from_numpy(f['resampler_raw__flatten'].copy()).reshape(1,16,768).to(args.device)
        with torch.inference_mode():
            bf.half()
            half,_=bf.resampler(bf.resampler_queries,x.half(),x.half(),need_weights=False)
            bf.float()
            x=x.float();q=bf.resampler_queries
            y,weights=bf.resampler(q,x,x,need_weights=True,average_attn_weights=False)
            parts=contributions(bf.resampler,x,weights)
            reconstructed=sum(parts.values())+bf.resampler.out_proj.bias
            record=dict(sample_id=row['sample_id'],source_group=row['source_group'],
                variant=row.get('variant'),fp16_vs_cache=relative(half,cached),
                fp32_vs_fp16=relative(y,half),decomposition_error=relative(reconstructed,y),sources={})
            sizes=[x.shape[1]-256,64,64,64,64];start=0
            for (name,part),size in zip(parts.items(),sizes):
                mass=weights[:,:,:,start:start+size].sum(-1)
                mask=torch.zeros((1,x.shape[1]),dtype=torch.bool,device=args.device)
                mask[:,start:start+size]=True
                ablated,_=bf.resampler(q,x,x,key_padding_mask=mask,need_weights=False)
                record['sources'][name]=dict(tokens=size,attention_mass_per_head=mass.mean((0,2)).cpu().tolist(),
                    attention_mass_mean=float(mass.mean()),
                    enrichment_over_uniform=float(mass.mean())/(size/x.shape[1]),
                    projected_contribution_norm=float(part.norm()),
                    projected_contribution_relative_norm=float(part.norm()/y.norm()),
                    source_removed_relative_change=relative(ablated,y))
                start+=size
            entropy=-(weights*weights.clamp_min(1e-30).log()).sum(-1)/np.log(x.shape[1])
            record['normalized_entropy_per_head']=entropy.mean((0,2)).cpu().tolist()
            record['query_attention_variation_rms']=float((weights-weights.mean(2,keepdim=True)).square().mean().sqrt())
            record['replay_acceptable']=record['fp16_vs_cache']<1e-3 and record['decomposition_error']<1e-5
            # 保留每个query/head的权重，供下载后分析原图/旋转差异。
            filename='%04d.npz'%len(report['records'])
            np.savez_compressed(out/filename,attention=weights.cpu().numpy(),
                output=y.cpu().numpy(),**{name:part.cpu().numpy() for name,part in parts.items()})
            record['arrays']=filename
        report['records'].append(record)
        write_json(out/'attention_report.json',report)
        print(row['sample_id'],record['fp16_vs_cache'],flush=True)
    report['complete']=True
    report['all_replays_acceptable']=all(r['replay_acceptable'] for r in report['records'])
    write_json(out/'attention_report.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['features','checkpoint','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cuda:0')
    run(p.parse_args())
