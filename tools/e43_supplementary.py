"""CLIP只作副指标；按PNG哈希保存旁表，不改写已完成的采集记录。"""
import argparse
import numpy as np
import torch
from tools.e43_protocol import *


def supplement(phase,device):
    from eval.metrics import extract_clip_image_features
    rows=read(OUT/'split_sources.json')[phase];pairs=[]
    for row in rows:
        for seed in ([42] if phase in ['pilot','train'] else [42,123]):
            folder=OUT/phase/'samples'/row['id']/('seed%d'%seed)
            record=read(folder/'complete.json')
            extra=folder/'b1_complete.json'
            outputs=record['outputs']+([read(extra)['output']] if extra.exists() else [])
            for value in outputs:
                assert sha(value['output_png'])==value['output_png_sha256']
                pairs.append((value,row))
    paths=sorted({p for v,r in pairs for p in [v['output_png'],r['gt'],r['reference']]})
    assert all(Path(p).exists() for p in paths)
    model=Path('models/clip');assert model.is_dir()
    weights={str(p):sha(p) for p in sorted(model.rglob('*')) if p.is_file()}
    features=extract_clip_image_features(paths,batch_size=8,device=device,model_name=str(model))
    assert len(features)==len(paths) and np.isfinite(features).all()
    table=dict(zip(paths,features));values={}
    for v,r in pairs:
        g=table[v['output_png']]
        values[v['output_png_sha256']]=dict(CLIP_real=float(g@table[r['gt']]),CLIP_texture=float(g@table[r['reference']]))
    write(OUT/phase/'supplementary_metrics.json',dict(values=values,rows=len(pairs),unique_images=len(paths),
        device=device,model_files_sha256=weights,GT_used_only_for_evaluation=True,code_commit=commit()))
    print('E43 CLIP',phase,len(pairs),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True,choices=['pilot','train','validation','holdout'])
    p.add_argument('--device',default='cpu');a=p.parse_args();torch.set_num_threads(4);supplement(a.phase,a.device)
