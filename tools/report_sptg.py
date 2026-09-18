"""SPTG 四组配对核验、逐样本折中指标与拼图。"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

def only(root, pattern):
    matches=list(root.rglob(pattern))
    if len(matches)!=1:
        raise ValueError(f'需要唯一文件：{root}/{pattern}')
    return matches[0]

MODES=('baseline','strong_texture','sptg_rho05','sptg_rho10')
METRICS=('struct_edge_f1','struct_iou','clip_i_texture','tpf_patch_sim','tpf_gram_l1',
         'tcf_lab_delta','leak_colored_frac','leak_mean_saturation','leak_value_shift','leak_edge_density')


def report(root, source_run):
    root,source=Path(root),Path(source_run)
    out=root/'report'
    out.mkdir(parents=True,exist_ok=True)
    result={'samples':{},'note':'六张诊断样本；纹理收益需多指标和目视判断，不能只看 CLIP。'}
    for sid in (2,5,14,18,22,23):
        name=f'sample_{sid:06d}'
        rows,images,traces={},{},{}
        reference=json.loads(only(source/name/'on/e5/token','probe.json').read_text(encoding='utf-8'))
        old=Image.open(only(source/name/'off/e5/token','generated_*.png')).convert('RGB')
        for mode in MODES:
            run=root/name/mode/'e5'
            metrics=json.loads((run/'metrics_per_sample.json').read_text(encoding='utf-8'))
            if len(metrics)!=1 or int(metrics[0]['sample_id'])!=sid:
                raise ValueError('样本不匹配')
            rows[mode]=metrics[0]
            images[mode]=Image.open(only(run/'token','generated_*.png')).convert('RGB')
            trace=json.loads(only(run/'token','sptg.json').read_text(encoding='utf-8'))
            expected={'baseline':(1.,0.),'strong_texture':(1.2,0.),'sptg_rho05':(1.2,.5),'sptg_rho10':(1.2,1.)}[mode]
            if (trace['mode']!=mode or (trace['texture_weight'],trace['rho'])!=expected or
                    trace['metadata']!=reference['metadata'] or
                    [r['step_index'] for r in trace['records']]!=list(range(50))):
                raise ValueError(f'{name}/{mode} 参数或条件不一致')
            if any(not np.isfinite(r['actual_delta_rms']) or not np.isfinite(r['actual_projection_delta_rms']) for r in trace['records']):
                raise ValueError('修正量非有限')
            traces[mode]=trace
        if not np.array_equal(np.asarray(images['baseline']),np.asarray(old)):
            raise ValueError(f'{name} baseline 未复现 E5')
        if any(r['applied'] or r['actual_delta_rms']!=0 for r in traces['baseline']['records']):
            raise ValueError('baseline 不应施加干预')
        if any(r['actual_projection_delta_rms']!=0 for r in traces['strong_texture']['records']):
            raise ValueError('strong texture 不应投影')
        sample={}
        for mode in MODES:
            for key in ('generation_seed','prompt','sketch_path','texture_path'):
                if rows[mode][key]!=rows['baseline'][key]:
                    raise ValueError('输入条件不一致')
            sample[mode]=dict(metrics={k:rows[mode].get(k) for k in METRICS},
                delta_vs_baseline={k:rows[mode][k]-rows['baseline'][k] for k in METRICS if isinstance(rows[mode].get(k),(int,float)) and isinstance(rows['baseline'].get(k),(int,float))},
                delta_vs_strong={k:rows[mode][k]-rows['strong_texture'][k] for k in METRICS if isinstance(rows[mode].get(k),(int,float)) and isinstance(rows['strong_texture'].get(k),(int,float))},
                projection_steps=[r['step_index'] for r in traces[mode]['records'] if r['actual_projection_delta_rms']>0])
        result['samples'][name]=sample
        width=256
        height=round(old.height*width/old.width)
        canvas=Image.new('RGB',(width*4,height+26),'white')
        draw=ImageDraw.Draw(canvas)
        for col,mode in enumerate(MODES):
            canvas.paste(images[mode].resize((width,height)),(col*width,26))
            draw.text((col*width+4,5),mode,fill='black')
        canvas.save(out/f'{name}.png')
    result['status']='pass'
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(out)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--source-run',required=True)
    args=parser.parse_args()
    report(args.run_dir,args.source_run)
