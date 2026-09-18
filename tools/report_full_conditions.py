"""核验 rho=0 图像一致性、分解完整性及关闭审计，汇总冲突时间分布。"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def only(root, pattern):
    matches = list(root.rglob(pattern))
    if len(matches) != 1:
        raise ValueError(f'需要唯一文件：{root}/{pattern}')
    return matches[0]


def report(root, local_response_root=None):
    root = Path(root)
    results = {}
    for sid in (2, 5, 14, 18, 22, 23):
        folder = root/f'sample_{sid:06d}'
        off, on = [folder/b/'e5' for b in ('off', 'on')]
        image_off, image_on = [np.asarray(Image.open(only(b/'token', 'generated_*.png')).convert('RGB')) for b in (off, on)]
        if not np.array_equal(image_off, image_on):
            raise ValueError(f'{sid} rho=0 未复现 E5')
        metrics = [json.loads((b/'metrics_per_sample.json').read_text(encoding='utf-8')) for b in (off,on)]
        if any(len(rows) != 1 or int(rows[0]['sample_id']) != sid for rows in metrics):
            raise ValueError('样本 ID 不匹配')
        for key in ('generation_seed','prompt','sketch_path','texture_path'):
            if metrics[0][0][key] != metrics[1][0][key]:
                raise ValueError(f'{sid} 条件不匹配：{key}')
        path = only(on/'token', 'probe.json')
        data = json.loads(path.read_text(encoding='utf-8'))
        records = data['records']
        local_path = None
        comparisons = []
        if local_response_root:
            local_path = only(Path(local_response_root)/f'sample_{sid:06d}'/'on/e5/token', 'probe.json')
            local_data = json.loads(local_path.read_text(encoding='utf-8'))
            if local_data['metadata'] != data['metadata']:
                raise ValueError('与原局部响应实验的输入条件不一致')
        if [r['step_index'] for r in records] != list(range(50)):
            raise ValueError('完整分解必须观测 50 步')
        for r in records:
            if r['step_index'] in (0,25,49) and len(r['shutdown_audits']) != 2:
                raise ValueError('缺少关闭审计')
            if any(v > max(r['repeat_max_abs']*10,1e-6) for v in r['shutdown_audits'].values()):
                raise ValueError('纹理关闭审计失败')
            with np.load(path.parent/f"step_{r['step_index']:02d}.npz") as arrays:
                if not all(np.isfinite(arrays[k]).all() for k in arrays.files):
                    raise ValueError('分解数据存在非有限值')
                if not np.allclose(arrays['sketch_guidance'],arrays['epsilon_s'].astype('float32')-arrays['epsilon_0'].astype('float32'),rtol=0,atol=1e-7):
                    raise ValueError('sketch 差分不一致')
                if not np.allclose(arrays['texture_guidance'],arrays['epsilon_st'].astype('float32')-arrays['epsilon_s'].astype('float32'),rtol=0,atol=1e-7):
                    raise ValueError('texture 差分不一致')
                if local_path:
                    with np.load(local_path.parent/f"step_{r['step_index']:02d}.npz") as previous:
                        def cosine(a, b, w):
                            a,b,w=a.astype('float64'),b.astype('float64'),w.astype('float64')
                            norm=np.sqrt((a*a*w).sum()*(b*b*w).sum())
                            return float((a*b*w).sum()/norm) if norm else None
                        comparisons.append(dict(step=r['step_index'], regions={name: dict(
                            full_vs_local_sketch_cosine=cosine(arrays['sketch_guidance'],previous['sketch_delta_h0.2'],arrays['weight_'+name]),
                            full_vs_local_texture_cosine=cosine(arrays['texture_guidance'],previous['texture_delta_h0.2'],arrays['weight_'+name]),
                            local_response_cosine=cosine(previous['sketch_delta_h0.2'],previous['texture_delta_h0.2'],arrays['weight_'+name]))
                            for name in ('global','inner','boundary','background')}))
        results[f'{sid:06d}'] = dict(pixel_identical=True,
            comparison_to_20percent_response=comparisons,
            repeat_max_abs=max(r['repeat_max_abs'] for r in records),
            reconstruction_fp32_max_abs=max(r['reconstruction_fp32_max_abs'] for r in records),
            regions={name: dict(negative_steps=[r['step_index'] for r in records
                if r['regions'][name]['above_repeat_floor'] and r['regions'][name]['opposing']],
                timeline=[dict(step=r['step_index'], **r['regions'][name]) for r in records])
                for name in ('global','inner','boundary','background')})
    out = root/'report'
    out.mkdir(parents=True, exist_ok=True)
    result = dict(status='pass', samples=results,
                  note='pass 表示 rho=0 与关闭审计通过；不证明关闭条件分支在分布内或冲突有害。')
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(out)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--local-response-root')
    args=parser.parse_args()
    report(args.run_dir,args.local_response_root)
