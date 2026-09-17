"""核对固定触发干预，输出逐样本指标差值和五组对照拼图。"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

MODES = ('baseline', 'boundary', 'weaken_texture', 'strengthen_sketch', 'global')
METRICS = ('struct_edge_f1', 'struct_iou', 'clip_i_texture', 'tpf_patch_sim',
           'tpf_gram_l1', 'tcf_lab_delta', 'leak_colored_frac')


def only(root, pattern):
    matches = list(root.rglob(pattern))
    if len(matches) != 1:
        raise ValueError(f'需要唯一文件：{root}/{pattern}，实际 {len(matches)}')
    return matches[0]


def summarize(run_dir, source_run):
    root, source = Path(run_dir), Path(source_run)
    out = root / 'report'
    out.mkdir(parents=True, exist_ok=True)
    result = {'samples': {}, 'metric_delta_definition': 'variant minus baseline',
              'note': '六张为有意选择的诊断集；指标与图像需共同判断，不估计总体效果。'}
    for sid in (2, 5, 14, 18, 22, 23):
        name = f'sample_{sid:06d}'
        records, images, traces = {}, {}, {}
        old = Image.open(only(source / name / 'on', 'generated_*.png')).convert('RGB')
        for mode in MODES:
            folder = root / name / mode / 'e5'
            rows = json.loads((folder / 'metrics_per_sample.json').read_text(encoding='utf-8'))
            if len(rows) != 1 or int(rows[0]['sample_id']) != sid:
                raise ValueError(f'{name}/{mode} 样本不匹配')
            records[mode] = rows[0]
            images[mode] = Image.open(only(folder / 'token', 'generated_*.png')).convert('RGB')
            trace = json.loads(only(folder, 'intervention.json').read_text(encoding='utf-8'))
            if trace['mode'] != mode or [r['step_index'] for r in trace['records']] != list(range(50)):
                raise ValueError(f'{name}/{mode} 日志不完整')
            if any(r['applied'] and r['step_index'] not in trace['trigger_steps'] for r in trace['records']):
                raise ValueError(f'{name}/{mode} 在计划外施加干预')
            traces[mode] = trace
        baseline = np.array(images['baseline'])
        if not np.array_equal(baseline, np.array(old)):
            raise ValueError(f'{name} baseline 与原实验不一致，停止解释干预效果')
        sample_result = {}
        for mode in MODES:
            trace = traces[mode]
            if trace['trigger_steps'] != traces['baseline']['trigger_steps'] or trace['metadata'] != traces['baseline']['metadata']:
                raise ValueError(f'{name}/{mode} 触发计划或条件不一致')
            for key in ('generation_seed', 'prompt', 'sketch_path', 'texture_path'):
                if records[mode][key] != records['baseline'][key]:
                    raise ValueError(f'{name}/{mode} 输入不同：{key}')
            applied = [r['step_index'] for r in trace['records'] if r['applied']]
            identical = np.array_equal(baseline, np.array(images[mode]))
            if not applied and not identical:
                raise ValueError(f'{name}/{mode} 未施加干预但图像改变')
            sample_result[mode] = dict(trigger_steps=trace['trigger_steps'], applied_steps=applied,
                pixel_identical_to_baseline=identical,
                metrics={k: records[mode].get(k) for k in METRICS},
                metric_delta={k: records[mode][k] - records['baseline'][k] for k in METRICS
                              if isinstance(records[mode].get(k), (int, float)) and isinstance(records['baseline'].get(k), (int, float))},
                correction_rms_sum=sum(r['correction_rms'] for r in trace['records']))
        result['samples'][name] = sample_result
        # 仅为评审拼图缩小显示，指标仍来自原尺寸输出。
        tile_w = 256
        tile_h = round(old.height * tile_w / old.width)
        canvas = Image.new('RGB', (tile_w * len(MODES), tile_h + 28), 'white')
        draw = ImageDraw.Draw(canvas)
        for i, mode in enumerate(MODES):
            canvas.paste(images[mode].resize((tile_w, tile_h)), (tile_w * i, 28))
            draw.text((tile_w * i + 4, 7), mode, fill='black')
        canvas.save(out / f'{name}.png')
    result['status'] = 'pass'
    (out / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(f'六样本五组核对通过：{out}')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--source-run', required=True)
    args = parser.parse_args()
    summarize(args.run_dir, args.source_run)
