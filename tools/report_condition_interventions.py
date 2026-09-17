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


def summarize(run_dir, source_run, matched_budget_root=None):
    root, source = Path(run_dir), Path(source_run)
    out = root / 'report'
    out.mkdir(parents=True, exist_ok=True)
    modes = ('baseline', 'boundary', 'weaken_texture_matched', 'strengthen_sketch_matched') if matched_budget_root else MODES
    sample_ids = (5, 18, 23) if matched_budget_root else (2, 5, 14, 18, 22, 23)
    result = {'samples': {}, 'metric_delta_definition': 'variant minus baseline',
              'note': '有意选择的诊断集；指标与图像需共同判断，不估计总体效果。'}
    if matched_budget_root:
        result.update(matched_budget_root=str(matched_budget_root), rms_relative_tolerance=0.05)
    for sid in sample_ids:
        name = f'sample_{sid:06d}'
        records, images, traces = {}, {}, {}
        old = Image.open(only(source / name / 'on', 'generated_*.png')).convert('RGB')
        for mode in modes:
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
        budget = None
        if matched_budget_root:
            budget_folder = Path(matched_budget_root) / name / 'boundary' / 'e5'
            budget = json.loads(only(budget_folder, 'intervention.json').read_text(encoding='utf-8'))
            budget_image = Image.open(only(budget_folder / 'token', 'generated_*.png')).convert('RGB')
            if not np.array_equal(np.array(images['boundary']), np.array(budget_image)):
                raise ValueError(f'{name} boundary 未复现参考预算实验')
            if (budget['metadata'] != traces['boundary']['metadata'] or
                    budget['trigger_steps'] != traces['boundary']['trigger_steps'] or
                    budget['records'] != traces['boundary']['records']):
                raise ValueError(f'{name} boundary 修正轨迹未复现')
        for mode in modes:
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
            if mode.endswith('_matched'):
                expected = [r['correction_rms'] for r in budget['records']]
                if trace['budget_rms'] != expected:
                    raise ValueError(f'{name}/{mode} 固定预算不一致')
                errors, scales = [], []
                for r, target in zip(trace['records'], expected):
                    if r['target_rms'] != target or not np.isfinite(r['correction_rms']):
                        raise ValueError(f'{name}/{mode} 目标幅度或实际幅度无效')
                    error = abs(r['correction_rms']-target)/target if target else 0.
                    if (target == 0 and (r['applied'] or r['correction_rms'] != 0)) or error > 0.05:
                        raise ValueError(f'{name}/{mode} 第 {r["step_index"]} 步幅度不匹配：{error:.2%}')
                    errors.append(error)
                    if target:
                        scales.append(r['match_scale'])
                sample_result[mode]['amplitude_match'] = dict(max_relative_error=max(errors),
                    mean_relative_error=float(np.mean(errors)), active_scale_min=min(scales) if scales else None,
                    active_scale_max=max(scales) if scales else None)
        result['samples'][name] = sample_result
        # 仅为评审拼图缩小显示，指标仍来自原尺寸输出。
        tile_w = 256
        tile_h = round(old.height * tile_w / old.width)
        canvas = Image.new('RGB', (tile_w * len(modes), tile_h + 28), 'white')
        draw = ImageDraw.Draw(canvas)
        for i, mode in enumerate(modes):
            canvas.paste(images[mode].resize((tile_w, tile_h)), (tile_w * i, 28))
            draw.text((tile_w * i + 4, 7), mode, fill='black')
        canvas.save(out / f'{name}.png')
    result['status'] = 'pass'
    (out / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(f'{len(sample_ids)} 样本 {len(modes)} 组核对通过：{out}')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--matched-budget-root', default=None)
    args = parser.parse_args()
    summarize(args.run_dir, args.source_run, args.matched_budget_root)
