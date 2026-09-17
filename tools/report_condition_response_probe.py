"""核对探针开关原图一致性并汇总响应；不将负夹角自动判为结构损害。"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


def response_timeline(records):
    """仅合并连续且两档都反向的已观测步；稀疏探针不插值。"""
    timeline = {}
    for region in ('global', 'inner', 'boundary', 'background'):
        events = []
        eligible = []
        for record in records:
            responses = record['regions'][region]['responses']
            if all(s['above_repeat_floor'] and s['cosine'] is not None for s in responses.values()):
                eligible.append(record)
                if all(s['cosine'] < 0 for s in responses.values()):
                    events.append(record['step_index'])
        intervals = []
        for step in events:
            if intervals and step == intervals[-1][1] + 1:
                intervals[-1][1] = step
            else:
                intervals.append([step, step])
        minimum = min(eligible, key=lambda r: r['regions'][region]['responses']['0.2']['cosine']) if eligible else None
        timeline[region] = {
            'observed_count': len(records), 'above_floor_count': len(eligible),
            'both_negative_steps': events, 'consecutive_observed_intervals': intervals,
            'minimum_cosine_h02': minimum['regions'][region]['responses']['0.2']['cosine'] if minimum else None,
            'minimum_step_h02': minimum['step_index'] if minimum else None,
        }
    return timeline


def report(baseline_dir, probe_dir, expected_count, output_dir, expected_steps=None):
    expected_steps = [0, 5, 15, 25, 49] if expected_steps is None else expected_steps
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    def rows(root):
        data = json.loads((Path(root) / 'metrics_per_sample.json').read_text(encoding='utf-8'))
        indexed = {str(r['sample_id']): r for r in data}
        if len(data) != expected_count or len(indexed) != expected_count:
            raise ValueError(f'{root} 样本数量或 ID 不完整')
        return indexed
    baseline, probe = rows(baseline_dir), rows(probe_dir)
    if baseline.keys() != probe.keys():
        raise ValueError('两组样本 ID 不一致')
    mismatches, all_rows, stability = [], [], []
    timelines = {}
    for sid, row in probe.items():
        ref = baseline[sid]
        for key in ('generation_seed', 'prompt', 'sketch_path', 'texture_path'):
            if key not in row or key not in ref or row[key] != ref[key]:
                raise ValueError(f'{sid} 两组条件不一致或缺失：{key}')
        original = Path(row.get('source_gen_path') or row['gen_path'])
        with Image.open(original) as a, Image.open(ref.get('source_gen_path') or ref['gen_path']) as b:
            a, b = a.convert('RGB'), b.convert('RGB')
            if a.size != b.size or a.tobytes() != b.tobytes():
                mismatches.append(sid)
        folder = original.parent / 'condition_response_probe'
        data = json.loads((folder / 'probe.json').read_text(encoding='utf-8'))
        records = data['records']
        if data['steps'] != expected_steps or data['fractions'] != [0.1, 0.2]:
            raise ValueError(f'{sid} 与约定的步骤、两档扰动不一致')
        if [r['step_index'] for r in records] != data['steps']:
            raise ValueError(f'{sid} 探针步骤不完整')
        timelines[sid] = response_timeline(records)
        for r in records:
            with np.load(folder / f"step_{r['step_index']:02d}.npz") as arrays:
                required = ['repeat_delta'] + [f'{branch}_delta_h{h:g}'
                    for branch in ('sketch', 'texture') for h in data['fractions']]
                for key in required:
                    if not np.isfinite(arrays[key]).all():
                        raise ValueError(f'{sid} {key} 存在非有限数值')
            if set(r['regions']) != {'global', 'inner', 'boundary', 'background'}:
                raise ValueError(f'{sid} 区域记录不完整')
            for region, entry in r['regions'].items():
                if set(entry['responses']) != {'0.1', '0.2'}:
                    raise ValueError(f'{sid} 扰动记录不完整')
                for h, stats in entry['responses'].items():
                    all_rows.append(dict(sample_id=sid, step=r['step_index'], region=region, fraction=h,
                                         repeat_max_abs=r['repeat_max_abs'], repeat_rms=entry['repeat_rms'], **stats))
                for comparison, branches in entry['stability'].items():
                    for branch, stats in branches.items():
                        stability.append(dict(sample_id=sid, step=r['step_index'], region=region,
                                              comparison=comparison, branch=branch, **stats))
    def write_csv(name, data):
        with (out / name).open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sorted({k for r in data for k in r}))
            writer.writeheader()
            writer.writerows(data)
    write_csv('responses.csv', all_rows)
    write_csv('stability.csv', stability)
    (out / 'response_timeline.json').write_text(
        json.dumps(timelines, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    result = dict(status='pass' if not mismatches else 'fail', expected_count=expected_count,
                  pixel_identical_count=expected_count-len(mismatches), pixel_mismatches=mismatches,
                  probe_record_count=expected_count * len(expected_steps),
                  repeat_max_abs=max(r['repeat_max_abs'] for r in all_rows),
                  note='pass 仅表示采样原图一致且探针完整，不证明冲突投影有效。')
    (out / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    if mismatches:
        raise ValueError(f'探针开关导致原图像素不同：{mismatches}')
    print(f'通过：{expected_count} 张原图逐像素一致；响应记录见 {out}')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', required=True)
    parser.add_argument('--probe-dir', required=True)
    parser.add_argument('--expected-count', type=int, default=32)
    parser.add_argument('--expected-steps', type=int, nargs='+', default=[0, 5, 15, 25, 49])
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    report(args.baseline_dir, args.probe_dir, args.expected_count, args.output_dir, args.expected_steps)
