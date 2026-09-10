"""汇总 E9-B 推理期干预的逐样本差异与残差轨迹。"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image


METRICS = (
    "clip_i_texture",
    "tpf_patch_sim",
    "tpf_gram_l1",
    "target_color_delta_e",
    "leak_colored_frac",
    "leak_edge_density",
    "struct_edge_f1",
    "struct_iou",
)
TRACE_METRICS = (
    "alpha",
    "runtime_scale",
    "base_rms",
    "detail_rms",
    "detail_highpass_rms",
    "residual_rms",
    "residual_relative_rms",
    "mask_coverage",
)


def read_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def metric_value(row, name):
    try:
        return float(row[name])
    except (KeyError, TypeError, ValueError):
        return np.nan


def image_mae(left_path, right_path):
    if not left_path or not right_path or not os.path.isfile(left_path) or not os.path.isfile(right_path):
        return np.nan
    with Image.open(left_path) as left, Image.open(right_path) as right:
        left_array = np.asarray(left.convert("RGB"), dtype=np.float32)
        right_array = np.asarray(right.convert("RGB").resize(left.size), dtype=np.float32)
    return float(np.abs(left_array - right_array).mean() / 255.0)


def discover_variants(root):
    found = {}
    for name in sorted(os.listdir(root)):
        metrics = os.path.join(root, name, "e9_b_diagnosis", "metrics_per_sample.csv")
        if os.path.isfile(metrics):
            found[name] = metrics
    return found


def summarize_deltas(root, baseline_name):
    variants = discover_variants(root)
    if baseline_name not in variants:
        raise FileNotFoundError(f"未找到基线 {baseline_name} 的逐样本指标")
    baseline_rows = {row["sample_id"]: row for row in read_rows(variants[baseline_name])}
    output = []
    for name, metrics_path in variants.items():
        rows = {row["sample_id"]: row for row in read_rows(metrics_path)}
        shared = sorted(set(baseline_rows) & set(rows), key=lambda value: int(value))
        summary = {"variant": name, "baseline": baseline_name, "shared_samples": len(shared)}
        pixel_maes = []
        for sample_id in shared:
            pixel_maes.append(image_mae(baseline_rows[sample_id].get("gen_path"), rows[sample_id].get("gen_path")))
        valid_maes = np.asarray([value for value in pixel_maes if np.isfinite(value)])
        summary["pixel_mae_mean"] = float(valid_maes.mean()) if len(valid_maes) else np.nan
        for metric in METRICS:
            deltas = np.asarray([
                metric_value(rows[sample_id], metric) - metric_value(baseline_rows[sample_id], metric)
                for sample_id in shared
            ])
            deltas = deltas[np.isfinite(deltas)]
            summary[f"{metric}_delta_mean"] = float(deltas.mean()) if len(deltas) else np.nan
            summary[f"{metric}_delta_median"] = float(np.median(deltas)) if len(deltas) else np.nan
        output.append(summary)
    return output


def summarize_trace(root):
    grouped = defaultdict(lambda: defaultdict(list))
    for trace_path in glob.glob(os.path.join(root, "*", "e9_b_diagnosis", "**", "local_detail_trace.jsonl"), recursive=True):
        variant = os.path.relpath(trace_path, root).split(os.sep)[0]
        with open(trace_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    grouped[variant][int(row["step_index"])].append(row)
    summary = []
    for variant in sorted(grouped):
        for step_index in sorted(grouped[variant]):
            rows = grouped[variant][step_index]
            row = {
                "variant": variant,
                "step_index": step_index,
                "timestep": int(np.mean([item["timestep"] for item in rows])),
                "trace_count": len(rows),
            }
            for metric in TRACE_METRICS:
                values = np.asarray([float(item[metric]) for item in rows])
                row[f"{metric}_mean"] = float(values.mean())
            summary.append(row)
    return summary


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="汇总 E9-B 局部旁路诊断结果")
    parser.add_argument("--diagnostic-root", required=True)
    parser.add_argument("--baseline", default="alpha_100")
    args = parser.parse_args()
    delta_rows = summarize_deltas(args.diagnostic_root, args.baseline)
    trace_rows = summarize_trace(args.diagnostic_root)
    write_csv(os.path.join(args.diagnostic_root, "diagnostic_variant_deltas.csv"), delta_rows)
    write_csv(os.path.join(args.diagnostic_root, "local_detail_trace_summary.csv"), trace_rows)
    with open(os.path.join(args.diagnostic_root, "diagnostic_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"baseline": args.baseline, "variant_deltas": delta_rows, "trace_rows": trace_rows}, f,
                  ensure_ascii=False, indent=2)
    print(f"[E9-B 诊断] 已写入：{args.diagnostic_root}")


if __name__ == "__main__":
    main()
