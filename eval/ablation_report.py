#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

from eval.eval_utils import json_safe


METRIC_SECTIONS = {
    "quality": [
        ("FID ↓", "FID", "FID_std", "lower", ".2f"),
        ("CLIP-I Real ↑", "clip_i_real_mean", "clip_i_real_std", "higher", ".4f"),
        (
            "CLIP-I Texture ↑",
            "clip_i_texture_mean",
            "clip_i_texture_std",
            "higher",
            ".4f",
        ),
    ],
    "texture": [
        ("TCF-LAB ↓", "tcf_lab_delta_mean", "tcf_lab_delta_std", "lower", ".4f"),
        ("TCF-HSV ↓", "tcf_hsv_l1_mean", "tcf_hsv_l1_std", "lower", ".4f"),
        ("TCF-RGB-L2 ↓", "tcf_rgb_l2_mean", "tcf_rgb_l2_std", "lower", ".4f"),
        ("TPF-Patch ↑", "tpf_patch_sim_mean", "tpf_patch_sim_std", "higher", ".4f"),
        ("TPF-Gram ↓", "tpf_gram_l1_mean", "tpf_gram_l1_std", "lower", ".4f"),
    ],
    "leakage": [
        (
            "Leak Colored ↓",
            "leak_colored_frac_mean",
            "leak_colored_frac_std",
            "lower",
            ".4f",
        ),
        (
            "Leak Saturation ↓",
            "leak_mean_saturation_mean",
            "leak_mean_saturation_std",
            "lower",
            ".4f",
        ),
        (
            "Leak Value Shift ↓",
            "leak_value_shift_mean",
            "leak_value_shift_std",
            "lower",
            ".4f",
        ),
        (
            "Boundary Edge Density ↓",
            "leak_edge_density_mean",
            "leak_edge_density_std",
            "lower",
            ".4f",
        ),
    ],
    "structure": [
        (
            "Edge F1 ↑",
            "struct_edge_f1_mean",
            "struct_edge_f1_std",
            "higher",
            ".4f",
        ),
        (
            "Sketch IoU ↑",
            "struct_iou_mean",
            "struct_iou_std",
            "higher",
            ".4f",
        ),
        (
            "Edge L1 ↓",
            "struct_edge_l1_mean",
            "struct_edge_l1_std",
            "lower",
            ".4f",
        ),
    ],
}

SECTION_TITLES = {
    "quality": "Generation Quality",
    "texture": "Texture Fidelity",
    "leakage": "Texture Leakage",
    "structure": "Structure Preservation",
}


def _finite(value):
    return isinstance(value, (int, float, np.number)) and math.isfinite(float(value))


def _load_json(path, default=None):
    if not os.path.isfile(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        print(f"[ablation_report] WARNING: failed to read {path}: {exc}")
        return default


def collect_image_paths(exp_dir, pattern="*.png"):
    paths = []
    for root, _, files in os.walk(exp_dir):
        for filename in files:
            if filename == "generated.png":
                paths.append(os.path.join(root, filename))
    return sorted(paths)


def load_sample_path_map(exp_dir):
    rows = _load_json(os.path.join(exp_dir, "metrics_per_sample.json"))
    if rows is None:
        rows = _load_json(os.path.join(exp_dir, "per_image_metrics.json"), [])
    return {row.get("uid"): row for row in rows if row.get("uid")}


def collect_pair_paths(exp_dir):
    rows = _load_json(os.path.join(exp_dir, "metrics_per_sample.json"))
    if rows is None:
        rows = _load_json(os.path.join(exp_dir, "per_image_metrics.json"), [])
    pairs = []
    for row in rows:
        gen_path = row.get("gen_path")
        if gen_path and os.path.isfile(gen_path):
            pairs.append(
                (
                    gen_path,
                    row.get("target_path"),
                    row.get("texture_path"),
                    row.get("sketch_path"),
                    row.get("mask_path"),
                )
            )
    return pairs


def _aggregate_per_sample(rows):
    result = {}
    keys = sorted({key for row in rows for key in row})
    for key in keys:
        values = [float(row[key]) for row in rows if _finite(row.get(key))]
        if values:
            result[f"{key}_mean"] = float(np.mean(values))
            result[f"{key}_std"] = float(np.std(values))
            result[f"{key}_valid"] = len(values)
    return result


def compute_experiment_metrics(
    exp_name,
    exp_dir,
    real_image_paths=None,
    batch_size=16,
    device="cuda",
):
    del real_image_paths, batch_size, device
    metrics = {"experiment": exp_name}
    summary = _load_json(os.path.join(exp_dir, "metrics_summary.json"))
    if summary is None:
        summary = _load_json(os.path.join(exp_dir, "summary_metrics.json"))
    if isinstance(summary, list) and summary:
        if len(summary) == 1:
            metrics.update(summary[0])
        else:
            for row in summary:
                mode = row.get("mode", "unknown")
                for key, value in row.items():
                    if key != "mode":
                        metrics[f"{mode}_{key}"] = value
    elif isinstance(summary, dict):
        metrics.update(summary)
    else:
        rows = _load_json(os.path.join(exp_dir, "metrics_per_sample.json"))
        if rows is None:
            rows = _load_json(os.path.join(exp_dir, "per_image_metrics.json"), [])
        if rows:
            metrics.update(_aggregate_per_sample(rows))
        else:
            print(f"[ablation_report] WARNING: no metrics found for {exp_name}")

    if not _finite(metrics.get("clip_i_real_mean")) and _finite(
        metrics.get("CLIP-I")
    ):
        metrics["clip_i_real_mean"] = metrics["CLIP-I"]
        metrics["clip_i_real_std"] = metrics.get("CLIP-I_std")
    if _finite(metrics.get("clip_i_real_mean")):
        metrics["CLIP-I"] = metrics["clip_i_real_mean"]
        metrics["CLIP-I_std"] = metrics.get("clip_i_real_std")

    diagnostics = _load_json(os.path.join(exp_dir, "diagnostics.json"), {})
    metrics["diagnostics"] = diagnostics
    return metrics


def format_metric(value, fmt=".4f", null_str="—"):
    if not _finite(value):
        return null_str
    return format(float(value), fmt)


def format_mean_std(metrics, mean_key, std_key, fmt):
    mean = metrics.get(mean_key)
    if not _finite(mean):
        return "—"
    std = metrics.get(std_key)
    if _finite(std):
        return f"{format(float(mean), fmt)} ± {format(float(std), fmt)}"
    return format(float(mean), fmt)


def _best_indices(all_metrics, spec):
    _, mean_key, _, direction, _ = spec
    valid = [
        (index, float(metrics[mean_key]))
        for index, metrics in enumerate(all_metrics)
        if _finite(metrics.get(mean_key))
    ]
    if not valid:
        return []
    best_value = (
        min(value for _, value in valid)
        if direction == "lower"
        else max(value for _, value in valid)
    )
    return [
        index
        for index, value in valid
        if math.isclose(value, best_value, rel_tol=1e-9, abs_tol=1e-12)
    ]


def _diagnostics_markdown(all_metrics):
    lines = [
        "## Diagnostics Summary",
        "",
        "| Experiment | Samples | Generated | Real | Texture | Sketch | Dataset Mask | "
        "Valid FID | Valid CLIP-I | Valid Leakage | Valid Structure | Empty Garment | "
        "Empty Outside | Empty Boundary |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: |",
    ]
    for metrics in all_metrics:
        diagnostics = metrics.get("diagnostics") or {}
        values = [
            metrics.get("experiment", "?"),
            diagnostics.get("num_samples", "—"),
            diagnostics.get("num_generated_found", "—"),
            diagnostics.get("num_real_found", "—"),
            diagnostics.get("num_texture_found", "—"),
            diagnostics.get("num_sketch_found", "—"),
            diagnostics.get("num_mask_found", "—"),
            diagnostics.get("num_valid_for_fid", "—"),
            diagnostics.get("num_valid_for_clip_i", "—"),
            diagnostics.get("num_valid_for_leakage", "—"),
            diagnostics.get("num_valid_for_structure", "—"),
            diagnostics.get("number_of_empty_garment_masks", "—"),
            diagnostics.get("number_of_empty_outside_masks", "—"),
            diagnostics.get("number_of_empty_boundary_masks", "—"),
        ]
        lines.append("| " + " | ".join(str(value) for value in values) + " |")

    lines.extend(["", "### Skipped Metrics", ""])
    found_reason = False
    for metrics in all_metrics:
        reasons = (metrics.get("diagnostics") or {}).get(
            "skipped_metrics_and_reasons", {}
        )
        for reason, count in reasons.items():
            found_reason = True
            lines.append(
                f"- `{metrics.get('experiment', '?')}`: {reason} ({count} samples)"
            )
    if not found_reason:
        lines.append("- None")
    return "\n".join(lines)


def _build_table(all_metrics, specs, title):
    best = {
        mean_key: _best_indices(all_metrics, spec)
        for spec in specs
        for mean_key in [spec[1]]
    }
    lines = [
        f"## {title}",
        "",
        "| Experiment | " + " | ".join(spec[0] for spec in specs) + " |",
        "| --- | " + " | ".join(["---"] * len(specs)) + " |",
    ]
    for index, metrics in enumerate(all_metrics):
        values = []
        for _, mean_key, std_key, _, fmt in specs:
            value = format_mean_std(metrics, mean_key, std_key, fmt)
            if index in best[mean_key] and value != "—":
                value = f"**{value}**"
            values.append(value)
        lines.append(
            f"| {metrics.get('experiment', '?')} | " + " | ".join(values) + " |"
        )
    return "\n".join(lines)


def build_ablation_table(all_metrics, category="all"):
    if not all_metrics:
        return "(no data)"
    sections = (
        list(METRIC_SECTIONS)
        if category == "all"
        else [category] if category in METRIC_SECTIONS else list(METRIC_SECTIONS)
    )
    lines = ["# Ablation Study Results", ""]
    for section in sections:
        lines.append(
            _build_table(
                all_metrics,
                METRIC_SECTIONS[section],
                SECTION_TITLES[section],
            )
        )
        lines.append("")
    lines.append(_diagnostics_markdown(all_metrics))
    return "\n".join(lines)


def build_comprehensive_table(all_metrics):
    if not all_metrics:
        return "(no data)"
    specs = [
        spec
        for section in ("quality", "texture", "leakage", "structure")
        for spec in METRIC_SECTIONS[section]
    ]
    lines = [
        "# Comprehensive Ablation Table",
        "",
        _build_table(all_metrics, specs, "All Metrics"),
        "",
        "Values are shown as mean ± std. Missing or invalid values are shown as —.",
        "",
        "## Best Metrics",
        "",
    ]
    for spec in specs:
        label, mean_key, _, _, _ = spec
        indices = _best_indices(all_metrics, spec)
        if not indices:
            lines.append(f"- **{label}**: No valid values")
            continue
        names = ", ".join(all_metrics[index]["experiment"] for index in indices)
        lines.append(f"- **{label}**: {names}")
    lines.extend(["", _diagnostics_markdown(all_metrics)])
    return "\n".join(lines)


def export_csv(all_metrics, path):
    rows = []
    for metrics in all_metrics:
        row = {key: value for key, value in metrics.items() if key != "diagnostics"}
        diagnostics = metrics.get("diagnostics") or {}
        for key, value in diagnostics.items():
            if key != "skipped_metrics_and_reasons":
                row[f"diagnostics_{key}"] = value
        rows.append(json_safe(row))
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_radar_html(all_metrics, output_path):
    if not all_metrics:
        return
    radar_specs = [
        ("CLIP-I Real", "clip_i_real_mean", True),
        ("Texture Match", "tpf_patch_sim_mean", True),
        ("Color Fidelity", "tcf_lab_delta_mean", False),
        ("Low Leakage", "leak_colored_frac_mean", False),
        ("Structure", "struct_edge_f1_mean", True),
    ]
    ranges = {}
    for _, key, _ in radar_specs:
        values = [
            float(metrics[key])
            for metrics in all_metrics
            if _finite(metrics.get(key))
        ]
        ranges[key] = (min(values), max(values)) if values else (None, None)

    datasets = []
    for metrics in all_metrics:
        values = []
        for _, key, higher_is_better in radar_specs:
            value = metrics.get(key)
            minimum, maximum = ranges[key]
            if not _finite(value) or minimum is None:
                values.append(None)
                continue
            if math.isclose(minimum, maximum):
                normalized = 50.0
            else:
                normalized = (float(value) - minimum) / (maximum - minimum) * 100.0
            values.append(round(normalized if higher_is_better else 100.0 - normalized, 2))
        datasets.append({"label": metrics.get("experiment", "?"), "data": values})

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Ablation Radar Chart</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.5.0/dist/chart.umd.js"></script>
</head>
<body>
<canvas id="radar"></canvas>
<script>
new Chart(document.getElementById('radar'), {{
  type: 'radar',
  data: {{
    labels: {json.dumps([spec[0] for spec in radar_specs])},
    datasets: {json.dumps(datasets)}
  }},
  options: {{ scales: {{ r: {{ min: 0, max: 100 }} }} }}
}});
</script>
</body>
</html>"""
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(html)


def main():
    parser = argparse.ArgumentParser(
        description="Generate ablation tables from benchmark outputs."
    )
    parser.add_argument("--experiments_dir", required=True)
    parser.add_argument("--real_images_dir", default=None)
    parser.add_argument("--real_images_list", default=None)
    parser.add_argument("--output_dir", default="eval_outputs/report")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clip_model_path", default=None)
    parser.add_argument("--experiment_names", default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    experiment_root = Path(args.experiments_dir)
    if not experiment_root.is_dir():
        raise SystemExit(
            f"ERROR: --experiments_dir '{args.experiments_dir}' is not a directory"
        )
    if args.experiment_names:
        experiment_dirs = [
            experiment_root / name.strip()
            for name in args.experiment_names.split(",")
            if name.strip()
        ]
    else:
        experiment_dirs = sorted(
            directory for directory in experiment_root.iterdir() if directory.is_dir()
        )

    all_metrics = []
    for experiment_dir in experiment_dirs:
        if not experiment_dir.is_dir():
            print(
                f"[ablation_report] WARNING: experiment missing: {experiment_dir}"
            )
            continue
        all_metrics.append(
            compute_experiment_metrics(
                experiment_dir.name,
                str(experiment_dir),
                batch_size=args.batch_size,
                device=args.device,
            )
        )
    if not all_metrics:
        raise SystemExit("ERROR: no experiment metrics found")

    comprehensive = build_comprehensive_table(all_metrics)
    categorized = build_ablation_table(all_metrics)
    with open(
        os.path.join(args.output_dir, "comprehensive_table.md"),
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(comprehensive)
    with open(
        os.path.join(args.output_dir, "ablation_tables.md"),
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(categorized)
    export_csv(all_metrics, os.path.join(args.output_dir, "ablation_results.csv"))
    with open(
        os.path.join(args.output_dir, "ablation_results.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(json_safe(all_metrics), handle, indent=2, ensure_ascii=False)
    build_radar_html(all_metrics, os.path.join(args.output_dir, "radar_chart.html"))
    print(comprehensive)


if __name__ == "__main__":
    main()
