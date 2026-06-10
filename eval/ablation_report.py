#!/usr/bin/env python3
"""
Ablation Report Generator for Multimodal Garment Generation.

Produces paper-ready tables showing:
  - Generation Quality (FID, CLIP-I, SSIM)
  - Texture Strength (TSS, TCF, TPF)
  - Texture Leakage (LR, BAS, BCS)
  - Structure Preservation (Edge F1, Sketch IoU)

Usage:
  python -m eval.ablation_report \
    --experiments_dir eval_outputs/ablation_suite \
    --real_images_dir /path/to/real/test/images \
    --output_dir eval_outputs/report

For each experiment directory containing generated images, this script calculates
all metrics and produces:
  - ablation_table.md (Markdown table, ready for paper)
  - ablation_table.csv (CSV for further analysis)
  - ablation_full.json (all per-sample data)
  - radar_chart.html (visual comparison)
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Ensure the project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


# ============================================================================
# Metric computation helpers
# ============================================================================

def collect_image_paths(exp_dir: str, pattern: str = "*.png") -> List[str]:
    """Collect all generated images under an experiment directory."""
    paths = []
    for root, _, files in os.walk(exp_dir):
        for fn in files:
            if fn.endswith(".png") and "grid" not in fn.lower():
                paths.append(os.path.join(root, fn))
    return sorted(paths)


def collect_pair_paths(exp_dir: str) -> List[Tuple[str, str, str, str]]:
    """
    Walk experiment directory and find (gen, target, texture, sketch) tuples.
    Expects structure like:
        exp_dir/mode/uid/generated.png
    with corresponding target/texture/sketch in the data directory.
    Returns list of (gen_path, target_path, texture_path, sketch_path, mask_path).
    If target/texture/sketch not available in metadata, those entries will be None.
    """
    pairs = []
    meta_file = os.path.join(exp_dir, "per_image_metrics.json")
    if os.path.exists(meta_file):
        with open(meta_file, "r") as f:
            rows = json.load(f)
        for row in rows:
            gen = row.get("gen_path", "")
            if not gen or not os.path.exists(gen):
                continue
            pairs.append((
                gen,
                row.get("target_path"),
                row.get("texture_path"),
                row.get("sketch_path"),
                row.get("mask_path"),
            ))
    else:
        # Fallback: just collect all generated images
        for p in collect_image_paths(exp_dir):
            pairs.append((p, None, None, None, None))
    return pairs


# ============================================================================
# Core: compute all metrics for an experiment
# ============================================================================

def compute_experiment_metrics(
    exp_name: str,
    exp_dir: str,
    real_image_paths: Optional[List[str]] = None,
    batch_size: int = 16,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Compute all evaluation metrics for one experiment.

    Returns a flat dict with metric_name -> value.
    """
    from eval.metrics import (
        compute_clip_i,
        compute_fid_from_paths,
        compute_texture_color_fidelity,
        compute_texture_leakage,
        compute_texture_pattern_fidelity,
        compute_structure_preservation,
    )

    metrics = {"experiment": exp_name}

    pairs = collect_pair_paths(exp_dir)
    gen_paths = [p[0] for p in pairs if os.path.exists(p[0])]
    if not gen_paths:
        print(f"[ablation_report] WARNING: No generated images found in {exp_dir}")
        return metrics

    print(f"[ablation_report] {exp_name}: {len(gen_paths)} generated images")

    # ---- FID ----
    if real_image_paths and len(real_image_paths) >= 100:
        try:
            fid = compute_fid_from_paths(gen_paths, real_image_paths, batch_size=batch_size, device=device)
            metrics["FID"] = round(fid, 2)
            print(f"  FID: {fid:.2f}")
        except Exception as e:
            print(f"  FID: ERROR ({e})")
            metrics["FID"] = None
    else:
        print("  FID: SKIPPED (need >= 100 real images)")

    # ---- CLIP-I ----
    if real_image_paths and len(real_image_paths) >= 10:
        try:
            clip = compute_clip_i(gen_paths[:len(real_image_paths)], real_image_paths[:len(gen_paths)],
                                  batch_size=batch_size, device=device)
            metrics["CLIP-I"] = round(clip["clip_i_mean"], 4)
            metrics["CLIP-I_std"] = round(clip["clip_i_std"], 4)
            print(f"  CLIP-I: {clip['clip_i_mean']:.4f} ± {clip['clip_i_std']:.4f}")
        except Exception as e:
            print(f"  CLIP-I: ERROR ({e})")
            metrics["CLIP-I"] = None

    # ---- Per-image metrics (averaged) ----
    tcf_vals = defaultdict(list)
    tpf_vals = defaultdict(list)
    leak_vals = defaultdict(list)
    struct_vals = defaultdict(list)

    for gen, target, texture, sketch, mask in pairs:
        if not os.path.exists(gen):
            continue

        if texture:
            try:
                tcf = compute_texture_color_fidelity(gen, texture, mask)
                for k, v in tcf.items():
                    tcf_vals[k].append(v)
                tpf = compute_texture_pattern_fidelity(gen, texture, mask)
                for k, v in tpf.items():
                    tpf_vals[k].append(v)
            except Exception:
                pass

        try:
            leak = compute_texture_leakage(gen, mask)
            for k, v in leak.items():
                leak_vals[k].append(v)
        except Exception:
            pass

        if sketch:
            try:
                struct = compute_structure_preservation(gen, sketch, mask)
                for k, v in struct.items():
                    struct_vals[k].append(v)
            except Exception:
                pass

    # Aggregate
    for name, vals in [("TCF", tcf_vals), ("TPF", tpf_vals), ("Leak", leak_vals), ("Struct", struct_vals)]:
        for k, v_list in vals.items():
            if v_list:
                metrics[f"{k}_mean"] = round(np.mean(v_list), 4)
                metrics[f"{k}_std"] = round(np.std(v_list), 4)

    return metrics


# ============================================================================
# Table formatting
# ============================================================================

def format_metric(val, fmt=".4f", null_str="—"):
    """Format a metric value for table display."""
    if val is None:
        return null_str
    if isinstance(val, float):
        return f"{val:{fmt}}"
    return str(val)


def build_ablation_table(all_metrics: List[Dict], category: str = "all") -> str:
    """
    Build a Markdown-format ablation table from a list of experiment metrics dicts.

    Categories:
      "quality"  — FID, CLIP-I
      "texture"  — TCF, TPF (texture strength)
      "leakage"  — leak_* (texture leakage)
      "structure" — struct_* (structure preservation)
      "all"      — all combined
    """
    if not all_metrics:
        return "(no data)"

    # Define what columns to show
    section_configs = {
        "quality": {
            "title": "Generation Quality",
            "keys": ["FID", "CLIP-I"],
        },
        "texture": {
            "title": "Texture Strength",
            "keys": ["tcf_lab_delta_mean", "tcf_hsv_l1_mean", "tpf_patch_sim_mean", "tpf_gram_l1_mean"],
            "labels": {
                "tcf_lab_delta_mean": "TCF-LAB ↓",
                "tcf_hsv_l1_mean": "TCF-HSV ↓",
                "tpf_patch_sim_mean": "TPF-Patch ↑",
                "tpf_gram_l1_mean": "TPF-Gram ↓",
            },
        },
        "leakage": {
            "title": "Texture Leakage",
            "keys": ["leak_colored_frac_mean", "leak_mean_saturation_mean", "leak_edge_density_mean"],
            "labels": {
                "leak_colored_frac_mean": "LR-Colored ↓",
                "leak_mean_saturation_mean": "LR-Sat ↓",
                "leak_edge_density_mean": "BAS ↓",
            },
        },
        "structure": {
            "title": "Structure Preservation",
            "keys": ["struct_edge_f1_mean", "struct_iou_mean", "struct_edge_l1_mean"],
            "labels": {
                "struct_edge_f1_mean": "Edge F1 ↑",
                "struct_iou_mean": "Sketch IoU ↑",
                "struct_edge_l1_mean": "Edge L1 ↓",
            },
        },
    }

    if category == "all":
        sections_to_build = list(section_configs.keys())
    else:
        sections_to_build = [category] if category in section_configs else list(section_configs.keys())

    lines = ["# Ablation Study Results", ""]

    for cat in sections_to_build:
        cfg = section_configs[cat]
        lines.append(f"## {cfg['title']}")
        lines.append("")

        keys = cfg["keys"]
        labels = cfg.get("labels", {k: k for k in keys})

        # Header
        header = "| Experiment | " + " | ".join(labels[k] for k in keys if k in labels) + " |"
        sep = "|" + "|".join([" --- " for _ in range(len(keys) + 1)]) + "|"
        lines.append(header)
        lines.append(sep)

        # Data rows
        for m in all_metrics:
            exp_name = m.get("experiment", "?")
            vals = []
            for k in keys:
                val = m.get(k)
                fmt = ".4f"
                if k == "FID":
                    fmt = ".2f"
                vals.append(format_metric(val, fmt))
            lines.append("| " + exp_name + " | " + " | ".join(vals) + " |")

        lines.append("")

    # Legend
    lines.append("---")
    lines.append("")
    lines.append("**Legend:**")
    lines.append("- **FID ↓**: Fréchet Inception Distance (lower = better quality)")
    lines.append("- **CLIP-I ↑**: CLIP Image similarity (higher = more semantically similar)")
    lines.append("- **TCF-LAB ↓**: Texture Color Fidelity — LAB mean color distance (lower = colors closer to texture)")
    lines.append("- **TCF-HSV ↓**: Histogram distance to texture (lower = tone distribution matches texture better)")
    lines.append("- **TPF-Patch ↑**: Texture Pattern Fidelity — patch-level similarity (higher = local patterns match better)")
    lines.append("- **TPF-Gram ↓**: VGG Gram matrix L1 (lower = style/texture matches better)")
    lines.append("- **LR-Colored ↓**: Leakage Rate — fraction of background pixels with noticeable color")
    lines.append("- **LR-Sat ↓**: Background mean saturation")
    lines.append("- **BAS ↓**: Boundary Artifact Score — edge density at garment boundary")
    lines.append("- **Edge F1 ↑**: Edge alignment with sketch (higher = better structure)")
    lines.append("- **Sketch IoU ↑**: Foreground overlap with sketch-derived mask")
    lines.append("- **Edge L1 ↓**: Mean edge difference vs sketch")
    lines.append("")
    lines.append("Arrows: ↑ = higher is better, ↓ = lower is better.")

    return "\n".join(lines)


def build_comprehensive_table(all_metrics: List[Dict]) -> str:
    """
    Build a single comprehensive table suitable for paper.
    Rows = experiments, columns = all key metrics.
    """
    if not all_metrics:
        return "(no data)"

    key_metrics = [
        # (display_name, dict_key, fmt, arrow)
        ("FID ↓", "FID", ".2f", "↓"),
        ("CLIP-I ↑", "CLIP-I", ".4f", "↑"),
        ("TCF-LAB ↓", "tcf_lab_delta_mean", ".4f", "↓"),
        ("TCF-HSV ↓", "tcf_hsv_l1_mean", ".4f", "↓"),
        ("TPF-Patch ↑", "tpf_patch_sim_mean", ".4f", "↑"),
        ("TPF-Gram ↓", "tpf_gram_l1_mean", ".4f", "↓"),
        ("LR-Colored ↓", "leak_colored_frac_mean", ".4f", "↓"),
        ("LR-Sat ↓", "leak_mean_saturation_mean", ".4f", "↓"),
        ("BAS ↓", "leak_edge_density_mean", ".2f", "↓"),
        ("Edge F1 ↑", "struct_edge_f1_mean", ".4f", "↑"),
        ("IoU ↑", "struct_iou_mean", ".4f", "↑"),
        ("Edge L1 ↓", "struct_edge_l1_mean", ".2f", "↓"),
    ]

    lines = ["# Comprehensive Ablation Table", ""]
    lines.append("| Experiment | " + " | ".join(name for name, _, _, _ in key_metrics) + " |")
    lines.append("|" + "|".join([" --- " for _ in range(len(key_metrics) + 1)]) + "|")

    for m in all_metrics:
        exp = m.get("experiment", "?")
        vals = []
        for _, key, fmt, _ in key_metrics:
            val = m.get(key)
            vals.append(format_metric(val, fmt))
        lines.append("| " + exp + " | " + " | ".join(vals) + " |")

    lines.append("")
    lines.append("**Bold** = best in column. See `ablation_results.json` for standard deviations.")

    # Highlight best per column
    best_row_indices = {}
    for _, key, _, direction in key_metrics:
        values = []
        for m in all_metrics:
            v = m.get(key)
            if v is not None and isinstance(v, (int, float)):
                values.append(v)
            else:
                values.append(float("inf") if "↓" in direction else float("-inf"))
        if values:
            if "↓" in direction:
                best = min(values)
            else:
                best = max(values)
            best_row_indices[key] = [i for i, v in enumerate(values) if v == best]

    lines.append("")
    for key, indices in best_row_indices.items():
        for idx in indices:
            name = all_metrics[idx].get("experiment", "?")
            lines.append(f"- Best **{key}**: {name}")

    return "\n".join(lines)


# ============================================================================
# CSV export
# ============================================================================

def export_csv(all_metrics: List[Dict], path: str):
    """Export all metrics to CSV."""
    import csv
    if not all_metrics:
        return
    all_keys = sorted(set().union(*(m.keys() for m in all_metrics)))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for row in all_metrics:
            writer.writerow({k: row.get(k, "") for k in all_keys})


# ============================================================================
# HTML radar chart
# ============================================================================

def build_radar_html(all_metrics: List[Dict], output_path: str):
    """Generate an HTML file with radar charts comparing experiments."""
    if not all_metrics or len(all_metrics) < 2:
        return

    # Normalize metrics to 0-100 scale (invert "lower is better")
    radar_metrics = {
        "Quality": ("CLIP-I", True, 0.6, 1.0),
        "No FID Drop": ("FID", False, 10.0, 200.0),
        "Texture Match": ("tpf_patch_sim_mean", True, 0.0, 1.0),
        "Color Fidelity": ("tcf_lab_delta_mean", False, 0.0, 50.0),
        "Low Leakage": ("leak_colored_frac_mean", False, 0.0, 0.5),
        "Structure": ("struct_edge_f1_mean", True, 0.0, 1.0),
    }

    datasets = []
    labels = list(radar_metrics.keys())
    for m in all_metrics:
        values = []
        for _, (key, higher_better, vmin, vmax) in radar_metrics.items():
            v = m.get(key)
            if v is None or not isinstance(v, (int, float)):
                values.append(0)
            else:
                # Normalize and invert if needed
                norm = max(0.0, min(1.0, (v - vmin) / max(vmax - vmin, 1e-6) * 100))
                if not higher_better:
                    norm = 100.0 - norm
                values.append(round(norm, 1))
        datasets.append({
            "label": m.get("experiment", "?"),
            "data": values,
        })

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Ablation Radar Chart</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.5.0/dist/chart.umd.js"
 integrity="sha384-iU8HYtnGQ8Cy4zl7gbNMOhsDTTKX02BTXptVP/vqAWIaTfM7isw76iyZCsjL2eVi" crossorigin="anonymous"></script>
<style>
  :root {{ color-scheme: light }}
  body {{ font-family: -apple-system, sans-serif; background: #fff; color: #222; max-width: 900px; margin: 40px auto; }}
  h1 {{ text-align: center; }}
  .chart-container {{ width: 600px; height: 600px; margin: 0 auto; }}
</style>
</head>
<body>
<h1>Ablation Study — Radar Comparison</h1>
<div class="chart-container">
  <canvas id="radar"></canvas>
</div>
<script>
const ctx = document.getElementById('radar').getContext('2d');
new Chart(ctx, {{
  type: 'radar',
  data: {{
    labels: {json.dumps(labels)},
    datasets: {json.dumps(datasets)}
  }},
  options: {{
    scales: {{ r: {{ min: 0, max: 100, ticks: {{ stepSize: 20 }} }} }},
    plugins: {{ legend: {{ position: 'bottom' }} }}
  }}
}});
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


# ============================================================================
# Main entry point
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Generate paper-ready ablation tables from experiment outputs."
    )
    ap.add_argument("--experiments_dir", required=True,
                    help="Directory containing subdirectories, one per experiment (e.g., eval_outputs/ablation_suite)")
    ap.add_argument("--real_images_dir", default=None,
                    help="Directory of real/ground-truth images for FID/CLIP-I computation")
    ap.add_argument("--real_images_list", default=None,
                    help="JSON file with list of real image paths (alternative to --real_images_dir)")
    ap.add_argument("--output_dir", default="eval_outputs/report",
                    help="Where to save the report files")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--experiment_names", default=None,
                    help="Comma-separated list of subdirectory names to evaluate (default: all)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find experiment directories
    exp_root = Path(args.experiments_dir)
    if not exp_root.is_dir():
        print(f"ERROR: --experiments_dir '{args.experiments_dir}' is not a directory.")
        sys.exit(1)

    if args.experiment_names:
        exp_dirs = [exp_root / name.strip() for name in args.experiment_names.split(",")]
    else:
        exp_dirs = sorted([d for d in exp_root.iterdir() if d.is_dir()])

    # Collect real image paths
    real_paths = None
    if args.real_images_list:
        with open(args.real_images_list, "r") as f:
            real_paths = json.load(f)
    elif args.real_images_dir:
        real_dir = Path(args.real_images_dir)
        real_paths = sorted([str(p) for p in real_dir.glob("*.png")] +
                            [str(p) for p in real_dir.glob("*.jpg")] +
                            [str(p) for p in real_dir.glob("*.jpeg")])

    if real_paths:
        print(f"[ablation_report] Found {len(real_paths)} real images for FID/CLIP-I")
    else:
        print("[ablation_report] No real images provided — FID/CLIP-I will be skipped.")

    # Compute metrics for each experiment
    all_metrics = []
    for exp_dir in exp_dirs:
        if not exp_dir.is_dir():
            continue
        exp_name = exp_dir.name
        print(f"\n{'='*60}")
        print(f"Evaluating: {exp_name}")
        print(f"{'='*60}")
        metrics = compute_experiment_metrics(
            exp_name, str(exp_dir), real_paths,
            batch_size=args.batch_size, device=args.device,
        )
        all_metrics.append(metrics)

    if not all_metrics:
        print("ERROR: No experiments with data found.")
        sys.exit(1)

    # Generate outputs
    # 1. Comprehensive table (paper-ready)
    comp_table = build_comprehensive_table(all_metrics)
    comp_path = os.path.join(args.output_dir, "comprehensive_table.md")
    with open(comp_path, "w", encoding="utf-8") as f:
        f.write(comp_table)
    print(f"\n[report] Comprehensive table → {comp_path}")

    # 2. Categorized tables
    cat_table = build_ablation_table(all_metrics, "all")
    cat_path = os.path.join(args.output_dir, "ablation_tables.md")
    with open(cat_path, "w", encoding="utf-8") as f:
        f.write(cat_table)
    print(f"[report] Categorized tables → {cat_path}")

    # 3. CSV
    csv_path = os.path.join(args.output_dir, "ablation_results.csv")
    export_csv(all_metrics, csv_path)
    print(f"[report] CSV → {csv_path}")

    # 4. Full JSON
    json_path = os.path.join(args.output_dir, "ablation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"[report] JSON → {json_path}")

    # 5. Radar chart HTML
    radar_path = os.path.join(args.output_dir, "radar_chart.html")
    build_radar_html(all_metrics, radar_path)
    print(f"[report] Radar chart → {radar_path}")

    # 6. Print summary to console
    print("\n" + "=" * 80)
    print("SUMMARY: Comprehensive Ablation Table")
    print("=" * 80)
    print(comp_table)


if __name__ == "__main__":
    main()
