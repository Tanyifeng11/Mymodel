#!/usr/bin/env python3
"""汇总 E7a 固定对照评测：多 seed、子集和 paired bootstrap。"""

import argparse
import csv
import json
import math
import os
import statistics
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from eval.benchmark_utils import ensure_dir, write_csv, write_json, write_markdown_table
from models.attribute_token_mask import find_spans


SUMMARY_METRICS = {
    "FID": "lower",
    "KID": "lower",
    "clip_i_real_mean": "higher",
    "clip_i_texture_mean": "higher",
    "tpf_patch_sim_mean": "higher",
    "tcf_lab_delta_mean": "lower",
    "struct_edge_f1_mean": "higher",
    "struct_iou_mean": "higher",
    "leak_colored_frac_mean": "lower",
    "leak_edge_density_mean": "lower",
}

PER_SAMPLE_METRICS = {
    "clip_i_real": "higher",
    "clip_i_texture": "higher",
    "tpf_patch_sim": "higher",
    "tcf_lab_delta": "lower",
    "struct_edge_f1": "higher",
    "struct_iou": "higher",
    "leak_colored_frac": "lower",
    "leak_edge_density": "lower",
}


def finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def parse_list(value, cast=str):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def load_summary(run_dir):
    for filename in ("summary_metrics.json", "metrics_summary.json"):
        path = os.path.join(run_dir, filename)
        if os.path.isfile(path):
            data = load_json(path)
            if isinstance(data, list) and len(data) == 1:
                return data[0]
            if isinstance(data, dict):
                return data
            raise ValueError(f"invalid summary file: {path}")
    raise FileNotFoundError(f"summary missing: {run_dir}")


def load_rows(run_dir):
    path = os.path.join(run_dir, "metrics_per_sample.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def group_names(row):
    prompt = row.get("prompt", "")
    groups = ["all"]
    groups.append("pattern_hit" if find_spans(prompt, "pattern") else "pattern_fallback")
    if str(row.get("conflict_bucket", "")) == "no_text_color":
        groups.append("no_text_color")
    elif finite_float(row.get("color_conflict_score")) is not None and float(row["color_conflict_score"]) >= 0.25:
        groups.append("color_conflict")
    return groups


def mean_std(values):
    if not values:
        return None, None
    return float(statistics.fmean(values)), (float(statistics.stdev(values)) if len(values) > 1 else 0.0)


def bootstrap(values, rng, iterations):
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return None, None, None
    draws = rng.integers(0, values.size, size=(iterations, values.size))
    estimates = values[draws].mean(axis=1)
    return float(values.mean()), float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def collect(eval_root, experiments, seeds):
    summaries, samples = {}, {}
    for experiment in experiments:
        summaries[experiment] = {}
        samples[experiment] = {}
        for seed in seeds:
            run_dir = os.path.join(eval_root, f"seed_{seed}", experiment)
            summaries[experiment][seed] = load_summary(run_dir)
            rows = load_rows(run_dir)
            samples[experiment][seed] = {str(row["sample_id"]): row for row in rows}
    return summaries, samples


def validate_backends(summaries, experiments, seeds):
    for key in ("FID_backend", "KID_backend"):
        values = {
            summaries[experiment][seed].get(key)
            for experiment in experiments
            for seed in seeds
        }
        if None in values or len(values) != 1:
            raise ValueError(f"inconsistent {key}: {sorted(map(str, values))}")


def build_seed_summary_rows(summaries, experiments, seeds):
    rows = []
    for experiment in experiments:
        for seed in seeds:
            summary = summaries[experiment][seed]
            row = {
                "experiment": experiment,
                "generation_seed": seed,
                "count": summary.get("count"),
                "FID_backend": summary.get("FID_backend"),
                "KID_backend": summary.get("KID_backend"),
            }
            for metric in SUMMARY_METRICS:
                row[metric] = finite_float(summary.get(metric))
            rows.append(row)
    return rows


def build_multiseed_rows(seed_rows, experiments):
    rows = []
    for experiment in experiments:
        selected = [row for row in seed_rows if row["experiment"] == experiment]
        row = {
            "experiment": experiment,
            "num_generation_seeds": len(selected),
            "generation_seeds": ",".join(str(item["generation_seed"]) for item in selected),
            "count_per_seed": selected[0]["count"],
            "FID_backend": selected[0]["FID_backend"],
            "KID_backend": selected[0]["KID_backend"],
        }
        for metric in SUMMARY_METRICS:
            values = [item[metric] for item in selected]
            if any(value is None for value in values):
                raise ValueError(f"missing {metric} for {experiment}")
            row[f"{metric}_seed_mean"], row[f"{metric}_seed_std"] = mean_std(values)
        rows.append(row)
    return rows


def build_subset_rows(samples, experiments, seeds):
    per_seed = []
    for experiment in experiments:
        for seed in seeds:
            rows = list(samples[experiment][seed].values())
            for group in ("all", "pattern_hit", "pattern_fallback", "color_conflict", "no_text_color"):
                selected = [row for row in rows if group in group_names(row)]
                result = {
                    "experiment": experiment,
                    "generation_seed": seed,
                    "subset": group,
                    "count": len(selected),
                }
                for metric in PER_SAMPLE_METRICS:
                    values = [finite_float(row.get(metric)) for row in selected]
                    values = [value for value in values if value is not None]
                    result[f"{metric}_mean"] = float(statistics.fmean(values)) if values else None
                per_seed.append(result)

    aggregate = []
    for experiment in experiments:
        for group in ("all", "pattern_hit", "pattern_fallback", "color_conflict", "no_text_color"):
            selected = [
                row for row in per_seed
                if row["experiment"] == experiment and row["subset"] == group
            ]
            result = {
                "experiment": experiment,
                "subset": group,
                "num_generation_seeds": len(selected),
                "count_per_seed_mean": float(statistics.fmean(row["count"] for row in selected)),
            }
            for metric in PER_SAMPLE_METRICS:
                values = [row[f"{metric}_mean"] for row in selected]
                values = [value for value in values if value is not None]
                result[f"{metric}_seed_mean"], result[f"{metric}_seed_std"] = mean_std(values)
            aggregate.append(result)
    return per_seed, aggregate


def build_bootstrap_rows(samples, comparisons, seeds, iterations, bootstrap_seed):
    output = []
    groups = ("all", "pattern_hit", "pattern_fallback", "color_conflict", "no_text_color")
    for comparison_index, (candidate, reference) in enumerate(comparisons):
        for group_index, group in enumerate(groups):
            common = []
            for seed in seeds:
                candidate_rows = samples[candidate][seed]
                reference_rows = samples[reference][seed]
                for sample_id in sorted(set(candidate_rows) & set(reference_rows)):
                    if group in group_names(reference_rows[sample_id]):
                        common.append((candidate_rows[sample_id], reference_rows[sample_id]))
            for metric_index, (metric, direction) in enumerate(PER_SAMPLE_METRICS.items()):
                deltas = []
                for candidate_row, reference_row in common:
                    candidate_value = finite_float(candidate_row.get(metric))
                    reference_value = finite_float(reference_row.get(metric))
                    if candidate_value is None or reference_value is None:
                        continue
                    delta = candidate_value - reference_value
                    deltas.append(delta if direction == "higher" else -delta)
                rng = np.random.default_rng(
                    bootstrap_seed + 10000 * comparison_index + 1000 * group_index + metric_index
                )
                mean, low, high = bootstrap(deltas, rng, iterations)
                output.append(
                    {
                        "candidate": candidate,
                        "reference": reference,
                        "subset": group,
                        "metric": metric,
                        "direction": direction,
                        "paired_count": len(deltas),
                        "improvement_mean": mean,
                        "improvement_ci95_low": low,
                        "improvement_ci95_high": high,
                        "ci_excludes_zero": (low > 0 or high < 0) if low is not None else None,
                    }
                )
    return output


def save_tables(output_dir, tables):
    ensure_dir(output_dir)
    for name, rows in tables.items():
        write_json(os.path.join(output_dir, f"{name}.json"), rows)
        write_csv(os.path.join(output_dir, f"{name}.csv"), rows)
        write_markdown_table(
            os.path.join(output_dir, f"{name}.md"), rows, title=name.replace("_", " ").title()
        )


def main():
    parser = argparse.ArgumentParser(description="Summarize E7a controlled evaluations.")
    parser.add_argument("--eval_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--experiments", default="e5,e7a_on,e7a_off")
    parser.add_argument("--generation_seeds", default="42,123,2026")
    parser.add_argument("--comparisons", default="e7a_on:e5,e7a_on:e7a_off")
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    args = parser.parse_args()

    experiments = parse_list(args.experiments)
    seeds = parse_list(args.generation_seeds, int)
    comparisons = [tuple(item.split(":", 1)) for item in parse_list(args.comparisons)]
    if len(seeds) < 2 or args.bootstrap_samples < 100:
        raise ValueError("at least two seeds and 100 bootstrap samples are required")
    if any(len(pair) != 2 or pair[0] not in experiments or pair[1] not in experiments for pair in comparisons):
        raise ValueError("comparisons must be candidate:reference names from --experiments")

    summaries, samples = collect(args.eval_root, experiments, seeds)
    validate_backends(summaries, experiments, seeds)
    seed_rows = build_seed_summary_rows(summaries, experiments, seeds)
    subset_per_seed, subset_multiseed = build_subset_rows(samples, experiments, seeds)
    bootstrap_rows = build_bootstrap_rows(
        samples, comparisons, seeds, args.bootstrap_samples, args.bootstrap_seed
    )
    save_tables(args.output_dir, {
        "per_seed_summary": seed_rows,
        "multiseed_summary": build_multiseed_rows(seed_rows, experiments),
        "subset_per_seed": subset_per_seed,
        "subset_multiseed": subset_multiseed,
        "paired_bootstrap": bootstrap_rows,
    })
    print(f"[e7a-control] report saved to {args.output_dir}")


if __name__ == "__main__":
    main()
