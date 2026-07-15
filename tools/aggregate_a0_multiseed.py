#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import statistics
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from eval.benchmark_utils import ensure_dir, write_csv, write_json, write_markdown_table


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
}

PER_SAMPLE_METRICS = {
    "clip_i_real": "higher",
    "clip_i_texture": "higher",
    "tpf_patch_sim": "higher",
    "tcf_lab_delta": "lower",
    "struct_edge_f1": "higher",
    "struct_iou": "higher",
    "leak_colored_frac": "lower",
}


def parse_csv_list(value, cast=str):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def load_summary(run_dir):
    for filename in ("summary_metrics.json", "metrics_summary.json"):
        path = os.path.join(run_dir, filename)
        if not os.path.isfile(path):
            continue
        payload = load_json(path)
        if isinstance(payload, list) and len(payload) == 1:
            return payload[0]
        if isinstance(payload, dict):
            return payload
        raise ValueError(f"expected exactly one summary row: {path}")
    raise FileNotFoundError(f"missing summary metrics in {run_dir}")


def finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def sample_stdev(values):
    return statistics.stdev(values) if len(values) > 1 else None


def collect_seed_summaries(eval_root, experiments, seeds):
    rows = []
    for experiment in experiments:
        for seed in seeds:
            run_dir = os.path.join(eval_root, f"seed_{seed}", experiment)
            summary = load_summary(run_dir)
            row = {
                "experiment": experiment,
                "generation_seed": seed,
                "run_dir": os.path.abspath(run_dir),
                "count": summary.get("count"),
                "FID_backend": summary.get("FID_backend"),
            }
            for metric in SUMMARY_METRICS:
                row[metric] = finite_float(summary.get(metric))
            rows.append(row)
    return rows


def aggregate_seed_summaries(seed_rows, experiments):
    output = []
    for experiment in experiments:
        selected = [row for row in seed_rows if row["experiment"] == experiment]
        result = {
            "experiment": experiment,
            "num_generation_seeds": len(selected),
            "generation_seeds": [row["generation_seed"] for row in selected],
            "count_per_seed": selected[0].get("count") if selected else None,
        }
        backends = {row.get("FID_backend") for row in selected}
        if len(backends) != 1:
            raise ValueError(f"inconsistent FID backends for {experiment}: {backends}")
        result["FID_backend"] = next(iter(backends)) if backends else None
        for metric in SUMMARY_METRICS:
            values = [row[metric] for row in selected if row[metric] is not None]
            if len(values) != len(selected):
                raise ValueError(f"missing {metric} for {experiment}")
            result[f"{metric}_seed_mean"] = statistics.fmean(values)
            result[f"{metric}_seed_std"] = sample_stdev(values)
        output.append(result)
    return output


def load_per_sample(run_dir, generation_seed):
    path = os.path.join(run_dir, "metrics_per_sample.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    rows = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(generation_seed), str(row["sample_id"]))
            rows[key] = row
    return rows


def paired_bootstrap(values, rng, samples):
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        raise ValueError("paired bootstrap needs at least two finite pairs")
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        draw = rng.integers(0, array.size, size=array.size)
        estimates[index] = array[draw].mean()
    return float(array.mean()), float(np.quantile(estimates, 0.025)), float(
        np.quantile(estimates, 0.975)
    )


def build_bootstrap_rows(
    eval_root,
    experiments,
    seeds,
    reference,
    bootstrap_samples,
    bootstrap_seed,
):
    cache = {}
    for experiment in experiments:
        combined = {}
        for seed in seeds:
            run_dir = os.path.join(eval_root, f"seed_{seed}", experiment)
            combined.update(load_per_sample(run_dir, seed))
        cache[experiment] = combined

    output = []
    reference_rows = cache[reference]
    for experiment in experiments:
        if experiment == reference:
            continue
        candidate_rows = cache[experiment]
        common = sorted(set(reference_rows) & set(candidate_rows))
        if not common:
            raise ValueError(f"no paired samples for {experiment} vs {reference}")
        for metric_index, (metric, direction) in enumerate(PER_SAMPLE_METRICS.items()):
            deltas = []
            for key in common:
                candidate = finite_float(candidate_rows[key].get(metric))
                baseline = finite_float(reference_rows[key].get(metric))
                if candidate is None or baseline is None:
                    continue
                raw_delta = candidate - baseline
                deltas.append(raw_delta if direction == "higher" else -raw_delta)
            rng = np.random.default_rng(
                bootstrap_seed + metric_index + 1000 * experiments.index(experiment)
            )
            mean, low, high = paired_bootstrap(deltas, rng, bootstrap_samples)
            output.append(
                {
                    "experiment": experiment,
                    "reference": reference,
                    "metric": metric,
                    "direction": direction,
                    "paired_count": len(deltas),
                    "improvement_mean": mean,
                    "improvement_ci95_low": low,
                    "improvement_ci95_high": high,
                    "ci_excludes_zero": low > 0 or high < 0,
                }
            )
    return output


def build_direction_rows(seed_rows, experiments, reference):
    reference_by_seed = {
        row["generation_seed"]: row
        for row in seed_rows
        if row["experiment"] == reference
    }
    output = []
    for experiment in experiments:
        if experiment == reference:
            continue
        candidate_rows = [row for row in seed_rows if row["experiment"] == experiment]
        for metric, direction in SUMMARY_METRICS.items():
            improvements = []
            for candidate in candidate_rows:
                baseline = reference_by_seed[candidate["generation_seed"]]
                delta = candidate[metric] - baseline[metric]
                improvements.append(delta if direction == "higher" else -delta)
            output.append(
                {
                    "experiment": experiment,
                    "reference": reference,
                    "metric": metric,
                    "direction": direction,
                    "improved_seed_count": sum(value > 0 for value in improvements),
                    "num_generation_seeds": len(improvements),
                    "improvement_seed_mean": statistics.fmean(improvements),
                    "improvement_seed_std": sample_stdev(improvements),
                }
            )
    return output


def main():
    parser = argparse.ArgumentParser(description="Aggregate A0 validation runs across models and seeds.")
    parser.add_argument("--eval_root", required=True)
    parser.add_argument("--experiments", required=True)
    parser.add_argument("--generation_seeds", default="42,123,2026")
    parser.add_argument("--reference", default="e2b_color_safe_gate")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    args = parser.parse_args()

    experiments = parse_csv_list(args.experiments)
    seeds = parse_csv_list(args.generation_seeds, int)
    if args.reference not in experiments:
        raise ValueError(f"reference {args.reference!r} is not in experiments")
    if len(seeds) < 2:
        raise ValueError("A0 requires at least two generation seeds")
    if args.bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")

    seed_rows = collect_seed_summaries(args.eval_root, experiments, seeds)
    aggregate_rows = aggregate_seed_summaries(seed_rows, experiments)
    direction_rows = build_direction_rows(seed_rows, experiments, args.reference)
    bootstrap_rows = build_bootstrap_rows(
        args.eval_root,
        experiments,
        seeds,
        args.reference,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )

    ensure_dir(args.output_dir)
    for filename, rows in (
        ("a0_per_seed", seed_rows),
        ("a0_multiseed_summary", aggregate_rows),
        ("a0_seed_direction", direction_rows),
        ("a0_paired_bootstrap", bootstrap_rows),
    ):
        write_json(os.path.join(args.output_dir, f"{filename}.json"), rows)
        write_csv(os.path.join(args.output_dir, f"{filename}.csv"), rows)
        write_markdown_table(
            os.path.join(args.output_dir, f"{filename}.md"),
            rows,
            title=filename.replace("_", " ").title(),
        )

    for row in aggregate_rows:
        print(
            f"[A0] {row['experiment']} "
            f"FID={row['FID_seed_mean']:.6f}±{row['FID_seed_std']:.6f} "
            f"KID={row['KID_seed_mean']:.6f}±{row['KID_seed_std']:.6f}"
        )


if __name__ == "__main__":
    main()
