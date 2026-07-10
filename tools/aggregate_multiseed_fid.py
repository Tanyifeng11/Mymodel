#!/usr/bin/env python3
import argparse
import json
import math
import os
import statistics
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from eval.benchmark_utils import ensure_dir, write_csv, write_json


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _summary_rows(run_dir):
    for name in ("summary_metrics.json", "metrics_summary.json"):
        path = os.path.join(run_dir, name)
        if not os.path.isfile(path):
            continue
        payload = _load_json(path)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("experiments"), list):
            return payload["experiments"]
        if isinstance(payload, dict):
            return [payload]
    raise FileNotFoundError(f"missing summary_metrics.json in {run_dir}")


def _manifest(run_dir):
    path = os.path.join(run_dir, "experiment_manifest.json")
    return _load_json(path) if os.path.isfile(path) else {}


def collect_seed_rows(run_dirs):
    records = []
    for run_dir in run_dirs:
        manifest = _manifest(run_dir)
        for summary in _summary_rows(run_dir):
            fid = summary.get("FID")
            backend = summary.get("FID_backend") or manifest.get("fid_backend")
            generation_seed = summary.get("generation_seed")
            if generation_seed is None:
                generation_seed = manifest.get("generation_seed")
            if fid is None or not math.isfinite(float(fid)):
                raise ValueError(f"invalid FID in {run_dir}: {fid}")
            if not backend:
                raise ValueError(f"missing FID_backend in {run_dir}")
            if generation_seed is None:
                raise ValueError(f"missing generation_seed in {run_dir}")
            records.append(
                {
                    "run_dir": os.path.abspath(run_dir),
                    "run_name": manifest.get("run_name", os.path.basename(run_dir)),
                    "mode": summary.get("mode", "unknown"),
                    "generation_seed": int(generation_seed),
                    "generation_seed_policy": summary.get(
                        "generation_seed_policy",
                        manifest.get("generation_seed_policy"),
                    ),
                    "evaluation_protocol": manifest.get("evaluation_protocol"),
                    "count": summary.get("count"),
                    "FID": float(fid),
                    "FID_backend": backend,
                }
            )
    return records


def aggregate_seed_rows(records):
    grouped = {}
    for record in records:
        grouped.setdefault(record["mode"], []).append(record)

    output = []
    for mode, rows in sorted(grouped.items()):
        seeds = [row["generation_seed"] for row in rows]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"duplicate generation seeds for mode={mode}: {seeds}")

        for field in (
            "FID_backend",
            "evaluation_protocol",
            "count",
            "generation_seed_policy",
        ):
            values = {row[field] for row in rows}
            if len(values) != 1:
                raise ValueError(
                    f"inconsistent {field} for mode={mode}: {sorted(map(str, values))}"
                )

        rows = sorted(rows, key=lambda row: row["generation_seed"])
        fid_values = [row["FID"] for row in rows]
        output.append(
            {
                "mode": mode,
                "num_generation_seeds": len(rows),
                "generation_seeds": [row["generation_seed"] for row in rows],
                "FID_values": fid_values,
                "FID": statistics.fmean(fid_values),
                "FID_std": (
                    statistics.stdev(fid_values) if len(fid_values) > 1 else None
                ),
                "FID_backend": rows[0]["FID_backend"],
                "evaluation_protocol": rows[0]["evaluation_protocol"],
                "count_per_seed": rows[0]["count"],
                "generation_seed_policy": rows[0]["generation_seed_policy"],
            }
        )
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate independent fixed-benchmark FID runs across seeds."
    )
    parser.add_argument("--run_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    records = collect_seed_rows(args.run_dirs)
    summary = aggregate_seed_rows(records)
    ensure_dir(args.output_dir)
    write_json(os.path.join(args.output_dir, "fid_per_seed.json"), records)
    write_csv(os.path.join(args.output_dir, "fid_per_seed.csv"), records)
    write_json(os.path.join(args.output_dir, "fid_multiseed_summary.json"), summary)
    write_csv(os.path.join(args.output_dir, "fid_multiseed_summary.csv"), summary)

    for row in summary:
        std_text = "null" if row["FID_std"] is None else f"{row['FID_std']:.6f}"
        print(
            f"[fid-multiseed] mode={row['mode']} seeds={row['generation_seeds']} "
            f"FID={row['FID']:.6f} std={std_text} "
            f"backend={row['FID_backend']}"
        )


if __name__ == "__main__":
    main()
