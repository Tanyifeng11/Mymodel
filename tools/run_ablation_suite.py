#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from eval.benchmark_utils import write_csv, write_json, write_markdown_table, write_manifest, ensure_dir
from tools.run_fixed_benchmark import build_argparser as build_bench_argparser, run_benchmark


def default_suite():
    return [
        {"name": "token", "mode": "token"},
        {"name": "spatial_minimal", "mode": "spatial"},
        {"name": "hybrid_minimal", "mode": "hybrid"},
        {"name": "spatial_bfm_like", "mode": "spatial_bfm_like"},
        {"name": "spatial_style_off", "mode": "spatial"},
        {"name": "spatial_style_on", "mode": "spatial"},
        {"name": "spatial_dropout_off", "mode": "spatial"},
        {"name": "spatial_dropout_on", "mode": "spatial"},
    ]


def main():
    ap = argparse.ArgumentParser(description="Run ablation suite by evaluating multiple mode/checkpoint settings.")
    ap.add_argument("--dataset_json", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split_path", default="eval/benchmarks/fixed_val_split.json")
    ap.add_argument("--num_samples", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--texture_ckpt", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mode_ckpt_map_json", required=True, help="JSON file: {mode_or_name: gam_ckpt_path}")
    ap.add_argument("--output_dir", default="eval_outputs/ablation_suite")
    ap.add_argument("--texture_preprocess_mode", default="crop_tile", choices=["plain_resize", "crop_tile", "plain"])
    args = ap.parse_args()

    import json

    with open(args.mode_ckpt_map_json, "r", encoding="utf-8") as f:
        mode_map = json.load(f)

    ensure_dir(args.output_dir)
    suite = default_suite()
    agg_rows = []
    for exp in suite:
        exp_name = exp["name"]
        exp_mode = exp["mode"]
        gam_ckpt = mode_map.get(exp_name, mode_map.get(exp_mode))
        if not gam_ckpt:
            continue

        bench_parser = build_bench_argparser()
        bench_args = bench_parser.parse_args(
            [
                "--dataset_json",
                args.dataset_json,
                "--data_root",
                args.data_root,
                "--split_path",
                args.split_path,
                "--num_samples",
                str(args.num_samples),
                "--seed",
                str(args.seed),
                "--gam_ckpt",
                gam_ckpt,
                "--texture_ckpt",
                args.texture_ckpt,
                "--device",
                args.device,
                "--modes",
                exp_mode,
                "--texture_preprocess_mode",
                args.texture_preprocess_mode,
                "--output_dir",
                args.output_dir,
                "--run_name",
                exp_name,
            ]
        )
        run_benchmark(bench_args)
        summary_json = os.path.join(args.output_dir, exp_name, "summary_metrics.json")
        if os.path.exists(summary_json):
            with open(summary_json, "r", encoding="utf-8") as f:
                rows = json.load(f)
            for r in rows:
                r["experiment"] = exp_name
                agg_rows.append(r)

    write_csv(os.path.join(args.output_dir, "ablation_results.csv"), agg_rows)
    write_json(os.path.join(args.output_dir, "ablation_results.json"), agg_rows)
    write_markdown_table(os.path.join(args.output_dir, "ablation_results.md"), agg_rows, title="Ablation Results")
    write_manifest(
        os.path.join(args.output_dir, "experiment_manifest.json"),
        {
            "task": "ablation_suite",
            "mode_ckpt_map_json": args.mode_ckpt_map_json,
            "split_path": args.split_path,
            "texture_preprocess_mode": args.texture_preprocess_mode,
        },
    )


if __name__ == "__main__":
    main()
