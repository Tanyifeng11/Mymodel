#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from eval.benchmark_utils import (
    create_or_load_fixed_split,
    ensure_dir,
    sample_uid,
    write_csv,
    write_json,
    write_manifest,
    write_markdown_table,
)
from eval.metrics import evaluate_pair


def parse_modes(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def mode_to_flags(mode_name):
    if mode_name == "token":
        return {"texture_condition_mode": "token", "fusion_type": "minimal"}
    if mode_name == "spatial":
        return {"texture_condition_mode": "spatial", "fusion_type": "minimal"}
    if mode_name == "hybrid":
        return {"texture_condition_mode": "hybrid", "fusion_type": "minimal"}
    if mode_name == "spatial_bfm_like":
        return {"texture_condition_mode": "spatial", "fusion_type": "bfm_like"}
    raise ValueError(f"Unsupported mode: {mode_name}")


def run_one_inference(args, sample, mode_name, out_dir):
    uid = sample_uid(sample)
    sample_out = os.path.join(out_dir, mode_name, uid)
    ensure_dir(sample_out)
    flags = mode_to_flags(mode_name)
    sketch_path = os.path.join(args.data_root, sample["sketch"])
    texture_path = os.path.join(args.data_root, sample["texture"])
    cmd = [
        "python",
        "inference_IMAGGarment-1.py",
        "--GAM_model_ckpt",
        args.gam_ckpt,
        "--texture_ckpt",
        args.texture_ckpt,
        "--sketch_path",
        sketch_path,
        "--texture_path",
        texture_path,
        "--prompt",
        sample["prompt"],
        "--output_path",
        sample_out,
        "--device",
        args.device,
        "--texture_condition_mode",
        flags["texture_condition_mode"],
        "--fusion_type",
        flags["fusion_type"],
        "--texture_preprocess_mode",
        args.texture_preprocess_mode,
        "--alpha1",
        str(args.alpha1),
        "--alpha2",
        str(args.alpha2),
        "--alpha3",
        str(args.alpha3),
        "--alpha4",
        str(args.alpha4),
    ]
    subprocess.run(cmd, check=True)
    src = os.path.join(sample_out, os.path.basename(sketch_path))
    dst = os.path.join(sample_out, "generated.png")
    if os.path.exists(src):
        shutil.move(src, dst)
    return dst


def make_grid(image_paths, save_path, cols=4):
    from PIL import Image

    images = [Image.open(p).convert("RGB") for p in image_paths if os.path.exists(p)]
    if not images:
        return
    w = max(i.size[0] for i in images)
    h = max(i.size[1] for i in images)
    rows = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))
    for i, img in enumerate(images):
        grid.paste(img.resize((w, h), Image.BICUBIC), ((i % cols) * w, (i // cols) * h))
    grid.save(save_path)


def run_benchmark(args):
    split = create_or_load_fixed_split(args.dataset_json, args.split_path, num_samples=args.num_samples, seed=args.seed)
    run_dir = os.path.join(args.output_dir, args.run_name)
    ensure_dir(run_dir)

    modes = parse_modes(args.modes)
    per_image_rows = []
    by_mode = {m: [] for m in modes}

    for mode in modes:
        for sample in split:
            gen_path = run_one_inference(args, sample, mode, run_dir)
            target_path = os.path.join(args.data_root, sample["target"]) if sample.get("target") else None
            texture_path = os.path.join(args.data_root, sample["texture"]) if sample.get("texture") else None
            mask_path = os.path.join(args.data_root, sample["mask"]) if sample.get("mask") else None
            metrics = evaluate_pair(gen_path, target_path=target_path, texture_path=texture_path, mask_path=mask_path)
            row = {"mode": mode, "uid": sample_uid(sample), "gen_path": gen_path, **metrics}
            per_image_rows.append(row)
            by_mode[mode].append(row)

        mode_grid_paths = [r["gen_path"] for r in by_mode[mode]]
        make_grid(mode_grid_paths, os.path.join(run_dir, f"grid_{mode}.png"), cols=4)

    summary_rows = []
    for mode, rows in by_mode.items():
        if not rows:
            continue
        keys = [k for k in rows[0].keys() if isinstance(rows[0].get(k), (int, float))]
        row = {"mode": mode, "count": len(rows)}
        for k in keys:
            row[k] = sum(float(r.get(k, 0.0)) for r in rows) / max(1, len(rows))
        summary_rows.append(row)

    write_csv(os.path.join(run_dir, "per_image_metrics.csv"), per_image_rows)
    write_json(os.path.join(run_dir, "per_image_metrics.json"), per_image_rows)
    write_csv(os.path.join(run_dir, "summary_metrics.csv"), summary_rows)
    write_json(os.path.join(run_dir, "summary_metrics.json"), summary_rows)
    write_markdown_table(os.path.join(run_dir, "summary_metrics.md"), summary_rows, title="Fixed Benchmark Summary")
    write_manifest(
        os.path.join(run_dir, "experiment_manifest.json"),
        {
            "task": "fixed_benchmark",
            "run_name": args.run_name,
            "modes": modes,
            "seed": args.seed,
            "split_path": args.split_path,
            "dataset_json": args.dataset_json,
            "data_root": args.data_root,
            "gam_ckpt": args.gam_ckpt,
            "texture_ckpt": args.texture_ckpt,
            "texture_preprocess_mode": args.texture_preprocess_mode,
            "alpha": [args.alpha1, args.alpha2, args.alpha3, args.alpha4],
        },
    )


def build_argparser():
    ap = argparse.ArgumentParser(description="Run fixed validation benchmark across multiple modes.")
    ap.add_argument("--dataset_json", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split_path", default="eval/benchmarks/fixed_val_split.json")
    ap.add_argument("--num_samples", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gam_ckpt", required=True)
    ap.add_argument("--texture_ckpt", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--modes", default="token,spatial,hybrid,spatial_bfm_like")
    ap.add_argument("--texture_preprocess_mode", default="crop_tile", choices=["plain_resize", "crop_tile", "plain"])
    ap.add_argument("--alpha1", type=float, default=1.0)
    ap.add_argument("--alpha2", type=float, default=1.0)
    ap.add_argument("--alpha3", type=float, default=0.7)
    ap.add_argument("--alpha4", type=float, default=0.5)
    ap.add_argument("--output_dir", default="eval_outputs")
    ap.add_argument("--run_name", default="step_000000")
    return ap


def main():
    args = build_argparser().parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
