#!/usr/bin/env python3
import argparse
import math
import os
import shutil
import subprocess
import sys
from collections import Counter

import numpy as np

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
from eval.eval_utils import (
    existing_file,
    extract_generated_panel,
    prepare_evaluation_masks,
    safe_open_rgb,
    save_mask_debug,
)
from eval.metrics import (
    compute_clip_i_values,
    compute_fid_from_paths,
    evaluate_full,
)


PER_SAMPLE_METRICS = [
    "ssim",
    "tcf_lab_delta",
    "tcf_hsv_l1",
    "tcf_rgb_l2",
    "tpf_patch_sim",
    "tpf_gram_l1",
    "clip_i_texture",
    "clip_i_real",
    "leak_colored_frac",
    "leak_mean_saturation",
    "leak_value_shift",
    "leak_edge_density",
    "struct_edge_f1",
    "struct_edge_precision",
    "struct_edge_recall",
    "struct_iou",
    "struct_edge_l1",
    "edge_f1",
    "edge_precision",
    "edge_recall",
    "sketch_iou",
    "edge_l1",
]


def parse_modes(value):
    return [item.strip() for item in value.split(",") if item.strip()]


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


def _resolve_path(data_root, value, override_dir=None):
    if not value:
        return None
    candidates = []
    if os.path.isabs(value):
        candidates.append(value)
    if override_dir:
        candidates.extend(
            [
                os.path.join(override_dir, value),
                os.path.join(override_dir, os.path.basename(value)),
            ]
        )
    candidates.append(os.path.join(data_root, value))
    for candidate in candidates:
        if existing_file(candidate):
            return os.path.normpath(candidate)
    return os.path.normpath(candidates[0]) if candidates else None


def sample_paths(args, sample):
    return {
        "target_path": _resolve_path(
            args.data_root, sample.get("target"), args.real_images_dir
        ),
        "texture_path": _resolve_path(
            args.data_root, sample.get("texture"), args.texture_images_dir
        ),
        "sketch_path": _resolve_path(
            args.data_root, sample.get("sketch"), args.sketch_images_dir
        ),
        "mask_path": _resolve_path(
            args.data_root, sample.get("mask"), args.mask_dir
        ),
    }


def run_one_inference(args, sample, mode_name, out_dir, paths):
    uid = sample_uid(sample)
    sample_out = os.path.join(out_dir, mode_name, uid)
    ensure_dir(sample_out)
    dst = os.path.join(sample_out, "generated.png")
    comparison_path = os.path.join(sample_out, "comparison_grid.png")

    if existing_file(dst):
        extract_generated_panel(dst, comparison_path=comparison_path)
        return dst
    if args.metrics_only:
        return None
    if not existing_file(paths["sketch_path"]):
        return None
    if not existing_file(paths["texture_path"]):
        return None

    flags = mode_to_flags(mode_name)
    src = os.path.join(sample_out, os.path.basename(paths["sketch_path"]))
    if existing_file(src):
        shutil.move(src, dst)
        extract_generated_panel(dst, comparison_path=comparison_path)
        return dst

    cmd = [
        "python",
        "inference_IMAGGarment-1.py",
        "--GAM_model_ckpt",
        args.gam_ckpt,
        "--texture_ckpt",
        args.texture_ckpt,
        "--sketch_path",
        paths["sketch_path"],
        "--texture_path",
        paths["texture_path"],
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
    if existing_file(src):
        shutil.move(src, dst)
    if not existing_file(dst):
        return None
    extract_generated_panel(dst, comparison_path=comparison_path)
    return dst


def make_grid(image_paths, save_path, cols=4):
    from PIL import Image

    images = [
        Image.open(path).convert("RGB")
        for path in image_paths
        if existing_file(path)
    ]
    if not images:
        return
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    rows = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (cols * width, rows * height), (255, 255, 255))
    for index, image in enumerate(images):
        grid.paste(
            image.resize((width, height), Image.BICUBIC),
            ((index % cols) * width, (index // cols) * height),
        )
    grid.save(save_path)


def _image_files(directory):
    if not directory or not os.path.isdir(directory):
        return []
    paths = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                paths.append(os.path.join(root, filename))
    return sorted(paths)


def _is_finite(value):
    return isinstance(value, (int, float, np.number)) and math.isfinite(float(value))


def _append_reason(counter, reason):
    if reason:
        counter[str(reason)] += 1


def _aggregate_rows(rows, mode):
    mode_rows = [row for row in rows if row.get("mode") == mode]
    summary = {"mode": mode, "count": len(mode_rows)}
    keys = sorted({key for row in mode_rows for key in row})
    for key in keys:
        values = [float(row[key]) for row in mode_rows if _is_finite(row.get(key))]
        if not values:
            if key in PER_SAMPLE_METRICS:
                summary[f"{key}_mean"] = None
                summary[f"{key}_std"] = None
            continue
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
        summary[f"{key}_valid"] = len(values)
    return summary


def _assign_clip_metric(
    rows,
    reference_key,
    output_key,
    args,
    diagnostics,
    reason_counts,
):
    for row in rows:
        row[output_key] = float("nan")
    pairs = [
        (index, row["gen_path"], row.get(reference_key))
        for index, row in enumerate(rows)
        if existing_file(row.get("gen_path")) and existing_file(row.get(reference_key))
    ]
    diagnostics[f"num_valid_for_{output_key}"] = len(pairs)
    if not pairs:
        reason = f"{output_key}: no valid generated/reference pairs"
        print(f"[benchmark] WARNING: {reason}")
        _append_reason(reason_counts, reason)
        return
    try:
        values = compute_clip_i_values(
            [item[1] for item in pairs],
            [item[2] for item in pairs],
            batch_size=args.clip_batch_size,
            device=args.device,
            model_name=args.clip_model_path,
        )
        for (row_index, _, _), value in zip(pairs, values):
            rows[row_index][output_key] = float(value)
    except Exception as exc:
        reason = f"{output_key} failed: {exc}"
        print(f"[benchmark] WARNING: {reason}")
        _append_reason(reason_counts, reason)


def _write_progress(run_dir, rows):
    write_csv(os.path.join(run_dir, "metrics_per_sample.csv"), rows)
    write_json(os.path.join(run_dir, "metrics_per_sample.json"), rows)
    write_csv(os.path.join(run_dir, "per_image_metrics.csv"), rows)
    write_json(os.path.join(run_dir, "per_image_metrics.json"), rows)


def run_benchmark(args):
    split = create_or_load_fixed_split(
        args.dataset_json,
        args.split_path,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    run_dir = os.path.join(args.output_dir, args.run_name)
    ensure_dir(run_dir)
    modes = parse_modes(args.modes)
    reason_counts = Counter()

    print(
        f"[benchmark] split_samples={len(split)}, requested_samples={args.num_samples}, "
        f"split_path={args.split_path}"
    )
    if len(split) != args.num_samples:
        raise RuntimeError(
            f"Fixed split contains {len(split)} samples, but {args.num_samples} "
            f"were requested. Use the same split as the compared experiments."
        )

    manifest = {
        "task": "fixed_benchmark",
        "status": "running",
        "run_name": args.run_name,
        "modes": modes,
        "seed": args.seed,
        "requested_samples": args.num_samples,
        "split_samples": len(split),
        "split_path": args.split_path,
        "dataset_json": args.dataset_json,
        "data_root": args.data_root,
        "gam_ckpt": args.gam_ckpt,
        "texture_ckpt": args.texture_ckpt,
        "clip_model_path": args.clip_model_path,
        "texture_preprocess_mode": args.texture_preprocess_mode,
        "alpha": [args.alpha1, args.alpha2, args.alpha3, args.alpha4],
        "metrics_only": args.metrics_only,
    }
    write_manifest(os.path.join(run_dir, "experiment_manifest.json"), manifest)

    rows = []
    for mode in modes:
        for sample_index, sample in enumerate(split, start=1):
            uid = sample_uid(sample)
            paths = sample_paths(args, sample)
            print(
                f"[benchmark] generating mode={mode}, "
                f"sample={sample_index}/{len(split)}, uid={uid}"
            )
            try:
                gen_path = run_one_inference(
                    args, sample, mode, run_dir, paths=paths
                )
            except Exception as exc:
                gen_path = None
                reason = f"inference failed for {uid}: {exc}"
                print(f"[benchmark] WARNING: {reason}")
                _append_reason(reason_counts, reason)
            row = {
                "mode": mode,
                "uid": uid,
                "gen_path": gen_path
                or os.path.join(run_dir, mode, uid, "generated.png"),
                **paths,
            }
            for metric in PER_SAMPLE_METRICS:
                row[metric] = float("nan")
            rows.append(row)

        make_grid(
            [
                row["gen_path"]
                for row in rows
                if row["mode"] == mode and existing_file(row["gen_path"])
            ],
            os.path.join(run_dir, f"grid_{mode}.png"),
            cols=4,
        )

    diagnostics = {
        "num_samples": len(rows),
        "num_generated_found": sum(existing_file(row["gen_path"]) for row in rows),
        "num_real_found": sum(existing_file(row["target_path"]) for row in rows),
        "num_texture_found": sum(existing_file(row["texture_path"]) for row in rows),
        "num_sketch_found": sum(existing_file(row["sketch_path"]) for row in rows),
        "num_mask_found": sum(existing_file(row["mask_path"]) for row in rows),
        "num_valid_for_fid": 0,
        "num_valid_for_clip_i": 0,
        "num_valid_for_clip_i_texture": 0,
        "num_valid_for_clip_i_real": 0,
        "num_valid_for_leakage": 0,
        "num_valid_for_structure": 0,
        "average_garment_mask_area": None,
        "average_outside_mask_area": None,
        "average_boundary_mask_area": None,
        "number_of_empty_garment_masks": 0,
        "number_of_empty_outside_masks": 0,
        "number_of_empty_boundary_masks": 0,
        "skipped_metrics_and_reasons": {},
    }

    debug_dir = os.path.join(run_dir, "debug_masks")
    for row_index, row in enumerate(rows, start=1):
        print(
            f"[benchmark] evaluating sample={row_index}/{len(rows)}, uid={row['uid']}"
        )
        generated = safe_open_rgb(row["gen_path"])
        if generated is None:
            reason = f"generated image missing: {row['gen_path']}"
            _append_reason(reason_counts, reason)
            row["metric_warnings"] = [reason]
            _write_progress(run_dir, rows)
            continue

        mask_bundle = prepare_evaluation_masks(
            generated.size,
            mask_path=row["mask_path"],
            sketch_path=row["sketch_path"],
            target_path=row["target_path"],
            gen_path=row["gen_path"],
        )
        row.update(mask_bundle["stats"])
        if row["garment_mask_pixels"] < args.min_valid_pixels:
            diagnostics["number_of_empty_garment_masks"] += 1
        if row["outside_mask_pixels"] < args.min_valid_pixels:
            diagnostics["number_of_empty_outside_masks"] += 1
        if row["boundary_mask_pixels"] < args.min_valid_pixels:
            diagnostics["number_of_empty_boundary_masks"] += 1
        if args.fail_on_empty_masks and (
            row["garment_mask_pixels"] < args.min_valid_pixels
            or row["outside_mask_pixels"] < args.min_valid_pixels
            or row["boundary_mask_pixels"] < args.min_valid_pixels
        ):
            raise RuntimeError(
                f"empty evaluation mask for uid={row['uid']}: "
                f"garment={row['garment_mask_pixels']}, "
                f"outside={row['outside_mask_pixels']}, "
                f"boundary={row['boundary_mask_pixels']}"
            )
        if row_index <= args.debug_save_masks:
            save_mask_debug(mask_bundle, debug_dir, row["uid"])

        metrics = evaluate_full(
            row["gen_path"],
            target_path=row["target_path"],
            texture_path=row["texture_path"],
            sketch_path=row["sketch_path"],
            mask_path=row["mask_path"],
            compute_leakage=bool(args.compute_leakage),
            compute_structure=bool(args.compute_structure),
            min_valid_pixels=args.min_valid_pixels,
            mask_bundle=mask_bundle,
        )
        metric_warnings = metrics.pop("metric_warnings", [])
        metric_warnings.extend(mask_bundle.get("warnings", []))
        for reason in metric_warnings:
            _append_reason(reason_counts, reason)
        row.update(metrics)
        row["metric_warnings"] = sorted(set(metric_warnings))
        _write_progress(run_dir, rows)

    if args.compute_clip_i:
        _assign_clip_metric(
            rows,
            "texture_path",
            "clip_i_texture",
            args,
            diagnostics,
            reason_counts,
        )
        _assign_clip_metric(
            rows,
            "target_path",
            "clip_i_real",
            args,
            diagnostics,
            reason_counts,
        )
    else:
        _append_reason(reason_counts, "CLIP-I disabled by --compute_clip_i 0")
    diagnostics["num_valid_for_clip_i"] = max(
        diagnostics["num_valid_for_clip_i_texture"],
        diagnostics["num_valid_for_clip_i_real"],
    )

    fid_value = float("nan")
    if args.compute_fid:
        gen_paths = [row["gen_path"] for row in rows if existing_file(row["gen_path"])]
        real_paths = [
            row["target_path"]
            for row in rows
            if existing_file(row["target_path"])
        ]
        if not real_paths:
            real_paths = _image_files(args.real_images_dir)
        diagnostics["num_valid_for_fid"] = min(len(gen_paths), len(real_paths))
        if len(gen_paths) < 2 or len(real_paths) < 2:
            reason = (
                f"FID skipped: generated={len(gen_paths)}, real={len(real_paths)}; "
                "at least 2 each are required"
            )
            print(f"[benchmark] WARNING: {reason}")
            _append_reason(reason_counts, reason)
        else:
            try:
                fid_value = compute_fid_from_paths(
                    gen_paths,
                    real_paths,
                    batch_size=args.fid_batch_size,
                    device=args.device,
                )
            except Exception as exc:
                reason = f"FID failed: {exc}"
                print(f"[benchmark] WARNING: {reason}")
                _append_reason(reason_counts, reason)
    else:
        _append_reason(reason_counts, "FID disabled by --compute_fid 0")

    diagnostics["num_valid_for_leakage"] = sum(
        _is_finite(row.get("leak_colored_frac")) for row in rows
    )
    diagnostics["num_valid_for_structure"] = sum(
        _is_finite(row.get("struct_edge_f1")) for row in rows
    )
    for source_key, output_key in (
        ("garment_mask_area", "average_garment_mask_area"),
        ("outside_mask_area", "average_outside_mask_area"),
        ("boundary_mask_area", "average_boundary_mask_area"),
    ):
        values = [float(row[source_key]) for row in rows if _is_finite(row.get(source_key))]
        diagnostics[output_key] = float(np.mean(values)) if values else None

    diagnostics["skipped_metrics_and_reasons"] = dict(reason_counts)
    for name in (
        "num_valid_for_fid",
        "num_valid_for_clip_i",
        "num_valid_for_leakage",
        "num_valid_for_structure",
    ):
        if diagnostics[name] == 0:
            print(f"[benchmark] WARNING: {name}=0; related summary metrics are null")

    summary_rows = []
    for mode in modes:
        summary = _aggregate_rows(rows, mode)
        summary["FID"] = fid_value if _is_finite(fid_value) else None
        summary["FID_std"] = None
        summary_rows.append(summary)

    _write_progress(run_dir, rows)
    write_csv(os.path.join(run_dir, "metrics_summary.csv"), summary_rows)
    write_json(os.path.join(run_dir, "metrics_summary.json"), summary_rows)
    write_csv(os.path.join(run_dir, "summary_metrics.csv"), summary_rows)
    write_json(os.path.join(run_dir, "summary_metrics.json"), summary_rows)
    write_markdown_table(
        os.path.join(run_dir, "summary_metrics.md"),
        summary_rows,
        title="Fixed Benchmark Summary",
    )
    write_json(os.path.join(run_dir, "diagnostics.json"), diagnostics)

    manifest.update(
        {
            "status": "completed",
            "generated_images": diagnostics["num_generated_found"],
            "diagnostics_path": os.path.join(run_dir, "diagnostics.json"),
        }
    )
    write_manifest(os.path.join(run_dir, "experiment_manifest.json"), manifest)
    return summary_rows, diagnostics


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Run fixed validation benchmark across multiple modes."
    )
    parser.add_argument("--dataset_json", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument(
        "--split_path", default="eval/benchmarks/fixed_val_split.json"
    )
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gam_ckpt", required=True)
    parser.add_argument("--texture_ckpt", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--modes", default="token,spatial,hybrid,spatial_bfm_like"
    )
    parser.add_argument(
        "--texture_preprocess_mode",
        default="crop_tile",
        choices=["plain_resize", "crop_tile", "plain"],
    )
    parser.add_argument("--alpha1", type=float, default=1.0)
    parser.add_argument("--alpha2", type=float, default=1.0)
    parser.add_argument("--alpha3", type=float, default=0.7)
    parser.add_argument("--alpha4", type=float, default=0.5)
    parser.add_argument("--output_dir", default="eval_outputs")
    parser.add_argument("--run_name", default="step_000000")
    parser.add_argument("--real_images_dir", default=None)
    parser.add_argument("--texture_images_dir", default=None)
    parser.add_argument("--sketch_images_dir", default=None)
    parser.add_argument("--mask_dir", default=None)
    parser.add_argument(
        "--clip_model_path", default="openai/clip-vit-large-patch14"
    )
    parser.add_argument("--compute_fid", type=int, choices=[0, 1], default=1)
    parser.add_argument("--compute_clip_i", type=int, choices=[0, 1], default=1)
    parser.add_argument("--compute_leakage", type=int, choices=[0, 1], default=1)
    parser.add_argument("--compute_structure", type=int, choices=[0, 1], default=1)
    parser.add_argument("--debug_save_masks", type=int, default=0)
    parser.add_argument(
        "--fail_on_empty_masks", type=int, choices=[0, 1], default=0
    )
    parser.add_argument("--min_valid_pixels", type=int, default=50)
    parser.add_argument("--clip_batch_size", type=int, default=16)
    parser.add_argument("--fid_batch_size", type=int, default=16)
    parser.add_argument(
        "--metrics_only",
        action="store_true",
        help="Only recompute metrics from existing generated.png files.",
    )
    return parser


def main():
    args = build_argparser().parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
