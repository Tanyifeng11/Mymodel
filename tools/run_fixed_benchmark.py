#!/usr/bin/env python3
import argparse
import math
import os
import re
import shutil
import subprocess
import sys
from collections import Counter

import numpy as np
from PIL import Image

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
from eval.distribution_diagnostics import compute_kid_from_features
from eval.metrics import (
    compute_clip_i_values,
    compute_fid_from_paths,
    evaluate_full,
    extract_inception_features,
)
from garment_mask_utils import mask_backend_info
from color_conflict_utils import (
    compute_color_conflict,
    conflict_bucket,
    delta_e_rgb,
    dominant_rgb_from_pil,
    summarize_conflict_rows,
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
    "prompt_color_delta_e",
    "target_color_delta_e",
    "gt_floor_delta_e",
    "edge_f1",
    "edge_precision",
    "edge_recall",
    "sketch_iou",
    "edge_l1",
]


def parse_modes(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def generation_seed_for_sample(base_seed, sample):
    """Return a stable, distinct seed for each fixed benchmark sample."""
    return int(base_seed) + int(sample["sample_id"])


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


def experiment_to_flags(run_name, args):
    configs = {
        "e0_baseline": {
            "texture_condition_mode": "token",
            "layer_group_enabled": 0,
            "use_texture_gate": 0,
        },
        "e1_grouped": {
            "texture_condition_mode": "token",
            "layer_group_enabled": 1,
            "use_texture_gate": 0,
        },
        "e2a_region": {
            "texture_condition_mode": "token",
            "layer_group_enabled": 1,
            "use_texture_gate": 0,
            "use_palette_tokens": 0,
        },
        "e3a_palette_token": {
            "texture_condition_mode": "token",
            "layer_group_enabled": 1,
            "use_texture_gate": 0,
            "use_palette_tokens": 1,
            "num_palette_tokens": 4,
        },
        "e3": {
            "texture_condition_mode": "token",
            "layer_group_enabled": 1,
            "use_texture_gate": 0,
            "use_palette_tokens": 1,
            "num_palette_tokens": 4,
        },
        "e3_palette_token": {
            "texture_condition_mode": "token",
            "layer_group_enabled": 1,
            "use_texture_gate": 0,
            "use_palette_tokens": 1,
            "num_palette_tokens": 4,
        },
        "e4a_balanced_gate": {
            "texture_condition_mode": "token",
            "layer_group_enabled": 1,
            "use_texture_gate": 0,
            "use_palette_tokens": 1,
            "num_palette_tokens": 4,
            "use_balanced_fusion_gate": 1,
            "balanced_gate_hidden_dim": 64,
            "balanced_gate_scale": 0.2,
            "balanced_gate_min": 0.8,
            "balanced_gate_max": 1.2,
        },
        "e4b_boundary_fid_safe": {
            "texture_condition_mode": "token",
            "layer_group_enabled": 1,
            "use_texture_gate": 0,
            "use_palette_tokens": 1,
            "num_palette_tokens": 4,
            "use_balanced_fusion_gate": 1,
            "balanced_gate_hidden_dim": 64,
            "balanced_gate_scale": 0.2,
            "balanced_gate_min": 0.8,
            "balanced_gate_max": 1.2,
        },
        "e4b_boundary_safe": {
            "texture_condition_mode": "token",
            "layer_group_enabled": 1,
            "use_texture_gate": 0,
            "use_palette_tokens": 1,
            "num_palette_tokens": 4,
            "use_balanced_fusion_gate": 1,
            "balanced_gate_hidden_dim": 64,
            "balanced_gate_scale": 0.2,
            "balanced_gate_min": 0.8,
            "balanced_gate_max": 1.2,
        },
        "E4d_lite_conflict_aware_gate": {
            "texture_condition_mode": "token",
            "layer_group_enabled": 1,
            "use_texture_gate": 0,
            "use_palette_tokens": 1,
            "num_palette_tokens": 4,
            "use_balanced_fusion_gate": 1,
            "balanced_gate_hidden_dim": 64,
            "balanced_gate_scale": 0.2,
            "balanced_gate_min": 0.8,
            "balanced_gate_max": 1.2,
            "use_conflict_aware_gate": 1,
        },
        "E4d_lite_conflict_aware_gate_v2": {
            "texture_condition_mode": "token",
            "layer_group_enabled": 1,
            "use_texture_gate": 0,
            "use_palette_tokens": 1,
            "num_palette_tokens": 4,
            "use_balanced_fusion_gate": 1,
            "balanced_gate_hidden_dim": 64,
            "balanced_gate_scale": 0.2,
            "balanced_gate_min": 0.8,
            "balanced_gate_max": 1.2,
            "use_conflict_aware_gate": 1,
        },
        "e4d_lite_conflict_aware_gate_v2": {
            "texture_condition_mode": "token",
            "layer_group_enabled": 1,
            "use_texture_gate": 0,
            "use_palette_tokens": 1,
            "num_palette_tokens": 4,
            "use_balanced_fusion_gate": 1,
            "balanced_gate_hidden_dim": 64,
            "balanced_gate_scale": 0.2,
            "balanced_gate_min": 0.8,
            "balanced_gate_max": 1.2,
            "use_conflict_aware_gate": 1,
        },
        "e4d_lite_conflict_aware_gate": {
            "texture_condition_mode": "token",
            "layer_group_enabled": 1,
            "use_texture_gate": 0,
            "use_palette_tokens": 1,
            "num_palette_tokens": 4,
            "use_balanced_fusion_gate": 1,
            "balanced_gate_hidden_dim": 64,
            "balanced_gate_scale": 0.2,
            "balanced_gate_min": 0.8,
            "balanced_gate_max": 1.2,
            "use_conflict_aware_gate": 1,
        },
        "e2b_gate": {
            "texture_condition_mode": "token",
            "layer_group_enabled": 1,
            "use_texture_gate": 1,
            "use_palette_tokens": 0,
            "gate_type": "layer",
            "gate_init": "identity",
            "gate_min": 0.7,
            "gate_max": 1.3,
        },
        "e2b_safe_gate": {
            "texture_condition_mode": "token",
            "layer_group_enabled": 1,
            "use_texture_gate": 1,
            "gate_type": "layer",
            "gate_init": "identity",
            "gate_min": 0.7,
            "gate_max": 1.3,
        },
        "e2b_color_safe_gate": {
            "texture_condition_mode": "token",
            "layer_group_enabled": 1,
            "use_texture_gate": 1,
            "gate_type": "layer",
            "gate_init": "identity",
            "gate_min": 0.7,
            "gate_max": 1.3,
        },
        "e5_tcpm_lite": {
            "texture_condition_mode": "token",
            "layer_group_enabled": 1,
            "use_texture_gate": 1,
            "gate_type": "layer",
            "gate_init": "identity",
            "gate_min": 0.7,
            "gate_max": 1.3,
            "use_tcpm_lite": 1,
        },
        "e5_tcpm_lite_e5": {
            "texture_condition_mode": "token",
            "layer_group_enabled": 1,
            "use_texture_gate": 1,
            "gate_type": "layer",
            "gate_init": "identity",
            "gate_min": 0.7,
            "gate_max": 1.3,
            "use_tcpm_lite": 1,
        },
        "e5_tcpm_lite_e12": {
            "texture_condition_mode": "token",
            "layer_group_enabled": 1,
            "use_texture_gate": 1,
            "gate_type": "layer",
            "gate_init": "identity",
            "gate_min": 0.7,
            "gate_max": 1.3,
            "use_tcpm_lite": 1,
        },
        "e7a": {
            "texture_condition_mode": "token",
            "layer_group_enabled": 1,
            "use_texture_gate": 1,
            "gate_type": "layer",
            "gate_init": "identity",
            "gate_min": 0.7,
            "gate_max": 1.3,
            "use_tcpm_lite": 1,
            "use_aa_tcr_fuse": 1,
        },
        "e5b_tcpm_detail_adapter": {
            "texture_condition_mode": "token",
            "layer_group_enabled": 1,
            "use_texture_gate": 1,
            "gate_type": "layer",
            "gate_init": "identity",
            "gate_min": 0.7,
            "gate_max": 1.3,
            "use_tcpm_lite": 1,
        },
    }
    config = dict(configs.get(run_name, {}))
    config.setdefault("use_texture_gate", args.use_texture_gate)
    config.setdefault("use_palette_tokens", args.use_palette_tokens)
    config.setdefault("num_palette_tokens", args.num_palette_tokens)
    config.setdefault("layer_group_enabled", args.layer_group_enabled)
    config.setdefault("gate_type", args.gate_type)
    config.setdefault("gate_init", args.gate_init)
    config.setdefault("gate_min", args.gate_min)
    config.setdefault("gate_max", args.gate_max)
    config.setdefault("use_balanced_fusion_gate", args.use_balanced_fusion_gate)
    config.setdefault("use_tcpm_lite", args.use_tcpm_lite)
    config.setdefault("use_aa_tcr_fuse", args.use_aa_tcr_fuse)
    config.setdefault("use_text_guided_resampler", args.use_text_guided_resampler)
    config.setdefault("balanced_gate_hidden_dim", args.balanced_gate_hidden_dim)
    config.setdefault("balanced_gate_scale", args.balanced_gate_scale)
    config.setdefault("balanced_gate_min", args.balanced_gate_min)
    config.setdefault("balanced_gate_max", args.balanced_gate_max)
    config.setdefault("use_conflict_aware_gate", args.use_conflict_aware_gate)
    return config


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


def _sample_output_paths(out_dir, mode_name, sample):
    uid = sample_uid(sample)
    sample_id = sample["sample_id"]
    sample_out = os.path.join(out_dir, mode_name, f"{sample_id}_{uid}")
    return {
        "uid": uid,
        "sample_id": sample_id,
        "sample_out": sample_out,
        "generated": os.path.join(sample_out, f"generated_{sample_id}.png"),
        "legacy_generated": os.path.join(sample_out, "generated.png"),
        "comparison": os.path.join(sample_out, f"comparison_{sample_id}.png"),
    }


def _sidecar_text_path(image_path):
    stem, _ = os.path.splitext(image_path)
    return f"{stem}.txt"


def _inference_mask_path(sample_out, paths):
    sketch_path = paths.get("sketch_path")
    if not sketch_path:
        return None
    stem = os.path.splitext(os.path.basename(sketch_path))[0]
    return os.path.join(sample_out, f"{stem}_mask.png")


def _sample_text_description(args, sample, mode_name, paths, role, image_path, status):
    flags = mode_to_flags(mode_name)
    experiment_flags = experiment_to_flags(args.run_name, args)
    texture_condition_mode = experiment_flags.get(
        "texture_condition_mode", flags["texture_condition_mode"]
    )
    lines = [
        f"role: {role}",
        f"run_name: {args.run_name}",
        f"mode: {mode_name}",
        f"sample_id: {sample['sample_id']}",
        f"dataset_index: {sample.get('idx')}",
        f"uid: {sample_uid(sample)}",
        f"generation_status: {status}",
        f"generation_seed: {generation_seed_for_sample(args.generation_seed, sample)}",
        f"prompt: {sample.get('prompt', '')}",
        f"image_path: {image_path}",
        f"target_path: {paths.get('target_path')}",
        f"sketch_path: {paths.get('sketch_path')}",
        f"texture_path: {paths.get('texture_path')}",
        f"mask_path: {paths.get('mask_path')}",
        f"texture_condition_mode: {texture_condition_mode}",
        f"texture_preprocess_mode: {args.texture_preprocess_mode}",
        f"fusion_type: {flags['fusion_type']}",
        f"layer_group_enabled: {int(experiment_flags['layer_group_enabled'])}",
        f"use_texture_gate: {int(experiment_flags['use_texture_gate'])}",
        f"use_palette_tokens: {int(experiment_flags['use_palette_tokens'])}",
        f"num_palette_tokens: {int(experiment_flags['num_palette_tokens'])}",
        f"gate_type: {experiment_flags['gate_type']}",
        f"gate_init: {experiment_flags['gate_init']}",
        f"gate_min: {experiment_flags['gate_min']}",
        f"gate_max: {experiment_flags['gate_max']}",
        f"use_balanced_fusion_gate: {int(experiment_flags['use_balanced_fusion_gate'])}",
        f"balanced_gate_scale: {experiment_flags['balanced_gate_scale']}",
        f"use_conflict_aware_gate: {int(experiment_flags['use_conflict_aware_gate'])}",
        f"use_aa_tcr_fuse: {int(experiment_flags['use_aa_tcr_fuse'])}",
        f"use_text_guided_resampler: {int(experiment_flags['use_text_guided_resampler'])}",
        f"conflict_texture_suppress_strength: {args.conflict_texture_suppress_strength}",
        f"conflict_palette_suppress_strength: {args.conflict_palette_suppress_strength}",
        f"conflict_deltae_norm: {args.conflict_deltae_norm}",
        f"conflict_threshold: {args.conflict_threshold}",
        f"alpha1: {args.alpha1}",
        f"alpha2: {args.alpha2}",
        f"alpha3: {args.alpha3}",
        f"alpha4: {args.alpha4}",
    ]
    return "\n".join(lines) + "\n"


def _sample_conflict_info(args, sample, paths):
    explicit_score = sample.get("color_conflict_score")
    if explicit_score is not None:
        try:
            score = float(explicit_score)
        except (TypeError, ValueError):
            score = 0.0
        has_text = bool(sample.get("has_text_color", score > 0.0))
        return {
            "text_color": sample.get("text_color", ""),
            "text_color_rgb": sample.get("text_color_rgb", [0, 0, 0]),
            "ref_palette_rgb": sample.get("ref_palette_rgb", [0, 0, 0]),
            "has_text_color": has_text,
            "color_conflict_score": score,
            "color_delta_e": sample.get("color_delta_e", None),
            "conflict_bucket": conflict_bucket(score, has_text),
        }
    ref_image = safe_open_rgb(paths.get("texture_path"))
    return compute_color_conflict(
        sample.get("prompt", ""),
        ref_image=ref_image,
        deltae_norm=args.conflict_deltae_norm,
    )


def _write_image_sidecar(image_path, description):
    if not image_path or not existing_file(image_path):
        return None
    text_path = _sidecar_text_path(image_path)
    ensure_dir(os.path.dirname(text_path) or ".")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(description)
    return text_path


def _write_sample_image_sidecars(
    args, sample, mode_name, paths, output_paths, status
):
    if not args.write_text_sidecars:
        return
    sample_out = output_paths["sample_out"]
    image_roles = [
        ("generated", output_paths["generated"]),
        ("comparison", output_paths["comparison"]),
        ("inference_mask", _inference_mask_path(sample_out, paths)),
    ]
    for role, image_path in image_roles:
        description = _sample_text_description(
            args, sample, mode_name, paths, role, image_path, status
        )
        _write_image_sidecar(image_path, description)


def _existing_generation_candidates(args, sample, mode_name, out_dir):
    current = _sample_output_paths(out_dir, mode_name, sample)
    uid = current["uid"]
    sample_id = current["sample_id"]
    candidates = [
        current["generated"],
        current["legacy_generated"],
    ]
    if args.reuse_from_dir:
        source_mode = os.path.join(args.reuse_from_dir, mode_name)
        candidates.extend(
            [
                os.path.join(
                    source_mode,
                    f"{sample_id}_{uid}",
                    f"generated_{sample_id}.png",
                ),
                os.path.join(source_mode, f"{sample_id}_{uid}", "generated.png"),
                os.path.join(source_mode, uid, f"generated_{sample_id}.png"),
                os.path.join(source_mode, uid, "generated.png"),
            ]
        )
    return current, [path for path in candidates if existing_file(path)]


def run_one_inference(args, sample, mode_name, out_dir, paths):
    output_paths, existing_candidates = _existing_generation_candidates(
        args, sample, mode_name, out_dir
    )
    sample_out = output_paths["sample_out"]
    ensure_dir(sample_out)
    dst = output_paths["generated"]
    comparison_path = output_paths["comparison"]

    if existing_candidates and not args.overwrite:
        if not args.skip_existing and not args.resume_generation:
            raise FileExistsError(
                f"generated image already exists for sample_id="
                f"{sample['sample_id']}; enable --skip_existing 1 or "
                "--resume_generation 1, or use --overwrite 1"
            )
        source = existing_candidates[0]
        if os.path.normcase(os.path.abspath(source)) != os.path.normcase(
            os.path.abspath(dst)
        ):
            if os.path.normcase(os.path.dirname(os.path.abspath(source))) == (
                os.path.normcase(os.path.abspath(sample_out))
            ):
                shutil.move(source, dst)
            else:
                shutil.copy2(source, dst)
            status = "reused_existing"
        else:
            status = "skipped_existing"
        extract_generated_panel(dst, comparison_path=comparison_path)
        _write_sample_image_sidecars(
            args, sample, mode_name, paths, output_paths, status
        )
        return dst, status

    if args.metrics_only:
        return None, "missing"
    if not existing_file(paths["sketch_path"]):
        return None, "missing_sketch"
    if not existing_file(paths["texture_path"]):
        return None, "missing_texture"

    flags = mode_to_flags(mode_name)
    generation_seed = generation_seed_for_sample(args.generation_seed, sample)
    experiment_flags = experiment_to_flags(args.run_name, args)
    texture_condition_mode = experiment_flags.get(
        "texture_condition_mode", flags["texture_condition_mode"]
    )
    src = os.path.join(sample_out, os.path.basename(paths["sketch_path"]))
    if existing_file(src) and not args.overwrite:
        shutil.move(src, dst)
        extract_generated_panel(dst, comparison_path=comparison_path)
        _write_sample_image_sidecars(
            args, sample, mode_name, paths, output_paths, "reused_existing"
        )
        return dst, "reused_existing"

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
        "--seed",
        str(generation_seed),
        "--texture_condition_mode",
        texture_condition_mode,
        "--fusion_type",
        flags["fusion_type"],
        "--layer_group_enabled",
        str(int(experiment_flags["layer_group_enabled"])),
        "--texture_preprocess_mode",
        args.texture_preprocess_mode,
        "--use_texture_gate",
        str(int(experiment_flags["use_texture_gate"])),
        "--use_palette_tokens",
        str(int(experiment_flags["use_palette_tokens"])),
        "--num_palette_tokens",
        str(int(experiment_flags["num_palette_tokens"])),
        "--gate_type",
        experiment_flags["gate_type"],
        "--gate_init",
        experiment_flags["gate_init"],
        "--gate_min",
        str(experiment_flags["gate_min"]),
        "--gate_max",
        str(experiment_flags["gate_max"]),
        "--use_balanced_fusion_gate",
        str(int(experiment_flags["use_balanced_fusion_gate"])),
        "--balanced_gate_hidden_dim",
        str(int(experiment_flags["balanced_gate_hidden_dim"])),
        "--balanced_gate_scale",
        str(experiment_flags["balanced_gate_scale"]),
        "--balanced_gate_min",
        str(experiment_flags["balanced_gate_min"]),
        "--balanced_gate_max",
        str(experiment_flags["balanced_gate_max"]),
        "--use_conflict_aware_gate",
        str(int(experiment_flags["use_conflict_aware_gate"])),
        "--use_tcpm_lite",
        str(int(experiment_flags["use_tcpm_lite"])),
        "--use_aa_tcr_fuse",
        str(int(experiment_flags["use_aa_tcr_fuse"])),
        "--use_text_guided_resampler",
        str(int(experiment_flags["use_text_guided_resampler"])),
        "--conflict_texture_suppress_strength",
        str(args.conflict_texture_suppress_strength),
        "--conflict_palette_suppress_strength",
        str(args.conflict_palette_suppress_strength),
        "--conflict_deltae_norm",
        str(args.conflict_deltae_norm),
        "--conflict_threshold",
        str(args.conflict_threshold),
        "--alpha1",
        str(args.alpha1),
        "--alpha2",
        str(args.alpha2),
        "--alpha3",
        str(args.alpha3),
        "--alpha4",
        str(args.alpha4),
    ]
    if args.save_balanced_gate_trace:
        trace_path = os.path.join(sample_out, "gate_trace.jsonl")
        cmd.extend(
            [
                "--balanced_gate_trace_path",
                trace_path,
                "--balanced_gate_trace_sample_id",
                sample["sample_id"],
            ]
        )
    subprocess.run(cmd, check=True)
    if existing_file(src):
        shutil.move(src, dst)
    if not existing_file(dst):
        return None, "generation_missing_output"
    extract_generated_panel(dst, comparison_path=comparison_path)
    _write_sample_image_sidecars(
        args, sample, mode_name, paths, output_paths, "generated"
    )
    return dst, "generated"


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


def _aggregate_rows(rows, mode, category=None):
    mode_rows = [
        row
        for row in rows
        if row.get("mode") == mode
        and (category is None or row.get("category") == category)
    ]
    summary = {"mode": mode, "count": len(mode_rows)}
    if category is not None:
        summary["category"] = category
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


def _aggregate_conflict_buckets(rows):
    bucket_order = [
        "no_text_color",
        "low_conflict",
        "mid_conflict",
        "high_conflict",
    ]
    output = []
    for mode in sorted({row.get("mode") for row in rows}):
        mode_rows = [row for row in rows if row.get("mode") == mode]
        for bucket in bucket_order:
            bucket_rows = [row for row in mode_rows if row.get("conflict_bucket") == bucket]
            summary = {
                "mode": mode,
                "conflict_bucket": bucket,
                "num_samples": len(bucket_rows),
            }
            for key in (
                "color_conflict_score",
                "tcf_lab_delta",
                "tcf_hsv_l1",
                "tcf_rgb_l2",
                "tpf_patch_sim",
                "leak_colored_frac",
                "leak_edge_density",
                "edge_f1",
                "sketch_iou",
                "edge_l1",
                "clip_i_real",
                "clip_i_texture",
                "prompt_color_delta_e",
            ):
                values = [float(row[key]) for row in bucket_rows if _is_finite(row.get(key))]
                summary[f"{key}_mean"] = float(np.mean(values)) if values else None
                summary[f"{key}_valid"] = len(values)
            output.append(summary)
    return output


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


def _evaluation_text_description(row, role, image_path):
    lines = [
        f"role: {role}",
        f"run_name: {row.get('run_name')}",
        f"mode: {row.get('mode')}",
        f"sample_id: {row.get('sample_id')}",
        f"dataset_index: {row.get('dataset_index')}",
        f"uid: {row.get('uid')}",
        f"prompt: {row.get('prompt', '')}",
        f"image_path: {image_path}",
        f"source_gen_path: {row.get('source_gen_path')}",
        f"source_target_path: {row.get('source_target_path')}",
        f"texture_path: {row.get('texture_path')}",
        f"sketch_path: {row.get('sketch_path')}",
        f"evaluation_protocol: {row.get('evaluation_protocol')}",
    ]
    return "\n".join(lines) + "\n"


def _grid_text_description(rows, mode, grid_path):
    lines = [
        "role: generated_grid",
        f"run_name: {rows[0].get('run_name') if rows else ''}",
        f"mode: {mode}",
        f"image_path: {grid_path}",
        f"num_images: {len(rows)}",
        "samples:",
    ]
    for row in rows:
        lines.append(
            f"- sample_id={row.get('sample_id')} uid={row.get('uid')} "
            f"prompt={row.get('prompt', '')}"
        )
    return "\n".join(lines) + "\n"


def _prepare_evaluation_images(rows, metrics_dir, resize_size, write_text_sidecars=True):
    from PIL import Image

    generated_dir = os.path.join(metrics_dir, "generated")
    real_dir = os.path.join(metrics_dir, "real")
    ensure_dir(generated_dir)
    ensure_dir(real_dir)

    for row in rows:
        source_gen = row["gen_path"]
        source_target = row["target_path"]
        row["source_gen_path"] = source_gen
        row["source_target_path"] = source_target
        filename = f"{row['mode']}_{row['sample_id']}.png"
        evaluation_gen = os.path.join(generated_dir, filename)
        evaluation_target = os.path.join(real_dir, filename)

        if existing_file(source_gen) and not existing_file(evaluation_gen):
            with Image.open(source_gen) as source_image:
                image = source_image.convert("RGB")
            if resize_size:
                image.resize((resize_size, resize_size), Image.BICUBIC).save(
                    evaluation_gen
                )
            else:
                image.save(evaluation_gen)
        if existing_file(source_target) and not existing_file(evaluation_target):
            with Image.open(source_target) as source_image:
                image = source_image.convert("RGB")
            if resize_size:
                image.resize((resize_size, resize_size), Image.BICUBIC).save(
                    evaluation_target
                )
            else:
                image.save(evaluation_target)

        row["gen_path"] = evaluation_gen
        row["target_path"] = evaluation_target
        row["evaluation_protocol"] = (
            f"resize_generated_real_to_{resize_size}"
            if resize_size
            else "original_image_size"
        )
        if write_text_sidecars:
            _write_image_sidecar(
                evaluation_gen,
                _evaluation_text_description(
                    row, "evaluation_generated", evaluation_gen
                ),
            )
            _write_image_sidecar(
                evaluation_target,
                _evaluation_text_description(row, "evaluation_real", evaluation_target),
            )
    return rows


def run_benchmark(args):
    if args.progress_write_interval < 1:
        raise ValueError("progress_write_interval must be at least 1")
    sample_id_end = (
        args.num_samples if args.sample_id_end is None else args.sample_id_end
    )
    split = create_or_load_fixed_split(
        args.dataset_json,
        args.split_path,
        num_samples=args.num_samples,
        seed=args.seed,
        sample_id_start=args.sample_id_start,
        sample_id_end=sample_id_end,
    )
    run_dir = os.path.join(args.output_dir, args.run_name)
    ensure_dir(run_dir)
    metrics_dir = args.metrics_output_dir or run_dir
    ensure_dir(metrics_dir)
    modes = parse_modes(args.modes)
    reason_counts = Counter()
    expected_num_samples = len(split) * len(modes)
    expected_range_size = sample_id_end - args.sample_id_start
    sample_ids = [sample["sample_id"] for sample in split]
    duplicate_sample_ids = sorted(
        sample_id
        for sample_id, count in Counter(sample_ids).items()
        if count > 1
    )
    existing_before_generation = 0
    for mode in modes:
        for sample in split:
            _, candidates = _existing_generation_candidates(
                args, sample, mode, run_dir
            )
            existing_before_generation += int(bool(candidates))

    print(
        f"[benchmark] split_samples={len(split)}, requested_range="
        f"[{args.sample_id_start}, {sample_id_end}), split_path={args.split_path}"
    )
    if len(split) != expected_range_size:
        raise RuntimeError(
            f"Fixed split contains {len(split)} selected samples, but "
            f"{expected_range_size} were requested"
        )
    if duplicate_sample_ids:
        raise RuntimeError(
            f"duplicate sample_ids in fixed split: {duplicate_sample_ids}"
        )

    manifest = {
        "task": "fixed_benchmark",
        "status": "running",
        "run_name": args.run_name,
        "modes": modes,
        "seed": args.seed,
        "generation_seed": args.generation_seed,
        "generation_seed_policy": "base_seed_plus_sample_id",
        "requested_samples": args.num_samples,
        "split_samples": len(split),
        "sample_id_start": args.sample_id_start,
        "sample_id_end": sample_id_end,
        "split_path": args.split_path,
        "dataset_json": args.dataset_json,
        "data_root": args.data_root,
        "gam_ckpt": args.gam_ckpt,
        "texture_ckpt": args.texture_ckpt,
        "clip_model_path": args.clip_model_path,
        "texture_preprocess_mode": args.texture_preprocess_mode,
        "experiment_flags": experiment_to_flags(args.run_name, args),
        "conflict_texture_suppress_strength": args.conflict_texture_suppress_strength,
        "conflict_palette_suppress_strength": args.conflict_palette_suppress_strength,
        "conflict_deltae_norm": args.conflict_deltae_norm,
        "conflict_threshold": args.conflict_threshold,
        "alpha": [args.alpha1, args.alpha2, args.alpha3, args.alpha4],
        "metrics_only": args.metrics_only,
        "resume_generation": bool(args.resume_generation),
        "skip_existing": bool(args.skip_existing),
        "overwrite": bool(args.overwrite),
        "reuse_from_dir": args.reuse_from_dir,
        "evaluation_protocol": args.evaluation_protocol,
        "mask_policy": args.mask_policy,
        "compute_fid": bool(args.compute_fid),
        "compute_kid": bool(args.compute_kid),
        "kid_subsets": args.kid_subsets,
        "kid_subset_size": args.kid_subset_size,
    }
    write_manifest(
        os.path.join(metrics_dir, "experiment_manifest.json"), manifest
    )

    rows = []
    generation_status_counts = Counter()
    for mode in modes:
        for sample_index, sample in enumerate(split, start=1):
            uid = sample_uid(sample)
            paths = sample_paths(args, sample)
            print(
                f"[benchmark] generation mode={mode}, "
                f"sample={sample_index}/{len(split)}, "
                f"sample_id={sample['sample_id']}, uid={uid}"
            )
            try:
                gen_path, generation_status = run_one_inference(
                    args, sample, mode, run_dir, paths=paths
                )
            except Exception as exc:
                if args.run_name == "e2b_gate" or args.use_texture_gate:
                    raise
                gen_path = None
                generation_status = "failed"
                reason = f"inference failed for {uid}: {exc}"
                print(f"[benchmark] WARNING: {reason}")
                _append_reason(reason_counts, reason)
            generation_status_counts[generation_status] += 1
            output_paths = _sample_output_paths(run_dir, mode, sample)
            conflict_info = _sample_conflict_info(args, sample, paths)
            row = {
                "mode": mode,
                "run_name": args.run_name,
                "sample_id": sample["sample_id"],
                "dataset_index": sample["idx"],
                "uid": uid,
                "category": sample.get("category"),
                "filename": sample.get("filename"),
                "prompt": sample["prompt"],
                "gen_path": gen_path
                or output_paths["generated"],
                "generation_status": generation_status,
                "generation_seed": generation_seed_for_sample(
                    args.generation_seed, sample
                ),
                **paths,
                "text_color": conflict_info["text_color"],
                "text_color_rgb": conflict_info["text_color_rgb"],
                "ref_palette_rgb": conflict_info["ref_palette_rgb"],
                "has_text_color": bool(conflict_info["has_text_color"]),
                "color_conflict_score": float(conflict_info["color_conflict_score"]),
                "color_delta_e": conflict_info.get("color_delta_e"),
                "conflict_bucket": conflict_info["conflict_bucket"],
            }
            for metric in PER_SAMPLE_METRICS:
                row[metric] = float("nan")
            rows.append(row)

        grid_rows = [
            row
            for row in rows
            if row["mode"] == mode and existing_file(row["gen_path"])
        ][: args.grid_max_images]
        grid_path = os.path.join(run_dir, f"grid_{mode}.png")
        make_grid([row["gen_path"] for row in grid_rows], grid_path, cols=4)
        if args.write_text_sidecars:
            _write_image_sidecar(
                grid_path, _grid_text_description(grid_rows, mode, grid_path)
            )

    final_generated_count = sum(
        existing_file(row["gen_path"]) for row in rows
    )
    generated_sample_keys = []
    for mode in modes:
        mode_dir = os.path.join(run_dir, mode)
        if not os.path.isdir(mode_dir):
            continue
        for root, _, files in os.walk(mode_dir):
            for filename in files:
                match = re.fullmatch(r"generated_(\d{6})\.png", filename)
                if match:
                    generated_sample_keys.append(
                        f"{mode}:{match.group(1)}"
                    )
    duplicate_generated_sample_ids = sorted(
        key
        for key, count in Counter(generated_sample_keys).items()
        if count > 1
    )
    all_duplicate_sample_ids = sorted(
        set(duplicate_sample_ids + duplicate_generated_sample_ids)
    )
    missing_sample_ids = [
        f"{row['mode']}:{row['sample_id']}"
        for row in rows
        if not existing_file(row["gen_path"])
    ]
    if final_generated_count != expected_num_samples:
        print(
            f"[benchmark] WARNING: final_generated_count={final_generated_count}, "
            f"expected_num_samples={expected_num_samples}"
        )
        print(
            "[benchmark] WARNING: missing_sample_ids="
            + ",".join(missing_sample_ids)
        )

    diagnostics = {
        "expected_num_samples": expected_num_samples,
        "expected_unique_sample_ids": len(split),
        "existing_before_generation": existing_before_generation,
        "newly_generated": generation_status_counts["generated"],
        "skipped_existing": (
            generation_status_counts["skipped_existing"]
            + generation_status_counts["reused_existing"]
        ),
        "reused_from_existing_results": generation_status_counts[
            "reused_existing"
        ],
        "final_generated_count": final_generated_count,
        "duplicate_sample_ids": all_duplicate_sample_ids,
        "duplicate_generated_sample_ids": duplicate_generated_sample_ids,
        "missing_sample_ids": missing_sample_ids,
        "resume_generation": bool(args.resume_generation),
        "skip_existing": bool(args.skip_existing),
        "overwrite": bool(args.overwrite),
        "evaluation_protocol": args.evaluation_protocol,
        "num_samples": len(rows),
        "num_generated_found": final_generated_count,
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
    diagnostics.update(summarize_conflict_rows(rows, threshold=args.conflict_threshold))
    write_json(os.path.join(metrics_dir, "benchmark_samples.json"), rows)
    _write_progress(metrics_dir, rows)
    write_json(os.path.join(metrics_dir, "diagnostics.json"), diagnostics)

    preflight_errors = []
    if final_generated_count != expected_num_samples:
        preflight_errors.append(
            f"generated images incomplete: {final_generated_count}/"
            f"{expected_num_samples}"
        )
    if all_duplicate_sample_ids:
        preflight_errors.append(
            f"duplicate sample ids: {all_duplicate_sample_ids}"
        )
    for key, label in (
        ("num_real_found", "real images"),
        ("num_texture_found", "texture images"),
        ("num_sketch_found", "sketch images"),
    ):
        if diagnostics[key] != expected_num_samples:
            preflight_errors.append(
                f"{label} incomplete: {diagnostics[key]}/"
                f"{expected_num_samples}"
            )
    if preflight_errors:
        diagnostics["preflight_errors"] = preflight_errors
        write_json(os.path.join(metrics_dir, "diagnostics.json"), diagnostics)
        raise RuntimeError(
            "benchmark preflight failed: " + "; ".join(preflight_errors)
        )

    rows = _prepare_evaluation_images(
        rows, metrics_dir, args.evaluation_resize, bool(args.write_text_sidecars)
    )
    diagnostics["evaluation_resize"] = args.evaluation_resize or None
    diagnostics["evaluation_protocol"] = (
        f"resize_generated_real_to_{args.evaluation_resize}"
        if args.evaluation_resize
        else "original_image_size"
    )
    diagnostics["num_evaluation_generated_found"] = sum(
        existing_file(row["gen_path"]) for row in rows
    )
    diagnostics["num_evaluation_real_found"] = sum(
        existing_file(row["target_path"]) for row in rows
    )
    if (
        diagnostics["num_evaluation_generated_found"] != expected_num_samples
        or diagnostics["num_evaluation_real_found"] != expected_num_samples
    ):
        write_json(
            os.path.join(metrics_dir, "diagnostics.json"), diagnostics
        )
        raise RuntimeError(
            "evaluation image preparation incomplete: "
            f"generated={diagnostics['num_evaluation_generated_found']}, "
            f"real={diagnostics['num_evaluation_real_found']}, "
            f"expected={expected_num_samples}"
        )
    if args.evaluation_resize:
        invalid_sizes = []
        expected_size = (args.evaluation_resize, args.evaluation_resize)
        for row in rows:
            for key in ("gen_path", "target_path"):
                if not existing_file(row[key]):
                    continue
                with Image.open(row[key]) as image:
                    if image.size != expected_size or image.mode != "RGB":
                        invalid_sizes.append(
                            f"{row['sample_id']}:{key}:{image.mode}:{image.size}"
                        )
        diagnostics["invalid_resized_images"] = invalid_sizes
        if invalid_sizes:
            write_json(
                os.path.join(metrics_dir, "diagnostics.json"), diagnostics
            )
            raise RuntimeError(
                "resize evaluation preflight failed: "
                + ", ".join(invalid_sizes[:20])
            )
    _write_progress(metrics_dir, rows)

    debug_dir = os.path.join(metrics_dir, "debug_masks")
    for row_index, row in enumerate(rows, start=1):
        print(
            f"[benchmark] evaluating sample={row_index}/{len(rows)}, uid={row['uid']}"
        )
        generated = safe_open_rgb(row["gen_path"])
        if generated is None:
            reason = f"generated image missing: {row['gen_path']}"
            _append_reason(reason_counts, reason)
            row["metric_warnings"] = [reason]
            _write_progress(metrics_dir, rows)
            continue

        mask_bundle = prepare_evaluation_masks(
            generated.size,
            mask_path=row["mask_path"],
            sketch_path=row["sketch_path"],
            target_path=row["target_path"],
            gen_path=row["gen_path"],
            mask_policy=args.mask_policy,
        )
        row.update(mask_bundle["stats"])
        garment_mask = mask_bundle.get("garment")
        mask_image = None
        if garment_mask is not None:
            mask_image = Image.fromarray(garment_mask.astype(np.uint8) * 255, mode="L")
        gen_rgb = dominant_rgb_from_pil(generated, mask=mask_image)
        if row.get("has_text_color"):
            row["generated_dominant_rgb"] = list(gen_rgb)
            row["prompt_color_delta_e"] = delta_e_rgb(row["text_color_rgb"], gen_rgb)
        else:
            row["generated_dominant_rgb"] = [0, 0, 0]
            row["prompt_color_delta_e"] = float("nan")
        # CTD 主指标 (docs/ctd_stage_a_spec.md §4): 参照物是真实目标图的主色, 不过
        # COLOR_TABLE, 所以无法靠坍缩颜色分辨率来刷 —— 它的下界是可测的 GT floor(=0)。
        # 对无颜色词样本也要算 —— 那是 D1 护栏(防全局阻尼)的读数。
        target_image = safe_open_rgb(row["target_path"])
        if target_image is not None:
            tgt_rgb = dominant_rgb_from_pil(target_image, mask=mask_image)
            row["target_dominant_rgb"] = list(tgt_rgb)
            row["target_color_delta_e"] = delta_e_rgb(tgt_rgb, gen_rgb)
            if row.get("has_text_color"):
                row["gt_floor_delta_e"] = delta_e_rgb(row["text_color_rgb"], tgt_rgb)
            else:
                row["gt_floor_delta_e"] = float("nan")
        else:
            row["target_dominant_rgb"] = [0, 0, 0]
            row["target_color_delta_e"] = float("nan")
            row["gt_floor_delta_e"] = float("nan")
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
            if args.write_text_sidecars:
                for name in ("garment", "outside", "boundary"):
                    debug_path = os.path.join(debug_dir, f"{row['uid']}_{name}.png")
                    _write_image_sidecar(
                        debug_path,
                        _evaluation_text_description(
                            row, f"debug_mask_{name}", debug_path
                        ),
                    )

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
        if (
            row_index % args.progress_write_interval == 0
            or row_index == len(rows)
        ):
            _write_progress(metrics_dir, rows)

    mask_confidences = [
        float(row["mask_confidence"])
        for row in rows
        if _is_finite(row.get("mask_confidence"))
    ]
    diagnostics["average_mask_confidence"] = (
        float(np.mean(mask_confidences)) if mask_confidences else None
    )
    diagnostics["number_of_low_confidence_masks"] = sum(
        bool(row.get("mask_low_confidence")) for row in rows
    )
    diagnostics["number_of_mask_fallbacks"] = sum(
        bool(row.get("mask_fallback")) for row in rows
    )
    diagnostics["mask_source_counts"] = dict(
        Counter(row.get("mask_source") or "missing" for row in rows)
    )
    # mask 形态学后端与几何决定了全部 mask 派生指标的取值。历史上这两项都没被
    # 记录, 导致跨 run 的 leak / boundary / edge / IoU / TCF / TPF 无法判断可比性。
    diagnostics.update(mask_backend_info())
    diagnostics["mask_geometry_counts"] = dict(
        Counter(
            "%sx%s" % (row.get("mask_width"), row.get("mask_height")) for row in rows
        )
    )

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
    diagnostics.update(summarize_conflict_rows(rows, threshold=args.conflict_threshold))

    fid_value = float("nan")
    fid_backend = None
    kid_mean = float("nan")
    kid_std = float("nan")
    kid_backend = None
    if args.compute_fid or args.compute_kid:
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
        elif args.compute_fid:
            try:
                fid_value, fid_backend = compute_fid_from_paths(
                    gen_paths,
                    real_paths,
                    batch_size=args.fid_batch_size,
                    device=args.device,
                    return_backend=True,
                )
            except Exception as exc:
                reason = f"FID failed: {exc}"
                print(f"[benchmark] WARNING: {reason}")
                _append_reason(reason_counts, reason)
        if len(gen_paths) >= 2 and len(real_paths) >= 2 and args.compute_kid:
            try:
                real_features = extract_inception_features(
                    real_paths,
                    batch_size=args.fid_batch_size,
                    device=args.device,
                )
                gen_features = extract_inception_features(
                    gen_paths,
                    batch_size=args.fid_batch_size,
                    device=args.device,
                )
                kid = compute_kid_from_features(
                    real_features,
                    gen_features,
                    subsets=args.kid_subsets,
                    subset_size=args.kid_subset_size,
                    seed=args.kid_seed,
                )
                kid_mean = kid["kid_mean"]
                kid_std = kid["kid_std"]
                kid_backend = "torchvision_inception_v3_pool3"
            except Exception as exc:
                reason = f"KID failed: {exc}"
                print(f"[benchmark] WARNING: {reason}")
                _append_reason(reason_counts, reason)
    if not args.compute_fid:
        _append_reason(reason_counts, "FID disabled by --compute_fid 0")
    if not args.compute_kid:
        _append_reason(reason_counts, "KID disabled by --compute_kid 0")

    diagnostics["num_valid_for_leakage"] = sum(
        _is_finite(row.get("leak_colored_frac")) for row in rows
    )
    diagnostics["num_valid_for_structure"] = sum(
        _is_finite(row.get("struct_edge_f1")) for row in rows
    )
    diagnostics["fid_backend"] = fid_backend
    diagnostics["kid_subsets"] = args.kid_subsets if args.compute_kid else None
    diagnostics["kid_subset_size"] = args.kid_subset_size if args.compute_kid else None
    diagnostics["kid_backend"] = kid_backend
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
        summary["FID_backend"] = fid_backend
        summary["KID"] = kid_mean if _is_finite(kid_mean) else None
        summary["KID_std"] = kid_std if _is_finite(kid_std) else None
        summary["KID_backend"] = kid_backend
        summary["generation_seed"] = args.generation_seed
        summary["generation_seed_policy"] = "base_seed_plus_sample_id"
        summary_rows.append(summary)
    bucket_rows = _aggregate_conflict_buckets(rows)
    category_rows = []
    categories = sorted(
        {row.get("category") for row in rows if row.get("category")}
    )
    for mode in modes:
        for category in categories:
            category_rows.append(_aggregate_rows(rows, mode, category=category))

    _write_progress(metrics_dir, rows)
    write_csv(os.path.join(metrics_dir, "category_metrics.csv"), category_rows)
    write_json(os.path.join(metrics_dir, "category_metrics.json"), category_rows)
    write_csv(os.path.join(metrics_dir, "conflict_bucket_metrics.csv"), bucket_rows)
    write_json(os.path.join(metrics_dir, "conflict_bucket_metrics.json"), bucket_rows)
    write_csv(os.path.join(metrics_dir, "metrics_summary.csv"), summary_rows)
    write_json(os.path.join(metrics_dir, "metrics_summary.json"), summary_rows)
    write_csv(os.path.join(metrics_dir, "summary_metrics.csv"), summary_rows)
    write_json(os.path.join(metrics_dir, "summary_metrics.json"), summary_rows)
    write_markdown_table(
        os.path.join(metrics_dir, "summary_metrics.md"),
        summary_rows,
        title="Fixed Benchmark Summary",
    )
    write_json(os.path.join(metrics_dir, "diagnostics.json"), diagnostics)

    manifest.update(
        {
            "status": "completed",
            "generated_images": diagnostics["num_generated_found"],
            "fid_backend": fid_backend,
            "diagnostics_path": os.path.join(
                metrics_dir, "diagnostics.json"
            ),
        }
    )
    write_manifest(
        os.path.join(metrics_dir, "experiment_manifest.json"), manifest
    )
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
    parser.add_argument(
        "--generation_seed",
        type=int,
        default=42,
        help=(
            "Base generation seed. Each sample uses generation_seed + sample_id; "
            "--seed remains reserved for the fixed data split."
        ),
    )
    parser.add_argument("--sample_id_start", type=int, default=0)
    parser.add_argument("--sample_id_end", type=int, default=None)
    parser.add_argument(
        "--resume_generation", type=int, choices=[0, 1], default=1
    )
    parser.add_argument("--skip_existing", type=int, choices=[0, 1], default=1)
    parser.add_argument("--overwrite", type=int, choices=[0, 1], default=0)
    parser.add_argument(
        "--reuse_from_dir",
        default=None,
        help="Existing experiment directory whose generated results can be reused.",
    )
    parser.add_argument("--gam_ckpt", required=True)
    parser.add_argument("--texture_ckpt", required=True)
    parser.add_argument("--layer_group_enabled", type=int, choices=[0, 1], default=1)
    parser.add_argument("--use_texture_gate", type=int, choices=[0, 1], default=0)
    parser.add_argument("--use_palette_tokens", type=int, choices=[0, 1], default=0)
    parser.add_argument("--num_palette_tokens", type=int, default=4)
    parser.add_argument("--gate_type", default="layer")
    parser.add_argument("--gate_init", default="identity")
    parser.add_argument("--gate_min", type=float, default=0.7)
    parser.add_argument("--gate_max", type=float, default=1.3)
    parser.add_argument("--use_balanced_fusion_gate", type=int, choices=[0, 1], default=0)
    parser.add_argument("--balanced_gate_hidden_dim", type=int, default=64)
    parser.add_argument("--balanced_gate_scale", type=float, default=0.2)
    parser.add_argument("--balanced_gate_min", type=float, default=0.8)
    parser.add_argument("--balanced_gate_max", type=float, default=1.2)
    parser.add_argument("--use_conflict_aware_gate", type=int, choices=[0, 1], default=0)
    parser.add_argument("--use_tcpm_lite", type=int, choices=[0, 1], default=0)
    parser.add_argument("--use_aa_tcr_fuse", type=int, choices=[0, 1], default=0)
    parser.add_argument("--use_text_guided_resampler", type=int, choices=[-1, 0, 1], default=-1)
    parser.add_argument("--conflict_texture_suppress_strength", type=float, default=0.1)
    parser.add_argument("--conflict_palette_suppress_strength", type=float, default=0.4)
    parser.add_argument("--conflict_deltae_norm", type=float, default=50.0)
    parser.add_argument("--conflict_threshold", type=float, default=0.70)
    parser.add_argument("--save_balanced_gate_trace", type=int, choices=[0, 1], default=0)
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
    parser.add_argument(
        "--metrics_output_dir",
        default=None,
        help="Optional separate directory for metrics and resized evaluation images.",
    )
    parser.add_argument(
        "--evaluation_resize",
        type=int,
        default=0,
        help="Resize generated and real images to this square size before evaluation.",
    )
    parser.add_argument(
        "--evaluation_protocol",
        default="original_image_size",
        choices=[
            "original_image_size",
            "resize_generated_real_to_256",
        ],
    )
    parser.add_argument("--real_images_dir", default=None)
    parser.add_argument("--texture_images_dir", default=None)
    parser.add_argument("--sketch_images_dir", default=None)
    parser.add_argument("--mask_dir", default=None)
    parser.add_argument(
        "--clip_model_path", default="openai/clip-vit-large-patch14"
    )
    parser.add_argument("--compute_fid", type=int, choices=[0, 1], default=1)
    parser.add_argument("--compute_kid", type=int, choices=[0, 1], default=0)
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
    parser.add_argument("--kid_subsets", type=int, default=50)
    parser.add_argument("--kid_subset_size", type=int, default=100)
    parser.add_argument("--kid_seed", type=int, default=42)
    parser.add_argument(
        "--mask_policy",
        choices=["auto", "sketch_only"],
        default="auto",
        help="A0 should use sketch_only to prevent target/generated mask fallback.",
    )
    parser.add_argument("--grid_max_images", type=int, default=100)
    parser.add_argument(
        "--progress_write_interval",
        type=int,
        default=1,
        help="Write per-sample progress files every N evaluated samples.",
    )
    parser.add_argument("--write_text_sidecars", type=int, choices=[0, 1], default=1)
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
