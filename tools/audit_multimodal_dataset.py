#!/usr/bin/env python3
import argparse
import hashlib
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from color_conflict_utils import compute_color_conflict, delta_e_rgb, dominant_rgb_from_pil
from eval.benchmark_utils import create_or_load_fixed_split, write_csv, write_json
from eval.metrics import compute_structure_preservation
from garment_mask_utils import estimate_cloth_foreground_mask


def resolve_path(data_root, value):
    if not value:
        return None
    if os.path.isabs(value):
        return value
    candidates = [os.path.join(data_root, value)]
    if os.path.normpath(value).split(os.sep)[0].lower() == "cloth":
        candidates.append(os.path.join(data_root, "gt", os.path.basename(value)))
    return next((path for path in candidates if os.path.isfile(path)), candidates[0])


def open_rgb(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def average_hash(image):
    gray = np.asarray(image.convert("L").resize((8, 8), Image.BILINEAR), dtype=np.float32)
    bits = gray >= gray.mean()
    return f"{int(''.join('1' if value else '0' for value in bits.flat), 2):016x}"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description="Audit multimodal validation samples")
    parser.add_argument("--dataset_json", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split_path", required=True)
    parser.add_argument("--output_dir", default="eval_outputs/phase2_diagnostics/dataset_audit")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected_width", type=int, default=384)
    parser.add_argument("--expected_height", type=int, default=512)
    parser.add_argument("--blank_texture_std", type=float, default=8.0)
    parser.add_argument("--color_conflict_threshold", type=float, default=0.55)
    parser.add_argument("--texture_target_delta_e_threshold", type=float, default=35.0)
    parser.add_argument("--structure_f1_threshold", type=float, default=0.10)
    return parser.parse_args()


def main():
    args = parse_args()
    split = create_or_load_fixed_split(
        args.dataset_json, args.split_path, num_samples=args.num_samples, seed=args.seed
    )
    rows = []
    exact_groups, perceptual_groups = defaultdict(list), defaultdict(list)

    for sample in split:
        sample_id = str(sample["sample_id"])
        paths = {
            "target": resolve_path(args.data_root, sample.get("target")),
            "texture": resolve_path(args.data_root, sample.get("texture")),
            "sketch": resolve_path(args.data_root, sample.get("sketch")),
        }
        images = {key: open_rgb(path) if path and os.path.isfile(path) else None for key, path in paths.items()}
        reasons = []
        row = {
            "sample_id": sample_id,
            "dataset_index": sample.get("idx"),
            "prompt": sample.get("prompt", ""),
            **{f"{key}_path": path for key, path in paths.items()},
        }
        for key, image in images.items():
            missing = image is None
            row[f"{key}_missing_or_corrupt"] = missing
            if missing:
                reasons.append(f"{key}_missing_or_corrupt")
                continue
            width, height = image.size
            row[f"{key}_width"] = width
            row[f"{key}_height"] = height
            row[f"{key}_needs_training_resize"] = (width, height) != (
                args.expected_width,
                args.expected_height,
            )
            aspect = width / max(height, 1)
            dimension_anomaly = width < 64 or height < 64 or aspect < 0.20 or aspect > 5.0
            row[f"{key}_dimension_anomaly"] = dimension_anomaly
            if dimension_anomaly:
                reasons.append(f"{key}_dimension")

        texture = images["texture"]
        if texture is not None:
            texture_std = float(np.asarray(texture, dtype=np.float32).std())
            row["texture_rgb_std"] = texture_std
            row["texture_near_blank"] = texture_std < args.blank_texture_std
            if row["texture_near_blank"]:
                reasons.append("texture_near_blank")

        target = images["target"]
        if target is not None:
            mask_image, mask_info = estimate_cloth_foreground_mask(target, *target.size)
            row.update(mask_info)
            if mask_info["mask_low_confidence"]:
                reasons.append("target_mask_low_confidence")
            conflict = compute_color_conflict(sample.get("prompt", ""), ref_image=target, mask=mask_image)
            row.update({f"text_target_{key}": value for key, value in conflict.items()})
            if conflict["has_text_color"] and conflict["color_conflict_score"] >= args.color_conflict_threshold:
                reasons.append("text_target_color_conflict")
            exact_groups[file_sha256(paths["target"])].append(sample_id)
            perceptual_groups[average_hash(target)].append(sample_id)

        if texture is not None:
            texture_conflict = compute_color_conflict(sample.get("prompt", ""), ref_image=texture)
            row.update({f"text_texture_{key}": value for key, value in texture_conflict.items()})
            if (
                texture_conflict["has_text_color"]
                and texture_conflict["color_conflict_score"] >= args.color_conflict_threshold
            ):
                reasons.append("text_texture_color_conflict")

        if target is not None and texture is not None:
            target_rgb = dominant_rgb_from_pil(target, mask_image)
            texture_rgb = dominant_rgb_from_pil(texture)
            texture_target_delta_e = delta_e_rgb(target_rgb, texture_rgb)
            row["target_palette_rgb"] = list(target_rgb)
            row["texture_palette_rgb"] = list(texture_rgb)
            row["texture_target_color_delta_e"] = texture_target_delta_e
            if texture_target_delta_e >= args.texture_target_delta_e_threshold:
                reasons.append("texture_target_color_mismatch")

        if target is not None and images["sketch"] is not None:
            structure = compute_structure_preservation(paths["target"], paths["sketch"], target_path=paths["target"])
            for key in ("struct_edge_f1", "struct_edge_precision", "struct_edge_recall", "struct_iou", "struct_edge_l1"):
                row[key] = structure.get(key)
            edge_f1 = structure.get("struct_edge_f1")
            if edge_f1 is not None and np.isfinite(edge_f1) and edge_f1 < args.structure_f1_threshold:
                reasons.append("sketch_target_structure_mismatch")

        row["suspicious_reasons"] = reasons
        rows.append(row)

    by_id = {row["sample_id"]: row for row in rows}
    for label, groups in (("exact_target_duplicate", exact_groups), ("perceptual_target_duplicate", perceptual_groups)):
        for group_id, sample_ids in groups.items():
            if len(sample_ids) < 2:
                continue
            for sample_id in sample_ids:
                by_id[sample_id][label] = True
                by_id[sample_id][f"{label}_group"] = group_id
                by_id[sample_id]["suspicious_reasons"].append(label)

    for row in rows:
        reasons = sorted(set(row["suspicious_reasons"]))
        row["suspicious_reasons"] = ";".join(reasons)
        row["suspicious_score"] = len(reasons)
    suspicious = sorted(
        [row for row in rows if row["suspicious_score"] > 0],
        key=lambda row: (-row["suspicious_score"], row["sample_id"]),
    )
    summary = {
        "diagnostic": "multimodal_dataset_audit",
        "num_samples": len(rows),
        "num_suspicious": len(suspicious),
        "num_clean": len(rows) - len(suspicious),
        "reason_counts": {
            reason: sum(reason in row["suspicious_reasons"].split(";") for row in rows)
            for reason in sorted({reason for row in rows for reason in row["suspicious_reasons"].split(";") if reason})
        },
    }
    write_csv(os.path.join(args.output_dir, "all_samples.csv"), rows)
    write_csv(os.path.join(args.output_dir, "suspicious_samples.csv"), suspicious)
    write_json(os.path.join(args.output_dir, "summary.json"), summary)
    print(summary)


if __name__ == "__main__":
    main()
