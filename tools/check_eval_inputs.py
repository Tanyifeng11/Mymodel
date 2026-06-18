#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
from glob import glob

import numpy as np
from PIL import Image, ImageDraw

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from eval.benchmark_utils import sample_uid, write_json
from eval.eval_utils import (
    existing_file,
    extract_generated_panel,
    prepare_evaluation_masks,
    safe_open_rgb,
)


def resolve_path(data_root, value, override_dir=None):
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


def find_generated(eval_output_dir, uid):
    matches = glob(
        os.path.join(eval_output_dir, "**", uid, "generated.png"),
        recursive=True,
    )
    if not matches:
        return None
    path = sorted(matches)[0]
    extract_generated_panel(
        path,
        comparison_path=os.path.join(os.path.dirname(path), "comparison_grid.png"),
    )
    return path


def placeholder(label, size):
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), label, fill="red")
    return image


def load_panel(path, size, label):
    image = safe_open_rgb(path)
    if image is None:
        return placeholder(f"{label}: missing", size)
    return image.resize(size, Image.BICUBIC)


def mask_panel(mask, size, label):
    if mask is None:
        return placeholder(f"{label}: missing", size)
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L").convert("RGB")
    return image.resize(size, Image.NEAREST)


def save_visual_grid(rows, output_path, count, seed):
    if not rows:
        return
    selected = rows[:]
    random.Random(seed).shuffle(selected)
    selected = selected[: min(count, len(selected))]
    panel_size = (192, 256)
    labels = [
        "generated",
        "real",
        "texture",
        "sketch",
        "garment",
        "outside",
        "boundary",
    ]
    grid = Image.new(
        "RGB",
        (panel_size[0] * len(labels), panel_size[1] * len(selected)),
        "white",
    )
    draw = ImageDraw.Draw(grid)
    for row_index, row in enumerate(selected):
        panels = [
            load_panel(row["generated_path"], panel_size, "generated"),
            load_panel(row["target_path"], panel_size, "real"),
            load_panel(row["texture_path"], panel_size, "texture"),
            load_panel(row["sketch_path"], panel_size, "sketch"),
            mask_panel(row["_mask_bundle"]["garment"], panel_size, "garment"),
            mask_panel(row["_mask_bundle"]["outside"], panel_size, "outside"),
            mask_panel(row["_mask_bundle"]["boundary"], panel_size, "boundary"),
        ]
        for column, panel in enumerate(panels):
            x = column * panel_size[0]
            y = row_index * panel_size[1]
            grid.paste(panel, (x, y))
            draw.rectangle((x, y, x + 190, y + 20), fill="white")
            draw.text((x + 4, y + 4), labels[column], fill="black")
    grid.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="Validate evaluation inputs.")
    parser.add_argument("--dataset_json", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--eval_output_dir", required=True)
    parser.add_argument("--split_path", default=None)
    parser.add_argument("--real_images_dir", default=None)
    parser.add_argument("--texture_images_dir", default=None)
    parser.add_argument("--sketch_images_dir", default=None)
    parser.add_argument("--mask_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--visual_samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_valid_pixels", type=int, default=50)
    args = parser.parse_args()

    source_path = (
        args.split_path
        if args.split_path and os.path.isfile(args.split_path)
        else args.dataset_json
    )
    with open(source_path, "r", encoding="utf-8") as handle:
        samples = json.load(handle)

    output_dir = args.output_dir or os.path.join(
        args.eval_output_dir, "input_check"
    )
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for index, sample in enumerate(samples):
        normalized = {
            "idx": sample.get("idx", index),
            "prompt": sample.get("prompt")
            or (
                sample.get("caption", "")
                if isinstance(sample.get("caption", ""), str)
                else sample.get("caption", [""])[0]
            ),
            "target": sample.get("target") or sample.get("cloth"),
            "texture": sample.get("texture") or sample.get("color"),
            "sketch": sample.get("sketch"),
            "mask": sample.get("mask"),
        }
        uid = sample_uid(normalized)
        generated_path = find_generated(args.eval_output_dir, uid)
        target_path = resolve_path(
            args.data_root, normalized["target"], args.real_images_dir
        )
        texture_path = resolve_path(
            args.data_root, normalized["texture"], args.texture_images_dir
        )
        sketch_path = resolve_path(
            args.data_root, normalized["sketch"], args.sketch_images_dir
        )
        mask_path = resolve_path(
            args.data_root, normalized["mask"], args.mask_dir
        )
        generated = safe_open_rgb(generated_path)
        size = generated.size if generated is not None else (384, 512)
        mask_bundle = prepare_evaluation_masks(
            size,
            mask_path=mask_path,
            sketch_path=sketch_path,
            target_path=target_path,
            gen_path=generated_path,
        )
        stats = mask_bundle["stats"]
        row = {
            "uid": uid,
            "generated_path": generated_path,
            "target_path": target_path,
            "texture_path": texture_path,
            "sketch_path": sketch_path,
            "mask_path": mask_path,
            "generated_found": existing_file(generated_path),
            "target_found": existing_file(target_path),
            "texture_found": existing_file(texture_path),
            "sketch_found": existing_file(sketch_path),
            "mask_found": existing_file(mask_path),
            **stats,
            "mask_warnings": mask_bundle["warnings"],
            "_mask_bundle": mask_bundle,
        }
        rows.append(row)

    report_rows = [
        {key: value for key, value in row.items() if key != "_mask_bundle"}
        for row in rows
    ]
    summary = {
        "num_samples": len(rows),
        "num_generated_found": sum(row["generated_found"] for row in rows),
        "num_real_found": sum(row["target_found"] for row in rows),
        "num_texture_found": sum(row["texture_found"] for row in rows),
        "num_sketch_found": sum(row["sketch_found"] for row in rows),
        "num_mask_found": sum(row["mask_found"] for row in rows),
        "num_valid_garment_masks": sum(
            row["garment_mask_pixels"] >= args.min_valid_pixels for row in rows
        ),
        "num_valid_outside_masks": sum(
            row["outside_mask_pixels"] >= args.min_valid_pixels for row in rows
        ),
        "num_valid_boundary_masks": sum(
            row["boundary_mask_pixels"] >= args.min_valid_pixels for row in rows
        ),
        "samples": report_rows,
    }
    write_json(
        os.path.join(output_dir, "eval_input_check_report.json"),
        summary,
    )

    markdown = [
        "# Evaluation Input Check",
        "",
        f"- Samples: {summary['num_samples']}",
        f"- Generated found: {summary['num_generated_found']}",
        f"- Real found: {summary['num_real_found']}",
        f"- Texture found: {summary['num_texture_found']}",
        f"- Sketch found: {summary['num_sketch_found']}",
        f"- Dataset masks found: {summary['num_mask_found']}",
        f"- Valid garment masks: {summary['num_valid_garment_masks']}",
        f"- Valid outside masks: {summary['num_valid_outside_masks']}",
        f"- Valid boundary masks: {summary['num_valid_boundary_masks']}",
        "",
        "| UID | Generated | Real | Texture | Sketch | Mask Source | Garment Pixels | "
        "Outside Pixels | Boundary Pixels |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in report_rows:
        markdown.append(
            "| {uid} | {generated_found} | {target_found} | {texture_found} | "
            "{sketch_found} | {mask_source} | {garment_mask_pixels} | "
            "{outside_mask_pixels} | {boundary_mask_pixels} |".format(**row)
        )
    with open(
        os.path.join(output_dir, "eval_input_check_report.md"),
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write("\n".join(markdown))

    save_visual_grid(
        rows,
        os.path.join(output_dir, "eval_input_visual_grid.png"),
        args.visual_samples,
        args.seed,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
