#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from eval.benchmark_utils import ensure_dir, write_csv, write_json
from eval.distribution_diagnostics import compute_distribution_metrics
from garment_mask_utils import estimate_cloth_foreground_mask, mask_image_to_bool


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def image_map(directory):
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"image directory does not exist: {directory}")
    return {
        name: os.path.join(directory, name)
        for name in sorted(os.listdir(directory))
        if name.lower().endswith(IMAGE_EXTENSIONS)
    }


def composite(image, foreground, background_rgb):
    array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    array[~foreground] = np.asarray(background_rgb, dtype=np.uint8)
    return Image.fromarray(array, "RGB")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare original and background-normalized FID")
    parser.add_argument("--experiment_dir", default=None)
    parser.add_argument("--real_dir", default=None)
    parser.add_argument("--generated_dir", default=None)
    parser.add_argument("--output_dir", default="eval_outputs/phase2_diagnostics/background_fid")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clean_fid", type=int, choices=[0, 1], default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    real_dir = args.real_dir or (os.path.join(args.experiment_dir, "real") if args.experiment_dir else None)
    generated_dir = args.generated_dir or (
        os.path.join(args.experiment_dir, "generated") if args.experiment_dir else None
    )
    if not real_dir or not generated_dir:
        raise ValueError("provide --experiment_dir or both --real_dir and --generated_dir")

    real = image_map(real_dir)
    generated = image_map(generated_dir)
    names = sorted(set(real) & set(generated))[: args.num_samples]
    if len(names) < 2:
        raise ValueError(f"need at least 2 filename-matched pairs, got {len(names)}")

    borders = []
    for name in names:
        array = np.asarray(Image.open(real[name]).convert("RGB"), dtype=np.float32)
        borders.append(np.concatenate([array[0], array[-1], array[:, 0], array[:, -1]], axis=0))
    mean_background = np.rint(np.concatenate(borders, axis=0).mean(axis=0)).astype(np.uint8)

    variants = {
        "original": ((255, 255, 255), False),
        "white_background": ((255, 255, 255), True),
        "mean_background": (mean_background, True),
    }
    variant_paths = {}
    for variant in variants:
        real_out = os.path.join(args.output_dir, variant, "real")
        fake_out = os.path.join(args.output_dir, variant, "generated")
        ensure_dir(real_out)
        ensure_dir(fake_out)
        variant_paths[variant] = (real_out, fake_out, [], [])

    rows = []
    for name in names:
        real_image = Image.open(real[name]).convert("RGB")
        fake_image = Image.open(generated[name]).convert("RGB").resize(real_image.size, Image.BICUBIC)
        width, height = real_image.size
        real_mask_image, real_info = estimate_cloth_foreground_mask(real_image, width, height)
        fake_mask_image, fake_info = estimate_cloth_foreground_mask(fake_image, width, height)
        real_mask = mask_image_to_bool(real_mask_image)
        fake_mask = mask_image_to_bool(fake_mask_image)
        row = {"filename": name}
        row.update({f"real_{key}": value for key, value in real_info.items()})
        row.update({f"generated_{key}": value for key, value in fake_info.items()})
        rows.append(row)

        for variant, (background, normalize) in variants.items():
            real_out, fake_out, real_paths, fake_paths = variant_paths[variant]
            real_result = composite(real_image, real_mask, background) if normalize else real_image
            fake_result = composite(fake_image, fake_mask, background) if normalize else fake_image
            real_path = os.path.join(real_out, name)
            fake_path = os.path.join(fake_out, name)
            real_result.save(real_path)
            fake_result.save(fake_path)
            real_paths.append(real_path)
            fake_paths.append(fake_path)

    metrics = {}
    for variant, (real_out, fake_out, real_paths, fake_paths) in variant_paths.items():
        metrics[variant] = compute_distribution_metrics(
            real_paths,
            fake_paths,
            device=args.device,
            batch_size=args.batch_size,
            seed=args.seed,
            clean_fid=bool(args.clean_fid),
            real_dir=real_out,
            fake_dir=fake_out,
        )
    summary = {
        "diagnostic": "background_fid",
        "real_dir": real_dir,
        "generated_dir": generated_dir,
        "num_pairs": len(names),
        "mean_background_rgb": mean_background.tolist(),
        "real_low_confidence_masks": sum(bool(row["real_mask_low_confidence"]) for row in rows),
        "generated_low_confidence_masks": sum(
            bool(row["generated_mask_low_confidence"]) for row in rows
        ),
        "metrics": metrics,
    }
    write_csv(os.path.join(args.output_dir, "mask_diagnostics.csv"), rows)
    write_json(os.path.join(args.output_dir, "summary.json"), summary)
    print(summary)


if __name__ == "__main__":
    main()
