#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

try:
    import cv2
except ImportError as exc:
    raise SystemExit("This script requires opencv-python or opencv-python-headless.") from exc


def open_rgb(path):
    return Image.open(path).convert("RGB")


def image_to_np(img):
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def extract_panel(img, mode):
    if mode == "full":
        return img
    if mode == "right" or (mode == "auto" and img.width >= img.height * 2.0):
        panel_w = img.width // 3
        return img.crop((panel_w * 2, 0, img.width, img.height))
    return img


def load_generated(path, panel_mode):
    return extract_panel(open_rgb(path), panel_mode)


def load_mask(path, size, panel_mode):
    if path is None:
        return None
    mask = Image.open(path).convert("L")
    mask = extract_panel(mask, panel_mode) if mask.width >= mask.height * 2.0 else mask
    mask = mask.resize(size, Image.NEAREST)
    return np.asarray(mask, dtype=np.uint8) > 127


def auto_foreground_mask(img):
    arr = image_to_np(img)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = hsv[..., 1] / 255.0
    val = hsv[..., 2] / 255.0
    # Works for product photos with pale white/gray backgrounds.
    mask = (sat > 0.10) | (val < 0.88)
    mask = mask.astype(np.uint8) * 255
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask > 127


def resize_np_mask(mask, size):
    if mask is None:
        return None
    out = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").resize(size, Image.NEAREST)
    return np.asarray(out, dtype=np.uint8) > 127


def masked_pixels(arr, mask):
    if mask is None or not mask.any():
        return arr.reshape(-1, 3)
    return arr[mask]


def lab_mean_std(img, mask=None):
    arr = image_to_np(img)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
    px = masked_pixels(lab, mask)
    return px.mean(axis=0), px.std(axis=0)


def color_delta(img_a, mask_a, img_b, mask_b):
    mean_a, std_a = lab_mean_std(img_a, mask_a)
    mean_b, std_b = lab_mean_std(img_b, mask_b)
    mean_delta = float(np.linalg.norm(mean_a - mean_b))
    std_delta = float(np.linalg.norm(std_a - std_b))
    return mean_delta, std_delta


def hsv_hist(img, mask=None, bins=(24, 16, 16)):
    arr = image_to_np(img)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    if mask is not None and mask.any():
        px = hsv[mask]
    else:
        px = hsv.reshape(-1, 3)
    hist, _ = np.histogramdd(
        px.astype(np.float32),
        bins=bins,
        range=((0, 180), (0, 256), (0, 256)),
    )
    hist = hist.astype(np.float32).ravel()
    hist /= max(float(hist.sum()), 1.0)
    return hist


def hist_l1(img_a, mask_a, img_b, mask_b):
    return float(np.abs(hsv_hist(img_a, mask_a) - hsv_hist(img_b, mask_b)).sum())


def leak_metrics(gen_img, mask):
    arr = image_to_np(gen_img)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
    if mask is None or not mask.any():
        return 0.0, 0.0

    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    dilated = np.asarray(mask_img.filter(ImageFilter.MaxFilter(13)), dtype=np.uint8) > 127
    outside = ~dilated
    if outside.sum() < 16:
        return 0.0, 0.0

    outside_hsv = hsv[outside]
    sat = outside_hsv[:, 1] / 255.0
    val = outside_hsv[:, 2] / 255.0
    colored_frac = float(((sat > 0.16) & (val < 0.96)).mean())
    mean_sat = float(sat.mean())
    return colored_frac, mean_sat


def find_cases(root, image_name):
    root = Path(root)
    return sorted(root.rglob(image_name))


def paired_mask_path(image_path):
    candidate = image_path.with_name(f"{image_path.stem}_mask{image_path.suffix}")
    return candidate if candidate.exists() else None


def score_row(row, has_target):
    if has_target:
        return (
            0.35 * row["target_lab_mean_delta"]
            + 0.12 * row["target_lab_std_delta"]
            + 12.0 * row["texture_hist_l1"]
            + 90.0 * row["leak_colored_frac"]
            + 35.0 * row["leak_mean_saturation"]
        )
    return (
        0.45 * row["texture_lab_mean_delta"]
        + 0.15 * row["texture_lab_std_delta"]
        + 18.0 * row["texture_hist_l1"]
        + 90.0 * row["leak_colored_frac"]
        + 35.0 * row["leak_mean_saturation"]
    )


def evaluate_case(image_path, args, texture_img, target_img=None, target_mask=None):
    gen_img = load_generated(image_path, args.panel)
    mask_path = args.mask_path or paired_mask_path(image_path)
    gen_mask = load_mask(mask_path, gen_img.size, args.panel)

    texture_resized = texture_img.resize(gen_img.size, Image.BICUBIC)
    texture_mean, texture_std = color_delta(gen_img, gen_mask, texture_resized, None)
    texture_hist = hist_l1(gen_img, gen_mask, texture_resized, None)
    leak_frac, leak_sat = leak_metrics(gen_img, gen_mask)

    row = {
        "case": str(image_path.parent),
        "image": str(image_path),
        "mask": str(mask_path) if mask_path else "",
        "texture_lab_mean_delta": texture_mean,
        "texture_lab_std_delta": texture_std,
        "texture_hist_l1": texture_hist,
        "leak_colored_frac": leak_frac,
        "leak_mean_saturation": leak_sat,
    }

    if target_img is not None:
        target_resized = target_img.resize(gen_img.size, Image.BICUBIC)
        target_mask_resized = resize_np_mask(target_mask, gen_img.size)
        target_mean, target_std = color_delta(gen_img, gen_mask, target_resized, target_mask_resized)
        target_hist = hist_l1(gen_img, gen_mask, target_resized, target_mask_resized)
        row.update(
            {
                "target_lab_mean_delta": target_mean,
                "target_lab_std_delta": target_std,
                "target_hist_l1": target_hist,
            }
        )

    row["score_lower_is_better"] = score_row(row, target_img is not None)
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Rank generated garment parameter sets by color match, texture match, and leakage."
    )
    parser.add_argument("--root", required=True, help="Directory containing generated result folders.")
    parser.add_argument("--image_name", default="sketch2.png")
    parser.add_argument("--texture_path", required=True, help="Reference texture image.")
    parser.add_argument("--target_path", default=None, help="Original product image, optional but recommended.")
    parser.add_argument("--target_mask_path", default=None, help="Optional foreground mask for target_path.")
    parser.add_argument("--mask_path", default=None, help="Use one mask for every generated image.")
    parser.add_argument("--panel", default="auto", choices=["auto", "right", "full"])
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--out_json", default=None)
    args = parser.parse_args()

    texture_img = open_rgb(args.texture_path)
    target_img = open_rgb(args.target_path) if args.target_path else None
    target_mask = None
    if target_img is not None:
        if args.target_mask_path:
            target_mask_img = Image.open(args.target_mask_path).convert("L")
            target_mask = np.asarray(target_mask_img, dtype=np.uint8) > 127
        else:
            target_mask = auto_foreground_mask(target_img)

    cases = find_cases(args.root, args.image_name)
    if not cases:
        raise SystemExit(f"No images named {args.image_name!r} found under {args.root!r}.")

    rows = [evaluate_case(path, args, texture_img, target_img, target_mask) for path in cases]
    rows.sort(key=lambda r: r["score_lower_is_better"])

    fieldnames = list(rows[0].keys())
    print("\nRanked results, lower score is better:\n")
    print(
        f"{'rank':>4}  {'score':>9}  {'target_dE':>9}  {'tex_dE':>8}  "
        f"{'tex_hist':>8}  {'leak%':>7}  case"
    )
    for idx, row in enumerate(rows, start=1):
        target_de = row.get("target_lab_mean_delta", float("nan"))
        print(
            f"{idx:>4}  {row['score_lower_is_better']:>9.3f}  "
            f"{target_de:>9.3f}  {row['texture_lab_mean_delta']:>8.3f}  "
            f"{row['texture_hist_l1']:>8.3f}  {row['leak_colored_frac'] * 100:>6.2f}%  "
            f"{row['case']}"
        )

    if args.out_csv:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
