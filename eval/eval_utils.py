import math
import os

import numpy as np
from PIL import Image, ImageFilter

from garment_mask_utils import (
    build_sketch_garment_mask,
    estimate_cloth_foreground_mask,
    mask_diagnostics,
    mask_image_to_bool,
)


def existing_file(path):
    return bool(path) and os.path.isfile(path)


def safe_open_rgb(path):
    if not existing_file(path):
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def extract_generated_panel(path, comparison_path=None):
    image = safe_open_rgb(path)
    if image is None:
        return False

    width, height = image.size
    panel_width = width // 3
    looks_like_grid = (
        width % 3 == 0
        and width / max(height, 1) > 1.5
        and 0.45 <= panel_width / max(height, 1) <= 1.25
    )
    if not looks_like_grid:
        return True

    if comparison_path and not os.path.exists(comparison_path):
        os.makedirs(os.path.dirname(comparison_path) or ".", exist_ok=True)
        image.save(comparison_path)
    generated = image.crop((2 * panel_width, 0, width, height))
    generated.save(path)
    return True


def _mask_from_path(path, size):
    if not existing_file(path):
        return None
    try:
        mask = Image.open(path).convert("L").resize(size, Image.NEAREST)
    except Exception:
        return None
    arr = np.asarray(mask, dtype=np.uint8)
    candidates = [arr > 127, arr <= 127]
    scored = []
    for candidate in candidates:
        area = float(candidate.mean())
        border = np.concatenate(
            [candidate[0], candidate[-1], candidate[:, 0], candidate[:, -1]]
        )
        valid_area = 0.001 < area < 0.95
        scored.append((valid_area, -float(border.mean()), -abs(area - 0.45), candidate))
    return max(scored, key=lambda item: item[:3])[3]


def sketch_to_garment_mask(
    sketch, size, line_threshold=245, close_size=5, return_info=False
):
    width, height = size
    mask_image, info = build_sketch_garment_mask(
        sketch,
        width,
        height,
        line_threshold=line_threshold,
        close_size=close_size,
    )
    garment = mask_image_to_bool(mask_image)
    return (garment, info) if return_info else garment


def estimate_foreground_mask(image, size, return_info=False):
    width, height = size
    mask_image, info = estimate_cloth_foreground_mask(image, width, height)
    mask = mask_image_to_bool(mask_image)
    return (mask, info) if return_info else mask


def derive_region_masks(garment_mask, kernel_size=9):
    if garment_mask is None:
        return None, None
    kernel_size = max(3, int(kernel_size) | 1)
    try:
        import cv2

        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask_u8 = garment_mask.astype(np.uint8) * 255
        dilated = cv2.dilate(mask_u8, kernel) > 127
        eroded = cv2.erode(mask_u8, kernel) > 127
    except ImportError:
        mask_image = Image.fromarray(garment_mask.astype(np.uint8) * 255, mode="L")
        dilated = np.asarray(mask_image.filter(ImageFilter.MaxFilter(kernel_size))) > 127
        eroded = np.asarray(mask_image.filter(ImageFilter.MinFilter(kernel_size))) > 127
    outside = ~dilated
    boundary = dilated & ~eroded
    return outside, boundary


def prepare_evaluation_masks(
    size,
    mask_path=None,
    sketch_path=None,
    target_path=None,
    gen_path=None,
    kernel_size=9,
    mask_policy="auto",
):
    if mask_policy not in ("auto", "sketch_only"):
        raise ValueError(f"unsupported mask_policy: {mask_policy}")
    warnings = []
    source = None
    mask_info = None
    sketch_mask = None
    sketch_info = None
    garment = _mask_from_path(mask_path, size) if mask_policy == "auto" else None
    if garment is not None:
        source = "dataset_mask"
        mask_info = mask_diagnostics(garment, source)
        mask_info["mask_confidence"] = 1.0
        mask_info["mask_low_confidence"] = False
    else:
        if mask_path and mask_policy == "auto":
            warnings.append(f"mask unreadable or missing: {mask_path}")
        sketch = safe_open_rgb(sketch_path)
        sketch_mask, sketch_info = sketch_to_garment_mask(
            sketch, size, return_info=True
        )
        garment = None
        if sketch_mask is not None and sketch_mask.any():
            if mask_policy == "sketch_only" or not sketch_info["mask_low_confidence"]:
                garment = sketch_mask
                mask_info = sketch_info
                source = sketch_info["mask_source"]
            else:
                warnings.append(
                    "low-confidence sketch mask: "
                    f"confidence={sketch_info['mask_confidence']:.4f}"
                )
        elif sketch_path:
            warnings.append(f"could not derive garment mask from sketch: {sketch_path}")

    if garment is None and mask_policy == "auto":
        target_mask, target_info = estimate_foreground_mask(
            safe_open_rgb(target_path), size, return_info=True
        )
        if (
            target_mask is not None
            and target_mask.any()
            and not target_info["mask_low_confidence"]
        ):
            garment = target_mask
            mask_info = target_info
            mask_info["mask_fallback"] = True
            mask_info["mask_fallback_reason"] = "low_confidence_sketch"
            source = target_info["mask_source"]

    if (
        garment is None
        and mask_policy == "auto"
        and sketch_mask is not None
        and sketch_mask.any()
    ):
        garment = sketch_mask
        mask_info = sketch_info
        source = sketch_info["mask_source"]

    if garment is None and mask_policy == "auto":
        garment, mask_info = estimate_foreground_mask(
            safe_open_rgb(gen_path), size, return_info=True
        )
        if garment is not None and garment.any():
            source = "generated_foreground"
            mask_info["mask_source"] = source
            warnings.append("garment mask estimated from generated image")
        else:
            garment = None
            mask_info = None

    outside, boundary = derive_region_masks(garment, kernel_size=kernel_size)
    total = max(1, size[0] * size[1])

    def count(mask):
        return int(mask.sum()) if mask is not None else 0

    stats = {
        "mask_source": source,
        "garment_mask_pixels": count(garment),
        "outside_mask_pixels": count(outside),
        "boundary_mask_pixels": count(boundary),
        "garment_mask_area": count(garment) / total,
        "outside_mask_area": count(outside) / total,
        "boundary_mask_area": count(boundary) / total,
    }
    if mask_info:
        stats.update(mask_info)
        stats["mask_source"] = source
    return {
        "garment": garment,
        "outside": outside,
        "boundary": boundary,
        "stats": stats,
        "warnings": warnings,
    }


def save_mask_debug(mask_bundle, output_dir, uid):
    os.makedirs(output_dir, exist_ok=True)
    for name in ("garment", "outside", "boundary"):
        mask = mask_bundle.get(name)
        if mask is None:
            continue
        image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
        image.save(os.path.join(output_dir, f"{uid}_{name}.png"))


def finite_or_none(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return finite_or_none(value)
