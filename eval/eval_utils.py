import math
import os
from collections import deque

import numpy as np
from PIL import Image, ImageFilter


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


def sketch_to_garment_mask(sketch, size, line_threshold=245, dilate_size=9):
    if sketch is None:
        return None
    width, height = size
    gray = sketch.convert("L").resize((width, height), Image.BILINEAR)
    line = np.asarray(gray, dtype=np.uint8) < line_threshold
    dilate_size = max(3, int(dilate_size) | 1)
    barrier = np.asarray(
        Image.fromarray(line.astype(np.uint8) * 255, mode="L").filter(
            ImageFilter.MaxFilter(dilate_size)
        )
    ) > 0

    passable = ~barrier
    outside = np.zeros((height, width), dtype=bool)
    queue = deque()

    def push(y, x):
        if passable[y, x] and not outside[y, x]:
            outside[y, x] = True
            queue.append((y, x))

    for x in range(width):
        push(0, x)
        push(height - 1, x)
    for y in range(height):
        push(y, 0)
        push(y, width - 1)

    while queue:
        y, x = queue.popleft()
        if y > 0:
            push(y - 1, x)
        if y + 1 < height:
            push(y + 1, x)
        if x > 0:
            push(y, x - 1)
        if x + 1 < width:
            push(y, x + 1)

    garment = ~outside
    return garment


def estimate_foreground_mask(image, size):
    if image is None:
        return None
    image = image.convert("RGB").resize(size, Image.BICUBIC)
    arr = np.asarray(image, dtype=np.float32)
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]], axis=0)
    background = np.median(border, axis=0)
    color_distance = np.linalg.norm(arr - background[None, None, :], axis=2)
    value = arr.mean(axis=2)
    mask = (color_distance > 18.0) | (value < 235.0)
    try:
        import cv2

        kernel = np.ones((5, 5), dtype=np.uint8)
        mask_u8 = mask.astype(np.uint8) * 255
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
        mask = mask_u8 > 127
    except ImportError:
        pass
    return mask


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
):
    warnings = []
    source = None
    garment = _mask_from_path(mask_path, size)
    if garment is not None:
        source = "dataset_mask"
    else:
        if mask_path:
            warnings.append(f"mask unreadable or missing: {mask_path}")
        sketch = safe_open_rgb(sketch_path)
        garment = sketch_to_garment_mask(sketch, size)
        if garment is not None and garment.any():
            source = "sketch"
        else:
            garment = None
            if sketch_path:
                warnings.append(f"could not derive garment mask from sketch: {sketch_path}")

    if garment is None:
        garment = estimate_foreground_mask(safe_open_rgb(target_path), size)
        if garment is not None and garment.any():
            source = "target_foreground"
        else:
            garment = None

    if garment is None:
        garment = estimate_foreground_mask(safe_open_rgb(gen_path), size)
        if garment is not None and garment.any():
            source = "generated_foreground"
            warnings.append("garment mask estimated from generated image")
        else:
            garment = None

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
