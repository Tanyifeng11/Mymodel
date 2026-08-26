from collections import deque
import warnings

import numpy as np
from PIL import Image, ImageFilter

# ---------------------------------------------------------------------------
# cv2 后端探测。
#
# 历史教训: 既有全部评测报告都跑在 cv2 缺失的环境里, 走了下面的 Pillow fallback。
# 两条路径的形态学运算并不等价 —— 草图 mask 的置信度判定会变, auto 策略下
# sketch_flood_fill 的通过率在 70/500(Pillow) 与 364/500(cv2) 之间跳。也就是说
# 全部 mask 派生指标(leak / boundary / edge / IoU / TCF / TPF)在两条后端之间不可比。
#
# opencv-python 在 requirements.txt 里是声明了的, 但非 headless 版依赖 libGL,
# 在无图形栈的节点上 `import cv2` 会抛 ImportError 并被静默吞掉。所以这里:
#   1. 捕获所有异常并保留真实原因(不只是 ImportError);
#   2. 退回 Pillow 时发出 RuntimeWarning, 不再静默;
#   3. 通过 mask_backend_info() 暴露后端, 供评测脚本写进 diagnostics。
# ---------------------------------------------------------------------------
CV2_IMPORT_ERROR = None
try:
    import cv2
except Exception as exc:  # noqa: BLE001 - 常见为 opencv-python 缺 libGL.so.1
    cv2 = None
    CV2_IMPORT_ERROR = "%s: %s" % (type(exc).__name__, exc)
    warnings.warn(
        "garment_mask_utils: cv2 unavailable (%s); falling back to Pillow morphology. "
        "Mask results WILL differ from the cv2 path, so every mask-derived metric "
        "becomes incomparable across backends. Install opencv-python-headless, or "
        "record the backend via mask_backend_info() before comparing any numbers."
        % CV2_IMPORT_ERROR,
        RuntimeWarning,
        stacklevel=2,
    )


def mask_backend_info():
    """返回当前 mask 形态学后端, 供评测脚本写进 diagnostics.json。

    任何跨 run 的 mask 派生指标比较, 都必须先确认这两个 run 的 mask_backend 相同。
    """
    return {
        "mask_backend": "pillow_fallback" if cv2 is None else "opencv",
        "cv2_version": None if cv2 is None else str(getattr(cv2, "__version__", "unknown")),
        "cv2_import_error": CV2_IMPORT_ERROR,
    }


def _odd(value, minimum=3):
    return max(minimum, int(value) | 1)


def _binary_close(mask, kernel_size):
    kernel_size = _odd(kernel_size)
    mask_u8 = mask.astype(np.uint8) * 255
    if cv2 is not None:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        return cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel) > 127
    image = Image.fromarray(mask_u8, mode="L")
    image = image.filter(ImageFilter.MaxFilter(kernel_size))
    image = image.filter(ImageFilter.MinFilter(kernel_size))
    return np.asarray(image) > 127


def _binary_dilate(mask, kernel_size):
    kernel_size = _odd(kernel_size)
    mask_u8 = mask.astype(np.uint8) * 255
    if cv2 is not None:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        return cv2.dilate(mask_u8, kernel) > 127
    image = Image.fromarray(mask_u8, mode="L")
    return np.asarray(image.filter(ImageFilter.MaxFilter(kernel_size))) > 127


def _flood_fill_enclosed(barrier):
    height, width = barrier.shape
    passable = ~barrier
    if cv2 is not None:
        _, labels = cv2.connectedComponents(passable.astype(np.uint8), connectivity=4)
        border_labels = np.unique(
            np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
        )
        border_labels = border_labels[border_labels != 0]
        outside = np.isin(labels, border_labels)
        return ~outside

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
    return ~outside


def _component_stats(mask):
    total_area = int(mask.sum())
    if total_area == 0:
        return 0, 0.0
    if cv2 is not None:
        count, _, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.asarray([])
        largest = int(areas.max()) if len(areas) else 0
        return max(0, count - 1), largest / total_area
    return 1, 1.0


def _filter_components(mask, minimum_area_ratio=0.001, relative_area=0.02):
    if cv2 is None or not mask.any():
        return mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(areas.max())
    minimum = max(int(mask.size * minimum_area_ratio), int(largest * relative_area))
    keep = np.zeros_like(mask, dtype=bool)
    for label, area in enumerate(areas, start=1):
        if int(area) >= minimum:
            keep |= labels == label
    return keep


def _convex_hull_mask(line):
    ys, xs = np.where(line)
    output = np.zeros_like(line, dtype=np.uint8)
    if len(xs) < 3:
        return output.astype(bool)
    if cv2 is not None:
        points = np.stack([xs, ys], axis=1).astype(np.int32)
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(output, hull, 255)
        return output > 127

    # This path is only used when OpenCV is unavailable.
    output[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] = 255
    return output > 127


def mask_diagnostics(mask, source, fallback=False, fallback_reason=None):
    height, width = mask.shape
    area_ratio = float(mask.mean())
    component_count, largest_component_ratio = _component_stats(mask)
    border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    border_touch_ratio = float(border.mean())

    if 0.05 <= area_ratio <= 0.90:
        area_score = 1.0
    elif area_ratio < 0.05:
        area_score = max(0.0, area_ratio / 0.05)
    else:
        area_score = max(0.0, (0.98 - area_ratio) / 0.08)
    component_score = min(1.0, largest_component_ratio / 0.80)
    border_score = max(0.0, 1.0 - border_touch_ratio / 0.15)
    confidence = 0.45 * area_score + 0.40 * component_score + 0.15 * border_score
    if fallback:
        confidence = min(confidence, 0.55)
    confidence = float(np.clip(confidence, 0.0, 1.0))

    hard_valid = (
        0.05 <= area_ratio <= 0.90
        and largest_component_ratio >= 0.80
        and border_touch_ratio <= 0.35
    )
    return {
        "mask_source": source,
        "mask_confidence": confidence,
        "mask_low_confidence": bool(confidence < 0.65 or not hard_valid),
        "mask_fallback": bool(fallback),
        "mask_fallback_reason": fallback_reason,
        "mask_area_ratio": area_ratio,
        "mask_component_count": int(component_count),
        "mask_largest_component_ratio": float(largest_component_ratio),
        "mask_border_touch_ratio": border_touch_ratio,
        "mask_width": int(width),
        "mask_height": int(height),
    }


def build_sketch_garment_mask(
    sketch,
    width,
    height,
    line_threshold=245,
    close_size=5,
    seal_size=3,
):
    if sketch is None:
        empty = np.zeros((height, width), dtype=bool)
        return Image.fromarray(empty.astype(np.uint8) * 255, mode="L"), mask_diagnostics(
            empty,
            "sketch_missing",
            fallback=True,
            fallback_reason="missing_sketch",
        )

    gray = sketch.convert("L").resize((width, height), Image.BILINEAR)
    line = np.asarray(gray, dtype=np.uint8) < int(line_threshold)
    if not line.any():
        empty = np.zeros((height, width), dtype=bool)
        return Image.fromarray(empty.astype(np.uint8) * 255, mode="L"), mask_diagnostics(
            empty,
            "sketch_empty",
            fallback=True,
            fallback_reason="no_line_pixels",
        )

    closed_line = _binary_close(line, close_size)
    barrier = _binary_dilate(closed_line, seal_size)
    garment = _filter_components(_flood_fill_enclosed(barrier))
    diagnostics = mask_diagnostics(garment, "sketch_flood_fill")

    if diagnostics["mask_low_confidence"]:
        hull = _filter_components(_convex_hull_mask(closed_line))
        hull_diagnostics = mask_diagnostics(
            hull,
            "sketch_convex_hull",
            fallback=True,
            fallback_reason="low_confidence_flood_fill",
        )
        if 0.02 <= hull_diagnostics["mask_area_ratio"] <= 0.95:
            garment = hull
            diagnostics = hull_diagnostics

    image = Image.fromarray(garment.astype(np.uint8) * 255, mode="L")
    return image, diagnostics


def estimate_cloth_foreground_mask(image, width, height):
    if image is None:
        empty = np.zeros((height, width), dtype=bool)
        return Image.fromarray(empty.astype(np.uint8) * 255, mode="L"), mask_diagnostics(
            empty,
            "target_missing",
            fallback=True,
            fallback_reason="missing_target",
        )

    rgb = image.convert("RGB").resize((width, height), Image.BICUBIC)
    array = np.asarray(rgb, dtype=np.float32)
    border = np.concatenate(
        [array[0], array[-1], array[:, 0], array[:, -1]], axis=0
    )
    background = np.median(border, axis=0)
    color_distance = np.linalg.norm(array - background[None, None, :], axis=2)
    value = array.mean(axis=2)
    foreground = (color_distance > 18.0) | (value < 235.0)
    foreground = _binary_close(foreground, 5)

    if cv2 is not None and foreground.any():
        contours, _ = cv2.findContours(
            foreground.astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if contours:
            largest = max(cv2.contourArea(contour) for contour in contours)
            filled = np.zeros_like(foreground, dtype=np.uint8)
            minimum = max(16.0, largest * 0.02)
            for contour in contours:
                if cv2.contourArea(contour) >= minimum:
                    cv2.drawContours(filled, [contour], -1, 255, thickness=-1)
            foreground = filled > 127

    foreground = _filter_components(foreground)
    diagnostics = mask_diagnostics(foreground, "target_foreground")
    output = Image.fromarray(foreground.astype(np.uint8) * 255, mode="L")
    return output, diagnostics


def mask_image_to_bool(mask_image):
    if mask_image is None:
        return None
    return np.asarray(mask_image.convert("L"), dtype=np.uint8) > 127
