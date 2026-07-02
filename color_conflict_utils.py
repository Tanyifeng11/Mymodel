import re
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image


COLOR_TABLE = {
    "red": (220, 30, 30),
    "blue": (40, 80, 220),
    "black": (20, 20, 20),
    "white": (235, 235, 235),
    "pink": (235, 110, 170),
    "green": (45, 150, 70),
    "yellow": (230, 205, 40),
    "purple": (130, 70, 170),
    "brown": (120, 75, 45),
    "gray": (135, 135, 135),
    "grey": (135, 135, 135),
    "orange": (230, 130, 35),
    "beige": (205, 180, 135),
    "\u7ea2\u8272": (220, 30, 30),
    "\u7ea2": (220, 30, 30),
    "\u84dd\u8272": (40, 80, 220),
    "\u84dd": (40, 80, 220),
    "\u9ed1\u8272": (20, 20, 20),
    "\u9ed1": (20, 20, 20),
    "\u767d\u8272": (235, 235, 235),
    "\u767d": (235, 235, 235),
    "\u7c89\u8272": (235, 110, 170),
    "\u7c89": (235, 110, 170),
    "\u7eff\u8272": (45, 150, 70),
    "\u7eff": (45, 150, 70),
    "\u9ec4\u8272": (230, 205, 40),
    "\u9ec4": (230, 205, 40),
    "\u7d2b\u8272": (130, 70, 170),
    "\u7d2b": (130, 70, 170),
    "\u68d5\u8272": (120, 75, 45),
    "\u68d5": (120, 75, 45),
    "\u7070\u8272": (135, 135, 135),
    "\u7070": (135, 135, 135),
}

EN_COLOR_KEYS = sorted(
    [k for k in COLOR_TABLE if re.fullmatch(r"[a-z]+", k)],
    key=len,
    reverse=True,
)
ZH_COLOR_KEYS = sorted(
    [k for k in COLOR_TABLE if not re.fullmatch(r"[a-z]+", k)],
    key=len,
    reverse=True,
)


def extract_text_color(prompt: str) -> Tuple[Optional[str], Optional[Tuple[int, int, int]]]:
    text = (prompt or "").lower()
    for color in EN_COLOR_KEYS:
        if re.search(rf"\b{re.escape(color)}\b", text):
            return color, COLOR_TABLE[color]
    raw_prompt = prompt or ""
    for color in ZH_COLOR_KEYS:
        if color in raw_prompt:
            return color, COLOR_TABLE[color]
    return None, None


def rgb_to_lab(rgb: Any) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float32).reshape(-1, 3) / 255.0
    arr = np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)
    mat = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    )
    xyz = arr @ mat.T
    xyz /= np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    delta = 6.0 / 29.0
    f_xyz = np.where(
        xyz > delta**3,
        np.cbrt(xyz),
        xyz / (3.0 * delta * delta) + 4.0 / 29.0,
    )
    lab = np.stack(
        [
            116.0 * f_xyz[:, 1] - 16.0,
            500.0 * (f_xyz[:, 0] - f_xyz[:, 1]),
            200.0 * (f_xyz[:, 1] - f_xyz[:, 2]),
        ],
        axis=-1,
    )
    return lab.reshape(np.asarray(rgb).shape[:-1] + (3,))


def delta_e_rgb(rgb_a: Any, rgb_b: Any) -> float:
    lab_a = rgb_to_lab(rgb_a).reshape(-1, 3)[0]
    lab_b = rgb_to_lab(rgb_b).reshape(-1, 3)[0]
    return float(np.linalg.norm(lab_a - lab_b))


def _dominant_rgb_from_array(arr: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[int, int, int]:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.max() <= 1.5 and arr.min() >= -1.5:
        arr = (arr + 1.0) * 127.5 if arr.min() < 0.0 else arr * 255.0
    arr = np.clip(arr, 0, 255)
    pixels = arr.reshape(-1, 3)
    if mask is not None:
        m = np.asarray(mask)
        if m.ndim == 3:
            m = m[0]
        if m.shape[:2] == arr.shape[:2]:
            keep = m.reshape(-1) > 0.5
            if keep.sum() > 16:
                pixels = pixels[keep]
    if pixels.shape[0] > 4096:
        step = max(1, pixels.shape[0] // 4096)
        pixels = pixels[::step]
    rgb = np.median(pixels, axis=0)
    return tuple(int(round(float(x))) for x in rgb)


def dominant_rgb_from_pil(image: Image.Image, mask: Optional[Image.Image] = None) -> Tuple[int, int, int]:
    rgb = image.convert("RGB")
    mask_arr = None
    if mask is not None:
        mask_arr = np.asarray(mask.convert("L").resize(rgb.size, Image.NEAREST), dtype=np.float32) / 255.0
    return _dominant_rgb_from_array(np.asarray(rgb), mask_arr)


def dominant_rgb_from_tensor(tensor: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[int, int, int]:
    arr = tensor.detach().float().cpu().numpy()
    mask_arr = None if mask is None else mask.detach().float().cpu().numpy()
    return _dominant_rgb_from_array(arr, mask_arr)


def conflict_bucket(score: float, has_text_color: bool) -> str:
    if not has_text_color:
        return "no_text_color"
    if score < 0.25:
        return "low_conflict"
    if score < 0.55:
        return "mid_conflict"
    return "high_conflict"


def compute_color_conflict(
    prompt: str,
    ref_rgb: Optional[Tuple[int, int, int]] = None,
    ref_image: Optional[Image.Image] = None,
    ref_tensor: Optional[torch.Tensor] = None,
    mask: Optional[Any] = None,
    deltae_norm: float = 50.0,
) -> Dict[str, Any]:
    text_color, text_rgb = extract_text_color(prompt)
    if ref_rgb is None:
        if ref_tensor is not None:
            ref_rgb = dominant_rgb_from_tensor(ref_tensor, mask if torch.is_tensor(mask) else None)
        elif ref_image is not None:
            ref_rgb = dominant_rgb_from_pil(ref_image, mask if isinstance(mask, Image.Image) else None)
        else:
            ref_rgb = (0, 0, 0)

    has_text_color = text_rgb is not None
    if has_text_color:
        delta_e = delta_e_rgb(text_rgb, ref_rgb)
        norm = max(float(deltae_norm), 1e-6)
        score = max(0.0, min(1.0, delta_e / norm))
    else:
        delta_e = 0.0
        score = 0.0
        text_rgb = (0, 0, 0)

    return {
        "text_color": text_color or "",
        "text_color_rgb": list(text_rgb),
        "ref_palette_rgb": list(ref_rgb),
        "has_text_color": bool(has_text_color),
        "color_conflict_score": float(score),
        "color_delta_e": float(delta_e),
        "conflict_bucket": conflict_bucket(float(score), bool(has_text_color)),
    }


def summarize_conflict_rows(rows, threshold: float = 0.55) -> Dict[str, Any]:
    scores = []
    buckets = {
        "no_text_color": 0,
        "low_conflict": 0,
        "mid_conflict": 0,
        "high_conflict": 0,
    }
    for row in rows:
        has_text = bool(row.get("has_text_color"))
        try:
            score = float(row.get("color_conflict_score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        scores.append(score)
        buckets[conflict_bucket(score, has_text)] += 1
    return {
        "mean_conflict_score": float(np.mean(scores)) if scores else 0.0,
        "num_conflict_score_gt_threshold": int(sum(s > float(threshold) for s in scores)),
        "no_text_color_count": buckets["no_text_color"],
        "low_conflict_count": buckets["low_conflict"],
        "mid_conflict_count": buckets["mid_conflict"],
        "high_conflict_count": buckets["high_conflict"],
    }
