"""真实局部条纹的行位移矫正与局部方向/频率/置信度；无训练。"""

from dataclasses import dataclass
import math

import cv2
import numpy as np
from PIL import Image

from models.pattern_geometry import axial_distance, estimate_orientation_structure_tensor


LOCAL_VERSION = "e26_local64_row_rectification_v1"


@dataclass(frozen=True)
class LocalPatternField:
    orientation: np.ndarray
    frequency: np.ndarray
    confidence: np.ndarray
    valid: np.ndarray


def patch_geometry(image):
    gray = np.asarray(image.convert("L"), dtype=float) / 255
    theta, coherence = estimate_orientation_structure_tensor(image)
    height, width = gray.shape
    power = abs(np.fft.fft2((gray - gray.mean()) * np.outer(np.hanning(height), np.hanning(width)))) ** 2
    fy, fx = np.meshgrid(np.fft.fftfreq(height), np.fft.fftfreq(width), indexing="ij")
    radius = np.hypot(fx, fy)
    weights = power * ((radius >= 1.5 / min(width, height)) & (radius <= .20))
    z = np.sum(weights * np.exp(2j * np.arctan2(fy, fx))) / max(weights.sum(), 1e-12)
    fft_theta = (np.degrees(np.angle(z)) / 2 + 90) % 180
    axis = 0 if axial_distance(theta, 90) < 45 else 1
    profile = gray.mean(axis)
    spectrum = abs(np.fft.rfft((profile - profile.mean()) * np.hanning(len(profile)), n=2048))
    frequencies = np.fft.rfftfreq(2048)
    spectrum[(frequencies < 1.5 / len(profile)) | (frequencies > .20)] = 0
    peak = int(np.argmax(spectrum))
    prominence = float(spectrum[max(0, peak - 16):peak + 17].sum() / max(spectrum.sum(), 1e-12))
    agreement = axial_distance(theta, fft_theta)
    valid = bool(gray.std() >= .02 and coherence >= .25 and abs(z) >= .35 and agreement <= 20)
    confidence = min(coherence, float(abs(z))) * math.exp(-agreement / 20) * min(1., prominence * 3)
    return {"orientation": float(theta), "frequency": float(frequencies[peak]),
            "confidence": float(confidence if valid else 0), "valid": valid}


def estimate_local_pattern_field(image, patch_size=64, stride=32):
    height, width = image.height, image.width
    values = []
    for y in range(0, height - patch_size + 1, stride):
        row = []
        for x in range(0, width - patch_size + 1, stride):
            row.append(patch_geometry(image.crop((x, y, x + patch_size, y + patch_size))))
        values.append(row)
    arrays = {key: np.array([[p[key] for p in row] for row in values])
              for key in ("orientation", "frequency", "confidence", "valid")}
    # 方向使用 pi 周期插值，避免 179° 与 1° 被平均成 90°。
    theta = np.deg2rad(arrays["orientation"] * 2)
    sine = cv2.resize(np.sin(theta), (width, height), interpolation=cv2.INTER_LINEAR)
    cosine = cv2.resize(np.cos(theta), (width, height), interpolation=cv2.INTER_LINEAR)
    orientation = (np.degrees(np.arctan2(sine, cosine)) / 2) % 180
    return LocalPatternField(orientation.astype(np.float32),
        cv2.resize(arrays["frequency"].astype(np.float32), (width, height)),
        cv2.resize(arrays["confidence"].astype(np.float32), (width, height)),
        cv2.resize(arrays["valid"].astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0)


def rectify_reference(image):
    theta, coherence = estimate_orientation_structure_tensor(image)
    vertical = axial_distance(theta, 90) < 45
    rgb = np.asarray(image.convert("RGB"))
    oriented = rgb if vertical else rgb.transpose(1, 0, 2)
    gray = cv2.cvtColor(oriented, cv2.COLOR_RGB2GRAY).astype(float) / 255
    profiles = gray - cv2.GaussianBlur(gray, (0, 0), 12, 0)
    template = np.median(profiles, axis=0)
    spectrum = abs(np.fft.rfft(template - template.mean()))
    spectrum[:2] = 0
    frequency = max(2, int(np.argmax(spectrum)))
    max_shift = max(1, int(gray.shape[1] / frequency / 4))
    shifts = np.arange(-max_shift, max_shift + 1)
    margin = max_shift + 8
    scores = np.stack([np.sum(profiles[:, margin:-margin] *
                      np.roll(template, int(shift))[None, margin:-margin], axis=1) for shift in shifts], -1)
    displacement = shifts[np.argmax(scores, axis=-1)].astype(np.float32)
    displacement = cv2.GaussianBlur(displacement[:, None], (1, 17), 3).ravel()
    yy, xx = np.indices(gray.shape, dtype=np.float32)
    uv = np.stack((xx + displacement[:, None], yy), -1)
    rectified = cv2.remap(oriented, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    if not vertical:
        rectified = rectified.transpose(1, 0, 2)
        uv = uv.transpose(1, 0, 2)[..., ::-1]
    return Image.fromarray(rectified, "RGB"), uv, {"orientation": float(theta),
            "coherence": float(coherence), "max_displacement_px": float(abs(displacement).max()),
            "normal_period_px": float(gray.shape[1] / frequency), "method": "bounded row-profile correlation"}


def render_confidence_aware(rectified, garment_field, local_field, garment_mask):
    source = np.asarray(rectified.convert("RGB"))
    uv = garment_field.uv
    mapped = cv2.remap(source, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    confidence = cv2.remap(local_field.confidence, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101) * garment_field.confidence
    # 置信度 < .25 的位置只使用参考中位颜色；其余位置连续混合。
    confidence = np.clip((confidence - .25) / .5, 0, 1)
    fallback = np.median(source.reshape(-1, 3), axis=0)
    pixels = confidence[..., None] * mapped + (1 - confidence[..., None]) * fallback
    pixels[np.asarray(garment_mask) == 0] = 255
    return Image.fromarray(np.round(pixels).astype(np.uint8), "RGB"), confidence
