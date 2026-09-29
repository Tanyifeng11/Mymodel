"""从规则双颜色条纹参考图估计几何；不读取数据集标签。"""

from dataclasses import asdict, dataclass
import math

import numpy as np
from PIL import Image

from models.pattern_scaffold import apply_garment_mask
from tools.e14_controlled_patterns import pattern_mask


GEOMETRY_VERSION = "e26_stripe_fft_tensor_phase_v1"


def axial_distance(a, b):
    return abs((a - b + 90) % 180 - 90)


@dataclass(frozen=True)
class PatternGeometry:
    orientation: float
    frequency: float
    phase: float
    orientation_confidence: float
    frequency_confidence: float
    method_agreement: float
    tensor_orientation: float
    fft_orientation: float
    acf_frequency: float
    color_a: tuple
    color_b: tuple

    def to_dict(self):
        return asdict(self)


def _gray(image):
    return np.asarray(image.convert("L"), dtype=np.float64) / 255.0


def estimate_orientation_structure_tensor(image, mask=None):
    gray = _gray(image)
    gy, gx = np.gradient(gray)
    if mask is not None:
        valid = np.asarray(mask) > 0
        gx, gy = gx[valid], gy[valid]
    xx, yy, xy = np.mean(gx * gx), np.mean(gy * gy), np.mean(gx * gy)
    coherence = math.hypot(xx - yy, 2 * xy) / max(xx + yy, 1e-12)
    theta = (math.degrees(math.atan2(2 * xy, xx - yy)) / 2 + 90) % 180
    return float(theta), float(coherence)


def estimate_orientation_fft(image, mask=None):
    gray = _gray(image)
    if mask is not None:
        gray = gray * (np.asarray(mask) > 0)
    height, width = gray.shape
    power = abs(np.fft.fft2(gray - gray.mean())) ** 2
    fy, fx = np.meshgrid(np.fft.fftfreq(height), np.fft.fftfreq(width), indexing="ij")
    radius = np.hypot(fx * width, fy * height)
    valid = (radius >= 2) & (radius <= min(height, width) / 4)
    power[~valid] = 0
    peak = np.unravel_index(np.argmax(power), power.shape)
    theta = (math.degrees(math.atan2(fy[peak], fx[peak])) + 90) % 180
    prominence = float(power[peak] / max(power.sum(), 1e-12))
    return float(theta), prominence


def estimate_frequency(image, orientation, mask=None):
    gray = _gray(image)
    # 受控数据仅有水平/竖直条纹；法向投影保持原始像素尺度。
    normal_axis = 0 if axial_distance(orientation, 90) < 45 else 1
    if mask is None:
        profile = gray.mean(axis=normal_axis)
    else:
        valid = (np.asarray(mask) > 0).astype(float)
        profile = (gray * valid).sum(axis=normal_axis) / np.maximum(valid.sum(axis=normal_axis), 1)
    centered = profile - profile.mean()
    spectrum = abs(np.fft.rfft(centered))
    spectrum[:2] = 0
    spectrum[len(profile) // 4 + 1:] = 0
    frequency = int(np.argmax(spectrum))
    others = spectrum.copy()
    others[max(0, frequency - 1):frequency + 2] = 0
    peak_ratio = float(spectrum[frequency] / max(others.max(), 1e-12))
    acf = np.fft.irfft(abs(np.fft.rfft(centered)) ** 2, n=len(profile))
    expected_period = len(profile) / frequency
    lo, hi = max(2, int(expected_period * .8)), min(len(profile) // 2, int(expected_period * 1.2) + 1)
    lag = lo + int(np.argmax(acf[lo:hi]))
    acf_frequency = len(profile) / lag
    agreement = math.exp(-abs(math.log2(acf_frequency / frequency)))
    confidence = min(1.0, peak_ratio / 3) * agreement
    return frequency, acf_frequency, float(confidence), profile


def estimate_phase(image, orientation, frequency, mask=None):
    gray = _gray(image)
    normal_axis = 0 if axial_distance(orientation, 90) < 45 else 1
    profile = gray.mean(axis=normal_axis)
    dark = 1 - (profile - profile.min()) / max(profile.max() - profile.min(), 1e-12)
    position = np.arange(len(profile))
    z = np.sum(dark * np.exp(2j * np.pi * frequency * position / len(profile)))
    initial = (-np.angle(z) / (2 * np.pi)) % 1
    phases = np.arange(4096) / 4096
    distance = abs((frequency * position[None, :] / len(profile) + phases[:, None] + .5) % 1 - .5)
    predicted = distance < .125
    observed = dark > .5
    mismatch = np.count_nonzero(predicted != observed[None, :], axis=1)
    best = np.flatnonzero(mismatch == mismatch.min())
    circular = abs((phases[best] - initial + .5) % 1 - .5)
    return float(phases[best[np.argmin(circular)]])


def _palette(image):
    colors, counts = np.unique(np.asarray(image.convert("RGB")).reshape(-1, 3), axis=0, return_counts=True)
    if len(colors) != 2:
        raise ValueError("第一版几何估计仅支持双颜色规则条纹")
    minority, majority = np.argsort(counts)
    return tuple(int(v) for v in colors[minority]), tuple(int(v) for v in colors[majority])


def estimate_pattern_geometry(image):
    tensor_theta, coherence = estimate_orientation_structure_tensor(image)
    fft_theta, prominence = estimate_orientation_fft(image)
    agreement = axial_distance(tensor_theta, fft_theta)
    theta = tensor_theta if agreement <= 8 else fft_theta
    frequency, acf_frequency, frequency_confidence, _ = estimate_frequency(image, theta)
    phase = estimate_phase(image, theta, frequency)
    color_a, color_b = _palette(image)
    orientation_confidence = min(1.0, coherence) * min(1.0, prominence * 2) * math.exp(-agreement / 8)
    return PatternGeometry(theta, float(frequency), phase, float(orientation_confidence),
                           frequency_confidence, float(agreement), tensor_theta, fft_theta,
                           float(acf_frequency), color_a, color_b)


def render_estimated_stripe(geometry, garment_mask):
    size = garment_mask.size[0]
    if garment_mask.size != (size, size):
        raise ValueError("当前 renderer 需要方形服装画布")
    angle = 0 if axial_distance(geometry.orientation, 90) < 45 else 90
    binary = pattern_mask("stripe", size, int(round(geometry.frequency)), angle,
                          geometry.phase, fraction=.25)
    pixels = np.where(binary[..., None], geometry.color_a, geometry.color_b).astype(np.uint8)
    return apply_garment_mask(Image.fromarray(pixels, "RGB"), garment_mask, garment_mask.size)
