"""E25 受控条纹的非学习 oracle 渲染；与 E18.1/E23 的像素规则相同。"""

from dataclasses import dataclass

import numpy as np
from PIL import Image

from tools.e14_controlled_patterns import pattern_mask


RENDERER_VERSION = "e14_pattern_mask_fraction025_exact_v1"


@dataclass(frozen=True)
class StripePatternSpec:
    orientation: int
    frequency: int
    phase: float
    color_a: tuple
    color_b: tuple
    quarter_turns: int


def render_stripe_field(spec, size=256):
    """orientation=90 为竖条纹，0 为横条纹；不从图像预测任何参数。"""
    if spec.orientation not in (0, 90):
        raise ValueError("E25 受控数据只含水平、竖直条纹")
    binary = pattern_mask("stripe", size, spec.frequency, 0, spec.phase, fraction=.25)
    pixels = np.where(binary[..., None], spec.color_a, spec.color_b).astype(np.uint8)
    image = Image.fromarray(pixels, "RGB")
    if (90 + 90 * spec.quarter_turns) % 180 != spec.orientation:
        raise ValueError("参考旋转次数和方向 metadata 不一致")
    for _ in range(spec.quarter_turns):
        image = image.transpose(Image.Transpose.ROTATE_90)
    return image


def apply_garment_mask(pattern, garment_mask, size):
    mask = garment_mask.convert("L")
    if mask.size != size:
        raise ValueError("scaffold 必须使用验证集原始 garment mask")
    if not set(np.unique(np.asarray(mask))).issubset({0, 255}):
        raise ValueError("scaffold 需要二值 mask，避免背景混入纹样")
    return Image.composite(pattern.resize(size, Image.Resampling.BILINEAR),
                           Image.new("RGB", size, "white"), mask)
