"""平面服装上的显式 reference 坐标场。"""

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


FIELD_VERSION = "e26_bbox_uv_interior_v1"


@dataclass(frozen=True)
class PatternCoordinateField:
    uv: np.ndarray
    confidence: np.ndarray
    orientation: tuple
    scale: tuple
    bbox: tuple


def garment_coordinates(garment_mask):
    mask = np.asarray(garment_mask.convert("L")) > 0
    yy, xx = np.where(mask)
    xmin, xmax, ymin, ymax = int(xx.min()), int(xx.max()), int(yy.min()), int(yy.max())
    grid_y, grid_x = np.indices(mask.shape)
    xg = (grid_x - xmin) / max(xmax - xmin, 1)
    yg = (grid_y - ymin) / max(ymax - ymin, 1)
    return xg, yg, (xmin, ymin, xmax, ymax), mask


def build_affine_pattern_field(garment_mask, orientation, frequency, phase, source_size=(256, 256)):
    xg, yg, bbox, mask = garment_coordinates(garment_mask)
    uv = np.stack((xg * (source_size[0] - 1), yg * (source_size[1] - 1)), -1).astype(np.float32)
    interior = cv2.erode(mask.astype(np.uint8), np.ones((17, 17), np.uint8)) > 0
    return PatternCoordinateField(uv, interior.astype(np.float32), (float(orientation),),
                                  (float(frequency), float(phase)), bbox)


def build_plaid_field(garment_mask, axis1, axis2, source_size=(256, 256)):
    field = build_affine_pattern_field(garment_mask, axis1[0], axis1[1], axis1[2], source_size)
    return PatternCoordinateField(field.uv, field.confidence, (float(axis1[0]), float(axis2[0])),
                                  (float(axis1[1]), float(axis2[1])), field.bbox)


def build_lattice_field(garment_mask, v1, v2, origin, source_size=(256, 256)):
    field = build_affine_pattern_field(garment_mask, 0, 1, 0, source_size)
    return PatternCoordinateField(field.uv, field.confidence, (0., 90.),
                                  (float(v1[0]), float(v2[1])), field.bbox)


def sample_reference_with_field(reference, field, garment_mask):
    source = np.asarray(reference.convert("RGB"))
    mapped = cv2.remap(source, field.uv[..., 0], field.uv[..., 1],
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    # 低置信边界沿用简单 appearance 映射，几何只在 eroded interior 强制。
    fallback = np.asarray(reference.resize(garment_mask.size, Image.Resampling.BILINEAR))
    pixels = np.where(field.confidence[..., None] > 0, mapped, fallback)
    pixels = np.where((np.asarray(garment_mask) > 0)[..., None], pixels, 255)
    return Image.fromarray(pixels.astype(np.uint8), "RGB")
