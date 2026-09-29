"""平面服装上的显式 reference 坐标场。"""

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


FIELD_VERSION = "e26_canonical_affine_interior_v2"


@dataclass(frozen=True)
class PatternCoordinateField:
    uv: np.ndarray
    confidence: np.ndarray
    orientation: tuple
    scale: tuple
    bbox: tuple
    canonical_reference: np.ndarray
    canonical_target: np.ndarray
    reference_basis: np.ndarray
    target_basis: np.ndarray
    origin: np.ndarray


def garment_coordinates(garment_mask):
    mask = np.asarray(garment_mask.convert("L")) > 0
    yy, xx = np.where(mask)
    xmin, xmax, ymin, ymax = int(xx.min()), int(xx.max()), int(yy.min()), int(yy.max())
    grid_y, grid_x = np.indices(mask.shape)
    xg = (grid_x - xmin) / max(xmax - xmin, 1)
    yg = (grid_y - ymin) / max(ymax - ymin, 1)
    return xg, yg, (xmin, ymin, xmax, ymax), mask


def _normal(orientation):
    angle = np.deg2rad(orientation - 90)
    return np.array((np.cos(angle), np.sin(angle)))


def _build(garment_mask, reference_basis, target_basis, origin, source_size, orientation, scale):
    xg, yg, bbox, mask = garment_coordinates(garment_mask)
    xy = np.stack((xg * (source_size[0] - 1), yg * (source_size[1] - 1)), -1)
    canonical_target = xy @ target_basis.T + origin
    uv = ((canonical_target - origin) @ np.linalg.inv(reference_basis).T).astype(np.float32)
    sy, sx = np.indices((source_size[1], source_size[0]))
    canonical_reference = np.stack((sx, sy), -1) @ reference_basis.T + origin
    interior = cv2.erode(mask.astype(np.uint8), np.ones((17, 17), np.uint8)) > 0
    return PatternCoordinateField(uv, interior.astype(np.float32), tuple(orientation), tuple(scale), bbox,
                                  canonical_reference.astype(np.float32), canonical_target.astype(np.float32),
                                  reference_basis, target_basis, origin)


def build_affine_pattern_field(garment_mask, orientation, frequency, phase, source_size=(256, 256),
                               target_orientation=None, target_frequency=None, target_phase=None):
    normal = _normal(orientation)
    tangent = np.array((-normal[1], normal[0]))
    reference_basis = np.stack((normal * frequency, tangent)) / source_size[0]
    target_normal = _normal(orientation if target_orientation is None else target_orientation)
    target_tangent = np.array((-target_normal[1], target_normal[0]))
    target_basis = np.stack((target_normal * (frequency if target_frequency is None else target_frequency),
                             target_tangent)) / source_size[0]
    field = _build(garment_mask, reference_basis, target_basis, np.array((phase, 0.)),
                   source_size, (orientation,), (frequency, phase))
    if target_phase is not None:
        field.canonical_target[..., 0] += target_phase - phase
        field.uv[:] += np.linalg.solve(reference_basis, np.array((target_phase - phase, 0.)))
    return field


def build_plaid_field(garment_mask, axis1, axis2, source_size=(256, 256), target_axes=None):
    axes = (axis1, axis2)
    reference_basis = np.stack([_normal(a[0]) * a[1] / source_size[0] for a in axes])
    target_basis = (reference_basis.copy() if target_axes is None else
                    np.stack([_normal(a[0]) * a[1] / source_size[0] for a in target_axes]))
    return _build(garment_mask, reference_basis, target_basis, np.array((axis1[2], axis2[2])),
                  source_size, (axis1[0], axis2[0]), (axis1[1], axis2[1]))


def build_lattice_field(garment_mask, v1, v2, origin, source_size=(256, 256), target_vectors=None):
    reference_basis = np.linalg.inv(np.stack((v1, v2), -1))
    target_basis = (reference_basis.copy() if target_vectors is None else
                    np.linalg.inv(np.stack(target_vectors, -1)))
    return _build(garment_mask, reference_basis, target_basis,
                  -reference_basis @ np.asarray(origin), source_size,
                  (float(np.degrees(np.arctan2(v1[1], v1[0]))),
                   float(np.degrees(np.arctan2(v2[1], v2[0])))),
                  (float(np.linalg.norm(v1)), float(np.linalg.norm(v2))))


def sample_reference_with_field(reference, field, garment_mask):
    source = np.asarray(reference.convert("RGB"))
    mapped = cv2.remap(source, field.uv[..., 0], field.uv[..., 1],
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    # 低置信边界沿用简单 appearance 映射，几何只在 eroded interior 强制。
    fallback = np.asarray(reference.resize(garment_mask.size, Image.Resampling.BILINEAR))
    pixels = np.where(field.confidence[..., None] > 0, mapped, fallback)
    pixels = np.where((np.asarray(garment_mask) > 0)[..., None], pixels, 255)
    return Image.fromarray(pixels.astype(np.uint8), "RGB")
