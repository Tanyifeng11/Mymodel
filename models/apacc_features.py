"""E30 的冻结外观与局部纹样特征；只读取参考 RGB 和估计前景。"""

import hashlib
import time

import cv2
import numpy as np
import torch
from PIL import Image

from models.confidence_local_correspondence import foreground
from models.local_pattern_field import patch_geometry


def load_dino(device):
    error = None
    for attempt in range(5):
        try:
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14',
                                   trust_repo=True).to(device).eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            digest = hashlib.sha256()
            for value in model.state_dict().values():
                digest.update(value.detach().cpu().numpy().tobytes())
            return model, digest.hexdigest()
        except Exception as exc:
            error = exc
            time.sleep(min(2 ** attempt, 16))
    raise RuntimeError('DINOv2 权重获取失败（重试 5 次）') from error


@torch.inference_mode()
def extract_dense_features(image, mask, model, device, grid=64):
    """DINOv2-S/14 的末四层均值，几何沿用 E26 的 patch_geometry。"""
    rgb = np.asarray(image.convert('RGB'))
    h, w = rgb.shape[:2]
    resized = cv2.resize(rgb, (448, 448), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized.copy()).permute(2, 0, 1).float()[None].to(device) / 255
    mean = tensor.new_tensor([.485, .456, .406])[None, :, None, None]
    std = tensor.new_tensor([.229, .224, .225])[None, :, None, None]
    layers = model.get_intermediate_layers((tensor - mean) / std, n=4, reshape=True)
    appearance = torch.stack(layers).mean(0)
    appearance = torch.nn.functional.interpolate(appearance, (grid, grid), mode='bilinear', align_corners=False)[0]
    appearance = torch.nn.functional.normalize(appearance, dim=0).cpu().numpy().transpose(1, 2, 0)

    # 32px 局部窗口在 16x16 粗网格上测量，避免在 4px cell 上估计频率。
    coarse = 16
    geo = np.zeros((coarse, coarse, 4), np.float32)
    for gy in range(coarse):
        for gx in range(coarse):
            cx, cy = round((gx + .5) * w / coarse), round((gy + .5) * h / coarse)
            x0, y0 = min(max(cx - 16, 0), w - 32), min(max(cy - 16, 0), h - 32)
            info = patch_geometry(image.crop((x0, y0, x0 + 32, y0 + 32)))
            angle = np.deg2rad(2 * info['orientation'])
            geo[gy, gx] = [np.cos(angle), np.sin(angle),
                            np.log(max(info['frequency'], 1e-4)), info['confidence']]
    geometry = cv2.resize(geo, (grid, grid), interpolation=cv2.INTER_LINEAR)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32) / 255
    color = cv2.resize(lab, (grid, grid), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    detail = abs(gray - cv2.GaussianBlur(gray, (0, 0), 4))
    selfsim = cv2.resize(cv2.GaussianBlur(detail, (0, 0), 8), (grid, grid))
    occupancy = cv2.resize(mask.astype(np.float32), (grid, grid), interpolation=cv2.INTER_AREA)
    return {'appearance': appearance, 'geometry': geometry, 'color': color,
            'self_similarity': selfsim, 'occupancy': occupancy}


def estimated_foreground(image):
    return foreground(image)
