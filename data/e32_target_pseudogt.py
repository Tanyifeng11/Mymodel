"""E32 固定自动监督；只在离线监督构建时读取 target RGB。"""

import cv2
import numpy as np
import torch
from PIL import Image

from garment_mask_utils import build_sketch_garment_mask, estimate_cloth_foreground_mask
from models.local_pattern_field import patch_geometry


SIZE = (384, 512)  # PIL 为 W,H；field 为 H=64,W=48。
COARSE = (12, 16)
FIELD = (48, 64)
TAU = .25


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def image_at(path):
    return Image.open(path).convert('RGB').resize(SIZE, Image.Resampling.BICUBIC)


def masks(sketch, target=None):
    # 推理 mask 只能从 sketch 得到。target mask 只给监督分支使用。
    input_mask, info = build_sketch_garment_mask(sketch, *SIZE)
    input_mask = np.asarray(input_mask) > 127
    if target is None:
        return input_mask, info
    gt_mask, gt_info = estimate_cloth_foreground_mask(target, *SIZE)
    gt_mask = np.asarray(gt_mask) > 127
    interior = cv2.erode(gt_mask.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    return input_mask, interior, dict(input=info, supervision=gt_info)


def lab_hist(image, mask=None):
    lab = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2LAB).astype(np.float32) / 255
    values = lab.reshape(-1, 3) if mask is None else lab[mask]
    if not len(values):
        return np.zeros(24, np.float32)
    hist = np.concatenate([np.histogram(values[:, c], bins=8, range=(0, 1))[0] for c in range(3)]).astype(np.float32)
    return unit(hist)


def statistical_descriptor(patch):
    """16 radial FFT +8 radial ACF +8 radial self-sim +6 Lab；不含方向标签。"""
    rgb = np.asarray(patch)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    centered = gray - gray.mean()
    window = np.outer(np.hanning(64), np.hanning(64))
    power = abs(np.fft.fft2(centered * window)) ** 2
    fy, fx = np.meshgrid(np.fft.fftfreq(64), np.fft.fftfreq(64), indexing='ij')
    radius = np.hypot(fx, fy)
    radial = np.array([np.log1p(power[(radius >= i/64) & (radius < (i+1)/64)].mean())
                       for i in range(1, 17)], np.float32)
    acf = np.fft.fftshift(np.fft.ifft2(abs(np.fft.fft2(centered))**2).real)
    yy, xx = np.indices((64, 64))
    distance = np.hypot(xx-32, yy-32)
    ac = np.array([acf[(distance >= 2*i+1) & (distance < 2*i+3)].mean()
                   for i in range(8)], np.float32) / max(float(acf[32, 32]), 1e-6)
    # 各方向平均的固定 lag 自相似，不用 E31 shift identity 充当真值。
    sim = []
    variance = max(float(centered.var()), 1e-5)
    for lag in range(2, 18, 2):
        diffs = [(gray-np.roll(gray, (dy, dx), (0, 1)))**2
                 for dy, dx in [(0, lag), (lag, 0), (lag, lag), (lag, -lag)]]
        sim.append(np.exp(-np.mean(diffs)/variance))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32) / 255
    color = np.r_[lab.mean((0, 1)), lab.std((0, 1))]
    return np.r_[unit(radial), unit(ac), unit(np.asarray(sim, np.float32)), color].astype(np.float32)


class FrozenFeatures:
    def __init__(self, dino, device):
        self.dino, self.device = dino, device
        # 固定随机投影，不拟合 dev/test，也不重新选择投影。
        rng = np.random.default_rng(32042)
        self.projection = np.linalg.qr(rng.normal(size=(384, 26)))[0].astype(np.float32)

    @torch.inference_mode()
    def perceptual(self, image, invariant):
        rgb = np.asarray(image)
        versions = [np.rot90(rgb, k).copy() for k in range(4 if invariant else 1)]
        resized = np.stack([cv2.resize(v, (448, 448), interpolation=cv2.INTER_AREA) for v in versions])
        t = torch.from_numpy(resized).permute(0, 3, 1, 2).float().to(self.device)/255
        mean = t.new_tensor([.485, .456, .406])[None, :, None, None]
        std = t.new_tensor([.229, .224, .225])[None, :, None, None]
        layers = self.dino.get_intermediate_layers((t-mean)/std, n=4, reshape=True)
        maps = torch.stack(layers).mean(0).cpu().numpy().transpose(0, 2, 3, 1)
        aligned = [cv2.resize(np.rot90(v, -k).copy(), COARSE) for k, v in enumerate(maps)]
        return unit(aligned[0]), unit(np.mean(aligned, axis=0))

    def extract(self, image, supervision_mask=None, appearance=False):
        dino, invariant = self.perceptual(image, appearance)
        rgb = np.asarray(image)
        geometry = np.zeros((16, 12, 4), np.float32)
        descriptors = np.zeros((16, 12, 64), np.float32) if appearance else None
        occupancy = np.ones((16, 12), np.float32)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32) / 255
        mean = cv2.GaussianBlur(lab, (0, 0), 8)
        std = np.sqrt(np.maximum(cv2.GaussianBlur(lab*lab, (0, 0), 8)-mean*mean, 0))
        for y in range(16):
            for x in range(12):
                cy, cx = round((y+.5)*512/16), round((x+.5)*384/12)
                y0, x0 = min(max(cy-32, 0), 448), min(max(cx-32, 0), 320)
                patch = image.crop((x0, y0, x0+64, y0+64))
                g = patch_geometry(patch)
                if supervision_mask is not None:
                    occupancy[y, x] = supervision_mask[y0:y0+64, x0:x0+64].mean()
                theta = np.deg2rad(2*g['orientation'])
                conf = g['confidence'] if occupancy[y, x] >= .95 else 0.
                geometry[y, x] = [np.cos(theta), np.sin(theta), np.log(max(g['frequency'], 1e-4)), conf]
                if appearance:
                    stat = statistical_descriptor(patch)
                    projected = unit(invariant[y, x] @ self.projection)
                    descriptors[y, x] = np.r_[stat, projected]
        color = cv2.resize(np.concatenate([mean, std], -1), COARSE)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
        selfsim = cv2.resize(abs(gray-cv2.GaussianBlur(gray, (0, 0), 8)), COARSE)[..., None]
        inputs = np.concatenate([dino, geometry, color, selfsim], -1)
        result = dict(reference=inputs.astype(np.float16), geometry=geometry, occupancy=occupancy,
                      histogram=lab_hist(image, supervision_mask))
        if appearance:
            result['appearance'] = descriptors
        return result


def upsample_geometry(geometry):
    resized = cv2.resize(geometry, FIELD, interpolation=cv2.INTER_LINEAR)
    resized[..., :2] = unit(resized[..., :2])
    return resized


def structure_input(sketch, mask):
    # 七个通道：RGB sketch, mask, x, y, distance；绝无 target RGB。
    rgb = np.asarray(sketch).astype(np.float32)/255
    yy, xx = np.indices((512, 384), dtype=np.float32)
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)/384
    return np.concatenate([rgb, mask[..., None].astype(np.float32),
                           (xx/383)[..., None], (yy/511)[..., None], distance[..., None]], -1)
