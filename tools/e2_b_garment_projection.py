"""固定衣身锚图的全衣身旧纹样投影。"""
import cv2
import numpy as np
import torch

from models.e2_lab_anchor import rgb_to_lab


def interior(mask):
    return cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5) >= 16


def lab(pixels):
    return cv2.cvtColor(np.asarray(pixels, np.float32) / 255, cv2.COLOR_RGB2LAB)


def highpass(field):
    return field - cv2.blur(field, (17, 17))


def basis(positive, anchor):
    template = anchor - positive
    template -= positive * (np.sum(template * positive) / max(float(np.sum(positive * positive)), 1e-6))
    return template, float(np.mean(template ** 2))


def projection(current, positive, template, energy):
    return float(abs(np.mean((current - positive) * template)) / energy) if energy >= .25 else None


class GarmentProjection:
    def __init__(self, positive, anchor, mask, device):
        self.inside = interior(mask)
        self.positive = highpass(lab(positive))[self.inside]
        self.template, self.energy = basis(self.positive, highpass(lab(anchor))[self.inside])
        weights = np.zeros((*mask.shape, 3), dtype=np.float32)
        weights[self.inside] = self.template
        # 把反射边界高通作用到固定模板的伴随上，避免GPU反射填充的非确定性反传。
        padded = np.pad(weights, ((8, 8), (8, 8), (0, 0)))
        blurred = cv2.blur(padded, (17, 17), borderType=cv2.BORDER_CONSTANT)
        y, x = [np.pad(np.arange(n), 8, mode='reflect') for n in mask.shape]
        folded = np.zeros_like(weights)
        np.add.at(folded, (y[:, None], x[None, :]), blurred)
        scale = self.template.size * max(self.energy, .25)
        self.coefficients = torch.from_numpy((weights - folded).transpose(2, 0, 1).copy()).to(device) / scale
        self.offset = float(np.sum(self.positive * self.template) / scale)

    def exact(self, pixels):
        return projection(highpass(lab(pixels))[self.inside], self.positive, self.template, self.energy)

    def surrogate(self, rgb):
        if self.energy < .25:
            return rgb.sum() * 0
        return ((rgb_to_lab(rgb)[0] * self.coefficients).sum() - self.offset).abs()
