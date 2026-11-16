"""连续VAE正目标的旧纹样代理；原PNG/OpenCV验收仍须独立执行。"""
import cv2
import numpy as np
import torch

from models.e2_lab_anchor import rgb_to_lab


class ConsistentGarmentProjection:
    """统一analytic Lab，固定模板，以显式目标差保证正目标残留为零。

    positive为[1,3,H,W]的连续RGB张量；anchor为同尺寸uint8 RGB锚图。
    当前RGB须与positive使用相同device/dtype；模板构造不参与反传。
    """

    @torch.no_grad()
    def __init__(self, positive, anchor, mask):
        assert positive.ndim == 4 and positive.shape[:2] == (1, 3)
        assert positive.dtype in (torch.float32, torch.float64)
        anchor = np.asarray(anchor)
        mask = np.asarray(mask, dtype=bool)
        assert anchor.dtype == np.uint8 and anchor.shape == (*mask.shape, 3)
        assert tuple(positive.shape[-2:]) == mask.shape
        self.inside = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5) >= 16
        assert self.inside.any()

        anchor_rgb = torch.from_numpy(anchor.copy()).permute(2, 0, 1)[None].to(positive) / 255
        self.positive_lab = rgb_to_lab(positive.detach()).clone()
        anchor_lab = rgb_to_lab(anchor_rgb)
        # 固定模板/伴随在CPU双精度构造；不把连续positive转为PNG。
        as_array = lambda value: value[0].permute(1, 2, 0).double().cpu().numpy()
        highpass = lambda value: value - cv2.blur(value, (17, 17))
        target = highpass(as_array(self.positive_lab))[self.inside]
        old = highpass(as_array(anchor_lab))[self.inside] - target
        old -= target * ((old * target).sum() / max(float((target * target).sum()), 1e-6))
        self.energy = float(np.mean(old ** 2))

        weights = np.zeros((*mask.shape, 3), dtype=np.float64)
        weights[self.inside] = old
        # 反射边界高通的伴随预先折回，反传只剩固定线性点积。
        padded = np.pad(weights, ((8, 8), (8, 8), (0, 0)))
        blurred = cv2.blur(padded, (17, 17), borderType=cv2.BORDER_CONSTANT)
        y, x = [np.pad(np.arange(n), 8, mode='reflect') for n in mask.shape]
        folded = np.zeros_like(weights)
        np.add.at(folded, (y[:, None], x[None, :]), blurred)
        coefficients = (weights - folded) / (old.size * max(self.energy, .25))
        self.coefficients = torch.from_numpy(coefficients.transpose(2, 0, 1).copy()).to(positive)

    def surrogate(self, rgb):
        assert rgb.shape == self.positive_lab.shape
        assert rgb.dtype == self.positive_lab.dtype and rgb.device == self.positive_lab.device
        if self.energy < .25:
            return rgb.sum() * 0
        difference = rgb_to_lab(rgb) - self.positive_lab
        return (difference[0] * self.coefficients).sum().abs()
