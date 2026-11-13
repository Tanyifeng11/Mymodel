"""保留原损失作A臂；B臂使用相对色度、局部频谱和不抵消的旧边缘监督。"""
import torch
from torch.nn import functional as F

from models.e2_lab_anchor import rgb_to_lab
from tools.e2_b_material_losses import loss_terms as original_terms, weighted


def geometry(mask):
    """只保留mask内侧轮廓；原草图继续原样传给冻结M0。"""
    import cv2
    import numpy as np
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    return np.clip(1 - distance / 8, 0, 1) * mask


def gradients(value):
    return torch.cat((F.pad(value[..., 1:] - value[..., :-1], (0, 1, 0, 0)),
                      F.pad(value[..., 1:, :] - value[..., :-1, :], (0, 0, 0, 1))), 1)


def suppression(anchor, target, weight):
    """目标没有或较弱的旧边缘提供监督权重；正确目标本身的损失为零。"""
    old, new = [gradients(rgb_to_lab(x) / 100).abs().mean(1, keepdim=True)
                for x in (anchor, target)]
    return (((old - new).clamp_min(0) / (old + new + .005)) * (weight >= .999)).detach()


def spectrum(current, target, coords):
    """已注册的受控目标允许比较复数频谱，保留相位和两个格纹方向。"""
    crops = lambda field: torch.stack([field[0, :, y:y+64, x:x+64] for x, y in coords])
    a, b = crops(current), crops(target)
    losses = []
    for size in (64, 32):
        def tiles(field):
            if size == 64:
                return field
            return field.reshape(-1, 3, 2, 32, 2, 32).permute(0, 2, 4, 1, 3, 5).reshape(-1, 3, 32, 32)
        p, q = tiles(a), tiles(b)
        p, q = p - p.mean((-2, -1), keepdim=True), q - q.mean((-2, -1), keepdim=True)
        scale = q.square().mean((-2, -1), keepdim=True).sqrt().clamp_min(.02).detach()
        difference = torch.fft.rfft2(p / scale, norm='ortho') - torch.fft.rfft2(q / scale, norm='ortho')
        energy = difference.abs().square()
        focus = difference.detach().abs()
        focus = focus / focus.amax((-2, -1), keepdim=True).clamp_min(1e-8)
        losses.append((focus * energy).mean())
    return sum(losses) / len(losses)


def loss_terms(rgb, goal, base, weight, old_template, coords, old_support):
    terms = original_terms(rgb, goal, base, weight, old_template, coords)
    current, target = rgb_to_lab(rgb) / 100, rgb_to_lab(goal) / 100
    interior = (weight >= .999).to(weight.dtype)
    mean = lambda x: (x * interior).sum((-2, -1), keepdim=True) / interior.sum().clamp_min(1)
    a, b = current[:, 1:], target[:, 1:]
    ma, mb = mean(a), mean(b)
    # 2 Lab单位下限防止近灰色和量化噪声被过度放大；不将整件a、b设为常数。
    mean_scale = mean(b.square()).mean(1, keepdim=True).sqrt().clamp_min(.02).detach()
    spatial_scale = mean((b - mb).square()).mean(1, keepdim=True).sqrt().clamp_min(.02).detach()
    terms['chroma'] = ((ma - mb).abs() / mean_scale).mean() / 2
    terms['chroma'] += weighted(((a - ma) - (b - mb)).abs() / spatial_scale, interior) / 2
    terms['palette'] = terms['palette'] / mean_scale.squeeze()
    terms['pattern'] = spectrum(current, target, coords)
    terms['old'] = weighted(gradients(current - target).abs(), old_support)
    return terms
