"""Lab目标与显式旧纹样投影；合成目标和损失使用同一颜色定义。"""
import cv2
import numpy as np
import torch
from torch.nn import functional as F

from tools.e2_state_b_pilot import lab_tensor, weighted_mean, pooled_chroma


def lab_to_rgb(lab):
    """不裁剪色域，供合成目标检测RGB越界。"""
    fy = (lab[..., 0] + 16) / 116
    f = np.stack((fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200), -1)
    xyz = np.where(f > 6 / 29, f ** 3, (f - 16 / 116) / 7.787037)
    xyz *= np.array([.95047, 1., 1.08883])
    matrix = np.array([[.4124564, .3575761, .1804375], [.2126729, .7151522, .0721750], [.0193339, .1191920, .9503041]])
    linear = xyz @ np.linalg.inv(matrix).T
    return np.where(linear > .0031308, 1.055 * np.maximum(linear, 0) ** (1 / 2.4) - .055, 12.92 * linear)


def make_targets(anchor, references, mask, weight):
    rgb = np.asarray(anchor, np.float32) / 255
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    ab = (lab[..., 1:] * weight[..., None]).sum((0, 1)) / weight.sum()
    mean_l = float((lab[..., 0] * weight).sum() / weight.sum())
    scan = np.zeros((1001, 3), np.float64)
    scan[:, 0] = np.linspace(0, 100, 1001)
    scan[:, 1:] = ab
    gamut = lab_to_rgb(scan)
    valid = ((gamut >= .001) & (gamut <= .999)).all(1)
    lo, hi = scan[valid, 0].min() + .2, scan[valid, 0].max() - .2
    inside = mask.astype(np.float32)
    base = cv2.GaussianBlur(lab[..., 0] * inside, (0, 0), 24) / cv2.GaussianBlur(inside, (0, 0), 24).clip(.001)
    room = max(.1, min(15., (hi - lo) / 4, .8 * (mean_l - lo), .8 * (hi - mean_l)))
    lower, upper = -100., 100.
    for _ in range(24):
        offset = (lower + upper) / 2
        current = np.clip(base + offset, lo + room, hi - room)
        if (current * weight).sum() / weight.sum() < mean_l:
            lower = offset
        else:
            upper = offset
    base = np.clip(base + (lower + upper) / 2, lo + room, hi - room)
    textures = {}
    amplitude = 1.
    for name, picture in references.items():
        ref = np.asarray(picture, np.float32) @ np.array([.2126, .7152, .0722]) / 255
        texture = ref - cv2.GaussianBlur(ref, (0, 0), 8)
        texture -= (texture * weight).sum() / weight.sum()
        texture *= 10 / max(texture.std(), 1e-6)
        textures[name] = texture
        for sign, room_map in ((1, hi - base), (-1, base - lo)):
            select = (sign * texture > 1e-6) & (weight > 0)
            amplitude = min(amplitude, float((room_map[select] / (sign * texture[select])).min()))
    pictures = {}
    for name, texture in textures.items():
        target_lab = np.empty_like(lab)
        target_lab[..., 0] = base + amplitude * texture
        target_lab[..., 1:] = ab
        target_rgb = lab_to_rgb(target_lab)
        assert target_rgb[weight > 0].min() >= -1e-5 and target_rgb[weight > 0].max() <= 1.00001
        # 色域约束发生在L*对比度上；这里仅量化浮点误差，不进行颜色修复。
        mixed = rgb * (1 - weight[..., None]) + target_rgb * weight[..., None]
        pictures[name] = np.rint(mixed.clip(0, 1) * 255).astype(np.uint8)
    return pictures, dict(ab=ab.tolist(), joint_contrast=amplitude, gamut_l=[float(lo), float(hi)], sigma=24.)


def remove_periodic(patch, condition):
    period = int(condition[1:3])
    phase = np.arange(64) % period
    profile = patch.mean(0 if condition[0] == 'v' else 1)
    repeated = np.array([profile[phase == i].mean() for i in range(period)])[phase]
    return patch - (repeated[None] if condition[0] == 'v' else repeated[:, None])


def template(anchor, positive, coords, condition, device):
    def gray(pic):
        rgb = np.asarray(pic, np.float32) / 255
        light = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0] / 100
        return light - cv2.blur(light, (17, 17), borderType=cv2.BORDER_CONSTANT)
    old = gray(anchor) - gray(positive)
    values = [remove_periodic(old[y:y+64, x:x+64], condition) for x, y in coords]
    return torch.tensor(np.stack(values), device=device).float()


def image_losses(prediction, goal, weight, old_template, coords):
    current, target = lab_tensor(prediction), lab_tensor(goal)
    highpass = lambda x: x - F.avg_pool2d(x, 17, 1, 8)
    error = highpass(current[:, :1]) - highpass(target[:, :1])
    gray = weighted_mean(error.abs(), weight)
    patches = torch.stack([error[0, 0, y:y+64, x:x+64] for x, y in coords])
    energy = old_template.square().mean((1, 2))
    valid = (energy >= (.5 / 100) ** 2).float()
    old = (((patches * old_template).mean((1, 2)) / energy.clamp_min(1e-6)).abs() * valid).sum() / valid.sum().clamp_min(1)
    pooled_weight = F.avg_pool2d(weight, 32)
    residual = pooled_chroma(prediction, weight, pooled_weight) - pooled_chroma(goal, weight, pooled_weight)
    local = weighted_mean(residual.abs(), pooled_weight)
    mean_error = ((current[:, 1:] - target[:, 1:]) * weight).sum((2, 3)) / weight.sum((2, 3)).clamp_min(1)
    color = local + mean_error.abs().mean()
    return dict(gray=gray, old=old, color=color), residual
