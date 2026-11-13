"""统一VAE可达目标，并分别记录颜色、纹样和残留的梯度。"""
import numpy as np
import torch
from torch.nn import functional as F

from models.e2_lab_anchor import rgb_to_lab
from tools.e2_b_material_losses import highpass, weighted
from tools.e2_b_joint_losses import gradients, spectrum

TASKS = ('pattern', 'residue', 'color')


def templates(anchor, positive, coords):
    a, p = (highpass(rgb_to_lab(x) / 100) for x in (anchor, positive))
    patches = lambda x: torch.stack([x[0, :, y:y+64, x0:x0+64] for x0, y in coords])
    old, new = patches(a - p), patches(p)
    old = old - new * ((old * new).sum((1, 2, 3), keepdim=True)
                      / new.square().sum((1, 2, 3), keepdim=True).clamp_min(1e-8))
    return old.detach()


def palette(field, positions):
    angle = torch.arange(16, device=field.device).float() * torch.pi / 16
    directions = torch.stack((angle.cos(), angle.sin()))
    return (field[0, 1:].flatten(1)[:, positions].T @ directions).sort(0).values


def terms(rgb, positive, original, base, weight, template, coords):
    current, target, source = (rgb_to_lab(x) / 100 for x in (rgb, positive, original))
    interior = (weight >= .999).to(weight.dtype)
    mean = lambda x: (x * interior).sum((-2, -1), keepdim=True) / interior.sum().clamp_min(1)
    positions = interior[0, 0].flatten().nonzero().flatten()
    if len(positions) > 2048:
        # 避免规则等距取样恰好锁定条纹相位，输入/目标使用同一固定随机位置。
        indices = np.sort(np.random.default_rng(20261114).choice(len(positions), 2048, replace=False))
        positions = positions[torch.as_tensor(indices, device=rgb.device)]
    error = current - target
    hp = highpass(error)
    projection = []
    patches = torch.stack([hp[0, :, y:y+64, x:x+64] for x, y in coords])
    for size in (64, 32):
        split = lambda x: x if size == 64 else x.reshape(-1, 3, 2, 32, 2, 32).permute(0, 2, 4, 1, 3, 5).reshape(-1, 3, 32, 32)
        a, b = split(patches), split(template)
        energy = b.square().mean((1, 2, 3))
        valid = energy > 2.5e-5
        values = (a * b).mean((1, 2, 3)).abs() / energy.clamp_min(1e-8)
        projection.append((values * valid).sum() / valid.sum().clamp_min(1))
    # 残留同时约束局部旧模板投影和整个衣身内部的高频/边缘误差。
    old = sum(projection) / len(projection)
    edge = weighted(gradients(error).abs(), interior)
    high = weighted(hp.abs(), interior)
    scale = mean((target[:, 1:] - mean(target[:, 1:])).square()).mean(1, keepdim=True).sqrt().clamp_min(.02).detach()
    spatial_ab = weighted((error[:, 1:] / scale).abs(), interior)
    mean_ab = (mean(error[:, 1:]).abs() / .02).mean()
    pc, pt, ps = (palette(x, positions) for x in (current, target, source))
    # 原参考的颜色目标只惩罚超过同一VAE重建所造成的误差，不与潜变量目标争夺不可达像素。
    palette_excess = F.relu((pc - ps).abs().mean() - (pt - ps).abs().mean().detach()) / .02
    sw = F.avg_pool2d(weight, 64, 16)
    low = F.avg_pool2d(error[:, :1] * weight, 64, 16) / sw.clamp_min(1e-6)
    return dict(rgb=weighted((rgb-positive).abs(), weight),
                frequency=spectrum(current, target, coords), high=high,
                projection=old, edge=edge, chroma=spatial_ab, mean_ab=mean_ab,
                palette=palette_excess, light=weighted(low.abs(), sw),
                protection=weighted((rgb-base).abs(), 1-weight))


def groups(values):
    return dict(pattern=values['latent'] + 2*values['rgb'] + values['frequency'] + values['high'] + 2*values['protection'],
                residue=values['projection'] + 10*values['edge'],
                color=values['chroma'] + values['mean_ab'] + values['palette'] + 5*values['light'])


def task_gradients(losses, parameters):
    result = []
    for index, task in enumerate(TASKS):
        parts = torch.autograd.grad(losses[task], parameters, retain_graph=index < len(TASKS)-1, allow_unused=True)
        result.append(torch.cat([torch.zeros_like(p).flatten() if g is None else g.detach().flatten()
                                 for p, g in zip(parameters, parts)]))
    return torch.stack(result)


def merge(gradients_, weights, pcgrad, seed):
    original = gradients_ * weights[:, None]
    projected = original.clone()
    if pcgrad:
        rng = np.random.default_rng(seed)
        for i in range(len(projected)):
            for j in rng.permutation(len(projected)):
                dot = torch.dot(projected[i], original[j])
                if dot < 0:
                    projected[i] -= dot / original[j].square().sum().clamp_min(1e-18) * original[j]
    return projected.sum(0)


def alignment(gradients_, direction):
    denominator = gradients_.norm(dim=1) * direction.norm()
    return ((gradients_ @ direction) / denominator.clamp_min(1e-18)).cpu().tolist()


def assign(parameters, gradient):
    offset = 0
    for p in parameters:
        p.grad = gradient[offset:offset+p.numel()].view_as(p).clone()
        offset += p.numel()
    assert offset == gradient.numel()
