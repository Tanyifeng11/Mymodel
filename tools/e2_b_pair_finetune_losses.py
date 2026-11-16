"""B 小试的幅度配对、绝对对齐及逐约束参数位移保护。"""
import numpy as np
import torch

from models.e2_lab_anchor import rgb_to_lab
from tools.e2_b_material_losses import highpass


ARMS = ('a075', 'a100', 'a125')
SLOPE_EPSILON = 1e-6


def projection(rgb, positive, base_positive, weight):
    """以基准正样本为固定基底；orth 是相对本档目标的正交能量比。"""
    h, p, base = [highpass(rgb_to_lab(value.float())[:, :1] / 100)
                  for value in (rgb, positive, base_positive)]
    inside = (weight >= .999).to(h.dtype)
    count = inside.sum()
    assert bool(count > 0), '衣内评价区域为空'
    mean = lambda value: (value * inside).sum() / count
    base_energy, target_energy = mean(base.square()), mean(p.square())
    assert bool(base_energy.detach() > 1e-8), '基准目标高频能量不足'
    assert bool(target_energy.detach() > 1e-8), '本档目标高频能量不足'
    x = mean(p * base) / base_energy
    y = mean(h * base) / base_energy
    g = mean(h * p) / target_energy
    orth = mean((h - g * p).square()) / target_energy
    return dict(x=x, y=y, g=g, orth=orth)


def pair_coefficients(y_low, y_high, x_low, x_high):
    """分支顺序反传的系数：只保存标量，避免同时保留两条采样计算图。"""
    error = ((y_high - y_low) - (x_high - x_low)).detach().float()
    return dict(low=-2 * error, high=2 * error, error=error, pair=error.square())


def pair_terms(outputs, positives, weight):
    """原始增量差监督；日志斜率才除以实际可达目标跨度。"""
    values = {arm: projection(outputs[arm], positives[arm], positives['a100'], weight)
              for arm in ARMS}
    low, base, high = [values[arm] for arm in ARMS]
    target_span = high['x'] - low['x']
    assert bool(target_span.detach() > SLOPE_EPSILON), '实际目标跨度过小或方向相反'
    output_span = high['y'] - low['y']
    return dict(pair=(output_span - target_span).square(),
                absolute=(base['g'] - 1).square(),
                details=dict(arms=values, target_span=target_span,
                             output_span=output_span, slope=output_span / target_span))


def project_displacement(displacement, gradients, margins, max_sweeps=4096, tolerance=1e-6):
    """投影实际 AdamW 位移，逐约束margin以原始位移范数归一化。"""
    norms = gradients.norm(dim=1)
    margins = torch.as_tensor(margins, dtype=gradients.dtype, device=gradients.device)
    assert margins.shape == norms.shape
    active = norms > 1e-12
    rows = gradients[active] / norms[active, None]
    length = displacement.norm()
    if not len(rows) or float(length) == 0:
        return displacement.clone(), dict(feasible=True, minimum_dot=0., relative_correction=0.)
    required = margins[active]
    gram = (rows @ rows.T).double().cpu().numpy()
    offset = (rows @ (displacement / length)).double().cpu().numpy() - required.double().cpu().numpy()
    dual = np.zeros(len(rows))
    for _ in range(max_sweeps):
        for i in range(len(rows)):
            dual[i] = max(0., dual[i] - (offset[i] + gram[i] @ dual) / max(gram[i, i], 1e-12))
        if (offset + gram @ dual).min() >= -tolerance / 4:
            break
    delta = displacement + length * (rows.T @ torch.as_tensor(dual, dtype=rows.dtype, device=rows.device))
    dots = (rows @ delta) / length
    feasible = bool(((dots - required) >= -tolerance).all())
    report = dict(feasible=feasible, minimum_dot=float(dots.min()),
                  descent_margin=float(required.min()), maximum_margin=float(required.max()),
                  relative_correction=float((delta - displacement).norm() / length))
    return delta if feasible else torch.zeros_like(delta), report
