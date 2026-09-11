"""同 latent 条件预测的输出端阻断；不混合两个独立采样轨迹。"""
import torch
import torch.nn.functional as F


def block_local_detail_output(on, off, garment_mask):
    mask = F.interpolate(garment_mask.to(device=on.device, dtype=torch.float32),
                         size=on.shape[-2:], mode='nearest') > 0.5
    # 二值 mask 用 where 保证区域外与 off 逐值相等，区域内与 on 逐值相等。
    used = torch.where(mask, on, off)
    outside = (~mask).expand_as(on)
    delta = (on.float()-off.float())
    stats = {'mask_coverage': float(mask.float().mean().item()),
             'outside_elements': int(outside.sum().item()),
             'outside_delta_before_rms': float(delta[outside].square().mean().sqrt().item()) if outside.any() else None,
             'outside_delta_after_max': float((used.float()-off.float())[outside].abs().max().item()) if outside.any() else None}
    return used, stats
