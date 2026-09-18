"""SPTG 第一步：完整条件增量观测。rho=0 直接返回原预测，不重组 FP16 张量。"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.condition_intervention import isolated_forward
from models.condition_response_probe import pair_stats, region_weights


def local_geometry(a, b, floor):
    # 每个窗口沿通道和 3x3 空间累计；边缘只计入实际存在的像素。
    pool = lambda x: F.avg_pool2d(x.sum(1, keepdim=True), 3, 1, 1, count_include_pad=False)
    aa, bb, dot = pool(a.square()), pool(b.square()), pool(a*b)
    noise = pool(floor.square()).clamp_min(1e-24)
    valid = (aa > 100*noise) & (bb > 100*noise)
    cosine = dot / (aa*bb).sqrt().clamp_min(1e-24)
    return cosine.clamp(-1, 1), valid


class FullConditionProbe:
    def __init__(self, sketch, texture, mask, folder, metadata):
        self.sketch, self.texture = list(sketch), list(texture)
        if not self.sketch or not self.texture or mask is None or mask.shape[:2] != (1, 1):
            raise ValueError('完整分解需要单样本 mask 及两路 attention 处理器')
        self.mask = mask
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.report = dict(metadata=metadata, prediction_space='epsilon_before_cfg',
            definition='gs=epsilon_s-epsilon_0; gt=epsilon_st-epsilon_s; fixed E5 text routing',
            rho=0, sketch_weight=1, texture_weight=1, local_window=3, region_kernel=9,
            records=[])

    @torch.no_grad()
    def observe(self, prediction, forward, step, timestep):
        """forward(sketch_on, scrub_texture)；texture scale 在这里临时归零。"""
        processors = self.sketch + self.texture
        maps = [(p, hasattr(p, 'attn_map'), getattr(p, 'attn_map', None)) for p in processors]
        audits = {}
        try:
            with isolated_forward(processors, prediction):
                repeat = forward(True, False)
                for p in self.texture:
                    p.scale = 0.
                eps_s = forward(True, False)
                eps_0 = forward(False, False)
                if step in (0, 25, 49):
                    # 主分支保留 token 切分和文本路由；清空 token 检查隐藏的纹理依赖。
                    audits['texture_off_sketch_on_max_abs'] = float((eps_s.float()-forward(True, True).float()).abs().max())
                    audits['texture_off_sketch_off_max_abs'] = float((eps_0.float()-forward(False, True).float()).abs().max())
        finally:
            for p, existed, value in maps:
                if existed:
                    p.attn_map = value
                elif hasattr(p, 'attn_map'):
                    delattr(p, 'attn_map')
        floor = repeat.float()-prediction.float()
        a, b = eps_s.float()-eps_0.float(), prediction.float()-eps_s.float()
        cosine, valid = local_geometry(a, b, floor)
        weights = region_weights(self.mask.to(prediction.device), prediction.shape[-2:], 9)
        row = dict(step_index=step, timestep=int(timestep), repeat_max_abs=float(floor.abs().max()),
                   shutdown_audits=audits, regions={},
                   reconstruction_fp32_max_abs=float((eps_0.float()+a+b-prediction.float()).abs().max()),
                   epsilon_rms={k: float(x.float().square().mean().sqrt()) for k, x in
                                [('text_only', eps_0), ('sketch_only', eps_s), ('full', prediction)]})
        for name, weight in weights.items():
            stats = pair_stats(a, b, weight)
            floor_rms = pair_stats(floor, floor, weight)['a_rms']
            stats['above_repeat_floor'] = stats['a_rms'] is not None and min(stats['a_rms'], stats['b_rms']) > 10*max(floor_rms or 0, 1e-12)
            eligible = weight*valid
            stats['local_valid_weight'] = float(eligible.sum())
            stats['local_negative_fraction'] = float((eligible*(cosine < 0)).sum()/eligible.sum()) if float(eligible.sum()) else None
            stats['local_strong_negative_fraction'] = float((eligible*(cosine < -.1)).sum()/eligible.sum()) if float(eligible.sum()) else None
            row['regions'][name] = stats
        arrays = dict(epsilon_0=eps_0, epsilon_s=eps_s, epsilon_st=prediction,
                      sketch_guidance=a, texture_guidance=b, repeat_delta=floor,
                      local_cosine=cosine, local_valid=valid, **{f'weight_{k}':v for k,v in weights.items()})
        if any(not torch.isfinite(v).all() for v in arrays.values()):
            raise ValueError('完整条件分解出现非有限值')
        if any(v > max(row['repeat_max_abs']*10, 1e-6) for v in audits.values()):
            raise ValueError('纹理关闭审计失败：清空纹理 token 后预测改变')
        np.savez_compressed(self.folder/f'step_{step:02d}.npz', **{k:v.cpu().numpy() for k,v in arrays.items()})
        self.report['records'].append(row)
        (self.folder/'probe.json').write_text(json.dumps(self.report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        self.last_response = (a, b, floor)
        return prediction
