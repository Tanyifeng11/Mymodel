"""完整条件 guidance 的局部负投影；保持 E5 文本路由及外层 CFG。"""
import json

import torch
import torch.nn.functional as F

from models.full_condition_probe import FullConditionProbe, local_geometry
from models.condition_response_probe import region_weights


SETTINGS = {'baseline': (1., 0.), 'strong_texture': (1.2, 0.),
            'sptg_rho05': (1.2, .5), 'sptg_rho10': (1.2, 1.)}


def combine_guidance(prediction, gs, gt, floor, texture_weight, rho):
    # 重叠 3x3 窗口统计系数，再作用于中心像素的通道向量。
    # 这不是独立、不重叠向量空间的正交投影；rho=1 不保证每个窗口都变为正交。
    pool = lambda x: F.avg_pool2d(x.sum(1, keepdim=True), 3, 1, 1, count_include_pad=False)
    aa, dot = pool(gs.square()), pool(gs*gt)
    cosine, valid = local_geometry(gs, gt, floor)
    coefficient = torch.where(valid, (dot/aa.clamp_min(1e-24)).clamp_max(0), 0.)
    correction = -rho*coefficient*gs
    strong = prediction if texture_weight == 1 else (prediction.float()+(texture_weight-1)*gt).to(prediction.dtype)
    # baseline 精确直通；避免 FP16 的减法重组破坏 rho=0 对照。
    result = strong if rho == 0 else (prediction.float()+(texture_weight-1)*gt+texture_weight*correction).to(prediction.dtype)
    if not torch.isfinite(result).all():
        raise ValueError('SPTG 产生非有限预测')
    after, after_valid = local_geometry(gs, gt+correction, floor)
    return result, dict(coefficient=coefficient, valid=valid, cosine=cosine,
                        correction=correction, strong=strong, after_cosine=after, after_valid=after_valid)


class SPTG(FullConditionProbe):
    def __init__(self, sketch, texture, mask, folder, metadata, mode):
        super().__init__(sketch, texture, mask, folder, metadata)
        self.texture_weight, self.rho = SETTINGS[mode]
        self.report.update(rho=self.rho, texture_weight=self.texture_weight, sampling_mode=mode)
        self.trace = dict(mode=mode, metadata=metadata, sketch_weight=1.,
                          texture_weight=self.texture_weight, rho=self.rho,
                          prediction_space='epsilon_before_cfg', local_window=3,
                          projection='overlapping_window_coefficient_at_center', records=[])

    @torch.no_grad()
    def observe(self, prediction, forward, step, timestep):
        super().observe(prediction, forward, step, timestep)
        gs, gt, floor = self.last_response
        result, diagnostics = combine_guidance(prediction, gs, gt, floor, self.texture_weight, self.rho)
        weights = region_weights(self.mask.to(prediction.device), prediction.shape[-2:], 9)
        delta = result.float()-prediction.float()
        projection_delta = result.float()-diagnostics['strong'].float()
        row = dict(step_index=step, timestep=int(timestep), applied=bool(torch.any(delta != 0)),
                   actual_delta_rms=float(delta.square().mean().sqrt()),
                   actual_projection_delta_rms=float(projection_delta.square().mean().sqrt()), regions={})
        for name,w in weights.items():
            active = w*diagnostics['valid']
            count = float(active.sum())
            row['regions'][name] = dict(
                negative_fraction=float((active*(diagnostics['cosine']<0)).sum()/count) if count else None,
                after_negative_fraction=float((active*(diagnostics['after_cosine']<0)).sum()/count) if count else None,
                actual_delta_rms=float(((delta.square()*w).sum()/(w.sum()*delta.shape[1])).sqrt()) if float(w.sum()) else None)
        self.trace['records'].append(row)
        (self.folder/'sptg.json').write_text(json.dumps(self.trace,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        return result
