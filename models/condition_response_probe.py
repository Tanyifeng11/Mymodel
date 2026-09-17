"""同 latent 的条件强度响应；仅观测，不返回替代采样预测。"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def region_weights(mask, size, kernel):
    """先在输入图分辨率构造互斥区域，再面积下采样，避免细边界消失。"""
    mask = (mask.float() > 0.5).float()
    radius = kernel // 2
    padded = F.pad(mask, (radius,) * 4, value=0)
    outer = F.max_pool2d(padded, kernel, stride=1)
    inner = -F.max_pool2d(-padded, kernel, stride=1)
    weights = {"inner": inner, "boundary": outer - inner, "background": 1 - outer}
    weights = {k: F.interpolate(v, size=size, mode="area") for k, v in weights.items()}
    return {"global": torch.ones_like(weights["inner"]), **weights}


def pair_stats(a, b, weight):
    """空间权重只乘一次；空区域/零方向的夹角记为 null。"""
    a, b, weight = a.double(), b.double(), weight.double()
    count = float(weight.sum()) * a.shape[1]
    aa = float((a.square() * weight).sum())
    bb = float((b.square() * weight).sum())
    dot = float((a * b * weight).sum())
    valid = count > 0 and aa > 0 and bb > 0
    cosine = max(-1.0, min(1.0, dot / (aa * bb) ** 0.5)) if valid else None
    return {
        "pixels": float(weight.sum()), "dot": dot,
        "a_weighted_l2": aa ** 0.5, "b_weighted_l2": bb ** 0.5,
        "a_rms": (aa / count) ** 0.5 if count else None,
        "b_rms": (bb / count) ** 0.5 if count else None,
        "cosine": cosine,
        "opposing": dot < 0 if valid else None,
        "removed_norm_fraction": max(0.0, -cosine) if valid else None,
    }


class ConditionResponseProbe:
    def __init__(self, sketch_processors, texture_processors, mask, output_dir,
                 steps=(0, 5, 15, 25, 49), fractions=(0.1, 0.2), region_kernel=9,
                 metadata=None):
        self.sketch = list(sketch_processors)
        self.texture = list(texture_processors)
        self.steps = set(steps)
        self.fractions = tuple(fractions)
        if not self.sketch or not self.texture:
            raise ValueError("响应探针需要 sketch 和 texture attention 处理器")
        if not self.steps or min(self.steps) < 0:
            raise ValueError("探针步骤必须是非负且非空")
        if len(self.fractions) < 2 or len(set(self.fractions)) != len(self.fractions) or not all(0 < h < 1 for h in self.fractions):
            raise ValueError("至少需要两个不同的扰动比例，且均在 (0, 1) 内")
        if region_kernel < 1 or region_kernel % 2 != 1:
            raise ValueError("区域核必须为正奇数，单位为输入 mask 像素")
        if mask is None or mask.ndim != 4 or mask.shape[:2] != (1, 1):
            raise ValueError("响应探针仅支持单样本的 sketch mask [1,1,H,W]")
        if not all(float(p.scale) > 0 for p in self.sketch + self.texture):
            raise ValueError("响应探针要求两路注入强度均为正")
        self.mask = mask
        self.kernel = region_kernel
        self.folder = Path(output_dir)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.report = {
            "schema_version": 1, "prediction_space": "epsilon_before_cfg",
            "steps": sorted(self.steps), "fractions": list(self.fractions),
            "region_kernel_input_pixels": region_kernel,
            "mask_shape": list(mask.shape), "metadata": metadata or {}, "records": [],
        }

    @torch.no_grad()
    def observe(self, prediction, forward, step, timestep):
        if step not in self.steps:
            return
        if prediction.shape[0] != 1:
            raise ValueError("响应探针仅支持 batch=1")
        processors = self.sketch + self.texture
        scales = [p.scale for p in processors]
        # 额外前向也不能污染原有 gate 日志和 last_* 诊断状态。
        diagnostic = [{k: v for k, v in vars(p).items() if k.startswith('last_')}
                      for p in processors]
        traces = [(p, getattr(p, 'balanced_gate_trace_enabled', False)) for p in processors]
        devices = [prediction.device.index] if prediction.is_cuda else []
        responses = {}
        try:
            for p, _ in traces:
                if hasattr(p, 'balanced_gate_trace_enabled'):
                    p.balanced_gate_trace_enabled = False
            with torch.random.fork_rng(devices=devices):
                repeat = forward()
                floor = repeat.float() - prediction.float()
                for h in self.fractions:
                    for p, scale in zip(processors, scales):
                        p.scale = scale * (1 - h) if p in self.sketch else scale
                    weaker_sketch = forward()
                    for p, scale in zip(processors, scales):
                        p.scale = scale * (1 - h) if p in self.texture else scale
                    weaker_texture = forward()
                    responses[h] = (prediction.float() - weaker_sketch.float(),
                                    prediction.float() - weaker_texture.float())
        finally:
            for p, scale, saved in zip(processors, scales, diagnostic):
                p.scale = scale
                for key in list(vars(p)):
                    if key.startswith('last_') and key not in saved:
                        delattr(p, key)
                for key, value in saved.items():
                    setattr(p, key, value)
            for p, enabled in traces:
                if hasattr(p, 'balanced_gate_trace_enabled'):
                    p.balanced_gate_trace_enabled = enabled

        weights = region_weights(self.mask.to(prediction.device), prediction.shape[-2:], self.kernel)
        row = {"step_index": step, "timestep": int(timestep),
               "sketch_scales": [float(x) for x in scales[:len(self.sketch)]],
               "texture_scales": [float(x) for x in scales[len(self.sketch):]],
               "repeat_max_abs": float(floor.abs().max()), "regions": {}}
        arrays = {"repeat_delta": floor.cpu().numpy()}
        for h, (a, b) in responses.items():
            arrays[f"sketch_delta_h{h:g}"] = a.cpu().numpy()
            arrays[f"texture_delta_h{h:g}"] = b.cpu().numpy()
        for name, weight in weights.items():
            arrays[f"weight_{name}"] = weight.cpu().numpy()
            floor_rms = pair_stats(floor, floor, weight)["a_rms"]
            entry = {"repeat_rms": floor_rms, "responses": {}, "stability": {}}
            for h, (a, b) in responses.items():
                stats = pair_stats(a, b, weight)
                stats["above_repeat_floor"] = (stats["a_rms"] is not None and
                    min(stats["a_rms"], stats["b_rms"]) > 10 * max(floor_rms or 0, 1e-12))
                entry["responses"][f"{h:g}"] = stats
            base_h = self.fractions[0]
            for h in self.fractions[1:]:
                entry["stability"][f"{base_h:g}_vs_{h:g}"] = {
                    label: pair_stats(responses[base_h][idx] / base_h,
                                      responses[h][idx] / h, weight)
                    for idx, label in enumerate(("sketch", "texture"))}
            row["regions"][name] = entry
        np.savez_compressed(self.folder / f"step_{step:02d}.npz", **arrays)
        self.report["records"].append(row)
        (self.folder / "probe.json").write_text(
            json.dumps(self.report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
