"""E15 诊断共用：固定协议、参考条件、BF 分层捕获与谱指标。

协议与 113719／113817 保持一致：相同 32 样本、相同 VAE 后验噪声、
相同 9 个去噪步、相同参考条件集合。E15 只做前向诊断，不训练。
"""

import json
import random
from pathlib import Path

import numpy as np

# 与 113719／113817 完全一致的固定去噪步。
FULL_STEPS = [1, 101, 141, 181, 221, 261, 481, 781, 981]
# random 对应旧报告的 wrong_1；zero_image 是全零参考图，不同于 zero_tokens。
CONDITIONS = ["matched", "rot90", "color_nearest", "random", "zero_image"]
CONTRASTS = ["rot90", "color_nearest", "random", "zero_image"]
# D1 分层顺序即出图顺序；TCPM 不在 BF 预训练权重内，另行说明。
REFERENCE_LAYERS = ["clip_patch", "cnn1", "cnn2", "cnn3", "cnn4",
                    "fused", "resampler_out", "mlp_out", "pre_ln", "final"]
TEXTURE_LAYERS = 16
GROUP_RANGES = {"G1": tuple(range(0, 4)), "G2": tuple(range(4, 8)),
                "G3": tuple(range(8, 12)), "G4": tuple(range(12, 16))}
REGIONS = ["interior", "boundary", "background"]
# 相邻层响应低于该倍数且多类对照同向时记为候选瓶颈。
BOTTLENECK_RATIO = 0.3


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def sample_indices(dataset, count=32):
    """与 e14_matched_denoising 相同的固定抽样，保证跨轮样本一致。"""
    if not 3 <= count <= len(dataset):
        raise ValueError("样本数不合法")
    return sorted(random.Random(42).sample(range(len(dataset)), count))


class LayerCapture:
    """零侵入捕获 BF 各层表示；resampler 的 K/V 输入即 fused tokens。"""

    def __init__(self, conditioner):
        self.conditioner = conditioner
        self.current = {}
        self.query = None
        self.attention = None
        modules = [
            (conditioner.stage1, "cnn1", "output"),
            (conditioner.stage2, "cnn2", "output"),
            (conditioner.stage3, "cnn3", "output"),
            (conditioner.stage4, "cnn4", "output"),
            (conditioner.resampler, "resampler_out", "output"),
            (conditioner.token_mlp, "mlp_out", "output"),
            (conditioner.token_norm, "pre_ln", "input"),
            (conditioner.token_norm, "final", "output"),
        ]
        self.handles = [self._register(module, name, kind) for module, name, kind in modules]
        self.handles.append(conditioner.resampler.register_forward_pre_hook(self._resampler_inputs))

    def _register(self, module, name, kind):
        if kind == "output":
            def hook(_module, _inputs, output, _name=name):
                # MultiheadAttention 等模块返回 tuple，取第一个张量。
                value = output[0] if isinstance(output, (tuple, list)) else output
                self.current[_name] = value.detach()
            return module.register_forward_hook(hook)

        def pre_hook(_module, inputs, _name=name):
            self.current[_name] = inputs[0].detach()
        return module.register_forward_pre_hook(pre_hook)

    def _resampler_inputs(self, _module, args):
        self.query = args[0].detach()
        self.current["fused"] = args[1].detach()

    def reset(self):
        self.current = {}
        self.query = None
        self.attention = None

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def flatten_representation(tensor):
    """[1, ...] -> 一维 float32 numpy，保留全部空间结构。"""
    if tensor.shape[0] != 1:
        raise ValueError("逐样本捕获只接受 batch=1")
    return tensor.detach().float().reshape(-1).cpu().numpy()


def gram_statistics(matrix):
    """[N, D] 的中心化 effective rank、前四主成分占比与成对距离均值。"""
    values = matrix.astype(np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    weights = np.linalg.eigvalsh(centered @ centered.T)[::-1]
    weights = np.clip(weights, 0.0, None)
    total = float(weights.sum())
    if total <= 0.0:
        return {"effective_rank": 0.0, "pc_variance": [],
                "pairwise_cosine_mean": None, "pairwise_l2_mean": None}
    share = weights / total
    positive = share[share > 0]
    norms = np.linalg.norm(values, axis=1)
    dots = values @ values.T
    denom = np.maximum(np.outer(norms, norms), 1e-12)
    good = norms > 0
    n = len(matrix)
    rows, cols = np.triu_indices(n, k=1)
    if good.sum() > 1:
        keep = np.ix_(np.flatnonzero(good), np.flatnonzero(good))
        cosine_mean = float(((dots[keep] / denom[keep])[np.triu_indices(int(good.sum()), k=1)]).mean())
    else:
        cosine_mean = None
    squared = np.maximum(norms[:, None] ** 2 + norms[None, :] ** 2 - 2.0 * dots, 0.0)
    return {
        "effective_rank": float(np.exp(-(positive * np.log(positive)).sum())),
        "pc_variance": [float(v) for v in share[:4]],
        "pairwise_cosine_mean": cosine_mean,
        "pairwise_l2_mean": float(np.sqrt(squared)[rows, cols].mean()),
    }


def summary_stats(values):
    array = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if array.size == 0:
        return {"mean": None, "median": None, "min": None, "max": None, "count": 0}
    return {"mean": float(array.mean()), "median": float(np.median(array)),
            "min": float(array.min()), "max": float(array.max()), "count": int(array.size)}


def normalized_response(reference, variant, zero):
    """||ref-variant|| / ||ref-zero||；分母为零时返回 None。"""
    denominator = float(np.linalg.norm(reference - zero))
    if denominator <= 1e-12:
        return None
    return float(np.linalg.norm(reference - variant) / denominator)


def relative_change(reference, variant):
    base = float(np.linalg.norm(reference))
    if base <= 1e-12:
        return None
    return float(np.linalg.norm(reference - variant) / base)


def compact_table(header, rows, width=88):
    """终端可读的定宽表；服务器网页终端只有 91 列。"""
    lines = ["  ".join(str(h).rjust(9) for h in header)]
    for row in rows:
        lines.append("  ".join(("%.4f" % v).rjust(9) if isinstance(v, float) else str(v).rjust(9)
                                for v in row))
    return "\n".join(line[:width] for line in lines)


def format_heatmap(labels, columns, matrix, fmt="%+.4f"):
    """热图文本：行=timestep/区域，列=层/条件。"""
    header = "".ljust(14) + " ".join(str(c).rjust(8) for c in columns)
    lines = [header]
    for label, values in zip(labels, matrix):
        cells = " ".join(("      -" if v is None else (fmt % v).rjust(8)) for v in values)
        lines.append(str(label).ljust(14) + " " + cells)
    return "\n".join(lines)