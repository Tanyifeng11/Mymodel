#!/usr/bin/env python3
"""从独立验证清单里挑出参考图案明显可见的样本，供 E9 首轮定性对照使用。

筛选只依据参考图自身的纹理统计，与任何一组生成结果无关，因此不会偏向 A 或 B。
判据是"图案可见"的粗代理，不是图案类别标注，也不用于评价生成质量。
"""

import argparse
import json
import os
import random
import sys

import numpy as np
from PIL import Image


CAPTION_KEYWORDS = ("stripe", "striped", "plaid", "check", "checked", "checkered", "gingham",
                    "houndstooth", "polka", "dot", "dotted", "floral", "flower", "print",
                    "printed", "pattern", "patterned", "graphic", "logo", "letter", "text",
                    "geometric", "argyle", "leopard", "animal", "camo", "camouflage", "tartan")
SUMMARY_FIELDS = ("sample_id", "score", "band_energy", "edge_density", "color_spread",
                  "keyword_hit", "category", "texture")


def to_gray(path, size):
    """统一到固定边长的灰度图，避免不同参考图分辨率影响频谱统计。"""
    with Image.open(path) as image:
        resized = image.convert("L").resize((size, size), Image.BICUBIC)
    return np.asarray(resized, dtype=np.float32) / 255.0


def band_energy(gray):
    """中高频能量占比：条纹、格纹和重复花纹在这一段明显强于纯色和渐变。"""
    windowed = gray - float(gray.mean())
    window = np.hanning(gray.shape[0])
    windowed = windowed * window[:, None] * window[None, :]
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(windowed))) ** 2
    size = gray.shape[0]
    center = size / 2.0
    rows = np.arange(size) - center
    radius = np.sqrt(rows[:, None] ** 2 + rows[None, :] ** 2) / center
    total = float(spectrum.sum())
    if total <= 0.0:
        return 0.0
    # 去掉最低频(整体亮度与缓变阴影)与最高频(压缩噪声)后取中频占比。
    band = (radius > 0.06) & (radius < 0.65)
    return float(spectrum[band].sum() / total)


def edge_density(gray):
    """有向梯度密度：重复纹样的边缘比纯色布料密集得多。"""
    dy = np.abs(np.diff(gray, axis=0))[:, :-1]
    dx = np.abs(np.diff(gray, axis=1))[:-1, :]
    magnitude = np.sqrt(dx ** 2 + dy ** 2)
    return float((magnitude > 0.06).mean())


def color_spread(path, size):
    """RGB 通道标准差均值：区分多色纹样与单色布料。"""
    with Image.open(path) as image:
        resized = image.convert("RGB").resize((size, size), Image.BICUBIC)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return float(array.reshape(-1, 3).std(axis=0).mean())


def keyword_hit(prompt):
    lowered = " " + " ".join(str(prompt or "").lower().replace("-", " ").split()) + " "
    return any(" " + word + " " in lowered for word in CAPTION_KEYWORDS)


def score_sample(sample, data_root, size):
    texture_path = os.path.join(data_root, str(sample.get("texture", "")))
    if not os.path.isfile(texture_path):
        raise FileNotFoundError("参考图不存在：%s" % texture_path)
    gray = to_gray(texture_path, size)
    metrics = {
        "band_energy": band_energy(gray),
        "edge_density": edge_density(gray),
        "color_spread": color_spread(texture_path, size),
        "keyword_hit": keyword_hit(sample.get("prompt")),
    }
    # 频谱与边缘是主判据；描述词只做小幅加权，避免完全依赖标注质量。
    metrics["score"] = (2.0 * metrics["band_energy"] + 1.5 * metrics["edge_density"]
                        + 0.5 * metrics["color_spread"] + (0.15 if metrics["keyword_hit"] else 0.0))
    return metrics


def select(samples, count, min_band_energy, min_edge_density):
    eligible = [sample for sample in samples
                if sample["band_energy"] >= min_band_energy
                and sample["edge_density"] >= min_edge_density]
    ranked = sorted(eligible, key=lambda item: (-item["score"], item["sample_id"]))
    return ranked[:count], len(eligible)


def build_color_swap(selected, seed):
    """同色异纹替换：颜色分布接近但纹样不同，用于检查模型是否真的用了图案信息。

    仅记录配对关系，不改写清单；生成时由评测脚本按 texture_override 使用。
    """
    if len(selected) < 2:
        return []
    order = sorted(selected, key=lambda item: item["color_spread"])
    pairs = []
    for first, second in zip(order[::2], order[1::2]):
        pairs.append({"sample_id": first["sample_id"], "swap_sample_id": second["sample_id"],
                      "texture": first["texture"], "swap_texture": second["texture"],
                      "color_spread": first["color_spread"], "swap_color_spread": second["color_spread"]})
    random.Random(seed).shuffle(pairs)
    return pairs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split_path", required=True, help="build_bf_test_manifest.py 生成的完整验证划分")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_split", required=True, help="图案子集划分，供 run_fixed_benchmark 使用")
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--swap_json", default=None, help="可选：同色异纹配对，用于图案敏感性抽查")
    parser.add_argument("--count", type=int, default=48, help="子集样本数，建议 32–64")
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--min_band_energy", type=float, default=0.10)
    parser.add_argument("--min_edge_density", type=float, default=0.05)
    parser.add_argument("--swap_count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if args.count <= 0:
        raise ValueError("count 必须大于 0")
    with open(args.split_path, encoding="utf-8") as handle:
        samples = json.load(handle)
    if not samples:
        raise ValueError("输入划分为空：%s" % args.split_path)

    scored = []
    for sample in samples:
        metrics = score_sample(sample, args.data_root, args.resize)
        scored.append({**sample, **metrics})

    selected, eligible_count = select(scored, args.count, args.min_band_energy, args.min_edge_density)
    if len(selected) < args.count:
        print("[e9-pattern] 满足阈值的样本只有 %s 个，少于请求的 %s 个；"
              "请放宽阈值或扩大输入划分。" % (eligible_count, args.count), file=sys.stderr, flush=True)
        if not selected:
            return 1

    # 输出仍是 run_fixed_benchmark 认识的划分格式，sample_id 重新连续编号。
    split = []
    for position, sample in enumerate(selected):
        item = {key: value for key, value in sample.items() if key not in SUMMARY_FIELDS[1:5]}
        item["sample_id"] = "%06d" % position
        item["source_sample_id"] = sample["sample_id"]
        split.append(item)

    os.makedirs(os.path.dirname(args.output_split) or ".", exist_ok=True)
    with open(args.output_split, "w", encoding="utf-8") as handle:
        json.dump(split, handle, ensure_ascii=False, indent=2)

    if args.summary_json:
        os.makedirs(os.path.dirname(args.summary_json) or ".", exist_ok=True)
        payload = {
            "scope": "频谱中频能量与边缘密度的图案可见性代理，仅用于挑选定性对照样本，"
                     "不是图案类别标注，也不是质量指标。",
            "split_path": os.path.abspath(args.split_path),
            "total_candidates": len(scored), "eligible_count": eligible_count,
            "selected_count": len(split), "requested_count": args.count,
            "thresholds": {"min_band_energy": args.min_band_energy,
                           "min_edge_density": args.min_edge_density},
            "selected": [{field: sample[field] for field in SUMMARY_FIELDS} for sample in selected],
        }
        with open(args.summary_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    if args.swap_json:
        pairs = build_color_swap(selected, args.seed)[: max(args.swap_count, 0)]
        os.makedirs(os.path.dirname(args.swap_json) or ".", exist_ok=True)
        with open(args.swap_json, "w", encoding="utf-8") as handle:
            json.dump({"scope": "同色异纹参考替换配对，用于检查模型是否使用了图案信息。",
                       "pairs": pairs}, handle, ensure_ascii=False, indent=2)

    print(json.dumps({"total_candidates": len(scored), "eligible_count": eligible_count,
                      "selected_count": len(split), "output_split": args.output_split},
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
