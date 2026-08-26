#!/usr/bin/env python
"""标定 TxtCF 与 TCF-LAB 的 GT floor(地板值)。

两个指标都是"生成图的颜色离某个参照物有多远", 但参照物和估计子都不同:

  TxtCF  (prompt_color_delta_e) = deltaE( COLOR_TABLE[颜色词],  median_RGB(gen[mask]) )
  TCF-LAB(tcf_lab_delta)        = || mean_LAB(gen[mask]) - mean_LAB(texture) ||

把 gen 换成**真实目标图**, 就得到各自的地板: 一个完美模型在这两个指标上能达到
的最好值。没有地板, 48.09 这种绝对值读不出含义; 两个指标之间的比值更是无法解释
(参照物不同 + 估计子不同, 双重混淆)。

本脚本不需要 GPU、不需要模型权重、不需要生成图 —— 地板只是数据集本身的性质。

用法:
    python tools/compute_color_floors.py \
        --per_sample_csv eval_outputs/report_e7a/metrics_per_sample.csv \
        --data_root /mnt/f/fuxian/dataset/datasets/BF/training
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --no_cv2 必须在导入 garment_mask_utils 之前生效: 该模块用 try/except ImportError
# 探测 cv2, 缺失时退回 Pillow 的 MaxFilter/MinFilter。两条路径的形态学运算并不等价,
# 所以 cv2 的有无会改变草图 mask 的置信度判定, 进而改变 auto 策略的 fallback 比例。
# 这是一个静默的环境依赖 —— 集群与本地若不一致, 全部 mask 派生指标都不可比。
if "--no_cv2" in sys.argv:
    sys.modules["cv2"] = None

from color_conflict_utils import delta_e_rgb, dominant_rgb_from_pil
from eval.eval_utils import prepare_evaluation_masks, safe_open_rgb
from eval.metrics import _pil_to_np, _rgb_to_lab

import garment_mask_utils as _gmu

BUCKETS = ["all", "has_text_color", "no_text_color", "color_conflict", "high_conflict"]


def _finite(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _stem(path):
    return os.path.splitext(os.path.basename(path or ""))[0]


def _buckets_of(rec):
    names = ["all"]
    if not rec["has_text_color"]:
        names.append("no_text_color")
        return names
    names.append("has_text_color")
    score = rec["color_conflict_score"]
    if score is not None:
        if score >= 0.25:
            names.append("color_conflict")
        if score >= 0.55:
            names.append("high_conflict")
    return names


def _resolve(data_root, sample_id):
    return {
        "target": os.path.join(data_root, "cloth", sample_id + ".jpg"),
        "texture": os.path.join(data_root, "texture", sample_id + ".jpg"),
        "sketch": os.path.join(data_root, "sketch", sample_id + ".jpg"),
    }


def compute_row(row, data_root, mask_policy, size=None):
    """返回单样本的两个地板值, 无法计算时返回 None。

    size: (width, height)。报告里 mask 是在**生成图尺寸**上构建的(评测脚本用
    384x512), 而 BF 原图是 256x256。要让地板值与报告数字可比, 必须在同一
    几何下重建 mask —— 否则 garment_mask_pixels 会差一个面积比(0.333)。
    """
    sample_id = _stem(row.get("texture_path")) or _stem(row.get("sketch_path"))
    paths = _resolve(data_root, sample_id)
    if not all(os.path.isfile(p) for p in paths.values()):
        return None

    target = safe_open_rgb(paths["target"])
    texture = safe_open_rgb(paths["texture"])
    if target is None or texture is None:
        return None

    if size is not None and tuple(target.size) != tuple(size):
        target = target.resize(tuple(size), Image.BICUBIC)

    bundle = prepare_evaluation_masks(
        target.size,
        sketch_path=paths["sketch"],
        target_path=paths["target"],
        gen_path=paths["target"],
        mask_policy=mask_policy,
    )
    mask = bundle.get("garment")
    if mask is None or int(mask.sum()) < 50:
        return None

    rec = {
        "sample_id": row.get("sample_id"),
        "dataset_id": sample_id,
        "has_text_color": str(row.get("has_text_color")).lower() in ("true", "1"),
        "color_conflict_score": _finite(row.get("color_conflict_score")),
        "garment_mask_pixels": int(mask.sum()),
        "mask_source": bundle.get("stats", {}).get("mask_source"),
    }

    # TxtCF 地板: 与 prompt_color_delta_e 同一个估计子(mask 内 median RGB -> LAB)
    if rec["has_text_color"] and row.get("text_color_rgb"):
        text_rgb = json.loads(row["text_color_rgb"])
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
        target_dominant = dominant_rgb_from_pil(target, mask=mask_image)
        rec["text_color_rgb"] = list(text_rgb)
        rec["target_dominant_rgb"] = list(target_dominant)
        rec["txtcf_floor"] = delta_e_rgb(text_rgb, target_dominant)
    else:
        rec["txtcf_floor"] = float("nan")

    # TCF-LAB 地板: 与 tcf_lab_delta 同一个估计子(mask 内 mean LAB vs 纹理图整图 mean LAB)
    texture_resized = texture.resize(target.size, Image.BICUBIC)
    target_lab = _rgb_to_lab(_pil_to_np(target))
    texture_lab = _rgb_to_lab(_pil_to_np(texture_resized))
    rec["tcf_lab_floor"] = float(
        np.linalg.norm(
            target_lab[mask].mean(axis=0) - texture_lab.reshape(-1, 3).mean(axis=0)
        )
    )
    return rec


def summarize(records):
    summary = {}
    for name in BUCKETS:
        selected = [r for r in records if name in _buckets_of(r)]
        if not selected:
            continue
        txtcf = [r["txtcf_floor"] for r in selected if _finite(r["txtcf_floor"]) is not None]
        tcf = [r["tcf_lab_floor"] for r in selected if _finite(r["tcf_lab_floor"]) is not None]
        summary[name] = {
            "n": len(selected),
            "txtcf_floor_n": len(txtcf),
            "txtcf_floor_mean": float(np.mean(txtcf)) if txtcf else None,
            "txtcf_floor_median": float(np.median(txtcf)) if txtcf else None,
            "tcf_lab_floor_n": len(tcf),
            "tcf_lab_floor_mean": float(np.mean(tcf)) if tcf else None,
            "tcf_lab_floor_median": float(np.median(tcf)) if tcf else None,
        }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--per_sample_csv",
        default="eval_outputs/report_e7a/metrics_per_sample.csv",
        help="定义样本集合与 text_color_rgb 的现有 per-sample CSV",
    )
    ap.add_argument("--data_root", default="/mnt/f/fuxian/dataset/datasets/BF/training")
    ap.add_argument("--mask_policy", default="sketch_only", choices=["auto", "sketch_only"])
    ap.add_argument("--width", type=int, default=384,
                    help="重建 mask 的宽度, 需与评测生成图一致(评测脚本默认 384)")
    ap.add_argument("--height", type=int, default=512,
                    help="重建 mask 的高度, 需与评测生成图一致(评测脚本默认 512)")
    ap.add_argument("--native_size", action="store_true",
                    help="改用图像原生尺寸(BF 为 256x256), 仅用于观察分辨率影响")
    ap.add_argument("--no_cv2", action="store_true",
                    help="屏蔽 cv2, 强制 garment_mask_utils 走 Pillow fallback(诊断环境依赖用)")
    ap.add_argument("--out_json", default="eval_outputs/color_floors.json")
    ap.add_argument("--out_csv", default="eval_outputs/color_floors_per_sample.csv")
    args = ap.parse_args()

    size = None if args.native_size else (args.width, args.height)

    with open(args.per_sample_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print("[info] samples in csv = %d" % len(rows))
    print("[info] data_root      = %s" % args.data_root)
    print("[info] mask_policy    = %s" % args.mask_policy)
    print("[info] mask geometry  = %s" % ("native" if size is None else "%dx%d" % size))
    print("[info] cv2 backend    = %s" % ("Pillow fallback" if _gmu.cv2 is None else "opencv"))

    records, skipped = [], 0
    for row in rows:
        rec = compute_row(row, args.data_root, args.mask_policy, size=size)
        if rec is None:
            skipped += 1
        else:
            records.append(rec)
    print("[info] usable         = %d (skipped %d)" % (len(records), skipped))

    summary = summarize(records)

    if os.path.dirname(args.out_json):
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "summary": summary}, f, indent=2, ensure_ascii=False)
    if records:
        fields = sorted({k for r in records for k in r})
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)

    print()
    header = "%-16s %5s | %-21s | %-21s" % (
        "subset", "n", "TxtCF floor mean/med", "TCF-LAB floor mean/med",
    )
    print(header)
    print("-" * len(header))
    for name, s in summary.items():
        def fmt(mean_key, med_key):
            if s[mean_key] is None:
                return "%21s" % "n/a"
            return "%9.3f / %9.3f" % (s[mean_key], s[med_key])
        print(
            "%-16s %5d | %-21s | %-21s"
            % (name, s["n"], fmt("txtcf_floor_mean", "txtcf_floor_median"),
               fmt("tcf_lab_floor_mean", "tcf_lab_floor_median"))
        )
    print()
    print("[ok] wrote %s" % args.out_json)


if __name__ == "__main__":
    main()
