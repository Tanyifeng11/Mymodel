#!/usr/bin/env python
"""校验 compute_color_floors.py 复现的 mask 是否与既有报告一致。

地板值只有在 mask 与报告同源时才可与报告里的 E5/E7a 数字比较。
本脚本对齐 sample_id, 比较 garment_mask_pixels 与 mask_source。
"""
import argparse
import csv
import sys

import numpy as np


def load(path, key="sample_id"):
    with open(path, newline="", encoding="utf-8") as f:
        return {r[key]: r for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report_csv", default="eval_outputs/report_e7a/metrics_per_sample.csv")
    ap.add_argument("--floor_csv", default="eval_outputs/color_floors_per_sample.csv")
    args = ap.parse_args()

    report = load(args.report_csv)
    floor = load(args.floor_csv)

    mine, theirs, diffs = {}, {}, []
    for sid, fr in floor.items():
        rr = report.get(sid)
        if rr is None:
            continue
        ms = fr.get("mask_source") or "?"
        mine[ms] = mine.get(ms, 0) + 1
        rs = rr.get("mask_source") or "?"
        theirs[rs] = theirs.get(rs, 0) + 1
        try:
            a = float(rr["garment_mask_pixels"])
            b = float(fr["garment_mask_pixels"])
        except (KeyError, TypeError, ValueError):
            continue
        if a > 0:
            diffs.append(abs(a - b) / a)

    print("复现的 mask_source :", mine)
    print("报告的 mask_source :", theirs)
    if not diffs:
        print("[warn] 没有可比较的 garment_mask_pixels")
        return 1
    d = np.asarray(diffs)
    print()
    print("garment_mask_pixels 相对差异  n=%d" % len(d))
    print("  mean = %.5f" % d.mean())
    print("  p95  = %.5f" % np.percentile(d, 95))
    print("  max  = %.5f" % d.max())
    print("  完全一致 = %d/%d" % (int((d == 0).sum()), len(d)))
    print()
    if d.max() < 1e-9:
        print("[ok] mask 逐样本完全一致, 地板值可直接与报告中的 E5/E7a 数字比较。")
    elif d.mean() < 0.02:
        print("[ok] mask 平均差异 <2%, 地板值可比(存在轻微 resize 差异)。")
    else:
        print("[FAIL] mask 差异过大, 地板值不可与报告数字直接比较。先统一 mask 提取逻辑。")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
