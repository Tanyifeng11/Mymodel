#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段三: 训练采样配置 (AA-TCR Fuse)

方案 3.3 要求 batch 内 solid 占 40%~50%, 但训练集实际 solid 占 64.1%,
纯色样本会淹没纹理相关的学习信号。本脚本产出逐样本采样权重, 直接喂给
torch.utils.data.WeightedRandomSampler 即可。

做法: 按 (类别组 x solid标志) 划分单元格, 先按目标比例分配 solid / non-solid
的概率质量, 再在每一组内部按类别做幂次平滑(count^-alpha)重分配, 格内均匀。
这样 solid 比例可以精确命中, 同时长尾类别不会被完全淹没, 也不会被过度放大。

  * alpha=0   完全保持原始类别分布
  * alpha=0.5 平方根平滑(默认), 长尾适度抬升
  * alpha=1   完全均匀, 极端长尾会被过度采样, 不建议

脚本自带蒙特卡洛模拟, 会打印按该权重实际抽样得到的分布, 用于核对是否达标。

只读 JSON, 不打开图片, 无第三方依赖。

用法:
    python scripts/prepare_balanced_sampling.py
    python scripts/prepare_balanced_sampling.py --target-solid 0.45 --alpha 0.5
"""

import argparse
import collections
import json
import os
import random
import sys


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.path.join(here, "data"))
    ap.add_argument("--target-solid", type=float, default=0.45,
                    help="目标 solid 比例, 方案建议 0.40~0.50")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="类别平滑指数, 0=保持原分布, 1=完全均匀")
    ap.add_argument("--min-count", type=int, default=50,
                    help="样本数低于此值的类别合并为 other")
    ap.add_argument("--keep-non-garment", action="store_true",
                    help="保留手袋/发带等非服装样本(默认剔除)")
    ap.add_argument("--sim", type=int, default=200000, help="模拟抽样次数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not 0.0 < args.target_solid < 1.0:
        print("[error] --target-solid 必须在 (0,1) 之间")
        return 1

    proc = os.path.join(args.data_root, "processed")
    src = os.path.join(proc, "train_normalized.json")
    if not os.path.exists(src):
        print("[error] 找不到 %s, 请先跑 scripts/normalize_attributes.py" % src)
        return 1

    records = load_json(src)
    print("读入训练集 %d 条" % len(records))

    # ---- 1. 过滤 ----
    kept, dropped = [], collections.Counter()
    for r in records:
        cat = r["attributes_normalized"]["category_coarse"]
        if cat == "non_garment" and not args.keep_non_garment:
            dropped["non_garment"] += 1
            continue
        if cat == "unknown":
            dropped["category_unknown"] += 1
            continue
        if not r.get("texture_candidates"):
            dropped["no_texture"] += 1
            continue
        kept.append(r)
    print("可训练 %d 条, 剔除 %d 条 %s"
          % (len(kept), sum(dropped.values()), dict(dropped) or "{}"))

    # ---- 2. 类别分组, 长尾并入 other ----
    raw_counts = collections.Counter(
        r["attributes_normalized"]["category_coarse"] for r in kept)
    merged = {c for c, n in raw_counts.items() if n < args.min_count}
    group_of = lambda r: ("other"
                          if r["attributes_normalized"]["category_coarse"] in merged
                          else r["attributes_normalized"]["category_coarse"])
    if merged:
        print("并入 other 的长尾类别: %s"
              % {c: raw_counts[c] for c in sorted(merged, key=lambda x: -raw_counts[x])})

    # ---- 3. 单元格 (组, solid) ----
    cells = collections.defaultdict(list)
    for i, r in enumerate(kept):
        cells[(group_of(r), r["is_solid"])].append(i)

    n_solid = sum(1 for r in kept if r["is_solid"])
    print("原始 solid %d / non-solid %d (solid %.1f%%) -> 目标 %.1f%%"
          % (n_solid, len(kept) - n_solid, 100.0 * n_solid / len(kept),
             100.0 * args.target_solid))

    # ---- 4. 概率质量分配 ----
    # 先分 solid / non-solid 两块, 每块内部按类别做 count^-alpha 平滑
    cell_prob = {}
    for solid_flag, mass in [(True, args.target_solid), (False, 1.0 - args.target_solid)]:
        sub = {k: v for k, v in cells.items() if k[1] == solid_flag}
        if not sub:
            print("[warn] 没有 solid=%s 的样本, 该部分概率质量将被丢弃" % solid_flag)
            continue
        sm = {k: len(v) ** (1.0 - args.alpha) for k, v in sub.items()}
        tot = sum(sm.values())
        for k in sub:
            cell_prob[k] = mass * sm[k] / tot

    # ---- 5. 逐样本权重: 格内均匀 ----
    weights = [0.0] * len(kept)
    for k, idxs in cells.items():
        if not idxs or k not in cell_prob:
            continue
        w = cell_prob[k] / len(idxs)
        for i in idxs:
            weights[i] = w
    s = sum(weights)
    weights = [w / s for w in weights]  # 归一化, 便于阅读

    # ---- 6. 模拟验证 ----
    rng = random.Random(args.seed)
    picks = rng.choices(range(len(kept)), weights=weights, k=args.sim)
    sim_solid = sum(1 for i in picks if kept[i]["is_solid"]) / args.sim
    sim_cat = collections.Counter(group_of(kept[i]) for i in picks)
    sim_cat = {k: round(v / args.sim, 4) for k, v in sim_cat.most_common()}
    orig_cat = collections.Counter(group_of(r) for r in kept)
    orig_cat = {k: round(v / len(kept), 4) for k, v in orig_cat.most_common()}

    print("\n模拟抽样 %d 次:" % args.sim)
    print("  solid 实际 %.1f%%  (目标 %.1f%%)" % (sim_solid * 100, args.target_solid * 100))
    ok = abs(sim_solid - args.target_solid) < 0.01
    print("  solid 比例%s" % ("达标" if ok else " 偏离超过 1 个百分点, 请检查"))
    print("  类别分布 原始 -> 采样后:")
    for c in sorted(set(orig_cat) | set(sim_cat), key=lambda x: -orig_cat.get(x, 0)):
        print("    %-16s %6.2f%% -> %6.2f%%"
              % (c, orig_cat.get(c, 0) * 100, sim_cat.get(c, 0) * 100))

    ratio = [w / (1.0 / len(kept)) for w in weights]
    print("  单样本权重相对均匀采样的倍数: min %.2fx  max %.2fx" % (min(ratio), max(ratio)))
    if max(ratio) > 20:
        print("  [warn] 最大倍数偏高, 少数样本会被反复抽到, 可调低 --alpha")

    if args.dry_run:
        print("\n[dry-run] 未写文件")
        return 0

    # ---- 7. 输出 ----
    os.makedirs(proc, exist_ok=True)
    p = os.path.join(proc, "train_sampling_weights.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"ids": [r["id"] for r in kept], "weights": weights}, f)
    print("\n写出 %s" % p)

    cfg = {
        "_usage": "weights 与 ids 一一对应, 顺序即 train_sampling_weights.json 中的顺序。"
                  "用 WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)。",
        "params": {
            "target_solid": args.target_solid,
            "alpha": args.alpha,
            "min_count": args.min_count,
            "keep_non_garment": args.keep_non_garment,
            "seed": args.seed,
        },
        "n_trainable": len(kept),
        "dropped": dict(dropped),
        "merged_into_other": sorted(merged),
        "original_solid_ratio": round(n_solid / len(kept), 4),
        "simulated_solid_ratio": round(sim_solid, 4),
        "category_ratio_original": orig_cat,
        "category_ratio_sampled": sim_cat,
        "weight_multiplier_range": [round(min(ratio), 3), round(max(ratio), 3)],
    }
    p = os.path.join(proc, "sampling_config.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1)
    print("写出 %s" % p)

    p = os.path.join(proc, "train_trainable.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=1)
    print("写出 %s (%d 条, 已剔除非服装)" % (p, len(kept)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
