#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段四: 固定验证集划分 (AA-TCR Fuse)

从 data/processed/val_normalized.json 切出三个固定评测集:

    dev100    - 覆盖均衡的小验证集, 用于 E7b 单变量消融快速筛选
    dev500    - dev100 的超集, 分布接近真实, 用于中等规模比较
    val_full  - 全量验证集(剔除非服装), 用于最终 FID / KID

设计要点:
  * 固定 seed, 结果可复现; 同一 seed 重跑得到完全相同的划分。
  * dev100 ⊂ dev500 ⊂ val_full, 三者嵌套, 指标可直接对比。
  * dev100 按 (类别 × solid/non-solid) 轮转取样, 保证纹理类型和类别都有覆盖;
    solid 压到约 50%, 否则纯色样本会淹没纹理相关指标。
  * dev500 = dev100 + 按真实分布补齐, 保留分布形状以便 FID 有意义。

只读 JSON, 不打开图片, 无第三方依赖。

用法:
    python scripts/create_eval_splits.py
    python scripts/create_eval_splits.py --seed 42 --dev100 100 --dev500 500
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


def is_eligible(rec):
    """能进评测集的基本条件。返回 (bool, 原因)。"""
    a = rec["attributes_normalized"]
    if a["category_coarse"] == "non_garment":
        return False, "non_garment"
    if a["category_coarse"] == "unknown":
        return False, "category_unknown"
    if not rec.get("texture_candidates"):
        return False, "no_texture"
    if not rec.get("mask") or not rec.get("sketch") or not rec.get("target_image"):
        return False, "missing_modality"
    return True, "ok"


def is_high_quality(rec):
    """dev100 额外要求: 属性干净, 无 caption 冲突, 颜色和图案都可用。"""
    if rec["attribute_conflicts"]:
        return False
    aux = rec["aux_supervision"]
    return aux["color"] and aux["pattern"]


def round_robin_pick(pool, quota, rng, keyfunc):
    """
    按 keyfunc 分组后轮转取样, 让各组尽量均匀出现。
    组内随机, 组间按组大小降序轮转。返回选中的样本列表。
    """
    cells = collections.defaultdict(list)
    for r in pool:
        cells[keyfunc(r)].append(r)
    for k in cells:
        rng.shuffle(cells[k])

    order = sorted(cells.keys(), key=lambda k: -len(cells[k]))
    picked, exhausted = [], set()
    while len(picked) < quota and len(exhausted) < len(order):
        for k in order:
            if len(picked) >= quota:
                break
            if k in exhausted:
                continue
            if cells[k]:
                picked.append(cells[k].pop())
            else:
                exhausted.add(k)
    return picked


def build_dev100(pool, n, rng):
    """solid / non-solid 各半, 每半内按类别轮转。"""
    hq = [r for r in pool if is_high_quality(r)]
    # 高质量样本不够就放宽到全池, 保证数量优先
    src = hq if len(hq) >= n * 2 else pool

    solid = [r for r in src if r["is_solid"]]
    nonsolid = [r for r in src if not r["is_solid"]]

    n_solid = min(n // 2, len(solid))
    n_non = min(n - n_solid, len(nonsolid))
    n_solid = min(n - n_non, len(solid))  # 一边不够时另一边补足

    cat = lambda r: r["attributes_normalized"]["category_coarse"]
    out = round_robin_pick(solid, n_solid, rng, cat)
    out += round_robin_pick(nonsolid, n_non, rng, cat)
    rng.shuffle(out)
    return out, len(hq)


def build_dev500(pool, seed_set, n, rng):
    """dev100 为种子, 其余按 (类别, solid) 分层比例补齐, 保留真实分布形状。"""
    seed_ids = {r["id"] for r in seed_set}
    rest = [r for r in pool if r["id"] not in seed_ids]
    need = n - len(seed_set)
    if need <= 0:
        return list(seed_set)

    key = lambda r: (r["attributes_normalized"]["category_coarse"], r["is_solid"])
    cells = collections.defaultdict(list)
    for r in rest:
        cells[key(r)].append(r)
    for k in cells:
        rng.shuffle(cells[k])

    total = len(rest)
    # 按比例分配, 先给整数份额
    alloc = {k: int(need * len(v) / total) for k, v in cells.items()}
    picked = []
    for k, cnt in alloc.items():
        picked.extend(cells[k][:cnt])
        cells[k] = cells[k][cnt:]

    # 余数用剩余样本随机补齐
    leftover = [r for v in cells.values() for r in v]
    rng.shuffle(leftover)
    picked.extend(leftover[: need - len(picked)])

    out = list(seed_set) + picked
    rng.shuffle(out)
    return out


def describe(records, name):
    def dist(key):
        c = collections.Counter()
        for r in records:
            v = r["attributes_normalized"][key]
            for x in (v if isinstance(v, list) else [v]):
                c[x] += 1
        return dict(c.most_common())

    n = len(records)
    n_solid = sum(1 for r in records if r["is_solid"])
    return {
        "name": name,
        "n": n,
        "n_solid": n_solid,
        "solid_ratio": round(n_solid / n, 4) if n else 0.0,
        "n_categories": len(dist("category_coarse")),
        "n_with_conflict": sum(1 for r in records if r["attribute_conflicts"]),
        "distributions": {k: dist(k) for k in
                          ["category_coarse", "color", "material", "pattern"]},
    }


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.path.join(here, "data"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dev100", type=int, default=100)
    ap.add_argument("--dev500", type=int, default=500)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    proc = os.path.join(args.data_root, "processed")
    src = os.path.join(proc, "val_normalized.json")
    if not os.path.exists(src):
        print("[error] 找不到 %s, 请先跑 scripts/normalize_attributes.py" % src)
        return 1

    records = load_json(src)
    print("读入验证集 %d 条" % len(records))

    pool, dropped = [], collections.Counter()
    for r in records:
        ok, why = is_eligible(r)
        (pool.append(r) if ok else dropped.update([why]))
    print("可用 %d 条, 剔除 %d 条 %s" % (len(pool), sum(dropped.values()), dict(dropped)))

    rng = random.Random(args.seed)
    d100, n_hq = build_dev100(pool, args.dev100, rng)
    d500 = build_dev500(pool, d100, args.dev500, rng)

    print("高质量候选(无冲突且颜色图案可用) %d 条" % n_hq)

    # 嵌套性校验: 失败说明构造逻辑有 bug, 直接报错而不是写出错误文件
    ids100, ids500 = {r["id"] for r in d100}, {r["id"] for r in d500}
    idsfull = {r["id"] for r in pool}
    assert ids100 <= ids500 <= idsfull, "嵌套关系被破坏"
    assert len(ids100) == len(d100) and len(ids500) == len(d500), "存在重复 id"

    splits = [("dev100", d100), ("dev500", d500), ("val_full", pool)]
    summary = {}
    for name, recs in splits:
        info = describe(recs, name)
        summary[name] = info
        print("\n-- %s: n=%d, solid %.1f%%, 类别 %d, 含冲突 %d"
              % (name, info["n"], info["solid_ratio"] * 100,
                 info["n_categories"], info["n_with_conflict"]))
        print("   类别: %s" % json.dumps(info["distributions"]["category_coarse"],
                                        ensure_ascii=False))

    summary["_meta"] = {
        "seed": args.seed,
        "nested": "dev100 subset of dev500 subset of val_full",
        "source": "data/processed/val_normalized.json",
        "excluded_from_val_full": dict(dropped),
        "note_fid": "dev100 样本量过小, FID/KID 方差很大, 只用于看趋势; "
                    "正式 FID 请用 val_full, 中间比较用 dev500。",
    }

    if args.dry_run:
        print("\n[dry-run] 未写文件")
        return 0

    for name, recs in splits:
        p = os.path.join(proc, "%s.json" % name)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False, indent=1)
        print("写出 %s (%d 条)" % (p, len(recs)))

    # id 清单单独存一份, 方便评测脚本核对用的是不是同一批样本
    p = os.path.join(proc, "eval_split_ids.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({name: [r["id"] for r in recs] for name, recs in splits},
                  f, ensure_ascii=False, indent=1)
    print("写出 %s" % p)

    p = os.path.join(proc, "eval_split_statistics.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print("写出 %s" % p)

    print("\n提示: dev100 只有 %d 条, FID/KID 在这个量级方差很大," % len(d100))
    print("      建议只用它看趋势和做 E7b 快速筛选, 正式 FID 用 val_full。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
