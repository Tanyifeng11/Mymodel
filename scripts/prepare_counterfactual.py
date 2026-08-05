#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段五: 反事实干预评测集 (AA-TCR Fuse, 方案第 9 节)

为每个选中样本构造四组条件, 相同 seed, 每次只改一个因素:

    C0  原始文本 + 原始纹理           基准
    C1  改一个文本属性 + 原始纹理      测文本属性响应
    C2  原始文本 + 同类别替换纹理      测纹理属性响应
    C3  改一个文本属性 + 替换纹理      测冲突时的控制优先级

重要: 这些样本没有真实 GT, 按方案第 12 节只能用于评测和特征级对比,
不能计算普通扩散损失。输出里对每条都写明了这一点。

文本改写只做颜色替换, 因为颜色是 caption 里表层词最可枚举、替换后最不会
破坏句子合法性的属性; 材质和图案改写容易产生不自然的句子, 反而引入混淆。
替换纹理要求同 category_coarse 但目标属性不同, 构成困难负样本。

只读 JSON, 不打开图片, 无第三方依赖。

用法:
    python scripts/prepare_counterfactual.py
    python scripts/prepare_counterfactual.py --n 60 --source dev500
"""

import argparse
import collections
import json
import os
import random
import re
import sys

# 颜色 -> caption 中可能出现的表层词。用于定位和替换。
COLOR_SURFACE = {
    "black": ["black"],
    "white": ["white", "cream", "ivory", "off-white"],
    "gray": ["gray", "grey", "charcoal"],
    "beige": ["beige", "tan", "camel", "khaki", "nude"],
    "brown": ["brown"],
    "red": ["red", "maroon", "burgundy", "crimson", "scarlet"],
    "orange": ["orange", "coral", "peach", "rust"],
    "yellow": ["yellow", "mustard"],
    "green": ["green", "olive", "mint", "emerald", "sage"],
    "blue": ["blue", "navy", "turquoise", "indigo", "cobalt"],
    "purple": ["purple", "lilac", "lavender", "mauve", "violet", "plum"],
    "pink": ["pink", "rose", "blush", "fuchsia", "magenta"],
    "gold": ["gold"],
    "silver": ["silver"],
}

# 替换目标优先选视觉差异大的颜色, 避免 gray->beige 这种难以评测的组合
COLOR_FAR = {
    "black": "white", "white": "black", "gray": "red", "beige": "blue",
    "brown": "blue", "red": "blue", "orange": "blue", "yellow": "purple",
    "green": "red", "blue": "red", "purple": "yellow", "pink": "green",
    "gold": "blue", "silver": "red",
}


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_color_mention(caption, color):
    """在 caption 中定位该颜色的表层词, 返回 (surface, start, end) 或 None。"""
    cap = caption.lower()
    for w in COLOR_SURFACE.get(color, []):
        m = re.search(r"\b" + re.escape(w) + r"\b", cap)
        if m:
            return w, m.start(), m.end()
    return None


def rewrite_color(caption, old_surface, span, new_color):
    """只替换 caption 中定位到的那一处颜色词, 保留原始大小写风格。"""
    s, e = span
    orig = caption[s:e]
    repl = new_color.capitalize() if orig[:1].isupper() else new_color
    return caption[:s] + repl + caption[e:]


def pick_texture_donor(rec, pool_by_cat, rng, max_try=60):
    """
    找同类别但属性不同的纹理供体, 构成困难负样本。
    优先图案不同, 其次颜色不同; 都找不到就返回 None。
    """
    cat = rec["attributes_normalized"]["category_coarse"]
    cands = pool_by_cat.get(cat, [])
    if len(cands) < 2:
        return None, None

    a = rec["attributes_normalized"]
    my_pat, my_col = set(a["pattern"]), set(a["color"])
    fallback = None
    for _ in range(min(max_try, len(cands) * 3)):
        d = cands[rng.randrange(len(cands))]
        if d["id"] == rec["id"]:
            continue
        b = d["attributes_normalized"]
        pat_diff = not (set(b["pattern"]) & my_pat)
        col_diff = not (set(b["color"]) & my_col)
        if pat_diff and col_diff:
            return d, "pattern_and_color_differ"
        if (pat_diff or col_diff) and fallback is None:
            fallback = (d, "pattern_differs" if pat_diff else "color_differs")
    return fallback if fallback else (None, None)


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.path.join(here, "data"))
    ap.add_argument("--source", default="val_full",
                    choices=["dev100", "dev500", "val_full"],
                    help="从哪个评测集里选样本")
    ap.add_argument("--exclude", default="dev500",
                    choices=["none", "dev100", "dev500"],
                    help="排除哪个集合的样本。默认排除 dev500, 避免在 E7b "
                         "选型集上做干预评测造成软泄漏")
    ap.add_argument("--n", type=int, default=80, help="方案建议 40~100")
    ap.add_argument("--seed", type=int, default=42, help="划分用随机种子")
    ap.add_argument("--gen-seed", type=int, default=12345,
                    help="写进输出的生成种子, 四组共用")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    proc = os.path.join(args.data_root, "processed")
    src = os.path.join(proc, "%s.json" % args.source)
    if not os.path.exists(src):
        print("[error] 找不到 %s, 请先跑 scripts/create_eval_splits.py" % src)
        return 1

    records = load_json(src)
    print("读入 %s: %d 条" % (args.source, len(records)))

    # 排除选型集, 保证干预评测的独立性
    if args.exclude != "none":
        ex_path = os.path.join(proc, "%s.json" % args.exclude)
        if os.path.exists(ex_path):
            ex_ids = {r["id"] for r in load_json(ex_path)}
            before = len(records)
            records = [r for r in records if r["id"] not in ex_ids]
            print("排除 %s 后剩 %d 条 (剔除 %d)"
                  % (args.exclude, len(records), before - len(records)))
            if not records:
                print("[error] 排除后没有样本了, 请改小 --exclude 或换 --source")
                return 1
        else:
            print("[warn] 找不到 %s, 跳过排除" % ex_path)

    # 供体池用全量验证集, 池子越大越容易找到合适的困难负样本
    full = os.path.join(proc, "val_full.json")
    donors = load_json(full) if os.path.exists(full) else records
    pool_by_cat = collections.defaultdict(list)
    for r in donors:
        pool_by_cat[r["attributes_normalized"]["category_coarse"]].append(r)

    # ---- 候选筛选: 属性干净, 且 caption 里能定位到颜色词 ----
    rng = random.Random(args.seed)
    cands, reject = [], collections.Counter()
    for r in records:
        a = r["attributes_normalized"]
        if r["attribute_conflicts"]:
            reject["has_conflict"] += 1
            continue
        if not r["aux_supervision"]["color"]:
            reject["color_unusable"] += 1
            continue
        colors = [c for c in a["color"] if c in COLOR_SURFACE]
        if len(colors) != 1:
            # 多色样本改写后语义歧义大, 单色最干净
            reject["not_single_color"] += 1
            continue
        hit = find_color_mention(r["caption"], colors[0])
        if not hit:
            reject["color_not_in_caption"] += 1
            continue
        cands.append((r, colors[0], hit))
    print("候选 %d 条, 排除 %s" % (len(cands), dict(reject)))

    if not cands:
        print("[error] 没有可用候选")
        return 1

    # 按类别轮转, 保证类别覆盖
    by_cat = collections.defaultdict(list)
    for item in cands:
        by_cat[item[0]["attributes_normalized"]["category_coarse"]].append(item)
    for k in by_cat:
        rng.shuffle(by_cat[k])
    order = sorted(by_cat, key=lambda k: -len(by_cat[k]))

    out, used, exhausted = [], set(), set()
    while len(out) < args.n and len(exhausted) < len(order):
        for cat in order:
            if len(out) >= args.n:
                break
            if cat in exhausted:
                continue
            if not by_cat[cat]:
                exhausted.add(cat)
                continue
            rec, color, (surface, s, e) = by_cat[cat].pop()
            if rec["id"] in used:
                continue

            donor, why = pick_texture_donor(rec, pool_by_cat, rng)
            if donor is None:
                reject["no_texture_donor"] += 1
                continue

            new_color = COLOR_FAR.get(color, "blue")
            new_cap = rewrite_color(rec["caption"], surface, (s, e), new_color)
            if new_cap == rec["caption"]:
                reject["rewrite_noop"] += 1
                continue

            used.add(rec["id"])
            a = rec["attributes_normalized"]
            out.append({
                "id": rec["id"],
                "category_coarse": a["category_coarse"],
                "gen_seed": args.gen_seed,
                "sketch": rec["sketch"],
                "mask": rec["mask"],
                "reference_target": rec["target_image"],
                "_gt_note": "reference_target 只对 C0 是真 GT; C1/C2/C3 无 GT, "
                            "禁止用于扩散损失, 仅供评测与特征级对比",
                "intervention": {
                    "text_attribute": "color",
                    "color_original": color,
                    "color_modified": new_color,
                    "surface_replaced": surface,
                    "texture_donor_id": donor["id"],
                    "texture_donor_reason": why,
                },
                "conditions": {
                    "C0": {"caption": rec["caption"],
                           "texture": rec["texture_candidates"][0],
                           "desc": "原始文本 + 原始纹理"},
                    "C1": {"caption": new_cap,
                           "texture": rec["texture_candidates"][0],
                           "desc": "改颜色 + 原始纹理"},
                    "C2": {"caption": rec["caption"],
                           "texture": donor["texture_candidates"][0],
                           "desc": "原始文本 + 替换纹理"},
                    "C3": {"caption": new_cap,
                           "texture": donor["texture_candidates"][0],
                           "desc": "改颜色 + 替换纹理"},
                },
                "expected": {
                    # 方案第 9 节的预设控制规则
                    "C1": "颜色应跟随文本变为 %s; 结构和材质保持" % new_color,
                    "C2": "材质图案应跟随新纹理; 文本未改, 类别与版型保持",
                    "C3": "显式颜色应由文本主导(%s), 材质图案由纹理主导" % new_color,
                    "all": "草图结构、Leak、Boundary 在四组间应保持稳定",
                },
            })

    print("构造 %d 个样本 x 4 组 = %d 次生成" % (len(out), len(out) * 4))
    if len(out) < args.n:
        print("[warn] 只凑到 %d 个(目标 %d), 原因: %s" % (len(out), args.n, dict(reject)))

    cat_dist = collections.Counter(o["category_coarse"] for o in out)
    print("类别分布: %s" % json.dumps(dict(cat_dist.most_common()), ensure_ascii=False))
    donor_dist = collections.Counter(o["intervention"]["texture_donor_reason"] for o in out)
    print("供体质量: %s" % dict(donor_dist))

    if args.dry_run:
        print("\n[dry-run] 未写文件")
        return 0

    payload = {
        "_meta": {
            "source_split": args.source,
            "excluded_split": args.exclude,
            "independence": "样本取自 %s 且已排除 %s, 与 E7b 选型集不重叠"
                            % (args.source, args.exclude),
            "n_samples": len(out),
            "n_generations": len(out) * 4,
            "gen_seed": args.gen_seed,
            "split_seed": args.seed,
            "rule": "四组必须使用同一 gen_seed, 每次只改变一个因素",
            "no_gt_warning": "C1/C2/C3 没有真实目标图。按方案第 12 节, "
                             "这些样本只能用于评测或特征级对比学习, "
                             "不得参与普通扩散重建损失。",
            "category_distribution": dict(cat_dist.most_common()),
            "donor_quality": dict(donor_dist),
        },
        "samples": out,
    }
    p = os.path.join(proc, "counterfactual_test.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print("\n写出 %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
