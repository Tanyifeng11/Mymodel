#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段二: 属性标签标准化 (AA-TCR Fuse)

读 data/raw/*.jsonl, 按 configs/attribute_vocab.json 把原始 category/color/
material/pattern/sleeve/collar 映射为统一标签, 写出 attributes_normalized。

只处理文本字段, 不打开任何图片, 无 torch/PIL/numpy 依赖, CPU 几秒跑完。

用法:
    python scripts/normalize_attributes.py
    python scripts/normalize_attributes.py --dry-run          # 只看报告不写文件
    python scripts/normalize_attributes.py --data-root data   # 自定义根目录

输出:
    data/processed/train_normalized.json
    data/processed/val_normalized.json
    data/processed/attribute_statistics.json
    data/processed/unmapped_report.json     # 落到 unknown 的原始值, 用于迭代词表
"""

import argparse
import collections
import json
import os
import re
import sys

# ---------------------------------------------------------------- 词表加载

def load_vocab(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def norm_text(v):
    """原始值 -> 小写去空白, 空值统一为 None。"""
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("", "none", "null", "n/a", "na", "unknown"):
        return None
    return re.sub(r"\s+", " ", s)


# ---------------------------------------------------------------- 单属性映射
# 每个 map_* 返回 (canonical_label, reason)
# reason: exact / rule / leak:<field> / empty / unmapped / not_applicable

def map_category(raw, vocab):
    s = norm_text(raw)
    if s is None:
        return "unknown", "empty"
    cfg = vocab["category"]
    if s in cfg["exact"]:
        return cfg["exact"][s], "exact"
    for pat, lab in cfg["rules"]:
        if pat == "_comment":
            continue
        if pat in s:
            return lab, "rule"
    return "unknown", "unmapped"


def map_color_one(raw, vocab):
    s = norm_text(raw)
    if s is None:
        return None, "empty"
    cfg = vocab["color"]
    if s in cfg["leak"]:
        return None, "leak:" + cfg["leak"][s]
    if s in cfg["exact"]:
        return cfg["exact"][s], "exact"
    # 剥离 light/dark/pastel/metallic... 等修饰词后重试
    toks = re.split(r"[\s\-]+", s)
    stripped = [t for t in toks if t not in cfg["modifiers"]]
    if stripped and len(stripped) < len(toks):
        cand = " ".join(stripped)
        if cand in cfg["exact"]:
            return cfg["exact"][cand], "rule"
        if len(stripped) == 1 and stripped[0] in cfg["canonical"]:
            return stripped[0], "rule"
    # 末词兜底: "medium wash blue" -> blue
    if stripped:
        last = stripped[-1]
        if last in cfg["exact"]:
            return cfg["exact"][last], "rule"
    return None, "unmapped"


def map_simple_list(raw_list, vocab, field):
    """material / pattern: 原始已较干净, 只做 exact + canonical 校验。"""
    cfg = vocab[field]
    canon = set(cfg["canonical"])
    out, reasons = [], []
    for raw in raw_list or []:
        s = norm_text(raw)
        if s is None:
            reasons.append(("", "empty"))
            continue
        if s in canon:
            lab = s
            r = "exact"
        elif s in cfg["exact"]:
            lab = cfg["exact"][s]
            r = "exact"
        else:
            reasons.append((s, "unmapped"))
            continue
        if lab not in out:
            out.append(lab)
        reasons.append((s, r))
    return out, reasons


def map_sleeve(raw, vocab, category):
    cfg = vocab["sleeve"]
    if category in vocab["category"]["no_sleeve_collar"]:
        return "not_applicable", "not_applicable"
    s = norm_text(raw)
    if s is None:
        return "unknown", "empty"
    # 先试长度规则: "long-sleeved mesh" 里的 long 是有效信息, 不该被材质词吃掉
    for pat, lab in cfg["rules"]:
        if pat == "_comment":
            continue
        if pat in s:
            return lab, "rule" if lab != "unknown" else "unsupported"
    if s in cfg["leak_exact"]:
        return "unknown", "leak:material_or_finish"
    if s in cfg["shape_only_exact"]:
        return "unknown", "leak:shape_only"
    return "unknown", "unmapped"


def map_collar(raw, vocab, category):
    cfg = vocab["collar"]
    if category in vocab["category"]["no_sleeve_collar"]:
        return "not_applicable", "not_applicable"
    s = norm_text(raw)
    if s is None:
        return "unknown", "empty"
    if s in cfg["leak_exact"]:
        return "unknown", "leak:material_or_generic"
    for pat, lab in cfg["rules"]:
        if pat == "_comment":
            continue
        if pat in s:
            return lab, "rule" if lab != "unknown" else "unsupported"
    return "unknown", "unmapped"


# ------------------------------------------------- caption 冲突检测 (方案 3.2)
# caption 明确说了另一个颜色/图案, 而标签给的不同 -> 标 conflict,
# 该属性不参与辅助监督, 但样本本身仍用于扩散训练。

_COLOR_SURFACE = {
    "black": ["black"], "white": ["white", "cream", "ivory", "off-white"],
    "gray": ["gray", "grey", "charcoal"], "beige": ["beige", "tan", "camel", "khaki", "nude"],
    "brown": ["brown"], "red": ["red", "maroon", "burgundy", "crimson", "scarlet"],
    "orange": ["orange", "coral", "peach", "rust"],
    "yellow": ["yellow", "mustard"],
    "green": ["green", "olive", "mint", "teal green", "emerald", "sage"],
    "blue": ["blue", "navy", "denim blue", "turquoise", "indigo", "cobalt"],
    "purple": ["purple", "lilac", "lavender", "mauve", "violet", "plum"],
    "pink": ["pink", "rose", "blush", "fuchsia", "magenta"],
    "gold": ["gold"], "silver": ["silver"],
    "multicolor": ["multicolor", "multicolored", "multi-color", "rainbow", "colorful"],
}

_PATTERN_SURFACE = {
    "solid": ["solid"], "floral": ["floral", "flower"], "striped": ["striped", "stripe"],
    "plaid": ["plaid", "checkered", "gingham", "tartan"],
    "polka dot": ["polka dot", "polka-dot"],
    "animal print": ["leopard", "zebra", "animal print", "snakeskin"],
    "camouflage": ["camo", "camouflage"],
    "geometric": ["geometric"], "abstract": ["abstract"],
    "embroidered": ["embroider"], "lace/openwork": ["lace", "openwork", "crochet"],
    "graphic/text print": ["graphic print", "text print", "logo print", "printed graphic"],
    "color block": ["color block", "colorblock", "color-block"],
}


def _mentioned(caption_lc, surface_map):
    hits = set()
    for label, words in surface_map.items():
        for w in words:
            if re.search(r"\b" + re.escape(w) + r"\b", caption_lc):
                hits.add(label)
                break
    return hits


def detect_conflicts(caption, norm):
    """返回冲突属性名集合。caption 提到了该属性的值, 且与标签完全不交集才算冲突。"""
    cap = (caption or "").lower()
    conflicts = []

    cap_colors = _mentioned(cap, _COLOR_SURFACE)
    lab_colors = set(norm["color"])
    if cap_colors and lab_colors and not (cap_colors & lab_colors):
        conflicts.append("color")

    cap_pats = _mentioned(cap, _PATTERN_SURFACE)
    lab_pats = set(norm["pattern"])
    if cap_pats and lab_pats and not (cap_pats & lab_pats):
        conflicts.append("pattern")

    return conflicts


# ---------------------------------------------------------------- 单样本处理

def normalize_record(rec, vocab, stats, unmapped):
    attrs = rec.get("attributes") or {}

    cat, r = map_category(attrs.get("category"), vocab)
    stats["category"][r] += 1
    if r == "unmapped":
        unmapped["category"][norm_text(attrs.get("category"))] += 1

    colors = []
    for raw in attrs.get("color") or []:
        lab, r = map_color_one(raw, vocab)
        stats["color"][r] += 1
        if r == "unmapped":
            unmapped["color"][norm_text(raw)] += 1
        if lab and lab not in colors:
            colors.append(lab)
    if not colors:
        colors = ["unknown"]

    mats, mr = map_simple_list(attrs.get("material"), vocab, "material")
    for s, r in mr:
        stats["material"][r] += 1
        if r == "unmapped":
            unmapped["material"][s] += 1
    if not mats:
        mats = ["unknown"]

    pats, pr = map_simple_list(attrs.get("pattern"), vocab, "pattern")
    for s, r in pr:
        stats["pattern"][r] += 1
        if r == "unmapped":
            unmapped["pattern"][s] += 1
    if not pats:
        pats = ["unknown"]

    sl, r = map_sleeve(attrs.get("sleeve"), vocab, cat)
    stats["sleeve"][r] += 1
    if r == "unmapped":
        unmapped["sleeve"][norm_text(attrs.get("sleeve"))] += 1

    co, r = map_collar(attrs.get("collar"), vocab, cat)
    stats["collar"][r] += 1
    if r == "unmapped":
        unmapped["collar"][norm_text(attrs.get("collar"))] += 1

    norm = {
        "category_coarse": cat,
        "category_raw": attrs.get("category"),
        "color": colors,
        "material": mats,
        "pattern": pats,
        "sleeve": sl,
        "collar": co,
    }

    conflicts = detect_conflicts(rec.get("caption"), norm)

    out = dict(rec)
    out["attributes_normalized"] = norm
    out["attribute_conflicts"] = conflicts
    # 方案 3.3: 采样用。solid 的定义是 pattern 恰为 [solid]
    out["is_solid"] = norm["pattern"] == ["solid"]
    # 方案 5.2: 只有非冲突且非 unknown 的属性才进辅助损失
    out["aux_supervision"] = {
        "color": "color" not in conflicts and norm["color"] != ["unknown"],
        "material": norm["material"] != ["unknown"],
        "pattern": "pattern" not in conflicts and norm["pattern"] != ["unknown"],
    }
    return out


# ---------------------------------------------------------------- 统计汇总

def summarize(records, split):
    def dist(key):
        c = collections.Counter()
        for r in records:
            v = r["attributes_normalized"][key]
            if isinstance(v, list):
                for x in v:
                    c[x] += 1
            else:
                c[v] += 1
        return dict(c.most_common())

    n = len(records)
    n_solid = sum(1 for r in records if r["is_solid"])
    return {
        "split": split,
        "n": n,
        "n_solid": n_solid,
        "n_non_solid": n - n_solid,
        "solid_ratio": round(n_solid / n, 4) if n else 0.0,
        "distributions": {k: dist(k) for k in
                          ["category_coarse", "color", "material", "pattern", "sleeve", "collar"]},
        "conflicts": {
            "color": sum(1 for r in records if "color" in r["attribute_conflicts"]),
            "pattern": sum(1 for r in records if "pattern" in r["attribute_conflicts"]),
            "any": sum(1 for r in records if r["attribute_conflicts"]),
        },
        "aux_usable": {
            k: sum(1 for r in records if r["aux_supervision"][k])
            for k in ["color", "material", "pattern"]
        },
        "unknown_counts": {
            "category_coarse": sum(1 for r in records
                                   if r["attributes_normalized"]["category_coarse"] == "unknown"),
            "color": sum(1 for r in records if r["attributes_normalized"]["color"] == ["unknown"]),
            "material": sum(1 for r in records if r["attributes_normalized"]["material"] == ["unknown"]),
            "pattern": sum(1 for r in records if r["attributes_normalized"]["pattern"] == ["unknown"]),
            "sleeve": sum(1 for r in records if r["attributes_normalized"]["sleeve"] == "unknown"),
            "collar": sum(1 for r in records if r["attributes_normalized"]["collar"] == "unknown"),
        },
        "non_garment": sum(1 for r in records
                           if r["attributes_normalized"]["category_coarse"] == "non_garment"),
    }


# ---------------------------------------------------------------- 主流程

def read_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                print("  [warn] %s:%d 解析失败: %s" % (os.path.basename(path), i, e))
    return out


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.path.join(here, "data"))
    ap.add_argument("--vocab", default=os.path.join(here, "configs", "attribute_vocab.json"))
    ap.add_argument("--dry-run", action="store_true", help="只打印报告, 不写文件")
    args = ap.parse_args()

    vocab = load_vocab(args.vocab)
    raw_dir = os.path.join(args.data_root, "raw")
    out_dir = os.path.join(args.data_root, "processed")

    splits = [
        ("train", "bf_fashion_training_full.jsonl", "train_normalized.json"),
        ("val", "bf_fashion_validation_garment_only.jsonl", "val_normalized.json"),
    ]

    all_stats, all_unmapped = {}, {}

    for split, src, dst in splits:
        src_path = os.path.join(raw_dir, src)
        if not os.path.exists(src_path):
            print("[error] 找不到 %s" % src_path)
            return 1

        print("\n==== %s: %s" % (split, src))
        records = read_jsonl(src_path)
        print("  读入 %d 条" % len(records))

        stats = collections.defaultdict(collections.Counter)
        unmapped = collections.defaultdict(collections.Counter)
        out = [normalize_record(r, vocab, stats, unmapped) for r in records]

        summary = summarize(out, split)
        all_stats[split] = summary
        all_unmapped[split] = {k: dict(v.most_common(50)) for k, v in unmapped.items()}

        print("  category  -> %d 类, unknown %d, non_garment %d"
              % (len(summary["distributions"]["category_coarse"]),
                 summary["unknown_counts"]["category_coarse"], summary["non_garment"]))
        for k in ["color", "material", "pattern", "sleeve", "collar"]:
            print("  %-9s -> %d 类, unknown %d"
                  % (k, len(summary["distributions"][k]), summary["unknown_counts"][k]))
        print("  solid %d / non-solid %d  (solid %.1f%%)"
              % (summary["n_solid"], summary["n_non_solid"], summary["solid_ratio"] * 100))
        print("  caption 冲突: color %d, pattern %d, 合计 %d"
              % (summary["conflicts"]["color"], summary["conflicts"]["pattern"],
                 summary["conflicts"]["any"]))
        print("  可用于辅助监督: color %d, material %d, pattern %d"
              % (summary["aux_usable"]["color"], summary["aux_usable"]["material"],
                 summary["aux_usable"]["pattern"]))

        if not args.dry_run:
            os.makedirs(out_dir, exist_ok=True)
            p = os.path.join(out_dir, dst)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=1)
            print("  写出 %s" % p)

    if not args.dry_run:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "attribute_statistics.json"), "w", encoding="utf-8") as f:
            json.dump(all_stats, f, ensure_ascii=False, indent=1)
        with open(os.path.join(out_dir, "unmapped_report.json"), "w", encoding="utf-8") as f:
            json.dump(all_unmapped, f, ensure_ascii=False, indent=1)
        print("\n写出 attribute_statistics.json / unmapped_report.json -> %s" % out_dir)
    else:
        print("\n[dry-run] 未写任何文件")

    print("\n下一步: 查看 data/processed/unmapped_report.json,")
    print("把仍落在 unknown 的高频原始值补进 configs/attribute_vocab.json 后重跑本脚本。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
