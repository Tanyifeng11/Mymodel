#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
属性 token mask 构造器 (AA-TCR Fuse 前置依赖)

方案第 133 行要求"根据颜色、材质和图案词的位置构造文本属性 mask"。
本模块把 caption 中的属性词定位到 CLIP tokenizer 的 token 下标上, 产出

    {"color": BoolTensor[B,77], "material": ..., "pattern": ...}

难点在 CLIP 用的是 BPE: "off-white" / "burgundy" 会被切成多个子词, 必须把
一个属性短语覆盖的所有子词 token 全部标上, 漏标会让交叉注意力只看到半个词。

做法: 用 tokenizer 的 offset mapping 把每个 token 映射回原文字符区间, 再与
正则匹配到的属性词字符区间求交。offset mapping 只有 fast tokenizer 才有;
慢速 tokenizer 会自动回退到逐 token 解码对齐, 结果一致但稍慢。

方案第 441 行把"token mask 对齐错误"列为头号排查项, 所以这里带自检:
    python -m models.attribute_token_mask --self-test
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

import torch

# ---------------------------------------------------------------- 属性表层词
# 与 scripts/prepare_counterfactual.py 的 COLOR_SURFACE 保持一致, 并扩展
# material / pattern。多词短语必须排在单词前面, 否则 "light blue" 会先被
# "blue" 吃掉半截 -> 见 _compile 里的长度降序排序。

COLOR_WORDS: Dict[str, List[str]] = {
    "black": ["black"],
    "white": ["white", "cream", "ivory", "off-white", "off white"],
    "gray": ["gray", "grey", "charcoal", "heather gray", "heather grey"],
    "beige": ["beige", "tan", "camel", "khaki", "nude"],
    "brown": ["brown", "chocolate"],
    "red": ["red", "maroon", "burgundy", "crimson", "scarlet", "wine"],
    "orange": ["orange", "coral", "peach", "rust", "salmon", "apricot"],
    "yellow": ["yellow", "mustard", "lemon"],
    "green": ["green", "olive", "mint", "emerald", "sage", "lime"],
    "blue": ["blue", "navy", "turquoise", "indigo", "cobalt", "teal", "aqua"],
    "purple": ["purple", "lilac", "lavender", "mauve", "violet", "plum", "eggplant"],
    "pink": ["pink", "rose", "blush", "fuchsia", "magenta"],
    "gold": ["gold", "golden"],
    "silver": ["silver"],
    # 只移除 ombre/gradient/tie-dye(描述分布方式, 归 pattern);
    # multicolored/colorful 仍是颜色描述, 留在 color。
    "multicolor": ["multicolor", "multicolored", "multi-color", "rainbow",
                   "colorful"],
}

MATERIAL_WORDS: Dict[str, List[str]] = {
    "denim": ["denim", "jean"],
    "knit": ["knit", "knitted", "cable knit", "ribbed knit"],
    "lace": ["lace"],
    "leather": ["leather", "faux leather", "pu leather"],
    "fur": ["fur", "faux fur", "shearling"],
    "sequin": ["sequin", "sequins", "sequined"],
    "velvet": ["velvet"],
    "mesh": ["mesh"],
    "wool": ["wool", "woolen", "woollen"],
    "satin": ["satin"],
    "suede": ["suede"],
    "corduroy": ["corduroy"],
    "tweed": ["tweed"],
    "chiffon": ["chiffon"],
    "fleece": ["fleece"],
    "silk": ["silk"],
    "cotton": ["cotton"],
    "linen": ["linen"],
}

PATTERN_WORDS: Dict[str, List[str]] = {
    "solid": ["solid"],
    "lace/openwork": ["openwork", "crochet", "eyelet", "lace-trimmed", "sheer"],
    # sequins 移到 material, embroidered 不再包含 sequins
    "embroidered": ["embroidered", "embroidery", "beaded", "applique",
                    "embellished", "embellishments"],
    # "logo"/"patch" 在真实数据上精确率 <0.5 (solid 上误触发多于非 solid 命中),
    # 已剔除; 保留短语形式的 "logo print" 等仍可命中。
    "graphic/text print": ["graphic print", "text print", "logo print",
                           "printed graphic", "graphic", "print", "printed",
                           "motif"],
    "floral": ["floral", "flower", "flowered"],
    "striped": ["striped", "stripe", "stripes", "pinstripe"],
    "color block": ["color block", "colorblock", "color-block", "colorblocked",
                    "patchwork"],
    "geometric": ["geometric"],
    "animal print": ["animal print", "leopard", "zebra", "snakeskin", "cheetah"],
    "plaid": ["plaid", "checkered", "check", "gingham", "tartan"],
    "abstract": ["abstract"],
    "polka dot": ["polka dot", "polka dots", "polka-dot"],
    "camouflage": ["camouflage", "camo"],
    # 泛指词: caption 常只说 "with an intricate pattern" 而不点名具体图案。
    # 不加这些的话, 非 solid 样本里只有 26% 能定位到 pattern token。
    "generic": ["pattern", "patterns", "patterned", "scalloped"],
    # 从 color 移过来: 这些描述的是分布方式而非单一颜色。
    # multicolored/colorful 不放这里 —— 它们是颜色描述, 归 color 一路。
    "multicolor_pattern": ["ombre", "gradient", "tie-dye"],
}

ATTRIBUTE_WORDS: Dict[str, Dict[str, List[str]]] = {
    "color": COLOR_WORDS,
    "material": MATERIAL_WORDS,
    "pattern": PATTERN_WORDS,
}

ATTRIBUTE_NAMES: Tuple[str, ...] = ("color", "material", "pattern")


def _compile(words: Dict[str, List[str]]) -> re.Pattern:
    """长短语优先, 避免 'polka dot' 被 'dot' 抢先匹配。"""
    surfaces = sorted({w for v in words.values() for w in v}, key=len, reverse=True)
    # 词边界用 \b, 但短语内部允许连字符和空格
    return re.compile(r"\b(?:" + "|".join(re.escape(s) for s in surfaces) + r")\b",
                      re.IGNORECASE)


_PATTERNS = {k: _compile(v) for k, v in ATTRIBUTE_WORDS.items()}

# 表层词 -> 归一化标签 的反查表, 供 caption/标签交叉校验用。
# 例: "burgundy" -> "red", "sequins" -> "sequin"。
_SURFACE_TO_LABEL: Dict[str, Dict[str, str]] = {
    attr: {s.lower(): label for label, surfaces in words.items() for s in surfaces}
    for attr, words in ATTRIBUTE_WORDS.items()
}


def _check_overlaps() -> Dict[Tuple[str, str], List[str]]:
    """
    自检用: 返回跨属性重复的表层词。同一个 token 若同时命中两个属性,
    会被两路各注入一次残差, 实际权重变成 alpha_a + alpha_b, 污染消融结论。
    这里应当返回空字典。
    """
    out: Dict[Tuple[str, str], List[str]] = {}
    names = list(ATTRIBUTE_WORDS.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            sa = set(_SURFACE_TO_LABEL[a])
            sb = set(_SURFACE_TO_LABEL[b])
            dup = sorted(sa & sb)
            if dup:
                out[(a, b)] = dup
    return out


def find_spans(caption: str, attribute: str) -> List[Tuple[int, int]]:
    """返回 caption 中该属性所有表层词的字符区间 [(start, end), ...]。"""
    return [(m.start(), m.end()) for m in _PATTERNS[attribute].finditer(caption or "")]


def find_labels(caption: str, attribute: str) -> List[str]:
    """caption 里该属性命中的归一化标签集合(去重, 保持出现顺序)。"""
    seen, out = set(), []
    for m in _PATTERNS[attribute].finditer(caption or ""):
        lab = _SURFACE_TO_LABEL[attribute].get(m.group(0).lower())
        if lab and lab not in seen:
            seen.add(lab)
            out.append(lab)
    return out


# ---------------------------------------------------------------- token 对齐

def _offsets_fast(tokenizer, caption: str, max_length: int):
    enc = tokenizer(
        caption,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    return enc["offset_mapping"], enc["input_ids"]


def _offsets_slow(tokenizer, caption: str, max_length: int):
    """
    慢速 tokenizer 没有 offset mapping, 用逐 token 解码在原文里顺序推进定位。
    CLIP BPE 的 </w> 结尾标记要剥掉才能匹配回原文。
    """
    ids = tokenizer(caption, padding="max_length", truncation=True,
                    max_length=max_length).input_ids
    toks = tokenizer.convert_ids_to_tokens(ids)
    special = set(tokenizer.all_special_tokens or [])
    low = (caption or "").lower()
    offsets, cursor = [], 0
    for t in toks:
        if t in special:
            offsets.append((0, 0))
            continue
        piece = t[:-4] if t.endswith("</w>") else t
        piece = piece.lstrip("Ġ▁").lower()
        if not piece:
            offsets.append((0, 0))
            continue
        idx = low.find(piece, cursor)
        if idx < 0:
            offsets.append((0, 0))
            continue
        offsets.append((idx, idx + len(piece)))
        cursor = idx + len(piece)
    return offsets, ids


def build_attribute_masks(
    captions: Sequence[str],
    tokenizer,
    max_length: Optional[int] = None,
    device=None,
    attributes: Sequence[str] = ATTRIBUTE_NAMES,
    ground_truth_attrs: Optional[Sequence[Optional[Dict[str, List[str]]]]] = None,
) -> Dict[str, torch.Tensor]:
    """
    captions: 长度 B 的字符串列表
    ground_truth_attrs: 可选, 长度 B 的列表, 每项是 {"color": ["blue"], ...} 或 None。
                        传了就做 caption/标签交叉校验: caption 命中的词与标签不一致时,
                        该属性 mask 置空(防止属性辅助损失从错误 token 预测标签)。

    返回 {attr: BoolTensor[B, max_length]}, True 表示该 token 属于此属性。

    注意 mask 覆盖的是**子词级**位置: 一个被 BPE 切开的属性词, 其所有子词
    token 都会被标为 True。
    """
    if max_length is None:
        max_length = getattr(tokenizer, "model_max_length", 77)
    is_fast = bool(getattr(tokenizer, "is_fast", False))

    out = {a: torch.zeros(len(captions), max_length, dtype=torch.bool)
           for a in attributes}

    for bi, cap in enumerate(captions):
        cap = cap or ""
        if is_fast:
            offsets, _ = _offsets_fast(tokenizer, cap, max_length)
        else:
            offsets, _ = _offsets_slow(tokenizer, cap, max_length)

        gt = ground_truth_attrs[bi] if ground_truth_attrs else None

        for attr in attributes:
            spans = find_spans(cap, attr)
            if not spans:
                continue

            # caption/标签交叉校验 (方案 3.2)。两边用的是同一套归一化标签
            # (见 scripts/normalize_attributes.py), 所以直接取交集判断。
            # caption 说 "blue" 而标签是 ["brown"] -> 这条的 color mask 指向一个
            # 与监督标签矛盾的 token, 属性辅助损失会学成从 blue 预测 brown。
            # 置空后走 fallback(EOS), 至少不会学到错误对齐。
            if gt and gt.get(attr):
                gt_labels = {str(lb).lower() for lb in gt[attr]}
                gt_labels.discard("unknown")
                cap_labels = set(find_labels(cap, attr))
                if gt_labels and cap_labels and not (cap_labels & gt_labels):
                    continue

            m = out[attr][bi]
            for ti, (s, e) in enumerate(offsets):
                if ti >= max_length:
                    break
                if e <= s:          # special token / padding
                    continue
                for (cs, ce) in spans:
                    if s < ce and cs < e:   # 字符区间相交即命中
                        m[ti] = True
                        break

    if device is not None:
        out = {k: v.to(device) for k, v in out.items()}
    return out


def masks_from_batch_captions(batch, tokenizer, device=None):
    """训练循环里的便捷入口: batch 需带 'caption' 字段(字符串列表)。"""
    caps = batch.get("caption")
    if caps is None:
        return None
    if isinstance(caps, str):
        caps = [caps]
    return build_attribute_masks(caps, tokenizer, device=device)


# ---------------------------------------------------------------- 自检

def _self_test():
    from transformers import CLIPTokenizer
    import os

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "stable-diffusion-v1-5")
    tok = CLIPTokenizer.from_pretrained(path, subfolder="tokenizer")
    print("tokenizer is_fast =", getattr(tok, "is_fast", False))

    cases = [
        ("a blue denim jacket with striped lining", {
            "color": ["blue"], "material": ["denim"], "pattern": ["striped"]}),
        ("an off-white lace dress with floral embroidery", {
            "color": ["off-white"], "material": ["lace"],
            "pattern": ["floral", "embroidery"]}),
        ("a burgundy corduroy skirt", {
            "color": ["burgundy"], "material": ["corduroy"], "pattern": []}),
        ("black leather biker jacket with polka dot scarf", {
            "color": ["black"], "material": ["leather"], "pattern": ["polka dot"]}),
        ("a plain garment", {"color": [], "material": [], "pattern": []}),
    ]

    ok = True
    for cap, expect in cases:
        masks = build_attribute_masks([cap], tok)
        ids = tok(cap, padding="max_length", truncation=True,
                  max_length=tok.model_max_length).input_ids
        print("\ncaption: %s" % cap)
        for attr in ATTRIBUTE_NAMES:
            m = masks[attr][0]
            sel = [tok.convert_ids_to_tokens([ids[i]])[0]
                   for i in range(len(ids)) if m[i]]
            # 还原成可读字符串再比对。BPE 会把 "off-white" 切成 off/-/white,
            # 拼回来是 "off - white", 所以两边都去掉空格和连字符再比子串。
            joined = "".join(s.replace("</w>", " ") for s in sel).strip()
            norm = re.sub(r"[\s\-]+", "", joined.lower())
            exp = expect[attr]
            hit = all(re.sub(r"[\s\-]+", "", e.lower()) in norm for e in exp) \
                  if exp else (len(sel) == 0)
            ok = ok and hit
            print("  %-8s n=%d  %-38s expect=%s  %s"
                  % (attr, int(m.sum()), joined or "(none)", exp,
                     "OK" if hit else "FAIL"))

    # 边界: 空 caption 和超长 caption 不应崩
    build_attribute_masks(["", "blue " * 200], tok)
    print("\n空/超长 caption: 未崩溃 OK")

    # 跨属性词去重: 同一个词落在两个属性里会被两路重复注入残差
    dup = _check_overlaps()
    print("\n跨属性重复词: %s" % (dup if dup else "无 OK"))
    ok = ok and not dup

    # 去重后的定向检查
    dedup_cases = [
        # sequins 只归 material, 不再同时进 pattern
        ("a dress with sequins",  {"material": True,  "pattern": False}),
        # ombre 只归 pattern(分布方式), 不进 color
        ("an ombre silk jacket",  {"color": False, "material": True, "pattern": True}),
        # colorful 是颜色描述留在 color; gradient 是分布方式归 pattern。
        # 两路各命中各自的词, 但命中的不是同一个 token, 不构成重复注入。
        ("a colorful gradient top", {"color": True, "pattern": True}),
    ]
    for cap, exp in dedup_cases:
        masks = build_attribute_masks([cap], tok)
        got = {a: bool(masks[a][0].any()) for a in ATTRIBUTE_NAMES}
        hit = all(got[a] == v for a, v in exp.items())
        ok = ok and hit
        print("  %-26s %s  expect=%s  %s"
              % (cap, got, exp, "OK" if hit else "FAIL"))

    # caption/标签交叉校验: caption 说 blue, 标签是 brown -> color mask 应置空
    cap = "a blue denim jacket"
    m_ok = build_attribute_masks([cap], tok,
                                 ground_truth_attrs=[{"color": ["blue"]}])
    m_bad = build_attribute_masks([cap], tok,
                                  ground_truth_attrs=[{"color": ["brown"]}])
    m_unk = build_attribute_masks([cap], tok,
                                  ground_truth_attrs=[{"color": ["unknown"]}])
    xc = (bool(m_ok["color"][0].any())
          and not bool(m_bad["color"][0].any())
          and bool(m_unk["color"][0].any())
          and bool(m_bad["material"][0].any()))   # 只影响冲突的那个属性
    ok = ok and xc
    print("\ncaption/标签交叉校验: 一致=%s 冲突=%s unknown不拦=%s material不受影响=%s  %s"
          % (bool(m_ok["color"][0].any()), bool(m_bad["color"][0].any()),
             bool(m_unk["color"][0].any()), bool(m_bad["material"][0].any()),
             "OK" if xc else "FAIL"))

    print("\n总体: %s" % ("通过" if ok else "存在失败项"))
    return 0 if ok else 1


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    sys.exit(_self_test() if a.self_test else 0)
