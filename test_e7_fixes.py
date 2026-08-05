#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E7 修复验证脚本
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import CLIPTokenizer
from models.attribute_token_mask import build_attribute_masks, ATTRIBUTE_NAMES
from models.attribute_text_texture_fuser import AttributeTextTextureFuser

def test_fallback():
    """验证 pattern 空 mask 的 fallback 机制"""
    print("=" * 60)
    print("测试 1: Pattern 空 mask fallback")
    print("=" * 60)

    model_path = os.path.join(os.path.dirname(__file__), "models/stable-diffusion-v1-5")
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")

    # 这些 caption 都不包含 pattern 词
    captions = [
        "a blue jacket",
        "red dress with long sleeves",
        "black pants",
        "green shirt",
    ]

    masks = build_attribute_masks(captions, tokenizer)

    print(f"\n输入 {len(captions)} 条 caption, 都不含 pattern 词")
    print(f"Color mask 覆盖率: {masks['color'].any(dim=-1).float().mean():.1%}")
    print(f"Material mask 覆盖率: {masks['material'].any(dim=-1).float().mean():.1%}")
    print(f"Pattern mask 覆盖率: {masks['pattern'].any(dim=-1).float().mean():.1%}")

    B = len(captions)
    torch.manual_seed(0)
    text_embeds = torch.randn(B, 77, 768)
    tex_tokens = torch.randn(B, 16, 768)
    input_ids = tokenizer(
        captions, padding="max_length", truncation=True,
        max_length=77, return_tensors="pt"
    ).input_ids

    def make(fallback):
        """
        alpha 和 to_out 都零初始化时残差恒为 0, 任何分支的 alpha 梯度都是 0 ——
        那是零初始化的性质, 不是稀疏问题。要观察 fallback 的效果必须先让模块
        脱离零点, 所以这里放开 alpha 并随机化 to_out, 模拟训练几步之后的状态。
        """
        torch.manual_seed(1)
        f = AttributeTextTextureFuser(hidden_dim=768,
                                      empty_mask_fallback=fallback)
        with torch.no_grad():
            for a in ATTRIBUTE_NAMES:
                f.alpha[a].fill_(1.0)
                torch.nn.init.normal_(f.blocks[a].to_out.weight, std=0.02)
        return f

    results = {}
    for name, fb in (("无 fallback", False), ("有 fallback", True)):
        f = make(fb)
        out = f(tex_tokens, text_embeds, masks, input_ids=input_ids)
        f.zero_grad()
        out.pow(2).mean().backward()
        # to_out 的梯度才是"pattern 这条路是否在学"的直接证据
        w_grad = f.blocks["pattern"].to_out.weight.grad
        results[name] = {
            "res_pattern": f.last_stats.get("res_pattern", 0.0),
            "alpha_grad": abs(f.alpha["pattern"].grad.item()),
            "wgrad": float(w_grad.abs().sum()) if w_grad is not None else 0.0,
        }

    for name, r in results.items():
        print(f"\n{name}:")
        print(f"  res_pattern           = {r['res_pattern']:.4f}")
        print(f"  |alpha_pattern.grad|  = {r['alpha_grad']:.3e}")
        print(f"  sum|to_out.grad|      = {r['wgrad']:.3e}")

    no_fb, fb = results["无 fallback"], results["有 fallback"]
    # pattern mask 全空 -> 不开 fallback 时这条路一点梯度都没有
    ok = (no_fb["alpha_grad"] < 1e-12 and no_fb["wgrad"] < 1e-12
          and fb["alpha_grad"] > 1e-9 and fb["wgrad"] > 1e-9)
    print("\n" + ("✅ 通过: mask 全空时, 无 fallback 则 pattern 完全无梯度, "
                  "开 fallback 后梯度非 0" if ok else
                  "❌ 失败: fallback 未能给 pattern 分支带来梯度"))
    return ok


def test_word_overlap():
    """验证跨属性词去重"""
    print("\n" + "=" * 60)
    print("测试 2: 跨属性词去重")
    print("=" * 60)

    model_path = os.path.join(os.path.dirname(__file__), "models/stable-diffusion-v1-5")
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")

    # sequins 以前既在 material 又在 pattern, 现在只留在 material
    # ombre/gradient 以前在 color, 现在只在 pattern
    test_cases = [
        ("a dress with sequins", {"material": True, "pattern": False}),
        ("an ombre jacket", {"color": False, "pattern": True}),
        ("a gradient skirt", {"color": False, "pattern": True}),
    ]

    all_ok = True
    for caption, expect in test_cases:
        masks = build_attribute_masks([caption], tokenizer)
        has_material = bool(masks["material"][0].any())
        has_color = bool(masks["color"][0].any())
        has_pattern = bool(masks["pattern"][0].any())

        print(f"\nCaption: '{caption}'")
        print(f"  Material mask: {has_material}, Color mask: {has_color}, Pattern mask: {has_pattern}")

        ok = True
        for attr, expected in expect.items():
            actual = {"material": has_material, "color": has_color, "pattern": has_pattern}[attr]
            if actual != expected:
                print(f"  ❌ {attr} 期望 {expected}, 实际 {actual}")
                ok = False
                all_ok = False

        if ok:
            print("  ✅ 通过")

    return all_ok


def test_collate_fn():
    """验证 collate_fn 能处理字符串字段"""
    print("\n" + "=" * 60)
    print("测试 3: collate_fn 处理字符串")
    print("=" * 60)

    sys.path.insert(0, os.path.dirname(__file__))
    from train_GAM_texture_joint import collate_fn

    batch = [
        {"tensor": torch.randn(3, 4), "caption": "blue dress", "index": torch.tensor(0)},
        {"tensor": torch.randn(3, 4), "caption": "red shirt", "index": torch.tensor(1)},
    ]

    try:
        result = collate_fn(batch)

        # 验证 tensor 字段被 stack 了
        assert result["tensor"].shape == (2, 3, 4), f"tensor shape 错误: {result['tensor'].shape}"
        assert result["index"].shape == (2,), f"index shape 错误: {result['index'].shape}"

        # 验证字符串字段保持为 list
        assert isinstance(result["caption"], list), f"caption 应该是 list, 实际是 {type(result['caption'])}"
        assert result["caption"] == ["blue dress", "red shirt"], f"caption 内容错误: {result['caption']}"

        print("  ✅ 通过: tensor 字段正确 stack, 字符串字段保持为 list")
        return True
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        return False


if __name__ == "__main__":
    results = []

    results.append(("Pattern fallback", test_fallback()))
    results.append(("跨属性词去重", test_word_overlap()))
    results.append(("collate_fn 字符串处理", test_collate_fn()))

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    for name, ok in results:
        status = "✅ 通过" if ok else "❌ 失败"
        print(f"{name}: {status}")

    all_pass = all(ok for _, ok in results)
    print("\n整体: " + ("✅ 全部通过" if all_pass else "❌ 存在失败"))
    sys.exit(0 if all_pass else 1)
