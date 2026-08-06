#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 AA-TCR Fuse 在"出厂设置"下能否真的学起来 —— 跑真实的优化器循环, 看梯度。

背景: 模块残差是 alpha * to_out(...)。两者同时零初始化会互锁,
    d(loss)/d(alpha) ∝ to_out(...) = 0,  d(loss)/d(to_out) ∝ alpha = 0,
梯度恒为 0, 模块永远学不动。模块自检里的梯度检查是先手动把 alpha 设成 1、
把 to_out 随机化之后才做的, 模拟的是"训练若干步之后", 抓不到这个问题。

现在出厂设置是方案 A: alpha=0(保住"初始严格退化为 E5"的硬约束) +
to_out 小随机初始化。本脚本第一条即测该配置, 并保留"双零"对照证明死锁真实存在。

判据: 出厂配置梯度范数 > 0, 且双零对照恒为 0。两者都满足才 exit 0。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from models.attribute_text_texture_fuser import AttributeTextTextureFuser, ATTRIBUTE_NAMES


def run(alpha_init, zero_out, steps=20, lr=5e-5, tag=""):
    torch.manual_seed(0)
    B, Nt, L, D = 4, 16, 77, 768
    tex = torch.randn(B, Nt, D)
    txt = torch.randn(B, L, D)
    target = torch.randn(B, Nt, D)
    ids = torch.zeros(B, L, dtype=torch.long)
    for bi, ln in enumerate([6, 9, 5, 7]):
        ids[bi, ln] = 49407

    masks = {a: torch.zeros(B, L, dtype=torch.bool) for a in ATTRIBUTE_NAMES}
    masks["color"][:, 2] = True
    masks["material"][0, 5] = True
    masks["pattern"][1, 9] = True      # 只有 1/4 样本有 pattern 词

    f = AttributeTextTextureFuser(hidden_dim=D, alpha_init=alpha_init)
    # to_out 现在出厂就是小随机初始化(方案 A)。要复现死锁必须显式置零,
    # 不能再靠构造函数 —— 否则这个脚本会因为"改好了"而静默失去回归价值。
    if zero_out:
        with torch.no_grad():
            for a in ATTRIBUTE_NAMES:
                torch.nn.init.zeros_(f.blocks[a].to_out.weight)

    before = {n: p.detach().clone() for n, p in f.named_parameters()}
    opt = torch.optim.AdamW(f.parameters(), lr=lr)

    gnorms = []
    for _ in range(steps):
        opt.zero_grad()
        out = f(tex, txt, masks, input_ids=ids)
        loss = torch.nn.functional.mse_loss(out, target)
        loss.backward()
        g = torch.nn.utils.clip_grad_norm_(f.parameters(), 1.0)
        gnorms.append(float(g))
        opt.step()

    moved = {n: float((p.detach() - before[n]).abs().max())
             for n, p in f.named_parameters()}
    n_moved = sum(1 for v in moved.values() if v > 1e-12)
    max_g = max(gnorms)

    print(f"\n--- {tag} (alpha_init={alpha_init}, to_out {'零' if zero_out else '随机'}初始化) ---")
    print(f"  grad_norm: 首步={gnorms[0]:.3e}  末步={gnorms[-1]:.3e}  最大={max_g:.3e}")
    # 注意: AdamW 的解耦权重衰减在梯度为 0 时照样会改动权重, 所以"参数变了"
    # 不能证明在学习。真正的判据是梯度范数, 以及 alpha 有没有离开 0。
    print(f"  参数发生变化的张量数: {n_moved}/{len(moved)} (含 weight decay 的漂移, 不作判据)")
    for a in ATTRIBUTE_NAMES:
        print(f"  alpha_{a}: {float(before[f'alpha.{a}']):.6f} -> "
              f"{float(dict(f.named_parameters())[f'alpha.{a}']):.6f}")
    return max_g, gnorms


print("=" * 66)
print("问题: alpha 和 to_out 双重零初始化时, 模块能学起来吗?")
print("=" * 66)

# 第一条测的是 E7a 真正会跑的出厂配置(方案 A: alpha=0 + to_out 小随机)。
# 后三条是对照, 其中"双零"一路要靠脚本显式置零才能复现。
a_g, _ = run(0.0, False, tag="E7a 出厂配置(方案 A)")
d_g, _ = run(0.0, True,  tag="对照: 双重零初始化(旧配置)")
c_g, _ = run(1.0, True,  tag="对照: 只 to_out 零初始化")

print("\n" + "=" * 66)
print("结论 (判据: 梯度范数是否恒为 0)")
print("=" * 66)
if a_g == 0.0:
    print("❌ E7a 出厂配置: 梯度范数全程恒为 0, 模块处于死锁, 学不到任何东西。")
    print("   残差 = alpha * to_out(...)  两者都是 0 时:")
    print("       d(loss)/d(alpha)  ∝ to_out(...) = 0")
    print("       d(loss)/d(to_out) ∝ alpha       = 0")
    print("   互为对方的梯度来源, 双零即互锁, 永远出不来。")
else:
    print(f"✅ E7a 出厂配置: 最大梯度范数 {a_g:.3e}, 模块能学")
print(f"   双重零初始化(旧): 最大梯度范数 {d_g:.3e}  <- 恒 0 即为死锁")
print(f"   只 to_out 零初始化: 最大梯度范数 {c_g:.3e}")

sys.exit(0 if (a_g > 0.0 and d_g == 0.0) else 1)
