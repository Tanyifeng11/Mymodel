#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AA-TCR Fuse: 属性感知的文本-纹理融合模块 (方案第 4 节)

放在 TCPM-Lite (E5) 之后、与文本拼接之前:

    tex_tokens = bf(...)                        # [B, 16, 768]
    tex_tokens = tcpm_lite(tex_tokens, enc_h)   # E5, 冻结
    tex_tokens = aa_tcr(tex_tokens, enc_h, attr_masks)   # <- 本模块
    enc_h = torch.cat([enc_h, tex_tokens], dim=1)

与 E5 的区别: E5 把整句文本做全局均值池化, 一个标量门控乘上去, 无法区分
"蓝色"和"牛仔"分别该影响纹理的哪一部分。本模块对 color / material / pattern
三类属性各做一次交叉注意力, 让纹理 token 分别去查询对应的属性词。

方案第 4 节的形式:
    A = softmax(Q(T_a) K(V)^T / sqrt(d));  V_fused = V + a * A^T W_v(T_a)
这里等价地实现为"纹理 token 作 query, 属性文本 token 作 key/value",
A^T 即为 [B, 16, N_text], 数值上一致但 mask 处理更安全(见下)。

三个关键设计:

1. alpha 零初始化, 且 **只有** alpha 零初始化。训练开始时输出与 E5 逐位相同,
   保证不会一上来就破坏已有的草图结构和边界表现(方案第 12 节硬约束 4)。
   输出投影 to_out 必须是非零的小随机初始化 —— 两处同时置零会造成梯度死锁,
   模块永远学不动, 详见 _AttributeCrossAttention.__init__ 的注释。

2. 空 mask 必须显式置零。真实数据里 80% 的 caption 根本没提图案词,
   pattern mask 整行为 False; 对全 -inf 的行做 softmax 会得到 NaN 并在
   一步之内污染整个网络。这里对空行直接输出 0 而不是走 softmax。

3. 全程 fp32 计算注意力再转回原 dtype。fp16 下 -inf 掩码容易溢出。

自检:
    python -m models.attribute_text_texture_fuser --self-test
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

ATTRIBUTE_NAMES = ("color", "material", "pattern")


class _AttributeCrossAttention(nn.Module):
    """单个属性的多头交叉注意力: 纹理 token 查询该属性的文本 token。"""

    def __init__(self, dim: int, num_heads: int = 4, head_dim: Optional[int] = None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim or max(32, dim // num_heads)
        inner = self.num_heads * self.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.norm_tex = nn.LayerNorm(dim)
        self.norm_txt = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(dim, inner, bias=False)
        self.to_v = nn.Linear(dim, inner, bias=False)
        self.to_out = nn.Linear(inner, dim, bias=False)

        # to_out 必须是非零的小随机初始化, 不能置零。
        # 残差是 alpha * to_out(...)。若 alpha 与 to_out 同时为 0:
        #     d(loss)/d(alpha)  ∝ to_out(...) = 0
        #     d(loss)/d(to_out) ∝ alpha       = 0
        # 两者互为对方唯一的梯度来源, 双零即互锁, 永远出不来(见
        # test_e7a_deadlock.py)。初始残差恒为 0 这一硬约束由外层 alpha=0
        # 单独保证即可, 这里只需保证 to_out 有梯度可回传。
        # std=0.02 而非默认 kaiming: alpha 离 0 后残差方向虽是随机的, 幅度
        # 足够小, 不会一上来就把已有的草图结构带偏。
        nn.init.normal_(self.to_out.weight, std=0.02)

    def forward(self, tex: torch.Tensor, txt: torch.Tensor,
                mask: torch.Tensor, txt_lengths: Optional[torch.Tensor] = None,
                fallback: bool = True):
        """
        tex  [B, Nt, D]  纹理 token
        txt  [B, L,  D]  文本 token (完整 77 个)
        mask [B, L]      bool, True 表示该 token 属于当前属性
        txt_lengths [B]  可选, 每条文本的有效长度(含 BOS/EOS), 用于定位 EOS
        fallback         mask 全空时是否退回 EOS/全句; False 则残差恒 0(旧行为)
        返回 (residual [B, Nt, D], 真实命中率 frac 用于日志)
        """
        B, Nt, D = tex.shape
        L = txt.shape[1]
        H, Dh = self.num_heads, self.head_dim

        q = self.to_q(self.norm_tex(tex)).view(B, Nt, H, Dh).transpose(1, 2)
        t = self.norm_txt(txt)
        k = self.to_k(t).view(B, -1, H, Dh).transpose(1, 2)
        v = self.to_v(t).view(B, -1, H, Dh).transpose(1, 2)

        # 检测空 mask: 该样本该属性一个词都没有
        has_any = mask.any(dim=-1)                            # [B]

        # Fallback: 空 mask 退回到该句的 EOS token(没有 txt_lengths 时退回全句
        # 有效 token)。语义上成立——caption 没提这个属性, 就用整句语义去查纹理,
        # 代价是这些样本的残差不再恒为 0, 于是每个样本都能给 alpha 贡献梯度。
        # 全向量化, 不用 Python 循环, 否则 B 大时每步都要在 host 上同步。
        mask_fb = mask
        if fallback and not bool(has_any.all()):
            if txt_lengths is not None:
                eos_idx = (txt_lengths.to(mask.device).long() - 1).clamp(0, L - 1)
                fb = torch.zeros_like(mask)
                fb.scatter_(1, eos_idx[:, None], True)
            else:
                # 长度未知: 退回整行(等价于对全句文本 token 做注意力池化)
                fb = torch.ones_like(mask)
            mask_fb = torch.where(has_any[:, None], mask, fb)

        # fp32 计算注意力, 避免 fp16 下 -inf 掩码溢出
        attn = torch.matmul(q.float(), k.float().transpose(-1, -2)) * self.scale
        m = mask_fb[:, None, None, :].to(torch.bool)          # [B,1,1,L]
        attn = attn.masked_fill(~m, float("-inf"))

        # fallback 之后理论上不会再有整行 -inf, 但 txt_lengths 传错仍可能出现,
        # 留一道保险: 空行换成全 0 logit, 结果由下面的 valid 置零。
        valid = mask_fb.any(dim=-1)                           # [B]
        if not bool(valid.all()):
            attn = torch.where(valid[:, None, None, None], attn,
                               torch.zeros_like(attn))

        attn = attn.softmax(dim=-1).to(v.dtype)
        out = torch.matmul(attn, v)                           # [B,H,Nt,Dh]
        out = out.transpose(1, 2).reshape(B, Nt, H * Dh)
        out = self.to_out(out)
        out = out * valid[:, None, None].to(out.dtype)

        with torch.no_grad():
            frac = has_any.float().mean()   # 真实词命中率, 与 fallback 无关
        return out, frac


class AttributeTextTextureFuser(nn.Module):
    """
    对 color / material / pattern 各做一路交叉注意力, 结果按可学习的
    per-attribute alpha 加回纹理 token。

    alpha 全部零初始化 => 模块初始输出 == 输入, 精确退化为 E5。
    注意只有 alpha 零初始化; to_out 是小随机初始化, 否则梯度死锁。
"""

    def __init__(self,
                 hidden_dim: int = 768,
                 attributes: Sequence[str] = ATTRIBUTE_NAMES,
                 num_heads: int = 4,
                 head_dim: Optional[int] = None,
                 alpha_init: float = 0.0,
                 max_alpha: Optional[float] = None,
                 empty_mask_fallback: bool = True):
        super().__init__()
        self.attributes = tuple(attributes)
        self.blocks = nn.ModuleDict({
            a: _AttributeCrossAttention(hidden_dim, num_heads, head_dim)
            for a in self.attributes
        })
        self.alpha = nn.ParameterDict({
            a: nn.Parameter(torch.tensor(float(alpha_init)))
            for a in self.attributes
        })
        # 限幅可选。开着能防止训练中期 alpha 冲高把结构带崩, 但会引入梯度截断,
        # 默认关闭, 由 E7b 消融决定是否需要。
        self.max_alpha = max_alpha
        # 空 mask 是否退回 EOS/全句。关掉即恢复旧行为(空样本残差恒 0),
        # 留作 E7b 消融项: "去掉 fallback" 对 pattern 分支的影响。
        self.empty_mask_fallback = bool(empty_mask_fallback)
        self.last_stats: Dict[str, float] = {}

    def _alpha(self, a: str, dtype):
        v = self.alpha[a]
        if self.max_alpha is not None:
            v = torch.tanh(v / self.max_alpha) * self.max_alpha
        return v.to(dtype)

    def forward(self,
                texture_tokens: torch.Tensor,
                text_embeds: torch.Tensor,
                attribute_masks: Optional[Dict[str, torch.Tensor]] = None,
                text_lengths: Optional[torch.Tensor] = None,
                input_ids: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        texture_tokens   [B, Nt, D]   TCPM-Lite 的输出
        text_embeds      [B, L,  D]   CLIP 文本编码器输出 (L=77)
        attribute_masks  {attr: BoolTensor[B, L]}, 由
                         models.attribute_token_mask.build_attribute_masks 产出
        text_lengths     [B] 可选, 每条 caption 的有效 token 数(含 BOS/EOS)。
        input_ids        [B, L] 可选, 传了就自动推出 text_lengths(EOS 位置)。

        返回 [B, Nt, D], 形状与输入一致, 下游拼接逻辑不用改。
        """
        if texture_tokens is None or text_embeds is None or not attribute_masks:
            return texture_tokens

        # fallback 需要知道 EOS 在哪。优先用调用方给的长度; 否则从 input_ids
        # 里取 argmax——CLIP 的 EOS id 是词表里最大的那个, 且 padding 也用它,
        # 所以 argmax 取到的是第一个 EOS, 正是句末。
        lengths = text_lengths
        if lengths is None and input_ids is not None:
            lengths = input_ids.to(torch.long).argmax(dim=-1) + 1
        if lengths is not None:
            lengths = lengths.to(texture_tokens.device)

        out = texture_tokens
        stats: Dict[str, float] = {}
        for a in self.attributes:
            m = attribute_masks.get(a)
            if m is None:
                continue
            m = m.to(texture_tokens.device)
            if m.shape[-1] != text_embeds.shape[1]:
                # 长度对不上说明 tokenizer 配置和 mask 构造用的不是同一套,
                # 静默截断会导致属性错位, 宁可直接报错
                raise ValueError(
                    "attribute mask 长度 %d 与 text_embeds 长度 %d 不一致"
                    % (m.shape[-1], text_embeds.shape[1]))

            res, frac = self.blocks[a](
                out, text_embeds, m,
                txt_lengths=lengths if self.empty_mask_fallback else None,
                fallback=self.empty_mask_fallback)
            al = self._alpha(a, res.dtype)
            out = out + al * res

            with torch.no_grad():
                # alpha 分开记 raw 和 effective: 开了 max_alpha 限幅后两者会分叉,
                # 只存 raw 的话换 max_alpha 重启会让有效 alpha 悄悄跳变。
                stats["alpha_%s" % a] = float(al.detach().float().cpu())
                stats["alpha_%s_raw" % a] = float(
                    self.alpha[a].detach().float().cpu())
                stats["cover_%s" % a] = float(frac.detach().float().cpu())
                stats["res_%s" % a] = float(
                    res.detach().float().norm(dim=-1).mean().cpu())

        with torch.no_grad():
            d = (out - texture_tokens).detach().float()
            stats["delta_norm"] = float(d.norm(dim=-1).mean().cpu())
            stats["input_norm"] = float(
                texture_tokens.detach().float().norm(dim=-1).mean().cpu())
            stats["rel_change"] = stats["delta_norm"] / max(stats["input_norm"], 1e-6)
            stats["max_alpha"] = float(self.max_alpha) if self.max_alpha else 0.0
        self.last_stats = stats
        return out

    def effective_alphas(self) -> Dict[str, float]:
        """当前生效的 alpha(已过限幅), 存 checkpoint 时一并落盘。"""
        with torch.no_grad():
            return {a: float(self._alpha(a, torch.float32).cpu())
                    for a in self.attributes}

    def attribute_parameters(self):
        """方便训练脚本只给本模块设独立学习率。"""
        return list(self.parameters())


# ---------------------------------------------------------------- 自检

def _self_test():
    torch.manual_seed(0)
    B, Nt, L, D = 4, 16, 77, 768
    tex = torch.randn(B, Nt, D)
    txt = torch.randn(B, L, D)
    # 假 input_ids: EOS id 取最大值, argmax 应还原出句长
    ids = torch.zeros(B, L, dtype=torch.long)
    for bi, ln in enumerate([6, 9, 5, 7]):
        ids[bi, ln] = 49407

    masks = {a: torch.zeros(B, L, dtype=torch.bool) for a in ATTRIBUTE_NAMES}
    masks["color"][0, 2] = True
    masks["color"][1, 3:5] = True          # 多子词
    masks["material"][0, 5] = True
    masks["material"][2, 7] = True
    # 样本 3 三个属性全空; pattern 只有样本 1 有 -> 覆盖真实数据的空 mask 情形
    masks["pattern"][1, 9:11] = True

    f = AttributeTextTextureFuser(hidden_dim=D)
    ok = True

    out = f(tex, txt, masks, input_ids=ids)
    same = torch.allclose(out, tex, atol=1e-6)
    print("1. alpha=0 时精确退化为 E5      : %s (max|diff|=%.2e)"
          % ("OK" if same else "FAIL", (out - tex).abs().max()))
    ok &= same

    finite = torch.isfinite(out).all().item()
    print("2. 空 mask 不产生 NaN/Inf       : %s" % ("OK" if finite else "FAIL"))
    ok &= finite

    # 放开 alpha 后应当真的改变输出, 且仍然有限。
    # 这里只动 alpha, 不再手动重初始化 to_out —— 出厂 to_out 就是小随机值,
    # 测的必须是真正会跑起来的那个状态。
    with torch.no_grad():
        for a in ATTRIBUTE_NAMES:
            f.alpha[a].fill_(1.0)
    out2 = f(tex, txt, masks, input_ids=ids)
    finite2 = torch.isfinite(out2).all().item()
    changed = not torch.allclose(out2, tex, atol=1e-6)
    print("3. alpha>0 时输出改变且有限     : %s"
          % ("OK" if (finite2 and changed) else "FAIL"))
    ok &= finite2 and changed

    # fallback 开启后, 三属性全空的样本(索引3)也必须产生残差 —— 这正是
    # pattern 分支从 18% 样本拿梯度变成 100% 样本拿梯度的关键。
    d3 = (out2[3] - tex[3]).abs().max().item()
    print("4. 全空样本走 fallback 有残差   : %s (max|diff|=%.2e)"
          % ("OK" if d3 > 1e-6 else "FAIL", d3))
    ok &= d3 > 1e-6

    # 关掉 fallback 应恢复旧行为: 全空样本残差恰为 0
    f_nf = AttributeTextTextureFuser(hidden_dim=D, empty_mask_fallback=False)
    with torch.no_grad():
        for a in ATTRIBUTE_NAMES:
            f_nf.alpha[a].fill_(1.0)
    out_nf = f_nf(tex, txt, masks, input_ids=ids)
    d3nf = (out_nf[3] - tex[3]).abs().max().item()
    print("5. 关 fallback 时全空残差为 0   : %s (max|diff|=%.2e)"
          % ("OK" if d3nf < 1e-6 else "FAIL", d3nf))
    ok &= d3nf < 1e-6

    # 反向传播: 对比 pattern 梯度在开/关 fallback 下的差别
    out2.pow(2).mean().backward()
    out_nf.pow(2).mean().backward()
    g = f.alpha["color"].grad
    gok = g is not None and torch.isfinite(g).all().item()
    pg = f.alpha["pattern"].grad
    pg_nf = f_nf.alpha["pattern"].grad
    print("6. 梯度可回传且有限             : %s (alpha_color.grad=%.3e)"
          % ("OK" if gok else "FAIL", g.item() if g is not None else float("nan")))
    ok &= gok
    # pattern 只有 1/4 样本有真实词。开 fallback 后 4/4 样本都贡献梯度。
    pgok = pg is not None and abs(pg.item()) > 1e-9
    print("7. fallback 后 pattern 有梯度   : %s (fallback=%.3e vs 无=%.3e)"
          % ("OK" if pgok else "FAIL", pg.item() if pg is not None else float("nan"),
             pg_nf.item() if pg_nf is not None else float("nan")))
    ok &= pgok

    # fp16
    try:
        fh = AttributeTextTextureFuser(hidden_dim=D).half()
        with torch.no_grad():
            for a in ATTRIBUTE_NAMES:
                fh.alpha[a].fill_(1.0)
        oh = fh(tex.half(), txt.half(), masks, input_ids=ids)
        hok = torch.isfinite(oh).all().item()
    except Exception as e:                                   # noqa: BLE001
        hok, e_ = False, e
        print("   fp16 异常:", e_)
    print("8. fp16 前向无 NaN/Inf          : %s" % ("OK" if hok else "FAIL"))
    ok &= hok

    # 形状不变
    sok = out2.shape == tex.shape
    print("9. 输出形状与输入一致           : %s %s"
          % ("OK" if sok else "FAIL", tuple(out2.shape)))
    ok &= sok

    # mask 长度不匹配应报错而不是静默
    try:
        f(tex, txt, {"color": torch.zeros(B, 20, dtype=torch.bool)})
        rok = False
    except ValueError:
        rok = True
    print("10. mask 长度不匹配时报错       : %s" % ("OK" if rok else "FAIL"))
    ok &= rok

    # max_alpha 限幅下 raw 与 effective 必须都被记录
    fc = AttributeTextTextureFuser(hidden_dim=D, max_alpha=0.5)
    with torch.no_grad():
        fc.alpha["color"].fill_(10.0)       # 远超限幅
    fc(tex, txt, masks, input_ids=ids)
    eff = fc.effective_alphas()["color"]
    raw = fc.last_stats["alpha_color_raw"]
    cok = abs(eff) <= 0.5 + 1e-6 and abs(raw - 10.0) < 1e-6
    print("11. max_alpha raw/effective 分开记录: %s (raw=%.2f eff=%.4f)"
          % ("OK" if cok else "FAIL", raw, eff))
    ok &= cok

    # 12. 出厂状态就跑一遍真实优化器循环。这一条是补上之前测试设计的漏洞:
    # 前面几条都是先手动把 alpha 设成 1 才测梯度, 模拟的是"训练若干步之后",
    # 而 E7a 真正跑的是出厂状态。alpha 与 to_out 同时置零会互锁, 梯度恒为 0,
    # 只有这一条能抓到。判据是梯度范数, 以及 alpha 有没有离开 0 ——
    # AdamW 的解耦权重衰减在梯度为 0 时照样会挪动权重, "参数变了"不作判据。
    fg = AttributeTextTextureFuser(hidden_dim=D)     # 完全默认, 不动任何权重
    target = torch.randn(B, Nt, D)
    opt = torch.optim.AdamW(fg.parameters(), lr=1e-3)
    gnorms = []
    for _ in range(20):
        opt.zero_grad()
        loss = (fg(tex, txt, masks, input_ids=ids) - target).pow(2).mean()
        loss.backward()
        gnorms.append(float(torch.nn.utils.clip_grad_norm_(fg.parameters(), 1.0)))
        opt.step()
    alpha_moved = max(abs(float(fg.alpha[a].detach())) for a in ATTRIBUTE_NAMES)
    gok2 = max(gnorms) > 1e-8 and alpha_moved > 1e-8
    print("12. 出厂状态非死锁(可训练)      : %s (grad_norm 首=%.3e 末=%.3e, "
          "max|alpha|=%.3e)"
          % ("OK" if gok2 else "FAIL", gnorms[0], gnorms[-1], alpha_moved))
    if not gok2:
        print("    -> alpha 与 to_out 不可同时零初始化: 残差 alpha*to_out(...) 的"
              "两个因子互为对方唯一梯度来源, 双零即互锁。")
    ok &= gok2

    # 13. 出厂状态下 to_out 必须非零(上一条的直接成因, 单独报一行便于定位)
    zok = all(float(fg.blocks[a].to_out.weight.abs().max()) > 0
              for a in ATTRIBUTE_NAMES)
    print("13. 出厂 to_out 非零初始化      : %s" % ("OK" if zok else "FAIL"))
    ok &= zok

    n = sum(p.numel() for p in f.parameters())
    print("\n新增参数量: %.2fM" % (n / 1e6))
    print("last_stats: %s" % {k: round(v, 4) for k, v in f.last_stats.items()})
    print("\n总体: %s" % ("通过" if ok else "存在失败项"))
    return 0 if ok else 1


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    sys.exit(_self_test() if a.self_test else 0)
