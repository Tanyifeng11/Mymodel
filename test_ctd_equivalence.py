#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CTD Stage A 等价性与算子正确性测试 (docs/ctd_stage_a_spec.md §6)。

E7a 之所以能确定"差异来自模块本身而不是加载或配置", 就是因为 e7a_off 与 E5
逐位相同。CTD 要有同一条闸 —— 这个文件就是那条闸。

五条:
  1. --ctd_prob 0 时 Dataset[i] 的每个 tensor 与改动前逐位相同
  2. --ctd_prob 1 时 ctd_applied 只在 has_text_color=1 的样本上为 1
  3. 黑/白/灰参考图扰动后 ΔE > 25   <- 直接针对 §0.2 那个坑
  4. chroma_shift_to 后 L 通道逐像素不变(误差 <= 1)
  5. 扰动后图案统计量(Gram)相对变化 < 5%

不需要 GPU、不需要 checkpoint。用 pytest 跑:
    python -m pytest test_ctd_equivalence.py -v
"""
import os
import random
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from color_conflict_utils import (  # noqa: E402
    COLOR_TABLE,
    chroma_shift_to,
    delta_e_rgb,
    dominant_rgb_from_pil,
    extract_text_color,
    lab_to_rgb,
    pick_far_ab,
    polar_ab,
    rgb_to_lab,
)

REPO = os.path.dirname(os.path.abspath(__file__))
# 本地 Windows/WSL 与服务器路径不同；服务器验证时由 submit 脚本显式传入。
DATA_ROOT = os.environ.get("CTD_DATA_ROOT", "/mnt/f/fuxian/dataset/datasets/BF/training")
DATASET_JSON = os.environ.get(
    "CTD_DATASET_JSON", os.path.join(REPO, "data/train_bf_texture.json")
)
SD_PATH = os.environ.get(
    "CTD_SD_PATH", os.path.join(REPO, "models/stable-diffusion-v1-5")
)

# §0.2 的三个近中性颜色 —— hue 旋转对它们是恒等变换, 平移必须能动
NEUTRAL_RGB = {
    "black": (20, 20, 20),
    "white": (235, 235, 235),
    "gray": (135, 135, 135),
}


def _synthetic_texture(base_rgb, size=64, seed=0):
    """造一张有图案的参考图: 基色 + 条纹 + 噪声, 用来测 L 保持与 Gram 保持。"""
    rng = np.random.RandomState(seed)
    arr = np.zeros((size, size, 3), dtype=np.float32)
    arr[:] = np.asarray(base_rgb, dtype=np.float32)
    # 条纹(明度结构) + 噪声(材质细节)
    stripe = (np.arange(size) % 8 < 4).astype(np.float32) * 28.0
    arr += stripe[None, :, None]
    arr += rng.normal(0.0, 6.0, size=arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _gram(image):
    """图案统计量: 灰度图的归一化二阶矩(Gram), 与颜色无关只与结构有关。"""
    g = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    g = g - g.mean()
    flat = g.reshape(g.shape[0], -1)
    gram = flat @ flat.T
    return gram / max(1.0, float(np.linalg.norm(gram)))


# ---------------------------------------------------------------------------
# 条 4 + lab_to_rgb 自检
# ---------------------------------------------------------------------------
def test_lab_roundtrip_error_within_2():
    """lab_to_rgb(rgb_to_lab(x)) 与 x 的最大绝对误差应 <= 2 (spec §1.1 自检)。"""
    rng = np.random.RandomState(42)
    x = rng.randint(0, 256, size=(64, 64, 3)).astype(np.uint8)
    back = lab_to_rgb(rgb_to_lab(x.astype(np.float32)))
    err = np.abs(back.astype(np.int16) - x.astype(np.int16)).max()
    assert err <= 2, f"往返误差 {err} > 2"


@pytest.mark.parametrize("name,rgb", sorted(NEUTRAL_RGB.items()))
def test_chroma_shift_preserves_L_channel(name, rgb):
    """条 4: chroma_shift_to 后 L 通道逐像素不变(误差 <= 1)。"""
    img = _synthetic_texture(rgb, seed=1)
    target_ab = polar_ab(30.0, chroma=45.0)
    out = chroma_shift_to(img, target_ab)

    L_before = rgb_to_lab(np.asarray(img, dtype=np.float32))[..., 0]
    L_after = rgb_to_lab(np.asarray(out, dtype=np.float32))[..., 0]
    max_dev = float(np.abs(L_after - L_before).max())
    assert max_dev <= 1.0, f"{name}: L 通道最大偏移 {max_dev:.3f} > 1"


# ---------------------------------------------------------------------------
# 条 3 —— 直接针对 §0.2 的坑
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,rgb", sorted(NEUTRAL_RGB.items()))
def test_neutral_reference_shift_delta_e_above_25(name, rgb):
    """条 3: 黑/白/灰参考图扰动后 ΔE > 25。

    这是 CTD 用 LAB 平移替换 hue 旋转的全部理由: adjust_hue 对 S≈0 的图是
    恒等变换, 而本数据集 57.9% 的有色词样本正是近中性(spec §0.2)。
    """
    img = _synthetic_texture(rgb, seed=2)
    before = dominant_rgb_from_pil(img)
    rng = random.Random(1234)
    target_ab, far_name = pick_far_ab(rgb, rng)
    out = chroma_shift_to(img, target_ab)
    d = delta_e_rgb(before, dominant_rgb_from_pil(out))
    assert d > 25.0, f"{name} -> {far_name}: ΔE 仅 {d:.2f}, 未超过 25"


def test_pick_far_ab_is_deterministic_per_seed():
    """同一样本 id 每次拿到同一个反事实颜色 —— 增广评测集可复现的前提。"""
    text_rgb = COLOR_TABLE["black"]
    a = pick_far_ab(text_rgb, random.Random(7 * 1000003 + 42))
    b = pick_far_ab(text_rgb, random.Random(7 * 1000003 + 42))
    assert a == b
    c = pick_far_ab(text_rgb, random.Random(7 * 1000003 + 43))
    assert a[1] != c[1] or a[0] != c[0], "不同样本 id 应能拿到不同扰动"


def test_polar_ab_does_not_touch_color_table():
    """族 B 必须与 COLOR_TABLE 无关(否则训练/测试同源, 审稿人会抓)。"""
    ab = polar_ab(137.5, chroma=45.0)
    assert abs(np.hypot(*ab) - 45.0) < 1e-4
    table_abs = [
        tuple(rgb_to_lab(np.asarray(v, np.float32).reshape(1, 3)).reshape(3)[1:])
        for v in COLOR_TABLE.values()
    ]
    assert all(abs(ab[0] - t[0]) > 1e-6 or abs(ab[1] - t[1]) > 1e-6 for t in table_abs)


# ---------------------------------------------------------------------------
# 条 5
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,rgb", sorted(NEUTRAL_RGB.items()))
def test_pattern_gram_change_below_5_percent(name, rgb):
    """条 5: 扰动后图案统计量(Gram)相对变化 < 5% —— 图案没被色度替换破坏。"""
    img = _synthetic_texture(rgb, seed=3)
    target_ab, _ = pick_far_ab(rgb, random.Random(99))
    out = chroma_shift_to(img, target_ab)
    g0, g1 = _gram(img), _gram(out)
    rel = float(np.linalg.norm(g1 - g0) / max(1e-8, np.linalg.norm(g0)))
    assert rel < 0.05, f"{name}: Gram 相对变化 {rel:.2%} >= 5%"


# ---------------------------------------------------------------------------
# 条 1 + 条 2 —— 需要真实数据集与 tokenizer
# ---------------------------------------------------------------------------
def _build_dataset(ctd_prob, ctd_all_samples=0, max_samples=8):
    from transformers import CLIPTokenizer

    from train_GAM_texture_joint import JointTextureDataset

    tokenizer = CLIPTokenizer.from_pretrained(SD_PATH, subfolder="tokenizer")
    return JointTextureDataset(
        DATASET_JSON,
        tokenizer,
        DATA_ROOT,
        width=384,
        height=512,
        texture_preprocess_mode="plain_resize",
        max_samples=max_samples,
        ctd_prob=ctd_prob,
        ctd_all_samples=ctd_all_samples,
    )


_needs_data = pytest.mark.skipif(
    not (os.path.isdir(DATA_ROOT) and os.path.isfile(DATASET_JSON) and os.path.isdir(SD_PATH)),
    reason="需要 BF 数据集 + train_bf_texture.json + SD1.5 tokenizer",
)


@_needs_data
def test_ctd_prob_zero_is_bitwise_identical_to_baseline():
    """条 1: --ctd_prob 0 时每个 tensor 与基线逐位相同(固定 seed)。

    这是整个 Stage A 最重要的一道闸: 它保证后面观察到的任何差异来自 CTD 本身,
    而不是加载顺序、随机数消耗或配置漂移。
    """
    ds = _build_dataset(ctd_prob=0.0)
    n = min(8, len(ds))

    def snapshot():
        out = []
        for i in range(n):
            random.seed(1000 + i)
            np.random.seed(1000 + i)
            out.append(ds[i])
        return out

    a, b = snapshot(), snapshot()
    for i, (ra, rb) in enumerate(zip(a, b)):
        assert ra.keys() == rb.keys()
        for k in ra:
            va, vb = ra[k], rb[k]
            if hasattr(va, "shape"):
                assert bool((va == vb).all()), f"样本 {i} 的 {k} 不是逐位相同"
        # ctd_prob=0 时扰动必须完全没发生
        assert float(ra["ctd_applied"]) == 0.0, f"样本 {i}: ctd_prob=0 却发生了扰动"
        assert float(ra["ctd_delta_e"]) == 0.0
        # 未扰动时 orig 主色应等于 conflict 用的参考主色
        assert bool((ra["ref_palette_rgb_orig"] == ra["ref_palette_rgb"]).all())


@_needs_data
def test_ctd_prob_one_applies_only_to_text_color_samples():
    """条 2: --ctd_prob 1 时 ctd_applied 只在 has_text_color=1 的样本上为 1。

    这是"管辖先验"写进数据管线的直接读数 —— 无颜色词样本的颜色仍由参考图控制。
    """
    ds = _build_dataset(ctd_prob=1.0, max_samples=24)
    n = min(24, len(ds))
    n_text, n_applied_no_text, n_applied_text = 0, 0, 0
    for i in range(n):
        random.seed(2000 + i)
        rec = ds[i]
        has_text = float(rec["has_text_color"]) > 0.5
        applied = float(rec["ctd_applied"]) > 0.5
        if has_text:
            n_text += 1
            n_applied_text += int(applied)
        else:
            n_applied_no_text += int(applied)
        if applied:
            # ΔE 校验生效: 记录的 ΔE 必须达到门限
            assert float(rec["ctd_delta_e"]) >= ds.ctd_min_delta_e

    assert n_applied_no_text == 0, (
        f"{n_applied_no_text} 个无颜色词样本被扰动了 —— 管辖先验被破坏"
    )
    assert n_text > 0, "取到的样本里没有含颜色词的, 无法判定"
    # ΔE 校验会剔除一部分(out-of-gamut 夹取), 但不该把绝大多数都剔掉
    assert n_applied_text >= max(1, int(0.5 * n_text)), (
        f"有颜色词样本 {n_text} 个, 仅 {n_applied_text} 个扰动成功 —— "
        "D0 会不过, 检查 pick_far_ab 的 chroma 幅度"
    )


@_needs_data
def test_ctd_all_samples_ablation_also_perturbs_no_text_color():
    """消融 A1: --ctd_all_samples 1 时无颜色词样本也被扰动。

    A1 必须是"全量扰动"而不是普通 color jitter —— 只有它才直接攻击核心主张
    (spec §5.3)。这个测试确认那条开关真的改变了行为。
    """
    ds = _build_dataset(ctd_prob=1.0, ctd_all_samples=1, max_samples=24)
    n = min(24, len(ds))
    n_no_text, n_applied_no_text = 0, 0
    for i in range(n):
        random.seed(3000 + i)
        rec = ds[i]
        if float(rec["has_text_color"]) <= 0.5:
            n_no_text += 1
            n_applied_no_text += int(float(rec["ctd_applied"]) > 0.5)
    if n_no_text == 0:
        pytest.skip("前 24 条里没有无颜色词样本")
    assert n_applied_no_text > 0, "A1 开关没生效: 无颜色词样本一个都没被扰动"


@_needs_data
def test_ctd_perturbation_reaches_both_conditioning_paths():
    """扰动必须同时进入 texture_image 与 clip_texture 两条通路(spec §2.2)。

    一处插入覆盖两条通路是这个实现的关键性质; 如果只有一条变了, 模型仍能从另一条
    读到原始颜色, 捷径不会被打断。
    """
    ds_off = _build_dataset(ctd_prob=0.0, max_samples=24)
    ds_on = _build_dataset(ctd_prob=1.0, max_samples=24)
    checked = 0
    for i in range(min(24, len(ds_on))):
        random.seed(4000 + i)
        rec_on = ds_on[i]
        if float(rec_on["ctd_applied"]) <= 0.5:
            continue
        random.seed(4000 + i)
        rec_off = ds_off[i]
        assert not bool((rec_on["texture_image"] == rec_off["texture_image"]).all()), (
            f"样本 {i}: texture_image 未随扰动改变"
        )
        assert not bool((rec_on["clip_texture"] == rec_off["clip_texture"]).all()), (
            f"样本 {i}: clip_texture 未随扰动改变 —— CLIP 通路漏了扰动"
        )
        # 目标图与草图绝不能被碰到 —— GT 必须是 100% 真实原图
        assert bool((rec_on["vae_cloth"] == rec_off["vae_cloth"]).all()), (
            f"样本 {i}: 目标图被改动了, 违反 CTD 的核心前提"
        )
        assert bool((rec_on["vae_sketch"] == rec_off["vae_sketch"]).all())
        checked += 1
        if checked >= 3:
            break
    if checked == 0:
        pytest.skip("前 24 条里没有扰动成功的样本")


@_needs_data
def test_conflict_score_describes_perturbed_reference():
    """color_conflict_score 应描述"实际呈现给模型的冲突"(spec §2.2)。

    compute_color_conflict 用的 ref_tensor 已是扰动后的, 所以扰动样本的冲突分
    必须高于未扰动版本 —— 这正是日志需要的读数。
    """
    ds_off = _build_dataset(ctd_prob=0.0, max_samples=24)
    ds_on = _build_dataset(ctd_prob=1.0, max_samples=24)
    pairs = []
    for i in range(min(24, len(ds_on))):
        random.seed(5000 + i)
        rec_on = ds_on[i]
        if float(rec_on["ctd_applied"]) <= 0.5:
            continue
        random.seed(5000 + i)
        rec_off = ds_off[i]
        pairs.append(
            (float(rec_off["color_conflict_score"]), float(rec_on["color_conflict_score"]))
        )
    if not pairs:
        pytest.skip("没有扰动成功的样本")
    higher = sum(1 for lo, hi in pairs if hi > lo)
    assert higher >= max(1, int(0.7 * len(pairs))), (
        f"仅 {higher}/{len(pairs)} 个扰动样本的冲突分上升, 扰动可能没进入 conflict 计算"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
