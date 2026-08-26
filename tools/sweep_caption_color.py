#!/usr/bin/env python
"""caption 颜色响应实验 —— CTD 的立项闸门。

问题: E5 的 TxtCF 已优于 GT floor(见 docs/e0_e7a_settlement.md §0.2), 所以
"文本颜色被忽略"这个缺陷用现有指标测不出来。但那只证明**指标无法发现缺陷**,
不证明模型会**响应** caption 的颜色变化 —— 也可能它无论 caption 写什么都生成
原型化的饱和色。这两件事完全不同, 现有全部指标都区分不了。

本实验: 固定 sketch / 参考图 / seed / 采样器, 只把 caption 里的颜色词扫过
COLOR_TABLE 的若干档, 看生成图 mask 内主色跟不跟。

三种结果对应三种决策(docs/ctd_stage_a_spec.md §0.5):
  跟           -> CTD 只是强化已有的轴, 主线应转向纹理侧余量
  不跟(被钉住)  -> 这才是真缺陷, 且现有指标看不见, CTD 正对着它
  跟但图案变    -> 颜色与图案纠缠, 直接进 Stage B 频带分离

分三段跑, 只有 generate 需要 GPU:
    python tools/sweep_caption_color.py --stage plan   ...   # 无 GPU
    python tools/sweep_caption_color.py --stage generate ...  # 需要 GPU
    python tools/sweep_caption_color.py --stage score  ...   # 无 GPU
"""
import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from color_conflict_utils import (
    COLOR_TABLE,
    EN_COLOR_KEYS,
    delta_e_rgb,
    dominant_rgb_from_pil,
    rgb_to_lab,
)
from eval.eval_utils import prepare_evaluation_masks, safe_open_rgb
from garment_mask_utils import mask_backend_info

# 默认扫这 12 档(COLOR_TABLE 的英文键, 去掉 grey 与 gray 的重复)
DEFAULT_COLORS = [
    "black", "white", "gray", "red", "blue", "green",
    "yellow", "orange", "purple", "pink", "brown", "beige",
]


# ---------------------------------------------------------------------------
# caption 颜色词定位与替换
# ---------------------------------------------------------------------------
def find_color_words(caption):
    """返回 caption 里所有 COLOR_TABLE 英文颜色词的 (start, end, name)。"""
    text = caption or ""
    hits = []
    for name in EN_COLOR_KEYS:
        for m in re.finditer(r"\b%s\b" % re.escape(name), text, flags=re.IGNORECASE):
            hits.append((m.start(), m.end(), name))
    hits.sort()
    kept = []
    for h in hits:
        if kept and h[0] < kept[-1][1]:
            continue  # 被更靠前的匹配覆盖
        kept.append(h)
    return kept


def replace_color_word(caption, new_name):
    """把 caption 里唯一的颜色词替换成 new_name; 不唯一则返回 None。

    只接受"恰好一个颜色词"的样本 —— 多个颜色词会让"改了哪一个"变成混淆变量。
    """
    hits = find_color_words(caption)
    if len(hits) != 1:
        return None
    start, end, _ = hits[0]
    return caption[:start] + new_name + caption[end:]


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------
def stage_preflight(args):
    """检查 mask 后端与输入文件, 不合格就直接失败, 不浪费 GPU 时间。

    mask 后端是这里最要紧的一项: garment_mask_utils 在 cv2 缺失时会静默退回
    Pillow 形态学, 两条路径产出的 mask 不同(实测 40 样本里 35 个像素数不同、
    24 个 mask_source 翻转)。既有全部报告都跑在 Pillow 路径上, 从未被记录过。
    """
    info = mask_backend_info()
    print("[preflight] mask_backend = %s" % info["mask_backend"])
    print("[preflight] cv2_version  = %s" % info["cv2_version"])
    if info["cv2_import_error"]:
        print("[preflight] cv2_import_error = %s" % info["cv2_import_error"])

    ok = True
    want = args.require_mask_backend
    if want != "any" and info["mask_backend"] != want:
        ok = False
        print()
        print("[preflight][FAIL] 要求 mask_backend=%s, 实际 %s" % (want, info["mask_backend"]))
        if info["mask_backend"] == "pillow_fallback":
            print("  cv2 导入失败。常见原因: opencv-python 非 headless 版依赖 libGL,")
            print("  在无图形栈的计算节点上 `import cv2` 会抛 ImportError。")
            print("  修法: pip install opencv-python-headless  (并卸掉 opencv-python)")
        print("  确认要在当前后端下继续: 设 REQUIRE_MASK_BACKEND=any")

    for label, path in (
        ("per_sample_csv", args.per_sample_csv),
        ("data_root/sketch", os.path.join(args.data_root, "sketch")),
        ("data_root/texture", os.path.join(args.data_root, "texture")),
        ("data_root/cloth", os.path.join(args.data_root, "cloth")),
    ):
        exists = os.path.exists(path)
        print("[preflight] %-18s %s  %s" % (label, "OK  " if exists else "MISS", path))
        ok = ok and exists

    if args.stage in ("generate", "all") or args.check_ckpt:
        for label, path in (("gam_ckpt", args.gam_ckpt), ("texture_ckpt", args.texture_ckpt)):
            exists = bool(path) and os.path.exists(path)
            print("[preflight] %-18s %s  %s" % (label, "OK  " if exists else "MISS", path))
            ok = ok and exists

    print("[preflight] %s" % ("通过" if ok else "未通过"))
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------
def stage_plan(args):
    with open(args.per_sample_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    colors = [c.strip() for c in args.colors.split(",") if c.strip()]
    for c in colors:
        if c not in COLOR_TABLE:
            raise SystemExit("颜色 %r 不在 COLOR_TABLE 里" % c)

    items, rejected = [], {"no_caption": 0, "not_exactly_one_color": 0, "missing_files": 0}
    for row in rows:
        if len(items) >= args.num_samples * len(colors):
            break
        caption = row.get("prompt") or row.get("caption") or ""
        if not caption:
            rejected["no_caption"] += 1
            continue
        hits = find_color_words(caption)
        if len(hits) != 1:
            rejected["not_exactly_one_color"] += 1
            continue

        stem = os.path.splitext(os.path.basename(row.get("sketch_path") or ""))[0]
        paths = {
            "sketch": os.path.join(args.data_root, "sketch", stem + ".jpg"),
            "texture": os.path.join(args.data_root, "texture", stem + ".jpg"),
            "target": os.path.join(args.data_root, "cloth", stem + ".jpg"),
        }
        if not all(os.path.isfile(p) for p in paths.values()):
            rejected["missing_files"] += 1
            continue

        base_id = row.get("sample_id") or stem
        # seed 只由 base 样本决定 —— 同一 base 的所有颜色变体共用一个 seed,
        # 否则颜色效应会和噪声效应混在一起, 整个实验作废。
        seed = int(args.generation_seed) + int(re.sub(r"\D", "", base_id) or 0)
        original_name = hits[0][2].lower()
        original_rgb = COLOR_TABLE.get(original_name)
        for color in colors:
            prompt = replace_color_word(caption, color)
            if prompt is None:
                continue
            # 按 RGB 判定"是否原始颜色", 不按名字 —— grey/gray 是同义词
            # (COLOR_TABLE 里两者都映射到 (135,135,135)), 按名字比会漏判。
            is_original = (
                original_rgb is not None
                and tuple(COLOR_TABLE[color]) == tuple(original_rgb)
            )
            items.append(
                {
                    "base_id": base_id,
                    "dataset_id": stem,
                    "color": color,
                    "requested_rgb": list(COLOR_TABLE[color]),
                    "original_color": original_name,
                    "is_original": is_original,
                    "prompt": prompt,
                    "original_prompt": caption,
                    "seed": seed,
                    "sketch_path": paths["sketch"],
                    "texture_path": paths["texture"],
                    "target_path": paths["target"],
                    "out_dir": os.path.join(args.out_dir, "gen", base_id, color),
                }
            )

    n_base = len({it["base_id"] for it in items})
    plan = {
        "config": {k: v for k, v in vars(args).items()},
        "colors": colors,
        "num_base_samples": n_base,
        "num_items": len(items),
        "rejected": rejected,
        "mask_backend": mask_backend_info(),
        "items": items,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "plan.json"), "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    print("[plan] base 样本 = %d, 颜色档 = %d, 待生成 = %d 张"
          % (n_base, len(colors), len(items)))
    print("[plan] 剔除: %s" % rejected)
    print("[plan] 每个 base 的 %d 个变体共用同一 seed(颜色是唯一变量)" % len(colors))
    print("[plan] -> %s" % os.path.join(args.out_dir, "plan.json"))
    if items:
        s = items[0]
        print("\n示例:")
        print("  原 caption : %s" % s["original_prompt"])
        print("  改写后     : %s" % s["prompt"])
        print("  seed       : %d" % s["seed"])
    return 0


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------
def _inference_cmd(args, item, flags, experiment_flags):
    """与 tools/run_fixed_benchmark.py 逐 flag 对齐, 只改 prompt / seed / output。

    重要: run_fixed_benchmark **不传** --width/--height/--guidance_scale/--ipa_scale/
    --num_inference_steps/--sketch_scale/--texture_mode/--texture_num_tokens, 所以
    既有 E5/E7a 报告全部跑在 inference 的默认生成参数上(guidance=7.0, ipa=1.0,
    sketch_scale=0.6, steps=50; width/height 从 GAM checkpoint metadata 读)。
    这里也一律不传 —— 传了扫描结果就与报告不在同一生成条件下, 判定无法迁移。
    需要偏离时用 --override_generation_params 显式开启, 并接受不可比。
    """
    cmd = [
        sys.executable, os.path.join(REPO, "inference_IMAGGarment-1.py"),
        "--GAM_model_ckpt", args.gam_ckpt,
        "--texture_ckpt", args.texture_ckpt,
        "--sketch_path", item["sketch_path"],
        "--texture_path", item["texture_path"],
        "--prompt", item["prompt"],
        "--output_path", item["out_dir"],
        "--device", args.device,
        "--seed", str(item["seed"]),
        "--texture_condition_mode",
        experiment_flags.get("texture_condition_mode", flags["texture_condition_mode"]),
        "--fusion_type", flags["fusion_type"],
        "--layer_group_enabled", str(int(experiment_flags["layer_group_enabled"])),
        "--texture_preprocess_mode", args.texture_preprocess_mode,
        "--use_texture_gate", str(int(experiment_flags["use_texture_gate"])),
        "--use_palette_tokens", str(int(experiment_flags["use_palette_tokens"])),
        "--num_palette_tokens", str(int(experiment_flags["num_palette_tokens"])),
        "--gate_type", experiment_flags["gate_type"],
        "--gate_init", experiment_flags["gate_init"],
        "--gate_min", str(experiment_flags["gate_min"]),
        "--gate_max", str(experiment_flags["gate_max"]),
        "--use_balanced_fusion_gate",
        str(int(experiment_flags["use_balanced_fusion_gate"])),
        "--balanced_gate_hidden_dim",
        str(int(experiment_flags["balanced_gate_hidden_dim"])),
        "--balanced_gate_scale", str(experiment_flags["balanced_gate_scale"]),
        "--balanced_gate_min", str(experiment_flags["balanced_gate_min"]),
        "--balanced_gate_max", str(experiment_flags["balanced_gate_max"]),
        "--use_conflict_aware_gate",
        str(int(experiment_flags["use_conflict_aware_gate"])),
        "--use_tcpm_lite", str(int(experiment_flags["use_tcpm_lite"])),
        "--use_aa_tcr_fuse", str(int(experiment_flags["use_aa_tcr_fuse"])),
        "--conflict_texture_suppress_strength",
        str(args.conflict_texture_suppress_strength),
        "--conflict_palette_suppress_strength",
        str(args.conflict_palette_suppress_strength),
        "--conflict_deltae_norm", str(args.conflict_deltae_norm),
        "--conflict_threshold", str(args.conflict_threshold),
        "--alpha1", str(args.alpha1),
        "--alpha2", str(args.alpha2),
        "--alpha3", str(args.alpha3),
        "--alpha4", str(args.alpha4),
    ]
    if args.override_generation_params:
        cmd += [
            "--width", str(args.width),
            "--height", str(args.height),
            "--guidance_scale", str(args.guidance_scale),
            "--ipa_scale", str(args.ipa_scale),
        ]
    return cmd


def stage_generate(args):
    from tools.run_fixed_benchmark import experiment_to_flags, mode_to_flags

    with open(os.path.join(args.out_dir, "plan.json"), encoding="utf-8") as f:
        plan = json.load(f)

    flags = mode_to_flags(args.mode)
    experiment_flags = experiment_to_flags(args.run_name, args)

    done = failed = skipped = 0
    for i, item in enumerate(plan["items"], 1):
        os.makedirs(item["out_dir"], exist_ok=True)
        existing = [p for p in os.listdir(item["out_dir"]) if p.endswith(".png")]
        if existing and not args.overwrite:
            skipped += 1
            continue
        cmd = _inference_cmd(args, item, flags, experiment_flags)
        if args.dry_run:
            if i == 1:
                print("[dry-run] 首条命令:\n  %s" % " ".join(cmd))
            continue
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            failed += 1
            print("[fail] %s/%s: %s" % (item["base_id"], item["color"],
                                        proc.stderr.strip()[-300:]))
        else:
            done += 1
        if i % 25 == 0:
            print("[gen] %d/%d  done=%d skip=%d fail=%d"
                  % (i, len(plan["items"]), done, skipped, failed))

    print("[gen] 完成: done=%d skipped=%d failed=%d" % (done, skipped, failed))
    return 1 if failed and not done else 0


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------
def _find_generated(out_dir):
    if not os.path.isdir(out_dir):
        return None
    cands = [p for p in sorted(os.listdir(out_dir)) if p.lower().endswith(".png")]
    if not cands:
        return None
    for p in cands:
        if "generated" in p.lower():
            return os.path.join(out_dir, p)
    return os.path.join(out_dir, cands[0])


def stage_score(args):
    with open(os.path.join(args.out_dir, "plan.json"), encoding="utf-8") as f:
        plan = json.load(f)

    try:
        from eval.metrics import compute_texture_pattern_fidelity
    except Exception:  # noqa: BLE001
        compute_texture_pattern_fidelity = None

    rows, missing = [], 0
    for item in plan["items"]:
        gen_path = _find_generated(item["out_dir"])
        if gen_path is None:
            missing += 1
            continue
        gen = safe_open_rgb(gen_path)
        if gen is None:
            missing += 1
            continue

        bundle = prepare_evaluation_masks(
            gen.size,
            sketch_path=item["sketch_path"],
            target_path=item["target_path"],
            gen_path=gen_path,
            mask_policy=args.mask_policy,
        )
        mask = bundle.get("garment")
        if mask is None or int(mask.sum()) < 50:
            missing += 1
            continue
        mask_img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")

        gen_dom = dominant_rgb_from_pil(gen, mask=mask_img)
        texture = safe_open_rgb(item["texture_path"])
        ref_dom = dominant_rgb_from_pil(texture) if texture is not None else (0, 0, 0)

        rec = {
            "base_id": item["base_id"],
            "dataset_id": item["dataset_id"],
            "color": item["color"],
            "is_original": item["is_original"],
            "seed": item["seed"],
            "gen_dominant_rgb": list(gen_dom),
            "requested_rgb": item["requested_rgb"],
            "reference_rgb": list(ref_dom),
            "delta_to_requested": delta_e_rgb(item["requested_rgb"], gen_dom),
            "delta_to_reference": delta_e_rgb(ref_dom, gen_dom),
        }
        rec["follows_text"] = int(rec["delta_to_requested"] < rec["delta_to_reference"])

        if compute_texture_pattern_fidelity is not None:
            try:
                tpf = compute_texture_pattern_fidelity(
                    gen_path, item["texture_path"], mask_bundle=bundle
                )
                rec["tpf_patch_sim"] = tpf.get("tpf_patch_sim")
            except Exception:  # noqa: BLE001 - VGG 权重缺失等, 不阻塞主判定
                rec["tpf_patch_sim"] = float("nan")
        rows.append(rec)

    if not rows:
        print("[score] 没有可评分的生成图(missing=%d)。先跑 --stage generate。" % missing)
        return 1

    # ---- 按 base 样本聚合 ----
    by_base = {}
    for r in rows:
        by_base.setdefault(r["base_id"], []).append(r)

    per_base = []
    for base_id, group in sorted(by_base.items()):
        labs = np.array([rgb_to_lab(np.array(r["gen_dominant_rgb"], dtype=np.float32)
                                   .reshape(1, 3)).reshape(3) for r in group])
        # response_range: 12 个输出主色之间的最大两两 LAB 距离。
        # 它是"模型对 caption 颜色变化的响应幅度", 也是本实验的核心量。
        if len(labs) > 1:
            d = np.linalg.norm(labs[:, None, :] - labs[None, :, :], axis=-1)
            response_range = float(d.max())
        else:
            response_range = 0.0
        tpfs = [r.get("tpf_patch_sim") for r in group
                if r.get("tpf_patch_sim") is not None
                and math.isfinite(float(r.get("tpf_patch_sim")))]
        per_base.append({
            "base_id": base_id,
            "n_colors": len(group),
            "response_range": response_range,
            "follow_rate": float(np.mean([r["follows_text"] for r in group])),
            "mean_delta_to_requested": float(np.mean([r["delta_to_requested"] for r in group])),
            "mean_delta_to_reference": float(np.mean([r["delta_to_reference"] for r in group])),
            "tpf_range": float(max(tpfs) - min(tpfs)) if len(tpfs) > 1 else float("nan"),
            "tpf_mean": float(np.mean(tpfs)) if tpfs else float("nan"),
        })

    rr = np.array([b["response_range"] for b in per_base])
    fr = np.array([b["follow_rate"] for b in per_base])
    tr = np.array([b["tpf_range"] for b in per_base], dtype=float)
    tr_ok = tr[np.isfinite(tr)]

    summary = {
        "n_base": len(per_base),
        "n_items": len(rows),
        "missing": missing,
        "mask_backend": mask_backend_info(),
        "mask_policy": args.mask_policy,
        "response_range_mean": float(rr.mean()),
        "response_range_median": float(np.median(rr)),
        "response_range_p10": float(np.percentile(rr, 10)),
        "follow_rate_mean": float(fr.mean()),
        "tpf_range_median": float(np.median(tr_ok)) if tr_ok.size else None,
        "tpf_mean": float(np.nanmean([b["tpf_mean"] for b in per_base])),
    }

    # ---- 判定 ----
    hi, lo = args.response_high, args.response_low
    med = summary["response_range_median"]
    if med < lo:
        verdict = "PINNED"
        detail = ("输出主色几乎不随 caption 变(中位响应 %.1f < %.1f) -> 被参考图钉住。"
                  "这是真缺陷且现有指标看不见, CTD 正对着它, 按规格开工。" % (med, lo))
    elif med >= hi and summary["follow_rate_mean"] >= 0.5:
        if tr_ok.size and np.median(tr_ok) > args.tpf_range_max:
            verdict = "FOLLOWS_BUT_PATTERN_DRIFTS"
            detail = ("颜色跟随(中位响应 %.1f), 但图案同时漂移(tpf 极差中位 %.4f > %.4f)"
                      " -> 颜色与图案纠缠, 直接进 Stage B 频带分离。"
                      % (med, np.median(tr_ok), args.tpf_range_max))
        else:
            verdict = "FOLLOWS"
            detail = ("输出主色跟随 caption(中位响应 %.1f, 跟随率 %.2f)且图案稳定"
                      " -> 模型本来就服从文本颜色, CTD 价值有限;"
                      " 主线应转向纹理侧余量(TCF-LAB 离地板还有 6~8.5)。"
                      % (med, summary["follow_rate_mean"]))
    else:
        verdict = "PARTIAL"
        detail = ("介于两者之间(中位响应 %.1f, 跟随率 %.2f) -> 扩大样本量或"
                  "收紧到高冲突档再判一次。" % (med, summary["follow_rate_mean"]))
    summary["verdict"] = verdict
    summary["verdict_detail"] = detail

    with open(os.path.join(args.out_dir, "sweep_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_base": per_base}, f, indent=2, ensure_ascii=False)
    for name, data in (("sweep_per_item.csv", rows), ("sweep_per_base.csv", per_base)):
        fields = sorted({k for r in data for k in r})
        with open(os.path.join(args.out_dir, name), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(data)

    print("[score] base=%d items=%d missing=%d  backend=%s"
          % (summary["n_base"], summary["n_items"], missing,
             summary["mask_backend"]["mask_backend"]))
    print()
    print("  响应幅度 response_range  mean=%.2f  median=%.2f  p10=%.2f"
          % (summary["response_range_mean"], med, summary["response_range_p10"]))
    print("  跟随率   follow_rate     mean=%.3f" % summary["follow_rate_mean"])
    if summary["tpf_range_median"] is not None:
        print("  图案漂移 tpf_range      median=%.4f (tpf_mean=%.4f)"
              % (summary["tpf_range_median"], summary["tpf_mean"]))
    else:
        print("  图案漂移 tpf_range      不可用 —— tpf 全为 NaN")
        print("    (compute_texture_pattern_fidelity 的 Gram 项需要 torchvision VGG 权重,")
        print("     离线节点上可能取不到。此时无法判定 FOLLOWS_BUT_PATTERN_DRIFTS,")
        print("     颜色是否跟随的主判定不受影响。)")
    print()
    print("  判定: %s" % verdict)
    print("  %s" % detail)
    return 0


# ---------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="plan",
                    choices=["preflight", "plan", "generate", "score", "all"])
    ap.add_argument("--per_sample_csv",
                    default="eval_outputs/report_e7a/metrics_per_sample.csv")
    ap.add_argument("--data_root", default="/mnt/f/fuxian/dataset/datasets/BF/training")
    ap.add_argument("--out_dir", default="output_eval/caption_color_sweep/e5")
    ap.add_argument("--colors", default=",".join(DEFAULT_COLORS))
    ap.add_argument("--num_samples", type=int, default=48,
                    help="base 样本数; 总生成量 = num_samples x 颜色档数")
    ap.add_argument("--generation_seed", type=int, default=42)
    ap.add_argument("--mask_policy", default="sketch_only", choices=["auto", "sketch_only"])

    # 生成侧(与 run_fixed_benchmark 对齐)
    ap.add_argument("--gam_ckpt", default="")
    ap.add_argument("--texture_ckpt", default="")
    ap.add_argument("--run_name", default="e5_tcpm_lite")
    ap.add_argument("--mode", default="token")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--texture_preprocess_mode", default="plain_resize")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry_run", action="store_true")

    # 默认不传给 inference —— 既有报告全部跑在 inference 默认值上(见 _inference_cmd)。
    # 只有显式开启 --override_generation_params 时才传, 此时扫描结果与报告不可比。
    ap.add_argument("--override_generation_params", action="store_true",
                    help="传 width/height/guidance_scale/ipa_scale。会破坏与既有报告的可比性")
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--guidance_scale", type=float, default=3.0)
    ap.add_argument("--ipa_scale", type=float, default=0.8)

    # experiment_to_flags 需要的默认值(与 run_fixed_benchmark 保持一致)
    ap.add_argument("--use_texture_gate", type=int, choices=[0, 1], default=1)
    ap.add_argument("--use_palette_tokens", type=int, choices=[0, 1], default=0)
    ap.add_argument("--num_palette_tokens", type=int, default=4)
    ap.add_argument("--layer_group_enabled", type=int, choices=[0, 1], default=1)
    ap.add_argument("--gate_type", default="layer")
    ap.add_argument("--gate_init", default="identity")
    ap.add_argument("--gate_min", type=float, default=0.7)
    ap.add_argument("--gate_max", type=float, default=1.3)
    ap.add_argument("--use_balanced_fusion_gate", type=int, choices=[0, 1], default=0)
    ap.add_argument("--balanced_gate_hidden_dim", type=int, default=64)
    ap.add_argument("--balanced_gate_scale", type=float, default=0.2)
    ap.add_argument("--balanced_gate_min", type=float, default=0.8)
    ap.add_argument("--balanced_gate_max", type=float, default=1.2)
    ap.add_argument("--use_conflict_aware_gate", type=int, choices=[0, 1], default=0)
    ap.add_argument("--use_tcpm_lite", type=int, choices=[0, 1], default=1)
    ap.add_argument("--use_aa_tcr_fuse", type=int, choices=[0, 1], default=0)

    # 与 run_fixed_benchmark 同默认值, 逐 flag 传给 inference
    ap.add_argument("--conflict_texture_suppress_strength", type=float, default=0.1)
    ap.add_argument("--conflict_palette_suppress_strength", type=float, default=0.4)
    ap.add_argument("--conflict_deltae_norm", type=float, default=50.0)
    ap.add_argument("--conflict_threshold", type=float, default=0.70)
    ap.add_argument("--alpha1", type=float, default=1.0)
    ap.add_argument("--alpha2", type=float, default=1.0)
    ap.add_argument("--alpha3", type=float, default=0.7)
    ap.add_argument("--alpha4", type=float, default=0.5)

    # 判定阈值
    ap.add_argument("--response_low", type=float, default=10.0,
                    help="中位 response_range 低于此值判为 PINNED")
    ap.add_argument("--response_high", type=float, default=20.0,
                    help="中位 response_range 高于此值且跟随率>=0.5 判为 FOLLOWS")
    ap.add_argument("--tpf_range_max", type=float, default=0.010,
                    help="tpf 极差中位超过此值视为图案漂移")

    # preflight
    ap.add_argument("--require_mask_backend", default="opencv",
                    choices=["opencv", "pillow_fallback", "any"],
                    help="preflight 要求的 mask 形态学后端; any 表示不检查")
    ap.add_argument("--check_ckpt", action="store_true",
                    help="preflight 时也检查 checkpoint(generate 阶段会自动检查)")
    return ap


def main():
    args = build_parser().parse_args()
    if args.stage in ("preflight", "all"):
        rc = stage_preflight(args)
        if rc:
            return rc
    if args.stage in ("plan", "all"):
        rc = stage_plan(args)
        if rc:
            return rc
    if args.stage in ("generate", "all"):
        rc = stage_generate(args)
        if rc:
            return rc
    if args.stage in ("score", "all"):
        return stage_score(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
