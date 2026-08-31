#!/usr/bin/env python
"""渲染 CTD 的 S2/S3 增广冲突评测集 (docs/ctd_stage_a_spec.md §3)。

`run_fixed_benchmark.py` 的生成是把 --texture_path 传给 inference 的子进程调用,
所以离线把扰动后的参考图渲染到磁盘、另写一份 JSON, 就完全不用碰生成代码。

三套评测集(spec §5.1):
  S1 标准集       原始参考图              护栏
  S2 族 A(训练同族) pick_far_ab           主战场(in-distribution)
  S3 族 B(保留族)   polar_ab, 与表无关     泛化主张(out-of-distribution)

族 A / 族 B 分开是为了避免"训练测试同源" —— 训练用 COLOR_TABLE 里的远色, 泛化
测试用与表无关的 LAB 极坐标。两套都保留真实目标图, 所以 FID/LPIPS 全套可算。

无颜色词的样本两个集合都保留原始参考图 —— 它们是 no_text_color 对照(D1 护栏)。

纯 CPU、无 GPU、无训练:
    python tools/prepare_ctd_eval_sets.py --out_dir data/ctd_eval
"""
import argparse
import csv
import json
import os
import random
import sys

from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from color_conflict_utils import (  # noqa: E402
    chroma_shift_to,
    delta_e_rgb,
    dominant_rgb_from_pil,
    extract_text_color,
    pick_far_ab,
    pick_gamut_aware_far_ab,
    pick_gamut_aware_polar_ab,
    polar_ab,
)
from garment_mask_utils import mask_backend_info  # noqa: E402


def _rel(path, root):
    """把绝对路径转成相对 data_root 的形式, 与 data/*.json 的写法一致。"""
    try:
        return os.path.relpath(path, root).replace(os.sep, "/")
    except ValueError:
        return path.replace(os.sep, "/")


def _load_rows(args):
    """从 E7A 的 per-sample CSV 读评测集 —— 它就是 500 样本 val split 的落地形式。

    spec §3 写的 eval/benchmarks/phase1_bf_val_split.json 在本仓库不存在, 而这份
    CSV 是既有 E5/E7a 报告实际跑的那 500 条(含 texture_path / prompt / sample_id),
    用它可以保证 S1/S2/S3 与既有基线逐样本对齐。
    """
    with open(args.per_sample_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        caption = r.get("prompt") or r.get("caption") or ""
        texture = r.get("texture_path") or ""
        cloth = r.get("source_target_path") or r.get("target_path") or ""
        sketch = r.get("sketch_path") or ""
        if not (caption and texture and cloth and sketch):
            continue
        out.append(
            {
                "sample_id": r.get("sample_id") or r.get("uid") or "",
                "caption": caption,
                "texture_abs": texture,
                "cloth_abs": cloth,
                "sketch_abs": sketch,
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_sample_csv",
                    default="eval_outputs/report_e7a/metrics_per_sample.csv")
    ap.add_argument("--data_root", default="/mnt/f/fuxian/dataset/datasets/BF/training")
    ap.add_argument("--out_dir", default="data/ctd_eval")
    ap.add_argument("--ctd_seed", type=int, default=42,
                    help="必须与训练的 --ctd_seed 一致, 否则 S2 不是'训练同族'")
    ap.add_argument("--min_delta_e", type=float, default=15.0)
    ap.add_argument("--chroma", type=float, default=45.0,
                    help="legacy 策略下族 B 的 LAB 色度半径")
    ap.add_argument("--ctd_target_strategy", choices=["legacy", "gamut_aware"], default="legacy",
                    help="目标色策略；legacy 保持旧行为，gamut_aware 按参考图亮度选择可达目标")
    ap.add_argument("--ctd_eval_pair_mode", choices=["intersection", "independent"], default="intersection",
                    help="A/B 样本保留方式；intersection 保持旧行为，independent 允许两集合独立保留")
    ap.add_argument("--gamut_requested_chroma", type=float, default=90.0,
                    help="gamut_aware 族 B 的请求色度；实际输出仍由色域安全回退限制")
    ap.add_argument("--gamut_candidate_count", type=int, default=8,
                    help="gamut_aware 族 B 的黄金角候选数")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = _load_rows(args)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("没有读到样本, 检查 --per_sample_csv")

    tex_a = os.path.join(args.out_dir, "texA")
    tex_b = os.path.join(args.out_dir, "texB")
    os.makedirs(tex_a, exist_ok=True)
    os.makedirs(tex_b, exist_ok=True)

    set_s1, set_a, set_b, manifest, skipped = [], [], [], [], []
    n_no_text = 0

    for i, row in enumerate(rows):
        sid = row["sample_id"] or ("%06d" % i)
        caption = row["caption"]
        _, text_rgb = extract_text_color(caption)

        base_entry = {
            "caption": caption,
            "cloth": _rel(row["cloth_abs"], args.data_root),
            "sketch": _rel(row["sketch_abs"], args.data_root),
            "sample_id": sid,
        }
        # S1 保留原始参考图，是 D1/D2 的护栏；S2/S3 只是同一批样本的反事实版本。
        set_s1.append(
            dict(base_entry, texture=_rel(row["texture_abs"], args.data_root))
        )

        if text_rgb is None:
            # 无颜色词: 两个集合都用原始参考图 —— no_text_color 对照(D1 护栏)
            n_no_text += 1
            entry = dict(base_entry, texture=_rel(row["texture_abs"], args.data_root))
            set_a.append(dict(entry))
            set_b.append(dict(entry))
            manifest.append({
                "sample_id": sid, "has_text_color": 0, "perturbed": 0,
                "reason": "no_text_color",
            })
            continue

        src = Image.open(row["texture_abs"]).convert("RGB")
        orig_rgb = dominant_rgb_from_pil(src)

        # 族 A: 与训练同一条 RNG 路径 —— 同一样本拿到同一个反事实颜色。
        rng = random.Random(args.ctd_seed * 1000003 + i)
        base_hue = (i * 137.5 + 60.0) % 360.0
        if args.ctd_target_strategy == "legacy":
            ab_a, color_name = pick_far_ab(text_rgb, rng)
            info_a = {"target_name": color_name}
            ab_b = polar_ab(base_hue, chroma=args.chroma)
            info_b = {"hue_deg": base_hue, "requested_chroma": args.chroma}
        else:
            ab_a, info_a = pick_gamut_aware_far_ab(
                orig_rgb, text_rgb, rng, min_delta_e=args.min_delta_e
            )
            rng_b = random.Random(args.ctd_seed * 1000003 + i + 7919)
            ab_b, info_b = pick_gamut_aware_polar_ab(
                orig_rgb,
                text_rgb,
                base_hue,
                rng_b,
                min_delta_e=args.min_delta_e,
                requested_chroma=args.gamut_requested_chroma,
                candidate_count=args.gamut_candidate_count,
            )

        img_a = chroma_shift_to(src, ab_a) if ab_a is not None else None
        img_b = chroma_shift_to(src, ab_b) if ab_b is not None else None
        de_a = delta_e_rgb(orig_rgb, dominant_rgb_from_pil(img_a)) if img_a is not None else 0.0
        de_b = delta_e_rgb(orig_rgb, dominant_rgb_from_pil(img_b)) if img_b is not None else 0.0

        rec = {
            "sample_id": sid,
            "has_text_color": 1,
            "text_color_rgb": list(text_rgb),
            "setA_target_ab": list(ab_a) if ab_a is not None else None,
            "setA_color_name": info_a.get("target_name", ""),
            "setA_selector": info_a,
            "setA_delta_e": de_a,
            "setB_target_ab": list(ab_b) if ab_b is not None else None,
            "setB_hue_deg": info_b.get("hue_deg", base_hue),
            "setB_selector": info_b,
            "setB_delta_e": de_b,
            "reference_rgb_orig": list(orig_rgb),
        }

        # ΔE 校验: 不达标的候选不计入对应集合。independent 模式不让 A/B 互相拖累。
        ok_a, ok_b = de_a >= args.min_delta_e, de_b >= args.min_delta_e
        rec["setA_perturbed"] = int(ok_a)
        rec["setB_perturbed"] = int(ok_b)
        rec["perturbed"] = int(ok_a and ok_b)
        if not ok_a:
            rec["setA_reason"] = "delta_e_below_min(%.2f)" % de_a
        if not ok_b:
            rec["setB_reason"] = "delta_e_below_min(%.2f)" % de_b

        keep_a = ok_a and (args.ctd_eval_pair_mode == "independent" or ok_b)
        keep_b = ok_b and (args.ctd_eval_pair_mode == "independent" or ok_a)
        if keep_a:
            pa = os.path.join(tex_a, "%s.png" % sid)
            img_a.save(pa)
            set_a.append(dict(base_entry, texture=os.path.abspath(pa)))
        if keep_b:
            pb = os.path.join(tex_b, "%s.png" % sid)
            img_b.save(pb)
            set_b.append(dict(base_entry, texture=os.path.abspath(pb)))
        if not (ok_a and ok_b):
            skipped.append(rec)
        manifest.append(rec)

        if (i + 1) % 100 == 0:
            print("[prep] %d/%d  已渲染 A=%d B=%d  skipped=%d"
                  % (i + 1, len(rows), len(set_a), len(set_b), len(skipped)))

    os.makedirs("data", exist_ok=True)
    ps1_json = os.path.join("data", "ctd_eval_setS1.json")
    pa_json = os.path.join("data", "ctd_eval_setA.json")
    pb_json = os.path.join("data", "ctd_eval_setB.json")
    for path, payload in ((ps1_json, set_s1), (pa_json, set_a), (pb_json, set_b)):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    perturbed_a = [m for m in manifest if m.get("setA_perturbed")]
    perturbed_b = [m for m in manifest if m.get("setB_perturbed")]
    perturbed = [m for m in manifest if m.get("perturbed")]
    des_a = [m["setA_delta_e"] for m in perturbed_a]
    des_b = [m["setB_delta_e"] for m in perturbed_b]
    summary = {
        "n_rows": len(rows),
        "n_setS1": len(set_s1),
        "n_no_text_color": n_no_text,
        "n_perturbed": len(perturbed),
        "n_feasible_setA": len(perturbed_a),
        "n_feasible_setB": len(perturbed_b),
        "n_skipped": len(skipped),
        "mean_delta_e_setA": (sum(des_a) / len(des_a)) if des_a else 0.0,
        "mean_delta_e_setB": (sum(des_b) / len(des_b)) if des_b else 0.0,
        "min_delta_e_gate": args.min_delta_e,
        "ctd_seed": args.ctd_seed,
        "ctd_target_strategy": args.ctd_target_strategy,
        "ctd_eval_pair_mode": args.ctd_eval_pair_mode,
        "chroma": args.chroma,
        "gamut_requested_chroma": args.gamut_requested_chroma,
        "gamut_candidate_count": args.gamut_candidate_count,
        "mask_backend": mask_backend_info(),
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "items": manifest}, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "skipped.json"), "w", encoding="utf-8") as f:
        json.dump(skipped, f, indent=2, ensure_ascii=False)

    print()
    print("[prep] 样本 %d  无颜色词 %d  A可行 %d  B可行 %d  交集 %d  有失败 %d"
          % (len(rows), n_no_text, len(perturbed_a), len(perturbed_b), len(perturbed), len(skipped)))
    print("[prep] 实测 ΔE  族A 均值 %.2f   族B 均值 %.2f"
          % (summary["mean_delta_e_setA"], summary["mean_delta_e_setB"]))
    print("[prep] -> %s (%d 条)" % (ps1_json, len(set_s1)))
    print("[prep] -> %s (%d 条)" % (pa_json, len(set_a)))
    print("[prep] -> %s (%d 条)" % (pb_json, len(set_b)))
    print("[prep] -> %s/manifest.json, skipped.json" % args.out_dir)
    if len(perturbed) and summary["mean_delta_e_setA"] < 25.0:
        print("[prep][warn] 族A 平均 ΔE < 25 —— D0 可能不过, 考虑调大 chroma 幅度")
    return 0


if __name__ == "__main__":
    sys.exit(main())
