"""汇总 E26 的阶段报告、完整性、内部纹样与非目标区域漂移。"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tools import e25_spatial_diagnosis as e25
from tools.e15_common import write_json
from tools.e20_utilization import case_stat
from tools.e26_b import _correlation


def local_similarity(image, target, inner):
    values = []
    for y in range(0, image.height - 63, 32):
        for x in range(0, image.width - 63, 32):
            region = inner[y:y+64, x:x+64]
            if region.mean() < .90:
                continue
            a, b = image.crop((x, y, x+64, y+64)), target.crop((x, y, x+64, y+64))
            if np.asarray(b.convert("L"))[region].std() >= 5:
                values.append(_correlation(a, b, region))
    return float(np.mean(values)) if values else None


def self_similarity(image, inner):
    gray = np.asarray(image.convert("L"), dtype=float)
    values = []
    for dy, dx in ((0, 8), (0, 16), (0, 32), (8, 0), (16, 0), (32, 0)):
        shifted = np.roll(gray, (-dy, -dx), axis=(0, 1))
        valid = inner & np.roll(inner, (-dy, -dx), axis=(0, 1))
        if dy:
            valid[-dy:, :] = False
        if dx:
            valid[:, -dx:] = False
        a, b = gray[valid], shifted[valid]
        a, b = a - a.mean(), b - b.mean()
        values.append(float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)))
    return np.asarray(values)


def supplementary(out, stage, cases, arms):
    rows = []
    for ref in cases["references"]:
        sk = cases["sketches"][ref["id"] % 2]
        mask = np.asarray(Image.open(out / sk["mask"]).convert("L")) > 0
        inner = cv2.erode(mask.astype(np.uint8), np.ones((17, 17), np.uint8)) > 0
        background = cv2.dilate(mask.astype(np.uint8), np.ones((17, 17), np.uint8)) == 0
        ptype = ref.get("pattern_type", ref.get("group", "stripe"))
        for arm in arms(ptype):
            for seed in (42, 43):
                pair = []
                for variant in ref["variants"]:
                    name = f"c{ref['id']:02d}_s{seed}_{variant['variant']}"
                    if stage == "B_coordinate_field":
                        folder = out / stage / ptype / arm
                        if ptype == "stripe" and arm == "scalar":
                            folder = out / "A_geometry/generations"
                        scaffold_arm = "scalar" if ptype == "plaid" and arm == "scalar" else "field"
                        scaffold_path = out / stage / ptype / "scaffolds" / f"c{ref['id']:02d}_{variant['variant']}_{scaffold_arm}.png"
                        if ptype == "stripe" and arm == "scalar":
                            scaffold_path = out / "A_geometry/scaffolds" / f"c{ref['id']:02d}_s42_{variant['variant']}.png"
                    else:
                        folder = out / stage / arm
                        scaffold_path = out / stage / "scaffolds" / f"c{ref['id']:02d}_{variant['variant']}_{arm}.png"
                    path = folder / f"{name}.png"
                    image, target = Image.open(path).convert("RGB"), Image.open(scaffold_path).convert("RGB")
                    meta = json.loads((folder / f"{name}.json").read_text())
                    row = {"case": ref["id"], "seed": seed, "variant": variant["variant"],
                           "method": f"{ptype}_{arm}", "path": str(path.relative_to(out)),
                           "local_patch_similarity": local_similarity(image, target, inner),
                           "local_self_similarity_error": float(abs(self_similarity(image, inner) - self_similarity(target, inner)).mean())}
                    rows.append(row)
                    pair.append((meta, image))
                rows[-1].update(non_target_contour_drift=abs(pair[0][0]["contour_f1"] - pair[1][0]["contour_f1"]),
                                non_target_background_drift=float(abs(np.asarray(pair[0][1], dtype=float) -
                                np.asarray(pair[1][1], dtype=float))[background].mean() / 255))
    methods = {}
    for key in sorted({r["method"] for r in rows}):
        methods[key] = {metric: case_stat([(r["case"], r[metric]) for r in rows if r["method"] == key and
                       r.get(metric) is not None]) for metric in ("local_patch_similarity", "local_self_similarity_error",
                       "non_target_contour_drift", "non_target_background_drift")}
    write_json(out / stage / "supplementary_metrics.json", {"methods": methods, "rows": rows,
               "self_similarity_offsets_px": [[8, 0], [16, 0], [32, 0], [0, 8], [0, 16], [0, 32]],
               "interpretation": "64px 局部块和自相似只比较生成图与各自 scaffold；不能替代真实对应 GT"})
    return methods


def audit_integrity(root, out):
    folders = {"A_estimated": (out / "A_geometry/generations", 128),
               "B_stripe_field": (out / "B_coordinate_field/stripe/field", 32),
               "B_plaid_scalar": (out / "B_coordinate_field/plaid/scalar", 32),
               "B_plaid_field": (out / "B_coordinate_field/plaid/field", 32),
               "B_repeated_field": (out / "B_coordinate_field/repeated_print/field", 32),
               "C_raw_uv": (out / "C_real_reference/raw_uv", 40),
               "C_rectified_confidence": (out / "C_real_reference/rectified_confidence", 40)}
    counts, errors, seen_noise = {}, [], {}
    for key, (folder, expected) in folders.items():
        metadata = sorted(folder.glob("c*.json"))
        counts[key] = {"expected": expected, "actual": len(metadata)}
        if len(metadata) != expected:
            errors.append(f"{key}: count")
        for path in metadata:
            row = json.loads(path.read_text())
            image_path = out / row["path"]
            if not image_path.exists() or e25.file_sha(image_path) != row["output_sha256"]:
                errors.append(str(path.relative_to(out)))
            if key.startswith("C_"):
                noise_key = (row["case"], row["seed"], row["variant"])
                if noise_key in seen_noise and seen_noise[noise_key] != row["noise_sha256"]:
                    errors.append(f"C noise mismatch: {noise_key}")
                seen_noise[noise_key] = row["noise_sha256"]
    a = json.loads((out / "protocol.json").read_text())
    b = json.loads((out / "B_coordinate_field/protocol.json").read_text())
    c = json.loads((out / "C_real_reference/protocol.json").read_text())
    same = all(a["checkpoint_sha256"] == p["checkpoint_sha256"] for p in (b, c)) and all(
        a[f"{module}_hash"] == b[f"{module}_hash"] == c["frozen_model_hashes"][module] for module in ("vae", "unet"))
    if not same:
        errors.append("stage model mismatch")
    return {"pass": not errors, "generated_counts": counts, "generated_images_total": sum(v["actual"] for v in counts.values()),
            "errors": errors, "same_checkpoint_vae_unet": same, "C_matched_noise_pass": not any("noise" in e for e in errors)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root, out = Path(args.root), Path(args.out)
    source = e25.source_path(root, "e25")
    shutil.copytree(source / "C_refine/on_low", out / "A_geometry/oracle_baseline", dirs_exist_ok=True)
    a = json.loads((out / "A_geometry/end_to_end_report.json").read_text())
    b = json.loads((out / "B_coordinate_field/audit/report.json").read_text())
    c = json.loads((out / "C_real_reference/report.json").read_text())
    cases_b = json.loads((out / "B_coordinate_field/cases.json").read_text())
    cases_c = json.loads((out / "C_real_reference/cases.json").read_text())
    extra_b = supplementary(out, "B_coordinate_field", cases_b,
                            lambda p: ("scalar", "field") if p != "repeated_print" else ("field",))
    extra_c = supplementary(out, "C_real_reference", cases_c, lambda _: ("raw_uv", "rectified_confidence"))
    integrity = audit_integrity(root, out)
    original_cases = json.loads((out / "cases.json").read_text())
    estimates = [json.loads(line) for line in (out / "A_geometry/estimates.jsonl").read_text().splitlines()]
    phase_records = [{"case": r["case"], "variant": r["variant"], "estimated_phase": r["geometry"]["phase"],
                      "metadata_phase": r["phase_gt"], "cyclic_scalar_error": float(abs((r["geometry"]["phase"] - r["phase_gt"] + .5) % 1 - .5))}
                     for r in estimates]
    write_json(out / "A_geometry/phase_audit.json", {"rows": phase_records,
        "interpretation": "scalar phase 元数据未随 reference 旋转改写；直接差值含坐标原点/正负轴歧义。Oracle/estimated 的相位兼容性以相同 renderer 图像逐像素差核验。"})
    oracle_diffs = []
    for r in estimates:
        name = f"c{r['case']:02d}_s42_{r['variant']}.png"
        oracle = source / "A_oracle" / name
        if oracle.exists():
            diff = abs(np.asarray(Image.open(out / r["scaffold"]), dtype=float) - np.asarray(Image.open(oracle), dtype=float))
            oracle_diffs.append(float(diff.max()))
    lattice = [json.loads(line) for line in (out / "B_coordinate_field/estimates.jsonl").read_text().splitlines()
               if json.loads(line)["pattern_type"] == "repeated_print"]
    lattice_errors = [float(np.linalg.norm(np.asarray(r["estimate"]["v1"]) - [32, 0]) +
                             np.linalg.norm(np.asarray(r["estimate"]["v2"]) - [0, 32])) for r in lattice]
    report = {"decision": json.loads((out / "decision_summary.json").read_text()),
              "A_geometry": json.loads((out / "A_geometry/report.json").read_text()), "A_end_to_end": a,
              "B_coordinate_field": b, "C_real_reference": c, "B_supplementary": extra_b,
              "C_supplementary": extra_c, "integrity": integrity,
              "A_scaffold_oracle_max_rgb_difference": max(oracle_diffs) if oracle_diffs else None,
              "actual_A_frequencies": sorted({r["frequency"] for r in original_cases["references"]}),
              "B_lattice_geometry": {"estimated_vectors_original": [[32, 0], [0, 32]],
                  "gt_vector_error_px": max(lattice_errors), "coverage": len(lattice), "axis_aligned_synthetic_only": True},
              "report_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "limits": ["复用实际 E25 的32组，频率包含旧8组的4/10；与文档列举的6/7/9/11/13/15存在差异，未悄然更换验证集。",
                         "B 是平面 bbox UV baseline；条纹方向与 scalar 持平，plaid 保留双轴和图案优于 scalar；未证明复杂服装形变对应。",
                         "C1/C2 独立报告；C3 没有完整服装输入，未评测；不训练新模型。",
                         "CI 按 reference case 聚合后 bootstrap，不把种子/旋转当独立样本。"]}
    write_json(out / "report.json", report)
    write_json(out / "integrity_report.json", integrity)
    files = [{"path": str(p.relative_to(out)), "bytes": p.stat().st_size, "sha256": e25.file_sha(p)}
             for p in sorted(out.rglob("*")) if p.is_file() and p.suffix in (".png", ".jpg", ".npz", ".json", ".jsonl")
             and p.name != "artifact_manifest.json"]
    write_json(out / "artifact_manifest.json", {"files": files, "count": len(files)})
    print(json.dumps({"decision": report["decision"], "integrity": integrity, "artifacts": len(files)}), flush=True)
    if not integrity["pass"]:
        raise RuntimeError("E26 产物完整性检查失败")


if __name__ == "__main__":
    main()
