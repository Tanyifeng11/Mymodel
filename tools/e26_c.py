"""E26-C：真实局部参考，先矫正，再做置信度感知的 bbox 对应；不训练。"""

import argparse
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_tensor

from models.local_pattern_field import (LOCAL_VERSION, estimate_local_pattern_field,
    rectify_reference, render_confidence_aware)
from models.pattern_coordinate_field import build_affine_pattern_field, sample_reference_with_field
from models.pattern_geometry import axial_distance, estimate_orientation_structure_tensor
from tools import e23_mechanism as e23
from tools import e25_spatial_diagnosis as e25
from tools.e15_common import write_json
from tools.e18_pattern_gold import pixel_hash
from tools.e20_utilization import case_stat, load_pipeline
from tools.e22_4_generation import module_hashes
from tools.e22_o4_metrics import measure, orientation
from tools.e26_audit import spectral_period, structural_drift
from tools.e26_b import _correlation


SOURCE_GROUPS = {"C1_planar_crop": (13, 17, 30, 36),
                 "C2_local_garment_crop": (2, 3, 19, 23, 25, 85)}
ARMS, SEEDS = ("raw_uv", "rectified_confidence"), (42, 43)
# C 文档未给数值门槛；在生成前保存这些探索性门槛，不能事后改动。
THRESHOLDS = {"follow_min": .90, "follow_ci_low_min": .75,
              "orientation_median_max_deg": 5., "identity_mean_min": .50,
              "confidence_coverage_min": .50, "contour_f1_min": .90,
              "leakage_max": .02, "mapping_cycle_max_px": .001}


def prepare(root, out):
    decision = json.loads((out / "decision_summary.json").read_text())
    required = ("B_stripe_field_pass", "B_plaid_field_pass", "B_repeat_field_pass",
                "B_correspondence_audit_pass")
    if not all(decision[k] for k in required):
        raise RuntimeError("B 未通过，停止 C")
    gold = e25.source_path(root, "e18") / "gold"
    manifest = json.loads((gold / "manifest.json").read_text())
    selected = {r["review_index"]: r for r in manifest["rows"] if r["variant"] == "original"}
    base = out / "C_real_reference"
    for folder in ("inputs", "rectified", "fields", "scaffolds", "previews"):
        (base / folder).mkdir(parents=True, exist_ok=True)
    sketches = json.loads((out / "cases.json").read_text())["sketches"]
    refs, geometry = [], []
    for group, indices in SOURCE_GROUPS.items():
        for review_id in indices:
            source = selected[review_id]
            original = Image.open(gold / source["image"]).convert("RGB")
            assert pixel_hash(original) == source["sha256"]
            case = len(refs)
            sk = sketches[case % 2]
            mask = Image.open(out / sk["mask"]).convert("L")
            inner = cv2.erode((np.asarray(mask) > 0).astype(np.uint8), np.ones((17, 17), np.uint8)) > 0
            variants = []
            for name, image in (("original", original), ("rot90", original.transpose(Image.Transpose.ROTATE_90))):
                prefix = f"c{case:02d}_{name}"
                relative = f"C_real_reference/inputs/{prefix}.png"
                image.save(out / relative)
                theta_gt = 90 if source["orientation"] == "vertical" else 0
                if name == "rot90":
                    theta_gt = (theta_gt + 90) % 180
                # 人工方向标签仅用于评测，估计器和 renderer 不接收标签。
                rectified, rect_uv, rect_info = rectify_reference(image)
                local = estimate_local_pattern_field(rectified)
                raw_local = estimate_local_pattern_field(image)
                theta, _ = estimate_orientation_structure_tensor(rectified)
                valid_frequencies = local.frequency[local.valid]
                frequency = float(np.median(valid_frequencies)) * rectified.width if len(valid_frequencies) else 2.
                field = build_affine_pattern_field(mask, theta, max(2., frequency), 0., image.size)
                raw = sample_reference_with_field(image, field, mask)
                mapped, blend = render_confidence_aware(rectified, field, local, mask)
                rectified_path = base / "rectified" / f"{prefix}.png"
                rectified.save(rectified_path)
                for arm, scaffold in (("raw_uv", raw), ("rectified_confidence", mapped)):
                    scaffold.save(base / "scaffolds" / f"{prefix}_{arm}.png")
                # 行位移的逆映射是减去同一行位移；只检验解析映射一致性，没有真实 UV GT。
                yy, xx = np.indices((image.height, image.width), dtype=np.float32)
                grid = np.stack((xx, yy), -1)
                displacement = rect_uv - grid
                inverse_displacement = cv2.remap(displacement, rect_uv[..., 0], rect_uv[..., 1],
                                                cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                rect_cycle_error = float(np.linalg.norm(rect_uv - inverse_displacement - grid, axis=-1).max())
                composed = cv2.remap(rect_uv, field.uv[..., 0], field.uv[..., 1], cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REFLECT_101)
                reference_theta = cv2.remap(local.orientation, field.uv[..., 0], field.uv[..., 1],
                                            cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
                reference_frequency = cv2.remap(local.frequency, field.uv[..., 0], field.uv[..., 1],
                                                cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                sx = (field.bbox[2] - field.bbox[0]) / (image.width - 1)
                sy = (field.bbox[3] - field.bbox[1]) / (image.height - 1)
                radians = np.deg2rad(reference_theta)
                target_theta = np.degrees(np.arctan2(np.sin(radians) * sy, np.cos(radians) * sx)) % 180
                normal = radians - np.pi / 2
                target_frequency = reference_frequency * np.hypot(np.cos(normal) / sx, np.sin(normal) / sy)
                field_path = base / "fields" / f"{prefix}.npz"
                np.savez_compressed(field_path, rectification_uv=rect_uv, garment_to_rectified_uv=field.uv,
                    garment_to_reference_uv=composed, orientation=local.orientation,
                    frequency=local.frequency, confidence=local.confidence, valid=local.valid,
                    raw_orientation=raw_local.orientation, raw_frequency=raw_local.frequency,
                    raw_confidence=raw_local.confidence, raw_valid=raw_local.valid, target_blend=blend,
                    target_orientation=target_theta, target_frequency=target_frequency,
                    canonical_reference=field.canonical_reference, canonical_target=field.canonical_target)
                geom = {"case": case, "review_index": review_id, "group": group, "variant": name,
                        "theta_gt": theta_gt, "rectification": rect_info,
                        "local_valid_fraction": float(local.valid.mean()),
                        "raw_local_valid_fraction": float(raw_local.valid.mean()),
                        "confidence_coverage": float((blend[inner] >= .5).mean()),
                        "median_local_confidence": float(np.median(local.confidence)),
                        "local_frequency_median_cycles_per_pixel": frequency / rectified.width,
                        "mapping_cycle_error_px": rect_cycle_error,
                        "field_sha256": e25.file_sha(field_path),
                        "reference_sha256": e25.file_sha(out / relative),
                        "rectified_sha256": e25.file_sha(rectified_path),
                        "mask_sha256": e25.file_sha(out / sk["mask"])}
                geometry.append(geom)
                variants.append({"variant": name, "path": relative, "theta": theta_gt})
            refs.append({"id": case, "review_index": review_id, "group": group,
                         "source_group": source["source_group"], "gold_pixel_sha256": source["sha256"],
                         "variants": variants})
    cases = {"references": refs, "sketches": sketches}
    write_json(base / "cases.json", cases)
    write_json(base / "geometry_report.json", {"rows": geometry, "thresholds": THRESHOLDS,
               "groups_fixed_before_generation": SOURCE_GROUPS,
               "C3_full_garment": "not_evaluated_no_full_garment_reference",
               "no_real_correspondence_gt": True, "source_identity": manifest["source_identity"]})
    return cases


@torch.inference_mode()
def generate(root, out, cases):
    base = out / "C_real_reference"
    pipe, _, _, modules, width, height = load_pipeline(root, "cuda:0")
    pipe.set_progress_bar_config(disable=True)
    frozen = module_hashes(modules)
    banks = e23.token_bank(pipe, cases, out, width, height, "E5")
    start = e25.timestep_for(pipe.scheduler, .15)
    write_json(base / "protocol.json", {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
        cwd=root, text=True).strip(), "local_version": LOCAL_VERSION, "seeds": SEEDS,
        "strength": .15, "steps": 50, "cfg": 7, "texture_path": "on", "arms": ARMS,
        "thresholds": THRESHOLDS, "groups": SOURCE_GROUPS, "canvas": [width, height],
        "scheduler": type(pipe.scheduler).__name__, "frozen_model_hashes": frozen,
        "checkpoint_sha256": e25.file_sha(root / "output/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt"),
        "interpretation": "探索性真实局部参考；仅检验映射后的纹样保留，没有真实 reference→garment UV 标签"})
    for ref in cases["references"]:
        sk = cases["sketches"][ref["id"] % 2]
        mask = Image.open(out / sk["mask"]).convert("L")
        sketch = Image.open(out / sk["path"]).convert("RGB")
        assert mask.size == (width, height)
        for variant in ref["variants"]:
            reference = Image.open(out / variant["path"]).convert("RGB")
            for arm in ARMS:
                scaffold_path = base / "scaffolds" / f"c{ref['id']:02d}_{variant['variant']}_{arm}.png"
                scaffold = Image.open(scaffold_path).convert("RGB")
                pixels = to_tensor(scaffold)[None].to(pipe.device, pipe.vae.dtype) * 2 - 1
                z0 = pipe.vae.encode(pixels).latent_dist.mean * pipe.vae.config.scaling_factor
                folder = base / arm
                folder.mkdir(exist_ok=True)
                for seed in SEEDS:
                    name = f"c{ref['id']:02d}_s{seed}_{variant['variant']}"
                    path, metadata = folder / f"{name}.png", folder / f"{name}.json"
                    if path.exists() and metadata.exists():
                        continue
                    noise = torch.randn(z0.shape, generator=torch.Generator(device=pipe.device).manual_seed(seed),
                                        device=pipe.device, dtype=z0.dtype)
                    image, noise_sha, initial_sha = e25.refine_one(pipe, z0, noise, start, sketch, mask,
                        reference, banks[ref["id"], variant["variant"]], True, mask.size)
                    image.save(path)
                    metrics = measure(image, mask, sketch, reference, variant["theta"], sk["roi"])
                    write_json(metadata, {"case": ref["id"], "group": ref["group"], "arm": arm,
                        "review_index": ref["review_index"], "seed": seed, "variant": variant["variant"],
                        "path": str(path.relative_to(out)), "contour_f1": metrics["contour_f1"],
                        "sketch_iou": metrics["sketch_iou"], "leakage": metrics["leakage"],
                        "noise_sha256": noise_sha, "initial_latent_sha256": initial_sha,
                        "reference_sha256": e25.file_sha(out / variant["path"]),
                        "mask_sha256": e25.file_sha(out / sk["mask"]),
                        "scaffold_sha256": e25.file_sha(scaffold_path), "output_sha256": e25.file_sha(path),
                        "local_version": LOCAL_VERSION, **start})
                    print("[e26-C]", ref["group"], arm, name, flush=True)
    after = module_hashes(modules)
    frozen_pass = frozen == after
    write_json(base / "frozen_check.json", {"pass": frozen_pass, "before": frozen, "after": after})
    report(out, cases, frozen_pass)


def report(out, cases, frozen_pass):
    base = out / "C_real_reference"
    geom = json.loads((base / "geometry_report.json").read_text())["rows"]
    geometry = {(r["case"], r["variant"]): r for r in geom}
    methods = {}
    for group in SOURCE_GROUPS:
        for arm in ARMS:
            rows, paired = [], []
            for ref in cases["references"]:
                if ref["group"] != group:
                    continue
                sk = cases["sketches"][ref["id"] % 2]
                mask = Image.open(out / sk["mask"]).convert("L")
                inner = cv2.erode((np.asarray(mask) > 0).astype(np.uint8), np.ones((17, 17), np.uint8)) > 0
                templates = {v["variant"]: Image.open(base / "scaffolds" /
                    f"c{ref['id']:02d}_{v['variant']}_{arm}.png").convert("RGB") for v in ref["variants"]}
                for seed in SEEDS:
                    pair = []
                    for variant in ref["variants"]:
                        name = f"c{ref['id']:02d}_s{seed}_{variant['variant']}"
                        image = Image.open(base / arm / f"{name}.png").convert("RGB")
                        old = json.loads((base / arm / f"{name}.json").read_text())
                        target = templates[variant["variant"]]
                        wrong = templates["rot90" if variant["variant"] == "original" else "original"]
                        direction = orientation(image, sk["roi"])
                        target_direction = orientation(target, sk["roi"])
                        similarity = _correlation(image, target, inner)
                        axis = 0 if variant["theta"] == 90 else 1
                        period, target_period = spectral_period(image, sk["roi"], axis), spectral_period(target, sk["roi"], axis)
                        row = {"case": ref["id"], "seed": seed, "variant": variant["variant"],
                            "orientation": direction["theta"], "orientation_valid": direction["valid"],
                            "orientation_error": axial_distance(direction["theta"], variant["theta"]),
                            "scaffold_orientation_error": axial_distance(direction["theta"], target_direction["theta"]),
                            "identity_similarity": similarity, "response_margin": similarity - _correlation(image, wrong, inner),
                            "period_valid": period["valid"] and target_period["valid"],
                            "period_log2_error": abs(math.log2(period["cycles_per_pixel"] / target_period["cycles_per_pixel"])),
                            "contour_f1": old["contour_f1"], "sketch_iou": old["sketch_iou"], "leakage": old["leakage"],
                            **structural_drift(image, target, mask)}
                        rows.append(row)
                        pair.append(row)
                    follow = all(r["orientation_valid"] and r["orientation_error"] <= 20 and
                                 r["response_margin"] > .05 for r in pair) and axial_distance(pair[0]["orientation"], pair[1]["orientation"]) >= 70
                    paired.append((ref["id"], float(follow)))
            stats = {m: case_stat([(r["case"], r[m]) for r in rows]) for m in
                     ("identity_similarity", "response_margin", "contour_f1", "sketch_iou", "leakage",
                      "boundary_rgb_deviation", "contour_displacement_px", "background_rgb_deviation")}
            stats.update(pattern_follow=case_stat(paired), cases=len(set(r["case"] for r in rows)), images=len(rows),
                         orientation_median_error=float(np.median([r["orientation_error"] for r in rows])),
                         scaffold_orientation_median_error=float(np.median([r["scaffold_orientation_error"] for r in rows])),
                         period_median_log2_error=float(np.median([r["period_log2_error"] for r in rows])),
                         period_coverage=float(np.mean([r["period_valid"] for r in rows])))
            methods[f"{group}_{arm}"] = stats
            write_json(base / f"{group}_{arm}_cases.json", rows)
    gates = {"frozen_pass": bool(frozen_pass)}
    for group in SOURCE_GROUPS:
        stats = methods[f"{group}_rectified_confidence"]
        local_rows = [r for r in geom if r["group"] == group]
        checks = {"follow": stats["pattern_follow"]["mean"] >= THRESHOLDS["follow_min"] and
                  stats["pattern_follow"]["ci95"][0] > THRESHOLDS["follow_ci_low_min"],
                  "orientation": stats["orientation_median_error"] <= THRESHOLDS["orientation_median_max_deg"],
                  "identity": stats["identity_similarity"]["mean"] >= THRESHOLDS["identity_mean_min"],
                  "confidence": all(r["confidence_coverage"] >= THRESHOLDS["confidence_coverage_min"] for r in local_rows),
                  "structure": stats["contour_f1"]["mean"] >= THRESHOLDS["contour_f1_min"] and
                  stats["leakage"]["mean"] <= THRESHOLDS["leakage_max"],
                  "mapping_consistency": all(r["mapping_cycle_error_px"] <= THRESHOLDS["mapping_cycle_max_px"] for r in local_rows)}
        gates[group] = {"pass": bool(all(checks.values()) and frozen_pass), "checks": checks}
    write_json(base / "report.json", {"methods": methods, "gates": gates, "thresholds": THRESHOLDS,
        "C3_full_garment": "not_evaluated_no_full_garment_reference", "reference_pairs": 10,
        "limits": "C1/C2 各自探索性评测，源样本身份未经 fabric identity 验证；相关性只衡量低强度生成对 scaffold 的保留；没有真实对应 GT；不能声明矫正优于 raw，需按两臂结果比较。"})
    decision = json.loads((out / "decision_summary.json").read_text())
    passed = all(gates[g]["pass"] for g in SOURCE_GROUPS)
    decision.update(C_real_reference_pass=passed,
                    C1_planar_crop_pass=gates["C1_planar_crop"]["pass"],
                    C2_local_garment_crop_pass=gates["C2_local_garment_crop"]["pass"],
                    C3_full_garment_pass=None, best_representation="bbox_uv_reference_sampling",
                    next_route="confidence_aware_correspondence" if passed else "real_reference_rectification")
    write_json(out / "decision_summary.json", decision)
    preview(out, cases)
    print(json.dumps({"gates": gates, "methods": {k: {"follow": s["pattern_follow"]["mean"],
        "theta_error": s["orientation_median_error"], "identity": s["identity_similarity"]["mean"]}
        for k, s in methods.items()}}), flush=True)


def preview(out, cases):
    base = out / "C_real_reference"
    for group in SOURCE_GROUPS:
        refs = [r for r in cases["references"] if r["group"] == group]
        sheet = Image.new("RGB", (5 * 192, len(refs) * 220), "white")
        draw = ImageDraw.Draw(sheet)
        for i, ref in enumerate(refs):
            prefix = f"c{ref['id']:02d}"
            paths = (base / "inputs" / f"{prefix}_original.png",
                     base / "rectified" / f"{prefix}_original.png",
                     base / "raw_uv" / f"{prefix}_s42_original.png",
                     base / "rectified_confidence" / f"{prefix}_s42_original.png",
                     base / "rectified_confidence" / f"{prefix}_s42_rot90.png")
            for j, (path, label) in enumerate(zip(paths, ("reference", "rectified", "raw UV", "confidence", "rot90"))):
                image = Image.open(path).convert("RGB")
                image.thumbnail((186, 192))
                sheet.paste(image, (j * 192, i * 220 + 22))
                draw.text((j * 192 + 3, i * 220 + 3), f"{ref['review_index']} {label}", fill="black")
        sheet.save(base / "previews" / f"{group}.jpg", quality=92)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stage", choices=("prepare", "generate", "report", "all"), default="all")
    args = parser.parse_args()
    root, out = Path(args.root), Path(args.out)
    torch.set_num_threads(4)
    cases = prepare(root, out) if args.stage in ("prepare", "all") else json.loads((out / "C_real_reference/cases.json").read_text())
    if args.stage in ("generate", "all"):
        generate(root, out, cases)
    elif args.stage == "report":
        report(out, cases, json.loads((out / "C_real_reference/frozen_check.json").read_text())["pass"])


if __name__ == "__main__":
    main()
