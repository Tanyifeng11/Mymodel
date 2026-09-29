"""用保存的 E26 图像补齐对应、结构及配对评测；不重新生成或训练。"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from eval.eval_utils import estimate_foreground_mask
from models.pattern_coordinate_field import (FIELD_VERSION, build_affine_pattern_field,
    build_plaid_field, build_lattice_field, garment_coordinates, sample_reference_with_field)
from models.pattern_geometry import axial_distance, estimate_pattern_geometry
from tools import e25_spatial_diagnosis as e25
from tools.e26_b import estimate_plaid_axes, estimate_lattice, _correlation
from tools.e22_o4_metrics import contour, orientation
from tools.e20_utilization import case_stat
from tools.e15_common import write_json


def structural_drift(image, scaffold, mask_image):
    mask = np.asarray(mask_image) > 0
    inner = cv2.erode(mask.astype(np.uint8), np.ones((17, 17), np.uint8)) > 0
    dilated = cv2.dilate(mask.astype(np.uint8), np.ones((17, 17), np.uint8)) > 0
    boundary, background = dilated & ~inner, ~dilated
    actual = np.asarray(image, dtype=float) / 255
    expected = np.asarray(scaffold, dtype=float) / 255
    fg = estimate_foreground_mask(image, image.size)
    target_edge, pred_edge = contour(mask), contour(fg)
    target_distance = cv2.distanceTransform((~target_edge).astype(np.uint8), cv2.DIST_L2, 5)
    pred_distance = cv2.distanceTransform((~pred_edge).astype(np.uint8), cv2.DIST_L2, 5)
    displacement = .5 * (target_distance[pred_edge].mean() + pred_distance[target_edge].mean())
    return {"boundary_rgb_deviation": float(abs(actual - expected)[boundary].mean()),
            "contour_displacement_px": float(displacement),
            "boundary_occupancy": float(fg[boundary].mean()),
            "boundary_occupancy_error": float(np.mean(fg[boundary] != mask[boundary])),
            "background_rgb_deviation": float(abs(actual - 1)[background].mean())}


def spectral_period(image, roi, axis):
    x, y, w, h = roi
    gray = np.asarray(image.convert("L"), dtype=float)[y:y+h, x:x+w] / 255
    profile = gray.mean(axis)
    signal = profile - profile.mean()
    energy = float(np.std(signal))
    spectrum = abs(np.fft.rfft(signal * np.hanning(len(signal)), n=4096))
    frequencies = np.fft.rfftfreq(4096)
    spectrum[(frequencies < 1.5 / len(signal)) | (frequencies > .20)] = 0
    peak = int(np.argmax(spectrum))
    return {"cycles_per_pixel": float(frequencies[peak]), "energy": energy,
            "valid": bool(energy >= .02 and peak > 0)}


def audit(root, out):
    base = out / "B_coordinate_field"
    cases = json.loads((base / "cases.json").read_text())
    original = json.loads((base / "report.json").read_text())
    folder = base / "audit"
    fields_folder = folder / "fields"
    fields_folder.mkdir(parents=True, exist_ok=True)
    groups, mappings = {}, []
    for ref in cases["references"]:
        ptype = ref["pattern_type"]
        sk = cases["sketches"][ref["id"] % 2]
        mask = Image.open(out / sk["mask"]).convert("L")
        xg, yg, bbox, mask_array = garment_coordinates(mask)
        inner = cv2.erode(mask_array.astype(np.uint8), np.ones((17, 17), np.uint8)) > 0
        templates = {v["variant"]: Image.open(base / ptype / "scaffolds" /
                     f"c{ref['id']:02d}_{v['variant']}_field.png").convert("RGB") for v in ref["variants"]}
        for variant in ref["variants"]:
            reference = Image.open(out / variant["path"]).convert("RGB")
            if ptype == "stripe":
                geom = estimate_pattern_geometry(reference)
                field = build_affine_pattern_field(mask, geom.orientation, geom.frequency, geom.phase, reference.size)
            elif ptype == "plaid":
                field = build_plaid_field(mask, *estimate_plaid_axes(reference), reference.size)
            else:
                field = build_lattice_field(mask, *estimate_lattice(reference), (0, 0), reference.size)
            # GT 为本实验预定的 bbox affine W；不使用估计器或 field builder 计算 GT。
            yy, xx = np.indices(mask_array.shape)
            truth_uv = np.stack(((xx - bbox[0]) * (reference.width - 1) / (bbox[2] - bbox[0]),
                                 (yy - bbox[1]) * (reference.height - 1) / (bbox[3] - bbox[1])), -1)
            corr_error = float(np.linalg.norm(field.uv - truth_uv, axis=-1)[inner].mean())
            regenerated = sample_reference_with_field(reference, field, mask)
            scaffold = templates[variant["variant"]]
            diff = abs(np.asarray(regenerated, dtype=float) - np.asarray(scaffold, dtype=float))
            name = f"c{ref['id']:02d}_{variant['variant']}"
            path = fields_folder / f"{name}.npz"
            np.savez_compressed(path, uv=field.uv, confidence=field.confidence,
                                canonical_reference=field.canonical_reference,
                                canonical_target=field.canonical_target, reference_basis=field.reference_basis,
                                target_basis=field.target_basis, origin=field.origin)
            mappings.append({"case": ref["id"], "pattern_type": ptype, "variant": variant["variant"],
                             "correspondence_error_px": corr_error,
                             "canonical_cycle_error": float(np.max(abs(field.uv @ field.reference_basis.T +
                                                                       field.origin - field.canonical_target))),
                             "scaffold_regeneration_max_rgb_difference": float(diff.max()),
                             "field_sha256": e25.file_sha(path), "field_version": FIELD_VERSION})
            arms = ("scalar", "field") if ptype != "repeated_print" else ("field",)
            for arm in arms:
                for seed in (42, 43):
                    name = f"c{ref['id']:02d}_s{seed}_{variant['variant']}"
                    source_folder = out / "A_geometry/generations" if ptype == "stripe" and arm == "scalar" else base / ptype / arm
                    image = Image.open(source_folder / f"{name}.png").convert("RGB")
                    old = json.loads((source_folder / f"{name}.json").read_text())
                    drift = structural_drift(image, scaffold, mask)
                    expected_theta = variant["theta"] if ptype == "stripe" else 90
                    direction = orientation(image, sk["roi"])
                    ori_error = axial_distance(direction["theta"], expected_theta)
                    periods = []
                    for axis in ((0 if expected_theta == 90 else 1,) if ptype == "stripe" else (0, 1)):
                        pred, truth = spectral_period(image, sk["roi"], axis), spectral_period(scaffold, sk["roi"], axis)
                        periods.append({"axis": axis, "predicted": pred, "target": truth,
                                        "log2_error": abs(math.log2(pred["cycles_per_pixel"] / truth["cycles_per_pixel"]))})
                    matched = _correlation(image, scaffold, inner)
                    wrong = templates["rot90" if variant["variant"] == "original" else "original"]
                    row = {"case": ref["id"], "seed": seed, "variant": variant["variant"],
                           "orientation_error": ori_error, "orientation_valid": direction["valid"],
                           "orientation": direction["theta"], "periods": periods,
                           "identity_similarity": matched, "response_margin": matched - _correlation(image, wrong, inner),
                           "contour_f1": old["contour_f1"], "sketch_iou": old["sketch_iou"],
                           "leakage": old["leakage"], **drift}
                    groups.setdefault(f"{ptype}_{arm}", []).append(row)
    summaries = {}
    for key, rows in groups.items():
        index = {(r["case"], r["seed"], r["variant"]): r for r in rows}
        paired = []
        for row in rows:
            if row["variant"] != "original":
                continue
            rotated = index[row["case"], row["seed"], "rot90"]
            if key.startswith("stripe"):
                follow = (row["orientation_valid"] and rotated["orientation_valid"] and
                          row["orientation_error"] <= 20 and rotated["orientation_error"] <= 20 and
                          axial_distance(row["orientation"], rotated["orientation"]) >= 70)
            else:
                follow = row["response_margin"] > .05 and rotated["response_margin"] > .05
            original_path = _image_path(out, base, key, row)
            rotated_path = _image_path(out, base, key, rotated)
            mask = Image.open(out / cases["sketches"][row["case"] % 2]["mask"]).convert("L")
            outside = cv2.dilate((np.asarray(mask) > 0).astype(np.uint8), np.ones((17, 17), np.uint8)) == 0
            rgb1, rgb2 = np.asarray(Image.open(original_path), dtype=float), np.asarray(Image.open(rotated_path), dtype=float)
            paired.append({"case": row["case"], "follow": float(follow),
                           "contour_drift": abs(row["contour_f1"] - rotated["contour_f1"]),
                           "background_drift": float(abs(rgb1 - rgb2)[outside].mean() / 255)})
        summary = {metric: case_stat([(r["case"], r[metric]) for r in rows]) for metric in
                   ("identity_similarity", "contour_f1", "sketch_iou", "leakage", "boundary_rgb_deviation",
                    "contour_displacement_px", "boundary_occupancy_error", "background_rgb_deviation")}
        summary.update({"pattern_follow": case_stat([(r["case"], r["follow"]) for r in paired]),
                        "non_target_contour_drift": case_stat([(r["case"], r["contour_drift"]) for r in paired]),
                        "non_target_background_drift": case_stat([(r["case"], r["background_drift"]) for r in paired]),
                        "orientation_median_error": float(np.median([r["orientation_error"] for r in rows])) if key.startswith("stripe") else None,
                        "period_median_log2_error": float(np.median([p["log2_error"] for r in rows for p in r["periods"]])),
                        "pattern_period_coverage": float(np.mean([p["predicted"]["valid"] for r in rows for p in r["periods"]]))})
        summaries[key] = summary
        write_json(folder / f"{key}_cases.json", rows)
    map_pass = all(r["correspondence_error_px"] < 1e-3 and r["scaffold_regeneration_max_rgb_difference"] == 0 for r in mappings)
    gates = dict(original["gates"])
    gates["correspondence_audit_pass"] = bool(map_pass)
    gates["stripe_field_pass"] = bool(summaries["stripe_field"]["pattern_follow"]["mean"] >= .75)
    write_json(folder / "report.json", {"methods": summaries, "mapping": mappings, "gates": gates,
              "interpretation": "首版是 bbox UV 映射；stripe scalar/field 用方向配对评测。plaid/print 的像素相关仍为相位敏感指标。",
              "no_claim": "不证明复杂变形对应或未见 motif 泛化。"})
    decision = json.loads((out / "decision_summary.json").read_text())
    decision.update(B_stripe_field_pass=gates["stripe_field_pass"], B_correspondence_audit_pass=map_pass,
                    C_real_reference_pass=None, best_representation="bbox_uv_reference_sampling",
                    next_route="real_reference_rectification" if all(gates.values()) else "revise_coordinate_field")
    write_json(out / "decision_summary.json", decision)
    print(json.dumps({"gates": gates, "methods": {k: {"follow": v["pattern_follow"]["mean"],
                     "period_error": v["period_median_log2_error"]} for k, v in summaries.items()}}), flush=True)


def _image_path(out, base, key, row):
    name = f"c{row['case']:02d}_s{row['seed']}_{row['variant']}.png"
    if key == "stripe_scalar":
        return out / "A_geometry/generations" / name
    ptype, arm = key.rsplit("_", 1)
    return base / ptype / arm / name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    audit(Path(args.root), Path(args.out))


if __name__ == "__main__":
    main()
