"""E26-B：解析 reference→garment 坐标场，分类型比较 scalar 与 field。"""

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from models.pattern_coordinate_field import (FIELD_VERSION, build_affine_pattern_field,
    build_lattice_field, build_plaid_field, sample_reference_with_field)
from models.pattern_geometry import estimate_pattern_geometry, render_estimated_stripe
from tools import e23_mechanism as e23
from tools import e25_spatial_diagnosis as e25
from tools.e15_common import write_json
from tools.e18_1_clean_patterns import PALETTES
from tools.e20_utilization import case_stat, load_pipeline
from tools.e22_4_generation import module_hashes
from tools.e22_o4_metrics import measure


COUNT = 8
SEEDS = (42, 43)
PATTERNS = ("stripe", "plaid", "repeated_print")


def _axis_frequency(gray, axis):
    profile = gray.mean(axis=axis)
    spectrum = abs(np.fft.rfft(profile - profile.mean()))
    spectrum[:2], spectrum[65:] = 0, 0
    return int(np.argmax(spectrum))


def estimate_plaid_axes(image):
    gray = np.asarray(image.convert("L"), dtype=np.float64) / 255
    return (90., _axis_frequency(gray, 0), 0.), (0., _axis_frequency(gray, 1), 0.)


def _period(profile):
    centered = profile - profile.mean()
    acf = np.fft.irfft(abs(np.fft.rfft(centered)) ** 2, n=len(profile))
    # 最短明显自相似峰对应单个 motif，而非第二/第三个重复周期。
    peaks = [i for i in range(12, 49) if acf[i] > acf[i - 1] and acf[i] >= acf[i + 1]]
    strong = [i for i in peaks if acf[i] >= .8 * max(acf[j] for j in peaks)]
    return min(strong)


def estimate_lattice(image):
    gray = np.asarray(image.convert("L"), dtype=np.float64) / 255
    return (float(_period(gray.mean(0))), 0.), (0., float(_period(gray.mean(1))))


def make_plaid(fx, fy, phase_x, phase_y, palette):
    y, x = np.mgrid[:256, :256]
    dx = abs((x * fx / 256 + phase_x + .5) % 1 - .5)
    dy = abs((y * fy / 256 + phase_y + .5) % 1 - .5)
    lines = (dx < .085) | (dy < .085)
    pixels = np.where(lines[..., None], palette[0], palette[1]).astype(np.uint8)
    return Image.fromarray(pixels, "RGB")


def make_repeated_print(case):
    tile = np.full((32, 32, 3), 224, dtype=np.uint8)
    colors = ((150, 35, 44), (30, 91, 130), (125, 65, 120), (65, 116, 50))
    color = colors[case % len(colors)]
    offset = case % 4
    tile[5 + offset:23 + offset, 6:10] = color
    tile[19 + offset:23 + offset, 6:24] = color
    tile[7 + offset:11 + offset, 18:23] = (30, 30, 30)
    return Image.fromarray(np.tile(tile, (8, 8, 1)), "RGB")


def prepare(root, out):
    decision = json.loads((out / "decision_summary.json").read_text())
    if not decision["A_end_to_end_pass"]:
        raise RuntimeError("E26-A 未通过，不能进入 E26-B")
    source = e25.source_path(root, "e25")
    old = json.loads((source / "cases.json").read_text())
    shutil.copytree(source / "inputs", out / "inputs", dirs_exist_ok=True)
    base = out / "B_coordinate_field"
    refs = []
    for ref in old["references"][:COUNT]:
        refs.append({**ref, "pattern_type": "stripe"})
    plaid_specs = ((6, 11), (7, 13), (9, 15), (6, 13), (7, 11), (9, 13), (6, 15), (11, 15))
    for i in range(COUNT):
        fx, fy = plaid_specs[i]
        image = make_plaid(fx, fy, .23 + .07 * (i % 3), .47 + .06 * (i % 2), PALETTES[i % 4])
        refs.append(_save_pair(out, 8 + i, "plaid", image,
                               {"frequency_x": fx, "frequency_y": fy}))
    for i in range(COUNT):
        refs.append(_save_pair(out, 16 + i, "repeated_print", make_repeated_print(i),
                               {"lattice_v1": [32, 0], "lattice_v2": [0, 32]}))
    cases = {"references": refs, "sketches": old["sketches"],
             "protocol": "8 paired cases/type; 2 seeds; synthetic planar patterns"}
    write_json(base / "cases.json", cases)
    geometry = []
    for ref in refs:
        sk = old["sketches"][ref["id"] % 2]
        mask = Image.open(out / sk["mask"]).convert("L")
        for variant in ref["variants"]:
            reference = Image.open(out / variant["path"]).convert("RGB")
            if ref["pattern_type"] == "stripe":
                estimate = estimate_pattern_geometry(reference)
                field = build_affine_pattern_field(mask, estimate.orientation, estimate.frequency,
                                                   estimate.phase, reference.size)
                params = estimate.to_dict()
            elif ref["pattern_type"] == "plaid":
                axis1, axis2 = estimate_plaid_axes(reference)
                field = build_plaid_field(mask, axis1, axis2, reference.size)
                params = {"axis1": axis1, "axis2": axis2}
            else:
                v1, v2 = estimate_lattice(reference)
                field = build_lattice_field(mask, v1, v2, (0, 0), reference.size)
                params = {"v1": v1, "v2": v2}
            ptype = ref["pattern_type"]
            folder = base / ptype / "scaffolds"
            folder.mkdir(parents=True, exist_ok=True)
            name = f"c{ref['id']:02d}_{variant['variant']}"
            image = sample_reference_with_field(reference, field, mask)
            path = folder / f"{name}_field.png"
            image.save(path)
            if ptype == "plaid":
                scalar = render_estimated_stripe(estimate_pattern_geometry(reference), mask)
                scalar.save(folder / f"{name}_scalar.png")
            geometry.append({"case": ref["id"], "pattern_type": ptype, "variant": variant["variant"],
                             "reference_sha256": e25.file_sha(out / variant["path"]),
                             "mask_sha256": e25.file_sha(out / sk["mask"]),
                             "field_sha256": e25.file_sha(path), "estimate": params,
                             "field_version": FIELD_VERSION, "bbox": field.bbox,
                             "interior_fraction": float(field.confidence.mean())})
    with (base / "estimates.jsonl").open("w") as stream:
        for row in geometry:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    plaid_ok = all((r["estimate"]["axis1"][1], r["estimate"]["axis2"][1]) ==
                   ((ref["frequency_x"], ref["frequency_y"]) if r["variant"] == "original" else
                    (ref["frequency_y"], ref["frequency_x"]))
                   for r in geometry if r["pattern_type"] == "plaid"
                   for ref in [refs[r["case"]]])
    repeat_ok = all(r["estimate"]["v1"][0] == 32 and r["estimate"]["v2"][1] == 32
                    for r in geometry if r["pattern_type"] == "repeated_print")
    report = {"plaid_axes_pass": bool(plaid_ok), "repeated_lattice_pass": bool(repeat_ok),
              "reference_pairs_per_type": COUNT, "field_version": FIELD_VERSION,
              "estimated_without_gt": True}
    write_json(base / "geometry_report.json", report)
    return report


def _save_pair(out, case_id, ptype, image, gt):
    variants = []
    for name, variant in (("original", image), ("rot90", image.transpose(Image.Transpose.ROTATE_90))):
        relative = f"inputs/e26_b_{ptype}_{case_id:02d}_{name}.png"
        variant.save(out / relative)
        variants.append({"variant": name, "path": relative})
    return {"id": case_id, "pattern_type": ptype, "variants": variants, **gt}


def _correlation(image, target, mask):
    a = np.asarray(image.convert("L"), dtype=float)[mask]
    b = np.asarray(target.convert("L"), dtype=float)[mask]
    a, b = a - a.mean(), b - b.mean()
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


@torch.inference_mode()
def generate(root, out):
    base = out / "B_coordinate_field"
    geom = json.loads((base / "geometry_report.json").read_text())
    if not geom["plaid_axes_pass"] or not geom["repeated_lattice_pass"]:
        raise RuntimeError("B 的解析几何未通过，停止生成")
    cases = json.loads((base / "cases.json").read_text())
    pipe, _, _, modules, width, height = load_pipeline(root, "cuda:0")
    pipe.set_progress_bar_config(disable=True)
    frozen = module_hashes(modules)
    banks = e23.token_bank(pipe, cases, out, width, height, "E5")
    start = e25.timestep_for(pipe.scheduler, .15)
    protocol = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                "field_version": FIELD_VERSION, "seeds": SEEDS, "strength": .15,
                "steps": 50, "cfg": 7, "texture_path": "on", "scheduler": type(pipe.scheduler).__name__,
                "vae_hash": frozen["vae"], "unet_hash": frozen["unet"],
                "checkpoint_sha256": e25.file_sha(root / "output/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt")}
    write_json(base / "protocol.json", protocol)
    for ref in cases["references"]:
        sk = cases["sketches"][ref["id"] % 2]
        mask = Image.open(out / sk["mask"]).convert("L")
        sketch = Image.open(out / sk["path"]).convert("RGB")
        assert mask.size == (width, height)
        for variant in ref["variants"]:
            reference = Image.open(out / variant["path"]).convert("RGB")
            ptype = ref["pattern_type"]
            arms = ("field", "scalar") if ptype == "plaid" else ("field",)
            for arm in arms:
                scaffold = base / ptype / "scaffolds" / f"c{ref['id']:02d}_{variant['variant']}_{arm}.png"
                pixels = to_tensor(Image.open(scaffold).convert("RGB"))[None].to(pipe.device, pipe.vae.dtype) * 2 - 1
                z0 = pipe.vae.encode(pixels).latent_dist.mean * pipe.vae.config.scaling_factor
                folder = base / ptype / arm
                folder.mkdir(parents=True, exist_ok=True)
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
                    structure = measure(image, mask, sketch, reference, 90, sk["roi"])
                    row = {"case": ref["id"], "pattern_type": ptype, "arm": arm,
                           "variant": variant["variant"], "seed": seed, "path": str(path.relative_to(out)),
                           "contour_f1": structure["contour_f1"], "sketch_iou": structure["sketch_iou"],
                           "leakage": structure["leakage"], "background_white_mae": structure["background_white_mae"],
                           "noise_sha256": noise_sha, "initial_latent_sha256": initial_sha,
                           "reference_sha256": e25.file_sha(out / variant["path"]),
                           "mask_sha256": e25.file_sha(out / sk["mask"]),
                           "scaffold_sha256": e25.file_sha(scaffold), "output_sha256": e25.file_sha(path),
                           "field_version": FIELD_VERSION, **start}
                    write_json(metadata, row)
                    print("[e26-B]", ptype, arm, name, flush=True)
    report(root, out, cases, frozen == module_hashes(modules))


def report(root, out, cases, frozen_pass):
    base = out / "B_coordinate_field"
    results = {}
    for ptype in PATTERNS:
        refs = [r for r in cases["references"] if r["pattern_type"] == ptype]
        arms = ("scalar", "field") if ptype == "plaid" else ("field",)
        if ptype == "stripe":
            arms = ("scalar", "field")
        for arm in arms:
            rows = []
            for ref in refs:
                sk = cases["sketches"][ref["id"] % 2]
                mask = np.asarray(Image.open(out / sk["mask"]).convert("L")) > 0
                inner = cv2.erode(mask.astype(np.uint8), np.ones((17, 17), np.uint8)) > 0
                templates = {v["variant"]: Image.open(base / ptype / "scaffolds" /
                    f"c{ref['id']:02d}_{v['variant']}_field.png").convert("RGB") for v in ref["variants"]}
                for seed in SEEDS:
                    for variant in ref["variants"]:
                        name = f"c{ref['id']:02d}_s{seed}_{variant['variant']}"
                        if ptype == "stripe" and arm == "scalar":
                            path = out / "A_geometry" / "generations" / f"{name}.png"
                            metadata = out / "A_geometry" / "generations" / f"{name}.json"
                        else:
                            path = base / ptype / arm / f"{name}.png"
                            metadata = base / ptype / arm / f"{name}.json"
                        image = Image.open(path).convert("RGB")
                        row = json.loads(metadata.read_text())
                        matched = _correlation(image, templates[variant["variant"]], inner)
                        other = "rot90" if variant["variant"] == "original" else "original"
                        margin = matched - _correlation(image, templates[other], inner)
                        rows.append({"case": ref["id"], "seed": seed, "variant": variant["variant"],
                                     "identity_similarity": matched, "response_margin": margin,
                                     "contour_f1": row["contour_f1"], "leakage": row["leakage"]})
            by_pair = {(r["case"], r["seed"], r["variant"]): r for r in rows}
            flips = [(case, float(by_pair[case, seed, "original"]["response_margin"] > .05 and
                                  by_pair[case, seed, "rot90"]["response_margin"] > .05))
                     for case in [r["id"] for r in refs] for seed in SEEDS]
            stats = {metric: case_stat([(r["case"], r[metric]) for r in rows]) for metric in
                     ("identity_similarity", "response_margin", "contour_f1", "leakage")}
            stats["pattern_follow"] = case_stat(flips)
            stats["cases"], stats["images"] = len(refs), len(rows)
            results[f"{ptype}_{arm}"] = stats
    stripe = results["stripe_field"]["pattern_follow"]["mean"] >= .75
    plaid = (results["plaid_field"]["identity_similarity"]["mean"] >
             results["plaid_scalar"]["identity_similarity"]["mean"] + .10 and
             results["plaid_field"]["pattern_follow"]["mean"] >=
             results["plaid_scalar"]["pattern_follow"]["mean"])
    repeated = results["repeated_print_field"]["pattern_follow"]["mean"] >= .75
    gates = {"stripe_field_pass": bool(stripe), "plaid_field_pass": bool(plaid),
             "repeat_field_pass": bool(repeated), "frozen_pass": bool(frozen_pass)}
    write_json(base / "report.json", {"methods": results, "gates": gates,
              "limits": "8 synthetic planar pairs per pattern; B pattern similarity is interior grayscale correlation"})
    decision = json.loads((out / "decision_summary.json").read_text())
    decision.update(B_stripe_field_pass=gates["stripe_field_pass"],
                    B_plaid_field_pass=gates["plaid_field_pass"],
                    B_repeat_field_pass=gates["repeat_field_pass"],
                    next_route="real_reference_rectification" if all(gates.values()) else "revise_coordinate_field")
    write_json(out / "decision_summary.json", decision)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stage", choices=("prepare", "generate", "all"), default="all")
    args = parser.parse_args()
    root, out = Path(args.root), Path(args.out)
    torch.set_num_threads(4)
    if args.stage in ("prepare", "all"):
        prepare(root, out)
    if args.stage in ("generate", "all"):
        generate(root, out)


if __name__ == "__main__":
    main()
