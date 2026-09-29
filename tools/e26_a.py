"""E26-A：reference 几何估计、estimated scaffold 与冻结 E5 对照。"""

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from models.pattern_geometry import (GEOMETRY_VERSION, axial_distance,
                                     estimate_pattern_geometry, render_estimated_stripe)
from models.pattern_scaffold import RENDERER_VERSION
from tools import e23_mechanism as e23
from tools import e25_spatial_diagnosis as e25
from tools.e15_common import write_json
from tools.e20_utilization import case_stat, load_pipeline
from tools.e22_4_generation import module_hashes


SEEDS = (42, 43)
COUNT = 32
CONFIDENCE_MIN = {"orientation": .25, "frequency": .20}


def paths(root, out):
    source = e25.source_path(root, "e25")
    cases = json.loads((source / "cases.json").read_text())
    assert len(cases["references"]) == COUNT
    return source, cases


def geometry_stage(root, out):
    source, cases = paths(root, out)
    out.mkdir(parents=True, exist_ok=True)
    folder = out / "A_geometry"
    (folder / "scaffolds").mkdir(parents=True, exist_ok=True)
    shutil.copytree(source / "inputs", out / "inputs", dirs_exist_ok=True)
    write_json(out / "cases.json", cases)
    rows = []
    frequencies = sorted({ref["frequency"] for ref in cases["references"]})
    for ref, sk in e23.selected(cases, 0, COUNT):
        mask = Image.open(out / sk["mask"]).convert("L")
        sketch = Image.open(out / sk["path"]).convert("RGB")
        for variant in ref["variants"]:
            reference_path = out / variant["path"]
            reference = Image.open(reference_path).convert("RGB")
            geometry = estimate_pattern_geometry(reference)
            scaffold = render_estimated_stripe(geometry, mask)
            name = f"c{ref['id']:02d}_s42_{variant['variant']}"
            scaffold_path = folder / "scaffolds" / f"{name}.png"
            scaffold.save(scaffold_path)
            templates = e25.templates_for(ref, variant, mask, mask.size, sk["roi"], frequencies)
            metrics = e25.evaluate_image(scaffold, ref, sk, variant, 42, mask, sketch, reference, templates)
            row = {"case": ref["id"], "variant": variant["variant"], "reference": variant["path"],
                   "geometry": geometry.to_dict(), "orientation_gt": variant["theta"],
                   "frequency_gt": ref["frequency"], "phase_gt": ref["phase"],
                   "orientation_error": axial_distance(geometry.orientation, variant["theta"]),
                   "frequency_log2_error": abs(math.log2(geometry.frequency / ref["frequency"])),
                   "low_confidence": (geometry.orientation_confidence < CONFIDENCE_MIN["orientation"]
                                      or geometry.frequency_confidence < CONFIDENCE_MIN["frequency"]),
                   "scaffold": str(scaffold_path.relative_to(out)), "scaffold_metrics": metrics,
                   "reference_sha256": e25.file_sha(reference_path),
                   "scaffold_sha256": e25.file_sha(scaffold_path),
                   "mask_sha256": e25.file_sha(out / sk["mask"]),
                   "renderer_version": RENDERER_VERSION, "geometry_version": GEOMETRY_VERSION}
            rows.append(row)
        print("[e26-A geometry]", ref["id"] + 1, "/", COUNT, flush=True)
    with (folder / "estimates.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    by_pair = {(row["case"], row["variant"]): row for row in rows}
    rotations = [axial_distance(by_pair[i, "rot90"]["geometry"]["orientation"],
                                by_pair[i, "original"]["geometry"]["orientation"] + 90)
                 for i in range(COUNT)]
    ori = [row["orientation_error"] for row in rows]
    freq = [row["frequency_log2_error"] for row in rows]
    stats = {"orientation_median_error": float(np.median(ori)),
             "orientation_p95_error": float(np.quantile(ori, .95)),
             "rot90_median_error": float(np.median(rotations)),
             "frequency_median_log2_error": float(np.median(freq)),
             "low_confidence_rate": float(np.mean([row["low_confidence"] for row in rows])),
             "scaffold_orientation_accuracy": float(np.mean([r["scaffold_metrics"]["direction"]["correct"] for r in rows])),
             "scaffold_period_accuracy": float(np.mean([r["scaffold_metrics"]["period"]["correct"] for r in rows]))}
    passed = (stats["orientation_median_error"] <= 3 and stats["orientation_p95_error"] <= 8
              and stats["rot90_median_error"] <= 3 and stats["frequency_median_log2_error"] <= .10)
    report = {"gate_pass": bool(passed), "statistics": stats, "cases": COUNT,
              "reference_variants": len(rows), "confidence_min": CONFIDENCE_MIN,
              "estimation_uses_gt_metadata": False}
    write_json(folder / "report.json", report)
    protocol = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                "source_e25_cases_sha256": e25.file_sha(source / "cases.json"),
                "geometry_version": GEOMETRY_VERSION, "renderer_version": RENDERER_VERSION,
                "steps": 50, "cfg": 7, "strength": .15, "texture_path": "on", "seeds": SEEDS,
                "geometry_gate": {"orientation_median_deg": 3, "orientation_p95_deg": 8,
                                  "rot90_median_deg": 3, "frequency_median_log2": .10},
                "end_to_end_gate": {"paired_flip": .90, "ci95_lower_strict": .75,
                                       "generated_orientation_median_deg": 5}}
    write_json(out / "protocol.json", protocol)
    write_json(out / "decision_summary.json", {"A_geometry_pass": bool(passed),
                                               "A_end_to_end_pass": None,
                                               "next_route": "e26_a_refine" if passed else "improve_geometry_estimation"})
    return report


@torch.inference_mode()
def refine_stage(root, out):
    source, cases = paths(root, out)
    geometry_report = json.loads((out / "A_geometry" / "report.json").read_text())
    if not geometry_report["gate_pass"]:
        return
    folder = out / "A_geometry" / "generations"
    folder.mkdir(parents=True, exist_ok=True)
    pipe, _, _, modules, width, height = load_pipeline(root, "cuda:0")
    pipe.set_progress_bar_config(disable=True)
    frozen = module_hashes(modules)
    token_bank_path = e25.source_path(root, "e23_m") / "tokens.pt"
    token_bank = torch.load(token_bank_path, map_location="cpu", weights_only=False)
    protocol_path = out / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol.update(checkpoint_sha256=e25.file_sha(root / "output/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt"),
                    token_bank_sha256=e25.file_sha(token_bank_path), frozen_module_hashes=frozen,
                    vae_hash=frozen["vae"], unet_hash=frozen["unet"],
                    scheduler=type(pipe.scheduler).__name__,
                    scheduler_config=e23.normalize_scheduler(pipe))
    write_json(protocol_path, protocol)
    start = e25.timestep_for(pipe.scheduler, .15)
    frequencies = sorted({ref["frequency"] for ref in cases["references"]})
    oracle_rows = e25.load_rows(source / "C_refine" / "on_low", COUNT)
    oracle_index = {(r["case"], r["seed"], r["variant"]): r for r in oracle_rows}
    assert len(oracle_index) == COUNT * len(SEEDS) * 2
    for ref, sk in e23.selected(cases, 0, COUNT):
        mask = Image.open(out / sk["mask"]).convert("L")
        sketch = Image.open(out / sk["path"]).convert("RGB")
        assert mask.size == (width, height)
        for variant in ref["variants"]:
            reference = Image.open(out / variant["path"]).convert("RGB")
            scaffold_path = out / "A_geometry" / "scaffolds" / f"c{ref['id']:02d}_s42_{variant['variant']}.png"
            scaffold = Image.open(scaffold_path).convert("RGB")
            pixels = to_tensor(scaffold)[None].to(pipe.device, pipe.vae.dtype) * 2 - 1
            z0 = pipe.vae.encode(pixels).latent_dist.mean * pipe.vae.config.scaling_factor
            latent_sha = e25.sha(z0.detach().cpu().numpy().tobytes())
            templates = e25.templates_for(ref, variant, mask, mask.size, sk["roi"], frequencies)
            for seed in SEEDS:
                name = f"c{ref['id']:02d}_s{seed}_{variant['variant']}"
                path, metadata = folder / f"{name}.png", folder / f"{name}.json"
                if metadata.exists() and path.exists():
                    continue
                noise = torch.randn(z0.shape, generator=torch.Generator(device=pipe.device).manual_seed(seed),
                                    device=pipe.device, dtype=z0.dtype)
                image, noise_sha, initial_sha = e25.refine_one(pipe, z0, noise, start, sketch, mask,
                    reference, token_bank["E5"][ref["id"], variant["variant"]], True, mask.size)
                image.save(path)
                row = e25.evaluate_image(image, ref, sk, variant, seed, mask, sketch, reference, templates)
                assert noise_sha == oracle_index[ref["id"], seed, variant["variant"]]["noise_sha256"]
                row.update(path=str(path.relative_to(out)), texture_path="on", **start,
                           noise_sha256=noise_sha, initial_latent_sha256=initial_sha,
                           latent_sha256=latent_sha, scaffold_sha256=e25.file_sha(scaffold_path),
                           mask_sha256=e25.file_sha(out / sk["mask"]),
                           reference_sha256=e25.file_sha(out / variant["path"]),
                           output_sha256=e25.file_sha(path), renderer_version=RENDERER_VERSION,
                           geometry_version=GEOMETRY_VERSION)
                write_json(metadata, row)
                print("[e26-A refine]", name, flush=True)
    rows = e25.load_rows(folder, COUNT)
    stats = e25.summary(rows, COUNT)
    oracle = e25.summary(oracle_rows, COUNT)
    pairs = e25.paired(rows)
    report = {"estimated": stats, "oracle": oracle,
              "contour_f1_estimated": case_stat([(r["case"], r["contour_f1"]) for r in rows]),
              "contour_f1_oracle": case_stat([(r["case"], r["contour_f1"]) for r in oracle_rows]),
              "paired_flip": stats["paired_flip_all"],
              "generated_orientation_median_error": float(np.median([r["direction"]["error"] for r in rows])),
              "period_median_log2_error": float(np.median([r["period"]["log2_error"] for r in rows if r["period"]["valid"]])),
              "coverage": stats["readout_coverage"],
              "frozen_pass": module_hashes(modules) == frozen,
              "paired_cases": len(pairs)}
    report["gate_pass"] = bool(report["paired_flip"]["mean"] >= .90
                               and report["paired_flip"]["ci95"][0] > .75
                               and report["generated_orientation_median_error"] <= 5
                               and report["frozen_pass"])
    write_json(out / "A_geometry" / "end_to_end_report.json", report)
    decision = json.loads((out / "decision_summary.json").read_text())
    decision.update(A_end_to_end_pass=report["gate_pass"],
                    next_route="explicit_coordinate_field" if report["gate_pass"] else "improve_geometry_estimation")
    write_json(out / "decision_summary.json", decision)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stage", choices=("geometry", "refine", "all"), default="all")
    args = parser.parse_args()
    root, out = Path(args.root), Path(args.out)
    torch.set_num_threads(4)
    if args.stage in ("geometry", "all"):
        geometry_stage(root, out)
    if args.stage in ("refine", "all"):
        refine_stage(root, out)


if __name__ == "__main__":
    main()
