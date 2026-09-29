"""E25：冻结 E5 的 oracle scaffold、VAE 与 texture ON/OFF 去噪诊断。"""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from garment_mask_utils import mask_backend_info
from models.pattern_scaffold import (RENDERER_VERSION, StripePatternSpec,
                                     apply_garment_mask, render_stripe_field)
from tools import e23_mechanism as e23
from tools.e15_common import write_json
from tools.e18_1_clean_patterns import PALETTES
from tools.e20_utilization import case_stat, load_pipeline
from tools.e22_4_generation import module_hashes
from tools.e22_o4_metrics import axial_distance, measure


SEEDS = (42, 43)
STRENGTHS = {"low": .15, "mid": .35, "high": .55}
GROUPS = tuple(f"{state}_{level}" for state in ("off", "on") for level in STRENGTHS)
SAFETY_TOLERANCE = .02


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_path(root, relative):
    for base in (root / "output_eval", root):
        path = base / relative
        if path.exists():
            return path
    raise FileNotFoundError(relative)


def spec_for(ref, variant):
    palette = PALETTES[ref["palette"]]
    turns = ref["id"] % 2 + int(variant["variant"] == "rot90")
    return StripePatternSpec(int(variant["theta"]), int(ref["frequency"]),
                             float(ref["phase"]), palette[0], palette[1], turns)


def period_signature(image, roi, orientation):
    x, y, w, h = roi
    crop = np.asarray(image.convert("L"), dtype=np.float64)[y:y+h, x:x+w] / 255.
    signal = crop.mean(0 if orientation == 90 else 1)
    signal -= signal.mean()
    energy = float(np.sqrt(np.mean(signal ** 2)))
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal)), n=2048))[1:129]
    spectrum /= max(np.linalg.norm(spectrum), 1e-12)
    return spectrum, energy


def period_readout(image, roi, orientation, templates, target_frequency):
    spectrum, energy = period_signature(image, roi, orientation)
    frequencies = sorted(templates)
    scores = [float(np.dot(spectrum, templates[f])) for f in frequencies]
    valid = energy >= .02
    estimated = frequencies[int(np.argmax(scores))] if valid else None
    return {"estimate": estimated, "valid": bool(valid), "energy": energy,
            "candidate_scores": dict(zip(frequencies, scores)),
            "correct": bool(valid and estimated == target_frequency),
            "log2_error": abs(math.log2(estimated / target_frequency)) if valid else None,
            "metric": "phase-invariant 1D FFT closed-set template, target-normal axis"}


def templates_for(ref, variant, mask, size, roi, frequencies):
    spec = spec_for(ref, variant)
    templates = {}
    for frequency in frequencies:
        candidate = StripePatternSpec(spec.orientation, frequency, spec.phase,
                                      spec.color_a, spec.color_b, spec.quarter_turns)
        image = apply_garment_mask(render_stripe_field(candidate), mask, size)
        templates[frequency] = period_signature(image, roi, spec.orientation)[0]
    return templates


def evaluate_image(image, ref, sketch_row, variant, seed, mask, sketch, reference, templates):
    result = measure(image, mask, sketch, reference, variant["theta"], sketch_row["roi"])
    return {"case": ref["id"], "reference": ref["id"], "sketch": sketch_row["id"],
            "seed": seed, "variant": variant["variant"], "orientation_gt": variant["theta"],
            "frequency_gt": ref["frequency"], "phase_gt": ref["phase"],
            "period": period_readout(image, sketch_row["roi"], int(variant["theta"]),
                                      templates, ref["frequency"]), **result}


def paired(rows):
    index = {(r["reference"], r["sketch"], r["seed"], r["variant"]): r for r in rows}
    result = []
    for (case, sketch, seed, variant), original in index.items():
        if variant != "original":
            continue
        rotated = index[case, sketch, seed, "rot90"]
        readable = original["direction"]["valid"] and rotated["direction"]["valid"]
        delta = axial_distance(original["direction"]["theta"], rotated["direction"]["theta"])
        result.append({"case": case, "sketch": sketch, "seed": seed,
                       "readable": bool(readable), "delta_theta": delta,
                       "flip": float(readable and original["direction"]["correct"]
                                     and rotated["direction"]["correct"] and delta >= 70)})
    return result


def summary(rows, count):
    assert len(rows) == count * len({r["seed"] for r in rows}) * 2
    pairs = paired(rows)
    readable = [r for r in rows if r["direction"]["valid"]]
    readable_pairs = [r for r in pairs if r["readable"]]
    stats = {name: case_stat([(r["case"], value(r)) for r in rows]) for name, value in {
        "orientation_accuracy": lambda r: float(r["direction"]["correct"]),
        "orientation_error": lambda r: r["direction"]["error"],
        "readout_coverage": lambda r: float(r["direction"]["valid"]),
        "period_accuracy": lambda r: float(r["period"]["correct"]),
        "period_coverage": lambda r: float(r["period"]["valid"]),
        "sketch_iou": lambda r: r["sketch_iou"],
        "edge_f1": lambda r: r["edge_f1"],
        "leakage": lambda r: r["leakage"],
    }.items()}
    stats["paired_flip_all"] = case_stat([(r["case"], r["flip"]) for r in pairs])
    stats["paired_coverage"] = case_stat([(r["case"], float(r["readable"])) for r in pairs])
    stats["paired_flip_readable"] = (case_stat([(r["case"], r["flip"]) for r in readable_pairs])
                                      if readable_pairs else None)
    stats["orientation_error_median"] = float(np.median([r["direction"]["error"] for r in rows]))
    stats["orientation_error_readable"] = (case_stat([(r["case"], r["direction"]["error"])
                                                        for r in readable]) if readable else None)
    valid_period = [r for r in rows if r["period"]["valid"]]
    stats["period_log2_error"] = (case_stat([(r["case"], r["period"]["log2_error"])
                                             for r in valid_period]) if valid_period else None)
    stats["cases"], stats["images"], stats["pairs"] = count, len(rows), len(pairs)
    return stats


def safety(rows, baseline):
    contrast = {}
    for key in ("leakage", "sketch_iou", "edge_f1"):
        contrast[key] = case_stat([(r["case"], r[key] - baseline[r["case"], r["seed"], r["variant"]][key])
                                   for r in rows])
    passed = (contrast["leakage"]["ci95"][1] <= SAFETY_TOLERANCE and
              contrast["sketch_iou"]["ci95"][0] >= -SAFETY_TOLERANCE and
              contrast["edge_f1"]["ci95"][0] >= -SAFETY_TOLERANCE)
    return {"passed": bool(passed), "versus_E5": contrast}


def pattern_gate(stats):
    flip = stats["paired_flip_all"]
    return bool(flip["mean"] >= .75 and flip["ci95"][0] > .5)


def strong_gate(stats, safety_report):
    return bool(pattern_gate(stats) and safety_report["passed"])


def load_rows(folder, count):
    rows = [json.loads(path.read_text()) for path in folder.glob("c*.json")
            if int(path.name[1:3]) < count]
    return sorted(rows, key=lambda r: (r["case"], r["seed"], r["variant"]))


def prepare_scaffolds(source, cases, out, size):
    folder = out / "A_oracle"
    folder.mkdir(exist_ok=True)
    frequencies = sorted({ref["frequency"] for ref in cases["references"]})
    rows = []
    for ref, sk in e23.selected(cases, 0, 32):
        mask = Image.open(out / sk["mask"]).convert("L")
        sketch = Image.open(out / sk["path"]).convert("RGB")
        for variant in ref["variants"]:
            name = f"c{ref['id']:02d}_s42_{variant['variant']}"
            path, metadata = folder / f"{name}.png", folder / f"{name}.json"
            spec = spec_for(ref, variant)
            reference = Image.open(out / variant["path"]).convert("RGB")
            rendered = render_stripe_field(spec)
            assert np.array_equal(np.asarray(rendered), np.asarray(reference)), name
            if not path.exists():
                apply_garment_mask(rendered, mask, size).save(path)
            scaffold = Image.open(path).convert("RGB")
            pixels = np.asarray(scaffold)
            assert np.all(pixels[np.asarray(mask) == 0] == 255)
            templates = templates_for(ref, variant, mask, size, sk["roi"], frequencies)
            row = evaluate_image(scaffold, ref, sk, variant, 42, mask, sketch, reference, templates)
            row.update(path=str(path.relative_to(out)), mask_area=float((np.asarray(mask) > 0).mean()),
                       renderer_version=RENDERER_VERSION, source_reference_id=ref["id"],
                       reference_sha256=file_sha(out / variant["path"]),
                       mask_sha256=file_sha(out / sk["mask"]), scaffold_sha256=file_sha(path))
            write_json(metadata, row)
            rows.append(row)
        print("[e25-A]", ref["id"] + 1, "/32", flush=True)
    stats = summary(rows, 32)
    pairs = paired(rows)
    passed = bool(stats["orientation_accuracy"]["mean"] == 1. and
                  stats["paired_flip_all"]["mean"] == 1. and
                  stats["period_accuracy"]["mean"] == 1. and
                  all(r["delta_theta"] >= 70 for r in pairs))
    report = {"gate_pass": passed, "statistics": stats,
              "calibration": "all 32 held-out references; period is closed-set FFT template readout"}
    write_json(folder / "report.json", report)
    return report


@torch.inference_mode()
def audit_vae(pipe, cases, out, size):
    folder = out / "B_vae"
    folder.mkdir(exist_ok=True)
    frequencies = sorted({ref["frequency"] for ref in cases["references"]})
    rows = []
    for ref, sk in e23.selected(cases, 0, 32):
        mask = Image.open(out / sk["mask"]).convert("L")
        sketch = Image.open(out / sk["path"]).convert("RGB")
        for variant in ref["variants"]:
            name = f"c{ref['id']:02d}_s42_{variant['variant']}"
            path, latent_path = folder / f"{name}.png", folder / f"{name}.pt"
            scaffold = Image.open(out / "A_oracle" / f"{name}.png").convert("RGB")
            pixels = to_tensor(scaffold)[None].to(pipe.device, pipe.vae.dtype) * 2 - 1
            z0 = pipe.vae.encode(pixels).latent_dist.mean * pipe.vae.config.scaling_factor
            torch.save(z0.cpu(), latent_path)
            decoded = pipe.vae.decode(z0 / pipe.vae.config.scaling_factor, return_dict=False)[0]
            decoded = (decoded / 2 + .5).clamp(0, 1).float().cpu()[0].permute(1, 2, 0).numpy()
            image = Image.fromarray((decoded * 255).round().astype(np.uint8))
            image.save(path)
            reference = Image.open(out / variant["path"]).convert("RGB")
            templates = templates_for(ref, variant, mask, size, sk["roi"], frequencies)
            row = evaluate_image(image, ref, sk, variant, 42, mask, sketch, reference, templates)
            row.update(path=str(path.relative_to(out)), latent_path=str(latent_path.relative_to(out)),
                       latent_sha256=file_sha(latent_path), image_sha256=file_sha(path))
            write_json(folder / f"{name}.json", row)
            rows.append(row)
        print("[e25-B]", ref["id"] + 1, "/32", flush=True)
    stats = summary(rows, 32)
    passed = bool(stats["orientation_accuracy"]["mean"] >= .95 and
                  stats["paired_flip_all"]["mean"] >= .95 and
                  stats["period_accuracy"]["mean"] >= .9)
    report = {"gate_pass": passed, "statistics": stats,
              "thresholds": {"orientation_accuracy": .95, "paired_flip": .95,
                             "period_accuracy": .9}}
    write_json(folder / "report.json", report)
    return report


def timestep_for(scheduler, strength):
    scheduler.set_timesteps(50)
    remaining = round(50 * strength)
    index = 50 - remaining
    timestep = scheduler.timesteps[index]
    alpha = float(scheduler.alphas_cumprod[int(timestep)])
    return {"strength": strength, "scheduler_index": index, "timestep": int(timestep),
            "remaining_steps": remaining, "alpha_bar": alpha,
            "sigma": math.sqrt(1 - alpha)}


@torch.inference_mode()
def refine_one(pipe, z0, noise, start, sketch, mask, reference, tokens, texture_on, size):
    timestep = torch.tensor([start["timestep"]], device=pipe.device, dtype=torch.long)
    noisy = pipe.scheduler.add_noise(z0, noise, timestep)
    saved_latents, saved_timesteps, saved_tokens = pipe.prepare_latents, pipe.scheduler.set_timesteps, pipe.get_image_embeds

    def clipped_timesteps(steps, *args, **kwargs):
        saved_timesteps(steps, *args, **kwargs)
        assert steps == 50
        pipe.scheduler.timesteps = pipe.scheduler.timesteps[start["scheduler_index"]:]

    pipe.prepare_latents = MethodType(lambda self, *args, **kwargs: noisy.clone(), pipe)
    pipe.scheduler.set_timesteps = clipped_timesteps
    if texture_on:
        pair = e23.tree(tokens, pipe.device)
        pipe.get_image_embeds = MethodType(lambda self, **kwargs: pair, pipe)
    try:
        generated = pipe(prompt="a cloth", null_prompt="", negative_prompt=" worst quality, low quality",
            ref_image=to_tensor(sketch)[None] * 2 - 1,
            texture_clip_image=reference if texture_on else None,
            width=size[0], height=size[1], num_inference_steps=50, guidance_scale=7.,
            sketch_scale=.6, ipa_scale=1., texture_mode="patch_resampled",
            texture_condition_mode="token", texture_preprocess_mode="plain_resize",
            texture_num_tokens=16, texture_scale=1.,
            spatial_mask=to_tensor(mask)[None].to(pipe.device, torch.float16),
            generator=torch.Generator(device=pipe.device).manual_seed(42))[0]
    finally:
        pipe.prepare_latents, pipe.scheduler.set_timesteps, pipe.get_image_embeds = (
            saved_latents, saved_timesteps, saved_tokens)
    return generated, sha(noise.detach().cpu().numpy().tobytes()), sha(noisy.detach().cpu().numpy().tobytes())


def run_group(pipe, cases, banks, out, group, count, baseline, size):
    folder = out / "C_refine" / group
    folder.mkdir(parents=True, exist_ok=True)
    state, level = group.split("_")
    start = timestep_for(pipe.scheduler, STRENGTHS[level])
    frequencies = sorted({ref["frequency"] for ref in cases["references"]})
    for ref, sk in e23.selected(cases, 0, count):
        mask = Image.open(out / sk["mask"]).convert("L")
        sketch = Image.open(out / sk["path"]).convert("RGB")
        for seed in SEEDS:
            for variant in ref["variants"]:
                name = f"c{ref['id']:02d}_s{seed}_{variant['variant']}"
                path, metadata = folder / f"{name}.png", folder / f"{name}.json"
                if metadata.exists():
                    continue
                reference = Image.open(out / variant["path"]).convert("RGB")
                latent_path = out / "B_vae" / f"c{ref['id']:02d}_s42_{variant['variant']}.pt"
                z0 = torch.load(latent_path, map_location=pipe.device, weights_only=False)
                generator = torch.Generator(device=pipe.device).manual_seed(seed)
                noise = torch.randn(z0.shape, generator=generator, device=pipe.device, dtype=z0.dtype)
                image, noise_sha, initial_sha = refine_one(pipe, z0, noise, start, sketch, mask, reference,
                    banks["E5"][ref["id"], variant["variant"]], state == "on", size)
                image.save(path)
                templates = templates_for(ref, variant, mask, size, sk["roi"], frequencies)
                row = evaluate_image(image, ref, sk, variant, seed, mask, sketch, reference, templates)
                row.update(group=group, path=str(path.relative_to(out)),
                           texture_path=state, **start, noise_sha256=noise_sha,
                           initial_latent_sha256=initial_sha,
                           scaffold_sha256=file_sha(out / "A_oracle" / f"c{ref['id']:02d}_s42_{variant['variant']}.png"),
                           latent_sha256=file_sha(latent_path), mask_sha256=file_sha(out / sk["mask"]),
                           reference_sha256=file_sha(out / variant["path"]))
                write_json(metadata, row)
                print("[e25-C]", group, name, flush=True)
    rows = load_rows(folder, count)
    stats = summary(rows, count)
    safety_report = safety(rows, baseline)
    report = {"group": group, "cases": count, "statistics": stats, "safety": safety_report,
              "pattern_pass": pattern_gate(stats),
              "strong_pass": strong_gate(stats, safety_report), "timestep": start}
    write_json(out / "metrics" / f"{group}_{count}.json", report)
    return report


def confirmation_groups(pilot):
    # 预定选择：保留最低噪声 ON/OFF 配对，必要时只扩一个失败对照或转折边界。
    promising = [level for level in STRENGTHS if any(
        pilot[f"{state}_{level}"]["statistics"]["paired_flip_all"]["mean"] >= .75
        for state in ("off", "on"))]
    if not promising:
        return ("off_low", "on_low", "off_mid")
    if promising == ["low"]:
        return ("off_low", "on_low", "off_mid", "on_mid")
    highest = promising[-1]
    return tuple(dict.fromkeys(("off_low", "on_low", f"off_{highest}", f"on_{highest}")))


def paired_initial_audit(out, level, count):
    off = load_rows(out / "C_refine" / f"off_{level}", count)
    on = load_rows(out / "C_refine" / f"on_{level}", count)
    assert len(off) == len(on)
    for a, b in zip(off, on):
        assert (a["case"], a["seed"], a["variant"]) == (b["case"], b["seed"], b["variant"])
        assert a["noise_sha256"] == b["noise_sha256"]
        assert a["initial_latent_sha256"] == b["initial_latent_sha256"]


def decide(confirmed):
    passing = {name for name, report in confirmed.items() if report["pattern_pass"]}
    if any(f"off_{level}" in passing and f"on_{level}" in confirmed and
           f"on_{level}" not in passing for level in STRENGTHS):
        return "spatial_global_arbitration"
    if (any(f"{state}_low" in passing for state in ("off", "on")) and
            "off_mid" in confirmed and "on_mid" in confirmed and
            not any(f"{state}_mid" in passing for state in ("off", "on"))):
        return "pattern_preserving_refinement"
    if any(f"on_{level}" in passing for level in STRENGTHS):
        return "explicit_pattern_correspondence"
    return "pattern_preserving_refinement"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    out = root / "output_eval/e25"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics").mkdir(exist_ok=True)
    torch.set_num_threads(4)
    source = source_path(root, "e23_m")
    previous = source_path(root, "e24")
    cases = json.loads((source / "cases.json").read_text())
    assert len(cases["references"]) == 32
    assert sorted({r["frequency"] for r in cases["references"]}) == [6, 7, 9, 11, 13, 15]
    shutil.copytree(source / "inputs", out / "inputs", dirs_exist_ok=True)
    write_json(out / "cases.json", cases)
    baseline = {(r["case"], r["seed"], r["variant"]): r for r in load_rows(previous / "G0", 32)}
    assert len(baseline) == 128
    size = Image.open(out / cases["sketches"][0]["mask"]).size
    backend = mask_backend_info()
    previous_protocol = json.loads((previous / "protocol.json").read_text())
    assert backend["mask_backend"] == previous_protocol["mask_backend"]["mask_backend"] == "opencv"
    protocol = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                "source_cases_sha256": file_sha(source / "cases.json"),
                "e24_baseline_report_sha256": file_sha(previous / "report.json"),
                "steps": 50, "cfg": 7, "seeds": SEEDS, "strengths": STRENGTHS,
                "pilot_cases": 8, "validation_cases": 32, "training_steps": 0,
                "renderer": RENDERER_VERSION, "mask_backend": backend,
                "period_metric": "closed-set 1D FFT template, six held-out validation frequencies",
                "gate_clean": "orientation=1; paired_flip=1; period_accuracy=1",
                "gate_vae": "orientation>=.95; paired_flip>=.95; period_accuracy>=.9",
                "gate_c": "flip>=.75 and case-bootstrap CI lower>.5, plus E24 safety tolerances .02"}
    write_json(out / "protocol.json", protocol)
    clean = prepare_scaffolds(source, cases, out, size)
    initial = {"oracle_clean_pass": clean["gate_pass"], "vae_pass": None,
               "texture_off_low_pass": None, "texture_off_mid_pass": None,
               "texture_off_high_pass": None, "texture_on_low_pass": None,
               "texture_on_mid_pass": None, "texture_on_high_pass": None,
               "training_steps": 0, "next_route": None}
    write_json(out / "decision_summary.json", initial)
    if not clean["gate_pass"]:
        initial.update(reason="oracle scaffold or readout calibration failed; repair before VAE")
        write_json(out / "decision_summary.json", initial)
        return

    pipe, _, _, modules, width, height = load_pipeline(root, "cuda:0")
    assert (width, height) == size
    pipe.set_progress_bar_config(disable=True)
    frozen = module_hashes(modules)
    token_bank = torch.load(source / "tokens.pt", map_location="cpu", weights_only=False)
    from tools.e23_mechanism import normalize_scheduler
    protocol.update(source_tokens_sha256=file_sha(source / "tokens.pt"),
                    checkpoint_sha256=file_sha(root / "output/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt"),
                    frozen_module_hashes=frozen, scheduler=type(pipe.scheduler).__name__,
                    scheduler_config=normalize_scheduler(pipe))
    write_json(out / "protocol.json", protocol)
    vae = audit_vae(pipe, cases, out, size)
    initial["vae_pass"] = vae["gate_pass"]
    if not vae["gate_pass"]:
        initial.update(next_route="vae_pattern_bottleneck",
                       reason="oracle passes but E5 VAE does not preserve orientation/period")
        initial["frozen_pass"] = module_hashes(modules) == frozen
        write_json(out / "decision_summary.json", initial)
        return

    pilot = {group: run_group(pipe, cases, token_bank, out, group, 8, baseline, size) for group in GROUPS}
    for level in STRENGTHS:
        paired_initial_audit(out, level, 8)
    write_json(out / "metrics" / "pilot.json", pilot)
    chosen = confirmation_groups(pilot)
    confirmed = {group: run_group(pipe, cases, token_bank, out, group, 32, baseline, size)
                 for group in chosen}
    for level in STRENGTHS:
        if f"off_{level}" in confirmed and f"on_{level}" in confirmed:
            paired_initial_audit(out, level, 32)
    write_json(out / "metrics" / "confirmation.json", confirmed)
    for group in GROUPS:
        initial[f"texture_{group}_pass"] = (confirmed[group]["strong_pass"] if group in confirmed else None)
    initial.update(confirmation_groups=chosen, next_route=decide(confirmed),
                   best_flip_accuracy=max(report["statistics"]["paired_flip_all"]["mean"]
                                          for report in confirmed.values()),
                   best_orientation_error=min(report["statistics"]["orientation_error"]["mean"]
                                              for report in confirmed.values()),
                   safety_pass_groups=[g for g, r in confirmed.items() if r["safety"]["passed"]],
                   semantic_pass_groups=[g for g, r in confirmed.items() if r["pattern_pass"]],
                   strong_pass_groups=[g for g, r in confirmed.items() if r["strong_pass"]],
                   frozen_pass=module_hashes(modules) == frozen)
    write_json(out / "decision_summary.json", initial)


if __name__ == "__main__":
    main()
