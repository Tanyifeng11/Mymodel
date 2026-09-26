"""A3-1：只旋转归一化；A2 基线、裁剪对照和真值角度诊断分开报告。"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from models.identity_geometry_pattern import IdentityGeometryPattern
from models.pattern_canonicalization import fft_orientation, rotation_only
from tools.e15_common import write_json
from tools.e18_joint_alignment import supervised_contrastive, token_vector
from tools.e19_2_a2 import SETTINGS, geometry_evaluation, identity_evaluation, vectors
from tools.e19_2_identity import PATTERNS, encode
from tools.e19_frequency_audit import load_images
from tools.e19_handcrafted import digest


def angle_audit(angles, confidence, rows):
    truth = torch.tensor([r["angle"] for r in rows], device=angles.device)
    error = torch.abs((angles * 180 / math.pi - truth + 45) % 90 - 45).cpu().numpy()
    result = {"mae_degrees_mod90": float(error.mean()), "p95_degrees_mod90": float(np.quantile(error, .95)),
              "mean_confidence": float(confidence.mean()), "by_class": {}, "by_angle": {}}
    for key, target in (("pattern", "by_class"), ("angle", "by_angle")):
        for group in sorted({r[key] for r in rows}):
            idx = [i for i, r in enumerate(rows) if r[key] == group]
            result[target][str(group)] = {"mae": float(error[idx].mean()), "p95": float(np.quantile(error[idx], .95))}
    return result


def angle_color_similarity(values, rows):
    x = values.cpu().numpy()
    attrs = {key: np.array([r[key] for r in rows]) for key in ("pattern", "palette", "angle", "frequency")}
    select = (attrs["pattern"][:, None] == attrs["pattern"][None, :])
    select &= attrs["frequency"][:, None] == attrs["frequency"][None, :]
    select &= attrs["palette"][:, None] != attrs["palette"][None, :]
    select &= attrs["angle"][:, None] != attrs["angle"][None, :]
    return float((x @ x.T)[select].mean())


def evaluate_identity(model, images, data):
    values = vectors(model, images)
    result = identity_evaluation(values, data)
    for split in ("primary", "extra"):
        result["splits"][split]["same_pattern_cross_color_angle_same_scale"] = angle_color_similarity(values[split], data[split])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root, device = Path(args.root), args.device
    out = root / "e19_2_a3"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    data = json.loads((root / "e19_2_a2/data/manifest.json").read_text())["splits"]
    source = {k: load_images(root / "e19_2_a2/data", rows).to(device) for k, rows in data.items()}
    protocol = {"stage": "A3-1 rotation only", "data": "unchanged E19.2-A2 manifest and splits",
                "arms": ["crop_control", "fft_rotation", "oracle_rotation"], "seeds": [42, 43, 44],
                "steps": 500, "lr": .0002, "batch": 16, "weight_decay": .0001,
                "loss": "same A2 identity supervised contrastive, temperature .12",
                "initialization": "same original E19.1 identity CNN per seed and same sampled batches in all arms",
                "angle_readout": "new fixed fourth circular moment of full 2D FFT, Hann window, radius 2..N/4, angle modulo 90; not a trained capability of the old 36D token",
                "geometry": "original E19.1 geometry tokens on ORIGINAL input; frozen, never canonicalized",
                "sampling": "unit-scale rotation followed by identical 90x90 center crop for every arm; identity fixed resize to 128 unchanged",
                "oracle": "uses synthetic angle ONLY in separately named positive-control arm, cannot authorize A3-2/B",
                "gate": {"rotation": "every seed and heldout set: 45deg BA>=.80 and >=A2+.15; overall BA>=A2+.03; angle/color same-scale cosine>=.80 and >=A2+.05; margin>=.30; all probes converge",
                         "crop_control": "mean paired gain across 3 seeds and 2 splits: overall BA>=.03 and 45deg BA>=.10",
                         "estimator": "each heldout class modulo90 angle p95<=5 degrees",
                         "geometry": "weights and heldout token outputs exactly unchanged; clean direction margin and frequency metrics exactly equal to A2"},
                "scope": "controlled classes; print=chevron prototype; no scale normalization, real fabrics or generation"}
    write_json(out / "protocol.json", protocol)
    transformed = {k: {} for k in protocol["arms"]}
    audit = {}
    with torch.no_grad():
        for split, images in source.items():
            angles, confidence = fft_orientation(images)
            audit[split] = angle_audit(angles, confidence, data[split])
            oracle = torch.tensor([((r["angle"] + 45) % 90 - 45) * math.pi / 180 for r in data[split]], device=device)
            for arm, theta in (("crop_control", torch.zeros_like(angles)), ("fft_rotation", angles), ("oracle_rotation", oracle)):
                transformed[arm][split] = rotation_only(images, theta)
            write_json(out / ("angles_%s.json" % split), [{"texture": r["texture"], "predicted_mod90": float(a * 180 / math.pi), "confidence": float(c)} for r, a, c in zip(data[split], angles, confidence)])
    write_json(out / "angle_audit.json", audit)
    sheet = Image.new("RGB", (4 * 128, 3 * 152), "white")
    draw = ImageDraw.Draw(sheet)
    for col, kind in enumerate(PATTERNS):
        idx = next(i for i, r in enumerate(data["primary"]) if r["pattern"] == kind and r["angle"] == 45 and r["palette"] == 2)
        for row, arm in enumerate(("original", "crop_control", "fft_rotation")):
            x = source["primary"][idx] if arm == "original" else transformed[arm]["primary"][idx]
            image = Image.fromarray((x.cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255).astype("uint8")).resize((128, 128))
            sheet.paste(image, (128 * col, 152 * row))
            draw.text((128 * col, 152 * row + 130), kind + " " + arm, fill="black")
    sheet.save(out / "canonicalization_preview.png")
    print("[a3-angle]", {k: v["p95_degrees_mod90"] for k, v in audit.items()}, flush=True)
    clean = {}
    for name, folder, manifest in (("primary", root / "e18_1/clean", "validation.json"),
                                    ("extra", root / "e19_1/extra_validation", "manifest.json")):
        rows = json.loads((folder / manifest).read_text())
        clean[name] = (load_images(folder, rows).to(device), rows)
    a2 = json.loads((root / "e19_2_a2/report.json").read_text())
    lookup = {(r["pattern"], r["palette"], r["frequency"], r["angle"], r["phase"]): i for i, r in enumerate(data["train"])}
    reports = {}
    for seed in protocol["seeds"]:
        model = IdentityGeometryPattern().to(device)
        model.load_state_dict(torch.load(root / ("e19_2_a2/branches_%d.pt" % seed), map_location=device, weights_only=False)["model"])
        with torch.inference_mode():
            baseline = evaluate_identity(model, source, data)
            cached = {k: encode(model.geometry_tokens, source[k]).cpu() for k in ("primary", "extra")}
        frozen_hash = (digest(model.geometry), digest(model.mapper))
        initial = {k: v.clone() for k, v in model.geometry.cnn.state_dict().items()}
        seed_report = {"a2_baseline": baseline, "arms": {}}
        for arm in protocol["arms"]:
            torch.manual_seed(seed)
            rng = random.Random(seed)
            model.identity.load_state_dict(initial)
            model.identity.train().requires_grad_(True)
            optimizer = torch.optim.AdamW(model.identity.parameters(), lr=.0002, weight_decay=.0001)
            labels = torch.arange(4, device=device).repeat_interleave(4)
            losses = []
            for step in range(1, 501):
                selected = []
                for kind in PATTERNS:
                    fs = rng.sample(list(SETTINGS["train"][0]), 4)
                    angles = rng.sample(list(SETTINGS["train"][2]), 4)
                    for color, f, a in zip(range(4), fs, angles):
                        selected.append(lookup[kind, color, f, a, rng.choice(SETTINGS["train"][1])])
                loss = supervised_contrastive(token_vector(model.identity_tokens(transformed[arm]["train"][selected])), labels)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.identity.parameters(), 1.)
                optimizer.step()
                if step == 1 or step % 100 == 0:
                    losses.append({"step": step, "loss": float(loss.detach())})
                    print("[a3-train]", seed, arm, losses[-1], flush=True)
            model.identity.eval().requires_grad_(False)
            with torch.inference_mode():
                result = evaluate_identity(model, transformed[arm], data)
            seed_report["arms"][arm] = {"identity": result, "losses": losses}
            torch.save({"model": model.state_dict(), "seed": seed, "arm": arm, "protocol": protocol}, out / ("%s_%d.pt" % (arm, seed)))
            print("[a3-arm]", seed, arm, {k: (v["accuracy"], v["by_angle"]["45"]) for k, v in result["splits"].items()}, flush=True)
            del optimizer
        with torch.inference_mode():
            geo = {k: geometry_evaluation(model, x, r) for k, (x, r) in clean.items()}
            changes = {k: float((encode(model.geometry_tokens, source[k]).cpu() - x).abs().max()) for k, x in cached.items()}
        frozen = frozen_hash == (digest(model.geometry), digest(model.mapper)) and max(changes.values()) == 0
        assert frozen and all(p.grad is None for p in model.geometry.parameters())
        seed_report.update(geometry=geo, frozen=frozen, geometry_output_change=changes)
        reports[str(seed)] = seed_report
        write_json(out / "partial.json", reports)
    gates = {"rotation": True, "crop_control": True, "estimator": True, "geometry": True}
    gains, gains45 = [], []
    for split in ("primary", "extra"):
        gates["estimator"] &= all(v["p95"] <= 5 for v in audit[split]["by_class"].values())
    for seed, q in reports.items():
        pred = q["arms"]["fft_rotation"]["identity"]
        gates["rotation"] &= pred["probe_converged"] and q["a2_baseline"]["probe_converged"]
        gates["geometry"] &= q["frozen"] and q["geometry"] == a2["seeds"][seed]["geometry_after"]
        for split in ("primary", "extra"):
            v, base = pred["splits"][split], q["a2_baseline"]["splits"][split]
            ctrl = q["arms"]["crop_control"]["identity"]["splits"][split]
            gates["rotation"] &= bool(v["by_angle"]["45"] >= .8 and v["by_angle"]["45"] >= base["by_angle"]["45"] + .15 and
                                       v["accuracy"] >= base["accuracy"] + .03 and v["same_pattern_cross_color_angle_same_scale"] >= .8 and
                                       v["same_pattern_cross_color_angle_same_scale"] >= base["same_pattern_cross_color_angle_same_scale"] + .05 and v["geometry"]["margin"] >= .3)
            gains.append(v["accuracy"] - ctrl["accuracy"])
            gains45.append(v["by_angle"]["45"] - ctrl["by_angle"]["45"])
    gates["crop_control"] = bool(np.mean(gains) >= .03 and np.mean(gains45) >= .1 and
                                  all(q["arms"]["crop_control"]["identity"]["probe_converged"] for q in reports.values()))
    passed = all(gates.values())
    result = {"protocol": protocol, "angle_audit": audit, "seeds": reports, "gates": gates,
              "mean_gain_over_crop_control": float(np.mean(gains)), "mean_45_gain_over_crop_control": float(np.mean(gains45)),
              "a3_rotation_pass": passed, "complete": True,
              "next": "A3-2 scale normalization" if passed else "stop; inspect estimator/oracle/crop-control before changing identity representation"}
    write_json(out / "report.json", result)
    write_json(out / "stage_status.json", {"a3_rotation_pass": passed, "scale_executed": False, "b_executed": False,
                                           "generation_executed": False, "next": result["next"]})
    print("[a3-final]", gates, passed, flush=True)


if __name__ == "__main__":
    main()
