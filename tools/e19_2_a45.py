"""A4 变换审计与 A5 纯尺度压力测试；冻结 A3-1 模型和重建后的原探针。"""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, recall_score

from models.harmonic_period import harmonic_period
from models.identity_geometry_pattern import IdentityGeometryPattern
from models.pattern_canonicalization import fft_orientation, rotation_only, rotation_scale
from tools.e14_controlled_patterns import pattern_mask
from tools.e15_common import write_json
from tools.e18_1_clean_patterns import PALETTES
from tools.e18_joint_alignment import token_vector
from tools.e19_2_a2 import geometry_evaluation
from tools.e19_2_a33 import fixed_probe
from tools.e19_2_identity import PATTERNS, encode, motif_mask
from tools.e19_frequency_audit import load_images
from tools.e19_handcrafted import digest


SCALES = (.5, .75, 1., 1.25, 1.5, 2.)


def make_stress(folder):
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in (142, 143, 144):
        phases = np.random.default_rng(seed).uniform(.05, .45, 2)
        for base in (4, 10):
            for phase_id, phase in enumerate(phases):
                for angle in (30, 45, 60):
                    for kind in PATTERNS:
                        for scale in SCALES:
                            frequency = base / scale
                            mask = (motif_mask(256, frequency, angle, phase) if kind == "repeated_print"
                                    else pattern_mask(kind, 256, frequency, angle, phase, fraction=.25))
                            for color, palette in enumerate(PALETTES):
                                image = Image.fromarray(np.where(mask[..., None], palette[0], palette[1]).astype("uint8"))
                                name = "%d_%s_f%d_p%d_a%d_s%.2f_c%d.png" % (seed, kind, base, phase_id, angle, scale, color)
                                image.save(folder / name)
                                rows.append({"texture": name, "seed": seed, "pattern": kind, "base_frequency": base,
                                             "frequency": frequency, "phase": float(phase), "angle": angle,
                                             "scale": scale, "palette": color,
                                             "sha256": hashlib.sha256(image.tobytes()).hexdigest()})
    write_json(folder / "manifest.json", rows)
    return rows


def relations(vectors, rows, scale_key="frequency"):
    x = vectors.numpy()
    sim = x @ x.T
    attrs = {k: np.array([r[k] for r in rows]) for k in ("pattern", "palette", "angle", "phase", scale_key)}
    same = {k: v[:, None] == v[None, :] for k, v in attrs.items()}
    common = same["angle"] & same["phase"]
    if scale_key == "scale":
        base = np.array([r["base_frequency"] for r in rows])
        common &= base[:, None] == base[None, :]
    cross_scale = common & same["pattern"] & same["palette"] & ~same[scale_key]
    cross_color = common & same["pattern"] & ~same["palette"] & same[scale_key]
    different = common & ~same["pattern"] & same["palette"] & same[scale_key]
    pos = float(sim[cross_scale].mean()) if cross_scale.any() else None
    neg = float(sim[different].mean())
    return {"same_pattern_cross_scale_similarity": pos,
            "same_pattern_cross_color_similarity": float(sim[cross_color].mean()),
            "same_color_different_pattern_cosine": neg, "separation": 1 - neg,
            "margin": None if pos is None else pos - neg}


def classify(probe, values, rows):
    truth = np.array([r["pattern"] for r in rows])
    pred = probe.predict(values.numpy())
    result = {"accuracy": float(balanced_accuracy_score(truth, pred)),
              "recall": dict(zip(PATTERNS, recall_score(truth, pred, labels=list(PATTERNS), average=None).tolist()))}
    result["by_angle"] = {}
    for angle in sorted({r["angle"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["angle"] == angle]
        result["by_angle"][str(angle)] = float(balanced_accuracy_score(truth[idx], pred[idx]))
    return result


def source_key(row):
    return tuple(row[k] for k in ("seed", "base_frequency", "phase", "angle", "pattern", "palette"))


def stress_evaluation(probe, vectors, rows):
    anchor = {source_key(r): i for i, r in enumerate(rows) if r["scale"] == 1.}
    report = {"all": relations(vectors, rows, "scale"), "by_scale": {}}
    for scale in SCALES:
        idx = [i for i, r in enumerate(rows) if r["scale"] == scale]
        local_rows = [rows[i] for i in idx]
        value = classify(probe, vectors[idx], local_rows)
        baseline = [anchor[source_key(r)] for r in local_rows]
        value["paired_scale1_similarity"] = float((vectors[idx] * vectors[baseline]).sum(-1).mean())
        value["separation"] = relations(vectors[idx], local_rows, "scale")["separation"]
        report["by_scale"][str(scale)] = value
    normal = [i for i, r in enumerate(rows) if .75 <= r["scale"] <= 1.5]
    report["normal_relations"] = relations(vectors[normal], [rows[i] for i in normal], "scale")
    def passed(scales):
        return all(report["by_scale"][str(s)]["accuracy"] >= .85 and
                   min(report["by_scale"][str(s)]["recall"].values()) >= .75 and
                   report["by_scale"][str(s)]["paired_scale1_similarity"] >= .8 for s in scales)
    report["normal_pass"] = passed((.75, 1., 1.25, 1.5)) and report["normal_relations"]["same_pattern_cross_scale_similarity"] >= .8 and report["normal_relations"]["margin"] >= .3
    report["full_pass"] = passed(SCALES) and report["all"]["same_pattern_cross_scale_similarity"] >= .8 and report["all"]["margin"] >= .3
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root, device = Path(args.root), args.device
    out = root / "e19_2_a45"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    protocol = {"models": "A3-1 fft_rotation seeds 42/43/44; every parameter frozen",
                "probe": "A3-1 original train-only probe deterministically reconstructed once, baseline verified, then fixed across A4/A5",
                "a4": {"S0": "rotation only", "S1": "actual rotation_scale code with scale=1; same warp/crop and encoder resize, no artificial extra resampling",
                       "S2": "oracle period scale normalization", "S3": "harmonic-aware period scale normalization"},
                "a5": "analytic rerender of same template with frequency=base_frequency/magnification; avoids input resize artifacts; orientations 30/45/60 and exact 25% foreground/color balance",
                "a5_scales": SCALES, "a5_base_frequencies": [4, 10], "a5_phase_seeds": [142, 143, 144],
                "gate": "all model seeds: each normal .75/1/1.25/1.5 scale BA>=.85, per-class recall>=.75, anchor cosine>=.8; normal cross-scale cosine>=.8 and identity margin>=.3; geometry unchanged",
                "routing": "normal pass -> B with stated scale range; normal fail -> equal-budget A6; full pass additionally covers .5/2 extremes",
                "limits": "synthetic four-class patterns; print=chevron prototype; no claim about real-garment scale distribution"}
    write_json(out / "protocol.json", protocol)
    data = json.loads((root / "e19_2_a2/data/manifest.json").read_text())["splits"]
    source = {k: load_images(root / "e19_2_a2/data", r).to(device) for k, r in data.items()}
    transformed = {s: {} for s in ("S0", "S1", "S2", "S3")}
    with torch.no_grad():
        for split, images in source.items():
            angle, _ = fft_orientation(images)
            transformed["S0"][split] = rotation_only(images, angle)
            transformed["S1"][split] = rotation_scale(images, angle, images.new_full((len(images),), 3.))
            transformed["S2"][split] = rotation_scale(images, angle, images.new_tensor([r["frequency"] for r in data[split]]))
            transformed["S3"][split] = rotation_scale(images, angle, harmonic_period(images, angle)["scalar"])
    pixel_identity = {k: float((transformed["S0"][k] - transformed["S1"][k]).abs().max()) for k in source}
    stress_rows = make_stress(out / "stress")
    stress = load_images(out / "stress", stress_rows).to(device)
    with torch.no_grad():
        stress = rotation_only(stress, fft_orientation(stress)[0])
    old_report = json.loads((root / "e19_2_a3/report.json").read_text())
    clean = {}
    for name, folder, manifest in (("primary", root / "e18_1/clean", "validation.json"),
                                    ("extra", root / "e19_1/extra_validation", "manifest.json")):
        rows = json.loads((folder / manifest).read_text())
        clean[name] = (load_images(folder, rows).to(device), rows)
    reports = {}
    with torch.inference_mode():
        for seed in (42, 43, 44):
            model = IdentityGeometryPattern().to(device).eval().requires_grad_(False)
            model.load_state_dict(torch.load(root / ("e19_2_a3/fft_rotation_%d.pt" % seed), map_location=device, weights_only=False)["model"])
            frozen_hash = digest(model)
            original = {k: encode(model.identity_tokens, x).cpu() for k, x in transformed["S0"].items()}
            probe, converged = fixed_probe(token_vector(original["train"].to(device)).cpu(), data["train"])
            joblib.dump(probe, out / ("fixed_probe_%d.joblib" % seed))
            audit = {}
            for arm in transformed:
                audit[arm] = {}
                for split in ("primary", "extra"):
                    tokens = original[split] if arm == "S0" else encode(model.identity_tokens, transformed[arm][split]).cpu()
                    v = token_vector(tokens.to(device)).cpu()
                    result = classify(probe, v, data[split])
                    result["relations"] = relations(v, data[split])
                    result["feature_cosine_to_S0"] = float(F.cosine_similarity(tokens.flatten(1), original[split].flatten(1)).mean())
                    result["relative_feature_l2_to_S0"] = float(((tokens - original[split]).flatten(1).norm(dim=-1) / original[split].flatten(1).norm(dim=-1)).mean())
                    if arm == "S0":
                        assert abs(result["accuracy"] - old_report["seeds"][str(seed)]["arms"]["fft_rotation"]["identity"]["splits"][split]["accuracy"]) < 1e-10
                    result["class_accuracy_change_from_S0"] = {k: result["recall"][k] - audit["S0"][split]["recall"][k] for k in PATTERNS} if arm != "S0" else dict.fromkeys(PATTERNS, 0.)
                    audit[arm][split] = result
            stress_vectors = token_vector(encode(model.identity_tokens, stress)).cpu()
            a5 = stress_evaluation(probe, stress_vectors, stress_rows)
            geometry = {k: geometry_evaluation(model, x, r) for k, (x, r) in clean.items()}
            unchanged = frozen_hash == digest(model) and geometry == old_report["seeds"][str(seed)]["geometry"]
            assert unchanged
            reports[str(seed)] = {"a4": audit, "a5": a5, "probe_converged": converged, "frozen": unchanged, "geometry": geometry}
            write_json(out / "partial.json", reports)
            print("[a45-seed]", seed, "normal", a5["normal_pass"], "full", a5["full_pass"], {k: v["accuracy"] for k, v in a5["by_scale"].items()}, flush=True)
    normal = all(q["a5"]["normal_pass"] and q["probe_converged"] and q["frozen"] for q in reports.values())
    full = normal and all(q["a5"]["full_pass"] for q in reports.values())
    result = {"protocol": protocol, "s1_s0_max_pixel_difference": pixel_identity, "seeds": reports,
              "a5_normal_pass": normal, "a5_full_pass": full, "complete": True,
              "next": "B: rotation only, no scale normalization" if normal else "A6: equal-budget scale-invariance training"}
    write_json(out / "report.json", result)
    write_json(out / "stage_status.json", {"a4_complete": True, "a5_complete": True, "a5_normal_pass": normal,
                                           "a5_full_pass": full, "a6_executed": False, "b_executed": False, "next": result["next"]})
    print("[a45-final]", normal, full, flush=True)


if __name__ == "__main__":
    main()
