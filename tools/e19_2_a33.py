"""A3-3：先诊断横纵基频，再冻结 A3-1 模型进行三种尺度处理的配对复测。"""

import argparse
import hashlib
import json
import math
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models.harmonic_period import harmonic_period
from models.identity_geometry_pattern import IdentityGeometryPattern
from models.pattern_canonicalization import fft_orientation, fft_period, rotation_only, rotation_scale
from tools.e15_common import write_json
from tools.e18_1_clean_patterns import PALETTES
from tools.e18_joint_alignment import token_vector
from tools.e19_2_a2 import geometry_evaluation, identity_evaluation
from tools.e19_2_identity import PATTERNS, encode
from tools.e19_frequency_audit import load_images
from tools.e19_handcrafted import digest


def diagnostic_data(folder):
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    y, x = np.mgrid[:256, :256].astype(float)
    cases = [("stripe", f, 0) for f in (4, 6, 8, 10, 14)]
    cases += [("plaid", u, v) for u, v in ((4, 4), (8, 8), (4, 8), (8, 4), (4, 6), (6, 10), (10, 14), (14, 4))]
    for seed in (42, 43, 44):
        rng = np.random.default_rng(seed)
        phases = rng.uniform(.04, .46, (3, 2))
        for kind, fu, fv in cases:
            for angle in (0, 45, 90):
                a = np.deg2rad(angle)
                for phase_id, (pu, pv) in enumerate(phases):
                    u = (x * np.cos(a) + y * np.sin(a)) * fu / 256 + pu
                    v = (-x * np.sin(a) + y * np.cos(a)) * fv / 256 + pv
                    du, dv = np.abs((u + .5) % 1 - .5), np.abs((v + .5) % 1 - .5)
                    score = du if kind == "stripe" else np.minimum(du, dv)
                    mask = np.zeros(256 * 256, dtype=bool)
                    mask[np.argsort(score.ravel(), kind="stable")[:16384]] = True
                    mask = mask.reshape(256, 256)
                    for color, palette in enumerate(PALETTES):
                        image = Image.fromarray(np.where(mask[..., None], palette[0], palette[1]).astype("uint8"))
                        name = "%d_%s_u%d_v%d_a%d_p%d_c%d.png" % (seed, kind, fu, fv, angle, phase_id, color)
                        image.save(folder / name)
                        rows.append({"texture": name, "seed": seed, "pattern": kind, "fu": fu, "fv": fv,
                                     "angle": angle, "phase_u": float(pu), "phase_v": float(pv), "palette": color,
                                     "sha256": hashlib.sha256(image.tobytes()).hexdigest()})
    write_json(folder / "manifest.json", rows)
    return rows


def diagnostic_metrics(prediction, active, expected):
    selected = expected > 0
    effective = np.where(active, prediction, 0)
    p, y = effective[selected], expected[selected]
    harmonic = (p == 2 * y) | (2 * p == y)
    result = {"active_axis_accuracy": float((p == y).mean()),
              "joint_xy_accuracy": float((effective == expected).all(-1).mean()),
              "frequency_mae_cycles_per_image": float(np.abs(p - y).mean()),
              "period_mae_pixels": float(np.abs(128 / np.maximum(p, 1) - 128 / y).mean()),
              "harmonic_error_rate": float(harmonic.mean()),
              "four_to_eight_count": int(((y == 4) & (p == 8)).sum()),
              "eight_to_four_count": int(((y == 8) & (p == 4)).sum()),
              "axis": {}}
    for axis in range(2):
        idx = selected[:, axis]
        if idx.any():
            result["axis"]["x" if axis == 0 else "y"] = float((effective[idx, axis] == expected[idx, axis]).mean())
    return result


def run_diagnostic(out, device):
    rows = diagnostic_data(out / "diagnostic")
    images = load_images(out / "diagnostic", rows).to(device)
    angle, _ = fft_orientation(images)
    pred = harmonic_period(images, angle)
    xy, active = pred["xy"].cpu().numpy(), pred["active"].cpu().numpy()
    # 标签只在评估中决定 canonical x/y 轴的对应；估计器未接收标签。
    quarters = np.rint((np.array([r["angle"] for r in rows]) - angle.cpu().numpy() * 180 / math.pi) / 90).astype(int)
    expected = np.array([[r["fu"], r["fv"]] if q % 2 == 0 else [r["fv"], r["fu"]] for r, q in zip(rows, quarters)])
    result = {"all": diagnostic_metrics(xy, active, expected), "by_seed_class": {}, "by_angle": {}}
    for seed in (42, 43, 44):
        result["by_seed_class"][str(seed)] = {}
        for kind in ("stripe", "plaid"):
            idx = [i for i, r in enumerate(rows) if r["seed"] == seed and r["pattern"] == kind]
            result["by_seed_class"][str(seed)][kind] = diagnostic_metrics(xy[idx], active[idx], expected[idx])
    for a in (0, 45, 90):
        idx = [i for i, r in enumerate(rows) if r["angle"] == a]
        result["by_angle"][str(a)] = diagnostic_metrics(xy[idx], active[idx], expected[idx])
    # 原最强峰只定义了一个标量，不能伪造不同横纵周期的旧基线。
    old = fft_period(images).cpu().numpy()
    iso = [i for i, r in enumerate(rows) if r["pattern"] == "stripe" or r["fu"] == r["fv"]]
    result["old_peak_scalar_eligible_accuracy"] = float(np.mean([old[i] == rows[i]["fu"] for i in iso]))
    result["scalar_baseline_scope"] = "stripe and isotropic plaid only; old radial argmax cannot estimate anisotropic xy"
    result["records"] = [dict(row, predicted_xy=xy[i].tolist(), active=active[i].tolist(), expected_xy=expected[i].tolist(),
                               window_agreement=pred["window_agreement"][i].cpu().tolist()) for i, row in enumerate(rows)]
    passed = all(v["joint_xy_accuracy"] >= .95 and v["active_axis_accuracy"] >= .95 and
                 v["frequency_mae_cycles_per_image"] <= .5 and v["harmonic_error_rate"] <= .01
                 for q in result["by_seed_class"].values() for v in q.values())
    result["pass"] = passed
    write_json(out / "diagnostic_report.json", result)
    print("[a33-diagnostic]", passed, result["all"], flush=True)
    return result


def fixed_probe(train, rows):
    probe = make_pipeline(StandardScaler(), PCA(n_components=32, random_state=42), LogisticRegression(C=1., max_iter=10000, random_state=42))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        probe.fit(train.numpy(), [r["pattern"] for r in rows])
    return probe, not any(issubclass(w.category, ConvergenceWarning) for w in caught)


def fixed_scores(probe, values, rows):
    labels = np.array([r["pattern"] for r in rows])
    pred = probe.predict(values.numpy())
    result = {"accuracy": float(balanced_accuracy_score(labels, pred)),
              "recall": recall_score(labels, pred, labels=list(PATTERNS), average=None).tolist(), "by_angle": {}}
    for angle in (30, 45, 60):
        idx = [i for i, r in enumerate(rows) if r["angle"] == angle]
        result["by_angle"][str(angle)] = float(balanced_accuracy_score(labels[idx], pred[idx]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root, device = Path(args.root), args.device
    out = root / "e19_2_a33"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    protocol = {"trainable_parameters": "none", "identity_checkpoint": "A3-1 fft_rotation for each seed 42/43/44, frozen identically for all arms",
                "estimator": "A3-1 fixed rotation, axis-wise mean removal; full plus 3 perpendicular strips; first positive local correlation maximum >=.60 and >=.80 of best, quadratic lag refinement and median vote",
                "development": "only former training frequencies 3/5/8/12 and angles 0/15/75/90/105/165; no heldout tuning",
                "diagnostic": "3 phase seeds, 4 colors, 0/45/90 degrees, stripe and unequal-axis plaid including 4/8 reversals",
                "diagnostic_gate": "every seed/class joint xy and active-axis accuracy>=.95, frequency MAE<=.5, harmonic error<=.01",
                "arms": ["no_scale", "peak_scale", "harmonic_scale"],
                "probes": "primary same fixed probe fit on no-scale training vectors; per-arm refitted probe is diagnostic only; neither updates encoder",
                "representation_gate": "every seed/set fixed-probe BA>=.85 and no more than .03 below no-scale; every class recall>=.75; every angle BA>=.80; strict color/angle/scale similarity>=.80; period accuracy>=.95; original geometry exactly unchanged",
                "scope": "controlled representation; appearance/BF/U-Net not loaded; no generation"}
    write_json(out / "protocol.json", protocol)
    with torch.inference_mode():
        diagnostic = run_diagnostic(out, device)
    if not diagnostic["pass"]:
        write_json(out / "stage_status.json", {"diagnostic_pass": False, "scale_retest_executed": False, "b_executed": False,
                                               "next": "stop at period estimator diagnostic"})
        return
    data = json.loads((root / "e19_2_a2/data/manifest.json").read_text())["splits"]
    source = {k: load_images(root / "e19_2_a2/data", r).to(device) for k, r in data.items()}
    transformed = {k: {} for k in protocol["arms"]}
    periods = {}
    with torch.inference_mode():
        for split, images in source.items():
            angles, _ = fft_orientation(images)
            old, new = fft_period(images), harmonic_period(images, angles)["scalar"]
            periods[split] = {"old_accuracy": float((old == images.new_tensor([r["frequency"] for r in data[split]])).float().mean()),
                              "new_accuracy": float((new == images.new_tensor([r["frequency"] for r in data[split]])).float().mean())}
            transformed["no_scale"][split] = rotation_only(images, angles)
            transformed["peak_scale"][split] = rotation_scale(images, angles, old)
            transformed["harmonic_scale"][split] = rotation_scale(images, angles, new)
    write_json(out / "existing_set_periods.json", periods)
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
            before = digest(model)
            cached = {k: encode(model.geometry_tokens, source[k]).cpu() for k in ("primary", "extra")}
            values = {arm: {k: token_vector(encode(model.identity_tokens, x)).cpu() for k, x in images.items()} for arm, images in transformed.items()}
            probe, converged = fixed_probe(values["no_scale"]["train"], data["train"])
            arms = {}
            for arm, vector in values.items():
                refit = identity_evaluation(vector, data)
                fixed = {k: fixed_scores(probe, vector[k], data[k]) for k in ("primary", "extra")}
                arms[arm] = {"fixed_probe": fixed, "refit_diagnostic": refit}
                print("[a33-retest]", seed, arm, {k: v["accuracy"] for k, v in fixed.items()}, flush=True)
            geo = {k: geometry_evaluation(model, x, r) for k, (x, r) in clean.items()}
            changes = {k: float((encode(model.geometry_tokens, source[k]).cpu() - x).abs().max()) for k, x in cached.items()}
            frozen = before == digest(model) and max(changes.values()) == 0
            assert frozen
            reports[str(seed)] = {"arms": arms, "fixed_probe_converged": converged, "frozen": frozen,
                                  "geometry": geo, "geometry_output_change": changes}
    gates = {"period": all(periods[k]["new_accuracy"] >= .95 for k in ("primary", "extra")), "identity": True, "geometry": True}
    for seed, q in reports.items():
        gates["identity"] &= q["fixed_probe_converged"]
        gates["geometry"] &= q["frozen"] and q["geometry"] == old_report["seeds"][seed]["geometry"]
        for split in ("primary", "extra"):
            v = q["arms"]["harmonic_scale"]["fixed_probe"][split]
            base = q["arms"]["no_scale"]["fixed_probe"][split]
            sim = q["arms"]["harmonic_scale"]["refit_diagnostic"]["splits"][split]["geometry"]["all_three_changed_similarity"]
            gates["identity"] &= bool(v["accuracy"] >= .85 and v["accuracy"] >= base["accuracy"] - .03 and min(v["recall"]) >= .75 and
                                       min(v["by_angle"].values()) >= .8 and sim >= .8)
    passed = all(gates.values())
    write_json(out / "report.json", {"protocol": protocol, "diagnostic_pass": True, "periods": periods,
                                     "seeds": reports, "gates": gates, "a33_pass": passed, "complete": True})
    write_json(out / "stage_status.json", {"diagnostic_pass": True, "scale_retest_executed": True, "a33_pass": passed,
                                           "b_executed": False, "generation_executed": False,
                                           "next": "E19.2-B bidirectional swap" if passed else "stop; inspect frozen identity scale response"})
    print("[a33-final]", gates, passed, flush=True)


if __name__ == "__main__":
    main()
