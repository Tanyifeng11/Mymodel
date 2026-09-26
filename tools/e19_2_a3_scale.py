"""A3-2：方向门控通过后，隔离预测周期归一化与真值周期诊断。"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from models.identity_geometry_pattern import IdentityGeometryPattern
from models.pattern_canonicalization import fft_orientation, fft_period, rotation_scale
from tools.e15_common import write_json
from tools.e18_joint_alignment import supervised_contrastive, token_vector
from tools.e19_2_a2 import SETTINGS, geometry_evaluation
from tools.e19_2_a3 import evaluate_identity
from tools.e19_2_identity import PATTERNS, encode
from tools.e19_frequency_audit import load_images
from tools.e19_handcrafted import digest


def period_audit(predicted, rows):
    p = predicted.cpu().numpy()
    truth = np.array([r["frequency"] for r in rows])
    result = {"accuracy": float((p == truth).mean()), "mae": float(np.abs(p - truth).mean()),
              "minimum_prediction": float(p.min()), "by_class": {}, "by_angle": {}, "by_frequency": {}}
    for key, field in (("pattern", "by_class"), ("angle", "by_angle"), ("frequency", "by_frequency")):
        for group in sorted({r[key] for r in rows}):
            idx = [i for i, r in enumerate(rows) if r[key] == group]
            result[field][str(group)] = {"accuracy": float((p[idx] == truth[idx]).mean()),
                                       "mae": float(np.abs(p[idx] - truth[idx]).mean())}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root, device = Path(args.root), args.device
    out = root / "e19_2_a3"
    rotation = json.loads((out / "report.json").read_text())
    assert rotation["a3_rotation_pass"], "scale experiment requires rotation gate"
    torch.set_num_threads(4)
    data = json.loads((root / "e19_2_a2/data/manifest.json").read_text())["splits"]
    source = {k: load_images(root / "e19_2_a2/data", rows).to(device) for k, rows in data.items()}
    protocol = {"stage": "A3-2", "arms": ["predicted_period", "oracle_period"], "seeds": [42, 43, 44],
                "baseline": "A3-1 fft_rotation checkpoints, same splits and 500-step training budget",
                "steps": 500, "lr": .0002, "batch": 16, "weight_decay": .0001,
                "initialization": "original E19.1 identity CNN, not an additional 500-step continuation",
                "angle": "same fixed FFT estimate in BOTH arms; no true angle used",
                "period": "new fixed radial distance of strongest 2D FFT power peak rounded to cycles/image; may confuse lattice harmonics, audited separately",
                "target_frequency": 3, "target_source": "minimum TRAIN frequency, never fitted to validation",
                "resampling": "joint rotation and target/predicted frequency affine sampling, then same 90x90 central crop",
                "oracle": "only diagnostic period labels; cannot authorize token swap",
                "loss": "unchanged A2/A3 identity supervised contrastive; original geometry frozen on original images",
                "gate": {"period_readout": "every heldout class frequency accuracy>=.95 and MAE<=.5",
                         "identity": "every seed/set: BA>=.85, no more than .03 below rotation-only; every class recall>=.75; every angle BA>=.80; strict color/angle/scale similarity>=.80; margin>=.30; probe converged",
                         "geometry": "weights and original-input tokens unchanged; direction/period metrics identical to A3-1"}}
    write_json(out / "scale_protocol.json", protocol)
    transformed = {arm: {} for arm in protocol["arms"]}
    audit = {}
    with torch.no_grad():
        for split, images in source.items():
            angle, _ = fft_orientation(images)
            frequency = fft_period(images)
            audit[split] = period_audit(frequency, data[split])
            true_frequency = images.new_tensor([r["frequency"] for r in data[split]])
            transformed["predicted_period"][split] = rotation_scale(images, angle, frequency)
            transformed["oracle_period"][split] = rotation_scale(images, angle, true_frequency)
            write_json(out / ("period_%s.json" % split), [{"texture": r["texture"], "prediction": float(f), "actual": r["frequency"]} for r, f in zip(data[split], frequency)])
    write_json(out / "period_audit.json", audit)
    print("[a3-scale-period]", {k: (v["accuracy"], v["mae"]) for k, v in audit.items()}, flush=True)
    clean = {}
    for name, folder, manifest in (("primary", root / "e18_1/clean", "validation.json"),
                                    ("extra", root / "e19_1/extra_validation", "manifest.json")):
        rows = json.loads((folder / manifest).read_text())
        clean[name] = (load_images(folder, rows).to(device), rows)
    lookup = {(r["pattern"], r["palette"], r["frequency"], r["angle"], r["phase"]): i for i, r in enumerate(data["train"])}
    reports = {}
    for seed in protocol["seeds"]:
        model = IdentityGeometryPattern().to(device)
        model.load_state_dict(torch.load(out / ("fft_rotation_%d.pt" % seed), map_location=device, weights_only=False)["model"])
        before = (digest(model.geometry), digest(model.mapper))
        initial = {k: v.clone() for k, v in model.geometry.cnn.state_dict().items()}
        with torch.inference_mode():
            cached = {k: encode(model.geometry_tokens, source[k]).cpu() for k in ("primary", "extra")}
        arms = {}
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
                    print("[a3-scale-train]", seed, arm, losses[-1], flush=True)
            model.identity.eval().requires_grad_(False)
            with torch.inference_mode():
                evaluation = evaluate_identity(model, transformed[arm], data)
            arms[arm] = {"identity": evaluation, "losses": losses}
            torch.save({"model": model.state_dict(), "seed": seed, "arm": arm, "protocol": protocol}, out / ("scale_%s_%d.pt" % (arm, seed)))
            print("[a3-scale-arm]", seed, arm, {k: v["accuracy"] for k, v in evaluation["splits"].items()}, flush=True)
            del optimizer
        with torch.inference_mode():
            geo = {k: geometry_evaluation(model, x, r) for k, (x, r) in clean.items()}
            changes = {k: float((encode(model.geometry_tokens, source[k]).cpu() - x).abs().max()) for k, x in cached.items()}
        frozen = before == (digest(model.geometry), digest(model.mapper)) and max(changes.values()) == 0
        assert frozen and all(p.grad is None for p in model.geometry.parameters())
        reports[str(seed)] = {"arms": arms, "geometry": geo, "frozen": frozen, "geometry_output_change": changes}
        write_json(out / "scale_partial.json", reports)
    gates = {"period_readout": True, "identity": True, "geometry": True}
    for split in ("primary", "extra"):
        gates["period_readout"] &= all(v["accuracy"] >= .95 and v["mae"] <= .5 for v in audit[split]["by_class"].values())
    for seed, q in reports.items():
        pred = q["arms"]["predicted_period"]["identity"]
        gates["identity"] &= pred["probe_converged"]
        gates["geometry"] &= q["frozen"] and q["geometry"] == rotation["seeds"][seed]["geometry"]
        for split in ("primary", "extra"):
            v = pred["splits"][split]
            base = rotation["seeds"][seed]["arms"]["fft_rotation"]["identity"]["splits"][split]
            gates["identity"] &= bool(v["accuracy"] >= .85 and v["accuracy"] >= base["accuracy"] - .03 and
                                       min(v["recall"]) >= .75 and min(v["by_angle"].values()) >= .8 and
                                       v["geometry"]["all_three_changed_similarity"] >= .8 and v["geometry"]["margin"] >= .3)
    passed = all(gates.values())
    result = {"protocol": protocol, "period_audit": audit, "seeds": reports, "gates": gates,
              "scale_pass": passed, "complete": True,
              "next": "E19.2-B token swap" if passed else "stop before B; inspect predicted versus oracle period and identity normalization"}
    write_json(out / "scale_report.json", result)
    write_json(out / "stage_status.json", {"a3_rotation_pass": True, "scale_executed": True, "scale_pass": passed,
                                           "b_executed": False, "generation_executed": False, "next": result["next"]})
    print("[a3-scale-final]", gates, passed, flush=True)


if __name__ == "__main__":
    main()
