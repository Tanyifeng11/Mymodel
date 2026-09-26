"""B 周期读出审计：恢复已知频谱坐标，训练集留频率校准；不更新模型。"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LinearRegression

from models.identity_geometry_pattern import IdentityGeometryPattern
from models.pattern_canonicalization import fft_orientation, rotation_only
from tools.e15_common import write_json
from tools.e19_1_frequency import conditioning
from tools.e19_2_b_swap import swap_records, expected_labels
from tools.e19_2_identity import encode
from tools.e19_frequency_audit import load_images
from tools.e19_handcrafted import bf_tokens, digest


def spectral_coordinate(features, threshold):
    """频率 bin 坐标读出；选择与主峰成整数倍关系的显著较低峰。"""
    spectra = features[..., 4:].mean(1).clip(0).reshape(-1, 2, 16)
    coordinates = []
    for axes in spectra:
        estimates, energies = [], []
        for profile in axes:
            strongest = int(profile.argmax()) + 1
            peaks = [i + 1 for i, value in enumerate(profile) if
                     (i == 0 or value >= profile[i - 1]) and (i == 15 or value >= profile[i + 1]) and
                     value >= threshold * profile.max()]
            candidates = [p for p in peaks if 1 <= round(strongest / p) <= 4 and
                          abs(strongest - round(strongest / p) * p) <= .25]
            estimates.append(2 * min(candidates or [strongest]))
            energies.append(np.linalg.norm(profile))
        active = np.array(energies) >= .2 * max(energies)
        coordinates.append(min(np.array(estimates)[active]))
    return np.array(coordinates, dtype=float)[:, None]


def decoded_features(tcpm, text, appearance, identity, geometry, mapper, records):
    result = []
    for start in range(0, len(records), 32):
        batch = records[start:start + 32]
        tokens = torch.cat([appearance[[r["appearance"] for r in batch]], identity[[r["identity"] for r in batch]],
                            geometry[[r["geometry"] for r in batch]]], 1).half()
        final = tcpm(tokens, text.expand(len(batch), -1, -1))
        # 固定正交映射的左逆；不读取 donor 标签，不反解 TCPM 的门控。
        restored = final[:, -4:].float() @ mapper.basis.float().T
        restored /= (mapper.basis.shape[1] ** .5 * mapper.rms)
        result.append(restored.cpu().numpy())
    return np.concatenate(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root, device = Path(args.root), args.device
    out = root / "e19_2_b"
    torch.set_num_threads(4)
    original = json.loads((out / "report.json").read_text())
    data = json.loads((out / "data/manifest.json").read_text())["splits"]
    protocol = {"reason": "generic matched-trained PCA/Ridge period probe fails matched calibration; original B report retained",
                "features": "post-TCPM geometry token segment projected through known frozen mapper transpose into original 36D coordinates",
                "period_rule": "first significant local spectral peak that divides strongest peak with harmonic order<=4; ignore axes below 20% maximum energy",
                "selection": "threshold from [.7,.8,.9,.95,1.] and physical-bin versus affine calibration by train-only leave-one-frequency-out MAE; final affine fit uses train only",
                "scope": "structured geometry-segment readout, not proof that U-Net uses geometry or that pooled generic readout works",
                "gate": "same B donor-attribution thresholds; other attribute predictions reused verbatim; no neural training"}
    write_json(out / "period_audit_protocol.json", protocol)
    source = {k: load_images(out / "data", r).to(device) for k, r in data.items()}
    canonical = {k: rotation_only(x, fft_orientation(x)[0]) for k, x in source.items()}
    pipe, bf, mapper, text, width, height = conditioning(root, device)
    modules = {"bf": bf, "clip": pipe.image_encoder, "tcpm": pipe.tcpm_lite}
    before = {k: digest(v) for k, v in modules.items()}
    appearance = {}
    cache = out / "appearance_cache.pt"
    with torch.inference_mode():
        if cache.exists():
            saved = torch.load(cache, map_location=device, weights_only=False)
            assert saved["bf_hash"] == before["bf"]
            appearance = saved["appearance"]
        else:
            for split, rows in data.items():
                tokens = []
                for row in rows:
                    image = Image.open(out / "data" / row["texture"]).convert("RGB")
                    tokens.append(bf_tokens(pipe, bf, image, text, width, height))
                appearance[split] = torch.cat(tokens)
                print("[period-cache]", split, len(rows), flush=True)
            torch.save({"bf_hash": before["bf"], "appearance": {k: v.cpu() for k, v in appearance.items()}}, cache)
        reports = {}
        for seed in (42, 43, 44):
            model = IdentityGeometryPattern().to(device).eval().requires_grad_(False)
            model.load_state_dict(torch.load(root / ("e19_2_a3/fft_rotation_%d.pt" % seed), map_location=device, weights_only=False)["model"])
            model_hash = digest(model)
            identity = {k: encode(model.identity_tokens, x).half() for k, x in canonical.items()}
            geometry = {k: encode(model.geometry_tokens, x).half() for k, x in source.items()}
            records = [{"appearance": i, "identity": i, "geometry": i} for i in range(len(data["train"]))]
            train = decoded_features(pipe.tcpm_lite, text, appearance["train"], identity["train"], geometry["train"], model.mapper, records)
            targets = np.array([r["frequency"] for r in data["train"]])
            selection = []
            for threshold in (.7, .8, .9, .95, 1.):
                x = spectral_coordinate(train, threshold)
                selection.append({"threshold": threshold, "calibration": "physical_bin", "train_leave_frequency_out_mae": float(np.abs(x[:, 0] - targets).mean())})
                errors = []
                for f in sorted(set(targets)):
                    fit, val = targets != f, targets == f
                    head = LinearRegression().fit(x[fit], targets[fit])
                    errors.extend(abs(head.predict(x[val]) - targets[val]))
                selection.append({"threshold": threshold, "calibration": "affine", "train_leave_frequency_out_mae": float(np.mean(errors))})
            selected = min(selection, key=lambda v: v["train_leave_frequency_out_mae"])
            chosen = selected["threshold"]
            head = LinearRegression().fit(spectral_coordinate(train, chosen), targets)
            result = {"train_only_selection": selection, "selected": selected, "threshold": chosen, "affine_coefficient": head.coef_.tolist(),
                      "affine_intercept": float(head.intercept_), "splits": {}}
            for split in ("primary", "extra"):
                records = swap_records(data[split])
                decoded = decoded_features(pipe.tcpm_lite, text, appearance[split], identity[split], geometry[split], model.mapper, records)
                coordinates = spectral_coordinate(decoded, chosen)
                predictions = coordinates[:, 0] if selected["calibration"] == "physical_bin" else head.predict(coordinates)
                old = json.loads((out / ("records_%d_%s_pooled.json" % (seed, split))).read_text())
                assert len(old) == len(records)
                result["splits"][split] = {}
                for name in sorted({r["intervention"] for r in records}):
                    idx = [i for i, r in enumerate(records) if r["intervention"] == name]
                    error = np.array([abs(predictions[i] - old[i]["expected"]["period"]) for i in idx])
                    joint = []
                    for i in idx:
                        entry = old[i]
                        good = all(entry["predicted"][k] == entry["expected"][k] for k in ("color", "identity"))
                        good &= entry["expected"]["orientation"] < 0 or entry["predicted"]["orientation"] == entry["expected"]["orientation"]
                        joint.append(good and abs(predictions[i] - entry["expected"]["period"]) <= .75)
                    value = dict(original["seeds"][str(seed)]["splits"][split]["pooled"][name])
                    value.update(period_mae=float(error.mean()), period_within_075=float((error <= .75).mean()), joint_success=float(np.mean(joint)))
                    result["splits"][split][name] = value
                write_json(out / ("structured_period_%d_%s.json" % (seed, split)),
                           [dict(record, period_prediction=float(predictions[i])) for i, record in enumerate(records)])
                print("[b-period-audit]", seed, split, {k: round(v["period_mae"], 3) for k, v in result["splits"][split].items()}, flush=True)
            result["model_frozen"] = model_hash == digest(model)
            reports[str(seed)] = result
    frozen = {k: before[k] == digest(v) for k, v in modules.items()}
    assert all(frozen.values()) and all(q["model_frozen"] for q in reports.values())
    passed = all(v["color_accuracy"] >= .9 and v["identity_accuracy"] >= .85 and
                 (v["orientation_accuracy"] is None or v["orientation_accuracy"] >= .9) and v["period_mae"] <= .75 and
                 v["period_within_075"] >= .85 and v["joint_success"] >= .8 for q in reports.values() for s in q["splits"].values() for v in s.values())
    write_json(out / "period_audit_report.json", {"protocol": protocol, "seeds": reports, "freeze_audit": frozen,
                                                  "structured_readout_pass": passed, "complete": True})
    status = json.loads((out / "stage_status.json").read_text())
    status.update(period_audit_complete=True, structured_readout_pass=passed,
                  next="C: U-Net causal test" if passed else "stop before C; final period attribution unresolved")
    write_json(out / "stage_status.json", status)
    print("[b-period-final]", passed, flush=True)


if __name__ == "__main__":
    main()
