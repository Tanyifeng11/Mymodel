"""B：冻结三路与 TCPM，在最终 conditioning 上验证 donor attribution，不加载 U-Net。"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models.identity_geometry_pattern import IdentityGeometryPattern
from models.pattern_canonicalization import fft_orientation, rotation_only
from tools.e15_common import write_json
from tools.e18_joint_alignment import token_vector
from tools.e19_1_frequency import conditioning
from tools.e19_2_identity import PATTERNS, encode, make_data
from tools.e19_frequency_audit import load_images
from tools.e19_handcrafted import bf_tokens, digest


TARGETS = ("color", "identity", "orientation", "period")


def swap_records(rows):
    lookup = {(r["frequency"], r["phase"], r["angle"], r["palette"], r["pattern"]): i for i, r in enumerate(rows)}
    fs = sorted({r["frequency"] for r in rows})
    records = []
    for i, r in enumerate(rows):
        f, p, a, c, kind = (r[k] for k in ("frequency", "phase", "angle", "palette", "pattern"))
        def ref(freq=f, angle=a, color=c, pattern=kind):
            return lookup[freq, p, angle, color, pattern]
        def add(name, app, identity, geo):
            records.append({"intervention": name, "original": i, "appearance": app, "identity": identity, "geometry": geo})
        add("matched", i, i, i)
        # 所有有序类别对均出现，包含 A->B 与 B->A。
        for other in PATTERNS:
            if other != kind:
                add("identity_only", i, ref(pattern=other), i)
                donor = ref(freq=fs[(fs.index(f) + 1) % len(fs)], angle=90 - a, color=(c + 1) % 4, pattern=other)
                add("identity_geometry_joint", i, donor, donor)
        # 用条纹作为方向 donor，避免对四重对称的格纹/点阵伪造横竖标签。
        add("geometry_orientation", i, i, ref(angle=90 - a, pattern="stripe"))
        add("geometry_period", i, i, ref(freq=fs[(fs.index(f) + 1) % len(fs)], pattern="stripe"))
        add("appearance_color_only", ref(color=(c + 1) % 4), i, i)
        add("appearance_conflicting", ref(freq=fs[(fs.index(f) + 1) % len(fs)], angle=90 - a,
                                          color=(c + 1) % 4, pattern=PATTERNS[(PATTERNS.index(kind) + 1) % 4]), i, i)
    return records


def expected_labels(rows, records):
    return {"color": np.array([rows[r["appearance"]]["palette"] for r in records]),
            "identity": np.array([rows[r["identity"]]["pattern"] for r in records]),
            "orientation": np.array([rows[r["geometry"]]["angle"] if rows[r["geometry"]]["pattern"] == "stripe" else -1 for r in records]),
            "period": np.array([rows[r["geometry"]]["frequency"] for r in records])}


def condition_vectors(tcpm, text, appearance, identity, geometry, records):
    pooled, owned = [], {k: [] for k in TARGETS}
    n_app, n_id = appearance.shape[1], identity.shape[1]
    for start in range(0, len(records), 32):
        batch = records[start:start + 32]
        tokens = torch.cat([appearance[[r["appearance"] for r in batch]], identity[[r["identity"] for r in batch]],
                            geometry[[r["geometry"] for r in batch]]], 1).half()
        final = tcpm(tokens, text.expand(len(batch), -1, -1))
        pooled.append(token_vector(final).float().cpu().numpy())
        owned["color"].append(token_vector(final[:, :n_app]).float().cpu().numpy())
        owned["identity"].append(token_vector(final[:, n_app:n_app + n_id]).float().cpu().numpy())
        geo = token_vector(final[:, n_app + n_id:]).float().cpu().numpy()
        owned["orientation"].append(geo)
        owned["period"].append(geo)
    return {"pooled": {k: np.concatenate(pooled) for k in TARGETS},
            "owned_segment_diagnostic": {k: np.concatenate(v) for k, v in owned.items()}}


def fit_readouts(features, targets):
    probes, converged = {}, True
    for target in TARGETS:
        idx = np.flatnonzero(targets[target] >= 0) if target == "orientation" else np.arange(len(targets[target]))
        head = Ridge(alpha=1.) if target == "period" else LogisticRegression(C=1., max_iter=10000, random_state=42)
        probe = make_pipeline(StandardScaler(), PCA(n_components=min(32, len(idx) - 1), random_state=42), head)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            probe.fit(features[target][idx], targets[target][idx])
        converged &= not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        probes[target] = probe
    return probes, converged


def score_readouts(probes, features, expected, records, rows):
    predictions = {k: p.predict(features[k]) for k, p in probes.items()}
    report = {}
    for name in sorted({r["intervention"] for r in records}):
        idx = np.array([i for i, r in enumerate(records) if r["intervention"] == name])
        good, scores = [], {}
        for target in TARGETS:
            eligible = idx[expected[target][idx] >= 0] if target == "orientation" else idx
            if target == "period":
                error = np.abs(predictions[target][eligible] - expected[target][eligible])
                scores["period_mae"] = float(error.mean())
                correct = error <= .75
                scores["period_within_075"] = float(correct.mean())
            else:
                correct = predictions[target][eligible] == expected[target][eligible]
                scores[target + "_accuracy"] = float(correct.mean()) if len(eligible) else None
            mask = np.ones(len(records), dtype=bool)
            mask[eligible] = correct
            good.append(mask[idx])
        scores["joint_success"] = float(np.stack(good).all(0).mean())
        scores["count"] = len(idx)
        scores["orientation_count"] = int((expected["orientation"][idx] >= 0).sum())
        report[name] = scores
    # 保存逐样本期望值与预测；目标跟 donor、非目标跟 original 可逐对审计。
    records_out = []
    for i, record in enumerate(records):
        records_out.append(dict(record, expected={k: expected[k][i].item() for k in TARGETS},
                                predicted={k: predictions[k][i].item() for k in TARGETS}))
    return report, records_out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root, device = Path(args.root), args.device
    out = root / "e19_2_b"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    a5 = json.loads((root / "e19_2_a45/report.json").read_text())
    # 保留 A5 严格 gate=False；依据用户允许的能力边界进入受限诊断，不事后改门槛。
    normal_scales = ("0.75", "1.0", "1.25", "1.5")
    bounded = all(q["a5"]["by_scale"][s]["accuracy"] >= .9 for q in a5["seeds"].values() for s in normal_scales)
    bounded &= all(q["a5"]["normal_relations"]["same_pattern_cross_scale_similarity"] >= .9 and q["frozen"] for q in a5["seeds"].values())
    assert bounded, "normal-scale bounded diagnostic requires strong observed accuracy and geometry"
    protocol = {"scope": "conditioning representation only; no U-Net or sampling",
                "route": "A5 bounded normal-scale route; its strict gate remains unchanged",
                "route_caveat": "seed42 dots recall at unscaled 1x is .6875; normal scaling improves it; preserve all seeds and report matched readout failures",
                "identity": "A3-1 fft_rotation checkpoint, NO scale canonicalization",
                "appearance": "frozen existing E18.2 BF appearance tokens, as used in E19-C",
                "geometry": "frozen original E19.1 geometry tokens from ORIGINAL donor reference",
                "interface": "concat BF appearance tokens + 4 identity tokens + 4 geometry tokens, then original frozen TCPM",
                "probes": "train-only matched references; final pooled mean/std primary, owned token segment probes diagnostic; orientation fitted only on unambiguous stripes; period Ridge continuous cycles/image",
                "interventions": "all ordered identity swaps; geometry orientation/period separately; appearance color-only and conflicting donor; joint identity+geometry swap",
                "gate": "both sets and all seeds: matched and all interventions color>=.90 identity>=.85 orientation>=.90 where defined; period MAE<=.75 and within-.75 rate>=.85; joint>=.80; probes converge; all weights unchanged",
                "limits": "controlled 0/90 degrees for attribution; no claim about ambiguous plaid/dot orientation or real garment scales; not directly comparable with old E19-C 62.5% simultaneous-reference replacement"}
    write_json(out / "protocol.json", protocol)
    settings = {"train": ((3, 5, 8, 12), (0., .23, .47), (0, 90)),
                "primary": ((4, 10), (.11, .36), (0, 90)), "extra": ((6, 14), (.07, .19, .41), (0, 90))}
    data = make_data(out / "data", settings)
    source = {k: load_images(out / "data", r).to(device) for k, r in data.items()}
    canonical = {k: rotation_only(x, fft_orientation(x)[0]) for k, x in source.items()}
    pipe, bf, mapper, text, width, height = conditioning(root, device)
    frozen_modules = {"bf": bf, "clip": pipe.image_encoder, "tcpm": pipe.tcpm_lite}
    frozen_hashes = {k: digest(v) for k, v in frozen_modules.items()}
    appearance = {}
    with torch.inference_mode():
        for split, rows in data.items():
            tokens = []
            for i, row in enumerate(rows):
                image = Image.open(out / "data" / row["texture"]).convert("RGB")
                tokens.append(bf_tokens(pipe, bf, image, text, width, height))
                if (i + 1) % 64 == 0:
                    print("[b-appearance]", split, i + 1, flush=True)
            appearance[split] = torch.cat(tokens)
    reports = {}
    with torch.inference_mode():
        for seed in (42, 43, 44):
            model = IdentityGeometryPattern().to(device).eval().requires_grad_(False)
            model.load_state_dict(torch.load(root / ("e19_2_a3/fft_rotation_%d.pt" % seed), map_location=device, weights_only=False)["model"])
            before = digest(model)
            identity = {k: encode(model.identity_tokens, x).half() for k, x in canonical.items()}
            geometry = {k: encode(model.geometry_tokens, x).half() for k, x in source.items()}
            train_records = [{"intervention": "matched", "original": i, "appearance": i, "identity": i, "geometry": i} for i in range(len(data["train"]))]
            train_targets = expected_labels(data["train"], train_records)
            train_features = condition_vectors(pipe.tcpm_lite, text, appearance["train"], identity["train"], geometry["train"], train_records)
            probes = {view: fit_readouts(f, train_targets) for view, f in train_features.items()}
            result = {"probe_converged": {k: value[1] for k, value in probes.items()}, "splits": {}}
            for split in ("primary", "extra"):
                records = swap_records(data[split])
                labels = expected_labels(data[split], records)
                features = condition_vectors(pipe.tcpm_lite, text, appearance[split], identity[split], geometry[split], records)
                result["splits"][split] = {}
                for view in probes:
                    scores, details = score_readouts(probes[view][0], features[view], labels, records, data[split])
                    result["splits"][split][view] = scores
                    write_json(out / ("records_%d_%s_%s.json" % (seed, split, view)), details)
                print("[b-swap]", seed, split, {k: round(v["joint_success"], 4) for k, v in result["splits"][split]["pooled"].items()}, flush=True)
            result["identity_geometry_frozen"] = before == digest(model)
            reports[str(seed)] = result
            write_json(out / "partial.json", reports)
    freeze = {k: frozen_hashes[k] == digest(v) for k, v in frozen_modules.items()}
    assert all(freeze.values()) and all(q["identity_geometry_frozen"] for q in reports.values())
    passed = all(q["probe_converged"]["pooled"] and all(v["color_accuracy"] >= .9 and v["identity_accuracy"] >= .85 and
                 (v["orientation_accuracy"] is None or v["orientation_accuracy"] >= .9) and v["period_mae"] <= .75 and
                 v["period_within_075"] >= .85 and v["joint_success"] >= .8
                 for s in q["splits"].values() for v in s["pooled"].values()) for q in reports.values())
    result = {"protocol": protocol, "seeds": reports, "freeze_audit": freeze, "b_pass": passed, "complete": True,
              "next": "C: U-Net causal and matched-wrong test" if passed else "stop before C; final-conditioning donor attribution not established"}
    write_json(out / "report.json", result)
    write_json(out / "stage_status.json", {"b_complete": True, "b_pass": passed, "c_executed": False, "generation_executed": False, "next": result["next"]})
    route = json.loads((root / "e19_2_a45/stage_status.json").read_text())
    route.update(b_executed=True, a6_executed=False, next=result["next"],
                 route_decision="bounded normal-scale diagnostic; strict A5 gate remains false because seed42 unscaled dots recall is below .75, while ordinary scaling improves that recall",
                 b_report="../e19_2_b/report.json")
    write_json(root / "e19_2_a45/stage_status.json", route)
    print("[b-final]", passed, flush=True)


if __name__ == "__main__":
    main()
