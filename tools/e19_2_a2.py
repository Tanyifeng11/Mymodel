"""E19.2-A2：独立身份 CNN + 冻结 E19.1 几何分支，仅做表示层门控。"""

import argparse
import json
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models.explicit_pattern import pattern_features
from models.identity_geometry_pattern import IdentityGeometryPattern
from tools.e15_common import write_json
from tools.e18_joint_alignment import supervised_contrastive, token_vector
from tools.e19_2_identity import PATTERNS, encode, identity_geometry, make_data
from tools.e19_frequency_audit import evaluate, load_images
from tools.e19_handcrafted import digest, summary


SETTINGS = {
    "train": ((3, 5, 8, 12), (0., .23, .47), (0, 15, 75, 90, 105, 165)),
    "primary": ((4, 10), (.11, .36), (30, 45, 60)),
    "extra": ((6, 14), (.07, .19, .41), (30, 45, 60)),
}


def vectors(model, images):
    return {k: token_vector(encode(model.identity_tokens, v)).cpu() for k, v in images.items()}


def identity_evaluation(values, rows):
    probe = make_pipeline(StandardScaler(), PCA(n_components=32, random_state=42),
                          LogisticRegression(C=1., max_iter=10000, random_state=42))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        probe.fit(values["train"].numpy(), [r["pattern"] for r in rows["train"]])
    result = {"probe_converged": not any(issubclass(w.category, ConvergenceWarning) for w in caught),
              "probe_iterations": probe[-1].n_iter_.tolist(), "splits": {}}
    for split in ("primary", "extra"):
        data, x = rows[split], values[split]
        labels = np.array([r["pattern"] for r in data])
        predictions = probe.predict(x.numpy())
        similarity = (x @ x.T).numpy()
        same = labels[:, None] == labels[None, :]
        # 正样本必须同时跨 angle、color、scale，不能只靠同模板重着色通过。
        positive = same.copy()
        for key in ("angle", "palette", "frequency"):
            attribute = np.array([r[key] for r in data])
            positive &= attribute[:, None] != attribute[None, :]
        geometry = identity_geometry(x, data)
        geometry["all_three_changed_similarity"] = float(similarity[positive].mean())
        geometry["all_three_changed_pairs"] = int(positive.sum())
        value = {"accuracy": float(balanced_accuracy_score(labels, predictions)),
                 "recall": recall_score(labels, predictions, labels=list(PATTERNS), average=None).tolist(),
                 "geometry": geometry}
        for key in ("angle", "frequency"):
            value["by_" + key] = {}
            for group in sorted({r[key] for r in data}):
                idx = [i for i, row in enumerate(data) if row[key] == group]
                value["by_" + key][str(group)] = float(balanced_accuracy_score(labels[idx], predictions[idx]))
        result["splits"][split] = value
    return result


def geometry_evaluation(model, images, rows):
    output = encode(model.geometry, images)
    target = pattern_features(images)
    guessed = [dict(row, axis="vertical" if bool(p[:, 0].mean() > p[:, 1].mean()) else "horizontal")
               for p, row in zip(output, rows)]
    return {"direction": summary(token_vector(model.mapper(output)).cpu(), rows),
            "frequency": evaluate(output, target, rows)["aggregate"],
            "predicted_axis_frequency": evaluate(output, target, guessed)["aggregate"]}


def interventions(model, images, rows):
    identity = encode(model.identity_tokens, images)
    geometry = encode(model.geometry_tokens, images)
    rotated = torch.rot90(images, 1, (-2, -1))
    identity_rot = encode(model.identity_tokens, rotated)
    geometry_rot = encode(model.geometry_tokens, rotated)
    # 旋转使用精确像素 rot90。格纹/点阵可有四重对称，几何变化门控只用于条纹。
    iv, ir = token_vector(identity), token_vector(identity_rot)
    gv, gr = token_vector(geometry), token_vector(geometry_rot)
    stripe = [i for i, row in enumerate(rows) if row["pattern"] == "stripe"]
    id_cos = (iv * ir).sum(-1)
    geo_distance = 1 - (gv * gr).sum(-1)
    return {"identity_rotation_cosine": float(id_cos.mean()),
            "identity_rotation_cosine_by_class": {k: float(id_cos[[i for i, r in enumerate(rows) if r["pattern"] == k]].mean()) for k in PATTERNS},
            "stripe_identity_rotation_distance": float((1 - id_cos[stripe]).mean()),
            "stripe_geometry_rotation_distance": float(geo_distance[stripe].mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root, device = Path(args.root), args.device
    out = root / "e19_2_a2"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    data = make_data(out / "data", SETTINGS)
    for split in ("primary", "extra"):
        for key in ("angle", "frequency", "phase"):
            assert not ({r[key] for r in data["train"]} & {r[key] for r in data[split]})
    protocol = {
        "classes": PATTERNS, "settings": SETTINGS, "counts": {k: len(v) for k, v in data.items()},
        "seeds": [42, 43, 44], "steps": 500, "batch": 16, "lr": .0002, "weight_decay": .0001,
        "initialization": "independent identity CNN copied from E19.1 CNN; geometry restored from original E19.1 fft_bias, never E19.2-A",
        "trainable": "identity CNN only; no shared storage with geometry",
        "tokens": "four 768D identity tokens and four 768D geometry tokens, separately retained",
        "loss": "identity supervised contrastive temperature .12, no orientation/frequency/generator losses",
        "positive_batch": "each class has four distinct colors, four distinct frequencies, four distinct angles; phases sampled from train only",
        "gate": {"identity": "every seed and split: BA>=.85; improvement over same initialization>=.05; every class recall>=.75; every heldout angle BA>=.80; strict three-attribute invariance cosine>=.80; identity margin>=.30",
                 "geometry": "weights/output exactly unchanged; primary+extra clean direction margin>=99% initial, positive in every frequency/phase; frequency and predicted-axis frequency accuracy>=.99",
                 "independence": "each class mean identity rot90 cosine>=.90; stripe geometry cosine-distance>=.05 and identity distance<=half geometry distance"},
        "scope": "controlled representation only; print=chevron prototype; no BF/TCPM/U-Net, real fabrics or generation",
        "comparison_limit": "new strict split excludes 30/45/60 from identity training; earlier A trained at 30 and is not a strict paired baseline",
    }
    write_json(out / "protocol.json", protocol)
    images = {k: load_images(out / "data", rows).to(device) for k, rows in data.items()}
    clean_sets = {}
    for name, folder, manifest in (("primary", root / "e18_1/clean", "validation.json"),
                                    ("extra", root / "e19_1/extra_validation", "manifest.json")):
        rows = json.loads((folder / manifest).read_text())
        clean_sets[name] = (load_images(folder, rows).to(device), rows)
    lookup = {(r["pattern"], r["palette"], r["frequency"], r["angle"], r["phase"]): i for i, r in enumerate(data["train"])}
    reports = {}
    for seed in protocol["seeds"]:
        torch.manual_seed(seed)
        rng = random.Random(seed)
        model = IdentityGeometryPattern().to(device)
        old = torch.load(root / ("e19_1/fft_bias/encoder_%d.pt" % seed), map_location=device, weights_only=False)
        model.geometry.load_state_dict(old["encoder"])
        model.identity.load_state_dict(model.geometry.cnn.state_dict())
        model.mapper.load_state_dict(torch.load(root / "e19/a/pattern_branch.pt", map_location=device, weights_only=False)["mapper"])
        assert not ({p.data_ptr() for p in model.identity.parameters()} & {p.data_ptr() for p in model.geometry.parameters()})
        frozen_hashes = (digest(model.geometry), digest(model.mapper))
        with torch.inference_mode():
            initial = identity_evaluation(vectors(model, images), data)
            geo_before = {k: geometry_evaluation(model, x, r) for k, (x, r) in clean_sets.items()}
            cached = {k: encode(model.geometry_tokens, x).cpu() for k, x in images.items() if k != "train"}
        optimizer = torch.optim.AdamW(model.identity.parameters(), lr=.0002, weight_decay=.0001)
        labels = torch.arange(4, device=device).repeat_interleave(4)
        losses = []
        for step in range(1, 501):
            selected = []
            for kind in PATTERNS:
                frequencies = rng.sample(list(SETTINGS["train"][0]), 4)
                angles = rng.sample(list(SETTINGS["train"][2]), 4)
                for color, frequency, angle in zip(range(4), frequencies, angles):
                    selected.append(lookup[kind, color, frequency, angle, rng.choice(SETTINGS["train"][1])])
            loss = supervised_contrastive(token_vector(model.identity_tokens(images["train"][selected])), labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.identity.parameters(), 1.)
            optimizer.step()
            if step == 1 or step % 100 == 0:
                losses.append({"step": step, "loss": float(loss.detach())})
                print("[a2-train]", seed, losses[-1], flush=True)
        model.eval().requires_grad_(False)
        with torch.inference_mode():
            final = identity_evaluation(vectors(model, images), data)
            geo_after = {k: geometry_evaluation(model, x, r) for k, (x, r) in clean_sets.items()}
            changes = {k: float((encode(model.geometry_tokens, images[k]).cpu() - v).abs().max()) for k, v in cached.items()}
            independence = {k: interventions(model, images[k], data[k]) for k in ("primary", "extra")}
        frozen = frozen_hashes == (digest(model.geometry), digest(model.mapper)) and max(changes.values()) == 0
        assert frozen and all(p.grad is None for p in model.geometry.parameters())
        gate = {"identity": final["probe_converged"] and initial["probe_converged"], "geometry": frozen, "independence": True}
        for k in ("primary", "extra"):
            v, g, i = final["splits"][k], geo_after[k], independence[k]
            gate["identity"] &= bool(v["accuracy"] >= .85 and v["accuracy"] - initial["splits"][k]["accuracy"] >= .05 and
                                      min(v["recall"]) >= .75 and min(v["by_angle"].values()) >= .8 and
                                      v["geometry"]["all_three_changed_similarity"] >= .8 and v["geometry"]["margin"] >= .3)
            gate["geometry"] &= bool(g["direction"]["margin"] >= .99 * geo_before[k]["direction"]["margin"] and
                                      g["direction"]["minimum_group_margin"] > 0 and
                                      g["frequency"]["frequency_exact_accuracy"] >= .99 and
                                      g["predicted_axis_frequency"]["frequency_exact_accuracy"] >= .99)
            gate["independence"] &= bool(min(i["identity_rotation_cosine_by_class"].values()) >= .9 and
                                          i["stripe_geometry_rotation_distance"] >= .05 and
                                          i["stripe_identity_rotation_distance"] <= .5 * i["stripe_geometry_rotation_distance"])
        reports[str(seed)] = {"initial": initial, "final": final, "geometry_before": geo_before,
                              "geometry_after": geo_after, "geometry_max_output_change": changes,
                              "frozen": frozen, "independence": independence, "losses": losses,
                              "gate": gate, "pass": all(gate.values())}
        torch.save({"model": model.state_dict(), "seed": seed, "protocol": protocol}, out / ("branches_%d.pt" % seed))
        write_json(out / "partial.json", reports)
        print("[a2-seed]", seed, gate, {k: v["accuracy"] for k, v in final["splits"].items()}, flush=True)
        del optimizer, model
    passed = all(v["pass"] for v in reports.values())
    result = {"protocol": protocol, "seeds": reports, "complete": True, "a2_pass": passed,
              "next": "E19.2-B token swap" if passed else "stop at representation; identity invariance not stable",
              "generation_executed": False, "real_fabric_validated": False}
    write_json(out / "report.json", result)
    write_json(out / "stage_status.json", {"a2_pass": passed, "b_executed": False, "generation_executed": False,
                                           "next": result["next"]})
    print("[a2-final]", passed, flush=True)


if __name__ == "__main__":
    main()
