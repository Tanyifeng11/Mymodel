"""复核 identity 探针收敛，并记录方向几何；不重新训练 encoder。"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models.explicit_pattern import ExplicitPatternTokens
from models.frequency_pattern import FrequencyBiasedEncoder
from tools.e15_common import write_json
from tools.e18_joint_alignment import token_vector
from tools.e19_2_identity import PATTERNS, encode, identity_geometry
from tools.e19_frequency_audit import load_images
from tools.e19_handcrafted import summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root, device = Path(args.root), args.device
    out = root / "e19_2"
    torch.set_num_threads(4)
    data = json.loads((out / "data/manifest.json").read_text())["splits"]
    images = {k: load_images(out / "data", rows).to(device) for k, rows in data.items()}
    mapper = ExplicitPatternTokens().to(device)
    mapper.load_state_dict(torch.load(root / "e19/a/pattern_branch.pt", map_location=device, weights_only=False)["mapper"])
    mapper.eval().requires_grad_(False)
    old_rows = json.loads((root / "e18_1/clean/validation.json").read_text())
    old_images = load_images(root / "e18_1/clean", old_rows).to(device)
    reports = {}
    with torch.inference_mode():
        for seed in (42, 43, 44):
            model = FrequencyBiasedEncoder().to(device).eval().requires_grad_(False)
            model.load_state_dict(torch.load(root / ("e19_1/fft_bias/encoder_%d.pt" % seed), map_location=device, weights_only=False)["encoder"])
            before_direction = summary(token_vector(mapper(encode(model, old_images))).cpu(), old_rows)
            model.load_state_dict(torch.load(out / ("identity_%d.pt" % seed), map_location=device, weights_only=False)["encoder"])
            vectors = {k: token_vector(mapper(encode(model, v))).cpu() for k, v in images.items()}
            after_direction = summary(token_vector(mapper(encode(model, old_images))).cpu(), old_rows)
            probe = make_pipeline(StandardScaler(), PCA(n_components=32, random_state=42),
                                  LogisticRegression(C=1., max_iter=10000, random_state=42))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                probe.fit(vectors["train"].numpy(), [r["pattern"] for r in data["train"]])
            converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
            result = {"probe_converged": converged, "probe_iterations": probe[-1].n_iter_.tolist(),
                      "pattern_direction_before": before_direction, "pattern_direction_after": after_direction}
            for split in ("primary", "extra"):
                rows = data[split]
                labels = np.array([r["pattern"] for r in rows])
                predictions = probe.predict(vectors[split].numpy())
                geometry = identity_geometry(vectors[split], rows)
                value = {"category_accuracy": float(balanced_accuracy_score(labels, predictions)),
                         "per_class_recall": recall_score(labels, predictions, labels=list(PATTERNS), average=None).tolist(),
                         "confusion": confusion_matrix(labels, predictions, labels=list(PATTERNS)).tolist(),
                         "geometry": geometry, "by_angle": {}, "by_frequency": {}}
                for key in ("angle", "frequency"):
                    for group in sorted({r[key] for r in rows}):
                        indices = [i for i, r in enumerate(rows) if r[key] == group]
                        value["by_" + key][str(group)] = float(balanced_accuracy_score(labels[indices], predictions[indices]))
                value["pass"] = bool(converged and value["category_accuracy"] >= .85 and min(value["per_class_recall"]) >= .75 and
                                     geometry["same_pattern_different_color_cross_template"] >= .8 and geometry["margin"] >= .3)
                result[split] = value
            reports[str(seed)] = result
            print("[e19.2-audit]", seed, converged, {k: (result[k]["category_accuracy"], result[k]["by_angle"]) for k in ("primary", "extra")}, flush=True)
    passed = all(v[k]["pass"] for v in reports.values() for k in ("primary", "extra"))
    result = {"class_order": PATTERNS, "seeds": reports, "a_controlled_pass": passed, "complete": True,
              "encoder_training": "none; same frozen identity checkpoints, only readout max_iter increased 2000->10000",
              "direction_scope": "pattern tokens before BF concatenation; not final conditioning or generation",
              "next": "bidirectional token swap" if passed else "stop in representation; do not enter B/C/generation"}
    write_json(out / "a_audit.json", result)
    write_json(out / "stage_status.json", {"identity_training_job": 114112, "a_controlled_pass": passed,
                                          "b_executed": False, "c_real_executed": False, "generation_executed": False,
                                          "authoritative_gate": "a_audit.json", "original_report": "a_report.json",
                                          "next": result["next"]})
    print("[e19.2-audit-final]", passed, flush=True)


if __name__ == "__main__":
    main()
