"""Attribute E17-A nonlinear decodability to CLIP or BF CNN tokens."""

import argparse
import json
from pathlib import Path

import numpy as np

from tools.e15_common import write_json
from tools.e15_linear_probe import gram_blocks, kernel_pca, spectral_basis
from tools.e17_probe import tasks, train_network


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = Path(args.probe_dir)
    rows = json.loads((root / "samples.json").read_text(encoding="utf-8"))
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    with np.load(root / "features.npz") as features:
        fused = features["fused"]
    clip_length = fused.shape[1] - 4 * 64
    sources = {"clip": fused[:, :clip_length], "cnn_all": fused[:, clip_length:],
               "cnn3_4": fused[:, clip_length + 2 * 64:]}
    result = {"protocol": {"probe_dir": str(root), "clip_tokens": clip_length,
                           "cnn_tokens_per_stage": 64,
                           "splits": "reuse E17-A outer test and inner validation"},
              "tasks": {}}
    task_labels = {name: dict(zip(indices, labels))
                   for name, (indices, labels) in tasks(rows).items()}
    for task in ("orientation", "pattern", "frequency"):
        entry = report["tasks"][task]
        classes = np.array(sorted(entry["classes"]))
        scores = {source: {model: [] for model in ("linear", "mlp", "attention")}
                  for source in sources}
        for fold, split in enumerate(entry["splits"]):
            test = np.array(split["test"], dtype=int)
            val = np.array(split["validation"], dtype=int)
            train = np.array(sorted(set(split["train"]) - set(val)), dtype=int)
            labels = {name: np.array([task_labels[task][i] for i in indices])
                      for name, indices in (("train", train), ("val", val), ("test", test))}
            for source, tokens in sources.items():
                matrix = tokens.reshape(len(rows), -1).astype(np.float32)
                kernel_train, kernel_test = gram_blocks(matrix[split["train"]], matrix[test])
                weights, vectors = spectral_basis(kernel_train)
                train_pca, test_pca = kernel_pca(weights, vectors, kernel_test,
                                                 min(64, len(split["train"]) - 1))
                predicted = LogisticRegression(C=1.0, max_iter=2000).fit(
                    train_pca, [task_labels[task][i] for i in split["train"]]).predict(test_pca)
                scores[source]["linear"].append(float(balanced_accuracy_score(labels["test"], predicted)))
                for model in ("mlp", "attention"):
                    predicted = train_network(model, tokens[train].astype(np.float32), labels["train"],
                                              tokens[val].astype(np.float32), labels["val"],
                                              tokens[test].astype(np.float32), classes,
                                              args.seed + fold, args.epochs, args.device)
                    score = balanced_accuracy_score(labels["test"], classes[predicted])
                    scores[source][model].append(float(score))
                    print("[e17-source] %s fold=%d %s %s %.3f" %
                          (task, fold, source, model, score), flush=True)
        result["tasks"][task] = {source: {model: {"folds": values,
                                                  "mean": float(np.mean(values))}
                                          for model, values in models.items()}
                                  for source, models in scores.items()}
    write_json(root / "source_report.json", result)


if __name__ == "__main__":
    main()
