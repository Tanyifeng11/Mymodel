"""E17-A: frozen fused representation, three readouts and pixel-domain controls."""

import argparse
import csv
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tools.e15_common import LayerCapture, write_json
from tools.e15_linear_probe import (axis_label, build_dataset, image_labels,
                                    gram_blocks, kernel_pca, spectral_basis)


def labels_file(path):
    with open(path, encoding="utf-8-sig") as handle:
        return {r["texture"]: r for r in csv.DictReader(handle)
                if r.get("confirmed") == "1" and r.get("pattern")}


def dominant_frequency(gray):
    power = np.abs(np.fft.fft2(gray - gray.mean())) ** 2
    fy = np.fft.fftfreq(gray.shape[0])[:, None]
    fx = np.fft.fftfreq(gray.shape[1])[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    bins = np.linspace(0.06, 0.45, 65)
    energy = np.histogram(radius, bins=bins, weights=power)[0]
    return float((bins[np.argmax(energy)] + bins[np.argmax(energy) + 1]) / 2)


def extract(args, output):
    import torch

    dataset = build_dataset(args)
    annotations = labels_file(args.labels)
    location = {r.get("texture", r.get("color")): i for i, r in enumerate(dataset.data)}
    tagged = {location[k] for k in annotations if k in location}
    random_ids = set(random.Random(args.seed).sample(range(len(dataset)), args.count))
    indices = sorted(random_ids | tagged)
    from tools.e15_diagnosis import build_models

    ctx = {}
    build_models(SimpleNamespace(checkpoint=args.checkpoint, base_model=args.base_model,
                                 clip_model=args.clip_model, device=args.device), ctx)
    capture = LayerCapture(ctx["bf"])
    rows = []
    fused = []
    fft = []
    gradients = []
    try:
        with torch.inference_mode():
            for index in indices:
                batch = dataset[index]
                capture.reset()
                visual = ctx["vision"](batch["clip_texture_image"].to(ctx["device"], ctx["dtype"]),
                                       output_hidden_states=True)
                ctx["model"].get_texture_condition_tokens(
                    visual, batch["texture_image"][None].to(ctx["device"], ctx["dtype"]))
                tokens = capture.current["fused"][0].float().cpu().numpy()
                entry = image_labels(batch["texture_image"])
                gray = batch["texture_image"].float().mean(0).numpy()
                gy, gx = np.gradient(gray)
                gradient = np.array([np.abs(gx).mean(), np.abs(gy).mean(),
                                     gx.std(), gy.std()], dtype=np.float32)
                path = dataset.data[index].get("texture", dataset.data[index].get("color"))
                annotation = annotations.get(path, {})
                rows.append({"index": index, "texture": path,
                             "source_group": annotation.get("source_group") or path,
                             "pattern": annotation.get("pattern", ""),
                             "axis": entry["axis"], "anisotropy": entry["anisotropy"],
                             "frequency": dominant_frequency(gray)})
                fused.append(tokens.astype(np.float16))
                fft.append(entry["image__gray_fft128"])
                gradients.append(gradient)
                print("[e17-a] extracted %d/%d index=%d" % (len(rows), len(indices), index), flush=True)
    finally:
        capture.remove()
    np.savez_compressed(output / "features.npz", fused=np.stack(fused),
                        fft=np.stack(fft), gradient=np.stack(gradients))
    write_json(output / "samples.json", rows)
    return rows, {"fused": np.stack(fused), "fft": np.stack(fft),
                  "gradient": np.stack(gradients)}


def tasks(rows, min_pattern=6, orientation_count=120):
    from collections import Counter

    oriented = [i for i in sorted(range(len(rows)), key=lambda j: -rows[j]["anisotropy"])
                if rows[i]["axis"]][:orientation_count]
    counts = Counter(r["pattern"] for r in rows if r["pattern"])
    patterns = [i for i, r in enumerate(rows) if r["pattern"] and counts[r["pattern"]] >= min_pattern]
    frequency = sorted(oriented, key=lambda i: rows[i]["frequency"])
    quarter = len(frequency) // 4
    frequency = frequency[:quarter] + frequency[-quarter:]
    return {
        "orientation": (oriented, [rows[i]["axis"] for i in oriented]),
        "pattern": (patterns, [rows[i]["pattern"] for i in patterns]),
        "frequency": (frequency, ["coarse"] * quarter + ["fine"] * quarter),
    }


def train_network(kind, train_x, train_y, val_x, val_y, test_x, classes, seed, epochs, device):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    d = train_x.shape[-1]
    if kind == "mlp":
        def pool(x):
            return np.concatenate([x.mean(1), x.std(1)], axis=1)
        train_x, val_x, test_x = map(pool, (train_x, val_x, test_x))
        mean = train_x.mean(0)
        std = np.maximum(train_x.std(0), 1e-4)
        train_x, val_x, test_x = [(x - mean) / std for x in (train_x, val_x, test_x)]
        model = nn.Sequential(nn.Linear(2 * d, 128), nn.GELU(), nn.Dropout(0.3),
                              nn.Linear(128, len(classes)))
    else:
        mean = train_x.mean((0, 1), keepdims=True)
        std = np.maximum(train_x.std((0, 1), keepdims=True), 1e-4)
        train_x, val_x, test_x = [(x - mean) / std for x in (train_x, val_x, test_x)]

        class AttentionProbe(nn.Module):
            def __init__(self):
                super().__init__()
                self.project = nn.Linear(d, 64)
                self.position = nn.Parameter(torch.randn(1, train_x.shape[1], 64) * 0.02)
                self.relate = nn.TransformerEncoderLayer(64, 4, 128, dropout=0.1,
                                                         batch_first=True)
                self.query = nn.Parameter(torch.randn(1, 4, 64) * 0.02)
                self.attn = nn.MultiheadAttention(64, 4, batch_first=True)
                self.classifier = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, len(classes)))

            def forward(self, x):
                z = self.relate(self.project(x) + self.position[:, :x.shape[1]])
                out, _ = self.attn(self.query.expand(len(x), -1, -1), z, z,
                                   need_weights=False)
                return self.classifier(out.mean(1))

        model = AttentionProbe()
    model.to(device)
    arrays = [torch.from_numpy(np.asarray(x, np.float32)) for x in (train_x, val_x, test_x)]
    targets = [torch.as_tensor(np.searchsorted(classes, y), dtype=torch.long)
               for y in (train_y, val_y)]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    best, state, stale = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        for ids in torch.randperm(len(arrays[0])).split(16):
            logits = model(arrays[0][ids].to(device))
            loss = nn.functional.cross_entropy(logits, targets[0][ids].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_logits = torch.cat([model(x.to(device)) for x in arrays[1].split(16)])
            val_loss = nn.functional.cross_entropy(val_logits, targets[1].to(device)).item()
        if val_loss < best - 1e-4:
            best, stale = val_loss, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 12:
                break
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        return torch.cat([model(x.to(device)) for x in arrays[2].split(16)]).argmax(1).cpu().numpy()


def evaluate(args, rows, arrays, output):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedGroupKFold, train_test_split

    report = {"protocol": {"checkpoint": args.checkpoint, "manifest": args.manifest,
                           "sample_count": len(rows), "seed": args.seed,
                           "split": "3-fold stratified grouped outer test; inner validation only for early stopping",
                           "source_note": "source_group is provisional where fabric identity is unverified"},
              "tasks": {}}
    for task, (positions, names) in tasks(rows, orientation_count=args.orientation_count).items():
        y = np.asarray(names)
        ids = np.asarray(positions)
        groups = np.asarray([rows[i]["source_group"] for i in ids])
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2 or counts.min() < 3:
            report["tasks"][task] = {"skipped": "insufficient class samples"}
            continue
        splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=args.seed)
        results = {name: [] for name in ("linear", "mlp", "attention", "image_fft", "image_gradient")}
        folds = []
        for fold, (train, test) in enumerate(splitter.split(ids, y, groups)):
            inner_train, val = train_test_split(train, test_size=0.2, stratify=y[train],
                                                random_state=args.seed + fold)
            for name, data in (("linear", arrays["fused"].reshape(len(rows), -1)),
                               ("image_fft", arrays["fft"]),
                               ("image_gradient", arrays["gradient"])):
                matrix = data[ids].astype(np.float32)
                kernel_train, kernel_test = gram_blocks(matrix[train], matrix[test])
                weights, vectors = spectral_basis(kernel_train)
                train_pca, test_pca = kernel_pca(weights, vectors, kernel_test, min(64, len(train) - 1))
                pred = LogisticRegression(C=1.0, max_iter=2000).fit(train_pca, y[train]).predict(test_pca)
                results[name].append(float(balanced_accuracy_score(y[test], pred)))
            x = arrays["fused"][ids].astype(np.float32)
            for name in ("mlp", "attention"):
                pred_idx = train_network(name, x[inner_train], y[inner_train], x[val], y[val],
                                         x[test], classes, args.seed + fold, args.epochs, args.device)
                results[name].append(float(balanced_accuracy_score(y[test], classes[pred_idx])))
            folds.append({"train": ids[train].tolist(), "test": ids[test].tolist(),
                          "validation": ids[val].tolist()})
            print("[e17-a] %s fold=%d %s" % (task, fold, {k: round(v[-1], 3) for k, v in results.items()}), flush=True)
        report["tasks"][task] = {"samples": len(ids), "classes": dict(zip(classes.tolist(), counts.tolist())),
                                 "scores": {k: {"folds": v, "mean": float(np.mean(v)),
                                                "std": float(np.std(v))} for k, v in results.items()},
                                 "splits": folds}
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "data-root", "checkpoint", "base-model", "clip-model", "labels", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--orientation-count", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reuse-features", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.reuse_features:
        rows = json.loads((output / "samples.json").read_text(encoding="utf-8"))
        with np.load(output / "features.npz") as features:
            arrays = {k: features[k] for k in features.files}
    else:
        rows, arrays = extract(args, output)
    report = evaluate(args, rows, arrays, output)
    for task, entry in report["tasks"].items():
        if "scores" in entry:
            print(task, {k: round(v["mean"], 4) for k, v in entry["scores"].items()}, flush=True)


if __name__ == "__main__":
    main()
