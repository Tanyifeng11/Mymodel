"""Fit direction probes on clean training templates; test unseen frequencies/phases."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from transformers import CLIPImageProcessor

from tools.e14_pattern_probe import training_preprocess
from tools.e15_common import write_json
from tools.e18_gold_probe import conditioner, vectorize


def extract(rows, root, bf, vision, processor, device):
    result = {name: [] for name in ("cnn3_4", "fused", "final")}
    with torch.inference_mode():
        for row in rows:
            with Image.open(root / row["texture"]) as image:
                cnn, clip = training_preprocess(image.convert("RGB"), processor, 512, 384)
            vision_output = vision(clip.to(device, torch.float16), output_hidden_states=True)
            patches = vision_output.hidden_states[-1][:, 1:, :].float()
            vectors = vectorize(bf, patches, cnn.to(device))
            for name in result:
                result[name].append(vectors[name].float().cpu().numpy()[0])
    return {name: np.stack(values).astype(np.float32) for name, values in result.items()}


def probe(train_x, train_y, validation_x, validation_rows):
    scaler = StandardScaler()
    a = scaler.fit_transform(train_x)
    b = scaler.transform(validation_x)
    pca = PCA(n_components=min(8, len(train_y) - 1), svd_solver="randomized", random_state=42)
    a, b = pca.fit_transform(a), pca.transform(b)
    classifier = LogisticRegression(C=1.0, max_iter=2000, random_state=42).fit(a, train_y)
    labels = np.array([int(row["axis"] == "horizontal") for row in validation_rows])
    predictions = classifier.predict(b)
    frequencies = sorted({row["frequency"] for row in validation_rows})
    by_frequency = {str(frequency): float(balanced_accuracy_score(
        labels[[i for i, row in enumerate(validation_rows) if row["frequency"] == frequency]],
        predictions[[i for i, row in enumerate(validation_rows) if row["frequency"] == frequency]]))
        for frequency in frequencies}
    groups = sorted({row["source_group"] for row in validation_rows})
    both_correct = sum(all(predictions[i] == labels[i] for i, row in enumerate(validation_rows)
                           if row["source_group"] == group) for group in groups)
    return {"balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
            "by_frequency": by_frequency, "paired_both_correct": int(both_correct),
            "pair_count": len(groups), "train_accuracy": float(classifier.score(a, train_y)),
            "predictions": predictions.tolist()}


def main():
    from train_texture_adapter import load_image_encoder_flexible

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "checkpoint", "clip-model", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = Path(args.data)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    train_rows = json.loads((root / "train.json").read_text(encoding="utf-8"))
    validation_rows = json.loads((root / "validation.json").read_text(encoding="utf-8"))
    if {row["frequency"] for row in train_rows} & {row["frequency"] for row in validation_rows}:
        raise ValueError("frequency leakage")
    if {row["phase"] for row in train_rows} & {row["phase"] for row in validation_rows}:
        raise ValueError("phase leakage")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    bf = conditioner(state["bf_texture_conditioner"]).to(args.device, dtype=torch.float32).eval()
    bf.requires_grad_(False)
    vision = load_image_encoder_flexible(args.clip_model, args.device, torch.float16).eval()
    vision.requires_grad_(False)
    processor = CLIPImageProcessor()
    train = extract(train_rows, root, bf, vision, processor, args.device)
    validation = extract(validation_rows, root, bf, vision, processor, args.device)
    train_y = np.array([int(row["axis"] == "horizontal") for row in train_rows])
    scores = {name: probe(train[name], train_y, validation[name], validation_rows)
              for name in ("cnn3_4", "fused", "final")}
    write_json(output / "report.json", {
        "checkpoint": args.checkpoint, "data": str(root),
        "train_images": len(train_rows), "validation_images": len(validation_rows),
        "protocol": "linear probe fitted on train templates only; test unseen frequencies and phases",
        "scores": scores})
    print("[e18.1-probe]", {name: (round(value["balanced_accuracy"], 3),
                                    value["paired_both_correct"]) for name, value in scores.items()},
          flush=True)


if __name__ == "__main__":
    main()
