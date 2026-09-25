"""Source-held-out orientation readout at CNN, fused and final tokens on Pattern-Gold."""

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

from models.bf_texture_module import BFTextureConditioner
from tools.e14_pattern_probe import training_preprocess
from tools.e15_common import write_json
from tools.e18_joint_alignment import balanced_fused_vector, token_vector


def conditioner(source):
    channels = tuple(source[f"stage{i}.0.weight"].shape[0] for i in range(1, 5))
    bf = BFTextureConditioner(clip_embeddings_dim=source["token_source_proj.0.0.weight"].shape[0],
                              num_tokens=source["resampler_queries"].shape[1], stage_channels=channels)
    if "direct_readout.selection_marker" in source:
        bf.configure_direct_readout(source_layout="select")
    bf.load_state_dict(source, strict=True)
    return bf


def vectorize(bf, patches, cnn):
    f1, f2, f3, f4 = bf._encode_texture_features(cnn)
    pooled = [bf._stage_to_tokens(x) for x in (f1, f2, f3, f4)]
    projected = [bf.token_source_proj[0](patches)] + [proj(x) for proj, x in
                 zip(bf.token_source_proj[1:], pooled)]
    fused = torch.cat(projected, dim=1)
    if bf.direct_readout is None:
        tokens = bf(clip_vision_tokens=patches, texture_images=cnn)[0]
    else:
        tokens = bf.direct_readout(fused, bf.stage_token_hw)
    cnn_parts = [torch.cat([x.mean(1), x.std(1, unbiased=False)], dim=1)
                 for x in pooled[2:]]
    return {"cnn3_4": torch.cat(cnn_parts, dim=1),
            "fused": balanced_fused_vector(fused), "final": token_vector(tokens)}


def evaluate(x, labels, groups):
    predictions = np.empty(len(labels), dtype=int)
    folds = []
    for group in sorted(set(groups)):
        test = np.flatnonzero(groups == group)
        train = np.flatnonzero(groups != group)
        scaler = StandardScaler()
        a, b = scaler.fit_transform(x[train]), scaler.transform(x[test])
        pca = PCA(n_components=min(8, len(train) - 1), svd_solver="randomized", random_state=42)
        a, b = pca.fit_transform(a), pca.transform(b)
        model = LogisticRegression(C=1.0, max_iter=2000, random_state=42).fit(a, labels[train])
        predictions[test] = model.predict(b)
        folds.append({"source_group": group, "correct": int((predictions[test] == labels[test]).sum()),
                      "count": len(test)})
    return {"balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
            "pair_both_correct": sum(fold["correct"] == 2 for fold in folds),
            "source_count": len(folds), "folds": folds, "predictions": predictions.tolist()}


def main():
    from train_texture_adapter import load_image_encoder_flexible

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("gold", "checkpoint", "clip-model", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = Path(args.gold)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    gold = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    bf = conditioner(state["bf_texture_conditioner"]).to(args.device, dtype=torch.float32).eval()
    bf.requires_grad_(False)
    vision = load_image_encoder_flexible(args.clip_model, args.device, torch.float16).eval()
    vision.requires_grad_(False)
    processor = CLIPImageProcessor()
    oriented = [row for row in gold["rows"] if row["orientation"]]
    features = {key: [] for key in ("cnn3_4", "fused", "final", "image_fft")}
    with torch.inference_mode():
        for row in oriented:
            with Image.open(root / row["image"]) as image:
                image = image.convert("RGB")
                gray = np.asarray(image.convert("L").resize((128, 128)), dtype=np.float32) / 255
                features["image_fft"].append(np.log1p(np.abs(np.fft.fftshift(
                    np.fft.fft2(gray - gray.mean())))).ravel())
                cnn, clip = training_preprocess(image, processor, 512, 384)
            visual = vision(clip.to(args.device, torch.float16), output_hidden_states=True)
            patches = visual.hidden_states[-1][:, 1:, :].float()
            vectors = vectorize(bf, patches, cnn.to(args.device))
            for key, value in vectors.items():
                features[key].append(value.float().cpu().numpy()[0])
            print("[e18-probe] %s" % row["image"], flush=True)
    labels = np.asarray([0 if row["orientation"] == "vertical" else 1 for row in oriented])
    groups = np.asarray([row["source_group"] for row in oriented])
    scores = {key: evaluate(np.stack(values).astype(np.float32), labels, groups)
              for key, values in features.items()}
    report = {"checkpoint": args.checkpoint, "gold": str(root / "manifest.json"),
              "protocol": "leave one original reference out; both rotations held out together; train-fold scaler/PCA",
              "sample_count": len(oriented), "source_count": len(set(groups)),
              "source_identity": gold["source_identity"], "scores": scores}
    write_json(output / "report.json", report)
    print("[e18-probe]", {k: (round(v["balanced_accuracy"], 3), v["pair_both_correct"])
                            for k, v in scores.items()}, flush=True)


if __name__ == "__main__":
    main()
