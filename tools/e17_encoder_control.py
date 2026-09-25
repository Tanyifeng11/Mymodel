"""E17-C conditional control: supervise only the BF encoder with pattern direction/frequency."""

import argparse
import json
import random
from pathlib import Path

import numpy as np

from models.bf_texture_module import BFTextureConditioner
from tools.e15_common import write_json
from tools.e15_linear_probe import axis_label, orientation_angle
from tools.e17_probe import dominant_frequency


def choose_examples(manifest, root, scan_count, per_axis, seed):
    from PIL import Image

    rows = json.loads(Path(manifest).read_text(encoding="utf-8"))
    picked = random.Random(seed).sample(range(len(rows)), min(scan_count, len(rows)))
    groups = {"horizontal": [], "vertical": []}
    for index in picked:
        path = Path(root) / rows[index].get("texture", rows[index].get("color"))
        with Image.open(path) as image:
            gray = np.asarray(image.convert("L").resize((128, 128)), dtype=np.float32) / 255.0
        angle, anisotropy = orientation_angle(gray)
        axis = axis_label(angle)
        if axis:
            groups[axis].append((anisotropy, index, dominant_frequency(gray)))
    selected = []
    for axis in groups:
        selected.extend((index, axis, freq) for _, index, freq in
                        sorted(groups[axis], reverse=True)[:per_axis])
    if min(len(groups["horizontal"]), len(groups["vertical"])) < 30:
        raise ValueError("training split has too few oriented patterns")
    median = float(np.median([entry[2] for entry in selected]))
    return rows, [(index, 0 if axis == "vertical" else 1, int(freq >= median))
                  for index, axis, freq in selected], median


def main():
    import torch
    from PIL import Image
    from torch import nn
    from transformers import CLIPImageProcessor

    from train_texture_adapter import load_image_encoder_flexible
    from tools.e14_pattern_probe import training_preprocess

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "data-root", "direct-checkpoint", "base-checkpoint",
                 "clip-model", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--scan-count", type=int, default=3000)
    parser.add_argument("--per-axis", type=int, default=400)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rows, examples, threshold = choose_examples(args.manifest, args.data_root,
                                                 args.scan_count, args.per_axis, args.seed)
    state = torch.load(args.direct_checkpoint, map_location="cpu", weights_only=False)
    source = state["bf_texture_conditioner"]
    channels = tuple(source[f"stage{i}.0.weight"].shape[0] for i in range(1, 5))
    bf = BFTextureConditioner(clip_embeddings_dim=source["token_source_proj.0.0.weight"].shape[0],
                              num_tokens=source["resampler_queries"].shape[1],
                              stage_channels=channels)
    layout = "select" if "direct_readout.selection_marker" in source else "mean"
    bf.configure_direct_readout(source_layout=layout)
    bf.load_state_dict(source, strict=True)
    bf.to(args.device, dtype=torch.float32).eval().requires_grad_(False)
    encoder_params = []
    for module in list((bf.stage1, bf.stage2, bf.stage3, bf.stage4)) + list(bf.token_source_proj[1:]):
        module.requires_grad_(True)
        encoder_params.extend(module.parameters())
    vision = load_image_encoder_flexible(args.clip_model, args.device, torch.float16).eval()
    vision.requires_grad_(False)
    processor = CLIPImageProcessor()
    head = nn.Sequential(nn.Linear(768, 16), nn.GELU(), nn.Flatten(),
                         nn.Linear(4 * 64 * 16, 128), nn.GELU(), nn.Linear(128, 4)).to(args.device)
    optimizer = torch.optim.AdamW([{"params": encoder_params, "lr": 2e-5},
                                   {"params": head.parameters(), "lr": 1e-3}], weight_decay=0.01)
    by_axis = {axis: [row for row in examples if row[1] == axis] for axis in (0, 1)}
    losses = []
    for step in range(1, args.steps + 1):
        batch_rows = [random.choice(by_axis[i % 2]) for i in range(args.batch_size)]
        cnn_images, clip_images = [], []
        for index, _, _ in batch_rows:
            path = Path(args.data_root) / rows[index].get("texture", rows[index].get("color"))
            with Image.open(path) as image:
                cnn, clip = training_preprocess(image.convert("RGB"), processor, 512, 384)
            cnn_images.append(cnn[0])
            clip_images.append(clip)
        with torch.no_grad():
            visual = vision(torch.cat(clip_images).to(args.device, torch.float16),
                            output_hidden_states=True)
            patches = visual.hidden_states[-1][:, 1:, :].float()
        fused = bf._build_patch_tokens(patches, torch.stack(cnn_images).to(args.device))[0]
        # Only the four spatial BF sources feed the supervision head.
        spatial = fused[:, -4 * 64:]
        logits = head(spatial)
        axis_target = torch.tensor([r[1] for r in batch_rows], device=args.device)
        freq_target = torch.tensor([r[2] for r in batch_rows], device=args.device)
        loss = (nn.functional.cross_entropy(logits[:, :2], axis_target)
                + nn.functional.cross_entropy(logits[:, 2:], freq_target))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 1 or step % 25 == 0:
            print("[e17-c] step=%d loss=%.4f" % (step, loss.item()), flush=True)
            losses.append({"step": step, "loss": float(loss.item())})
    updated = {key: value.detach().cpu() for key, value in bf.state_dict().items()}
    state["bf_texture_conditioner"] = updated
    state.setdefault("meta", {})["e17_encoder_control"] = {
        "steps": args.steps, "supervision": "FFT orientation and dominant frequency",
        "direct_readout_frozen": True, "unet_frozen": True}
    torch.save(state, output / "joint_model.pt")
    baseline = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    for key, value in updated.items():
        flat_key = "bf_texture_conditioner." + key
        if flat_key in baseline and (key.startswith("stage") or key.startswith("token_source_proj.")):
            baseline[flat_key] = value
    torch.save(baseline, output / "pytorch_model.bin")
    write_json(output / "train_report.json", {
        "examples": len(examples), "axis_counts": {str(k): len(v) for k, v in by_axis.items()},
        "frequency_threshold": threshold, "train_manifest": args.manifest,
        "direct_checkpoint": args.direct_checkpoint, "losses": losses,
        "trained_modules": ["stage1", "stage2", "stage3", "stage4", "token_source_proj.1-4"],
        "frozen_modules": ["CLIP vision", "direct_readout", "resampler", "U-Net", "TCPM"]})


if __name__ == "__main__":
    main()
