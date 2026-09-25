"""E18-B: matched BF/readout joint controls, with optional orientation alignment."""

import argparse
import copy
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from transformers import CLIPImageProcessor

from models.bf_texture_module import BFTextureConditioner
from tools.e14_pattern_probe import training_preprocess
from tools.e17_encoder_control import choose_examples
from tools.e15_common import write_json


def build_conditioner(source):
    channels = tuple(source[f"stage{i}.0.weight"].shape[0] for i in range(1, 5))
    bf = BFTextureConditioner(clip_embeddings_dim=source["token_source_proj.0.0.weight"].shape[0],
                              num_tokens=source["resampler_queries"].shape[1],
                              stage_channels=channels)
    if "direct_readout.selection_marker" not in source:
        raise ValueError("E18 requires E17 select readout")
    bf.configure_direct_readout(source_layout="select")
    bf.load_state_dict(source, strict=True)
    return bf


def balanced_fused_vector(fused, area=64):
    clip = fused[:, :-4 * area]
    sources = [clip] + [fused[:, -(4 - i) * area:-(3 - i) * area if i < 3 else None]
                        for i in range(4)]
    means = torch.stack([x.mean(1) for x in sources]).mean(0)
    stds = torch.stack([x.std(1, unbiased=False) for x in sources]).mean(0)
    return F.normalize(torch.cat([means, stds], dim=1), dim=1)


def token_vector(tokens):
    return F.normalize(torch.cat([tokens.mean(1), tokens.std(1, unbiased=False)], dim=1), dim=1)


def supervised_contrastive(vectors, labels, temperature=0.12):
    similarity = vectors @ vectors.T / temperature
    diagonal = torch.eye(len(labels), dtype=torch.bool, device=vectors.device)
    positives = labels[:, None].eq(labels[None, :]) & ~diagonal
    normalizer = torch.logsumexp(similarity.masked_fill(diagonal, -1e4), dim=1)
    return -((similarity - normalizer[:, None]) * positives).sum(1).div(positives.sum(1)).mean()


def training_batch(rows, examples, root, processor, rng):
    by_axis = {axis: [entry for entry in examples if entry[1] == axis] for axis in (0, 1)}
    chosen = rng.sample(by_axis[0], 2) + rng.sample(by_axis[1], 2)
    natural, augmented, clip_images, labels = [], [], [], []
    for index, axis, _ in chosen:
        path = root / rows[index].get("texture", rows[index].get("color"))
        with Image.open(path) as image:
            image = image.convert("RGB")
            brightness, contrast = rng.uniform(0.8, 1.2), rng.uniform(0.9, 1.1)
            for variant, label in ((image, axis),
                                   (image.transpose(Image.Transpose.ROTATE_90), 1 - axis)):
                original, clip = training_preprocess(variant, processor, 512, 384)
                adjusted = ImageEnhance.Contrast(ImageEnhance.Brightness(variant).enhance(brightness))
                changed, _ = training_preprocess(adjusted.enhance(contrast), processor, 512, 384)
                natural.append(original[0])
                augmented.append(changed[0])
                clip_images.append(clip)
                labels.append(label)
    return (torch.stack(natural), torch.stack(augmented), torch.cat(clip_images),
            torch.as_tensor(labels, dtype=torch.long))


def main():
    from train_texture_adapter import load_image_encoder_flexible

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "data-root", "start-checkpoint", "clip-model", "selection", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--arm", choices=("b0", "b1"), required=True)
    parser.add_argument("--scan-count", type=int, default=3000)
    parser.add_argument("--per-axis", type=int, default=400)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    selection_path = Path(args.selection)
    if selection_path.exists():
        selected = json.loads(selection_path.read_text(encoding="utf-8"))
        if selected["manifest"] != args.manifest or selected["seed"] != args.seed:
            raise ValueError("selection protocol differs between B0 and B1")
        rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        examples = selected["examples"]
    else:
        rows, examples, _ = choose_examples(args.manifest, args.data_root,
                                            args.scan_count, args.per_axis, args.seed)
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(selection_path, {"manifest": args.manifest, "seed": args.seed,
                                    "scan_count": args.scan_count, "per_axis": args.per_axis,
                                    "examples": examples,
                                    "axis_label_source": "FFT axis of BF training images only"})
    state = torch.load(args.start_checkpoint, map_location="cpu", weights_only=False)
    bf = build_conditioner(state["bf_texture_conditioner"])
    teacher = copy.deepcopy(bf).to(args.device, dtype=torch.float32).eval().requires_grad_(False)
    teacher.direct_readout = None  # Preserved E5/GAM resampler is the common target for B0/B1.
    bf.to(args.device, dtype=torch.float32).eval().requires_grad_(False)
    trainable = []
    for module in (bf.stage3, bf.stage4, bf.token_source_proj[3],
                   bf.token_source_proj[4], bf.direct_readout):
        module.requires_grad_(True)
        trainable.extend(module.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=2e-5, weight_decay=0.01)
    vision = load_image_encoder_flexible(args.clip_model, args.device, torch.float16).eval()
    vision.requires_grad_(False)
    processor = CLIPImageProcessor()
    losses = []
    for step in range(1, args.steps + 1):
        natural, augmented, clip_images, labels = training_batch(
            rows, examples, Path(args.data_root), processor, rng)
        labels = labels.to(args.device)
        with torch.no_grad():
            visual = vision(clip_images.to(args.device, torch.float16), output_hidden_states=True)
            patches = visual.hidden_states[-1][:, 1:, :].float()
            base_tokens = teacher(clip_vision_tokens=patches,
                                  texture_images=natural.to(args.device))[0]
        fused = bf._build_patch_tokens(patches, augmented.to(args.device))[0]
        tokens = bf.direct_readout(fused, bf.stage_token_hw)
        preservation = F.smooth_l1_loss(tokens, base_tokens)
        orientation = preservation.new_zeros(())
        if args.arm == "b1":
            source = balanced_fused_vector(fused)
            final = token_vector(tokens)
            orientation = (supervised_contrastive(source, labels)
                           + supervised_contrastive(final, labels)
                           + 0.5 * F.mse_loss(source @ source.T, final @ final.T))
        loss = preservation + 0.2 * orientation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if step == 1 or step % 25 == 0:
            item = {"step": step, "preservation": float(preservation.item()),
                    "orientation": float(orientation.item()), "total": float(loss.item())}
            losses.append(item)
            print("[e18-%s] %s" % (args.arm, item), flush=True)
    state["bf_texture_conditioner"] = {key: value.detach().cpu()
                                       for key, value in bf.state_dict().items()}
    state.setdefault("meta", {})["e18_b"] = {
        "arm": args.arm, "steps": args.steps, "selection": args.selection,
        "base_objective": "distill frozen E5/GAM texture tokens under brightness/contrast changes",
        "orientation_objective": "raw fused/final supervised contrastive, paired 90 degree rotations"
        if args.arm == "b1" else None,
        "unet_frozen": True, "clip_frozen": True}
    torch.save(state, output / "joint_model.pt")
    write_json(output / "train_report.json", {
        "arm": args.arm, "steps": args.steps, "examples": len(examples),
        "trainable_parameters": sum(p.numel() for p in trainable),
        "trained": ["stage3", "stage4", "token_source_proj.3-4", "direct_readout"],
        "frozen": ["CLIP", "BF stage1-2", "U-Net", "TCPM", "E5/GAM resampler teacher"],
        "start_checkpoint": args.start_checkpoint, "selection": args.selection,
        "pair_protocol": "each source contributes original and rot90 with the same color perturbation",
        "losses": losses})


if __name__ == "__main__":
    main()
