"""E18.2 B2: supervise only fused; every downstream parameter stays frozen."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from transformers import CLIPImageProcessor

from tools.e15_common import write_json
from tools.e18_joint_alignment import (
    balanced_fused_vector, build_conditioner, supervised_contrastive, training_batch,
)


TRAINED_PREFIXES = ("stage3.", "stage4.", "token_source_proj.3.", "token_source_proj.4.")


def configure_trainable(bf):
    bf.eval().requires_grad_(False)
    parameters = []
    for name, parameter in bf.named_parameters():
        if name.startswith(TRAINED_PREFIXES):
            parameter.requires_grad_(True)
            parameters.append(parameter)
    return parameters


def audit_changes(initial, final):
    changed = [name for name in initial if not torch.equal(initial[name], final[name])]
    unexpected = [name for name in changed if not name.startswith(TRAINED_PREFIXES)]
    if unexpected:
        raise ValueError("Frozen BF tensors changed: %s" % unexpected)
    return {"frozen_tensors_unchanged": True, "changed_keys": changed,
            "checked_frozen_tensors": sum(not name.startswith(TRAINED_PREFIXES) for name in initial)}


def main():
    from train_texture_adapter import load_image_encoder_flexible

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "data-root", "selection", "start-checkpoint", "clip-model", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    selected = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    if selected["manifest"] != args.manifest or selected["seed"] != args.seed:
        raise ValueError("Use the unchanged E18.1 sample selection and seed")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    state = torch.load(args.start_checkpoint, map_location="cpu", weights_only=False)
    initial = state["bf_texture_conditioner"]
    bf = build_conditioner(initial).to(args.device, dtype=torch.float32)
    parameters = configure_trainable(bf)
    optimizer = torch.optim.AdamW(parameters, lr=2e-5, weight_decay=0.01)
    vision = load_image_encoder_flexible(args.clip_model, args.device, torch.float16).eval()
    vision.requires_grad_(False)
    processor = CLIPImageProcessor()
    losses = []
    for step in range(1, args.steps + 1):
        _, augmented, clip_images, labels = training_batch(
            rows, selected["examples"], Path(args.data_root), processor, rng)
        labels = labels.to(args.device)
        with torch.no_grad():
            visual = vision(clip_images.to(args.device, torch.float16), output_hidden_states=True)
            patches = visual.hidden_states[-1][:, 1:, :].float()
        fused = bf._build_patch_tokens(patches, augmented.to(args.device))[0]
        direction = supervised_contrastive(balanced_fused_vector(fused), labels)
        # Keep the E18-B1 fused term's coefficient. There is no other training loss.
        loss = 0.2 * direction
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if step == 1 or step % 25 == 0:
            item = {"step": step, "fused_direction": float(direction.item()), "total": float(loss.item())}
            losses.append(item)
            print("[e18.2-b2]", item, flush=True)
    final = {name: value.detach().cpu() for name, value in bf.state_dict().items()}
    audit = audit_changes(initial, final)
    state["bf_texture_conditioner"] = final
    configuration = {"arm": "b2_fused_only", "steps": args.steps, "seed": args.seed,
                     "lr": 2e-5, "weight_decay": 0.01, "orientation_weight": 0.2,
                     "temperature": 0.12, "trained_prefixes": TRAINED_PREFIXES,
                     "loss": "fused supervised contrastive only; exact E18-B1 fused term",
                     "fusion": "projected source concatenation; no extra fusion parameters",
                     "frozen": ["BF stage1/2", "CLIP", "direct readout", "resampler",
                                "final token projection", "U-Net", "TCPM"],
                     "unet_tcpm": "not instantiated; checkpoint entries preserved",
                     "start_checkpoint": args.start_checkpoint, "selection": args.selection,
                     "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                     "selection_sha256": hashlib.sha256(Path(args.selection).read_bytes()).hexdigest()}
    state.setdefault("meta", {})["e18_2"] = configuration
    torch.save(state, output / "joint_model.pt")
    write_json(output / "train_report.json", {
        **configuration, "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "freeze_audit": audit, "losses": losses})
    print("[e18.2-freeze]", audit, flush=True)


if __name__ == "__main__":
    main()
