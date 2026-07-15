#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
import sys
from types import SimpleNamespace

import torch
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from eval.benchmark_utils import sample_uid
from garment_mask_utils import build_sketch_garment_mask


def load_inference_module(project_root):
    path = os.path.join(project_root, "inference_IMAGGarment-1.py")
    spec = importlib.util.spec_from_file_location("imaggarment_inference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_path(data_root, value):
    return value if os.path.isabs(value) else os.path.join(data_root, value)


def output_path(run_dir, sample):
    uid = sample_uid(sample)
    sample_dir = os.path.join(run_dir, "token", f"{sample['sample_id']}_{uid}")
    return sample_dir, os.path.join(
        sample_dir, f"generated_{sample['sample_id']}.png"
    )


def prepare_args(args):
    return SimpleNamespace(
        GAM_model_ckpt=args.gam_ckpt,
        texture_ckpt=args.texture_ckpt,
        seed=args.generation_seed,
        device=args.device,
        base_model_path="auto",
        vae_model_path="auto",
        image_encoder_path="auto",
        width=None,
        height=None,
        texture_num_tokens=16,
        force_texture_num_tokens_override=False,
        texture_condition_mode="token",
        layer_group_enabled=1,
        use_texture_gate=1,
        use_palette_tokens=0,
        num_palette_tokens=4,
        palette_branch_scale_init=0.0,
        gate_type="layer",
        gate_init="identity",
        gate_min=0.7,
        gate_max=1.3,
        use_balanced_fusion_gate=0,
        balanced_gate_hidden_dim=64,
        balanced_gate_scale=0.2,
        balanced_gate_min=0.8,
        balanced_gate_max=1.2,
        use_conflict_aware_gate=0,
        use_tcpm_lite=1,
        tcpm_hidden_ratio=0.25,
        tcpm_residual_scale_init=0.0,
        conflict_texture_suppress_strength=0.1,
        conflict_palette_suppress_strength=0.4,
        conflict_threshold=0.70,
        alpha1=1.0,
        alpha2=1.0,
        alpha3=0.7,
        alpha4=0.5,
        lora_rank=128,
    )


@torch.inference_mode()
def generate_one(pipe, model_args, sample, data_root, seed, destination):
    sketch_path = resolve_path(data_root, sample["sketch"])
    texture_path = resolve_path(data_root, sample["texture"])
    sketch = Image.open(sketch_path).convert("RGB").resize(
        (model_args.width, model_args.height), Image.BILINEAR
    )
    texture = Image.open(texture_path).convert("RGB")
    image_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )
    vae_sketch = image_transform(sketch).unsqueeze(0)
    mask_image, _ = build_sketch_garment_mask(
        sketch, model_args.width, model_args.height
    )
    spatial_mask = transforms.ToTensor()(mask_image).unsqueeze(0)
    generator = torch.Generator(device=model_args.device).manual_seed(seed)

    output = pipe(
        ref_image=vae_sketch,
        prompt=sample["prompt"],
        texture_clip_image=texture,
        texture_embeds=None,
        null_prompt="",
        negative_prompt=" worst quality, low quality",
        width=model_args.width,
        height=model_args.height,
        num_images_per_prompt=1,
        guidance_scale=7.0,
        sketch_scale=0.6,
        ipa_scale=1.0,
        generator=generator,
        num_inference_steps=50,
        texture_mode="patch_resampled",
        texture_num_tokens=model_args.texture_num_tokens,
        texture_scale=1.0,
        texture_condition_mode="token",
        use_palette_tokens=False,
        num_palette_tokens=4,
        use_conflict_aware_gate=False,
        conflict_texture_suppress_strength=0.1,
        conflict_palette_suppress_strength=0.4,
        conflict_deltae_norm=50.0,
        conflict_threshold=0.70,
        fusion_type="minimal",
        texture_preprocess_mode="plain_resize",
        alpha1=1.0,
        alpha2=1.0,
        alpha3=0.7,
        alpha4=0.5,
        spatial_mask=spatial_mask,
        debug_spatial=False,
        force_texture_num_tokens_override=False,
    )
    output[0].convert("RGB").save(destination)


def main():
    parser = argparse.ArgumentParser(
        description="Generate one BF test shard with one resident E5 model."
    )
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--gam_ckpt", required=True)
    parser.add_argument("--texture_ckpt", required=True)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--generation_seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()

    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError(
            f"invalid shard: shard_id={args.shard_id}, num_shards={args.num_shards}"
        )
    with open(args.split_path, "r", encoding="utf-8") as handle:
        split = json.load(handle)
    start = args.shard_id * len(split) // args.num_shards
    end = (args.shard_id + 1) * len(split) // args.num_shards
    shard = split[start:end]
    run_dir = os.path.join(args.output_dir, args.run_name)
    status_dir = os.path.join(run_dir, "shards")
    os.makedirs(status_dir, exist_ok=True)

    pending = []
    for sample in shard:
        sample_dir, destination = output_path(run_dir, sample)
        if args.overwrite or not os.path.isfile(destination):
            pending.append((sample, sample_dir, destination))
    print(
        f"[shard] id={args.shard_id}/{args.num_shards}, range=[{start},{end}), "
        f"samples={len(shard)}, pending={len(pending)}"
    )

    generated = 0
    if pending:
        inference = load_inference_module(args.project_root)
        model_args = prepare_args(args)
        pipe, _ = inference.prepare(model_args)
        print("[shard] E5 loaded once; generation starts")
        for index, (sample, sample_dir, destination) in enumerate(pending, start=1):
            os.makedirs(sample_dir, exist_ok=True)
            seed = args.generation_seed + int(sample["sample_id"])
            generate_one(
                pipe, model_args, sample, args.data_root, seed, destination
            )
            generated += 1
            print(
                f"[shard] {index}/{len(pending)} sample_id={sample['sample_id']} "
                f"category={sample.get('category')} seed={seed}"
            )

    missing = []
    for sample in shard:
        _, destination = output_path(run_dir, sample)
        if not os.path.isfile(destination):
            missing.append(sample["sample_id"])
    status = {
        "status": "completed" if not missing else "incomplete",
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "sample_id_start": start,
        "sample_id_end": end,
        "num_samples": len(shard),
        "newly_generated": generated,
        "missing_sample_ids": missing,
    }
    status_path = os.path.join(status_dir, f"shard_{args.shard_id:04d}.json")
    with open(status_path, "w", encoding="utf-8") as handle:
        json.dump(status, handle, ensure_ascii=False, indent=2)
    print(json.dumps(status, ensure_ascii=False))
    if missing:
        raise RuntimeError(f"shard generation incomplete: {len(missing)} missing")


if __name__ == "__main__":
    main()
