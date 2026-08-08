"""固定草图与简易文本，验证不同纹理是否能驱动不同的服装生成结果。"""

import argparse
import importlib.util
import itertools
import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Plain-prompt texture-control test")
    parser.add_argument("--GAM_model_ckpt", required=True)
    parser.add_argument("--texture_ckpt", required=True)
    parser.add_argument("--sketch_path", required=True)
    parser.add_argument("--texture_paths", nargs="+", required=True)
    parser.add_argument("--output_dir", default="eval_outputs/plain_prompt_texture_control")
    parser.add_argument("--prompt", default="a cloth")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fixed_seed", type=int, default=42)
    parser.add_argument("--min_pairwise_mse", type=float, default=0.002)

    # E7a 的默认推理配置；若要检查旧模型，可以显式传 --use_aa_tcr_fuse 0。
    parser.add_argument("--texture_condition_mode", default="token", choices=["token", "spatial", "hybrid"])
    parser.add_argument("--texture_mode", default="patch_resampled", choices=["patch_resampled", "legacy_pooled"])
    parser.add_argument("--texture_preprocess_mode", default="plain_resize", choices=["plain_resize", "crop_tile", "plain"])
    parser.add_argument("--texture_num_tokens", type=int, default=16)
    parser.add_argument("--texture_scale", type=float, default=1.0)
    parser.add_argument("--use_tcpm_lite", type=int, choices=[0, 1], default=1)
    parser.add_argument("--use_aa_tcr_fuse", type=int, choices=[0, 1], default=1)
    parser.add_argument("--use_texture_gate", type=int, choices=[0, 1], default=1)
    parser.add_argument("--layer_group_enabled", type=int, choices=[0, 1], default=1)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--sketch_scale", type=float, default=0.6)
    parser.add_argument("--ipa_scale", type=float, default=1.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--base_model_path", default="auto")
    parser.add_argument("--vae_model_path", default="auto")
    parser.add_argument("--image_encoder_path", default="auto")
    parser.add_argument("--force_texture_num_tokens_override", action="store_true")

    # inference_IMAGGarment-1.py 的 prepare() 需要这些配置项。
    parser.add_argument("--tcpm_hidden_ratio", type=float, default=0.25)
    parser.add_argument("--tcpm_residual_scale_init", type=float, default=0.0)
    parser.add_argument("--use_palette_tokens", type=int, choices=[0, 1], default=0)
    parser.add_argument("--num_palette_tokens", type=int, default=4)
    parser.add_argument("--palette_branch_scale_init", type=float, default=0.0)
    parser.add_argument("--gate_type", default="layer")
    parser.add_argument("--gate_init", default="identity")
    parser.add_argument("--gate_min", type=float, default=0.7)
    parser.add_argument("--gate_max", type=float, default=1.3)
    parser.add_argument("--use_balanced_fusion_gate", type=int, choices=[0, 1], default=0)
    parser.add_argument("--balanced_gate_hidden_dim", type=int, default=64)
    parser.add_argument("--balanced_gate_scale", type=float, default=0.2)
    parser.add_argument("--balanced_gate_min", type=float, default=0.8)
    parser.add_argument("--balanced_gate_max", type=float, default=1.2)
    parser.add_argument("--balanced_gate_trace_path", default="")
    parser.add_argument("--balanced_gate_trace_sample_id", default="")
    parser.add_argument("--use_conflict_aware_gate", type=int, choices=[0, 1], default=0)
    parser.add_argument("--conflict_texture_suppress_strength", type=float, default=0.1)
    parser.add_argument("--conflict_palette_suppress_strength", type=float, default=0.4)
    parser.add_argument("--conflict_deltae_norm", type=float, default=50.0)
    parser.add_argument("--conflict_threshold", type=float, default=0.70)
    parser.add_argument("--fusion_type", default="minimal", choices=["minimal", "bfm_like"])
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--alpha1", type=float, default=2.0)
    parser.add_argument("--alpha2", type=float, default=2.0)
    parser.add_argument("--alpha3", type=float, default=1.5)
    parser.add_argument("--alpha4", type=float, default=1.0)
    parser.add_argument("--debug_spatial", action="store_true")
    return parser.parse_args()


def mse(left, right):
    a = np.asarray(left, dtype=np.float32) / 255.0
    b = np.asarray(right, dtype=np.float32) / 255.0
    return float(np.mean((a - b) ** 2))


def make_pair_grid(texture_paths, images, width, height, output_path):
    thumb_width = max(128, width // 3)
    row_height = height + 28
    canvas = Image.new("RGB", (thumb_width + width, len(images) * row_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (texture_path, image) in enumerate(zip(texture_paths, images)):
        y = index * row_height
        texture = Image.open(texture_path).convert("RGB").resize((thumb_width, height))
        canvas.paste(texture, (0, y + 28))
        canvas.paste(image, (thumb_width, y + 28))
        draw.text((4, y + 6), f"texture {index}: {os.path.basename(texture_path)}", fill="black")
        draw.text((thumb_width + 4, y + 6), f"generated {index}", fill="black")
    canvas.save(output_path)


def main():
    args = parse_args()
    if len(args.texture_paths) < 2:
        raise ValueError("至少提供两张纹理图")
    args.texture_path = args.texture_paths[0]  # prepare() 读取该字段。
    args.seed = args.fixed_seed
    args.output_path = os.path.join(args.output_dir, "_inference_tmp")
    os.makedirs(args.output_dir, exist_ok=True)

    spec = importlib.util.spec_from_file_location(
        "imag_infer", os.path.join(os.path.dirname(__file__), "..", "inference_IMAGGarment-1.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pipe, _ = module.prepare(args)

    sketch = Image.open(args.sketch_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize([args.height, args.width]),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    ref_image = transform(sketch).unsqueeze(0)

    images = []
    for index, texture_path in enumerate(args.texture_paths):
        texture = Image.open(texture_path).convert("RGB")
        generator = torch.Generator(device=args.device).manual_seed(args.fixed_seed)
        image = pipe(
            ref_image=ref_image,
            prompt=args.prompt,
            texture_clip_image=texture,
            texture_embeds=None,
            null_prompt="",
            negative_prompt=" worst quality, low quality",
            width=args.width,
            height=args.height,
            num_images_per_prompt=1,
            guidance_scale=args.guidance_scale,
            sketch_scale=args.sketch_scale,
            ipa_scale=args.ipa_scale,
            generator=generator,
            num_inference_steps=args.num_inference_steps,
            texture_mode=args.texture_mode,
            texture_condition_mode=args.texture_condition_mode,
            texture_preprocess_mode=args.texture_preprocess_mode,
            texture_num_tokens=args.texture_num_tokens,
            texture_scale=args.texture_scale,
        )[0]
        image.save(os.path.join(args.output_dir, f"generated_{index:02d}.png"))
        images.append(image)

    pairs = [
        {"left": i, "right": j, "mse": mse(images[i], images[j])}
        for i, j in itertools.combinations(range(len(images)), 2)
    ]
    minimum = min(item["mse"] for item in pairs)
    passed = minimum >= args.min_pairwise_mse
    make_pair_grid(
        args.texture_paths,
        images,
        args.width,
        args.height,
        os.path.join(args.output_dir, "texture_to_garment_grid.png"),
    )
    result = {
        "prompt": args.prompt,
        "fixed_seed": args.fixed_seed,
        "texture_paths": args.texture_paths,
        "min_pairwise_mse": args.min_pairwise_mse,
        "pairwise_mse": pairs,
        "minimum_mse": minimum,
        "passed": passed,
    }
    with open(os.path.join(args.output_dir, "result.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit("FAIL: 至少一对不同纹理的生成图差异不足")


if __name__ == "__main__":
    main()
