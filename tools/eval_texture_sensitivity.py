import argparse
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid, save_image

import importlib.util



def img_to_tensor(img):
    return transforms.ToTensor()(img)


def l2_dist(a, b):
    return torch.mean((a - b) ** 2).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--GAM_model_ckpt", required=True)
    ap.add_argument("--texture_ckpt", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--sketch_path", required=True)
    ap.add_argument("--texture_paths", nargs="+", required=True)
    ap.add_argument("--output_dir", default="./texture_eval")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--texture_mode", default="patch_resampled", choices=["patch_resampled", "legacy_pooled"])
    ap.add_argument("--texture_num_tokens", type=int, default=16)
    ap.add_argument("--texture_scale", type=float, default=1.0)
    ap.add_argument("--fixed_seed", type=int, default=1234)
    # 下面这些 prepare() 都会读, 缺一个就 AttributeError。默认值对齐
    # inference_IMAGGarment-1.py, 但与训练相关的几个必须显式传对:
    # E5 是 token 模式 + TCPM-lite, 用默认的 spatial/0 加载出来的模型是错的。
    ap.add_argument("--texture_condition_mode", default="token",
                    choices=["token", "spatial", "hybrid"])
    ap.add_argument("--use_tcpm_lite", type=int, default=1, choices=[0, 1])
    ap.add_argument("--texture_preprocess_mode", default="plain_resize",
                    choices=["plain_resize", "crop_tile", "plain"])
    ap.add_argument("--use_texture_gate", type=int, default=1, choices=[0, 1])
    ap.add_argument("--layer_group_enabled", type=int, default=1, choices=[0, 1])
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--guidance_scale", type=float, default=7.0)
    ap.add_argument("--sketch_scale", type=float, default=0.6)
    ap.add_argument("--ipa_scale", type=float, default=1.0)
    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--base_model_path", default="auto")
    ap.add_argument("--vae_model_path", default="auto")
    ap.add_argument("--image_encoder_path", default="auto")
    ap.add_argument("--tcpm_hidden_ratio", type=float, default=0.25)
    ap.add_argument("--tcpm_residual_scale_init", type=float, default=0.0)
    ap.add_argument("--use_palette_tokens", type=int, default=0, choices=[0, 1])
    ap.add_argument("--num_palette_tokens", type=int, default=4)
    ap.add_argument("--palette_branch_scale_init", type=float, default=0.0)
    ap.add_argument("--gate_type", default="layer")
    ap.add_argument("--gate_init", default="identity")
    ap.add_argument("--gate_min", type=float, default=0.7)
    ap.add_argument("--gate_max", type=float, default=1.3)
    ap.add_argument("--use_balanced_fusion_gate", type=int, default=0, choices=[0, 1])
    ap.add_argument("--balanced_gate_hidden_dim", type=int, default=64)
    ap.add_argument("--balanced_gate_scale", type=float, default=0.2)
    ap.add_argument("--balanced_gate_min", type=float, default=0.8)
    ap.add_argument("--balanced_gate_max", type=float, default=1.2)
    ap.add_argument("--balanced_gate_trace_path", default="")
    ap.add_argument("--balanced_gate_trace_sample_id", default="")
    ap.add_argument("--use_conflict_aware_gate", type=int, default=0, choices=[0, 1])
    ap.add_argument("--conflict_texture_suppress_strength", type=float, default=0.1)
    ap.add_argument("--conflict_palette_suppress_strength", type=float, default=0.4)
    ap.add_argument("--conflict_deltae_norm", type=float, default=50.0)
    ap.add_argument("--conflict_threshold", type=float, default=0.70)
    ap.add_argument("--fusion_type", default="minimal", choices=["minimal", "bfm_like"])
    ap.add_argument("--lora_rank", type=int, default=4)
    ap.add_argument("--alpha1", type=float, default=2.0)
    ap.add_argument("--alpha2", type=float, default=2.0)
    ap.add_argument("--alpha3", type=float, default=1.5)
    ap.add_argument("--alpha4", type=float, default=1.0)
    ap.add_argument("--debug_spatial", action="store_true")
    ap.add_argument("--force_texture_num_tokens_override", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_path", default="./texture_eval/_tmp")
    ap.add_argument("--texture_path", default="")
    args = ap.parse_args()

    # prepare() 读 args.texture_path, 但本工具是多张纹理轮流跑, 先占位
    if not args.texture_path:
        args.texture_path = args.texture_paths[0]

    os.makedirs(args.output_dir, exist_ok=True)
    spec = importlib.util.spec_from_file_location("imag_infer", os.path.join(os.path.dirname(__file__), "..", "inference_IMAGGarment-1.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pipe, generator = mod.prepare(args)

    # prepare() 会就地把分辨率写回 args(优先 checkpoint metadata, 兜底 512x640)。
    # 原来这里写死 640x512, 与训练分辨率不符会影响结论, 现在跟着 checkpoint 走。
    width, height = args.width, args.height
    print(f"[info] inference resolution = {height} x {width}")

    sketch = Image.open(args.sketch_path).convert("RGB")
    outputs = []
    for tp in args.texture_paths:
        tex = Image.open(tp).convert("RGB")
        local_gen = torch.Generator(device=args.device).manual_seed(args.fixed_seed)
        out = pipe(
            ref_image=transforms.Compose([
                transforms.Resize([height, width]),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])(sketch).unsqueeze(0),
            prompt=args.prompt,
            texture_clip_image=tex,
            texture_embeds=None,
            null_prompt="",
            negative_prompt=" worst quality, low quality",
            width=width,
            height=height,
            num_images_per_prompt=1,
            guidance_scale=args.guidance_scale,
            sketch_scale=args.sketch_scale,
            ipa_scale=args.ipa_scale,
            generator=local_gen,
            num_inference_steps=args.num_inference_steps,
            texture_mode=args.texture_mode,
            texture_num_tokens=args.texture_num_tokens,
            texture_scale=args.texture_scale,
        )[0]
        outputs.append(out)

    tensors = [img_to_tensor(x) for x in outputs]
    # 单独存每张图, 只看网格不好判断具体哪张纹理对应哪个结果
    for i, (t, tp) in enumerate(zip(tensors, args.texture_paths)):
        stem = os.path.splitext(os.path.basename(tp))[0]
        save_image(t, os.path.join(args.output_dir, f"out_{i}_{stem}.png"))

    dists = []
    for i in range(len(tensors)):
        for j in range(i + 1, len(tensors)):
            dists.append(((i, j), l2_dist(tensors[i], tensors[j])))

    with open(os.path.join(args.output_dir, "pairwise_l2.txt"), "w", encoding="utf-8") as f:
        f.write(f"prompt: {args.prompt}\n")
        f.write(f"sketch: {args.sketch_path}\n")
        for i, tp in enumerate(args.texture_paths):
            f.write(f"[{i}] {tp}\n")
        f.write("\npairwise L2 (越大说明纹理影响越强):\n")
        for (i, j), d in dists:
            f.write(f"{i}-{j}: {d:.6f}\n")
        if dists:
            vals = [d for _, d in dists]
            f.write("\nmean=%.6f  min=%.6f  max=%.6f\n"
                    % (sum(vals) / len(vals), min(vals), max(vals)))

    grid = make_grid(tensors, nrow=len(tensors))
    save_image(grid, os.path.join(args.output_dir, "texture_sensitivity_grid.png"))
    print("Saved:", os.path.join(args.output_dir, "texture_sensitivity_grid.png"))
    print("Distances:", dists)
    if dists:
        vals = [d for _, d in dists]
        print("mean L2 = %.6f" % (sum(vals) / len(vals)))


if __name__ == "__main__":
    main()
