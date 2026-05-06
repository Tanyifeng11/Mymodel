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
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    spec = importlib.util.spec_from_file_location("imag_infer", os.path.join(os.path.dirname(__file__), "..", "inference_IMAGGarment-1.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pipe, generator = mod.prepare(args)

    sketch = Image.open(args.sketch_path).convert("RGB")
    outputs = []
    for tp in args.texture_paths:
        tex = Image.open(tp).convert("RGB")
        local_gen = torch.Generator(device=args.device).manual_seed(args.fixed_seed)
        out = pipe(
            ref_image=transforms.Compose([
                transforms.Resize([640, 512]),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])(sketch).unsqueeze(0),
            prompt=args.prompt,
            texture_clip_image=tex,
            texture_embeds=None,
            null_prompt="",
            negative_prompt=" worst quality, low quality",
            width=512,
            height=640,
            num_images_per_prompt=1,
            guidance_scale=7.0,
            sketch_scale=0.6,
            ipa_scale=1.0,
            generator=local_gen,
            num_inference_steps=50,
            texture_mode=args.texture_mode,
            texture_num_tokens=args.texture_num_tokens,
            texture_scale=args.texture_scale,
        )[0]
        outputs.append(out)

    tensors = [img_to_tensor(x) for x in outputs]
    dists = []
    for i in range(len(tensors)):
        for j in range(i + 1, len(tensors)):
            dists.append(((i, j), l2_dist(tensors[i], tensors[j])))

    with open(os.path.join(args.output_dir, "pairwise_l2.txt"), "w", encoding="utf-8") as f:
        for (i, j), d in dists:
            f.write(f"{i}-{j}: {d:.6f}\n")

    grid = make_grid(tensors, nrow=len(tensors))
    save_image(grid, os.path.join(args.output_dir, "texture_sensitivity_grid.png"))
    print("Saved:", os.path.join(args.output_dir, "texture_sensitivity_grid.png"))
    print("Distances:", dists)


if __name__ == "__main__":
    main()
