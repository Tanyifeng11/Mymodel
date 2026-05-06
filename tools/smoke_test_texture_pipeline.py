import argparse
import importlib.util
import os
import torch
from PIL import Image
from torchvision import transforms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--GAM_model_ckpt", required=True)
    ap.add_argument("--texture_ckpt", required=True)
    ap.add_argument("--sketch_path", required=True)
    ap.add_argument("--texture_path", required=True)
    ap.add_argument("--prompt", default="a cloth")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--texture_mode", default="patch_resampled", choices=["patch_resampled", "legacy_pooled"])
    ap.add_argument("--texture_num_tokens", type=int, default=16)
    ap.add_argument("--image_encoder_path", default="h94/IP-Adapter")
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("imag_infer", os.path.join(os.path.dirname(__file__), "..", "inference_IMAGGarment-1.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    pipe, _ = mod.prepare(args)
    sketch = Image.open(args.sketch_path).convert("RGB")
    texture = Image.open(args.texture_path).convert("RGB")

    vae_sketch = transforms.Compose([
        transforms.Resize([640, 512]),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])(sketch).unsqueeze(0)

    gen = torch.Generator(device=args.device).manual_seed(1234)
    out = pipe(
        ref_image=vae_sketch,
        prompt=args.prompt,
        texture_clip_image=texture,
        texture_embeds=None,
        null_prompt="",
        negative_prompt=" worst quality, low quality",
        width=512,
        height=640,
        num_images_per_prompt=1,
        guidance_scale=5.5,
        sketch_scale=0.6,
        ipa_scale=1.0,
        generator=gen,
        num_inference_steps=2,
        texture_mode=args.texture_mode,
        texture_num_tokens=args.texture_num_tokens,
    )

    print(f"[smoke] success, outputs={len(out)}")


if __name__ == "__main__":
    main()
