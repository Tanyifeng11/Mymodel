#!/usr/bin/env python3
import argparse
import math
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from eval.benchmark_utils import create_or_load_fixed_split, ensure_dir, write_csv, write_json
from eval.distribution_diagnostics import compute_distribution_metrics, resolve_device
from eval.metrics import compute_ssim, lpips_like


def resolve_path(data_root, value):
    if not value:
        return None
    if os.path.isabs(value):
        return value
    candidates = [os.path.join(data_root, value)]
    if os.path.normpath(value).split(os.sep)[0].lower() == "cloth":
        candidates.append(os.path.join(data_root, "gt", os.path.basename(value)))
    return next((path for path in candidates if os.path.isfile(path)), candidates[0])


def image_to_tensor(image, width, height):
    image = image.convert("RGB").resize((width, height), Image.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1), image


def tensor_to_image(tensor):
    array = tensor.detach().float().clamp(-1, 1).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(np.rint((array + 1.0) * 127.5).astype(np.uint8), "RGB")


def psnr(real, recon):
    a = np.asarray(real, dtype=np.float32) / 255.0
    b = np.asarray(recon, dtype=np.float32) / 255.0
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else float(10.0 * math.log10(1.0 / mse))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the VAE reconstruction ceiling")
    parser.add_argument("--dataset_json", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split_path", required=True)
    parser.add_argument("--vae_model_path", required=True)
    parser.add_argument("--output_dir", default="eval_outputs/phase2_diagnostics/vae_reconstruction")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--metric_batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--latent_mode", choices=["mode", "sample"], default="mode")
    parser.add_argument("--clean_fid", type=int, choices=[0, 1], default=0)
    parser.add_argument("--overwrite", type=int, choices=[0, 1], default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    from diffusers import AutoencoderKL

    device = resolve_device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" and device != "cpu" else torch.float32
    split = create_or_load_fixed_split(
        args.dataset_json, args.split_path, num_samples=args.num_samples, seed=args.seed
    )
    real_dir = os.path.join(args.output_dir, "real")
    recon_dir = os.path.join(args.output_dir, "reconstruction")
    ensure_dir(real_dir)
    ensure_dir(recon_dir)

    vae = AutoencoderKL.from_pretrained(args.vae_model_path, torch_dtype=dtype).to(device).eval()
    rows = []
    real_paths, recon_paths = [], []
    for start in range(0, len(split), args.batch_size):
        samples = split[start : start + args.batch_size]
        tensors, resized_images, output_pairs = [], [], []
        for sample in samples:
            sample_id = str(sample["sample_id"])
            real_out = os.path.join(real_dir, f"real_{sample_id}.png")
            recon_out = os.path.join(recon_dir, f"recon_{sample_id}.png")
            source = resolve_path(args.data_root, sample.get("target"))
            if not source or not os.path.isfile(source):
                raise FileNotFoundError(f"target image missing: {source}")
            tensor, resized = image_to_tensor(Image.open(source), args.width, args.height)
            tensors.append(tensor)
            resized_images.append(resized)
            output_pairs.append((real_out, recon_out, source, sample_id))

        batch = torch.stack(tensors).to(device=device, dtype=dtype)
        with torch.inference_mode():
            posterior = vae.encode(batch).latent_dist
            latents = posterior.mode() if args.latent_mode == "mode" else posterior.sample()
            decoded = vae.decode(latents).sample

        for real, recon_tensor, paths in zip(resized_images, decoded, output_pairs):
            real_out, recon_out, source, sample_id = paths
            recon = tensor_to_image(recon_tensor)
            if args.overwrite or not os.path.isfile(real_out):
                real.save(real_out)
            if args.overwrite or not os.path.isfile(recon_out):
                recon.save(recon_out)
            rows.append({
                "sample_id": sample_id,
                "source_path": source,
                "real_path": real_out,
                "reconstruction_path": recon_out,
                "ssim": compute_ssim(real, recon),
                "psnr": psnr(real, recon),
                "lpips_like": lpips_like(real, recon),
            })
            real_paths.append(real_out)
            recon_paths.append(recon_out)

    distribution = compute_distribution_metrics(
        real_paths,
        recon_paths,
        device=device,
        batch_size=args.metric_batch_size,
        seed=args.seed,
        clean_fid=bool(args.clean_fid),
        real_dir=real_dir,
        fake_dir=recon_dir,
    )
    summary = {
        "diagnostic": "vae_reconstruction",
        "vae_model_path": args.vae_model_path,
        "device": device,
        "latent_mode": args.latent_mode,
        "width": args.width,
        "height": args.height,
        **distribution,
        "mean_ssim": float(np.mean([row["ssim"] for row in rows])),
        "mean_psnr": float(np.mean([row["psnr"] for row in rows])),
        "mean_lpips_like": float(np.mean([row["lpips_like"] for row in rows])),
    }
    write_csv(os.path.join(args.output_dir, "per_sample_metrics.csv"), rows)
    write_json(os.path.join(args.output_dir, "summary.json"), summary)
    print(summary)


if __name__ == "__main__":
    main()
