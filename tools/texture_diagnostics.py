#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import random
from pathlib import Path


def solid_texture(color, size=(512, 640)):
    from PIL import Image
    return Image.new("RGB", size, color=color)


def striped_texture(size=(512, 640), c1=(230, 230, 230), c2=(50, 50, 50), step=24):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", size, c1)
    dr = ImageDraw.Draw(img)
    for x in range(0, size[0], step):
        dr.rectangle([x, 0, x + step // 2, size[1]], fill=c2)
    return img


def plaid_texture(size=(512, 640)):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", size, (210, 210, 210))
    dr = ImageDraw.Draw(img)
    for x in range(0, size[0], 40):
        dr.line((x, 0, x, size[1]), fill=(30, 30, 120), width=6)
    for y in range(0, size[1], 40):
        dr.line((0, y, size[0], y), fill=(120, 30, 30), width=6)
    return img


def noise_texture(size=(512, 640)):
    from PIL import Image
    img = Image.new("RGB", size)
    px = img.load()
    for y in range(size[1]):
        for x in range(size[0]):
            px[x, y] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return img


def mean_rgb(image, mask=None):
    from PIL import Image
    img = image.convert("RGB")
    pixels = list(img.getdata())
    if mask is None:
        n = max(1, len(pixels))
        r = sum(p[0] for p in pixels) / n
        g = sum(p[1] for p in pixels) / n
        b = sum(p[2] for p in pixels) / n
        return [r, g, b]
    m_img = mask.convert("L").resize(img.size, Image.NEAREST)
    m = list(m_img.getdata())
    weighted = [p for p, mv in zip(pixels, m) if mv > 0]
    if len(weighted) == 0:
        weighted = pixels
    n = max(1, len(weighted))
    r = sum(p[0] for p in weighted) / n
    g = sum(p[1] for p in weighted) / n
    b = sum(p[2] for p in weighted) / n
    return [r, g, b]


def run_case(args, texture_path, output_path):
    cmd = [
        "python",
        "inference_IMAGGarment-1.py",
        "--GAM_model_ckpt",
        args.gam_ckpt,
        "--texture_ckpt",
        args.texture_ckpt,
        "--sketch_path",
        args.sketch_path,
        "--texture_path",
        texture_path,
        "--prompt",
        args.prompt,
        "--output_path",
        str(output_path.parent),
        "--device",
        args.device,
        "--texture_condition_mode",
        args.texture_condition_mode,
        "--fusion_type",
        args.fusion_type,
        "--texture_preprocess_mode",
        args.texture_preprocess_mode,
        "--alpha1",
        str(args.alpha1),
        "--alpha2",
        str(args.alpha2),
        "--alpha3",
        str(args.alpha3),
        "--alpha4",
        str(args.alpha4),
    ]
    subprocess.run(cmd, check=True)
    return output_path


def build_grid(images, cols=4):
    from PIL import Image
    if len(images) == 0:
        return None
    w = max(img.size[0] for img in images)
    h = max(img.size[1] for img in images)
    rows = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))
    for idx, img in enumerate(images):
        grid.paste(img.resize((w, h), Image.BICUBIC), ((idx % cols) * w, (idx // cols) * h))
    return grid


def main():
    ap = argparse.ArgumentParser(description="Texture usage diagnostics runner.")
    ap.add_argument("--gam_ckpt", required=True)
    ap.add_argument("--texture_ckpt", required=True)
    ap.add_argument("--sketch_path", required=True)
    ap.add_argument("--real_texture_path", default=None, help="Optional real texture path used in null-texture test.")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--output_dir", default="diagnostics_output")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--texture_condition_mode", default="spatial", choices=["token", "spatial", "hybrid"])
    ap.add_argument("--fusion_type", default="minimal", choices=["minimal", "bfm_like"])
    ap.add_argument("--texture_preprocess_mode", default="crop_tile", choices=["plain_resize", "crop_tile", "plain"])
    ap.add_argument("--conflict_prompt", default=None, help="Optional prompt for prompt-vs-texture conflict test.")
    ap.add_argument("--alpha1", type=float, default=1.0)
    ap.add_argument("--alpha2", type=float, default=1.0)
    ap.add_argument("--alpha3", type=float, default=0.7)
    ap.add_argument("--alpha4", type=float, default=0.5)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    tmp_tex = out_dir / "textures"
    tmp_tex.mkdir(parents=True, exist_ok=True)

    tests = {
        "null_texture": {
            "gray": solid_texture((127, 127, 127)),
            "zero_black": solid_texture((0, 0, 0)),
            "random_noise": noise_texture(),
        },
        "solid_sweep": {
            "red": solid_texture((220, 40, 40)),
            "green": solid_texture((40, 220, 40)),
            "blue": solid_texture((40, 40, 220)),
            "yellow": solid_texture((220, 220, 40)),
            "magenta": solid_texture((220, 40, 220)),
            "cyan": solid_texture((40, 220, 220)),
        },
        "pattern_sweep": {
            "stripes": striped_texture(),
            "plaid": plaid_texture(),
            "floral_like": noise_texture(),
            "denim_like": solid_texture((70, 90, 150)),
            "camouflage_like": striped_texture(c1=(80, 110, 70), c2=(120, 90, 60), step=36),
        },
        "conflict_test": {
            "red_texture": solid_texture((220, 40, 40)),
            "blue_texture": solid_texture((40, 40, 220)),
            "green_texture": solid_texture((40, 220, 40)),
        },
    }

    summary = {"tests": {}, "args": vars(args)}
    sketch_name = Path(args.sketch_path).name

    for test_name, items in tests.items():
        test_out = out_dir / test_name
        test_out.mkdir(parents=True, exist_ok=True)
        generated = []
        summary["tests"][test_name] = {}
        for name, img in items.items():
            tex_path = tmp_tex / f"{test_name}_{name}.png"
            if img is None:
                # null test uses first provided real texture path fallback as sketch file sibling
                continue
            img.save(tex_path)
            local_args = argparse.Namespace(**vars(args))
            if test_name == "conflict_test" and args.conflict_prompt:
                local_args.prompt = args.conflict_prompt
            run_case(local_args, str(tex_path), test_out / sketch_name)
            out_img_path = test_out / sketch_name
            if out_img_path.exists():
                from PIL import Image
                out_img = Image.open(out_img_path).convert("RGB")
                generated.append(out_img)
                summary["tests"][test_name][name] = {"output": str(out_img_path), "mean_rgb": mean_rgb(out_img)}
        if test_name == "null_texture" and args.real_texture_path:
            run_case(args, args.real_texture_path, test_out / sketch_name)
            out_img_path = test_out / sketch_name
            if out_img_path.exists():
                from PIL import Image
                out_img = Image.open(out_img_path).convert("RGB")
                generated.insert(0, out_img)
                summary["tests"][test_name]["real"] = {"output": str(out_img_path), "mean_rgb": mean_rgb(out_img)}

        grid = build_grid(generated, cols=3)
        if grid is not None:
            grid.save(test_out / "grid.png")

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
