#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from eval.benchmark_utils import ensure_dir, write_csv, write_json, write_manifest, write_markdown_table
from eval.metrics import evaluate_pair, histogram_rgb, hist_l1, lpips_like


def _solid(path, color, size=(512, 640)):
    from PIL import Image

    Image.new("RGB", size, color=color).save(path)


def _noise(path, size=(512, 640)):
    import random
    from PIL import Image

    img = Image.new("RGB", size)
    px = img.load()
    for y in range(size[1]):
        for x in range(size[0]):
            px[x, y] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    img.save(path)


def run_infer(args, texture_path, mode, fusion_type, out_dir):
    ensure_dir(out_dir)
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
        out_dir,
        "--device",
        args.device,
        "--texture_condition_mode",
        mode,
        "--fusion_type",
        fusion_type,
        "--texture_preprocess_mode",
        args.texture_preprocess_mode,
    ]
    subprocess.run(cmd, check=True)
    src = os.path.join(out_dir, os.path.basename(args.sketch_path))
    dst = os.path.join(out_dir, "generated.png")
    if os.path.exists(src):
        shutil.move(src, dst)
    return dst


def main():
    ap = argparse.ArgumentParser(description="Analyze texture reliance with controlled texture perturbations.")
    ap.add_argument("--gam_ckpt", required=True)
    ap.add_argument("--texture_ckpt", required=True)
    ap.add_argument("--sketch_path", required=True)
    ap.add_argument("--real_texture_path", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--output_dir", default="eval_outputs/texture_reliance")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--modes", default="token,spatial,hybrid")
    ap.add_argument("--texture_preprocess_mode", default="crop_tile", choices=["plain_resize", "crop_tile", "plain"])
    args = ap.parse_args()

    ensure_dir(args.output_dir)
    tex_dir = os.path.join(args.output_dir, "textures")
    ensure_dir(tex_dir)
    gray_tex = os.path.join(tex_dir, "gray.png")
    zero_tex = os.path.join(tex_dir, "zero_black.png")
    noise_tex = os.path.join(tex_dir, "noise.png")
    red_tex = os.path.join(tex_dir, "red.png")
    blue_tex = os.path.join(tex_dir, "blue.png")
    _solid(gray_tex, (127, 127, 127))
    _solid(zero_tex, (0, 0, 0))
    _solid(red_tex, (220, 40, 40))
    _solid(blue_tex, (40, 40, 220))
    _noise(noise_tex)

    variants = {
        "real": args.real_texture_path,
        "gray": gray_tex,
        "zero_black": zero_tex,
        "noise": noise_tex,
        "red": red_tex,
        "blue": blue_tex,
    }

    rows = []
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        fusion = "minimal"
        if mode == "spatial_bfm_like":
            mode = "spatial"
            fusion = "bfm_like"
        base_path = run_infer(args, variants["real"], mode, fusion, os.path.join(args.output_dir, mode, "real"))
        from PIL import Image

        base_img = Image.open(base_path).convert("RGB")
        base_hist = histogram_rgb(base_img)
        for v_name, v_tex in variants.items():
            gen_path = run_infer(args, v_tex, mode, fusion, os.path.join(args.output_dir, mode, v_name))
            img = Image.open(gen_path).convert("RGB")
            row = {"mode": mode, "fusion_type": fusion, "variant": v_name, "gen_path": gen_path}
            if v_name != "real":
                row["output_lpips_like_vs_real"] = lpips_like(img, base_img)
                row["output_hist_l1_vs_real"] = hist_l1(histogram_rgb(img), base_hist)
            row.update(evaluate_pair(gen_path, texture_path=v_tex))
            rows.append(row)

    summary = {}
    for r in rows:
        key = (r["mode"], r["fusion_type"])
        summary.setdefault(key, [])
        summary[key].append(r)
    summary_rows = []
    for (m, f), vals in summary.items():
        s = {"mode": m, "fusion_type": f, "count": len(vals)}
        num_keys = [k for k, v in vals[0].items() if isinstance(v, (int, float))]
        for k in num_keys:
            s[k] = sum(float(x.get(k, 0.0)) for x in vals) / max(1, len(vals))
        summary_rows.append(s)

    write_csv(os.path.join(args.output_dir, "texture_reliance.csv"), rows)
    write_json(os.path.join(args.output_dir, "texture_reliance.json"), rows)
    write_markdown_table(os.path.join(args.output_dir, "texture_reliance.md"), summary_rows, title="Texture Reliance Summary")
    write_manifest(
        os.path.join(args.output_dir, "experiment_manifest.json"),
        {
            "task": "texture_reliance",
            "modes": args.modes,
            "prompt": args.prompt,
            "sketch_path": args.sketch_path,
            "real_texture_path": args.real_texture_path,
            "texture_preprocess_mode": args.texture_preprocess_mode,
        },
    )


if __name__ == "__main__":
    main()
