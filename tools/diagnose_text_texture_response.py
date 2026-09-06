"""用已有权重检查条件响应；只做推理，不以响应幅度判定生成质量。"""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms

from test_plain_prompt_texture_control import build_parser


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def difference(left, right):
    if left.shape != right.shape:
        return {"comparable": False, "left_shape": list(left.shape), "right_shape": list(right.shape)}
    left, right = left.float(), right.float()
    if not (torch.isfinite(left).all() and torch.isfinite(right).all()):
        return {"comparable": False, "nonfinite": True}
    delta = left - right
    rms = float(left.square().mean().sqrt())
    return {
        "comparable": True,
        "exact_equal": bool(torch.equal(left, right)),
        "mse": float(delta.square().mean()),
        "relative_rms": float(delta.square().mean().sqrt()) / max(rms, 1e-8),
        "max_abs": float(delta.abs().max()),
    }


class Trace:
    """只保留每次推理的首次条件调用；第二次通常是 CFG 无条件调用。"""

    def __init__(self, pipe):
        self.values = {}
        self.calls = {}
        self.handles = []
        self.hook(pipe.text_encoder, "text_tokens", lambda args, out: out[0])
        self.hook(pipe.image_encoder, "clip_visual_tokens", lambda args, out: out.hidden_states[-1])
        bf = pipe.bf_texture_conditioner
        if bf is None:
            raise RuntimeError("未加载 BFTextureConditioner，不能按 BF 路径诊断")
        self.hook(bf.resampler, "patches_before_resampler", lambda args, out: args[1])
        self.hook(bf, "tokens_after_bf", lambda args, out: out[0])
        if pipe.use_tcpm_lite:
            self.hook(pipe.tcpm_lite, "tokens_after_tcpm", lambda args, out: out)
        if pipe.aa_tcr_fuser is not None:
            self.hook(pipe.aa_tcr_fuser, "tokens_after_e7a", lambda args, out: out)
        self.handles.append(pipe.unet.register_forward_pre_hook(self.unet_input, with_kwargs=True))
        self.hook(pipe.unet, "first_cond_noise", lambda args, out: out[0])

    def save(self, name, tensor):
        self.calls[name] = self.calls.get(name, 0) + 1
        if name not in self.values:
            self.values[name] = tensor.detach().float().cpu().clone()

    def hook(self, module, name, select):
        def callback(module, args, output):
            if name == "first_cond_noise" and self.calls.get(name) == 1:
                self.save("first_uncond_noise", select(args, output))
            self.save(name, select(args, output))
        self.handles.append(module.register_forward_hook(callback))

    def unet_input(self, module, args, kwargs):
        self.save("first_latent_input", args[0])
        self.save("unet_condition", kwargs["encoder_hidden_states"])

    def reset(self):
        self.values = {}
        self.calls = {}

    def summary(self):
        return {
            name: {
                "shape": list(t.shape), "calls": self.calls.get(name, 1),
                "finite": bool(torch.isfinite(t).all()),
                "rms": float(t.square().mean().sqrt()) if torch.isfinite(t).all() else None,
            }
            for name, t in self.values.items()
        }

    def close(self):
        for handle in self.handles:
            handle.remove()


def build_cases(args):
    cases = [
        {"name": f"texture_{i}", "prompt": args.prompt, "texture": str(Path(p).resolve())}
        for i, p in enumerate(args.texture_paths)
    ]
    cases += [
        {"name": f"text_{i}", "prompt": p, "texture": cases[0]["texture"]}
        for i, p in enumerate(args.text_prompts)
    ]
    # None 会走管线的关闭分支逻辑；不以全零 token 冒充关闭分支。
    cases += [
        {"name": "texture_off", "prompt": args.prompt, "texture": None},
        {**cases[0], "name": "repeat_baseline"},
    ]
    return cases


def save_grid(path, records, images):
    cell_w, cell_h = 280, 340
    grid = Image.new("RGB", (cell_w * 4, cell_h * ((len(images) + 3) // 4)), "white")
    draw = ImageDraw.Draw(grid)
    for i, (record, picture) in enumerate(zip(records, images)):
        x, y = (i % 4) * cell_w, (i // 4) * cell_h
        # 默认提示词为英文，避免服务器 PIL 默认字体缺少中文字形。
        label = f"{record['name']}\n{record['prompt']}".encode("ascii", "replace").decode()
        draw.text((x + 4, y + 4), label, fill="black")
        if record["texture"]:
            with Image.open(record["texture"]) as ref:
                grid.paste(ref.convert("RGB").resize((48, 48)), (x + cell_w - 52, y + 4))
        picture = picture.copy()
        picture.thumbnail((cell_w - 8, cell_h - 60))
        grid.paste(picture, (x + 4, y + 56))
    grid.save(path)


def main():
    parser = build_parser()
    parser.description = "E5/E7a 文本与纹理响应诊断（不训练、不计算 FID）"
    parser.add_argument("--text_prompts", nargs="+", default=["a red cloth", "a blue cloth"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    args = parser.parse_args()
    if args.texture_condition_mode != "token" or args.use_palette_tokens:
        raise ValueError("本诊断限定 token 模式且不启用 palette，与 E5/E7a 对照配置一致")
    if len(args.texture_paths) < 2:
        raise ValueError("至少提供两张不同纹理图")
    for p in [args.GAM_model_ckpt, args.texture_ckpt, args.sketch_path, *args.texture_paths]:
        if not Path(p).is_file():
            raise FileNotFoundError(p)
    out = Path(args.output_dir)
    # 拒绝复用目录，避免历史图像造成假性的零响应。
    out.mkdir(parents=True, exist_ok=False)
    args.texture_path = args.texture_paths[0]
    args.seed = args.seeds[0]
    args.output_path = str(out / "_unused")
    write_json(out / "requested_args.json", vars(args))
    root = Path(__file__).resolve().parents[1]
    # 保存实际运行代码的指纹，便于排查服务器代码与本地版本不一致。
    source_paths = [
        "tools/diagnose_text_texture_response.py", "inference_IMAGGarment-1.py",
        "pipelines/IMAGGarment_pipeline.py", "adapter/attention_processor.py",
        "models/bf_texture_module.py", "models/tcpm_lite.py",
        "models/attribute_text_texture_fuser.py",
    ]
    write_json(out / "code_sha256.json", {
        p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in source_paths
    })

    spec = importlib.util.spec_from_file_location(
        "imag_response_infer", Path(__file__).resolve().parents[1] / "inference_IMAGGarment-1.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pipe, _ = module.prepare(args)
    for value in vars(pipe).values():
        if isinstance(value, torch.nn.Module):
            value.eval()
    torch.backends.cudnn.benchmark = False
    trace = Trace(pipe)
    write_json(out / "effective_args.json", vars(args))
    write_json(out / "texture_metadata.json", pipe.texture_meta)
    transform = transforms.Compose([
        transforms.Resize([args.height, args.width]),
        transforms.ToTensor(), transforms.Normalize([0.5], [0.5]),
    ])
    with Image.open(args.sketch_path) as sketch:
        ref_image = transform(sketch.convert("RGB")).unsqueeze(0)
    cases = build_cases(args)
    summary = {"checkpoint": str(Path(args.GAM_model_ckpt).resolve()), "seeds": {}}
    try:
        for seed in args.seeds:
            seed_dir = out / f"seed_{seed}"
            seed_dir.mkdir()
            images, records = [], []
            baseline = None
            for case in cases:
                trace.reset()
                torch.manual_seed(seed)
                np.random.seed(seed)
                generator = torch.Generator(device=args.device).manual_seed(seed)
                texture = None
                if case["texture"]:
                    with Image.open(case["texture"]) as source:
                        texture = source.convert("RGB")

                def final_latents(step, timestep, latents):
                    if step == args.num_inference_steps - 1:
                        trace.save("final_latents", latents)

                print(f"[diagnose] seed={seed} case={case['name']} prompt={case['prompt']}", flush=True)
                with torch.inference_mode():
                    picture = pipe(
                        ref_image=ref_image, prompt=case["prompt"], texture_clip_image=texture,
                        texture_embeds=None, null_prompt="", negative_prompt=" worst quality, low quality",
                        width=args.width, height=args.height, num_images_per_prompt=1,
                        guidance_scale=args.guidance_scale, sketch_scale=args.sketch_scale,
                        ipa_scale=args.ipa_scale, generator=generator,
                        num_inference_steps=args.num_inference_steps,
                        texture_mode=args.texture_mode, texture_condition_mode=args.texture_condition_mode,
                        texture_preprocess_mode=args.texture_preprocess_mode,
                        texture_num_tokens=args.texture_num_tokens, texture_scale=args.texture_scale,
                        callback=final_latents, callback_steps=1,
                    )[0]
                picture.save(seed_dir / f"{case['name']}.png")
                pixels = np.asarray(picture.convert("RGB"), dtype=np.uint8)
                trace.save("output_pixels", torch.from_numpy(pixels.copy()).float() / 255)
                if baseline is None:
                    baseline = dict(trace.values)
                record = {
                    **case, "pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                    "pixel_std": float(pixels.astype(np.float32).std()),
                    "trace": trace.summary(),
                    "vs_baseline": {
                        name: difference(baseline[name], value)
                        for name, value in trace.values.items() if name in baseline
                    },
                    "stages_absent_vs_baseline": sorted(set(baseline) - set(trace.values)),
                }
                write_json(seed_dir / f"{case['name']}.json", record)
                records.append(record)
                images.append(picture)
            save_grid(seed_dir / "grid.png", records, images)
            summary["seeds"][str(seed)] = records
            write_json(out / "summary.json", summary)
    finally:
        trace.close()
    print(f"[done] {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
