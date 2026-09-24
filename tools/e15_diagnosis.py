"""E15-Diagnosis 入口：固定 32 样本与五参考条件，按 D1–D4 顺序做前向诊断。

D1 参考信息衰减曲线、D2 16 层 residual 热图、D3 timestep×区域离线重算、
D4 分组层消融。全部只读前向，不训练、不改权重。
"""

import argparse
from pathlib import Path

from tools.e15_common import CONDITIONS, FULL_STEPS, REGIONS, sample_indices, write_json


def build_inputs(args):
    """固定样本清单、配对与颜色近邻，与 113719／113817 完全一致。"""
    from PIL import Image
    from transformers import CLIPTokenizer

    from tools.e14_denoising_complete import color_pairs
    from tools.e14_matched_denoising import choose_pairs
    from tools.e14_pattern_probe import pixel_hash
    from train_texture_adapter import MyDataset

    tokenizer = CLIPTokenizer.from_pretrained(args.base_model, subfolder="tokenizer",
                                             local_files_only=True)
    dataset = MyDataset(args.manifest, tokenizer, height=512, width=384,
                        image_root_path=args.data_root, texture_preprocess_mode="plain_resize",
                        t_drop_rate=0, i_drop_rate=0, ti_drop_rate=0)
    indices = sample_indices(dataset, args.count)
    hashes = {}
    for index in indices:
        row = dataset.data[index]
        with Image.open(Path(args.data_root) / row.get("texture", row.get("color"))) as image:
            hashes[index] = pixel_hash(image.convert("RGB"))
    pairs = choose_pairs(dataset.data, indices, hashes)
    colors = color_pairs(dataset.data, indices, hashes, args.data_root)
    return dataset, indices, hashes, pairs, colors


def load_regions(root, indices):
    """读取已核验的 384x512 目标 mask 并分成内部／边界／背景。"""
    from PIL import Image

    from tools.e14_denoising_regions import partition

    regions = {}
    for index in indices:
        path = Path(root) / ("%05d_mask.png" % index)
        if not path.is_file():
            raise ValueError("缺少 mask：%s" % path)
        with Image.open(path) as image:
            mask = image.convert("L")
        if mask.size != (384, 512):
            raise ValueError("需要 384x512 的已核验 mask：%s" % path)
        parts = partition(mask, (64, 48))
        if not all(part.any() for part in parts.values()):
            raise ValueError("mask 存在空区域：%s" % path)
        regions[index] = parts
    return regions


def build_models(args, ctx):
    import torch
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel

    from checkpoint_utils import load_texture_warmstart
    from models.bf_texture_module import BFTextureConditioner
    from train_texture_adapter import (AttnProcessor, IPAttnProcessor, TextureAdapter,
                                       load_image_encoder_flexible)

    device, dtype = args.device, torch.float16
    scheduler = DDPMScheduler.from_pretrained(args.base_model, subfolder="scheduler",
                                              local_files_only=True)
    if scheduler.config.prediction_type != "epsilon":
        raise ValueError("当前诊断要求 epsilon 预测")
    text = CLIPTextModel.from_pretrained(args.base_model, subfolder="text_encoder",
                                         local_files_only=True).to(device, dtype).eval()
    vision = load_image_encoder_flexible(args.clip_model, device, dtype).eval()
    vae = AutoencoderKL.from_pretrained(args.base_model, subfolder="vae",
                                        local_files_only=True).to(device, dtype).eval()
    unet = UNet2DConditionModel.from_pretrained(args.base_model, subfolder="unet",
                                                local_files_only=True)
    processors = {}
    for name in unet.attn_processors:
        if name.endswith("attn1.processor"):
            processors[name] = AttnProcessor()
            continue
        if name.startswith("mid_block"):
            hidden = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            hidden = list(reversed(unet.config.block_out_channels))[int(name.split(".")[1])]
        else:
            hidden = unet.config.block_out_channels[int(name.split(".")[1])]
        processors[name] = IPAttnProcessor(hidden_size=hidden,
                                           cross_attention_dim=unet.config.cross_attention_dim,
                                           num_tokens=16)
    unet.set_attn_processor(processors)
    conditioner = BFTextureConditioner(clip_embeddings_dim=vision.config.hidden_size,
                                       cross_attention_dim=unet.config.cross_attention_dim,
                                       num_tokens=16)
    model = TextureAdapter(unet, torch.nn.ModuleList(unet.attn_processors.values()), conditioner)
    state = torch.load(args.checkpoint, map_location="cpu")
    filled = load_texture_warmstart(model, state)
    del state
    model.to(device=device, dtype=dtype).requires_grad_(False)
    unet.eval()
    ctx.update({"device": device, "dtype": dtype, "scheduler": scheduler, "text": text,
                "vision": vision, "vae": vae, "unet": unet, "model": model, "bf": conditioner,
                "filled_palette_keys": filled, "height": 512, "width": 384})
    return ctx


def d4_gate(ctx):
    """仅在 D3 显示内部正收益且边界负收益时才跑 D4。"""
    path = Path(ctx["output"]) / "d3_timestep_region" / "report.json"
    if not path.is_file():
        return False, "缺少 D3 结果"
    report = __import__("json").loads(path.read_text(encoding="utf-8"))
    entry = report["delta"]["random"]
    interior = [value for position, value in enumerate(entry["interior"])
                if report["timesteps"][position] in (1, 181, 481, 781, 981)]
    boundary = [value for position, value in enumerate(entry["boundary"])
                if report["timesteps"][position] in (1, 181, 481, 781, 981)]
    hit = (sum(interior) / len(interior) > 0) and (sum(boundary) / len(boundary) < 0)
    return hit, "interior %.6f boundary %.6f" % (sum(interior) / len(interior),
                                                 sum(boundary) / len(boundary))


def summarize(ctx, results):
    lines = ["# E15-Diagnosis 汇总", ""]
    config = ctx["args"]
    lines.append("- 样本 %d，时间步 %s" % (config["count"], FULL_STEPS))
    lines.append("- 条件 %s" % CONDITIONS)
    lines.append("- checkpoint %s" % config["checkpoint"])
    lines.append("")
    d1 = results.get("d1")
    if d1:
        lines.append("## D1 参考信息衰减（条件对 matched 的归一化响应均值）")
        lines.append("")
        lines.append("| layer | dim | D_rot90 | D_color | D_random | D_zero | effective rank | PC1 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for layer in d1["protocol"]["layers"]:
            item = d1["layers"][layer]
            spectrum = item["correct_spectrum"]
            pc1 = spectrum["pc_variance"][0] if spectrum["pc_variance"] else float("nan")
            lines.append("| %s | %d | %.4f | %.4f | %.4f | %.4f | %.3f | %.3f |" % (
                layer, item["dimension"], item["d_rot90"]["mean"], item["d_color_nearest"]["mean"],
                item["d_random"]["mean"], item["d_zero"]["mean"], spectrum["effective_rank"], pc1))
        lines.append("")
        lines.append("- 来源 attention 占比：" + ", ".join(
            "%s=%.3f" % (name, value) for name, value in
            zip(d1["source_attention"]["names"], d1["source_attention"]["mean_share"])))
        lines.append("- 候选瓶颈（相邻层同时 < %.2f 倍）：%s" % (
            d1["bottleneck_ratio_threshold"], d1["bottlenecks"] or "无"))
        lines.append("")
    d2 = results.get("d2")
    if d2:
        lines.append("## D2 texture residual 热图（R = RMS 差分比）")
        lines.append("")
        for name in ("r_rot90", "r_color_nearest", "r_random"):
            matrix = d2[name]
            lines.append("- %s 每层均值：%s" % (name, " ".join(
                "%.3f" % (sum(row[i] for row in matrix) / len(matrix)) for i in range(len(matrix[0])))))
        lines.append("- epsilon 相对响应：" + ", ".join(
            "%s=%.5f" % (name, sum(values) / len(values)) for name, values in
            d2["epsilon_relative_response"].items()))
        lines.append("")
    d3 = results.get("d3")
    if d3:
        lines.append("## D3 timestep x 区域（ΔL = L_condition - L_matched，正=正确参考更好）")
        lines.append("")
        lines.append("时间步 %s" % d3["timesteps"])
        for name in d3["delta"]:
            for region in REGIONS:
                lines.append("- %s / %s：%s" % (name, region, " ".join(
                    "%+.5f" % value for value in d3["delta"][name][region])))
        lines.append("- 内部为正的样本数（random）：%s" % d3["interior_positive_samples"]["random"])
        lines.append("")
    d4 = results.get("d4")
    if d4:
        lines.append("## D4 分组层消融（base steps 均值）")
        lines.append("")
        lines.append("| config | " + " | ".join(REGIONS) + " |")
        lines.append("|---" * (len(REGIONS) + 1) + "|")
        for config, entry in d4["delta"].items():
            lines.append("| %s | %s |" % (config, " | ".join(
                "%+.6f" % entry[region]["base_steps_random_mean"] for region in REGIONS)))
    text = "\n".join(lines) + "\n"
    (Path(ctx["output"]) / "SUMMARY.md").write_text(text, encoding="utf-8")
    print(text, flush=True)
    return text


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    stages = [item.strip() for item in args.stages.split(",") if item.strip()]
    unknown = [item for item in stages if item not in {"d1", "d2", "d3", "d4"}]
    if unknown:
        raise ValueError("未知阶段：%s" % unknown)
    ctx = {"args": vars(args), "output": output}
    results = {}
    if "d3" in stages:
        if not args.previous_report:
            raise ValueError("D3 需要 --previous-report")
        from tools.e15_stages import d3_timestep_region

        results["d3"] = d3_timestep_region(ctx)
    if any(item in stages for item in ("d1", "d2", "d4")):
        if not args.mask_root:
            raise ValueError("D1/D2/D4 需要 --mask-root")
        dataset, indices, hashes, pairs, colors = build_inputs(args)
        ctx.update({"dataset": dataset, "indices": indices, "hashes": hashes,
                    "pairs": pairs, "colors": colors})
        ctx["regions"] = load_regions(args.mask_root, indices)
        ctx["args"]["indices"] = indices
        build_models(args, ctx)
        from tools.e15_stages import (build_token_bank, d1_reference_trace,
                                      d2_residual_trace, d4_layer_ablation)

        build_token_bank(ctx)
        if "d1" in stages:
            # D1 仍需要 CLIP vision 编码参考图，因此在其后才释放编码器。
            results["d1"] = d1_reference_trace(ctx)
        # D2/D4 仍需 text encoder 构造条件；只有 CLIP 图像编码器可以提前释放。
        del ctx["vision"]
        import torch

        torch.cuda.empty_cache()
        if "d2" in stages:
            results["d2"] = d2_residual_trace(ctx)
        if "d4" in stages:
            if args.force_d4:
                results["d4"] = d4_layer_ablation(ctx)
            else:
                hit, detail = d4_gate(ctx)
                print("d4 gate:", hit, detail, flush=True)
                if hit:
                    results["d4"] = d4_layer_ablation(ctx)
                else:
                    results["d4_skipped"] = detail
    if any(item in stages for item in ("d1", "d2", "d3", "d4")):
        summarize(ctx, results)
    write_json(output / "run_config.json", {"args": vars(args), "stages": stages,
                                            "results": sorted(results)})
    print("E15 finished", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["manifest", "data-root", "checkpoint", "base-model", "clip-model", "output"]:
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--mask-root")
    parser.add_argument("--previous-report")
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stages", default="d1,d2,d3,d4")
    parser.add_argument("--force-d4", action="store_true")
    run(parser.parse_args())
