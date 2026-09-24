"""E15-D5 生成验证：固定 32 样本、单 seed，比较 texture 层开关与参考条件。

两个套件共用一次模型加载、同 latent、同 prompt、同 sketch：

- layers：只改 texture attention 层开关（baseline/no_g4/no_g3/only_g3/no_texture），
  验证 D4 的分组结论在真实生成上是否成立。
- reference：只改喂给模型的参考图（matched/rot90/color_near/wrong_ref），
  外加一个换 seed 的基线用于标定随机波动尺度，回答"生成结果是否真的取决于参考纹样"。

两套件都输出生成图、全套标准指标、区域指标与相对参考配置的逐像素差异。
"""

import argparse
import importlib.util
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tools.e15_common import sample_indices, write_json

# 层索引与 D4 的 G1-G4 分组一致（0-based，共 16 层 texture cross-attention）。
# 空串表示全开；其余为「要关闭的层」。
LAYER_CONFIGS = {
    "baseline": "",
    "no_g4": "12,13,14,15",
    "no_g3": "8,9,10,11",
    "only_g3": "0,1,2,3,4,5,6,7,12,13,14,15",
    "no_texture": ",".join(str(i) for i in range(16)),
}
# 参考条件套件：不改层，只换喂进去的参考图；matched_seed1 用于标定噪声尺度。
REFERENCE_CONFIGS = [("matched", {}), ("rot90", {"texture": "rot90"}),
                     ("color_near", {"texture": "color_near"}),
                     ("wrong_ref", {"texture": "wrong_ref"}),
                     ("matched_seed1", {"seed_offset": 1})]
DEFAULT_OPTIONS = {"spec": "", "texture": "matched", "seed_offset": 0}
REGIONS = ["interior", "boundary", "background"]
# 汇总表列：与文档要求的 CLIP-texture / TPF / Edge / IoU / Leak 对齐。
KEY_METRICS = ["clip_texture", "tpf_patch_sim", "tpf_gram_l1", "tcf_lab_delta", "ssim",
               "struct_edge_f1", "struct_iou", "leak_colored_frac", "leak_mean_saturation",
               "leak_edge_density", "region_l1_interior", "region_l1_boundary",
               "region_l1_background", "interior_tpf", "interior_gram_l1",
               "pixel_diff_vs_ref"]


def load_inference_module():
    """文件名带连字符，只能按路径加载。"""
    path = Path(__file__).resolve().parents[1] / "inference_IMAGGarment-1.py"
    spec = importlib.util.spec_from_file_location("e15_imag_infer", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["e15_imag_infer"] = module
    spec.loader.exec_module(module)
    return module


def build_inference_args(args):
    """prepare() 与 generate_one() 用到的全部参数；默认值取自 E5 已跑通口径。"""
    return argparse.Namespace(
        GAM_model_ckpt=args.checkpoint,
        texture_ckpt=args.texture_ckpt or args.checkpoint,
        # auto 表示沿用 checkpoint metadata 的 SD 根目录与 VAE 子目录。
        base_model_path=args.base_model_path,
        vae_model_path=args.vae_model_path,
        image_encoder_path=args.clip_model,
        device=args.device,
        width=None, height=None,  # None = 沿用 checkpoint metadata 的分辨率
        seed=args.seed,
        prompt="a cloth",
        sketch_path="", texture_path="", output_path="",
        num_inference_steps=args.steps,
        guidance_scale=7.0, sketch_scale=0.6, ipa_scale=1.0,
        texture_mode="patch_resampled", texture_condition_mode="token",
        texture_num_tokens=16, texture_scale=1.0,
        texture_preprocess_mode="plain_resize", fusion_type="minimal",
        use_tcpm_lite=1, use_texture_gate=1, layer_group_enabled=1, use_aa_tcr_fuse=0,
        tcpm_hidden_ratio=0.25, tcpm_residual_scale_init=0.0,
        use_palette_tokens=0, num_palette_tokens=4, palette_branch_scale_init=0.0,
        gate_type="layer", gate_init="identity", gate_min=0.7, gate_max=1.3,
        use_balanced_fusion_gate=0, balanced_gate_hidden_dim=64, balanced_gate_scale=0.2,
        balanced_gate_min=0.8, balanced_gate_max=1.2,
        balanced_gate_trace_path="", balanced_gate_trace_sample_id="",
        use_conflict_aware_gate=0, conflict_texture_suppress_strength=0.1,
        conflict_palette_suppress_strength=0.4, conflict_deltae_norm=50.0,
        conflict_threshold=0.70,
        lora_rank=4, alpha1=2.0, alpha2=2.0, alpha3=1.5, alpha4=1.0,
        nexus_prompt=None, use_text_guided_resampler=-1,
        disable_nexus_adapter=False, disable_texture_film=False,
        use_local_detail_adapter=-1, local_detail_scale=1.0,
        local_detail_step_start=0, local_detail_step_end=10 ** 9,
        local_detail_token_permutation="none", local_detail_permutation_seed=42,
        local_detail_donor_texture_path="", local_detail_input_transform="none",
        local_detail_probe_dir="", local_detail_output_block=0,
        local_detail_trace_path="", local_detail_trace_sample_id="",
        condition_intervention="none", condition_intervention_budget="",
        condition_intervention_source="", condition_intervention_dir="",
        sptg_mode="none", sptg_dir="",
        full_condition_probe_dir="", condition_response_probe_dir="",
        condition_response_probe_steps=[0, 5, 15, 25, 49],
        condition_response_probe_fractions=[0.1, 0.2],
        condition_response_probe_region_kernel=9,
        debug_spatial=False, force_texture_num_tokens_override=False,
        disable_texture_layers="",
    )


def build_samples(args):
    """固定 32 样本：索引与 113719/113817/D1-D4 完全一致。"""
    from transformers import CLIPTokenizer

    from train_texture_adapter import MyDataset

    tokenizer = CLIPTokenizer.from_pretrained(args.base_model, subfolder="tokenizer",
                                             local_files_only=True)
    dataset = MyDataset(args.manifest, tokenizer, height=512, width=384,
                        image_root_path=args.data_root,
                        texture_preprocess_mode="plain_resize",
                        t_drop_rate=0, i_drop_rate=0, ti_drop_rate=0)
    root = Path(args.data_root)
    samples = []
    for index in sample_indices(dataset, args.count):
        row = dataset.data[index]
        samples.append({
            "index": index,
            "caption": row["caption"],
            "sketch": str(root / row["sketch"]),
            "texture": str(root / row.get("texture", row.get("color"))),
            "cloth": str(root / row["cloth"]),
        })
    return dataset, samples


def build_reference_variants(args, dataset, samples):
    """按 D1-D3 的同一配对协议准备 rot90 / 同色异纹 / 完全错误参考三种纹理。"""
    from PIL import Image as PILImage

    from tools.e14_denoising_complete import color_pairs
    from tools.e14_matched_denoising import choose_pairs
    from tools.e14_pattern_probe import pixel_hash

    root = Path(args.data_root)
    rows = dataset.data
    indices = [sample["index"] for sample in samples]
    hashes = {}
    for index in indices:
        with PILImage.open(root / rows[index].get("texture", rows[index].get("color"))) as image:
            hashes[index] = pixel_hash(image.convert("RGB"))
    wrong = choose_pairs(rows, indices, hashes)
    nearest = color_pairs(rows, indices, hashes, root)
    folder = Path(args.output_reference or args.output) / "_textures"
    folder.mkdir(parents=True, exist_ok=True)
    variants, detail = {}, {}
    for sample in samples:
        index = sample["index"]
        rotated = folder / ("%05d_rot90.png" % index)
        with PILImage.open(sample["texture"]) as image:
            image.convert("RGB").rotate(90, expand=True).save(rotated)
        color_index = nearest[index]["index"]
        wrong_index = wrong[index][0]
        variants[index] = {
            "matched": sample["texture"],
            "rot90": str(rotated),
            "color_near": str(root / rows[color_index].get("texture",
                                                          rows[color_index].get("color"))),
            "wrong_ref": str(root / rows[wrong_index].get("texture",
                                                         rows[wrong_index].get("color"))),
        }
        detail[index] = {"color_near_index": color_index, "wrong_ref_index": wrong_index,
                         "hellinger_distance": nearest[index]["hellinger_distance"]}
    return variants, detail


def load_regions(mask_root, samples, size):
    """把已核验的目标 mask 分成内部/边界/背景；尺寸不符时最近邻对齐。"""
    from tools.e14_denoising_regions import partition

    regions = {}
    for sample in samples:
        path = Path(mask_root) / ("%05d_mask.png" % sample["index"])
        mask = Image.open(path).convert("L")
        if mask.size != size:
            mask = mask.resize(size, Image.NEAREST)
        regions[sample["index"]] = partition(mask, (size[1], size[0]))
    return regions


def region_metrics(gen_path, sample, regions):
    """区域指标：相对目标 cloth 的 L1，以及内部区域的纹理匹配度。"""
    from eval.metrics import _compute_gram_l1, _pil_to_np, patch_texture_similarity

    gen = Image.open(gen_path).convert("RGB")
    cloth = Image.open(sample["cloth"]).convert("RGB").resize(gen.size, Image.BICUBIC)
    texture = Image.open(sample["texture"]).convert("RGB").resize(gen.size, Image.BICUBIC)
    diff = np.abs(_pil_to_np(gen).astype(np.float32) -
                  _pil_to_np(cloth).astype(np.float32)).mean(axis=2) / 255.0
    result = {}
    for name in REGIONS:
        result["region_l1_" + name] = float(diff[regions[name]].mean())
    interior = Image.fromarray(regions["interior"].astype(np.uint8) * 255, mode="L")
    result["interior_tpf"] = patch_texture_similarity(gen, texture, mask=interior, patch=8)
    try:
        result["interior_gram_l1"] = _compute_gram_l1(gen, texture, mask=interior)
    except Exception as exc:  # VGG 权重缺失时不影响其余指标
        result["interior_gram_l1"] = float("nan")
        result["interior_gram_error"] = str(exc)
    return result


def config_layers(pipe, module):
    """按 UNet 顺序返回 16 个 texture processor 的名字，用于核对层索引。"""
    return [name for name, proc in pipe.unet.attn_processors.items()
            if isinstance(proc, module.IPAttnProcessor2_0)]


def normalize_options(options):
    merged = dict(DEFAULT_OPTIONS)
    merged.update(options)
    return merged


def generate_config(args, module, pipe, namespace, samples, name, options,
                    variants, output):
    """一个配置跑满全部样本；逐样本重置 generator 保证跨配置 latent 相同。"""
    import torch

    options = normalize_options(options)
    folder = Path(output) / name
    folder.mkdir(parents=True, exist_ok=True)
    info = module.apply_texture_layer_disable(pipe, options["spec"])
    records = []
    for position, sample in enumerate(samples):
        texture = (variants or {}).get(sample["index"], {}).get(options["texture"],
                                                              sample["texture"])
        seed = args.seed + options["seed_offset"]
        namespace.sketch_path = sample["sketch"]
        namespace.texture_path = texture
        namespace.prompt = sample["caption"]
        gen_path = folder / ("%05d_gen.png" % sample["index"])
        started = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        generator = torch.Generator(device=args.device).manual_seed(seed)
        with torch.inference_mode():
            image, _grid, _mask = module.generate_one(
                pipe, generator, namespace, str(folder),
                sketch_path=sample["sketch"], texture_path=texture,
                prompt=sample["caption"], out_name="%05d_grid.png" % sample["index"])
        image.save(gen_path)
        records.append({"config": name, "index": sample["index"], "gen": str(gen_path),
                        "caption": sample["caption"], "seed": seed,
                        "texture_used": texture,
                        "seconds": round(time.time() - started, 2)})
        print("[d5] %s %d/%d index=%d %.1fs" % (name, position + 1, len(samples),
                                                sample["index"], records[-1]["seconds"]),
              flush=True)
    return records, info


def evaluate_config(args, samples, records, regions):
    """逐样本算标准指标与区域指标，最后批量补 CLIP-texture。"""
    from eval.metrics import compute_clip_i_values, evaluate_full

    for record, sample in zip(records, samples):
        metrics = evaluate_full(
            record["gen"], target_path=sample["cloth"], texture_path=sample["texture"],
            sketch_path=sample["sketch"],
            mask_path=str(Path(args.mask_root) / ("%05d_mask.png" % record["index"])))
        metrics.update(region_metrics(record["gen"], sample, regions[record["index"]]))
        metrics.pop("metric_warnings", None)
        record["metrics"] = metrics
    try:
        values = compute_clip_i_values([r["gen"] for r in records],
                                       [s["texture"] for s in samples],
                                       device=args.device, model_name=args.clip_model)
        for record, value in zip(records, values):
            record["metrics"]["clip_texture"] = float(value)
    except Exception as exc:
        for record in records:
            record["metrics"]["clip_texture"] = float("nan")
        print("[d5] CLIP-texture 跳过：%s" % exc, flush=True)
    return records


def add_pixel_diff(records, reference):
    """把每个配置与参考配置的逐像素差异补进指标，作为"改动有多大"的直接尺度。"""
    base = {record["index"]: record["gen"] for record in records
            if record["config"] == reference}
    for record in records:
        if record["config"] == reference or record["index"] not in base:
            record["metrics"]["pixel_diff_vs_ref"] = 0.0
            continue
        with Image.open(record["gen"]) as image:
            mine = np.asarray(image.convert("RGB"), dtype=np.float32)
        with Image.open(base[record["index"]]) as image:
            theirs = np.asarray(image.convert("RGB"), dtype=np.float32)
        record["metrics"]["pixel_diff_vs_ref"] = float(np.abs(mine - theirs).mean())
    return records


def aggregate(records):
    keys = sorted({key for record in records for key, value in record["metrics"].items()
                   if isinstance(value, (int, float))})
    result = {}
    for key in keys:
        values = np.array([record["metrics"].get(key, np.nan) for record in records],
                          dtype=np.float64)
        finite = values[np.isfinite(values)]
        result[key] = {"mean": float(finite.mean()) if finite.size else None,
                       "std": float(finite.std()) if finite.size else None,
                       "count": int(finite.size)}
    return result


def contact_sheet(names, records, samples, output):
    """每样本一行、每配置一列的缩略对照图，供人工核验纹样方向。"""
    cell = (192, 256)
    sheet = Image.new("RGB", (cell[0] * len(names), cell[1] * len(samples) + 12), "white")
    draw = ImageDraw.Draw(sheet)
    for column, name in enumerate(names):
        draw.text((column * cell[0] + 4, 1), name, fill="black")
    lookup = {(record["config"], record["index"]): record["gen"] for record in records}
    for row, sample in enumerate(samples):
        for column, name in enumerate(names):
            path = lookup.get((name, sample["index"]))
            if path is None or not Path(path).is_file():
                continue
            with Image.open(path) as image:
                sheet.paste(image.convert("RGB").resize(cell, Image.BILINEAR),
                            (column * cell[0], row * cell[1] + 12))
    path = Path(output) / "contact_sheet.png"
    sheet.save(path)
    return str(path)


def write_summary(output, suite, results, names):
    """配置对比表；delta 表里正号只表示数值更大，方向由指标语义决定。"""
    metrics = [key for key in KEY_METRICS if key in results["aggregate"][names[0]]]
    lines = ["# E15-D5 生成验证（%s）" % suite, "",
             "协议：%d 样本（与 D1-D4 同索引）、seed=%d、同 latent/prompt/sketch、%d 步。" %
             (results["protocol"]["samples"], results["protocol"]["seed"],
              results["protocol"]["steps"]),
             "参考配置：%s；pixel_diff_vs_ref 是同 latent 下与参考配置的逐像素平均差。" %
             results["protocol"]["reference"], "",
             "| config | " + " | ".join(metrics) + " |",
             "|---" * (len(metrics) + 1) + "|"]
    for name in names:
        entry = results["aggregate"][name]
        lines.append("| %s | %s |" % (name, " | ".join(
            "-" if entry[key]["mean"] is None else "%.4f" % entry[key]["mean"]
            for key in metrics)))
    lines += ["", "## 相对参考配置的变化", "",
              "| config | " + " | ".join(metrics) + " |",
              "|---" * (len(metrics) + 1) + "|"]
    for name in names:
        if name == results["protocol"]["reference"]:
            continue
        entry, base = results["aggregate"][name], results["aggregate"][results["protocol"]["reference"]]
        lines.append("| %s | %s |" % (name, " | ".join(
            "-" if entry[key]["mean"] is None or base[key]["mean"] is None
            else "%+.4f" % (entry[key]["mean"] - base[key]["mean"]) for key in metrics)))
    text = "\n".join(lines) + "\n"
    (Path(output) / "SUMMARY.md").write_text(text, encoding="utf-8")
    print(text, flush=True)
    return text


def run_suite(args, module, pipe, namespace, dataset, samples, regions, suite, output):
    """跑一个套件：逐配置生成 + 评测 + 汇总。"""
    if suite == "layers":
        configs = [(name, {"spec": spec}) for name, spec in LAYER_CONFIGS.items()]
        variants, detail = None, {}
        reference = "baseline"
    elif suite == "reference":
        configs = REFERENCE_CONFIGS
        variants, detail = build_reference_variants(args, dataset, samples)
        reference = "matched"
    else:
        raise ValueError("未知套件：%s" % suite)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    info = {}
    for name, options in configs:
        config_records, layer_info = generate_config(args, module, pipe, namespace,
                                                    samples, name, options, variants,
                                                    output)
        info[name] = dict(normalize_options(options), **layer_info)
        evaluate_config(args, samples, config_records, regions)
        records.extend(config_records)
    add_pixel_diff(records, reference)
    names = [name for name, _ in configs]
    for name in names:
        write_json(output / name / "metrics.json",
                   [record for record in records if record["config"] == name])
    results = {
        "protocol": {"suite": suite, "samples": len(samples), "seed": args.seed,
                     "steps": args.steps, "reference": reference,
                     "indices": [sample["index"] for sample in samples],
                     "texture_layers": config_layers(pipe, module), "configs": info,
                     "pairing_detail": detail,
                     "note": "只改 texture 层开关或参考图；latent/prompt/sketch 固定。"},
        "per_sample": records,
        "aggregate": {name: aggregate([record for record in records
                                       if record["config"] == name]) for name in names},
        "contact_sheet": contact_sheet(names, records, samples, output),
    }
    write_json(output / "report.json", results)
    write_summary(output, suite, results, names)
    return results


def run(args):
    import torch

    module = load_inference_module()
    dataset, samples = build_samples(args)
    namespace = build_inference_args(args)
    pipe, _generator = module.prepare(namespace)
    for value in vars(pipe).values():
        if isinstance(value, torch.nn.Module):
            value.eval()
    torch.backends.cudnn.benchmark = False
    print("[d5] resolution %dx%d, texture layers=%d" %
          (namespace.width, namespace.height, len(config_layers(pipe, module))), flush=True)
    regions = load_regions(args.mask_root, samples, (namespace.width, namespace.height))
    for suite in [item.strip() for item in args.suite.split(",") if item.strip()]:
        output = args.output if suite == "layers" else args.output_reference
        if suite != "layers" and not output:
            raise ValueError("套件 %s 需要 --output-reference" % suite)
        print("[d5] suite=%s output=%s" % (suite, output), flush=True)
        run_suite(args, module, pipe, namespace, dataset, samples, regions, suite, output)
    print("E15-D5 finished", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["manifest", "data-root", "checkpoint", "texture-ckpt", "base-model",
                 "clip-model", "mask-root", "output"]:
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--output-reference")
    parser.add_argument("--suite", default="layers")
    parser.add_argument("--base-model-path", default="auto")
    parser.add_argument("--vae-model-path", default="auto")
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())
