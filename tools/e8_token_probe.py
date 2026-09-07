"""同输入比较 BF/TCPM 条件分支；不加载 UNet，不生成图片，不代表完整扩散等价性。"""

import copy
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from checkpoint_utils import extract_texture_metadata
from models.bf_texture_module import BFTextureConditioner
from models.tcpm_lite import TCPMLite
from models.text_guided_queries import guidance_config_from_checkpoint, text_content_mask


NEGATIVE_PROMPT = " worst quality, low quality"  # 与实际 inference 入口一致。


def build_conditioner(checkpoint, device="cpu", dtype=torch.float32):
    state, meta = checkpoint["bf_texture_conditioner"], checkpoint.get("meta", {})
    query = state["resampler_queries"]
    if int(meta.get("texture_num_tokens", query.shape[1])) != query.shape[1]:
        raise ValueError("BF 查询数量与 checkpoint 元数据不一致")
    bf = BFTextureConditioner(
        clip_embeddings_dim=state["token_source_proj.0.0.weight"].numel(),
        cross_attention_dim=query.shape[-1], num_tokens=query.shape[1],
        stage_channels=tuple(state[key].shape[0] for key in (
            "stage1.0.weight", "stage2.0.weight", "stage3.0.weight", "stage4.0.weight")),
        texture_mode=meta.get("texture_mode", "patch_resampled"),
        **guidance_config_from_checkpoint(state, meta),
    )
    bf.load_state_dict(state, strict=True)
    tcpm = TCPMLite(query.shape[-1], hidden_ratio=float(meta.get("tcpm_hidden_ratio", 0.25)))
    tcpm.load_state_dict(checkpoint["tcpm_lite"], strict=True)
    # 正式推理中的 BF/TCPM 保留构造后的 train 状态；禁止擅自切换 MHA 计算路径。
    return tuple(module.to(device=device, dtype=dtype).requires_grad_(False) for module in (bf, tcpm))


def make_zero_init(baseline, text_config):
    bf, tcpm = copy.deepcopy(baseline)
    bf.configure_text_guidance(**text_config)
    bf.requires_grad_(False)
    return bf, tcpm


@torch.inference_mode()
def capture_features(model, inputs, enabled):
    bf, tcpm = model
    values = {}

    def before_resampler(module, args):
        values["queries"] = args[0].detach().float().cpu()
        values["patch_tokens"] = args[1].detach().float().cpu()

    handle = bf.resampler.register_forward_pre_hook(before_resampler)
    try:
        tokens, _ = bf(**inputs, apply_text_guidance=enabled)
    finally:
        handle.remove()
    values["bf_tokens"] = tokens.float().cpu()
    values["tcpm_tokens"] = tcpm(tokens, inputs["text_embeds"]).float().cpu()
    stats = dict(bf.text_guidance.last_stats) if enabled and bf.text_guidance is not None else {}
    return values, stats


def token_difference(reference, candidate):
    if reference.shape != candidate.shape:
        raise ValueError("条件特征形状不一致: %s / %s" % (reference.shape, candidate.shape))
    a, b = reference.float(), candidate.float()
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError("条件特征包含 NaN/Inf")
    delta_rms = float((b - a).square().mean().sqrt())
    reference_rms = float(a.square().mean().sqrt())
    return {
        "exact_equal": bool(torch.equal(a, b)),
        "reference_rms": reference_rms, "delta_rms": delta_rms,
        "relative_rms": delta_rms / reference_rms if reference_rms else (0.0 if delta_rms == 0 else None),
        "max_abs_delta": float((b - a).abs().max()),
        "mean_token_cosine": float(F.cosine_similarity(a, b, dim=-1).mean()),
    }


@torch.inference_mode()
def probe_sample(models, positive_inputs, negative_inputs, sample_id):
    cases = {"e5": (models["e5"], False), "e8a": (models["e8a"], False),
             "e8b_on": (models["e8b"], True), "e8b_off": (models["e8b"], False),
             "e5_zero_init": (models["e5_zero_init"], True)}
    captures, stats = {}, []
    for branch, inputs in (("conditional", positive_inputs), ("unconditional", negative_inputs)):
        captures[branch] = {}
        for name, (model, enabled) in cases.items():
            features, guidance = capture_features(model, inputs, enabled)
            captures[branch][name] = features
            if name == "e8b_on":
                stats.append({"sample_id": sample_id, "branch": branch, **guidance})
    captures["cond_minus_uncond"] = {
        name: {stage: captures["conditional"][name][stage] - captures["unconditional"][name][stage]
               for stage in ("bf_tokens", "tcpm_tokens")} for name in cases
    }
    rows = []
    pairs = [("e8a", "e5"), ("e8b_on", "e5"), ("e8b_off", "e5"),
             ("e8b_on", "e8a"), ("e8b_on", "e8b_off"), ("e5_zero_init", "e5")]
    for branch, data in captures.items():
        for candidate, reference in pairs:
            for stage in data[reference]:
                rows.append({
                    "sample_id": sample_id, "comparison": candidate + "_vs_" + reference,
                    "branch": branch, "stage": stage,
                    **token_difference(data[reference][stage], data[candidate][stage]),
                })
    return rows, stats


def summarize_tokens(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["comparison"], row["branch"], row["stage"])].append(row)
    results = []
    for (comparison, branch, stage), items in sorted(groups.items()):
        result = dict(comparison=comparison, branch=branch, stage=stage, count=len(items),
                      exact_equal_count=sum(row["exact_equal"] for row in items))
        for key in ("relative_rms", "delta_rms", "max_abs_delta", "mean_token_cosine"):
            values = [row[key] for row in items if row[key] is not None and math.isfinite(row[key])]
            result[key + "_mean"] = sum(values) / len(values) if values else None
            result[key + "_max"] = max(values) if values else None
        results.append(result)
    return results


def read_samples(path, count, data_root):
    samples = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(samples, list) or count < 1 or len(samples) < count:
        raise ValueError("样本文件应为已有 benchmark_samples.json 列表，且数量不少于指定值")
    selected = []
    for row in samples[:count]:
        texture = Path(row["texture_path"])
        if not texture.is_file():
            texture = Path(data_root) / "texture" / Path(row["texture_path"]).name
        if not texture.is_file():
            raise FileNotFoundError(str(texture))
        selected.append({"sample_id": str(row["sample_id"]), "prompt": row["prompt"],
                         "texture_path": str(texture)})
    if len({row["sample_id"] for row in selected}) != count:
        raise ValueError("诊断样本编号重复")
    return selected


def run_token_probe(checkpoints, texture_checkpoint, args):
    # 延迟导入，单独检查权重时不要求 diffusers/transformers。
    from diffusers.image_processor import VaeImageProcessor
    from PIL import Image
    from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

    samples = read_samples(args.samples_json, args.num_samples, args.data_root)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    device = torch.device(args.device)
    config = guidance_config_from_checkpoint(checkpoints["e8b"]["bf_texture_conditioner"],
                                            checkpoints["e8b"].get("meta", {}))
    if not config["text_guidance_dim"]:
        raise ValueError("E8b checkpoint 中没有文本查询模块")
    for name, checkpoint in checkpoints.items():
        meta = checkpoint.get("meta", {})
        if meta.get("texture_mode", "patch_resampled") != "patch_resampled":
            raise ValueError("%s 不是本次实验的 patch_resampled 模式" % name)
        if not meta.get("use_tcpm_lite", 1) or meta.get("use_aa_tcr_fuse", 0):
            raise ValueError("%s 的 TCPM/AA-TCR 配置与本次探针不兼容" % name)
    models = {name: build_conditioner(checkpoint, device, dtype) for name, checkpoint in checkpoints.items()}
    torch.manual_seed(42)
    models["e5_zero_init"] = make_zero_init(models["e5"], config)
    base_path = args.base_model_path
    if base_path == "auto":
        base_path = checkpoints["e5"].get("meta", {}).get("pretrained_model_name_or_path")
    if not base_path:
        raise ValueError("E5 元数据缺少基础模型路径，请指定 --base-model-path")
    image_path = args.image_encoder_path
    if image_path == "auto":
        image_path = extract_texture_metadata(texture_checkpoint).get(
            "image_encoder_path", "openai/clip-vit-large-patch14")
    tokenizer = CLIPTokenizer.from_pretrained(base_path, subfolder="tokenizer", local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(base_path, subfolder="text_encoder",
                                                local_files_only=True).to(device=device, dtype=dtype).eval()
    try:
        vision = CLIPVisionModelWithProjection.from_pretrained(image_path, local_files_only=True)
    except Exception:
        vision = CLIPVisionModelWithProjection.from_pretrained(
            image_path, subfolder="models/image_encoder", local_files_only=True)
    vision = vision.to(device=device, dtype=dtype).eval()
    processor = CLIPImageProcessor()
    image_processor = VaeImageProcessor(vae_scale_factor=8, do_convert_rgb=True, do_normalize=False)
    text_encoder.requires_grad_(False)
    vision.requires_grad_(False)

    def encode_text(prompt):
        ids = tokenizer([prompt], padding="max_length", max_length=tokenizer.model_max_length,
                        truncation=True, return_tensors="pt")
        attention = ids.attention_mask.to(device) if getattr(text_encoder.config, "use_attention_mask", False) else None
        return (text_encoder(ids.input_ids.to(device), attention_mask=attention)[0],
                text_content_mask(ids.input_ids.to(device), tokenizer.eos_token_id))

    rows, gate_rows = [], []
    with torch.inference_mode():
        negative_text, negative_mask = encode_text(NEGATIVE_PROMPT)
        for index, sample in enumerate(samples):
            with Image.open(sample["texture_path"]) as original:
                image = original.convert("RGB")
            clip = vision(processor(images=[image], return_tensors="pt").pixel_values.to(device, dtype=dtype),
                          output_hidden_states=True)
            texture = image_processor.preprocess([image], height=args.height, width=args.width).to(device, dtype=dtype)
            text, mask = encode_text(sample["prompt"])
            inputs = dict(clip_image_embeds=clip.image_embeds, clip_vision_tokens=clip.hidden_states[-1][:, 1:, :],
                          texture_images=texture * 2 - 1, text_embeds=text, text_mask=mask)
            negative = {key: torch.zeros_like(value) for key, value in inputs.items()
                        if key not in ("text_embeds", "text_mask")}
            negative.update(text_embeds=negative_text, text_mask=negative_mask)
            result, stats = probe_sample(models, inputs, negative, sample["sample_id"])
            rows.extend(result)
            gate_rows.extend(stats)
            if (index + 1) % 10 == 0 or index == 0:
                print("[token] %d/%d" % (index + 1, len(samples)), flush=True)
    return rows, gate_rows, {
        "sample_count": len(samples), "samples": samples, "dtype": args.dtype,
        "base_model_path": str(base_path), "image_encoder_path": str(image_path),
        "width": args.width, "height": args.height, "negative_prompt": NEGATIVE_PROMPT,
        "scope": "仅 BF/TCPM 条件分支；零门控检查不代表完整 UNet/扩散生成等价性。",
        "input_policy": "各组共享同一 CLIP 编码器及文本编码器；若元数据路径不同，需另查实际推理加载。",
    }
