import argparse
import csv
import importlib.util
import itertools
import json
import math
from collections import deque
import os
import random
import re
import subprocess
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import VGG19_Weights, vgg19
from torchvision.utils import make_grid, save_image

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from adapter.attention_processor import (
    IPAttnProcessor2_0,
    LogoCacheCAttnProcessor2_0,
    LogoCacheSAttnProcessor2_0,
    LogoRefSAttnProcessor2_0,
)
from models.bf_texture_module import BFTextureConditioner
from models.multiscale_texture_encoder import MultiScaleTextureEncoder
from models.palette_tokenizer import PaletteTokenMLP
from models.spatial_injection import SpatialInjectionAdapter
from texture_preprocess import preprocess_texture_image
from color_conflict_utils import compute_color_conflict

try:
    _repo_checkpoint_spec = importlib.util.find_spec("repo_utils.checkpoint_utils")
except ModuleNotFoundError:
    _repo_checkpoint_spec = None
if _repo_checkpoint_spec is not None:
    from repo_utils.checkpoint_utils import extract_texture_metadata
else:
    from checkpoint_utils import extract_texture_metadata



# =========================
# Phase 1: Ti-MGD layer group routing
# =========================
def _get_layer_group(name: str) -> str:
    """
    Map UNet attention processor name to frequency group.
    "semantic" = text-only, "detail" = texture-dominant.
    """
    if "down_blocks.2" in name or "down_blocks.3" in name:
        return "semantic"
    if "mid_block" in name:
        return "semantic"
    if "up_blocks.0" in name or "up_blocks.1" in name:
        return "semantic"
    return "detail"


def _get_detail_text_scale(name: str) -> float:
    if "up_blocks" in name:
        return 0.15
    return 0.05

# =========================
# Dataset
# =========================
def sketch_to_garment_mask(
    sketch: Image.Image,
    width: int,
    height: int,
    line_threshold: int = 245,
    dilate_size: int = 9,
) -> Image.Image:
    """
    Estimate a garment-region mask from a black-line sketch on white background.
    The sketch lines are treated as barriers, then the outside background is
    flood-filled from image borders. The remaining region is the garment mask.
    """
    dilate_size = max(3, int(dilate_size) | 1)
    gray = sketch.convert("L").resize((width, height), Image.BILINEAR)
    line = np.asarray(gray) < line_threshold
    line_img = Image.fromarray((line.astype(np.uint8) * 255), mode="L")
    barrier = np.asarray(line_img.filter(ImageFilter.MaxFilter(dilate_size))) > 0

    h, w = barrier.shape
    passable = ~barrier
    outside = np.zeros((h, w), dtype=bool)
    q = deque()

    def push(y, x):
        if passable[y, x] and not outside[y, x]:
            outside[y, x] = True
            q.append((y, x))

    for x in range(w):
        push(0, x)
        push(h - 1, x)
    for y in range(h):
        push(y, 0)
        push(y, w - 1)

    while q:
        y, x = q.popleft()
        if y > 0:
            push(y - 1, x)
        if y + 1 < h:
            push(y + 1, x)
        if x > 0:
            push(y, x - 1)
        if x + 1 < w:
            push(y, x + 1)

    mask = ~outside
    area = float(mask.mean())
    if area < 0.02 or area > 0.95:
        ys, xs = np.where(line)
        if len(xs) == 0:
            mask = np.ones((h, w), dtype=bool)
        else:
            pad_x = max(8, int(0.06 * w))
            pad_y = max(8, int(0.06 * h))
            x0 = max(0, int(xs.min()) - pad_x)
            x1 = min(w, int(xs.max()) + pad_x)
            y0 = max(0, int(ys.min()) - pad_y)
            y1 = min(h, int(ys.max()) + pad_y)
            mask = np.zeros((h, w), dtype=bool)
            mask[y0:y1, x0:x1] = True

    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    return mask_img.filter(ImageFilter.MaxFilter(5))


class JointTextureDataset(Dataset):
    def __init__(
        self,
        json_path,
        tokenizer,
        image_root,
        width=512,
        height=640,
        texture_preprocess_mode="crop_tile",
        conflict_deltae_norm=50.0,
    ):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.image_root = image_root
        self.width = width
        self.height = height
        self.texture_preprocess_mode = texture_preprocess_mode
        self.conflict_deltae_norm = float(conflict_deltae_norm)

        self.vae_tf = transforms.Compose(
            [
                transforms.Resize((height, width)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.mask_tf = transforms.Compose(
            [
                transforms.Resize(
                    (height, width),
                    interpolation=transforms.InterpolationMode.NEAREST,
                ),
                transforms.ToTensor(),
            ]
        )
        self.clip_proc = CLIPImageProcessor()

    def _load(self, p):
        return Image.open(os.path.join(self.image_root, p)).convert("RGB")

    def __getitem__(self, i):
        it = self.data[i]

        cloth = self._load(it["cloth"])
        sketch = self._load(it["sketch"]).resize(cloth.size)

        texture_path = it.get("texture", it.get("color", it["cloth"]))
        texture = self._load(texture_path)

        texture_tensor = preprocess_texture_image(
            texture,
            width=self.width,
            height=self.height,
            mode=self.texture_preprocess_mode,
        )
        texture_for_clip = transforms.ToPILImage()(
            (texture_tensor * 0.5 + 0.5).clamp(0, 1)
        )

        caption = it["caption"] if isinstance(it["caption"], str) else it["caption"][0]
        input_ids = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        has_mask = 0
        if "mask" in it and it["mask"]:
            mask = self.mask_tf(
                Image.open(os.path.join(self.image_root, it["mask"])).convert("L")
            )
            mask = (mask > 0.5).float()
            has_mask = 1
        else:
            mask = self.mask_tf(sketch_to_garment_mask(sketch, self.width, self.height))
            mask = (mask > 0.5).float()

        conflict_info = compute_color_conflict(
            caption,
            ref_tensor=texture_tensor,
            deltae_norm=self.conflict_deltae_norm,
        )

        return {
            "vae_cloth": self.vae_tf(cloth),
            "vae_sketch": self.vae_tf(sketch),
            "clip_texture": self.clip_proc(
                images=texture_for_clip, return_tensors="pt"
            ).pixel_values[0],
            "texture_image": texture_tensor,
            "garment_mask": mask,
            "has_mask": torch.tensor(has_mask, dtype=torch.float32),
            "input_ids": input_ids,
            "text_color_rgb": torch.tensor(conflict_info["text_color_rgb"], dtype=torch.float32),
            "ref_palette_rgb": torch.tensor(conflict_info["ref_palette_rgb"], dtype=torch.float32),
            "has_text_color": torch.tensor(float(conflict_info["has_text_color"]), dtype=torch.float32),
            "color_conflict_score": torch.tensor(conflict_info["color_conflict_score"], dtype=torch.float32),
        }

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    return {k: torch.stack([x[k] for x in batch]) for k in batch[0].keys()}


# =========================
# Utils
# =========================
def load_checkpoint_file(path):
    return torch.load(path, map_location="cpu")


def safe_git_hash(default="unknown"):
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return default


def resolve_image_encoder_path(cli_image_encoder_path, texture_meta):
    if cli_image_encoder_path and cli_image_encoder_path != "auto":
        return cli_image_encoder_path
    if isinstance(texture_meta, dict):
        ckpt_path = texture_meta.get("image_encoder_path")
        if ckpt_path:
            return ckpt_path
    return "openai/clip-vit-large-patch14"


def load_image_encoder_flexible(image_encoder_path, device=None, dtype=None):
    try:
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path)
    except Exception:
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            image_encoder_path, subfolder="models/image_encoder"
        )
    if device is not None or dtype is not None:
        image_encoder = image_encoder.to(device=device, dtype=dtype)
    return image_encoder


def override_args_from_texture_meta(args, texture_meta):
    if not isinstance(texture_meta, dict):
        return

    if "texture_num_tokens" in texture_meta and not args.force_bf_num_tokens_override:
        args.bf_num_tokens = int(texture_meta["texture_num_tokens"])
    if "bf_base_channels" in texture_meta:
        args.bf_base_channels = int(texture_meta["bf_base_channels"])
    if "clip_hidden_layer" in texture_meta:
        args.clip_hidden_layer = int(texture_meta["clip_hidden_layer"])
    if "texture_mode" in texture_meta:
        args.texture_mode = str(texture_meta["texture_mode"])
    if "texture_preprocess_mode" in texture_meta:
        args.texture_preprocess_mode = str(texture_meta["texture_preprocess_mode"])
    if "width" in texture_meta and texture_meta["width"] and not args.force_resolution_override:
        args.width = int(texture_meta["width"])
    if "height" in texture_meta and texture_meta["height"] and not args.force_resolution_override:
        args.height = int(texture_meta["height"])


def save_training_manifest(args, resolved_image_encoder_path):
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": safe_git_hash(),
        "output_dir": args.output_dir,
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
        "pretrained_vae_model_path": args.pretrained_vae_model_path,
        "texture_condition_mode": args.texture_condition_mode,
        "texture_preprocess_mode": args.texture_preprocess_mode,
        "alpha": [args.alpha1, args.alpha2, args.alpha3, args.alpha4],
        "lambda_style": args.lambda_style,
        "style_loss_type": args.style_loss_type,
        "lambda_patch_style": args.lambda_patch_style,
        "lambda_edge": args.lambda_edge,
        "lambda_texture_color": args.lambda_texture_color,
        "lambda_texture_gram": args.lambda_texture_gram,
        "lambda_region_texture": args.lambda_region_texture,
        "lambda_region_color_lab": args.lambda_region_color_lab,
        "lambda_boundary": args.lambda_boundary,
        "lambda_leak": args.lambda_leak,
        "region_kernel_size": args.region_kernel_size,
        "layer_group_enabled": args.layer_group_enabled,
        "use_palette_tokens": args.use_palette_tokens,
        "num_palette_tokens": args.num_palette_tokens,
        "palette_branch_scale_init": args.palette_branch_scale_init,
        "palette_mlp_lr": args.palette_mlp_lr,
        "gate_min": args.gate_min,
        "gate_max": args.gate_max,
        "use_conflict_aware_gate": args.use_conflict_aware_gate,
        "conflict_texture_suppress_strength": args.conflict_texture_suppress_strength,
        "conflict_palette_suppress_strength": args.conflict_palette_suppress_strength,
        "conflict_deltae_norm": args.conflict_deltae_norm,
        "conflict_threshold": args.conflict_threshold,
        "joint_t_drop_rate": args.joint_t_drop_rate,
        "joint_i_drop_rate": args.joint_i_drop_rate,
        "joint_ti_drop_rate": args.joint_ti_drop_rate,
        "hybrid_drop_token_rate": args.hybrid_drop_token_rate,
        "hybrid_drop_spatial_rate": args.hybrid_drop_spatial_rate,
        "train_spatial_only": args.train_spatial_only,
        "reload_texture_adapter_after_gam_init": args.reload_texture_adapter_after_gam_init,
        "vis_every_n_steps": args.vis_every_n_steps,
        "num_vis_samples": args.num_vis_samples,
        "fixed_vis_json": args.fixed_vis_json,
        "image_encoder_path": resolved_image_encoder_path,
        "dataset_json_path": args.dataset_json_path,
        "width": args.width,
        "height": args.height,
    }
    with open(
        os.path.join(args.output_dir, "experiment_manifest.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def set_unet_trainable(unet):
    # 冻结 base UNet
    for p in unet.parameters():
        p.requires_grad = False

    # 只开放 attention processors
    for proc in unet.attn_processors.values():
        for p in proc.parameters():
            p.requires_grad = True


def set_texture_token_enabled(unet, enabled):
    if hasattr(unet, "module"):
        unet = unet.module
    for proc in unet.attn_processors.values():
        if isinstance(proc, IPAttnProcessor2_0):
            proc.use_ip_adapter = bool(enabled)


def set_palette_token_enabled(unet, enabled):
    if hasattr(unet, "module"):
        unet = unet.module
    for proc in unet.attn_processors.values():
        if isinstance(proc, IPAttnProcessor2_0):
            proc.use_palette_tokens = bool(enabled)


def _is_palette_key(key):
    return (
        "palette_branch_scale" in key
        or "to_k_palette" in key
        or "to_v_palette" in key
        or "palette" in key
    )


def _is_balanced_gate_key(key):
    return "balanced_gate" in key


def _collect_palette_summary(unet, palette_tokens=None):
    if hasattr(unet, "module"):
        unet = unet.module
    scales = []
    for proc in unet.attn_processors.values():
        if isinstance(proc, IPAttnProcessor2_0) and hasattr(proc, "palette_branch_scale"):
            scales.append(float(proc.palette_branch_scale.detach().float().cpu().item()))
    summary = {
        "palette_branch_scale": float(np.mean(scales)) if scales else 0.0,
        "palette_token_norm": 0.0,
    }
    if palette_tokens is not None:
        summary["palette_token_norm"] = float(
            palette_tokens.detach().float().norm(dim=-1).mean().cpu().item()
        )
    return summary


def _collect_balanced_gate_summary(unet):
    texture_values = []
    palette_values = []
    for _, proc in _iter_unet_processors(unet):
        if not isinstance(proc, IPAttnProcessor2_0):
            continue
        if getattr(proc, "last_balanced_texture_gate", None) is not None:
            texture_values.append(float(proc.last_balanced_texture_gate.detach().cpu().item()))
        if getattr(proc, "last_balanced_palette_gate", None) is not None:
            palette_values.append(float(proc.last_balanced_palette_gate.detach().cpu().item()))
    return {
        "balanced_texture_gate": float(np.mean(texture_values)) if texture_values else 1.0,
        "balanced_palette_gate": float(np.mean(palette_values)) if palette_values else 1.0,
    }


def _get_balanced_gate_params(unet, only_trainable=False):
    params = []
    for _, proc in _iter_unet_processors(unet):
        if not isinstance(proc, IPAttnProcessor2_0):
            continue
        if not getattr(proc, "use_balanced_fusion_gate", False):
            continue
        for name, p in proc.named_parameters():
            if not name.startswith("balanced_gate_"):
                continue
            if only_trainable and not p.requires_grad:
                continue
            params.append(p)
    return params


def _balanced_gate_l2(unet, device):
    params = _get_balanced_gate_params(unet, only_trainable=True)
    if not params:
        return torch.zeros((), device=device)
    values = [(p.float() * p.float()).mean() for p in params]
    return torch.stack(values).mean().to(device)


def load_partial_state(module, state_dict, key, name, strict=False):
    if key not in state_dict:
        print(f"[load] {name}: key '{key}' not found, keep init.")
        return
    missing, unexpected = module.load_state_dict(state_dict[key], strict=strict)
    print(f"[load] {name}: missing={len(missing)} unexpected={len(unexpected)}")


def print_load_key_details(prefix, missing, unexpected, max_items=32):
    if missing:
        print(f"{prefix} missing keys (first {min(len(missing), max_items)}):")
        for key in list(missing)[:max_items]:
            print(f"  MISSING {key}")
    if unexpected:
        print(f"{prefix} unexpected keys (first {min(len(unexpected), max_items)}):")
        for key in list(unexpected)[:max_items]:
            print(f"  UNEXPECTED {key}")


def load_joint_checkpoint_into_models(
    state_dict,
    unet,
    ref_unet,
    bf,
    spatial_texture_encoder,
    spatial_injection,
    palette_token_mlp=None,
    accelerator=None,
):
    if not isinstance(state_dict, dict):
        return

    load_partial_state(unet, state_dict, "unet", "unet", strict=False)
    load_partial_state(ref_unet, state_dict, "ref_unet", "ref_unet", strict=False)
    load_partial_state(
        bf, state_dict, "bf_texture_conditioner", "bf_texture_conditioner", strict=False
    )
    load_partial_state(
        spatial_texture_encoder,
        state_dict,
        "spatial_texture_encoder",
        "spatial_texture_encoder",
        strict=False,
    )
    load_partial_state(
        spatial_injection,
        state_dict,
        "spatial_injection",
        "spatial_injection",
        strict=False,
    )
    if palette_token_mlp is not None:
        load_partial_state(
            palette_token_mlp,
            state_dict,
            "palette_token_mlp",
            "palette_token_mlp",
            strict=False,
        )

    if "texture_adapter" in state_dict:
        unet_raw = accelerator.unwrap_model(unet) if accelerator is not None else (unet.module if hasattr(unet, "module") else unet)
        attn_module_list = nn.ModuleList(unet_raw.attn_processors.values())
        missing, unexpected = attn_module_list.load_state_dict(state_dict["texture_adapter"], strict=False)
        gate_missing = [k for k in missing if "texture_gate_delta" in k or "gate" in k]
        palette_missing = [k for k in missing if _is_palette_key(k)]
        balanced_missing = [k for k in missing if _is_balanced_gate_key(k)]
        nongate_missing = [k for k in missing if k not in gate_missing and k not in palette_missing and k not in balanced_missing]
        nongate_unexpected = [k for k in unexpected if "texture_gate_delta" not in k and "gate" not in k and not _is_palette_key(k) and not _is_balanced_gate_key(k)]
        print(f"[load] texture_adapter(attn processors): missing={len(missing)} unexpected={len(unexpected)}")
        if gate_missing:
            print(f"[Expected missing gate keys] {gate_missing[:16]}")
        if palette_missing:
            print(f"[Expected missing palette keys] {palette_missing[:16]}")
        if balanced_missing:
            print(f"[Expected missing balanced gate keys] {balanced_missing[:16]}")
        if nongate_missing:
            print(f"[Unexpected non-gate missing keys] {nongate_missing[:16]}")
        if unexpected:
            print(f"[Unexpected keys] {list(unexpected)[:16]}")
        if nongate_unexpected:
            print(f"[Unexpected non-gate unexpected keys] {nongate_unexpected[:16]}")


def adapt_bf_state_for_token_count(bf, bf_state, log_prefix="[load]"):
    bf_state = dict(bf_state)
    key = "resampler_queries"
    if key not in bf_state:
        return bf_state

    current = bf.state_dict().get(key)
    saved = bf_state[key]
    if current is None or tuple(saved.shape) == tuple(current.shape):
        return bf_state
    if saved.ndim != 3 or current.ndim != 3 or saved.shape[0] != current.shape[0] or saved.shape[2] != current.shape[2]:
        return bf_state

    adapted = current.detach().clone()
    n = min(saved.shape[1], current.shape[1])
    adapted[:, :n, :] = saved[:, :n, :]
    bf_state[key] = adapted
    print(
        f"{log_prefix} adapted bf_texture_conditioner.{key}: "
        f"checkpoint={tuple(saved.shape)} current={tuple(current.shape)} copied_tokens={n}"
    )
    return bf_state


def load_texture_adapter_branch(texture_state, unet, bf, log_prefix="[load]", debug=False):
    if not isinstance(texture_state, dict):
        return

    if "texture_adapter" in texture_state:
        adapter_modules = nn.ModuleList(unet.attn_processors.values())
        missing, unexpected = adapter_modules.load_state_dict(
            texture_state["texture_adapter"], strict=False
        )
        palette_missing = [k for k in missing if _is_palette_key(k)]
        balanced_missing = [k for k in missing if _is_balanced_gate_key(k)]
        nonpalette_missing = [k for k in missing if not _is_palette_key(k) and not _is_balanced_gate_key(k)]
        nonpalette_unexpected = [k for k in unexpected if not _is_palette_key(k) and not _is_balanced_gate_key(k)]
        print(
            f"{log_prefix} texture_adapter -> unet.attn_processors: "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
        if palette_missing:
            print(f"{log_prefix} expected missing palette keys: {palette_missing[:16]}")
        if balanced_missing:
            print(f"{log_prefix} expected missing balanced gate keys: {balanced_missing[:16]}")
        if nonpalette_missing:
            print(f"{log_prefix} WARNING non-palette missing keys: {nonpalette_missing[:16]}")
        if nonpalette_unexpected:
            print(f"{log_prefix} WARNING non-palette unexpected keys: {nonpalette_unexpected[:16]}")
        if debug:
            print(
                f"{log_prefix} texture_adapter key counts: "
                f"checkpoint={len(texture_state['texture_adapter'])} "
                f"current={len(adapter_modules.state_dict())}"
            )
            print_load_key_details(
                f"{log_prefix} texture_adapter -> unet.attn_processors",
                missing,
                unexpected,
            )

    if "bf_texture_conditioner" in texture_state:
        bf_state = adapt_bf_state_for_token_count(
            bf,
            texture_state["bf_texture_conditioner"],
            log_prefix=log_prefix,
        )
        missing, unexpected = bf.load_state_dict(
            bf_state, strict=False
        )
        print(
            f"{log_prefix} bf_texture_conditioner: "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
        if debug:
            print_load_key_details(
                f"{log_prefix} bf_texture_conditioner",
                missing,
                unexpected,
            )


def infer_checkpoint_step(path):
    if not path:
        return None
    candidates = [path, os.path.dirname(path)]
    for candidate in candidates:
        match = re.search(r"checkpoint-(\d+)", candidate)
        if match:
            return int(match.group(1))
    return None


def save_training_checkpoint(
    accelerator,
    unet,
    ref_unet,
    bf,
    spatial_texture_encoder,
    spatial_injection,
    palette_token_mlp,
    output_dir,
    global_step,
    args,
    resolved_image_encoder_path,
    aliases=None,
):
    save_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
    os.makedirs(save_dir, exist_ok=True)

    def unwrap(model):
        if accelerator is not None:
            return accelerator.unwrap_model(model)
        return model.module if hasattr(model, "module") else model

    unet_raw = unwrap(unet)
    ref_unet_raw = unwrap(ref_unet)
    bf_raw = unwrap(bf)
    spatial_texture_encoder_raw = unwrap(spatial_texture_encoder)
    spatial_injection_raw = unwrap(spatial_injection)
    palette_token_mlp_raw = unwrap(palette_token_mlp) if palette_token_mlp is not None else None
    texture_adapter_raw = nn.ModuleList(unet_raw.attn_processors.values())

    payload = {
        "checkpoint_format": "gam_texture_joint_v3",
        "unet": unet_raw.state_dict(),
        "ref_unet": ref_unet_raw.state_dict(),
        "texture_adapter": texture_adapter_raw.state_dict(),
        "bf_texture_conditioner": bf_raw.state_dict(),
        "spatial_texture_encoder": spatial_texture_encoder_raw.state_dict(),
        "spatial_injection": spatial_injection_raw.state_dict(),
        "palette_token_mlp": (
            palette_token_mlp_raw.state_dict() if palette_token_mlp_raw is not None else {}
        ),
        "meta": {
            "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
            "pretrained_vae_model_path": args.pretrained_vae_model_path,
            "texture_num_tokens": args.bf_num_tokens,
            "texture_mode": args.texture_mode,
            "texture_condition_mode": args.texture_condition_mode,
            "texture_preprocess_mode": args.texture_preprocess_mode,
            "lambda_style": args.lambda_style,
            "style_loss_type": args.style_loss_type,
            "lambda_patch_style": args.lambda_patch_style,
            "lambda_texture_color": args.lambda_texture_color,
            "lambda_texture_gram": args.lambda_texture_gram,
            "lambda_region_texture": args.lambda_region_texture,
            "lambda_region_color_lab": args.lambda_region_color_lab,
            "lambda_boundary": args.lambda_boundary,
            "lambda_leak": args.lambda_leak,
            "region_kernel_size": args.region_kernel_size,
            "layer_group_enabled": args.layer_group_enabled,
            "use_palette_tokens": args.use_palette_tokens,
            "num_palette_tokens": args.num_palette_tokens,
            "palette_branch_scale_init": args.palette_branch_scale_init,
            "palette_mlp_lr": args.palette_mlp_lr,
            "use_texture_gate": args.use_texture_gate,
            "gate_type": args.gate_type,
            "gate_init": args.gate_init,
            "gate_reg_weight": args.gate_reg_weight,
            "gate_min": args.gate_min,
            "gate_max": args.gate_max,
            "use_balanced_fusion_gate": args.use_balanced_fusion_gate,
            "balanced_gate_hidden_dim": args.balanced_gate_hidden_dim,
            "balanced_gate_scale": args.balanced_gate_scale,
            "balanced_gate_min": args.balanced_gate_min,
            "balanced_gate_max": args.balanced_gate_max,
            "balanced_gate_reg_weight": args.balanced_gate_reg_weight,
            "use_conflict_aware_gate": args.use_conflict_aware_gate,
            "conflict_texture_suppress_strength": args.conflict_texture_suppress_strength,
            "conflict_palette_suppress_strength": args.conflict_palette_suppress_strength,
            "conflict_deltae_norm": args.conflict_deltae_norm,
            "conflict_threshold": args.conflict_threshold,
            "joint_t_drop_rate": args.joint_t_drop_rate,
            "joint_i_drop_rate": args.joint_i_drop_rate,
            "joint_ti_drop_rate": args.joint_ti_drop_rate,
            "reload_texture_adapter_after_gam_init": args.reload_texture_adapter_after_gam_init,
            "image_encoder_path": resolved_image_encoder_path,
            "clip_hidden_layer": args.clip_hidden_layer,
            "alpha": [args.alpha1, args.alpha2, args.alpha3, args.alpha4],
            "bf_base_channels": args.bf_base_channels,
            "width": args.width,
            "height": args.height,
        },
    }
    torch.save(payload, os.path.join(save_dir, "joint_model.pt"))

    aliases = aliases or []
    for alias_name in aliases:
        alias_path = os.path.join(output_dir, alias_name)
        if os.path.lexists(alias_path):
            if os.path.islink(alias_path):
                os.unlink(alias_path)
            else:
                print(f"[warn] skip checkpoint alias because path exists: {alias_path}")
                continue
        rel_target = os.path.relpath(save_dir, start=output_dir)
        try:
            os.symlink(rel_target, alias_path, target_is_directory=True)
        except OSError as e:
            print(f"[warn] failed to create checkpoint symlink {alias_path} -> {rel_target}: {e}")
    return save_dir


def _iter_unet_processors(unet):
    unet_raw = unet.module if hasattr(unet, "module") else unet
    return list(unet_raw.attn_processors.items())


def _collect_gate_stats(unet):
    rows = []
    gate_values = []
    for name, proc in _iter_unet_processors(unet):
        if not hasattr(proc, "texture_gate_delta") or proc.texture_gate_delta is None:
            continue
        delta = float(proc.texture_gate_delta.detach().float().cpu().item())
        raw_value = float(math.exp(delta))
        gate_min = float(getattr(proc, "gate_min", 0.7))
        gate_max = float(getattr(proc, "gate_max", 1.3))
        value = float(min(max(raw_value, gate_min), gate_max))
        layer_group = getattr(proc, "layer_group", "all")
        rows.append(
            {
                "layer_name": name,
                "layer_group": layer_group,
                "gate_delta": delta,
                "gate_raw_value": raw_value,
                "gate_value": value,
            }
        )
        gate_values.append((layer_group, raw_value, value))
    if not gate_values:
        return rows, {}
    raw_values = np.array([raw for _, raw, _ in gate_values], dtype=np.float32)
    values = np.array([v for _, _, v in gate_values], dtype=np.float32)
    summary = {
        "gate_raw_mean": float(raw_values.mean()),
        "gate_raw_std": float(raw_values.std()),
        "gate_raw_min": float(raw_values.min()),
        "gate_raw_max": float(raw_values.max()),
        "gate_mean": float(values.mean()),
        "gate_std": float(values.std()),
        "gate_min": float(values.min()),
        "gate_max": float(values.max()),
    }
    for group in ("semantic", "detail", "all"):
        group_values = [v for g, _, v in gate_values if g == group]
        if group_values:
            summary[f"{group}_gate_mean"] = float(np.mean(group_values))
    return rows, summary


def _get_gate_params(unet, only_trainable=False):
    params = []
    seen = set()
    for _, proc in _iter_unet_processors(unet):
        if not hasattr(proc, "texture_gate_delta") or proc.texture_gate_delta is None:
            continue
        p = proc.texture_gate_delta
        if only_trainable and not p.requires_grad:
            continue
        if id(p) in seen:
            continue
        seen.add(id(p))
        params.append(p)
    return params


def _detached_gate_l2(unet, device):
    gate_params = _get_gate_params(unet, only_trainable=True)
    if not gate_params:
        return torch.tensor(0.0, device=device)
    values = [p.detach().float() * p.detach().float() for p in gate_params]
    return torch.stack(values).mean()


def _add_gate_l2_grad(unet, weight):
    if weight <= 0:
        return
    gate_params = _get_gate_params(unet, only_trainable=True)
    if not gate_params:
        return
    scale = float(2.0 * weight / max(1, len(gate_params)))
    for p in gate_params:
        grad = p.detach().float() * scale
        grad = grad.to(device=p.device, dtype=p.dtype)
        if p.grad is None:
            p.grad = grad.clone()
        else:
            p.grad = p.grad + grad


def _find_shared_attention_processors(unet):
    seen = {}
    shared = []
    for name, proc in _iter_unet_processors(unet):
        proc_id = id(proc)
        if proc_id in seen:
            shared.append((seen[proc_id], name))
        else:
            seen[proc_id] = name
    return shared


# =========================
# Loss
# =========================
class VGGGramStyleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        feats = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
        self.l3 = feats[:18]
        self.l4 = feats[:27]
        for p in self.parameters():
            p.requires_grad = False

    @staticmethod
    def gram(x):
        x = x.float().contiguous()
        b, c, h, w = x.shape
        x = x.view(b, c, h * w)
        return (x @ x.transpose(1, 2)) / (c * h * w + 1e-6)

    def forward(self, pred, target, mask=None):
        pred = pred.float().contiguous()
        target = target.float().contiguous()
        if mask is not None:
            mask = mask.float().contiguous()
            pred = pred * mask
            target = target * mask
            pred = pred.contiguous()
            target = target.contiguous()
        p3, t3 = self.l3(pred), self.l3(target)
        p4, t4 = self.l4(pred), self.l4(target)
        return F.l1_loss(self.gram(p3), self.gram(t3)) + F.l1_loss(
            self.gram(p4), self.gram(t4)
        )

    def patch_cosine_loss(self, pred, target, mask=None, patch_size=8, stride=8):
        pred = pred.float().contiguous()
        target = target.float().contiguous()
        if mask is not None:
            mask = mask.float().contiguous()
            pred = pred * mask
            target = target * mask
            pred = pred.contiguous()
            target = target.contiguous()
        p = F.unfold(pred.contiguous(), kernel_size=patch_size, stride=stride)
        t = F.unfold(target.contiguous(), kernel_size=patch_size, stride=stride)
        p = F.normalize(p, dim=1)
        t = F.normalize(t, dim=1)
        return 1.0 - (p * t).sum(dim=1).mean()


def reconstruct_x0(noisy_latents, noise_pred, timesteps, noise_scheduler):
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(
        device=noisy_latents.device, dtype=noisy_latents.dtype
    )
    alpha_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
    x0_hat = (noisy_latents - sqrt_one_minus_alpha_t * noise_pred) / torch.clamp(
        sqrt_alpha_t, min=1e-6
    )
    return x0_hat


def rgb_to_gray(x):
    return (
        0.2989 * x[:, 0:1, :, :]
        + 0.5870 * x[:, 1:2, :, :]
        + 0.1140 * x[:, 2:3, :, :]
    )


def sobel_edges(x):
    x = x.float().contiguous()
    x01 = (x + 1.0) * 0.5
    gray = rgb_to_gray(x01).contiguous()
    kx = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        device=gray.device,
        dtype=gray.dtype,
    ).unsqueeze(0)
    ky = torch.tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        device=gray.device,
        dtype=gray.dtype,
    ).unsqueeze(0)
    gx = F.conv2d(gray.contiguous(), kx.contiguous(), padding=1)
    gy = F.conv2d(gray.contiguous(), ky.contiguous(), padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def masked_edge_l1(pred, target, mask=None):
    pred = pred.float().contiguous()
    target = target.float().contiguous()
    pred_edge = sobel_edges(pred)
    target_edge = sobel_edges(target)
    if mask is not None:
        mask = mask.float().contiguous()
        mask = F.interpolate(mask, size=pred_edge.shape[-2:], mode="nearest").contiguous()
        pred_edge = pred_edge * mask
        target_edge = target_edge * mask
        pred_edge = pred_edge.contiguous()
        target_edge = target_edge.contiguous()
    return F.l1_loss(pred_edge, target_edge)


def _weighted_mean_per_sample(loss_per_sample, sample_weight=None):
    if sample_weight is None:
        return loss_per_sample.mean()
    weight = sample_weight.to(device=loss_per_sample.device, dtype=loss_per_sample.dtype).view(-1)
    denom = weight.sum().clamp_min(1.0)
    return (loss_per_sample * weight).sum() / denom


def _masked_channel_mean_std(x, mask=None):
    x = x.float().contiguous()
    x = ((x + 1.0) * 0.5).clamp(0.0, 1.0)
    if mask is None:
        mean = x.mean(dim=(2, 3))
        std = x.std(dim=(2, 3), unbiased=False)
        return mean, std

    mask = mask.float().contiguous()
    mask = F.interpolate(mask, size=x.shape[-2:], mode="nearest").contiguous()
    mask = mask.to(device=x.device, dtype=x.dtype).contiguous()
    denom = mask.sum(dim=(2, 3)).clamp_min(1.0)
    mean = (x * mask).sum(dim=(2, 3)) / denom
    var = (((x - mean[:, :, None, None]) * mask) ** 2).sum(dim=(2, 3)) / denom
    return mean, torch.sqrt(var + 1e-6)


def texture_color_stat_loss(pred, texture, garment_mask=None, sample_weight=None):
    pred = pred.float().contiguous()
    texture = texture.float().contiguous()
    if garment_mask is not None:
        garment_mask = garment_mask.float().contiguous()
    pred_mean, pred_std = _masked_channel_mean_std(pred, garment_mask)
    texture_mean, texture_std = _masked_channel_mean_std(texture, None)
    mean_loss = F.smooth_l1_loss(pred_mean, texture_mean, reduction="none").mean(dim=1)
    std_loss = F.smooth_l1_loss(pred_std, texture_std, reduction="none").mean(dim=1)
    return _weighted_mean_per_sample(mean_loss + std_loss, sample_weight=sample_weight)


def rgb_to_lab_normalized(x):
    x = x.float().contiguous()
    x = ((x + 1.0) * 0.5).clamp(0.0, 1.0)
    rgb = torch.where(
        x > 0.04045,
        torch.pow((x + 0.055) / 1.055, 2.4),
        x / 12.92,
    )
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    xyz_x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
    xyz_y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    xyz_z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883
    epsilon = 0.008856
    kappa = 903.3

    def lab_f(t):
        return torch.where(t > epsilon, torch.pow(t.clamp_min(1e-6), 1.0 / 3.0), (kappa * t + 16.0) / 116.0)

    fx = lab_f(xyz_x)
    fy = lab_f(xyz_y)
    fz = lab_f(xyz_z)
    lab_l = (116.0 * fy - 16.0) / 100.0
    lab_a = 500.0 * (fx - fy) / 128.0
    lab_b = 200.0 * (fy - fz) / 128.0
    return torch.cat([lab_l, lab_a, lab_b], dim=1).contiguous()


def masked_lab_mean(x, mask=None):
    lab = rgb_to_lab_normalized(x)
    if mask is None:
        return lab.mean(dim=(2, 3))
    mask = mask.float().contiguous()
    mask = F.interpolate(mask, size=lab.shape[-2:], mode="nearest").contiguous()
    mask = mask.to(device=lab.device, dtype=lab.dtype).contiguous()
    denom = mask.sum(dim=(2, 3)).clamp_min(1.0)
    return (lab * mask).sum(dim=(2, 3)) / denom


def region_color_lab_loss(pred, texture, garment_mask=None, sample_weight=None):
    pred_mean = masked_lab_mean(pred, garment_mask)
    texture_mean = masked_lab_mean(texture, None)
    loss_per_sample = F.smooth_l1_loss(pred_mean, texture_mean, reduction="none").mean(dim=1)
    return _weighted_mean_per_sample(loss_per_sample, sample_weight=sample_weight)


def build_region_masks(mask, kernel_size=9):
    """
    Split a garment mask into inner body, boundary band, and outside regions.
    mask: [B, 1, H, W], values in [0, 1].
    """
    k = max(1, int(kernel_size))
    if k % 2 == 0:
        k += 1

    mask = mask.float().contiguous().clamp(0.0, 1.0)
    dilated = F.max_pool2d(mask.contiguous(), kernel_size=k, stride=1, padding=k // 2).contiguous()
    eroded = -F.max_pool2d((-mask).contiguous(), kernel_size=k, stride=1, padding=k // 2)
    eroded = eroded.contiguous()

    body = eroded.clamp(0.0, 1.0).contiguous()
    boundary = (dilated - eroded).clamp(0.0, 1.0).contiguous()
    outside = (1.0 - dilated).clamp(0.0, 1.0).contiguous()
    return body, boundary, outside


def masked_l1_loss(pred, target, mask):
    pred = pred.float().contiguous()
    target = target.float().contiguous()
    mask = mask.float().contiguous()
    if mask.shape[-2:] != pred.shape[-2:]:
        mask = F.interpolate(mask, size=pred.shape[-2:], mode="nearest").contiguous()
    mask = mask.to(device=pred.device, dtype=pred.dtype).contiguous()
    if mask.shape[1] == 1 and pred.shape[1] != 1:
        mask = mask.expand(-1, pred.shape[1], -1, -1).contiguous()
    denom = mask.sum().clamp_min(1.0)
    return (torch.abs(pred - target) * mask).sum() / denom


# =========================
# Validation vis
# =========================
@torch.no_grad()
def run_mode_validation_vis(
    out_dir,
    step,
    modes,
    unet,
    ref_unet,
    bf,
    spatial_texture_encoder,
    spatial_injection,
    palette_token_mlp,
    image_encoder,
    text_encoder,
    vae,
    batch,
    noise_scheduler,
    args,
):
    os.makedirs(out_dir, exist_ok=True)

    latents = vae.encode(batch["vae_cloth"]).latent_dist.sample() * vae.config.scaling_factor
    ref_latents = (
        vae.encode(batch["vae_sketch"]).latent_dist.sample() * vae.config.scaling_factor
    )
    t = torch.full(
        (latents.shape[0],),
        noise_scheduler.config.num_train_timesteps - 1,
        device=latents.device,
        dtype=torch.long,
    )

    _ = ref_unet(ref_latents, torch.zeros_like(t), None, return_dict=False)
    sa = {
        n: ref_unet.attn_processors[n].cache["hidden_states"]
        for n in ref_unet.attn_processors.keys()
        if "attn1" in n and hasattr(ref_unet.attn_processors[n], "cache")
    }

    clip_out = image_encoder(batch["clip_texture"], output_hidden_states=True)
    text_h = text_encoder(batch["input_ids"])[0]

    for mode in modes:
        enc_h = text_h

        if mode in ("token", "hybrid"):
            tex_tokens, _ = bf(
                clip_image_embeds=clip_out.image_embeds,
                texture_images=batch["texture_image"],
                clip_vision_tokens=clip_out.hidden_states[args.clip_hidden_layer][
                    :, 1:, :
                ],
                texture_mode=args.texture_mode,
            )
            enc_h = torch.cat([enc_h, tex_tokens], dim=1)
            if args.use_palette_tokens:
                palette_tokens = palette_token_mlp(batch["texture_image"])
                enc_h = torch.cat([enc_h, palette_tokens], dim=1)

        if mode in ("spatial", "hybrid"):
            texture_feats = spatial_texture_encoder(batch["texture_image"])
            spatial_injection.set_features(texture_feats)
            spatial_injection.set_mask(batch["garment_mask"].float())
        else:
            spatial_injection.clear_features()
        set_texture_token_enabled(unet, mode in ("token", "hybrid"))
        set_palette_token_enabled(unet, args.use_palette_tokens and mode in ("token", "hybrid"))

        cross_attention_kwargs = {"sa_hidden_states": sa}
        if args.use_balanced_fusion_gate:
            cross_attention_kwargs["balanced_gate_timestep"] = (
                t.float() / float(noise_scheduler.config.num_train_timesteps)
            )
        if args.use_conflict_aware_gate:
            cross_attention_kwargs["color_conflict_score"] = batch["color_conflict_score"].to(
                device=latents.device,
                dtype=latents.dtype,
            )

        noise_pred = unet(
            latents,
            t,
            encoder_hidden_states=enc_h,
            cross_attention_kwargs=cross_attention_kwargs,
        ).sample

        x0_hat = reconstruct_x0(latents, noise_pred, t, noise_scheduler)
        decoded = vae.decode(x0_hat / vae.config.scaling_factor).sample

        stacked = []
        n_show = min(args.num_vis_samples, decoded.shape[0])
        for i in range(n_show):
            sketch = (batch["vae_sketch"][i : i + 1].float() + 1) * 0.5
            texture = (batch["texture_image"][i : i + 1].float() + 1) * 0.5
            gen = (decoded[i : i + 1].float() + 1) * 0.5
            target = (batch["vae_cloth"][i : i + 1].float() + 1) * 0.5
            stacked.extend([sketch[0], texture[0], gen[0], target[0]])

        grid = make_grid(torch.stack(stacked), nrow=4)
        mode_dir = os.path.join(out_dir, f"step_{step:06d}", mode)
        os.makedirs(mode_dir, exist_ok=True)
        save_image(grid, os.path.join(mode_dir, "x0_hat_grid.png"))
    spatial_injection.clear_features()
    set_texture_token_enabled(unet, False)


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()

    # model/data
    ap.add_argument("--pretrained_model_name_or_path", required=True)
    ap.add_argument("--pretrained_vae_model_path", required=True)
    ap.add_argument("--image_encoder_path", default="auto")
    ap.add_argument("--dataset_json_path", required=True)
    ap.add_argument("--data_root_path", required=True)

    # checkpoints
    ap.add_argument("--gam_init_ckpt", type=str, default="")
    ap.add_argument("--texture_adapter_ckpt", required=True)
    ap.add_argument("--resume_from_checkpoint", type=str, default="")
    ap.add_argument(
        "--reload_texture_adapter_after_gam_init",
        action="store_true",
        help=(
            "Load texture_adapter_ckpt again after gam_init_ckpt. "
            "Use this when adapting a GAM checkpoint to a dataset-specific "
            "texture adapter, e.g. BF-Fashion."
        ),
    )
    ap.add_argument(
        "--start_global_step",
        type=int,
        default=-1,
        help=(
            "Starting step used for checkpoint numbering. "
            "Use -1 to infer from gam_init_ckpt/resume_from_checkpoint path."
        ),
    )
    ap.add_argument("--output_dir", default="joint_texture_output")

    # train
    ap.add_argument("--train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--dataloader_num_workers", type=int, default=1)
    ap.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    ap.add_argument("--max_train_steps", type=int, default=-1)
    ap.add_argument("--num_train_epochs", type=int, default=5)
    ap.add_argument("--checkpointing_epochs", type=int, default=1)
    ap.add_argument("--learning_rate", type=float, default=5e-5)
    ap.add_argument("--num_warmup_steps", type=int, default=500)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument(
        "--debug_trainable_params",
        action="store_true",
        help="Print trainable UNet parameter indices and names for DDP unused-parameter debugging.",
    )
    ap.add_argument(
        "--debug_checkpoint_load",
        action="store_true",
        help="Print missing/unexpected checkpoint keys when loading texture/GAM checkpoints.",
    )
    ap.add_argument("--report_to", type=str, default="tensorboard", choices=["tensorboard", "wandb", "all", "none"])
    ap.add_argument("--wandb_project", type=str, default="IMAGGarment-1")
    ap.add_argument("--wandb_run_name", type=str, default=None)
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])

    # conditioning
    ap.add_argument("--bf_num_tokens", type=int, default=16)
    ap.add_argument("--bf_base_channels", type=int, default=32)
    ap.add_argument(
        "--force_bf_num_tokens_override",
        action="store_true",
        help="Keep CLI bf_num_tokens even when texture checkpoint metadata has another value.",
    )
    ap.add_argument(
        "--texture_mode",
        type=str,
        default="patch_resampled",
        choices=["patch_resampled", "legacy_pooled"],
    )
    ap.add_argument("--clip_hidden_layer", type=int, default=-1)
    ap.add_argument(
        "--texture_condition_mode",
        type=str,
        default="hybrid",
        choices=["token", "spatial", "hybrid"],
    )
    ap.add_argument(
        "--fusion_type",
        type=str,
        default="minimal",
        choices=["minimal", "bfm_like"],
        help="Deprecated: decoupled spatial no longer uses fusion_type.",
    )
    ap.add_argument(
        "--texture_preprocess_mode",
        type=str,
        default="plain_resize",
        choices=["plain_resize", "crop_tile", "plain"],
    )
    ap.add_argument(
        "--layer_group_enabled",
        type=int,
        default=0,
        choices=[0, 1],
        help="Enable Ti-MGD-style layer-grouped texture routing for token mode.",
    )
    ap.add_argument("--use_texture_gate", type=int, default=0, choices=[0, 1])
    ap.add_argument("--use_palette_tokens", type=int, default=0, choices=[0, 1])
    ap.add_argument("--num_palette_tokens", type=int, default=4)
    ap.add_argument("--palette_branch_scale_init", type=float, default=0.0)
    ap.add_argument("--palette_mlp_lr", type=float, default=5e-5)
    ap.add_argument("--gate_type", type=str, default="layer")
    ap.add_argument("--gate_init", type=str, default="identity")
    ap.add_argument("--gate_reg_weight", type=float, default=0.0)
    ap.add_argument("--gate_min", type=float, default=0.7)
    ap.add_argument("--gate_max", type=float, default=1.3)
    ap.add_argument("--freeze_except_gate", type=int, default=0, choices=[0, 1])
    ap.add_argument("--log_gate_stats", type=int, default=0, choices=[0, 1])
    ap.add_argument("--use_balanced_fusion_gate", type=int, default=0, choices=[0, 1])
    ap.add_argument("--balanced_gate_hidden_dim", type=int, default=64)
    ap.add_argument("--balanced_gate_scale", type=float, default=0.2)
    ap.add_argument("--balanced_gate_min", type=float, default=0.8)
    ap.add_argument("--balanced_gate_max", type=float, default=1.2)
    ap.add_argument("--balanced_gate_reg_weight", type=float, default=1e-4)
    ap.add_argument("--use_conflict_aware_gate", type=int, default=0, choices=[0, 1])
    ap.add_argument("--conflict_texture_suppress_strength", type=float, default=0.1)
    ap.add_argument("--conflict_palette_suppress_strength", type=float, default=0.4)
    ap.add_argument("--conflict_deltae_norm", type=float, default=50.0)
    ap.add_argument("--conflict_threshold", type=float, default=0.70)
    ap.add_argument("--ddp_find_unused_parameters", type=int, default=-1, choices=[-1, 0, 1])
    ap.add_argument("--disable_gradient_checkpointing", type=int, default=1, choices=[0, 1])
    ap.add_argument("--alpha1", type=float, default=1.0)
    ap.add_argument("--alpha2", type=float, default=1.0)
    ap.add_argument("--alpha3", type=float, default=0.7)
    ap.add_argument("--alpha4", type=float, default=0.5)

    # losses
    ap.add_argument("--lambda_style", type=float, default=1.0)
    ap.add_argument(
        "--style_loss_type",
        type=str,
        default="gram",
        choices=["gram", "gram+patch"],
    )
    ap.add_argument("--lambda_patch_style", type=float, default=0.0)
    ap.add_argument("--lambda_edge", type=float, default=0.05)
    ap.add_argument("--lambda_texture_color", type=float, default=0.0)
    ap.add_argument("--lambda_texture_gram", type=float, default=0.0)
    ap.add_argument(
        "--lambda_region_texture",
        type=float,
        default=0.0,
        help="Extra texture color-stat loss on the eroded garment body region.",
    )
    ap.add_argument(
        "--lambda_region_color_lab",
        type=float,
        default=0.0,
        help="LAB mean color consistency loss between garment/body region and texture reference.",
    )
    ap.add_argument(
        "--lambda_boundary",
        type=float,
        default=0.0,
        help="L1 reconstruction loss on the garment boundary band to suppress boundary texture spill.",
    )
    ap.add_argument(
        "--lambda_leak",
        type=float,
        default=0.0,
        help="L1 reconstruction loss outside the dilated garment mask to suppress texture leakage.",
    )
    ap.add_argument(
        "--region_kernel_size",
        type=int,
        default=9,
        help="Odd morphology kernel size used to build body/boundary/outside masks.",
    )

    # dropout
    ap.add_argument("--joint_t_drop_rate", type=float, default=0.2)
    ap.add_argument("--joint_i_drop_rate", type=float, default=0.05)
    ap.add_argument("--joint_ti_drop_rate", type=float, default=0.05)
    ap.add_argument("--hybrid_drop_token_rate", type=float, default=0.25)
    ap.add_argument("--hybrid_drop_spatial_rate", type=float, default=0.25)
    ap.add_argument("--train_spatial_only", action="store_true")

    # vis
    ap.add_argument("--val_vis_steps", type=int, default=500)
    ap.add_argument("--vis_every_n_steps", type=int, default=0)
    ap.add_argument("--num_vis_samples", type=int, default=4)
    ap.add_argument("--fixed_vis_json", type=str, default=None)

    # resolution
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument(
        "--force_resolution_override",
        action="store_true",
        help="Keep CLI width/height even when texture checkpoint metadata has another value.",
    )

    args = ap.parse_args()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False

    is_token_mode = args.texture_condition_mode in ("token", "hybrid")
    is_spatial_mode = args.texture_condition_mode in ("spatial", "hybrid")

    if (
        args.joint_t_drop_rate + args.joint_i_drop_rate + args.joint_ti_drop_rate
    ) > 1.0:
        raise ValueError("joint dropout probabilities sum must be <= 1.0")
    if (args.hybrid_drop_token_rate + args.hybrid_drop_spatial_rate) > 1.0:
        raise ValueError("hybrid branch dropout probabilities sum must be <= 1.0")

    os.makedirs(args.output_dir, exist_ok=True)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, "logs")
    )
    log_with = None if args.report_to == "none" else args.report_to
    if args.ddp_find_unused_parameters >= 0:
        find_unused_parameters = bool(args.ddp_find_unused_parameters)
    else:
        find_unused_parameters = True
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)
    accelerator_mixed_precision = args.mixed_precision
    if accelerator_mixed_precision is None:
        accelerator_mixed_precision = "fp16" if torch.cuda.is_available() else "no"
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=accelerator_mixed_precision,
        project_config=accelerator_project_config,
        log_with=log_with,
        kwargs_handlers=[ddp_kwargs],
    )
    if accelerator.is_main_process:
        print(f"[info] ddp_find_unused_parameters = {find_unused_parameters}")

    # ---- texture ckpt meta ----
    texture_state = load_checkpoint_file(args.texture_adapter_ckpt)
    texture_meta = extract_texture_metadata(texture_state)
    if accelerator.is_main_process and texture_meta:
        print(f"[train_GAM_texture_joint] texture checkpoint meta: {texture_meta}")
    override_args_from_texture_meta(args, texture_meta)

    if accelerator.is_main_process:
        print(f"[info] effective bf_num_tokens = {args.bf_num_tokens}")
        print(f"[info] effective bf_base_channels = {args.bf_base_channels}")
        print(f"[info] effective clip_hidden_layer = {args.clip_hidden_layer}")
        print(f"[info] effective texture_mode = {args.texture_mode}")
        print(f"[info] effective texture_preprocess_mode = {args.texture_preprocess_mode}")
        print(f"[info] effective resolution = {args.height} x {args.width}")
        print(f"[info] layer_group_enabled = {args.layer_group_enabled}, texture_condition_mode = {args.texture_condition_mode}")

    # ---- models ----
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )
    ref_unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_vae_model_path)

    # ---- attention processors ----
    attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = (
            None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.") :].split(".")[0])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        else:
            block_id = int(name[len("down_blocks.") :].split(".")[0])
            hidden_size = unet.config.block_out_channels[block_id]

        if cross_attention_dim is None:
            attn_procs[name] = LogoRefSAttnProcessor2_0(name, hidden_size)
        else:
            if args.layer_group_enabled:
                layer_group = _get_layer_group(name)
                detail_ts = _get_detail_text_scale(name)
            else:
                layer_group = "all"
                detail_ts = 0.1
            attn_procs[name] = IPAttnProcessor2_0(
                hidden_size, cross_attention_dim,
                num_tokens=args.bf_num_tokens,
                layer_group=layer_group,
                detail_text_scale=detail_ts,
                use_texture_gate=bool(args.use_texture_gate),
                gate_type=args.gate_type,
                gate_init=args.gate_init,
                gate_reg_weight=args.gate_reg_weight,
                gate_min=args.gate_min,
                gate_max=args.gate_max,
                use_palette_tokens=bool(args.use_palette_tokens),
                num_palette_tokens=args.num_palette_tokens,
                palette_branch_scale_init=args.palette_branch_scale_init,
                use_balanced_fusion_gate=bool(args.use_balanced_fusion_gate),
                balanced_gate_hidden_dim=args.balanced_gate_hidden_dim,
                balanced_gate_scale=args.balanced_gate_scale,
                balanced_gate_min=args.balanced_gate_min,
                balanced_gate_max=args.balanced_gate_max,
                use_conflict_aware_gate=bool(args.use_conflict_aware_gate),
                conflict_texture_suppress_strength=args.conflict_texture_suppress_strength,
                conflict_palette_suppress_strength=args.conflict_palette_suppress_strength,
                conflict_threshold=args.conflict_threshold,
            )
    unet.set_attn_processor(attn_procs)
    if args.disable_gradient_checkpointing and hasattr(unet, "disable_gradient_checkpointing"):
        unet.disable_gradient_checkpointing()
        if accelerator.is_main_process:
            print("[info] unet gradient checkpointing disabled")
    shared_processors = _find_shared_attention_processors(unet)
    if accelerator.is_main_process:
        if shared_processors:
            print(f"[WARNING] shared attention processor instances detected: {shared_processors[:16]}")
        else:
            print("[info] no shared attention processor instances detected")

    attn_procs2 = {}
    for name in ref_unet.attn_processors.keys():
        cross_attention_dim = (
            None
            if name.endswith("attn1.processor")
            else ref_unet.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size = ref_unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.") :].split(".")[0])
            hidden_size = list(reversed(ref_unet.config.block_out_channels))[block_id]
        else:
            block_id = int(name[len("down_blocks.") :].split(".")[0])
            hidden_size = ref_unet.config.block_out_channels[block_id]

        if cross_attention_dim is None:
            attn_procs2[name] = LogoCacheSAttnProcessor2_0(name, hidden_size)
        else:
            attn_procs2[name] = LogoCacheCAttnProcessor2_0(
                name, hidden_size, hidden_size
            )
    ref_unet.set_attn_processor(attn_procs2)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    image_encoder_path = resolve_image_encoder_path(args.image_encoder_path, texture_meta)
    if accelerator.is_main_process:
        print(f"[train_GAM_texture_joint] resolved image encoder path: {image_encoder_path}")
    image_encoder = load_image_encoder_flexible(image_encoder_path)
    image_encoder.requires_grad_(False)

    set_unet_trainable(unet)

    # ref_unet 只是 cache/reference，用冻结版
    for p in ref_unet.parameters():
        p.requires_grad = False

    bf = BFTextureConditioner(
        clip_embeddings_dim=image_encoder.config.hidden_size,
        cross_attention_dim=unet.config.cross_attention_dim,
        num_tokens=args.bf_num_tokens,
    )
    palette_token_mlp = PaletteTokenMLP(
        cross_attention_dim=unet.config.cross_attention_dim,
        num_palette_tokens=args.num_palette_tokens,
    )

    # 旧 token 路线初始化
    if accelerator.is_main_process:
        print(f"[load] texture branch init from: {args.texture_adapter_ckpt}")
    load_texture_adapter_branch(
        texture_state,
        unet,
        bf,
        log_prefix="[load]",
        debug=args.debug_checkpoint_load,
    )

    # decoupled texture-first spatial branch
    spatial_texture_encoder = MultiScaleTextureEncoder(stage_channels=(64, 128, 256, 256))
    spatial_injection = SpatialInjectionAdapter(
        unet=unet,
        fusion_channels=(64, 128, 256, 256),
        target_channels=(
            unet.config.block_out_channels[0],
            unet.config.block_out_channels[1],
            unet.config.block_out_channels[2],
            unet.config.block_out_channels[-1],
        ),
        alphas=(args.alpha1, args.alpha2, args.alpha3, args.alpha4),
    )

    # gam 初始化
    if args.gam_init_ckpt:
        if accelerator.is_main_process:
            print(f"[resume] loading gam_init_ckpt: {args.gam_init_ckpt}")
        init_state = torch.load(args.gam_init_ckpt, map_location="cpu")
        load_joint_checkpoint_into_models(
            init_state,
            unet,
            ref_unet,
            bf,
            spatial_texture_encoder,
            spatial_injection,
            palette_token_mlp,
            accelerator=accelerator,
        )
        if args.reload_texture_adapter_after_gam_init:
            if accelerator.is_main_process:
                print(
                    "[load] reloading texture branch after gam_init_ckpt "
                    f"from: {args.texture_adapter_ckpt}"
                )
            load_texture_adapter_branch(
                texture_state,
                unet,
                bf,
                log_prefix="[load after gam_init]",
                debug=args.debug_checkpoint_load,
            )

    # resume 继续训练
    if args.resume_from_checkpoint:
        if accelerator.is_main_process:
            print(f"[resume] loading resume_from_checkpoint: {args.resume_from_checkpoint}")
        resume_state = torch.load(args.resume_from_checkpoint, map_location="cpu")
        load_joint_checkpoint_into_models(
            resume_state,
            unet,
            ref_unet,
            bf,
            spatial_texture_encoder,
            spatial_injection,
            palette_token_mlp,
            accelerator=accelerator,
        )

    # bf/token + spatial branch 是否训练
    if args.train_spatial_only:
        for p in unet.parameters():
            p.requires_grad = False
        for p in ref_unet.parameters():
            p.requires_grad = False
        for p in bf.parameters():
            p.requires_grad = False
        if palette_token_mlp is not None:
            for p in palette_token_mlp.parameters():
                p.requires_grad = False
        use_spatial_train = is_spatial_mode
    else:
        for p in bf.parameters():
            p.requires_grad = is_token_mode
        if palette_token_mlp is not None:
            for p in palette_token_mlp.parameters():
                p.requires_grad = is_token_mode and bool(args.use_palette_tokens)
        use_spatial_train = is_spatial_mode

    for p in spatial_texture_encoder.parameters():
        p.requires_grad = use_spatial_train
    for p in spatial_injection.parameters():
        p.requires_grad = use_spatial_train

    if args.freeze_except_gate:
        for p in unet.parameters():
            p.requires_grad = False
        for _, proc in _iter_unet_processors(unet):
            if hasattr(proc, "texture_gate_delta") and proc.texture_gate_delta is not None:
                proc.texture_gate_delta.requires_grad = True
        for p in bf.parameters():
            p.requires_grad = False
        if palette_token_mlp is not None:
            for p in palette_token_mlp.parameters():
                p.requires_grad = False
        for p in spatial_texture_encoder.parameters():
            p.requires_grad = False
        for p in spatial_injection.parameters():
            p.requires_grad = False
        if accelerator.is_main_process:
            print("[info] freeze_except_gate=1, only texture_gate_delta remains trainable.")

    # 显式构造 trainable params
    trainable_param_groups = []
    trainable_params = []
    seen = set()

    def add_params(params, lr=None):
        unique = []
        for p in params:
            if p.requires_grad and id(p) not in seen:
                unique.append(p)
                seen.add(id(p))
        if unique:
            trainable_param_groups.append({
                "params": unique,
                "lr": args.learning_rate if lr is None else lr,
            })
            trainable_params.extend(unique)

    # 1. 只训练 UNet 的 attention processors（spatial-only 时不训练）
    if not args.train_spatial_only:
        add_params(nn.ModuleList(unet.attn_processors.values()).parameters())

    # 2. BF token conditioner
    add_params(bf.parameters())
    if palette_token_mlp is not None:
        add_params(palette_token_mlp.parameters(), lr=args.palette_mlp_lr)

    # 3. spatial 分支
    if use_spatial_train:
        add_params(spatial_texture_encoder.parameters())
        add_params(spatial_injection.parameters())  # SpatialInjectionAdapter only exposes proj params

    if args.freeze_except_gate:
        trainable_param_groups = []
        trainable_params = []
        gate_params = []
        for _, proc in _iter_unet_processors(unet):
            if hasattr(proc, "texture_gate_delta") and proc.texture_gate_delta is not None:
                gate_params.append(proc.texture_gate_delta)
        if gate_params:
            trainable_param_groups.append({"params": gate_params, "lr": args.learning_rate})
            trainable_params.extend(gate_params)

    if args.debug_trainable_params and accelerator.is_main_process:
        print("[debug] trainable UNet parameters:")
        for idx, (name, p) in enumerate(unet.named_parameters()):
            if p.requires_grad:
                print(f"[debug] unet_param[{idx}] {name} {tuple(p.shape)}")
        if args.freeze_except_gate:
            print("[debug] gate-only trainable parameters:")
            for idx, (name, proc) in enumerate(_iter_unet_processors(unet)):
                if hasattr(proc, "texture_gate_delta") and proc.texture_gate_delta is not None:
                    print(f"[debug] gate[{idx}] {name} {tuple(proc.texture_gate_delta.shape)}")

    # dataset
    ds = JointTextureDataset(
        args.dataset_json_path,
        tokenizer,
        args.data_root_path,
        width=args.width,
        height=args.height,
        texture_preprocess_mode=args.texture_preprocess_mode,
        conflict_deltae_norm=args.conflict_deltae_norm,
    )
    dl = DataLoader(
        ds,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    dataset_len = len(ds)
    steps_per_epoch = max(
        1,
        math.ceil(
            dataset_len
            / (
                args.train_batch_size
                * accelerator.num_processes
                * args.gradient_accumulation_steps
            )
        ),
    )
    if args.max_train_steps > 0:
        target_global_step = args.max_train_steps
        total_epochs = max(1, math.ceil(target_global_step / steps_per_epoch))
    else:
        total_epochs = max(1, args.num_train_epochs)
        target_global_step = total_epochs * steps_per_epoch
    checkpoint_interval_steps = max(1, args.checkpointing_epochs * steps_per_epoch)

    optimizer = torch.optim.AdamW(trainable_param_groups, lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=target_global_step,
    )

    fixed_vis_batch = None
    if args.fixed_vis_json and os.path.exists(args.fixed_vis_json):
        with open(args.fixed_vis_json, "r", encoding="utf-8") as f:
            fixed_indices = json.load(f)
        fixed_indices = fixed_indices[: args.num_vis_samples]
        fixed_vis_items = [ds[int(i)] for i in fixed_indices]
        fixed_vis_batch = collate_fn(fixed_vis_items)

    (
        unet,
        ref_unet,
        bf,
        spatial_texture_encoder,
        spatial_injection,
        palette_token_mlp,
        optimizer,
        dl,
        lr_scheduler,
    ) = accelerator.prepare(
        unet,
        ref_unet,
        bf,
        spatial_texture_encoder,
        spatial_injection,
        palette_token_mlp,
        optimizer,
        dl,
        lr_scheduler,
    )
    spatial_injection_module = accelerator.unwrap_model(spatial_injection)
    spatial_injection_module.bind_unet(accelerator.unwrap_model(unet))
    spatial_injection_module.enable()

    if accelerator.is_main_process and args.report_to != "none":
        init_kwargs = {}
        if args.report_to in ("wandb", "all"):
            init_kwargs = {
                "wandb": {
                    "name": args.wandb_run_name,
                    "entity": args.wandb_entity,
                    "mode": args.wandb_mode,
                }
            }
        accelerator.init_trackers(
            args.wandb_project,
            config=vars(args),
            init_kwargs=init_kwargs,
        )

    text_encoder.to(accelerator.device)
    vae.to(accelerator.device)
    image_encoder.to(accelerator.device)
    use_style_loss_module = (
        args.lambda_style > 0
        or args.lambda_patch_style > 0
        or args.lambda_texture_gram > 0
    )
    style_loss_fn = VGGGramStyleLoss().to(accelerator.device) if use_style_loss_module else None

    save_training_manifest(args, image_encoder_path)

    noise_scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        prediction_type="epsilon",
    )

    warned_no_mask_once = False
    drop_counts = {"t": 0, "i": 0, "ti": 0, "total": 0}
    branch_drop_counts = {"token": 0, "spatial": 0, "total": 0}

    if args.start_global_step >= 0:
        global_step = args.start_global_step
    else:
        global_step = (
            infer_checkpoint_step(args.resume_from_checkpoint)
            or infer_checkpoint_step(args.gam_init_ckpt)
            or 0
        )

    if accelerator.is_main_process:
        print(
            f"[train] dataset_size={dataset_len}, batch_size={args.train_batch_size}, "
            f"num_gpus={accelerator.num_processes}"
        )
        print(
            f"[train] steps_per_epoch={steps_per_epoch}, total_epochs={total_epochs}, "
            f"total_steps={target_global_step}"
        )
        print(
            f"[train] start_global_step={global_step}, "
            f"ckpt_interval={checkpoint_interval_steps} "
            f"(~{checkpoint_interval_steps // max(1, steps_per_epoch)} epoch(s))"
        )
        print(f"[info] use_texture_gate = {bool(args.use_texture_gate)}")
        print(f"[info] use_palette_tokens = {bool(args.use_palette_tokens)}")
        print(f"[info] use_balanced_fusion_gate = {bool(args.use_balanced_fusion_gate)}")
        print(f"[info] use_conflict_aware_gate = {bool(args.use_conflict_aware_gate)}")
        print(
            "[info] conflict suppression: "
            f"texture={args.conflict_texture_suppress_strength}, "
            f"palette={args.conflict_palette_suppress_strength}, "
            f"deltae_norm={args.conflict_deltae_norm}, "
            f"threshold={args.conflict_threshold}"
        )
        print(f"[info] num_palette_tokens = {args.num_palette_tokens}")
        print(f"[info] gate_type = {args.gate_type}")
        print(f"[info] gate_min = {args.gate_min}, gate_max = {args.gate_max}")
        gate_rows, gate_summary = _collect_gate_stats(unet)
        print(f"[info] number_of_gate_params = {len(gate_rows)}")
        if gate_rows:
            for row in gate_rows[:32]:
                print(
                    f"[gate] {row['layer_name']} | group={row['layer_group']} | "
                    f"delta={row['gate_delta']:.6f} | "
                    f"raw={row['gate_raw_value']:.6f} | gate={row['gate_value']:.6f}"
                )
            print(f"[gate] summary = {gate_summary}")

    total_steps_done = 0
    null_input_ids = tokenizer(
        "",
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids[0]

    while global_step < target_global_step:
        for batch in dl:
            current_epoch = global_step // max(1, steps_per_epoch)
            with accelerator.accumulate(unet):
                drop_token_branch = False
                drop_spatial_branch = False
                if args.texture_condition_mode == "hybrid":
                    r_branch = random.random()
                    if r_branch < args.hybrid_drop_token_rate:
                        drop_token_branch = True
                    elif r_branch < (
                        args.hybrid_drop_token_rate + args.hybrid_drop_spatial_rate
                    ):
                        drop_spatial_branch = True
                    branch_drop_counts["total"] += 1
                    if drop_token_branch:
                        branch_drop_counts["token"] += 1
                    if drop_spatial_branch:
                        branch_drop_counts["spatial"] += 1

                # ---- 先复制，再做 dropout ----
                input_ids = batch["input_ids"].clone()
                texture_image_target = batch["texture_image"].clone()
                texture_image = batch["texture_image"].clone()
                clip_texture = batch["clip_texture"].clone()

                joint_i_rate = args.joint_i_drop_rate
                joint_t_rate = args.joint_t_drop_rate
                joint_ti_rate = args.joint_ti_drop_rate
                if args.train_spatial_only:
                    joint_i_rate = 0.0
                    joint_ti_rate = 0.0
                if args.texture_condition_mode == "hybrid" and (
                    drop_token_branch or drop_spatial_branch
                ):
                    # branch dropout step should stay pure: do not apply image-related joint drop
                    joint_i_rate = 0.0
                    joint_ti_rate = 0.0

                bsz = input_ids.shape[0]
                texture_condition_weight = torch.ones(
                    bsz, device=accelerator.device, dtype=torch.float32
                )
                for bi in range(bsz):
                    r = random.random()
                    if r < joint_i_rate:
                        texture_image[bi] = 0.0
                        clip_texture[bi] = 0.0
                        texture_condition_weight[bi] = 0.0
                        drop_counts["i"] += 1
                    elif r < (joint_i_rate + joint_t_rate):
                        input_ids[bi] = null_input_ids.to(input_ids.device)
                        drop_counts["t"] += 1
                    elif r < (
                        joint_i_rate
                        + joint_t_rate
                        + joint_ti_rate
                    ):
                        input_ids[bi] = null_input_ids.to(input_ids.device)
                        texture_image[bi] = 0.0
                        clip_texture[bi] = 0.0
                        texture_condition_weight[bi] = 0.0
                        drop_counts["ti"] += 1
                    drop_counts["total"] += 1

                # ---- 用 dropout 后的条件编码 ----
                with torch.no_grad():
                    latents = (
                        vae.encode(batch["vae_cloth"]).latent_dist.sample()
                        * vae.config.scaling_factor
                    )
                    ref_latents = (
                        vae.encode(batch["vae_sketch"]).latent_dist.sample()
                        * vae.config.scaling_factor
                    )
                    text_h = text_encoder(input_ids)[0]
                    clip_out = image_encoder(clip_texture, output_hidden_states=True)

                use_token = is_token_mode and (
                    not drop_token_branch
                )
                use_spatial = is_spatial_mode and (
                    not drop_spatial_branch
                )

                enc_h = text_h
                palette_tokens = None
                if use_token:
                    tex_tokens, _ = bf(
                        clip_image_embeds=clip_out.image_embeds,
                        texture_images=texture_image,
                        clip_vision_tokens=clip_out.hidden_states[
                            args.clip_hidden_layer
                        ][:, 1:, :],
                        texture_mode=args.texture_mode,
                    )
                    if tex_tokens.shape[1] != args.bf_num_tokens:
                        raise RuntimeError(
                            "Texture token count mismatch: "
                            f"got {tex_tokens.shape[1]}, expected {args.bf_num_tokens}. "
                            "Check texture checkpoint metadata and --force_bf_num_tokens_override."
                        )
                    enc_h = torch.cat([enc_h, tex_tokens], dim=1)
                    if args.use_palette_tokens:
                        palette_tokens = palette_token_mlp(texture_image)
                        palette_tokens = palette_tokens * texture_condition_weight.view(-1, 1, 1).to(
                            device=palette_tokens.device, dtype=palette_tokens.dtype
                        )
                        enc_h = torch.cat([enc_h, palette_tokens], dim=1)

                if use_spatial:
                    texture_feats = spatial_texture_encoder(texture_image)
                    spatial_injection_module.set_features(texture_feats)
                    spatial_injection_module.set_mask(batch["garment_mask"].float())
                else:
                    spatial_injection_module.clear_features()
                set_texture_token_enabled(unet, use_token)
                set_palette_token_enabled(unet, use_token and bool(args.use_palette_tokens))

                # ---- diffusion 前向 ----
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device,
                    dtype=torch.long,
                )
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # ---- ref_unet cache ----
                ref_timesteps = torch.zeros_like(timesteps)
                _ = ref_unet(ref_latents, ref_timesteps, None, return_dict=False)
                sa = {
                    n: ref_unet.attn_processors[n].cache["hidden_states"]
                    for n in ref_unet.attn_processors.keys()
                    if "attn1" in n and hasattr(ref_unet.attn_processors[n], "cache")
                }

                cross_attention_kwargs = {"sa_hidden_states": sa}
                if args.use_balanced_fusion_gate:
                    cross_attention_kwargs["balanced_gate_timestep"] = (
                        timesteps.float() / float(noise_scheduler.config.num_train_timesteps)
                    )
                if args.use_conflict_aware_gate:
                    cross_attention_kwargs["color_conflict_score"] = batch["color_conflict_score"].to(
                        device=noisy_latents.device,
                        dtype=noisy_latents.dtype,
                    )

                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=enc_h,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                loss_denoise = F.mse_loss(
                    noise_pred.float(), noise.float(), reduction="mean"
                )

                zero_aux_loss = loss_denoise.new_tensor(0.0)
                loss_style = zero_aux_loss
                loss_patch = zero_aux_loss
                loss_edge = zero_aux_loss
                loss_texture_color = zero_aux_loss
                loss_texture_gram = zero_aux_loss
                loss_region_texture = zero_aux_loss
                loss_region_color_lab = zero_aux_loss
                loss_boundary = zero_aux_loss
                loss_leak = zero_aux_loss

                use_region_losses = (
                    args.lambda_region_texture > 0
                    or args.lambda_region_color_lab > 0
                    or args.lambda_boundary > 0
                    or args.lambda_leak > 0
                )
                texture_loss_active = (use_token or use_spatial) and (
                    texture_condition_weight.sum().item() > 0
                )
                use_decoded_losses = (
                    args.lambda_style > 0
                    or (
                        args.style_loss_type == "gram+patch"
                        and args.lambda_patch_style > 0
                    )
                    or args.lambda_edge > 0
                    or (
                        texture_loss_active
                        and (
                            args.lambda_texture_color > 0
                            or args.lambda_texture_gram > 0
                            or args.lambda_region_texture > 0
                            or args.lambda_region_color_lab > 0
                        )
                    )
                    or args.lambda_boundary > 0
                    or args.lambda_leak > 0
                )

                if use_decoded_losses:
                    x0_hat = reconstruct_x0(
                        noisy_latents, noise_pred, timesteps, noise_scheduler
                    )
                    decoded = vae.decode(x0_hat / vae.config.scaling_factor).sample
                    target = batch["vae_cloth"]
                    mask = batch["garment_mask"]

                    if (
                        batch["has_mask"].sum().item() < batch["has_mask"].numel()
                        and not warned_no_mask_once
                        and accelerator.is_main_process
                    ):
                        print(
                            "[train_GAM_texture_joint] WARNING: some samples have no "
                            "garment mask, fallback to full-image style loss."
                        )
                        warned_no_mask_once = True

                    with torch.cuda.amp.autocast(enabled=False):
                        decoded_loss = decoded.float().contiguous()
                        target_loss = target.float().contiguous()
                        mask_loss = mask.float().contiguous()
                        texture_target_loss = texture_image_target.float().contiguous()

                        if use_region_losses:
                            body_mask, boundary_mask, outside_mask = build_region_masks(
                                mask_loss, kernel_size=args.region_kernel_size
                            )
                        else:
                            body_mask = boundary_mask = outside_mask = None

                        if args.lambda_style > 0:
                            loss_style = style_loss_fn(
                                decoded_loss, target_loss, mask=mask_loss
                            )
                        if (
                            args.style_loss_type == "gram+patch"
                            and args.lambda_patch_style > 0
                        ):
                            loss_patch = style_loss_fn.patch_cosine_loss(
                                decoded_loss, target_loss, mask=mask_loss
                            )
                        if args.lambda_edge > 0:
                            loss_edge = masked_edge_l1(
                                decoded_loss,
                                batch["vae_sketch"].float().contiguous(),
                                mask=mask_loss,
                            )
                        if texture_loss_active:
                            if args.lambda_texture_color > 0:
                                loss_texture_color = texture_color_stat_loss(
                                    decoded_loss,
                                    texture_target_loss,
                                    garment_mask=mask_loss,
                                    sample_weight=texture_condition_weight,
                                )
                            if args.lambda_texture_gram > 0:
                                keep_texture = texture_condition_weight > 0
                                loss_texture_gram = style_loss_fn(
                                    decoded_loss[keep_texture].contiguous(),
                                    texture_target_loss[keep_texture].contiguous(),
                                    mask=mask_loss[keep_texture].contiguous(),
                                )
                            if args.lambda_region_texture > 0:
                                loss_region_texture = texture_color_stat_loss(
                                    decoded_loss,
                                    texture_target_loss,
                                    garment_mask=body_mask,
                                    sample_weight=texture_condition_weight,
                                )
                            if args.lambda_region_color_lab > 0:
                                loss_region_color_lab = region_color_lab_loss(
                                    decoded_loss,
                                    texture_target_loss,
                                    garment_mask=body_mask,
                                    sample_weight=texture_condition_weight,
                                )

                        if args.lambda_boundary > 0:
                            loss_boundary = masked_l1_loss(
                                decoded_loss,
                                target_loss,
                                boundary_mask,
                            )
                        if args.lambda_leak > 0:
                            loss_leak = masked_l1_loss(
                                decoded_loss,
                                target_loss,
                                outside_mask,
                            )

                loss = loss_denoise + args.lambda_style * loss_style
                if (
                    args.style_loss_type == "gram+patch"
                    and args.lambda_patch_style > 0
                ):
                    loss = loss + args.lambda_patch_style * loss_patch
                loss = loss + args.lambda_edge * loss_edge
                loss = loss + args.lambda_texture_color * loss_texture_color
                loss = loss + args.lambda_texture_gram * loss_texture_gram
                loss = loss + args.lambda_region_texture * loss_region_texture
                loss = loss + args.lambda_region_color_lab * loss_region_color_lab
                loss = loss + args.lambda_boundary * loss_boundary
                loss = loss + args.lambda_leak * loss_leak
                loss_gate = _detached_gate_l2(unet, loss.device) if args.use_texture_gate else loss.new_tensor(0.0)
                loss_balanced_gate = (
                    _balanced_gate_l2(unet, loss.device)
                    if args.use_balanced_fusion_gate
                    else loss.new_tensor(0.0)
                )
                if args.use_balanced_fusion_gate and args.balanced_gate_reg_weight > 0:
                    loss = loss + args.balanced_gate_reg_weight * loss_balanced_gate

                if not torch.isfinite(loss.detach()).all():
                    loss_items = {
                        "loss_total": loss,
                        "loss_denoise": loss_denoise,
                        "loss_style": loss_style,
                        "loss_patch": loss_patch,
                        "loss_edge": loss_edge,
                        "loss_texture_color": loss_texture_color,
                        "loss_texture_gram": loss_texture_gram,
                        "loss_region_texture": loss_region_texture,
                        "loss_region_color_lab": loss_region_color_lab,
                        "loss_boundary": loss_boundary,
                        "loss_leak": loss_leak,
                        "loss_gate": loss_gate,
                        "loss_balanced_gate": loss_balanced_gate,
                    }
                    print("[train_GAM_texture_joint] non-finite loss detected")
                    for loss_name, loss_value in loss_items.items():
                        value = loss_value.detach().float()
                        is_finite = torch.isfinite(value).all().item()
                        print(
                            f"  {loss_name}: value={value.item():.8g}, "
                            f"finite={bool(is_finite)}"
                        )
                    raise RuntimeError("non-finite loss detected")

                accelerator.backward(loss)

                if accelerator.sync_gradients and args.use_texture_gate and args.gate_reg_weight > 0:
                    _add_gate_l2_grad(unet, args.gate_reg_weight)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        trainable_params, args.max_grad_norm
                    )
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                else:
                    grad_norm = None

                if accelerator.sync_gradients:
                    global_step += 1
                    total_steps_done += 1

                if accelerator.sync_gradients and args.report_to != "none":
                    palette_summary = _collect_palette_summary(unet, palette_tokens)
                    balanced_summary = _collect_balanced_gate_summary(unet)
                    grad_norm_log = None
                    if grad_norm is not None:
                        grad_norm_log = (
                            float(grad_norm.item())
                            if hasattr(grad_norm, "item")
                            else float(grad_norm)
                        )
                    accelerator.log(
                        {
                            "train/loss": loss.detach().float().item(),
                            "train/loss_denoise": loss_denoise.detach().float().item(),
                            "train/loss_style": loss_style.detach().float().item(),
                            "train/loss_patch": loss_patch.detach().float().item(),
                            "train/loss_edge": loss_edge.detach().float().item(),
                            "train/loss_texture_color": loss_texture_color.detach().float().item(),
                            "train/loss_texture_gram": loss_texture_gram.detach().float().item(),
                            "train/loss_region_texture": loss_region_texture.detach().float().item(),
                            "train/loss_region_color_lab": loss_region_color_lab.detach().float().item(),
                            "train/loss_boundary": loss_boundary.detach().float().item(),
                            "train/loss_leak": loss_leak.detach().float().item(),
                            "train/loss_gate": loss_gate.detach().float().item(),
                            "train/loss_balanced_gate": loss_balanced_gate.detach().float().item(),
                            "train/palette_branch_scale": palette_summary["palette_branch_scale"],
                            "train/palette_token_norm": palette_summary["palette_token_norm"],
                            "train/balanced_texture_gate": balanced_summary["balanced_texture_gate"],
                            "train/balanced_palette_gate": balanced_summary["balanced_palette_gate"],
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "train/grad_norm": grad_norm_log,
                            "train/drop_t_rate": drop_counts["t"] / max(1, drop_counts["total"]),
                            "train/drop_i_rate": drop_counts["i"] / max(1, drop_counts["total"]),
                            "train/drop_ti_rate": drop_counts["ti"] / max(1, drop_counts["total"]),
                            "train/drop_token_branch_rate": branch_drop_counts["token"] / max(1, branch_drop_counts["total"]),
                            "train/drop_spatial_branch_rate": branch_drop_counts["spatial"] / max(1, branch_drop_counts["total"]),
                            "train/use_token": float(use_token),
                            "train/use_spatial": float(use_spatial),
                            "train/texture_condition_keep_rate": texture_condition_weight.mean().item(),
                            "train/mean_conflict_score": batch["color_conflict_score"].detach().float().mean().item(),
                            "train/high_conflict_count": (
                                batch["color_conflict_score"].detach().float() >= args.conflict_threshold
                            ).sum().item(),
                            "train/no_text_color_count": (
                                batch["has_text_color"].detach().float() < 0.5
                            ).sum().item(),
                            "train/encoder_hidden_tokens": enc_h.shape[1],
                            "train/batch_size": bsz,
                            "train/epoch": float(current_epoch) + (float(global_step % max(1, steps_per_epoch)) / max(1, steps_per_epoch)),
                            "train/use_layer_group": float(args.layer_group_enabled),
                        },
                        step=global_step,
                    )

                if (
                    accelerator.sync_gradients
                    and accelerator.is_main_process
                    and (global_step == 1 or global_step % 100 == 0)
                ):
                    palette_summary = _collect_palette_summary(unet, palette_tokens)
                    balanced_summary = _collect_balanced_gate_summary(unet)
                    grad_norm_val = (
                        float(grad_norm.item())
                        if grad_norm is not None and not isinstance(grad_norm, float)
                        else grad_norm
                    )
                    print(
                        f"step={global_step}, epoch={current_epoch + 1}/{total_epochs}, "
                        f"loss_total={loss.item():.6f}, "
                        f"loss_denoise={loss_denoise.item():.6f}, "
                        f"loss_style={loss_style.item():.6f}, "
                        f"loss_patch={loss_patch.item():.6f}, "
                        f"loss_edge={loss_edge.item():.6f}, "
                        f"loss_tex_color={loss_texture_color.item():.6f}, "
                        f"loss_tex_gram={loss_texture_gram.item():.6f}, "
                        f"loss_region_tex={loss_region_texture.item():.6f}, "
                        f"loss_region_color_lab={loss_region_color_lab.item():.6f}, "
                        f"loss_boundary={loss_boundary.item():.6f}, "
                        f"loss_leak={loss_leak.item():.6f}, "
                        f"loss_gate={loss_gate.item():.6f}, "
                        f"loss_balanced_gate={loss_balanced_gate.item():.6f}, "
                        f"palette_scale={palette_summary['palette_branch_scale']:.6f}, "
                        f"palette_token_norm={palette_summary['palette_token_norm']:.6f}, "
                        f"balanced_texture_gate={balanced_summary['balanced_texture_gate']:.6f}, "
                        f"balanced_palette_gate={balanced_summary['balanced_palette_gate']:.6f}, "
                        f"grad_norm={grad_norm_val}, "
                        f"drop_t={drop_counts['t'] / max(1, drop_counts['total']):.3f}, "
                        f"drop_i={drop_counts['i'] / max(1, drop_counts['total']):.3f}, "
                        f"drop_ti={drop_counts['ti'] / max(1, drop_counts['total']):.3f}, "
                        f"drop_token_branch={branch_drop_counts['token'] / max(1, branch_drop_counts['total']):.3f}, "
                        f"drop_spatial_branch={branch_drop_counts['spatial'] / max(1, branch_drop_counts['total']):.3f}, "
                        f"mean_conflict={batch['color_conflict_score'].detach().float().mean().item():.3f}, "
                        f"high_conflict={(batch['color_conflict_score'].detach().float() >= args.conflict_threshold).sum().item()}, "
                        f"encoder_hidden_states={tuple(enc_h.shape)}"
                    )

                if (
                    accelerator.sync_gradients
                    and accelerator.is_main_process
                    and args.log_gate_stats
                    and global_step > 0
                    and global_step % max(1, checkpoint_interval_steps) == 0
                ):
                    gate_rows, gate_summary = _collect_gate_stats(unet)
                    gate_csv = os.path.join(args.output_dir, "gate_stats.csv")
                    gate_jsonl = os.path.join(args.output_dir, "gate_stats.jsonl")
                    if gate_rows:
                        write_header = not os.path.exists(gate_csv)
                        with open(gate_csv, "a", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(
                                f,
                                fieldnames=[
                                    "global_step",
                                    "layer_name",
                                    "layer_group",
                                    "gate_delta",
                                    "gate_raw_value",
                                    "gate_value",
                                ],
                            )
                            if write_header:
                                writer.writeheader()
                            for row in gate_rows:
                                writer.writerow({"global_step": global_step, **row})
                        with open(gate_jsonl, "a", encoding="utf-8") as f:
                            for row in gate_rows:
                                f.write(json.dumps({"global_step": global_step, **row}, ensure_ascii=False) + "\n")
                        if args.report_to != "none":
                            accelerator.log({f"gate/{k}": v for k, v in gate_summary.items()}, step=global_step)

                if (
                    accelerator.sync_gradients
                    and
                    accelerator.is_main_process
                    and args.val_vis_steps > 0
                    and global_step % args.val_vis_steps == 0
                    and global_step > 0
                ):
                    vis_batch = (
                        fixed_vis_batch
                        if fixed_vis_batch is not None
                        else {k: v[: args.num_vis_samples] for k, v in batch.items()}
                    )
                    vis_batch = {
                        k: v.to(accelerator.device) if hasattr(v, "to") else v
                        for k, v in vis_batch.items()
                    }
                    run_mode_validation_vis(
                        out_dir=os.path.join(args.output_dir, "val_outputs"),
                        step=global_step,
                        modes=["token", "spatial", "hybrid"],
                        unet=unet,
                        ref_unet=ref_unet,
                        bf=bf,
                        spatial_texture_encoder=spatial_texture_encoder,
                        spatial_injection=spatial_injection_module,
                        palette_token_mlp=palette_token_mlp,
                        image_encoder=image_encoder,
                        text_encoder=text_encoder,
                        vae=vae,
                        batch=vis_batch,
                        noise_scheduler=noise_scheduler,
                        args=args,
                    )

                if (
                    accelerator.sync_gradients
                    and
                    accelerator.is_main_process
                    and args.vis_every_n_steps > 0
                    and global_step % args.vis_every_n_steps == 0
                    and global_step > 0
                ):
                    vis_batch = (
                        fixed_vis_batch
                        if fixed_vis_batch is not None
                        else {k: v[: args.num_vis_samples] for k, v in batch.items()}
                    )
                    vis_batch = {
                        k: v.to(accelerator.device) if hasattr(v, "to") else v
                        for k, v in vis_batch.items()
                    }
                    run_mode_validation_vis(
                        out_dir=os.path.join(args.output_dir, "training_vis"),
                        step=global_step,
                        modes=[args.texture_condition_mode],
                        unet=unet,
                        ref_unet=ref_unet,
                        bf=bf,
                        spatial_texture_encoder=spatial_texture_encoder,
                        spatial_injection=spatial_injection_module,
                        palette_token_mlp=palette_token_mlp,
                        image_encoder=image_encoder,
                        text_encoder=text_encoder,
                        vae=vae,
                        batch=vis_batch,
                        noise_scheduler=noise_scheduler,
                        args=args,
                    )

                if (
                    accelerator.sync_gradients
                    and
                    accelerator.is_main_process
                    and checkpoint_interval_steps > 0
                    and global_step % checkpoint_interval_steps == 0
                    and global_step > 0
                ):
                    completed_epoch = min(
                        total_epochs,
                        max(1, math.ceil(global_step / max(1, steps_per_epoch))),
                    )
                    save_dir = save_training_checkpoint(
                        accelerator,
                        unet,
                        ref_unet,
                        bf,
                        spatial_texture_encoder,
                        spatial_injection,
                        palette_token_mlp,
                        args.output_dir,
                        global_step,
                        args,
                        image_encoder_path,
                        aliases=[f"checkpoint-epoch-{completed_epoch:03d}"],
                    )
                    if accelerator.is_main_process:
                        print(
                            f"[info] checkpoint saved to {save_dir} "
                            f"(epoch {completed_epoch}/{total_epochs})"
                        )

                if accelerator.sync_gradients and global_step >= target_global_step:
                    break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_dir = save_training_checkpoint(
            accelerator,
            unet,
            ref_unet,
            bf,
            spatial_texture_encoder,
            spatial_injection,
            palette_token_mlp,
            args.output_dir,
            global_step,
            args,
            image_encoder_path,
            aliases=[
                "checkpoint-final",
                f"checkpoint-epoch-{total_epochs:03d}",
            ],
        )
        print(f"[info] final full checkpoint saved to {save_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
