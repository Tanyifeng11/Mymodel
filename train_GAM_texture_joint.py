import argparse
import importlib.util
import itertools
import json
import os
import random
import subprocess
from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import VGG19_Weights, vgg19
from torchvision.utils import make_grid, save_image

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
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
from models.multiscale_texture_encoder import (
    MultiScaleSketchEncoder,
    MultiScaleTextureEncoder,
)
from models.spatial_fusion import MultiScaleFusion
from models.spatial_injection import SpatialInjectionAdapter
from texture_preprocess import preprocess_texture_image

if importlib.util.find_spec("repo_utils.checkpoint_utils") is not None:
    from repo_utils.checkpoint_utils import extract_texture_metadata
else:
    from checkpoint_utils import extract_texture_metadata


# =========================
# Dataset
# =========================
class JointTextureDataset(Dataset):
    def __init__(
        self,
        json_path,
        tokenizer,
        image_root,
        width=512,
        height=640,
        texture_preprocess_mode="crop_tile",
    ):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.image_root = image_root
        self.width = width
        self.height = height
        self.texture_preprocess_mode = texture_preprocess_mode

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
            mask = torch.ones(1, self.height, self.width)

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

    if "texture_num_tokens" in texture_meta:
        args.bf_num_tokens = int(texture_meta["texture_num_tokens"])
    if "bf_base_channels" in texture_meta:
        args.bf_base_channels = int(texture_meta["bf_base_channels"])
    if "clip_hidden_layer" in texture_meta:
        args.clip_hidden_layer = int(texture_meta["clip_hidden_layer"])
    if "texture_mode" in texture_meta:
        args.texture_mode = str(texture_meta["texture_mode"])
    if "width" in texture_meta and texture_meta["width"]:
        args.width = int(texture_meta["width"])
    if "height" in texture_meta and texture_meta["height"]:
        args.height = int(texture_meta["height"])


def save_training_manifest(args, resolved_image_encoder_path):
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": safe_git_hash(),
        "output_dir": args.output_dir,
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
        "pretrained_vae_model_path": args.pretrained_vae_model_path,
        "texture_condition_mode": args.texture_condition_mode,
        "fusion_type": args.fusion_type,
        "texture_preprocess_mode": args.texture_preprocess_mode,
        "alpha": [args.alpha1, args.alpha2, args.alpha3, args.alpha4],
        "lambda_style": args.lambda_style,
        "style_loss_type": args.style_loss_type,
        "lambda_patch_style": args.lambda_patch_style,
        "lambda_edge": args.lambda_edge,
        "joint_t_drop_rate": args.joint_t_drop_rate,
        "joint_i_drop_rate": args.joint_i_drop_rate,
        "joint_ti_drop_rate": args.joint_ti_drop_rate,
        "hybrid_drop_token_rate": args.hybrid_drop_token_rate,
        "hybrid_drop_spatial_rate": args.hybrid_drop_spatial_rate,
        "train_spatial_only": args.train_spatial_only,
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


def load_partial_state(module, state_dict, key, name, strict=False):
    if key not in state_dict:
        print(f"[load] {name}: key '{key}' not found, keep init.")
        return
    missing, unexpected = module.load_state_dict(state_dict[key], strict=strict)
    print(f"[load] {name}: missing={len(missing)} unexpected={len(unexpected)}")


def load_joint_checkpoint_into_models(
    state_dict,
    unet,
    ref_unet,
    bf,
    spatial_texture_encoder,
    spatial_sketch_encoder,
    spatial_fusion,
    spatial_injection,
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
        spatial_sketch_encoder,
        state_dict,
        "spatial_sketch_encoder",
        "spatial_sketch_encoder",
        strict=False,
    )
    load_partial_state(
        spatial_fusion, state_dict, "spatial_fusion", "spatial_fusion", strict=False
    )
    load_partial_state(
        spatial_injection,
        state_dict,
        "spatial_injection",
        "spatial_injection",
        strict=False,
    )

    if "texture_adapter" in state_dict:
        attn_module_list = nn.ModuleList(unet.attn_processors.values())
        missing, unexpected = attn_module_list.load_state_dict(
            state_dict["texture_adapter"], strict=False
        )
        print(
            f"[load] texture_adapter(attn processors): "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )


def save_training_checkpoint(
    accelerator,
    unet,
    ref_unet,
    bf,
    spatial_texture_encoder,
    spatial_sketch_encoder,
    spatial_fusion,
    spatial_injection,
    output_dir,
    global_step,
    args,
    resolved_image_encoder_path,
):
    save_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
    os.makedirs(save_dir, exist_ok=True)

    payload = {
        "checkpoint_format": "gam_texture_joint_v2",
        "unet": accelerator.unwrap_model(unet).state_dict(),
        "ref_unet": accelerator.unwrap_model(ref_unet).state_dict(),
        "texture_adapter": accelerator.unwrap_model(
            nn.ModuleList(unet.attn_processors.values())
        ).state_dict(),
        "bf_texture_conditioner": accelerator.unwrap_model(bf).state_dict(),
        "spatial_texture_encoder": accelerator.unwrap_model(
            spatial_texture_encoder
        ).state_dict(),
        "spatial_sketch_encoder": accelerator.unwrap_model(
            spatial_sketch_encoder
        ).state_dict(),
        "spatial_fusion": accelerator.unwrap_model(spatial_fusion).state_dict(),
        "spatial_injection": accelerator.unwrap_model(spatial_injection).state_dict(),
        "meta": {
            "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
            "pretrained_vae_model_path": args.pretrained_vae_model_path,
            "texture_num_tokens": args.bf_num_tokens,
            "texture_mode": args.texture_mode,
            "texture_condition_mode": args.texture_condition_mode,
            "fusion_type": args.fusion_type,
            "texture_preprocess_mode": args.texture_preprocess_mode,
            "lambda_style": args.lambda_style,
            "style_loss_type": args.style_loss_type,
            "lambda_patch_style": args.lambda_patch_style,
            "joint_t_drop_rate": args.joint_t_drop_rate,
            "joint_i_drop_rate": args.joint_i_drop_rate,
            "joint_ti_drop_rate": args.joint_ti_drop_rate,
            "image_encoder_path": resolved_image_encoder_path,
            "clip_hidden_layer": args.clip_hidden_layer,
            "alpha": [args.alpha1, args.alpha2, args.alpha3, args.alpha4],
            "bf_base_channels": args.bf_base_channels,
            "width": args.width,
            "height": args.height,
        },
    }
    torch.save(payload, os.path.join(save_dir, "joint_model.pt"))
    return save_dir


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
        b, c, h, w = x.shape
        x = x.view(b, c, h * w)
        return (x @ x.transpose(1, 2)) / (c * h * w + 1e-6)

    def forward(self, pred, target, mask=None):
        if mask is not None:
            pred = pred * mask
            target = target * mask
        p3, t3 = self.l3(pred), self.l3(target)
        p4, t4 = self.l4(pred), self.l4(target)
        return F.l1_loss(self.gram(p3), self.gram(t3)) + F.l1_loss(
            self.gram(p4), self.gram(t4)
        )

    def patch_cosine_loss(self, pred, target, mask=None, patch_size=8, stride=8):
        if mask is not None:
            pred = pred * mask
            target = target * mask
        p = F.unfold(pred, kernel_size=patch_size, stride=stride)
        t = F.unfold(target, kernel_size=patch_size, stride=stride)
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
    x01 = (x + 1.0) * 0.5
    gray = rgb_to_gray(x01)
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
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def masked_edge_l1(pred, target, mask=None):
    pred_edge = sobel_edges(pred)
    target_edge = sobel_edges(target)
    if mask is not None:
        mask = F.interpolate(mask, size=pred_edge.shape[-2:], mode="nearest")
        pred_edge = pred_edge * mask
        target_edge = target_edge * mask
    return F.l1_loss(pred_edge, target_edge)


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
    spatial_sketch_encoder,
    spatial_fusion,
    spatial_injection,
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

        if mode in ("spatial", "hybrid"):
            sketch_feats = spatial_sketch_encoder(batch["vae_sketch"])
            texture_feats = spatial_texture_encoder(batch["texture_image"])
            spatial_fusion.set_fusion_type(args.fusion_type)
            spatial_injection.set_features(spatial_fusion(sketch_feats, texture_feats))
        else:
            spatial_injection.clear_features()

        noise_pred = unet(
            latents,
            t,
            encoder_hidden_states=enc_h,
            cross_attention_kwargs={"sa_hidden_states": sa},
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
    ap.add_argument("--output_dir", default="joint_texture_output")

    # train
    ap.add_argument("--train_batch_size", type=int, default=1)
    ap.add_argument("--max_train_steps", type=int, default=20000)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--num_warmup_steps", type=int, default=500)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    # conditioning
    ap.add_argument("--bf_num_tokens", type=int, default=16)
    ap.add_argument("--bf_base_channels", type=int, default=32)
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
    )
    ap.add_argument(
        "--texture_preprocess_mode",
        type=str,
        default="plain_resize",
        choices=["plain_resize", "crop_tile", "plain"],
    )
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

    # dropout
    ap.add_argument("--joint_t_drop_rate", type=float, default=0.3)
    ap.add_argument("--joint_i_drop_rate", type=float, default=0.05)
    ap.add_argument("--joint_ti_drop_rate", type=float, default=0.05)
    ap.add_argument("--hybrid_drop_token_rate", type=float, default=0.25)
    ap.add_argument("--hybrid_drop_spatial_rate", type=float, default=0.25)
    ap.add_argument("--train_spatial_only", action="store_true")

    # vis
    ap.add_argument("--val_vis_steps", type=int, default=0)
    ap.add_argument("--vis_every_n_steps", type=int, default=0)
    ap.add_argument("--num_vis_samples", type=int, default=4)
    ap.add_argument("--fixed_vis_json", type=str, default=None)

    # resolution
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=640)

    args = ap.parse_args()

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
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="fp16" if torch.cuda.is_available() else "no",
        project_config=accelerator_project_config,
    )

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
            attn_procs[name] = IPAttnProcessor2_0(
                hidden_size, cross_attention_dim, num_tokens=args.bf_num_tokens
            )
    unet.set_attn_processor(attn_procs)

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

    # 旧 token 路线初始化
    if "texture_adapter" in texture_state:
        adapter_modules = nn.ModuleList(unet.attn_processors.values())
        missing, unexpected = adapter_modules.load_state_dict(
            texture_state["texture_adapter"], strict=False
        )
        if accelerator.is_main_process:
            print(
                f"[load] texture_adapter -> unet.attn_processors: "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )

    if "bf_texture_conditioner" in texture_state:
        missing, unexpected = bf.load_state_dict(
            texture_state["bf_texture_conditioner"], strict=False
        )
        if accelerator.is_main_process:
            print(
                f"[load] bf_texture_conditioner: "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )

    # 新 spatial 分支
    spatial_texture_encoder = MultiScaleTextureEncoder(stage_channels=(64, 128, 256, 256))
    spatial_sketch_encoder = MultiScaleSketchEncoder(stage_channels=(64, 128, 256, 256))
    spatial_fusion = MultiScaleFusion(
        (64, 128, 256, 256),
        (64, 128, 256, 256),
        (64, 128, 256, 256),
        fusion_type=args.fusion_type,
    )
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
    spatial_injection.enable()

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
            spatial_sketch_encoder,
            spatial_fusion,
            spatial_injection,
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
            spatial_sketch_encoder,
            spatial_fusion,
            spatial_injection,
        )

    # bf/token + spatial branch 是否训练
    if args.train_spatial_only:
        for p in unet.parameters():
            p.requires_grad = False
        for p in ref_unet.parameters():
            p.requires_grad = False
        for p in bf.parameters():
            p.requires_grad = False
        use_spatial_train = True
    else:
        for p in bf.parameters():
            p.requires_grad = args.texture_condition_mode in ("token", "hybrid")
        use_spatial_train = args.texture_condition_mode in ("spatial", "hybrid")

    for module in [spatial_texture_encoder, spatial_sketch_encoder, spatial_fusion]:
        for p in module.parameters():
            p.requires_grad = use_spatial_train
    for p in spatial_injection.parameters():
        p.requires_grad = use_spatial_train

    # 显式构造 trainable params
    trainable_param_groups = []
    trainable_params = []
    seen = set()

    def add_params(params):
        unique = []
        for p in params:
            if p.requires_grad and id(p) not in seen:
                unique.append(p)
                seen.add(id(p))
        if unique:
            trainable_param_groups.append({
                "params": unique,
                "lr": args.learning_rate,
            })
            trainable_params.extend(unique)

    # 1. 只训练 UNet 的 attention processors（spatial-only 时不训练）
    if not args.train_spatial_only:
        add_params(nn.ModuleList(unet.attn_processors.values()).parameters())

    # 2. BF token conditioner
    add_params(bf.parameters())

    # 3. spatial 分支
    if use_spatial_train:
        add_params(spatial_texture_encoder.parameters())
        add_params(spatial_sketch_encoder.parameters())
        add_params(spatial_fusion.parameters())
        add_params(spatial_injection.parameters())  # SpatialInjectionAdapter only exposes proj params

    optimizer = torch.optim.AdamW(trainable_param_groups, lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # dataset
    ds = JointTextureDataset(
        args.dataset_json_path,
        tokenizer,
        args.data_root_path,
        width=args.width,
        height=args.height,
        texture_preprocess_mode=args.texture_preprocess_mode,
    )
    dl = DataLoader(
        ds,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
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
        spatial_sketch_encoder,
        spatial_fusion,
        spatial_injection,
        optimizer,
        dl,
        lr_scheduler,
    ) = accelerator.prepare(
        unet,
        ref_unet,
        bf,
        spatial_texture_encoder,
        spatial_sketch_encoder,
        spatial_fusion,
        spatial_injection,
        optimizer,
        dl,
        lr_scheduler,
    )

    text_encoder.to(accelerator.device)
    vae.to(accelerator.device)
    image_encoder.to(accelerator.device)
    style_loss_fn = VGGGramStyleLoss().to(accelerator.device)

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
    global_step = 0

    null_input_ids = tokenizer(
        "",
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids[0]

    while global_step < args.max_train_steps:
        for batch in dl:
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
                for bi in range(bsz):
                    r = random.random()
                    if r < joint_i_rate:
                        texture_image[bi] = 0.0
                        clip_texture[bi] = 0.0
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

                use_token = args.texture_condition_mode in ("token", "hybrid") and (
                    not drop_token_branch
                )
                use_spatial = args.texture_condition_mode in ("spatial", "hybrid") and (
                    not drop_spatial_branch
                )

                enc_h = text_h
                if use_token:
                    tex_tokens, _ = bf(
                        clip_image_embeds=clip_out.image_embeds,
                        texture_images=texture_image,
                        clip_vision_tokens=clip_out.hidden_states[
                            args.clip_hidden_layer
                        ][:, 1:, :],
                        texture_mode=args.texture_mode,
                    )
                    enc_h = torch.cat([enc_h, tex_tokens], dim=1)

                if use_spatial:
                    sketch_feats = spatial_sketch_encoder(batch["vae_sketch"])
                    texture_feats = spatial_texture_encoder(texture_image)
                    spatial_fusion.set_fusion_type(args.fusion_type)
                    fused_feats = spatial_fusion(sketch_feats, texture_feats)
                    spatial_injection.set_features(fused_feats)
                else:
                    spatial_injection.clear_features()

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

                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=enc_h,
                    cross_attention_kwargs={"sa_hidden_states": sa},
                ).sample

                loss_denoise = F.mse_loss(
                    noise_pred.float(), noise.float(), reduction="mean"
                )

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

                loss_style = style_loss_fn(
                    decoded.float(), target.float(), mask=mask.float()
                )
                loss_patch = torch.tensor(0.0, device=loss_style.device)
                if (
                    args.style_loss_type == "gram+patch"
                    and args.lambda_patch_style > 0
                ):
                    loss_patch = style_loss_fn.patch_cosine_loss(
                        decoded.float(), target.float(), mask=mask.float()
                    )
                loss_edge = masked_edge_l1(
                    decoded.float(),
                    batch["vae_sketch"].float(),
                    mask=batch["garment_mask"].float(),
                )

                loss = loss_denoise + args.lambda_style * loss_style
                if (
                    args.style_loss_type == "gram+patch"
                    and args.lambda_patch_style > 0
                ):
                    loss = loss + args.lambda_patch_style * loss_patch
                loss = loss + args.lambda_edge * loss_edge

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        trainable_params, args.max_grad_norm
                    )
                else:
                    grad_norm = None

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if accelerator.is_main_process and global_step % 100 == 0:
                    grad_norm_val = (
                        float(grad_norm.item())
                        if grad_norm is not None and not isinstance(grad_norm, float)
                        else grad_norm
                    )
                    print(
                        f"step={global_step}, "
                        f"loss_total={loss.item():.6f}, "
                        f"loss_denoise={loss_denoise.item():.6f}, "
                        f"loss_style={loss_style.item():.6f}, "
                        f"loss_patch={loss_patch.item():.6f}, "
                        f"loss_edge={loss_edge.item():.6f}, "
                        f"grad_norm={grad_norm_val}, "
                        f"drop_t={drop_counts['t'] / max(1, drop_counts['total']):.3f}, "
                        f"drop_i={drop_counts['i'] / max(1, drop_counts['total']):.3f}, "
                        f"drop_ti={drop_counts['ti'] / max(1, drop_counts['total']):.3f}, "
                        f"drop_token_branch={branch_drop_counts['token'] / max(1, branch_drop_counts['total']):.3f}, "
                        f"drop_spatial_branch={branch_drop_counts['spatial'] / max(1, branch_drop_counts['total']):.3f}, "
                        f"encoder_hidden_states={tuple(enc_h.shape)}"
                    )

                if (
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
                        spatial_sketch_encoder=spatial_sketch_encoder,
                        spatial_fusion=spatial_fusion,
                        spatial_injection=spatial_injection,
                        image_encoder=image_encoder,
                        text_encoder=text_encoder,
                        vae=vae,
                        batch=vis_batch,
                        noise_scheduler=noise_scheduler,
                        args=args,
                    )

                if (
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
                        spatial_sketch_encoder=spatial_sketch_encoder,
                        spatial_fusion=spatial_fusion,
                        spatial_injection=spatial_injection,
                        image_encoder=image_encoder,
                        text_encoder=text_encoder,
                        vae=vae,
                        batch=vis_batch,
                        noise_scheduler=noise_scheduler,
                        args=args,
                    )

                if (
                    accelerator.is_main_process
                    and global_step % 2000 == 0
                    and global_step > 0
                ):
                    save_dir = save_training_checkpoint(
                        accelerator,
                        unet,
                        ref_unet,
                        bf,
                        spatial_texture_encoder,
                        spatial_sketch_encoder,
                        spatial_fusion,
                        spatial_injection,
                        args.output_dir,
                        global_step,
                        args,
                        image_encoder_path,
                    )
                    print(f"[info] checkpoint saved to {save_dir}")

            global_step += 1
            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_dir = save_training_checkpoint(
            accelerator,
            unet,
            ref_unet,
            bf,
            spatial_texture_encoder,
            spatial_sketch_encoder,
            spatial_fusion,
            spatial_injection,
            args.output_dir,
            global_step,
            args,
            image_encoder_path,
        )
        print(f"[info] final full checkpoint saved to {save_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
