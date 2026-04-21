import os
import re
import math
import random
import argparse
import importlib.util
from pathlib import Path
import json
import itertools
import time

import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from torchvision.models import vgg19, VGG19_Weights
from torchvision.utils import make_grid, save_image
from PIL import Image
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from adapter.utils import is_torch2_available
from models.bf_texture_module import BFTextureConditioner
from texture_preprocess import preprocess_texture_image
if importlib.util.find_spec("repo_utils.checkpoint_utils") is not None:
    from repo_utils.checkpoint_utils import extract_texture_metadata
else:
    from checkpoint_utils import extract_texture_metadata

if is_torch2_available():
    from adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from adapter.attention_processor import IPAttnProcessor, AttnProcessor


logger = get_logger(__name__)


def load_image_encoder_flexible(image_encoder_path, device=None, dtype=None):
    try:
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path)
    except Exception:
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path, subfolder="models/image_encoder")
    if device is not None or dtype is not None:
        image_encoder = image_encoder.to(device=device, dtype=dtype)
    return image_encoder


class MyDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        json_file,
        tokenizer,
        size=512,
        height=640,
        width=512,
        t_drop_rate=0.05,
        i_drop_rate=0.05,
        ti_drop_rate=0.05,
        image_root_path="",
        texture_preprocess_mode="crop_tile",
        texture_crop_scale_min=0.4,
        texture_crop_scale_max=0.9,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.size = size
        self.height = height
        self.width = width
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path
        self.texture_preprocess_mode = texture_preprocess_mode
        self.texture_crop_scale_min = texture_crop_scale_min
        self.texture_crop_scale_max = texture_crop_scale_max

        with open(json_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.transform = transforms.Compose(
            [
                transforms.Resize([self.height, self.width], interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop([self.height, self.width]),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.clip_image_processor = CLIPImageProcessor()

    def _resolve_path(self, rel_path: str) -> str:
        return os.path.join(self.image_root_path, rel_path)

    def __getitem__(self, idx):
        item = self.data[idx]

        text = item["caption"]
        texture = item.get("texture", item.get("color", None))
        cloth = item["cloth"]

        if texture is None:
            raise KeyError(f"Sample {idx} has neither 'texture' nor 'color' field: {item}")

        texture_image = Image.open(self._resolve_path(texture)).convert("RGB")
        texture_tensor_for_cond = preprocess_texture_image(
            texture_image,
            width=self.width,
            height=self.height,
            mode=self.texture_preprocess_mode,
            crop_scale_min=self.texture_crop_scale_min,
            crop_scale_max=self.texture_crop_scale_max,
        )
        texture_image_for_cond = transforms.ToPILImage()((texture_tensor_for_cond * 0.5 + 0.5).clamp(0, 1))

        clip_texture_image = self.clip_image_processor(images=texture_image_for_cond, return_tensors="pt").pixel_values

        cloth_image = Image.open(self._resolve_path(cloth)).convert("RGB")
        cloth_image = self.transform(cloth_image)

        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_image_embed = 1

        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        return {
            "image": cloth_image,
            "texture_image": texture_tensor_for_cond,
            "texture_ref": self.transform(texture_image),
            "text_input_ids": text_input_ids,
            "clip_texture_image": clip_texture_image,
            "drop_image_embed": drop_image_embed,
        }

    def __len__(self):
        return len(self.data)


class VGGStyleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        feats = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
        self.layers = nn.ModuleList([feats[:4], feats[4:9], feats[9:18], feats[18:27]])
        for p in self.parameters():
            p.requires_grad = False

    @staticmethod
    def _gram(x):
        b, c, h, w = x.shape
        x = x.view(b, c, h * w)
        gram = x @ x.transpose(1, 2)
        return gram / (c * h * w)

    def forward(self, pred, target):
        loss = 0.0
        x = pred
        y = target
        for layer in self.layers:
            x = layer(x)
            y = layer(y)
            loss = loss + F.l1_loss(self._gram(x), self._gram(y))
        return loss


def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    texture_images = torch.stack([example["texture_image"] for example in data])
    texture_refs = torch.stack([example["texture_ref"] for example in data])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    clip_texture_images = torch.cat([example["clip_texture_image"] for example in data], dim=0)
    drop_image_embeds = [example["drop_image_embed"] for example in data]

    return {
        "images": images,
        "texture_images": texture_images,
        "texture_refs": texture_refs,
        "text_input_ids": text_input_ids,
        "clip_texture_images": clip_texture_images,
        "drop_image_embeds": drop_image_embeds,
    }


class TextureAdapter(torch.nn.Module):
    def __init__(self, unet, adapter_modules, bf_texture_conditioner, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.adapter_modules = adapter_modules
        self.bf_texture_conditioner = bf_texture_conditioner
        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def get_texture_condition_tokens(self, clip_outputs, texture_images, texture_mode="patch_resampled", clip_hidden_layer=-1):
        clip_image_embeds = clip_outputs.image_embeds
        clip_patch_tokens = clip_outputs.hidden_states[clip_hidden_layer][:, 1:, :]
        return self.bf_texture_conditioner(
            clip_image_embeds=clip_image_embeds,
            texture_images=texture_images,
            clip_vision_tokens=clip_patch_tokens,
            texture_mode=texture_mode,
        )[0]

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, clip_outputs, texture_images, texture_mode="patch_resampled", **kwargs):
        if texture_images is None:
            raise ValueError("texture_images must be provided for BF texture conditioning.")
        texture_tokens = self.get_texture_condition_tokens(clip_outputs, texture_images, texture_mode=texture_mode, clip_hidden_layer=kwargs.get("clip_hidden_layer", -1))
        encoder_hidden_states = torch.cat([encoder_hidden_states, texture_tokens], dim=1)
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return noise_pred, texture_tokens

    def load_from_checkpoint(self, ckpt_path: str):
        state_dict = torch.load(ckpt_path, map_location="cpu")
        meta = extract_texture_metadata(state_dict)
        if meta:
            print(f"[TextureAdapter] checkpoint meta: {meta}")
        if "bf_texture_conditioner" in state_dict:
            self.bf_texture_conditioner.load_state_dict(state_dict["bf_texture_conditioner"], strict=False)
        adapter_sd = state_dict.get("texture_adapter", state_dict.get("color_adapter", state_dict.get("ip_adapter", None)))
        if adapter_sd is None:
            raise KeyError(f"Cannot find adapter weights in checkpoint {ckpt_path}. Available keys: {list(state_dict.keys())}")
        self.adapter_modules.load_state_dict(adapter_sd, strict=False)
        print(f"Successfully loaded weights from checkpoint {ckpt_path}")


def save_texture_adapter_checkpoint(accelerator, model, save_path, meta=None):
    unwrapped = accelerator.unwrap_model(model)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    state = {
        "texture_adapter": unwrapped.adapter_modules.state_dict(),
        "bf_texture_conditioner": unwrapped.bf_texture_conditioner.state_dict(),
        "meta": meta or {},
    }
    torch.save(state, save_path)


def parse_step_from_ckpt_path(ckpt_path: str) -> int:
    if ckpt_path is None or ckpt_path == "":
        return 0
    m = re.search(r"checkpoint-(\d+)", ckpt_path)
    return int(m.group(1)) if m else 0


def compute_loss(noise_pred, noise, loss_type="mse", huber_c=0.1):
    if loss_type == "mse":
        return F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
    if loss_type == "huber":
        return F.huber_loss(noise_pred.float(), noise.float(), reduction="mean", delta=huber_c)
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def reconstruct_x0(noisy_latents, noise_pred, timesteps, noise_scheduler):
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
    alpha_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
    x0_hat = (noisy_latents - sqrt_one_minus_alpha_t * noise_pred) / torch.clamp(sqrt_alpha_t, min=1e-6)
    return x0_hat


def set_trainable_texture_blocks(unet, unfreeze_mid_block=True, unfreeze_up_blocks=2, unfreeze_attention_only=True):
    for p in unet.parameters():
        p.requires_grad = False

    for name, module in unet.attn_processors.items():
        if hasattr(module, "parameters"):
            for p in module.parameters():
                p.requires_grad = True

    def should_train(name):
        mid_match = unfreeze_mid_block and name.startswith("mid_block")
        up_match = False
        if name.startswith("up_blocks") and unfreeze_up_blocks > 0:
            try:
                idx = int(name.split(".")[1])
                up_match = idx >= (len(unet.up_blocks) - unfreeze_up_blocks)
            except Exception:
                up_match = False
        block_hit = mid_match or up_match
        if not block_hit:
            return False
        if not unfreeze_attention_only:
            return True
        keys = ["attn", "transformer", "to_q", "to_k", "to_v", "to_out"]
        return any(k in name for k in keys)

    for name, p in unet.named_parameters():
        if should_train(name):
            p.requires_grad = True


def parse_args():
    parser = argparse.ArgumentParser(description="Texture Adapter training script.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--pretrained_texture_adapter_path", type=str, default=None)
    parser.add_argument("--pretrained_color_adapter_path", type=str, default=None)
    parser.add_argument("--data_json_file", type=str, required=True)
    parser.add_argument("--data_root_path", type=str, default="", required=True)
    parser.add_argument("--image_encoder_path", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--i_drop_rate", type=float, default=0.05)
    parser.add_argument("--t_drop_rate", type=float, default=0.05)
    parser.add_argument("--ti_drop_rate", type=float, default=0.05)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--wandb_project", type=str, default="Mymodel")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--bf_num_tokens", type=int, default=16)
    parser.add_argument("--bf_base_channels", type=int, default=32)
    parser.add_argument("--texture_mode", type=str, default="patch_resampled", choices=["patch_resampled", "legacy_pooled"])
    parser.add_argument("--texture_preprocess_mode", type=str, default="crop_tile", choices=["plain_resize", "crop_tile", "plain"])
    parser.add_argument("--texture_crop_scale_min", type=float, default=0.4)
    parser.add_argument("--texture_crop_scale_max", type=float, default=0.9)
    parser.add_argument("--lambda_texture_style", type=float, default=0.1)
    parser.add_argument("--lambda_texture_global", type=float, default=0.0)
    parser.add_argument("--texture_loss_target_mode", type=str, default="conditioned_texture", choices=["conditioned_texture", "raw_texture"])
    parser.add_argument("--fixed_seed", type=int, default=1234)
    parser.add_argument("--clip_hidden_layer", type=int, default=-1)
    parser.add_argument("--unfreeze_mid_block", action="store_true", default=True)
    parser.add_argument("--no_unfreeze_mid_block", action="store_false", dest="unfreeze_mid_block")
    parser.add_argument("--unfreeze_up_blocks", type=int, default=2)
    parser.add_argument("--unfreeze_attention_only", action="store_true", default=True)
    parser.add_argument("--no_unfreeze_attention_only", action="store_false", dest="unfreeze_attention_only")
    parser.add_argument("--validation_steps", type=int, default=1000)
    parser.add_argument("--validation_num_textures", type=int, default=4)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["constant", "constant_with_warmup", "cosine", "cosine_with_restarts", "linear"])
    parser.add_argument("--lr_warmup_steps", type=int, default=300)
    parser.add_argument("--loss_type", type=str, default="huber", choices=["mse", "huber"])
    parser.add_argument("--huber_c", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args


def run_texture_validation(accelerator, args, texture_adapter, tokenizer, text_encoder, image_encoder, vae, noise_scheduler, val_items, step, weight_dtype):
    if len(val_items) == 0:
        return
    os.makedirs(os.path.join(args.output_dir, "val_grids"), exist_ok=True)
    model = accelerator.unwrap_model(texture_adapter)
    images = []

    fixed_latent = torch.randn((1, vae.config.latent_channels, args.height // 8, args.width // 8), device=accelerator.device, dtype=weight_dtype, generator=torch.Generator(device=accelerator.device).manual_seed(args.fixed_seed))
    for item in val_items:
        text_ids = tokenizer(item["caption"], padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt").input_ids.to(accelerator.device)
        with torch.no_grad():
            text_h = text_encoder(text_ids)[0]
        texture_img = Image.open(item["texture_path"]).convert("RGB")
        clip_tex = CLIPImageProcessor()(images=texture_img, return_tensors="pt").pixel_values.to(accelerator.device, dtype=weight_dtype)
        texture_tensor = transforms.Compose([
            transforms.Resize([args.height, args.width]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])(texture_img).unsqueeze(0).to(accelerator.device, dtype=weight_dtype)

        with torch.no_grad():
            clip_out = image_encoder(clip_tex, output_hidden_states=True)
            latents = fixed_latent.clone()
            t = torch.tensor([noise_scheduler.num_train_timesteps - 1], device=accelerator.device, dtype=torch.long)
            noise_pred, _ = model(latents, t, text_h, clip_out, texture_tensor, texture_mode=args.texture_mode, clip_hidden_layer=args.clip_hidden_layer)
            x0_hat = reconstruct_x0(latents, noise_pred, t, noise_scheduler)
            decoded = vae.decode(x0_hat / vae.config.scaling_factor).sample
            decoded = (decoded / 2 + 0.5).clamp(0, 1)
        images.append(decoded[0].cpu())

    grid = make_grid(images, nrow=len(images))
    grid_path = os.path.join(args.output_dir, "val_grids", f"texture_sensitivity_step_{step}.png")
    save_image(grid, grid_path)


def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    pretrained_adapter_path = args.pretrained_texture_adapter_path if args.pretrained_texture_adapter_path is not None else args.pretrained_color_adapter_path

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    image_encoder = load_image_encoder_flexible(args.image_encoder_path)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)

    bf_texture_conditioner = BFTextureConditioner(
        clip_embeddings_dim=image_encoder.config.hidden_size,
        cross_attention_dim=unet.config.cross_attention_dim,
        num_tokens=args.bf_num_tokens,
        base_channels=args.bf_base_channels,
        texture_mode=args.texture_mode,
    )

    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            raise ValueError(f"Unexpected attention processor name: {name}")

        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=args.bf_num_tokens)
            attn_procs[name].load_state_dict(weights, strict=False)

    unet.set_attn_processor(attn_procs)
    set_trainable_texture_blocks(unet, args.unfreeze_mid_block, args.unfreeze_up_blocks, args.unfreeze_attention_only)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())

    texture_adapter = TextureAdapter(unet=unet, adapter_modules=adapter_modules, bf_texture_conditioner=bf_texture_conditioner, ckpt_path=pretrained_adapter_path)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)

    style_loss_fn = VGGStyleLoss().to(accelerator.device)

    params_to_opt = [p for p in texture_adapter.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay, betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_epsilon)

    train_dataset = MyDataset(
        args.data_json_file,
        tokenizer=tokenizer,
        size=args.resolution,
        height=args.height,
        width=args.width,
        i_drop_rate=args.i_drop_rate,
        t_drop_rate=args.t_drop_rate,
        ti_drop_rate=args.ti_drop_rate,
        image_root_path=args.data_root_path,
        texture_preprocess_mode=args.texture_preprocess_mode,
        texture_crop_scale_min=args.texture_crop_scale_min,
        texture_crop_scale_max=args.texture_crop_scale_max,
    )

    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.train_batch_size, num_workers=args.dataloader_num_workers)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer, num_warmup_steps=args.lr_warmup_steps, num_training_steps=max_train_steps)

    texture_adapter, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(texture_adapter, optimizer, train_dataloader, lr_scheduler)

    if accelerator.is_main_process:
        init_kwargs = {}
        if args.report_to in ("wandb", "all"):
            init_kwargs = {"wandb": {"name": args.wandb_run_name, "entity": args.wandb_entity, "mode": args.wandb_mode}}
        accelerator.init_trackers(args.wandb_project, config=vars(args), init_kwargs=init_kwargs)

    checkpoint_meta = {
        "texture_num_tokens": args.bf_num_tokens,
        "texture_mode": args.texture_mode,
        "image_encoder_path": args.image_encoder_path,
        "clip_hidden_layer": args.clip_hidden_layer,
        "stage_token_hw": list(texture_adapter.bf_texture_conditioner.stage_token_hw),
        "texture_preprocess_mode": args.texture_preprocess_mode,
        "bf_base_channels": args.bf_base_channels,
        "clip_embeddings_dim": image_encoder.config.hidden_size,
        "texture_loss_target_mode": args.texture_loss_target_mode,
    }

    resume_step = parse_step_from_ckpt_path(pretrained_adapter_path)
    global_step = resume_step

    with open(args.data_json_file, "r", encoding="utf-8") as f:
        raw_items = json.load(f)
    val_items = []
    for item in raw_items[: args.validation_num_textures]:
        texture = item.get("texture", item.get("color"))
        if texture is None:
            continue
        val_items.append({"caption": item["caption"], "texture_path": os.path.join(args.data_root_path, texture)})

    for epoch in range(args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(texture_adapter):
                with torch.no_grad():
                    latents = vae.encode(batch["images"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                clip_outputs = image_encoder(batch["clip_texture_images"].to(accelerator.device, dtype=weight_dtype), output_hidden_states=True)
                image_embeds = clip_outputs.image_embeds
                image_embeds_ = []
                for image_embed, drop_image_embed in zip(image_embeds, batch["drop_image_embeds"]):
                    image_embeds_.append(torch.zeros_like(image_embed) if drop_image_embed == 1 else image_embed)
                image_embeds = torch.stack(image_embeds_)
                clip_outputs.image_embeds = image_embeds

                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["text_input_ids"].to(accelerator.device))[0]

                noise_pred, texture_tokens = texture_adapter(
                    noisy_latents=noisy_latents,
                    timesteps=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    clip_outputs=clip_outputs,
                    texture_images=batch["texture_images"].to(accelerator.device, dtype=weight_dtype),
                    texture_mode=args.texture_mode,
                    clip_hidden_layer=args.clip_hidden_layer,
                )

                loss_eps = compute_loss(noise_pred, noise, loss_type=args.loss_type, huber_c=args.huber_c)

                x0_hat = reconstruct_x0(noisy_latents, noise_pred, timesteps, noise_scheduler)
                decoded_pred = vae.decode(x0_hat / vae.config.scaling_factor).sample
                decoded_pred = decoded_pred.float()
                if args.texture_loss_target_mode == "conditioned_texture":
                    texture_ref = batch["texture_images"].to(accelerator.device, dtype=torch.float32)
                else:
                    texture_ref = batch["texture_refs"].to(accelerator.device, dtype=torch.float32)
                loss_style = style_loss_fn(decoded_pred, texture_ref)

                if args.lambda_texture_global > 0:
                    pred_global = F.normalize(decoded_pred.mean(dim=(2, 3)), dim=-1)
                    tex_global = F.normalize(texture_ref.mean(dim=(2, 3)), dim=-1)
                    loss_global = 1.0 - (pred_global * tex_global).sum(dim=-1).mean()
                else:
                    loss_global = torch.tensor(0.0, device=decoded_pred.device, dtype=decoded_pred.dtype)

                loss = loss_eps + args.lambda_texture_style * loss_style + args.lambda_texture_global * loss_global

                avg_loss = accelerator.gather(loss.detach().repeat(bsz)).mean().item()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_opt, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    global_step += 1
                    accelerator.log(
                        {
                            "train/loss": avg_loss,
                            "train/loss_eps": loss_eps.detach().item(),
                            "train/loss_style": loss_style.detach().item(),
                            "train/loss_global": loss_global.detach().item(),
                            "train/texture_token_count": texture_tokens.shape[1],
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "train/data_time": load_data_time,
                            "train/step_time": time.perf_counter() - begin,
                            "train/epoch": epoch,
                        },
                        step=global_step,
                    )

                    if accelerator.is_main_process and global_step % args.validation_steps == 0:
                        run_texture_validation(accelerator, args, texture_adapter, tokenizer, text_encoder, image_encoder, vae, noise_scheduler, val_items, global_step, weight_dtype)

                    if global_step % args.save_steps == 0:
                        save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_dir, safe_serialization=False)
                        if accelerator.is_main_process:
                            save_texture_adapter_checkpoint(accelerator, texture_adapter, os.path.join(save_dir, "texture_adapter.bin"), meta=checkpoint_meta)

            begin = time.perf_counter()

    final_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    accelerator.save_state(final_dir, safe_serialization=False)
    if accelerator.is_main_process:
        save_texture_adapter_checkpoint(accelerator, texture_adapter, os.path.join(final_dir, "texture_adapter.bin"), meta=checkpoint_meta)


if __name__ == "__main__":
    main()
