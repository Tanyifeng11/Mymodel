import argparse
import gc
import itertools
import json
import os
import re
import time

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

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

try:
    from repo_utils.checkpoint_utils import (
        detect_gam_checkpoint_format,
        extract_texture_metadata,
        load_checkpoint_file,
    )
except Exception:
    from utils.checkpoint_utils import (
        detect_gam_checkpoint_format,
        extract_texture_metadata,
        load_checkpoint_file,
    )


class JointTextureDataset(Dataset):
    def __init__(self, json_path, tokenizer, image_root, height=448, width=320):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.image_root = image_root
        self.height = height
        self.width = width

        self.vae_tf = transforms.Compose(
            [
                transforms.Resize((self.height, self.width)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
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

        caption = it["caption"] if isinstance(it["caption"], str) else it["caption"][0]
        input_ids = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        return {
            "vae_cloth": self.vae_tf(cloth),
            "vae_sketch": self.vae_tf(sketch),
            "clip_texture": self.clip_proc(images=texture, return_tensors="pt").pixel_values[0],
            "texture_image": self.vae_tf(texture),
            "input_ids": input_ids,
        }

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    return {k: torch.stack([x[k] for x in batch]) for k in batch[0].keys()}


def get_weight_dtype(accelerator: Accelerator):
    if accelerator.mixed_precision == "fp16":
        return torch.float16
    if accelerator.mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def set_processors_only_trainable(model):
    for p in model.parameters():
        p.requires_grad = False

    for proc in model.attn_processors.values():
        for p in proc.parameters():
            p.requires_grad = True


def count_trainable_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def move_batch_to_device(batch, device, weight_dtype):
    out = {}
    for k, v in batch.items():
        if k == "input_ids":
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v.to(device, dtype=weight_dtype, non_blocking=True)
    return out


def maybe_enable_xformers(unet, ref_unet, accelerator):
    if accelerator.is_main_process:
        print("[info] skip xformers for custom attention processors.")


def build_unet_attn_processors(unet, num_tokens):
    st = unet.state_dict()
    attn_procs = {}

    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim

        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            hidden_size = list(reversed(unet.config.block_out_channels))[int(name[len("up_blocks.")])]
        else:
            hidden_size = unet.config.block_out_channels[int(name[len("down_blocks.")])]

        if cross_attention_dim is None:
            proc = LogoRefSAttnProcessor2_0(name, hidden_size)
            layer_name = name.split(".processor")[0]

            weights = {}
            k_name = layer_name + ".to_k.weight"
            v_name = layer_name + ".to_v.weight"
            if k_name in st:
                weights["to_k_ref.weight"] = st[k_name]
            if v_name in st:
                weights["to_v_ref.weight"] = st[v_name]
            if weights:
                proc.load_state_dict(weights, strict=False)

            attn_procs[name] = proc
        else:
            attn_procs[name] = IPAttnProcessor2_0(
                hidden_size,
                cross_attention_dim,
                num_tokens=num_tokens,
            )

    return attn_procs


def build_ref_unet_attn_processors(ref_unet):
    attn_procs = {}

    for name in ref_unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else ref_unet.config.cross_attention_dim

        if name.startswith("mid_block"):
            hidden_size = ref_unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            hidden_size = list(reversed(ref_unet.config.block_out_channels))[int(name[len("up_blocks.")])]
        else:
            hidden_size = ref_unet.config.block_out_channels[int(name[len("down_blocks.")])]

        if cross_attention_dim is None:
            attn_procs[name] = LogoCacheSAttnProcessor2_0(name, hidden_size)
        else:
            attn_procs[name] = LogoCacheCAttnProcessor2_0(name, hidden_size, hidden_size)

    return attn_procs


def build_bf_texture_conditioner(args, clip_embeddings_dim, cross_attention_dim):
    attempts = [
        {
            "clip_embeddings_dim": clip_embeddings_dim,
            "cross_attention_dim": cross_attention_dim,
            "num_tokens": args.bf_num_tokens,
            "base_channels": args.bf_base_channels,
            "texture_mode": args.texture_mode,
        },
        {
            "clip_embeddings_dim": clip_embeddings_dim,
            "cross_attention_dim": cross_attention_dim,
            "num_tokens": args.bf_num_tokens,
            "base_channels": args.bf_base_channels,
        },
        {
            "clip_embeddings_dim": clip_embeddings_dim,
            "cross_attention_dim": cross_attention_dim,
            "num_tokens": args.bf_num_tokens,
        },
    ]

    last_err = None
    for kwargs in attempts:
        try:
            return BFTextureConditioner(**kwargs)
        except TypeError as e:
            last_err = e

    raise last_err


def override_args_from_texture_meta(args, texture_meta, accelerator):
    if not texture_meta:
        return

    if "texture_num_tokens" in texture_meta:
        ckpt_tokens = int(texture_meta["texture_num_tokens"])
        if ckpt_tokens != args.bf_num_tokens and accelerator.is_main_process:
            print(f"[warning] bf_num_tokens mismatch: cli={args.bf_num_tokens}, ckpt={ckpt_tokens}. use ckpt value.")
        args.bf_num_tokens = ckpt_tokens

    if "clip_hidden_layer" in texture_meta:
        ckpt_layer = int(texture_meta["clip_hidden_layer"])
        if ckpt_layer != args.clip_hidden_layer and accelerator.is_main_process:
            print(f"[warning] clip_hidden_layer mismatch: cli={args.clip_hidden_layer}, ckpt={ckpt_layer}. use ckpt value.")
        args.clip_hidden_layer = ckpt_layer

    if "texture_mode" in texture_meta:
        ckpt_mode = str(texture_meta["texture_mode"])
        if ckpt_mode != args.texture_mode and accelerator.is_main_process:
            print(f"[warning] texture_mode mismatch: cli={args.texture_mode}, ckpt={ckpt_mode}. use ckpt value.")
        args.texture_mode = ckpt_mode

    if "bf_base_channels" in texture_meta:
        ckpt_base_channels = int(texture_meta["bf_base_channels"])
        if ckpt_base_channels != args.bf_base_channels and accelerator.is_main_process:
            print(f"[warning] bf_base_channels mismatch: cli={args.bf_base_channels}, ckpt={ckpt_base_channels}. use ckpt value.")
        args.bf_base_channels = ckpt_base_channels


def load_gam_init_checkpoint(gam_ckpt_path, unet, ref_unet, accelerator):
    if not gam_ckpt_path:
        return

    state = load_checkpoint_file(gam_ckpt_path)
    fmt = detect_gam_checkpoint_format(state)

    if accelerator.is_main_process:
        print(f"[load_gam_init_checkpoint] path={gam_ckpt_path}")
        print(f"[load_gam_init_checkpoint] detected format: {fmt}")

    if fmt == "legacy_module":
        model_sd = state["module"]

        unet_dict = {}
        ref_unet_dict = {}

        for k, v in model_sd.items():
            if k.startswith("unet."):
                unet_dict[k.replace("unet.", "", 1)] = v
            elif k.startswith("ref_unet."):
                ref_unet_dict[k.replace("ref_unet.", "", 1)] = v

        if unet_dict:
            msg = unet.load_state_dict(unet_dict, strict=False)
            if accelerator.is_main_process:
                print(f"[load_gam_init_checkpoint] loaded legacy unet: {msg}")

        if ref_unet_dict:
            msg = ref_unet.load_state_dict(ref_unet_dict, strict=False)
            if accelerator.is_main_process:
                print(f"[load_gam_init_checkpoint] loaded legacy ref_unet: {msg}")

    elif fmt == "gam_texture_joint_v1":
        if "unet" in state:
            msg = unet.load_state_dict(state["unet"], strict=False)
            if accelerator.is_main_process:
                print(f"[load_gam_init_checkpoint] loaded joint unet: {msg}")

        if "ref_unet" in state:
            msg = ref_unet.load_state_dict(state["ref_unet"], strict=False)
            if accelerator.is_main_process:
                print(f"[load_gam_init_checkpoint] loaded joint ref_unet: {msg}")

        meta = extract_texture_metadata(state)
        if meta and accelerator.is_main_process:
            print(f"[load_gam_init_checkpoint] metadata: {meta}")

    else:
        raise ValueError(f"Unsupported GAM init checkpoint format: {fmt}")


def load_texture_checkpoint_into_models(texture_state, adapter_modules, bf, accelerator):
    adapter_sd = texture_state.get(
        "texture_adapter",
        texture_state.get("color_adapter", texture_state.get("ip_adapter", None)),
    )
    bf_sd = texture_state.get("bf_texture_conditioner", None)

    if adapter_sd is None:
        raise KeyError(f"Cannot find texture adapter weights in checkpoint. Available keys: {list(texture_state.keys())}")
    if bf_sd is None:
        raise KeyError(f"Cannot find bf_texture_conditioner in checkpoint. Available keys: {list(texture_state.keys())}")

    msg1 = adapter_modules.load_state_dict(adapter_sd, strict=False)
    msg2 = bf.load_state_dict(bf_sd, strict=False)

    if accelerator.is_main_process:
        print(f"[load_texture_checkpoint] adapter load msg: {msg1}")
        print(f"[load_texture_checkpoint] bf load msg: {msg2}")


def save_joint_model_file(accelerator, unet, ref_unet, bf, save_path, args):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    unet_unwrapped = accelerator.unwrap_model(unet)
    ref_unet_unwrapped = accelerator.unwrap_model(ref_unet)
    bf_unwrapped = accelerator.unwrap_model(bf)

    adapter_modules = torch.nn.ModuleList(unet_unwrapped.attn_processors.values())

    torch.save(
        {
            "checkpoint_format": "gam_texture_joint_v1",
            "unet": unet_unwrapped.state_dict(),
            "ref_unet": ref_unet_unwrapped.state_dict(),
            "texture_adapter": adapter_modules.state_dict(),
            "bf_texture_conditioner": bf_unwrapped.state_dict(),
            "meta": {
                "texture_num_tokens": args.bf_num_tokens,
                "texture_mode": args.texture_mode,
                "image_encoder_path": args.image_encoder_path,
                "clip_hidden_layer": args.clip_hidden_layer,
                "bf_base_channels": args.bf_base_channels,
                "height": args.height,
                "width": args.width,
            },
        },
        save_path,
    )
    return save_path


def parse_step_from_checkpoint_name(path: str) -> int:
    if not path:
        return 0
    name = os.path.basename(os.path.normpath(path))
    m = re.search(r"checkpoint-(\d+)", name)
    return int(m.group(1)) if m else 0


def get_resume_step(resume_dir: str) -> int:
    trainer_state_path = os.path.join(resume_dir, "trainer_state.json")
    if os.path.exists(trainer_state_path):
        with open(trainer_state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        return int(state.get("global_step", 0))
    return parse_step_from_checkpoint_name(resume_dir)


def save_training_checkpoint(accelerator, unet, ref_unet, bf, output_dir, global_step, args):
    save_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
    os.makedirs(save_dir, exist_ok=True)

    # 完整训练状态：模型、优化器、scheduler、grad scaler 等
    accelerator.save_state(save_dir, safe_serialization=False)

    # 推理友好的单文件权重
    model_path = os.path.join(save_dir, "joint_model.pt")
    save_joint_model_file(accelerator, unet, ref_unet, bf, model_path, args)

    trainer_state = {
        "global_step": int(global_step),
        "image_encoder_path": args.image_encoder_path,
        "height": int(args.height),
        "width": int(args.width),
        "bf_num_tokens": int(args.bf_num_tokens),
        "bf_base_channels": int(args.bf_base_channels),
        "clip_hidden_layer": int(args.clip_hidden_layer),
        "texture_mode": args.texture_mode,
        "train_batch_size": int(args.train_batch_size),
        "learning_rate": float(args.learning_rate),
    }
    with open(os.path.join(save_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
        json.dump(trainer_state, f, ensure_ascii=False, indent=2)

    return save_dir


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--pretrained_model_name_or_path", required=True)
    ap.add_argument("--pretrained_vae_model_path", required=True)
    ap.add_argument("--image_encoder_path", required=True)

    ap.add_argument("--dataset_json_path", required=True)
    ap.add_argument("--data_root_path", required=True)

    ap.add_argument("--gam_init_ckpt", type=str, default="")
    ap.add_argument("--texture_adapter_ckpt", required=True)
    ap.add_argument("--resume_from_checkpoint", type=str, default="")

    ap.add_argument("--output_dir", default="joint_texture_output")

    ap.add_argument("--train_batch_size", type=int, default=1)
    ap.add_argument("--max_train_steps", type=int, default=20000)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--num_warmup_steps", type=int, default=500)

    ap.add_argument("--bf_num_tokens", type=int, default=16)
    ap.add_argument("--bf_base_channels", type=int, default=32)
    ap.add_argument("--texture_mode", type=str, default="patch_resampled", choices=["patch_resampled", "legacy_pooled"])
    ap.add_argument("--clip_hidden_layer", type=int, default=-1)

    ap.add_argument("--height", type=int, default=448)
    ap.add_argument("--width", type=int, default=320)

    ap.add_argument("--dataloader_num_workers", type=int, default=0)
    ap.add_argument("--save_steps", type=int, default=2000)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    # tracking / wandb
    ap.add_argument("--report_to", type=str, default="none", choices=["none", "wandb", "tensorboard", "all"])
    ap.add_argument("--wandb_project", type=str, default="Mymodel")
    ap.add_argument("--wandb_run_name", type=str, default=None)
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])

    return ap.parse_args()


def main():
    args = parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # 先解析 resume step，给 scheduler 和日志用
    resume_step_for_scheduler = 0
    if args.resume_from_checkpoint:
        resume_step_for_scheduler = get_resume_step(args.resume_from_checkpoint)

    logging_dir = os.path.join(args.output_dir, "logs")
    project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        log_with=None if args.report_to == "none" else args.report_to,
        project_config=project_config,
    )
    weight_dtype = get_weight_dtype(accelerator)

    os.makedirs(args.output_dir, exist_ok=True)

    texture_state = load_checkpoint_file(args.texture_adapter_ckpt)
    texture_meta = extract_texture_metadata(texture_state)
    if accelerator.is_main_process and texture_meta:
        print(f"[train_GAM_texture_joint] texture checkpoint meta: {texture_meta}")
    override_args_from_texture_meta(args, texture_meta, accelerator)

    if accelerator.is_main_process:
        print(f"[info] effective bf_num_tokens = {args.bf_num_tokens}")
        print(f"[info] effective bf_base_channels = {args.bf_base_channels}")
        print(f"[info] effective clip_hidden_layer = {args.clip_hidden_layer}")
        print(f"[info] effective texture_mode = {args.texture_mode}")
        print(f"[info] effective resolution = {args.height} x {args.width}")

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    ref_unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    vae = AutoencoderKL.from_pretrained(args.pretrained_vae_model_path)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)

    unet.set_attn_processor(build_unet_attn_processors(unet, args.bf_num_tokens))
    ref_unet.set_attn_processor(build_ref_unet_attn_processors(ref_unet))

    bf = build_bf_texture_conditioner(
        args=args,
        clip_embeddings_dim=image_encoder.config.hidden_size,
        cross_attention_dim=unet.config.cross_attention_dim,
    )

    is_resuming = bool(args.resume_from_checkpoint)

    if accelerator.is_main_process and is_resuming:
        print(f"[resume] resume mode enabled, checkpoint dir: {args.resume_from_checkpoint}")
        if args.gam_init_ckpt:
            print("[resume] ignore --gam_init_ckpt because --resume_from_checkpoint is set.")

    if not is_resuming and args.gam_init_ckpt:
        load_gam_init_checkpoint(args.gam_init_ckpt, unet, ref_unet, accelerator)

    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    if not is_resuming:
        load_texture_checkpoint_into_models(texture_state, adapter_modules, bf, accelerator)

    del texture_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)

    vae.eval()
    text_encoder.eval()
    image_encoder.eval()

    # 只训练 processors + bf
    set_processors_only_trainable(unet)
    set_processors_only_trainable(ref_unet)
    bf.requires_grad_(True)

    unet.enable_gradient_checkpointing()
    ref_unet.enable_gradient_checkpointing()
    maybe_enable_xformers(unet, ref_unet, accelerator)

    if accelerator.is_main_process:
        print(f"[info] trainable params in unet: {count_trainable_params(unet) / 1e6:.2f}M")
        print(f"[info] trainable params in ref_unet: {count_trainable_params(ref_unet) / 1e6:.2f}M")
        print(f"[info] trainable params in bf: {count_trainable_params(bf) / 1e6:.2f}M")
        print(f"[info] mixed precision = {accelerator.mixed_precision}")

    trainable_params = [
        p for p in itertools.chain(unet.parameters(), ref_unet.parameters(), bf.parameters())
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        foreach=False,
    )

    # resume 时让 scheduler 的 total steps 接着走
    scheduler_total_steps = resume_step_for_scheduler + args.max_train_steps
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=scheduler_total_steps,
    )

    ds = JointTextureDataset(
        args.dataset_json_path,
        tokenizer,
        args.data_root_path,
        height=args.height,
        width=args.width,
    )

    dl = DataLoader(
        ds,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.dataloader_num_workers > 0),
    )

    unet, ref_unet, bf, optimizer, dl, scheduler = accelerator.prepare(
        unet, ref_unet, bf, optimizer, dl, scheduler
    )

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)

    resume_step = 0
    if is_resuming:
        accelerator.load_state(args.resume_from_checkpoint)
        resume_step = get_resume_step(args.resume_from_checkpoint)
        if accelerator.is_main_process:
            print(f"[resume] loaded full training state from: {args.resume_from_checkpoint}")
            print(f"[resume] resume_step = {resume_step}")

    if accelerator.is_main_process and args.report_to != "none":
        init_kwargs = {}
        if args.report_to in ["wandb", "all"]:
            wandb_kwargs = {"mode": args.wandb_mode}
            if args.wandb_run_name:
                wandb_kwargs["name"] = args.wandb_run_name
            if args.wandb_entity:
                wandb_kwargs["entity"] = args.wandb_entity
            init_kwargs["wandb"] = wandb_kwargs

        tracker_name = args.wandb_project if args.report_to in ["wandb", "all"] else "train_GAM_texture_joint"
        accelerator.init_trackers(
            tracker_name,
            config=vars(args),
            init_kwargs=init_kwargs,
        )

        accelerator.log(
            {
                "dataset/num_samples": len(ds),
                "dataset/batch_size": args.train_batch_size,
                "dataset/height": args.height,
                "dataset/width": args.width,
                "model/trainable_unet_M": count_trainable_params(accelerator.unwrap_model(unet)) / 1e6,
                "model/trainable_ref_unet_M": count_trainable_params(accelerator.unwrap_model(ref_unet)) / 1e6,
                "model/trainable_bf_M": count_trainable_params(accelerator.unwrap_model(bf)) / 1e6,
            },
            step=resume_step,
        )

    # prepare/load_state 后重新拿一遍 trainable_params，供 clip_grad_norm_ 使用
    trainable_params = [
        p for p in itertools.chain(unet.parameters(), ref_unet.parameters(), bf.parameters())
        if p.requires_grad
    ]

    noise_scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        prediction_type="epsilon",
    )

    global_step = resume_step
    target_step = resume_step + args.max_train_steps

    unet.train()
    ref_unet.train()
    bf.train()

    begin = time.perf_counter()

    while global_step < target_step:
        for batch in dl:
            load_data_time = time.perf_counter() - begin

            batch = move_batch_to_device(batch, accelerator.device, weight_dtype)

            with torch.no_grad():
                with accelerator.autocast():
                    latents = vae.encode(batch["vae_cloth"]).latent_dist.sample() * 0.18215
                    ref_latents = vae.encode(batch["vae_sketch"]).latent_dist.sample() * 0.18215
                    text_h = text_encoder(batch["input_ids"])[0]
                    clip_out = image_encoder(batch["clip_texture"], output_hidden_states=True)

            noise = torch.randn_like(latents)
            t = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=latents.device,
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, t)

            with accelerator.autocast():
                tex_tokens, _ = bf(
                    clip_image_embeds=clip_out.image_embeds,
                    texture_images=batch["texture_image"],
                    clip_vision_tokens=clip_out.hidden_states[args.clip_hidden_layer][:, 1:, :],
                    texture_mode=args.texture_mode,
                )

                enc_h = torch.cat([text_h, tex_tokens], dim=1)

                _ = ref_unet(ref_latents, torch.zeros_like(t), None, return_dict=False)
                sa_hidden_states = {
                    n: ref_unet.attn_processors[n].cache["hidden_states"]
                    for n in ref_unet.attn_processors.keys()
                    if "attn1" in n
                }

                pred = unet(
                    noisy_latents,
                    t,
                    encoder_hidden_states=enc_h,
                    cross_attention_kwargs={"sa_hidden_states": sa_hidden_states},
                ).sample

                loss = F.mse_loss(pred.float(), noise.float(), reduction="mean")

            accelerator.backward(loss)
            grad_norm = accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            step_time = time.perf_counter() - begin

            if accelerator.is_main_process and global_step % 50 == 0:
                msg = f"step={global_step}, loss={loss.item():.6f}, encoder_hidden_states={tuple(enc_h.shape)}"
                if torch.cuda.is_available():
                    mem = torch.cuda.max_memory_allocated() / 1024**3
                    msg += f", max_mem={mem:.2f}GB"
                print(msg)

            if args.report_to != "none":
                log_dict = {
                    "train/loss": loss.detach().item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/data_time": load_data_time,
                    "train/step_time": step_time,
                    "train/encoder_hidden_states_len": enc_h.shape[1],
                }
                if grad_norm is not None:
                    try:
                        log_dict["train/grad_norm"] = float(grad_norm.detach().item())
                    except Exception:
                        pass
                if torch.cuda.is_available():
                    log_dict["train/max_mem_gb"] = torch.cuda.max_memory_allocated() / 1024**3
                accelerator.log(log_dict, step=global_step)

            if accelerator.is_main_process and global_step > 0 and global_step % args.save_steps == 0:
                save_dir = save_training_checkpoint(accelerator, unet, ref_unet, bf, args.output_dir, global_step, args)
                print(f"[info] saved full checkpoint to {save_dir}")

            del latents, ref_latents, text_h, clip_out
            del tex_tokens, enc_h, noisy_latents, pred, loss, sa_hidden_states, noise, t

            if torch.cuda.is_available() and global_step % 50 == 0:
                torch.cuda.empty_cache()

            if global_step >= target_step:
                break

            begin = time.perf_counter()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_dir = save_training_checkpoint(accelerator, unet, ref_unet, bf, args.output_dir, global_step, args)
        print(f"[info] final full checkpoint saved to {save_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()