import os
import random
import argparse
from pathlib import Path
import json
import itertools
import time

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from adapter.utils import is_torch2_available
from models.bf_texture_module import BFTextureConditioner

if is_torch2_available():
    from adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from adapter.attention_processor import IPAttnProcessor, AttnProcessor


logger = get_logger(__name__)


class MyDataset(torch.utils.data.Dataset):
    """
    Expected json format:
    [
        {
            "caption": "a white t-shirt",
            "texture": "texture/xxx.jpg",   # preferred
            "cloth": "cloth/xxx.jpg"
        }
    ]

    Backward compatibility:
    - if "texture" is absent, will fallback to "color"
    """

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
        clip_texture_image = self.clip_image_processor(images=texture_image, return_tensors="pt").pixel_values

        cloth_image = Image.open(self._resolve_path(cloth)).convert("RGB")
        cloth_image = self.transform(cloth_image)

        # classifier-free style condition dropping
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
            "texture_image": self.transform(texture_image),
            "text_input_ids": text_input_ids,
            "clip_texture_image": clip_texture_image,
            "drop_image_embed": drop_image_embed,
        }

    def __len__(self):
        return len(self.data)


def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    texture_images = torch.stack([example["texture_image"] for example in data])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    clip_texture_images = torch.cat([example["clip_texture_image"] for example in data], dim=0)
    drop_image_embeds = [example["drop_image_embed"] for example in data]

    return {
        "images": images,
        "texture_images": texture_images,
        "text_input_ids": text_input_ids,
        "clip_texture_images": clip_texture_images,
        "drop_image_embeds": drop_image_embeds,
    }


class TextureAdapter(torch.nn.Module):
    def __init__(
        self,
        unet,
        adapter_modules,
        bf_texture_conditioner,
        ckpt_path=None,
    ):
        super().__init__()
        self.unet = unet
        self.adapter_modules = adapter_modules
        self.bf_texture_conditioner = bf_texture_conditioner

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds, texture_images):
        if texture_images is None:
            raise ValueError("texture_images must be provided for BF texture conditioning.")
        texture_tokens, _ = self.bf_texture_conditioner(image_embeds, texture_images)
        encoder_hidden_states = torch.cat([encoder_hidden_states, texture_tokens], dim=1)
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))
        orig_bf_sum = torch.sum(torch.stack([torch.sum(p) for p in self.bf_texture_conditioner.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")
        if "bf_texture_conditioner" in state_dict:
            self.bf_texture_conditioner.load_state_dict(state_dict["bf_texture_conditioner"], strict=False)
            print("Loaded bf_texture_conditioner weights.")
        else:
            print("No bf_texture_conditioner key found in checkpoint, using random init for BF conditioner.")

        if "texture_adapter" in state_dict:
            adapter_sd = state_dict["texture_adapter"]
        elif "color_adapter" in state_dict:
            adapter_sd = state_dict["color_adapter"]
        elif "ip_adapter" in state_dict:
            adapter_sd = state_dict["ip_adapter"]
        else:
            raise KeyError(
                f"Cannot find adapter weights in checkpoint {ckpt_path}. "
                f"Available keys: {list(state_dict.keys())}"
            )

        self.adapter_modules.load_state_dict(adapter_sd, strict=False)

        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))
        new_bf_sum = torch.sum(torch.stack([torch.sum(p) for p in self.bf_texture_conditioner.parameters()]))

        if "bf_texture_conditioner" in state_dict:
            assert orig_bf_sum != new_bf_sum, "Weights of bf_texture_conditioner did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")


def save_texture_adapter_checkpoint(accelerator, model, save_path):
    unwrapped = accelerator.unwrap_model(model)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    state = {
        "texture_adapter": unwrapped.adapter_modules.state_dict(),
        "bf_texture_conditioner": unwrapped.bf_texture_conditioner.state_dict(),
    }

    torch.save(state, save_path)

    print(f"Saved texture adapter checkpoint to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Texture Adapter training script.")

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_texture_adapter_path",
        type=str,
        default=None,
        help="Path to pretrained texture adapter model. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--pretrained_color_adapter_path",
        type=str,
        default=None,
        help="Backward-compatible alias for old color adapter checkpoints.",
    )
    parser.add_argument(
        "--data_json_file",
        type=str,
        default=None,
        required=True,
        help="Training data json file.",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="",
        required=True,
        help="Training data root path.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        required=True,
        help="Path to CLIP image encoder.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory where checkpoints and logs will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Logging directory.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Resolution for input images.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Training image width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=640,
        help="Training image height.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size per device.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of workers for dataloader.",
    )
    parser.add_argument(
        "--i_drop_rate",
        type=float,
        default=0.05,
        help="Probability to drop texture image condition for CFG-style training.",
    )
    parser.add_argument(
        "--t_drop_rate",
        type=float,
        default=0.05,
        help="Probability to drop text condition for CFG-style training.",
    )
    parser.add_argument(
        "--ti_drop_rate",
        type=float,
        default=0.05,
        help="Probability to drop both text and texture image condition for CFG-style training.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help="Save checkpoint every X optimizer steps.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Mixed precision type.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help='Reporting backend: "tensorboard", "wandb", "comet_ml", or "all".',
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="IMAGGarment-1",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Optional Weights & Biases run name.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Optional Weights & Biases team/user entity.",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="Weights & Biases mode.",
    )
    parser.add_argument(
        "--bf_num_tokens",
        type=int,
        default=4,
        help="Number of BF texture conditioning tokens.",
    )
    parser.add_argument(
        "--bf_base_channels",
        type=int,
        default=32,
        help="Base channel width for BF texture conditioner.",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training.")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    pretrained_adapter_path = (
        args.pretrained_texture_adapter_path
        if args.pretrained_texture_adapter_path is not None
        else args.pretrained_color_adapter_path
    )

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)

    # freeze backbones
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)

    bf_texture_conditioner = BFTextureConditioner(
        clip_embeddings_dim=image_encoder.config.projection_dim,
        cross_attention_dim=unet.config.cross_attention_dim,
        num_tokens=args.bf_num_tokens,
        base_channels=args.bf_base_channels,
    )

    # init adapter modules on UNet attention processors
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
            attn_procs[name] = IPAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
            )
            attn_procs[name].load_state_dict(weights)

    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())

    texture_adapter = TextureAdapter(
        unet=unet,
        adapter_modules=adapter_modules,
        bf_texture_conditioner=bf_texture_conditioner,
        ckpt_path=pretrained_adapter_path,
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)

    params_to_opt = itertools.chain(
        texture_adapter.adapter_modules.parameters(),
        texture_adapter.bf_texture_conditioner.parameters(),
    )
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)

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
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    texture_adapter, optimizer, train_dataloader = accelerator.prepare(
        texture_adapter, optimizer, train_dataloader
    )
    if accelerator.is_main_process:
        init_kwargs = None
        if args.report_to in ("wandb", "all"):
            init_kwargs = {
                "wandb": {
                    "project": args.wandb_project,
                    "name": args.wandb_run_name,
                    "entity": args.wandb_entity,
                    "mode": args.wandb_mode,
                }
            }
        accelerator.init_trackers("texture_adapter_training", config=vars(args), init_kwargs=init_kwargs)
        accelerator.log(
            {
                "dataset/num_samples": len(train_dataset),
                "dataset/num_batches_per_epoch": len(train_dataloader),
                "dataset/total_batch_size": args.train_batch_size * accelerator.num_processes,
            },
            step=0,
        )

    global_step = 0
    for epoch in range(args.num_train_epochs):
        begin = time.perf_counter()
        epoch_loss_sum = 0.0
        epoch_loss_count = 0

        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin

            with accelerator.accumulate(texture_adapter):
                with torch.no_grad():
                    latents = vae.encode(batch["images"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                timesteps = torch.randint(
                    0,
                    noise_scheduler.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                with torch.no_grad():
                    image_embeds = image_encoder(
                        batch["clip_texture_images"].to(accelerator.device, dtype=weight_dtype)
                    ).image_embeds

                image_embeds_ = []
                for image_embed, drop_image_embed in zip(image_embeds, batch["drop_image_embeds"]):
                    if drop_image_embed == 1:
                        image_embeds_.append(torch.zeros_like(image_embed))
                    else:
                        image_embeds_.append(image_embed)
                image_embeds = torch.stack(image_embeds_)

                with torch.no_grad():
                    encoder_hidden_states = text_encoder(
                        batch["text_input_ids"].to(accelerator.device)
                    )[0]

                noise_pred = texture_adapter(
                    noisy_latents=noisy_latents,
                    timesteps=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    image_embeds=image_embeds,
                    texture_images=batch["texture_images"].to(accelerator.device, dtype=weight_dtype),
                )

                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
                epoch_loss_sum += avg_loss
                epoch_loss_count += 1

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.is_main_process:
                    print(
                        f"Epoch {epoch}, step {step}, data_time: {load_data_time:.4f}, "
                        f"time: {time.perf_counter() - begin:.4f}, step_loss: {avg_loss:.6f}"
                    )
                if accelerator.sync_gradients:
                    accelerator.log(
                        {
                            "train/loss": avg_loss,
                            "train/step_loss": loss.detach().item(),
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "train/data_time": load_data_time,
                            "train/step_time": time.perf_counter() - begin,
                            "train/epoch": epoch,
                        },
                        step=global_step,
                    )

            global_step += 1

            if global_step % args.save_steps == 0:
                save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_dir, safe_serialization=False)

                if accelerator.is_main_process:
                    save_texture_adapter_checkpoint(
                        accelerator,
                        texture_adapter,
                        os.path.join(save_dir, "texture_adapter.bin"),
                    )

            begin = time.perf_counter()

        if accelerator.is_main_process and epoch_loss_count > 0:
            epoch_loss = epoch_loss_sum / epoch_loss_count
            accelerator.log(
                {
                    "train/epoch_loss": epoch_loss,
                    "train/epoch": epoch,
                },
                step=global_step,
            )
            print(f"Epoch {epoch} finished, epoch_loss: {epoch_loss:.6f}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
